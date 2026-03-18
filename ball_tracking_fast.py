"""
Optimized Ball Tracking and Score Detection Pipeline
=====================================================
Same behavior as ball_tracking_analysis.py but with:
- Frame producer thread (decode ahead, overlap I/O with inference)
- Batched digit inference (1-2 GPU calls per score cycle instead of 4-8)
- Optional async video writer thread

Does not modify ball_tracking_analysis.py; imports and reuses its components.
"""

import cv2
import numpy as np
import json
import sys
import os
import time
import threading
import queue
from collections import deque, Counter
from pathlib import Path

# Ensure sort and ball_tracking_analysis are importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, 'sort'))

from sort import Sort
from ultralytics import YOLO

# Import from original module (no changes to it)
import ball_tracking_analysis as bta
from ball_tracking_analysis import (
    resolve_model_path,
    BALL_MODEL_PATH,
    DIGIT_MODEL_PATH,
    BALL_CONF_THRESHOLD,
    DIGIT_CONF_THRESHOLD,
    MIN_DIGIT_CONF_RELIABLE,
    SCORE_STABLE_RUNS,
    TRACKER_MAX_AGE,
    TRACKER_MIN_HITS,
    TRACKER_IOU_THRESHOLD,
    RALLY_BALL_SEEN_FRAMES,
    RALLY_BALL_MISSING_FRAMES,
    TableCalibration,
    BallTracker,
    RallyAggregator,
    DataLogger,
    RealtimeStats,
    interactive_frame_setup,
    _run_benchmark,
    PoseFeatureExtractor,
    POSE_TARGET_FPS,
    POSE_MODEL_PATH,
    _MEDIAPIPE_AVAILABLE,
)

# Unique sentinel object for end of stream (use identity check: `item is END_SENTINEL`)
class _Sentinel:
    pass
END_SENTINEL = _Sentinel()


# =============================================================================
# FRAME PRODUCER THREAD
# =============================================================================
def frame_producer_thread(cap, frame_queue):
    """Read frames from cap and put (frame_idx, frame) into frame_queue. Puts END_SENTINEL when done."""
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(END_SENTINEL)
                break
            frame_queue.put((frame_idx, frame))
            frame_idx += 1
    except Exception:
        frame_queue.put(END_SENTINEL)


# =============================================================================
# VIDEO WRITER THREAD (OPTIONAL)
# =============================================================================
def video_writer_thread(write_queue, output_path, fourcc, fps, write_w, write_h, output_size):
    """Consume (frame_idx, frame) from write_queue; resize if output_size; write. Stops on END_SENTINEL."""
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (write_w, write_h))
    try:
        while True:
            item = write_queue.get()
            if item is END_SENTINEL:
                break
            frame_idx, frame = item
            if output_size:
                frame = cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
    finally:
        writer.release()


# =============================================================================
# BATCHED SCORE DETECTOR (same API as ScoreDetector, batched inference)
# =============================================================================
def _preprocess_for_detection(crop, clahe=None):
    """Same as ScoreDetector.preprocess_for_detection (copied to avoid modifying original)."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    min_size = 64
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return processed


def _parse_digit_result(boxes, conf_threshold=DIGIT_CONF_THRESHOLD, max_score=30):
    """From one result's boxes, compute score (int or None), num_detections, mean_conf, min_conf. Same logic as original."""
    if not boxes or len(boxes) == 0:
        return None, 0, 0.0, 0.0
    digits = []
    for box in boxes:
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        cls = int(box.cls.item())
        conf = box.conf.item()
        if cls < 0 or cls > 9:
            continue
        cx = (bx1 + bx2) / 2
        digits.append({'digit': str(cls), 'x': cx, 'conf': conf})
    if not digits:
        return None, 0, 0.0, 0.0
    digits.sort(key=lambda d: d['x'])
    score_str = ''.join(d['digit'] for d in digits[:2])
    try:
        score = int(score_str)
        if score > max_score:
            return None, 0, 0.0, 0.0
        num_det = len(digits)
        mean_conf = sum(d['conf'] for d in digits) / num_det
        min_conf = min(d['conf'] for d in digits)
        return score, num_det, mean_conf, min_conf
    except ValueError:
        return None, 0, 0.0, 0.0


# ROI keys in order: score then rounds (matches update_scores + detect_rounds)
ROI_KEYS_ORDER = [
    ('player1', 'player1_score', 'score'),
    ('player2', 'player2_score', 'score'),
    ('player1', 'player1_rounds', 'rounds'),
    ('player2', 'player2_rounds', 'rounds'),
]


class ScoreDetectorBatched:
    """Same external API as ScoreDetector; uses one batched YOLO call per score cycle (4 ROIs at once)."""

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # TensorRT engines are typically built with batch size 1; batch inference will fail
        self._is_tensorrt = str(model_path).endswith('.engine') or Path(model_path).suffix == '.engine'
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.score_history = {'player1': deque(maxlen=5), 'player2': deque(maxlen=5)}
        self.current_scores = {'player1': None, 'player2': None}
        self.stable_scores = {'player1': None, 'player2': None}
        self._stable_candidate = {'player1': None, 'player2': None}
        self._stable_run = {'player1': 0, 'player2': 0}
        self._rounds_candidate = {'player1': None, 'player2': None}
        self._rounds_run = {'player1': 0, 'player2': 0}
        self.rounds = {'player1': 0, 'player2': 0}
        self.score_roi_obscured = {'player1': False, 'player2': False}
        self.rounds_roi_obscured = {'player1': False, 'player2': False}
        self.last_processed_frame = None
        self._imgsz = 320

    def stop(self):
        pass

    def _crop_and_preprocess(self, frame, roi):
        if roi is None:
            return None
        x1, y1, x2, y2 = roi
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None
        processed = _preprocess_for_detection(crop, clahe=self._clahe)
        return processed if processed is not None else crop

    def _build_batch(self, frame, rois):
        """Build list of 4 images (one per ROI, preprocessed). Pad/resize to same size for batch."""
        images = []
        for _player, roi_key, _kind in ROI_KEYS_ORDER:
            roi = rois.get(roi_key)
            img = self._crop_and_preprocess(frame, roi)
            if img is None:
                # Placeholder so batch indices stay aligned
                img = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            else:
                if img.shape[0] != self._imgsz or img.shape[1] != self._imgsz:
                    img = cv2.resize(img, (self._imgsz, self._imgsz), interpolation=cv2.INTER_LINEAR)
            images.append(img)
        return images

    def _run_batch(self, images):
        """Run model on list of images; return list of Results (one per image).
        TensorRT engines are usually built with batch size 1, so run one image at a time when using .engine."""
        if not images:
            return []
        if self._is_tensorrt:
            results_list = []
            for img in images:
                r = self.model(img, conf=DIGIT_CONF_THRESHOLD, imgsz=self._imgsz, verbose=False)
                if r is not None:
                    # predict returns list of Results; take first
                    results_list.append(r[0] if hasattr(r, '__getitem__') else r)
                else:
                    results_list.append(None)
            return results_list
        results = self.model(images, conf=DIGIT_CONF_THRESHOLD, imgsz=self._imgsz, verbose=False)
        if results is None:
            return []
        return list(results) if hasattr(results, '__iter__') and not isinstance(results, (str, dict)) else [results]

    def update_scores_and_rounds(self, frame, rois, frame_idx):
        """
        Run batched digit detection for all 4 ROIs, then update scores and rounds (same logic as original).
        Returns (current_scores, rounds).
        """
        images = self._build_batch(frame, rois)
        results_list = self._run_batch(images)

        # Map batch index -> (player, roi_key, kind)
        roi_info = list(ROI_KEYS_ORDER)  # [(player, roi_key, 'score'|'rounds'), ...]

        # Parse each result
        parsed = []
        for i, (player, roi_key, kind) in enumerate(roi_info):
            score_val, num_det, mean_conf, min_conf = None, 0, 0.0, 0.0
            if i < len(results_list) and results_list[i] is not None:
                boxes = results_list[i].boxes if hasattr(results_list[i], 'boxes') else []
                score_val, num_det, mean_conf, min_conf = _parse_digit_result(boxes, max_score=30)
            parsed.append((player, roi_key, kind, score_val, num_det, min_conf))

        # Update score state (players 1 & 2 from first two ROIs)
        for player, roi_key, kind, score_val, num_det, min_conf in parsed:
            if kind != 'score':
                continue
            reliable = (num_det >= 1 and min_conf >= MIN_DIGIT_CONF_RELIABLE)
            self.score_roi_obscured[player] = not reliable
            if reliable and score_val is not None:
                self.score_history[player].append(score_val)
                if len(self.score_history[player]) >= 3:
                    counts = Counter(self.score_history[player])
                    most_common = counts.most_common(1)[0]
                    if most_common[1] >= 2:
                        voted = most_common[0]
                        self.current_scores[player] = voted
                        if voted == self._stable_candidate[player]:
                            self._stable_run[player] += 1
                        else:
                            self._stable_candidate[player] = voted
                            self._stable_run[player] = 1
                        if self._stable_run[player] >= SCORE_STABLE_RUNS:
                            self.stable_scores[player] = voted
            else:
                self._stable_run[player] = 0

        # Update rounds (last two ROIs)
        for player, roi_key, kind, score_val, num_det, min_conf in parsed:
            if kind != 'rounds':
                continue
            reliable = (num_det >= 1 and min_conf >= MIN_DIGIT_CONF_RELIABLE)
            self.rounds_roi_obscured[player] = not reliable
            if reliable and score_val is not None and score_val <= 5:
                if score_val == self._rounds_candidate[player]:
                    self._rounds_run[player] += 1
                else:
                    self._rounds_candidate[player] = score_val
                    self._rounds_run[player] = 1
                if self._rounds_run[player] >= SCORE_STABLE_RUNS:
                    self.rounds[player] = score_val
            else:
                self._rounds_run[player] = 0

        self.last_processed_frame = frame_idx
        return self.current_scores, self.rounds

    def update_scores(self, frame, rois, frame_idx):
        """API compatibility: run batch and return current_scores."""
        self.update_scores_and_rounds(frame, rois, frame_idx)
        return self.current_scores

    def detect_rounds(self, frame, rois):
        """API compatibility: must be called after update_scores in same cycle; rounds already updated."""
        return self.rounds

    def draw_scores(self, frame, rois, player_names=None):
        """Same drawing as original ScoreDetector.draw_scores (copied)."""
        h, w = frame.shape[:2]
        p1_name = (player_names[0] if player_names else None) or "Player 1"
        p2_name = (player_names[1] if player_names else None) or "Player 2"
        cv2.rectangle(frame, (20, 20), (200, 100), (0, 70, 0), -1)
        p1_score = self.current_scores['player1']
        p1_score_text = str(p1_score) if p1_score is not None else "--"
        if self.score_roi_obscured['player1']:
            p1_score_text += " (obscured)"
        cv2.putText(frame, p1_name, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, p1_score_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        sets_p1 = str(self.rounds['player1']) + (" (obscured)" if self.rounds_roi_obscured['player1'] else "")
        cv2.putText(frame, f"Sets: {sets_p1}", (120, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.rectangle(frame, (w - 200, 20), (w - 20, 100), (70, 0, 0), -1)
        p2_score = self.current_scores['player2']
        p2_score_text = str(p2_score) if p2_score is not None else "--"
        if self.score_roi_obscured['player2']:
            p2_score_text += " (obscured)"
        cv2.putText(frame, p2_name, (w - 190, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, p2_score_text, (w - 190, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        sets_p2 = str(self.rounds['player2']) + (" (obscured)" if self.rounds_roi_obscured['player2'] else "")
        cv2.putText(frame, f"Sets: {sets_p2}", (w - 90, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        for roi_name, roi in rois.items():
            if roi:
                x1, y1, x2, y2 = roi
                color = (0, 255, 0) if 'player1' in roi_name else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


# =============================================================================
# MANUAL SCORE TRACKER  (replaces ScoreDetectorBatched when score_mode='manual')
# =============================================================================
class ManualScoreTracker:
    """
    Drop-in replacement for ScoreDetectorBatched when the user enters scores manually.
    No OCR is run. Scores are adjusted via key presses (1/2 to increment, [/] to decrement).
    Z key swaps score display without changing pose or ROIs.
    """

    def __init__(self, initial_scores=None, initial_rounds=None):
        self.current_scores = {'player1': 0, 'player2': 0}
        self.stable_scores  = {'player1': 0, 'player2': 0}
        self.rounds         = {'player1': 0, 'player2': 0}
        self.score_roi_obscured  = {'player1': False, 'player2': False}
        self.rounds_roi_obscured = {'player1': False, 'player2': False}
        if initial_scores:
            self.current_scores.update(initial_scores)
            self.stable_scores.update(initial_scores)
        if initial_rounds:
            self.rounds.update(initial_rounds)

    def stop(self):
        pass

    def adjust(self, player, delta):
        """Increment or decrement a player's score. player='player1' or 'player2', delta=+1|-1."""
        v = max(0, self.current_scores[player] + delta)
        self.current_scores[player] = v
        self.stable_scores[player]  = v

    def adjust_rounds(self, player, delta):
        """Increment or decrement a player's set count."""
        v = max(0, self.rounds[player] + delta)
        self.rounds[player] = v

    def swap_scores(self):
        """Swap score+sets display between players (no ROI/pose change)."""
        s = self.current_scores
        r = self.rounds
        s['player1'], s['player2'] = s['player2'], s['player1']
        r['player1'], r['player2'] = r['player2'], r['player1']
        self.stable_scores['player1'] = s['player1']
        self.stable_scores['player2'] = s['player2']

    def draw_scores(self, frame, rois, player_names=None):
        """Draw score overlay without ROI rectangles (no OCR regions in manual mode)."""
        h, w = frame.shape[:2]
        p1_name = (player_names[0] if player_names else None) or "Player 1"
        p2_name = (player_names[1] if player_names else None) or "Player 2"
        cv2.rectangle(frame, (20, 20), (200, 100), (0, 70, 0), -1)
        cv2.putText(frame, p1_name, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, str(self.current_scores['player1']), (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Sets: {self.rounds['player1']}", (120, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.rectangle(frame, (w - 200, 20), (w - 20, 100), (70, 0, 0), -1)
        cv2.putText(frame, p2_name, (w - 190, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, str(self.current_scores['player2']), (w - 190, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Sets: {self.rounds['player2']}", (w - 90, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        # Manual mode hint
        cv2.putText(frame, "MANUAL SCORES  1/2:+  [/]:- Z:swap sets", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 60), 1)


# =============================================================================
# POSE SKELETON OVERLAY (debug)
# =============================================================================
def draw_pose_skeleton(frame, landmarks_px, color, thickness=2, joint_radius=4):
    """
    Draw a stick-figure skeleton on frame. landmarks_px: dict name -> (x, y) int.
    Used for debugging pose tracking; can be removed or gated by a flag later.
    """
    if not landmarks_px:
        return
    # Connections: (start_name, end_name)
    pairs = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
    ]
    for a, b in pairs:
        if a in landmarks_px and b in landmarks_px:
            pt1 = tuple(landmarks_px[a])
            pt2 = tuple(landmarks_px[b])
            cv2.line(frame, pt1, pt2, color, thickness)
    for name, pt in landmarks_px.items():
        cv2.circle(frame, tuple(pt), joint_radius, color, -1)


# =============================================================================
# CONFIG LOADING (no dependency on original's load_config_and_run)
# =============================================================================
def load_config(config_path):
    """Load JSON config; return (rois, table_corners). table_corners is list of 4 points or empty."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    rois = {}
    for k, v in config['rois'].items():
        rois[k] = tuple(v) if v else None
    table_corners = config.get('table_corners')
    if table_corners is None:
        table_corners = []
    return rois, table_corners


# =============================================================================
# OPTIMIZED RUN (producer thread + batched score + optional async write)
# =============================================================================
def optimized_run(
    video_path,
    output_dir="tracking_output",
    save_video=True,
    inference_size=None,
    score_interval_sec=1,
    output_size=None,
    use_tensorrt=False,
    config_path=None,
    async_write=True,
    score_mode="auto",        # "auto" (OCR) | "manual" (key-driven)
    initial_scores=None,      # {"player1": 0, "player2": 0}  — manual mode only
    initial_rounds=None,      # {"player1": 0, "player2": 0}  — manual mode only
    player_names=None,        # ["Player 1 name", "Player 2 name"]
):
    """
    Run tracking with optimized pipeline: frame producer thread, batched digit inference,
    optional async video writer. Uses same ROIs/config as original.
    """
    ball_model_path = resolve_model_path(BALL_MODEL_PATH, require_engine=use_tensorrt)
    digit_model_path = resolve_model_path(DIGIT_MODEL_PATH, require_engine=use_tensorrt)
    if use_tensorrt:
        for name, path in [("Ball", ball_model_path), ("Digit", digit_model_path)]:
            if not Path(path).exists():
                print(f"Error: --use-tensorrt set but {name} engine not found: {path}")
                return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if config_path:
        rois, table_corners = load_config(config_path)
        if len(table_corners) == 4:
            table_calibration = TableCalibration(table_corners)
            table_calibration = table_calibration if table_calibration.is_valid() else None
        else:
            table_calibration = None
        print(f"Loaded config from {config_path}")
    else:
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            cap.release()
            return
        result = interactive_frame_setup(first_frame)
        if result is None:
            print("Setup cancelled.")
            cap.release()
            return
        rois, table_corners = result
        table_calibration = None
        if len(table_corners) == 4:
            table_calibration = TableCalibration(table_corners)
            table_calibration = table_calibration if table_calibration.is_valid() else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("\n" + "=" * 60)
    print("OPTIMIZED PIPELINE (producer thread + batched score)")
    print("=" * 60)
    print(f"Video: {video_path} | {frame_width}x{frame_height} @ {fps} FPS")
    if inference_size:
        print(f"Ball inference size: {inference_size[0]}x{inference_size[1]}")
    else:
        print(f"Ball inference size: full resolution")
    print(f"Ball model:   {ball_model_path}")
    print(f"Digit model:  {digit_model_path} (batched)")
    if async_write:
        print("Video write:  async (writer thread)")
    else:
        print("Video write:  sync")

    print("\nLoading models...")
    ball_tracker = BallTracker(ball_model_path, fps, table_calibration=table_calibration, inference_size=inference_size)
    if score_mode == "manual":
        score_detector = ManualScoreTracker(initial_scores=initial_scores, initial_rounds=initial_rounds)
        print("Score mode:   MANUAL (use 1/2 to score, [/] to undo, Z to swap)")
    else:
        score_detector = ScoreDetectorBatched(digit_model_path)
        print("Score mode:   AUTO (OCR)")
    logger = DataLogger(output_dir, with_meters=(table_calibration is not None and table_calibration.is_valid()))
    rally_aggregator = RallyAggregator(fps, logger, table_calibration)
    stats = RealtimeStats(fps)
    pose_extractor = PoseFeatureExtractor(fps)
    logger.save_config(rois, video_path, fps, table_corners=table_corners)
    print("  Models loaded!")
    if _MEDIAPIPE_AVAILABLE:
        print(f"  Pose logging: ENABLED  (target {POSE_TARGET_FPS:.0f} FPS, file: {logger.pose_file.name})")
    else:
        print("  Pose logging: DISABLED (mediapipe not installed)")

    pose_interval = max(1, int(fps / POSE_TARGET_FPS))

    write_w = output_size[0] if output_size else frame_width
    write_h = output_size[1] if output_size else frame_height
    video_writer = None
    write_queue = None
    writer_thread = None
    output_video_path = None
    if save_video:
        output_video_path = Path(output_dir) / f"tracked_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if async_write:
            write_queue = queue.Queue(maxsize=8)
            writer_thread = threading.Thread(
                target=video_writer_thread,
                args=(write_queue, output_video_path, fourcc, fps, write_w, write_h, output_size),
            )
            writer_thread.start()
        else:
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (write_w, write_h))
        print(f"Output video: {output_video_path} ({write_w}x{write_h})")

    cv2.namedWindow("Ball Tracking - Real-Time", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Tracking - Real-Time", 1280, 720)

    score_detect_interval = max(1, int(fps * score_interval_sec))
    frame_queue = queue.Queue(maxsize=4)
    producer = threading.Thread(target=frame_producer_thread, args=(cap, frame_queue), daemon=True)
    producer.start()

    frame_idx = 0
    last_score_log = None
    paused = False
    playback_speed = 1.0
    frame_delay_base = 1.0 / fps if fps > 0 else 1.0 / 30.0
    last_displayed_frame = None  # for pause: redraw without consuming queue
    sides_swapped = False
    score_freeze_until_frame = 0

    if score_mode == "manual":
        print("\nOptimized tracking started. Q=Quit P=Pause +/-=Speed S=Screenshot X=SwapSides")
        print("Manual score keys: 1/2=+1 point  [/]=-1 point  3/4=+1 set  Z=swap scores")
    else:
        print("\nOptimized tracking started. Q=Quit P=Pause +/-=Speed S=Screenshot X=SwapSides")

    try:
        while True:
            loop_start = time.time()

            if paused:
                if last_displayed_frame is not None:
                    display_frame = last_displayed_frame.copy()
                    stats.draw(display_frame, frame_idx, total_frames, paused, playback_speed)
                    cv2.imshow("Ball Tracking - Real-Time", display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = False
                    print("Resumed")
                elif key == ord('s'):
                    screenshot_path = Path(output_dir) / f"screenshot_{frame_idx}.jpg"
                    if last_displayed_frame is not None:
                        cv2.imwrite(str(screenshot_path), last_displayed_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                elif key == ord('x'):
                    sides_swapped = not sides_swapped
                    rois['player1_score'], rois['player2_score'] = rois['player2_score'], rois['player1_score']
                    rois['player1_rounds'], rois['player2_rounds'] = rois['player2_rounds'], rois['player1_rounds']
                    score_freeze_until_frame = frame_idx + int(fps * 8)
                    print(f"Sides swapped: Player 1 now on {'right' if sides_swapped else 'left'}")
                elif key == ord('1') and score_mode == "manual":
                    score_detector.adjust('player1', +1)
                    print(f"P1 score: {score_detector.current_scores['player1']}")
                elif key == ord('2') and score_mode == "manual":
                    score_detector.adjust('player2', +1)
                    print(f"P2 score: {score_detector.current_scores['player2']}")
                elif key == ord('[') and score_mode == "manual":
                    score_detector.adjust('player1', -1)
                    print(f"P1 score: {score_detector.current_scores['player1']}")
                elif key == ord(']') and score_mode == "manual":
                    score_detector.adjust('player2', -1)
                    print(f"P2 score: {score_detector.current_scores['player2']}")
                elif key == ord('z') and score_mode == "manual":
                    score_detector.swap_scores()
                    print(f"Scores swapped — P1:{score_detector.current_scores['player1']} P2:{score_detector.current_scores['player2']}")
                elif key == ord('3') and score_mode == "manual":
                    score_detector.adjust_rounds('player1', +1)
                    print(f"P1 sets: {score_detector.rounds['player1']}")
                elif key == ord('4') and score_mode == "manual":
                    score_detector.adjust_rounds('player2', +1)
                    print(f"P2 sets: {score_detector.rounds['player2']}")
                continue

            # Get next frame from producer (blocking)
            item = frame_queue.get()
            if item is END_SENTINEL:
                rally_aggregator.flush_final(frame_idx, frame_idx / fps if fps > 0 else 0)
                break
            frame_idx, frame = item

            t0 = time.perf_counter()
            tracks = ball_tracker.detect_and_track(frame)
            stats.record_phase('ball_infer', time.perf_counter() - t0)

            ball_detected = len(tracks) > 0
            max_speed = 0
            max_speed_mps = None
            tracks_with_meters = []
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                speed = ball_tracker.get_smoothed_speed(track_id)
                max_speed = max(max_speed, speed)
                x_m, y_m = ball_tracker.get_position_meters(track_id)
                speed_mps = ball_tracker.get_smoothed_speed_mps(track_id)
                if speed_mps is not None:
                    max_speed_mps = max(max_speed_mps or 0, speed_mps)
                logger.log_trajectory(frame_idx, track_id, cx, cy, speed, x_m=x_m, y_m=y_m, speed_mps=speed_mps)
                tracks_with_meters.append((track_id, x_m, y_m, speed_mps))

            stats.update_ball_stats(ball_detected, max_speed, max_speed_mps)

            # Pose extraction (every pose_interval frames, timed)
            if frame_idx % pose_interval == 0:
                t0 = time.perf_counter()
                ts = frame_idx / fps
                for pid in (1, 2):
                    feat = pose_extractor.update(frame, frame_idx, pid, ts, sides_swapped=sides_swapped)
                    if feat is not None:
                        logger.log_pose(feat)
                        # Debug: draw skeleton overlay (P1=green, P2=blue)
                        lm_px = feat.get('debug_landmarks')
                        if lm_px:
                            color = (0, 255, 0) if pid == 1 else (255, 0, 0)  # BGR
                            draw_pose_skeleton(frame, lm_px, color)
                stats.record_phase('pose_infer', time.perf_counter() - t0)

            if score_mode == "auto" and frame_idx % score_detect_interval == 0 and frame_idx >= score_freeze_until_frame:
                t0 = time.perf_counter()
                scores, rounds = score_detector.update_scores_and_rounds(frame, rois, frame_idx)
                stats.record_phase('score_infer', time.perf_counter() - t0)
            else:
                scores = score_detector.current_scores
                rounds = score_detector.rounds

            rally_aggregator.add_frame(frame_idx, frame_idx / fps, tracks_with_meters, score_detector.stable_scores, rounds)
            stats.rally_display = rally_aggregator.get_state_for_display()

            t0 = time.perf_counter()
            ball_tracker.draw_trajectories(frame, tracks)
            current_score_log = (scores['player1'], scores['player2'], rounds['player1'], rounds['player2'])
            if current_score_log != last_score_log:
                logger.log_score(frame_idx, frame_idx / fps, scores, rounds, score_detector.score_roi_obscured, score_detector.rounds_roi_obscured)
                last_score_log = current_score_log
            score_detector.draw_scores(frame, rois, player_names=player_names)
            stats.record_phase('draw', time.perf_counter() - t0)

            stats.update()

            if save_video:
                t0 = time.perf_counter()
                if async_write and write_queue is not None:
                    try:
                        write_frame = cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR) if output_size else frame
                        write_queue.put_nowait((frame_idx, write_frame))
                    except queue.Full:
                        pass
                elif video_writer is not None:
                    write_frame = cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR) if output_size else frame
                    video_writer.write(write_frame)
                stats.record_phase('write', time.perf_counter() - t0)

            stats.draw(frame, frame_idx, total_frames, paused, playback_speed)
            cv2.imshow("Ball Tracking - Real-Time", frame)
            last_displayed_frame = frame  # for pause: show this frame again when paused

            processing_time = time.time() - loop_start
            wait_time = max(1, int((frame_delay_base / playback_speed - processing_time) * 1000))
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = True
                print("Paused")
            elif key in [ord('+'), ord('=')]:
                playback_speed = min(4.0, playback_speed + 0.25)
                print(f"Playback speed: {playback_speed}x")
            elif key == ord('-'):
                playback_speed = max(0.25, playback_speed - 0.25)
                print(f"Playback speed: {playback_speed}x")
            elif key == ord('0'):
                playback_speed = 1.0
                print("Playback speed: 1.0x")
            elif key == ord('s'):
                screenshot_path = Path(output_dir) / f"screenshot_{frame_idx}.jpg"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"Screenshot saved: {screenshot_path}")
            elif key == ord('x'):
                sides_swapped = not sides_swapped
                rois['player1_score'], rois['player2_score'] = rois['player2_score'], rois['player1_score']
                rois['player1_rounds'], rois['player2_rounds'] = rois['player2_rounds'], rois['player1_rounds']
                score_freeze_until_frame = frame_idx + int(fps * 8)
                print(f"Sides swapped: Player 1 now on {'right' if sides_swapped else 'left'}")
            elif key == ord('1') and score_mode == "manual":
                score_detector.adjust('player1', +1)
                print(f"P1 score: {score_detector.current_scores['player1']}")
            elif key == ord('2') and score_mode == "manual":
                score_detector.adjust('player2', +1)
                print(f"P2 score: {score_detector.current_scores['player2']}")
            elif key == ord('[') and score_mode == "manual":
                score_detector.adjust('player1', -1)
                print(f"P1 score: {score_detector.current_scores['player1']}")
            elif key == ord(']') and score_mode == "manual":
                score_detector.adjust('player2', -1)
                print(f"P2 score: {score_detector.current_scores['player2']}")
            elif key == ord('z') and score_mode == "manual":
                score_detector.swap_scores()
                print(f"Scores swapped — P1:{score_detector.current_scores['player1']} P2:{score_detector.current_scores['player2']}")
            elif key == ord('3') and score_mode == "manual":
                score_detector.adjust_rounds('player1', +1)
                print(f"P1 sets: {score_detector.rounds['player1']}")
            elif key == ord('4') and score_mode == "manual":
                score_detector.adjust_rounds('player2', +1)
                print(f"P2 sets: {score_detector.rounds['player2']}")

    finally:
        score_detector.stop()
        pose_extractor.close()
        # Drain producer so it can exit (in case we broke out early)
        try:
            while True:
                frame_queue.get_nowait()
        except queue.Empty:
            pass
        producer.join(timeout=3.0)
        if write_queue is not None and writer_thread is not None:
            write_queue.put(END_SENTINEL)
            writer_thread.join(timeout=10.0)
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Trajectory log: {logger.trajectory_file}")
    print(f"Score log: {logger.score_file}")
    print(f"Rally log: {logger.rally_file}")
    print(f"Pose log: {logger.pose_file}")
    if save_video and output_video_path:
        print(f"Output video: {output_video_path}")
    print(f"Max ball speed recorded: {stats.max_speed:.0f} px/s")


# =============================================================================
# ENTRY POINT
# =============================================================================
def _parse_inference_size(s):
    try:
        w, h = s.lower().split('x')
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid inference size '{s}'. Expected WxH e.g. 1280x720")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimized Ball Tracking and Score Detection (producer thread + batched digits)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", default="tracking_output", help="Output directory")
    parser.add_argument("--config", "-c", default=None, help="Path to saved config JSON (skip interactive setup)")
    parser.add_argument("--no-video", action="store_true", help="Don't save output video")
    parser.add_argument("--ball-inference-size", metavar="WxH", default=None, type=_parse_inference_size,
                        help="Resize to WxH before ball inference (e.g. 640x360)")
    parser.add_argument("--output-size", metavar="WxH", default=None, type=_parse_inference_size,
                        help="Downscale saved video to WxH")
    parser.add_argument("--score-interval", metavar="SEC", type=float, default=2.0,
                        help="Seconds between score detector runs (default: 2.0)")
    parser.add_argument("--no-async-write", dest="async_write", action="store_false",
                        help="Disable background thread for video writing (async write is on by default)")
    parser.add_argument("--use-tensorrt", action="store_true", help="Require TensorRT .engine models")
    parser.add_argument("--manual-scores", action="store_true",
                        help="Disable OCR; enter scores manually with keyboard (1/2/[/]/3/4/Z keys)")
    parser.add_argument("--initial-scores", metavar="P1,P2", default="0,0",
                        help="Starting scores for manual mode, e.g. '3,5' (default: 0,0)")
    parser.add_argument("--initial-rounds", metavar="P1,P2", default="0,0",
                        help="Starting set counts for manual mode, e.g. '1,2' (default: 0,0)")
    parser.add_argument("--player-names", metavar="NAME1,NAME2", default=None,
                        help="Player display names, e.g. 'Alice,Bob'")
    parser.add_argument("--benchmark", metavar="N", type=int, default=0,
                        help="Run headless ball-inference benchmark over N frames and exit")

    parser.set_defaults(async_write=True)
    args = parser.parse_args()
    inf_size = args.ball_inference_size if args.ball_inference_size is not None else getattr(bta, 'BALL_INFERENCE_SIZE', None)

    if args.benchmark > 0:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video: {args.video}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _run_benchmark(cap, fps, fw, fh, inf_size, args.benchmark)
        cap.release()
        sys.exit(0)

    # Parse manual score/rounds initial values
    def _parse_two_ints(s, default=(0, 0)):
        try:
            a, b = s.split(',')
            return int(a.strip()), int(b.strip())
        except Exception:
            return default

    p1s, p2s = _parse_two_ints(args.initial_scores)
    p1r, p2r = _parse_two_ints(args.initial_rounds)
    player_names = None
    if args.player_names:
        parts = args.player_names.split(',', 1)
        player_names = [p.strip() for p in parts] if len(parts) == 2 else None

    optimized_run(
        args.video,
        output_dir=args.output,
        save_video=not args.no_video,
        inference_size=inf_size,
        score_interval_sec=args.score_interval,
        output_size=args.output_size,
        use_tensorrt=args.use_tensorrt,
        config_path=args.config,
        async_write=args.async_write,
        score_mode="manual" if args.manual_scores else "auto",
        initial_scores={'player1': p1s, 'player2': p2s},
        initial_rounds={'player1': p1r, 'player2': p2r},
        player_names=player_names,
    )
