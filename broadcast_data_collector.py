"""
Broadcast Data Collector
========================
Standalone data collection pipeline for broadcast table-tennis footage.
Imports and reuses existing classes from ball_tracking_analysis.py,
ball_tracking_fast.py, and broadcast_utils — does NOT modify any of them.

Features:
  1. ManualSceneGate  — press F to pause inference during cutscenes/replays,
                        press F again to resume (also resets the SORT tracker
                        so stale tracks don't bleed into the next rally)
  2. SkippableRallyAggregator — press D to discard a rally with a bad camera
                        angle; pose and trajectory data are not saved but score
                        tracking continues so the next rally starts cleanly.
                        Press U to undo.

Usage:
    python broadcast_data_collector.py match.mp4 --output match1_data --manual-scores
    python broadcast_data_collector.py match.mp4 --broadcast-model path/to/best.pt
"""

import cv2
import numpy as np
import json
import sys
import os
import time
import threading
import queue
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, 'sort'))

from sort import Sort
from ultralytics import YOLO

import ball_tracking_analysis as bta
from ball_tracking_analysis import (
    resolve_model_path,
    BALL_MODEL_PATH,
    DIGIT_MODEL_PATH,
    TRACKER_MAX_AGE,
    TRACKER_MIN_HITS,
    TRACKER_IOU_THRESHOLD,
    TableCalibration,
    BallTracker,
    RallyAggregator,
    DataLogger,
    RealtimeStats,
    interactive_frame_setup,
    PoseFeatureExtractor,
    POSE_TARGET_FPS,
    _MEDIAPIPE_AVAILABLE,
)
from ball_tracking_fast import (
    frame_producer_thread,
    video_writer_thread,
    ManualScoreTracker,
    ScoreDetectorBatched,
    draw_pose_skeleton,
    load_config,
    END_SENTINEL,
    _parse_inference_size,
)
from prediction_model_base import PredictionResult
from prediction_pipeline import load_prediction_model, build_packet, draw_prediction_overlay


# =============================================================================
# MANUAL SCENE GATE
# =============================================================================
class ManualSceneGate:
    """
    Pure manual toggle for scene skipping.

    Press F  →  start skipping inference (cutscene / replay).
    Press F  →  resume inference; also flags is_cut=True so the caller
                resets the SORT tracker, preventing stale tracks from
                bleeding into the next rally.
    """

    def __init__(self):
        self.skipping = False
        self.is_cut = False  # True for one frame when resuming from skip

    def toggle(self):
        """Toggle skipping state. Returns True if now skipping."""
        self.skipping = not self.skipping
        # Signal a tracker reset only when we resume (skip → play)
        self.is_cut = not self.skipping
        return self.skipping

    def clear_cut(self):
        """Call once per frame after consuming is_cut."""
        self.is_cut = False

    @property
    def should_process(self):
        return not self.skipping


# =============================================================================
# SKIPPABLE RALLY AGGREGATOR
# =============================================================================
class SkippableRallyAggregator(RallyAggregator):
    """
    Extends RallyAggregator with per-rally skip support.

    When mark_skipped() is called:
      - self.skipped is set to True
      - Accumulated samples are cleared
      - The caller should stop logging pose/trajectory for this rally

    When the rally ends (score change), the flushed CSV row includes skipped=1.
    The skipped flag resets automatically for the next rally.
    """

    _SKIPPED_COLUMN = "skipped"

    def __init__(self, fps, data_logger, table_calibration=None):
        super().__init__(fps, data_logger, table_calibration)
        self.skipped = False
        self._patch_rally_header(data_logger)

    def _patch_rally_header(self, logger):
        """Append the skipped column to the rally CSV header."""
        rally_file = getattr(logger, 'rally_file', None)
        if rally_file is None:
            return
        try:
            with open(rally_file, 'r') as f:
                existing = f.read().strip()
            if existing and self._SKIPPED_COLUMN not in existing:
                with open(rally_file, 'w') as f:
                    f.write(existing + f",{self._SKIPPED_COLUMN}\n")
        except OSError:
            pass

    def mark_skipped(self):
        """Discard current rally data and mark it as skipped."""
        self.skipped = True
        self.samples = []

    def undo_skip(self):
        """Resume collection for the current rally."""
        self.skipped = False

    def _flush_rally(self, end_frame, end_time, point_winner):
        """Flush rally row with the skipped flag appended."""
        if self.skipped:
            self.samples = []
        original_log = self.logger.log_rally

        def _patched_log(record):
            record[self._SKIPPED_COLUMN] = 1 if self.skipped else 0
            original_log(record)

        self.logger.log_rally = _patched_log
        try:
            super()._flush_rally(end_frame, end_time, point_winner)
        finally:
            self.logger.log_rally = original_log

    def add_frame(self, frame_idx, timestamp_sec, tracks_with_meters, stable_scores, rounds):
        """Reset the skipped flag at the start of each new rally."""
        p1 = stable_scores.get('player1')
        p2 = stable_scores.get('player2')
        current_stable = (p1, p2)
        score_changed = (
            self.last_stable[0] is not None
            and current_stable[0] is not None
            and current_stable != self.last_stable
        )
        if score_changed:
            self.skipped = False
        super().add_frame(frame_idx, timestamp_sec, tracks_with_meters, stable_scores, rounds)


# =============================================================================
# DRAWING HELPERS
# =============================================================================
def _draw_scene_overlay(frame, gate, skipped_frames):
    """Draw the inference state indicator in the bottom-right corner."""
    h, w = frame.shape[:2]
    if gate.skipping:
        color = (0, 200, 200)
        text = f"SCENE SKIP — F to resume  (skipped: {skipped_frames})"
    else:
        color = (0, 200, 0)
        text = "COLLECTING — F to skip scene"
    cv2.putText(frame, text, (w - 500, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_rally_overlay(frame, rally_aggregator):
    """Draw the rally-discarded banner."""
    if not rally_aggregator.skipped:
        return
    h, w = frame.shape[:2]
    y1, y2 = h // 2 - 30, h // 2 + 30
    cv2.rectangle(frame, (w // 4, y1), (3 * w // 4, y2), (0, 0, 180), -1)
    cv2.putText(frame, "RALLY SKIPPED  (D=discard  U=undo)",
                (w // 4 + 20, h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _reset_ball_tracker(ball_tracker, frame):
    """Reset SORT tracker and clear trajectory history after a scene transition."""
    ball_tracker.tracker = Sort(
        max_age=TRACKER_MAX_AGE,
        min_hits=TRACKER_MIN_HITS,
        iou_threshold=TRACKER_IOU_THRESHOLD,
    )
    ball_tracker.trajectories.clear()
    ball_tracker.speed_history.clear()
    ball_tracker.trajectories_meters.clear()
    ball_tracker.speed_history_mps.clear()


# =============================================================================
# MAIN COLLECTION LOOP
# =============================================================================
def collect(
    video_path,
    output_dir="broadcast_data",
    save_video=True,
    inference_size=None,
    output_size=None,
    config_path=None,
    async_write=True,
    score_mode="manual",
    initial_scores=None,
    initial_rounds=None,
    player_names=None,
    broadcast_model=None,
    score_interval_sec=2.0,
    prediction_model=None,
):
    """Main data collection loop."""
    # ------------------------------------------------------------------
    # Model paths
    # ------------------------------------------------------------------
    ball_model_path = broadcast_model or resolve_model_path(BALL_MODEL_PATH)
    digit_model_path = resolve_model_path(DIGIT_MODEL_PATH)

    if not Path(ball_model_path).exists():
        print(f"Error: Ball model not found: {ball_model_path}")
        return

    # ------------------------------------------------------------------
    # Video capture
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # Interactive setup or load config
    # ------------------------------------------------------------------
    if config_path:
        rois, table_corners = load_config(config_path)
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ------------------------------------------------------------------
    # Table calibration (fixed — no optical-flow tracking)
    # ------------------------------------------------------------------
    table_calibration = None
    if len(table_corners) == 4:
        table_calibration = TableCalibration(table_corners)
        if not table_calibration.is_valid():
            table_calibration = None

    # ------------------------------------------------------------------
    # Scene gate (always on — purely manual)
    # ------------------------------------------------------------------
    scene_gate = ManualSceneGate()

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BROADCAST DATA COLLECTOR")
    print("=" * 60)
    print(f"Video : {video_path}  |  {frame_width}x{frame_height} @ {fps} FPS")
    print(f"Ball model : {ball_model_path}")
    if inference_size:
        print(f"Ball inference size : {inference_size[0]}x{inference_size[1]}")
    print(f"Table calibration : {'ON' if table_calibration else 'OFF (no corners marked)'}")
    print(f"Score mode : {score_mode.upper()}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("\nLoading models...")
    ball_tracker = BallTracker(
        ball_model_path, fps,
        table_calibration=table_calibration,
        inference_size=inference_size,
    )

    if score_mode == "manual":
        score_detector = ManualScoreTracker(
            initial_scores=initial_scores,
            initial_rounds=initial_rounds,
        )
    else:
        score_detector = ScoreDetectorBatched(digit_model_path)

    logger = DataLogger(output_dir, with_meters=(table_calibration is not None))
    rally_aggregator = SkippableRallyAggregator(fps, logger, table_calibration)
    stats = RealtimeStats(fps)
    pose_extractor = PoseFeatureExtractor(fps)
    logger.save_config(rois, video_path, fps, table_corners=table_corners)
    print("  Models loaded!")

    if prediction_model is not None:
        prediction_model.reset()
        print(f"  Prediction model: {type(prediction_model).__name__}")
    else:
        print("  Prediction model: none (disabled)")

    pose_interval = max(1, int(fps / POSE_TARGET_FPS))
    score_detect_interval = max(1, int(fps * score_interval_sec))

    # ------------------------------------------------------------------
    # Video writer
    # ------------------------------------------------------------------
    write_w = output_size[0] if output_size else frame_width
    write_h = output_size[1] if output_size else frame_height
    write_queue = None
    writer_thread = None
    video_writer = None
    output_video_path = None
    if save_video:
        output_video_path = Path(output_dir) / f"broadcast_{Path(video_path).stem}.mp4"
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

    # ------------------------------------------------------------------
    # OpenCV window + frame producer
    # ------------------------------------------------------------------
    cv2.namedWindow("Broadcast Data Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Broadcast Data Collector", 1280, 720)

    frame_queue = queue.Queue(maxsize=4)
    producer = threading.Thread(
        target=frame_producer_thread, args=(cap, frame_queue), daemon=True
    )
    producer.start()

    # ------------------------------------------------------------------
    # State variables
    # ------------------------------------------------------------------
    frame_idx = 0
    last_score_log = None
    paused = False
    playback_speed = 1.0
    frame_delay_base = 1.0 / fps if fps > 0 else 1.0 / 30.0
    last_displayed_frame = None
    sides_swapped = False
    score_freeze_until_frame = 0
    skipped_frames = 0

    # Prediction model state
    last_stable_scores_pred = (None, None)
    last_prediction = PredictionResult()
    pose_p1_latest = None
    pose_p2_latest = None
    start_time = time.time()

    print("\nBroadcast data collection started.")
    print("  Q=Quit  P=Pause  +/-=Speed  S=Screenshot  X=SwapSides  R=Re-mark corners")
    print("  F=Toggle scene skip  D=Discard rally  U=Undo discard")
    if score_mode == "manual":
        print("  1/2=+1 point  [/]=-1 point  3/4=+1 set  Z=swap scores")

    # ------------------------------------------------------------------
    # Keyboard handler (shared between paused and running states)
    # ------------------------------------------------------------------
    def handle_key(key):
        """Process a keypress. Returns 'quit', 'pause_toggle', or None."""
        nonlocal sides_swapped, score_freeze_until_frame, playback_speed, table_calibration

        if key == ord('q'):
            return 'quit'
        elif key == ord('p'):
            return 'pause_toggle'
        elif key == ord('f'):
            now_skipping = scene_gate.toggle()
            if not now_skipping:
                # Resuming — reset tracker so stale tracks are cleared
                _reset_ball_tracker(ball_tracker, last_displayed_frame)
            print(f"Scene skip: {'ON — inference paused' if now_skipping else 'OFF — collecting'}")
        elif key == ord('d'):
            rally_aggregator.mark_skipped()
            print(f"Rally #{rally_aggregator.rally_id} marked as SKIPPED")
        elif key == ord('u'):
            rally_aggregator.undo_skip()
            print(f"Rally #{rally_aggregator.rally_id} skip undone — collecting again")
        elif key == ord('x'):
            sides_swapped = not sides_swapped
            rois['player1_score'], rois['player2_score'] = (
                rois['player2_score'], rois['player1_score']
            )
            rois['player1_rounds'], rois['player2_rounds'] = (
                rois['player2_rounds'], rois['player1_rounds']
            )
            score_freeze_until_frame = frame_idx + int(fps * 8)
            print(f"Sides swapped: P1 now on {'right' if sides_swapped else 'left'}")
        elif key == ord('s') and last_displayed_frame is not None:
            p = Path(output_dir) / f"screenshot_{frame_idx}.jpg"
            cv2.imwrite(str(p), last_displayed_frame)
            print(f"Screenshot: {p}")
        elif key in [ord('+'), ord('=')]:
            playback_speed = min(4.0, playback_speed + 0.25)
            print(f"Playback speed: {playback_speed}x")
        elif key == ord('-'):
            playback_speed = max(0.25, playback_speed - 0.25)
            print(f"Playback speed: {playback_speed}x")
        elif key == ord('0'):
            playback_speed = 1.0
            print("Playback speed: 1.0x")
        elif key == ord('r'):
            print("Re-marking table corners...")
            result = interactive_frame_setup(last_displayed_frame)
            if result is not None:
                _, new_corners = result
                if len(new_corners) == 4:
                    tc = TableCalibration(new_corners)
                    if tc.is_valid():
                        table_calibration = tc
                        ball_tracker.set_table_calibration(tc)
                        rally_aggregator.set_table_calibration(tc)
                        print("Table corners updated!")
                    else:
                        print("Invalid corners, keeping previous.")
        elif score_mode == "manual":
            if key == ord('1'):
                score_detector.adjust('player1', +1)
                print(f"P1 score: {score_detector.current_scores['player1']}")
            elif key == ord('2'):
                score_detector.adjust('player2', +1)
                print(f"P2 score: {score_detector.current_scores['player2']}")
            elif key == ord('['):
                score_detector.adjust('player1', -1)
                print(f"P1 score: {score_detector.current_scores['player1']}")
            elif key == ord(']'):
                score_detector.adjust('player2', -1)
                print(f"P2 score: {score_detector.current_scores['player2']}")
            elif key == ord('z'):
                score_detector.swap_scores()
                print(f"Scores swapped — P1:{score_detector.current_scores['player1']} "
                      f"P2:{score_detector.current_scores['player2']}")
            elif key == ord('3'):
                score_detector.adjust_rounds('player1', +1)
                print(f"P1 sets: {score_detector.rounds['player1']}")
            elif key == ord('4'):
                score_detector.adjust_rounds('player2', +1)
                print(f"P2 sets: {score_detector.rounds['player2']}")
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    try:
        while True:
            loop_start = time.time()

            # ---- Pause ----
            if paused:
                if last_displayed_frame is not None:
                    display_frame = last_displayed_frame.copy()
                    _draw_scene_overlay(display_frame, scene_gate, skipped_frames)
                    _draw_rally_overlay(display_frame, rally_aggregator)
                    stats.draw(display_frame, frame_idx, total_frames, paused, playback_speed)
                    cv2.imshow("Broadcast Data Collector", display_frame)

                key = cv2.waitKey(30) & 0xFF
                action = handle_key(key)
                if action == 'quit':
                    break
                elif action == 'pause_toggle':
                    paused = False
                    print("Resumed")
                continue

            # ---- Get next frame ----
            item = frame_queue.get()
            if item is END_SENTINEL:
                rally_aggregator.flush_final(frame_idx, frame_idx / fps if fps > 0 else 0)
                break
            frame_idx, frame = item

            # ---- Scene gate: skip inference if active ----
            if not scene_gate.should_process:
                skipped_frames += 1
                _draw_scene_overlay(frame, scene_gate, skipped_frames)
                _draw_rally_overlay(frame, rally_aggregator)
                stats.draw(frame, frame_idx, total_frames, paused, playback_speed)
                cv2.imshow("Broadcast Data Collector", frame)
                last_displayed_frame = frame

                key = cv2.waitKey(1) & 0xFF
                action = handle_key(key)
                if action == 'quit':
                    break
                elif action == 'pause_toggle':
                    paused = True
                continue

            # ---- Ball detection & tracking ----
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
                if not rally_aggregator.skipped:
                    logger.log_trajectory(
                        frame_idx, track_id, cx, cy, speed,
                        x_m=x_m, y_m=y_m, speed_mps=speed_mps,
                    )
                tracks_with_meters.append((track_id, x_m, y_m, speed_mps))

            stats.update_ball_stats(ball_detected, max_speed, max_speed_mps)

            # ---- Pose extraction (skipped when rally is discarded) ----
            if frame_idx % pose_interval == 0 and not rally_aggregator.skipped:
                t0 = time.perf_counter()
                ts = frame_idx / fps
                for pid in (1, 2):
                    feat = pose_extractor.update(
                        frame, frame_idx, pid, ts, sides_swapped=sides_swapped
                    )
                    if feat is not None:
                        logger.log_pose(feat)
                        if pid == 1:
                            pose_p1_latest = feat
                        else:
                            pose_p2_latest = feat
                        lm_px = feat.get('debug_landmarks')
                        if lm_px:
                            color = (0, 255, 0) if pid == 1 else (255, 0, 0)
                            draw_pose_skeleton(frame, lm_px, color)
                stats.record_phase('pose_infer', time.perf_counter() - t0)

            # ---- Score detection ----
            if (score_mode == "auto"
                    and frame_idx % score_detect_interval == 0
                    and frame_idx >= score_freeze_until_frame):
                t0 = time.perf_counter()
                scores, rounds = score_detector.update_scores_and_rounds(frame, rois, frame_idx)
                stats.record_phase('score_infer', time.perf_counter() - t0)
            else:
                scores = score_detector.current_scores
                rounds = score_detector.rounds

            # ---- Rally aggregation ----
            rally_aggregator.add_frame(
                frame_idx, frame_idx / fps,
                tracks_with_meters,
                score_detector.stable_scores,
                rounds,
            )
            stats.rally_display = rally_aggregator.get_state_for_display()

            # ---- Win prediction ----
            if prediction_model is not None:
                current_stable_pred = (
                    score_detector.stable_scores.get('player1'),
                    score_detector.stable_scores.get('player2'),
                )
                packet = build_packet(
                    frame_idx, fps, ball_tracker, tracks, score_detector,
                    rally_aggregator, pose_p1_latest, pose_p2_latest, start_time,
                )
                if (
                    current_stable_pred[0] is not None
                    and current_stable_pred != last_stable_scores_pred
                    and last_stable_scores_pred[0] is not None
                ):
                    prediction_model.on_point_scored(packet)
                last_stable_scores_pred = current_stable_pred
                last_prediction = prediction_model.predict(packet)

            # ---- Drawing ----
            t0 = time.perf_counter()
            ball_tracker.draw_trajectories(frame, tracks)
            current_score_log = (
                scores['player1'], scores['player2'],
                rounds['player1'], rounds['player2'],
            )
            if current_score_log != last_score_log:
                logger.log_score(
                    frame_idx, frame_idx / fps, scores, rounds,
                    score_detector.score_roi_obscured,
                    score_detector.rounds_roi_obscured,
                )
                last_score_log = current_score_log
            score_detector.draw_scores(frame, rois, player_names=player_names)
            if prediction_model is not None:
                draw_prediction_overlay(frame, last_prediction, player_names=player_names)

            _draw_scene_overlay(frame, scene_gate, skipped_frames)
            _draw_rally_overlay(frame, rally_aggregator)

            stats.record_phase('draw', time.perf_counter() - t0)
            stats.update()

            # ---- Video writing ----
            if save_video:
                if async_write and write_queue is not None:
                    try:
                        wf = (cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR)
                              if output_size else frame)
                        write_queue.put_nowait((frame_idx, wf))
                    except queue.Full:
                        pass
                elif video_writer is not None:
                    wf = (cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR)
                          if output_size else frame)
                    video_writer.write(wf)

            # ---- Display ----
            stats.draw(frame, frame_idx, total_frames, paused, playback_speed)
            cv2.imshow("Broadcast Data Collector", frame)
            last_displayed_frame = frame

            processing_time = time.time() - loop_start
            wait_time = max(1, int((frame_delay_base / playback_speed - processing_time) * 1000))
            key = cv2.waitKey(wait_time) & 0xFF

            action = handle_key(key)
            if action == 'quit':
                break
            elif action == 'pause_toggle':
                paused = True
                print("Paused")

    finally:
        score_detector.stop()
        pose_extractor.close()
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
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Trajectory log : {logger.trajectory_file}")
    print(f"Score log      : {logger.score_file}")
    print(f"Rally log      : {logger.rally_file}")
    print(f"Pose log       : {logger.pose_file}")
    if save_video and output_video_path:
        print(f"Output video   : {output_video_path}")
    print(f"Skipped frames (scene gate) : {skipped_frames}")
    print(f"Max ball speed recorded     : {stats.max_speed:.0f} px/s")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Broadcast Data Collector — ITTF stream data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard controls (OpenCV window):
  Q          Quit
  P          Pause / Resume
  F          Toggle scene skip (pause / resume inference)
  D          Discard current rally (bad camera angle)
  U          Undo rally discard (resume collection)
  X          Swap player sides (pose + scores)
  R          Re-mark table corners
  S          Screenshot
  + / -      Speed up / slow down playback
  0          Reset playback speed
  1 / 2      +1 point P1 / P2  (manual score mode)
  [ / ]      -1 point P1 / P2  (manual score mode)
  3 / 4      +1 set P1 / P2    (manual score mode)
  Z          Swap score display (manual score mode)
""",
    )
    parser.add_argument("video", help="Path to broadcast video file")
    parser.add_argument("--output", "-o", default="broadcast_data",
                        help="Output directory (default: broadcast_data)")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to saved ROI config JSON")
    parser.add_argument("--no-video", action="store_true",
                        help="Do not save annotated output video")
    parser.add_argument("--ball-inference-size", metavar="WxH", default=None,
                        type=_parse_inference_size,
                        help="Resize frame before ball inference, e.g. 1280x720")
    parser.add_argument("--output-size", metavar="WxH", default=None,
                        type=_parse_inference_size,
                        help="Downscale saved video to WxH")
    parser.add_argument("--no-async-write", dest="async_write", action="store_false",
                        help="Disable background video-writer thread")
    parser.add_argument("--broadcast-model", default=None,
                        help="Path to fine-tuned broadcast ball model")
    parser.add_argument("--manual-scores", action="store_true",
                        help="Enter scores manually via keyboard (1/2/[/]/3/4/Z)")
    parser.add_argument("--score-interval", metavar="SEC", type=float, default=2.0,
                        help="Seconds between auto score detection (default: 2.0)")
    parser.add_argument("--initial-scores", metavar="P1,P2", default="0,0",
                        help="Starting scores, e.g. 3,5 (default: 0,0)")
    parser.add_argument("--initial-rounds", metavar="P1,P2", default="0,0",
                        help="Starting set counts, e.g. 1,2 (default: 0,0)")
    parser.add_argument("--player-names", metavar="NAME1,NAME2", default=None,
                        help="Player display names, e.g. Alice,Bob")
    parser.add_argument("--prediction-model", metavar="PATH", default=None,
                        help="Path to Python file containing a WinPredictionModel subclass "
                             "(e.g. xgb_win_predictor.py). Omit to disable win prediction.")
    parser.add_argument("--ittf-name1", metavar="NAME", default=None,
                        help="ITTF-format name of Player 1 for late fusion profile lookup, "
                             "e.g. 'CALDERANO Hugo'. Substring match is supported.")
    parser.add_argument("--ittf-name2", metavar="NAME", default=None,
                        help="ITTF-format name of Player 2 for late fusion profile lookup.")
    parser.add_argument("--sets-to-win", metavar="N", type=int, default=3,
                        help="Number of sets needed to win the match (default: 3, i.e. Best of 5).")

    parser.set_defaults(async_write=True)
    args = parser.parse_args()

    p1s, p2s = args.initial_scores.split(",")
    init_scores = {"player1": int(p1s), "player2": int(p2s)}
    p1r, p2r = args.initial_rounds.split(",")
    init_rounds = {"player1": int(p1r), "player2": int(p2r)}
    player_names = args.player_names.split(",") if args.player_names else None

    pred_model = None
    if args.prediction_model:
        from pathlib import Path as _Path
        pred_script = _Path(args.prediction_model).resolve()

        # LateFusionWinPredictor needs ITTF names at construction time —
        # load it directly rather than via load_prediction_model() so we can
        # pass constructor arguments.
        if pred_script.stem == "late_fusion_win_predictor":
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("late_fusion_win_predictor", str(pred_script))
            _mod  = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            ittf1 = args.ittf_name1 or (player_names[0] if player_names else "Player 1")
            ittf2 = args.ittf_name2 or (player_names[1] if player_names else "Player 2")
            pred_model = _mod.LateFusionWinPredictor(
                ittf_name1=ittf1,
                ittf_name2=ittf2,
                player_names=player_names,
            )
        else:
            pred_model = load_prediction_model(args.prediction_model)
            if player_names and hasattr(pred_model, '_player_names'):
                pred_model._player_names = player_names

    collect(
        video_path=args.video,
        output_dir=args.output,
        save_video=not args.no_video,
        inference_size=args.ball_inference_size,
        output_size=args.output_size,
        config_path=args.config,
        async_write=args.async_write,
        score_mode="manual" if args.manual_scores else "auto",
        initial_scores=init_scores,
        initial_rounds=init_rounds,
        player_names=player_names,
        broadcast_model=args.broadcast_model,
        score_interval_sec=args.score_interval,
        prediction_model=pred_model,
    )
