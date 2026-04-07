"""
Prediction Pipeline — Real-time CV + Win Prediction
====================================================
Runs the CV pipeline on a video file, live camera, or RTSP stream,
packages extracted features into PredictionDataPacket each frame,
sends them to a WinPredictionModel, and overlays the prediction on
the video feed.

Usage examples:
    # Pre-downloaded video
    python prediction_pipeline.py --source path/to/video.mp4

    # Live camera (0 = default webcam)
    python prediction_pipeline.py --source 0

    # RTSP stream
    python prediction_pipeline.py --source rtsp://...

    # With partner's trained model
    python prediction_pipeline.py --source video.mp4 --model path/to/my_model.py

    # Manual scores (no OCR)
    python prediction_pipeline.py --source video.mp4 --manual-scores

    # Load saved config (skip interactive setup)
    python prediction_pipeline.py --source video.mp4 --config config.json
"""

import cv2
import numpy as np
import sys
import os
import time
import json
import queue
import threading
import argparse
import importlib.util
from pathlib import Path
from collections import deque

# Ensure project root is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, 'sort'))

import ball_tracking_analysis as bta
from ball_tracking_analysis import (
    resolve_model_path,
    BALL_MODEL_PATH, DIGIT_MODEL_PATH,
    BALL_CONF_THRESHOLD,
    TableCalibration, BallTracker, RallyAggregator, DataLogger,
    RealtimeStats, PoseFeatureExtractor,
    interactive_frame_setup,
    POSE_TARGET_FPS, _MEDIAPIPE_AVAILABLE,
    RALLY_BALL_SEEN_FRAMES,
)
from ball_tracking_fast import (
    frame_producer_thread, video_writer_thread,
    ScoreDetectorBatched, ManualScoreTracker,
    load_config, draw_pose_skeleton, END_SENTINEL,
)
from prediction_model_base import (
    BallState, ScoreState, RallyState, PoseState,
    PredictionDataPacket, PredictionResult,
    WinPredictionModel, DummyPredictionModel,
)


# =============================================================================
# SOURCE PARSING (video file / camera index / RTSP URL)
# =============================================================================

def parse_source(source_str):
    """
    Return (source, is_live).
    - Integer string ("0", "1") → camera index, is_live=True
    - rtsp:// or http:// URL → stream URL, is_live=True
    - Anything else → file path, is_live=False
    """
    try:
        idx = int(source_str)
        return idx, True
    except ValueError:
        pass
    if source_str.startswith(("rtsp://", "http://", "https://")):
        return source_str, True
    return source_str, False


# =============================================================================
# DYNAMIC MODEL LOADING
# =============================================================================

def load_prediction_model(model_path):
    """
    Dynamically import a Python file and return the first WinPredictionModel
    subclass instance found. The file must define a class that inherits from
    WinPredictionModel.
    """
    path = Path(model_path).resolve()
    if not path.exists():
        print(f"Error: model file not found: {path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("user_model", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the first concrete WinPredictionModel subclass
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if (isinstance(obj, type)
                and issubclass(obj, WinPredictionModel)
                and obj is not WinPredictionModel):
            print(f"Loaded prediction model: {obj.__name__} from {path}")
            return obj()

    print(f"Error: no WinPredictionModel subclass found in {path}")
    sys.exit(1)


# =============================================================================
# BUILD DATA PACKET
# =============================================================================

def build_packet(
    frame_idx, fps, ball_tracker, tracks, score_detector,
    rally_aggregator, pose_p1, pose_p2, start_time,
    sets_to_win=3,
):
    """Assemble a PredictionDataPacket from current pipeline state."""
    timestamp_sec = frame_idx / fps if fps > 0 else 0.0

    # --- Ball state ---
    ball_detected = len(tracks) > 0
    pos_px = None
    pos_m = None
    speed_mps = None
    speed_pps = None
    trajectory = []

    if ball_detected:
        # Use the first (highest-confidence) track
        x1, y1, x2, y2, tid = tracks[0]
        tid = int(tid)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pos_px = (cx, cy)
        pos_m_vals = ball_tracker.get_position_meters(tid)
        if pos_m_vals[0] is not None:
            pos_m = pos_m_vals
        speed_mps = ball_tracker.get_smoothed_speed_mps(tid)
        speed_pps = ball_tracker.get_smoothed_speed(tid)
        if tid in ball_tracker.trajectories:
            trajectory = list(ball_tracker.trajectories[tid])

    ball = BallState(
        detected=ball_detected,
        position_px=pos_px,
        position_m=pos_m,
        speed_mps=speed_mps,
        speed_pps=speed_pps,
        trajectory_px=trajectory,
    )

    # --- Score state ---
    scores = score_detector.current_scores
    rounds = score_detector.rounds
    obscured = getattr(score_detector, 'score_roi_obscured', {})
    score = ScoreState(
        player1_score=scores.get('player1'),
        player2_score=scores.get('player2'),
        player1_sets=rounds.get('player1', 0),
        player2_sets=rounds.get('player2', 0),
        sets_to_win=sets_to_win,
        score_reliable=not (obscured.get('player1', False) or obscured.get('player2', False)),
    )

    # --- Rally state ---
    rd = rally_aggregator.get_state_for_display()
    rally_active = rd.get('state') == RallyAggregator.STATE_RALLY_ACTIVE

    # Compute rally duration
    rally_duration = 0.0
    if rally_active and rally_aggregator.rally_start_time is not None:
        rally_duration = timestamp_sec - rally_aggregator.rally_start_time

    # Compute live speed stats from samples
    mean_spd = None
    max_spd = None
    zones = [0] * 9
    if rally_aggregator.samples:
        speeds = [s[3] for s in rally_aggregator.samples if s[3] and s[3] > 0]
        if speeds:
            mean_spd = float(np.mean(speeds))
            max_spd = float(max(speeds))
        for _, xm, ym, _ in rally_aggregator.samples:
            idx = rally_aggregator._landing_zone_index(xm, ym)
            if idx is not None:
                zones[idx] += 1

    rally = RallyState(
        is_active=rally_active,
        rally_id=rally_aggregator.rally_id,
        rally_duration_sec=rally_duration,
        mean_speed_mps=mean_spd,
        max_speed_mps=max_spd,
        landing_zones=zones,
    )

    # --- Pose state ---
    # Strip debug_landmarks before sending to model
    p1_clean = {k: v for k, v in pose_p1.items() if k != 'debug_landmarks'} if pose_p1 else None
    p2_clean = {k: v for k, v in pose_p2.items() if k != 'debug_landmarks'} if pose_p2 else None
    pose = PoseState(player1=p1_clean, player2=p2_clean)

    return PredictionDataPacket(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        ball=ball,
        score=score,
        rally=rally,
        pose=pose,
        match_elapsed_sec=time.time() - start_time,
    )


# =============================================================================
# PREDICTION OVERLAY
# =============================================================================

def draw_prediction_overlay(frame, result, player_names=None):
    """Draw win probability bars at the top of the frame."""
    h, w = frame.shape[:2]
    p1_name = (player_names[0] if player_names else None) or "Player 1"
    p2_name = (player_names[1] if player_names else None) or "Player 2"

    bar_y = 110  # below the score boxes
    bar_h = 30
    bar_margin = 20
    bar_w = w - 2 * bar_margin

    # Background
    cv2.rectangle(frame, (bar_margin, bar_y), (bar_margin + bar_w, bar_y + bar_h + 20), (15, 15, 15), -1)

    if not result.model_ready:
        cv2.putText(frame, "Prediction: Model warming up...",
                    (bar_margin + 10, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        return

    p1_prob = max(0.0, min(1.0, result.player1_win_prob))
    p2_prob = 1.0 - p1_prob

    # Divider position
    split_x = bar_margin + int(bar_w * p1_prob)

    # Player 1 bar (green)
    if p1_prob > 0.01:
        cv2.rectangle(frame, (bar_margin, bar_y + 4), (split_x, bar_y + bar_h + 4), (0, 180, 0), -1)

    # Player 2 bar (blue)
    if p2_prob > 0.01:
        cv2.rectangle(frame, (split_x, bar_y + 4), (bar_margin + bar_w, bar_y + bar_h + 4), (180, 60, 0), -1)

    # Labels
    p1_pct = f"{p1_prob * 100:.0f}%"
    p2_pct = f"{p2_prob * 100:.0f}%"

    cv2.putText(frame, f"{p1_name} {p1_pct}",
                (bar_margin + 5, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    p2_text = f"{p2_pct} {p2_name}"
    text_size = cv2.getTextSize(p2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    cv2.putText(frame, p2_text,
                (bar_margin + bar_w - text_size[0] - 5, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Confidence indicator
    conf_text = f"conf: {result.confidence:.0%}"
    cv2.putText(frame, conf_text,
                (w // 2 - 30, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_prediction_pipeline(
    source,
    is_live=False,
    output_dir="prediction_output",
    save_video=True,
    inference_size=None,
    score_interval_sec=2.0,
    output_size=None,
    config_path=None,
    score_mode="auto",
    initial_scores=None,
    initial_rounds=None,
    player_names=None,
    prediction_model=None,
    predict_interval=1,
    no_display=False,
):
    """
    Run the full CV + prediction pipeline.

    Parameters
    ----------
    source : str or int
        Video file path, camera index, or stream URL.
    is_live : bool
        True for camera/stream sources (affects end-of-stream handling).
    prediction_model : WinPredictionModel or None
        If None, uses DummyPredictionModel.
    predict_interval : int
        Call model.predict() every N frames (default: every frame).
    no_display : bool
        If True, skip cv2.imshow (headless mode for testing).
    """
    if prediction_model is None:
        prediction_model = DummyPredictionModel()

    ball_model_path = resolve_model_path(BALL_MODEL_PATH)
    digit_model_path = resolve_model_path(DIGIT_MODEL_PATH)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Setup ROIs and table calibration ---
    if config_path:
        rois, table_corners = load_config(config_path)
        table_calibration = None
        if len(table_corners) == 4:
            tc = TableCalibration(table_corners)
            table_calibration = tc if tc.is_valid() else None
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
            tc = TableCalibration(table_corners)
            table_calibration = tc if tc.is_valid() else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Initialize components ---
    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Source: {source} | {frame_width}x{frame_height} @ {fps:.1f} FPS")
    print(f"Live: {is_live} | Display: {not no_display}")
    print(f"Prediction model: {type(prediction_model).__name__}")
    print(f"Predict interval: every {predict_interval} frame(s)")

    print("\nLoading models...")
    ball_tracker = BallTracker(ball_model_path, fps, table_calibration=table_calibration, inference_size=inference_size)

    if score_mode == "manual":
        score_detector = ManualScoreTracker(initial_scores=initial_scores, initial_rounds=initial_rounds)
        print("Score mode: MANUAL")
    else:
        score_detector = ScoreDetectorBatched(digit_model_path)
        print("Score mode: AUTO (OCR)")

    logger = DataLogger(output_dir, with_meters=(table_calibration is not None and table_calibration.is_valid()))
    rally_aggregator = RallyAggregator(fps, logger, table_calibration)
    stats = RealtimeStats(fps)
    pose_extractor = PoseFeatureExtractor(fps)
    logger.save_config(rois, str(source), fps, table_corners=table_corners)
    print("Models loaded!")

    pose_interval = max(1, int(fps / POSE_TARGET_FPS))
    score_detect_interval = max(1, int(fps * score_interval_sec))

    # --- Video writer ---
    write_w = output_size[0] if output_size else frame_width
    write_h = output_size[1] if output_size else frame_height
    write_queue_obj = None
    writer_thread_obj = None
    output_video_path = None

    if save_video:
        output_video_path = Path(output_dir) / f"prediction_{Path(str(source)).stem if not is_live else 'live'}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        write_queue_obj = queue.Queue(maxsize=8)
        writer_thread_obj = threading.Thread(
            target=video_writer_thread,
            args=(write_queue_obj, output_video_path, fourcc, fps, write_w, write_h, output_size),
        )
        writer_thread_obj.start()
        print(f"Output video: {output_video_path}")

    # --- Display window ---
    if not no_display:
        cv2.namedWindow("Prediction Pipeline", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Prediction Pipeline", 1280, 720)

    # --- Frame producer ---
    frame_queue = queue.Queue(maxsize=4)
    producer = threading.Thread(target=frame_producer_thread, args=(cap, frame_queue), daemon=True)
    producer.start()

    # --- Main loop ---
    frame_idx = 0
    last_score_log = None
    last_prediction = PredictionResult()
    last_stable_scores = (None, None)
    pose_p1_latest = None
    pose_p2_latest = None
    paused = False
    playback_speed = 1.0
    frame_delay_base = 1.0 / fps if fps > 0 else 1.0 / 30.0
    last_displayed_frame = None
    sides_swapped = False
    score_freeze_until_frame = 0
    start_time = time.time()

    prediction_model.reset()

    print("\nPipeline running. Q=Quit P=Pause +/-=Speed S=Screenshot X=SwapSides")
    if score_mode == "manual":
        print("Manual score keys: 1/2=+1 point  [/]=-1 point  3/4=+1 set  Z=swap scores")

    try:
        while True:
            loop_start = time.time()

            # --- Handle pause ---
            if paused:
                if last_displayed_frame is not None and not no_display:
                    cv2.imshow("Prediction Pipeline", last_displayed_frame)
                key = cv2.waitKey(30) & 0xFF if not no_display else 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = False
                elif key == ord('1') and score_mode == "manual":
                    score_detector.adjust('player1', +1)
                elif key == ord('2') and score_mode == "manual":
                    score_detector.adjust('player2', +1)
                elif key == ord('[') and score_mode == "manual":
                    score_detector.adjust('player1', -1)
                elif key == ord(']') and score_mode == "manual":
                    score_detector.adjust('player2', -1)
                elif key == ord('z') and score_mode == "manual":
                    score_detector.swap_scores()
                elif key == ord('3') and score_mode == "manual":
                    score_detector.adjust_rounds('player1', +1)
                elif key == ord('4') and score_mode == "manual":
                    score_detector.adjust_rounds('player2', +1)
                continue

            # --- Get next frame ---
            item = frame_queue.get()
            if item is END_SENTINEL:
                rally_aggregator.flush_final(frame_idx, frame_idx / fps if fps > 0 else 0)
                break
            frame_idx, frame = item

            # --- Ball detection + tracking ---
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

            # --- Pose extraction ---
            if frame_idx % pose_interval == 0:
                t0 = time.perf_counter()
                ts = frame_idx / fps
                for pid in (1, 2):
                    feat = pose_extractor.update(frame, frame_idx, pid, ts, sides_swapped=sides_swapped)
                    if feat is not None:
                        logger.log_pose(feat)
                        if pid == 1:
                            pose_p1_latest = feat
                        else:
                            pose_p2_latest = feat
                        # Debug skeleton overlay
                        lm_px = feat.get('debug_landmarks')
                        if lm_px:
                            color = (0, 255, 0) if pid == 1 else (255, 0, 0)
                            draw_pose_skeleton(frame, lm_px, color)
                stats.record_phase('pose_infer', time.perf_counter() - t0)

            # --- Score detection ---
            if score_mode == "auto" and frame_idx % score_detect_interval == 0 and frame_idx >= score_freeze_until_frame:
                t0 = time.perf_counter()
                score_detector.update_scores_and_rounds(frame, rois, frame_idx)
                stats.record_phase('score_infer', time.perf_counter() - t0)

            # --- Rally aggregation ---
            rally_aggregator.add_frame(frame_idx, frame_idx / fps, tracks_with_meters,
                                       score_detector.stable_scores, score_detector.rounds)
            stats.rally_display = rally_aggregator.get_state_for_display()

            # --- Check for point scored (for model hook) ---
            current_stable = (
                score_detector.stable_scores.get('player1'),
                score_detector.stable_scores.get('player2'),
            )
            if (current_stable[0] is not None and current_stable != last_stable_scores
                    and last_stable_scores[0] is not None):
                packet = build_packet(
                    frame_idx, fps, ball_tracker, tracks, score_detector,
                    rally_aggregator, pose_p1_latest, pose_p2_latest, start_time,
                )
                prediction_model.on_point_scored(packet)
            last_stable_scores = current_stable

            # --- Prediction ---
            if frame_idx % predict_interval == 0:
                packet = build_packet(
                    frame_idx, fps, ball_tracker, tracks, score_detector,
                    rally_aggregator, pose_p1_latest, pose_p2_latest, start_time,
                )
                last_prediction = prediction_model.predict(packet)

            # --- Draw overlays ---
            ball_tracker.draw_trajectories(frame, tracks)
            current_score_log = (
                score_detector.current_scores['player1'],
                score_detector.current_scores['player2'],
                score_detector.rounds['player1'],
                score_detector.rounds['player2'],
            )
            if current_score_log != last_score_log:
                logger.log_score(
                    frame_idx, frame_idx / fps,
                    score_detector.current_scores, score_detector.rounds,
                    score_detector.score_roi_obscured, score_detector.rounds_roi_obscured,
                )
                last_score_log = current_score_log

            score_detector.draw_scores(frame, rois, player_names=player_names)
            draw_prediction_overlay(frame, last_prediction, player_names=player_names)

            stats.update()
            stats.draw(frame, frame_idx, total_frames, paused, playback_speed)

            # --- Write output video ---
            if save_video and write_queue_obj is not None:
                try:
                    wf = cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR) if output_size else frame
                    write_queue_obj.put_nowait((frame_idx, wf))
                except queue.Full:
                    pass

            # --- Display ---
            if not no_display:
                cv2.imshow("Prediction Pipeline", frame)
                last_displayed_frame = frame

            # --- Key handling ---
            processing_time = time.time() - loop_start
            if is_live:
                wait_time = 1
            else:
                wait_time = max(1, int((frame_delay_base / playback_speed - processing_time) * 1000))

            key = cv2.waitKey(wait_time) & 0xFF if not no_display else 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = True
            elif key in [ord('+'), ord('=')]:
                playback_speed = min(4.0, playback_speed + 0.25)
            elif key == ord('-'):
                playback_speed = max(0.25, playback_speed - 0.25)
            elif key == ord('0'):
                playback_speed = 1.0
            elif key == ord('s'):
                sp = Path(output_dir) / f"screenshot_{frame_idx}.jpg"
                cv2.imwrite(str(sp), frame)
                print(f"Screenshot: {sp}")
            elif key == ord('x'):
                sides_swapped = not sides_swapped
                rois['player1_score'], rois['player2_score'] = rois['player2_score'], rois['player1_score']
                rois['player1_rounds'], rois['player2_rounds'] = rois['player2_rounds'], rois['player1_rounds']
                score_freeze_until_frame = frame_idx + int(fps * 8)
            elif key == ord('1') and score_mode == "manual":
                score_detector.adjust('player1', +1)
            elif key == ord('2') and score_mode == "manual":
                score_detector.adjust('player2', +1)
            elif key == ord('[') and score_mode == "manual":
                score_detector.adjust('player1', -1)
            elif key == ord(']') and score_mode == "manual":
                score_detector.adjust('player2', -1)
            elif key == ord('z') and score_mode == "manual":
                score_detector.swap_scores()
            elif key == ord('3') and score_mode == "manual":
                score_detector.adjust_rounds('player1', +1)
            elif key == ord('4') and score_mode == "manual":
                score_detector.adjust_rounds('player2', +1)

    finally:
        score_detector.stop()
        pose_extractor.close()
        # Drain producer
        try:
            while True:
                frame_queue.get_nowait()
        except queue.Empty:
            pass
        producer.join(timeout=3.0)
        if write_queue_obj is not None and writer_thread_obj is not None:
            write_queue_obj.put(END_SENTINEL)
            writer_thread_obj.join(timeout=10.0)
        cap.release()
        if not no_display:
            cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Trajectory log: {logger.trajectory_file}")
    print(f"Score log:      {logger.score_file}")
    print(f"Rally log:      {logger.rally_file}")
    print(f"Pose log:       {logger.pose_file}")
    if save_video and output_video_path:
        print(f"Output video:   {output_video_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def _parse_inference_size(s):
    try:
        w, h = s.lower().split('x')
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid size '{s}'. Expected WxH e.g. 1280x720")


def _parse_two_ints(s, default=(0, 0)):
    try:
        a, b = s.split(',')
        return int(a.strip()), int(b.strip())
    except Exception:
        return default


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediction Pipeline: CV features → win prediction → live overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", "-s", required=True,
                        help="Video file path, camera index (0, 1, ...), or RTSP/HTTP URL")
    parser.add_argument("--model", "-m", default=None,
                        help="Path to Python file containing a WinPredictionModel subclass")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to saved config JSON (skip interactive setup)")
    parser.add_argument("--output", "-o", default="prediction_output",
                        help="Output directory (default: prediction_output)")
    parser.add_argument("--no-video", action="store_true",
                        help="Don't save output video")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — no cv2.imshow window")
    parser.add_argument("--ball-inference-size", metavar="WxH", default=None,
                        type=_parse_inference_size,
                        help="Resize frame before ball inference (e.g. 640x360)")
    parser.add_argument("--output-size", metavar="WxH", default=None,
                        type=_parse_inference_size,
                        help="Downscale saved video to WxH")
    parser.add_argument("--score-interval", metavar="SEC", type=float, default=2.0,
                        help="Seconds between score detector runs (default: 2.0)")
    parser.add_argument("--manual-scores", action="store_true",
                        help="Disable OCR; enter scores manually with keyboard")
    parser.add_argument("--initial-scores", metavar="P1,P2", default="0,0",
                        help="Starting scores for manual mode (default: 0,0)")
    parser.add_argument("--initial-rounds", metavar="P1,P2", default="0,0",
                        help="Starting set counts for manual mode (default: 0,0)")
    parser.add_argument("--player-names", metavar="NAME1,NAME2", default=None,
                        help="Player display names, e.g. 'Alice,Bob'")
    parser.add_argument("--predict-interval", metavar="N", type=int, default=1,
                        help="Run prediction model every N frames (default: 1)")

    args = parser.parse_args()

    source, is_live = parse_source(args.source)
    inf_size = args.ball_inference_size or getattr(bta, 'BALL_INFERENCE_SIZE', None)

    # Load prediction model
    if args.model:
        pred_model = load_prediction_model(args.model)
    else:
        pred_model = DummyPredictionModel()
        print("No --model specified; using DummyPredictionModel (50/50)")

    p1s, p2s = _parse_two_ints(args.initial_scores)
    p1r, p2r = _parse_two_ints(args.initial_rounds)
    player_names = None
    if args.player_names:
        parts = args.player_names.split(',', 1)
        player_names = [p.strip() for p in parts] if len(parts) == 2 else None

    run_prediction_pipeline(
        source=source,
        is_live=is_live,
        output_dir=args.output,
        save_video=not args.no_video,
        inference_size=inf_size,
        score_interval_sec=args.score_interval,
        output_size=args.output_size,
        config_path=args.config,
        score_mode="manual" if args.manual_scores else "auto",
        initial_scores={'player1': p1s, 'player2': p2s},
        initial_rounds={'player1': p1r, 'player2': p2r},
        player_names=player_names,
        prediction_model=pred_model,
        predict_interval=args.predict_interval,
        no_display=args.no_display,
    )
