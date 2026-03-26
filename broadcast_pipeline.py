"""
Broadcast Pipeline for ITTF Stream Data Collection
====================================================
Wraps the same processing as ball_tracking_fast.py but adds:
- Scene change detection (auto-skips replays/cuts)
- Optical flow table corner tracking (handles camera sway)
- Broadcast-specific ball detection model support

Usage:
    python broadcast_pipeline.py match.mp4 --output match1_output --manual-scores
    python broadcast_pipeline.py match.mp4 --broadcast-model runs/detect/runs/ball_detector_broadcast/weights/best.pt
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

# Ensure imports work
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
    BALL_CONF_THRESHOLD,
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
from broadcast_utils.scene_classifier import SceneClassifier, GAMEPLAY, NON_GAMEPLAY
from broadcast_utils.table_tracker import TableTracker


def broadcast_run(
    video_path,
    output_dir="broadcast_output",
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
    table_color="blue",
    scene_filter=True,
    score_interval_sec=2,
):
    """
    Run the broadcast data collection pipeline.
    Same core loop as ball_tracking_fast.optimized_run but with
    scene classification and optical flow table tracking.
    """
    # Resolve ball model path
    ball_model_path = broadcast_model or resolve_model_path(BALL_MODEL_PATH)
    digit_model_path = resolve_model_path(DIGIT_MODEL_PATH)

    if not Path(ball_model_path).exists():
        print(f"Error: Ball model not found: {ball_model_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Interactive setup or load config
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

    # Initial table calibration
    table_calibration = None
    table_tracker = None
    if len(table_corners) == 4:
        table_calibration = TableCalibration(table_corners)
        if table_calibration.is_valid():
            table_tracker = TableTracker(table_corners)
        else:
            table_calibration = None

    # Scene classifier
    scene_classifier = SceneClassifier(table_color=table_color) if scene_filter else None

    print("\n" + "=" * 60)
    print("BROADCAST DATA COLLECTION PIPELINE")
    print("=" * 60)
    print(f"Video: {video_path} | {frame_width}x{frame_height} @ {fps} FPS")
    print(f"Ball model: {ball_model_path}")
    if inference_size:
        print(f"Ball inference size: {inference_size[0]}x{inference_size[1]}")
    print(f"Scene filter: {'ON (' + table_color + ')' if scene_filter else 'OFF'}")
    print(f"Table tracking: {'ON' if table_tracker else 'OFF (no corners marked)'}")
    print(f"Score mode: {score_mode.upper()}")

    print("\nLoading models...")
    ball_tracker = BallTracker(ball_model_path, fps, table_calibration=table_calibration, inference_size=inference_size)

    if score_mode == "manual":
        score_detector = ManualScoreTracker(initial_scores=initial_scores, initial_rounds=initial_rounds)
    else:
        score_detector = ScoreDetectorBatched(digit_model_path)

    logger = DataLogger(output_dir, with_meters=(table_calibration is not None))
    rally_aggregator = RallyAggregator(fps, logger, table_calibration)
    stats = RealtimeStats(fps)
    pose_extractor = PoseFeatureExtractor(fps)
    logger.save_config(rois, video_path, fps, table_corners=table_corners)
    print("  Models loaded!")

    pose_interval = max(1, int(fps / POSE_TARGET_FPS))
    score_detect_interval = max(1, int(fps * score_interval_sec))

    # Video writer setup
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

    cv2.namedWindow("Broadcast Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Broadcast Tracking", 1280, 720)

    # Frame producer
    frame_queue = queue.Queue(maxsize=4)
    producer = threading.Thread(target=frame_producer_thread, args=(cap, frame_queue), daemon=True)
    producer.start()

    frame_idx = 0
    last_score_log = None
    paused = False
    playback_speed = 1.0
    frame_delay_base = 1.0 / fps if fps > 0 else 1.0 / 30.0
    last_displayed_frame = None
    sides_swapped = False
    score_freeze_until_frame = 0
    skipped_frames = 0

    print("\nBroadcast tracking started.")
    print("  Q=Quit P=Pause +/-=Speed S=Screenshot X=SwapSides R=Re-mark table corners")
    if score_mode == "manual":
        print("  1/2=+1 point  [/]=-1 point  3/4=+1 set  Z=swap scores")

    try:
        while True:
            loop_start = time.time()

            # --- Pause handling (identical to ball_tracking_fast) ---
            if paused:
                if last_displayed_frame is not None:
                    display_frame = last_displayed_frame.copy()
                    # Draw scene state indicator
                    if scene_classifier:
                        _draw_scene_state(display_frame, scene_classifier.state, skipped_frames)
                    stats.draw(display_frame, frame_idx, total_frames, paused, playback_speed)
                    cv2.imshow("Broadcast Tracking", display_frame)

                key = cv2.waitKey(30) & 0xFF
                key = _handle_common_keys(
                    key, score_mode, score_detector, rois, frame_idx, fps, output_dir,
                    last_displayed_frame, sides_swapped, score_freeze_until_frame, playback_speed, paused
                )
                if key == "quit":
                    break
                elif key == "pause_toggle":
                    paused = False
                    print("Resumed")
                elif isinstance(key, dict):
                    sides_swapped = key.get("sides_swapped", sides_swapped)
                    score_freeze_until_frame = key.get("score_freeze_until_frame", score_freeze_until_frame)
                    playback_speed = key.get("playback_speed", playback_speed)
                continue

            # --- Get next frame ---
            item = frame_queue.get()
            if item is END_SENTINEL:
                rally_aggregator.flush_final(frame_idx, frame_idx / fps if fps > 0 else 0)
                break
            frame_idx, frame = item

            # --- Scene classification ---
            if scene_classifier:
                scene_state = scene_classifier.update(frame)

                if scene_classifier.is_cut:
                    # Reset SORT tracker on scene transition
                    ball_tracker.tracker = Sort(
                        max_age=TRACKER_MAX_AGE,
                        min_hits=TRACKER_MIN_HITS,
                        iou_threshold=TRACKER_IOU_THRESHOLD,
                    )
                    ball_tracker.trajectories.clear()
                    ball_tracker.speed_history.clear()
                    ball_tracker.trajectories_meters.clear()
                    ball_tracker.speed_history_mps.clear()

                    # Re-initialize table tracker with last good corners
                    if table_tracker:
                        table_tracker.reinitialize(table_tracker.smoothed_corners, frame)

                if not scene_classifier.should_process:
                    skipped_frames += 1
                    # Draw skip indicator and still show frame
                    _draw_scene_state(frame, scene_state, skipped_frames)
                    stats.draw(frame, frame_idx, total_frames, paused, playback_speed)
                    cv2.imshow("Broadcast Tracking", frame)
                    last_displayed_frame = frame

                    # Still handle keyboard in skipped frames
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = True
                    continue

            # --- Table corner tracking ---
            if table_tracker:
                new_corners = table_tracker.update(frame)
                if table_tracker.is_valid:
                    tc = TableCalibration(new_corners)
                    if tc.is_valid():
                        ball_tracker.set_table_calibration(tc)
                        rally_aggregator.set_table_calibration(tc)

            # --- Ball detection & tracking ---
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
                        lm_px = feat.get('debug_landmarks')
                        if lm_px:
                            color = (0, 255, 0) if pid == 1 else (255, 0, 0)
                            draw_pose_skeleton(frame, lm_px, color)
                stats.record_phase('pose_infer', time.perf_counter() - t0)

            # --- Score detection ---
            if score_mode == "auto" and frame_idx % score_detect_interval == 0 and frame_idx >= score_freeze_until_frame:
                t0 = time.perf_counter()
                scores, rounds = score_detector.update_scores_and_rounds(frame, rois, frame_idx)
                stats.record_phase('score_infer', time.perf_counter() - t0)
            else:
                scores = score_detector.current_scores
                rounds = score_detector.rounds

            # --- Rally aggregation ---
            rally_aggregator.add_frame(frame_idx, frame_idx / fps, tracks_with_meters, score_detector.stable_scores, rounds)
            stats.rally_display = rally_aggregator.get_state_for_display()

            # --- Drawing ---
            t0 = time.perf_counter()
            ball_tracker.draw_trajectories(frame, tracks)
            current_score_log = (scores['player1'], scores['player2'], rounds['player1'], rounds['player2'])
            if current_score_log != last_score_log:
                logger.log_score(frame_idx, frame_idx / fps, scores, rounds, score_detector.score_roi_obscured, score_detector.rounds_roi_obscured)
                last_score_log = current_score_log
            score_detector.draw_scores(frame, rois, player_names=player_names)

            # Draw table corners overlay
            if table_tracker:
                _draw_table_overlay(frame, table_tracker)

            # Draw scene state
            if scene_classifier:
                _draw_scene_state(frame, scene_classifier.state, skipped_frames)

            stats.record_phase('draw', time.perf_counter() - t0)
            stats.update()

            # --- Video writing ---
            if save_video:
                if async_write and write_queue is not None:
                    try:
                        wf = cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR) if output_size else frame
                        write_queue.put_nowait((frame_idx, wf))
                    except queue.Full:
                        pass
                elif video_writer is not None:
                    wf = cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR) if output_size else frame
                    video_writer.write(wf)

            # --- Display ---
            stats.draw(frame, frame_idx, total_frames, paused, playback_speed)
            cv2.imshow("Broadcast Tracking", frame)
            last_displayed_frame = frame

            processing_time = time.time() - loop_start
            wait_time = max(1, int((frame_delay_base / playback_speed - processing_time) * 1000))
            key = cv2.waitKey(wait_time) & 0xFF

            # --- Keyboard handling ---
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
            elif key == ord('r'):
                # Re-mark table corners interactively
                print("Re-marking table corners... (press on the current frame)")
                result = interactive_frame_setup(frame)
                if result is not None:
                    _, new_corners = result
                    if len(new_corners) == 4:
                        tc = TableCalibration(new_corners)
                        if tc.is_valid():
                            ball_tracker.set_table_calibration(tc)
                            rally_aggregator.set_table_calibration(tc)
                            if table_tracker:
                                table_tracker.reinitialize(new_corners, frame)
                            else:
                                table_tracker = TableTracker(new_corners)
                                table_tracker.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            print("Table corners updated!")
                        else:
                            print("Invalid corners, keeping previous.")
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
    print("BROADCAST PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Trajectory log: {logger.trajectory_file}")
    print(f"Score log: {logger.score_file}")
    print(f"Rally log: {logger.rally_file}")
    print(f"Pose log: {logger.pose_file}")
    if save_video and output_video_path:
        print(f"Output video: {output_video_path}")
    print(f"Skipped frames (non-gameplay): {skipped_frames}")
    print(f"Max ball speed recorded: {stats.max_speed:.0f} px/s")


# =============================================================================
# DRAWING HELPERS
# =============================================================================
def _draw_table_overlay(frame, table_tracker):
    """Draw tracked table corners on frame."""
    corners = table_tracker.get_corners_int()
    color = (0, 255, 0) if table_tracker.is_valid else (0, 0, 255)
    cv2.polylines(frame, [corners.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)
    for i, (x, y) in enumerate(corners):
        label = ["TL", "TR", "BR", "BL"][i]
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.putText(frame, label, (x + 8, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _draw_scene_state(frame, state, skipped_count):
    """Draw scene classification state indicator."""
    h, w = frame.shape[:2]
    if state == GAMEPLAY:
        color = (0, 200, 0)
        text = "GAMEPLAY"
    elif state == NON_GAMEPLAY:
        color = (0, 0, 200)
        text = f"NON-GAMEPLAY (skipped: {skipped_count})"
    else:
        color = (0, 200, 200)
        text = "CUT DETECTED"
    cv2.putText(frame, text, (w - 350, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _handle_common_keys(key, score_mode, score_detector, rois, frame_idx, fps, output_dir,
                        last_frame, sides_swapped, score_freeze_until_frame, playback_speed, paused):
    """Handle keyboard input during pause. Returns action string or dict of state updates."""
    if key == ord('q'):
        return "quit"
    elif key == ord('p'):
        return "pause_toggle"
    elif key == ord('s') and last_frame is not None:
        screenshot_path = Path(output_dir) / f"screenshot_{frame_idx}.jpg"
        cv2.imwrite(str(screenshot_path), last_frame)
        print(f"Screenshot saved: {screenshot_path}")
    elif key == ord('x'):
        sides_swapped = not sides_swapped
        rois['player1_score'], rois['player2_score'] = rois['player2_score'], rois['player1_score']
        rois['player1_rounds'], rois['player2_rounds'] = rois['player2_rounds'], rois['player1_rounds']
        score_freeze_until_frame = frame_idx + int(fps * 8)
        print(f"Sides swapped: Player 1 now on {'right' if sides_swapped else 'left'}")
        return {"sides_swapped": sides_swapped, "score_freeze_until_frame": score_freeze_until_frame}
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
    return None


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Broadcast Pipeline — ITTF stream data collection with scene filtering and table tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Path to broadcast video file")
    parser.add_argument("--output", "-o", default="broadcast_output", help="Output directory")
    parser.add_argument("--config", "-c", default=None, help="Path to saved config JSON")
    parser.add_argument("--no-video", action="store_true", help="Don't save output video")
    parser.add_argument("--ball-inference-size", metavar="WxH", default=None, type=_parse_inference_size,
                        help="Resize to WxH before ball inference (e.g. 640x360)")
    parser.add_argument("--output-size", metavar="WxH", default=None, type=_parse_inference_size,
                        help="Downscale saved video to WxH")
    parser.add_argument("--no-async-write", dest="async_write", action="store_false",
                        help="Disable background video writing thread")
    parser.add_argument("--broadcast-model", default=None,
                        help="Path to fine-tuned broadcast ball model (default: use original)")
    parser.add_argument("--table-color", choices=["blue", "green"], default="blue",
                        help="Table color for scene classification (default: blue)")
    parser.add_argument("--no-scene-filter", dest="scene_filter", action="store_false",
                        help="Disable scene change detection")
    parser.add_argument("--manual-scores", action="store_true",
                        help="Enter scores manually (1/2/[/]/3/4/Z keys)")
    parser.add_argument("--score-interval", metavar="SEC", type=float, default=2.0,
                        help="Seconds between auto score detection (default: 2.0)")
    parser.add_argument("--initial-scores", metavar="P1,P2", default="0,0",
                        help="Starting scores (default: 0,0)")
    parser.add_argument("--initial-rounds", metavar="P1,P2", default="0,0",
                        help="Starting set counts (default: 0,0)")
    parser.add_argument("--player-names", metavar="NAME1,NAME2", default=None,
                        help="Player display names, e.g. 'Alice,Bob'")

    parser.set_defaults(async_write=True, scene_filter=True)
    args = parser.parse_args()

    # Parse initial scores/rounds
    p1s, p2s = args.initial_scores.split(",")
    init_scores = {"player1": int(p1s), "player2": int(p2s)}
    p1r, p2r = args.initial_rounds.split(",")
    init_rounds = {"player1": int(p1r), "player2": int(p2r)}
    player_names = args.player_names.split(",") if args.player_names else None

    broadcast_run(
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
        table_color=args.table_color,
        scene_filter=args.scene_filter,
        score_interval_sec=args.score_interval,
    )
