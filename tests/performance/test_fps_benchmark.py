"""
Performance tests for the broadcast pipeline.

Weight-dependent tests are skipped when YOLO model files are not present.
Memory-bound tests run unconditionally.
"""

import time
import pytest
import numpy as np
from pathlib import Path
from collections import deque

from ball_tracking_analysis import (
    TRAJECTORY_LENGTH, SPEED_SMOOTHING_WINDOW,
    resolve_model_path, BALL_MODEL_PATH, DIGIT_MODEL_PATH,
)


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _ball_weights_available():
    path = resolve_model_path(BALL_MODEL_PATH)
    return Path(path).exists()


def _digit_weights_available():
    path = resolve_model_path(DIGIT_MODEL_PATH)
    return Path(path).exists()


# ---------------------------------------------------------------------------
# Memory stability (no weights needed)
# ---------------------------------------------------------------------------

class TestMemoryStability:

    def test_trajectory_deque_bounded(self):
        """Trajectory deque with maxlen=TRAJECTORY_LENGTH never exceeds the cap."""
        traj = deque(maxlen=TRAJECTORY_LENGTH)
        for i in range(TRAJECTORY_LENGTH + 50):
            traj.append((float(i), float(i)))
        assert len(traj) == TRAJECTORY_LENGTH, (
            f"Trajectory deque grew to {len(traj)}, expected {TRAJECTORY_LENGTH}"
        )

    def test_speed_history_deque_bounded(self):
        """Speed history deque with maxlen=SPEED_SMOOTHING_WINDOW stays bounded."""
        speed_hist = deque(maxlen=SPEED_SMOOTHING_WINDOW)
        for i in range(SPEED_SMOOTHING_WINDOW + 30):
            speed_hist.append(float(i * 10))
        assert len(speed_hist) == SPEED_SMOOTHING_WINDOW, (
            f"Speed history grew to {len(speed_hist)}, expected {SPEED_SMOOTHING_WINDOW}"
        )

    def test_multiple_tracks_each_bounded(self):
        """Multiple track trajectories each stay within TRAJECTORY_LENGTH."""
        trajectories = {}
        for track_id in range(5):
            trajectories[track_id] = deque(maxlen=TRAJECTORY_LENGTH)
            for i in range(TRAJECTORY_LENGTH + 20):
                trajectories[track_id].append((float(i), float(i)))
        for tid, traj in trajectories.items():
            assert len(traj) == TRAJECTORY_LENGTH


# ---------------------------------------------------------------------------
# Ball detection latency (requires weights)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ball_weights_available(), reason="Ball model weights not found")
class TestBallDetectionLatency:

    def test_single_inference_under_threshold(self):
        """Single-frame ball detection should complete in < 100 ms (median over 10 runs)."""
        from ball_tracking_analysis import BallTracker, TRACKER_MAX_AGE, TRACKER_MIN_HITS, TRACKER_IOU_THRESHOLD
        from sort import Sort

        model_path = resolve_model_path(BALL_MODEL_PATH)
        tracker = BallTracker(model_path, fps=30.0)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Warm-up
        for _ in range(3):
            tracker.detect_and_track(frame)

        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            tracker.detect_and_track(frame)
            latencies.append(time.perf_counter() - t0)

        median_ms = np.median(latencies) * 1000
        assert median_ms < 100, f"Ball detection median latency {median_ms:.1f} ms exceeds 100 ms"


# ---------------------------------------------------------------------------
# Score detection latency (requires weights)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _digit_weights_available(), reason="Digit model weights not found")
class TestScoreDetectionLatency:

    def test_batched_score_inference_under_threshold(self, sample_rois):
        """Batched score detection on 4 ROIs should complete in < 50 ms (median)."""
        from ball_tracking_fast import ScoreDetectorBatched

        model_path = resolve_model_path(DIGIT_MODEL_PATH)
        detector = ScoreDetectorBatched(model_path)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Warm-up
        for _ in range(2):
            detector.update_scores_and_rounds(frame, sample_rois, 0)

        latencies = []
        for i in range(10):
            t0 = time.perf_counter()
            detector.update_scores_and_rounds(frame, sample_rois, i)
            latencies.append(time.perf_counter() - t0)

        median_ms = np.median(latencies) * 1000
        assert median_ms < 50, f"Score detection median latency {median_ms:.1f} ms exceeds 50 ms"


# ---------------------------------------------------------------------------
# Full pipeline FPS (requires weights)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ball_weights_available(), reason="Ball model weights not found")
class TestFullPipelineFPS:

    def test_fps_above_minimum(self):
        """50 synthetic frames through ball detection should run at >= 15 FPS."""
        from ball_tracking_analysis import BallTracker

        model_path = resolve_model_path(BALL_MODEL_PATH)
        tracker = BallTracker(model_path, fps=30.0)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(50)]

        # Warm-up
        for _ in range(3):
            tracker.detect_and_track(frames[0])

        t_start = time.perf_counter()
        for frame in frames:
            tracker.detect_and_track(frame)
        elapsed = time.perf_counter() - t_start

        fps = len(frames) / elapsed
        assert fps >= 15, f"Pipeline FPS {fps:.1f} is below minimum of 15 FPS"
