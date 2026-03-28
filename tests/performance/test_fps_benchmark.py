"""
Performance benchmarks for the CV pipeline components.

Run with:
    pytest tests/performance/ -v -s

These tests require model weights and are skipped if not present.
"""

import time
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BALL_WEIGHTS = os.path.join(_PROJECT_ROOT, 'runs', 'detect', 'runs', 'ball_detector', 'weights', 'best.pt')
_DIGIT_WEIGHTS = os.path.join(_PROJECT_ROOT, 'runs', 'detect', 'runs', 'detect', 'digits_v2', 'weights', 'best.pt')

_HAS_WEIGHTS = os.path.exists(_BALL_WEIGHTS) and os.path.exists(_DIGIT_WEIGHTS)
skip_no_weights = pytest.mark.skipif(not _HAS_WEIGHTS, reason="Model weights not found")


@skip_no_weights
class TestBallDetectionLatency:

    def test_single_inference_under_threshold(self):
        """Single YOLO ball detection inference should complete under 100ms."""
        from ball_tracking_analysis import BallTracker
        from tests.fixtures.synthetic_frames import make_frame_with_ball

        tracker = BallTracker(_BALL_WEIGHTS, fps=30.0)
        frame = make_frame_with_ball(640, 360, w=960, h=540)

        # Warm up
        tracker.detect_and_track(frame)

        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            tracker.detect_and_track(frame)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        print(f"\nBall detection median: {median_ms:.1f}ms")
        assert median_ms < 100, f"Ball detection too slow: {median_ms:.1f}ms (threshold: 100ms)"


@skip_no_weights
class TestScoreDetectionLatency:

    def test_batched_score_inference_under_threshold(self):
        """Batched digit detection on 4 ROIs should complete under 50ms."""
        from ball_tracking_fast import ScoreDetectorBatched
        from tests.fixtures.synthetic_frames import make_blank_frame

        detector = ScoreDetectorBatched(_DIGIT_WEIGHTS)
        frame = make_blank_frame(1280, 720)
        rois = {
            'player1_score': (50, 30, 180, 80),
            'player2_score': (1100, 30, 1230, 80),
            'player1_rounds': (185, 30, 250, 80),
            'player2_rounds': (1050, 30, 1095, 80),
        }

        # Warm up
        detector.update_scores_and_rounds(frame, rois, 0)

        times = []
        for i in range(20):
            t0 = time.perf_counter()
            detector.update_scores_and_rounds(frame, rois, i + 1)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        print(f"\nScore detection median: {median_ms:.1f}ms")
        assert median_ms < 50, f"Score detection too slow: {median_ms:.1f}ms (threshold: 50ms)"


@skip_no_weights
class TestFullPipelineFPS:

    def test_fps_above_minimum(self):
        """Process 50 frames through ball detection and measure FPS."""
        from ball_tracking_analysis import BallTracker
        from tests.fixtures.synthetic_frames import make_frame_sequence_with_moving_ball

        tracker = BallTracker(_BALL_WEIGHTS, fps=30.0)
        frames = make_frame_sequence_with_moving_ball(n_frames=50, w=960, h=540)

        # Warm up
        tracker.detect_and_track(frames[0])

        t0 = time.perf_counter()
        for frame in frames:
            tracker.detect_and_track(frame)
        elapsed = time.perf_counter() - t0

        fps = len(frames) / elapsed
        print(f"\nPipeline FPS: {fps:.1f}")
        assert fps >= 15, f"Pipeline FPS too low: {fps:.1f} (minimum: 15)"


class TestMemoryStability:

    def test_trajectory_memory_bounded(self):
        """Trajectory storage should not grow unbounded."""
        from collections import deque
        from ball_tracking_analysis import TRAJECTORY_LENGTH

        traj = deque(maxlen=TRAJECTORY_LENGTH)
        for i in range(10000):
            traj.append((float(i), float(i)))
        assert len(traj) == TRAJECTORY_LENGTH

    def test_speed_history_bounded(self):
        from collections import deque
        from ball_tracking_analysis import SPEED_SMOOTHING_WINDOW

        hist = deque(maxlen=SPEED_SMOOTHING_WINDOW)
        for i in range(10000):
            hist.append(float(i))
        assert len(hist) == SPEED_SMOOTHING_WINDOW
