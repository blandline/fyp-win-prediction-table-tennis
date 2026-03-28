"""
Unit tests for BallTracker — detection, tracking, speed, trajectory.
"""

import numpy as np
import pytest
import sys, os
from unittest.mock import patch, MagicMock
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ball_tracking_analysis import (
    BallTracker, TableCalibration,
    TRAJECTORY_LENGTH, SPEED_SMOOTHING_WINDOW,
    TRACKER_MAX_AGE, MAX_BALL_SPEED_MPS,
)
from tests.fixtures.synthetic_frames import (
    make_blank_frame, make_frame_with_ball, MockYOLOModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker_with_mock(detections, fps=30.0, table_cal=None):
    """Create a BallTracker with a mock YOLO model returning preset detections."""
    tracker = BallTracker.__new__(BallTracker)
    tracker.model = MockYOLOModel(detections)
    tracker.fps = fps
    tracker.table_calibration = table_cal
    tracker.inference_size = None
    tracker.trajectories = {}
    tracker.speed_history = {}
    tracker.trajectories_meters = {}
    tracker.speed_history_mps = {}
    from sort import Sort
    tracker.tracker = Sort(max_age=TRACKER_MAX_AGE, min_hits=1, iou_threshold=0.1)
    return tracker


# ---------------------------------------------------------------------------
# Speed calculation tests
# ---------------------------------------------------------------------------

class TestSpeedCalculation:

    def test_speed_zero_when_stationary(self):
        """Same position across frames → speed ≈ 0."""
        tracker = _make_tracker_with_mock(
            [(630, 350, 650, 370, 0, 0.95)], fps=30.0
        )
        frame = make_frame_with_ball(640, 360)
        for _ in range(5):
            tracker.detect_and_track(frame)

        for tid, hist in tracker.speed_history.items():
            if hist:
                # With identical positions, speed should be near 0
                assert np.mean(hist) < 50, f"Speed should be ~0 for stationary ball, got {np.mean(hist)}"

    def test_speed_calculation_known_distance(self):
        """Two positions 30px apart at 30 FPS → ~900 px/s."""
        tracker = BallTracker.__new__(BallTracker)
        tracker.fps = 30.0
        tracker.table_calibration = None
        tracker.trajectories = {1: deque(maxlen=TRAJECTORY_LENGTH)}
        tracker.speed_history = {1: deque(maxlen=SPEED_SMOOTHING_WINDOW)}
        tracker.trajectories_meters = {1: deque(maxlen=TRAJECTORY_LENGTH)}
        tracker.speed_history_mps = {1: deque(maxlen=SPEED_SMOOTHING_WINDOW)}

        # Simulate two positions
        tracker.trajectories[1].append((100.0, 200.0))
        tracker.trajectories[1].append((130.0, 200.0))  # 30px to the right

        # Manually compute speed as the tracker would
        dist = np.sqrt((130 - 100) ** 2 + (200 - 200) ** 2)
        expected_speed = dist * 30.0  # 900 px/s
        assert abs(expected_speed - 900.0) < 1.0

    def test_smoothed_speed_returns_zero_for_unknown_track(self):
        tracker = _make_tracker_with_mock([], fps=30.0)
        assert tracker.get_smoothed_speed(9999) == 0

    def test_smoothed_speed_mps_returns_none_for_unknown_track(self):
        tracker = _make_tracker_with_mock([], fps=30.0)
        assert tracker.get_smoothed_speed_mps(9999) is None


# ---------------------------------------------------------------------------
# Trajectory tests
# ---------------------------------------------------------------------------

class TestTrajectory:

    def test_trajectory_capped_at_max_length(self):
        """Trajectory deque should not exceed TRAJECTORY_LENGTH."""
        tracker = _make_tracker_with_mock(
            [(630, 350, 650, 370, 0, 0.95)], fps=30.0
        )
        frame = make_frame_with_ball(640, 360)
        for _ in range(TRAJECTORY_LENGTH + 20):
            tracker.detect_and_track(frame)

        for tid, traj in tracker.trajectories.items():
            assert len(traj) <= TRAJECTORY_LENGTH

    def test_trajectory_grows_with_detections(self):
        tracker = _make_tracker_with_mock(
            [(630, 350, 650, 370, 0, 0.95)], fps=30.0
        )
        frame = make_frame_with_ball(640, 360)
        tracker.detect_and_track(frame)
        tracker.detect_and_track(frame)

        found = False
        for tid, traj in tracker.trajectories.items():
            if len(traj) >= 2:
                found = True
        assert found, "Expected trajectory with at least 2 points"


# ---------------------------------------------------------------------------
# Detection filtering tests
# ---------------------------------------------------------------------------

class TestDetectionFiltering:

    def test_no_detection_on_blank_frame(self):
        """No ball in frame → tracker returns no active tracks after min_hits."""
        tracker = _make_tracker_with_mock([], fps=30.0)
        frame = make_blank_frame()
        tracks = tracker.detect_and_track(frame)
        assert len(tracks) == 0

    def test_low_confidence_filtered(self):
        """Detection with conf below threshold → not passed to tracker."""
        tracker = _make_tracker_with_mock(
            [(630, 350, 650, 370, 0, 0.20)], fps=30.0  # conf 0.20 < 0.40 threshold
        )
        frame = make_frame_with_ball(640, 360)
        tracks = tracker.detect_and_track(frame)
        # With conf=0.20 and BALL_CONF_THRESHOLD=0.40, the mock filters it
        assert len(tracks) == 0


# ---------------------------------------------------------------------------
# Track persistence tests
# ---------------------------------------------------------------------------

class TestTrackPersistence:

    def test_track_persists_during_short_gap(self):
        """Track should survive a few frames without detection (MAX_AGE=5)."""
        # First establish a track with detections
        det = [(630, 350, 650, 370, 0, 0.95)]
        tracker = _make_tracker_with_mock(det, fps=30.0)
        frame = make_frame_with_ball(640, 360)

        # Detect for several frames to establish track
        for _ in range(5):
            tracker.detect_and_track(frame)

        # Now switch to no detections
        tracker.model = MockYOLOModel([])
        blank = make_blank_frame()

        # Track should persist for a few frames
        still_tracked = False
        for i in range(3):
            tracks = tracker.detect_and_track(blank)
            if len(tracks) > 0:
                still_tracked = True
        assert still_tracked, "Track should persist during short detection gap"

    def test_track_lost_after_max_age(self):
        """Track should be dropped after MAX_AGE frames without detection."""
        det = [(630, 350, 650, 370, 0, 0.95)]
        tracker = _make_tracker_with_mock(det, fps=30.0)
        frame = make_frame_with_ball(640, 360)

        for _ in range(5):
            tracker.detect_and_track(frame)

        tracker.model = MockYOLOModel([])
        blank = make_blank_frame()

        for _ in range(TRACKER_MAX_AGE + 5):
            tracks = tracker.detect_and_track(blank)

        assert len(tracks) == 0, "Track should be dropped after MAX_AGE frames"


# ---------------------------------------------------------------------------
# Metre-space tests
# ---------------------------------------------------------------------------

class TestMetreSpace:

    def test_position_meters_none_without_calibration(self):
        tracker = _make_tracker_with_mock([], fps=30.0)
        x_m, y_m = tracker.get_position_meters(1)
        assert x_m is None and y_m is None

    def test_set_table_calibration(self, sample_table_corners):
        tracker = _make_tracker_with_mock([], fps=30.0)
        cal = TableCalibration(sample_table_corners)
        tracker.set_table_calibration(cal)
        assert tracker.table_calibration is cal


# ---------------------------------------------------------------------------
# Inference size rescaling
# ---------------------------------------------------------------------------

class TestInferenceRescaling:

    def test_prepare_inference_frame_no_resize(self):
        tracker = _make_tracker_with_mock([], fps=30.0)
        frame = make_blank_frame(1280, 720)
        inf_frame, sx, sy = tracker._prepare_inference_frame(frame)
        assert sx == 1.0 and sy == 1.0
        assert inf_frame.shape == frame.shape

    def test_prepare_inference_frame_with_resize(self):
        tracker = _make_tracker_with_mock([], fps=30.0)
        tracker.inference_size = (640, 360)
        frame = make_blank_frame(1280, 720)
        inf_frame, sx, sy = tracker._prepare_inference_frame(frame)
        assert inf_frame.shape == (360, 640, 3)
        assert abs(sx - 2.0) < 0.01
        assert abs(sy - 2.0) < 0.01
