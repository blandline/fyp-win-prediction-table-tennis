"""
Shared pytest fixtures for the broadcast pipeline test suite.

Provides:
- tmp_output_dir: temporary directory for CSV/JSON output
- sample_rois: realistic ROI dict
- valid_table_corners: 4-point table quad in a 1280x720 frame
- MockYOLOModel / mock_yolo_model: configurable fake YOLO detector
- sample_packet: pre-filled PredictionDataPacket
- sample_pose_features: dict matching DataLogger.POSE_COLUMNS
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is on sys.path so all modules are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# MockYOLOModel — configurable fake YOLO that returns preset detections
# ---------------------------------------------------------------------------

class MockYOLOResult:
    """Mimics ultralytics Results object."""

    class _Boxes:
        def __init__(self, detections):
            self._dets = detections  # list of [x1,y1,x2,y2,conf]

        def __len__(self):
            return len(self._dets)

        def __iter__(self):
            return iter(self._boxes_list())

        def _boxes_list(self):
            return [MockYOLOResult._Box(d) for d in self._dets]

        @property
        def xyxy(self):
            if not self._dets:
                return np.empty((0, 4))
            return np.array([[d[0], d[1], d[2], d[3]] for d in self._dets])

        @property
        def conf(self):
            if not self._dets:
                return np.empty((0,))
            return np.array([d[4] for d in self._dets])

    class _Box:
        def __init__(self, det):
            self._det = det

        @property
        def xyxy(self):
            return [np.array(self._det[:4])]

        @property
        def conf(self):
            class _Conf:
                def __init__(self, v):
                    self._v = v
                def item(self):
                    return self._v
            return _Conf(self._det[4])

        @property
        def cls(self):
            class _Cls:
                def __init__(self, v):
                    self._v = v
                def item(self):
                    return self._v
            return _Cls(self._det[5] if len(self._det) > 5 else 0)

    def __init__(self, detections):
        self.boxes = self._Boxes(detections)


class MockYOLOModel:
    """
    Fake YOLO model that returns a configurable list of detections.

    Usage:
        model = MockYOLOModel(detections=[[100, 100, 120, 120, 0.9]])
        results = model(frame, conf=0.4, verbose=False)
        # results[0].boxes yields the configured detections
    """

    def __init__(self, detections=None):
        """
        detections: list of [x1, y1, x2, y2, conf] or
                    list of [x1, y1, x2, y2, conf, cls_id]
        """
        self._detections = detections or []
        self._call_count = 0

    def set_detections(self, detections):
        self._detections = detections

    def __call__(self, *args, **kwargs):
        self._call_count += 1
        return [MockYOLOResult(self._detections)]

    @property
    def call_count(self):
        return self._call_count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path_in_workspace):
    """Temporary directory for DataLogger output."""
    d = tmp_path_in_workspace / "output"
    d.mkdir(exist_ok=True)
    return str(d)


@pytest.fixture
def tmp_path_in_workspace(request, tmp_path_factory):
    """
    Temporary directory inside the workspace (avoids AppData/Temp sandbox restriction).
    Falls back to tmp_path_factory if workspace write is available.
    """
    import tempfile
    import pathlib
    # Use a subdirectory inside the workspace's .pytest_tmp folder
    base = pathlib.Path(_PROJECT_ROOT) / ".pytest_tmp"
    base.mkdir(exist_ok=True)
    # Create a unique subdir per test
    test_name = request.node.nodeid.replace("/", "_").replace("\\", "_").replace("::", "_").replace(" ", "_")
    test_name = test_name[:60]  # truncate long names
    d = base / test_name
    d.mkdir(exist_ok=True)
    yield d
    # Cleanup
    import shutil
    try:
        shutil.rmtree(str(d), ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def sample_rois():
    """Realistic ROI dict for a 1280x720 broadcast frame."""
    return {
        "player1_score": (50, 30, 200, 90),
        "player2_score": (1080, 30, 1230, 90),
        "player1_rounds": (200, 30, 280, 90),
        "player2_rounds": (1000, 30, 1080, 90),
    }


@pytest.fixture
def valid_table_corners():
    """
    4 corners (TL, TR, BR, BL) of a table in a 1280x720 frame.
    Chosen so the quad is wide enough to pass all quality checks.
    """
    return [
        (200, 300),   # TL
        (1080, 300),  # TR
        (1080, 500),  # BR
        (200, 500),   # BL
    ]


@pytest.fixture
def mock_yolo_model():
    """Return a fresh MockYOLOModel with no detections."""
    return MockYOLOModel()


@pytest.fixture
def sample_packet():
    """A pre-filled PredictionDataPacket with realistic values."""
    from prediction_model_base import (
        PredictionDataPacket, BallState, ScoreState, RallyState, PoseState,
    )
    return PredictionDataPacket(
        frame_idx=150,
        timestamp_sec=5.0,
        ball=BallState(
            detected=True,
            position_px=(320.0, 240.0),
            speed_pps=450.0,
            speed_mps=8.5,
            trajectory_px=[(310.0, 235.0), (315.0, 237.0), (320.0, 240.0)],
        ),
        score=ScoreState(
            player1_score=5,
            player2_score=3,
            player1_sets=1,
            player2_sets=0,
            score_reliable=True,
        ),
        rally=RallyState(
            is_active=True,
            rally_id=7,
            rally_duration_sec=2.3,
            mean_speed_mps=8.2,
            max_speed_mps=12.1,
            landing_zones=[0, 0, 1, 0, 2, 0, 1, 0, 0],
        ),
        pose=PoseState(
            player1={
                "v_hand_speed": 1.2,
                "com_height": 0.55,
                "v_torso_x": 0.05,
                "angle_elbow_dom": 110.0,
                "visibility_mean": 0.85,
            },
            player2={
                "v_hand_speed": 0.9,
                "com_height": 0.52,
                "v_torso_x": -0.03,
                "angle_elbow_dom": 105.0,
                "visibility_mean": 0.80,
            },
        ),
        match_elapsed_sec=300.0,
    )


@pytest.fixture
def sample_pose_features():
    """Dict with all DataLogger.POSE_COLUMNS keys populated."""
    return {
        "frame": 100,
        "timestamp_sec": 3.33,
        "player_id": 1,
        "visibility_mean": 0.88,
        "visibility_min": 0.70,
        "visibility_left_arm": 0.82,
        "visibility_right_arm": 0.91,
        "angle_trunk_lean": 5.2,
        "angle_elbow_dom": 112.0,
        "angle_elbow_left": 108.0,
        "angle_elbow_right": 115.0,
        "angle_knee_front": 155.0,
        "angle_knee_back": 160.0,
        "com_height": 0.54,
        "hand_height": 0.72,
        "hand_height_left": 0.70,
        "hand_height_right": 0.74,
        "stance_length": 0.45,
        "v_hand_speed": 1.1,
        "v_hand_speed_left": 0.9,
        "v_hand_speed_right": 1.3,
        "v_torso_x": 0.04,
        "v_com_y": -0.02,
    }
