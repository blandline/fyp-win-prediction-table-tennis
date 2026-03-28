"""
Shared pytest fixtures for the table tennis CV pipeline test suite.
"""

import sys
import os
import tempfile
import shutil
import pytest
import numpy as np

# Ensure project root and sort directory are on the path BEFORE any other imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_sort_dir = os.path.join(_project_root, 'sort')
if _sort_dir not in sys.path:
    sys.path.insert(0, _sort_dir)

from tests.fixtures.synthetic_frames import (
    make_blank_frame,
    make_frame_with_ball,
    make_frame_sequence_with_moving_ball,
    make_test_video,
    MockYOLOModel,
    MockBoxes,
    MockBox,
    MockResult,
)


@pytest.fixture
def blank_frame():
    """A 1280x720 black frame."""
    return make_blank_frame()


@pytest.fixture
def frame_with_ball():
    """A 1280x720 frame with a white ball at (640, 360)."""
    return make_frame_with_ball(640, 360)


@pytest.fixture
def tmp_output_dir():
    """Create a temporary output directory, cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="tt_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_table_corners():
    """
    Realistic table corner pixel coordinates for a 1280x720 frame.
    Order: TL, TR, BR, BL
    """
    return [
        (320, 200),    # top-left
        (960, 200),    # top-right
        (1050, 520),   # bottom-right
        (230, 520),    # bottom-left
    ]


@pytest.fixture
def sample_rois():
    """Sample score and round ROIs for a 1280x720 frame."""
    return {
        'player1_score': (50, 30, 180, 80),
        'player2_score': (1100, 30, 1230, 80),
        'player1_rounds': (185, 30, 250, 80),
        'player2_rounds': (1050, 30, 1095, 80),
    }


@pytest.fixture
def test_video_path(tmp_output_dir):
    """Create a short test .mp4 video and return its path."""
    path = os.path.join(tmp_output_dir, "test_input.mp4")
    make_test_video(path, n_frames=90, fps=30.0, w=640, h=360)
    return path


@pytest.fixture
def mock_ball_model_single():
    """Mock YOLO model that always detects one ball at (640, 360)."""
    return MockYOLOModel(detections=[
        (632, 352, 648, 368, 0, 0.95),  # ball class=0, conf=0.95
    ])


@pytest.fixture
def mock_ball_model_none():
    """Mock YOLO model that never detects anything."""
    return MockYOLOModel(detections=[])


@pytest.fixture
def mock_digit_detections_score_5():
    """Mock boxes representing detected digit '5'."""
    return MockBoxes([MockBox(10, 5, 30, 40, 5, 0.90)])


@pytest.fixture
def mock_digit_detections_score_11():
    """Mock boxes representing detected digits '1' and '1' → score 11."""
    return MockBoxes([
        MockBox(10, 5, 25, 40, 1, 0.92),
        MockBox(30, 5, 45, 40, 1, 0.88),
    ])
