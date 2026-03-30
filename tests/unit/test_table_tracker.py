"""
Unit tests for TableTracker and its helper functions
(broadcast_utils/table_tracker.py).

Tests corner ordering, convexity, area, quality checks, and
optical-flow tracking on synthetic frames.
"""

import pytest
import numpy as np

from broadcast_utils.table_tracker import (
    TableTracker, _order_corners, _is_convex, _quad_area,
    JUMP_THRESHOLD, MIN_QUAD_AREA, EMA_ALPHA,
)
from tests.fixtures.synthetic_frames import make_blank_frame


# ---------------------------------------------------------------------------
# Helper geometry functions
# ---------------------------------------------------------------------------

class TestOrderCorners:

    def test_already_ordered_tl_tr_br_bl(self):
        pts = [(100, 100), (500, 100), (500, 300), (100, 300)]
        ordered = _order_corners(pts)
        # TL should have smallest x and y
        assert ordered[0][0] < ordered[1][0]  # TL.x < TR.x
        assert ordered[3][0] < ordered[2][0]  # BL.x < BR.x

    def test_shuffled_corners_sorted_correctly(self):
        """Shuffled input -> same TL/TR/BR/BL result."""
        pts_ordered = [(100, 100), (500, 100), (500, 300), (100, 300)]
        shuffled = [pts_ordered[2], pts_ordered[0], pts_ordered[3], pts_ordered[1]]
        ordered = _order_corners(shuffled)
        np.testing.assert_allclose(ordered[0], [100, 100], atol=1)  # TL
        np.testing.assert_allclose(ordered[1], [500, 100], atol=1)  # TR


class TestIsConvex:

    def test_convex_rectangle(self):
        pts = np.array([[100, 100], [500, 100], [500, 300], [100, 300]], dtype=np.float32)
        assert _is_convex(pts) is True

    def test_convex_trapezoid(self):
        pts = np.array([[200, 300], [1080, 300], [1080, 500], [200, 500]], dtype=np.float32)
        assert _is_convex(pts) is True

    def test_concave_quad(self):
        """A 'bowtie' or concave shape."""
        pts = np.array([[100, 100], [500, 300], [500, 100], [100, 300]], dtype=np.float32)
        assert _is_convex(pts) is False


class TestQuadArea:

    def test_known_rectangle_area(self):
        """400 x 200 rectangle -> area = 80000."""
        pts = np.array([[100, 100], [500, 100], [500, 300], [100, 300]], dtype=np.float32)
        area = _quad_area(pts)
        assert abs(area - 80000.0) < 1.0, f"Expected 80000, got {area}"

    def test_zero_area_degenerate(self):
        pts = np.array([[100, 100], [100, 100], [100, 100], [100, 100]], dtype=np.float32)
        area = _quad_area(pts)
        assert area == 0.0


# ---------------------------------------------------------------------------
# TableTracker
# ---------------------------------------------------------------------------

_GOOD_CORNERS = [(200, 300), (1080, 300), (1080, 500), (200, 500)]


class TestTableTrackerInit:

    def test_init_stores_corners(self):
        tracker = TableTracker(_GOOD_CORNERS)
        np.testing.assert_allclose(tracker.corners, np.array(_GOOD_CORNERS, dtype=np.float32))

    def test_init_is_valid_true(self):
        tracker = TableTracker(_GOOD_CORNERS)
        assert tracker.is_valid is True

    def test_first_update_returns_initial_corners(self):
        tracker = TableTracker(_GOOD_CORNERS)
        frame = make_blank_frame(1280, 720)
        result = tracker.update(frame)
        np.testing.assert_allclose(result, np.array(_GOOD_CORNERS, dtype=np.float32), atol=1)


class TestTableTrackerUpdate:

    def test_stable_sequence_keeps_valid(self):
        """
        Feeding the same textured frame repeatedly keeps is_valid True.
        A blank frame has no gradient so LK flow may fail; use a noisy frame.
        """
        import numpy as np
        tracker = TableTracker(_GOOD_CORNERS)
        rng = np.random.default_rng(42)
        frame = rng.integers(50, 200, (720, 1280, 3), dtype=np.uint8)
        tracker.update(frame)  # prime prev_gray
        for _ in range(5):
            tracker.update(frame)
        # On a static frame, LK returns the same points -> no jump -> valid
        assert tracker.is_valid is True

    def test_get_corners_int_returns_int32(self):
        tracker = TableTracker(_GOOD_CORNERS)
        corners_int = tracker.get_corners_int()
        assert corners_int.dtype == np.int32
        assert corners_int.shape == (4, 2)


class TestTableTrackerReinitialize:

    def test_reinitialize_resets_corners(self):
        tracker = TableTracker(_GOOD_CORNERS)
        new_corners = [(300, 350), (980, 350), (980, 480), (300, 480)]
        tracker.reinitialize(new_corners)
        np.testing.assert_allclose(
            tracker.corners, np.array(new_corners, dtype=np.float32)
        )

    def test_reinitialize_sets_valid_true(self):
        tracker = TableTracker(_GOOD_CORNERS)
        # Force invalid
        tracker.is_valid = False
        tracker.reinitialize(_GOOD_CORNERS)
        assert tracker.is_valid is True

    def test_reinitialize_with_frame_sets_prev_gray(self):
        tracker = TableTracker(_GOOD_CORNERS)
        frame = make_blank_frame(1280, 720)
        tracker.reinitialize(_GOOD_CORNERS, frame=frame)
        assert tracker.prev_gray is not None


class TestTableTrackerQualityCheck:

    def test_tiny_quad_fails_quality_check(self):
        """A 5x5 px quad should fail the area check."""
        tiny_corners = [(100, 100), (105, 100), (105, 105), (100, 105)]
        tracker = TableTracker(tiny_corners)
        # Quality check is run on update; initial corners may pass init but fail update
        frame = make_blank_frame(640, 480)
        tracker.update(frame)  # prime
        result = tracker.update(frame)
        # is_valid may be False due to tiny area
        area = _quad_area(np.array(tiny_corners, dtype=np.float32))
        if area < MIN_QUAD_AREA:
            assert tracker.is_valid is False
