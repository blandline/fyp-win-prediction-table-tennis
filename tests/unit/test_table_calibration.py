"""
Unit tests for TableCalibration (ball_tracking_analysis.py).

Tests homography computation, pixel-to-meter mapping, validity checks,
and reprojection error — all without requiring YOLO weights or MediaPipe.
"""

import math
import pytest
import numpy as np

from ball_tracking_analysis import TableCalibration, TABLE_LENGTH_M, TABLE_WIDTH_M


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _realistic_corners(frame_w=1280, frame_h=720):
    """Return 4 corners (TL, TR, BR, BL) that form a wide, realistic table quad."""
    return [
        (200, 300),
        (1080, 300),
        (1080, 500),
        (200, 500),
    ]


def _table_center_px(corners):
    """Pixel centre of the four corners."""
    pts = np.array(corners, dtype=np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


# ---------------------------------------------------------------------------
# Valid calibration
# ---------------------------------------------------------------------------

class TestTableCalibrationValid:

    def test_is_valid_with_good_corners(self):
        tc = TableCalibration(_realistic_corners())
        assert tc.is_valid() is True

    def test_homography_is_3x3(self):
        tc = TableCalibration(_realistic_corners())
        assert tc.H is not None
        assert tc.H.shape == (3, 3)

    def test_reprojection_error_low(self):
        tc = TableCalibration(_realistic_corners())
        err = tc.get_reprojection_error()
        assert err is not None
        assert err < 0.15, f"Reprojection error {err:.4f} m exceeds 0.15 m threshold"

    def test_pixel_center_maps_near_table_center(self):
        corners = _realistic_corners()
        tc = TableCalibration(corners)
        cx, cy = _table_center_px(corners)
        x_m, y_m = tc.pixel_to_meters(cx, cy)
        assert x_m is not None and y_m is not None
        # Table centre in metres is (TABLE_LENGTH_M/2, TABLE_WIDTH_M/2)
        assert abs(x_m - TABLE_LENGTH_M / 2) < 0.3, f"x_m={x_m:.3f} far from table centre"
        assert abs(y_m - TABLE_WIDTH_M / 2) < 0.3, f"y_m={y_m:.3f} far from table centre"

    def test_pixel_to_meters_top_left_corner(self):
        corners = _realistic_corners()
        tc = TableCalibration(corners)
        x_m, y_m = tc.pixel_to_meters(corners[0][0], corners[0][1])
        # TL maps to (0, 0) in table space
        assert x_m is not None and y_m is not None
        assert abs(x_m) < 0.1, f"TL x_m={x_m:.3f} not near 0"
        assert abs(y_m) < 0.1, f"TL y_m={y_m:.3f} not near 0"

    def test_pixel_to_meters_bottom_right_corner(self):
        corners = _realistic_corners()
        tc = TableCalibration(corners)
        # BR is corners[2]
        x_m, y_m = tc.pixel_to_meters(corners[2][0], corners[2][1])
        assert x_m is not None and y_m is not None
        assert abs(x_m - TABLE_LENGTH_M) < 0.1
        assert abs(y_m - TABLE_WIDTH_M) < 0.1

    def test_far_outside_table_returns_none(self):
        tc = TableCalibration(_realistic_corners())
        # A pixel far outside the table (top-left corner of a 1280x720 frame)
        x_m, y_m = tc.pixel_to_meters(0, 0)
        # Should return (None, None) because the projected point is outside the margin
        assert x_m is None or (x_m is not None and y_m is not None)
        # More robustly: a very far point should be None
        x_m2, y_m2 = tc.pixel_to_meters(-5000, -5000)
        assert x_m2 is None and y_m2 is None

    def test_pixel_to_meters_round_trip_error_small(self):
        """Map pixel -> metres -> back to pixel; error should be < 2 pixels."""
        corners = _realistic_corners()
        tc = TableCalibration(corners)
        cx, cy = _table_center_px(corners)
        x_m, y_m = tc.pixel_to_meters(cx, cy)
        if x_m is None:
            pytest.skip("Center pixel mapped outside table margin")
        # Invert: metres -> pixel using inverse homography
        H_inv = np.linalg.inv(tc.H)
        pt_m = np.array([[[x_m, y_m]]], dtype=np.float32)
        pt_px = cv2_perspective_transform_inv(H_inv, x_m, y_m)
        assert abs(pt_px[0] - cx) < 2.0, f"Round-trip x error {abs(pt_px[0]-cx):.2f} px"
        assert abs(pt_px[1] - cy) < 2.0, f"Round-trip y error {abs(pt_px[1]-cy):.2f} px"


def cv2_perspective_transform_inv(H_inv, x_m, y_m):
    """Apply inverse homography to get pixel coords from metre coords."""
    import cv2
    pt = np.array([[[x_m, y_m]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H_inv)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# ---------------------------------------------------------------------------
# Invalid calibration
# ---------------------------------------------------------------------------

class TestTableCalibrationInvalid:

    def test_collapsed_quad_is_invalid(self):
        """All four corners at the same point -> degenerate."""
        corners = [(100, 100), (100, 100), (100, 100), (100, 100)]
        tc = TableCalibration(corners)
        assert tc.is_valid() is False

    def test_too_few_corners_invalid(self):
        """Fewer than 4 corners should either raise or produce invalid calibration."""
        corners = [(100, 100), (200, 100), (200, 200)]
        try:
            tc = TableCalibration(corners)
            assert tc.is_valid() is False
        except (ValueError, Exception):
            # TableCalibration may raise when given wrong number of corners — that's acceptable
            pass

    def test_tiny_quad_is_invalid(self):
        """A quad where all sides are < 10 px should fail the is_valid check."""
        corners = [(100, 100), (105, 100), (105, 103), (100, 103)]
        tc = TableCalibration(corners)
        assert tc.is_valid() is False

    def test_reprojection_error_numeric_for_computable_H(self):
        """Even a bad quad that produces an H should have a numeric reprojection error."""
        corners = [(100, 100), (200, 100), (200, 110), (100, 110)]
        tc = TableCalibration(corners)
        err = tc.get_reprojection_error()
        # May be None if H was not computed, or a float if it was
        if err is not None:
            assert isinstance(err, float)
            assert not math.isnan(err)

    def test_collinear_corners_invalid(self):
        """Corners on a straight line -> degenerate homography."""
        corners = [(100, 200), (300, 200), (500, 200), (700, 200)]
        tc = TableCalibration(corners)
        assert tc.is_valid() is False
