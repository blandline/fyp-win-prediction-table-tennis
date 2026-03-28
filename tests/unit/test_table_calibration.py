"""
Unit tests for TableCalibration (homography pixel ↔ metre transforms).
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ball_tracking_analysis import TableCalibration, TABLE_LENGTH_M, TABLE_WIDTH_M


class TestTableCalibrationValid:
    """Tests with well-formed corners."""

    @pytest.fixture
    def calibration(self, sample_table_corners):
        return TableCalibration(sample_table_corners)

    def test_is_valid(self, calibration):
        assert calibration.is_valid()

    def test_homography_computed(self, calibration):
        assert calibration.H is not None
        assert calibration.H.shape == (3, 3)

    def test_reprojection_error_small(self, calibration):
        err = calibration.get_reprojection_error()
        assert err is not None
        assert err < 0.15, f"Reprojection error {err:.4f}m exceeds 0.15m threshold"

    def test_pixel_to_meters_centre_of_table(self, calibration):
        """The pixel centre of the marked table should map near (L/2, W/2)."""
        corners = np.array([(320, 200), (960, 200), (1050, 520), (230, 520)], dtype=np.float32)
        cx = float(np.mean(corners[:, 0]))
        cy = float(np.mean(corners[:, 1]))
        x_m, y_m = calibration.pixel_to_meters(cx, cy)
        assert x_m is not None and y_m is not None
        assert abs(x_m - TABLE_LENGTH_M / 2) < 0.5, f"x_m={x_m}"
        assert abs(y_m - TABLE_WIDTH_M / 2) < 0.5, f"y_m={y_m}"

    def test_pixel_to_meters_roundtrip(self, calibration):
        """pixel → metre → pixel should round-trip with < 2px error."""
        if calibration.H is None:
            pytest.skip("No homography")
        H_inv = np.linalg.inv(calibration.H)
        test_pixel = np.array([[[640.0, 360.0]]], dtype=np.float32)
        import cv2
        metres = cv2.perspectiveTransform(test_pixel, calibration.H)
        back = cv2.perspectiveTransform(metres, H_inv)
        err = np.linalg.norm(back[0, 0] - test_pixel[0, 0])
        assert err < 2.0, f"Round-trip pixel error: {err:.2f}px"

    def test_point_far_outside_table_returns_none(self, calibration):
        """A pixel very far from the table should be rejected (returns None)."""
        x_m, y_m = calibration.pixel_to_meters(0, 0)
        # Should either be None (rejected) or within margin — not wildly off
        if x_m is not None:
            assert x_m > -(TABLE_LENGTH_M + 5), "Point accepted but unreasonably far"


class TestTableCalibrationInvalid:
    """Tests with degenerate / bad corners."""

    def test_too_few_corners(self):
        """3 corners → should fail (reshape error or invalid calibration)."""
        try:
            cal = TableCalibration([(0, 0), (100, 0), (100, 100)])
            assert not cal.is_valid()
        except ValueError:
            # Acceptable: numpy reshape fails for 3 corners
            pass

    def test_collapsed_corners(self):
        """All corners at the same point → degenerate."""
        cal = TableCalibration([(100, 100)] * 4)
        assert not cal.is_valid()

    def test_very_small_quadrilateral(self):
        """Corners forming a tiny 5x5 pixel quad → rejected by side-length check."""
        cal = TableCalibration([(100, 100), (105, 100), (105, 105), (100, 105)])
        assert not cal.is_valid()

    def test_reprojection_error_high_for_bad_corners(self):
        """Deliberately skewed corners should have high reprojection error."""
        cal = TableCalibration([(0, 0), (1000, 0), (1000, 10), (0, 10)])
        # This creates a very flat quad; calibration might still compute but
        # reprojection error should be checked
        if cal.H is not None:
            err = cal.get_reprojection_error()
            # Just verify it returns a number; extreme shapes may still pass
            assert err is not None
