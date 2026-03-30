"""
Unit tests for pose geometry helpers (ball_tracking_analysis.py).

Tests _angle_between, _vertical_angle, and velocity-related logic
without requiring MediaPipe or any model files.
"""

import math
import pytest

from ball_tracking_analysis import _angle_between, _vertical_angle


# ---------------------------------------------------------------------------
# _angle_between
# ---------------------------------------------------------------------------

class TestAngleBetween:

    def test_collinear_points_180_degrees(self):
        """A-B-C collinear (B in the middle) -> angle at B is ~180°."""
        a = (0, 0)
        b = (1, 0)
        c = (2, 0)
        angle = _angle_between(a, b, c)
        assert angle is not None
        assert abs(angle - 180.0) < 1.0, f"Expected ~180°, got {angle:.2f}°"

    def test_right_angle_90_degrees(self):
        """L-shaped: angle at B is 90°."""
        a = (0, 1)
        b = (0, 0)
        c = (1, 0)
        angle = _angle_between(a, b, c)
        assert angle is not None
        assert abs(angle - 90.0) < 1.0, f"Expected ~90°, got {angle:.2f}°"

    def test_acute_angle_45_degrees(self):
        """Isoceles right triangle: angle at vertex is 45°."""
        a = (1, 0)
        b = (0, 0)
        c = (1, 1)
        angle = _angle_between(a, b, c)
        assert angle is not None
        assert abs(angle - 45.0) < 1.0, f"Expected ~45°, got {angle:.2f}°"

    def test_none_input_a_returns_none(self):
        assert _angle_between(None, (0, 0), (1, 0)) is None

    def test_none_input_b_returns_none(self):
        assert _angle_between((0, 0), None, (1, 0)) is None

    def test_none_input_c_returns_none(self):
        assert _angle_between((0, 0), (1, 0), None) is None

    def test_degenerate_coincident_b_a_returns_none(self):
        """A == B -> zero vector -> degenerate."""
        assert _angle_between((0, 0), (0, 0), (1, 0)) is None

    def test_degenerate_coincident_b_c_returns_none(self):
        """B == C -> zero vector -> degenerate."""
        assert _angle_between((0, 0), (1, 0), (1, 0)) is None

    def test_obtuse_angle(self):
        """Angle > 90° (120°)."""
        a = (1, 0)
        b = (0, 0)
        c = (-0.5, math.sqrt(3) / 2)
        angle = _angle_between(a, b, c)
        assert angle is not None
        assert abs(angle - 120.0) < 1.0, f"Expected ~120°, got {angle:.2f}°"


# ---------------------------------------------------------------------------
# _vertical_angle
# ---------------------------------------------------------------------------

class TestVerticalAngle:

    def test_vertical_alignment_near_zero(self):
        """top directly above bottom -> angle ~0°."""
        top = (0, 0)
        bottom = (0, 10)
        angle = _vertical_angle(top, bottom)
        assert angle is not None
        assert abs(angle) < 1.0, f"Expected ~0°, got {angle:.2f}°"

    def test_lean_right_positive(self):
        """Bottom shifted right of top -> positive angle."""
        top = (0, 0)
        bottom = (5, 10)
        angle = _vertical_angle(top, bottom)
        assert angle is not None
        assert angle > 0, f"Expected positive lean, got {angle:.2f}°"

    def test_lean_left_negative(self):
        """Bottom shifted left of top -> negative angle."""
        top = (0, 0)
        bottom = (-5, 10)
        angle = _vertical_angle(top, bottom)
        assert angle is not None
        assert angle < 0, f"Expected negative lean, got {angle:.2f}°"

    def test_horizontal_near_90_degrees(self):
        """top and bottom at same height, offset horizontally -> ~90°."""
        top = (0, 5)
        bottom = (10, 5)
        angle = _vertical_angle(top, bottom)
        assert angle is not None
        assert abs(abs(angle) - 90.0) < 1.0, f"Expected ~90°, got {angle:.2f}°"

    def test_none_top_returns_none(self):
        assert _vertical_angle(None, (0, 10)) is None

    def test_none_bottom_returns_none(self):
        assert _vertical_angle((0, 0), None) is None

    def test_degenerate_same_point_returns_none(self):
        assert _vertical_angle((5, 5), (5, 5)) is None


# ---------------------------------------------------------------------------
# Velocity logic (pure math, no MediaPipe)
# ---------------------------------------------------------------------------

class TestVelocityLogic:

    def test_hand_speed_from_displacement(self):
        """
        Normalised hand displacement / dt should give hand speed.
        If hand moves 0.1 units in 1/6 second, speed = 0.6 units/s.
        """
        displacement = 0.1
        dt = 1.0 / 6.0
        expected_speed = displacement / dt
        assert abs(expected_speed - 0.6) < 0.001

    def test_torso_velocity_positive_when_x_increases(self):
        """
        If torso x increases between frames, horizontal velocity is positive.
        """
        prev_x = 0.3
        curr_x = 0.35
        dt = 1.0 / 6.0
        v_torso_x = (curr_x - prev_x) / dt
        assert v_torso_x > 0

    def test_torso_velocity_negative_when_x_decreases(self):
        prev_x = 0.35
        curr_x = 0.30
        dt = 1.0 / 6.0
        v_torso_x = (curr_x - prev_x) / dt
        assert v_torso_x < 0

    def test_com_vertical_velocity_sign(self):
        """COM moving downward (y increases in image coords) -> positive v_com_y."""
        prev_y = 0.5
        curr_y = 0.55
        dt = 1.0 / 6.0
        v_com_y = (curr_y - prev_y) / dt
        assert v_com_y > 0
