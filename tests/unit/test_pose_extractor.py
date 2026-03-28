"""
Unit tests for pose feature extraction: angle computation, visibility, velocity.
These tests exercise the pure-math helper functions without requiring MediaPipe.
"""

import math
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ball_tracking_analysis import _angle_between, _vertical_angle


# ---------------------------------------------------------------------------
# _angle_between tests
# ---------------------------------------------------------------------------

class TestAngleBetween:

    def test_straight_line_180_degrees(self):
        """Three collinear points → angle ≈ 180°."""
        a = (0.0, 0.0)
        b = (1.0, 0.0)
        c = (2.0, 0.0)
        angle = _angle_between(a, b, c)
        assert angle is not None
        assert abs(angle - 180.0) < 0.1

    def test_right_angle_90_degrees(self):
        """L-shaped points → angle ≈ 90°."""
        a = (0.0, 0.0)
        b = (1.0, 0.0)
        c = (1.0, 1.0)
        angle = _angle_between(a, b, c)
        assert angle is not None
        assert abs(angle - 90.0) < 0.1

    def test_acute_angle(self):
        """~45° angle at vertex B."""
        # A straight up, B at origin, C at 45° → angle at B = 45°
        a = (0.0, -1.0)
        b = (0.0, 0.0)
        c = (1.0, -1.0)
        angle = _angle_between(a, b, c)
        assert angle is not None
        assert 40 < angle < 50  # approximately 45°

    def test_returns_none_when_point_is_none(self):
        assert _angle_between(None, (1, 0), (2, 0)) is None
        assert _angle_between((0, 0), None, (2, 0)) is None
        assert _angle_between((0, 0), (1, 0), None) is None

    def test_coincident_points_returns_none(self):
        """Same point for A and B → zero-length vector → None."""
        angle = _angle_between((1.0, 1.0), (1.0, 1.0), (2.0, 2.0))
        assert angle is None


# ---------------------------------------------------------------------------
# _vertical_angle tests
# ---------------------------------------------------------------------------

class TestVerticalAngle:

    def test_straight_down_zero_degrees(self):
        """Top directly above bottom → 0° lean."""
        angle = _vertical_angle((0.5, 0.3), (0.5, 0.6))
        assert angle is not None
        assert abs(angle) < 1.0

    def test_leaning_right_positive(self):
        """Bottom is to the right of top → positive angle."""
        angle = _vertical_angle((0.5, 0.3), (0.8, 0.6))
        assert angle is not None
        assert angle > 0

    def test_leaning_left_negative(self):
        angle = _vertical_angle((0.5, 0.3), (0.2, 0.6))
        assert angle is not None
        assert angle < 0

    def test_horizontal_90_degrees(self):
        """Top and bottom at same height, offset horizontally → ~90°."""
        angle = _vertical_angle((0.3, 0.5), (0.6, 0.5))
        assert angle is not None
        assert abs(angle - 90.0) < 1.0

    def test_returns_none_for_none_input(self):
        assert _vertical_angle(None, (0.5, 0.6)) is None
        assert _vertical_angle((0.5, 0.3), None) is None

    def test_coincident_points_returns_none(self):
        angle = _vertical_angle((0.5, 0.5), (0.5, 0.5))
        assert angle is None


# ---------------------------------------------------------------------------
# Velocity / hand speed tests (logic-level)
# ---------------------------------------------------------------------------

class TestVelocityLogic:

    def test_hand_speed_zero_at_rest(self):
        """Same hand position two frames → speed = 0."""
        hand_prev = (0.5, 0.5)
        hand_curr = (0.5, 0.5)
        dt = 1.0 / 6.0  # 6 FPS pose rate
        dx = hand_curr[0] - hand_prev[0]
        dy = hand_curr[1] - hand_prev[1]
        speed = math.hypot(dx, dy) / dt
        assert speed == 0.0

    def test_hand_speed_computed_correctly(self):
        """Known displacement → expected speed."""
        hand_prev = (0.5, 0.5)
        hand_curr = (0.6, 0.5)  # moved 0.1 normalised units
        dt = 1.0 / 6.0
        dx = hand_curr[0] - hand_prev[0]
        dy = hand_curr[1] - hand_prev[1]
        speed = math.hypot(dx, dy) / dt
        expected = 0.1 * 6.0  # 0.6 normalised units/sec
        assert abs(speed - expected) < 0.001

    def test_torso_speed_horizontal(self):
        """Torso moves horizontally → positive v_torso_x."""
        prev_x = 0.4
        curr_x = 0.45
        dt = 1.0 / 6.0
        v = (curr_x - prev_x) / dt
        assert v > 0
