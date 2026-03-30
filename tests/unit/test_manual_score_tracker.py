"""
Unit tests for ManualScoreTracker (ball_tracking_fast.py).

Verifies score/set adjustment, clamping, swapping, and initial values.
No YOLO or video required.
"""

import pytest
import numpy as np

from ball_tracking_fast import ManualScoreTracker
from tests.fixtures.synthetic_frames import make_blank_frame


class TestManualScoreTrackerDefaults:

    def test_default_scores_zero(self):
        tracker = ManualScoreTracker()
        assert tracker.current_scores["player1"] == 0
        assert tracker.current_scores["player2"] == 0

    def test_default_rounds_zero(self):
        tracker = ManualScoreTracker()
        assert tracker.rounds["player1"] == 0
        assert tracker.rounds["player2"] == 0

    def test_default_stable_scores_zero(self):
        tracker = ManualScoreTracker()
        assert tracker.stable_scores["player1"] == 0
        assert tracker.stable_scores["player2"] == 0

    def test_stop_does_not_raise(self):
        tracker = ManualScoreTracker()
        tracker.stop()  # should be a no-op


class TestManualScoreTrackerAdjust:

    def test_adjust_player1_increment(self):
        tracker = ManualScoreTracker()
        tracker.adjust("player1", +1)
        assert tracker.current_scores["player1"] == 1

    def test_adjust_player2_increment(self):
        tracker = ManualScoreTracker()
        tracker.adjust("player2", +1)
        assert tracker.current_scores["player2"] == 1

    def test_adjust_decrement_clamps_at_zero(self):
        tracker = ManualScoreTracker()
        tracker.adjust("player1", -1)
        assert tracker.current_scores["player1"] == 0

    def test_adjust_multiple_increments(self):
        tracker = ManualScoreTracker()
        for _ in range(7):
            tracker.adjust("player1", +1)
        assert tracker.current_scores["player1"] == 7

    def test_stable_scores_mirror_current_after_adjust(self):
        tracker = ManualScoreTracker()
        tracker.adjust("player1", +3)
        assert tracker.stable_scores["player1"] == tracker.current_scores["player1"]


class TestManualScoreTrackerRounds:

    def test_adjust_rounds_increments(self):
        tracker = ManualScoreTracker()
        tracker.adjust_rounds("player1", +1)
        assert tracker.rounds["player1"] == 1

    def test_adjust_rounds_resets_point_scores(self):
        tracker = ManualScoreTracker()
        tracker.adjust("player1", +5)
        tracker.adjust("player2", +3)
        tracker.adjust_rounds("player1", +1)
        assert tracker.current_scores["player1"] == 0
        assert tracker.current_scores["player2"] == 0
        assert tracker.stable_scores["player1"] == 0

    def test_adjust_rounds_clamps_at_zero(self):
        tracker = ManualScoreTracker()
        tracker.adjust_rounds("player1", -1)
        assert tracker.rounds["player1"] == 0


class TestManualScoreTrackerSwap:

    def test_swap_scores_exchanges_values(self):
        tracker = ManualScoreTracker(
            initial_scores={"player1": 5, "player2": 3},
            initial_rounds={"player1": 1, "player2": 0},
        )
        tracker.swap_scores()
        assert tracker.current_scores["player1"] == 3
        assert tracker.current_scores["player2"] == 5
        assert tracker.rounds["player1"] == 0
        assert tracker.rounds["player2"] == 1

    def test_swap_updates_stable_scores(self):
        tracker = ManualScoreTracker(
            initial_scores={"player1": 4, "player2": 2},
        )
        tracker.swap_scores()
        assert tracker.stable_scores["player1"] == 2
        assert tracker.stable_scores["player2"] == 4


class TestManualScoreTrackerInitialValues:

    def test_initial_scores_honored(self):
        tracker = ManualScoreTracker(
            initial_scores={"player1": 7, "player2": 5},
        )
        assert tracker.current_scores["player1"] == 7
        assert tracker.current_scores["player2"] == 5

    def test_initial_rounds_honored(self):
        tracker = ManualScoreTracker(
            initial_rounds={"player1": 2, "player2": 1},
        )
        assert tracker.rounds["player1"] == 2
        assert tracker.rounds["player2"] == 1

    def test_draw_scores_does_not_raise(self):
        tracker = ManualScoreTracker()
        frame = make_blank_frame(1280, 720)
        rois = {}
        tracker.draw_scores(frame, rois, player_names=["Alice", "Bob"])
