"""
Unit tests for SceneClassifier (broadcast_utils/scene_classifier.py).

Tests state transitions, cut detection, table-color heuristic, and
the should_process property — all using synthetic frames.
"""

import pytest
import numpy as np

from broadcast_utils.scene_classifier import (
    SceneClassifier, GAMEPLAY, CUT_DETECTED, NON_GAMEPLAY,
    CUT_THRESHOLD, POLL_INTERVAL, MIN_TABLE_FRACTION,
)
from tests.fixtures.synthetic_frames import (
    make_gameplay_frame, make_non_gameplay_frame,
    make_blank_frame, make_large_diff_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed(classifier, frame, n=1):
    """Feed the same frame n times and return the last state."""
    state = None
    for _ in range(n):
        state = classifier.update(frame)
    return state


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:

    def test_initial_state_is_gameplay(self):
        sc = SceneClassifier("blue")
        assert sc.state == GAMEPLAY

    def test_should_process_true_initially(self):
        sc = SceneClassifier("blue")
        assert sc.should_process is True

    def test_is_cut_false_initially(self):
        sc = SceneClassifier("blue")
        assert sc.is_cut is False


# ---------------------------------------------------------------------------
# Stable gameplay
# ---------------------------------------------------------------------------

class TestStableGameplay:

    def test_identical_frames_stay_gameplay(self):
        """Identical frames produce zero diff -> stays GAMEPLAY."""
        sc = SceneClassifier("blue")
        frame = make_gameplay_frame(640, 480, "blue")
        sc.update(frame)  # first frame (no prev)
        for _ in range(5):
            state = sc.update(frame)
        assert state == GAMEPLAY

    def test_should_process_true_in_gameplay(self):
        sc = SceneClassifier("blue")
        frame = make_gameplay_frame(640, 480, "blue")
        sc.update(frame)
        sc.update(frame)
        assert sc.should_process is True


# ---------------------------------------------------------------------------
# Cut detection
# ---------------------------------------------------------------------------

class TestCutDetection:

    def test_large_diff_triggers_cut_detected(self):
        """Inverted frame has max diff -> transitions to CUT_DETECTED."""
        sc = SceneClassifier("blue")
        frame = make_gameplay_frame(640, 480, "blue")
        sc.update(frame)
        inverted = make_large_diff_frame(frame)
        state = sc.update(inverted)
        assert state == CUT_DETECTED

    def test_cut_to_gameplay_sets_is_cut(self):
        """After CUT_DETECTED, a frame with table color -> GAMEPLAY + is_cut=True."""
        sc = SceneClassifier("blue")
        frame = make_gameplay_frame(640, 480, "blue")
        sc.update(frame)
        sc.update(make_large_diff_frame(frame))  # -> CUT_DETECTED
        gameplay_frame = make_gameplay_frame(640, 480, "blue")
        sc.update(gameplay_frame)
        assert sc.state == GAMEPLAY
        assert sc.is_cut is True

    def test_cut_to_non_gameplay(self):
        """After CUT_DETECTED, a frame without table color -> NON_GAMEPLAY."""
        sc = SceneClassifier("blue")
        frame = make_gameplay_frame(640, 480, "blue")
        sc.update(frame)
        sc.update(make_large_diff_frame(frame))  # -> CUT_DETECTED
        non_gp = make_non_gameplay_frame(640, 480)
        sc.update(non_gp)
        assert sc.state == NON_GAMEPLAY

    def test_is_cut_false_on_second_gameplay_frame(self):
        """is_cut should be True only on the first frame after returning to GAMEPLAY."""
        sc = SceneClassifier("blue")
        frame = make_gameplay_frame(640, 480, "blue")
        sc.update(frame)
        sc.update(make_large_diff_frame(frame))
        gameplay_frame = make_gameplay_frame(640, 480, "blue")
        sc.update(gameplay_frame)  # is_cut = True here
        sc.update(gameplay_frame)  # is_cut should reset to False
        assert sc.is_cut is False


# ---------------------------------------------------------------------------
# Non-gameplay polling
# ---------------------------------------------------------------------------

class TestNonGameplayPolling:

    def test_non_gameplay_polls_at_interval(self):
        """After POLL_INTERVAL frames in NON_GAMEPLAY with table color, returns to GAMEPLAY."""
        sc = SceneClassifier("blue")
        frame = make_gameplay_frame(640, 480, "blue")
        sc.update(frame)
        sc.update(make_large_diff_frame(frame))  # CUT_DETECTED
        sc.update(make_non_gameplay_frame(640, 480))  # NON_GAMEPLAY
        assert sc.state == NON_GAMEPLAY

        gameplay_frame = make_gameplay_frame(640, 480, "blue")
        # Feed POLL_INTERVAL - 1 non-gameplay frames (no poll yet)
        for i in range(POLL_INTERVAL - 1):
            sc.update(make_non_gameplay_frame(640, 480))
        # On the POLL_INTERVAL-th frame, feed a gameplay frame
        sc.update(gameplay_frame)
        assert sc.state == GAMEPLAY
        assert sc.is_cut is True

    def test_should_process_false_in_non_gameplay(self):
        sc = SceneClassifier("blue")
        frame = make_gameplay_frame(640, 480, "blue")
        sc.update(frame)
        sc.update(make_large_diff_frame(frame))
        sc.update(make_non_gameplay_frame(640, 480))
        assert sc.should_process is False


# ---------------------------------------------------------------------------
# Green table preset
# ---------------------------------------------------------------------------

class TestGreenTablePreset:

    def test_green_table_recognized(self):
        sc = SceneClassifier("green")
        frame = make_gameplay_frame(640, 480, "green")
        sc.update(frame)
        sc.update(make_large_diff_frame(frame))  # CUT_DETECTED
        sc.update(make_gameplay_frame(640, 480, "green"))
        assert sc.state == GAMEPLAY
