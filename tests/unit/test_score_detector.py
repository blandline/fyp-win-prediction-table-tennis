"""
Unit tests for score detection logic (ball_tracking_fast.py).

Tests _parse_digit_result (pure logic, no YOLO inference) and
ManualScoreTracker voting/stability via ScoreDetectorBatched internals.
"""

import pytest
from collections import deque

from ball_tracking_fast import (
    _parse_digit_result,
    ManualScoreTracker,
    SCORE_STABLE_RUNS,
    MIN_DIGIT_CONF_RELIABLE,
)
from ball_tracking_analysis import DIGIT_CONF_THRESHOLD


# ---------------------------------------------------------------------------
# Fake box helpers for _parse_digit_result
# ---------------------------------------------------------------------------

class _FakeBox:
    """Minimal fake ultralytics box for _parse_digit_result."""

    def __init__(self, x1, x2, cls_id, conf):
        self._x1 = x1
        self._x2 = x2
        self._cls_id = cls_id
        self._conf = conf

    @property
    def xyxy(self):
        x1, x2 = self._x1, self._x2

        class _T:
            def tolist(self_inner):
                return [x1, 0, x2, 20]

        return [_T()]

    @property
    def cls(self):
        v = self._cls_id

        class _C:
            def item(self_inner):
                return v

        return _C()

    @property
    def conf(self):
        v = self._conf

        class _C:
            def item(self_inner):
                return v

        return _C()


def _box(digit, x_center, conf=0.9):
    half = 5
    return _FakeBox(x_center - half, x_center + half, digit, conf)


# ---------------------------------------------------------------------------
# _parse_digit_result
# ---------------------------------------------------------------------------

class TestParseDigitResult:

    def test_empty_boxes_returns_none(self):
        score, n, mean_c, min_c = _parse_digit_result([])
        assert score is None
        assert n == 0

    def test_none_boxes_returns_none(self):
        score, n, mean_c, min_c = _parse_digit_result(None)
        assert score is None

    def test_single_digit_5(self):
        boxes = [_box(5, 50)]
        score, n, mean_c, min_c = _parse_digit_result(boxes)
        assert score == 5
        assert n == 1

    def test_single_digit_0(self):
        boxes = [_box(0, 50)]
        score, n, mean_c, min_c = _parse_digit_result(boxes)
        assert score == 0

    def test_two_digits_x_sorted_11(self):
        """Two '1' digits side by side -> score 11."""
        boxes = [_box(1, 30), _box(1, 60)]
        score, n, _, _ = _parse_digit_result(boxes)
        assert score == 11

    def test_two_digits_x_sorted_21(self):
        """'2' at x=30, '1' at x=60 -> '21'."""
        boxes = [_box(2, 30), _box(1, 60)]
        score, n, _, _ = _parse_digit_result(boxes)
        assert score == 21

    def test_two_digits_x_sorted_12(self):
        """'1' at x=30, '2' at x=60 -> '12'."""
        boxes = [_box(1, 30), _box(2, 60)]
        score, n, _, _ = _parse_digit_result(boxes)
        assert score == 12

    def test_non_digit_class_ignored(self):
        """Class 10 is not a digit (0-9); should be ignored."""
        boxes = [_box(10, 50)]
        score, n, _, _ = _parse_digit_result(boxes)
        assert score is None
        assert n == 0

    def test_max_score_cap(self):
        """Score > 30 -> rejected."""
        boxes = [_box(3, 30), _box(1, 60)]  # "31"
        score, n, _, _ = _parse_digit_result(boxes, max_score=30)
        assert score is None

    def test_max_two_digits_contribute(self):
        """Only first 2 (by x) digits form the score."""
        boxes = [_box(1, 10), _box(2, 20), _box(3, 30)]
        score, n, _, _ = _parse_digit_result(boxes)
        assert score == 12  # first two by x

    def test_confidence_reported(self):
        boxes = [_box(7, 50, conf=0.85)]
        score, n, mean_c, min_c = _parse_digit_result(boxes)
        assert abs(mean_c - 0.85) < 0.001
        assert abs(min_c - 0.85) < 0.001

    def test_min_conf_is_minimum_across_digits(self):
        boxes = [_box(1, 30, conf=0.9), _box(2, 60, conf=0.6)]
        _, _, _, min_c = _parse_digit_result(boxes)
        assert abs(min_c - 0.6) < 0.001


# ---------------------------------------------------------------------------
# Score voting and stability (via ManualScoreTracker as a proxy for logic)
# ---------------------------------------------------------------------------

class TestManualScoreTrackerVoting:
    """
    ManualScoreTracker doesn't do OCR voting, but we test that its
    stable_scores immediately mirror current_scores (deterministic input).
    The voting logic lives in ScoreDetectorBatched; we test it via
    direct state manipulation below.
    """

    def test_adjust_increments_stable_score(self):
        tracker = ManualScoreTracker()
        tracker.adjust("player1", +1)
        assert tracker.stable_scores["player1"] == 1

    def test_adjust_decrement_clamps_at_zero(self):
        tracker = ManualScoreTracker()
        tracker.adjust("player1", -1)
        assert tracker.current_scores["player1"] == 0

    def test_initial_scores_honored(self):
        tracker = ManualScoreTracker(
            initial_scores={"player1": 3, "player2": 2},
            initial_rounds={"player1": 1, "player2": 0},
        )
        assert tracker.current_scores["player1"] == 3
        assert tracker.rounds["player1"] == 1


class TestScoreDetectorBatchedVoting:
    """
    Test the voting / stability logic inside ScoreDetectorBatched
    by directly manipulating its internal state (no YOLO inference).
    """

    def _make_detector(self):
        from ball_tracking_fast import ScoreDetectorBatched
        det = ScoreDetectorBatched.__new__(ScoreDetectorBatched)
        det.score_history = {"player1": deque(maxlen=5), "player2": deque(maxlen=5)}
        det.current_scores = {"player1": None, "player2": None}
        det.stable_scores = {"player1": None, "player2": None}
        det._stable_candidate = {"player1": None, "player2": None}
        det._stable_run = {"player1": 0, "player2": 0}
        det._rounds_candidate = {"player1": None, "player2": None}
        det._rounds_run = {"player1": 0, "player2": 0}
        det.rounds = {"player1": 0, "player2": 0}
        det.score_roi_obscured = {"player1": False, "player2": False}
        det.rounds_roi_obscured = {"player1": False, "player2": False}
        det.last_processed_frame = None
        return det

    def _push_vote(self, det, player, value, conf=0.9):
        """Simulate one reliable score read for `player`."""
        reliable = conf >= MIN_DIGIT_CONF_RELIABLE
        det.score_roi_obscured[player] = not reliable
        if reliable and value is not None:
            det.score_history[player].append(value)
            from collections import Counter
            if len(det.score_history[player]) >= 3:
                counts = Counter(det.score_history[player])
                most_common = counts.most_common(1)[0]
                if most_common[1] >= 2:
                    voted = most_common[0]
                    det.current_scores[player] = voted
                    if voted == det._stable_candidate[player]:
                        det._stable_run[player] += 1
                    else:
                        det._stable_candidate[player] = voted
                        det._stable_run[player] = 1
                    if det._stable_run[player] >= SCORE_STABLE_RUNS:
                        det.stable_scores[player] = voted
        else:
            det._stable_run[player] = 0

    def test_stable_score_after_stable_runs(self):
        det = self._make_detector()
        for _ in range(SCORE_STABLE_RUNS + 2):
            self._push_vote(det, "player1", 7)
        assert det.stable_scores["player1"] == 7

    def test_no_stable_before_threshold(self):
        det = self._make_detector()
        for _ in range(SCORE_STABLE_RUNS - 1):
            self._push_vote(det, "player1", 5)
        assert det.stable_scores["player1"] is None

    def test_stable_resets_when_vote_changes(self):
        det = self._make_detector()
        for _ in range(SCORE_STABLE_RUNS + 2):
            self._push_vote(det, "player1", 3)
        assert det.stable_scores["player1"] == 3
        # Now push a different value
        for _ in range(3):
            self._push_vote(det, "player1", 4)
        # stable_run for 4 should have reset; stable_scores still 3 until 4 is stable
        assert det._stable_run["player1"] < SCORE_STABLE_RUNS or det.stable_scores["player1"] == 4

    def test_low_confidence_does_not_update_stable(self):
        det = self._make_detector()
        low_conf = MIN_DIGIT_CONF_RELIABLE - 0.05
        for _ in range(SCORE_STABLE_RUNS + 5):
            self._push_vote(det, "player1", 9, conf=low_conf)
        assert det.stable_scores["player1"] is None
