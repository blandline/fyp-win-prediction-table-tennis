"""
Unit tests for score detection: digit parsing, voting, stability, obscuration.
"""

import numpy as np
import pytest
import sys, os
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ball_tracking_fast import _parse_digit_result, ScoreDetectorBatched, SCORE_STABLE_RUNS
from ball_tracking_analysis import MIN_DIGIT_CONF_RELIABLE, DIGIT_CONF_THRESHOLD
from tests.fixtures.synthetic_frames import MockBox, MockBoxes


# ---------------------------------------------------------------------------
# _parse_digit_result tests
# ---------------------------------------------------------------------------

class TestParseDigitResult:

    def test_single_digit(self):
        boxes = MockBoxes([MockBox(10, 5, 30, 40, 5, 0.90)])
        score, n, mean_c, min_c = _parse_digit_result(boxes)
        assert score == 5
        assert n == 1
        assert abs(mean_c - 0.90) < 0.01

    def test_two_digits_make_11(self):
        boxes = MockBoxes([
            MockBox(10, 5, 25, 40, 1, 0.92),
            MockBox(30, 5, 45, 40, 1, 0.88),
        ])
        score, n, mean_c, min_c = _parse_digit_result(boxes)
        assert score == 11
        assert n == 2
        assert abs(min_c - 0.88) < 0.01

    def test_digit_ordering_by_x(self):
        """Detections given in reverse x order should still yield correct score."""
        boxes = MockBoxes([
            MockBox(50, 5, 65, 40, 3, 0.90),  # right digit: 3
            MockBox(10, 5, 25, 40, 1, 0.85),  # left digit: 1
        ])
        score, n, _, _ = _parse_digit_result(boxes)
        assert score == 13  # "1" then "3"

    def test_non_digit_classes_ignored(self):
        """Classes > 9 (e.g. 'l', 'undefined') should be filtered out."""
        boxes = MockBoxes([
            MockBox(10, 5, 30, 40, 10, 0.90),  # class 10 = 'l'
            MockBox(40, 5, 60, 40, 7, 0.85),   # class 7 = digit
        ])
        score, n, _, _ = _parse_digit_result(boxes)
        assert score == 7
        assert n == 1

    def test_score_above_30_rejected(self):
        """Score > 30 should return None."""
        boxes = MockBoxes([
            MockBox(10, 5, 25, 40, 3, 0.90),
            MockBox(30, 5, 45, 40, 5, 0.90),
        ])
        score, _, _, _ = _parse_digit_result(boxes, max_score=30)
        assert score is None  # 35 > 30

    def test_empty_boxes(self):
        boxes = MockBoxes([])
        score, n, _, _ = _parse_digit_result(boxes)
        assert score is None
        assert n == 0

    def test_zero_score(self):
        boxes = MockBoxes([MockBox(10, 5, 30, 40, 0, 0.90)])
        score, _, _, _ = _parse_digit_result(boxes)
        assert score == 0

    def test_max_two_digits_used(self):
        """Only first 2 digits (by x) should be used even if 3 detected."""
        boxes = MockBoxes([
            MockBox(10, 5, 25, 40, 1, 0.90),
            MockBox(30, 5, 45, 40, 2, 0.90),
            MockBox(50, 5, 65, 40, 3, 0.90),
        ])
        score, n, _, _ = _parse_digit_result(boxes)
        assert score == 12  # first two digits: "1" "2"
        assert n == 3  # all 3 detected


# ---------------------------------------------------------------------------
# Score voting and stability tests
# ---------------------------------------------------------------------------

class TestScoreVotingAndStability:

    def _make_detector_state(self):
        """Create a minimal ScoreDetectorBatched-like object for testing voting logic."""

        class FakeScoreDetector:
            def __init__(self):
                self.score_history = {'player1': deque(maxlen=5), 'player2': deque(maxlen=5)}
                self.current_scores = {'player1': None, 'player2': None}
                self.stable_scores = {'player1': None, 'player2': None}
                self._stable_candidate = {'player1': None, 'player2': None}
                self._stable_run = {'player1': 0, 'player2': 0}

        return FakeScoreDetector()

    def _update_score(self, det, player, score_val, min_conf=0.90):
        """Simulate one score-detector cycle for a player."""
        from collections import Counter

        reliable = min_conf >= MIN_DIGIT_CONF_RELIABLE
        if reliable and score_val is not None:
            det.score_history[player].append(score_val)
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

    def test_voting_requires_majority(self):
        det = self._make_detector_state()
        for val in [5, 5, 4]:
            self._update_score(det, 'player1', val)
        assert det.current_scores['player1'] == 5

    def test_no_vote_without_three_readings(self):
        det = self._make_detector_state()
        self._update_score(det, 'player1', 5)
        self._update_score(det, 'player1', 5)
        # Only 2 readings → not enough for majority vote
        assert det.current_scores['player1'] is None

    def test_stability_requires_four_runs(self):
        det = self._make_detector_state()
        # Need at least 3 history entries for voting to kick in,
        # then SCORE_STABLE_RUNS consistent votes for stability.
        # Feed enough runs: first 3 build history, then stable_run counts from 1.
        # run 1-2: history < 3, no vote → stable_run stays 0
        # run 3: history=[7,7,7], vote=7, stable_run=1
        # run 4: stable_run=2
        # run 5: stable_run=3
        # run 6: stable_run=4 → committed
        for _ in range(5):
            self._update_score(det, 'player1', 7)
        assert det.stable_scores['player1'] is None  # stable_run=3, needs 4

        self._update_score(det, 'player1', 7)
        assert det.stable_scores['player1'] == 7

    def test_stability_resets_on_change(self):
        det = self._make_detector_state()
        # 6 runs to reach stability for score=5
        for _ in range(6):
            self._update_score(det, 'player1', 5)
        assert det.stable_scores['player1'] == 5

        # Feed enough 6s to make the majority vote flip to 6
        # History deque(maxlen=5): after 3 sixes → [5, 5, 6, 6, 6] → vote=6
        for _ in range(3):
            self._update_score(det, 'player1', 6)

        # The voted value changed from 5→6, so stable_run should be low (restarted)
        assert det._stable_candidate['player1'] == 6
        assert det._stable_run['player1'] < SCORE_STABLE_RUNS

    def test_confidence_below_reliable_threshold_ignored(self):
        det = self._make_detector_state()
        for _ in range(5):
            self._update_score(det, 'player1', 5, min_conf=0.30)  # below 0.45
        assert det.current_scores['player1'] is None
        assert det._stable_run['player1'] == 0
