"""
Unit tests for RallyAggregator (ball_tracking_analysis.py).

Verifies state transitions, point-winner detection, landing zones,
and speed aggregation — all without YOLO or MediaPipe.
"""

import csv
import os
import pytest

from ball_tracking_analysis import (
    RallyAggregator, DataLogger, TableCalibration,
    RALLY_BALL_SEEN_FRAMES, RALLY_BALL_MISSING_FRAMES,
    TABLE_LENGTH_M, TABLE_WIDTH_M,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_aggregator(tmp_dir, with_calibration=False):
    logger = DataLogger(tmp_dir)
    tc = None
    if with_calibration:
        corners = [(200, 300), (1080, 300), (1080, 500), (200, 500)]
        tc = TableCalibration(corners)
    return RallyAggregator(fps=30.0, data_logger=logger, table_calibration=tc), logger


def _ball_track(x_m=1.37, y_m=0.76, speed=8.0):
    """Single track tuple (track_id, x_m, y_m, speed_mps)."""
    return [(1, x_m, y_m, speed)]


def _no_ball():
    return []


def _scores(p1, p2):
    return {"player1": p1, "player2": p2}


def _rounds(r1=0, r2=0):
    return {"player1": r1, "player2": r2}


def _add_frames(agg, n, tracks, scores, rounds):
    """Feed n identical frames to the aggregator."""
    for i in range(n):
        agg.add_frame(i, i / 30.0, tracks, scores, rounds)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestRallyStateTransitions:

    def test_initial_state_is_between_points(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        assert agg.state == RallyAggregator.STATE_BETWEEN_POINTS

    def test_transitions_to_rally_active_after_threshold(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        # Bootstrap with initial scores
        agg.add_frame(0, 0.0, _no_ball(), _scores(0, 0), _rounds())
        for i in range(1, RALLY_BALL_SEEN_FRAMES + 1):
            agg.add_frame(i, i / 30.0, _ball_track(), _scores(0, 0), _rounds())
        assert agg.state == RallyAggregator.STATE_RALLY_ACTIVE

    def test_no_early_transition(self, tmp_output_dir):
        """Fewer than RALLY_BALL_SEEN_FRAMES frames -> still between_points."""
        agg, _ = _make_aggregator(tmp_output_dir)
        agg.add_frame(0, 0.0, _no_ball(), _scores(0, 0), _rounds())
        for i in range(1, RALLY_BALL_SEEN_FRAMES):
            agg.add_frame(i, i / 30.0, _ball_track(), _scores(0, 0), _rounds())
        assert agg.state == RallyAggregator.STATE_BETWEEN_POINTS

    def test_returns_to_between_points_after_missing_frames(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        agg.add_frame(0, 0.0, _no_ball(), _scores(0, 0), _rounds())
        # Activate rally
        for i in range(1, RALLY_BALL_SEEN_FRAMES + 1):
            agg.add_frame(i, i / 30.0, _ball_track(), _scores(0, 0), _rounds())
        assert agg.state == RallyAggregator.STATE_RALLY_ACTIVE
        # Now remove ball for RALLY_BALL_MISSING_FRAMES
        base = RALLY_BALL_SEEN_FRAMES + 1
        for i in range(RALLY_BALL_MISSING_FRAMES):
            agg.add_frame(base + i, (base + i) / 30.0, _no_ball(), _scores(0, 0), _rounds())
        assert agg.state == RallyAggregator.STATE_BETWEEN_POINTS


# ---------------------------------------------------------------------------
# Point winner detection
# ---------------------------------------------------------------------------

class TestPointWinner:

    def test_p2_wins_point(self, tmp_output_dir):
        """Score changes (5,3) -> (5,4) -> winner is p2."""
        agg, logger = _make_aggregator(tmp_output_dir)
        agg.add_frame(0, 0.0, _no_ball(), _scores(5, 3), _rounds())
        agg.add_frame(1, 0.033, _no_ball(), _scores(5, 4), _rounds())
        with open(logger.rally_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["point_winner"] == "p2"

    def test_p1_wins_point(self, tmp_output_dir):
        """Score changes (5,3) -> (6,3) -> winner is p1."""
        agg, logger = _make_aggregator(tmp_output_dir)
        agg.add_frame(0, 0.0, _no_ball(), _scores(5, 3), _rounds())
        agg.add_frame(1, 0.033, _no_ball(), _scores(6, 3), _rounds())
        with open(logger.rally_file) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["point_winner"] == "p1"

    def test_simultaneous_score_change_unknown(self, tmp_output_dir):
        """Both scores increment -> unknown winner."""
        agg, logger = _make_aggregator(tmp_output_dir)
        agg.add_frame(0, 0.0, _no_ball(), _scores(5, 3), _rounds())
        agg.add_frame(1, 0.033, _no_ball(), _scores(6, 4), _rounds())
        with open(logger.rally_file) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["point_winner"] == "unknown"

    def test_multiple_points_multiple_rows(self, tmp_output_dir):
        """Three score changes -> three rally rows."""
        agg, logger = _make_aggregator(tmp_output_dir)
        agg.add_frame(0, 0.0, _no_ball(), _scores(0, 0), _rounds())
        agg.add_frame(1, 0.033, _no_ball(), _scores(1, 0), _rounds())
        agg.add_frame(2, 0.067, _no_ball(), _scores(2, 0), _rounds())
        agg.add_frame(3, 0.100, _no_ball(), _scores(2, 1), _rounds())
        with open(logger.rally_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# Landing zones
# ---------------------------------------------------------------------------

class TestLandingZones:

    def test_center_of_table_zone_4(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        cx = TABLE_LENGTH_M / 2
        cy = TABLE_WIDTH_M / 2
        idx = agg._landing_zone_index(cx, cy)
        assert idx == 4, f"Expected zone 4 for table center, got {idx}"

    def test_top_left_corner_zone_0(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        idx = agg._landing_zone_index(0.01, 0.01)
        assert idx == 0

    def test_top_right_corner_zone_2(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        idx = agg._landing_zone_index(TABLE_LENGTH_M - 0.01, 0.01)
        assert idx == 2

    def test_bottom_left_corner_zone_6(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        idx = agg._landing_zone_index(0.01, TABLE_WIDTH_M - 0.01)
        assert idx == 6

    def test_bottom_right_corner_zone_8(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        idx = agg._landing_zone_index(TABLE_LENGTH_M - 0.01, TABLE_WIDTH_M - 0.01)
        assert idx == 8

    def test_outside_table_returns_none(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        idx = agg._landing_zone_index(-1.0, -1.0)
        assert idx is None

    def test_all_nine_zones_reachable(self, tmp_output_dir):
        agg, _ = _make_aggregator(tmp_output_dir)
        found = set()
        step_x = TABLE_LENGTH_M / 3
        step_y = TABLE_WIDTH_M / 3
        for row in range(3):
            for col in range(3):
                x = step_x * col + step_x / 2
                y = step_y * row + step_y / 2
                idx = agg._landing_zone_index(x, y)
                if idx is not None:
                    found.add(idx)
        assert found == set(range(9)), f"Not all zones reachable: {found}"


# ---------------------------------------------------------------------------
# Speed aggregation and flush_final
# ---------------------------------------------------------------------------

class TestSpeedAggregation:

    def test_flush_final_emits_one_row(self, tmp_output_dir):
        agg, logger = _make_aggregator(tmp_output_dir)
        agg.add_frame(0, 0.0, _no_ball(), _scores(0, 0), _rounds())
        agg.flush_final(30, 1.0)
        with open(logger.rally_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_mean_speed_le_max_speed(self, tmp_output_dir):
        """When samples are present, mean_speed_mps <= max_speed_mps."""
        agg, logger = _make_aggregator(tmp_output_dir, with_calibration=True)
        agg.add_frame(0, 0.0, _no_ball(), _scores(0, 0), _rounds())
        # Activate rally
        for i in range(1, RALLY_BALL_SEEN_FRAMES + 1):
            agg.add_frame(i, i / 30.0, _ball_track(speed=8.0), _scores(0, 0), _rounds())
        # Score change to flush
        agg.add_frame(RALLY_BALL_SEEN_FRAMES + 2, 0.5, _no_ball(), _scores(1, 0), _rounds())
        with open(logger.rally_file) as f:
            rows = list(csv.DictReader(f))
        if rows and rows[0]["mean_speed_mps"]:
            mean = float(rows[0]["mean_speed_mps"])
            max_ = float(rows[0]["max_speed_mps"])
            assert mean <= max_ + 0.001, f"mean {mean} > max {max_}"
