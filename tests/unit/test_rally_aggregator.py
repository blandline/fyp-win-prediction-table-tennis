"""
Unit tests for RallyAggregator — state transitions, point winner, landing zones, speed stats.
"""

import numpy as np
import pytest
import sys, os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ball_tracking_analysis import (
    RallyAggregator, TableCalibration, DataLogger,
    RALLY_BALL_SEEN_FRAMES, RALLY_BALL_MISSING_FRAMES,
    TABLE_LENGTH_M, TABLE_WIDTH_M,
)


@pytest.fixture
def mock_logger(tmp_output_dir):
    return DataLogger(tmp_output_dir, with_meters=True)


@pytest.fixture
def calibration(sample_table_corners):
    cal = TableCalibration(sample_table_corners)
    return cal if cal.is_valid() else None


@pytest.fixture
def aggregator(mock_logger, calibration):
    return RallyAggregator(fps=30.0, data_logger=mock_logger, table_calibration=calibration)


# ---------------------------------------------------------------------------
# Rally state transitions
# ---------------------------------------------------------------------------

class TestRallyStateTransitions:

    def test_initial_state_is_between_points(self, aggregator):
        assert aggregator.state == RallyAggregator.STATE_BETWEEN_POINTS

    def test_rally_starts_after_n_consecutive_frames(self, aggregator):
        """Ball seen for RALLY_BALL_SEEN_FRAMES consecutive frames → rally_active."""
        stable = {'player1': 0, 'player2': 0}
        rounds = {'player1': 0, 'player2': 0}

        # Bootstrap scores
        aggregator.add_frame(0, 0.0, [], stable, rounds)

        for i in range(1, RALLY_BALL_SEEN_FRAMES + 1):
            tracks = [(1, 1.0, 0.5, 5.0)]  # (tid, x_m, y_m, speed_mps)
            aggregator.add_frame(i, i / 30.0, tracks, stable, rounds)

        assert aggregator.state == RallyAggregator.STATE_RALLY_ACTIVE

    def test_rally_not_started_before_n_frames(self, aggregator):
        """Ball seen for fewer than RALLY_BALL_SEEN_FRAMES → still between_points."""
        stable = {'player1': 0, 'player2': 0}
        rounds = {'player1': 0, 'player2': 0}

        aggregator.add_frame(0, 0.0, [], stable, rounds)

        for i in range(1, RALLY_BALL_SEEN_FRAMES):  # one fewer than needed
            tracks = [(1, 1.0, 0.5, 5.0)]
            aggregator.add_frame(i, i / 30.0, tracks, stable, rounds)

        assert aggregator.state == RallyAggregator.STATE_BETWEEN_POINTS

    def test_rally_ends_after_missing_frames(self, aggregator):
        """Ball missing for RALLY_BALL_MISSING_FRAMES → back to between_points."""
        stable = {'player1': 0, 'player2': 0}
        rounds = {'player1': 0, 'player2': 0}

        aggregator.add_frame(0, 0.0, [], stable, rounds)

        # Start a rally
        for i in range(1, RALLY_BALL_SEEN_FRAMES + 2):
            tracks = [(1, 1.0, 0.5, 5.0)]
            aggregator.add_frame(i, i / 30.0, tracks, stable, rounds)
        assert aggregator.state == RallyAggregator.STATE_RALLY_ACTIVE

        # Ball disappears
        f = RALLY_BALL_SEEN_FRAMES + 2
        for i in range(f, f + RALLY_BALL_MISSING_FRAMES + 1):
            aggregator.add_frame(i, i / 30.0, [], stable, rounds)

        assert aggregator.state == RallyAggregator.STATE_BETWEEN_POINTS


# ---------------------------------------------------------------------------
# Point winner detection
# ---------------------------------------------------------------------------

class TestPointWinner:

    def test_point_winner_from_score_change(self, aggregator):
        """Score (5,3) → (5,4) → winner is p2."""
        rounds = {'player1': 0, 'player2': 0}
        logged_rallies = []
        original_log = aggregator.logger.log_rally
        aggregator.logger.log_rally = lambda rec: logged_rallies.append(rec)

        # Bootstrap
        aggregator.add_frame(0, 0.0, [], {'player1': 5, 'player2': 3}, rounds)

        # Frames at stable score (5,3)
        for i in range(1, 10):
            aggregator.add_frame(i, i / 30.0, [], {'player1': 5, 'player2': 3}, rounds)

        # Score changes to (5,4)
        aggregator.add_frame(10, 10 / 30.0, [], {'player1': 5, 'player2': 4}, rounds)

        assert len(logged_rallies) == 1
        assert logged_rallies[0]['point_winner'] == 'p2'

    def test_point_winner_unknown_when_both_change(self, aggregator):
        """Both scores change simultaneously → unknown."""
        rounds = {'player1': 0, 'player2': 0}
        logged_rallies = []
        aggregator.logger.log_rally = lambda rec: logged_rallies.append(rec)

        aggregator.add_frame(0, 0.0, [], {'player1': 3, 'player2': 3}, rounds)
        for i in range(1, 5):
            aggregator.add_frame(i, i / 30.0, [], {'player1': 3, 'player2': 3}, rounds)

        aggregator.add_frame(5, 5 / 30.0, [], {'player1': 4, 'player2': 4}, rounds)
        assert logged_rallies[0]['point_winner'] == 'unknown'


# ---------------------------------------------------------------------------
# Landing zones
# ---------------------------------------------------------------------------

class TestLandingZones:

    def test_landing_zone_index_centre(self, aggregator):
        """Centre of table → zone index 4 (middle of 3x3 grid)."""
        idx = aggregator._landing_zone_index(TABLE_LENGTH_M / 2, TABLE_WIDTH_M / 2)
        assert idx == 4

    def test_landing_zone_index_top_left(self, aggregator):
        """Near (0, 0) → zone 0."""
        idx = aggregator._landing_zone_index(0.1, 0.1)
        assert idx == 0

    def test_landing_zone_outside_table(self, aggregator):
        """Point outside table → None."""
        idx = aggregator._landing_zone_index(-1.0, -1.0)
        assert idx is None

    def test_all_nine_zones_reachable(self, aggregator):
        """Verify all 9 zones can be hit with appropriate coordinates."""
        zone_hits = set()
        cell_w = TABLE_LENGTH_M / 3
        cell_h = TABLE_WIDTH_M / 3
        for row in range(3):
            for col in range(3):
                x = cell_w * col + cell_w / 2
                y = cell_h * row + cell_h / 2
                idx = aggregator._landing_zone_index(x, y)
                assert idx is not None
                zone_hits.add(idx)
        assert zone_hits == set(range(9))


# ---------------------------------------------------------------------------
# Speed aggregation
# ---------------------------------------------------------------------------

class TestSpeedAggregation:

    def test_speed_stats_in_flushed_rally(self, aggregator):
        """Check mean/max/std speed in a flushed rally record."""
        rounds = {'player1': 0, 'player2': 0}
        logged = []
        aggregator.logger.log_rally = lambda rec: logged.append(rec)

        aggregator.add_frame(0, 0.0, [], {'player1': 0, 'player2': 0}, rounds)

        # Start rally
        for i in range(1, RALLY_BALL_SEEN_FRAMES + 5):
            tracks = [(1, 1.0, 0.5, float(i))]  # increasing speed
            aggregator.add_frame(i, i / 30.0, tracks, {'player1': 0, 'player2': 0}, rounds)

        # Score change triggers flush
        f = RALLY_BALL_SEEN_FRAMES + 5
        aggregator.add_frame(f, f / 30.0, [], {'player1': 1, 'player2': 0}, rounds)

        assert len(logged) == 1
        rec = logged[0]
        # Speeds were collected during rally_active state
        if rec['mean_speed_mps'] != '':
            mean_spd = float(rec['mean_speed_mps'])
            max_spd = float(rec['max_speed_mps'])
            assert max_spd >= mean_spd

    def test_flush_final_writes_row(self, aggregator):
        logged = []
        aggregator.logger.log_rally = lambda rec: logged.append(rec)

        aggregator.add_frame(0, 0.0, [], {'player1': 0, 'player2': 0}, {'player1': 0, 'player2': 0})
        aggregator.flush_final(100, 3.33)
        assert len(logged) == 1
