"""
Unit tests for prediction model interfaces.

Tests DummyPredictionModel, dataclass defaults, XGBoostWinPredictor
(without model file), and helper functions.
"""

import pytest
import numpy as np

from prediction_model_base import (
    BallState, ScoreState, RallyState, PoseState,
    PredictionDataPacket, PredictionResult,
    WinPredictionModel, DummyPredictionModel,
)
from xgb_win_predictor import XGBoostWinPredictor, _safe_mean, _safe_std, _FEATURE_NAMES


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestDataclassDefaults:

    def test_ball_state_defaults(self):
        b = BallState()
        assert b.detected is False
        assert b.position_px is None
        assert b.position_m is None
        assert b.speed_mps is None
        assert b.speed_pps is None
        assert b.trajectory_px == []

    def test_score_state_defaults(self):
        s = ScoreState()
        assert s.player1_score is None
        assert s.player2_score is None
        assert s.player1_sets == 0
        assert s.player2_sets == 0
        assert s.score_reliable is True

    def test_rally_state_defaults(self):
        r = RallyState()
        assert r.is_active is False
        assert r.rally_id == 0
        assert r.rally_duration_sec == 0.0
        assert len(r.landing_zones) == 9

    def test_pose_state_defaults(self):
        p = PoseState()
        assert p.player1 is None
        assert p.player2 is None

    def test_prediction_data_packet_defaults(self):
        pkt = PredictionDataPacket()
        assert pkt.frame_idx == 0
        assert pkt.timestamp_sec == 0.0
        assert isinstance(pkt.ball, BallState)
        assert isinstance(pkt.score, ScoreState)
        assert isinstance(pkt.rally, RallyState)
        assert isinstance(pkt.pose, PoseState)
        assert pkt.match_elapsed_sec == 0.0

    def test_prediction_result_defaults(self):
        r = PredictionResult()
        assert r.player1_win_prob == 0.5
        assert r.player2_win_prob == 0.5
        assert r.confidence == 0.0
        assert r.model_ready is False

    def test_prediction_result_explicit_fields(self):
        r = PredictionResult(
            player1_win_prob=0.7,
            player2_win_prob=0.3,
            confidence=0.4,
            model_ready=True,
        )
        assert r.player1_win_prob == 0.7
        assert r.model_ready is True


# ---------------------------------------------------------------------------
# DummyPredictionModel
# ---------------------------------------------------------------------------

class TestDummyPredictionModel:

    def test_predict_returns_50_50(self, sample_packet):
        model = DummyPredictionModel()
        result = model.predict(sample_packet)
        assert abs(result.player1_win_prob - 0.5) < 1e-6
        assert abs(result.player2_win_prob - 0.5) < 1e-6

    def test_predict_model_ready_false(self, sample_packet):
        model = DummyPredictionModel()
        result = model.predict(sample_packet)
        assert result.model_ready is False

    def test_predict_confidence_zero(self, sample_packet):
        model = DummyPredictionModel()
        result = model.predict(sample_packet)
        assert result.confidence == 0.0

    def test_on_point_scored_does_not_raise(self, sample_packet):
        model = DummyPredictionModel()
        model.on_point_scored(sample_packet)  # default no-op

    def test_reset_does_not_raise(self):
        model = DummyPredictionModel()
        model.reset()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:

    def test_safe_mean_empty_list(self):
        assert _safe_mean([]) == 0.0

    def test_safe_mean_single_value(self):
        assert abs(_safe_mean([5.0]) - 5.0) < 1e-6

    def test_safe_mean_multiple_values(self):
        assert abs(_safe_mean([1.0, 2.0, 3.0]) - 2.0) < 1e-6

    def test_safe_std_empty_list(self):
        assert _safe_std([]) == 0.0

    def test_safe_std_single_value(self):
        assert _safe_std([5.0]) == 0.0

    def test_safe_std_known_values(self):
        result = _safe_std([1.0, 3.0])
        expected = np.std([1.0, 3.0])
        assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# XGBoostWinPredictor (without model file)
# ---------------------------------------------------------------------------

class TestXGBoostWinPredictorNoModel:

    def _make_predictor(self):
        """Create predictor pointing to a non-existent model file."""
        return XGBoostWinPredictor(model_path="/nonexistent/model.pkl")

    def test_predict_returns_cached_50_50_without_model(self, sample_packet):
        model = self._make_predictor()
        result = model.predict(sample_packet)
        assert abs(result.player1_win_prob - 0.5) < 1e-6
        assert result.model_ready is False

    def test_ingest_packet_pose_buffers_values(self, sample_packet):
        model = self._make_predictor()
        model._ingest_packet_pose(sample_packet)
        assert len(model._pose_p1_hand_speed) == 1
        assert abs(model._pose_p1_hand_speed[0] - 1.2) < 0.001
        assert len(model._pose_p2_hand_speed) == 1

    def test_ingest_packet_pose_with_none_pose(self):
        model = self._make_predictor()
        pkt = PredictionDataPacket()  # pose.player1 = None
        model._ingest_packet_pose(pkt)
        assert len(model._pose_p1_hand_speed) == 0

    def test_extract_rally_features_returns_8_keys(self, sample_packet):
        model = self._make_predictor()
        model._ingest_packet_pose(sample_packet)
        features = model._extract_rally_features(sample_packet)
        from xgb_win_predictor import _RALLY_FEATURE_KEYS
        for key in _RALLY_FEATURE_KEYS:
            assert key in features, f"Missing feature key: {key}"

    def test_build_feature_vector_none_when_no_history(self, sample_packet):
        model = self._make_predictor()
        result = model._build_feature_vector(sample_packet)
        assert result is None

    def test_build_feature_vector_19_features_after_history(self, sample_packet):
        model = self._make_predictor()
        # Add some rally history manually
        from xgb_win_predictor import _RALLY_FEATURE_KEYS
        dummy_rally = {k: 0.5 for k in _RALLY_FEATURE_KEYS}
        model._rally_history.append(dummy_rally)
        X = model._build_feature_vector(sample_packet)
        assert X is not None
        assert X.shape == (1, len(_FEATURE_NAMES))
        assert X.shape[1] == 19

    def test_reset_clears_history_and_cache(self, sample_packet):
        model = self._make_predictor()
        from xgb_win_predictor import _RALLY_FEATURE_KEYS
        dummy_rally = {k: 1.0 for k in _RALLY_FEATURE_KEYS}
        model._rally_history.append(dummy_rally)
        model._ingest_packet_pose(sample_packet)
        model.reset()
        assert len(model._rally_history) == 0
        assert len(model._pose_p1_hand_speed) == 0
        assert model._cached_result.model_ready is False

    def test_on_point_scored_without_model_does_not_raise(self, sample_packet):
        model = self._make_predictor()
        model.on_point_scored(sample_packet)  # should not raise even without model
