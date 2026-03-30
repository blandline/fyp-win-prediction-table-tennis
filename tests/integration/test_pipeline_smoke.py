"""
Integration tests for the prediction pipeline.

Tests source parsing, packet building with mock components,
prediction model contract, and (optionally) a headless end-to-end run.
"""

import os
import sys
import csv
import pytest
import numpy as np

from prediction_model_base import (
    BallState, ScoreState, RallyState, PoseState,
    PredictionDataPacket, PredictionResult,
    DummyPredictionModel,
)
from prediction_pipeline import parse_source, build_packet


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------

class TestSourceParsing:

    def test_integer_source_is_live(self):
        src, is_live = parse_source("0")
        assert src == 0
        assert is_live is True

    def test_integer_source_1(self):
        src, is_live = parse_source("1")
        assert src == 1
        assert is_live is True

    def test_rtsp_source_is_live(self):
        src, is_live = parse_source("rtsp://192.168.1.1/stream")
        assert src == "rtsp://192.168.1.1/stream"
        assert is_live is True

    def test_http_source_is_live(self):
        src, is_live = parse_source("http://example.com/stream.m3u8")
        assert is_live is True

    def test_file_path_is_not_live(self):
        src, is_live = parse_source("path/to/video.mp4")
        assert src == "path/to/video.mp4"
        assert is_live is False

    def test_windows_file_path_is_not_live(self):
        src, is_live = parse_source(r"C:\Users\user\video.mp4")
        assert is_live is False


# ---------------------------------------------------------------------------
# Prediction model base contract
# ---------------------------------------------------------------------------

class TestPredictionModelBase:

    def test_dummy_model_returns_valid_result(self, sample_packet):
        model = DummyPredictionModel()
        result = model.predict(sample_packet)
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.player1_win_prob <= 1.0
        assert 0.0 <= result.player2_win_prob <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.model_ready is False

    def test_probabilities_sum_to_one(self, sample_packet):
        model = DummyPredictionModel()
        result = model.predict(sample_packet)
        assert abs(result.player1_win_prob + result.player2_win_prob - 1.0) < 1e-6

    def test_data_packet_defaults(self):
        pkt = PredictionDataPacket()
        assert pkt.frame_idx == 0
        assert pkt.ball.detected is False
        assert pkt.score.player1_score is None
        assert pkt.rally.is_active is False
        assert pkt.pose.player1 is None


# ---------------------------------------------------------------------------
# build_packet
# ---------------------------------------------------------------------------

class _MockBallTracker:
    """Minimal mock for build_packet."""
    def __init__(self):
        self.trajectories = {}

    def get_position_meters(self, tid):
        return None, None

    def get_smoothed_speed_mps(self, tid):
        return None

    def get_smoothed_speed(self, tid):
        return 0.0


class _MockScoreDetector:
    def __init__(self, p1=None, p2=None, r1=0, r2=0):
        self.current_scores = {"player1": p1, "player2": p2}
        self.stable_scores = {"player1": p1, "player2": p2}
        self.rounds = {"player1": r1, "player2": r2}
        self.score_roi_obscured = {"player1": False, "player2": False}


class _MockRallyAggregator:
    """Minimal mock for build_packet."""
    from ball_tracking_analysis import RallyAggregator
    STATE_BETWEEN_POINTS = "between_points"
    STATE_RALLY_ACTIVE = "rally_active"
    rally_id = 0
    rally_start_time = None
    samples = []

    def get_state_for_display(self):
        return {"state": self.STATE_BETWEEN_POINTS, "rally_id": 0}

    def _landing_zone_index(self, x_m, y_m):
        return None


class TestBuildPacket:

    def test_build_packet_no_detections(self):
        import time
        tracker = _MockBallTracker()
        score_det = _MockScoreDetector(p1=3, p2=1)
        rally_agg = _MockRallyAggregator()
        packet = build_packet(
            frame_idx=60,
            fps=30.0,
            ball_tracker=tracker,
            tracks=[],
            score_detector=score_det,
            rally_aggregator=rally_agg,
            pose_p1=None,
            pose_p2=None,
            start_time=time.time() - 5.0,
        )
        assert packet.frame_idx == 60
        assert abs(packet.timestamp_sec - 2.0) < 0.01
        assert packet.ball.detected is False
        assert packet.ball.position_px is None
        assert packet.score.player1_score == 3
        assert packet.score.player2_score == 1
        assert packet.rally.is_active is False

    def test_build_packet_with_pose(self):
        import time
        tracker = _MockBallTracker()
        score_det = _MockScoreDetector()
        rally_agg = _MockRallyAggregator()
        pose_p1 = {
            "v_hand_speed": 1.5,
            "debug_landmarks": [(0, 0)],  # should be stripped
        }
        packet = build_packet(
            frame_idx=30,
            fps=30.0,
            ball_tracker=tracker,
            tracks=[],
            score_detector=score_det,
            rally_aggregator=rally_agg,
            pose_p1=pose_p1,
            pose_p2=None,
            start_time=time.time(),
        )
        # debug_landmarks should be stripped
        assert "debug_landmarks" not in packet.pose.player1
        assert packet.pose.player1["v_hand_speed"] == 1.5

    def test_build_packet_timestamp_calculation(self):
        import time
        tracker = _MockBallTracker()
        score_det = _MockScoreDetector()
        rally_agg = _MockRallyAggregator()
        packet = build_packet(
            frame_idx=90,
            fps=30.0,
            ball_tracker=tracker,
            tracks=[],
            score_detector=score_det,
            rally_aggregator=rally_agg,
            pose_p1=None,
            pose_p2=None,
            start_time=time.time(),
        )
        # 90 frames / 30 fps = 3.0 seconds
        assert abs(packet.timestamp_sec - 3.0) < 0.01


# ---------------------------------------------------------------------------
# Full pipeline smoke (requires model weights — skipped if absent)
# ---------------------------------------------------------------------------

def _weights_available():
    """Check if ball model weights exist."""
    from pathlib import Path
    from ball_tracking_analysis import BALL_MODEL_PATH, resolve_model_path
    path = resolve_model_path(BALL_MODEL_PATH)
    return Path(path).exists()


@pytest.mark.skipif(not _weights_available(), reason="Ball model weights not found")
class TestPipelineSmokeWithModels:

    def test_pipeline_runs_on_synthetic_video(self, tmp_path_in_workspace):
        """Run the full pipeline headlessly on a synthetic video; verify CSV output."""
        import json
        from tests.fixtures.synthetic_frames import write_synthetic_video
        from prediction_pipeline import run_prediction_pipeline

        video_path = str(tmp_path_in_workspace / "test_video.mp4")
        write_synthetic_video(video_path, n_frames=30, fps=30.0, w=640, h=480)

        output_dir = str(tmp_path_in_workspace / "output")
        os.makedirs(output_dir, exist_ok=True)

        # Write a minimal config so the pipeline skips interactive ROI setup
        config_path = str(tmp_path_in_workspace / "config.json")
        config = {
            "video_path": video_path,
            "fps": 30.0,
            "rois": {
                "player1_score": [10, 10, 100, 60],
                "player2_score": [540, 10, 630, 60],
                "player1_rounds": [100, 10, 160, 60],
                "player2_rounds": [480, 10, 540, 60],
            },
            "ball_model": "runs/detect/runs/ball_detector/weights/best.pt",
            "digit_model": "runs/detect/runs/detect/digits_v2/weights/best.pt",
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        run_prediction_pipeline(
            source=video_path,
            is_live=False,
            output_dir=output_dir,
            save_video=False,
            no_display=True,
            score_mode="manual",
            config_path=config_path,
            prediction_model=DummyPredictionModel(),
        )

        # At least trajectory, score, and rally CSVs should exist
        csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
        assert len(csv_files) >= 3, f"Expected >= 3 CSV files, found: {csv_files}"
