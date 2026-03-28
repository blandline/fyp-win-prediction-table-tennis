"""
Integration smoke tests — verify the pipeline runs end-to-end without crashing.
These tests create synthetic videos and run the pipeline headlessly.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.fixtures.synthetic_frames import make_test_video


@pytest.fixture
def config_file(tmp_output_dir, sample_rois, sample_table_corners):
    """Write a config JSON so we can skip interactive setup."""
    config = {
        'video_path': 'test.mp4',
        'fps': 30.0,
        'rois': {k: list(v) if v else None for k, v in sample_rois.items()},
        'table_corners': [[float(p[0]), float(p[1])] for p in sample_table_corners],
        'ball_model': 'runs/detect/runs/ball_detector/weights/best.pt',
        'digit_model': 'runs/detect/runs/detect/digits_v2/weights/best.pt',
    }
    path = os.path.join(tmp_output_dir, "test_config.json")
    with open(path, 'w') as f:
        json.dump(config, f)
    return path


class TestPredictionModelBase:
    """Test that the prediction model interface works correctly."""

    def test_dummy_model_returns_valid_result(self):
        from prediction_model_base import DummyPredictionModel, PredictionDataPacket, PredictionResult
        model = DummyPredictionModel()
        packet = PredictionDataPacket()
        result = model.predict(packet)
        assert isinstance(result, PredictionResult)
        assert result.player1_win_prob == 0.5
        assert result.player2_win_prob == 0.5
        assert result.model_ready is False

    def test_data_packet_defaults(self):
        from prediction_model_base import PredictionDataPacket, BallState
        packet = PredictionDataPacket()
        assert packet.frame_idx == 0
        assert packet.ball.detected is False
        assert packet.score.player1_score is None
        assert packet.rally.is_active is False
        assert packet.pose.player1 is None

    def test_prediction_result_fields(self):
        from prediction_model_base import PredictionResult
        r = PredictionResult(player1_win_prob=0.7, player2_win_prob=0.3, confidence=0.9, model_ready=True)
        assert r.player1_win_prob == 0.7
        assert r.model_ready is True


class TestSourceParsing:
    """Test the parse_source function."""

    def test_integer_source(self):
        from prediction_pipeline import parse_source
        source, is_live = parse_source("0")
        assert source == 0
        assert is_live is True

    def test_rtsp_source(self):
        from prediction_pipeline import parse_source
        source, is_live = parse_source("rtsp://192.168.1.1/stream")
        assert source == "rtsp://192.168.1.1/stream"
        assert is_live is True

    def test_file_source(self):
        from prediction_pipeline import parse_source
        source, is_live = parse_source("video.mp4")
        assert source == "video.mp4"
        assert is_live is False

    def test_http_source(self):
        from prediction_pipeline import parse_source
        source, is_live = parse_source("http://example.com/stream")
        assert is_live is True


class TestBuildPacket:
    """Test the build_packet helper function."""

    def test_build_packet_no_detections(self):
        """Pipeline state with no detections → valid packet with defaults."""
        from prediction_pipeline import build_packet
        from ball_tracking_analysis import RallyAggregator, DataLogger
        import time, tempfile

        tmp_dir = tempfile.mkdtemp()
        logger = DataLogger(tmp_dir, with_meters=False)
        rally_agg = RallyAggregator(fps=30.0, data_logger=logger)

        # Minimal mock tracker
        class FakeTracker:
            trajectories = {}
            def get_position_meters(self, tid): return (None, None)
            def get_smoothed_speed_mps(self, tid): return None
            def get_smoothed_speed(self, tid): return 0

        class FakeScoreDetector:
            current_scores = {'player1': 0, 'player2': 0}
            stable_scores = {'player1': 0, 'player2': 0}
            rounds = {'player1': 0, 'player2': 0}
            score_roi_obscured = {'player1': False, 'player2': False}

        packet = build_packet(
            frame_idx=10, fps=30.0,
            ball_tracker=FakeTracker(), tracks=[],
            score_detector=FakeScoreDetector(),
            rally_aggregator=rally_agg,
            pose_p1=None, pose_p2=None,
            start_time=time.time(),
        )

        assert packet.frame_idx == 10
        assert packet.ball.detected is False
        assert packet.score.player1_score == 0
        assert packet.rally.is_active is False


class TestPipelineSmokeWithModels:
    """
    These tests require real YOLO model weights.
    They are skipped automatically if weights are not present.
    """

    @pytest.fixture
    def weights_available(self):
        ball_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  'runs', 'detect', 'runs', 'ball_detector', 'weights', 'best.pt')
        digit_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                   'runs', 'detect', 'runs', 'detect', 'digits_v2', 'weights', 'best.pt')
        if not os.path.exists(ball_path) or not os.path.exists(digit_path):
            pytest.skip("Model weights not found — skipping integration test")

    def test_pipeline_runs_on_synthetic_video(self, weights_available, tmp_output_dir, config_file):
        """Run the prediction pipeline on a short synthetic video headlessly."""
        from prediction_pipeline import run_prediction_pipeline
        from prediction_model_base import DummyPredictionModel

        video_path = os.path.join(tmp_output_dir, "smoke_test.mp4")
        make_test_video(video_path, n_frames=30, fps=30.0, w=640, h=360)

        run_prediction_pipeline(
            source=video_path,
            is_live=False,
            output_dir=tmp_output_dir,
            save_video=False,
            config_path=config_file,
            score_mode="manual",
            prediction_model=DummyPredictionModel(),
            no_display=True,
        )

        # Check that CSV files were created
        csv_files = [f for f in os.listdir(tmp_output_dir) if f.endswith('.csv')]
        assert len(csv_files) >= 3, f"Expected ≥3 CSV files, got {csv_files}"
