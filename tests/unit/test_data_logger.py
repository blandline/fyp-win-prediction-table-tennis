"""
Unit tests for DataLogger (ball_tracking_analysis.py).

Verifies CSV file creation, headers, row writing, and config JSON round-trip.
No YOLO or MediaPipe required.
"""

import csv
import json
import os
import pytest

from ball_tracking_analysis import DataLogger


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

class TestFileCreation:

    def test_output_directory_created(self, tmp_output_dir):
        sub = os.path.join(tmp_output_dir, "new_subdir")
        logger = DataLogger(sub)
        assert os.path.isdir(sub)

    def test_csv_files_created(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir)
        assert os.path.isfile(logger.trajectory_file)
        assert os.path.isfile(logger.score_file)
        assert os.path.isfile(logger.rally_file)
        assert os.path.isfile(logger.pose_file)

    def test_config_file_path_set(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir)
        assert logger.config_file is not None
        assert str(logger.config_file).endswith(".json")


# ---------------------------------------------------------------------------
# Trajectory CSV
# ---------------------------------------------------------------------------

class TestTrajectoryCSV:

    def test_trajectory_header_without_meters(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir, with_meters=False)
        with open(logger.trajectory_file) as f:
            header = f.readline().strip()
        assert header == "frame,track_id,x,y,speed_pps"

    def test_trajectory_header_with_meters(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir, with_meters=True)
        with open(logger.trajectory_file) as f:
            header = f.readline().strip()
        assert "x_m" in header
        assert "y_m" in header
        assert "speed_mps" in header

    def test_log_trajectory_row_without_meters(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir, with_meters=False)
        logger.log_trajectory(42, 1, 320.5, 240.1, 900.0)
        with open(logger.trajectory_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert int(rows[0]["frame"]) == 42
        assert int(rows[0]["track_id"]) == 1
        assert abs(float(rows[0]["x"]) - 320.5) < 0.01
        assert abs(float(rows[0]["speed_pps"]) - 900.0) < 0.1

    def test_log_trajectory_row_with_meters(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir, with_meters=True)
        logger.log_trajectory(10, 2, 100.0, 200.0, 500.0, x_m=1.2, y_m=0.5, speed_mps=8.3)
        with open(logger.trajectory_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert abs(float(rows[0]["x_m"]) - 1.2) < 0.001
        assert abs(float(rows[0]["speed_mps"]) - 8.3) < 0.001


# ---------------------------------------------------------------------------
# Score CSV
# ---------------------------------------------------------------------------

class TestScoreCSV:

    def test_score_csv_header(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir)
        with open(logger.score_file) as f:
            header = f.readline().strip()
        expected_cols = [
            "frame", "timestamp_sec", "player1_score", "player2_score",
            "player1_sets", "player2_sets",
            "player1_obscured", "player2_obscured",
            "player1_sets_obscured", "player2_sets_obscured",
        ]
        for col in expected_cols:
            assert col in header, f"Missing column '{col}' in score header"

    def test_log_score_row(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir)
        scores = {"player1": 5, "player2": 3}
        rounds = {"player1": 1, "player2": 0}
        logger.log_score(100, 3.33, scores, rounds)
        with open(logger.score_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert int(rows[0]["player1_score"]) == 5
        assert int(rows[0]["player2_score"]) == 3
        assert int(rows[0]["player1_sets"]) == 1

    def test_log_score_row_with_obscured(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir)
        scores = {"player1": None, "player2": 2}
        rounds = {"player1": 0, "player2": 0}
        obscured = {"player1": True, "player2": False}
        logger.log_score(50, 1.67, scores, rounds, obscured=obscured)
        with open(logger.score_file) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["player1_obscured"] == "1"
        assert rows[0]["player2_obscured"] == "0"


# ---------------------------------------------------------------------------
# Rally CSV
# ---------------------------------------------------------------------------

class TestRallyCSV:

    def test_rally_csv_header(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir)
        with open(logger.rally_file) as f:
            header = f.readline().strip()
        required = [
            "rally_id", "rally_start_frame", "rally_end_frame",
            "mean_speed_mps", "max_speed_mps", "speed_std_mps",
            "landing_zone_0", "landing_zone_8", "point_winner",
        ]
        for col in required:
            assert col in header, f"Missing column '{col}' in rally header"

    def test_log_rally_row(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir)
        record = {
            "rally_id": 3,
            "rally_start_frame": 100,
            "rally_end_frame": 200,
            "rally_start_time": 3.33,
            "rally_end_time": 6.67,
            "p1_score_start": 2,
            "p2_score_start": 1,
            "p1_sets_start": 0,
            "p2_sets_start": 0,
            "mean_speed_mps": "8.1234",
            "max_speed_mps": "12.5678",
            "speed_std_mps": "1.2345",
            "point_winner": "p1",
        }
        for i in range(9):
            record[f"landing_zone_{i}"] = i
        logger.log_rally(record)
        with open(logger.rally_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert int(rows[0]["rally_id"]) == 3
        assert rows[0]["point_winner"] == "p1"
        assert int(rows[0]["landing_zone_4"]) == 4

    def test_log_rally_row_no_samples(self, tmp_output_dir):
        """Rally with no ball samples should write empty speed fields."""
        logger = DataLogger(tmp_output_dir)
        record = {
            "rally_id": 0,
            "rally_start_frame": 0,
            "rally_end_frame": 50,
            "rally_start_time": 0.0,
            "rally_end_time": 1.67,
            "p1_score_start": 0,
            "p2_score_start": 0,
            "p1_sets_start": 0,
            "p2_sets_start": 0,
            "mean_speed_mps": "",
            "max_speed_mps": "",
            "speed_std_mps": "",
            "point_winner": "unknown",
        }
        for i in range(9):
            record[f"landing_zone_{i}"] = 0
        logger.log_rally(record)
        with open(logger.rally_file) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["mean_speed_mps"] == ""


# ---------------------------------------------------------------------------
# Pose CSV
# ---------------------------------------------------------------------------

class TestPoseCSV:

    def test_pose_csv_header(self, tmp_output_dir):
        logger = DataLogger(tmp_output_dir)
        with open(logger.pose_file) as f:
            header = f.readline().strip()
        for col in DataLogger.POSE_COLUMNS:
            assert col in header, f"Missing pose column '{col}'"

    def test_log_pose_row(self, tmp_output_dir, sample_pose_features):
        logger = DataLogger(tmp_output_dir)
        logger.log_pose(sample_pose_features)
        with open(logger.pose_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert int(rows[0]["frame"]) == sample_pose_features["frame"]
        assert int(rows[0]["player_id"]) == sample_pose_features["player_id"]
        assert abs(float(rows[0]["visibility_mean"]) - sample_pose_features["visibility_mean"]) < 0.001


# ---------------------------------------------------------------------------
# Config JSON round-trip
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:

    def test_save_and_reload_config(self, tmp_output_dir, valid_table_corners, sample_rois):
        logger = DataLogger(tmp_output_dir)
        rois = {k: v for k, v in sample_rois.items()}
        logger.save_config(rois, "test_video.mp4", 30.0, table_corners=valid_table_corners)
        with open(logger.config_file) as f:
            cfg = json.load(f)
        assert cfg["video_path"] == "test_video.mp4"
        assert abs(cfg["fps"] - 30.0) < 0.01
        assert "table_corners" in cfg
        assert len(cfg["table_corners"]) == 4
        assert "rois" in cfg

    def test_save_config_without_corners(self, tmp_output_dir, sample_rois):
        logger = DataLogger(tmp_output_dir)
        rois = {k: v for k, v in sample_rois.items()}
        logger.save_config(rois, "video.mp4", 25.0)
        with open(logger.config_file) as f:
            cfg = json.load(f)
        assert "table_corners" not in cfg
