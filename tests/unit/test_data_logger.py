"""
Unit tests for DataLogger — CSV format, file creation, config round-trip.
"""

import json
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ball_tracking_analysis import DataLogger


@pytest.fixture
def logger(tmp_output_dir):
    return DataLogger(tmp_output_dir, with_meters=True)


class TestFileCreation:

    def test_output_directory_created(self, tmp_output_dir):
        new_dir = os.path.join(tmp_output_dir, "new_output")
        # DataLogger should create the directory (one level)
        lg = DataLogger(new_dir, with_meters=False)
        assert os.path.isdir(new_dir)

    def test_csv_files_created(self, logger):
        assert logger.trajectory_file.exists()
        assert logger.score_file.exists()
        assert logger.rally_file.exists()
        assert logger.pose_file.exists()


class TestTrajectoryCSV:

    def test_trajectory_header_with_meters(self, logger):
        with open(logger.trajectory_file) as f:
            header = f.readline().strip()
        assert "frame,track_id,x,y,speed_pps,x_m,y_m,speed_mps" == header

    def test_trajectory_header_without_meters(self, tmp_output_dir):
        lg = DataLogger(tmp_output_dir + "_no_m", with_meters=False)
        with open(lg.trajectory_file) as f:
            header = f.readline().strip()
        assert header == "frame,track_id,x,y,speed_pps"

    def test_log_trajectory_row(self, logger):
        logger.log_trajectory(10, 1, 640.5, 360.2, 900.0, x_m=1.37, y_m=0.76, speed_mps=5.5)
        with open(logger.trajectory_file) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 data row
        parts = lines[1].strip().split(',')
        assert parts[0] == '10'  # frame
        assert parts[1] == '1'   # track_id
        assert float(parts[2]) == pytest.approx(640.5, abs=0.1)


class TestScoreCSV:

    def test_score_csv_header(self, logger):
        with open(logger.score_file) as f:
            header = f.readline().strip()
        assert "frame,timestamp_sec" in header
        assert "player1_score" in header

    def test_log_score_row(self, logger):
        scores = {'player1': 5, 'player2': 3}
        rounds = {'player1': 1, 'player2': 0}
        obscured = {'player1': False, 'player2': False}
        logger.log_score(100, 3.33, scores, rounds, obscured, obscured)
        with open(logger.score_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        parts = lines[1].strip().split(',')
        assert parts[2] == '5'   # player1_score
        assert parts[3] == '3'   # player2_score


class TestRallyCSV:

    def test_rally_csv_header(self, logger):
        with open(logger.rally_file) as f:
            header = f.readline().strip()
        assert "rally_id" in header
        assert "point_winner" in header
        assert "landing_zone_0" in header

    def test_log_rally_row(self, logger):
        record = {
            'rally_id': 0,
            'rally_start_frame': 10,
            'rally_end_frame': 200,
            'rally_start_time': 0.33,
            'rally_end_time': 6.67,
            'p1_score_start': 0,
            'p2_score_start': 0,
            'p1_sets_start': 0,
            'p2_sets_start': 0,
            'mean_speed_mps': '5.1234',
            'max_speed_mps': '8.5000',
            'speed_std_mps': '1.2000',
            'point_winner': 'p1',
        }
        for i in range(9):
            record[f'landing_zone_{i}'] = i
        logger.log_rally(record)

        with open(logger.rally_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        parts = lines[1].strip().split(',')
        assert parts[0] == '0'    # rally_id
        assert parts[-1] == 'p1'  # point_winner


class TestPoseCSV:

    def test_pose_csv_header(self, logger):
        with open(logger.pose_file) as f:
            header = f.readline().strip()
        cols = header.split(',')
        assert cols[0] == 'frame'
        assert 'angle_trunk_lean' in cols
        assert 'v_hand_speed' in cols

    def test_log_pose_row(self, logger):
        record = {col: '' for col in DataLogger.POSE_COLUMNS}
        record['frame'] = 50
        record['timestamp_sec'] = 1.67
        record['player_id'] = 1
        record['visibility_mean'] = 0.85
        logger.log_pose(record)

        with open(logger.pose_file) as f:
            lines = f.readlines()
        assert len(lines) == 2


class TestConfigRoundTrip:

    def test_save_and_reload_config(self, logger, sample_rois, sample_table_corners):
        logger.save_config(sample_rois, "test_video.mp4", 30.0, table_corners=sample_table_corners)

        with open(logger.config_file) as f:
            config = json.load(f)

        assert config['video_path'] == "test_video.mp4"
        assert config['fps'] == 30.0
        assert len(config['table_corners']) == 4
        assert config['rois']['player1_score'] == list(sample_rois['player1_score'])
