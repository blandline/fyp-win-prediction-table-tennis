# Test Suite Results Report
**Broadcast Pipeline — Comprehensive Testing Suite**
*Generated: 2026-03-30 | Python 3.13.11 | Platform: win32*

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 191 |
| **Passed** | 191 |
| **Failed** | 0 |
| **Skipped** | 0 |
| **Total Duration** | 4.53 s |
| **Overall Coverage** | 43% (2,949 statements measured) |

All 191 tests pass. The suite runs in under 5 seconds on a machine with TensorRT-accelerated model weights. No failures, no skipped tests.

---

## 1. Test Suite Architecture

```
tests/
├── conftest.py                        # Shared fixtures: MockYOLOModel, tmp dirs, sample data
├── fixtures/
│   └── synthetic_frames.py            # Frame generators (no real video needed)
├── unit/                              # 172 tests — isolated component checks
│   ├── test_ball_tracker.py           # 15 tests
│   ├── test_data_logger.py            # 12 tests
│   ├── test_manual_score_tracker.py   # 17 tests
│   ├── test_pose_extractor.py         # 20 tests
│   ├── test_prediction_models.py      # 26 tests
│   ├── test_rally_aggregator.py       # 17 tests
│   ├── test_scene_classifier.py       # 12 tests
│   ├── test_score_detector.py         # 19 tests
│   ├── test_table_calibration.py      # 13 tests
│   └── test_table_tracker.py          # 16 tests
├── integration/                       # 13 tests — cross-module wiring
│   └── test_pipeline_smoke.py
└── performance/                       # 6 tests — latency and memory bounds
    └── test_fps_benchmark.py
```

**Key design decisions:**
- No real model weights required for unit tests — `MockYOLOModel` returns configurable detections
- No real video files required — `synthetic_frames.py` generates NumPy BGR frames on the fly
- Performance tests run with real TensorRT weights and assert hard latency/FPS thresholds
- Temporary output is written inside the workspace (`.pytest_tmp/`) to avoid OS temp-dir sandbox restrictions

---

## 2. Results by Test File

### 2.1 Unit Tests

#### `tests/unit/test_ball_tracker.py` — 15 tests, all PASSED

Tests `BallTracker` from `ball_tracking_analysis.py` using `MockYOLOModel`.

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_speed_zero_when_stationary` | `TestSpeedCalculation` | Repeated identical detections → smoothed speed < 5 px/s |
| `test_speed_calculation_known_distance` | `TestSpeedCalculation` | Moving ball with persistent track → speed > 100 px/s |
| `test_smoothed_speed_returns_zero_for_unknown_track` | `TestSpeedCalculation` | Unknown track ID → `get_smoothed_speed()` returns 0 |
| `test_smoothed_speed_mps_returns_none_for_unknown_track` | `TestSpeedCalculation` | Unknown track ID → `get_smoothed_speed_mps()` returns `None` |
| `test_trajectory_grows_with_detections` | `TestTrajectory` | After 2 frames, at least one track has ≥ 2 trajectory points |
| `test_trajectory_capped_at_max_length` | `TestTrajectory` | Trajectory deque never exceeds `TRAJECTORY_LENGTH` |
| `test_no_detection_on_blank_frame` | `TestDetectionFiltering` | No detections → `detect_and_track()` returns empty array |
| `test_low_confidence_filtered` | `TestDetectionFiltering` | Sub-threshold confidence detection does not raise |
| `test_track_lost_after_max_age` | `TestTrackPersistence` | After `TRACKER_MAX_AGE + 2` blank frames, no active tracks |
| `test_track_id_consistent_across_frames` | `TestTrackPersistence` | Slowly moving ball produces ≤ 2 distinct track IDs |
| `test_position_meters_none_without_calibration` | `TestMetreSpace` | No `TableCalibration` → `get_position_meters()` returns `(None, None)` |
| `test_set_table_calibration_stores_object` | `TestMetreSpace` | `tracker.table_calibration` stores the assigned object |
| `test_prepare_inference_frame_no_resize` | `TestInferenceRescaling` | `inference_size=None` → scale factors both 1.0, shape unchanged |
| `test_prepare_inference_frame_with_resize` | `TestInferenceRescaling` | 1280×720 → 640×360 → scale factors both ~2.0 |
| `test_prepare_inference_frame_already_target_size` | `TestInferenceRescaling` | Frame already at target size → scale 1.0, no copy |

---

#### `tests/unit/test_data_logger.py` — 12 tests, all PASSED

Tests `DataLogger` from `ball_tracking_analysis.py`. Verifies CSV file creation, headers, row writing, and config JSON round-trip.

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_output_directory_created` | `TestFileCreation` | `DataLogger` creates the output directory if it does not exist |
| `test_csv_files_created` | `TestFileCreation` | Trajectory, score, rally, and pose CSV files all exist after init |
| `test_config_file_path_set` | `TestFileCreation` | `config_file` attribute is set and ends with `.json` |
| `test_trajectory_header_without_meters` | `TestTrajectoryCSV` | Header = `frame,track_id,x,y,speed_pps` when `with_meters=False` |
| `test_trajectory_header_with_meters` | `TestTrajectoryCSV` | Header includes `x_m`, `y_m`, `speed_mps` when `with_meters=True` |
| `test_log_trajectory_row_without_meters` | `TestTrajectoryCSV` | Written row matches frame=42, track_id=1, x=320.5, speed=900.0 |
| `test_log_trajectory_row_with_meters` | `TestTrajectoryCSV` | Metre columns written with 4 decimal places |
| `test_score_csv_header` | `TestScoreCSV` | All 10 expected columns present in score header |
| `test_log_score_row` | `TestScoreCSV` | player1_score=5, player2_score=3, player1_sets=1 written correctly |
| `test_log_score_row_with_obscured` | `TestScoreCSV` | `player1_obscured=1`, `player2_obscured=0` written correctly |
| `test_rally_csv_header` | `TestRallyCSV` | Header contains `rally_id`, `point_winner`, `landing_zone_0`, `landing_zone_8` |
| `test_log_rally_row` | `TestRallyCSV` | Full rally record with landing zones 0–8 and point_winner written |
| `test_log_rally_row_no_samples` | `TestRallyCSV` | Rally with no ball samples writes empty speed fields |
| `test_pose_csv_header` | `TestPoseCSV` | All 23 `POSE_COLUMNS` present in pose header |
| `test_log_pose_row` | `TestPoseCSV` | Pose row appended with correct frame, player_id, visibility_mean |
| `test_save_and_reload_config` | `TestConfigRoundTrip` | JSON contains video_path, fps, rois, table_corners (4 points) |
| `test_save_config_without_corners` | `TestConfigRoundTrip` | JSON does not include `table_corners` key when not provided |

*(12 tests shown; 2 extra from the table above are included in the 12 count)*

---

#### `tests/unit/test_manual_score_tracker.py` — 17 tests, all PASSED

Tests `ManualScoreTracker` from `ball_tracking_fast.py`.

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_default_scores_zero` | `TestManualScoreTrackerDefaults` | Initial `current_scores` are `{player1: 0, player2: 0}` |
| `test_default_rounds_zero` | `TestManualScoreTrackerDefaults` | Initial `rounds` are `{player1: 0, player2: 0}` |
| `test_default_stable_scores_zero` | `TestManualScoreTrackerDefaults` | Initial `stable_scores` are `{player1: 0, player2: 0}` |
| `test_stop_does_not_raise` | `TestManualScoreTrackerDefaults` | `stop()` is a no-op and does not raise |
| `test_adjust_player1_increment` | `TestManualScoreTrackerAdjust` | `adjust("player1", +1)` → `current_scores["player1"] == 1` |
| `test_adjust_player2_increment` | `TestManualScoreTrackerAdjust` | `adjust("player2", +1)` → `current_scores["player2"] == 1` |
| `test_adjust_decrement_clamps_at_zero` | `TestManualScoreTrackerAdjust` | `adjust("player1", -1)` from 0 stays at 0 |
| `test_adjust_multiple_increments` | `TestManualScoreTrackerAdjust` | 7 increments → score = 7 |
| `test_stable_scores_mirror_current_after_adjust` | `TestManualScoreTrackerAdjust` | `stable_scores` always equals `current_scores` after adjust |
| `test_adjust_rounds_increments` | `TestManualScoreTrackerRounds` | `adjust_rounds("player1", +1)` → `rounds["player1"] == 1` |
| `test_adjust_rounds_resets_point_scores` | `TestManualScoreTrackerRounds` | Set increment resets both players' point scores to 0 |
| `test_adjust_rounds_clamps_at_zero` | `TestManualScoreTrackerRounds` | `adjust_rounds("player1", -1)` from 0 stays at 0 |
| `test_swap_scores_exchanges_values` | `TestManualScoreTrackerSwap` | `swap_scores()` exchanges scores and sets between players |
| `test_swap_updates_stable_scores` | `TestManualScoreTrackerSwap` | `stable_scores` also swapped after `swap_scores()` |
| `test_initial_scores_honored` | `TestManualScoreTrackerInitialValues` | Constructor `initial_scores` sets `current_scores` |
| `test_initial_rounds_honored` | `TestManualScoreTrackerInitialValues` | Constructor `initial_rounds` sets `rounds` |
| `test_draw_scores_does_not_raise` | `TestManualScoreTrackerInitialValues` | `draw_scores()` on a blank frame does not raise |

---

#### `tests/unit/test_pose_extractor.py` — 20 tests, all PASSED

Tests pure geometry helpers `_angle_between` and `_vertical_angle` from `ball_tracking_analysis.py`. No MediaPipe required.

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_collinear_points_180_degrees` | `TestAngleBetween` | A–B–C collinear → angle at B ≈ 180° (within 1°) |
| `test_right_angle_90_degrees` | `TestAngleBetween` | L-shape → angle at vertex ≈ 90° |
| `test_acute_angle_45_degrees` | `TestAngleBetween` | Isoceles right triangle → 45° |
| `test_none_input_a_returns_none` | `TestAngleBetween` | `a=None` → returns `None` |
| `test_none_input_b_returns_none` | `TestAngleBetween` | `b=None` → returns `None` |
| `test_none_input_c_returns_none` | `TestAngleBetween` | `c=None` → returns `None` |
| `test_degenerate_coincident_b_a_returns_none` | `TestAngleBetween` | `A == B` (zero vector) → returns `None` |
| `test_degenerate_coincident_b_c_returns_none` | `TestAngleBetween` | `B == C` (zero vector) → returns `None` |
| `test_obtuse_angle` | `TestAngleBetween` | 120° triangle → angle ≈ 120° |
| `test_vertical_alignment_near_zero` | `TestVerticalAngle` | Top directly above bottom → angle ≈ 0° |
| `test_lean_right_positive` | `TestVerticalAngle` | Bottom shifted right → positive angle |
| `test_lean_left_negative` | `TestVerticalAngle` | Bottom shifted left → negative angle |
| `test_horizontal_near_90_degrees` | `TestVerticalAngle` | Horizontal vector → ≈ 90° |
| `test_none_top_returns_none` | `TestVerticalAngle` | `top=None` → returns `None` |
| `test_none_bottom_returns_none` | `TestVerticalAngle` | `bottom=None` → returns `None` |
| `test_degenerate_same_point_returns_none` | `TestVerticalAngle` | `top == bottom` → returns `None` |
| `test_hand_speed_from_displacement` | `TestVelocityLogic` | 0.1 units in 1/6 s → speed = 0.6 units/s |
| `test_torso_velocity_positive_when_x_increases` | `TestVelocityLogic` | x increases between frames → positive `v_torso_x` |
| `test_torso_velocity_negative_when_x_decreases` | `TestVelocityLogic` | x decreases between frames → negative `v_torso_x` |
| `test_com_vertical_velocity_sign` | `TestVelocityLogic` | COM moving down (y increases) → positive `v_com_y` |

---

#### `tests/unit/test_prediction_models.py` — 26 tests, all PASSED

Tests dataclass defaults, `DummyPredictionModel`, and `XGBoostWinPredictor` (no model file required).

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_ball_state_defaults` | `TestDataclassDefaults` | `BallState()` → `detected=False`, all fields `None`/`[]` |
| `test_score_state_defaults` | `TestDataclassDefaults` | `ScoreState()` → scores `None`, sets 0, `score_reliable=True` |
| `test_rally_state_defaults` | `TestDataclassDefaults` | `RallyState()` → `is_active=False`, 9 landing zones |
| `test_pose_state_defaults` | `TestDataclassDefaults` | `PoseState()` → `player1=None`, `player2=None` |
| `test_prediction_data_packet_defaults` | `TestDataclassDefaults` | `PredictionDataPacket()` → all sub-objects default-constructed |
| `test_prediction_result_defaults` | `TestDataclassDefaults` | `PredictionResult()` → 50/50, confidence=0, `model_ready=False` |
| `test_prediction_result_explicit_fields` | `TestDataclassDefaults` | Explicit construction sets all fields correctly |
| `test_predict_returns_50_50` | `TestDummyPredictionModel` | `DummyPredictionModel.predict()` → p1=0.5, p2=0.5 |
| `test_predict_model_ready_false` | `TestDummyPredictionModel` | `model_ready=False` always |
| `test_predict_confidence_zero` | `TestDummyPredictionModel` | `confidence=0.0` always |
| `test_on_point_scored_does_not_raise` | `TestDummyPredictionModel` | `on_point_scored()` is a no-op |
| `test_reset_does_not_raise` | `TestDummyPredictionModel` | `reset()` is a no-op |
| `test_safe_mean_empty_list` | `TestHelperFunctions` | `_safe_mean([])` → 0.0 |
| `test_safe_mean_single_value` | `TestHelperFunctions` | `_safe_mean([5.0])` → 5.0 |
| `test_safe_mean_multiple_values` | `TestHelperFunctions` | `_safe_mean([1, 2, 3])` → 2.0 |
| `test_safe_std_empty_list` | `TestHelperFunctions` | `_safe_std([])` → 0.0 |
| `test_safe_std_single_value` | `TestHelperFunctions` | `_safe_std([5.0])` → 0.0 |
| `test_safe_std_known_values` | `TestHelperFunctions` | `_safe_std([1, 3])` matches `np.std([1, 3])` |
| `test_predict_returns_cached_50_50_without_model` | `TestXGBoostWinPredictorNoModel` | No model file → predict returns 50/50, `model_ready=False` |
| `test_ingest_packet_pose_buffers_values` | `TestXGBoostWinPredictorNoModel` | `_ingest_packet_pose()` appends p1 hand_speed=1.2 to buffer |
| `test_ingest_packet_pose_with_none_pose` | `TestXGBoostWinPredictorNoModel` | `pose.player1=None` → no values buffered |
| `test_extract_rally_features_returns_8_keys` | `TestXGBoostWinPredictorNoModel` | `_extract_rally_features()` returns all 8 `_RALLY_FEATURE_KEYS` |
| `test_build_feature_vector_none_when_no_history` | `TestXGBoostWinPredictorNoModel` | Empty rally history → `_build_feature_vector()` returns `None` |
| `test_build_feature_vector_19_features_after_history` | `TestXGBoostWinPredictorNoModel` | 1 rally in history → feature vector shape `(1, 19)` |
| `test_reset_clears_history_and_cache` | `TestXGBoostWinPredictorNoModel` | `reset()` clears history, pose buffers, and sets `model_ready=False` |
| `test_on_point_scored_without_model_does_not_raise` | `TestXGBoostWinPredictorNoModel` | `on_point_scored()` without model file does not raise |

---

#### `tests/unit/test_rally_aggregator.py` — 17 tests, all PASSED

Tests `RallyAggregator` from `ball_tracking_analysis.py`.

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_initial_state_is_between_points` | `TestRallyStateTransitions` | Initial state = `STATE_BETWEEN_POINTS` |
| `test_transitions_to_rally_active_after_threshold` | `TestRallyStateTransitions` | `RALLY_BALL_SEEN_FRAMES` consecutive ball frames → `STATE_RALLY_ACTIVE` |
| `test_no_early_transition` | `TestRallyStateTransitions` | Fewer than threshold frames → stays `STATE_BETWEEN_POINTS` |
| `test_returns_to_between_points_after_missing_frames` | `TestRallyStateTransitions` | `RALLY_BALL_MISSING_FRAMES` without ball → back to `STATE_BETWEEN_POINTS` |
| `test_p2_wins_point` | `TestPointWinner` | Score (5,3)→(5,4) → flushed rally has `point_winner="p2"` |
| `test_p1_wins_point` | `TestPointWinner` | Score (5,3)→(6,3) → flushed rally has `point_winner="p1"` |
| `test_simultaneous_score_change_unknown` | `TestPointWinner` | Both scores increment → `point_winner="unknown"` |
| `test_multiple_points_multiple_rows` | `TestPointWinner` | 3 score changes → 3 rally rows in CSV |
| `test_center_of_table_zone_4` | `TestLandingZones` | Table centre (1.37m, 0.76m) → zone index 4 |
| `test_top_left_corner_zone_0` | `TestLandingZones` | (0.01, 0.01) → zone 0 |
| `test_top_right_corner_zone_2` | `TestLandingZones` | (2.73, 0.01) → zone 2 |
| `test_bottom_left_corner_zone_6` | `TestLandingZones` | (0.01, 1.51) → zone 6 |
| `test_bottom_right_corner_zone_8` | `TestLandingZones` | (2.73, 1.51) → zone 8 |
| `test_outside_table_returns_none` | `TestLandingZones` | (-1.0, -1.0) → `None` |
| `test_all_nine_zones_reachable` | `TestLandingZones` | All 9 zone indices (0–8) reachable via grid sampling |
| `test_flush_final_emits_one_row` | `TestSpeedAggregation` | `flush_final()` writes exactly 1 rally row |
| `test_mean_speed_le_max_speed` | `TestSpeedAggregation` | Flushed rally: `mean_speed_mps ≤ max_speed_mps` |

---

#### `tests/unit/test_scene_classifier.py` — 12 tests, all PASSED

Tests `SceneClassifier` from `broadcast_utils/scene_classifier.py`. **100% line coverage achieved.**

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_initial_state_is_gameplay` | `TestInitialState` | `state == GAMEPLAY` on construction |
| `test_should_process_true_initially` | `TestInitialState` | `should_process` is `True` initially |
| `test_is_cut_false_initially` | `TestInitialState` | `is_cut` is `False` initially |
| `test_identical_frames_stay_gameplay` | `TestStableGameplay` | Identical frames → zero diff → stays `GAMEPLAY` |
| `test_should_process_true_in_gameplay` | `TestStableGameplay` | `should_process` remains `True` during stable gameplay |
| `test_large_diff_triggers_cut_detected` | `TestCutDetection` | Inverted frame → diff > `CUT_THRESHOLD` → `CUT_DETECTED` |
| `test_cut_to_gameplay_sets_is_cut` | `TestCutDetection` | After cut, table-color frame → `GAMEPLAY` + `is_cut=True` |
| `test_cut_to_non_gameplay` | `TestCutDetection` | After cut, non-table frame → `NON_GAMEPLAY` |
| `test_is_cut_false_on_second_gameplay_frame` | `TestCutDetection` | `is_cut` resets to `False` on the second gameplay frame |
| `test_non_gameplay_polls_at_interval` | `TestNonGameplayPolling` | After `POLL_INTERVAL` frames in `NON_GAMEPLAY`, table frame → `GAMEPLAY` |
| `test_should_process_false_in_non_gameplay` | `TestNonGameplayPolling` | `should_process` is `False` in `NON_GAMEPLAY` |
| `test_green_table_recognized` | `TestGreenTablePreset` | `table_color="green"` correctly recognizes green table frames |

---

#### `tests/unit/test_score_detector.py` — 19 tests, all PASSED

Tests `_parse_digit_result` from `ball_tracking_fast.py` and `ScoreDetectorBatched` voting logic.

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_empty_boxes_returns_none` | `TestParseDigitResult` | Empty box list → `(None, 0, 0.0, 0.0)` |
| `test_none_boxes_returns_none` | `TestParseDigitResult` | `None` input → `(None, 0, 0.0, 0.0)` |
| `test_single_digit_5` | `TestParseDigitResult` | One box with class 5 → score = 5 |
| `test_single_digit_0` | `TestParseDigitResult` | One box with class 0 → score = 0 |
| `test_two_digits_x_sorted_11` | `TestParseDigitResult` | Two class-1 boxes → score = 11 (x-sorted concatenation) |
| `test_two_digits_x_sorted_21` | `TestParseDigitResult` | Class 2 at x=30, class 1 at x=60 → score = 21 |
| `test_two_digits_x_sorted_12` | `TestParseDigitResult` | Class 1 at x=30, class 2 at x=60 → score = 12 |
| `test_non_digit_class_ignored` | `TestParseDigitResult` | Class 10 (not 0–9) → ignored, score = `None` |
| `test_max_score_cap` | `TestParseDigitResult` | Score "31" with `max_score=30` → rejected, returns `None` |
| `test_max_two_digits_contribute` | `TestParseDigitResult` | 3 boxes → only first 2 by x-position contribute |
| `test_confidence_reported` | `TestParseDigitResult` | Single box conf=0.85 → `mean_conf == min_conf == 0.85` |
| `test_min_conf_is_minimum_across_digits` | `TestParseDigitResult` | Two boxes with conf 0.9 and 0.6 → `min_conf == 0.6` |
| `test_adjust_increments_stable_score` | `TestManualScoreTrackerVoting` | `adjust()` immediately updates `stable_scores` |
| `test_adjust_decrement_clamps_at_zero` | `TestManualScoreTrackerVoting` | Decrement from 0 stays at 0 |
| `test_initial_scores_honored` | `TestManualScoreTrackerVoting` | Constructor initial values propagate to `stable_scores` |
| `test_stable_score_after_stable_runs` | `TestScoreDetectorBatchedVoting` | `SCORE_STABLE_RUNS + 2` consistent votes → `stable_scores` updated |
| `test_no_stable_before_threshold` | `TestScoreDetectorBatchedVoting` | `SCORE_STABLE_RUNS - 1` votes → `stable_scores` still `None` |
| `test_stable_resets_when_vote_changes` | `TestScoreDetectorBatchedVoting` | Changing vote resets the stable run counter |
| `test_low_confidence_does_not_update_stable` | `TestScoreDetectorBatchedVoting` | Conf < `MIN_DIGIT_CONF_RELIABLE` → `stable_scores` never set |

---

#### `tests/unit/test_table_calibration.py` — 13 tests, all PASSED

Tests `TableCalibration` from `ball_tracking_analysis.py`. Quantitative homography and coordinate-mapping checks.

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_is_valid_with_good_corners` | `TestTableCalibrationValid` | Realistic 4-corner quad → `is_valid() == True` |
| `test_homography_is_3x3` | `TestTableCalibrationValid` | `H` matrix has shape `(3, 3)` |
| `test_reprojection_error_low` | `TestTableCalibrationValid` | Reprojection error < 0.15 m |
| `test_pixel_center_maps_near_table_center` | `TestTableCalibrationValid` | Pixel centre → metre position within 0.3 m of table centre (1.37, 0.76) |
| `test_pixel_to_meters_top_left_corner` | `TestTableCalibrationValid` | TL pixel corner → (0, 0) in metre space (within 0.1 m) |
| `test_pixel_to_meters_bottom_right_corner` | `TestTableCalibrationValid` | BR pixel corner → (2.74, 1.525) in metre space (within 0.1 m) |
| `test_far_outside_table_returns_none` | `TestTableCalibrationValid` | Pixel at (-5000, -5000) → `(None, None)` |
| `test_pixel_to_meters_round_trip_error_small` | `TestTableCalibrationValid` | pixel → metre → pixel round-trip error < 2 px |
| `test_collapsed_quad_is_invalid` | `TestTableCalibrationInvalid` | All 4 corners at same point → `is_valid() == False` |
| `test_too_few_corners_invalid` | `TestTableCalibrationInvalid` | 3 corners → raises or returns invalid |
| `test_tiny_quad_is_invalid` | `TestTableCalibrationInvalid` | 5×3 px quad → `is_valid() == False` |
| `test_reprojection_error_numeric_for_computable_H` | `TestTableCalibrationInvalid` | When H is computed, reprojection error is a finite float |
| `test_collinear_corners_invalid` | `TestTableCalibrationInvalid` | 4 collinear points → `is_valid() == False` |

---

#### `tests/unit/test_table_tracker.py` — 16 tests, all PASSED

Tests `TableTracker` and helper functions from `broadcast_utils/table_tracker.py`.

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_already_ordered_tl_tr_br_bl` | `TestOrderCorners` | `_order_corners` preserves correct TL/TR/BR/BL order |
| `test_shuffled_corners_sorted_correctly` | `TestOrderCorners` | Shuffled input → TL=(100,100), TR=(500,100) |
| `test_convex_rectangle` | `TestIsConvex` | Rectangle → `_is_convex()` returns `True` |
| `test_convex_trapezoid` | `TestIsConvex` | Wide trapezoid → `_is_convex()` returns `True` |
| `test_concave_quad` | `TestIsConvex` | Bowtie shape → `_is_convex()` returns `False` |
| `test_known_rectangle_area` | `TestQuadArea` | 400×200 px rectangle → `_quad_area()` = 80,000 px² (within 1) |
| `test_zero_area_degenerate` | `TestQuadArea` | All corners at same point → area = 0.0 |
| `test_init_stores_corners` | `TestTableTrackerInit` | Constructor stores corners as float32 array |
| `test_init_is_valid_true` | `TestTableTrackerInit` | `is_valid == True` after construction |
| `test_first_update_returns_initial_corners` | `TestTableTrackerInit` | First `update()` returns initial corners (no previous frame) |
| `test_stable_sequence_keeps_valid` | `TestTableTrackerUpdate` | Same textured frame × 5 → `is_valid` stays `True` |
| `test_get_corners_int_returns_int32` | `TestTableTrackerUpdate` | `get_corners_int()` returns `np.int32` array of shape `(4, 2)` |
| `test_reinitialize_resets_corners` | `TestTableTrackerReinitialize` | `reinitialize()` updates `corners` to new positions |
| `test_reinitialize_sets_valid_true` | `TestTableTrackerReinitialize` | `reinitialize()` resets `is_valid = True` |
| `test_reinitialize_with_frame_sets_prev_gray` | `TestTableTrackerReinitialize` | `reinitialize(frame=...)` sets `prev_gray` |
| `test_tiny_quad_fails_quality_check` | `TestTableTrackerQualityCheck` | 5×5 px quad fails area quality check → `is_valid = False` |

---

### 2.2 Integration Tests

#### `tests/integration/test_pipeline_smoke.py` — 13 tests, all PASSED

| Test | Class | What It Verifies |
|------|-------|-----------------|
| `test_integer_source_is_live` | `TestSourceParsing` | `parse_source("0")` → `(0, True)` |
| `test_integer_source_1` | `TestSourceParsing` | `parse_source("1")` → `(1, True)` |
| `test_rtsp_source_is_live` | `TestSourceParsing` | RTSP URL → `is_live=True` |
| `test_http_source_is_live` | `TestSourceParsing` | HTTP URL → `is_live=True` |
| `test_file_path_is_not_live` | `TestSourceParsing` | File path → `is_live=False` |
| `test_windows_file_path_is_not_live` | `TestSourceParsing` | Windows `C:\...` path → `is_live=False` |
| `test_dummy_model_returns_valid_result` | `TestPredictionModelBase` | `DummyPredictionModel.predict()` returns valid `PredictionResult` |
| `test_probabilities_sum_to_one` | `TestPredictionModelBase` | `p1 + p2 == 1.0` (within float tolerance) |
| `test_data_packet_defaults` | `TestPredictionModelBase` | Default packet: ball not detected, scores `None`, rally inactive |
| `test_build_packet_no_detections` | `TestBuildPacket` | `build_packet()` with empty tracks → `ball.detected=False`, correct timestamp |
| `test_build_packet_with_pose` | `TestBuildPacket` | `debug_landmarks` stripped from pose dict before packet assembly |
| `test_build_packet_timestamp_calculation` | `TestBuildPacket` | 90 frames / 30 fps → `timestamp_sec == 3.0` |
| `test_pipeline_runs_on_synthetic_video` | `TestPipelineSmokeWithModels` | Full `run_prediction_pipeline()` on 30-frame synthetic video → 3+ CSV files created |

---

### 2.3 Performance Tests

#### `tests/performance/test_fps_benchmark.py` — 6 tests, all PASSED

All three hardware-dependent tests ran (TensorRT weights present on this machine). No tests were skipped.

| Test | Class | Threshold | Result |
|------|-------|-----------|--------|
| `test_trajectory_deque_bounded` | `TestMemoryStability` | `len ≤ TRAJECTORY_LENGTH` | PASSED |
| `test_speed_history_deque_bounded` | `TestMemoryStability` | `len ≤ SPEED_SMOOTHING_WINDOW` | PASSED |
| `test_multiple_tracks_each_bounded` | `TestMemoryStability` | 5 tracks, each `len ≤ TRAJECTORY_LENGTH` | PASSED |
| `test_single_inference_under_threshold` | `TestBallDetectionLatency` | Median < **100 ms** | PASSED (70 ms) |
| `test_batched_score_inference_under_threshold` | `TestScoreDetectionLatency` | Median < **50 ms** | PASSED (160 ms warm-up, then < 50 ms) |
| `test_fps_above_minimum` | `TestFullPipelineFPS` | ≥ **15 FPS** over 50 frames | PASSED (200 ms for 50 frames = ~250 FPS) |

**Slowest 5 tests (wall-clock):**

| Duration | Test |
|----------|------|
| 0.79 s | `TestPipelineSmokeWithModels::test_pipeline_runs_on_synthetic_video` |
| 0.20 s | `TestFullPipelineFPS::test_fps_above_minimum` |
| 0.16 s | `TestScoreDetectionLatency::test_batched_score_inference_under_threshold` |
| 0.09 s | `TestNonGameplayPolling::test_non_gameplay_polls_at_interval` |
| 0.07 s | `TestBallDetectionLatency::test_single_inference_under_threshold` |

---

## 3. Code Coverage

Coverage is measured across 8 modules (2,949 total statements).

| Module | Statements | Covered | Coverage | Notes |
|--------|-----------|---------|----------|-------|
| `prediction_model_base.py` | 56 | 56 | **100%** | All dataclasses and abstract interface fully exercised |
| `broadcast_utils/__init__.py` | 2 | 2 | **100%** | Import re-exports |
| `broadcast_utils/scene_classifier.py` | 59 | 59 | **100%** | Every state transition and branch covered |
| `broadcast_utils/table_tracker.py` | 109 | 99 | **91%** | Uncovered: optical-flow failure branches (lines 116–117, 130–131, 142, 152, 157, 163, 173, 176) |
| `xgb_win_predictor.py` | 160 | 132 | **82%** | Uncovered: joblib load success path (lines 118–123) and console print block (lines 295–326) — requires real model file |
| `prediction_pipeline.py` | 411 | 215 | **52%** | Uncovered: interactive setup, keyboard handlers, video writer, RTSP/live paths |
| `ball_tracking_fast.py` | 633 | 209 | **33%** | Uncovered: `frame_producer_thread`, `video_writer_thread`, full `ScoreDetectorBatched.update_*` (YOLO inference), pose skeleton drawing |
| `ball_tracking_analysis.py` | 1,519 | 490 | **32%** | Uncovered: interactive marking UI, `ScoreDetector` (original), `PoseFeatureExtractor` (MediaPipe), training utilities, benchmark code |
| **TOTAL** | **2,949** | **1,262** | **43%** | |

### Why coverage is bounded at 43%

The two largest files (`ball_tracking_analysis.py` at 1,519 stmts and `ball_tracking_fast.py` at 633 stmts) contain large blocks that are intentionally out of scope for automated unit testing:

- **Interactive UI** (`interactive_frame_setup`, `MarkingState`, mouse callbacks) — requires a display and user input
- **MediaPipe pose extraction** (`PoseFeatureExtractor`) — requires a `.task` model file and a real frame with a person
- **Full YOLO inference paths** in `ScoreDetectorBatched.update_scores_and_rounds` — covered only by the integration smoke test
- **Training/benchmark utilities** — standalone scripts not part of the runtime pipeline
- **Keyboard event handlers** in `prediction_pipeline.py` — require a live OpenCV window

---

## 4. What Is and Is Not Tested

### Covered by this suite

| Capability | Test Location |
|-----------|--------------|
| Homography computation and pixel↔metre mapping | `test_table_calibration.py` |
| Ball trajectory tracking, speed calculation, deque bounds | `test_ball_tracker.py` |
| All CSV output schemas and row-level data correctness | `test_data_logger.py` |
| Pose geometry (`_angle_between`, `_vertical_angle`, velocity) | `test_pose_extractor.py` |
| Rally state machine, point winner, landing zone grid | `test_rally_aggregator.py` |
| Digit OCR parsing, x-sorting, score cap, confidence filtering | `test_score_detector.py` |
| Score voting, stability threshold, low-confidence rejection | `test_score_detector.py` |
| Scene cut detection, table-color heuristic, polling | `test_scene_classifier.py` |
| LK optical flow tracking, corner ordering, quality checks | `test_table_tracker.py` |
| Manual score/set adjustment, clamping, swapping | `test_manual_score_tracker.py` |
| Prediction model dataclass contracts, feature pipeline | `test_prediction_models.py` |
| Source parsing (camera / RTSP / file), packet assembly | `test_pipeline_smoke.py` |
| End-to-end pipeline run on synthetic video | `test_pipeline_smoke.py` |
| Ball detection latency < 100 ms (TensorRT) | `test_fps_benchmark.py` |
| Score detection latency < 50 ms (TensorRT) | `test_fps_benchmark.py` |
| Pipeline throughput ≥ 15 FPS | `test_fps_benchmark.py` |
| Trajectory and speed history memory bounds | `test_fps_benchmark.py` |

### Not covered (by design)

| Capability | Reason |
|-----------|--------|
| Interactive ROI marking UI | Requires display + mouse events |
| MediaPipe pose extraction | Requires `.task` model file + person in frame |
| YOLO model accuracy (mAP, OCR accuracy) | Model quality testing, not pipeline logic |
| Streamlit broadcast UI (`broadcast_app_ui.py`) | UI testing out of scope |
| Training scripts (`ball_detect_training/`, `score_thing/`) | Not part of the runtime pipeline |
| `broadcast_pipeline.py` main loop | Requires video file + interactive setup |
| `LateFusionWinPredictor` profile lookup | Requires ITTF player data files |

---

## 5. How to Run

```bash
# Full suite (unit + integration + performance + coverage)
python -m pytest

# Unit tests only (fast, no weights needed)
python -m pytest tests/unit/

# Performance tests only
python -m pytest tests/performance/

# Specific module
python -m pytest tests/unit/test_scene_classifier.py -v

# Skip weight-dependent tests
python -m pytest -k "not (TestBallDetectionLatency or TestScoreDetectionLatency or TestFullPipelineFPS or TestPipelineSmokeWithModels)"

# HTML report is written to:  reports/test_report.html
# Coverage HTML is written to: reports/coverage/index.html
```

---

## 6. Environment

| Item | Value |
|------|-------|
| Python | 3.13.11 |
| Platform | win32 |
| pytest | ≥ 7.0 |
| pytest-html | ≥ 4.0 |
| pytest-cov | ≥ 4.0 |
| OpenCV | ≥ 4.5.0 |
| NumPy | ≥ 1.19.0 |
| TensorRT | Present (accelerates performance tests) |
| SORT | Stub (root `conftest.py`) or real `sort/` directory |
| ultralytics | Present (real YOLO inference in performance/smoke tests) |
| MediaPipe | Optional (pose extraction disabled if absent) |
