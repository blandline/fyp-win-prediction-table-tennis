# Testing suite: what was tested and what the reports show

This document describes each automated test in the project, how tests are grouped, and how to interpret the generated **pytest HTML report** (`reports/test_report.html`) and **coverage report** (`reports/coverage/index.html`). Figures in the “outcome” section match a recent full `pytest` run on this project (89 tests). Total duration varies (roughly 7–20 s) with CPU/GPU and whether performance tests run real inference.

---

## 1. How to read the pytest HTML report

The file `reports/test_report.html` is a self-contained HTML page produced by **pytest-html**. It includes:

- **Summary counts**: total tests, passed, failed, skipped, and total duration.
- **Environment**: Python version, platform, pytest and plugin versions.
- **Per-test table**: each row is one test with **result** (Passed / Failed), **node id** (file and test name), and **duration**.
- **Failures**: expanded tracebacks and assertion messages for any failed test.

When you open the report in a browser, use the filters at the top to show only failed tests or only passed tests. Failed rows appear in red; passed rows in green.

---

## 2. How to read the coverage HTML report

The directory `reports/coverage/` is produced by **pytest-cov** and **coverage.py**. Open `reports/coverage/index.html` in a browser to see:

- **Overall and per-file coverage** for the packages configured in `pytest.ini`: `ball_tracking_analysis`, `ball_tracking_fast`, `prediction_pipeline`, and `prediction_model_base`.
- **Line-level highlighting**: executed lines vs missing lines for each file.

Coverage answers “which lines of the main modules were exercised during the test run,” not “whether the logic is correct.” High coverage on a module still allows failing tests elsewhere; low coverage means large parts of a file (often UI, training, or alternate code paths) were not run.

---

## 3. Test layout (by folder)

| Folder | Role |
|--------|------|
| `tests/unit/` | Isolated checks of one component (mocks, no full video pipeline unless noted). |
| `tests/integration/` | Cross-module behaviour: source parsing, packet building, end-to-end pipeline smoke. |
| `tests/performance/` | Latency/FPS thresholds on real YOLO weights when present; memory-bound structure checks. |

Shared fixtures live in `tests/conftest.py` and `tests/fixtures/synthetic_frames.py`. Root `conftest.py` supplies a **SORT stub** when `sort/` is missing and can stub **ultralytics** if it is not installed.

---

## 4. Integration tests (`tests/integration/test_pipeline_smoke.py`)

These verify the prediction stack and pipeline wiring.

| Test | What it exercises |
|------|-------------------|
| `TestPredictionModelBase::test_dummy_model_returns_valid_result` | `DummyPredictionModel.predict` returns a `PredictionResult` with 50/50 probabilities and `model_ready=False`. |
| `TestPredictionModelBase::test_data_packet_defaults` | `PredictionDataPacket` default state: frame index, ball not detected, scores unset, rally inactive, no pose. |
| `TestPredictionModelBase::test_prediction_result_fields` | Constructing `PredictionResult` with explicit probabilities and flags. |
| `TestSourceParsing::test_integer_source` | `parse_source("0")` → camera index `0`, live. |
| `TestSourceParsing::test_rtsp_source` | RTSP URL parsed as live stream. |
| `TestSourceParsing::test_file_source` | File path parsed as non-live. |
| `TestSourceParsing::test_http_source` | HTTP URL treated as live. |
| `TestBuildPacket::test_build_packet_no_detections` | `build_packet` with fake tracker/score detector and empty tracks yields correct frame index, no ball detection, default scores, inactive rally. |
| `TestPipelineSmokeWithModels::test_pipeline_runs_on_synthetic_video` | **Requires** ball and digit weights under `runs/...`. Writes a short synthetic MP4, runs `run_prediction_pipeline` headlessly with `DummyPredictionModel`, asserts at least three CSV files in the output directory. |

**Report implication:** If weights are missing, the smoke test is **skipped** (not failed); your HTML report would show fewer collected tests or an orange “Skipped” row for that test. With weights present, it appears as **Passed** when the pipeline completes and CSVs are written.

---

## 5. Performance tests (`tests/performance/test_fps_benchmark.py`)

| Test | What it exercises |
|------|-------------------|
| `TestBallDetectionLatency::test_single_inference_under_threshold` | **Requires weights.** `BallTracker` single-frame inference median under 100 ms (after warm-up). |
| `TestScoreDetectionLatency::test_batched_score_inference_under_threshold` | **Requires weights.** `ScoreDetectorBatched` on four ROIs median under 50 ms. |
| `TestFullPipelineFPS::test_fps_above_minimum` | **Requires weights.** 50 synthetic frames through ball detection; FPS must be ≥ 15. |
| `TestMemoryStability::test_trajectory_memory_bounded` | Trajectory `deque` with `maxlen=TRAJECTORY_LENGTH` does not grow past the cap. |
| `TestMemoryStability::test_speed_history_bounded` | Speed history `deque` respects `SPEED_SMOOTHING_WINDOW`. |

**Report implication:** Weight-dependent tests are **skipped** when `best.pt` paths are missing; they **pass** when hardware meets the asserted thresholds. The HTML report lists each with Passed or Skipped. Stdout timing lines (median ms, FPS) appear in the terminal when using `pytest -s`, not always in the HTML table.

---

## 6. Unit tests — ball tracker (`tests/unit/test_ball_tracker.py`)

Uses a **mock YOLO** (`MockYOLOModel`) and real `Sort` (or the root stub). Covers `BallTracker` behaviour from `ball_tracking_analysis`.

| Test | What it exercises |
|------|-------------------|
| `TestSpeedCalculation::test_speed_zero_when_stationary` | Repeated identical detections → mean speed in history stays near zero. |
| `TestSpeedCalculation::test_speed_calculation_known_distance` | Analytic check: 30 px per frame at 30 FPS → 900 px/s. |
| `TestSpeedCalculation::test_smoothed_speed_returns_zero_for_unknown_track` | Unknown track id → smoothed speed 0. |
| `TestSpeedCalculation::test_smoothed_speed_mps_returns_none_for_unknown_track` | Unknown track → m/s is `None`. |
| `TestTrajectory::test_trajectory_capped_at_max_length` | Trajectory length never exceeds `TRAJECTORY_LENGTH`. |
| `TestTrajectory::test_trajectory_grows_with_detections` | At least two points stored after two frames with detections. |
| `TestDetectionFiltering::test_no_detection_on_blank_frame` | No detections → no tracks. |
| `TestDetectionFiltering::test_low_confidence_filtered` | Confidence below ball threshold → nothing passed to tracker. |
| `TestTrackPersistence::test_track_persists_during_short_gap` | After establishing a track, blank frames for a few steps — expects track to still appear (**sensitive to SORT implementation**). |
| `TestTrackPersistence::test_track_lost_after_max_age` | After long gap, no active tracks. |
| `TestMetreSpace::test_position_meters_none_without_calibration` | No table calibration → metre position `None`. |
| `TestMetreSpace::test_set_table_calibration` | `set_table_calibration` stores the calibration object. |
| `TestInferenceRescaling::test_prepare_inference_frame_no_resize` | No `inference_size` → scale 1.0, same shape. |
| `TestInferenceRescaling::test_prepare_inference_frame_with_resize` | `inference_size` set → resized frame and scale factors ~2.0 for 1280×720 → 640×360. |

**Report implication:** One failure in this module indicates a mismatch between the test’s expectation of **track persistence during occlusion gaps** and the actual **SORT** behaviour (real `sort` package vs minimal stub in root `conftest.py`). Other tests in this file validate speed math, filtering, trajectory caps, and inference rescaling.

---

## 7. Unit tests — data logger (`tests/unit/test_data_logger.py`)

| Test | What it exercises |
|------|-------------------|
| `TestFileCreation::test_output_directory_created` | `DataLogger` creates the output directory. |
| `TestFileCreation::test_csv_files_created` | Trajectory, score, rally, and pose CSV paths exist. |
| `TestTrajectoryCSV::test_trajectory_header_with_meters` | Header includes metre and m/s columns when `with_meters=True`. |
| `TestTrajectoryCSV::test_trajectory_header_without_meters` | Reduced header without metre columns. |
| `TestTrajectoryCSV::test_log_trajectory_row` | One logged row matches frame, track id, coordinates. |
| `TestScoreCSV::test_score_csv_header` | Score file header contains expected column names. |
| `TestScoreCSV::test_log_score_row` | Player scores written correctly. |
| `TestRallyCSV::test_rally_csv_header` | Rally columns include `rally_id`, `point_winner`, landing zones. |
| `TestRallyCSV::test_log_rally_row` | Full rally record flushes one data row with expected ids and winner. |
| `TestPoseCSV::test_pose_csv_header` | Pose columns include lean and hand speed fields. |
| `TestPoseCSV::test_log_pose_row` | One pose row appended. |
| `TestConfigRoundTrip::test_save_and_reload_config` | `save_config` JSON contains video path, fps, corners, ROIs. |

---

## 8. Unit tests — pose helpers (`tests/unit/test_pose_extractor.py`)

Tests pure geometry helpers (`_angle_between`, `_vertical_angle`) and velocity formulas **without MediaPipe**.

| Test | What it exercises |
|------|-------------------|
| `TestAngleBetween::*` | Collinear ~180°, right angle ~90°, acute ~45°, `None` for missing points or degenerate segments. |
| `TestVerticalAngle::*` | Vertical alignment ~0°, left/right lean sign, horizontal ~90°, `None` guards. |
| `TestVelocityLogic::*` | Normalised hand displacement / dt for speed; torso horizontal velocity positive when x increases. |

---

## 9. Unit tests — rally aggregator (`tests/unit/test_rally_aggregator.py`)

| Test | What it exercises |
|------|-------------------|
| `TestRallyStateTransitions::*` | Initial `STATE_BETWEEN_POINTS`; transition to `STATE_RALLY_ACTIVE` after `RALLY_BALL_SEEN_FRAMES` with ball; no early start; return to between-points after `RALLY_BALL_MISSING_FRAMES` without ball. |
| `TestPointWinner::*` | Score change (5,3)→(5,4) logs rally with winner `p2`; simultaneous increments → `unknown`. |
| `TestLandingZones::*` | 3×3 grid indices for centre, corner, outside table, and all nine zones reachable. |
| `TestSpeedAggregation::*` | Flushed rally record has consistent mean vs max speed when present; `flush_final` emits one row. |

---

## 10. Unit tests — score / digits (`tests/unit/test_score_detector.py`)

| Test | What it exercises |
|------|-------------------|
| `TestParseDigitResult::*` | `_parse_digit_result` from `ball_tracking_fast`: single digit, 11 from two 1s, x-sorting for two-digit scores, non-digit classes ignored, max score cap, empty input, digit 0, at most two digits contribute to score while counting detections. |
| `TestScoreVotingAndStability::*` | Majority voting over history, minimum three readings, stable score after `SCORE_STABLE_RUNS`, reset when vote changes, ignoring readings below `MIN_DIGIT_CONF_RELIABLE`. |

---

## 11. Unit tests — table calibration (`tests/unit/test_table_calibration.py`)

| Test | What it exercises |
|------|-------------------|
| `TestTableCalibrationValid::*` | Valid corners → `is_valid`, 3×3 `H`, low reprojection error, pixel centre maps near table centre in metres, pixel↔metre round-trip under 2 px, sanity for far points. |
| `TestTableCalibrationInvalid::*` | Too few corners invalid or error; collapsed quad invalid; tiny quad invalid; bad quad still yields a numeric reprojection error when `H` exists. |

---

## 12. What the generated reports showed (representative run)

The following matches the embedded summary in `reports/test_report.html` from a successful report generation:

| Metric | Value |
|--------|--------|
| Total tests | 89 |
| Passed | 88 |
| Failed | 1 |
| Skipped | 0 |
| Duration | Varies; HTML report shows the wall-clock for the run that generated it |

The single failure was **`tests/unit/test_ball_tracker.py::TestTrackPersistence::test_track_persists_during_short_gap`**: after five frames with detections and three blank frames, the test expected at least one frame with a non-empty track list, but received none. That is consistent with using a **minimal SORT stub** that does not mirror full track coasting, or with stricter track expiry than the test assumes.

**Coverage (same run, from `reports/coverage/`):**

| Module | Statements (approx.) | Covered | Coverage (approx.) |
|--------|----------------------|---------|---------------------|
| `ball_tracking_analysis.py` | 1515 | ~501 | ~33% |
| `ball_tracking_fast.py` | 628 | ~185 | ~29% |
| `prediction_model_base.py` | 56 | ~55 | ~98% |
| `prediction_pipeline.py` | 411 | ~217 | ~53% |
| **All measured** | **~2610** | **~958** | **~37%** |

The coverage report highlights that most of the large tracking and fast-pipeline files are **only partly** exercised: many branches (interactive setup, alternate modes, training-only paths) are not hit by the current tests. `prediction_model_base.py` is almost fully covered because the tests heavily use its dataclasses and dummy model.

---

## 13. What is *not* covered by this pytest suite

- Standalone scripts such as `score_thing/test_scoreboard_on_video.py` and `segmentation_train/test_masks.py` are **not** pytest tests; they are manual or demo utilities.
- Training scripts, Streamlit UI (`app_ui.py`), and broadcast-specific modules are outside the four `--cov` packages unless imported indirectly.
- Model **accuracy** (mAP, digit OCR accuracy on real scoreboards) is not measured here—only parsing rules, stability logic, and optional latency/FPS thresholds.

---

## 14. Regenerating the reports

From the repository root:

```bash
python -m pytest
```

This refreshes `reports/test_report.html` and `reports/coverage/` per `pytest.ini`. After regenerating, update Section 12 if you need the written document to match a new run exactly (counts and coverage percentages will change with code and environment).
