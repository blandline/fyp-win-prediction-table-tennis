# Table Tennis Match Analysis — CV & ML Pipeline

A computer vision and machine learning system for analysing table tennis broadcast footage. The system tracks the ball, reads the scoreboard, estimates player pose, segments rallies, and optionally predicts match outcomes in real time.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Model Weights Setup](#4-model-weights-setup)
5. [Running the Broadcast Pipeline](#5-running-the-broadcast-pipeline)
   - [Option A — Streamlit UI (recommended)](#option-a--streamlit-ui-recommended)
   - [Option B — Broadcast Data Collector (CLI)](#option-b--broadcast-data-collector-cli)
   - [Option C — Broadcast Pipeline (automated scene detection)](#option-c--broadcast-pipeline-automated-scene-detection)
6. [Other Pipelines](#6-other-pipelines)
   - [Ball Tracking (non-broadcast)](#ball-tracking-non-broadcast)
   - [Prediction Pipeline](#prediction-pipeline)
7. [Output Files](#7-output-files)
8. [Running Tests](#8-running-tests)

---

## 1. Project Overview

This repository is a full-stack table tennis analysis system built around YOLO-based object detection and classical computer vision. It is designed to process broadcast-quality match footage and extract structured, per-rally data that can be used for downstream machine learning.

**Core capabilities:**

- **Ball tracking** — YOLO detection + SORT multi-object tracker with trajectory smoothing and speed estimation.
- **Score reading** — Automatic OCR of the on-screen scoreboard using a fine-tuned digit detection model, or manual keyboard entry.
- **Rally segmentation** — Automatic detection of rally start/end events from ball visibility and score changes.
- **Scene classification** — Frame-differencing and table-colour heuristics to skip replays, cutscenes, and non-gameplay segments.
- **Optical-flow table tracking** — Lucas–Kanade corner tracking to compensate for camera movement.
- **Pose estimation** — Optional MediaPipe pose landmarks logged per player per rally.
- **Win prediction** — Optional XGBoost or late-fusion models that produce a live win-probability overlay.
- **Dataset preparation** — Scripts to convert collected CSVs into ML-ready feature datasets.

---

## 2. Project Structure

```
dataset_prep/
├── broadcast_pipeline.py        # Broadcast pipeline with automatic scene detection
├── broadcast_data_collector.py  # Broadcast data collector with manual scene gate
├── broadcast_app_ui.py          # Streamlit UI that launches the collector
├── ball_tracking_fast.py        # Optimised ball tracking (non-broadcast)
├── ball_tracking_analysis.py    # Reference implementation with all components
├── prediction_pipeline.py       # Live win-prediction overlay pipeline
├── xgb_win_predictor.py         # XGBoost win prediction model
├── late_fusion_win_predictor.py # Late-fusion (CV + player profile) predictor
├── prediction_model_base.py     # Abstract base class for prediction models
├── broadcast_utils/
│   ├── scene_classifier.py      # Frame-diff + HSV scene classifier
│   └── table_tracker.py         # LK optical-flow table corner tracker
├── CV Pipeline/
│   ├── ML_Dataset_Prep.py       # Build ML feature dataset from collected CSVs
│   ├── Model_training.py        # Train XGBoost model on CV features
│   ├── fusion_model_training.py # Train late-fusion meta-model
│   └── xgb_model.pkl            # Trained XGBoost model (not in repo, see §4)
├── Player_Profile/
│   ├── train_model.py           # Train player profile model from ITTF data
│   └── ittf_h2h_data_2026.csv   # H2H data used for profile model
├── ball_detect_training/        # Scripts for building and training ball datasets
├── segmentation_train/          # Table/player segmentation model training
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── requirements.txt
└── pose_landmarker_lite.task    # MediaPipe pose model (not in repo, see §4)
```

---

## 3. Installation

### Prerequisites

- Python 3.8 or higher
- A GPU with CUDA support is strongly recommended for real-time YOLO inference. CPU-only mode works but will be significantly slower.

### Steps

**1. Clone the repository**

```bash
git clone <repo-url>
cd dataset_prep
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

`requirements.txt` includes all required packages:

| Package | Purpose |
|---|---|
| `opencv-python` | Video I/O and image processing |
| `ultralytics` | YOLOv8 ball and digit detection |
| `filterpy` | Kalman filter used by SORT |
| `numpy`, `Pillow` | Array and image utilities |
| `streamlit` | Broadcast data collector UI |
| `mediapipe` | Player pose estimation |
| `xgboost`, `scikit-learn`, `pandas`, `joblib` | Win prediction models |
| `pytest`, `pytest-html`, `pytest-cov` | Test suite |

**4. Install the SORT tracker**

The SORT multi-object tracker is a separate repository that must be cloned into the `sort/` directory at the repo root. It is excluded from version control by `.gitignore`.

```bash
git clone https://github.com/abewley/sort.git sort
```

After cloning, the directory structure should be:

```
dataset_prep/
└── sort/
    ├── sort.py
    └── ...
```

---

## 4. Model Weights Setup

Several trained model weight files are required to run the pipelines. These files are excluded from the repository by `.gitignore` (all `*.pt` files and the `runs/` directory are ignored). Place the provided weights at the exact paths listed below so the scripts can find them without any extra configuration.

### Required directory structure

Create the following directories before placing the weights:

```bash
# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path "runs\detect\runs\ball_detector\weights"
New-Item -ItemType Directory -Force -Path "runs\detect\runs\ball_detector_broadcast\weights"
New-Item -ItemType Directory -Force -Path "runs\detect\runs\detect\digits_v2\weights"

# macOS / Linux
mkdir -p runs/detect/runs/ball_detector/weights
mkdir -p runs/detect/runs/ball_detector_broadcast/weights
mkdir -p runs/detect/runs/detect/digits_v2/weights
```

### Weight files

| File | Expected path (relative to repo root) | Required for |
|---|---|---|
| Ball detection model | `runs/detect/runs/ball_detector/weights/best.pt` | All tracking pipelines |
| Broadcast ball model | `runs/detect/runs/ball_detector_broadcast/weights/best.pt` | Broadcast pipelines (recommended over the base model for broadcast footage) |
| Digit detection model | `runs/detect/runs/detect/digits_v2/weights/best.pt` | Automatic score OCR (not needed with `--manual-scores`) |
| MediaPipe pose model | `pose_landmarker_lite.task` (repo root) | Pose feature logging (optional; pipeline continues without it) |
| XGBoost win model | `CV Pipeline/xgb_model.pkl` | Win prediction with `xgb_win_predictor.py` |
| Late fusion model | `CV Pipeline/late_fusion_model.pkl` | Late-fusion win prediction (falls back to weighted average if absent) |
| Player profile model | `Player_Profile/tt_model_2026.pkl` | Late-fusion player profile branch |
| Player ELO dict | `Player_Profile/elo_dict_2026.pkl` | Late-fusion player profile branch |
| Feature columns | `Player_Profile/feature_columns.pkl` | Late-fusion player profile branch |
| Processed player data | `Player_Profile/processed_df.pkl` | Late-fusion player profile branch (falls back to CSV if absent) |

> **Minimum setup for the broadcast pipeline:** you need the broadcast ball model (`ball_detector_broadcast/weights/best.pt`) and, if using automatic score reading, the digit model. The pose model and prediction models are optional.

### Overriding weight paths at runtime

All pipelines accept a `--broadcast-model` flag to specify an alternative ball model path:

```bash
python broadcast_data_collector.py match.mp4 --broadcast-model path/to/custom_ball.pt
```

---

## 5. Running the Broadcast Pipeline

The broadcast pipeline is the primary workflow for collecting data from ITTF/WTT broadcast footage. There are three ways to run it.

### Option A — Streamlit UI (recommended)

The Streamlit UI provides a graphical interface for configuring and launching the data collector. Run it from the repo root:

```bash
streamlit run broadcast_app_ui.py
```

This opens a browser window. Configure the session using the form fields:

| Field | Description |
|---|---|
| **Input** | Path to a video file, or a camera index for live capture |
| **Output directory** | Where CSVs and the annotated video will be saved |
| **Player names** | Display names shown on the score overlay |
| **Best of** | Match format (e.g. Best of 7 → sets to win = 4) |
| **Score mode** | Auto OCR (requires digit model) or manual keyboard entry |
| **Broadcast ball model** | Optional path to override the default ball weights |
| **Prediction mode** | Disabled, CV-only (XGBoost), or Late Fusion (requires ITTF names) |
| **ROI mode** | Interactive (draw regions on first frame) or load from saved JSON |
| **Ball inference size** | Resize frame before YOLO inference (e.g. `640x360`) |
| **Auto scene** *(experimental)* | Use automatic scene classifier instead of manual F-key gate |
| **Table tracking** *(experimental)* | Enable optical-flow table corner tracking |

Click **Start Collection** to launch the collector as a separate process. An OpenCV window will open for interactive calibration and playback.

---

### Option B — Broadcast Data Collector (CLI)

`broadcast_data_collector.py` is the recommended command-line entry point for broadcast footage. It uses a **manual scene gate** (press `F` to pause/resume inference during replays and cutscenes) and supports **rally skipping** (press `D` to discard a rally with a bad camera angle).

#### Basic usage

```bash
# With automatic score OCR
python broadcast_data_collector.py path/to/match.mp4 --output match1_data

# With manual score entry (no digit model needed)
python broadcast_data_collector.py path/to/match.mp4 --output match1_data --manual-scores

# With broadcast-specific ball model and player names
python broadcast_data_collector.py path/to/match.mp4 \
    --output match1_data \
    --broadcast-model runs/detect/runs/ball_detector_broadcast/weights/best.pt \
    --player-names "MOREGARD Truls,WANG Chuqin" \
    --manual-scores

# Resume from a saved config (skip interactive ROI setup)
python broadcast_data_collector.py path/to/match.mp4 \
    --output match1_data \
    --config match1_data/config.json \
    --manual-scores

# Live camera feed
python broadcast_data_collector.py --camera 0 --manual-scores --no-video
```

#### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `video` | — | Path to video file. Omit when using `--camera`. |
| `--camera INDEX` | — | Camera index for live capture (e.g. `0`). Mutually exclusive with `video`. |
| `--output`, `-o` | `broadcast_data` | Output directory for CSVs and video. |
| `--config`, `-c` | — | Path to a saved ROI config JSON to skip interactive setup. |
| `--no-video` | off | Do not save the annotated output video. |
| `--broadcast-model` | *(default ball model)* | Path to a fine-tuned broadcast ball YOLO model. |
| `--manual-scores` | off | Enter scores via keyboard instead of using OCR. |
| `--score-interval SEC` | `2.0` | Seconds between automatic OCR runs. |
| `--initial-scores P1,P2` | `0,0` | Starting point scores, e.g. `3,5`. |
| `--initial-rounds P1,P2` | `0,0` | Starting set counts, e.g. `1,2`. |
| `--player-names NAME1,NAME2` | — | Player display names, e.g. `Alice,Bob`. |
| `--sets-to-win N` | `3` | Number of sets needed to win (Best of 5 = 3, Best of 7 = 4). |
| `--ball-inference-size WxH` | — | Resize frame before ball inference, e.g. `640x360`. |
| `--output-size WxH` | — | Downscale the saved video, e.g. `640x360`. |
| `--no-async-write` | off | Disable background video-writer thread (use if memory is limited). |
| `--prediction-model PATH` | — | Path to a Python file with a `WinPredictionModel` subclass for live win prediction. |
| `--ittf-name1 NAME` | — | ITTF-format name of Player 1 for late-fusion profile lookup, e.g. `CALDERANO Hugo`. |
| `--ittf-name2 NAME` | — | ITTF-format name of Player 2 for late-fusion profile lookup. |
| `--auto-scene` | off | *(Experimental)* Use automatic scene classifier instead of the F-key gate. |
| `--table-color` | `blue` | Table colour preset for the auto scene classifier: `blue` or `green`. |
| `--auto-table-track` | off | *(Experimental)* Enable optical-flow table corner tracking. |

#### Keyboard controls (OpenCV window)

During playback, the following keys are active:

| Key | Action |
|---|---|
| `Q` | Quit and save all data |
| `P` | Pause / Resume playback |
| `F` | Toggle scene skip — pause inference during replays/cutscenes, resume when gameplay returns |
| `D` | Discard current rally (bad camera angle; score tracking continues) |
| `U` | Undo rally discard |
| `R` | Re-mark table corners interactively |
| `X` | Swap player sides (flips score ROIs and pose assignment) |
| `S` | Save screenshot |
| `+` / `-` | Speed up / slow down playback |
| `0` | Reset playback speed |
| `1` / `2` | +1 point for Player 1 / Player 2 *(manual score mode)* |
| `[` / `]` | −1 point for Player 1 / Player 2 *(manual score mode)* |
| `3` / `4` | +1 set for Player 1 / Player 2 *(manual score mode)* |
| `Z` | Swap score display *(manual score mode)* |

#### Interactive setup

On the first frame, an OpenCV window will prompt you to define regions of interest (ROIs):

1. **Table corners** — click the four corners of the table (top-left, top-right, bottom-right, bottom-left).
2. **Score ROIs** — draw bounding boxes around each player's point score and set score on the scoreboard.

These are saved to `<output_dir>/config.json` and can be reused with `--config` to skip setup on subsequent runs.

---

### Option C — Broadcast Pipeline (automated scene detection)

`broadcast_pipeline.py` is an alternative to the data collector. Instead of a manual F-key gate, it uses an **automatic scene classifier** that detects cuts and replays via frame differencing and table-colour analysis. The SORT tracker and table tracker are automatically reset on scene changes.

```bash
# Basic run with automatic scene detection
python broadcast_pipeline.py path/to/match.mp4 --output match1_output --manual-scores

# With broadcast ball model and player names
python broadcast_pipeline.py path/to/match.mp4 \
    --output match1_output \
    --broadcast-model runs/detect/runs/ball_detector_broadcast/weights/best.pt \
    --player-names "HARIMOTO Tomokazu,WANG Chuqin" \
    --manual-scores

# Disable automatic scene detection (process all frames)
python broadcast_pipeline.py path/to/match.mp4 --no-scene-filter --manual-scores

# Green table (e.g. Commonwealth Games footage)
python broadcast_pipeline.py path/to/match.mp4 --table-color green --manual-scores
```

#### Key differences from the data collector

| Feature | `broadcast_data_collector.py` | `broadcast_pipeline.py` |
|---|---|---|
| Scene handling | Manual F-key gate | Automatic classifier (disable with `--no-scene-filter`) |
| Rally skipping | Yes (D/U keys) | No |
| Win prediction | Optional (`--prediction-model`) | Not supported |
| Table tracking | Optional (`--auto-table-track`) | Always active when 4 corners are valid |

#### CLI arguments (broadcast_pipeline.py)

| Argument | Default | Description |
|---|---|---|
| `video` | — | Path to broadcast video file (required) |
| `--output`, `-o` | `broadcast_output` | Output directory |
| `--config`, `-c` | — | Saved ROI config JSON |
| `--no-video` | off | Do not save annotated video |
| `--broadcast-model` | *(default ball model)* | Override ball YOLO weights |
| `--table-color` | `blue` | Table colour for scene classifier: `blue` or `green` |
| `--no-scene-filter` | off | Disable automatic scene detection |
| `--manual-scores` | off | Manual keyboard scoring |
| `--score-interval SEC` | `2.0` | Auto OCR interval in seconds |
| `--initial-scores P1,P2` | `0,0` | Starting point scores |
| `--initial-rounds P1,P2` | `0,0` | Starting set counts |
| `--player-names NAME1,NAME2` | — | Player display names |
| `--ball-inference-size WxH` | — | Resize before ball inference |
| `--output-size WxH` | — | Downscale saved video |
| `--no-async-write` | off | Disable background writer thread |

---

## 6. Other Pipelines

### Ball Tracking (non-broadcast)

`ball_tracking_fast.py` is the optimised tracking pipeline for non-broadcast footage (e.g. fixed-camera recordings). It does not include scene classification or table tracking.

```bash
python ball_tracking_fast.py path/to/match.mp4 --output tracking_output --manual-scores
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `video` | — | Path to input video (required) |
| `--output`, `-o` | `tracking_output` | Output directory |
| `--config`, `-c` | — | Saved ROI config JSON |
| `--manual-scores` | off | Manual keyboard scoring |
| `--ball-inference-size WxH` | — | Resize before ball inference |
| `--output-size WxH` | — | Downscale saved video |
| `--score-interval SEC` | `2.0` | Auto OCR interval |
| `--initial-scores P1,P2` | `0,0` | Starting scores |
| `--initial-rounds P1,P2` | `0,0` | Starting set counts |
| `--player-names NAME1,NAME2` | — | Player display names |
| `--benchmark N` | — | Run headless ball-inference benchmark over N frames and exit |

### Prediction Pipeline

`prediction_pipeline.py` adds a live win-probability overlay to any video or camera feed. It accepts video files, camera indices, and RTSP/HTTP streams.

```bash
# With XGBoost predictor
python prediction_pipeline.py \
    --source path/to/match.mp4 \
    --model xgb_win_predictor.py \
    --output prediction_output \
    --manual-scores

# With late-fusion predictor and ITTF player names
python prediction_pipeline.py \
    --source path/to/match.mp4 \
    --model late_fusion_win_predictor.py \
    --output prediction_output \
    --manual-scores

# Live camera, headless (no display window)
python prediction_pipeline.py --source 0 --no-display --no-video
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--source`, `-s` | — | Video file path, camera index, or RTSP/HTTP URL (required) |
| `--model`, `-m` | — | Path to Python file with a `WinPredictionModel` subclass. Omit for 50/50 dummy. |
| `--output`, `-o` | `prediction_output` | Output directory |
| `--config`, `-c` | — | Saved ROI config JSON |
| `--no-video` | off | Do not save output video |
| `--no-display` | off | Headless mode (no OpenCV window) |
| `--manual-scores` | off | Manual keyboard scoring |
| `--ball-inference-size WxH` | — | Resize before ball inference |
| `--predict-interval N` | `1` | Run prediction model every N frames |

---

## 7. Output Files

Each pipeline writes its output to the directory specified by `--output`. The following files are generated:

| File | Description |
|---|---|
| `rallies_<timestamp>.csv` | Per-rally summary: frame range, timestamps, scores, ball speed statistics, landing zone counts, point winner, and whether the rally was skipped. |
| `pose_<timestamp>.csv` | Per-frame pose landmarks for both players (when MediaPipe is available). |
| `trajectory_<timestamp>.csv` | Per-frame ball position, speed, and SORT track ID. |
| `config.json` | Saved ROI and table corner configuration. Pass to `--config` to skip interactive setup on subsequent runs. |
| `broadcast_session.json` | Full session metadata including video path, player names, score mode, model paths, and match format. Used by downstream ML preparation scripts. |
| `output_<timestamp>.mp4` | Annotated video with ball tracks, score overlay, pose skeleton, and (if enabled) win probability bar. Only written when `--no-video` is not set. |

---

## 8. Running Tests

The test suite covers unit, integration, and performance tests.

```bash
# Run all tests
pytest

# Run with HTML report (output to reports/)
pytest --html=reports/report.html

# Run only unit tests
pytest tests/unit/

# Run with coverage
pytest --cov=. --cov-report=html
```

> **Note:** The SORT tracker stub in `conftest.py` ensures unit tests run even when the `sort/` directory is not present. Integration and performance tests may require the full environment including model weights.
