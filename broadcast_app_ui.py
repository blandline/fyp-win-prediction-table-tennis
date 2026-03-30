"""
Broadcast Data Collector — Configuration UI
============================================
Run with:  streamlit run broadcast_app_ui.py

Configures and launches broadcast_data_collector.py in a separate process.
The collector opens its own OpenCV window for video playback and interaction.
"""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Broadcast Data Collector",
    page_icon="📡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
COLLECTOR_SCRIPT = SCRIPT_DIR / "broadcast_data_collector.py"

BALL_SIZE_OPTIONS = {
    "Full resolution (no resize)": None,
    "1280 × 720": (1280, 720),
    "960 × 540": (960, 540),
    "640 × 360": (640, 360),
}

OUTPUT_SIZE_OPTIONS = {
    "Same as input": None,
    "1920 × 1080": (1920, 1080),
    "1280 × 720": (1280, 720),
    "960 × 540": (960, 540),
    "640 × 360": (640, 360),
}


def _size_to_str(size):
    if size is None:
        return None
    return f"{size[0]}x{size[1]}"


def _build_args(cfg):
    """Convert config dict to CLI argument list for broadcast_data_collector.py."""
    if cfg.get("camera_index") is not None:
        args = [sys.executable, str(COLLECTOR_SCRIPT), "--camera", str(cfg["camera_index"])]
    else:
        args = [sys.executable, str(COLLECTOR_SCRIPT), cfg["video_path"]]
    args += ["--output", cfg["output_dir"]]

    if cfg.get("roi_config_path"):
        args += ["--config", cfg["roi_config_path"]]

    if not cfg.get("save_video", True):
        args.append("--no-video")

    if cfg.get("score_mode") == "manual":
        args.append("--manual-scores")
        p1s = cfg.get("initial_scores", {}).get("player1", 0)
        p2s = cfg.get("initial_scores", {}).get("player2", 0)
        p1r = cfg.get("initial_rounds", {}).get("player1", 0)
        p2r = cfg.get("initial_rounds", {}).get("player2", 0)
        args += ["--initial-scores", f"{p1s},{p2s}"]
        args += ["--initial-rounds", f"{p1r},{p2r}"]
    else:
        args += ["--score-interval", str(cfg.get("score_interval_sec", 2.0))]

    names = cfg.get("player_names", [])
    if names and len(names) == 2 and any(names):
        n1 = names[0].strip() or "Player 1"
        n2 = names[1].strip() or "Player 2"
        args += ["--player-names", f"{n1},{n2}"]

    inf = _size_to_str(cfg.get("inference_size"))
    if inf:
        args += ["--ball-inference-size", inf]

    out = _size_to_str(cfg.get("output_size"))
    if out:
        args += ["--output-size", out]

    if cfg.get("broadcast_model"):
        args += ["--broadcast-model", cfg["broadcast_model"]]

    if cfg.get("prediction_model"):
        args += ["--prediction-model", cfg["prediction_model"]]

    if cfg.get("ittf_name1"):
        args += ["--ittf-name1", cfg["ittf_name1"]]
    if cfg.get("ittf_name2"):
        args += ["--ittf-name2", cfg["ittf_name2"]]

    if cfg.get("sets_to_win"):
        args += ["--sets-to-win", str(cfg["sets_to_win"])]

    if cfg.get("auto_scene"):
        args.append("--auto-scene")
        args += ["--table-color", cfg.get("table_color", "blue")]

    if cfg.get("auto_table_track"):
        args.append("--auto-table-track")

    return args


# ---------------------------------------------------------------------------
# Sidebar — keyboard reference
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📡 Broadcast Collector")
    st.markdown("---")
    st.subheader("Keyboard Controls")
    st.markdown("""
| Key | Action |
|-----|--------|
| **Q** | Quit |
| **P** | Pause / Resume |
| **F** | Toggle scene skip |
| **D** | Discard current rally |
| **U** | Undo rally discard |
| **X** | Swap player sides |
| **R** | Re-mark table corners |
| **S** | Screenshot |
| **+** / **-** | Speed up / slow down |
| **0** | Reset speed |
""")
    st.markdown("**Manual score mode only:**")
    st.markdown("""
| Key | Action |
|-----|--------|
| **1** | +1 point Player 1 |
| **2** | +1 point Player 2 |
| **[** | −1 point Player 1 |
| **]** | −1 point Player 2 |
| **3** | +1 set Player 1 |
| **4** | +1 set Player 2 |
| **Z** | Swap score display |
""")
    st.markdown("---")
    st.markdown("**F — Scene skip toggle** *(manual mode only)*")
    st.caption(
        "Press **F** when a cutscene or replay starts to pause inference. "
        "Press **F** again when gameplay resumes — this also resets the ball "
        "tracker so stale tracks from the cutscene don't bleed into the next rally. "
        "**Disabled when Auto scene classifier is enabled.**"
    )
    st.markdown("**D — Rally discard**")
    st.caption(
        "Press **D** when the camera angle is wrong (close-up, behind-player, etc.). "
        "Pose and trajectory data for that rally are not saved. "
        "Score tracking continues so the next rally starts cleanly. "
        "Press **U** to undo."
    )
    st.markdown("---")
    st.caption("After clicking **Start Collection**, an OpenCV window will open on your desktop.")


# ---------------------------------------------------------------------------
# Main form
# ---------------------------------------------------------------------------
st.title("Broadcast Data Collector — Setup")
st.markdown("Configure the settings below, then click **▶ Start Collection**.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Input Source")
    input_source = st.radio(
        "Input type",
        ["Video file", "Camera (live feed)"],
        horizontal=True,
        help="Choose between a recorded video file or a live camera feed.",
    )

    if input_source == "Video file":
        video_path = st.text_input(
            "Path to broadcast video file",
            placeholder="C:/Videos/match.mp4",
            help="Full path to the MP4, AVI, or MOV broadcast recording.",
        )
        camera_index = None
    else:
        camera_index = st.number_input(
            "Camera index",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Index of the camera to use (0 = default webcam, 1 = second camera, etc.).",
        )
        video_path = None

    st.subheader("2. Output")
    output_dir = st.text_input(
        "Output directory",
        value="broadcast_data",
        help="Folder where CSVs and the annotated video will be saved.",
    )
    save_video = st.checkbox("Save annotated output video", value=True)

    st.subheader("3. Player Names")
    p1_name = st.text_input("Player 1 name", value="Player 1")
    p2_name = st.text_input("Player 2 name", value="Player 2")

    st.subheader("3b. Match Format")
    _best_of = st.selectbox(
        "Best of (total sets)",
        options=[3, 5, 7],
        index=1,
        help="Total number of sets in this match format (e.g. Best of 5).",
    )
    _sets_to_win_options = {3: 2, 5: 3, 7: 4}
    sets_to_win = _sets_to_win_options[_best_of]
    st.caption(
        f"Sets needed to win: **{sets_to_win}** "
        f"(first to {sets_to_win} sets wins the match)"
    )

    st.subheader("4. Broadcast Ball Model (optional)")
    broadcast_model = st.text_input(
        "Path to fine-tuned broadcast ball model",
        placeholder="runs/detect/runs/ball_detector_broadcast/weights/best.pt",
        help="Leave blank to use the default ball model. "
             "Use the broadcast-trained model for better accuracy on TV footage.",
    )

    st.subheader("5. Win Prediction (optional)")
    _pred_mode = st.radio(
        "Prediction mode",
        [
            "Disabled",
            "CV features only",
            "Late fusion (CV + player profile)",
        ],
        help=(
            "Disabled: no win probability overlay.\n\n"
            "CV features only: uses pose/ball/rally data from the video "
            "(requires CV Pipeline/xgb_model.pkl).\n\n"
            "Late fusion: combines CV features with career stats / Elo / H2H "
            "from the ITTF player profile model "
            "(requires CV Pipeline/late_fusion_model.pkl and "
            "Player_Profile/*.pkl — run fusion_model_training.py first)."
        ),
    )

    _default_cv_predictor     = str(SCRIPT_DIR / "xgb_win_predictor.py")
    _default_fusion_predictor = str(SCRIPT_DIR / "late_fusion_win_predictor.py")

    prediction_model_path = None
    ittf_name1 = None
    ittf_name2 = None

    if _pred_mode == "CV features only":
        prediction_model_path = _default_cv_predictor

    elif _pred_mode == "Late fusion (CV + player profile)":
        prediction_model_path = _default_fusion_predictor
        st.caption(
            "Enter ITTF-format names so the profile model can look up career stats. "
            "Format: **LASTNAME Firstname** (e.g. `CALDERANO Hugo`). "
            "A partial name works too (substring search)."
        )
        ittf_col1, ittf_col2 = st.columns(2)
        with ittf_col1:
            ittf_name1 = st.text_input(
                "Player 1 ITTF name",
                placeholder="CALDERANO Hugo",
                help="Used for profile model lookup only.",
            )
        with ittf_col2:
            ittf_name2 = st.text_input(
                "Player 2 ITTF name",
                placeholder="MOREGARD Truls",
                help="Used for profile model lookup only.",
            )

with col2:
    st.subheader("6. Score Mode")
    score_mode_label = st.radio(
        "How should scores be tracked?",
        ["Auto (OCR from scoreboard)", "Manual entry (keyboard)"],
        help=(
            "Auto: reads the scoreboard from the video using the digit-detection model.\n"
            "Manual: press 1/2/[/] during playback to record each point."
        ),
    )
    score_mode = "auto" if "Auto" in score_mode_label else "manual"

    if score_mode == "auto":
        score_interval = st.slider(
            "Score detection interval (seconds)",
            min_value=0.5, max_value=5.0, value=2.0, step=0.5,
            help="How often to run OCR on the scoreboard.",
        )
        init_score_p1 = init_score_p2 = 0
        init_rounds_p1 = init_rounds_p2 = 0
    else:
        score_interval = 2.0
        st.markdown("**Starting scores (optional)**")
        sc1, sc2 = st.columns(2)
        with sc1:
            init_score_p1 = st.number_input("P1 starting score", min_value=0, max_value=20, value=0)
            init_rounds_p1 = st.number_input("P1 starting sets", min_value=0, max_value=7, value=0)
        with sc2:
            init_score_p2 = st.number_input("P2 starting score", min_value=0, max_value=20, value=0)
            init_rounds_p2 = st.number_input("P2 starting sets", min_value=0, max_value=7, value=0)

    st.subheader("7. ROI / Table Layout")
    roi_mode = st.radio(
        "Score region and table setup",
        ["First run — set up interactively", "Load saved layout"],
        help=(
            "First run: an OpenCV window opens for you to draw score regions "
            "and mark 4 table corners with the mouse.\n\n"
            "Load saved: provide the path to a previously saved config JSON."
        ),
    )
    roi_config_path = None
    if roi_mode == "Load saved layout":
        roi_config_path = st.text_input(
            "Path to saved config JSON",
            placeholder="broadcast_data/config_20250101_120000.json",
        )

# ---------------------------------------------------------------------------
# Advanced settings
# ---------------------------------------------------------------------------
with st.expander("Advanced settings"):
    adv1, adv2 = st.columns(2)
    with adv1:
        ball_inf_label = st.selectbox(
            "Ball inference size",
            list(BALL_SIZE_OPTIONS.keys()),
            index=0,
            help="Resize each frame before ball detection. Smaller = faster but may miss the ball.",
        )
    with adv2:
        out_size_label = st.selectbox(
            "Output video size",
            list(OUTPUT_SIZE_OPTIONS.keys()),
            index=0,
            help="Downscale the saved output video. Does not affect processing quality.",
        )

# ---------------------------------------------------------------------------
# Auto-detection features (experimental)
# ---------------------------------------------------------------------------
with st.expander("Auto-detection features (experimental)", expanded=False):
    st.caption(
        "These features were removed from the default pipeline due to reliability issues. "
        "Enable at your own risk — manual controls (F key, fixed corners) remain the default."
    )
    auto_scene = st.checkbox(
        "Auto scene classifier",
        value=False,
        help=(
            "Use frame-differencing + table-color heuristics to automatically detect "
            "cutscenes and replays. When enabled, the F key is ignored. "
            "Default: manual F key toggle."
        ),
    )
    table_color = st.selectbox(
        "Table color (for scene classifier)",
        ["blue", "green"],
        disabled=not auto_scene,
        help="Dominant table surface color used by the scene classifier to detect gameplay.",
    )
    auto_table_track = st.checkbox(
        "Auto table corner tracking (optical flow)",
        value=False,
        help=(
            "Track the 4 table corners frame-to-frame using Lucas-Kanade optical flow "
            "so the calibration adapts to slow camera pans. "
            "Default: fixed corners from the setup step."
        ),
    )

# ---------------------------------------------------------------------------
# Start button
# ---------------------------------------------------------------------------
st.markdown("---")
start_col, _ = st.columns([1, 3])
with start_col:
    start = st.button("▶ Start Collection", type="primary", use_container_width=True)

if start:
    errors = []
    if input_source == "Video file":
        if not video_path:
            errors.append("Video file path is required.")
        elif not Path(video_path).exists():
            errors.append(f"Video file not found: `{video_path}`")
    if not output_dir:
        errors.append("Output directory is required.")
    if roi_mode == "Load saved layout" and roi_config_path and not Path(roi_config_path).exists():
        errors.append(f"Config JSON not found: `{roi_config_path}`")
    if broadcast_model and not Path(broadcast_model).exists():
        errors.append(f"Broadcast model not found: `{broadcast_model}`")
    if prediction_model_path and not Path(prediction_model_path).exists():
        errors.append(f"Prediction model script not found: `{prediction_model_path}`")
    if _pred_mode == "Late fusion (CV + player profile)":
        if not ittf_name1:
            errors.append("Player 1 ITTF name is required for late fusion mode.")
        if not ittf_name2:
            errors.append("Player 2 ITTF name is required for late fusion mode.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        cfg = {
            "video_path": str(video_path) if video_path else None,
            "camera_index": int(camera_index) if camera_index is not None else None,
            "output_dir": str(output_dir),
            "player_names": [p1_name.strip() or "Player 1", p2_name.strip() or "Player 2"],
            "score_mode": score_mode,
            "score_interval_sec": score_interval,
            "roi_config_path": str(roi_config_path) if roi_config_path else None,
            "save_video": save_video,
            "inference_size": BALL_SIZE_OPTIONS[ball_inf_label],
            "output_size": OUTPUT_SIZE_OPTIONS[out_size_label],
            "broadcast_model": str(broadcast_model) if broadcast_model else None,
            "prediction_model": str(prediction_model_path) if prediction_model_path else None,
            "ittf_name1": ittf_name1.strip() if ittf_name1 else None,
            "ittf_name2": ittf_name2.strip() if ittf_name2 else None,
            "sets_to_win": int(sets_to_win),
            "best_of": int(_best_of),
            "auto_scene": bool(auto_scene),
            "table_color": str(table_color),
            "auto_table_track": bool(auto_table_track),
        }
        if score_mode == "manual":
            cfg["initial_scores"] = {"player1": int(init_score_p1), "player2": int(init_score_p2)}
            cfg["initial_rounds"] = {"player1": int(init_rounds_p1), "player2": int(init_rounds_p2)}

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        session_cfg_path = Path(output_dir) / "broadcast_session.json"
        with open(session_cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        cli_args = _build_args(cfg)

        _setup_note = (
            "An OpenCV setup window will open — use the mouse to mark score regions "
            "and table corners, then press **Enter** to start.\n\n"
            if not roi_config_path else ""
        )
        _source_note = (
            f"Using **camera index {cfg['camera_index']}** as the live input source.\n\n"
            if cfg.get("camera_index") is not None
            else ""
        )
        st.info(
            "**Collection pipeline launched!**\n\n"
            + _source_note
            + _setup_note
            + "The tracking window will open on your desktop. "
            "Press **Q** to quit when done."
        )
        with st.expander("Command being run"):
            st.code(" ".join(str(a) for a in cli_args))

        subprocess.Popen(cli_args, cwd=str(SCRIPT_DIR))

        st.success(f"Session config saved to `{session_cfg_path}`")

        st.markdown("### Keyboard reference (active during collection)")
        if score_mode == "manual":
            st.markdown("""
| Key | Action |
|-----|--------|
| P | Pause / Resume |
| Q | Quit |
| **F** | Toggle scene skip — pauses inference during cutscenes/replays |
| **D** | Discard current rally (bad camera angle) |
| **U** | Undo rally discard |
| X | Swap player sides (pose + scores) |
| R | Re-mark table corners |
| S | Screenshot |
| + / − | Speed up / slow down |
| 0 | Reset speed |
| **1** | +1 point — Player 1 |
| **2** | +1 point — Player 2 |
| **[** | −1 point — Player 1 (undo) |
| **]** | −1 point — Player 2 (undo) |
| **3** | +1 set — Player 1 |
| **4** | +1 set — Player 2 |
| **Z** | Swap score display only |
""")
        else:
            st.markdown("""
| Key | Action |
|-----|--------|
| P | Pause / Resume |
| Q | Quit |
| **F** | Toggle scene skip — pauses inference during cutscenes/replays |
| **D** | Discard current rally (bad camera angle) |
| **U** | Undo rally discard |
| X | Swap player sides (pose + scores) |
| R | Re-mark table corners |
| S | Screenshot |
| + / − | Speed up / slow down |
| 0 | Reset speed |
""")

        st.markdown("---")
        st.markdown("**Output files** (saved to `" + output_dir + "`):")
        st.markdown("""
- `trajectories_*.csv` — ball position per frame (skipped when rally is discarded)
- `pose_frames_*.csv` — player pose features per frame (skipped when rally is discarded)
- `scores_*.csv` — score changes
- `rallies_*.csv` — one row per point, includes `skipped` column (1 = discarded)
- `config_*.json` — ROI and model config
- `broadcast_session.json` — this session's settings
""")
