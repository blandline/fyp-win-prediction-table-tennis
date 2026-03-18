"""
Table Tennis Tracking — Configuration UI
=========================================
Run with:  streamlit run app_ui.py

This app collects all settings and launches ball_tracking_fast.py in a
separate process. The tracking pipeline opens its own OpenCV window.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Table Tennis Tracker",
    page_icon="🏓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
FAST_SCRIPT = SCRIPT_DIR / "ball_tracking_fast.py"

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
    """Convert the config dict into a list of CLI arguments for ball_tracking_fast.py."""
    args = [sys.executable, str(FAST_SCRIPT), cfg["video_path"]]
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
    return args


# ---------------------------------------------------------------------------
# Sidebar — keyboard reference
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏓 TT Tracker")
    st.markdown("---")
    st.subheader("Keyboard Controls")
    st.markdown("""
| Key | Action |
|-----|--------|
| **P** | Pause / Resume |
| **Q** | Quit |
| **X** | Swap sides (pose + scores) |
| **S** | Screenshot |
| **+** / **-** | Speed up / down |
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
    st.caption("After clicking **Start Processing**, an OpenCV window will open on your desktop.")


# ---------------------------------------------------------------------------
# Main form
# ---------------------------------------------------------------------------
st.title("Table Tennis Match Tracker — Setup")
st.markdown("Fill in the settings below, then click **▶ Start Processing**.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Video File")
    video_path = st.text_input(
        "Path to video file",
        placeholder="C:/Videos/match.mp4",
        help="Full path to the MP4, AVI, or MOV file you want to process.",
    )

    st.subheader("2. Output")
    output_dir = st.text_input(
        "Output directory",
        value="tracking_output",
        help="Folder where logs, CSVs, and the output video will be saved.",
    )
    save_video = st.checkbox("Save annotated output video", value=True)

    st.subheader("3. Player Names")
    p1_name = st.text_input("Player 1 name", value="Player 1")
    p2_name = st.text_input("Player 2 name", value="Player 2")

with col2:
    st.subheader("4. Score Mode")
    score_mode_label = st.radio(
        "How should scores be tracked?",
        ["Auto (OCR from scoreboard)", "Manual entry (keyboard)"],
        help=(
            "Auto: reads the scoreboard from the video using a digit-detection model.\n"
            "Manual: you press keys during playback to record each point."
        ),
    )
    score_mode = "auto" if "Auto" in score_mode_label else "manual"

    if score_mode == "auto":
        score_interval = st.slider(
            "Score detection interval (seconds)",
            min_value=0.5, max_value=5.0, value=2.0, step=0.5,
            help="How often to run OCR on the scoreboard. Lower = more frequent but more CPU.",
        )
    else:
        score_interval = 2.0
        st.markdown("**Starting scores (optional)**")
        sc1, sc2 = st.columns(2)
        with sc1:
            init_score_p1 = st.number_input("P1 starting score", min_value=0, max_value=20, value=0)
            init_rounds_p1 = st.number_input("P1 starting sets", min_value=0, max_value=5, value=0)
        with sc2:
            init_score_p2 = st.number_input("P2 starting score", min_value=0, max_value=20, value=0)
            init_rounds_p2 = st.number_input("P2 starting sets", min_value=0, max_value=5, value=0)

    st.subheader("5. ROI / Table Layout")
    roi_mode = st.radio(
        "Score region and table setup",
        ["First run — set up interactively", "Load saved layout"],
        help=(
            "First run: when you click Start, an OpenCV window opens on your screen. "
            "Use the mouse to draw the score regions and mark 4 table corners. "
            "The layout is saved automatically.\n\n"
            "Load saved: provide the path to a previously saved config JSON."
        ),
    )
    roi_config_path = None
    if roi_mode == "Load saved layout":
        roi_config_path = st.text_input(
            "Path to saved config JSON",
            placeholder="tracking_output/config.json",
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
            help="Resize each frame before ball detection. Smaller = faster but may miss the ball at distance.",
        )
    with adv2:
        out_size_label = st.selectbox(
            "Output video size",
            list(OUTPUT_SIZE_OPTIONS.keys()),
            index=0,
            help="Downscale the saved output video. Does not affect processing.",
        )

# ---------------------------------------------------------------------------
# Start button
# ---------------------------------------------------------------------------
st.markdown("---")
start_col, _ = st.columns([1, 3])
with start_col:
    start = st.button("▶ Start Processing", type="primary", use_container_width=True)

if start:
    # --- Validate ---
    errors = []
    if not video_path:
        errors.append("Video file path is required.")
    elif not Path(video_path).exists():
        errors.append(f"Video file not found: `{video_path}`")
    if not output_dir:
        errors.append("Output directory is required.")
    if roi_mode == "Load saved layout" and roi_config_path and not Path(roi_config_path).exists():
        errors.append(f"Config JSON not found: `{roi_config_path}`")

    if errors:
        for e in errors:
            st.error(e)
    else:
        # --- Build config dict ---
        cfg = {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "player_names": [p1_name.strip() or "Player 1", p2_name.strip() or "Player 2"],
            "score_mode": score_mode,
            "score_interval_sec": score_interval,
            "roi_config_path": str(roi_config_path) if roi_config_path else None,
            "save_video": save_video,
            "inference_size": BALL_SIZE_OPTIONS[ball_inf_label],
            "output_size": OUTPUT_SIZE_OPTIONS[out_size_label],
        }
        if score_mode == "manual":
            cfg["initial_scores"] = {"player1": int(init_score_p1), "player2": int(init_score_p2)}
            cfg["initial_rounds"] = {"player1": int(init_rounds_p1), "player2": int(init_rounds_p2)}

        # --- Save session config ---
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        session_cfg_path = Path(output_dir) / "app_session.json"
        with open(session_cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # --- Build CLI args and launch ---
        cli_args = _build_args(cfg)
        st.info(
            "**Pipeline launched!**\n\n"
            + ("An OpenCV setup window will open — follow the on-screen instructions to mark score regions and table corners.\n\n"
               if not roi_config_path else "")
            + "The tracking window will open on your desktop. You can close it with **Q**."
        )
        with st.expander("Command being run"):
            st.code(" ".join(str(a) for a in cli_args))

        # Launch in background so Streamlit doesn't block
        subprocess.Popen(cli_args, cwd=str(SCRIPT_DIR))

        st.success(f"Session config saved to `{session_cfg_path}`")
        st.markdown("### Keyboard reference (active during tracking)")
        if score_mode == "manual":
            st.markdown("""
| Key | Action |
|-----|--------|
| P | Pause / Resume |
| Q | Quit |
| X | Swap sides (pose + scores together) |
| **1** | +1 point — Player 1 |
| **2** | +1 point — Player 2 |
| **[** | −1 point — Player 1 (undo) |
| **]** | −1 point — Player 2 (undo) |
| **3** | +1 set — Player 1 |
| **4** | +1 set — Player 2 |
| **Z** | Swap score display only |
| S | Screenshot |
| + / − | Speed up / down |
""")
        else:
            st.markdown("""
| Key | Action |
|-----|--------|
| P | Pause / Resume |
| Q | Quit |
| X | Swap sides (pose + scores) |
| S | Screenshot |
| + / − | Speed up / down |
| 0 | Reset speed |
""")
