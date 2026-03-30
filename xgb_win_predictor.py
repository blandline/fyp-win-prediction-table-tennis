"""
XGBoost Win Predictor
=====================
Implements WinPredictionModel using the trained XGBoost model from
'CV Pipeline/xgb_model.pkl'. Computes the same 19 features used during
training (ML_Dataset_Prep.py) in real-time from PredictionDataPacket data.

Feature computation:
- Per-rally pose features are buffered during each rally.
- On each point scored (on_point_scored), features are extracted, the
  rally is appended to history, and the full 19-feature vector is built
  from cumulative + last-3-rally means + score context.
- predict() returns the last cached result instantly (no per-frame inference).

Console output on each point:
  [Rally 3] P1: 61.5% | P2: 38.5% | Score: 3-2 | conf: 0.72
"""

from __future__ import annotations

import os
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Resolve model path relative to this file's directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_MODEL_PATH = _HERE / "CV Pipeline" / "xgb_model.pkl"

# ---------------------------------------------------------------------------
# Lazy import of joblib / xgboost so the pipeline starts even if unavailable
# ---------------------------------------------------------------------------
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from prediction_model_base import (
    WinPredictionModel,
    PredictionDataPacket,
    PredictionResult,
)


# ---------------------------------------------------------------------------
# Feature names — must match the column order used in Model_training.py
# ---------------------------------------------------------------------------
_FEATURE_NAMES = [
    "hand_speed_diff",
    "com_std_diff",
    "movement_diff",
    "elbow_diff",
    "visibility_avg",
    "ball_speed_mean",
    "ball_speed_max",
    "rally_duration_frames",
    "hand_speed_diff_last3",
    "com_std_diff_last3",
    "movement_diff_last3",
    "elbow_diff_last3",
    "visibility_avg_last3",
    "ball_speed_mean_last3",
    "ball_speed_max_last3",
    "rally_duration_frames_last3",
    "score_diff",
    "total_points",
    "rally_number",
]

_RALLY_FEATURE_KEYS = [
    "hand_speed_diff",
    "com_std_diff",
    "movement_diff",
    "elbow_diff",
    "visibility_avg",
    "ball_speed_mean",
    "ball_speed_max",
    "rally_duration_frames",
]


def _safe_mean(values: list) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


class XGBoostWinPredictor(WinPredictionModel):
    """
    Real-time XGBoost win predictor.

    Loads 'CV Pipeline/xgb_model.pkl' and computes the same 19 features
    used during training after each point is confirmed.
    """

    def __init__(self, model_path: Optional[str] = None, player_names: Optional[list] = None):
        self._model = None
        self._load_error: Optional[str] = None
        self._player_names = player_names or ["Player 1", "Player 2"]

        path = Path(model_path) if model_path else _MODEL_PATH
        if not _JOBLIB_AVAILABLE:
            self._load_error = "joblib not installed — cannot load XGBoost model"
            print(f"[XGBoostWinPredictor] WARNING: {self._load_error}")
        elif not path.exists():
            self._load_error = f"Model file not found: {path}"
            print(f"[XGBoostWinPredictor] WARNING: {self._load_error}")
        else:
            try:
                self._model = joblib.load(str(path))
                print(f"[XGBoostWinPredictor] Loaded model from {path}")
            except Exception as exc:
                self._load_error = str(exc)
                print(f"[XGBoostWinPredictor] ERROR loading model: {exc}")

        # Rally history: list of per-rally feature dicts
        self._rally_history: list[dict] = []

        # Pose buffers for the *current* rally (reset on each point)
        self._pose_p1_hand_speed: list[float] = []
        self._pose_p1_com_height: list[float] = []
        self._pose_p1_torso_x: list[float] = []
        self._pose_p1_elbow: list[float] = []
        self._pose_p1_vis: list[float] = []

        self._pose_p2_hand_speed: list[float] = []
        self._pose_p2_com_height: list[float] = []
        self._pose_p2_torso_x: list[float] = []
        self._pose_p2_elbow: list[float] = []
        self._pose_p2_vis: list[float] = []

        # Ball speed samples for the current rally
        self._ball_speeds: list[float] = []
        self._rally_start_frame: Optional[int] = None
        self._rally_end_frame: Optional[int] = None

        # Cached result — returned by predict() every frame
        self._cached_result = PredictionResult(
            player1_win_prob=0.5,
            player2_win_prob=0.5,
            confidence=0.0,
            model_ready=False,
        )
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _ingest_packet_pose(self, packet: PredictionDataPacket) -> None:
        """Buffer per-frame pose values from the packet into current-rally lists."""
        p1 = packet.pose.player1
        p2 = packet.pose.player2

        if p1:
            if "v_hand_speed" in p1:
                self._pose_p1_hand_speed.append(float(p1["v_hand_speed"] or 0))
            if "com_height" in p1:
                self._pose_p1_com_height.append(float(p1["com_height"] or 0))
            if "v_torso_x" in p1:
                self._pose_p1_torso_x.append(abs(float(p1["v_torso_x"] or 0)))
            if "angle_elbow_dom" in p1:
                self._pose_p1_elbow.append(float(p1["angle_elbow_dom"] or 0))
            if "visibility_mean" in p1:
                self._pose_p1_vis.append(float(p1["visibility_mean"] or 0))

        if p2:
            if "v_hand_speed" in p2:
                self._pose_p2_hand_speed.append(float(p2["v_hand_speed"] or 0))
            if "com_height" in p2:
                self._pose_p2_com_height.append(float(p2["com_height"] or 0))
            if "v_torso_x" in p2:
                self._pose_p2_torso_x.append(abs(float(p2["v_torso_x"] or 0)))
            if "angle_elbow_dom" in p2:
                self._pose_p2_elbow.append(float(p2["angle_elbow_dom"] or 0))
            if "visibility_mean" in p2:
                self._pose_p2_vis.append(float(p2["visibility_mean"] or 0))

    def _extract_rally_features(self, packet: PredictionDataPacket) -> dict:
        """Build a per-rally feature dict matching extract_rally_features() in ML_Dataset_Prep.py."""
        feat: dict = {}

        feat["hand_speed_diff"] = (
            _safe_mean(self._pose_p1_hand_speed) - _safe_mean(self._pose_p2_hand_speed)
        )
        feat["com_std_diff"] = (
            _safe_std(self._pose_p1_com_height) - _safe_std(self._pose_p2_com_height)
        )
        feat["movement_diff"] = (
            _safe_mean(self._pose_p1_torso_x) - _safe_mean(self._pose_p2_torso_x)
        )
        feat["elbow_diff"] = (
            _safe_mean(self._pose_p1_elbow) - _safe_mean(self._pose_p2_elbow)
        )
        feat["visibility_avg"] = (
            _safe_mean(self._pose_p1_vis) + _safe_mean(self._pose_p2_vis)
        ) / 2.0

        # Ball stats from the packet's rally state (already aggregated by the pipeline)
        feat["ball_speed_mean"] = float(packet.rally.mean_speed_mps or 0)
        feat["ball_speed_max"] = float(packet.rally.max_speed_mps or 0)

        # Rally duration in frames — rally_duration_sec * fps; fall back to frame diff
        fps_est = packet.frame_idx / max(packet.timestamp_sec, 1e-6) if packet.timestamp_sec > 0 else 30.0
        feat["rally_duration_frames"] = float(packet.rally.rally_duration_sec * fps_est)

        return feat

    def _build_feature_vector(self, packet: PredictionDataPacket) -> Optional[np.ndarray]:
        """Compute the full 19-feature vector from rally_history + current score."""
        if not self._rally_history:
            return None

        history_keys = _RALLY_FEATURE_KEYS
        n = len(self._rally_history)

        # Cumulative means over all rallies
        cumulative: dict = {}
        for k in history_keys:
            cumulative[k] = _safe_mean([r[k] for r in self._rally_history])

        # Last-3 rally means
        last3 = self._rally_history[-3:]
        for k in history_keys:
            cumulative[f"{k}_last3"] = _safe_mean([r[k] for r in last3])

        # Score features
        p1_score = packet.score.player1_score or 0
        p2_score = packet.score.player2_score or 0
        cumulative["score_diff"] = float(p1_score - p2_score)
        cumulative["total_points"] = float(p1_score + p2_score)
        cumulative["rally_number"] = float(n)

        row = [cumulative[k] for k in _FEATURE_NAMES]
        return np.array(row, dtype=np.float32).reshape(1, -1)

    def _reset_rally_buffers(self) -> None:
        """Clear per-rally pose + ball buffers at the start of a new rally."""
        self._pose_p1_hand_speed.clear()
        self._pose_p1_com_height.clear()
        self._pose_p1_torso_x.clear()
        self._pose_p1_elbow.clear()
        self._pose_p1_vis.clear()

        self._pose_p2_hand_speed.clear()
        self._pose_p2_com_height.clear()
        self._pose_p2_torso_x.clear()
        self._pose_p2_elbow.clear()
        self._pose_p2_vis.clear()

        self._ball_speeds.clear()
        self._rally_start_frame = None
        self._rally_end_frame = None

    # -----------------------------------------------------------------------
    # WinPredictionModel interface
    # -----------------------------------------------------------------------

    def predict(self, packet: PredictionDataPacket) -> PredictionResult:
        """
        Called every frame (or every --predict-interval frames).
        Buffers pose data and returns the last cached prediction instantly.
        """
        self._ingest_packet_pose(packet)
        with self._lock:
            return self._cached_result

    def on_point_scored(self, packet: PredictionDataPacket) -> None:
        """
        Called when a stable score change is confirmed (i.e., a point has been won).
        Extracts features for the just-finished rally, appends to history,
        runs XGBoost inference, prints result to console, caches for predict().
        """
        # Extract features for the rally that just ended
        rally_feat = self._extract_rally_features(packet)
        self._rally_history.append(rally_feat)

        # Reset pose buffers for the next rally
        self._reset_rally_buffers()

        # Build feature vector
        X = self._build_feature_vector(packet)
        if X is None or self._model is None:
            return

        try:
            X_filled = np.nan_to_num(X, nan=0.0)
            proba = self._model.predict_proba(X_filled)[0]
            p1_prob = float(proba[1])
            p2_prob = float(proba[0])
            # Confidence: distance from 50/50
            confidence = float(abs(p1_prob - 0.5) * 2.0)
        except Exception as exc:
            print(f"[XGBoostWinPredictor] Inference error: {exc}")
            return

        rally_num = len(self._rally_history)
        p1_score = packet.score.player1_score or 0
        p2_score = packet.score.player2_score or 0
        p1_name = self._player_names[0]
        p2_name = self._player_names[1]

        print(
            f"[Rally {rally_num:>3}] "
            f"{p1_name}: {p1_prob * 100:.1f}% | {p2_name}: {p2_prob * 100:.1f}% | "
            f"Score: {p1_score}-{p2_score} | "
            f"conf: {confidence:.2f}"
        )

        new_result = PredictionResult(
            player1_win_prob=p1_prob,
            player2_win_prob=p2_prob,
            confidence=confidence,
            model_ready=True,
        )
        with self._lock:
            self._cached_result = new_result

    def reset(self) -> None:
        """Reset all state for a new match / session."""
        self._rally_history.clear()
        self._reset_rally_buffers()
        with self._lock:
            self._cached_result = PredictionResult(
                player1_win_prob=0.5,
                player2_win_prob=0.5,
                confidence=0.0,
                model_ready=False,
            )
        print("[XGBoostWinPredictor] Reset — ready for new match.")
