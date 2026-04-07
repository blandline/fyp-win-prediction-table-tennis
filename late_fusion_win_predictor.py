"""
Late Fusion Win Predictor
=========================
Combines two win probability sources in real-time:

  1. cv_prob      — from XGBoostWinPredictor (pose/ball/rally CV features,
                    updated after each confirmed point)
  2. profile_prob — from the player-profile XGBoost model (career stats /
                    Elo / H2H, computed once at startup — static per match)

The fused probability is produced by a logistic regression meta-model trained
in 'CV Pipeline/fusion_model_training.py'.

Fallback behaviour:
  - If profile lookup fails (player not in ITTF data), falls back to cv_prob only.
  - If the fusion model pkl is missing, falls back to a simple weighted average
    using weights from 'CV Pipeline/fusion_weights.json'.
  - If neither is available, falls back to 50/50.

Console output after each point:
  [Rally  5] CV: 63.0% | Profile: 71.2% | FUSED: 68.4% | P1 vs P2 | conf: 0.77

Usage in broadcast_data_collector.py via --prediction-model:
  --prediction-model late_fusion_win_predictor.py
  --ittf-name1 "CALDERANO Hugo"
  --ittf-name2 "MOREGARD Truls"
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from prediction_model_base import (
    WinPredictionModel,
    PredictionDataPacket,
    PredictionResult,
)
from xgb_win_predictor import XGBoostWinPredictor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_CV_PIPELINE = _HERE / "CV Pipeline"
_PROFILE_DIR = _HERE / "Player_Profile"

_FUSION_MODEL_PATH  = _CV_PIPELINE / "late_fusion_model.pkl"
_FUSION_WEIGHTS_PATH = _CV_PIPELINE / "fusion_weights.json"

_PROFILE_MODEL_PATH    = _PROFILE_DIR / "tt_model_2026.pkl"
_PROFILE_ELO_PATH      = _PROFILE_DIR / "elo_dict_2026.pkl"
_PROFILE_FEATURES_PATH = _PROFILE_DIR / "feature_columns.pkl"
_PROFILE_DATA_PATH     = _PROFILE_DIR / "processed_df.pkl"
_PROFILE_CSV_PATH      = _PROFILE_DIR / "ittf_h2h_data_2026.csv"

_DEFAULT_ELO = 1500
_SESSION_PATH = _HERE / "broadcast_data" / "broadcast_session.json"


# ---------------------------------------------------------------------------
# Match progress & dynamic fusion weight helpers
# ---------------------------------------------------------------------------

def _load_sets_to_win_from_session() -> int:
    """Read sets_to_win from broadcast_session.json, default 3."""
    if _SESSION_PATH.exists():
        try:
            with open(_SESSION_PATH) as f:
                data = json.load(f)
            return int(data.get("sets_to_win", 3))
        except Exception:
            pass
    return 3


def _match_progress(p1_sets: int, p2_sets: int, p1_score: int, p2_score: int,
                    sets_to_win: int) -> float:
    """
    Compute how far into the match we are on a 0→1 scale.

    The total "work" needed to finish a match is ``sets_to_win`` sets where
    each set is worth ~11 points.  We measure how many of those points have
    been played (counting completed sets as 11 each) relative to the maximum
    possible total (a match that goes the distance = ``2*sets_to_win - 1``
    sets each lasting 11 points).

    Returns a float in [0, 1] where 0 = match just started and
    1 = someone is on match-point in a deciding set.
    """
    points_per_set = 11.0
    max_sets = 2 * sets_to_win - 1
    max_points = max_sets * points_per_set

    completed_set_points = (p1_sets + p2_sets) * points_per_set
    current_set_points = float(p1_score + p2_score)
    total_played = completed_set_points + current_set_points

    progress = min(1.0, total_played / max_points) if max_points > 0 else 0.0

    # Boost progress when a player is one set away from winning
    leading_sets = max(p1_sets, p2_sets)
    if leading_sets == sets_to_win - 1:
        progress = max(progress, 0.6)
        leading_score = max(p1_score, p2_score)
        if leading_score >= 8:
            progress = max(progress, 0.85)

    return progress


def _dynamic_fusion_weights(progress: float,
                            base_w_cv: float = 0.5,
                            base_w_profile: float = 0.5) -> tuple[float, float]:
    """
    Shift fusion weights so that profile weight decreases and CV weight
    increases as the match progresses.

    At progress=0 the weights are (0.35, 0.65) — profile dominates.
    At progress=1 the weights are (0.85, 0.15) — CV dominates.
    The transition follows a smooth cubic curve.
    """
    t = max(0.0, min(1.0, progress))
    t_smooth = 3 * t * t - 2 * t * t * t  # smoothstep

    w_cv_min, w_cv_max = 0.35, 0.85
    w_cv = w_cv_min + (w_cv_max - w_cv_min) * t_smooth
    w_profile = 1.0 - w_cv
    return w_cv, w_profile


# ---------------------------------------------------------------------------
# Profile model helpers (self-contained, no import from Player_Profile/)
# ---------------------------------------------------------------------------

def _load_profile_df():
    """
    Load the processed profile DataFrame, falling back to the raw CSV when
    processed_df.pkl can't be unpickled (pandas StringDtype version mismatch).
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError("pandas not installed")

    if _PROFILE_DATA_PATH.exists():
        try:
            return joblib.load(_PROFILE_DATA_PATH)
        except Exception:
            pass  # fall through to CSV rebuild

    if not _PROFILE_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Neither {_PROFILE_DATA_PATH.name} nor {_PROFILE_CSV_PATH.name} found in "
            f"{_PROFILE_DIR}. Run Player_Profile/train_model.py first."
        )

    import pandas as pd
    df = pd.read_csv(_PROFILE_CSV_PATH)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace('  ', ' ')
    df = df.rename(columns={
        'Win_B': 'Wins_B', 'Loses_A': 'Losses_A', 'Loses_B': 'Losses_B',
        'Head-To-Head': 'Head_To_Head', 'Best of 3/5': 'Best_of',
    })
    df['ID_A'] = df['ID_A'].astype(str).str.strip()
    df['ID_B'] = df['ID_B'].astype(str).str.strip()

    def _target(winner_str, pa, pb):
        w = str(winner_str).strip().lower()
        if str(pa).strip().lower() in w:
            return 0
        if str(pb).strip().lower() in w:
            return 1
        return float('nan')

    df['target'] = df.apply(lambda r: _target(r['Winner'], r['Player A'], r['Player B']), axis=1)
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)
    return df


def _load_profile_artifacts():
    """Load all profile model artifacts. Returns (model, elo_dict, feature_cols, df) or raises."""
    if not _JOBLIB_AVAILABLE:
        raise ImportError("joblib not installed")
    if not _PANDAS_AVAILABLE:
        raise ImportError("pandas not installed")
    for p in [_PROFILE_MODEL_PATH, _PROFILE_ELO_PATH, _PROFILE_FEATURES_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Profile artifact not found: {p}\n"
                "Run Player_Profile/train_model.py first."
            )
    model        = joblib.load(_PROFILE_MODEL_PATH)
    elo_dict     = joblib.load(_PROFILE_ELO_PATH)
    feature_cols = joblib.load(_PROFILE_FEATURES_PATH)
    profile_df   = _load_profile_df()
    return model, elo_dict, feature_cols, profile_df


def _get_player_stats(player_name: str, profile_df, elo_dict: dict) -> dict:
    mask_a = profile_df["Player A"].str.contains(player_name, case=False, na=False)
    mask_b = profile_df["Player B"].str.contains(player_name, case=False, na=False)
    matches = profile_df[mask_a | mask_b]
    if matches.empty:
        raise ValueError(f"Player not found in ITTF profile data: '{player_name}'")
    latest = matches.iloc[-1]
    is_a = latest["Player A"].lower().find(player_name.lower()) != -1
    prefix = "A" if is_a else "B"
    return {
        "name":    player_name.strip(),
        "id":      str(latest[f"ID_{prefix}"]),
        "gender":  latest.get(f"Gender_{prefix}", "M"),
        "age":     latest.get(f"Age_{prefix}", float("nan")),
        "events":  latest.get(f"Events_{prefix}", 0),
        "matches": latest.get(f"Matches_{prefix}", 0),
        "wins":    latest.get(f"Wins_{prefix}", 0),
        "titles":  latest.get(f"Titles_{prefix}", 0),
        "elo":     elo_dict.get(str(latest[f"ID_{prefix}"]), _DEFAULT_ELO),
    }


def _compute_profile_prob(p1_name: str, p2_name: str,
                           profile_model, elo_dict: dict,
                           feature_cols: list, profile_df,
                           best_of: int = 5) -> float:
    """Return P(p1 wins) from the profile XGBoost. Raises ValueError if lookup fails."""
    import pandas as pd

    p1 = _get_player_stats(p1_name, profile_df, elo_dict)
    p2 = _get_player_stats(p2_name, profile_df, elo_dict)

    pair = profile_df[
        ((profile_df["ID_A"] == p1["id"]) & (profile_df["ID_B"] == p2["id"])) |
        ((profile_df["ID_A"] == p2["id"]) & (profile_df["ID_B"] == p1["id"]))
    ]
    h2h_total = len(pair)
    h2h_rate_p1 = 0.5
    if h2h_total > 0:
        first_row = pair.iloc[0]
        p1_is_a = first_row["ID_A"] == p1["id"]
        p1_wins = len(pair[pair["target"] == (0 if p1_is_a else 1)])
        h2h_rate_p1 = p1_wins / h2h_total

    gender_match = f"{p1['gender']}_vs_{p2['gender']}"
    age_diff = (
        p1["age"] - p2["age"]
        if not (isinstance(p1["age"], float) and np.isnan(p1["age"]))
        and not (isinstance(p2["age"], float) and np.isnan(p2["age"]))
        else 0
    )

    row_dict = {
        "elo_diff":      p1["elo"] - p2["elo"],
        "age_diff":      age_diff,
        "events_diff":   p1["events"] - p2["events"],
        "matches_diff":  p1["matches"] - p2["matches"],
        "win_rate_diff": (
            p1["wins"] / (p1["matches"] + 1e-6)
            - p2["wins"] / (p2["matches"] + 1e-6)
        ),
        "titles_diff":   p1["titles"] - p2["titles"],
        "h2h_a":         h2h_rate_p1,
        "margin_abs":    0.0,
        "recent_diff":   0.0,
        "best_of":       best_of,
    }

    for col in [c for c in feature_cols if c.startswith("gender_")]:
        row_dict[col] = 1 if col == f"gender_{gender_match}" else 0

    input_df = pd.DataFrame([row_dict])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    prob = profile_model.predict_proba(input_df[feature_cols])[0]
    return float(prob[0])  # class 0 = p1 wins in profile model convention


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class LateFusionWinPredictor(WinPredictionModel):
    """
    Real-time late fusion win predictor.

    Combines CV (pose/ball) probability updated per point with a static
    pre-match profile probability derived from career/Elo/H2H data.

    Parameters
    ----------
    ittf_name1 : str
        ITTF-format name of Player 1, e.g. "CALDERANO Hugo".
        Used to look up career stats. Can be a partial substring.
    ittf_name2 : str
        ITTF-format name of Player 2.
    player_names : list[str], optional
        Display names for console output (defaults to ittf names).
    best_of : int
        Tournament format (3 or 5). Used by profile model. Default 5.
    """

    def __init__(
        self,
        ittf_name1: str = "Player 1",
        ittf_name2: str = "Player 2",
        player_names: Optional[list] = None,
        best_of: int = 5,
        sets_to_win: Optional[int] = None,
    ):
        self._ittf_name1 = ittf_name1
        self._ittf_name2 = ittf_name2
        self._player_names = player_names or [ittf_name1, ittf_name2]
        self._best_of = best_of

        if sets_to_win is not None:
            self._sets_to_win = sets_to_win
        else:
            self._sets_to_win = _load_sets_to_win_from_session()
        print(f"[LateFusionWinPredictor] Sets to win: {self._sets_to_win}")

        # Internal CV predictor (handles all per-rally pose/ball buffering)
        self._cv_predictor = XGBoostWinPredictor(player_names=self._player_names)

        # Fusion meta-model
        self._fusion_model = None
        self._fusion_weights: dict = {"w_cv": 0.5, "w_profile": 0.5}
        self._load_fusion_model()

        # Profile model (loaded lazily, errors handled gracefully)
        self._profile_model = None
        self._profile_elo: dict = {}
        self._profile_feature_cols: list = []
        self._profile_df = None
        self._profile_prob: Optional[float] = None
        self._profile_available = False
        self._load_profile_and_lookup()

        # Cached fused result
        self._cached_result = PredictionResult(
            player1_win_prob=0.5,
            player2_win_prob=0.5,
            confidence=0.0,
            model_ready=False,
        )
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Initialization helpers
    # -----------------------------------------------------------------------

    def _load_fusion_model(self) -> None:
        if not _JOBLIB_AVAILABLE:
            print("[LateFusionWinPredictor] WARNING: joblib not available — fusion disabled")
            return

        if _FUSION_MODEL_PATH.exists():
            try:
                self._fusion_model = joblib.load(_FUSION_MODEL_PATH)
                print(f"[LateFusionWinPredictor] Loaded fusion model from {_FUSION_MODEL_PATH.name}")
            except Exception as exc:
                print(f"[LateFusionWinPredictor] ERROR loading fusion model: {exc}")

        if _FUSION_WEIGHTS_PATH.exists():
            try:
                with open(_FUSION_WEIGHTS_PATH) as f:
                    self._fusion_weights = json.load(f)
                print(
                    f"[LateFusionWinPredictor] Fusion weights: "
                    f"w_cv={self._fusion_weights.get('w_cv', 0.5):.3f}, "
                    f"w_profile={self._fusion_weights.get('w_profile', 0.5):.3f}"
                )
            except Exception as exc:
                print(f"[LateFusionWinPredictor] Could not load fusion weights: {exc}")

    def _load_profile_and_lookup(self) -> None:
        """Load profile artifacts and pre-compute the static profile_prob."""
        try:
            (self._profile_model, self._profile_elo,
             self._profile_feature_cols, self._profile_df) = _load_profile_artifacts()
            print("[LateFusionWinPredictor] Profile model loaded.")
        except Exception as exc:
            print(f"[LateFusionWinPredictor] Profile model unavailable: {exc}")
            return

        try:
            self._profile_prob = _compute_profile_prob(
                self._ittf_name1, self._ittf_name2,
                self._profile_model, self._profile_elo,
                self._profile_feature_cols, self._profile_df,
                best_of=self._best_of,
            )
            self._profile_available = True
            p1 = self._player_names[0]
            p2 = self._player_names[1]
            print(
                f"[LateFusionWinPredictor] Pre-match profile probability: "
                f"{p1} {self._profile_prob * 100:.1f}% | {p2} {(1 - self._profile_prob) * 100:.1f}%"
            )
        except ValueError as exc:
            print(
                f"[LateFusionWinPredictor] Profile lookup failed: {exc}\n"
                f"  Falling back to CV-only prediction."
            )
            self._profile_available = False

    # -----------------------------------------------------------------------
    # Fusion logic
    # -----------------------------------------------------------------------

    def _fuse(self, cv_prob: float,
              p1_sets: int = 0, p2_sets: int = 0,
              p1_score: int = 0, p2_score: int = 0) -> tuple[float, float]:
        """
        Fuse cv_prob and profile_prob into a final probability.

        Weights shift dynamically: early in the match the profile (career stats,
        Elo, H2H) dominates; as the match progresses the CV features (current
        performance, momentum, score) take over.

        Returns (fused_prob, confidence).
        """
        if not self._profile_available or self._profile_prob is None:
            confidence = float(abs(cv_prob - 0.5) * 2.0)
            return cv_prob, confidence

        profile_prob = self._profile_prob

        progress = _match_progress(
            p1_sets, p2_sets, p1_score, p2_score, self._sets_to_win
        )
        w_cv, w_profile = _dynamic_fusion_weights(progress)

        if self._fusion_model is not None:
            X = np.array([[cv_prob, profile_prob]])
            raw_fused = float(self._fusion_model.predict_proba(X)[0][1])
            # Blend the meta-model output with dynamic weighting: the meta-
            # model's own estimate is treated as a "fair" mix, then we shift
            # towards CV or profile based on match progress.
            fused = w_cv * cv_prob + w_profile * profile_prob
            # Average with the meta-model so its learned calibration still
            # helps when the match is young and we trust profiles more.
            meta_weight = 1.0 - progress  # trust meta-model less as game progresses
            fused = meta_weight * raw_fused + (1.0 - meta_weight) * fused
        else:
            total = w_cv + w_profile
            fused = (w_cv * cv_prob + w_profile * profile_prob) / total if total > 0 else 0.5

        confidence = float(abs(fused - 0.5) * 2.0)
        return fused, confidence

    # -----------------------------------------------------------------------
    # WinPredictionModel interface
    # -----------------------------------------------------------------------

    def predict(self, packet: PredictionDataPacket) -> PredictionResult:
        """
        Called every frame. Delegates pose buffering to the internal CV predictor,
        returns cached fused result instantly.
        """
        self._cv_predictor._ingest_packet_pose(packet)
        with self._lock:
            return self._cached_result

    def on_point_scored(self, packet: PredictionDataPacket) -> None:
        """
        Called on each confirmed point. Gets cv_prob from CV predictor,
        fuses with static profile_prob, prints to console, caches result.
        """
        # Let CV predictor extract its rally features and update its history
        self._cv_predictor.on_point_scored(packet)

        # Get the latest CV probability from the CV predictor's cache
        cv_result = self._cv_predictor._cached_result
        if not cv_result.model_ready:
            # Not enough rally history yet — keep warming up
            return

        cv_prob = cv_result.player1_win_prob

        # Extract match state for dynamic weighting
        p1_score = packet.score.player1_score or 0
        p2_score = packet.score.player2_score or 0
        p1_sets = packet.score.player1_sets
        p2_sets = packet.score.player2_sets

        # Fuse with match-progress-aware dynamic weights
        fused_prob, confidence = self._fuse(
            cv_prob, p1_sets, p2_sets, p1_score, p2_score
        )
        p2_prob = 1.0 - fused_prob

        p1_name = self._player_names[0]
        p2_name = self._player_names[1]
        rally_num = len(self._cv_predictor._rally_history)

        progress = _match_progress(
            p1_sets, p2_sets, p1_score, p2_score, self._sets_to_win
        )
        w_cv, w_profile = _dynamic_fusion_weights(progress)

        if self._profile_available and self._profile_prob is not None:
            profile_prob = self._profile_prob
            method = "fusion" if self._fusion_model is not None else "weighted-avg"
            print(
                f"[Rally {rally_num:>3}] "
                f"CV: {cv_prob * 100:.1f}% | "
                f"Profile: {profile_prob * 100:.1f}% | "
                f"FUSED ({method}): {fused_prob * 100:.1f}% | "
                f"{p1_name} vs {p2_name} | "
                f"Sets: {p1_sets}-{p2_sets} | Score: {p1_score}-{p2_score} | "
                f"progress: {progress:.0%} w_cv={w_cv:.2f} w_prof={w_profile:.2f} | "
                f"conf: {confidence:.2f}"
            )
        else:
            print(
                f"[Rally {rally_num:>3}] "
                f"CV (no profile): {cv_prob * 100:.1f}% | "
                f"{p1_name} vs {p2_name} | "
                f"Sets: {p1_sets}-{p2_sets} | Score: {p1_score}-{p2_score} | "
                f"conf: {confidence:.2f}"
            )

        new_result = PredictionResult(
            player1_win_prob=fused_prob,
            player2_win_prob=p2_prob,
            confidence=confidence,
            model_ready=True,
        )
        with self._lock:
            self._cached_result = new_result

    def reset(self) -> None:
        """Reset for a new match. Profile prob stays (career stats don't change)."""
        self._cv_predictor.reset()
        with self._lock:
            self._cached_result = PredictionResult(
                player1_win_prob=0.5,
                player2_win_prob=0.5,
                confidence=0.0,
                model_ready=False,
            )
        print("[LateFusionWinPredictor] Reset — ready for new match.")
        if self._profile_available:
            print(
                f"  Pre-match profile still: "
                f"{self._player_names[0]} {self._profile_prob * 100:.1f}% | "
                f"{self._player_names[1]} {(1 - self._profile_prob) * 100:.1f}%"
            )
