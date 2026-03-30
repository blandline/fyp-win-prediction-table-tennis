"""
Late Fusion Model Training
==========================
Trains a meta-model that combines:
  - cv_prob:      P(Player1 wins) from the CV/pose XGBoost model (per-match summary)
  - profile_prob: P(Player1 wins) from the player-profile XGBoost model (career stats)

The meta-model is a logistic regression trained on one row per match.
It learns which source to trust more, and outputs the fused probability.

Outputs (saved next to this script):
  late_fusion_model.pkl   — trained LogisticRegression meta-model
  fusion_weights.json     — simple weighted-average fallback weights + metadata

Usage:
  1. Make sure Player_Profile/train_model.py has been run first (produces
     tt_model_2026.pkl, elo_dict_2026.pkl, feature_columns.pkl, processed_df.pkl).
  2. Run ML_Dataset_Prep.py to regenerate final_dataset.csv with player name columns.
  3. Run this script from the 'CV Pipeline' directory:
       cd "CV Pipeline"
       python fusion_model_training.py
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent                          # CV Pipeline/
_REPO = _HERE.parent                                   # repo root
_PROFILE_DIR = _REPO / "Player_Profile"

CV_DATASET     = _HERE / "final_dataset.csv"
CV_MODEL       = _HERE / "xgb_model.pkl"

PROFILE_MODEL     = _PROFILE_DIR / "tt_model_2026.pkl"
PROFILE_ELO       = _PROFILE_DIR / "elo_dict_2026.pkl"
PROFILE_FEATURES  = _PROFILE_DIR / "feature_columns.pkl"
PROFILE_DATA      = _PROFILE_DIR / "processed_df.pkl"
PROFILE_CSV       = _PROFILE_DIR / "ittf_h2h_data_2026.csv"

OUTPUT_MODEL   = _HERE / "late_fusion_model.pkl"
OUTPUT_WEIGHTS = _HERE / "fusion_weights.json"


# ---------------------------------------------------------------------------
# Load profile CSV (pandas-version-safe alternative to processed_df.pkl)
# ---------------------------------------------------------------------------

def _load_profile_df_from_csv() -> pd.DataFrame:
    """
    Rebuild the processed profile DataFrame from the raw CSV.
    Replicates the essential cleaning done in Player_Profile/train_model.py
    so that column names and types match what the profile model expects.
    """
    if not PROFILE_CSV.exists():
        raise FileNotFoundError(
            f"Neither processed_df.pkl nor {PROFILE_CSV} found.\n"
            "Run Player_Profile/train_model.py first."
        )
    df = pd.read_csv(PROFILE_CSV)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace('  ', ' ')
    df = df.rename(columns={
        'Win_B':    'Wins_B',
        'Loses_A':  'Losses_A',
        'Loses_B':  'Losses_B',
        'Head-To-Head': 'Head_To_Head',
        'Best of 3/5':  'Best_of',
    })
    df['ID_A'] = df['ID_A'].astype(str).str.strip()
    df['ID_B'] = df['ID_B'].astype(str).str.strip()

    # Recreate target so H2H win counts work in _build_profile_row
    def _get_target(winner_str, player_a, player_b):
        winner_str = str(winner_str).strip().lower()
        player_a   = str(player_a).strip().lower()
        player_b   = str(player_b).strip().lower()
        if player_a in winner_str:
            return 0
        if player_b in winner_str:
            return 1
        return np.nan

    df['target'] = df.apply(
        lambda r: _get_target(r['Winner'], r['Player A'], r['Player B']), axis=1
    )
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)
    print(f"  Profile CSV loaded: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

def load_artifacts():
    print("Loading CV model and dataset...")
    if not CV_DATASET.exists():
        raise FileNotFoundError(
            f"CV dataset not found: {CV_DATASET}\n"
            "Run ML_Dataset_Prep.py first to generate final_dataset.csv"
        )
    cv_df = pd.read_csv(CV_DATASET)

    if "player1_name" not in cv_df.columns or "player2_name" not in cv_df.columns:
        raise ValueError(
            "final_dataset.csv is missing player1_name / player2_name columns.\n"
            "Re-run ML_Dataset_Prep.py to regenerate it with name columns."
        )

    if not CV_MODEL.exists():
        raise FileNotFoundError(f"CV model not found: {CV_MODEL}")
    cv_model = joblib.load(CV_MODEL)
    print(f"  CV model: {CV_MODEL.name}")

    print("\nLoading player-profile model...")
    for path in [PROFILE_MODEL, PROFILE_ELO, PROFILE_FEATURES]:
        if not path.exists():
            raise FileNotFoundError(
                f"Profile artifact not found: {path}\n"
                "Run Player_Profile/train_model.py first."
            )
    profile_model = joblib.load(PROFILE_MODEL)
    elo_dict      = joblib.load(PROFILE_ELO)
    feature_cols  = joblib.load(PROFILE_FEATURES)

    # Load the processed DataFrame from CSV to avoid pandas pickle version issues.
    # processed_df.pkl may fail to unpickle across pandas versions (StringDtype error).
    if PROFILE_DATA.exists():
        try:
            profile_df = joblib.load(PROFILE_DATA)
            print(f"  Profile data:  loaded from {PROFILE_DATA.name} "
                  f"({len(profile_df)} rows)")
        except Exception as e:
            print(f"  processed_df.pkl could not be loaded ({e})\n"
                  f"  Falling back to CSV: {PROFILE_CSV.name}")
            profile_df = _load_profile_df_from_csv()
    else:
        print(f"  processed_df.pkl not found — loading from CSV: {PROFILE_CSV.name}")
        profile_df = _load_profile_df_from_csv()

    print(f"  Profile model: {PROFILE_MODEL.name}")

    return cv_df, cv_model, profile_model, elo_dict, feature_cols, profile_df


# ---------------------------------------------------------------------------
# Profile model inference (adapted from Player_Profile/prediction.py)
# ---------------------------------------------------------------------------

DEFAULT_ELO = 1500

def _get_player_stats(player_name: str, profile_df: pd.DataFrame, elo_dict: dict) -> dict:
    mask_a = profile_df["Player A"].str.contains(player_name, case=False, na=False)
    mask_b = profile_df["Player B"].str.contains(player_name, case=False, na=False)
    matches = profile_df[mask_a | mask_b]
    if matches.empty:
        raise ValueError(f"Player not found in profile data: '{player_name}'")
    latest = matches.iloc[-1]
    is_a = latest["Player A"].lower().find(player_name.lower()) != -1
    prefix = "A" if is_a else "B"
    return {
        "name":    player_name.strip(),
        "id":      str(latest[f"ID_{prefix}"]),
        "gender":  latest.get(f"Gender_{prefix}", "M"),
        "age":     latest.get(f"Age_{prefix}", np.nan),
        "events":  latest.get(f"Events_{prefix}", 0),
        "matches": latest.get(f"Matches_{prefix}", 0),
        "wins":    latest.get(f"Wins_{prefix}", 0),
        "titles":  latest.get(f"Titles_{prefix}", 0),
        "elo":     elo_dict.get(str(latest[f"ID_{prefix}"]), DEFAULT_ELO),
    }


def _build_profile_row(p1: dict, p2: dict, profile_df: pd.DataFrame,
                        feature_cols: list, best_of: int = 5) -> pd.DataFrame:
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
        if not pd.isna(p1["age"]) and not pd.isna(p2["age"])
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
    return input_df[feature_cols]


def get_profile_prob(player1_name: str, player2_name: str,
                     profile_model, elo_dict: dict,
                     feature_cols: list, profile_df: pd.DataFrame,
                     best_of: int = 5) -> float:
    """Return P(player1 wins) from the profile model. Raises ValueError if not found."""
    p1 = _get_player_stats(player1_name, profile_df, elo_dict)
    p2 = _get_player_stats(player2_name, profile_df, elo_dict)
    row = _build_profile_row(p1, p2, profile_df, feature_cols, best_of)
    prob = profile_model.predict_proba(row)[0]
    # class 0 = player A (p1) wins in training convention
    return float(prob[0])


# ---------------------------------------------------------------------------
# CV model probability for a match (last rally row, cumulative features)
# ---------------------------------------------------------------------------

_CV_FEATURE_COLS = [
    "hand_speed_diff", "com_std_diff", "movement_diff", "elbow_diff",
    "visibility_avg", "ball_speed_mean", "ball_speed_max", "rally_duration_frames",
    "hand_speed_diff_last3", "com_std_diff_last3", "movement_diff_last3",
    "elbow_diff_last3", "visibility_avg_last3", "ball_speed_mean_last3",
    "ball_speed_max_last3", "rally_duration_frames_last3",
    "score_diff", "total_points", "rally_number",
]

def get_cv_prob(match_rows: pd.DataFrame, cv_model) -> float:
    """Get P(Player1 wins) from CV model using the last rally row of the match."""
    last_row = match_rows.iloc[[-1]][_CV_FEATURE_COLS].fillna(0)
    prob = cv_model.predict_proba(last_row)[0]
    # CV model: label=1 means Player1 wins → prob[1]
    return float(prob[1])


# ---------------------------------------------------------------------------
# Build fusion training dataset
# ---------------------------------------------------------------------------

def build_fusion_dataset(cv_df, cv_model, profile_model, elo_dict,
                          feature_cols, profile_df):
    print("\nBuilding per-match fusion rows...")
    rows = []
    skipped = []

    match_ids = sorted(cv_df["match_id"].unique())
    for mid in match_ids:
        match_rows = cv_df[cv_df["match_id"] == mid]
        p1_name = match_rows["player1_name"].iloc[0]
        p2_name = match_rows["player2_name"].iloc[0]
        label   = int(match_rows["label"].iloc[0])

        cv_prob = get_cv_prob(match_rows, cv_model)

        try:
            profile_prob = get_profile_prob(
                p1_name, p2_name,
                profile_model, elo_dict, feature_cols, profile_df,
            )
            profile_found = True
        except ValueError as e:
            print(f"  [Match {mid}] Profile lookup failed ({e}) — will use CV-only fallback")
            profile_prob = None
            profile_found = False

        row = {
            "match_id":     mid,
            "player1_name": p1_name,
            "player2_name": p2_name,
            "cv_prob":      cv_prob,
            "profile_prob": profile_prob,
            "label":        label,
            "profile_found": profile_found,
        }
        rows.append(row)
        status = f"profile={profile_prob:.3f}" if profile_found else "profile=N/A"
        print(f"  Match {mid}: '{p1_name}' vs '{p2_name}' | "
              f"cv={cv_prob:.3f} | {status} | label={label}")

    df = pd.DataFrame(rows)
    df_with_profile = df[df["profile_found"]].copy()

    print(f"\nMatches with profile data: {len(df_with_profile)} / {len(df)}")
    if len(df_with_profile) < 2:
        raise ValueError(
            "Need at least 2 matches with profile data to train the fusion model.\n"
            "Check that player names in broadcast_session.json can be found in "
            "Player_Profile/ittf_h2h_data_2026.csv via substring search."
        )

    return df, df_with_profile


# ---------------------------------------------------------------------------
# Train fusion meta-model
# ---------------------------------------------------------------------------

def train_fusion_model(fusion_df: pd.DataFrame):
    """
    Train a logistic regression on [cv_prob, profile_prob] → label.
    Uses Leave-One-Out CV since N is small.
    """
    X = fusion_df[["cv_prob", "profile_prob"]].values
    y = fusion_df["label"].values

    print("\nTraining late fusion meta-model (Logistic Regression)...")
    print(f"  Training samples: {len(X)}")
    print(f"  Features: cv_prob, profile_prob")

    meta_model = LogisticRegression(max_iter=1000, random_state=42)

    # ---- Leave-One-Out cross-validation ----
    if len(X) >= 3:
        loo = LeaveOneOut()
        loo_preds = []
        loo_probs = []
        loo_true  = []

        for train_idx, test_idx in loo.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if len(np.unique(y_tr)) < 2:
                continue

            fold_model = LogisticRegression(max_iter=1000, random_state=42)
            fold_model.fit(X_tr, y_tr)
            loo_preds.append(fold_model.predict(X_te)[0])
            loo_probs.append(fold_model.predict_proba(X_te)[0][1])
            loo_true.append(y_te[0])

        if len(loo_true) >= 2:
            loo_acc = accuracy_score(loo_true, loo_preds)
            loo_auc = roc_auc_score(loo_true, loo_probs) if len(np.unique(loo_true)) > 1 else float("nan")
            print(f"\n  LOO-CV Accuracy : {loo_acc:.3f}")
            print(f"  LOO-CV ROC-AUC  : {loo_auc:.3f}" if not np.isnan(loo_auc)
                  else "  LOO-CV ROC-AUC  : N/A (only one class in test set)")
        else:
            print("  Not enough samples for full LOO-CV evaluation.")
    else:
        print("  Fewer than 3 samples — skipping LOO-CV.")

    # ---- Train on full dataset ----
    meta_model.fit(X, y)
    train_preds = meta_model.predict(X)
    train_probs = meta_model.predict_proba(X)[:, 1]
    train_acc   = accuracy_score(y, train_preds)
    print(f"\n  Train Accuracy  : {train_acc:.3f}")
    print(f"  Model coefs     : cv_prob={meta_model.coef_[0][0]:.4f}, "
          f"profile_prob={meta_model.coef_[0][1]:.4f}")
    print(f"  Intercept       : {meta_model.intercept_[0]:.4f}")

    return meta_model


# ---------------------------------------------------------------------------
# Compute simple weighted-average fallback weights
# ---------------------------------------------------------------------------

def compute_weighted_average_weights(fusion_df: pd.DataFrame) -> dict:
    """
    Determine naive weights by comparing individual model accuracy.
    Used as a simple fallback when only one source is available.
    """
    y = fusion_df["label"].values
    cv_acc = accuracy_score(y, (fusion_df["cv_prob"].values >= 0.5).astype(int))
    pr_acc = accuracy_score(y, (fusion_df["profile_prob"].values >= 0.5).astype(int))

    total = cv_acc + pr_acc if (cv_acc + pr_acc) > 0 else 1.0
    w_cv      = round(cv_acc / total, 4)
    w_profile = round(pr_acc / total, 4)

    print(f"\n  Simple weighted-average fallback:")
    print(f"    CV model accuracy      : {cv_acc:.3f}  → weight {w_cv}")
    print(f"    Profile model accuracy : {pr_acc:.3f}  → weight {w_profile}")

    return {"w_cv": w_cv, "w_profile": w_profile, "cv_accuracy": cv_acc, "pr_accuracy": pr_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LATE FUSION MODEL TRAINING")
    print("=" * 60)

    cv_df, cv_model, profile_model, elo_dict, feature_cols, profile_df = load_artifacts()

    all_df, fusion_df = build_fusion_dataset(
        cv_df, cv_model, profile_model, elo_dict, feature_cols, profile_df
    )

    meta_model = train_fusion_model(fusion_df)
    weights    = compute_weighted_average_weights(fusion_df)

    # ---- Save ----
    joblib.dump(meta_model, OUTPUT_MODEL)
    print(f"\nSaved meta-model → {OUTPUT_MODEL}")

    weights_data = {
        **weights,
        "n_matches_trained": int(len(fusion_df)),
        "n_matches_total":   int(len(all_df)),
        "matches": [
            {
                "match_id":     int(row["match_id"]),
                "player1_name": row["player1_name"],
                "player2_name": row["player2_name"],
                "cv_prob":      round(float(row["cv_prob"]), 4),
                "profile_prob": round(float(row["profile_prob"]), 4) if row["profile_found"] else None,
                "label":        int(row["label"]),
                "profile_found": bool(row["profile_found"]),
            }
            for _, row in all_df.iterrows()
        ],
    }
    with open(OUTPUT_WEIGHTS, "w") as f:
        json.dump(weights_data, f, indent=2)
    print(f"Saved fusion weights → {OUTPUT_WEIGHTS}")

    print("\n" + "=" * 60)
    print("FUSION MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Meta-model trained on {len(fusion_df)} matches.")
    print("To retrain: collect more matches, re-run ML_Dataset_Prep.py,")
    print("then re-run this script.")


if __name__ == "__main__":
    main()
