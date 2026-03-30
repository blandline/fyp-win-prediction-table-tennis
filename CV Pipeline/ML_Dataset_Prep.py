import os
import glob
import json
import pandas as pd
import numpy as np

# -------------------- LOAD FILES --------------------

def load_match_data(match_path):
    pose_file = glob.glob(os.path.join(match_path, "pose_frames*.csv"))[0]
    rallies_file = glob.glob(os.path.join(match_path, "rallies*.csv"))[0]
    scores_file = glob.glob(os.path.join(match_path, "scores*.csv"))[0]
    traj_file = glob.glob(os.path.join(match_path, "trajectories*.csv"))[0]

    pose = pd.read_csv(pose_file)
    rallies = pd.read_csv(rallies_file)
    scores = pd.read_csv(scores_file)
    traj = pd.read_csv(traj_file)

    return pose, rallies, scores, traj


# -------------------- CLEAN --------------------

def clean(df):
    return df.fillna(0)


# -------------------- FILTER BY FRAME --------------------

def get_rally_data(df, start_frame, end_frame):
    return df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]


# -------------------- SAFE FUNCTIONS --------------------

def safe_mean(series):
    return series.mean() if len(series) > 0 else 0

def safe_std(series):
    return series.std() if len(series) > 0 else 0


# -------------------- FEATURE EXTRACTION --------------------

def extract_rally_features(pose_r, traj_r):
    p1 = pose_r[pose_r["player_id"] == 1]
    p2 = pose_r[pose_r["player_id"] == 2]

    features = {}

    # Motion
    features["hand_speed_diff"] = safe_mean(p1["v_hand_speed"]) - safe_mean(p2["v_hand_speed"])

    # Stability
    features["com_std_diff"] = safe_std(p1["com_height"]) - safe_std(p2["com_height"])

    # Movement
    features["movement_diff"] = safe_mean(abs(p1["v_torso_x"])) - safe_mean(abs(p2["v_torso_x"]))

    # Technique
    features["elbow_diff"] = safe_mean(p1["angle_elbow_dom"]) - safe_mean(p2["angle_elbow_dom"])

    # Visibility
    features["visibility_avg"] = (
        safe_mean(p1["visibility_mean"]) + safe_mean(p2["visibility_mean"])
    ) / 2

    # Ball features (from trajectories)
    if "speed_mps" in traj_r.columns:
        features["ball_speed_mean"] = safe_mean(traj_r["speed_mps"])
        features["ball_speed_max"] = traj_r["speed_mps"].max() if len(traj_r) > 0 else 0
    else:
        features["ball_speed_mean"] = 0
        features["ball_speed_max"] = 0

    # Duration (using frame difference)
    features["rally_duration_frames"] = (
        traj_r["frame"].max() - traj_r["frame"].min()
    ) if len(traj_r) > 0 else 0

    return features


# -------------------- SCORE AT FRAME --------------------

def get_score_at_frame(scores, frame):
    s = scores[scores["frame"] <= frame]

    if len(s) == 0:
        return 0, 0

    last = s.iloc[-1]
    return last["player1_score"], last["player2_score"]


# -------------------- MATCH WINNER --------------------

def get_match_winner(scores):
    last = scores.iloc[-1]

    p1_sets = last["player1_sets"]
    p2_sets = last["player2_sets"]

    return 1 if p1_sets > p2_sets else 0


# -------------------- PLAYER NAMES FROM SESSION --------------------

def load_session_meta(match_path):
    """Read player names, sets_to_win and best_of from broadcast_session.json."""
    session_file = os.path.join(match_path, "broadcast_session.json")
    defaults = ("Player 1", "Player 2", 3, 5)
    if os.path.exists(session_file):
        try:
            with open(session_file, "r") as f:
                session = json.load(f)
            names = session.get("player_names", [])
            p1 = names[0] if len(names) >= 1 else "Player 1"
            p2 = names[1] if len(names) >= 2 else "Player 2"
            sets_to_win = int(session.get("sets_to_win", 3))
            best_of     = int(session.get("best_of", sets_to_win * 2 - 1))
            return p1, p2, sets_to_win, best_of
        except Exception:
            pass
    return defaults


# -------------------- PROCESS MATCH --------------------

def process_match(match_path, match_id):
    pose, rallies, scores, traj = load_match_data(match_path)

    pose = clean(pose)
    traj = clean(traj)

    winner = get_match_winner(scores)
    player1_name, player2_name, sets_to_win, best_of = load_session_meta(match_path)

    dataset = []
    history = []

    for i, rally in rallies.iterrows():
        start = rally["rally_start_frame"]
        end = rally["rally_end_frame"]

        pose_r = get_rally_data(pose, start, end)
        traj_r = get_rally_data(traj, start, end)

        rally_feat = extract_rally_features(pose_r, traj_r)
        history.append(rally_feat)

        df_hist = pd.DataFrame(history)

        # Aggregate ALL past rallies
        agg = df_hist.mean().to_dict()

        # Momentum (last 3 rallies)
        last3 = df_hist.tail(3).mean().to_dict()
        for k, v in last3.items():
            agg[f"{k}_last3"] = v

        # Score features
        p1_score, p2_score = get_score_at_frame(scores, end)

        agg["score_diff"] = p1_score - p2_score
        agg["total_points"] = p1_score + p2_score
        agg["rally_number"] = i + 1

        # Label
        agg["label"] = winner

        # Meta
        agg["match_id"] = match_id
        agg["player1_name"] = player1_name
        agg["player2_name"] = player2_name
        agg["sets_to_win"] = sets_to_win
        agg["best_of"] = best_of

        dataset.append(agg)

    return dataset


# -------------------- BUILD DATASET --------------------

def build_dataset(dataset_path):
    all_data = []

    match_folders = sorted(os.listdir(dataset_path))

    for idx, folder in enumerate(match_folders):
        match_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(match_path):
            continue

        print(f"Processing {folder}")

        match_data = process_match(match_path, idx)
        all_data.extend(match_data)

    return pd.DataFrame(all_data)


# -------------------- RUN --------------------

if __name__ == "__main__":
    dataset_path = "./Data"  # CHANGE THIS

    df = build_dataset(dataset_path)

    df.to_csv("final_dataset.csv", index=False)

    print("Dataset shape:", df.shape)
    print(df.head())