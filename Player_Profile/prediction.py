# predict_model.py
import pandas as pd
import numpy as np
import joblib

# ────────────────────────────────────────────────
# Load saved artifacts
# ────────────────────────────────────────────────

MODEL_PATH    = 'tt_model_2026.pkl'
ELO_PATH      = 'elo_dict_2026.pkl'
FEATURES_PATH = 'feature_columns.pkl'
DATA_PATH     = 'processed_df.pkl'

model        = joblib.load(MODEL_PATH)
elo_dict     = joblib.load(ELO_PATH)
feature_cols = joblib.load(FEATURES_PATH)

# Load processed data for lookups + H2H
try:
    df = joblib.load(DATA_PATH)
    print("Loaded processed pickle data for lookups")
except:
    print("Processed pickle not found → loading fallback...")
    df = pd.read_excel('ittf_h2h_data_2026.csv')  # adjust if needed

DEFAULT_ELO = 1500

# ────────────────────────────────────────────────
# Get latest stats for a player
# ────────────────────────────────────────────────

def get_latest_player_stats(player_name):
    if df is None:
        raise ValueError("Data not loaded")

    mask_a = df['Player A'].str.contains(player_name, case=False, na=False)
    mask_b = df['Player B'].str.contains(player_name, case=False, na=False)

    matches = df[mask_a | mask_b]
    if matches.empty:
        raise ValueError(f"Player not found: {player_name}")

    latest = matches.iloc[-1]

    is_a = latest['Player A'].lower().find(player_name.lower()) != -1
    prefix = 'A' if is_a else 'B'

    return {
        'name':   player_name.strip(),
        'id':     str(latest[f'ID_{prefix}']),
        'gender': latest.get(f'Gender_{prefix}', 'M'),
        'age':    latest.get(f'Age_{prefix}', np.nan),
        'events': latest.get(f'Events_{prefix}', 0),
        'matches':latest.get(f'Matches_{prefix}', 0),
        'wins':   latest.get(f'Wins_{prefix}', latest.get('Win_B', 0) if prefix == 'B' else latest.get('Wins_A', 0)),
        'titles': latest.get(f'Titles_{prefix}', 0),
        'elo':    elo_dict.get(str(latest[f'ID_{prefix}']), DEFAULT_ELO)
    }


# ────────────────────────────────────────────────
# Create prediction input + gather H2H info
# ────────────────────────────────────────────────

def create_prediction_row_and_h2h(p1, p2, best_of=5):
    # Find all matches between these two players
    pair = df[
        ((df['ID_A'] == p1['id']) & (df['ID_B'] == p2['id'])) |
        ((df['ID_A'] == p2['id']) & (df['ID_B'] == p1['id']))
    ].copy()

    h2h_total = len(pair)
    h2h_rate_p1 = 0.5
    most_recent_winner = "No previous match"
    p1_wins = 0
    p2_wins = 0

    if h2h_total > 0:
        # Determine which side is p1
        first_row = pair.iloc[0]
        p1_is_a_in_data = first_row['ID_A'] == p1['id']

        # Count wins
        if p1_is_a_in_data:
            p1_wins = len(pair[pair['target'] == 0])
            p2_wins = len(pair[pair['target'] == 1])
        else:
            p1_wins = len(pair[pair['target'] == 1])
            p2_wins = len(pair[pair['target'] == 0])

        h2h_rate_p1 = p1_wins / h2h_total if h2h_total > 0 else 0.5

        # Most recent match winner (last row = most recent)
        last_row = pair.iloc[-1]
        last_winner = last_row['Player A'] if last_row['target'] == 0 else last_row['Player B']
        most_recent_winner = f"Most recent: {last_winner} won"

    gender_match = f"{p1['gender']}_vs_{p2['gender']}"

    row_dict = {
        'elo_diff':       p1['elo'] - p2['elo'],
        'age_diff':       p1['age'] - p2['age'] if not pd.isna(p1['age']) and not pd.isna(p2['age']) else 0,
        'events_diff':    p1['events'] - p2['events'],
        'matches_diff':   p1['matches'] - p2['matches'],
        'win_rate_diff':  (p1['wins'] / (p1['matches'] + 1e-6)) - (p2['wins'] / (p2['matches'] + 1e-6)),
        'titles_diff':    p1['titles'] - p2['titles'],
        'h2h_a':          h2h_rate_p1,
        'margin_abs':     0.0,
        'recent_diff':    0.0,
        'best_of':        best_of,
    }

    # Manually create gender dummies
    possible_gender_dummies = [c for c in feature_cols if c.startswith('gender_')]
    for col in possible_gender_dummies:
        row_dict[col] = 1 if col == f'gender_{gender_match}' else 0

    input_df = pd.DataFrame([row_dict])

    # Align columns
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[feature_cols], {
        'total_matches': h2h_total,
        'p1_wins': p1_wins,
        'p2_wins': p2_wins,
        'most_recent': most_recent_winner
    }


# ────────────────────────────────────────────────
# Predict function – now includes H2H info
# ────────────────────────────────────────────────

def predict_match(player1_name, player2_name, best_of=5):
    try:
        p1 = get_latest_player_stats(player1_name)
        p2 = get_latest_player_stats(player2_name)
    except ValueError as e:
        return f"Error: {e}"

    input_df, h2h_info = create_prediction_row_and_h2h(p1, p2, best_of=best_of)

    prob = model.predict_proba(input_df)[0]
    prob_p1 = round(prob[0] * 100, 1)   # P(Player1 wins)
    prob_p2 = round(100 - prob_p1, 1)

    h2h_text = (
        f"\nHead-to-Head ({h2h_info['total_matches']} matches): "
        f"{player1_name} {h2h_info['p1_wins']} – {h2h_info['p2_wins']} {player2_name}\n"
        f"{h2h_info['most_recent']}"
    )

    return (
        f"{player1_name:<22} {prob_p1:5.1f}% chance to win\n"
        f"{player2_name:<22} {prob_p2:5.1f}% chance to win"
        f"{h2h_text}"
    )


# ────────────────────────────────────────────────
# Run examples
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nPredictions (with historical results):\n")
    
    print("Hugo Calderano vs Truls Moregard")
    print(predict_match("CALDERANO Hugo", "MOREGARD Truls"))
    print("-" * 60)

    print("Alexis Lebrun vs Dang Qiu")
    print(predict_match("LEBRUN Alexis", "QIU Dang"))
    print("-" * 60)

    # You can add more:
    # print(predict_match("GROTH Jonathan", "KARLSSON Kristian"))