import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ────────────────────────────────────────────────
# 1. Load data
# ────────────────────────────────────────────────

df = pd.read_csv('ittf_h2h_data_2026.csv')

# Clean & normalize column names
df.columns = df.columns.str.strip().str.replace('"', '').str.replace('  ', ' ')
df = df.rename(columns={
    'Win_B': 'Wins_B',
    'Loses_A': 'Losses_A',
    'Loses_B': 'Losses_B',
    'Head-To-Head': 'Head_To_Head',
    'Best of 3/5': 'Best_of',
})

print("Columns after loading and renaming:")
print(df.columns.tolist())

# Force IDs to string
df['ID_A'] = df['ID_A'].astype(str).str.strip()
df['ID_B'] = df['ID_B'].astype(str).str.strip()

# ────────────────────────────────────────────────
# 2. Target
# ────────────────────────────────────────────────

def get_target(winner_str, player_a, player_b):
    winner_str = str(winner_str).strip().lower()
    player_a = str(player_a).strip().lower()
    player_b = str(player_b).strip().lower()
    
    if player_a in winner_str:
        return 0
    if player_b in winner_str:
        return 1
    
    # Fallback
    if 'a' in winner_str or any(n in winner_str for n in ['calderano', 'groth', 'lebrun']):
        return 0
    if 'b' in winner_str or any(n in winner_str for n in ['moregard', 'karlsson', 'qiu']):
        return 1
    
    return np.nan

df['target'] = df.apply(lambda r: get_target(r['Winner'], r['Player A'], r['Player B']), axis=1)
df = df.dropna(subset=['target']).copy()
df['target'] = df['target'].astype(int)

print(f"Rows after dropping invalid winners: {len(df)}")

# ────────────────────────────────────────────────
# 3. Parse Score
# ────────────────────────────────────────────────

def parse_score(score_str):
    if pd.isna(score_str):
        return 0, 0
    score_str = str(score_str).strip().strip('"')
    if '-' not in score_str:
        return 0, 0
    try:
        a_str, b_str = score_str.split('-', 1)
        a = int(a_str.strip())
        b = int(b_str.strip())
        return a - b, abs(a - b)
    except:
        return 0, 0

df['margin_signed'], df['margin_abs'] = zip(*df['Score'].apply(parse_score))

# ────────────────────────────────────────────────
# 4. Best of
# ────────────────────────────────────────────────

def parse_best_of(val):
    if pd.isna(val):
        return 5
    val = str(val).lower().strip()
    if '3' in val:
        return 3
    if '5' in val or 'best of 5' in val:
        return 5
    return 5

df['best_of'] = df['Best_of'].apply(parse_best_of)

# ────────────────────────────────────────────────
# 5. Elo
# ────────────────────────────────────────────────

elo_dict = {}
DEFAULT_ELO = 1500
K_FACTOR = 32

def elo_update(winner_elo, loser_elo):
    expected = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    new_w = winner_elo + K_FACTOR * (1 - expected)
    new_l = loser_elo + K_FACTOR * (0 - (1 - expected))
    return new_w, new_l

df['elo_a_before'] = 0.0
df['elo_b_before'] = 0.0

for i in df.index:
    row = df.loc[i]
    id_a = row['ID_A']
    id_b = row['ID_B']
    
    elo_a = elo_dict.get(id_a, DEFAULT_ELO)
    elo_b = elo_dict.get(id_b, DEFAULT_ELO)
    
    df.at[i, 'elo_a_before'] = elo_a
    df.at[i, 'elo_b_before'] = elo_b
    
    if row['target'] == 0:
        new_a, new_b = elo_update(elo_a, elo_b)
    else:
        new_a, new_b = elo_update(elo_b, elo_a)
    
    elo_dict[id_a] = new_a
    elo_dict[id_b] = new_b

# ────────────────────────────────────────────────
# 6. Features
# ────────────────────────────────────────────────

df['elo_diff'] = df['elo_a_before'] - df['elo_b_before']
df['age_diff'] = df['Age_A'] - df['Age_B']
df['events_diff'] = df['Events_A'] - df['Events_B']
df['matches_diff'] = df['Matches_A'] - df['Matches_B']
df['win_rate_diff'] = (
    df['Wins_A'] / (df['Matches_A'] + 1e-6)
    - df['Wins_B'] / (df['Matches_B'] + 1e-6)
)
df['titles_diff'] = df['Titles_A'] - df['Titles_B']

def parse_h2h(val):
    if pd.isna(val):
        return 0.5
    val = str(val).strip().strip('"')
    if ':' in val:
        try:
            wa, wb = map(int, val.split(':'))
            return wa / (wa + wb + 1e-6)
        except:
            pass
    try:
        return float(val)
    except:
        return 0.5

df['h2h_a'] = df['Head_To_Head'].apply(parse_h2h)

df['gender_match'] = df['Gender_A'] + '_vs_' + df['Gender_B']

df['cum_wr_a'] = df.groupby('ID_A')['target'].transform(
    lambda x: (1 - x).cumsum().shift(1).fillna(0) / np.maximum(df.groupby('ID_A').cumcount().shift(1), 1)
)
df['cum_wr_b'] = df.groupby('ID_B')['target'].transform(
    lambda x: x.cumsum().shift(1).fillna(0) / np.maximum(df.groupby('ID_B').cumcount().shift(1), 1)
)
df['recent_diff'] = df['cum_wr_a'] - df['cum_wr_b']

# ────────────────────────────────────────────────
# 7. Feature list + encoding
# ────────────────────────────────────────────────

feature_cols = [
    'elo_diff', 'age_diff', 'events_diff', 'matches_diff',
    'win_rate_diff', 'titles_diff', 'h2h_a', 'margin_abs',
    'recent_diff', 'best_of'
]

df = pd.get_dummies(df, columns=['gender_match'], drop_first=True, prefix='gender')
gender_dummies = [c for c in df.columns if c.startswith('gender_')]
feature_cols += gender_dummies

X = df[feature_cols].fillna(0)
y = df['target']

# ────────────────────────────────────────────────
# 8. Train / Test split
# ────────────────────────────────────────────────

train_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]

# ────────────────────────────────────────────────
# 9. Train both models
# ────────────────────────────────────────────────

xgb_model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.025,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.75,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# ────────────────────────────────────────────────
# 10. Evaluate both
# ────────────────────────────────────────────────

def get_metrics(model, X_test, y_test, name):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'Model': name,
        'Accuracy':  accuracy_score(y_test, pred),
        'F1':        f1_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall':    recall_score(y_test, pred),
        'ROC-AUC':   roc_auc_score(y_test, prob)
    }, confusion_matrix(y_test, pred)

xgb_metrics, xgb_cm = get_metrics(xgb_model, X_test, y_test, 'XGBoost')
lr_metrics,  lr_cm  = get_metrics(lr_model,  X_test, y_test, 'Logistic Regression')

# ────────────────────────────────────────────────
# 11. Comparison table
# ────────────────────────────────────────────────

comparison = pd.DataFrame([xgb_metrics, lr_metrics])
print("\nModel Comparison:")
print(comparison.round(4))

comparison.to_csv('model_comparison_metrics.csv', index=False)

# ────────────────────────────────────────────────
# 12. Confusion matrices
# ────────────────────────────────────────────────

def save_cm(cm, title, fname):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

save_cm(xgb_cm, 'XGBoost Confusion Matrix', 'xgboost_confusion_matrix.png')
save_cm(lr_cm,  'Logistic Regression Confusion Matrix', 'logistic_confusion_matrix.png')

# Side-by-side
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
axes[0].set_title('XGBoost')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1])
axes[1].set_title('Logistic Regression')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.suptitle('Confusion Matrices Comparison')
plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved:")
print("• model_comparison_metrics.csv")
print("• xgboost_confusion_matrix.png")
print("• logistic_confusion_matrix.png")
print("• confusion_matrices_comparison.png")

# ────────────────────────────────────────────────
# Save models & artifacts
# ────────────────────────────────────────────────

joblib.dump(xgb_model,   'tt_model_2026.pkl')
joblib.dump(lr_model,    'logistic_baseline_2026.pkl')
joblib.dump(elo_dict,    'elo_dict_2026.pkl')
joblib.dump(feature_cols, 'feature_columns.pkl')
joblib.dump(df,          'processed_df.pkl')

print("\nTraining & evaluation completed.")