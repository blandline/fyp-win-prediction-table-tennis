# -------------------- IMPORTS --------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
import joblib


# -------------------- LOAD DATA --------------------

df = pd.read_csv("final_dataset.csv")

print("Dataset shape:", df.shape)


# -------------------- PREP FEATURES --------------------

drop_cols = ["label", "match_id", "player1_name", "player2_name", "sets_to_win", "best_of"]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["label"]
groups = df["match_id"]

X = X.fillna(0)


# -------------------- FEATURE SCALING --------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------- MODELS --------------------

log_model = LogisticRegression(max_iter=1000)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)


# -------------------- STRATIFIED GROUP K-FOLD --------------------

sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)


# -------------------- TRAINING LOOP --------------------

log_acc, log_f1, log_auc = [], [], []
xgb_acc, xgb_f1, xgb_auc = [], [], []

fold = 1

for train_idx, test_idx in sgkf.split(X, y, groups):
    print(f"\n--- Fold {fold} ---")

    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train_scaled = X_scaled[train_idx]
    X_test_scaled = X_scaled[test_idx]

    # 🔴 SAFETY CHECK (prevents crash)
    if len(np.unique(y_train)) < 2:
        print("Skipping fold due to single class in training set")
        continue

    # -------------------- Logistic Regression --------------------

    log_model.fit(X_train_scaled, y_train)

    log_probs = log_model.predict_proba(X_test_scaled)[:, 1]
    log_preds = (log_probs > 0.5).astype(int)

    log_acc.append(accuracy_score(y_test, log_preds))
    log_f1.append(f1_score(y_test, log_preds))
    log_auc.append(roc_auc_score(y_test, log_probs))

    print(f"LogReg AUC: {log_auc[-1]:.4f}")

    # -------------------- XGBoost --------------------

    xgb_model.fit(X_train, y_train)

    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs > 0.5).astype(int)

    xgb_acc.append(accuracy_score(y_test, xgb_preds))
    xgb_f1.append(f1_score(y_test, xgb_preds))
    xgb_auc.append(roc_auc_score(y_test, xgb_probs))

    print(f"XGBoost AUC: {xgb_auc[-1]:.4f}")

    fold += 1


# -------------------- FINAL RESULTS --------------------

print("\n===== FINAL RESULTS =====")

print("\nLogistic Regression:")
print("Accuracy:", np.mean(log_acc))
print("F1:", np.mean(log_f1))
print("AUC:", np.mean(log_auc))

print("\nXGBoost:")
print("Accuracy:", np.mean(xgb_acc))
print("F1:", np.mean(xgb_f1))
print("AUC:", np.mean(xgb_auc))


# -------------------- TRAIN FINAL MODEL --------------------

xgb_model.fit(X, y)

joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved!")


# -------------------- EXAMPLE PREDICTION --------------------

sample = X.iloc[[0]]

prob = xgb_model.predict_proba(sample)[0]

print("\nExample Prediction:")
print(f"P(Player1 wins): {prob[1]:.3f}")
print(f"P(Player2 wins): {prob[0]:.3f}")