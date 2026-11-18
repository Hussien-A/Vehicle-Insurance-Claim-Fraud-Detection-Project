"""
Train the fraud detection model with an evaluation setup that fits an imbalanced problem.

Why this script changed:
  Only ~11.5% of claims are fraud. A model that always says "not fraud" already scores
  ~88.5% accuracy, so accuracy says almost nothing here. The old pipeline selected models
  by accuracy and reported ~88% accuracy while catching only ~6% of fraud cases.

What this script does:
  1. Splits the data into train / validation / test (60 / 20 / 20, stratified).
     The preprocessor is fit on the training split only, so no information leaks.
  2. Trains several models with class weighting for the minority (fraud) class.
  3. Selects the model by validation PR-AUC (average precision) on the fraud class.
  4. Tunes the decision threshold on validation to maximise fraud-class F1.
  5. Reports final metrics ONCE on the untouched test split, next to two baselines:
     "always predict not fraud" and the previous approach (unweighted GB, threshold 0.5).
  6. Saves the model, preprocessor, threshold and metrics for the API.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
BASE = Path(__file__).resolve().parent.parent
DATA_PATH = os.environ.get("DATA_PATH", str(BASE / "dataset" / "car_insurance_fraud_dataset.csv"))
if not os.path.exists(DATA_PATH):
    DATA_PATH = str(BASE / "car_insurance_fraud_dataset.csv")
assert os.path.exists(DATA_PATH), f"Dataset not found: {DATA_PATH}"
MODELS_DIR = BASE / "saved_models"
REPORTS_DIR = BASE / "reports"


# ---------------------------------------------------------------------------
# Feature engineering (must stay in sync with api/preprocess.py)
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["policy_id", "fraud_reported", "FraudFound_P"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    X = X.fillna(X.mode().iloc[0])

    if "incident_date" in X.columns:
        X["incident_date"] = pd.to_datetime(X["incident_date"], errors="coerce")
        X["incident_year"] = X["incident_date"].dt.year
        X["incident_month"] = X["incident_date"].dt.month
        X["incident_day_of_week"] = X["incident_date"].dt.dayofweek
        X = X.drop(columns=["incident_date"])
    X["claim_ratio"] = (X["claim_amount"] / X["total_claim_amount"].replace(0, np.nan)).fillna(1)
    X["premium_per_deductible"] = (
        X["policy_annual_premium"] / X["policy_deductible"].replace(0, np.nan)
    ).fillna(X["policy_annual_premium"])
    X["police_report_available"] = (X["police_report_available"].str.upper() == "YES").astype(int)
    X["is_total_loss"] = X["incident_severity"].str.contains("Total", case=False, na=False).astype(int)
    X = X.drop(columns=["incident_city"])  # ~18k unique values: noise, not signal
    X["age_group"] = pd.cut(X["insured_age"], bins=[0, 30, 50, 120], labels=[0, 1, 2]).astype(float)
    X["incident_weekend"] = (X["incident_day_of_week"] >= 5).astype(int)
    X["log_claim_amount"] = np.log1p(X["claim_amount"])
    X["log_total_claim_amount"] = np.log1p(X["total_claim_amount"])
    X["vehicles_plus_injuries"] = X["number_of_vehicles_involved"] + X["bodily_injuries"]
    X["high_claim_ratio"] = (X["claim_ratio"] > 1.1).astype(int)
    return X


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )


def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))  # last point has no threshold
    return float(thresholds[i])


def fraud_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "fraud_precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "fraud_recall": round(recall_score(y_true, pred), 4),
        "fraud_f1": round(f1_score(y_true, pred), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Data and splits
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
y = (df["fraud_reported"] == "Y").astype(int).to_numpy()
X = build_features(df)

X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=SEED
)
print(f"Rows: {len(X)} | fraud rate: {y.mean():.2%}")
print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

preprocessor = make_preprocessor(X_train)
Xtr = preprocessor.fit_transform(X_train)
Xva = preprocessor.transform(X_val)
Xte = preprocessor.transform(X_test)
feature_names = list(preprocessor.get_feature_names_out())

# ---------------------------------------------------------------------------
# Candidates (all weight the fraud class up)
# ---------------------------------------------------------------------------
balanced_w = compute_sample_weight("balanced", y_train)
candidates = {
    "Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5), None),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=400, min_samples_leaf=10, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED,
        ),
        None,
    ),
    "Hist Gradient Boosting": (
        HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", early_stopping=True, validation_fraction=0.15, random_state=SEED,
        ),
        None,
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED),
        balanced_w,
    ),
}

results = {}
fitted = {}
print("\nValidation results (fraud class):")
for name, (model, weights) in candidates.items():
    if weights is not None:
        model.fit(Xtr, y_train, sample_weight=weights)
    else:
        model.fit(Xtr, y_train)
    p_val = model.predict_proba(Xva)[:, 1]
    thr = best_f1_threshold(y_val, p_val)
    results[name] = fraud_metrics(y_val, p_val, thr)
    fitted[name] = model
    r = results[name]
    print(f"  {name:<24} PR-AUC={r['pr_auc']:.3f}  ROC-AUC={r['roc_auc']:.3f}  "
          f"F1={r['fraud_f1']:.3f}  recall={r['fraud_recall']:.3f}  thr={r['threshold']:.3f}")

best_name = max(results, key=lambda n: results[n]["pr_auc"])
best_model = fitted[best_name]
threshold = results[best_name]["threshold"]
print(f"\nSelected by validation PR-AUC: {best_name} (threshold {threshold:.3f})")

# ---------------------------------------------------------------------------
# Final, one-time evaluation on the untouched test split
# ---------------------------------------------------------------------------
p_test = best_model.predict_proba(Xte)[:, 1]
test_metrics = fraud_metrics(y_test, p_test, threshold)

# Baseline 1: always predict "not fraud"
always_no = {
    "accuracy": round(accuracy_score(y_test, np.zeros_like(y_test)), 4),
    "fraud_recall": 0.0,
    "fraud_f1": 0.0,
}
# Baseline 2: the previous approach (unweighted Gradient Boosting, default 0.5 threshold)
old = GradientBoostingClassifier(random_state=SEED).fit(Xtr, y_train)
old_metrics = fraud_metrics(y_test, old.predict_proba(Xte)[:, 1], 0.5)

print("\n" + "=" * 60)
print("TEST SET (held out, used once)")
print("=" * 60)
print(f"{'':<34}{'Accuracy':>9}{'Recall':>9}{'F1':>8}")
print(f"{'Always predict not fraud':<34}{always_no['accuracy']:>9.3f}{0:>9.3f}{0:>8.3f}")
print(f"{'Previous approach (GB, thr 0.5)':<34}{old_metrics['accuracy']:>9.3f}"
      f"{old_metrics['fraud_recall']:>9.3f}{old_metrics['fraud_f1']:>8.3f}")
print(f"{'New: ' + best_name:<34}{test_metrics['accuracy']:>9.3f}"
      f"{test_metrics['fraud_recall']:>9.3f}{test_metrics['fraud_f1']:>8.3f}")
print(f"\nNew model PR-AUC {test_metrics['pr_auc']:.3f} (random = {y_test.mean():.3f}), "
      f"ROC-AUC {test_metrics['roc_auc']:.3f}")
print(classification_report(y_test, (p_test >= threshold).astype(int), target_names=["Not Fraud", "Fraud"]))
print("Confusion matrix (rows = true, cols = predicted):")
print(np.array(test_metrics["confusion_matrix"]))

# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
with open(MODELS_DIR / "best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(MODELS_DIR / "feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
with open(MODELS_DIR / "input_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
with open(MODELS_DIR / "threshold.json", "w") as f:
    json.dump({"model": best_name, "threshold": threshold}, f, indent=2)

report = {
    "data": {"rows": int(len(X)), "fraud_rate": round(float(y.mean()), 4),
             "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)}},
    "selection": {"metric": "validation PR-AUC (fraud class)", "selected": best_name},
    "validation": results,
    "test": {
        "selected_model": test_metrics,
        "baseline_always_not_fraud": always_no,
        "baseline_previous_approach": old_metrics,
    },
}
with open(REPORTS_DIR / "metrics.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nSaved model, preprocessor, threshold to saved_models/ and metrics to reports/metrics.json")
