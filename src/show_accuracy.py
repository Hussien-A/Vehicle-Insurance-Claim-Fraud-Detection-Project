
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "dataset" / "car_insurance_fraud_dataset.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE / "car_insurance_fraud_dataset.csv"

df = pd.read_csv(DATA_PATH)
df["FraudFound_P"] = (df["fraud_reported"] == "Y").astype(int)
y = df["FraudFound_P"]
drop_cols = ["policy_id", "fraud_reported", "FraudFound_P"]
X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns]).fillna(df.mode().iloc[0])
numeric_cols_raw = X_raw.select_dtypes(include=[np.number]).columns
X_raw[numeric_cols_raw] = X_raw[numeric_cols_raw].fillna(X_raw[numeric_cols_raw].median())

X = X_raw.copy()
if "incident_date" in X.columns:
    X["incident_date"] = pd.to_datetime(X["incident_date"], errors="coerce")
    X["incident_year"] = X["incident_date"].dt.year
    X["incident_month"] = X["incident_date"].dt.month
    X["incident_day_of_week"] = X["incident_date"].dt.dayofweek
    X = X.drop(columns=["incident_date"])
if "claim_amount" in X.columns and "total_claim_amount" in X.columns:
    X["claim_ratio"] = X["claim_amount"] / (X["total_claim_amount"].replace(0, np.nan))
    X["claim_ratio"] = X["claim_ratio"].fillna(1)
if "policy_annual_premium" in X.columns and "policy_deductible" in X.columns:
    X["premium_per_deductible"] = X["policy_annual_premium"] / (X["policy_deductible"].replace(0, np.nan))
    X["premium_per_deductible"] = X["premium_per_deductible"].fillna(X["policy_annual_premium"])
if "police_report_available" in X.columns:
    X["police_report_available"] = (X["police_report_available"].str.upper() == "YES").astype(int)
if "incident_severity" in X.columns:
    X["is_total_loss"] = (X["incident_severity"].str.contains("Total", case=False, na=False)).astype(int)
if "incident_city" in X.columns:
    X = X.drop(columns=["incident_city"])
if "insured_age" in X.columns:
    X["age_group"] = pd.cut(X["insured_age"], bins=[0, 30, 50, 120], labels=[0, 1, 2]).astype(float)
if "incident_day_of_week" in X.columns:
    X["incident_weekend"] = (X["incident_day_of_week"] >= 5).astype(int)
if "claim_amount" in X.columns:
    X["log_claim_amount"] = np.log1p(X["claim_amount"])
if "total_claim_amount" in X.columns:
    X["log_total_claim_amount"] = np.log1p(X["total_claim_amount"])
if "number_of_vehicles_involved" in X.columns and "bodily_injuries" in X.columns:
    X["vehicles_plus_injuries"] = X["number_of_vehicles_involved"] + X["bodily_injuries"]
if "claim_ratio" in X.columns:
    X["high_claim_ratio"] = (X["claim_ratio"] > 1.1).astype(int)

with open(BASE / "saved_models" / "preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
with open(BASE / "saved_models" / "best_model.pkl", "rb") as f:
    model = pickle.load(f)

X_processed = preprocessor.transform(X)
y_pred = model.predict(X_processed)
accuracy = accuracy_score(y, y_pred)

print("=" * 50)
print("  MODEL ACCURACY (full dataset)")
print("=" * 50)
print(f"  Accuracy: {accuracy:.4f}  ({accuracy:.2%})")
print("=" * 50)
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Not Fraud", "Fraud"]))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print("  (rows: true, cols: predicted)")
