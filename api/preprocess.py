import pandas as pd
import numpy as np
from datetime import datetime


def build_features_from_request(data: dict) -> pd.DataFrame:
    incident_date = data.get("incident_date")
    if isinstance(incident_date, str):
        try:
            dt = pd.to_datetime(incident_date)
        except Exception:
            dt = pd.Timestamp.now()
    else:
        dt = pd.Timestamp.now()

    claim_amount = float(data.get("claim_amount", 0))
    total_claim = float(data.get("total_claim_amount", 1))
    claim_ratio = claim_amount / total_claim if total_claim else 1.0

    premium = float(data.get("policy_annual_premium", 0))
    deductible = float(data.get("policy_deductible", 1))
    premium_per_deductible = premium / deductible if deductible else premium

    police = data.get("police_report_available", "No")
    if isinstance(police, str):
        police_report_available = 1 if police.upper() in ("YES", "Y", "1") else 0
    else:
        police_report_available = int(bool(police))

    severity = str(data.get("incident_severity", ""))
    is_total_loss = 1 if "Total" in severity else 0

    age = int(data.get("insured_age", 35))
    if age <= 30:
        age_group = 0.0
    elif age <= 50:
        age_group = 1.0
    else:
        age_group = 2.0
    day_of_week = dt.dayofweek
    incident_weekend = 1 if day_of_week >= 5 else 0
    log_claim_amount = np.log1p(claim_amount)
    log_total_claim_amount = np.log1p(total_claim)
    vehicles_plus_injuries = int(data.get("number_of_vehicles_involved", 1)) + int(data.get("bodily_injuries", 0))
    high_claim_ratio = 1 if claim_ratio > 1.1 else 0

    row = {
        "policy_state": str(data.get("policy_state", "GA")),
        "policy_deductible": float(data.get("policy_deductible", 500)),
        "policy_annual_premium": float(data.get("policy_annual_premium", 1000)),
        "insured_age": int(data.get("insured_age", 35)),
        "insured_sex": str(data.get("insured_sex", "MALE")),
        "insured_education_level": str(data.get("insured_education_level", "College")),
        "insured_occupation": str(data.get("insured_occupation", "Other")),
        "insured_hobbies": str(data.get("insured_hobbies", "other")),
        "incident_type": str(data.get("incident_type", "Multi-vehicle Collision")),
        "collision_type": str(data.get("collision_type", "Front")),
        "incident_severity": str(data.get("incident_severity", "Major Damage")),
        "authorities_contacted": str(data.get("authorities_contacted", "Police")),
        "incident_state": str(data.get("incident_state", "GA")),
        "incident_hour_of_the_day": int(data.get("incident_hour_of_the_day", 12)),
        "number_of_vehicles_involved": int(data.get("number_of_vehicles_involved", 1)),
        "bodily_injuries": int(data.get("bodily_injuries", 0)),
        "witnesses": int(data.get("witnesses", 0)),
        "police_report_available": police_report_available,
        "claim_amount": claim_amount,
        "total_claim_amount": total_claim,
        "incident_year": dt.year,
        "incident_month": dt.month,
        "incident_day_of_week": dt.dayofweek,
        "claim_ratio": claim_ratio,
        "premium_per_deductible": premium_per_deductible,
        "is_total_loss": is_total_loss,
        "age_group": age_group,
        "incident_weekend": incident_weekend,
        "log_claim_amount": log_claim_amount,
        "log_total_claim_amount": log_total_claim_amount,
        "vehicles_plus_injuries": vehicles_plus_injuries,
        "high_claim_ratio": high_claim_ratio,
    }
    return pd.DataFrame([row])
