import os
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "saved_models"
if not MODELS_DIR.exists():
    MODELS_DIR = BASE_DIR / "saved_models"

app = FastAPI(
    title="Vehicle Insurance Fraud Detection API",
    description="Predict whether an insurance claim is fraudulent (1) or legitimate (0).",
    version="1.0.0",
)

preprocessor = None
model = None


def load_artifacts():
    global preprocessor, model
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    model_path = MODELS_DIR / "best_model.pkl"
    if not preprocessor_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Saved model files not found. Run the Colab notebook and copy saved_models/ (preprocessor.pkl, best_model.pkl) into api/saved_models/ or project root saved_models/"
        )
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)


@app.on_event("startup")
def startup():
    try:
        load_artifacts()
    except FileNotFoundError as e:
        import sys
        print(str(e), file=sys.stderr)
        pass


class PredictionRequest(BaseModel):
    policy_state: str = Field("GA", description="Policy state code")
    policy_deductible: float = Field(500, ge=0)
    policy_annual_premium: float = Field(1000, ge=0)
    insured_age: int = Field(35, ge=0, le=120)
    insured_sex: str = Field("MALE", description="MALE, FEMALE, OTHER")
    insured_education_level: str = Field("College")
    insured_occupation: str = Field("Other")
    insured_hobbies: str = Field("other")
    incident_type: str = Field("Multi-vehicle Collision")
    collision_type: str = Field("Front")
    incident_severity: str = Field("Major Damage")
    authorities_contacted: str = Field("Police")
    incident_state: str = Field("GA")
    incident_date: str = Field("2024-01-15", description="YYYY-MM-DD")
    incident_hour_of_the_day: int = Field(12, ge=0, le=23)
    number_of_vehicles_involved: int = Field(1, ge=1)
    bodily_injuries: int = Field(0, ge=0)
    witnesses: int = Field(0, ge=0)
    police_report_available: str = Field("No", description="Yes or No")
    claim_amount: float = Field(1000, ge=0)
    total_claim_amount: float = Field(1000, ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_fraud: Optional[float] = None


@app.get("/")
def root():
    return {"message": "Vehicle Insurance Fraud Detection API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if preprocessor is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Add saved_models and restart.")
    try:
        from preprocess import build_features_from_request
        data = req.model_dump()
        df = build_features_from_request(data)
        X = preprocessor.transform(df)
        pred = int(model.predict(X)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])
        return PredictionResponse(
            prediction=pred,
            label="Fraud" if pred == 1 else "Legitimate",
            probability_fraud=proba,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
