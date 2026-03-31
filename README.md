# Vehicle Insurance Claim Fraud Detection

**Binary classification** for insurance claim fraud: predict whether a claim is **Fraud** or **Legitimate** from policy, insured, incident, and claim data. End-to-end ML project with EDA, feature engineering, training, FastAPI backend, and Streamlit UI.

---

## Features

- **Notebook (Colab):** EDA, preprocessing, feature engineering, multiple classifiers (LR, NB, KNN, SVM, DT, RF, GB, optional XGBoost), evaluation, save best model by accuracy.
- **Local training:** `src/train_local.py` — same pipeline, extra features, optional SMOTE, RF/GB/XGBoost with tuning, optional ensemble; refit on full data for high accuracy.
- **API:** FastAPI with `/predict` — load preprocessor + model from `saved_models/`, return prediction and fraud probability.
- **Web app:** Streamlit form (Policy & Insured, Incident & Claim), calls API, shows Fraud/Legitimate and probability.
- **Accuracy script:** `src/show_accuracy.py` — evaluate saved model on the dataset (accuracy, classification report, confusion matrix).

---

## Dataset

| Item | Description |
|------|-------------|
| **Source** | [Kaggle – Car Insurance Fraud Detection](https://www.kaggle.com/datasets/ahluwaliasaksham/car-insurance-fraud-detection-dataset) |
| **Target** | `fraud_reported` (Y/N) → mapped to `FraudFound_P` (1 = Fraud, 0 = Legitimate) |
| **Content** | Policy (state, deductible, premium), insured (age, sex, education, occupation, hobbies), incident (type, severity, date, vehicles, injuries, witnesses, police report), claim amounts |
| **Location** | `dataset/car_insurance_fraud_dataset.csv` (place your copy here or set path in notebook/script) |

---

## Project Structure

```
vehicle_insurance_fraud/
├── README.md
├── .gitignore
├── dataset/
│   └── car_insurance_fraud_dataset.csv       # Raw data (required for training / show_accuracy)
├── saved_models/                             # Created by notebook or src/train_local.py (required for API)
│   ├── preprocessor.pkl
│   ├── best_model.pkl
│   ├── feature_names.pkl
│   └── input_columns.pkl
├── scripts/
│   ├── run_api.sh                            # Start FastAPI (from project root)
│   └── run_streamlit.sh                     # Start Streamlit (from project root)
├── src/
│   ├── train_local.py                        # Local training (same pipeline + extras, RF/GB/XGB)
│   └── show_accuracy.py                      # Evaluate saved model on dataset
├── notebooks/
│   └── Vehicle_Insurance_Fraud_Detection.ipynb   # Colab: EDA, preprocessing, FE, training, save model
├── api/
│   ├── main.py                               # FastAPI app: /, /health, /predict
│   ├── preprocess.py                         # Build features from request (same as training)
│   ├── requirements.txt
│   └── show_accuracy.py                      # Wrapper to run src/show_accuracy.py from api/
└── streamlit_app/
    ├── app.py                                # Streamlit UI (form → API → result)
    ├── .streamlit/
    │   └── config.toml                       # Theme
    └── requirements.txt
```

---

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- Dataset CSV in `dataset/car_insurance_fraud_dataset.csv` (for training and `show_accuracy`)

---

## Installation

1. **Clone or download** the project and go to its root:

   ```bash
   cd vehicle_insurance_fraud
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies** (API + Streamlit + training):

   ```bash
   pip install -r api/requirements.txt -r streamlit_app/requirements.txt
   ```

   For **local training** with XGBoost/LightGBM (optional):

   ```bash
   pip install xgboost lightgbm
   ```

---

## Quick Start

You need **saved models** before running the API or Streamlit. Either train in Colab and copy `saved_models/` into the project root, or train locally (see below).

1. **Start the API** (from project root):

   ```bash
   ./scripts/run_api.sh
   ```

   Or manually:

   ```bash
   source .venv/bin/activate
   cd api && uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   - API: http://localhost:8000  
   - Docs: http://localhost:8000/docs  
   - Health: http://localhost:8000/health  

2. **Start the Streamlit app** (new terminal, from project root):

   ```bash
   ./scripts/run_streamlit.sh
   ```

   Or:

   ```bash
   source .venv/bin/activate
   streamlit run streamlit_app/app.py
   ```

   Open the URL shown (e.g. http://localhost:8501). Fill the form and click **Get Prediction**.

---

## Usage

### Training a model

**Option A – Google Colab (recommended for first run)**  
1. Open `notebooks/Vehicle_Insurance_Fraud_Detection.ipynb` in Colab (upload the notebook or open from repo).  
2. Ensure the dataset is at `dataset/car_insurance_fraud_dataset.csv` relative to the notebook (e.g. upload project and use `../dataset/` from `notebooks/`). The notebook uses `../dataset/` and `../saved_models/`.  
3. Run all cells (EDA → preprocessing → feature engineering → train → evaluate → save).  
4. Download the `saved_models/` folder from Colab and place it in `vehicle_insurance_fraud/` (project root).

**Option B – Local training**  
From project root with venv activated:

```bash
python src/train_local.py
```

This writes `saved_models/preprocessor.pkl` and `saved_models/best_model.pkl`. The script uses the same feature pipeline as the API (with extra engineered features). Optional: set env vars for data path, SMOTE, full-data refit, etc. (see script).

**Important:** The API expects **scikit-learn 1.4.0** to load the pickles. If you retrain with a different sklearn version, either keep that version when running the API or retrain after upgrading.

### API endpoints

| Method | Endpoint   | Description |
|--------|------------|-------------|
| GET    | `/`        | Welcome + link to docs |
| GET    | `/health`  | Status + whether model is loaded |
| POST   | `/predict` | JSON body with claim fields → `prediction`, `label`, `probability_fraud` |

Request body (all fields have defaults): `policy_state`, `policy_deductible`, `policy_annual_premium`, `insured_age`, `insured_sex`, `insured_education_level`, `insured_occupation`, `insured_hobbies`, `incident_type`, `collision_type`, `incident_severity`, `authorities_contacted`, `incident_state`, `incident_date` (YYYY-MM-DD), `incident_hour_of_the_day`, `number_of_vehicles_involved`, `bodily_injuries`, `witnesses`, `police_report_available`, `claim_amount`, `total_claim_amount`.  
See **http://localhost:8000/docs** for the full schema.

### Streamlit app

- Form: Policy & Insured (left), Incident & Claim (right).  
- Submit → request sent to API → result shown (Fraud/Legitimate + probability).  
- API URL: default `http://localhost:8000`. Override:

  ```bash
  STREAMLIT_API_URL=http://your-host:8000 streamlit run streamlit_app/app.py
  ```

### Check model accuracy

From project root (with venv activated and dataset in `dataset/`):

```bash
python src/show_accuracy.py
```

Or from `api/` (wrapper runs `src/show_accuracy.py` with correct cwd):

```bash
cd api && python show_accuracy.py
```

Prints accuracy, classification report, and confusion matrix for the saved model on the full dataset.

---

## Models and pipeline

- **Notebook:** Best model selected by **accuracy** among LR, NB, KNN, SVM, DT, RF, GB (optional XGBoost).  
- **src/train_local.py:** RandomizedSearchCV on RF, GB, XGBoost (optional LightGBM), optional ensemble; same preprocessing and feature engineering as the notebook plus extra features (e.g. age_group, incident_weekend, log claim amounts, high_claim_ratio).  
- **Preprocessing:** ColumnTransformer (OneHotEncoder for categoricals, StandardScaler for numerics); same pipeline in notebook, `src/train_local.py`, and `api/preprocess.py` so the API input matches training.

---

## Tech Stack

| Component   | Stack |
|------------|--------|
| Notebook   | pandas, numpy, matplotlib, seaborn, scikit-learn |
| Training   | scikit-learn, optional: xgboost, lightgbm, imbalanced-learn (SMOTE) |
| API        | FastAPI, Pydantic, uvicorn |
| Frontend   | Streamlit, requests |


