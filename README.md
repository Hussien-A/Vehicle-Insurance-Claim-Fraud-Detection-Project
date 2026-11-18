# Vehicle Insurance Claim Fraud Detection

**Binary classification** for insurance claim fraud: predict whether a claim is **Fraud** or **Legitimate** from policy, insured, incident, and claim data. End-to-end ML project with EDA, feature engineering, training, FastAPI backend, and Streamlit UI.

---

## Results

Only **11.5% of claims are fraud**, so a model that always answers "not fraud" already gets ~88.5% accuracy.
Accuracy is therefore the wrong yardstick here. The pipeline is evaluated on the **fraud class** instead:
how many fraud cases it catches (recall), how precise its flags are, and PR-AUC.

Held-out test set: 6,000 claims never used for training, model selection or threshold tuning.

| Model | Accuracy | Fraud recall | Fraud precision | Fraud F1 | PR-AUC |
|---|---|---|---|---|---|
| Always predict "not fraud" | 0.885 | 0.000 | – | 0.000 | 0.115 (chance) |
| Previous approach (selected by accuracy, threshold 0.5) | 0.883 | 0.028 | 0.365 | 0.051 | 0.279 |
| **Current: Hist Gradient Boosting, class-weighted, tuned threshold** | 0.765 | **0.645** | 0.276 | **0.386** | **0.282** |

**What changed and why**

1. **Model selection by validation PR-AUC on the fraud class**, not by accuracy. Accuracy rewarded models that ignore fraud.
2. **Class weighting** (`class_weight="balanced"`) so the minority class counts during training.
3. **Decision threshold tuned on a validation split** (0.57, maximising fraud F1) instead of the default 0.5. The API uses this threshold.
4. **Train / validation / test split (60 / 20 / 20)** with the preprocessor fit on the training split only. The test split is used once, for the table above.
5. **Removed full-dataset "accuracy"** reporting, which scored the model on its own training rows and overstated performance.

Result: the model now catches **444 of 688 fraud cases (64.5%)** in the test set, up from **2.8%**.
The trade-off is more false alarms (1,167 legitimate claims flagged), which is the usual cost of
catching fraud and can be moved along the precision/recall curve by changing the threshold.

Note that PR-AUC barely moved (0.279 → 0.282): the old model could already rank claims by risk about as well.
The big gain comes from **evaluating and deciding correctly** (the right metric and the right threshold), not from a stronger model.

**Honest limits:** PR-AUC of 0.28 is about 2.5× chance, so the features carry real but limited signal.
Better fraud detection would need richer data (claim history, repairer and network features) rather than more model tuning.

Full metrics for every candidate model are in `reports/metrics.json`.

---

## Features

- **Notebook (Colab):** EDA, preprocessing, feature engineering, multiple classifiers (LR, NB, KNN, SVM, DT, RF, GB, optional XGBoost), evaluation, save best model by accuracy.
- **Local training:** `src/train_local.py` — train/validation/test split, class-weighted LR, RF, GB and HistGradientBoosting, model selection by fraud-class PR-AUC, threshold tuning, and a final one-time test evaluation against baselines. Writes `reports/metrics.json`.
- **API:** FastAPI with `/predict` — load preprocessor + model from `saved_models/`, return prediction and fraud probability.
- **Web app:** Streamlit form (Policy & Insured, Incident & Claim), calls API, shows Fraud/Legitimate and probability.
- **Evaluation script:** `src/show_accuracy.py` — prints the held-out test results (fraud recall, precision, F1, PR-AUC, confusion matrix) next to the baselines.

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
│   ├── input_columns.pkl
│   └── threshold.json                        # Decision threshold tuned on validation data
├── reports/
│   └── metrics.json                          # Validation + test metrics written by src/train_local.py
├── scripts/
│   ├── run_api.sh                            # Start FastAPI (from project root)
│   └── run_streamlit.sh                     # Start Streamlit (from project root)
├── src/
│   ├── train_local.py                        # Training, model selection, threshold tuning, test evaluation
│   └── show_accuracy.py                      # Print held-out test metrics
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

This writes the preprocessor, model and `threshold.json` to `saved_models/`, and all metrics to `reports/metrics.json`. The script uses the same feature pipeline as the API. Set `DATA_PATH` to use a different CSV. Training takes under a minute on a laptop CPU.

> The Colab notebook is kept as the original exploration (EDA and first model comparison). Its model selection by accuracy is superseded by `src/train_local.py`, which produces the model the API serves.

**Important:** The API expects **scikit-learn 1.4.0** to load the pickles. If you retrain with a different sklearn version, either keep that version when running the API or retrain after upgrading.

### API endpoints

| Method | Endpoint   | Description |
|--------|------------|-------------|
| GET    | `/`        | Welcome + link to docs |
| GET    | `/health`  | Status + whether model is loaded |
| POST   | `/predict` | JSON body with claim fields → `prediction`, `label`, `probability_fraud`, `threshold` (fraud if probability ≥ threshold) |

Request body (all fields have defaults): `policy_state`, `policy_deductible`, `policy_annual_premium`, `insured_age`, `insured_sex`, `insured_education_level`, `insured_occupation`, `insured_hobbies`, `incident_type`, `collision_type`, `incident_severity`, `authorities_contacted`, `incident_state`, `incident_date` (YYYY-MM-DD), `incident_hour_of_the_day`, `number_of_vehicles_involved`, `bodily_injuries`, `witnesses`, `police_report_available`, `claim_amount`, `total_claim_amount`.  
See **http://localhost:8000/docs** for the full schema.

### Streamlit app

- Form: Policy & Insured (left), Incident & Claim (right).  
- Submit → request sent to API → result shown (Fraud/Legitimate + probability).  
- API URL: default `http://localhost:8000`. Override:

  ```bash
  STREAMLIT_API_URL=http://your-host:8000 streamlit run streamlit_app/app.py
  ```

### Check model performance

From project root (after running `src/train_local.py`):

```bash
python src/show_accuracy.py
```

Or from `api/` (wrapper runs `src/show_accuracy.py` with correct cwd):

```bash
cd api && python show_accuracy.py
```

Prints the held-out test metrics for the saved model next to the two baselines.

---

## Models and pipeline

- **Notebook:** Original exploration; compares LR, NB, KNN, SVM, DT, RF, GB by accuracy (kept for reference, see Results for why accuracy misleads here).  
- **src/train_local.py:** Class-weighted Logistic Regression, Random Forest, Gradient Boosting and HistGradientBoosting; selection by validation PR-AUC; threshold tuned for fraud F1; extra features (age_group, incident_weekend, log claim amounts, claim_ratio, high_claim_ratio).  
- **Preprocessing:** ColumnTransformer (OneHotEncoder for categoricals, StandardScaler for numerics); same pipeline in notebook, `src/train_local.py`, and `api/preprocess.py` so the API input matches training.

---

## Tech Stack

| Component   | Stack |
|------------|--------|
| Notebook   | pandas, numpy, matplotlib, seaborn, scikit-learn |
| Training   | scikit-learn |
| API        | FastAPI, Pydantic, uvicorn |
| Frontend   | Streamlit, requests |


