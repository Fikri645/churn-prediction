---
title: Customer Churn Predictor
emoji: 📉
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "5.9.1"
app_file: app/gradio_app.py
pinned: false
python_version: "3.11"
---

# Customer Churn Prediction

![CI](https://github.com/Fikri645/churn-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-orange)
[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fikri0o0/churn-prediction)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end MLOps pipeline predicting telecom customer churn. Built as a Data Scientist portfolio piece demonstrating the full workflow from raw data to a live web demo.

**[Live Demo →](https://huggingface.co/spaces/fikri0o0/churn-prediction)**  |  **[GitHub →](https://github.com/Fikri645/churn-prediction)**

---

## Highlights

| What | Detail |
|---|---|
| **Algorithm** | XGBoost (Optuna HPO, 50 trials) — validated best across 4-model comparison |
| **ROC-AUC** | **0.8486** (test set, 20% split) |
| **Business value** | Business-optimal decision threshold lifts net profit from $57.7K → $64.7K (+$7.2K) on the 1,409-customer test set |
| **Explainability** | SHAP TreeExplainer — global importance + per-customer bar chart |
| **Calibration** | Reliability diagram + Brier score — verifying probabilities are meaningful |
| **Experiments** | 4-model comparison (LR / LightGBM / XGBoost / Stacking) + 4-strategy SMOTE comparison |
| **Serving** | FastAPI REST endpoint (`/predict`, `/predict/batch`) |
| **UI** | Gradio — real-time SHAP + business cost estimate per prediction |
| **Experiment tracking** | MLflow — all trials, model comparison, SMOTE variants |
| **Drift monitoring** | Evidently HTML report |
| **Deployment** | Docker (API) + Hugging Face Spaces (Gradio UI) |

---

## Architecture

```
Raw CSV
  └─► preprocess.py       (sklearn Pipeline: impute → encode → scale)
        ├─► train.py       (XGBoost + Optuna 50-trial HPO + MLflow)
        ├─► experiments.py (model comparison + SMOTE + stacking + business metrics)
        └─► evaluate.py    (confusion matrix, ROC, SHAP, Expected Profit Curve)
              ├─► api/main.py          (FastAPI — /predict, /predict/batch)
              ├─► app/gradio_app.py    (HF Spaces — SHAP + business cost estimate)
              └─► monitoring/drift_report.py (Evidently)
```

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/Fikri645/churn-prediction
cd churn-prediction
pip install -r requirements.txt

# 2. Download dataset
python scripts/download_data.py

# 3. Train (Optuna HPO + MLflow)
python -m src.train

# 4. Evaluate + generate SHAP plots + Expected Profit Curve
python -m src.evaluate

# 5. Run model comparison experiments (SMOTE, LightGBM, Stacking — logged to MLflow)
python -m src.experiments

# 7. Run API locally
uvicorn api.main:app --reload

# 8. Run Gradio UI locally
python app/gradio_app.py

# 9. Generate drift report
python -m monitoring.drift_report
```

---

## Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                    # Telco-Customer-Churn.csv (not committed)
│   └── processed/              # train.csv, test.csv
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory Data Analysis
├── src/
│   ├── config.py               # paths, feature lists, constants
│   ├── preprocess.py           # sklearn ColumnTransformer Pipeline
│   ├── train.py                # XGBoost + Optuna + MLflow
│   ├── evaluate.py             # metrics, ROC, SHAP, calibration, profit curve
│   ├── business_metrics.py     # cost-sensitive metrics, Expected Profit Curve
│   └── experiments.py          # SMOTE comparison, model comparison, stacking
├── tests/
│   ├── test_business_metrics.py
│   ├── test_preprocess.py
│   └── test_api_schemas.py
├── api/
│   ├── main.py                 # FastAPI app
│   └── schemas.py              # Pydantic request/response models
├── app/
│   └── gradio_app.py           # Gradio UI (HF Spaces entry point)
├── monitoring/
│   └── drift_report.py         # Evidently drift detection
├── models/                     # Saved joblib pipeline
├── reports/figures/            # SHAP + confusion matrix + ROC plots
├── scripts/
│   └── download_data.py        # Dataset downloader
├── Dockerfile                  # FastAPI container
├── docker-compose.yml          # API + MLflow UI
└── requirements.txt
```

---

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 20 features, ~26% churn rate.

Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Features: tenure, monthly charges, contract type, internet service, add-on services (security, backup, streaming), payment method, demographics.

---

## Model & Results

### Preprocessing

- `TotalCharges`: whitespace → `NaN` → median impute
- Binary categoricals (`Yes`/`No`, `Male`/`Female`): mapped to 1/0
- Multi-class categoricals: OneHotEncoder
- Numerics: StandardScaler
- Class imbalance: `scale_pos_weight = n_negative / n_positive`

### Hyperparameter Search (Optuna)

50 trials with TPE sampler over: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`.

**Best hyperparameters found:**
- `n_estimators=600`, `max_depth=3`, `learning_rate=0.016`
- `subsample=0.59`, `colsample_bytree=0.70`
- `scale_pos_weight=2.77` (class imbalance correction)

**Test set results (1,409 customers):**

| Metric | Default threshold (0.5) | Optimal threshold |
|---|---|---|
| ROC-AUC | **0.8486** | 0.8486 (threshold-independent) |
| Recall (Churn) | 0.80 | higher |
| Precision (Churn) | 0.52 | lower |
| F1 | 0.63 | — |

### Class Imbalance Experiment (4 strategies compared)

| Strategy | ROC-AUC | Recall | Verdict |
|---|---|---|---|
| No balancing | 0.8391 | 0.53 | ❌ misses half of churners |
| **`scale_pos_weight` (chosen)** | **0.8391** | **0.76** | ✅ best recall for churn use case |
| SMOTE | 0.8396 | 0.62 | marginal AUC gain, lower recall |
| SMOTE-Tomek | 0.8378 | 0.61 | underperforms both |

**Finding:** `scale_pos_weight` is the right choice here — in churn, false negatives (missed churners) cost 5–10× more than false positives, so recall is the priority metric.

### Model Comparison (4 models, same train/test split)

| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic Regression (baseline) | 0.8407 | 0.6176 |
| LightGBM | 0.8264 | 0.5993 |
| **XGBoost — Optuna 50 trials** | **0.8488** | **0.6303** |
| Stacking (XGB + LR + LightGBM) | 0.8454 | 0.6047 |

**Finding:** LightGBM underperforms XGBoost on this specific dataset, and a stacking ensemble (LR + XGBoost + LightGBM with a logistic meta-learner) does **not** beat the single Optuna-tuned XGBoost (0.8454 < 0.8488). Sometimes the simpler model wins — so XGBoost stays in production.

### Business Value Analysis

Cost structure: FN (missed churner) = CLV lost ≈ $769 (12-mo) | FP (wrong contact) = $20 campaign cost | retention success rate = 30%

| Scenario | Contacts | Retained | Revenue saved | Campaign spend | Net profit |
|---|---|---|---|---|---|
| Default threshold (0.50) | 578 | 90 | $69,216 | $11,560 | $57,656 |
| **Optimal threshold (0.168)** | 943 | 109 | $83,521 | $18,860 | **$64,661** |

The optimal threshold is found by sweeping 200 thresholds and **maximising expected profit — not F1**. Because a missed churner costs ~38× a wasted outreach, the model should cast a wider net: lowering the threshold to 0.168 contacts 365 more customers, but saves an extra **$6,805 net** (+12.6%). This is the decision production teams actually care about. See `reports/figures/profit_curve.png`.

### Key SHAP Findings

1. **Month-to-month contract** — strongest churn driver; two-year contracts cut risk sharply
2. **Fiber optic internet** — higher churn than DSL, likely price sensitivity
3. **Short tenure** — new customers at highest risk; first 12 months critical
4. **No tech support / online security** — absence of add-ons increases risk
5. **High monthly charges** — especially without a long-term contract

---

## API Reference

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
       "tenure":12,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
       "StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":79.85,"TotalCharges":958.2}'

# Response
{"churn_probability": 0.7821, "churn_prediction": true, "risk_label": "High"}
```

---

## Docker

```bash
# Build and run API + MLflow UI
docker compose up --build

# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000
```

---

## What I Learned

- **The threshold is a business decision, not 0.5.** ROC-AUC measures ranking; profit measures the business. Sweeping thresholds against an explicit cost matrix (FN ≈ 38× FP) moved the decision point to 0.168 and added ~$6.8K net profit — with no change to the model itself.
- **Complexity isn't free, and doesn't always pay.** A stacking ensemble (LR + XGBoost + LightGBM) *underperformed* the single Optuna-tuned XGBoost (0.8454 vs 0.8488). Knowing when to stop adding models is part of the job.
- **Class imbalance matters at prediction time too.** `scale_pos_weight` in XGBoost beat SMOTE and SMOTE-Tomek on recall (0.76 vs 0.62/0.61) — the metric that matters when missing a churner is the expensive error.
- **SHAP beats feature importance for stakeholders.** Showing *per-customer* reasons for a prediction is more actionable than a global bar chart alone.
- **Optuna > GridSearch for HPO.** 50 TPE trials covers more of the space than an exhaustive 3-fold grid in the same wall time.
- **Evidently makes drift visible, not invisible.** Monthly charges distribution shifts first — good leading indicator before model performance degrades.
