---
title: Customer Churn Predictor
emoji: 📉
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "5.6.0"
app_file: app/gradio_app.py
pinned: false
python_version: "3.11"
---

# Customer Churn Prediction

End-to-end MLOps pipeline predicting telecom customer churn. Built as a Data Scientist portfolio piece demonstrating the full workflow from raw data to a live web demo.

**[Live Demo →](https://huggingface.co/spaces/fikri0o0/churn-prediction)**  |  **[MLflow Experiments →](#)**

---

## Highlights

| What | Detail |
|---|---|
| **Algorithm** | XGBoost with Optuna HPO (50 trials, TPE sampler) |
| **ROC-AUC** | 0.8486 (test set, 20% split) |
| **Explainability** | SHAP TreeExplainer — global + per-customer breakdown |
| **Serving** | FastAPI REST endpoint (`/predict`, `/predict/batch`) |
| **UI** | Gradio — single prediction + CSV batch upload |
| **Experiment tracking** | MLflow — params, metrics, model registry |
| **Drift monitoring** | Evidently HTML report |
| **Deployment** | Docker (API) + Hugging Face Spaces (Gradio UI) |

---

## Architecture

```
Raw CSV
  └─► preprocess.py  (sklearn Pipeline: impute → encode → scale)
        └─► train.py  (XGBoost + Optuna + MLflow logging)
              └─► evaluate.py  (confusion matrix, ROC, SHAP plots)
                    ├─► api/main.py  (FastAPI — /predict, /predict/batch)
                    ├─► app/gradio_app.py  (HF Spaces demo)
                    └─► monitoring/drift_report.py  (Evidently)
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

# 4. Evaluate + generate SHAP plots
python -m src.evaluate

# 5. Run API locally
uvicorn api.main:app --reload

# 6. Run Gradio UI locally
python app/gradio_app.py

# 7. Generate drift report
python -m monitoring.drift_report
```

---

## Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                    # Telco-Customer-Churn.csv (not committed)
│   └── processed/              # train.csv, test.csv
├── src/
│   ├── config.py               # paths, feature lists, constants
│   ├── preprocess.py           # sklearn ColumnTransformer Pipeline
│   ├── train.py                # XGBoost + Optuna + MLflow
│   └── evaluate.py             # metrics, ROC, SHAP plots
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

| Metric | Score |
|---|---|
| ROC-AUC | **0.8486** |
| Recall (Churn) | **0.80** — catches 4 in 5 churners |
| Precision (Churn) | 0.52 |
| F1 (Churn) | 0.63 |
| Accuracy | 0.75 |

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

- **Class imbalance matters at prediction time too.** `scale_pos_weight` in XGBoost fixes calibration; without it recall on the minority class collapses.
- **SHAP beats feature importance for stakeholders.** Showing *per-customer* reasons for a prediction is more actionable than a global bar chart alone.
- **Optuna > GridSearch for HPO.** 50 TPE trials covers more of the space than an exhaustive 3-fold grid in the same wall time.
- **Evidently makes drift visible, not invisible.** Monthly charges distribution shifts first — good leading indicator before model performance degrades.
