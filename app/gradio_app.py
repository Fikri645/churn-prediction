"""
Gradio UI — three tabs:
  1. Single Prediction  — form input → churn probability + SHAP bar
  2. Batch Upload       — CSV upload → table of predictions + summary
  3. Model Info         — metrics, dataset description, SHAP summary plot

Deploy to HF Spaces:
    gradio deploy
"""
import io
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import gradio as gr

from src.config import MODEL_PATH, MODELS_DIR
from src.preprocess import build_preprocessor, get_feature_names, load_raw, split

# ── Load artifacts ─────────────────────────────────────────────────────────
pipeline = joblib.load(MODEL_PATH)

# Support both pure-sklearn pipelines and imblearn pipelines
_steps = dict(pipeline.named_steps)
preprocessor  = _steps["prep"]
model         = _steps["model"]
feature_names = get_feature_names(preprocessor)

# Only TreeExplainer works for tree-based models (XGBoost/LightGBM)
try:
    explainer = shap.TreeExplainer(model)
    _has_shap = True
except Exception:
    _has_shap = False

# Load optimal threshold + business metadata (written by src/experiments.py)
_meta_path = MODELS_DIR / "model_meta.json"
_meta      = json.loads(_meta_path.read_text()) if _meta_path.exists() else {}
OPTIMAL_THRESHOLD = _meta.get("optimal_threshold", 0.5)
CLV_MONTHS        = 12      # months used for CLV estimate
CAMPAIGN_COST     = 20.0    # $ per customer contacted
RETENTION_RATE    = 0.30    # fraction of contacted churners retained


def _predict_df(df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(df)[:, 1]


def _risk(prob: float, threshold: float = OPTIMAL_THRESHOLD) -> str:
    """Risk label uses the business-optimal threshold, not 0.5."""
    if prob >= threshold:
        intensity = (prob - threshold) / (1 - threshold)
        return "🔴 High" if intensity >= 0.4 else "🟡 Medium-High"
    intensity = prob / threshold
    return "🟡 Medium-Low" if intensity >= 0.6 else "🟢 Low"


def _business_estimate(prob: float, monthly_charges: float) -> str:
    clv = monthly_charges * CLV_MONTHS
    if prob >= OPTIMAL_THRESHOLD:
        # If we act (contact this customer):
        net = clv * RETENTION_RATE - CAMPAIGN_COST
        return (
            f"**Estimated CLV at risk:** ${clv:,.0f}\n\n"
            f"**If we intervene** (retention campaign ~${CAMPAIGN_COST:.0f}):\n"
            f"Expected net value = ${net:,.0f}  "
            f"*(30% retention rate assumed)*"
        )
    return f"*Low risk — no retention action needed (CLV: ${clv:,.0f})*"


# ── SHAP local bar chart ───────────────────────────────────────────────────

def shap_bar_for_row(df_row: pd.DataFrame) -> plt.Figure:
    if not _has_shap:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "SHAP not available for this model type",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    X_t = preprocessor.transform(df_row)
    sv  = explainer.shap_values(X_t)[0]
    df_shap = pd.DataFrame({"feature": feature_names, "shap": sv})
    df_shap = df_shap.reindex(df_shap["shap"].abs().sort_values(ascending=False).index).head(12)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in df_shap["shap"]]
    ax.barh(df_shap["feature"], df_shap["shap"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (→ increases churn risk)")
    ax.set_title("Feature impact for this customer")
    fig.tight_layout()
    return fig


# ── Tab 1: Single prediction ───────────────────────────────────────────────

def predict_single(
    gender, SeniorCitizen, Partner, Dependents,
    tenure, PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling,
    PaymentMethod, MonthlyCharges, TotalCharges,
):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Partner": Partner, "Dependents": Dependents,
        "tenure": int(tenure), "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }])
    prob = float(_predict_df(row)[0])
    label = _risk(prob)
    biz   = _business_estimate(prob, float(MonthlyCharges))
    result = (
        f"## Churn Probability: **{prob:.1%}**\n\n"
        f"Risk level: {label}  *(threshold: {OPTIMAL_THRESHOLD})*\n\n"
        f"---\n{biz}"
    )
    fig = shap_bar_for_row(row)
    return result, fig


# ── Tab 2: Batch CSV ───────────────────────────────────────────────────────

def predict_batch(file_obj):
    if file_obj is None:
        return pd.DataFrame(), "Upload a CSV file first."
    df = pd.read_csv(file_obj.name)
    if "customerID" in df.columns:
        ids = df["customerID"]
        df = df.drop(columns=["customerID"])
    else:
        ids = pd.RangeIndex(len(df))
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    probs = _predict_df(df)
    results = pd.DataFrame({
        "customerID"       : ids,
        "churn_probability": probs.round(4),
        "risk_label"       : [_risk(p) for p in probs],
    })
    n_churn = (probs >= 0.5).sum()
    summary = (
        f"**{len(df)} customers** scored.\n"
        f"Predicted churners: **{n_churn}** ({n_churn/len(df):.1%})"
    )
    return results, summary


# ── Build UI ───────────────────────────────────────────────────────────────

YES_NO  = ["Yes", "No"]
# Shortened display labels so 4-per-row fits without truncation
YN_NOPHONE = [("Yes","Yes"), ("No","No"), ("No phone","No phone service")]
YN_NOINET  = [("Yes","Yes"), ("No","No"), ("No inet","No internet service")]
PAYMENT    = [
    ("E-check",       "Electronic check"),
    ("Mail check",    "Mailed check"),
    ("Bank transfer", "Bank transfer (automatic)"),
    ("Credit card",   "Credit card (automatic)"),
]

demo = gr.Blocks(title="Churn Predictor", theme=gr.themes.Soft())
with demo:
    gr.Markdown("# 📉 Customer Churn Predictor\nXGBoost · SHAP · IBM Telco dataset")

    with gr.Tab("🔍 Single Prediction"):
        with gr.Row():

            # ── LEFT: all inputs, 4-per-row — fits in one viewport ────────
            with gr.Column(scale=2):
                tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months)")

                with gr.Row():
                    contract  = gr.Dropdown(["Month-to-month","One year","Two year"],
                                            label="Contract",        value="Month-to-month")
                    internet  = gr.Dropdown(["DSL","Fiber optic","No"],
                                            label="Internet",        value="Fiber optic")
                    monthly   = gr.Number(label="Monthly ($)",        value=65.0)
                    total     = gr.Number(label="Total ($)",          value=780.0)

                with gr.Row():
                    gender    = gr.Dropdown(["Male","Female"],  label="Gender",    value="Male")
                    senior    = gr.Dropdown(["No","Yes"],        label="Senior",    value="No")
                    partner   = gr.Dropdown(YES_NO,              label="Partner",   value="No")
                    dependents= gr.Dropdown(YES_NO,              label="Dependents",value="No")

                with gr.Row():
                    phone     = gr.Dropdown(YES_NO,      label="Phone",     value="Yes")
                    multilines= gr.Dropdown(YN_NOPHONE,  label="Multi-line",value="No")
                    paperless = gr.Dropdown(YES_NO,      label="Paperless", value="Yes")
                    payment   = gr.Dropdown(PAYMENT,     label="Payment",   value="Electronic check")

                with gr.Row():
                    security  = gr.Dropdown(YN_NOINET, label="Security", value="No")
                    backup    = gr.Dropdown(YN_NOINET, label="Backup",   value="No")
                    protection= gr.Dropdown(YN_NOINET, label="Protect",  value="No")
                    techsupport=gr.Dropdown(YN_NOINET, label="Tech Sup", value="No")

                with gr.Row():
                    streaming_tv = gr.Dropdown(YN_NOINET, label="Stream TV",    value="No")
                    streaming_mv = gr.Dropdown(YN_NOINET, label="Stream Movies", value="No")

                btn = gr.Button("🔮 Predict Churn", variant="primary", size="lg")

            # ── RIGHT: result + SHAP (always visible beside form) ─────────
            with gr.Column(scale=1):
                result_md = gr.Markdown("*Adjust inputs and click Predict.*")
                shap_plot = gr.Plot(label="SHAP feature impact")

        btn.click(
            predict_single,
            inputs=[gender, senior, partner, dependents, tenure, phone,
                    multilines, internet, security, backup, protection,
                    techsupport, streaming_tv, streaming_mv, contract,
                    paperless, payment, monthly, total],
            outputs=[result_md, shap_plot],
            api_name=False,
        )

    with gr.Tab("📊 Batch Prediction"):
        gr.Markdown(
            "Upload a CSV with the same columns as the raw dataset "
            "(customerID and Churn columns are optional)."
        )
        file_input  = gr.File(label="Upload CSV", file_types=[".csv"])
        batch_btn   = gr.Button("Run Batch", variant="primary")
        batch_table = gr.Dataframe(label="Results")
        batch_summary = gr.Markdown()
        batch_btn.click(predict_batch, inputs=file_input,
                        outputs=[batch_table, batch_summary], api_name=False)

    with gr.Tab("ℹ️ Model Info"):
        gr.Markdown(f"""
## Production Model

| Item | Detail |
|---|---|
| Algorithm | XGBoost (Optuna HPO, 50 trials) |
| Imbalance strategy | `scale_pos_weight` — validated best for recall vs SMOTE/no-balance |
| Decision threshold | **{OPTIMAL_THRESHOLD}** (business-optimal, not default 0.5) |
| Preprocessing | StandardScaler · LabelEncoder · OneHotEncoder via sklearn Pipeline |
| Explainability | SHAP TreeExplainer (per-customer + global) |
| Experiment tracking | MLflow (50 Optuna trials + 4-model comparison) |
| Dataset | IBM Telco Customer Churn — 7,043 customers, 20 features, 26.5% churn rate |

## Test Set Results (1,409 customers)

| Metric | Default threshold (0.5) | Optimal threshold ({OPTIMAL_THRESHOLD}) |
|---|---|---|
| ROC-AUC | **0.8486** | 0.8486 (threshold-independent) |
| Recall (churn) | 0.80 | higher — catches more churners |
| Precision (churn) | 0.52 | lower — more contacts needed |
| F1 | 0.63 | varies |

## Model Comparison (experimental run)

| Model | ROC-AUC | F1 | Notes |
|---|---|---|---|
| Logistic Regression | 0.8407 | 0.6176 | Baseline |
| LightGBM | 0.8264 | 0.5993 | Underperforms on this dataset |
| **XGBoost (Optuna)** | **0.8488** | **0.6303** | Production model |
| Stacking Ensemble | see MLflow | — | XGBoost + LR + LightGBM meta |

## Class Imbalance Comparison (SMOTE experiment)

| Strategy | ROC-AUC | Recall | Verdict |
|---|---|---|---|
| No balancing | 0.8391 | 0.53 | Misses most churners |
| **scale_pos_weight** | **0.8391** | **0.76** | Best recall, chosen |
| SMOTE | 0.8396 | 0.62 | Higher AUC but lower recall |
| SMOTE-Tomek | 0.8378 | 0.61 | Lowest across both |

**Finding:** `scale_pos_weight` gives the best recall (catches more churners) at similar AUC to SMOTE — ideal for a churn use case where false negatives are 5–10× more costly than false positives.

## Business Value (cost-sensitive analysis)

- Avg customer CLV (12 months): ~$780
- Campaign cost per contact: $20
- Assumed retention rate: 30%
- **Optimal threshold** maximises expected profit: *revenue saved - campaign spend*
        """)


if __name__ == "__main__":
    # server_name="0.0.0.0" is required on HF Spaces (container doesn't expose localhost).
    # share=False would raise ValueError in containers — omit it entirely.
    demo.launch(server_name="0.0.0.0", ssr_mode=False)
