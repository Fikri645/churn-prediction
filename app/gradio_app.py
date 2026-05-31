"""
Gradio UI — three tabs:
  1. Single Prediction  — form input → churn probability + SHAP bar
  2. Batch Upload       — CSV upload → table of predictions + summary
  3. Model Info         — metrics, dataset description, SHAP summary plot

Deploy to HF Spaces:
    gradio deploy
"""
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import gradio as gr

from src.config import MODEL_PATH, SHAP_VALUES
from src.preprocess import build_preprocessor, get_feature_names, load_raw, split

# ── Load artifacts ─────────────────────────────────────────────────────────
pipeline = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps["prep"]
model        = pipeline.named_steps["model"]
feature_names = get_feature_names(preprocessor)
explainer     = shap.TreeExplainer(model)


def _predict_df(df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(df)[:, 1]


def _risk(prob: float) -> str:
    if prob >= 0.65: return "🔴 High"
    if prob >= 0.40: return "🟡 Medium"
    return "🟢 Low"


# ── SHAP local bar chart ───────────────────────────────────────────────────

def shap_bar_for_row(df_row: pd.DataFrame) -> plt.Figure:
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
    result = f"## Churn Probability: **{prob:.1%}**\n\nRisk level: {label}"
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

YES_NO       = ["Yes", "No"]
YN_NOPHONE   = ["Yes", "No", "No phone service"]
YN_NOINET    = ["Yes", "No", "No internet service"]

# Disable SSR — Gradio 5.x SSR causes TypeError in api_info when gr.Checkbox
# is present (additionalProperties:false schema bug). Safe to disable on HF Spaces.
demo = gr.Blocks(title="Churn Predictor", theme=gr.themes.Soft())
with demo:
    gr.Markdown("# 📉 Customer Churn Predictor\nXGBoost · SHAP · IBM Telco dataset")

    with gr.Tab("🔍 Single Prediction"):
        with gr.Row():

            # ── LEFT: form (3 compact columns) ────────────────────────────
            with gr.Column(scale=3):
                with gr.Row():
                    # Col A: Demographics + Phone
                    with gr.Column():
                        gender       = gr.Dropdown(["Male","Female"],  label="Gender",          value="Male")
                        senior       = gr.Dropdown(["No", "Yes"], value="No", label="Senior Citizen")
                        partner      = gr.Dropdown(YES_NO, label="Partner",          value="No")
                        dependents   = gr.Dropdown(YES_NO, label="Dependents",       value="No")
                        phone        = gr.Dropdown(YES_NO, label="Phone Service",    value="Yes")
                        multilines   = gr.Dropdown(YN_NOPHONE, label="Multiple Lines", value="No")

                    # Col B: Account & Billing
                    with gr.Column():
                        tenure       = gr.Slider(0, 72, value=12, step=1, label="Tenure (months)")
                        contract     = gr.Dropdown(
                            ["Month-to-month","One year","Two year"],
                            label="Contract", value="Month-to-month")
                        paperless    = gr.Dropdown(YES_NO, label="Paperless Billing", value="Yes")
                        payment      = gr.Dropdown([
                            "Electronic check","Mailed check",
                            "Bank transfer (automatic)","Credit card (automatic)"],
                            label="Payment Method", value="Electronic check")
                        monthly      = gr.Number(label="Monthly Charges ($)", value=65.0)
                        total        = gr.Number(label="Total Charges ($)",   value=780.0)

                    # Col C: Internet Services
                    with gr.Column():
                        internet     = gr.Dropdown(["DSL","Fiber optic","No"],
                                                   label="Internet Service", value="Fiber optic")
                        security     = gr.Dropdown(YN_NOINET, label="Online Security",   value="No")
                        backup       = gr.Dropdown(YN_NOINET, label="Online Backup",     value="No")
                        protection   = gr.Dropdown(YN_NOINET, label="Device Protection", value="No")
                        techsupport  = gr.Dropdown(YN_NOINET, label="Tech Support",      value="No")
                        streaming_tv = gr.Dropdown(YN_NOINET, label="Streaming TV",      value="No")
                        streaming_mv = gr.Dropdown(YN_NOINET, label="Streaming Movies",  value="No")

                btn = gr.Button("🔮 Predict Churn", variant="primary", size="lg")

            # ── RIGHT: output (appears beside the form, not below) ────────
            with gr.Column(scale=2):
                result_md = gr.Markdown("*Fill in the form and click Predict.*")
                shap_plot = gr.Plot(label="SHAP — feature impact for this customer")

        btn.click(
            predict_single,
            inputs=[gender, senior, partner, dependents, tenure, phone,
                    multilines, internet, security, backup, protection,
                    techsupport, streaming_tv, streaming_mv, contract,
                    paperless, payment, monthly, total],
            outputs=[result_md, shap_plot],
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
                        outputs=[batch_table, batch_summary])

    with gr.Tab("ℹ️ Model Info"):
        gr.Markdown("""
## Model

| Item | Detail |
|---|---|
| Algorithm | XGBoost (gradient-boosted trees) |
| HPO | Optuna TPE, 50 trials |
| Preprocessing | StandardScaler · LabelEncoder · OneHotEncoder via sklearn Pipeline |
| Explainability | SHAP TreeExplainer |
| Dataset | IBM Telco Customer Churn (7,043 customers, 20 features) |
| Target | Churn (Yes/No) — ~26% positive rate |

## Evaluation (test set, 20%)

| Metric | Score |
|---|---|
| ROC-AUC | see MLflow |
| F1 (Churn class) | see MLflow |

## Key Findings (SHAP)

Top drivers of churn:
1. **Month-to-month contract** — highest risk; two-year contracts drastically reduce churn.
2. **Fiber optic internet** — higher than DSL; may reflect price sensitivity.
3. **Short tenure** — new customers churn more; first 12 months are critical.
4. **No online security / tech support** — add-on absence increases risk.
5. **High monthly charges** — especially without a long-term contract.
        """)


if __name__ == "__main__":
    demo.launch(share=False, ssr_mode=False)
else:
    # HF Spaces entrypoint — disable SSR to avoid Gradio 5.x api_info bug
    demo.launch(ssr_mode=False)
