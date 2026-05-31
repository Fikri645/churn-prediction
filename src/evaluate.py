"""
Post-training evaluation:
  - Confusion matrix
  - ROC curve
  - SHAP global + local explanation
  - Save plots to reports/figures/

Usage:
    python -m src.evaluate
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    classification_report, roc_auc_score,
)

from src.config import (
    FIGURES_DIR, MODEL_PATH, SHAP_VALUES, TARGET,
)
from src.preprocess import build_preprocessor, get_feature_names, load_raw, split

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def load_pipeline():
    return joblib.load(MODEL_PATH)


def plot_confusion_matrix(pipeline, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test,
        display_labels=["No Churn", "Churn"],
        colorbar=False, ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_roc_curve(pipeline, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    path = FIGURES_DIR / "roc_curve.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def compute_shap(pipeline, X_train):
    """
    Compute SHAP values on the transformed training set.
    Uses TreeExplainer (fast, exact for XGBoost).
    """
    preprocessor = pipeline.named_steps["prep"]
    model        = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_train)
    feature_names = get_feature_names(preprocessor)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    np.save(SHAP_VALUES, shap_values)
    print(f"Saved SHAP values → {SHAP_VALUES}")

    return shap_values, X_transformed, feature_names, explainer


def plot_shap_summary(shap_values, X_transformed, feature_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        plot_type="dot", show=False,
    )
    plt.tight_layout()
    path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_shap_bar(shap_values, feature_names):
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
    df = df.nlargest(15, "importance")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df, y="feature", x="importance", hue="feature",
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Top 15 Features — Mean |SHAP value|")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("")
    fig.tight_layout()
    path = FIGURES_DIR / "shap_bar.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def run_evaluation():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline()
    df       = load_raw()
    X_train, X_test, y_train, y_test = split(df)

    # ── Classification report ──────────────────────────────────────────────
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_test, y_prob)

    print("── Classification Report ──────────────────────")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {auc:.4f}")

    plot_confusion_matrix(pipeline, X_test, y_test)
    plot_roc_curve(pipeline, X_test, y_test)

    # ── SHAP ───────────────────────────────────────────────────────────────
    print("\nComputing SHAP values (may take ~30 s)…")
    shap_values, X_transformed, feature_names, _ = compute_shap(pipeline, X_train)
    plot_shap_summary(shap_values, X_transformed, feature_names)
    plot_shap_bar(shap_values, feature_names)

    print("\nAll evaluation artifacts saved to reports/figures/")


if __name__ == "__main__":
    run_evaluation()
