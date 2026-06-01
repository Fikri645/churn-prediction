"""
Comparative experiments — run ONCE to find the best model.

Experiments:
  A. SMOTE comparison     : no_balance | scale_pos_weight | smote | smote_tomek
  B. Model comparison     : LogisticRegression | LightGBM | XGBoost | Stacking
  C. Best model evaluation: full business report + optimal threshold

Usage:
    python -m src.experiments
"""
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[warn] lightgbm not installed — skipping LightGBM experiments")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("[warn] imbalanced-learn not installed — skipping SMOTE experiments")

from src.config import (
    MLFLOW_EXPERIMENT, MODEL_PATH, MODELS_DIR,
    RANDOM_SEED, XGB_BASE_PARAMS,
)
from src.preprocess import build_preprocessor, load_raw, split
from src.business_metrics import (
    business_summary, find_optimal_threshold,
    plot_profit_curve, print_business_report, FIGURES_DIR,
)

EXPERIMENT_NAME = f"{MLFLOW_EXPERIMENT}-experiments"


# ── Helpers ────────────────────────────────────────────────────────────────

def _evaluate(pipeline, X_test, y_test) -> dict:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc"      : round(roc_auc_score(y_test, y_prob), 4),
        "f1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
    }


def _log_run(run_name: str, params: dict, metrics: dict, pipeline=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if pipeline is not None:
            mlflow.sklearn.log_model(pipeline, artifact_path="pipeline")
    print(f"  {run_name:35s}  AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}")


# ── Experiment A: SMOTE comparison ────────────────────────────────────────

def run_smote_comparison(X_train, X_test, y_train, y_test):
    if not HAS_IMBLEARN:
        print("[skip] imbalanced-learn not available")
        return

    print("\n[A] SMOTE Comparison")
    print("-" * 60)
    mlflow.set_experiment(EXPERIMENT_NAME)

    preprocessor = build_preprocessor()
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos

    base_xgb_params = {**XGB_BASE_PARAMS,
                       "n_estimators": 400, "max_depth": 4,
                       "learning_rate": 0.05, "subsample": 0.8,
                       "colsample_bytree": 0.8}

    variants = [
        ("no_balance",       None,        {**base_xgb_params}),
        ("scale_pos_weight", None,        {**base_xgb_params, "scale_pos_weight": spw}),
        ("smote",            SMOTE(random_state=RANDOM_SEED), {**base_xgb_params}),
        ("smote_tomek",      SMOTETomek(random_state=RANDOM_SEED), {**base_xgb_params}),
    ]

    results = []
    for name, sampler, xgb_params in variants:
        if sampler is None:
            pipe = Pipeline([
                ("prep",  preprocessor),
                ("model", XGBClassifier(**xgb_params)),
            ])
        else:
            pipe = ImbPipeline([
                ("prep",    preprocessor),
                ("sampler", sampler),
                ("model",   XGBClassifier(**xgb_params)),
            ])
        pipe.fit(X_train, y_train)
        m = _evaluate(pipe, X_test, y_test)
        results.append({"variant": name, **m})
        _log_run(f"smote/{name}", {"balance_strategy": name}, m)

    df = pd.DataFrame(results).set_index("variant")
    print("\n" + df.to_string())
    return df


# ── Experiment B: Model comparison ────────────────────────────────────────

def run_model_comparison(X_train, X_test, y_train, y_test):
    print("\n[B] Model Comparison")
    print("-" * 60)
    mlflow.set_experiment(EXPERIMENT_NAME)

    preprocessor = build_preprocessor()
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos

    # ── Logistic Regression (baseline) ─────────────────────────────────
    lr_pipe = Pipeline([
        ("prep",  build_preprocessor()),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_SEED, C=0.1)),
    ])
    lr_pipe.fit(X_train, y_train)
    m_lr = _evaluate(lr_pipe, X_test, y_test)
    _log_run("model/logistic_regression", {"model": "LogisticRegression"}, m_lr)

    # ── LightGBM ────────────────────────────────────────────────────────
    m_lgbm = {}
    lgbm_pipe = None
    if HAS_LGBM:
        lgbm_pipe = Pipeline([
            ("prep",  build_preprocessor()),
            ("model", LGBMClassifier(
                n_estimators=500, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                scale_pos_weight=spw,
                random_state=RANDOM_SEED, n_jobs=-1,
                verbose=-1)),
        ])
        lgbm_pipe.fit(X_train, y_train)
        m_lgbm = _evaluate(lgbm_pipe, X_test, y_test)
        _log_run("model/lightgbm", {"model": "LightGBM"}, m_lgbm)

    # ── XGBoost (best params from main train) ────────────────────────────
    xgb_pipe = Pipeline([
        ("prep",  build_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=600, max_depth=3, learning_rate=0.016,
            subsample=0.59, colsample_bytree=0.70,
            reg_alpha=7.67, reg_lambda=0.038,
            min_child_weight=3, scale_pos_weight=spw,
            **XGB_BASE_PARAMS)),
    ])
    xgb_pipe.fit(X_train, y_train)
    m_xgb = _evaluate(xgb_pipe, X_test, y_test)
    _log_run("model/xgboost", {"model": "XGBoost"}, m_xgb)

    # ── Stacking Ensemble ────────────────────────────────────────────────
    estimators = [("lr",  LogisticRegression(max_iter=1000, class_weight="balanced",
                                             random_state=RANDOM_SEED, C=0.1)),
                  ("xgb", XGBClassifier(
                      n_estimators=300, max_depth=3, learning_rate=0.05,
                      scale_pos_weight=spw, **XGB_BASE_PARAMS))]
    if HAS_LGBM:
        estimators.append(("lgbm", LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            scale_pos_weight=spw, random_state=RANDOM_SEED,
            n_jobs=1, verbose=-1)))  # n_jobs=1 inside stacking to avoid OOM

    stacking = Pipeline([
        ("prep", build_preprocessor()),
        ("model", StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            cv=3, passthrough=False,
            n_jobs=1,   # sequential — avoids joblib OOM inside StackingClassifier
        )),
    ])
    stacking.fit(X_train, y_train)
    m_stack = _evaluate(stacking, X_test, y_test)
    _log_run("model/stacking_ensemble", {"model": "Stacking(LR+XGB+LGBM)"}, m_stack)

    # ── Summary table ────────────────────────────────────────────────────
    rows = [
        {"model": "LogisticRegression (baseline)", **m_lr},
        {"model": "XGBoost (Optuna-tuned)",        **m_xgb},
        {"model": "Stacking Ensemble",             **m_stack},
    ]
    if m_lgbm:
        rows.insert(2, {"model": "LightGBM",  **m_lgbm})

    df = pd.DataFrame(rows).set_index("model")
    print("\n" + df.to_string())

    # ── Return best pipeline (by AUC) ────────────────────────────────────
    candidates = [("XGBoost", m_xgb["auc"], xgb_pipe),
                  ("Stacking", m_stack["auc"], stacking)]
    if m_lgbm:
        candidates.append(("LightGBM", m_lgbm["auc"], lgbm_pipe))

    best_name, best_auc, best_pipe = max(candidates, key=lambda x: x[1])
    print(f"\n  → Best model: {best_name}  (AUC={best_auc:.4f})")
    return df, best_name, best_pipe


# ── Experiment C: Best model — business evaluation + save ─────────────────

def run_business_evaluation(best_name, best_pipe,
                             X_test, y_test, df_raw_test):
    print("\n[C] Business Value Report")
    print("-" * 60)

    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    # compute business summary using actual MonthlyCharges from test set
    monthly = df_raw_test["MonthlyCharges"] if "MonthlyCharges" in df_raw_test.columns \
              else None
    summary = business_summary(y_test, y_prob, monthly)
    print_business_report(summary)

    # plot profit curve
    opt = find_optimal_threshold(y_test, y_prob,
          float(monthly.mean()) if monthly is not None else 65.0)
    fig = plot_profit_curve(opt)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = FIGURES_DIR / "profit_curve.png"
    fig.savefig(curve_path)
    print(f"  Saved: {curve_path}")

    # save metadata (optimal threshold + business summary)
    meta = {
        "best_model"      : best_name,
        "optimal_threshold": summary["optimal_threshold"],
        "business_summary": summary,
    }
    meta_path = MODELS_DIR / "model_meta.json"
    MODELS_DIR.mkdir(exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Saved: {meta_path}")

    # overwrite production model if stacking beats original XGBoost
    orig_pipe = joblib.load(MODEL_PATH)
    orig_auc  = roc_auc_score(y_test, orig_pipe.predict_proba(X_test)[:, 1])
    best_auc  = roc_auc_score(y_test, y_prob)

    if best_auc > orig_auc + 0.001:  # meaningful improvement
        joblib.dump(best_pipe, MODEL_PATH)
        print(f"\n  Production model UPDATED: {best_name} "
              f"AUC={best_auc:.4f} > XGBoost AUC={orig_auc:.4f}")
    else:
        print(f"\n  Production model KEPT: XGBoost AUC={orig_auc:.4f} "
              f"(best={best_auc:.4f}, delta < 0.001)")

    return summary


# ── Main ──────────────────────────────────────────────────────────────────

def run_all():
    df = load_raw()
    X_train, X_test, y_train, y_test = split(df)

    # keep raw test rows for MonthlyCharges (business metric)
    df_raw_test = df.loc[y_test.index] if hasattr(y_test, "index") else df.iloc[-len(y_test):]

    mlflow.set_experiment(EXPERIMENT_NAME)

    df_smote = run_smote_comparison(X_train, X_test, y_train, y_test)
    df_models, best_name, best_pipe = run_model_comparison(X_train, X_test, y_train, y_test)
    summary = run_business_evaluation(best_name, best_pipe, X_test, y_test, df_raw_test)

    return df_smote, df_models, summary


if __name__ == "__main__":
    run_all()
