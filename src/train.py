"""
Train XGBoost with Optuna HPO.
Logs every trial + final model to MLflow.
Saves the final sklearn Pipeline to models/.

Usage:
    python -m src.train
"""
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    MODEL_PATH, MODELS_DIR, MLFLOW_EXPERIMENT, MLFLOW_RUN_NAME,
    OPTUNA_N_TRIALS, RANDOM_SEED, XGB_BASE_PARAMS,
)
from src.preprocess import build_preprocessor, load_raw, split

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_pipeline(params: dict) -> Pipeline:
    preprocessor = build_preprocessor()
    model = XGBClassifier(**{**XGB_BASE_PARAMS, **params})
    return Pipeline([("prep", preprocessor), ("model", model)])


def evaluate(pipeline, X, y) -> dict:
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc":       roc_auc_score(y, y_prob),
        "f1":        f1_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred, zero_division=0),
    }


def train(n_trials: int = OPTUNA_N_TRIALS):
    df       = load_raw()
    X_train, X_test, y_train, y_test = split(df)

    # Class imbalance weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = neg / pos

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    def objective(trial):
        params = {
            "n_estimators"       : trial.suggest_int("n_estimators",   100, 800),
            "max_depth"          : trial.suggest_int("max_depth",         3,   9),
            "learning_rate"      : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"          : trial.suggest_float("subsample",      0.5,  1.0),
            "colsample_bytree"   : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"          : trial.suggest_float("reg_alpha",      1e-8, 10.0, log=True),
            "reg_lambda"         : trial.suggest_float("reg_lambda",     1e-8, 10.0, log=True),
            "min_child_weight"   : trial.suggest_int("min_child_weight",    1,  10),
            "scale_pos_weight"   : scale_pos,
        }
        pipe = make_pipeline(params)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)

        # Log each trial as a child run
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

        return metrics["auc"]

    print(f"Running {n_trials} Optuna trials…")
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as parent_run:
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = {**study.best_params, "scale_pos_weight": scale_pos}
        print(f"\nBest AUC: {study.best_value:.4f}")
        print(f"Best params: {best_params}")

        # Retrain on full training set with best params
        final_pipe = make_pipeline(best_params)
        final_pipe.fit(X_train, y_train)
        metrics   = evaluate(final_pipe, X_test, y_test)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(final_pipe, artifact_path="pipeline",
                                 registered_model_name="churn-xgb")

        print("\n── Final Test Metrics ──────────────────")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")

        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(final_pipe, MODEL_PATH)
        print(f"\nSaved pipeline → {MODEL_PATH}")
        print(f"MLflow run ID  : {parent_run.info.run_id}")

    return final_pipe, metrics


if __name__ == "__main__":
    train()
