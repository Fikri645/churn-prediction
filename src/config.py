"""
Central config — paths, feature lists, model hyperparameters.
Import this everywhere instead of scattering magic strings.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV      = DATA_RAW  / "Telco-Customer-Churn.csv"
TRAIN_CSV    = DATA_PROC / "train.csv"
TEST_CSV     = DATA_PROC / "test.csv"
MODEL_PATH   = MODELS_DIR / "xgb_pipeline.joblib"
SHAP_VALUES  = MODELS_DIR / "shap_values.npy"

# ── Target ─────────────────────────────────────────────────────────────────
TARGET = "Churn"

# ── Feature groups ─────────────────────────────────────────────────────────
DROP_COLS = ["customerID"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling",
]

MULTI_FEATURES = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]

SENIOR_FEATURE = ["SeniorCitizen"]  # already 0/1 in raw data

ALL_FEATURES = SENIOR_FEATURE + NUMERIC_FEATURES + BINARY_FEATURES + MULTI_FEATURES

# ── Train / test split ─────────────────────────────────────────────────────
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "churn-prediction"
MLFLOW_RUN_NAME   = "xgb-optuna"

# ── Model defaults (overridden by Optuna) ──────────────────────────────────
XGB_BASE_PARAMS = {
    "eval_metric"    : "auc",
    "use_label_encoder": False,
    "random_state"   : RANDOM_SEED,
    "n_jobs"         : -1,
}

OPTUNA_N_TRIALS = 50
