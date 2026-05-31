"""
Preprocessing pipeline:
  - Fix TotalCharges (whitespace → NaN → median impute)
  - Encode binary categoricals (Yes/No → 1/0)
  - One-hot encode multi-class categoricals
  - Standard-scale numeric features
  - Build a sklearn Pipeline usable for both training and inference
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    RAW_CSV, TARGET, DROP_COLS,
    NUMERIC_FEATURES, BINARY_FEATURES, MULTI_FEATURES, SENIOR_FEATURE,
    TEST_SIZE, RANDOM_SEED,
)


# ── Custom transformer: binary Yes/No → 1/0 ───────────────────────────────

class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Map Yes→1, No→0, Female→0, Male→1 across selected columns."""

    MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].map(self.MAP).fillna(0).astype(int)
        return X


# ── Load raw data ──────────────────────────────────────────────────────────

def load_raw(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges is read as object due to stray spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    # Drop useless columns
    df = df.drop(columns=DROP_COLS, errors="ignore")

    return df


# ── Split ──────────────────────────────────────────────────────────────────

def split(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_SEED, stratify=y)


# ── Build sklearn ColumnTransformer ───────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("encoder", BinaryEncoder()),
    ])

    multi_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("senior",  "passthrough",  SENIOR_FEATURE),
        ("numeric", numeric_pipe,   NUMERIC_FEATURES),
        ("binary",  binary_pipe,    BINARY_FEATURES),
        ("multi",   multi_pipe,     MULTI_FEATURES),
    ], remainder="drop")

    return preprocessor


# ── Feature names after transform ─────────────────────────────────────────

def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "senior":
            names += cols
        elif name == "numeric":
            names += cols
        elif name == "binary":
            names += cols
        elif name == "multi":
            ohe: OneHotEncoder = trans.named_steps["ohe"]
            names += list(ohe.get_feature_names_out(cols))
    return names


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.config import TRAIN_CSV, TEST_CSV

    df = load_raw()
    print(f"Loaded {len(df)} rows, {df.shape[1]} cols")
    print(f"Churn rate: {df[TARGET].mean():.2%}")

    X_train, X_test, y_train, y_test = split(df)
    X_train.assign(Churn=y_train.values).to_csv(TRAIN_CSV, index=False)
    X_test.assign(Churn=y_test.values).to_csv(TEST_CSV, index=False)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
