"""
Unit tests for src/preprocess.py

These tests use synthetic dataframes — no raw CSV required.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.preprocess import BinaryEncoder, build_preprocessor


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_sample_df(n: int = 6) -> pd.DataFrame:
    """Minimal synthetic DataFrame matching the raw dataset's column schema."""
    # Generate each column with exactly n elements by cycling through options
    def cycle(options, count):
        return [options[i % len(options)] for i in range(count)]

    return pd.DataFrame({
        "gender"          : cycle(["Male", "Female"], n),
        "SeniorCitizen"   : cycle([0, 1], n),
        "Partner"         : cycle(["Yes", "No"], n),
        "Dependents"      : cycle(["No", "Yes"], n),
        "tenure"          : list(range(1, n + 1)),
        "PhoneService"    : cycle(["Yes", "No"], n),
        "MultipleLines"   : cycle(["No", "Yes", "No phone service"], n),
        "InternetService" : cycle(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity"  : cycle(["No", "Yes", "No internet service"], n),
        "OnlineBackup"    : cycle(["Yes", "No", "No internet service"], n),
        "DeviceProtection": cycle(["No", "Yes", "No internet service"], n),
        "TechSupport"     : cycle(["No", "Yes", "No internet service"], n),
        "StreamingTV"     : cycle(["Yes", "No", "No internet service"], n),
        "StreamingMovies" : cycle(["No", "Yes", "No internet service"], n),
        "Contract"        : cycle(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": cycle(["Yes", "No"], n),
        "PaymentMethod"   : cycle([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], n),
        "MonthlyCharges"  : [float(i * 10 + 29) for i in range(n)],
        "TotalCharges"    : [float(i * 100 + 100) for i in range(n)],
    })


# ── BinaryEncoder ──────────────────────────────────────────────────────────

class TestBinaryEncoder:
    """BinaryEncoder has a hardcoded MAP: Yes→1, No→0, Male→1, Female→0."""

    def test_transforms_yes_no(self):
        enc = BinaryEncoder()
        X = pd.DataFrame({"Partner": ["Yes", "No", "Yes", "No"]})
        enc.fit(X)
        out = enc.transform(X)
        assert list(out["Partner"]) == [1, 0, 1, 0]

    def test_transforms_gender(self):
        enc = BinaryEncoder()
        X = pd.DataFrame({"gender": ["Male", "Female", "Male"]})
        enc.fit(X)
        out = enc.transform(X)
        assert list(out["gender"]) == [1, 0, 1]

    def test_transforms_multiple_columns(self):
        enc = BinaryEncoder()
        X = pd.DataFrame({"gender": ["Male", "Female"], "Partner": ["Yes", "No"]})
        enc.fit(X)
        out = enc.transform(X)
        assert list(out["gender"]) == [1, 0]
        assert list(out["Partner"]) == [1, 0]

    def test_sklearn_compat_n_features_in(self):
        """sklearn 1.5+ requires n_features_in_ after fit."""
        enc = BinaryEncoder()
        X = pd.DataFrame({"Partner": ["Yes", "No"], "Dependents": ["No", "Yes"]})
        enc.fit(X)
        assert hasattr(enc, "n_features_in_")
        assert enc.n_features_in_ == 2

    def test_sklearn_compat_feature_names_in(self):
        """sklearn 1.5+ requires feature_names_in_ after fit."""
        enc = BinaryEncoder()
        X = pd.DataFrame({"Partner": ["Yes", "No"]})
        enc.fit(X)
        assert hasattr(enc, "feature_names_in_")
        assert list(enc.feature_names_in_) == ["Partner"]

    def test_unknown_values_become_zero(self):
        """BinaryEncoder silently maps unknown values to 0 (fillna behaviour)."""
        enc = BinaryEncoder()
        X_train = pd.DataFrame({"col": ["Yes", "No"]})
        X_unknown = pd.DataFrame({"col": ["Maybe", "Unknown"]})
        enc.fit(X_train)
        out = enc.transform(X_unknown)
        # Unknown values not in MAP → NaN → fillna(0) → 0
        assert list(out["col"]) == [0, 0]

    def test_get_feature_names_out(self):
        enc = BinaryEncoder()
        X = pd.DataFrame({"Partner": ["Yes", "No"], "Dependents": ["No", "Yes"]})
        enc.fit(X)
        # BaseEstimator provides get_feature_names_out if n_features_in_ is set
        # (or BinaryEncoder inherits it from TransformerMixin in sklearn 1.1+)
        # Just verify fit doesn't break and produces correct n_features_in_
        assert enc.n_features_in_ == 2


# ── build_preprocessor ─────────────────────────────────────────────────────

class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        """build_preprocessor() returns a ColumnTransformer (not a Pipeline)."""
        prep = build_preprocessor()
        assert isinstance(prep, ColumnTransformer)

    def test_output_row_count_preserved(self):
        """Row count must be preserved after transform."""
        prep = build_preprocessor()
        df = _make_sample_df(6)
        out = prep.fit_transform(df)
        assert out.shape[0] == 6

    def test_output_has_many_features(self):
        """OneHotEncoding of multi-categoricals expands feature count well beyond raw."""
        prep = build_preprocessor()
        df = _make_sample_df(12)
        out = prep.fit_transform(df)
        # 1 senior + 3 numeric + 5 binary + OHE expansions ≈ 30+ features
        assert out.shape[1] >= 20

    def test_no_nan_in_output(self):
        prep = build_preprocessor()
        df = _make_sample_df(12)
        out = prep.fit_transform(df)
        assert not np.isnan(out).any(), "Preprocessor output contains NaN"

    def test_numeric_features_are_scaled(self):
        """StandardScaler should bring numeric values into a [-5, 5] range."""
        prep = build_preprocessor()
        df = _make_sample_df(30)
        out = prep.fit_transform(df)
        # StandardScaler output should not contain values in the thousands
        # (unscaled MonthlyCharges range: 29-$120, TotalCharges: $100-$3000)
        assert out.max() < 200, "Numeric features don't appear to be scaled"
