"""
Unit tests for src/preprocess.py

These tests use synthetic dataframes — no raw CSV required.
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocess import BinaryEncoder, build_preprocessor


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_sample_df(n: int = 5) -> pd.DataFrame:
    """Minimal synthetic DataFrame matching the raw dataset's column schema."""
    return pd.DataFrame({
        "gender"          : ["Male", "Female"] * (n // 2) + ["Male"] * (n % 2),
        "SeniorCitizen"   : [0, 1] * (n // 2) + [0] * (n % 2),
        "Partner"         : ["Yes", "No"] * (n // 2) + ["Yes"] * (n % 2),
        "Dependents"      : ["No"] * n,
        "tenure"          : list(range(1, n + 1)),
        "PhoneService"    : ["Yes"] * n,
        "MultipleLines"   : ["No", "Yes", "No phone service"] * (n // 3 + 1),
        "InternetService" : ["DSL", "Fiber optic", "No"] * (n // 3 + 1),
        "OnlineSecurity"  : ["No", "Yes", "No internet service"] * (n // 3 + 1),
        "OnlineBackup"    : ["Yes", "No", "No internet service"] * (n // 3 + 1),
        "DeviceProtection": ["No", "Yes", "No internet service"] * (n // 3 + 1),
        "TechSupport"     : ["No"] * n,
        "StreamingTV"     : ["Yes", "No"] * (n // 2) + ["No"] * (n % 2),
        "StreamingMovies" : ["No"] * n,
        "Contract"        : ["Month-to-month", "One year", "Two year"] * (n // 3 + 1),
        "PaperlessBilling": ["Yes", "No"] * (n // 2) + ["Yes"] * (n % 2),
        "PaymentMethod"   : [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ] * (n // 4 + 1),
        "MonthlyCharges"  : [float(i * 10 + 29) for i in range(n)],
        "TotalCharges"    : [float(i * 100 + 100) for i in range(n)],
    }).head(n)


# ── BinaryEncoder ──────────────────────────────────────────────────────────

class TestBinaryEncoder:
    def test_fit_transform_yes_no(self):
        enc = BinaryEncoder(mapping={"Yes": 1, "No": 0})
        X = pd.DataFrame({"col": ["Yes", "No", "Yes", "No"]})
        enc.fit(X)
        out = enc.transform(X)
        assert out.tolist() == [[1], [0], [1], [0]]

    def test_fit_transform_gender(self):
        enc = BinaryEncoder(mapping={"Male": 1, "Female": 0})
        X = pd.DataFrame({"gender": ["Male", "Female", "Male"]})
        enc.fit(X)
        out = enc.transform(X)
        assert out.tolist() == [[1], [0], [1]]

    def test_sklearn_compat_attributes(self):
        """sklearn 1.8 requires n_features_in_ and feature_names_in_ after fit."""
        enc = BinaryEncoder(mapping={"Yes": 1, "No": 0})
        X = pd.DataFrame({"col": ["Yes", "No"]})
        enc.fit(X)
        assert hasattr(enc, "n_features_in_")
        assert hasattr(enc, "feature_names_in_")
        assert enc.n_features_in_ == 1

    def test_get_feature_names_out(self):
        enc = BinaryEncoder(mapping={"Yes": 1, "No": 0})
        X = pd.DataFrame({"col": ["Yes", "No"]})
        enc.fit(X)
        names = enc.get_feature_names_out()
        assert len(names) == 1

    def test_raises_on_unknown_value(self):
        """Unknown values should raise KeyError (strict mapping)."""
        enc = BinaryEncoder(mapping={"Yes": 1, "No": 0})
        X = pd.DataFrame({"col": ["Yes", "No"]})
        enc.fit(X)
        X_bad = pd.DataFrame({"col": ["Maybe"]})
        with pytest.raises(Exception):
            enc.transform(X_bad)


# ── build_preprocessor ─────────────────────────────────────────────────────

class TestBuildPreprocessor:
    def test_returns_pipeline(self):
        from sklearn.pipeline import Pipeline
        prep = build_preprocessor()
        assert isinstance(prep, Pipeline)

    def test_output_shape_rows(self):
        """Row count must be preserved after transform."""
        prep = build_preprocessor()
        df = _make_sample_df(6)
        out = prep.fit_transform(df)
        assert out.shape[0] == 6

    def test_output_has_many_features(self):
        """OneHotEncoding of multi-categoricals should expand feature count."""
        prep = build_preprocessor()
        df = _make_sample_df(6)
        out = prep.fit_transform(df)
        assert out.shape[1] >= 20  # at minimum all raw + OHE expansions

    def test_no_nan_in_output(self):
        prep = build_preprocessor()
        df = _make_sample_df(6)
        out = prep.fit_transform(df)
        assert not np.isnan(out).any(), "Preprocessor output contains NaN"

    def test_numeric_features_are_scaled(self):
        """StandardScaler should produce values mostly in [-3, 3]."""
        prep = build_preprocessor()
        df = _make_sample_df(20)
        out = prep.fit_transform(df)
        # Check that range is not in the thousands (unscaled MonthlyCharges would be)
        assert out.max() < 100, "Numeric features don't appear to be scaled"
