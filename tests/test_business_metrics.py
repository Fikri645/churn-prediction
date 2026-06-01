"""
Unit tests for src/business_metrics.py

These tests use synthetic data — no trained model or raw dataset required.
"""
import numpy as np
import pandas as pd
import pytest

from src.business_metrics import (
    CAMPAIGN_COST,
    RETENTION_RATE,
    business_summary,
    expected_profit,
    find_optimal_threshold,
    plot_profit_curve,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_labels(n_pos: int = 80, n_neg: int = 200, seed: int = 42):
    """Synthetic binary labels + probabilities (higher prob for positives)."""
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * n_neg)
    prob_pos = rng.uniform(0.45, 1.0, n_pos)
    prob_neg = rng.uniform(0.0, 0.55, n_neg)
    probs = np.concatenate([prob_pos, prob_neg])
    return y, probs


# ── expected_profit ────────────────────────────────────────────────────────

class TestExpectedProfit:
    def test_returns_float(self):
        y, p = _make_labels()
        assert isinstance(expected_profit(y, p, threshold=0.5), float)

    def test_threshold_one_contacts_nobody(self):
        """At threshold=1.0 nobody is predicted positive → profit = 0."""
        y, p = _make_labels()
        # Cap probs below 1.0 so threshold=1.0 means no contacts
        p = np.clip(p, 0.0, 0.99)
        assert expected_profit(y, p, threshold=1.0) == 0.0

    def test_threshold_zero_contacts_everyone(self):
        """At threshold≈0, everyone is contacted — formula matches manual calc."""
        y, p = _make_labels()
        profit = expected_profit(y, p, threshold=0.01, avg_monthly=65.0)
        clv = 65.0 * 12
        tp_gain = clv * RETENTION_RATE - CAMPAIGN_COST
        fp_cost = CAMPAIGN_COST
        tp = int(y.sum())       # all positives predicted positive
        fp = int((y == 0).sum())
        expected = tp * tp_gain - fp * fp_cost
        assert abs(profit - expected) < 1e-6

    def test_higher_monthly_means_higher_profit(self):
        """More valuable customers → higher retention profit."""
        y, p = _make_labels()
        p_low  = expected_profit(y, p, threshold=0.3, avg_monthly=30.0)
        p_high = expected_profit(y, p, threshold=0.3, avg_monthly=100.0)
        assert p_high > p_low


# ── find_optimal_threshold ─────────────────────────────────────────────────

class TestFindOptimalThreshold:
    def test_returns_required_keys(self):
        y, p = _make_labels()
        result = find_optimal_threshold(y, p)
        for key in ("optimal_threshold", "expected_profit", "thresholds", "profits"):
            assert key in result

    def test_threshold_in_range(self):
        y, p = _make_labels()
        result = find_optimal_threshold(y, p)
        assert 0.0 < result["optimal_threshold"] < 1.0

    def test_optimal_beats_or_ties_default(self):
        """The swept optimum must be >= profit at default 0.5."""
        y, p = _make_labels()
        result = find_optimal_threshold(y, p)
        default = expected_profit(y, p, threshold=0.5)
        assert result["expected_profit"] >= default - 1e-6  # allow float noise

    def test_profits_array_length_matches_thresholds(self):
        y, p = _make_labels()
        result = find_optimal_threshold(y, p, n_thresholds=50)
        assert len(result["profits"]) == len(result["thresholds"]) == 50


# ── plot_profit_curve ──────────────────────────────────────────────────────

class TestPlotProfitCurve:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        y, p = _make_labels()
        result = find_optimal_threshold(y, p)
        fig = plot_profit_curve(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ── business_summary ──────────────────────────────────────────────────────

class TestBusinessSummary:
    def test_keys_present(self):
        y, p = _make_labels()
        s = business_summary(y, p)
        expected_keys = [
            "avg_monthly_charges", "avg_clv", "n_churners_in_test",
            "max_revenue_at_risk_$", "optimal_threshold",
            "max_expected_profit_$", "at_default_0.5",
            "at_optimal_threshold", "profit_improvement_$",
        ]
        for k in expected_keys:
            assert k in s, f"Missing key: {k}"

    def test_n_churners_correct(self):
        y, p = _make_labels(n_pos=80, n_neg=200)
        s = business_summary(y, p)
        assert s["n_churners_in_test"] == 80

    def test_profit_improvement_positive(self):
        y, p = _make_labels()
        s = business_summary(y, p)
        assert s["profit_improvement_$"] >= 0

    def test_uses_monthly_charges_series(self):
        """When a MonthlyCharges Series is supplied, avg should match."""
        y, p = _make_labels(n_pos=50, n_neg=100)
        charges = pd.Series([100.0] * 150)
        s = business_summary(y, p, monthly_charges_series=charges)
        assert abs(s["avg_monthly_charges"] - 100.0) < 1e-6

    def test_at_default_sub_dict_keys(self):
        y, p = _make_labels()
        s = business_summary(y, p)
        for sub in ("at_default_0.5", "at_optimal_threshold"):
            d = s[sub]
            for k in ("tp", "fp", "fn", "tn", "customers_contacted",
                      "revenue_saved_$", "campaign_spend_$", "net_profit_$"):
                assert k in d, f"Missing key {k} in {sub}"

    def test_contacts_plus_no_contact_equals_total(self):
        """tp+fp+tn+fn should equal total test set size."""
        y, p = _make_labels(n_pos=80, n_neg=200)
        s = business_summary(y, p)
        total = len(y)
        for key in ("at_default_0.5", "at_optimal_threshold"):
            d = s[key]
            assert d["tp"] + d["fp"] + d["tn"] + d["fn"] == total
