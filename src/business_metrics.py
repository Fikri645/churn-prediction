"""
Business-driven evaluation for churn prediction.

Cost structure (IBM Telco context):
  True Positive  → we contact churner, retain them
                   Gain = CLV × retention_rate - campaign_cost
  False Positive → we contact non-churner unnecessarily
                   Cost = campaign_cost
  False Negative → churner leaves undetected
                   Cost = CLV (revenue lost)
  True Negative  → correct, no action needed
                   Cost = 0

CLV (simplified) = MonthlyCharges × 12
Campaign cost    = $20 per contacted customer (industry estimate)
Retention rate   = 30% (industry average for proactive outreach)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.config import FIGURES_DIR

# ── Business parameters ────────────────────────────────────────────────────
CLV_MONTHS       = 12          # months of revenue treated as CLV
CAMPAIGN_COST    = 20.0        # $ per customer contacted
RETENTION_RATE   = 0.30        # fraction of contacted churners retained

AVG_MONTHLY      = 65.0        # fallback if dataset unavailable (dataset avg ≈ $65)


def _clv(avg_monthly: float) -> float:
    return avg_monthly * CLV_MONTHS


def expected_profit(y_true, y_prob, threshold: float,
                    avg_monthly: float = AVG_MONTHLY) -> float:
    """Expected profit of the retention campaign at a given decision threshold."""
    clv = _clv(avg_monthly)
    tp_gain = clv * RETENTION_RATE - CAMPAIGN_COST   # per true positive
    fp_cost = CAMPAIGN_COST                           # per false positive

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return tp * tp_gain - fp * fp_cost


def find_optimal_threshold(y_true, y_prob,
                           avg_monthly: float = AVG_MONTHLY,
                           n_thresholds: int = 200) -> dict:
    """Sweep thresholds and return the one maximising expected profit."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    profits    = [expected_profit(y_true, y_prob, t, avg_monthly) for t in thresholds]

    best_idx  = int(np.argmax(profits))
    best_t    = float(thresholds[best_idx])
    best_p    = float(profits[best_idx])

    return {
        "optimal_threshold": round(best_t, 3),
        "expected_profit"  : round(best_p, 2),
        "thresholds"       : thresholds,
        "profits"          : np.array(profits),
    }


def plot_profit_curve(result: dict, baseline_threshold: float = 0.5,
                      avg_monthly: float = AVG_MONTHLY) -> plt.Figure:
    """Expected Profit Curve with optimal and baseline thresholds marked."""
    ts = result["thresholds"]
    ps = result["profits"]
    opt_t = result["optimal_threshold"]
    opt_p = result["expected_profit"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, ps, color="#4C72B0", linewidth=2, label="Expected profit")
    ax.axvline(opt_t, color="#DD8452", linestyle="--", linewidth=1.5,
               label=f"Optimal threshold = {opt_t}  (${opt_p:,.0f})")
    ax.axvline(baseline_threshold, color="#55A868", linestyle=":",
               linewidth=1.5, label=f"Default threshold = {baseline_threshold}")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected profit ($)")
    ax.set_title("Expected Profit Curve\n(campaign cost $20/contact, 30% retention rate)")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def business_summary(y_true, y_prob,
                     monthly_charges_series: pd.Series = None,
                     threshold: float = None) -> dict:
    """
    Full business value report.
    Returns a dict with all key business metrics.
    """
    avg_monthly = float(monthly_charges_series.mean()) if monthly_charges_series is not None \
                  else AVG_MONTHLY
    clv = _clv(avg_monthly)

    # Find optimal threshold
    opt = find_optimal_threshold(y_true, y_prob, avg_monthly)
    opt_t = threshold if threshold is not None else opt["optimal_threshold"]

    y_pred_opt  = (y_prob >= opt_t).astype(int)
    y_pred_def  = (y_prob >= 0.5).astype(int)

    def _metrics(y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        retained = tp * RETENTION_RATE
        revenue_saved = retained * clv
        campaign_spend = (tp + fp) * CAMPAIGN_COST
        net = revenue_saved - campaign_spend
        return {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "customers_contacted"  : int(tp + fp),
            "predicted_retained"   : round(retained, 1),
            "revenue_saved_$"      : round(revenue_saved, 0),
            "campaign_spend_$"     : round(campaign_spend, 0),
            "net_profit_$"         : round(net, 0),
        }

    n_churners = int(y_true.sum())
    max_revenue_at_risk = n_churners * clv

    return {
        "avg_monthly_charges"     : round(avg_monthly, 2),
        "avg_clv"                 : round(clv, 2),
        "n_churners_in_test"      : n_churners,
        "max_revenue_at_risk_$"   : round(max_revenue_at_risk, 0),
        "optimal_threshold"       : opt["optimal_threshold"],
        "max_expected_profit_$"   : opt["expected_profit"],
        "at_default_0.5"          : _metrics(y_pred_def),
        "at_optimal_threshold"    : _metrics(y_pred_opt),
        "profit_improvement_$"    : round(
            opt["expected_profit"]
            - expected_profit(y_true, y_prob, 0.5, avg_monthly), 2
        ),
    }


def print_business_report(summary: dict) -> None:
    print("\n" + "=" * 55)
    print("  BUSINESS VALUE REPORT")
    print("=" * 55)
    print(f"  Avg monthly charges : ${summary['avg_monthly_charges']}")
    print(f"  Avg CLV (12 mo)     : ${summary['avg_clv']:,.0f}")
    print(f"  Churners in test set: {summary['n_churners_in_test']}")
    print(f"  Max revenue at risk : ${summary['max_revenue_at_risk_$']:,.0f}")
    print()
    print(f"  Optimal threshold   : {summary['optimal_threshold']}")
    print(f"  Max expected profit : ${summary['max_expected_profit_$']:,.0f}")
    print(f"  Profit vs default   : +${summary['profit_improvement_$']:,.0f}")
    print()

    for label, key in [("Default (0.50)", "at_default_0.5"),
                       (f"Optimal ({summary['optimal_threshold']})",
                        "at_optimal_threshold")]:
        m = summary[key]
        print(f"  [{label}]")
        print(f"    Contacts sent       : {m['customers_contacted']}")
        print(f"    Customers retained  : {m['predicted_retained']:.0f}")
        print(f"    Revenue saved       : ${m['revenue_saved_$']:,.0f}")
        print(f"    Campaign spend      : ${m['campaign_spend_$']:,.0f}")
        print(f"    Net profit          : ${m['net_profit_$']:,.0f}")
        print()
    print("=" * 55 + "\n")
