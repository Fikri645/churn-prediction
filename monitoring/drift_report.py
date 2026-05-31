"""
Data drift monitoring with Evidently.

Generates an HTML report comparing the training reference distribution
against a new "production" batch. Run this periodically on new inference data.

Usage:
    python -m monitoring.drift_report --new-data path/to/new_batch.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.report import Report

from src.config import TARGET, TRAIN_CSV
from src.preprocess import load_raw, split

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def generate_drift_report(new_data_path: str | None = None, output_name: str = "drift_report"):
    REPORT_DIR.mkdir(exist_ok=True)

    # Reference = training split
    df = load_raw()
    X_train, X_test, y_train, y_test = split(df)
    reference = X_train.copy()
    reference[TARGET] = y_train.values

    # Current = new data or test split (for demo)
    if new_data_path:
        current = pd.read_csv(new_data_path)
    else:
        print("No new data provided — using test split as current batch (demo mode).")
        current = X_test.copy()
        current[TARGET] = y_test.values

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    out_path = REPORT_DIR / f"{output_name}.html"
    report.save_html(str(out_path))
    print(f"Drift report saved → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-data", type=str, default=None,
                        help="Path to new CSV data (optional; uses test split if omitted)")
    parser.add_argument("--output", type=str, default="drift_report")
    args = parser.parse_args()
    generate_drift_report(args.new_data, args.output)
