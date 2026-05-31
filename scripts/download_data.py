"""
Download the IBM Telco Customer Churn dataset.

Sources tried in order:
  1. IBM GitHub (primary)
  2. Kaggle CLI (fallback — requires `kaggle.json` credentials)

Usage:
    python scripts/download_data.py
"""
from pathlib import Path
import requests
import sys

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"

IBM_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download_from_ibm():
    print(f"Downloading from IBM GitHub…")
    r = requests.get(IBM_URL, timeout=30)
    r.raise_for_status()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(r.content)
    print(f"Saved → {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


def download_from_kaggle():
    print("Trying Kaggle CLI…")
    import subprocess
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "blastchar/telco-customer-churn",
         "-p", str(OUT.parent), "--unzip"],
        check=True,
    )
    downloaded = OUT.parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if downloaded.exists():
        downloaded.rename(OUT)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    if OUT.exists():
        print(f"Dataset already exists: {OUT}")
        sys.exit(0)

    try:
        download_from_ibm()
    except Exception as e:
        print(f"IBM source failed ({e}), trying Kaggle…")
        try:
            download_from_kaggle()
        except Exception as e2:
            print(f"Both sources failed: {e2}")
            print("Manual download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
            sys.exit(1)
