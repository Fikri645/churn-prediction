"""
Download the IBM Telco Customer Churn dataset.

Sources tried in order:
  1. IBM GitHub (primary)
  2. Mirror (fallback)
  3. Kaggle CLI (requires kaggle.json credentials)

Usage:
    python scripts/download_data.py
"""
# -*- coding: utf-8 -*-
from pathlib import Path
import requests
import sys

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"

SOURCES = [
    ("IBM GitHub",
     "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
     "master/data/Telco-Customer-Churn.csv"),
    ("Mirror (dsrscientist)",
     "https://raw.githubusercontent.com/dsrscientist/dataset1/"
     "master/Telco-Customer-Churn.csv"),
]


def download_from_url(url: str, label: str):
    print(f"Trying {label} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(r.content)
    size_kb = OUT.stat().st_size / 1024
    print(f"Saved -> {OUT}  ({size_kb:.1f} KB)")


def download_from_kaggle():
    print("Trying Kaggle CLI ...")
    import subprocess
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "blastchar/telco-customer-churn",
         "-p", str(OUT.parent), "--unzip"],
        check=True,
    )
    downloaded = OUT.parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if downloaded.exists():
        downloaded.rename(OUT)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    if OUT.exists():
        print(f"Dataset already exists: {OUT}")
        sys.exit(0)

    for label, url in SOURCES:
        try:
            download_from_url(url, label)
            sys.exit(0)
        except Exception as e:
            print(f"  {label} failed: {e}")

    try:
        download_from_kaggle()
        sys.exit(0)
    except Exception as e:
        print(f"  Kaggle failed: {e}")

    print("\nAll sources failed.")
    print("Manual download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("Save as: data/raw/Telco-Customer-Churn.csv")
    sys.exit(1)
