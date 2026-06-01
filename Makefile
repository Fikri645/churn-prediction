# Churn Prediction — common development commands
# Requires: pip install -r requirements-dev.txt
# Windows users: run inside WSL or Git Bash

.PHONY: install data train evaluate experiments test lint api ui docker-up clean

## Install full development dependencies
install:
	pip install -r requirements-dev.txt

## Download the IBM Telco dataset from Kaggle
data:
	python scripts/download_data.py

## Train XGBoost with Optuna HPO (50 trials) — logs to MLflow
train:
	python -m src.train

## Evaluate production model — confusion matrix, ROC, SHAP, Profit Curve
evaluate:
	python -m src.evaluate

## Run model comparison experiments — SMOTE, LightGBM, Stacking (logs to MLflow)
experiments:
	python -m src.experiments

## Run all steps in order (train → evaluate → experiments)
all: train evaluate experiments

## Run unit tests
test:
	pytest tests/ -v --tb=short

## Run tests with coverage report
test-cov:
	pytest tests/ -v --tb=short --cov=src --cov=api --cov-report=term-missing

## Lint with ruff
lint:
	ruff check src/ tests/ api/ app/

## Start FastAPI server locally
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## Start Gradio UI locally
ui:
	python app/gradio_app.py

## Start FastAPI + MLflow UI via Docker Compose
docker-up:
	docker compose up --build

## Generate Evidently drift report
drift:
	python -m monitoring.drift_report

## Remove generated artifacts (keep models and data)
clean:
	find reports/figures/ -name "*.png" -delete 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
