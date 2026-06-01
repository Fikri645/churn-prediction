"""
Unit tests for api/schemas.py

Tests Pydantic validation rules without starting the FastAPI server.
"""
import pytest
from pydantic import ValidationError

from api.schemas import BatchPredictionResponse, CustomerFeatures, PredictionResponse


# ── Fixtures ───────────────────────────────────────────────────────────────

def _valid_customer(**overrides) -> dict:
    base = {
        "gender"          : "Female",
        "SeniorCitizen"   : 0,
        "Partner"         : "Yes",
        "Dependents"      : "No",
        "tenure"          : 12,
        "PhoneService"    : "Yes",
        "MultipleLines"   : "No",
        "InternetService" : "Fiber optic",
        "OnlineSecurity"  : "No",
        "OnlineBackup"    : "Yes",
        "DeviceProtection": "No",
        "TechSupport"     : "No",
        "StreamingTV"     : "Yes",
        "StreamingMovies" : "No",
        "Contract"        : "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod"   : "Electronic check",
        "MonthlyCharges"  : 79.85,
        "TotalCharges"    : 958.2,
    }
    base.update(overrides)
    return base


# ── CustomerFeatures ───────────────────────────────────────────────────────

class TestCustomerFeatures:
    def test_valid_customer_parses(self):
        c = CustomerFeatures(**_valid_customer())
        assert c.tenure == 12
        assert c.MonthlyCharges == pytest.approx(79.85)

    def test_invalid_gender_fails(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**_valid_customer(gender="Other"))

    def test_negative_tenure_fails(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**_valid_customer(tenure=-1))

    def test_tenure_above_72_fails(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**_valid_customer(tenure=73))

    def test_negative_monthly_charges_fails(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**_valid_customer(MonthlyCharges=-5.0))

    def test_negative_total_charges_fails(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**_valid_customer(TotalCharges=-1.0))

    def test_invalid_contract_type_fails(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**_valid_customer(Contract="Weekly"))

    def test_invalid_internet_service_fails(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**_valid_customer(InternetService="Cable"))

    def test_all_literal_fields_accept_valid_values(self):
        """Spot-check a few Literal fields."""
        # Internet service options
        for val in ["DSL", "Fiber optic", "No"]:
            c = CustomerFeatures(**_valid_customer(InternetService=val))
            assert c.InternetService == val

        # Contract options
        for val in ["Month-to-month", "One year", "Two year"]:
            c = CustomerFeatures(**_valid_customer(Contract=val))
            assert c.Contract == val

    def test_senior_citizen_only_0_or_1(self):
        CustomerFeatures(**_valid_customer(SeniorCitizen=0))
        CustomerFeatures(**_valid_customer(SeniorCitizen=1))
        with pytest.raises(ValidationError):
            CustomerFeatures(**_valid_customer(SeniorCitizen=2))

    def test_tenure_zero_is_valid(self):
        c = CustomerFeatures(**_valid_customer(tenure=0))
        assert c.tenure == 0

    def test_tenure_72_is_valid(self):
        c = CustomerFeatures(**_valid_customer(tenure=72))
        assert c.tenure == 72


# ── PredictionResponse ─────────────────────────────────────────────────────

class TestPredictionResponse:
    def test_valid_response(self):
        r = PredictionResponse(
            churn_probability=0.75,
            churn_prediction=True,
            risk_label="High",
        )
        assert r.churn_probability == pytest.approx(0.75)
        assert r.churn_prediction is True

    def test_invalid_risk_label_fails(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                churn_probability=0.5,
                churn_prediction=True,
                risk_label="Critical",
            )


# ── BatchPredictionResponse ────────────────────────────────────────────────

class TestBatchPredictionResponse:
    def test_valid_batch_response(self):
        preds = [
            PredictionResponse(churn_probability=0.8, churn_prediction=True, risk_label="High"),
            PredictionResponse(churn_probability=0.2, churn_prediction=False, risk_label="Low"),
        ]
        r = BatchPredictionResponse(predictions=preds, n_churners=1, churn_rate=0.5)
        assert len(r.predictions) == 2
        assert r.n_churners == 1
