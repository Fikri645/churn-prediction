"""Pydantic request / response schemas for the FastAPI endpoint."""
from pydantic import BaseModel, Field
from typing import Literal


class CustomerFeatures(BaseModel):
    """Single customer feature vector — mirrors the raw CSV columns (minus customerID)."""

    # Demographics
    gender           : Literal["Male", "Female"]
    SeniorCitizen    : Literal[0, 1]
    Partner          : Literal["Yes", "No"]
    Dependents       : Literal["Yes", "No"]

    # Services
    tenure           : int   = Field(..., ge=0, le=72, description="Months as customer")
    PhoneService     : Literal["Yes", "No"]
    MultipleLines    : Literal["Yes", "No", "No phone service"]
    InternetService  : Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity   : Literal["Yes", "No", "No internet service"]
    OnlineBackup     : Literal["Yes", "No", "No internet service"]
    DeviceProtection : Literal["Yes", "No", "No internet service"]
    TechSupport      : Literal["Yes", "No", "No internet service"]
    StreamingTV      : Literal["Yes", "No", "No internet service"]
    StreamingMovies  : Literal["Yes", "No", "No internet service"]

    # Account
    Contract         : Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling : Literal["Yes", "No"]
    PaymentMethod    : Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges   : float = Field(..., ge=0)
    TotalCharges     : float = Field(..., ge=0)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85, "TotalCharges": 958.2,
    }}}


class PredictionResponse(BaseModel):
    churn_probability : float = Field(..., description="P(Churn=1), range [0, 1]")
    churn_prediction  : bool  = Field(..., description="True if probability ≥ 0.5")
    risk_label        : Literal["High", "Medium", "Low"]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    n_churners : int
    churn_rate : float
