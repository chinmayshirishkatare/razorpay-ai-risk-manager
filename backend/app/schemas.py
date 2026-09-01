"""
schemas.py
----------
Pydantic request/response models for the AI Risk Manager API.
"""

from pydantic import BaseModel, Field
from typing import Literal


class TransactionInput(BaseModel):
    """Incoming transaction data submitted for risk evaluation."""

    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    is_international: bool = Field(..., description="Is this an international transaction?")
    merchant_category: Literal[
        "electronics", "grocery", "travel", "fashion",
        "food_delivery", "utilities", "gaming", "crypto_exchange"
    ] = Field(..., description="Merchant category")
    account_age_days: int = Field(..., ge=0, description="Age of the account in days")
    num_txns_last_hour: int = Field(..., ge=0, description="Number of transactions in the last hour")
    distance_from_home_km: float = Field(..., ge=0, description="Distance from home location in km")
    previous_fraud_flag: bool = Field(False, description="Has this account been flagged before?")
    failed_attempts_last_24h: int = Field(0, ge=0, description="Failed payment attempts in last 24h")
    device_trust_score: float = Field(..., ge=0, le=1, description="Device trust score (0=untrusted, 1=trusted)")

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 45000,
                "transaction_hour": 2,
                "is_international": True,
                "merchant_category": "crypto_exchange",
                "account_age_days": 5,
                "num_txns_last_hour": 4,
                "distance_from_home_km": 850.5,
                "previous_fraud_flag": False,
                "failed_attempts_last_24h": 2,
                "device_trust_score": 0.15,
            }
        }


class RiskAssessment(BaseModel):
    """Response returned after evaluating a transaction."""

    is_fraud_predicted: bool
    fraud_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_score: int = Field(..., description="Risk score out of 100")
    recommended_action: str
    contributing_factors: list[str]


class DashboardStats(BaseModel):
    """Aggregate stats shown on the dashboard."""

    total_transactions_evaluated: int
    flagged_as_fraud: int
    fraud_rate_percent: float
    average_risk_score: float
