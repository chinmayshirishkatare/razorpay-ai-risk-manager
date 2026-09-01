"""
model.py
--------
Loads the trained fraud-detection model and exposes a clean
`RiskEngine` class used by the FastAPI routes to score transactions.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MERCHANT_CATEGORIES = [
    "electronics", "grocery", "travel", "fashion",
    "food_delivery", "utilities", "gaming", "crypto_exchange"
]


class RiskEngine:
    """Wraps the trained model + scaler and provides a `.assess()` method."""

    def __init__(self):
        model_path = os.path.join(MODEL_DIR, "risk_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        features_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            raise FileNotFoundError(
                "Model files not found. Please run `python train_model.py` "
                "from the backend/ directory first."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_cols = joblib.load(features_path)

    def _build_feature_row(self, txn: dict) -> pd.DataFrame:
        row = {
            "amount": txn["amount"],
            "transaction_hour": txn["transaction_hour"],
            "is_international": int(txn["is_international"]),
            "account_age_days": txn["account_age_days"],
            "num_txns_last_hour": txn["num_txns_last_hour"],
            "distance_from_home_km": txn["distance_from_home_km"],
            "previous_fraud_flag": int(txn["previous_fraud_flag"]),
            "failed_attempts_last_24h": txn["failed_attempts_last_24h"],
            "device_trust_score": txn["device_trust_score"],
        }
        for cat in MERCHANT_CATEGORIES:
            row[f"cat_{cat}"] = 1 if txn["merchant_category"] == cat else 0

        df = pd.DataFrame([row])
        df = df[self.feature_cols]  # enforce correct column order
        return df

    def _contributing_factors(self, txn: dict) -> list[str]:
        """Human-readable explanation of why a transaction looks risky."""
        factors = []
        if txn["amount"] > 40000:
            factors.append("Unusually high transaction amount")
        if txn["is_international"]:
            factors.append("International transaction")
        if txn["account_age_days"] < 30:
            factors.append("New account (less than 30 days old)")
        if txn["num_txns_last_hour"] > 3:
            factors.append("High transaction velocity (multiple txns in last hour)")
        if txn["distance_from_home_km"] > 500:
            factors.append("Transaction far from usual location")
        if txn["previous_fraud_flag"]:
            factors.append("Account previously flagged for fraud")
        if txn["failed_attempts_last_24h"] > 1:
            factors.append("Multiple failed payment attempts in last 24h")
        if txn["device_trust_score"] < 0.3:
            factors.append("Low device trust score")
        if txn["transaction_hour"] in (1, 2, 3, 4):
            factors.append("Transaction made during unusual hours (1-4 AM)")
        if not factors:
            factors.append("No significant risk indicators detected")
        return factors

    def assess(self, txn: dict) -> dict:
        X = self._build_feature_row(txn)
        X_scaled = self.scaler.transform(X)

        proba = float(self.model.predict_proba(X_scaled)[0][1])
        prediction = bool(self.model.predict(X_scaled)[0])
        risk_score = int(round(proba * 100))

        if risk_score < 25:
            risk_level = "LOW"
            action = "Approve transaction automatically."
        elif risk_score < 50:
            risk_level = "MEDIUM"
            action = "Approve, but flag for periodic review."
        elif risk_score < 75:
            risk_level = "HIGH"
            action = "Hold for manual review / trigger OTP step-up verification."
        else:
            risk_level = "CRITICAL"
            action = "Block transaction and notify fraud team immediately."

        return {
            "is_fraud_predicted": prediction,
            "fraud_probability": round(proba, 4),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "recommended_action": action,
            "contributing_factors": self._contributing_factors(txn),
        }


# Singleton instance, loaded once at app startup
risk_engine: RiskEngine | None = None


def get_risk_engine() -> RiskEngine:
    global risk_engine
    if risk_engine is None:
        risk_engine = RiskEngine()
    return risk_engine
