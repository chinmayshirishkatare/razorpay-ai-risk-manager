"""
train_model.py
----------------
Generates a synthetic transaction dataset (simulating Razorpay-style
payment transactions) and trains a Random Forest classifier to detect
fraudulent / high-risk transactions.

Run:
    python train_model.py

Output:
    - backend/data/transactions.csv   (synthetic dataset)
    - backend/models/risk_model.pkl   (trained model)
    - backend/models/scaler.pkl       (feature scaler)
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

N_SAMPLES = 20000
MERCHANT_CATEGORIES = [
    "electronics", "grocery", "travel", "fashion",
    "food_delivery", "utilities", "gaming", "crypto_exchange"
]


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Generate a synthetic but realistic payment transaction dataset."""

    amount = np.round(np.random.exponential(scale=2500, size=n_samples), 2)
    amount = np.clip(amount, 10, 500000)

    transaction_hour = np.random.randint(0, 24, size=n_samples)

    is_international = np.random.choice(
        [0, 1], size=n_samples, p=[0.88, 0.12]
    )

    merchant_category = np.random.choice(MERCHANT_CATEGORIES, size=n_samples)

    account_age_days = np.random.randint(1, 3000, size=n_samples)

    num_txns_last_hour = np.random.poisson(lam=1.2, size=n_samples)

    distance_from_home_km = np.round(
        np.random.exponential(scale=15, size=n_samples), 1
    )
    distance_from_home_km = np.clip(distance_from_home_km, 0, 5000)

    previous_fraud_flag = np.random.choice(
        [0, 1], size=n_samples, p=[0.97, 0.03]
    )

    failed_attempts_last_24h = np.random.poisson(lam=0.3, size=n_samples)

    device_trust_score = np.round(np.random.uniform(0, 1, size=n_samples), 2)

    df = pd.DataFrame({
        "amount": amount,
        "transaction_hour": transaction_hour,
        "is_international": is_international,
        "merchant_category": merchant_category,
        "account_age_days": account_age_days,
        "num_txns_last_hour": num_txns_last_hour,
        "distance_from_home_km": distance_from_home_km,
        "previous_fraud_flag": previous_fraud_flag,
        "failed_attempts_last_24h": failed_attempts_last_24h,
        "device_trust_score": device_trust_score,
    })

    # ---- Risk-scoring heuristic used ONLY to label synthetic data ----
    risk_score = (
        0.30 * (df["amount"] > 40000).astype(int)
        + 0.20 * df["is_international"]
        + 0.15 * (df["account_age_days"] < 30).astype(int)
        + 0.15 * (df["num_txns_last_hour"] > 3).astype(int)
        + 0.10 * (df["distance_from_home_km"] > 500).astype(int)
        + 0.30 * df["previous_fraud_flag"]
        + 0.15 * (df["failed_attempts_last_24h"] > 1).astype(int)
        + 0.20 * (df["device_trust_score"] < 0.3).astype(int)
        + 0.10 * (df["transaction_hour"].isin([1, 2, 3, 4])).astype(int)
    )

    noise = np.random.normal(0, 0.12, size=n_samples)
    final_score = risk_score + noise

    threshold = np.quantile(final_score, 0.93)  # ~7% fraud rate
    df["is_fraud"] = (final_score > threshold).astype(int)

    return df


def build_features(df: pd.DataFrame):
    """One-hot encode categorical columns and return feature matrix + labels."""
    df_encoded = pd.get_dummies(df, columns=["merchant_category"], prefix="cat")

    # Ensure all merchant category columns always exist (for consistent inference)
    for cat in MERCHANT_CATEGORIES:
        col = f"cat_{cat}"
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    feature_cols = [
        "amount", "transaction_hour", "is_international",
        "account_age_days", "num_txns_last_hour", "distance_from_home_km",
        "previous_fraud_flag", "failed_attempts_last_24h", "device_trust_score",
    ] + [f"cat_{c}" for c in MERCHANT_CATEGORIES]

    X = df_encoded[feature_cols]
    y = df_encoded["is_fraud"]
    return X, y, feature_cols


def main():
    print("Generating synthetic transaction dataset...")
    df = generate_dataset()
    csv_path = os.path.join(DATA_DIR, "transactions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path} ({len(df)} rows)")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

    X, y, feature_cols = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    model_path = os.path.join(MODEL_DIR, "risk_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)

    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
