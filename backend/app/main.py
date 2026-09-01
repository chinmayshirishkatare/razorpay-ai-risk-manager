"""
main.py
-------
AI Risk Manager - FastAPI backend.

Endpoints:
    GET  /                      -> health check / welcome
    GET  /health                -> service health status
    POST /api/assess-risk       -> evaluate a transaction for fraud risk
    GET  /api/recent-transactions -> list recently evaluated transactions
    GET  /api/stats             -> dashboard summary statistics

Run:
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from collections import deque

from app.schemas import TransactionInput, RiskAssessment, DashboardStats
from app.model import get_risk_engine

app = FastAPI(
    title="AI Risk Manager API",
    description="Fraud & risk scoring engine for payment transactions (Razorpay AI Builder Internship - Track 2).",
    version="1.0.0",
)

# Allow the frontend dashboard (served separately) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of recently evaluated transactions (for demo/dashboard purposes)
RECENT_TRANSACTIONS = deque(maxlen=50)
STATS = {"total": 0, "flagged": 0, "score_sum": 0}


@app.get("/")
def root():
    return {
        "message": "AI Risk Manager API is running.",
        "docs": "/docs",
        "track": "Track 2: AI Risk Manager",
    }


@app.get("/health")
def health_check():
    try:
        get_risk_engine()
        return {"status": "healthy", "model_loaded": True}
    except FileNotFoundError as e:
        return {"status": "unhealthy", "model_loaded": False, "detail": str(e)}


@app.post("/api/assess-risk", response_model=RiskAssessment)
def assess_risk(transaction: TransactionInput):
    try:
        engine = get_risk_engine()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = engine.assess(transaction.model_dump())

    # Update in-memory dashboard state
    STATS["total"] += 1
    STATS["score_sum"] += result["risk_score"]
    if result["is_fraud_predicted"]:
        STATS["flagged"] += 1

    RECENT_TRANSACTIONS.appendleft({
        "timestamp": datetime.utcnow().isoformat(),
        "amount": transaction.amount,
        "merchant_category": transaction.merchant_category,
        "risk_level": result["risk_level"],
        "risk_score": result["risk_score"],
        "is_fraud_predicted": result["is_fraud_predicted"],
    })

    return RiskAssessment(**result)


@app.get("/api/recent-transactions")
def recent_transactions():
    return {"transactions": list(RECENT_TRANSACTIONS)}


@app.get("/api/stats", response_model=DashboardStats)
def stats():
    total = STATS["total"] or 1  # avoid divide-by-zero
    return DashboardStats(
        total_transactions_evaluated=STATS["total"],
        flagged_as_fraud=STATS["flagged"],
        fraud_rate_percent=round((STATS["flagged"] / total) * 100, 2),
        average_risk_score=round(STATS["score_sum"] / total, 2),
    )
