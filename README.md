# AI Risk Manager — Real-Time Fraud Detection Engine

**Track:** Track 2 — AI Risk Manager
**Submission:** Razorpay AI Builder Internship 2026

A machine-learning powered risk engine that evaluates payment transactions in
real time and flags potentially fraudulent activity, similar to the risk
systems used by payment gateways like Razorpay.

---

## Problem it solves

Payment platforms process millions of transactions daily. Manually reviewing
each one for fraud is impossible, and simple rule-based systems (e.g. "block
if amount > ₹50,000") are too rigid — they miss sophisticated fraud and
block too many genuine customers.

**AI Risk Manager** solves this by using a trained Random Forest classifier
that looks at multiple signals together — transaction amount, location,
account age, device trust, transaction velocity, time of day, and history —
to produce a **0–100 risk score**, a **risk level** (LOW / MEDIUM / HIGH /
CRITICAL), and a **recommended action**, along with a plain-English
explanation of *why* the transaction was flagged.

---

## Tech stack

| Layer          | Technology                          |
|----------------|--------------------------------------|
| ML Model       | scikit-learn (Random Forest Classifier) |
| Backend API    | FastAPI (Python)                    |
| Frontend       | HTML / CSS / vanilla JavaScript     |
| Data handling  | pandas, numpy                       |
| Model storage  | joblib                              |

---

## Project structure

```
razorpay-ai-risk-manager/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app + API routes
│   │   ├── model.py         # RiskEngine: loads model, runs inference
│   │   └── schemas.py       # Pydantic request/response models
│   ├── data/                # generated synthetic dataset (created on train)
│   ├── models/               # trained model files (created on train)
│   ├── train_model.py       # generates data + trains the model
│   └── requirements.txt
├── frontend/
│   └── index.html           # dashboard (form + live risk results + table)
├── .gitignore
└── README.md
```

---

## Setup instructions (VS Code) — पूर्ण मराठीत मार्गदर्शन

### पूर्वतयारी (Prerequisites)
- **Python 3.10+** इन्स्टॉल असणे आवश्यक आहे ([python.org](https://python.org) वरून डाउनलोड करा)
- **VS Code** इन्स्टॉल असणे आवश्यक
- VS Code मध्ये **Python extension** इन्स्टॉल करा (Extensions टॅबमधून)

### पायरी 1: प्रोजेक्ट उघडा
1. Zip फाईल extract करा
2. VS Code उघडा → `File > Open Folder` → `razorpay-ai-risk-manager` फोल्डर निवडा

### पायरी 2: Terminal उघडा
VS Code मध्ये `Terminal > New Terminal` वर क्लिक करा (किंवा `Ctrl + ~`)

### पायरी 3: Virtual environment तयार करा (recommended)
```bash
cd backend
python -m venv venv
```

Activate करा:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### पायरी 4: Dependencies इन्स्टॉल करा
```bash
pip install -r requirements.txt
```

### पायरी 5: Model train करा
हे synthetic transaction data generate करेल आणि fraud-detection model train करेल:
```bash
python train_model.py
```
हे यशस्वी झाल्यावर तुम्हाला accuracy report आणि "Training complete." असा मेसेज दिसेल.
`backend/models/` फोल्डरमध्ये `risk_model.pkl` फाईल तयार होईल.

### पायरी 6: Backend server सुरू करा
```bash
uvicorn app.main:app --reload --port 8000
```
सर्व्हर सुरू झाल्यावर टर्मिनलमध्ये दिसेल: `Uvicorn running on http://127.0.0.1:8000`

API डॉक्युमेंटेशन बघण्यासाठी ब्राउझरमध्ये उघडा: **http://localhost:8000/docs**

### पायरी 7: Frontend dashboard उघडा
`frontend/index.html` फाईलवर उजवं क्लिक करा → **"Open with Live Server"**
(किंवा VS Code मध्ये "Live Server" extension इन्स्टॉल करा, नंतर हे option दिसेल)

Live Server नसेल तर फक्त `index.html` फाईल डबल-क्लिक करून ब्राउझरमध्ये उघडू शकता —
पण backend `localhost:8000` वर आधीच चालू असणं गरजेचं आहे.

### पायरी 8: टेस्ट करा
Dashboard वर एक transaction भरा (उदा. जास्त amount, international, low device trust)
आणि "Assess risk" क्लिक करा — तुम्हाला लगेच risk score आणि reasons दिसतील.

---

## API Endpoints

| Method | Endpoint                     | Description                          |
|--------|-------------------------------|---------------------------------------|
| GET    | `/health`                    | Check if API and model are ready      |
| POST   | `/api/assess-risk`           | Submit a transaction, get risk score  |
| GET    | `/api/recent-transactions`   | List recently evaluated transactions  |
| GET    | `/api/stats`                 | Dashboard summary statistics          |

### Example request to `/api/assess-risk`
```json
{
  "amount": 45000,
  "transaction_hour": 2,
  "is_international": true,
  "merchant_category": "crypto_exchange",
  "account_age_days": 5,
  "num_txns_last_hour": 4,
  "distance_from_home_km": 850.5,
  "previous_fraud_flag": false,
  "failed_attempts_last_24h": 2,
  "device_trust_score": 0.15
}
```

### Example response
```json
{
  "is_fraud_predicted": true,
  "fraud_probability": 0.87,
  "risk_level": "CRITICAL",
  "risk_score": 87,
  "recommended_action": "Block transaction and notify fraud team immediately.",
  "contributing_factors": [
    "Unusually high transaction amount",
    "International transaction",
    "New account (less than 30 days old)",
    "Transaction far from usual location",
    "Low device trust score",
    "Transaction made during unusual hours (1-4 AM)"
  ]
}
```

---

## Build challenges & how they were solved

1. **No real fraud dataset available.** Real payment fraud datasets are
   private/confidential. Solved by writing a synthetic data generator
   (`train_model.py`) that creates realistic transaction patterns with a
   controllable, heuristic-based fraud rate (~7%), which is standard
   practice for prototyping risk models before connecting to real data.

2. **Class imbalance.** Fraud is rare, so a naive model would just predict
   "not fraud" every time and still score high accuracy. Solved using
   `class_weight="balanced"` in the Random Forest so the minority (fraud)
   class is weighted properly.

3. **Explainability.** A raw probability score isn't useful to a risk
   analyst. Solved by adding a rule-based `contributing_factors` layer on
   top of the model output, so every prediction comes with a human-readable
   explanation.

4. **Frontend-backend feature consistency.** Categorical fields
   (merchant category) need identical encoding between training and
   inference. Solved by persisting the exact `feature_columns` order with
   `joblib` and reconstructing it identically in `model.py`.

---

## Future scope
- Connect to a live transaction stream (Kafka / webhook) instead of manual form input
- Add a feedback loop where analyst decisions retrain the model
- Add SHAP-based explainability instead of rule-based factors
- Persist transaction history in a real database (PostgreSQL) instead of in-memory storage
