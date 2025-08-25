# ml/infer.py
import os, math, joblib, pandas as pd
from pathlib import Path

MODELS_DIR = Path(os.getenv("MODEL_DIR", "models"))

# Finance & policy assumptions (override via env)
CONF = {
    "discount": float(os.getenv("BTO_DISCOUNT", "0.20")),  # 20% proxy discount from resale
    "ltv": float(os.getenv("LTV", "0.80")),                # loan-to-value
    "interest_pa": float(os.getenv("RATE", "0.026")),      # annual interest (HDB example 2.6%)
    "tenure_years": int(os.getenv("TENURE", "25")),        # tenure years
    "msr": float(os.getenv("MSR", "0.30")),                # mortgage servicing ratio
}

def monthly_payment(principal, annual_rate, years):
    r = annual_rate / 12.0
    n = years * 12
    if r == 0: return principal / n
    return principal * (r * (1 + r)**n) / ((1 + r)**n - 1)

def required_income(price, conf=CONF):
    loan = price * conf["ltv"]
    pay = monthly_payment(loan, conf["interest_pa"], conf["tenure_years"])
    return pay / conf["msr"]

# Load models if present
def _try(path):
    return joblib.load(path) if path.exists() else None

MEAN = _try(MODELS_DIR / "resale_lgbm_mean.joblib")
Q10  = _try(MODELS_DIR / "resale_lgbm_q10.joblib")
Q50  = _try(MODELS_DIR / "resale_lgbm_q50.joblib")
Q90  = _try(MODELS_DIR / "resale_lgbm_q90.joblib")

REQUIRED_INPUTS = [
    "month","town","flat_type","flat_model",
    "storey_low","storey_high","floor_area_sqm",
    "lease_commence_year","remaining_lease_months"
]

def predict(records):
    """
    records: dict or list[dict] with keys REQUIRED_INPUTS.
    returns list of dicts with resale_mean (or median), p10/p50/p90 if available,
    plus bto_proxy and required_income.
    """
    if isinstance(records, dict):
        records = [records]
    X = pd.DataFrame(records)
    # Ensure month looks like "YYYY-MM" or date
    if X["month"].dtype == object:
        X["month"] = pd.to_datetime(X["month"], errors="coerce")

    out = []
    # central tendency: prefer Q50 if available, else MEAN
    central_model = Q50 or MEAN
    if central_model is None:
        raise RuntimeError("No model found. Train models first (ml/train.py).")

    central = central_model.predict(X)
    p10 = Q10.predict(X) if Q10 is not None else None
    p50 = Q50.predict(X) if Q50 is not None else None
    p90 = Q90.predict(X) if Q90 is not None else None

    for i in range(len(X)):
        resale_central = float(central[i])
        bto = resale_central * (1 - CONF["discount"])
        income = required_income(bto)
        row = {
            "resale_pred": resale_central,
            "bto_proxy": bto,
            "required_income": income,
        }
        if p10 is not None: row["p10"] = float(p10[i])
        if p50 is not None: row["p50"] = float(p50[i])
        if p90 is not None: row["p90"] = float(p90[i])
        out.append(row)
    return out
