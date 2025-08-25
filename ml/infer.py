# ml/infer.py
import os, math, json, joblib, sqlite3
from pathlib import Path
import pandas as pd

# ---------- finance config (env overridable) ----------
CONF = {
    "discount": float(os.getenv("BTO_DISCOUNT", "0.20")),  # 20% off resale for BTO proxy
    "ltv": float(os.getenv("LTV", "0.80")),                # loan-to-value
    "interest_pa": float(os.getenv("RATE", "0.026")),      # annual interest (e.g., 2.6%)
    "tenure_years": int(os.getenv("TENURE", "25")),        # years
    "msr": float(os.getenv("MSR", "0.30")),                # mortgage servicing ratio
}

# ---------- paths (absolute, robust) ----------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models")).resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    r = annual_rate / 12.0
    n = years * 12
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def required_income(price: float, conf=CONF) -> float:
    loan = price * conf["ltv"]
    pay = monthly_payment(loan, conf["interest_pa"], conf["tenure_years"])
    return pay / conf["msr"]

def _try(path: Path):
    return joblib.load(path) if path.exists() else None

# load if present
MEAN = _try(MODELS_DIR / "resale_lgbm_mean.joblib")
Q10  = _try(MODELS_DIR / "resale_lgbm_q10.joblib")
Q50  = _try(MODELS_DIR / "resale_lgbm_q50.joblib")
Q90  = _try(MODELS_DIR / "resale_lgbm_q90.joblib")

def _ensure_models():
    """If models are missing, run the one-step trainer and reload."""
    global MEAN, Q10, Q50, Q90
    if (MEAN is None) and (Q50 is None):
        try:
            # import here to avoid import-time overhead unless needed
            from ml import train as trainmod
            df = trainmod.load_data()
            trainmod.train_and_save(df)  # your one-step LightGBM training
            # reload
            MEAN = _try(MODELS_DIR / "resale_lgbm_mean.joblib")
            Q10  = _try(MODELS_DIR / "resale_lgbm_q10.joblib")
            Q50  = _try(MODELS_DIR / "resale_lgbm_q50.joblib")
            Q90  = _try(MODELS_DIR / "resale_lgbm_q90.joblib")
        except Exception as e:
            # leave models as None; predict() will raise a clean error
            pass

REQUIRED_INPUTS = [
    "month","town","flat_type","flat_model",
    "storey_low","storey_high","floor_area_sqm",
    "lease_commence_year","remaining_lease_months"
]

def predict(records):
    """
    records: dict or list[dict] with keys REQUIRED_INPUTS.
    returns list[dict] with resale_pred, bto_proxy, required_income, and optional p10/p50/p90.
    """
    _ensure_models()

    if isinstance(records, dict):
        records = [records]
    X = pd.DataFrame(records)
    # basic parsing/cleanup so the saved pipeline's FE works correctly
    if "month" in X.columns:
        X["month"] = pd.to_datetime(X["month"], errors="coerce")
    for c in ["town","flat_type","flat_model"]:
        if c in X.columns:
            X[c] = X[c].astype(str).str.strip().str.upper()

    # choose central model: prefer Q50 (median), else MEAN
    central_model = Q50 or MEAN
    if central_model is None:
        raise RuntimeError(
            "No model found and auto-training failed. "
            "Please run:  python -m ml.train  (from project root)"
        )

    central = central_model.predict(X)
    p10 = Q10.predict(X) if Q10 is not None else None
    p50 = Q50.predict(X) if Q50 is not None else None
    p90 = Q90.predict(X) if Q90 is not None else None

    out = []
    for i in range(len(X)):
        resale = float(central[i])
        bto = resale * (1 - CONF["discount"])
        income = required_income(bto)
        row = {
            "resale_pred": resale,
            "bto_proxy": bto,
            "required_income": income,
        }
        if p10 is not None: row["p10"] = float(p10[i])
        if p50 is not None: row["p50"] = float(p50[i])
        if p90 is not None: row["p90"] = float(p90[i])
        out.append(row)
    return out
