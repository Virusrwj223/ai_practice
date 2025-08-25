# ml/train.py
import os, sqlite3, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "db" / "hdb.db"
DB_PATH = DEFAULT_DB.expanduser().resolve()

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RAW_COLS = [
    "month", "town", "flat_type", "flat_model",
    "storey_low", "storey_high", "floor_area_sqm",
    "lease_commence_year", "remaining_lease_months"
]
TARGET = "resale_price"

def load_data() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    q = """
    SELECT r.month, t.name AS town, r.flat_type, r.flat_model,
           r.storey_low, r.storey_high, r.floor_area_sqm,
           r.lease_commence_year, r.remaining_lease_months, r.resale_price
    FROM resale_transaction r
    JOIN town t ON t.id = r.town_id
    """
    df = pd.read_sql(q, con)
    con.close()
    # ensure month is datetime
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month", TARGET, "floor_area_sqm", "storey_low", "storey_high", "lease_commence_year"])
    # tidy strings
    for c in ["town", "flat_type", "flat_model"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    # keep only necessary columns
    return df

def fe_transform(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for both training and inference."""
    X = X.copy()
    # month to datetime if needed
    if not np.issubdtype(X["month"].dtype, np.datetime64):
        X["month"] = pd.to_datetime(X["month"], errors="coerce")
    # core derived features
    X["storey_mid"] = (X["storey_low"].astype(float) + X["storey_high"].astype(float)) / 2.0
    X["flat_age"] = X["month"].dt.year - X["lease_commence_year"].astype("float")
    X["flat_age"] = X["flat_age"].clip(lower=0)
    X["remaining_lease_years"] = (X["remaining_lease_months"].fillna(0).astype(float)) / 12.0
    # time index for backtests (not fed to model)
    X["month_idx"] = (X["month"].dt.year - X["month"].dt.year.min()) * 12 + X["month"].dt.month
    return X

NUM_FEATS = ["floor_area_sqm", "storey_mid", "flat_age", "remaining_lease_years"]
CAT_FEATS = ["town", "flat_type", "flat_model"]

# ---------- ONLY CHANGE: make LightGBM a single tree for speed ----------
def make_pipeline(objective="regression", alpha=None, n_estimators=1, learning_rate=1.0, num_leaves=4):
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
        ],
        remainder="drop",
    )
    est_kwargs = dict(
        objective=objective,
        n_estimators=n_estimators,   # 1 boosting iteration
        learning_rate=learning_rate, # irrelevant with 1 estimator, but set high
        num_leaves=num_leaves,       # tiny tree
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42
    )
    if objective == "quantile":
        est_kwargs["alpha"] = alpha
    model = LGBMRegressor(**est_kwargs)
    pipe = Pipeline([
        ("fe", FunctionTransformer(fe_transform, validate=False)),
        ("pre", pre),
        ("est", model)
    ])
    return pipe
# -----------------------------------------------------------------------

def backtest(df: pd.DataFrame, pipe_builder) -> pd.DataFrame:
    df = df.copy().sort_values("month")
    months = sorted(df["month"].dt.to_period("M").unique())
    results = []
    # start backtesting after 60% of months to have enough training history
    start_i = int(len(months) * 0.6)
    for i in range(start_i, len(months) - 1):
        train_months = [m.to_timestamp() for m in months[:i+1]]
        test_month = months[i+1].to_timestamp()

        tr = df[df["month"].isin(train_months)]
        te = df[df["month"] == test_month]

        X_tr = tr[RAW_COLS]
        y_tr = tr[TARGET]
        X_te = te[RAW_COLS]
        y_te = te[TARGET]

        pipe = pipe_builder()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        mae = mean_absolute_error(y_te, pred)
        rmse = mean_squared_error(y_te, pred)  # (kept as-is)
        mape = float(np.mean(np.abs((y_te - pred) / np.maximum(y_te, 1e-8))) * 100.0)
        results.append({"test_month": str(test_month)[:10], "MAE": mae, "RMSE": rmse, "MAPE": mape})
    return pd.DataFrame(results)

def train_and_save(df: pd.DataFrame):
    # 1) Backtest (kept, but now very fast since each fit is a single tree)
    mean_builder = lambda: make_pipeline(objective="regression")
    bt = backtest(df, mean_builder)
    bt.to_csv(MODEL_DIR / "backtest_mean.csv", index=False)

    # 2) Fit final models on ALL data (each one is a single boosting round)
    X_all = df[RAW_COLS]
    y_all = df[TARGET]

    mean_pipe = make_pipeline(objective="regression")
    mean_pipe.fit(X_all, y_all)
    joblib.dump(mean_pipe, MODEL_DIR / "resale_lgbm_mean.joblib")

    q10_pipe = make_pipeline(objective="quantile", alpha=0.10)
    q10_pipe.fit(X_all, y_all)
    joblib.dump(q10_pipe, MODEL_DIR / "resale_lgbm_q10.joblib")

    q50_pipe = make_pipeline(objective="quantile", alpha=0.50)
    q50_pipe.fit(X_all, y_all)
    joblib.dump(q50_pipe, MODEL_DIR / "resale_lgbm_q50.joblib")

    q90_pipe = make_pipeline(objective="quantile", alpha=0.90)
    q90_pipe.fit(X_all, y_all)
    joblib.dump(q90_pipe, MODEL_DIR / "resale_lgbm_q90.joblib")

    # 3) Metadata / model card stub
    meta = {
        "train_rows": int(len(df)),
        "train_start": str(df["month"].min())[:10],
        "train_end": str(df["month"].max())[:10],
        "features_num": NUM_FEATS,
        "features_cat": CAT_FEATS,
        "target": TARGET,
        "backtest_summary": {
            "rows": int(len(bt)),
            "MAE_mean": float(bt["MAE"].mean()) if len(bt) else None,
            "RMSE_mean": float(bt["RMSE"].mean()) if len(bt) else None,
            "MAPE_mean": float(bt["MAPE"].mean()) if len(bt) else None,
        }
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved models to", MODEL_DIR)

if __name__ == "__main__":
    df = load_data()
    train_and_save(df)
    print("Done. Backtest saved to models/backtest_mean.csv")
