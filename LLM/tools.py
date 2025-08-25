# LLM/tools.py
"""
Agent tools (A2A-style) used by the UI/agent. Each tool is a pure function with a
stable JSON contract, easy to expose later via MCP or an API gateway.

Tools:
- t_price_estimates(town, flat_type, month?: str, bands=('low','mid','high'))
- t_low_supply(last_n_years=10, flat_type?: str, top_k=8)
"""
from __future__ import annotations

import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "db" / "hdb.db"

# Inference & finance parameters
from ml.infer import predict as price_predict, CONF as FINCONF, required_income

# --- Optional telemetry (no-ops if module not available) ---------------------
try:
    from monitoring.telemetry import log_tool, log_prediction
except Exception:  # pragma: no cover
    def log_tool(tool: str, args: dict, ok: bool, ms: float, err: str | None = None):
        return

    def log_prediction(*_args, **_kw):
        return
# -----------------------------------------------------------------------------

# --------------------------- helpers -----------------------------------------
def _latest_month(con: sqlite3.Connection) -> str:
    row = con.execute("SELECT MAX(month) FROM resale_transaction").fetchone()
    if not row or not row[0]:
        return pd.Timestamp.today().strftime("%Y-%m")
    return str(pd.to_datetime(row[0]).date())[:7]  # YYYY-MM

def _typical_features(con: sqlite3.Connection, town: str, flat_type: str) -> Tuple[pd.Series, str]:
    """Median numeric features + modal flat_model for a (town, flat_type)."""
    df = pd.read_sql(
        """
        SELECT month, flat_model, storey_low, storey_high, floor_area_sqm,
               lease_commence_year, remaining_lease_months
        FROM resale_transaction r
        JOIN town t ON t.id = r.town_id
        WHERE t.name = ? AND r.flat_type = ?;
        """,
        con,
        params=[town.upper(), flat_type.upper()],
        parse_dates=["month"],
    )
    if df.empty:
        raise ValueError(f"No data for {town}/{flat_type}")
    med = df.median(numeric_only=True)
    mode_model = df["flat_model"].dropna()
    mode_model = (mode_model.mode().iloc[0] if len(mode_model) else "IMPROVED")
    return med, str(mode_model)

@lru_cache(maxsize=1024)
def _floor_premiums(town: str, flat_type: str) -> Dict[str, float]:
    """
    Compute simple floor premiums from last 24 months for the segment:
    median(price by band) / overall median. Clamped to [0.95, 1.10] to reduce noise.
    """
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT r.month, r.storey_low, r.storey_high, r.resale_price
        FROM resale_transaction r
        JOIN town t ON t.id = r.town_id
        WHERE t.name = ? AND r.flat_type = ? AND r.resale_price IS NOT NULL;
        """,
        con,
        params=[town.upper(), flat_type.upper()],
        parse_dates=["month"],
    )
    con.close()

    if df.empty:
        return {"low": 1.0, "mid": 1.0, "high": 1.0}

    df = df.sort_values("month")
    cutoff = df["month"].max() - pd.DateOffset(months=24)
    df = df[df["month"] >= cutoff]

    df["storey_mid"] = (df["storey_low"].astype(float) + df["storey_high"].astype(float)) / 2.0
    def _band(x: float) -> str:
        if x <= 3:
            return "low"
        if x >= 10:
            return "high"
        return "mid"
    df["band"] = df["storey_mid"].apply(_band)

    overall = float(df["resale_price"].median() or df["resale_price"].mean())
    if overall <= 0:
        overall = 1.0

    ratios = (df.groupby("band")["resale_price"].median() / overall).to_dict()
    out = {
        "low": float(np.clip(ratios.get("low", 1.0), 0.95, 1.10)),
        "mid": float(np.clip(ratios.get("mid", 1.0), 0.95, 1.10)),
        "high": float(np.clip(ratios.get("high", 1.0), 0.95, 1.10)),
    }
    return out

_BAND_MAP = {"low": (1, 3), "mid": (4, 6), "high": (10, 12)}

# --------------------------- tools -------------------------------------------
def t_price_estimates(
    town: str,
    flat_type: str,
    month: str | None = None,
    bands: Tuple[str, ...] = ("low", "mid", "high"),
) -> dict:
    """
    Return low/mid/high band estimates for (town, flat_type[, month]) using:
      - typical features from historical data (median values),
      - ML resale estimate,
      - data-driven floor premiums,
      - affordability math (BTO proxy & required income).
    Response:
      {
        "tool": "price_estimates",
        "month": "YYYY-MM",
        "town": "...",
        "flat_type": "...",
        "rows": [{"band","resale_pred","bto_proxy","required_income", "floor_premium_applied"}, ...],
        "finance": {...},
        "premiums": {"low":..., "mid":..., "high":...}
      }
    """
    t0 = time.perf_counter()
    args_for_log = {"town": town, "flat_type": flat_type, "month": month, "bands": list(bands)}
    try:
        con = sqlite3.connect(DB_PATH)
        if not month:
            month = _latest_month(con)
        med, mode_model = _typical_features(con, town, flat_type)
        con.close()

        premiums = _floor_premiums(town, flat_type)

        rows = []
        for b in bands:
            lo, hi = _BAND_MAP.get(b, (4, 6))
            rec = {
                "month": month,
                "town": town.upper(),
                "flat_type": flat_type.upper(),
                "flat_model": mode_model.upper(),
                "storey_low": int(lo),
                "storey_high": int(hi),
                "floor_area_sqm": float(med.get("floor_area_sqm", 90.0)),
                "lease_commence_year": int(med.get("lease_commence_year", 1990)),
                "remaining_lease_months": int(med.get("remaining_lease_months", 300)),
            }
            base = price_predict(rec)[0]  # central estimate (q50 if present)
            # Apply floor premium to resale estimate
            adj_resale = float(base["resale_pred"]) * float(premiums.get(b, 1.0))
            adj_bto = adj_resale * (1.0 - float(FINCONF["discount"]))
            adj_income = required_income(adj_bto)

            rows.append(
                {
                    "band": b,
                    "resale_pred": adj_resale,
                    "bto_proxy": adj_bto,
                    "required_income": adj_income,
                    "floor_premium_applied": float(premiums.get(b, 1.0)),
                }
            )
            # per-row prediction telemetry (optional)
            try:
                log_prediction(town.upper(), flat_type.upper(), b, adj_resale, adj_bto, adj_income, "lgbm-single-tree")
            except Exception:
                pass

        out = {
            "tool": "price_estimates",
            "month": month,
            "town": town,
            "flat_type": flat_type,
            "rows": rows,
            "finance": FINCONF,
            "premiums": premiums,
        }
        ms = (time.perf_counter() - t0) * 1000
        try:
            log_tool("price_estimates", args_for_log, True, ms, None)
        except Exception:
            pass
        return out

    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        try:
            log_tool("price_estimates", args_for_log, False, ms, repr(e))
        except Exception:
            pass
        return {"tool": "price_estimates", "error": str(e)}

def t_low_supply(
    last_n_years: int = 10,
    flat_type: str | None = None,
    top_k: int = 8,
) -> dict:
    """
    Proxy for 'limited launches': we rank (town, flat_type) pairs by **low resale volume**
    over the last N years. This is a heuristic since we don't have BTO-launch data here.
    Response:
      {
        "tool": "low_supply",
        "cutoff": "YYYY-MM-01",
        "flat_type": "...|None",
        "items": [{"town","flat_type","n"}, ...]  # ascending by n
      }
    """
    t0 = time.perf_counter()
    args_for_log = {"last_n_years": last_n_years, "flat_type": flat_type, "top_k": top_k}
    try:
        cutoff = (pd.Timestamp.utcnow().to_period("M").to_timestamp() - pd.DateOffset(years=last_n_years))
        con = sqlite3.connect(DB_PATH)
        q = """
        SELECT t.name AS town, r.flat_type, COUNT(*) AS n
        FROM resale_transaction r
        JOIN town t ON t.id = r.town_id
        WHERE r.month >= ?
        GROUP BY t.name, r.flat_type
        ORDER BY n ASC;
        """
        df = pd.read_sql(q, con, params=[str(cutoff.date())])
        con.close()

        if flat_type:
            df = df[df["flat_type"].str.upper() == flat_type.upper()]
        df = df.sort_values("n", ascending=True).head(int(top_k))
        items = df.to_dict(orient="records")

        out = {
            "tool": "low_supply",
            "cutoff": str(cutoff.date()),
            "flat_type": flat_type,
            "items": items,
        }
        ms = (time.perf_counter() - t0) * 1000
        try:
            log_tool("low_supply", args_for_log, True, ms, None)
        except Exception:
            pass
        return out

    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        try:
            log_tool("low_supply", args_for_log, False, ms, repr(e))
        except Exception:
            pass
        return {"tool": "low_supply", "error": str(e)}
