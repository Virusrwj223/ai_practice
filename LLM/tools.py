# LLM/tools.py
"""
Tools are plain functions with stable JSON signatures.
This is A2A-friendly: any client can call them the same way,
and you could later expose them via MCP with the same contracts.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from ml.infer import predict as price_predict, CONF as FINCONF

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "db" / "hdb.db"

def t_price_estimates(town: str, flat_type: str, month: str | None = None,
                      bands: tuple[str, ...] = ("low", "mid", "high")) -> dict:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT month, flat_model, storey_low, storey_high, floor_area_sqm,
               lease_commence_year, remaining_lease_months
        FROM resale_transaction r
        JOIN town t ON t.id=r.town_id
        WHERE t.name=? AND r.flat_type=?;
    """, con, params=[town.upper(), flat_type.upper()])
    con.close()
    if df.empty:
        return {"error": f"No data for {town}/{flat_type}"}

    # default month = latest available in DB
    if not month:
        month = str(pd.to_datetime(df["month"]).max().date())[:7]

    # typical values per segment
    med = df.median(numeric_only=True)
    mode_model = df["flat_model"].mode().iloc[0] if not df["flat_model"].isna().all() else "IMPROVED"

    band_map = {"low": (1, 3), "mid": (4, 6), "high": (10, 12)}
    out_rows = []
    for b in bands:
        lo, hi = band_map.get(b, (4, 6))
        rec = {
            "month": month,
            "town": town.upper(),
            "flat_type": flat_type.upper(),
            "flat_model": str(mode_model).upper(),
            "storey_low": int(lo),
            "storey_high": int(hi),
            "floor_area_sqm": float(med.get("floor_area_sqm", 90.0)),
            "lease_commence_year": int(med.get("lease_commence_year", 1990)),
            "remaining_lease_months": int(med.get("remaining_lease_months", 300)),
        }
        pred = price_predict(rec)[0]  # central = q50 if present else mean
        pred["band"] = b
        out_rows.append(pred)

    return {"tool": "price_estimates", "month": month, "town": town, "flat_type": flat_type,
            "rows": out_rows, "finance": FINCONF}

def t_low_supply(last_n_years: int = 10, flat_type: str | None = None, top_k: int = 8) -> dict:
    cutoff = (datetime.utcnow().date().replace(day=1) - timedelta(days=365*last_n_years))
    con = sqlite3.connect(DB_PATH)
    q = """
    SELECT t.name AS town, r.flat_type, COUNT(*) AS n
    FROM resale_transaction r
    JOIN town t ON t.id=r.town_id
    WHERE r.month >= ?
    GROUP BY t.name, r.flat_type
    ORDER BY n ASC;
    """
    df = pd.read_sql(q, con, params=[str(cutoff)])
    con.close()
    if flat_type:
        df = df[df["flat_type"].str.upper() == flat_type.upper()]
    df = df.sort_values("n", ascending=True).head(top_k)
    return {"tool": "low_supply", "cutoff": str(cutoff), "flat_type": flat_type,
            "items": df.to_dict(orient="records")}
