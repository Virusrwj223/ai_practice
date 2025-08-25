# app/pages/01_Admin.py
import sys
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.drift import compute_drift  # import after sys.path bootstrap


LOG_DB = ROOT / "logs" / "telemetry.db"
MODEL_META = ROOT / "models" / "model_meta.json"

st.set_page_config(page_title="Admin • Telemetry & Drift", page_icon="📈", layout="wide")
st.title("📈 Admin · Telemetry & Drift")

# ---------------------------
# Helpers
# ---------------------------
def read_sql_safe(db_path: Path, query: str, expected_cols: list[str]) -> pd.DataFrame:
    """Read a table if it exists; otherwise return an empty DataFrame with expected columns."""
    if not db_path.exists():
        return pd.DataFrame(columns=expected_cols)
    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql(query, con)
        con.close()
        return df
    except Exception:
        return pd.DataFrame(columns=expected_cols)

def normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Force 'ts' column (if present) to datetime, preserving tz if present; coerce bad rows to NaT."""
    if len(df) and "ts" in df.columns:
        # handle plain strings, ISO with +00:00, etc.
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    return df

def cutoff_like_series(ts_series: pd.Series, days: int = 7) -> pd.Timestamp:
    """
    Return a cutoff timestamp 'days' before ts_series.max(), matching the series' dtype/tz.
    Falls back to UTC-now if the series is empty or all NaT.
    """
    if ts_series is not None and len(ts_series.dropna()) > 0:
        return ts_series.max() - pd.Timedelta(days=days)
    # fallback: naive UTC now (rare; only if no rows)
    return pd.Timestamp.utcnow() - pd.Timedelta(days=days)

# ---------------------------
# Telemetry
# ---------------------------
st.header("Tool & Router Telemetry")

tool_df = read_sql_safe(
    LOG_DB,
    "SELECT * FROM tool_calls ORDER BY ts DESC LIMIT 5000",
    ["ts", "tool", "args_json", "ok", "ms", "err"],
)
router_df = read_sql_safe(
    LOG_DB,
    "SELECT * FROM router_events ORDER BY ts DESC LIMIT 5000",
    ["ts", "ok", "tool", "raw_json", "err"],
)
pred_df = read_sql_safe(
    LOG_DB,
    "SELECT * FROM predictions ORDER BY ts DESC LIMIT 5000",
    ["ts", "town", "flat_type", "band", "resale", "bto", "required_income", "model_version"],
)

tool_df = normalize_ts(tool_df)
router_df = normalize_ts(router_df)
pred_df = normalize_ts(pred_df)

if not LOG_DB.exists():
    st.warning("No telemetry DB yet. Interact with the app to generate logs.")
else:
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tool calls (last 5k)", f"{len(tool_df):,}")
    col2.metric("Router events (last 5k)", f"{len(router_df):,}")
    col3.metric("Predictions (last 5k)", f"{len(pred_df):,}")

    ok_series = pd.to_numeric(router_df.get("ok", pd.Series([], dtype="float")), errors="coerce")
    ok_rate = float(ok_series.mean() * 100) if len(ok_series) else 0.0
    col4.metric("Router JSON OK %", f"{ok_rate:.1f}%")

    # Latency by tool
    st.subheader("Latency by tool")
    if len(tool_df):
        lat = (
            tool_df.assign(ms=pd.to_numeric(tool_df["ms"], errors="coerce"))
            .dropna(subset=["ms"])
            .groupby("tool")["ms"]
            .agg(avg_ms="mean", p95_ms=lambda s: float(np.percentile(s, 95)))
            .sort_values("p95_ms", ascending=False)
        )
        st.dataframe(lat.style.format({"avg_ms": "{:.1f}", "p95_ms": "{:.1f}"}))
    else:
        st.info("No tool telemetry yet.")

    # Recent tool errors
    st.subheader("Recent tool errors")
    if "ok" in tool_df.columns and "err" in tool_df.columns:
        errs = tool_df[tool_df["ok"].astype(str) == "0"][["ts", "tool", "err"]].head(50)
        st.dataframe(errs)
    else:
        st.info("No errors logged.")

    # Prediction distribution (last 7 days)
    st.subheader("Prediction distribution (last 7 days)")
    if len(pred_df):
        cutoff = cutoff_like_series(pred_df["ts"], days=7)  # <<< robust cutoff; matches ts dtype/tz
        recent = pred_df[pred_df["ts"] >= cutoff]
        if len(recent):
            agg = (
                recent.assign(
                    bto=pd.to_numeric(recent["bto"], errors="coerce"),
                    required_income=pd.to_numeric(recent["required_income"], errors="coerce"),
                )
                .dropna(subset=["bto"])
                .groupby("town")
                .agg(
                    n=("bto", "count"),
                    bto_min=("bto", "min"),
                    bto_avg=("bto", "mean"),
                    bto_max=("bto", "max"),
                )
                .sort_values("n", ascending=False)
                .head(20)
            )
            st.dataframe(agg.style.format({"bto_min": "{:.0f}", "bto_avg": "{:.0f}", "bto_max": "{:.0f}"}))
        else:
            st.info("No predictions in the last 7 days.")
    else:
        st.info("No prediction telemetry yet.")

# ---------------------------
# Drift
# ---------------------------
st.header("Data Drift (latest month vs. training reference)")

try:
    dr = compute_drift()

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Numeric drift (PSI vs. training means)")
        num = {k: v for k, v in dr.items() if isinstance(v, dict) and "psi_vs_mean" in v}
        if num:
            st.json(num)
        else:
            st.info("No numeric drift metrics available.")

    with colB:
        st.subheader("Top town share deltas")
        deltas = dr.get("town_share_delta", {})
        if deltas:
            ser = pd.Series(deltas).sort_values(ascending=False)
            st.bar_chart(ser)
        else:
            st.info("No categorical drift metrics available.")

    # Model meta (training window, features)
    st.subheader("Model meta")
    if MODEL_META.exists():
        st.json(json.loads(MODEL_META.read_text()))
    else:
        st.info("model_meta.json not found. Train the model to generate it (python -m ml.train).")

except Exception as e:
    st.error(f"Drift script error: {e}")

st.caption("Tip: Use this page to verify router health, tool latencies, and data stability before demos.")
