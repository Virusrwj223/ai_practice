# monitoring/drift.py
import json, sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db" / "hdb.db"
META = ROOT / "models" / "model_meta.json"

def psi(a: np.ndarray, b: np.ndarray, bins=10):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    q = np.quantile(a, np.linspace(0,1,bins+1))
    q = np.unique(q)
    if len(q) < 3: return 0.0
    a_hist, _ = np.histogram(a, bins=q)
    b_hist, _ = np.histogram(b, bins=q)
    a_rat = np.clip(a_hist / max(1,a_hist.sum()), 1e-6, 1)
    b_rat = np.clip(b_hist / max(1,b_hist.sum()), 1e-6, 1)
    return float(np.sum((b_rat - a_rat) * np.log(b_rat / a_rat)))

def latest_month_view():
    con = sqlite3.connect(DB)
    df = pd.read_sql("""
    SELECT r.month, t.name as town, r.flat_type, r.storey_low, r.storey_high,
           r.floor_area_sqm, r.lease_commence_year, r.remaining_lease_months
    FROM resale_transaction r JOIN town t ON t.id=r.town_id
    """, con, parse_dates=["month"])
    con.close()
    df["storey_mid"] = (df["storey_low"]+df["storey_high"])/2
    df["flat_age"] = df["month"].dt.year - df["lease_commence_year"]
    df["remaining_lease_years"] = df["remaining_lease_months"].fillna(0)/12
    latest = df[df["month"] == df["month"].max()]
    return latest

def compute_drift():
    ref = json.loads(META.read_text())
    latest = latest_month_view()
    out = {}
    for col in ["floor_area_sqm","storey_low","storey_high","remaining_lease_months"]:
        base_mean = ref["reference"]["num_means"][col]
        psi_val = psi(latest[col].to_numpy(), np.array([base_mean]*len(latest)))
        out[col] = {"psi_vs_mean": psi_val, "latest_mean": float(latest[col].mean())}
    # simple cat drift: top-5 share change
    top_towns = sorted(ref["reference"]["cat_freqs_town"], key=ref["reference"]["cat_freqs_town"].get, reverse=True)[:5]
    latest_town_share = latest["town"].value_counts(normalize=True).to_dict()
    out["town_share_delta"] = {t: float(latest_town_share.get(t,0) - ref["reference"]["cat_freqs_town"].get(t,0)) for t in top_towns}
    return out

if __name__ == "__main__":
    print(compute_drift())
