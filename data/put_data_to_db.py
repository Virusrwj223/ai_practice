# data/setup_and_load_hdb.py
import os, re
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR, DATA_DIR = os.path.join(ROOT, "db"), os.path.join(ROOT, "data")
os.makedirs(DB_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "resale.csv")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")
DB_PATH = os.path.join(DB_DIR, "hdb.db")
DB_URL = f"sqlite:///{DB_PATH}"

def is_sqlite(engine: Engine) -> bool:
    return engine.url.get_backend_name() == "sqlite"

def apply_schema(engine: Engine, schema_path: str):
    with open(schema_path, "r", encoding="utf-8") as f:
        ddl_sql = f.read()
    if is_sqlite(engine):
        raw = engine.raw_connection()
        try:
            raw.executescript(ddl_sql)   # <-- key fix
            raw.commit()
        finally:
            raw.close()
    else:
        with engine.begin() as conn:
            conn.exec_driver_sql(ddl_sql)

def parse_month(x):
    if pd.isna(x): return None
    return datetime.strptime(str(x).strip(), "%Y-%m").date().replace(day=1)

def parse_storey_range(rng):
    if pd.isna(rng): return (None, None)
    m = re.search(r"(\d+)\s*TO\s*(\d+)", str(rng).upper())
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def parse_remaining_lease(s):
    if pd.isna(s): return None
    txt = str(s).lower()
    y = re.search(r"(\d+)\s*year", txt)
    m = re.search(r"(\d+)\s*month", txt)
    years = int(y.group(1)) if y else 0
    months = int(m.group(1)) if m else 0
    return years * 12 + months

def main():
    engine = create_engine(DB_URL, future=True)
    apply_schema(engine, SCHEMA_PATH)

    df = pd.read_csv(CSV_PATH)

    df["month"] = df["month"].apply(parse_month)
    for c in ["town","flat_type","block","street_name","flat_model"]:
        df[c] = df[c].astype(str).str.strip().str.upper()

    lohi = df["storey_range"].apply(parse_storey_range)
    df["storey_low"] = [a for a,_ in lohi]
    df["storey_high"] = [b for _,b in lohi]

    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")
    df["lease_commence_year"] = pd.to_numeric(df["lease_commence_date"], errors="coerce").astype("Int64")
    df["remaining_lease_months"] = df["remaining_lease"].apply(parse_remaining_lease)

    df = df.dropna(subset=[
        "month","town","flat_type","block","street_name",
        "resale_price","floor_area_sqm","storey_low","storey_high"
    ])

    # upsert towns
    with engine.begin() as conn:
        for t in sorted(df["town"].unique()):
            conn.execute(
                text("INSERT INTO town(name) VALUES (:n) ON CONFLICT(name) DO NOTHING"),
                {"n": t}
            )

    with engine.begin() as conn:
        rows = conn.execute(text("SELECT name, id FROM town")).fetchall()
    town_map = {name: tid for name, tid in rows}
    df["town_id"] = df["town"].map(town_map)

    insert_sql = """
    INSERT INTO resale_transaction
    (month, town_id, block, street_name, flat_type, flat_model,
     storey_low, storey_high, floor_area_sqm, lease_commence_year,
     remaining_lease_months, resale_price, source_file, source_rownum)
    VALUES (:month, :town_id, :block, :street_name, :flat_type, :flat_model,
            :storey_low, :storey_high, :floor_area_sqm, :lease_commence_year,
            :remaining_lease_months, :resale_price, :source_file, :source_rownum)
    ON CONFLICT(month, town_id, block, street_name, flat_type, flat_model,
                storey_low, storey_high, floor_area_sqm, lease_commence_year, resale_price)
    DO NOTHING;
    """

    payload = []
    for i, r in df.iterrows():
        payload.append({
            "month": r["month"], "town_id": int(r["town_id"]),
            "block": r["block"], "street_name": r["street_name"],
            "flat_type": r["flat_type"], "flat_model": r.get("flat_model"),
            "storey_low": int(r["storey_low"]), "storey_high": int(r["storey_high"]),
            "floor_area_sqm": float(r["floor_area_sqm"]),
            "lease_commence_year": None if pd.isna(r["lease_commence_year"]) else int(r["lease_commence_year"]),
            "remaining_lease_months": None if pd.isna(r["remaining_lease_months"]) else int(r["remaining_lease_months"]),
            "resale_price": float(r["resale_price"]),
            "source_file": os.path.basename(CSV_PATH),
            "source_rownum": int(i) + 2
        })

    with engine.begin() as conn:
        conn.execute(text(insert_sql), payload)

    print(f"Load complete. DB at db/hdb.db")
    print("Check counts:")
    print("  sqlite3 db/hdb.db 'SELECT COUNT(*) FROM resale_transaction;'")

if __name__ == "__main__":
    main()
