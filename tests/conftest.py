# tests/conftest.py
import sqlite3
from pathlib import Path
import json
import importlib
import types
import pytest

SCHEMA = """
CREATE TABLE town (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);
CREATE TABLE resale_transaction (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  month TEXT NOT NULL,
  town_id INTEGER NOT NULL,
  block TEXT, street_name TEXT,
  flat_type TEXT NOT NULL,
  flat_model TEXT,
  storey_low INTEGER NOT NULL,
  storey_high INTEGER NOT NULL,
  floor_area_sqm REAL NOT NULL,
  lease_commence_year INTEGER,
  remaining_lease_months INTEGER,
  resale_price REAL,
  FOREIGN KEY(town_id) REFERENCES town(id)
);
"""

SEED_ROWS = [
    # month, town, flat_type, model, storey_low, storey_high, area, lcy, rem_mo, price
    ("2025-07-01", "ANG MO KIO", "4 ROOM", "IMPROVED",      2,  3, 90.0, 1982, 420, 470000),
    ("2025-08-01", "ANG MO KIO", "4 ROOM", "IMPROVED",      5,  6, 90.0, 1982, 408, 480000),
    ("2025-08-01", "ANG MO KIO", "4 ROOM", "IMPROVED",     11, 12, 90.0, 1982, 396, 500000),
    ("2025-08-01", "ANG MO KIO", "3 ROOM", "NEW GENERATION",1,  3, 67.0, 1979, 420, 350000),
    ("2025-08-01", "BISHAN",     "4 ROOM", "IMPROVED",      4,  6, 95.0, 1987, 360, 650000),
    ("2025-07-01", "BISHAN",     "3 ROOM", "NEW GENERATION",1,  3, 67.0, 1986, 420, 520000),
]

@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a throwaway SQLite DB with minimal schema + seed rows."""
    db = tmp_path / "hdb_test.db"
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.executescript(SCHEMA)

    # towns
    towns = sorted({r[1] for r in SEED_ROWS})
    for t in towns:
        cur.execute("INSERT INTO town(name) VALUES (?)", (t,))
    town_ids = {t: cur.execute("SELECT id FROM town WHERE name=?", (t,)).fetchone()[0] for t in towns}

    # rows
    for m, town, ftype, model, lo, hi, area, lcy, rem, price in SEED_ROWS:
        cur.execute(
            """INSERT INTO resale_transaction
               (month, town_id, block, street_name, flat_type, flat_model,
                storey_low, storey_high, floor_area_sqm, lease_commence_year,
                remaining_lease_months, resale_price)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (m, town_ids[town], "101", "TEST ST", ftype, model, lo, hi, area, lcy, rem, price),
        )
    con.commit(); con.close()
    return db

@pytest.fixture
def patch_llm_tools_db(monkeypatch, temp_db):
    """Point LLM.tools.DB_PATH at temp DB and clear any cached premiums."""
    import LLM.tools as tools
    monkeypatch.setattr(tools, "DB_PATH", temp_db, raising=False)
    try:
        tools._floor_premiums.cache_clear()
    except Exception:
        pass
    return tools

@pytest.fixture
def router_mod(monkeypatch, temp_db):
    """Reload LLM.router with DB_PATH pointing to temp DB so vocab (towns/flats) refreshes."""
    import LLM.router as router
    monkeypatch.setattr(router, "DB_PATH", temp_db, raising=False)
    import importlib
    router = importlib.reload(router)
    return router

@pytest.fixture
def stub_models(monkeypatch):
    """
    Prevent auto-training and use tiny stub models for ml.infer.
    Predict returns a simple linear function of area so we can assert monotonicity.
    """
    import ml.infer as infer

    class DummyModel:
        def predict(self, X):
            return (X["floor_area_sqm"].astype(float) * 5000.0).to_numpy()

    monkeypatch.setattr(infer, "_ensure_models", lambda: None, raising=False)
    monkeypatch.setattr(infer, "Q50", DummyModel(), raising=True)  # central model
    monkeypatch.setattr(infer, "MEAN", None, raising=True)
    monkeypatch.setattr(infer, "Q10", None, raising=True)
    monkeypatch.setattr(infer, "Q90", None, raising=True)
    return infer

@pytest.fixture
def tmp_meta(tmp_path: Path):
    """Create a models dir + empty/lean model_meta.json for drift tests."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    meta = models_dir / "model_meta.json"
    meta.write_text(json.dumps({
        "train_start": "2024-01-01",
        "train_end": "2025-06-30",
    }, indent=2))
    return meta, models_dir
