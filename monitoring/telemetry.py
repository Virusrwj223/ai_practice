# monitoring/telemetry.py
import os, time, json, sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DB = LOGS_DIR / "telemetry.db"

def _init():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS tool_calls(
      ts TEXT, tool TEXT, args_json TEXT, ok INTEGER, ms REAL, err TEXT
    );
    CREATE TABLE IF NOT EXISTS router_events(
      ts TEXT, ok INTEGER, tool TEXT, raw_json TEXT, err TEXT
    );
    CREATE TABLE IF NOT EXISTS predictions(
      ts TEXT, town TEXT, flat_type TEXT, band TEXT, resale REAL, bto REAL,
      required_income REAL, model_version TEXT
    );
    """)
    con.commit(); con.close()
_init()

def log_tool(tool:str, args:dict, ok:bool, ms:float, err:str|None=None):
    con = sqlite3.connect(DB); cur = con.cursor()
    cur.execute("INSERT INTO tool_calls VALUES (?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(), tool, json.dumps(args), int(ok), ms, err))
    con.commit(); con.close()

def log_router(ok:bool, tool:str|None, raw:str, err:str|None=None):
    con = sqlite3.connect(DB); cur = con.cursor()
    cur.execute("INSERT INTO router_events VALUES (?,?,?,?,?)",
                (datetime.utcnow().isoformat(), int(ok), tool, raw[:2000], err))
    con.commit(); con.close()

def log_prediction(town, flat_type, band, resale, bto, income, model_version):
    con = sqlite3.connect(DB); cur = con.cursor()
    cur.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(), town, flat_type, band, resale, bto, income, model_version))
    con.commit(); con.close()

def timed(tool_name, args_snapshot:dict):
    def deco(fn):
        def wrap(*a, **kw):
            t0 = time.perf_counter()
            try:
                out = fn(*a, **kw)
                ms = (time.perf_counter() - t0) * 1000
                log_tool(tool_name, args_snapshot, True, ms, None)
                return out
            except Exception as e:
                ms = (time.perf_counter() - t0) * 1000
                log_tool(tool_name, args_snapshot, False, ms, repr(e))
                raise
        return wrap
    return deco
