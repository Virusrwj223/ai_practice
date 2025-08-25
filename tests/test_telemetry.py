# tests/test_telemetry.py
import sqlite3

def test_telemetry_logging(tmp_path, monkeypatch):
    # Patch telemetry DB to a temp file
    from monitoring import telemetry as t
    monkeypatch.setattr(t, "DB", tmp_path / "telemetry.db", raising=False)
    t._init()

    t.log_tool("price_estimates", {"town":"ANG MO KIO"}, True, 12.3, None)
    t.log_prediction("ANG MO KIO","4 ROOM","mid", 480000, 384000, 4200, "v0")
    t.log_router(True, "price_estimates", '{"tool":"price_estimates"}', None)

    con = sqlite3.connect(t.DB)
    n_tool = con.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    n_pred = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_router = con.execute("SELECT COUNT(*) FROM router_events").fetchone()[0]
    con.close()

    assert n_tool == 1 and n_pred == 1 and n_router == 1
