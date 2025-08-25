# tests/test_tools_price.py
import types

def test_price_estimates_structure_and_bands(patch_llm_tools_db, monkeypatch):
    tools = patch_llm_tools_db

    # stub ml.infer.predict used inside tools to a fixed central price
    def stub_predict(rec):
        return [{"resale_pred": 480000.0}]
    monkeypatch.setattr(tools, "price_predict", stub_predict, raising=True)
    tools._floor_premiums.cache_clear()

    out = tools.t_price_estimates("ANG MO KIO", "4 ROOM", month="2025-08")
    assert out["tool"] == "price_estimates"
    rows = out["rows"]
    assert len(rows) == 3
    bands = {r["band"] for r in rows}
    assert bands == {"low","mid","high"}
    # with premiums applied, at least one band should differ
    vals = {r["resale_pred"] for r in rows}
    assert len(vals) >= 2
    for r in rows:
        assert set(["resale_pred","bto_proxy","required_income","floor_premium_applied"]).issubset(r.keys())
        assert r["required_income"] > 0
