# tests/test_tools_low_supply.py
def test_low_supply_all_and_filtered(patch_llm_tools_db):
    tools = patch_llm_tools_db
    out_all = tools.t_low_supply(last_n_years=2, flat_type=None, top_k=5)
    assert out_all["tool"] == "low_supply"
    assert len(out_all["items"]) >= 1

    out_ft = tools.t_low_supply(last_n_years=2, flat_type="4 ROOM", top_k=5)
    assert all(item["flat_type"].upper() == "4 ROOM" for item in out_ft["items"])

def test_low_supply_topk_boundary(patch_llm_tools_db):
    tools = patch_llm_tools_db
    out = tools.t_low_supply(last_n_years=2, flat_type=None, top_k=1)  # boundary
    assert len(out["items"]) == 1
