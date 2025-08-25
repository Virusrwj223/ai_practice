from LLM.router import llm_route
def test_router_price():
    r = llm_route("Price for 4-room in Ang Mo Kio with low/mid/high")
    assert r["tool"] in {"price_estimates","low_supply","final"}
