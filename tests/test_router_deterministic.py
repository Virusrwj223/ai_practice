# tests/test_router_deterministic.py
def test_router_price_detects_town_flat_and_month(router_mod):
    router = router_mod
    q = "Show prices for 4 ROOM in Ang Mo Kio for 2025-08"
    r = router.llm_route(q)
    assert r["tool"] == "price_estimates"
    args = r["args"]
    # town/flat_type matched from DB vocab
    assert args["town"].upper() == "ANG MO KIO"
    assert args["flat_type"].upper() == "4 ROOM"
    assert args.get("month") in ("2025-08","2025-08-01")  # normalized to YYYY-MM

def test_router_low_supply_keyword(router_mod):
    router = router_mod
    q = "Recommend towns with limited BTO launches (low supply) for 4 room"
    r = router.llm_route(q)
    assert r["tool"] == "low_supply"
    assert r["args"].get("flat_type", "").upper() in ("4 ROOM","")  # may or may not detect
