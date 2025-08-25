from LLM.tools import t_price_estimates
def test_price_tool_happy():
    data = t_price_estimates("ANG MO KIO","4 ROOM")
    assert "rows" in data and len(data["rows"]) >= 1
    assert "finance" in data
