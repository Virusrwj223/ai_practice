# LLM/agent.py
from typing import Callable, Dict
from LLM.router import llm_route
from LLM.writer import llm_write
from LLM.tools import t_price_estimates, t_low_supply

TOOL_REGISTRY: Dict[str, Callable[..., dict]] = {
    "price_estimates": t_price_estimates,
    "low_supply": t_low_supply,
}

def run_agent(user_text: str) -> dict:
    """Returns {"route": {...}, "data": {...}, "answer": "..."}"""
    route = llm_route(user_text)
    tool = route.get("tool", "final")
    args = route.get("args", {}) or {}

    if tool == "final":
        return {"route": route, "data": {"tool": "final", "args": args}, "answer": args.get("answer","")}

    fn = TOOL_REGISTRY.get(tool)
    if not fn:
        data = {"error": f"Unknown tool '{tool}'"}
        return {"route": route, "data": data, "answer": "Sorry, I can't handle that request yet."}

    data = fn(**args)
    # keep tool name in payload for the writer
    if isinstance(data, dict) and "tool" not in data:
        data["tool"] = tool

    answer = llm_write({"tool": tool, "result": data}, user_text)
    return {"route": route, "data": data, "answer": answer}
