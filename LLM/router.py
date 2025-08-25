# LLM/router.py
import json
from LLM.config import generate, MAX_NEW_TOKENS

ROUTER_SPEC = """You are a router. Choose ONE tool or finish.
TOOLS:
- price_estimates(town:str, flat_type:str, month?:str)
- low_supply(last_n_years:int=10, flat_type?:str)
Return STRICT JSON only:
{"tool":"price_estimates","args":{"town":"ANG MO KIO","flat_type":"4 ROOM","month":"2024-05"}}
or {"tool":"low_supply","args":{"last_n_years":10,"flat_type":"4 ROOM"}}
or {"tool":"final","args":{"answer":"..."}}"""

def llm_route(user_text: str) -> dict:
    prompt = ROUTER_SPEC + "\nUser: " + user_text + "\nJSON:"
    out = generate(prompt, max_new_tokens=MAX_NEW_TOKENS).strip()
    try:
        start, end = out.index("{"), out.rindex("}") + 1
        return json.loads(out[start:end])
    except Exception:
        # fallback default: price estimates with guessed args
        return {"tool": "price_estimates", "args": {"town": "ANG MO KIO", "flat_type": "4 ROOM"}}
