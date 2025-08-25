# LLM/router.py
import json, re, difflib, sqlite3
from pathlib import Path
from LLM.config import generate, MAX_NEW_TOKENS

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "db" / "hdb.db"

def _vocab(query):
    con = sqlite3.connect(DB_PATH)
    towns = [r[0] for r in con.execute("SELECT DISTINCT name FROM town").fetchall()]
    flats = [r[0] for r in con.execute("SELECT DISTINCT flat_type FROM resale_transaction").fetchall()]
    con.close()
    return [t.upper() for t in towns], [f.upper() for f in flats]

TOWNS, FLATS = _vocab("x")

def _norm(s): return re.sub(r"[^A-Z0-9]+", "", s.upper())

def _guess_month(text):
    m = re.search(r"(20\d{2})[-/ ]?(\d{1,2})", text)
    if not m: return None
    y, mm = m.group(1), int(m.group(2))
    if 1 <= mm <= 12:
        return f"{y}-{mm:02d}"
    return None

def _best_match(candidates, text):
    if not text: return None
    tok = _norm(text)
    # exact-ish: try each candidate as substring match
    for c in candidates:
        if _norm(c) in tok or tok in _norm(c):
            return c
    # fuzzy fallback
    got = difflib.get_close_matches(text.upper(), candidates, n=1, cutoff=0.75)
    return got[0] if got else None

def _deterministic_route(user_text: str):
    month = _guess_month(user_text)
    # crude intent: "limited launch" or "low supply" -> low_supply tool
    if re.search(r"(limited|few|scarce).*(launch|bto|supply)|low\s*supply", user_text, re.I):
        flat = _best_match(FLATS, user_text)
        return {"tool":"low_supply", "args":{"last_n_years":10, "flat_type": flat}}

    town = _best_match(TOWNS, user_text)
    flat = _best_match(FLATS, user_text)
    if town or flat or month:
        args = {"town": town or "ANG MO KIO", "flat_type": flat or "4 ROOM"}
        if month: args["month"] = month
        return {"tool":"price_estimates", "args": args}
    return None

ROUTER_SPEC = """You are a router. Choose ONE tool or finish.
TOOLS:
- price_estimates(town:str, flat_type:str, month?:str)
- low_supply(last_n_years:int=10, flat_type?:str)
Return STRICT JSON only.
"""

def llm_route(user_text: str) -> dict:
    # 1) Try deterministic extraction first
    det = _deterministic_route(user_text)
    if det: return det
    # 2) Fall back to LLM routing
    prompt = ROUTER_SPEC + "\nUser: " + user_text + "\nJSON:"
    out = generate(prompt, max_new_tokens=MAX_NEW_TOKENS).strip()
    try:
        start, end = out.index("{"), out.rindex("}") + 1
        return json.loads(out[start:end])
    except Exception:
        # stable default
        return {"tool": "price_estimates", "args": {"town": "ANG MO KIO", "flat_type": "4 ROOM"}}
