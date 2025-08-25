# LLM/writer.py
import json
from LLM.config import generate, MAX_NEW_TOKENS

WRITER_PROMPT = """You write a concise answer from structured tool data.
Rules:
- If 'low_supply' was used, state it's a proxy via low resale volume (BTO-launch data not available).
- If price table is present, mention discount & affordability assumptions briefly.
- Bullet points plus a short paragraph. No invented numbers.
DATA:
"""

def llm_write(data: dict, user_msg: str) -> str:
    j = json.dumps(data, ensure_ascii=False)
    prompt = WRITER_PROMPT + j + "\nUser: " + user_msg + "\nAnswer:"
    return generate(prompt, max_new_tokens=MAX_NEW_TOKENS).strip()
