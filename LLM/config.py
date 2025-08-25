# LLM/config.py
import os
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")  # tiny, free
MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))

@lru_cache(maxsize=1)
def get_llm_pipe():
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)
    return pipeline("text2text-generation", model=mdl, tokenizer=tok)

def generate(text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    pipe = get_llm_pipe()
    return pipe(text, max_new_tokens=max_new_tokens)[0]["generated_text"]
