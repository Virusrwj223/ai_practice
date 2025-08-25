# app/streamlit_app.py
import sys
from pathlib import Path

# --- ensure project root is importable BEFORE any local imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from ml.infer import CONF as FINCONF
from LLM import run_agent  # uses your refactored agent

st.set_page_config(page_title="HDB Price LLM (PoC)", page_icon="🏠", layout="wide")
st.title("HDB Price LLM (PoC)")

with st.sidebar:
    st.markdown("**Finance assumptions**")
    st.json(FINCONF, expanded=False)
    show_diag = st.checkbox("Show diagnostics", value=False)

query = st.text_area(
    "Ask:",
    value="Recommend towns with limited launches for 4-room flats, and include low/mid/high price estimates."
)

if st.button("Run"):
    res = run_agent(query)
    if show_diag:
        st.subheader("Route")
        st.code(res["route"])

    data = res["data"]
    if isinstance(data, dict) and data.get("tool") == "price_estimates" and "rows" in data:
        st.subheader(f"{data['town']} / {data['flat_type']} — {data['month']}")
        rows = pd.DataFrame(data["rows"])
        st.dataframe(rows[["band","resale_pred","bto_proxy","required_income"]])

    if isinstance(data, dict) and data.get("tool") == "low_supply" and "items" in data:
        st.subheader("Low-supply proxy (lowest resale volume)")
        st.dataframe(pd.DataFrame(data["items"]))

    # st.markdown(res["answer"])

