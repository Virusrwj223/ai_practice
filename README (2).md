# HDB Price Intelligence — Proof of Concept

A lightweight system that combines classic ML with a small LLM “agent” to analyze Singapore HDB resale data and produce price estimates, BTO proxy prices, and required household income. It runs entirely locally and includes telemetry, drift checks, and tests.

This Proof of Concept (POC) is entirely self-contained for minimal setup requirements.

---

## Setup

### 1) Prerequisites

- Python 3.11

### 2) Create and activate a virtualenv

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

- Main page: chat-style UX over the LLM agent.
- Admin page: View telemetry & drift.

---

## User Stories

- As a housing planner, I want to identify towns with historically limited launches for a given flat type so that I can prioritize areas for future supply planning.

- As a market analyst, I want price estimates for a specific town, flat type, and month so that I can benchmark affordability over time.

- As a prospective home buyer, I want the required household income for a BTO-like proxy price so that I can quickly assess budget and eligibility.

---

## Non-functional requirements

- Able to run on a typical laptop
- No user access control or security

---

## Architecture (high level)

The POC uses a clinet service architecture. It can be thought of as having a frontend (Streamlit), an LLM service, an ML service and a database. The frontend requests and responds with the LLM service. The LLM service requests and responds to the ML service. Both the LLM service and ML service uses the data from the database.

---

## Engineering Concepts / Design Choices

### Separation of Concerns (SoC)

- **UI** (Streamlit) is thin.
- **Agent** routes and writes text, not numbers.
- **Tools** compute numbers deterministically.
- **ML** owns training/inference; **DB** owns data; **monitoring** owns observability.

### A2A (Agents-as-APIs)

- Tools (`t_price_estimates`, `t_low_supply`) are **pure functions** with **stable JSON contracts**.
- Any client can call them; we can expose the same contracts via REST later without code changes.

### MCP Awareness

- The tools’ JSON signatures map cleanly to an MCP server in future. The LLM “client” could be MCP-compatible with minimal glue.

### Why SQL (SQLite) over NoSQL

- The dataset is **highly tabular** with joins/aggregations (town, flat_type, months).
- **SQLite** is trivial to set up locally, performant for this scale, and supports indices and constraints.

### Determinism over LLM randomness

- **Deterministic parser** pulls `town`, `flat_type`, `month` from user text using DB vocab; HF LLM is a fallback only.
- Numbers are produced by tools and ML, never invented by the LLM.

### Robustness

- `ml/infer.py` **auto-trains on first call** if models are missing (uses one-step LightGBM).
- `monitoring/telemetry.py` captures runtime health without extra infra.
- `monitoring/drift.py` computes a training reference if missing (self-healing meta).

---

## ML: What & Why

- **Model:** LightGBM regressor wrapped in an sklearn `Pipeline` with:
  - **Feature engineering:**
    - `storey_mid` = average of `storey_low/high`
    - `flat_age` = `year(month) - lease_commence_year` (clipped ≥ 0)
    - `remaining_lease_years` = months/12
  - **Preprocessing:** OneHotEncoder for `town`, `flat_type`, `flat_model`; numeric passthrough.
- **PoC speed choice:** `n_estimators=1` (single tree) → **training is instant**; adequate for PoC wiring and demos.
- **Quantile models:** optional q10/q50/q90 for intervals (kept but trained as one-step too).
- **Backtest:** rolling month-by-month split for sanity; summary saved as `models/backtest_mean.csv`.
- **Metadata:** `models/model_meta.json` includes training window and **reference stats** for drift.

---

## LLM Service: What & Why

- **Routing:** `LLM/router.py` tries a **deterministic parser** first (regex + fuzzy match to DB vocab for `town`, `flat_type`, `month`). Falls back to HF model (`flan-t5-small`) if needed.
- **Tools:**
  - `t_price_estimates` — computes estimates for **low/mid/high** bands. It calls ML once and applies **data-driven floor premiums** derived from the last 24 months in that town & flat type.
  - `t_low_supply` — a proxy for “limited BTO launches” using **low resale volume** over N years.
- **Writer:** `LLM/writer.py` produces a concise explanation from tool outputs. The LLM **never** fabricates numbers.
- **Why HF model:** free & tiny (CPU-friendly), just for light orchestration text. All numeric work stays deterministic.

---

## Quality Control (GitHub Actions)

Example CI pipeline (summary):

- **Checkout**
- **Setup Python**, install deps
- **Create DB** (`python data/put_data_to_db.py`)
- **Train** one-step model (`python -m ml.train`)
- **Run pytest** suite (fast, no network/model downloads needed)

This ensures a clean machine can **reproduce the build**, **train**, and **pass tests** end-to-end.

---

## 🧪 Testing Strategy

### Heuristics used

- **Boundary Value Analysis (BVA):**
  - `storey_low == storey_high`
  - negative `flat_age` clipped to 0
  - `top_k=1` in `t_low_supply`
- **Equivalence Partitioning (EP):**
  - single dict vs list input to `predict`
  - `t_low_supply` with and without `flat_type` filter
  - router: “price” vs “low-supply” intents

### Black-box vs White-box

- **Black-box:** router intent detection, tool outputs, full predict path (stub models).
- **White-box:** feature engineering invariants (monotonicity, clipping), drift computation internals.

### Unit tests (highlights)

- `tests/test_features.py` — FE correctness (BVA/EP).
- `tests/test_infer_predict.py` — predict returns required keys; affordability math; monotonicity with stub.
- `tests/test_tools_price.py` — tool contract & **distinct bands** via floor premiums.
- `tests/test_tools_low_supply.py` — filtering & top-k boundary.
- `tests/test_router_deterministic.py` — argument extraction from natural language.
- `tests/test_telemetry.py` — logs written to SQLite.
- `tests/test_drift.py` — reference auto-generation & drift outputs.

All tests use **temporary SQLite** DBs and **stubs** (no internet, no heavy training).

Run:

```bash
python -m pytest -q
```

---

## Observability

- **Telemetry** (`logs/telemetry.db`):
  - `tool_calls`: latency, errors
  - `router_events`: JSON-parse health & chosen tool
  - `predictions`: values + model_version
- **Admin page** shows:
  - latency averages/p95 by tool
  - latest errors
  - recent prediction distributions
  - drift snapshot
- **Drift** compares **latest month** vs training reference (numeric PSI proxy + categorical share deltas). Reference is created at training or on the fly if missing.

---

## Future Work

### Product/ML

- Train proper multi-tree LightGBM; tune features; add **calibrated prediction intervals**.
- Bring in **BTO launch** datasets; replace “low resale volume” proxy with real signals.
- Data validation (e.g., **Great Expectations**), schema contracts, and richer drift (KS tests, PSI per feature bins).

### LLM/Agents

- Swap in a stronger small model for routing (e.g., instruction-tuned 0.5–1B).
- Expose tools via **MCP server** or REST to enable IDE and external agent integrations.

### Deployment

- **FastAPI** service for `/predict` and `/recommend` endpoints.
- Containerize and deploy to **Azure** (App Service/Container Apps + Azure Database for PostgreSQL).
- Optional **Azure ML** or AKS for model hosting; add a **model registry**.

> This PoC **does not deploy to Azure** to avoid ongoing infrastructure costs for the interview setting. Everything runs locally to keep the footprint small and reproducible.

### Observability at scale

- Export telemetry to **OpenTelemetry**, ship to **Prometheus/Grafana** or Azure Monitor.
- Alerts on latency/error rates and drift thresholds.

### CI/CD

- Add coverage gates; pre-commit hooks (ruff/black); environment matrix (Windows/Linux).

---

## 📂 Project Structure

```
data/
  resale.csv
  put_data_to_db.py
db/
  schema.sql
  hdb.db
ml/
  train.py          # one-step LightGBM + backtest + model_meta
  infer.py          # loads models, affordability math, auto-train if missing
LLM/
  config.py         # HF pipeline cache + generate()
  router.py         # deterministic + LLM routing to tools
  tools.py          # price_estimates, low_supply (+ floor premiums)
  writer.py         # concise narrative from tool results
  agent.py          # route → tool → write (single entry point)
app/
  streamlit_app.py  # main UI
  pages/01_Admin.py # telemetry + drift dashboard
monitoring/
  telemetry.py      # logs to logs/telemetry.db
  drift.py          # drift vs training reference
models/
  ...joblib files + model_meta.json
logs/
  telemetry.db
tests/
  ...pytest files
```
