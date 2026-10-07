# decision-flow

An evaluation harness that measures the trade-offs of extracting decision handling in
enterprise systems from a **text-generation LLM** to a **Decision Model** (a model that
returns structured decisions), under **identical data, task definition, execution
conditions, and metrics**.

Methods compared:

| Method | Implementation | Probability handling |
|---|---|---|
| Rule-based | `src/adapters/rule.py` (keyword rules) | none (0/1 risk signal only) |
| General LLM | `src/adapters/llm.py` (prompt mode / json_schema-enforced mode) | self-reported values (**not calibrated**) |
| Decision Model | `src/adapters/mercury.py`, `pplx_decider.py`, `jev.py` | officially defined probabilities (**calibration target**) |
| Hybrid | `src/hybrid/pipeline.py` (rule → decision model → threshold → auto/review/block) | the above probabilities |

The harness does not assume Decision Models are "fast, cheap, and safe". It only reports
what the measured numbers show.

English | [日本語](README.ja.md)

## Setup

Python 3.11+. Dependencies are split into extras.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"          # pytest / numpy / scikit-learn
uv pip install -e ".[embed]"        # to use local embeddings (fastembed)
uv pip install -e ".[jev]"          # to use the Jev SDK (optional)
```

API keys go in `.env` (git-ignored). Resolution order: **process environment variables →
repo-root `.env`**.

```bash
cp .env.example .env   # LLM_API_KEY / DECIDER_API_KEY / TYPESAFE_API_KEY / PERPLEXITY_API_KEY
```

Key values are never written to code, logs, results, or reports (use `key_status()` to
check existence only).

## Running

Methods that call external APIs (`mercury` / `llm_prompt` / `llm_json_schema`) need a
route in `config/local.toml` (see *Configuration* below). Locally-complete methods
(`rule` / `embedding_lr`) run as-is.

```bash
# 1) Run all methods under identical conditions (smoke on synthetic data)
.venv/bin/python -m runners.benchmark \
  --dataset data/synthetic/support_classification.jsonl \
  --adapters rule,embedding_lr,mercury,llm_prompt,llm_json_schema,pplx_decider \
  --train-on data/synthetic/support_classification.jsonl

# 2) Choose thresholds on calib (thresholds are never searched on test)
.venv/bin/python -m runners.calibrate --calib data/calib/support_classification.jsonl

# 3) Recompute metrics from raw responses and emit a report
.venv/bin/python -m runners.report --dataset data/test/support_classification.jsonl \
  --calib-raw-dir results/raw_calib --calib-dataset data/calib/support_classification.jsonl \
  --hybrid results/hybrid_pplx_decider_A_rulecommit.json,results/hybrid_pplx_decider_B_nocommit.json

# 3b) Evaluate the Hybrid alone (name the layer-2 decision model; --no-rule-commit
# stops layer-1 immediate commits)
.venv/bin/python -m runners.hybrid_eval --dataset data/test/support_classification.jsonl \
  --raw-dir results/raw --model-method pplx_decider --no-rule-commit

# 3c) Restore calib raw responses from cache (a safety net when calib/test output
# names collide)
.venv/bin/python -m runners.recover_raw --dataset data/calib/support_classification.jsonl \
  --out results/raw_calib

# 4) Jev dry run (no API call; shows the request shape and estimated cost)
.venv/bin/python -m runners.jev_dry_run --token-estimate 300 --out results/jev_dry_run.json

# 5) Jev production run (confirm first; --allow-paid-models is mandatory)
.venv/bin/python -m runners.benchmark --dataset data/test/support_classification.jsonl \
  --adapters jev --allow-paid-models

# 6) Shadow mode (diff against existing decisions)
.venv/bin/python -m runners.shadow --existing existing_decisions.jsonl \
  --model-raw results/raw/support_classification__jev.jsonl

# 7) Stability (same-input repeats + paraphrase / option-order variants)
.venv/bin/python -m runners.stability --subset 50 --runs 20 --mode runs
.venv/bin/python -m runners.stability --subset 50 --mode variants

# 8) Throughput measurement (identical concurrency; unique run_index to bypass cache)
.venv/bin/python -m runners.throughput --n 60 --concurrency 12 \
  --methods llm_prompt,llm_json_schema,pplx_decider

# 9) Recalibration (learn on calib → evaluate on test)
.venv/bin/python -m runners.recalibrate

# 10) Assemble the final report (results/report.md, 8 sections)
.venv/bin/python -m runners.report --dataset data/test/support_classification.jsonl \
  --raw-dir results/raw --calib-raw-dir results/raw_calib \
  --hybrid results/hybrid_pplx_decider_A_rulecommit.json,results/hybrid_pplx_decider_B_nocommit.json \
  --throughput results/throughput.json --out results/report_test_partial.md
.venv/bin/python -m runners.final_report

# Tests
.venv/bin/python -m pytest -q
```

## Configuration (`config/default.toml`)

Local overrides go in `config/local.toml` (git-ignored) and are deep-merged.

| Section | Contents |
|---|---|
| `run` | concurrency, timeout, retry policy, seed (identical execution conditions) |
| `evaluation` | high-risk label/severity, recall target, bootstrap settings, stability N |
| `free_tier` | shared free-tier daily/per-minute limits and exhaustion behavior |
| `budget` | estimated cost caps for paid models (Perplexity $0.5 / Jev $1), external-send confirmation |
| `models.*` | model IDs, endpoints, unit prices. **Never hardcoded in code.** `models.llm` routes are swappable via `provider`/`endpoint`/`key_env`/`tier` |
| `embedding` | local embedding model (a fastembed-supported one) |

### General-LLM route (swapped by configuration)

`models.llm` works with **any OpenAI-compatible endpoint**. Model IDs, endpoints, and key
names are never hardcoded — set them per environment in `config/local.toml` (git-ignored):

```toml
[models.llm]
provider = "openai-compatible"
model = "<model id you use>"
endpoint = "<OpenAI-compatible chat/completions endpoint>"
key_env = "LLM_API_KEY"
tier = "free"   # when the route consumes the shared free-tier counter with mercury
```

Keys go in `.env` (git-ignored; see `.env.example` for the format). `tier = "external_free"`
means "an external API that does not consume mercury's shared free-tier counter" — daily and
per-minute limits do not apply.

For third-party independence, the generation model and the evaluation LLM are kept on
**separate lineages** (the lineages themselves are managed in `config/local.toml`).

Task definitions live in `config/tasks/*.toml` (categories, descriptions, risk_label, task
instruction) — swappable without touching code. Rule patterns live in `config/rules/*.toml`.
Thresholds are written to `config/thresholds.json` by `runners.calibrate`.

## Data format

JSONL (1 line = 1 item). `label` and `severity` are **independent columns**.

```json
{"id": "sc-001", "text": "inquiry body", "label": "urgent_claim", "severity": "high",
 "language": "ja", "ambiguous": false, "annotator_labels": ["urgent_claim", "billing"]}
```

Each dataset directory has a `manifest.json` (`source` / `license` / `data_class` /
`external_ok` / `provenance`). Sending `external_ok=false` (real data) to an external API
requires `--confirm-external`. Details: `data/schema.md`.

| Directory | Purpose |
|---|---|
| `data/synthetic/` | smoke tests only (never used in reports) |
| `data/calib/` | threshold selection and recalibration only |
| `data/test/` | final evaluation only (never used for tuning) |

## Cost guards and safety valves

- **`--allow-paid-models`**: mandatory for Jev (prevents accidental runs). Perplexity is
  controlled by the budget cap only.
- **Budget caps**: `budget.perplexity_usd` (default $0.5) / `budget.jev_usd` (default $1).
  Stops when the estimate would exceed them. Costs are **estimates** from API-returned token
  counts × published unit prices — not actual bills.
- **`--confirm-external`**: mandatory when sending `external_ok=false` data to an external API.
- **Free tier**: mercury's shared free-tier counter is managed in `free_tier`. A method that
  hits `daily_limit` is parked in pending; other methods, local computation, and reports keep
  running.
- **Cache**: stored in `results/cache/` keyed by (method, model, task, item, run index,
  instruction hash) — enables resuming across days and prevents mix-ups after instruction edits.
- **Audit log**: every request's input, probabilities, request ID, UTC timestamp, and model ID
  go to `results/audit/`.

## Layout

```
config/     default.toml, tasks/, rules/, local.toml (optional), thresholds.json (generated)
data/       synthetic/ calib/ test/, schema.md
src/
  tasks/      task definitions (categories and instructions)
  adapters/   base, rule, embedding_lr, llm, mercury, pplx_decider, jev, registry
  hybrid/     rule → decision model → threshold → auto/review/block
  evaluation/ classification, calibration, risk_coverage, bootstrap
  runners/    benchmark, calibrate, report, final_report, hybrid_eval, stability, throughput,
              recalibrate, recover_raw, jev_dry_run, shadow
  common/     env, config, dataset, http, ratelimit, budget, cache
results/    raw/ (test), raw_calib/ (calib), aggregate/, audit/, cache/, pending/, report.md
tests/
```

**Always write calib and test raw responses to separate directories** (`--raw-dir`). Output
file names are derived from the dataset name, so identical names overwrite each other (this
happened once). `runners.recover_raw` can restore from cache.

## Sources

API contracts, limits, and prices are recorded in code docstrings with source URLs from the
official documentation of each evaluated model (route details are managed in
`config/local.toml`). Unverified specs are never guessed; they are noted in the report's
"Constraints and caveats" section.
