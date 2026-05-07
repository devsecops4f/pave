# PAVE

Root Cause Analysis agent using LangGraph, evaluated on [`anon-ops/ops-lite`](https://huggingface.co/datasets/anon-ops/ops-lite).

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure LLM credentials
cat >> .env <<'EOF'
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=<your-model>
UTU_LLM_BASE_URL=<your-api-base-url>
UTU_LLM_API_KEY=<your-api-key>
EOF

# 3. Pull the dataset (~3.8 GB, 500 cases)
hf download anon-ops/ops-lite --repo-type dataset \
  --local-dir ./data/ops_lite

# 4. Seed the eval database
LLM_EVAL_DB_URL=sqlite:///./eval.db \
  uv run python scripts/seed_v3_db.py --root ./data/ops_lite
```

## Run evaluation

```bash
uv run rca llm-eval run config/eval/ops_lite_smoke.yaml -a pave
```

Remove `max_samples: 1` from `config/eval/ops_lite_smoke.yaml` (or pass `-l <N>`) to run more than one case. Add `--dashboard --dashboard-port 8766` for a real-time dashboard.

## Format check (no LLM cost)

Verify dataset layout and DB rows without spending tokens:

```bash
LLM_EVAL_DB_URL=sqlite:///./eval.db \
  uv run python scripts/check_preprocess.py config/eval/ops_lite_smoke.yaml
```

## Local dataset layout

If the dataset is already on disk, point `--root` at the directory containing `manifest.jsonl` and `cases/`:

```bash
LLM_EVAL_DB_URL=sqlite:///./eval.db \
  uv run python scripts/seed_v3_db.py --root /path/to/ops_lite
```
