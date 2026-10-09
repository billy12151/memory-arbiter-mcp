# mema × LoCoMo-Refined

Benchmark mema (memory-arbiter) on [LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined)
(1,382 questions over 10 long multi-session conversations; official judge `Qwen3-14B`).

## Methodology (report these alongside any number)

- **Pipeline per conversation** (each in its own isolated mema database — never
  the developer's production DB):
  1. Sessions are replayed chronologically. An extractor LLM distills durable
     facts per session (the same job a host Agent does before writing), and each
     fact is written via `memory(action="remember", workspace="locomo")` —
     write-time conflict judging (mini-clash mDeBERTa) and hybrid indexing run
     exactly as in production.
  2. Each question is answered from `memory(action="find", content_mode="full",
     limit=10)` recall only — the answerer sees recalled memories and nothing
     else.
  3. Predictions are scored with the **official refined judge prompt** (copied
     verbatim from `official_llm_judge.py`) using the official judge model
     `qwen3-14b` via an OpenAI-compatible endpoint.
- **Metrics**: judge accuracy (CORRECT fraction), reported overall, text-only
  (non-multimodal), and per question category (raw dataset categories 1–4).
- **Models**: judge = `qwen3-14b` (official); extractor + answerer configured
  via env — recorded in every results file.
- **Known simplifications (v1)**: writes do not pass structured `event_time`
  (session dates appear only inside extracted fact text); recall top-k = 10;
  one workspace per conversation.

## Layout

- `run_locomo.py` — the harness (replay → recall → answer → judge → summarize)
- `data/locomo_refined.json` — dataset (**not committed**: CC BY-NC 4.0;
  download from the upstream repo, see below)
- `official_*.py|.sh` — upstream scripts fetched verbatim for reference
- `run/` — per-conversation isolated DBs + predictions + `summary.json`
  (not committed)
- `env.local.sh` — API credentials template (not committed)

## Reproduce

```bash
cd <repo root>
curl -L -o evals/locomo/data/locomo_refined.json \
  https://raw.githubusercontent.com/mem-eval-suite/LoCoMo_refined/main/data/raw/locomo_refined.json

cp evals/locomo/env.local.sh.example evals/locomo/env.local.sh  # fill in keys
source evals/locomo/env.local.sh

# smoke: one conversation, 5 questions
.venv/bin/python evals/locomo/run_locomo.py --conv 0 --limit 5
# full run
.venv/bin/python evals/locomo/run_locomo.py --conv all
```

Results land in `evals/locomo/run/summary.json`.
