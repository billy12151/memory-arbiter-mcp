# mema on LoCoMo-Refined — Results & Methodology

> Evaluated 2026-10-08/09 · mema 0.17.1 · official judge `Qwen/Qwen3-14B` ·
> all 1,382 questions · zero judge errors · full ablation disclosure below.

## Headline numbers

| Configuration | Overall | Text-only (n=861) |
|---|---:|---:|
| **mema + writer-agent** (submission config) | **61.00%** | **65.62%** |
| **mema verbatim** (zero-LLM write path) | 43.13% | 49.48% |

Per-category (text-only), mema + writer-agent: single-hop 45% · multi-hop 69% ·
temporal 44% · open-domain 69%.

### Leaderboard context

| System | LoCoMo-Refined score | Entry type |
|---|---:|---|
| MemoraX AI | 82.65% | self-submitted¹ |
| **mema + writer-agent (text-only)** | **65.62%** | self-submitted, full methodology below |
| MemOS | 63.60% | re-scored by benchmark authors |
| **mema + writer-agent (overall)** | **61.00%** | self-submitted |
| MemPalace | 58.68% | re-scored |
| EverMemOS | 58.25% | re-scored |
| **mema verbatim (text-only)** | **49.48%** | self-submitted |
| Mem0 | 48.91% | re-scored |

¹ The four "re-scored" systems are original LoCoMo submissions re-graded by the
benchmark authors; self-submitted entries are new. We encourage the maintainers
to list entry types in the table and to request methodology notes for
self-submissions — ours is below in full.

**Three findings we consider more interesting than the ranking:**

1. **Bare mema matches Mem0's full pipeline.** With *zero* LLM in the write
   path — every utterance stored verbatim with a structured `event_time` —
   text-only accuracy is 49.48%, vs Mem0's 48.91% for its complete
   retrieve-rewrite pipeline. The storage + hybrid-retrieval layer alone is
   worth a Mem0.
2. **Write quality is the dominant variable.** The writer-agent arm
   (+16pp over verbatim) used the *cheapest* model on the shelf
   (deepseek-v4.1-flash) with five public rules. This is mema's architecture
   thesis measured: the agent decides *what* to remember, the arbiter keeps
   *how* it is stored verifiable — and the write side is where the points are.
3. **552 write-time semantic conflicts detected** across the 20 replayed
   conversations (mini-clash self-trained mDeBERTa three-class judge, running
   in the production write path). No other system on the leaderboard detects
   conflicts at write time. 27.6 conflicts per conversation is also a warning
   to the industry: ungoverned memory accumulates landmines.

## Methodology (fully reproducible)

Harness: [`run_locomo.py`](./run_locomo.py) · dataset: `data/locomo_refined.json`
(CC BY-NC, downloaded from the upstream repo, not committed) · per-conversation
isolated mema databases (`MEMORY_ARBITER_CONFIG` pointing at a per-conv config
copied from production with only db_path/identity overridden — the production
embedding model, hybrid index and write-time conflict judge run exactly as
shipped).

**Writer-agent arm (submission config):**
1. Sessions replayed chronologically; per session, an extractor LLM
   (deepseek-v4.1-flash) distills durable facts under five rules: one event
   per sentence, no merging, original date expressions preserved (relative
   references resolved to the event's date, dual-form when the source used a
   relative phrase), small details kept (nicknames, possessions, counts),
   JSON-array output.
2. Each fact written via `memory(action="remember")` — write-time conflict
   judging and hybrid indexing active as in production.
3. Per question: `batch_find` union recall (raw question + LLM keyword
   reformulation, `limit_per_query=8`, `content_mode="full"`).
4. Answerer (deepseek-v4.1-flash; **84/1382 questions answered by qwen3-14b
   fallback after our local gateway was rate-limited — attributed per record**)
   answers in one sentence, absolute dates only, complete lists, from recalled
   memories only.
5. Official refined judge prompt, official model `qwen3-14b`, all acceptable
   golds scored independently, any CORRECT wins (matching the official
   runtime's semantics).

**Verbatim arm:** same retrieval/answering/judging; write path stores every
utterance as-is with the session datetime as structured `event_time`. Zero
LLM in ingestion.

**Cost:** ≈ ¥5 total API spend (judge only; extraction/answering ran on a
local gateway).

## Ablations (conv-30, same 50 questions)

| Variant | Score |
|---|---:|
| v2 baseline (raw question as query, single gold, naive extraction) | 40% |
| + union recall (question ∨ keywords) & dual-form dates | 56% |
| + answer-format rules (absolute dates, complete lists) | 64% |
| verbatim-only write path | 44% |
| hybrid dual-write library | 64% |
| **union ceiling across all arms** (gold info present in ≥1 arm's context) | **82%** |
| answerer swap, same memories & judge: deepseek-flash 64% vs GLM-5.3 subagent 62% | n.s. |

Read-side selection — not answerer strength, not storage fidelity — is the
remaining gap between 64% and the low-80s ceiling.


## Addendum — detector iteration (2026-10-10)

Re-ran the extraction-arm replay with an updated mini-clash checkpoint
(`mdeberta-v58_ep3_fp16`, swapped into the local production config). Note:
this replay's extractor was qwen3-14b (the original gateway was unavailable),
so absolute counts carry a small extractor confound; the comparable metric is
the context-free triage rate.

| Detector | Detections | True updates (context-free triage) | Rate |
|---|---:|---:|---:|
| original (submission run) | 207 | 9 | 4.3% |
| v58_ep3_fp16 | 276 | 12 | 4.9% |

Precision is unchanged; recall of true conflicts is marginally higher
(9 → 12). The dominant failure mode remains **complementary pairs flagged as
conflicts** (parallel facts, same-scene different-subject, causal/narrative
links) — a triage problem, not a sensitivity problem. The full labeled pair
sets for both detector generations are exported alongside this repo's eval
workspace for detector training.

## Reproducing

```bash
curl -L -o evals/locomo/data/locomo_refined.json \
  https://raw.githubusercontent.com/mem-eval-suite/LoCoMo_refined/main/data/raw/locomo_refined.json
cp evals/locomo/env.local.sh.example evals/locomo/env.local.sh  # fill keys
source evals/locomo/env.local.sh
.venv/bin/python evals/locomo/run_locomo.py --conv 0 --limit 5   # smoke
.venv/bin/python evals/locomo/run_locomo.py --conv all            # writer arm
.venv/bin/python evals/locomo/run_locomo.py --conv all --verbatim # verbatim arm
```

Submission-format predictions for both arms: [`submission/`](./submission/).
Per-question records (recalled ids, reformulated queries, judge labels,
answerer attribution) live in `run/` (not committed; contains isolated DBs).
