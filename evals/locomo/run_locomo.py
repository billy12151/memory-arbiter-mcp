"""LoCoMo-Refined benchmark harness for mema (memory-arbiter).

Pipeline per conversation (each in a fully isolated mema database):
  1. replay sessions chronologically -> extractor LLM distills durable facts
     -> tools.memory(action="remember") writes each fact (write-time conflict
     judging and hybrid indexing run exactly as in production)
  2. for each question -> tools.memory(action="find") top-k recall
     -> answerer LLM answers using ONLY recalled memories
  3. official LoCoMo-Refined judge (refined prompt, Qwen3-14B) scores the pair

Isolation: each conversation builds a fresh runtime via MEMORY_ARBITER_CONFIG
pointing at run/<sample_id>/config.json (copied from the production config
with db_path/backup/identity overridden). The production database is never
touched.

Usage:
  export EVALUATOR_API_BASE=...      # judge (official: qwen3-14b)
  export EVALUATOR_API_KEY=...
  export MEMA_EVAL_ANSWERER_BASE=... # extractor + answerer
  export MEMA_EVAL_ANSWERER_KEY=...
  export MEMA_EVAL_ANSWERER_MODEL=...

  .venv/bin/python evals/locomo/run_locomo.py --conv 0        # one conversation
  .venv/bin/python evals/locomo/run_locomo.py --conv all      # everything
  .venv/bin/python evals/locomo/run_locomo.py --conv 0 --limit 5   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "locomo_refined.json"
RUN_DIR = HERE / "run"
PROD_CONFIG = Path.home() / ".config" / "memory-arbiter" / "config.json"

WORKSPACE = "locomo"
RECALL_TOP_K = 15

# Official LoCoMo-Refined judge prompt, verbatim from
# mem-eval-suite/LoCoMo_refined src/llm_judge.py ("refined" judge).
JUDGE_PROMPT = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold’s content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., “the/last/previous/just prior”) as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C → C, C′), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```
"""

EXTRACT_PROMPT = """You are the memory-writer for an AI assistant. Below is one session of a long conversation between {a} and {b}, which took place on {when}.

Extract the durable facts worth remembering across sessions: events (with dates), preferences, relationships, plans, commitments, opinions. Also capture small but concrete details people mention: nicknames people use for each other, possessions, counts ("rejected twice"), and specific items. Write each fact as one self-contained sentence.

IMPORTANT — dates must be absolute AND attached to the event: resolve any relative time ("last year", "next month", "last night", "two weeks ago", "the Saturday before …") to the date the EVENT happened (not the date of this session), and state BOTH forms when the source used one: absolute date plus the relative phrase, e.g. "On 30 October 2022 (last night), ...", "in 2022 (last year)". Keep names, dates and amounts verbatim when present.

Session transcript:
{transcript}

Return a JSON array of strings (possibly empty). Return ONLY the JSON array."""

REFORM_PROMPT = """Turn this question into a short keyword query for searching a memory database. Rules: 5-10 words, lead with the person/entity names, then the topic nouns, drop all question words (what/when/where/does/did). Output ONLY the query.

Question: {question}

Query:"""

ANSWER_PROMPT = """You are answering a question about {a} and {b}, using ONLY the memories recalled below. Read every memory carefully before answering.

Rules:
- If any memory contains or clearly implies the answer, answer directly and confidently in ONE concise sentence.
- For time questions, state the absolute date or month/year exactly as the memory gives it. NEVER append relative-time parentheticals like "(next month)" or "(last year)" — absolute only.
- If the question asks for multiple items (which cities, which events, what items), include ALL items the memories mention, comma-separated.
- Only reply exactly "I don't have that in memory." when nothing recalled is relevant.

Recalled memories:
{memories}

Question: {question}

Answer:"""


# ---------------------------------------------------------------- LLM client

class LLM:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=180)

    def chat(self, prompt: str, max_tokens: int = 512) -> str:
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        # Qwen3 thinking mode would burn output tokens on
                        # reasoning and leave content empty — judge/answerer
                        # need a direct answer.
                        "enable_thinking": False,
                    },
                )
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                text = msg.get("content") or msg.get("reasoning_content") or ""
                # some gateways inline <think> blocks even when disabled
                if "</think>" in text:
                    text = text.split("</think>", 1)[1]
                return text.strip()
            except Exception as exc:  # noqa: BLE001 — retry any transport/API error
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


def extract_json_object(text: str) -> str:
    """First {...} or [...] JSON object in text (fence-stripped). Mirrors the
    official llm_judge_runtime.extract_json_object behavior, extended to arrays."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return stripped[start : i + 1]
    raise ValueError(f"no JSON object found in: {text[:200]!r}")


# ------------------------------------------------------------ mema runtime

def build_isolated_runtime(conv_dir: Path):
    """Fresh mema runtime whose DB lives inside conv_dir. Uses the production
    config as the base (embedding + conflict-judge settings identical), with
    db_path / backup / identity overridden. Never touches the production DB."""
    cfg = json.loads(PROD_CONFIG.read_text())
    cfg["db_path"] = str(conv_dir / "memory.sqlite3")
    cfg["backup_jsonl"] = str(conv_dir / "backup.jsonl")
    cfg["client"] = "locomo-eval"
    cfg["agent_id"] = "locomo-eval"
    eval_cfg = conv_dir / "config.json"
    eval_cfg.write_text(json.dumps(cfg, indent=2))
    os.environ["MEMORY_ARBITER_CONFIG"] = str(eval_cfg)
    # imported lazily so the env var is honored
    from memory_arbiter.server import build_runtime

    return build_runtime().tools


def remember(tools, content: str, session_tag: str) -> None:
    res = tools.memory(
        action="remember",
        data={
            "content": content,
            "subject": f"locomo {session_tag}",
            "workspace": WORKSPACE,
            "tags": ["locomo", "benchmark"],
            "source_type": "agent_generated",
        },
    )
    if not res.get("ok"):
        raise RuntimeError(f"remember failed: {json.dumps(res, default=str)[:300]}")


def recall(tools, question: str, reform_q: str | None = None) -> list[dict]:
    """Union recall: the raw question AND a keyword reformulation carry different
    strengths (question-form favors vectors, keywords favor FTS+keyword-boost) —
    batch_find merges both into one deduped page."""
    queries = [{"id": "q", "query": question}]
    if reform_q and reform_q != question:
        queries.append({"id": "r", "query": reform_q})
    res = tools.memory(
        action="batch_find",
        data={
            "queries": queries,
            "limit_per_query": 8,
            "workspace": WORKSPACE,
            "content_mode": "full",
        },
    )
    if not res.get("ok"):
        raise RuntimeError(f"batch_find failed: {json.dumps(res, default=str)[:300]}")
    return res.get("data", {}).get("results", [])


# --------------------------------------------------------------- pipeline

def conversation_sessions(conv: dict):
    """(session_tag, datetime, [(speaker, text), ...]) in chronological order."""
    import re

    keys = sorted(
        (k for k in conv if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    for k in keys:
        n = k.split("_")[1]
        when = conv.get(f"session_{n}_date_time", "unknown date")
        turns = [(t["speaker"], t["text"]) for t in conv[k]]
        yield f"session-{n}", when, turns


def parse_session_dt(text: str):
    """'1:56 pm on 8 May, 2023' -> ISO datetime, or None."""
    import re
    from datetime import datetime

    m = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Z][a-z]+),?\s+(\d{4})", text or "")
    if not m:
        return None
    hh, mm, ap, day, mon, year = m.groups()
    hh = int(hh) % 12 + (12 if ap == "pm" else 0)
    months = {m_: i + 1 for i, m_ in enumerate(
        "January February March April May June July August September October November December".split())}
    if mon not in months:
        return None
    return datetime(int(year), months[mon], int(day), hh, int(mm)).isoformat()


def replay_verbatim(tools, conv: dict, log) -> int:
    """Zero-extraction arm: every utterance stored verbatim with the session
    datetime as event_time — mema used exactly as its README documents."""
    written = 0
    for tag, when, turns in conversation_sessions(conv["conversation"]):
        dt = parse_session_dt(when)
        for speaker, text in turns:
            text = (text or "").strip()
            if not text:
                continue
            data = {
                "content": text,
                "subject": f"locomo {tag} turn",
                "workspace": WORKSPACE,
                "tags": ["locomo", "verbatim"],
                "source_type": "agent_generated",
            }
            if dt:
                data["event_time"] = dt
            res = tools.memory(action="remember", data=data)
            if not res.get("ok"):
                raise RuntimeError(f"verbatim remember failed: {json.dumps(res, default=str)[:200]}")
            written += 1
        log(f"  {tag} ({when}): {len(turns)} turns stored verbatim")
    return written


def replay(tools, conv: dict, extractor: LLM, log) -> int:
    a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
    written = 0
    # conversation_sessions expects the inner conversation dict
    for tag, when, turns in conversation_sessions(conv["conversation"]):
        transcript = "\n".join(f"{s}: {t}" for s, t in turns)
        raw = extractor.chat(
            EXTRACT_PROMPT.format(a=a, b=b, when=when, transcript=transcript),
            max_tokens=2048,
        )
        try:
            facts = json.loads(extract_json_object(raw))
            if not isinstance(facts, list):
                facts = []
        except (ValueError, json.JSONDecodeError) as exc:
            log(
                f"  {tag}: EXTRACT PARSE FAILED ({exc}); raw head: {raw[:200]!r}"
            )
            facts = []
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                remember(tools, f"{fact.strip()}", tag)
                written += 1
        log(f"  {tag} ({when}): {len(turns)} turns, raw {len(raw)} chars -> {len(facts)} facts")
    return written


def chat_with_fallback(primary: LLM, fallback: LLM | None, prompt: str, max_tokens: int, log) -> tuple[str, str]:
    """(text, model_used) — try the primary answerer; on failure fall back and
    attribute the answer to the fallback model (disclosed per record)."""
    try:
        return primary.chat(prompt, max_tokens=max_tokens), primary.model
    except Exception as exc:  # noqa: BLE001
        if fallback is None:
            raise
        log(f"    [answerer fallback] primary failed ({str(exc)[:80]}); using {fallback.model}")
        return fallback.chat(prompt, max_tokens=max_tokens), fallback.model


def answer_questions(tools, conv: dict, answerer: LLM, judge: LLM, out_path: Path, log, limit=None, fallback: LLM | None = None):
    a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
    done = 0
    mode = "a" if out_path.exists() else "w"
    seen_questions = set()
    if mode == "a":
        for line in out_path.read_text().splitlines():
            try:
                seen_questions.add(json.loads(line)["question"])
            except Exception:  # noqa: BLE001 — tolerate a torn last line
                pass
    with out_path.open(mode) as fh:
        for qa in conv["qa"]:
            q = qa["question"]
            if q in seen_questions:
                continue
            if limit is not None and done >= limit:
                break
            reform_raw, _ = chat_with_fallback(answerer, fallback, REFORM_PROMPT.format(question=q), 48, log)
            reform_q = reform_raw.strip().strip('"')
            hits = recall(tools, q, reform_q)
            def _line(h):
                et = str(h.get("event_time") or "")[:10]
                return f"- {h.get('content', '')}" + (f" [date: {et}]" if et else "")

            memories = (
                "\n".join(_line(h) for h in hits if h.get("content"))
                or "(nothing recalled)"
            )
            pred_raw, a_model = chat_with_fallback(
                answerer, fallback, ANSWER_PROMPT.format(a=a, b=b, memories=memories, question=q), 256, log
            )
            pred = pred_raw.strip()
            gold = qa["answer"]
            # official judge semantics: score against EACH acceptable gold,
            # any CORRECT wins (official_llm_judge_runtime._evaluate_async)
            label, judge_raw = "ERROR", ""
            for gold_str in gold if isinstance(gold, list) else [str(gold)]:
                try:
                    judge_raw = judge.chat(
                        JUDGE_PROMPT.format(
                            question=q, gold_answer=gold_str, generated_answer=pred
                        ),
                        max_tokens=256,
                    )
                    single = json.loads(extract_json_object(judge_raw)).get("label", "ERROR")
                    if single == "CORRECT":
                        label = "CORRECT"
                        break
                    label = single if label != "CORRECT" else label
                except Exception as exc:  # noqa: BLE001
                    judge_raw = f"{exc}"
            rec = {
                "sample_id": conv.get("sample_id"),
                "question": q,
                "gold": gold,
                "category": qa.get("category"),
                "is_multi_modality": bool(qa.get("is_multi_modality")),
                "reform_query": reform_q,
                "answerer_model": a_model,
                "prediction": pred,
                "judge_label": label,
                "recalled_ids": [h.get("id") for h in hits],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done += 1
            log(f"  Q{done}: [{label}] {q[:60]}")
    return done


def summarize(run_dir: Path, qa_file: str = "predictions.jsonl") -> dict:
    records = []
    for p in sorted(run_dir.glob(f"*/{qa_file}")):
        for line in p.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    text_only = [r for r in records if not r.get("is_multi_modality")]

    def acc(rows):
        scored = [r for r in rows if r.get("judge_label") in ("CORRECT", "WRONG")]
        if not scored:
            return {"n": 0, "acc": None}
        c = sum(1 for r in scored if r["judge_label"] == "CORRECT")
        return {"n": len(scored), "acc": round(c / len(scored) * 100, 2)}

    summary = {
        "overall_all": acc(records),
        "overall_text_only": acc(text_only),
        "by_category_text_only": {
            str(cat): acc([r for r in text_only if r.get("category") == cat])
            for cat in sorted({r.get("category") for r in text_only})
        },
        "judge_errors": sum(1 for r in records if r.get("judge_label") not in ("CORRECT", "WRONG")),
    }
    (run_dir / f"summary_{qa_file.replace('.jsonl', '')}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="all", help="conversation index (0-9) or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="max questions per conversation (smoke)")
    ap.add_argument("--replay-only", action="store_true", help="ingest sessions, skip QA")
    ap.add_argument("--qa-file", default="predictions.jsonl", help="QA output filename (per-answerer reruns)")
    ap.add_argument("--verbatim", action="store_true", help="zero-extraction arm: store each utterance verbatim with event_time (mema-as-documented)")
    ap.add_argument("--hybrid", action="store_true", help="dual-write arm: curated facts AND verbatim utterances in one library")
    args = ap.parse_args()

    for var in ("EVALUATOR_API_BASE", "EVALUATOR_API_KEY"):
        if not os.getenv(var) and not args.replay_only:
            sys.exit(f"missing env {var} (see module docstring)")
    judge = LLM(
        os.getenv("EVALUATOR_API_BASE", ""),
        os.getenv("EVALUATOR_API_KEY", ""),
        os.getenv("EVALUATOR_MODEL", "qwen3-14b"),
    )
    answerer = LLM(
        os.getenv("MEMA_EVAL_ANSWERER_BASE", os.getenv("EVALUATOR_API_BASE", "")),
        os.getenv("MEMA_EVAL_ANSWERER_KEY", os.getenv("EVALUATOR_API_KEY", "")),
        os.getenv("MEMA_EVAL_ANSWERER_MODEL", ""),
    )
    fallback = None
    if os.getenv("MEMA_EVAL_FALLBACK_MODEL"):
        fallback = LLM(
            os.getenv("MEMA_EVAL_FALLBACK_BASE", os.getenv("EVALUATOR_API_BASE", "")),
            os.getenv("MEMA_EVAL_FALLBACK_KEY", os.getenv("EVALUATOR_API_KEY", "")),
            os.getenv("MEMA_EVAL_FALLBACK_MODEL"),
        )

    data = json.loads(DATA_PATH.read_text())
    RUN_DIR.mkdir(exist_ok=True)
    idxs = range(len(data)) if args.conv == "all" else [int(args.conv)]

    for i in idxs:
        conv = data[i]
        sid = conv.get("sample_id", f"conv-{i}")
        if args.verbatim:
            sid = f"{sid}-verbatim"
        if args.hybrid:
            sid = f"{sid}-hybrid"
        conv_dir = RUN_DIR / sid
        conv_dir.mkdir(exist_ok=True)

        def log(msg: str) -> None:
            print(f"[{sid}] {msg}", flush=True)

        log("building isolated runtime…")
        tools = build_isolated_runtime(conv_dir)
        replay_marker = conv_dir / "replayed.json"
        if not replay_marker.exists():
            if args.verbatim:
                n = replay_verbatim(tools, conv, log)
                replay_marker.write_text(json.dumps({"mode": "verbatim", "items_written": n}))
                log(f"verbatim replay complete: {n} utterances")
            elif args.hybrid:
                v = replay_verbatim(tools, conv, log)
                f = replay(tools, conv, answerer, log)
                replay_marker.write_text(json.dumps({"mode": "hybrid", "verbatim": v, "facts": f}))
                log(f"hybrid replay complete: {v} utterances + {f} facts")
            else:
                n = replay(tools, conv, answerer, log)
                replay_marker.write_text(json.dumps({"facts_written": n}))
                log(f"replay complete: {n} facts")
        else:
            log(f"replay already done: {replay_marker.read_text()}")
        if not args.replay_only:
            answered = answer_questions(
                tools, conv, answerer, judge, conv_dir / args.qa_file, log, args.limit, fallback=fallback
            )
            log(f"answered {answered} new questions")

    print(json.dumps(summarize(RUN_DIR, args.qa_file), indent=2, ensure_ascii=False))
    # llama.cpp metal teardown can assert on exit on macOS; results are already
    # flushed, so skip the crashy interpreter finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    # every exit path goes through os._exit: normal interpreter finalization
    # trips a llama.cpp Metal teardown assert on macOS, which the system
    # surfaces as repeated "Python quit unexpectedly" crash dialogs
    _code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001 — log, then skip crashy teardown
        import traceback

        traceback.print_exc()
        _code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
