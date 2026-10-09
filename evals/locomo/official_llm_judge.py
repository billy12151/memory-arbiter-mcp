"""LLM judge wrapper for LoCoMo evaluation.

Two named judges are available:

- ``original`` — the prompt used by the original LoCoMo paper. Lenient: it
  only checks topic/meaning overlap and treats equivalent date references as
  CORRECT.
- ``refined`` — the judge shipped by this repository. Stricter: it enforces
  time-granularity matching, forbids converting between relative and absolute
  time, and requires list-type gold answers to be fully covered. This is the
  default.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from llm_judge_runtime import AsyncBinaryJudge

# Official LoCoMo-Refined judge model.
DEFAULT_EVALUATOR_MODEL = "qwen3-14b"

DEFAULT_LLM_JUDGE = "refined"

JUDGE_PROMPTS = {
    "original": """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
""",
    "refined": """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
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
""",
}
SUPPORTED_LLM_JUDGES = frozenset(JUDGE_PROMPTS)
_JUDGES: dict[str, AsyncBinaryJudge] = {}
_WARNED_NON_QWEN_MODELS: set[str] = set()


def configure_llm_judge(
    *,
    name: str = DEFAULT_LLM_JUDGE,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> None:
    judge = _get_judge(name)
    default_model, default_base_url, default_api_key = resolve_evaluator_config()
    judge.configure(
        model=model or default_model,
        base_url=base_url if base_url is not None else default_base_url,
        api_key=api_key if api_key is not None else default_api_key,
    )


def resolve_llm_judge_config(
    *,
    name: str = DEFAULT_LLM_JUDGE,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, str | None]:
    _normalize_judge(name)
    default_model, default_base_url, default_api_key = resolve_evaluator_config()
    resolved = {
        "model": model or default_model,
        "base_url": base_url if base_url is not None else default_base_url,
        "api_key": api_key if api_key is not None else default_api_key,
    }
    _warn_if_non_qwen_model(str(resolved["model"] or ""))
    return resolved


def shutdown_llm_judge(name: str | None = None) -> None:
    if name is None:
        for judge in _JUDGES.values():
            judge.close()
        _JUDGES.clear()
        return
    judge = _JUDGES.pop(_normalize_judge(name), None)
    if judge is not None:
        judge.close()


async def ashutdown_llm_judge(name: str | None = None) -> None:
    if name is None:
        judges = list(_JUDGES.values())
        _JUDGES.clear()
        for judge in judges:
            await judge.aclose()
        return
    judge = _JUDGES.pop(_normalize_judge(name), None)
    if judge is not None:
        await judge.aclose()


async def evaluate_llm_judge(
    *,
    question: str,
    reference_answer: str | list[str],
    predicted_answer: str,
    name: str = DEFAULT_LLM_JUDGE,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Judge one QA pair and return a normalized score payload."""
    normalized_name = _normalize_judge(name)
    judge = _get_judge(normalized_name)
    if model is not None or base_url is not None or api_key is not None:
        default_model, default_base_url, default_api_key = resolve_evaluator_config()
        judge.configure(
            model=model or default_model,
            base_url=base_url if base_url is not None else default_base_url,
            api_key=api_key if api_key is not None else default_api_key,
        )
    score = await judge.evaluate(question, reference_answer, predicted_answer)
    return {
        "score": float(score),
        "reason": f"llm_judge_{normalized_name}",
        "name": normalized_name,
    }


def resolve_evaluator_config() -> tuple[str, str | None, str | None]:
    model = os.getenv("EVALUATOR_MODEL", "").strip() or DEFAULT_EVALUATOR_MODEL
    base_url = (
        os.getenv("EVALUATOR_API_BASE", "").strip()
        or os.getenv("EVALUATOR_BASE_URL", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
    )
    api_key = os.getenv("EVALUATOR_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip() or None
    return model, base_url or None, api_key or None


def is_qwen3_14b_model(model_name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(model_name or "").strip().lower())
    return normalized.endswith(("qwen314b", "qwenqwen314b"))


def _warn_if_non_qwen_model(model_name: str) -> None:
    candidate = str(model_name or "").strip()
    if not candidate or is_qwen3_14b_model(candidate):
        return
    if candidate in _WARNED_NON_QWEN_MODELS:
        return
    _WARNED_NON_QWEN_MODELS.add(candidate)
    print(
        (
            "[LoCoMo-Refined] Warning: evaluator model is not Qwen3-14B "
            f"({candidate}). Official LoCoMo-Refined judge baseline uses Qwen3-14B; "
            "results may not be directly comparable."
        ),
        file=sys.stderr,
        flush=True,
    )


def _get_judge(name: str) -> AsyncBinaryJudge:
    normalized = _normalize_judge(name)
    judge = _JUDGES.get(normalized)
    if judge is None:
        judge = AsyncBinaryJudge(
            prompt_template=JUDGE_PROMPTS[normalized],
            default_model=DEFAULT_EVALUATOR_MODEL,
            default_base_url=None,
            default_api_key=None,
        )
        _JUDGES[normalized] = judge
    return judge


def _normalize_judge(name: str) -> str:
    normalized = (name or DEFAULT_LLM_JUDGE).strip().lower()
    if normalized in SUPPORTED_LLM_JUDGES:
        return normalized
    raise ValueError(
        f"unsupported llm judge: {name}. "
        f"Supported judges: {', '.join(sorted(SUPPORTED_LLM_JUDGES))}"
    )
