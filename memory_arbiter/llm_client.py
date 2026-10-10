"""Console knowledge-workbench LLM channel (0.18.0, plan §B.4/B.5).

stdlib-only OpenAI-compatible client (urllib.request — same zero-dependency
stance as update_monitor), context assembly with a conservative token budget,
and the per-action harness: skill-file system prompt + thin structural
validation + one corrective retry + degraded passthrough.
"""
from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib import resources
from typing import Any

from .config import Settings

# Context-assembly reserves (plan §B.4.2): output headroom and the system /
# formatting overhead are subtracted from max_context_tokens before any
# memory body is admitted.
RESERVE_OUTPUT = 4096
RESERVE_PROMPT = 800

ACTIONS = ("summarize", "outline", "quiz", "feynman", "quiz_me", "timeline", "custom")

# last-successfully-used URL variant keyed by base_url ("plain" | "v1")
# (R2 P2-2: keyed so switching the URL invalidates immediately). Storing the
# variant instead of a bool survives base_urls that already end in /v1 —
# their plain path IS .../v1/chat/completions and a bool cache would poison
# the next call into .../v1/v1/... (adversarial review P0-1).
_V1_PROBE_CACHE: dict[str, str] = {}


class LLMError(Exception):
    """Transport/config failure. Never carries the api_key."""

    def __init__(self, message: str, *, status: int | None = None):
        self.status = status
        super().__init__(message)


def _settings_ready(settings: Settings) -> str | None:
    if not str(settings.llm_base_url or "").strip():
        return "llm.base_url is not configured — open Console → Settings → LLM workspace"
    if not str(settings.llm_model or "").strip():
        return "llm.model is not configured — open Console → Settings → LLM workspace"
    if not str(settings.llm_api_key or "").strip():
        return "llm.api_key is not configured — open Console → Settings → LLM workspace"
    if int(settings.llm_max_context_tokens or 0) <= 0:
        return "llm.max_context_tokens is not set — open Console → Settings → LLM workspace"
    return None


def _post_json(url: str, payload: dict[str, Any], *, api_key: str, timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "memory-arbiter-console/0.18.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001 - best-effort error body
            detail = ""
        raise LLMError(f"LLM endpoint returned HTTP {exc.code}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"LLM endpoint unreachable: {exc.reason}") from exc
    except (ValueError, OSError) as exc:
        raise LLMError(f"LLM endpoint response invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("LLM endpoint returned a non-object response")
    return data


def chat_completion(settings: Settings, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """POST {base_url}/chat/completions; returns {content, reasoning, usage}.

    Tries ``{base_url}/chat/completions`` first, then ``/v1/chat/completions``
    on 404 (probe cached per base_url). content=None (thinking models whose
    max_tokens budget is consumed by reasoning) raises LLMError with
    finish_reason context.
    """
    if (problem := _settings_ready(settings)) is not None:
        raise LLMError(problem)
    base = str(settings.llm_base_url).rstrip("/")
    plain_url = base + "/chat/completions"
    v1_url = base + "/v1/chat/completions"
    cached = _V1_PROBE_CACHE.get(base)
    if cached == "v1":
        urls = [v1_url, plain_url]
    else:
        # unknown or "plain": try the plain path first — for base_urls already
        # ending in /v1 the plain path IS the correct OpenAI-style endpoint.
        urls = [plain_url, v1_url]
    variant = "plain"
    last_error: LLMError | None = None
    for url in urls:
        try:
            data = _post_json(
                url,
                {
                    "model": settings.llm_model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": int(settings.llm_max_output_tokens or 8192),
                },
                api_key=str(settings.llm_api_key),
                timeout=int(settings.llm_timeout_seconds or 120),
            )
        except LLMError as exc:
            if exc.status == 404:
                last_error = exc
                continue
            raise
        variant = "v1" if url == v1_url else "plain"
        _V1_PROBE_CACHE[base] = variant
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if not str(content or "").strip():
            raise LLMError(
                "LLM returned empty content "
                f"(finish_reason={choice.get('finish_reason')}); try raising llm.max_output_tokens"
            )
        usage = data.get("usage") or {}
        return {
            "content": str(content),
            "reasoning": str(
                message.get("reasoning_content") or message.get("reasoning") or ""
            ) or None,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        }
    raise last_error or LLMError("LLM endpoint unreachable")


# ---- context assembly (plan §B.4.2) ----------------------------------------


def _est_tokens(text: str) -> int:
    """Conservative mixed-language estimate (R2 P1-5 fix): per-script sum,
    then a global safety factor — never the old ``max(len*1.1, ascii/4+cjk)``
    whose first arm always dominated (English overestimated ~4x)."""
    ascii_len = sum(1 for ch in text if ord(ch) < 128)
    cjk_len = len(text) - ascii_len
    return math.ceil((cjk_len * 1.0 + ascii_len / 4.0) * 1.15)


@dataclass
class ContextMaterial:
    memory_id: int
    subject: str
    event_time: str
    content: str

    def block(self) -> str:
        return f"## [{self.memory_id}] {self.subject} ({self.event_time})\n{self.content}"


def assemble_context(materials: list[ContextMaterial], settings: Settings) -> str:
    """Concatenate materials within the token budget or fail loudly with a
    per-memory estimate table — never silent truncation."""
    budget = int(settings.llm_max_context_tokens) - RESERVE_OUTPUT - RESERVE_PROMPT
    estimates = {m.memory_id: _est_tokens(m.block()) for m in materials}
    total = sum(estimates.values())
    if total <= budget:
        return "\n\n".join(m.block() for m in materials)
    lines = [f"selected content is {total} tokens over the {budget} budget (by {total - budget}) — remove some memories:"]
    for m in materials:
        lines.append(f"  [{m.memory_id}] {m.subject}: ~{estimates[m.memory_id]} tokens")
    if materials and estimates[max(estimates, key=lambda k: estimates[k])] > budget:
        single = max(estimates, key=lambda k: estimates[k])
        raise LLMError(
            "single memory "
            f"[{single}] is ~{estimates[single]} tokens, above the whole budget {budget} — "
            "raise llm.max_context_tokens or pick shorter memories. " + "; ".join(lines)
        )
    raise LLMError("; ".join(lines))


# ---- skill prompts (plan §B.5) ----------------------------------------------

_DISCIPLINE = """## 范围纪律（三条铁律）
1. 只准使用材料中出现的记忆内容，材料外一律不编造；
2. 不确定的表述明确标注「材料未覆盖」；
3. 引用具体记忆时带 [id] 标记（如 [101]）。"""


def load_skill(action: str) -> str:
    if action == "custom":
        action = "_custom"
    try:
        resource = resources.files("memory_arbiter.llm_skills").joinpath(f"{action}.md")
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, ModuleNotFoundError) as exc:
        raise LLMError(f"skill file for action '{action}' is missing: {exc}") from exc


# ---- thin structural validators (plan §B.5.2) --------------------------------

# non-capturing groups: findall must return full years ("2015"), not the
# "19"/"20" prefix a capturing group would yield.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ISO_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")


def validate_output(action: str, output: str, materials: list[ContextMaterial]) -> list[str]:
    """Quality gate (not a safety gate): structural checks per action.
    Returns the list of violations; empty = pass. Degraded output still
    reaches the user with a warning — nothing is swallowed."""
    errors: list[str] = []
    text = output or ""
    if action == "summarize":
        if len(text) < 100:
            errors.append("summary is under 100 characters")
        if "[101]" not in text and not re.search(r"\[\d+\]", text):
            errors.append("no [id] citation found")
    elif action == "outline":
        if len(re.findall(r"^##+ ", text, re.M)) < 2:
            errors.append("fewer than two ## headings")
    elif action == "quiz":
        # count questions ONLY in the section before the answers heading —
        # answer lines and headings are numbered too and double-counted into
        # a false over-limit (adversarial review P1-2). No trailing \s so
        # "1、题" (CJK enumeration comma) counts.
        split = re.split(r"^#{2,3}\s*(?:答案|answers?)\s*$", text, flags=re.M | re.I)
        question_section = split[0] if split else text
        numbered = re.findall(r"^#{2,3}\s|^\d{1,2}[.、)]", question_section, re.M)
        if not 3 <= len(numbered) <= 20:
            errors.append(f"question count {len(numbered)} outside 3..20")
        if "答案" not in text and "answer" not in text.lower():
            errors.append("no answer section found")
    elif action == "feynman":
        if len(text) < 200:
            errors.append("explanation under 200 characters")
    elif action == "quiz_me":
        questions = re.findall(r"[?？]", text)
        if not 3 <= len(questions) <= 15:
            errors.append(f"question count {len(questions)} outside 3..15")
    elif action == "timeline":
        material_times: set[str] = set()
        for m in materials:
            # stored event_time is a full ISO datetime ("2026-10-10T…"): the
            # date-part string itself counts, plus bare years — the  in
            # _ISO_RE never matches past the "T" (adversarial review P1-1).
            event_day = str(m.event_time or "")[:10]
            if len(event_day) == 10 and event_day[4] == "-" and event_day[7] == "-":
                material_times.add(event_day)
                material_times.add(event_day[:7])
            material_times.update(_YEAR_RE.findall(m.event_time or ""))
            material_times.update(_YEAR_RE.findall(m.content))
            material_times.update(_ISO_RE.findall(m.content))
        claimed = set(_YEAR_RE.findall(text)) | set(_ISO_RE.findall(text))
        fabricated = claimed - material_times
        if fabricated:
            errors.append(
                "timeline cites time points absent from the materials: "
                + ", ".join(sorted(fabricated))
            )
    elif action == "custom":
        if len(text) < 10:
            errors.append("output under 10 characters")
    return errors


# ---- action runner (plan §B.4.3) ----------------------------------------------

def run_action(
    settings: Settings,
    action: str,
    materials: list[ContextMaterial],
    custom_prompt: str = "",
) -> dict[str, Any]:
    if action not in ACTIONS:
        raise LLMError(f"unknown action '{action}'")
    if action == "custom" and not str(custom_prompt).strip():
        raise LLMError("custom_prompt is required for action=custom")
    if len(str(custom_prompt)) > 4000:
        raise LLMError("custom_prompt exceeds 4000 characters")
    if (problem := _settings_ready(settings)) is not None:
        raise LLMError(problem)

    context = assemble_context(materials, settings)
    system = load_skill(action)
    user_parts = [f"# 材料（{len(materials)} 条记忆）\n\n{context}"]
    if action == "custom":
        user_parts.append(f"# 用户指令\n\n{custom_prompt}")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    result: dict[str, Any] | None = None
    errors: list[str] = []
    for attempt in (1, 2):
        try:
            result = chat_completion(settings, messages)
        except LLMError:
            if attempt == 1:
                raise
            # attempt 2 hit a transport failure after attempt 1 already
            # produced (unvalidated) content — degrade with what we have
            # instead of surfacing a hard error (adversarial review P2-7).
            if result is not None:
                errors = validate_output(action, result["content"], materials)
                break
            raise
        errors = validate_output(action, result["content"], materials)
        if not errors:
            return {
                "output_markdown": result["content"],
                "reasoning": result["reasoning"],
                "usage": result["usage"],
                "attempt": attempt,
                "degraded": False,
                "warnings": [],
                "context_chars": len(context),
                "source_memory_ids": [m.memory_id for m in materials],
            }
        if attempt == 1:
            messages = messages + [
                {"role": "assistant", "content": result["content"]},
                {
                    "role": "user",
                    "content": "上一次输出未通过结构校验：" + "；".join(errors) + "。请修正这些问题重新输出。",
                },
            ]
    # both attempts failed the gate — degraded passthrough with warnings
    if result is None:  # unreachable in practice: attempt 1 raises on failure
        raise LLMError("no model output was produced")
    return {
        "output_markdown": result["content"],
        "reasoning": result["reasoning"],
        "usage": result["usage"],
        "attempt": 2,
        "degraded": True,
        "warnings": errors,
        "context_chars": len(context),
        "source_memory_ids": [m.memory_id for m in materials],
    }


def run_test_probe(settings: Settings) -> dict[str, Any]:
    """/api/llm/test — connectivity probe with a fixed one-line material,
    bypassing validators and retries (plan §B.3.1)."""
    if (problem := _settings_ready(settings)) is not None:
        raise LLMError(problem)
    result = chat_completion(
        settings,
        [
            {"role": "system", "content": "You are a connectivity probe. Reply with exactly: ok"},
            {"role": "user", "content": "ping"},
        ],
    )
    return {"reply": result["content"][:80], "model": settings.llm_model}
