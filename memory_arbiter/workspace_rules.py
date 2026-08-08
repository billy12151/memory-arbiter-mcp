"""Rule-first workspace normalization decisions (design 636 §2, §3, §5).

The resolver (db.resolve_workspace_canonical) handles the mechanical layer:
confirmed-alias short-circuit, exact match, vector nearest-canonical. This
module sits *on top* of the vector candidates and decides whether a proposed
merge should happen automatically (AUTO), be blocked/kept separate (KEEP), or
be surfaced to the user/agent for confirmation (ASK). Vector distance produces
*candidates*; rules decide. When rules can't decide with confidence they return
None and the caller may fall back to the model layer (Qwen, phase C).

All functions here are pure (no IO) so they are cheap to unit-test.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# ── workspace-string quality ────────────────────────────────────────────────

# Generic project-shell terms that are NOT workspace-identifying on their own —
# they describe a document *kind*, not a project. A workspace that is just one
# of these should be treated as low-signal (ASK, not AUTO-merge).
GENERIC_TERMS = {
    "实施计划", "项目二期", "经营方案", "月报", "周报", "复盘", "总结",
    "方案", "计划", "报告", "汇报", "纪要", "notes", "note", "plan",
    "report", "draft", "misc", "temp", "tmp", "project", "projects",
    "工作", "文档", "资料", "通用",
}

DEFAULT_TERMS = {"", "default", "默认", "none", "null", "unknown", "未知"}

# Reference / borrowed-material cues: content that merely *references* a project
# should stay in its own workspace, not be merged into the referenced one.
KEEP_CUES = (
    "参考", "借鉴", "模板", "经验", "月报", "指标", "复盘", "通用",
    "template", "reference", "benchmark",
)


def classify_workspace_quality(ws_raw: Optional[str]) -> str:
    """Return one of: empty | default | generic | specific | suspicious.

    - empty / default : no usable signal → ASK.
    - generic         : a document-kind word, not a project → ASK.
    - suspicious      : contains path/separators/very long → treat carefully.
    - specific        : a plausible project identifier → candidate-check still runs.
    """
    s = (ws_raw or "").strip()
    low = s.casefold()
    if not s:
        return "empty"
    if low in DEFAULT_TERMS:
        return "default"
    # path-like / injection-ish / absurdly long → suspicious
    if len(s) > 80 or re.search(r"[\\/\n\t]|https?://", s):
        return "suspicious"
    if low in {t.casefold() for t in GENERIC_TERMS}:
        return "generic"
    return "specific"


# ── evidence extraction (636 §3) ─────────────────────────────────────────────

def extract_evidence(record: Any, *, max_key_sentences: int = 3) -> dict[str, Any]:
    """Pull short, high-signal text from a memory record for candidate scoring.

    Long content is never fed wholesale downstream; we take title/subject, the
    first heading(s), the first paragraph, and a few key sentences. Accepts
    either a MemoryRecord-like object or a dict.
    """
    def _get(name: str) -> str:
        if isinstance(record, dict):
            return str(record.get(name) or "")
        return str(getattr(record, name, "") or "")

    subject = _get("subject").strip()
    content = _get("content")
    headings: list[str] = []
    for line in content.splitlines():
        st = line.strip()
        if st.startswith("#"):
            headings.append(st.lstrip("#").strip())
        if len(headings) >= 3:
            break

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    first_para = paragraphs[0] if paragraphs else ""

    # crude sentence split (CJK 。！？ + latin .!?)
    sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", first_para) if s.strip()]
    key_sentences = sentences[:max_key_sentences]

    return {
        "subject": subject,
        "title": subject or (headings[0] if headings else ""),
        "headings": headings,
        "first_para": first_para[:400],
        "key_sentences": key_sentences,
    }


# ── rule decision (636 §5) ───────────────────────────────────────────────────

def rule_decision(
    ws_raw: Optional[str],
    resolved: dict[str, Any],
    evidence: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Decide AUTO | KEEP | ASK | None over a resolver result.

    `resolved` is the dict from db.resolve_workspace_canonical. Returns
    {"decision": ..., "reason": ..., "canonical": <name or None>}.

    - AUTO : commit the merge/canonical now (confirmed/exact/mechanical, or a
             strong candidate hit with no KEEP/ASK veto).
    - KEEP : do NOT merge — keep the raw workspace as its own canonical
             (reference material, or an explicitly rejected pair).
    - ASK  : surface to the user/agent (empty/default/generic, near-tie
             candidates, weak/conflicting evidence).
    - None : rules can't decide with confidence → caller may invoke the model
             layer (Qwen). None never *commits* anything.
    """
    matched_by = resolved.get("matched_by")
    quality = classify_workspace_quality(ws_raw)
    ev = evidence or {}
    hint_text = " ".join([
        str(ws_raw or ""), ev.get("title", ""), ev.get("first_para", ""),
    ]).casefold()

    # AUTO: the mechanical layer already found high-confidence identity.
    if matched_by in {"confirmed_alias", "exact"}:
        return {"decision": "AUTO", "reason": matched_by, "canonical": resolved.get("canonical")}

    # KEEP: reference/borrowed material must not be merged into what it cites.
    # Only meaningful when a merge would otherwise happen (a candidate matched);
    # a bare generic workspace with no candidate is an ASK, handled below.
    if matched_by == "vector" and any(cue in hint_text for cue in KEEP_CUES):
        return {"decision": "KEEP", "reason": "reference_material", "canonical": ws_raw}

    # KEEP: explicitly rejected pair — the nearest candidate is a rejected one.
    rejected = set(resolved.get("rejected_canonicals") or [])
    similar = resolved.get("similar") or []
    if similar and similar[0].get("name") in rejected:
        return {"decision": "KEEP", "reason": "rejected_pair", "canonical": ws_raw}

    # ASK: no usable workspace signal.
    if quality in {"empty", "default", "generic", "suspicious"}:
        return {"decision": "ASK", "reason": f"low_signal_{quality}", "canonical": None}

    # A vector hit landed within threshold (resolver already applied it).
    if matched_by == "vector":
        # Near-tie between top-2 candidates → ambiguous, ask.
        if len(similar) >= 2:
            d0 = float(similar[0].get("distance") or 1.0)
            d1 = float(similar[1].get("distance") or 1.0)
            if abs(d1 - d0) < 0.05:
                return {"decision": "ASK", "reason": "candidate_near_tie", "canonical": None}
        return {"decision": "AUTO", "reason": "vector_strong", "canonical": resolved.get("canonical")}

    # New canonical, specific name, no candidate matched: keep it as-is (its own
    # workspace). Not ASK — a distinct specific project is the common case.
    if matched_by == "new" and quality == "specific":
        return {"decision": "AUTO", "reason": "new_specific_canonical", "canonical": resolved.get("canonical")}

    # Rules can't decide → let the model layer (phase C) suggest.
    return {"decision": None, "reason": "undecided", "canonical": None}
