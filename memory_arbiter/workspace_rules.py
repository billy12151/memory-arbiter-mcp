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
from typing import Any

from .constants import DEFAULT_TERMS, is_default_workspace_term  # re-exported below

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

# Single source: constants.DEFAULT_TERMS; re-exported here so
# existing ``from .workspace_rules import DEFAULT_TERMS`` keeps working.

# Case-folded view of GENERIC_TERMS for token-level guards.
_GENERIC_TERMS_CF = {t.casefold() for t in GENERIC_TERMS}

# Reference / borrowed-material cues: content that merely *references* a project
# should stay in its own workspace, not be merged into the referenced one.
KEEP_CUES = (
    "参考", "借鉴", "模板", "经验", "月报", "指标", "复盘", "通用",
    "template", "reference", "benchmark",
)


def classify_workspace_quality(ws_raw: str | None) -> str:
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


# ── shared vector admission (Shared admission — 期1 weak weighting / 期3 strict) ─────

# Short-name guard default: canonicals shorter than this never vector-admit
# (a 1-2 char name is within cosine reach of many unrelated projects).
DEFAULT_MIN_NAME_LEN = 3

# weak-isolation continuous weighting curve anchors :
# full +0.30 inside 0.15, linear decay to 0 at 0.30, 0 beyond. The 0.30 cap
# keeps the nudge at ~5% of a subject-medium hit (6.0) — same magnitude
# discipline as trust/recency bonuses.
WEAK_VECTOR_WEIGHT_MAX = 0.30
WEAK_VECTOR_DISTANCE_NEAR = 0.15
WEAK_VECTOR_DISTANCE_FAR = 0.30


def _ws_name_fold(name: str) -> str:
    """Fold to alphanumerics-only (separators/case dropped) for containment."""
    return re.sub(r"[\W_]+", "", name, flags=re.UNICODE).casefold()


def _ws_name_tokens(name: str) -> set[str]:
    """Split on non-alphanumeric separators; CJK runs stay single tokens."""
    return {tok.casefold() for tok in re.split(r"[\W_]+", name) if tok}


def _generic_only_proximity(query: str, record: str) -> bool:
    """True when two names look close only through non-identifying material.

    Two hazards, both measured on real embeddings:
      (a) pure substring containment — ``main`` ⊂ ``openclaw-main`` sits at
          cosine 0.132 (inside even the weak full-bonus zone) while sharing
          nothing but the substring;
      (b) token overlap made exclusively of GENERIC_TERMS — ``project-alpha``
          vs ``project-beta`` share only "project".
    """
    fq, fr = _ws_name_fold(query), _ws_name_fold(record)
    if fq and fr and fq != fr and (fq in fr or fr in fq):
        return True
    tokens_q = _ws_name_tokens(query)
    tokens_r = _ws_name_tokens(record)
    shared = tokens_q & tokens_r
    if shared and shared <= _GENERIC_TERMS_CF:
        if not (tokens_q - _GENERIC_TERMS_CF) & (tokens_r - _GENERIC_TERMS_CF):
            return True
    return False


def workspace_vector_distance(
    query_canonical: str | None,
    record_canonical: str | None,
    distance_map: dict[str, float] | None,
    *,
    min_name_len: int = DEFAULT_MIN_NAME_LEN,
) -> float | None:
    """Guarded cosine distance between two workspace canonicals (Shared admission).

    Returns the usable distance from ``distance_map`` (keyed by record
    canonical), or None when a guard fires and the caller must fall back to
    exact-equality semantics:
      - either side is a reserved default-pool term (default is insulated
        from the whole vector system, in both directions);
      - either side is shorter than ``min_name_len`` (short-name guard);
      - the pair is close only through generic/substring material;
      - the record canonical has no entry in the map (no vector / degraded).
    Identical canonicals always return 0.0 — the same workspace needs no
    vector evidence.
    """
    q = (query_canonical or "").strip()
    r = (record_canonical or "").strip()
    if not q or not r:
        return None
    if q == r:
        return 0.0
    if is_default_workspace_term(q) or is_default_workspace_term(r):
        return None
    if len(q) < min_name_len or len(r) < min_name_len:
        return None
    if _generic_only_proximity(q, r):
        return None
    if not distance_map:
        return None
    distance = distance_map.get(r)
    if distance is None:
        return None
    return float(distance)


def workspace_admit(
    query_canonical: str | None,
    record_canonical: str | None,
    distance_map: dict[str, float] | None,
    cutoff: float,
    *,
    min_name_len: int = DEFAULT_MIN_NAME_LEN,
) -> bool:
    """Vector-admission predicate: same-domain iff the guarded
    canonical-to-canonical distance is available and ≤ cutoff."""
    distance = workspace_vector_distance(
        query_canonical, record_canonical, distance_map, min_name_len=min_name_len,
    )
    return distance is not None and distance <= float(cutoff)


def weak_workspace_vector_weight(distance: float) -> float:
    """Weak-isolation weight from a guarded cosine distance ."""
    d = float(distance)
    if d < WEAK_VECTOR_DISTANCE_NEAR:
        return WEAK_VECTOR_WEIGHT_MAX
    if d <= WEAK_VECTOR_DISTANCE_FAR:
        span = WEAK_VECTOR_DISTANCE_FAR - WEAK_VECTOR_DISTANCE_NEAR
        return WEAK_VECTOR_WEIGHT_MAX * (WEAK_VECTOR_DISTANCE_FAR - d) / span
    return 0.0


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
    ws_raw: str | None,
    resolved: dict[str, Any],
    evidence: dict[str, Any] | None = None,
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

    # Rejected candidates the resolver already discarded. `similar` is the raw
    # (unfiltered) top-k; filter it so neither the KEEP check nor the near-tie
    # check reasons about a pair the user explicitly rejected.
    rejected = set(resolved.get("rejected_canonicals") or [])
    similar = resolved.get("similar") or []
    non_rejected = [s for s in similar if s.get("name") not in rejected]

    # KEEP: the resolver's chosen canonical is itself a rejected pair. The
    # resolver skips rejected candidates on the vector path, so this only bites
    # when the chosen canonical *equals* a rejected name (e.g. an exact/confirmed
    # path that a later rejection should override) — keep the memory separate.
    if resolved.get("canonical") in rejected:
        return {"decision": "KEEP", "reason": "rejected_pair", "canonical": ws_raw}

    # ASK: no usable workspace signal.
    if quality in {"empty", "default", "generic", "suspicious"}:
        return {"decision": "ASK", "reason": f"low_signal_{quality}", "canonical": None}

    # A vector hit landed within threshold (resolver already applied it).
    if matched_by == "vector":
        # Near-tie between the top-2 *non-rejected* candidates → ambiguous, ask.
        if len(non_rejected) >= 2:
            d0 = float(non_rejected[0].get("distance") or 1.0)
            d1 = float(non_rejected[1].get("distance") or 1.0)
            if abs(d1 - d0) < 0.05:
                return {"decision": "ASK", "reason": "candidate_near_tie", "canonical": None}
        return {"decision": "AUTO", "reason": "vector_strong", "canonical": resolved.get("canonical")}

    # New canonical, specific name.
    if matched_by == "new" and quality == "specific":
        # If there are near-miss candidates (vector found something, just below
        # the merge threshold), rules can't confidently keep them separate —
        # defer to the model layer (636 §6). With no candidates at all it's a
        # genuinely new distinct workspace → AUTO. Rejected candidates don't count.
        if non_rejected:
            return {"decision": None, "reason": "near_miss_candidates", "canonical": None}
        return {"decision": "AUTO", "reason": "new_specific_canonical", "canonical": resolved.get("canonical")}

    # Rules can't decide → let the model layer (phase C) suggest.
    return {"decision": None, "reason": "undecided", "canonical": None}
