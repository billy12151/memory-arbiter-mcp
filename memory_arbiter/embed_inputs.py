"""Shared embed-input builders for the recall-vector family (reuse batch C5,
2026-10-11): summary_embed_text / subject_tags_embed_text / embed_input_profile
were previously pinned together by a "Mirror ... exactly" comment in
pipeline/operations.py and a pipeline->tools reverse import — this leaf turns
that coupling into a code-level one. Stdlib-only by contract (no tools/
pipeline/db imports)."""
from __future__ import annotations

import json
from typing import Any

SUMMARY_SEGMENT_CHARS = 40
SUMMARY_TOTAL_CHARS = 800


def summary_embed_text(subject: Any, tags: Any, content: Any) -> str:
    """C3a canonical summary text: subject + sorted tags + each body
    segment's first 40 chars, capped ~800 chars total.

    Segments split on blank lines (local_text_units' paragraph shape);
    ownership voting needs topical signal, not full bodies — a misplaced
    memory keeps its subject/tag vocabulary even when bodies drift.
    """
    cleaned = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    parts = [f"{str(subject or '').strip()}\n{' '.join(sorted(cleaned))}".strip()]
    body = str(content or "")
    for segment in body.split("\n\n"):
        segment = segment.strip()
        if segment:
            parts.append(segment[:SUMMARY_SEGMENT_CHARS])
    return "\n".join(parts)[:SUMMARY_TOTAL_CHARS]


def subject_tags_embed_text(subject: Any, tags: Any) -> str:
    """Canonical text embedded into subject_tags_vec (owner spec: "subject
    + sorted tags"). Tags are sorted so tag ORDER never changes the
    vector; case/whitespace are left to the model like every other
    embedded body."""
    cleaned = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    body = f"{str(subject or '').strip()}\n{' '.join(sorted(cleaned))}".strip()
    return body


def embed_input_profile(record: dict[str, Any] | None) -> tuple[str, str, str]:
    """Derived-embedding input profile (0.16.12 P2-T1): the normalized
    (subject, tags, content) triple the two recall-vector refreshes re-embed
    from. Row-level equality of this triple means the refreshed vectors would
    be byte-identical, so the re-embed is skippable. Accepts tags in either
    shape — the raw JSON string (SQL rows) or an already-parsed list
    (missing_*_rows snapshots) — so profiles compare equal across sources.

    Must stay consistent with subject_tags_embed_text / summary_embed_text:
    strip each tag, drop empties, THEN sort — an unstripped join would let
    distinct embed inputs share a profile (second-round adversarial finding).
    Co-location in this leaf replaces the old mirror comment."""
    if not record:
        return ("", "", "")
    raw_tags = record.get("tags")
    if isinstance(raw_tags, str):
        try:
            tags = json.loads(raw_tags or "[]")
        except (TypeError, ValueError):
            tags = []
    elif isinstance(raw_tags, (list, tuple)):
        tags = list(raw_tags)
    else:
        tags = []
    cleaned = sorted(str(t).strip() for t in tags if str(t).strip())
    return (
        str(record.get("subject") or ""),
        " ".join(cleaned),
        str(record.get("content") or ""),
    )
