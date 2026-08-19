"""Deterministic local-text units used by the vNext vector index."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])\s*")


@dataclass(frozen=True)
class EvidenceUnit:
    kind: str
    text: str
    start_offset: int
    end_offset: int
    unit_index: int


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# Cheap SQL prefilter for "might be indexable" rows. Deliberately
# over-inclusive: SQLite TRIM strips only ASCII spaces and the subject is
# compared untrimmed, so whitespace-only rows (tab/newline/U+00A0/U+3000
# legacy artifacts) still pass. Callers pair it with has_indexable_text,
# which shares the extractor's exact semantics and is authoritative.
INDEXABLE_PREFILTER_SQL = "(COALESCE(m.subject,'')!='' OR TRIM(COALESCE(m.content,''))!='')"


def has_indexable_text(subject: str, content: str) -> bool:
    """True when local_text_units would publish at least one unit.

    Pending-set, stale-selection, and coverage SQL use the cheap prefilter
    above and then apply this check, so those sets can never disagree with
    what the indexer actually publishes (a zero-unit memory has no evidence
    row to mark it done and would otherwise stay pending forever).

    The fast path uses the extractor's own blank tests verbatim — a subject
    surviving _clean yields the subject unit, and any line that survives
    str.strip (the exact check local_text_units applies per line) feeds a
    paragraph whose cleaned text is non-empty — so it answers the same
    question without building the full unit list."""
    if _clean(subject or ""):
        return True
    return any(line.strip() for line in (content or "").splitlines())


def _locate_clean_text(content: str, cleaned: str, start: int, end: int) -> tuple[int, int]:
    """Locate normalized text without pretending normalized offsets are exact."""
    direct = content.find(cleaned, start, end)
    if direct >= 0:
        return direct, direct + len(cleaned)
    words = [word for word in re.split(r"\s+", cleaned) if word]
    if words:
        first = content.find(words[0], start, end)
        last = content.rfind(words[-1], first if first >= 0 else start, end)
        if first >= 0 and last >= first:
            return first, min(end, last + len(words[-1]))
    return start, end


def _split_long(text: str, start: int, *, size: int = 300, overlap: int = 60) -> Iterable[tuple[str, int, int]]:
    step = max(1, size - overlap)
    for offset in range(0, len(text), step):
        part = text[offset:offset + size].strip()
        if len(part) >= 8:
            yield part, start + offset, min(start + offset + size, start + len(text))
        if offset + size >= len(text):
            break


def local_text_units(subject: str, content: str) -> list[EvidenceUnit]:
    """Return complete-coverage local units without semantic fact extraction.

    Subject and Markdown headings are indexed independently. Body text follows
    existing line/sentence boundaries; long unpunctuated text uses an overlap
    fallback. The function never guesses entities, attributes, or scopes.
    """
    raw_units: list[tuple[str, str, int, int]] = []
    normalized_subject = _clean(subject)
    if normalized_subject:
        raw_units.append(("subject", normalized_subject, 0, 0))

    paragraphs: list[tuple[int, int, str]] = []
    paragraph_lines: list[str] = []
    paragraph_start = 0
    offset = 0

    def flush(end: int) -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            paragraphs.append((paragraph_start, end, " ".join(paragraph_lines)))
            paragraph_lines = []

    for line in (content or "").splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        heading = _HEADING_RE.match(raw.strip())
        if heading:
            flush(offset)
            title = _clean(heading.group(2))
            if title:
                raw_units.append(("heading", title, offset, offset + len(raw)))
        elif not raw.strip():
            flush(offset)
        else:
            if not paragraph_lines:
                paragraph_start = offset
            paragraph_lines.append(raw)
        offset += len(line)
    flush(len(content or ""))

    # Spec §4 rule 4/6: merge very short paragraphs with an adjacent one
    # instead of dropping them — coverage of retrievable text is mandatory.
    merged: list[tuple[int, int, str]] = []
    for start, end, paragraph in paragraphs:
        text = _clean(paragraph)
        if merged and (len(text) < 8 or len(merged[-1][2]) < 8):
            prev_start, _, prev_text = merged[-1]
            merged[-1] = (prev_start, end, _clean(f"{prev_text} {text}"))
        else:
            merged.append((start, end, text))
    paragraphs = [item for item in merged if item[2]]

    for start, end, paragraph in paragraphs:
        text = _clean(paragraph)
        sentences = [_clean(item) for item in _SENTENCE_BOUNDARY_RE.split(text) if _clean(item)]
        sentence_groups: list[str]
        if len(sentences) <= 1:
            sentence_groups = [text]
        else:
            sentence_groups = []
            index = 0
            while index < len(sentences):
                sentence_group: list[str] = []
                chars = 0
                cursor = index
                while cursor < len(sentences) and (
                    chars < 100 or (chars + len(sentences[cursor]) <= 240 and len(sentence_group) < 3)
                ):
                    sentence_group.append(sentences[cursor])
                    chars += len(sentences[cursor])
                    cursor += 1
                sentence_groups.append(" ".join(sentence_group))
                index = max(index + 1, cursor - 1)

        search_from = start
        for grouped_text in sentence_groups:
            group_start, group_end = _locate_clean_text(content or "", grouped_text, search_from, end)
            search_from = max(search_from, group_end)
            if len(grouped_text) <= 400:
                raw_units.append(("text", grouped_text, group_start, group_end))
            else:
                for part, part_start, part_end in _split_long(grouped_text, group_start):
                    raw_units.append(("text", part, part_start, part_end))

    return [
        EvidenceUnit(kind, text, start, end, index)
        for index, (kind, text, start, end) in enumerate(raw_units)
    ]


def evidence_content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()
