"""Deterministic local-text units used by the vNext vector index."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])\s*")


@dataclass(frozen=True)
class EvidenceUnit:
    kind: str
    text: str
    start_offset: int
    end_offset: int
    unit_index: int


@dataclass(frozen=True)
class _MappedText:
    """Normalized text whose characters retain exact source coordinates."""

    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def slice(self, start: int, end: int) -> "_MappedText":
        start = max(0, min(int(start), len(self.text)))
        end = max(start, min(int(end), len(self.text)))
        return _MappedText(self.text[start:end], self.starts[start:end], self.ends[start:end])

    def strip(self) -> "_MappedText":
        start = 0
        end = len(self.text)
        while start < end and self.text[start].isspace():
            start += 1
        while end > start and self.text[end - 1].isspace():
            end -= 1
        return self.slice(start, end)

    @property
    def source_span(self) -> tuple[int, int]:
        if not self.text:
            return 0, 0
        return self.starts[0], self.ends[-1]


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


def _normalize_with_map(content: str, start: int, end: int) -> _MappedText:
    """Collapse source whitespace while preserving exact source coordinates.

    The old implementation normalized text first and then searched for it in
    the source to recover offsets. Repeated words, semicolon-dense settings,
    and collapsed cross-line whitespace made that reverse lookup ambiguous and
    errors cascaded through the forward-only search cursor. Here coordinates
    are created during normalization, so no reverse lookup is needed.
    """
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = max(0, int(start))
    stop = min(len(content), max(index, int(end)))
    while index < stop:
        char = content[index]
        if char.isspace():
            run_start = index
            index += 1
            while index < stop and content[index].isspace():
                index += 1
            if chars and index < stop:
                chars.append(" ")
                starts.append(run_start)
                ends.append(index)
            continue
        chars.append(char)
        starts.append(index)
        ends.append(index + 1)
        index += 1
    return _MappedText("".join(chars), tuple(starts), tuple(ends)).strip()


def _sentence_slices(mapped: _MappedText) -> list[_MappedText]:
    """Split normalized text at sentence boundaries without losing mapping."""
    slices: list[_MappedText] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(mapped.text):
        item = mapped.slice(cursor, match.start()).strip()
        if item.text:
            slices.append(item)
        cursor = match.end()
    item = mapped.slice(cursor, len(mapped.text)).strip()
    if item.text:
        slices.append(item)
    return slices


def _long_parts(mapped: _MappedText, *, size: int = 300, overlap: int = 60) -> list[_MappedText]:
    parts: list[_MappedText] = []
    step = max(1, size - overlap)
    for offset in range(0, len(mapped.text), step):
        part = mapped.slice(offset, min(offset + size, len(mapped.text))).strip()
        if len(part.text) >= 8:
            parts.append(part)
        if offset + size >= len(mapped.text):
            break
    return parts


def local_text_units(subject: str, content: str) -> list[EvidenceUnit]:
    """Return complete-coverage local units without semantic fact extraction.

    Subject and Markdown headings are indexed independently. Body text follows
    existing line/sentence boundaries; long unpunctuated text uses an overlap
    fallback. Every body unit's offsets are derived directly from a
    normalization-to-source character map, so ``_clean(content[start:end])``
    contains the unit text even for repeated tokens and cross-line whitespace.
    The function never guesses entities, attributes, or scopes.
    """
    raw_units: list[tuple[str, str, int, int]] = []
    normalized_subject = _clean(subject)
    if normalized_subject:
        raw_units.append(("subject", normalized_subject, 0, 0))

    paragraph_spans: list[tuple[int, int]] = []
    heading_offsets: list[int] = []
    paragraph_start: int | None = None
    offset = 0

    def flush(end: int) -> None:
        nonlocal paragraph_start
        if paragraph_start is not None:
            paragraph_spans.append((paragraph_start, end))
            paragraph_start = None

    for line in (content or "").splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        heading = _HEADING_RE.match(raw.strip())
        if heading:
            flush(offset)
            heading_offsets.append(offset)
            title = _clean(heading.group(2))
            if title:
                raw_units.append(("heading", title, offset, offset + len(raw)))
        elif not raw.strip():
            flush(offset)
        elif paragraph_start is None:
            paragraph_start = offset
        offset += len(line)
    flush(len(content or ""))

    paragraphs: list[tuple[int, int, _MappedText]] = []
    for start, end in paragraph_spans:
        mapped = _normalize_with_map(content or "", start, end)
        if not mapped.text:
            continue
        previous_end = paragraphs[-1][1] if paragraphs else 0
        heading_between = any(previous_end <= pos < start for pos in heading_offsets)
        if (
            paragraphs
            and not heading_between
            and (len(mapped.text) < 8 or len(paragraphs[-1][2].text) < 8)
        ):
            prev_start, _, _ = paragraphs[-1]
            # Blank lines may separate the paragraphs; normalizing one
            # continuous source range keeps exact offsets without losing the
            # separator. Headings are an explicit merge barrier above.
            combined = _normalize_with_map(content or "", prev_start, end)
            paragraphs[-1] = (prev_start, end, combined)
        else:
            paragraphs.append((start, end, mapped))

    for _, _, paragraph in paragraphs:
        sentences = _sentence_slices(paragraph)
        groups: list[_MappedText]
        if len(sentences) <= 1:
            groups = [paragraph]
        else:
            groups = []
            index = 0
            while index < len(sentences):
                chars = 0
                cursor = index
                while cursor < len(sentences) and (
                    chars < 100
                    or (
                        chars + len(sentences[cursor].text) <= 240
                        and cursor - index < 3
                    )
                ):
                    chars += len(sentences[cursor].text)
                    cursor += 1
                group_source_start = sentences[index].source_span[0]
                group_source_end = sentences[cursor - 1].source_span[1]
                groups.append(
                    _normalize_with_map(
                        content or "", group_source_start, group_source_end,
                    )
                )
                index = max(index + 1, cursor - 1)

        for group in groups:
            if not group.text:
                continue
            if len(group.text) <= 400:
                start, end = group.source_span
                raw_units.append(("text", group.text, start, end))
            else:
                for part in _long_parts(group):
                    start, end = part.source_span
                    raw_units.append(("text", part.text, start, end))

    return [
        EvidenceUnit(kind, text, start, end, index)
        for index, (kind, text, start, end) in enumerate(raw_units)
    ]


def evidence_content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()
