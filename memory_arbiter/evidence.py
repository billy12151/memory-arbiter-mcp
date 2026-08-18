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

    for start, end, paragraph in paragraphs:
        text = _clean(paragraph)
        if len(text) < 8:
            continue
        sentences = [_clean(item) for item in _SENTENCE_BOUNDARY_RE.split(text) if _clean(item)]
        if len(sentences) <= 1:
            groups = [text]
        else:
            groups: list[str] = []
            index = 0
            while index < len(sentences):
                group: list[str] = []
                chars = 0
                cursor = index
                while cursor < len(sentences) and (
                    chars < 100 or (chars + len(sentences[cursor]) <= 240 and len(group) < 3)
                ):
                    group.append(sentences[cursor])
                    chars += len(sentences[cursor])
                    cursor += 1
                groups.append(" ".join(group))
                index = max(index + 1, cursor - 1)

        search_from = start
        for group in groups:
            group_start, group_end = _locate_clean_text(content or "", group, search_from, end)
            search_from = max(search_from, group_end)
            if len(group) <= 400:
                raw_units.append(("text", group, group_start, group_end))
            else:
                for part, part_start, part_end in _split_long(group, group_start):
                    raw_units.append(("text", part, part_start, part_end))

    return [
        EvidenceUnit(kind, text, start, end, index)
        for index, (kind, text, start, end) in enumerate(raw_units)
    ]


def evidence_content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()
