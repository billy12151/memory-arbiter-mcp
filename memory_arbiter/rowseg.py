"""Row-level segmentation for the conflict-detection channel (0.17.0 P2-2.1).

Sentence-level granularity for prose plus one-row-per-table-row vectors with
the header folded into every row's text. Search keeps the evidence-unit
granularity (dual-granularity, owner decision #5): this module feeds ONLY the
conflict pipeline and its ``memory_row`` store.

Design anchors (spike evidence, plan appendix A):
- R13: sentences beat whole units on R@5/MRR; markdown headings HURT recall —
  headings and the subject are deliberately NOT indexed here (the subject
  already has subject_tags_vec).
- R13: rows shorter than 8 chars are noise-dominated — filtered.
- Table heuristics reuse the difference_classifier shape (``\\|-{2,}`` or a
  high pipe count); row text pairs each cell with its header column so a row
  embeds as "服务:api-gateway 超时:500ms" — attribute and value travel
  together (R12: value-anchored location is what actually works).
"""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import _normalize_with_map, _sentence_slices

ROW_TEXT_MAX_CHARS = 200
ROW_MIN_CHARS = 8
# C3 fallback row cap: the leading-slice sentence row emitted when the content
# yields nothing splittable. The span is the SAME slice the text came from —
# content[start:end] == row text keeps the span contract intact.
FALLBACK_ROW_CHARS = 200
# 行上下文 envelope（owner 2026-09-25 方案 D1）：每部分独立截断，合计不超
# CONTEXT_MAX_CHARS——只进 Qwen prompt（属性名恢复），绝不进向量/索引。
CONTEXT_HEADING_CHARS = 80
CONTEXT_NEIGHBOR_CHARS = 110
CONTEXT_MAX_CHARS = 300


@dataclass(frozen=True)
class RowSegment:
    kind: str  # "subject" | "sentence" | "table_row"
    text: str
    start_offset: int
    end_offset: int
    row_index: int


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    return all(ch in "|-: \t" for ch in stripped)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _is_separator_row(stripped):
        return True
    return stripped.count("|") >= 2


def _split_cells(line: str) -> list[str]:
    parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return [cell for cell in parts if cell != ""]


def _merge_header_rows(rows: list[list[str]]) -> list[str]:
    """Multi-row header inheritance: join per column, drop duplicates."""
    width = max((len(r) for r in rows), default=0)
    merged: list[str] = []
    for col in range(width):
        seen: list[str] = []
        for r in rows:
            if col < len(r) and r[col] and r[col] not in seen:
                seen.append(r[col])
        merged.append(" ".join(seen))
    return merged


def _table_row_text(header: list[str], cells: list[str]) -> str:
    pieces: list[str] = []
    for index, cell in enumerate(cells):
        label = header[index] if index < len(header) else ""
        pieces.append(f"{label}:{cell}" if label else cell)
    return " ".join(pieces)[:ROW_TEXT_MAX_CHARS]


def _line_spans(content: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for line in content.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        spans.append((offset, offset + len(raw), raw))
        offset += len(line)
    return spans


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return (
        bool(stripped)
        and stripped.lstrip("#").startswith(" ")
        and stripped.startswith("#")
    )


def _emit_table_rows(
    block: list[tuple[int, int, str]], rows: list[tuple[str, str, int, int]]
) -> None:
    has_separator = any(_is_separator_row(raw) for _, _, raw in block)
    if not has_separator:
        # Best effort without a separator row: the first line is the header.
        header = _split_cells(block[0][2]) if len(block) >= 2 else []
        data_lines = block[1:]
    else:
        # Rows above the separator are header rows (multi-row merge); rows
        # below are data. A well-formed table has exactly one separator.
        first_separator = next(
            index for index, (_, _, raw) in enumerate(block) if _is_separator_row(raw)
        )
        header = _merge_header_rows(
            [_split_cells(raw) for _, _, raw in block[:first_separator]]
        )
        data_lines = block[first_separator + 1 :]
    for start, end, raw in data_lines:
        if _is_separator_row(raw):
            continue
        cells = _split_cells(raw)
        if not cells:
            continue
        text = _table_row_text(header, cells)
        if len(text) >= ROW_MIN_CHARS:
            rows.append(("table_row", text, start, end))


def row_context_text(
    content: str, start_offset: int, end_offset: int, *, max_chars: int = CONTEXT_MAX_CHARS,
) -> str:
    """行上下文 envelope（owner 2026-09-25 方案）：主行所属标题 + 前一行 +
    后一行，供 Qwen 恢复属性归属（「## 连接池配置」屏障下的「上限调整为
    200」）。合约为**属性名可从上下文恢复、值必须取自主行**——grounding
    只读 envelope 的 quote（主行），机制上挡住从上下文捞值。

    邻行 = 主行之前/之后最近的非空、非标题、非表格分隔行的原始行文本；
    标题 = 主行之前最近的 ``#`` 标题。跨行句（行级分段折叠空白后一行可跨
    多个源行）所在源行不算邻行。各部分独立截断（标题 80 / 邻行 110），
    合计再裁 max_chars；无任何可用片段返回 ""——调用方据此不设 context
    键，prompt 不渲染该段（写入/扫描/旧调用点向后兼容）。"""
    heading = ""
    prev_line = ""
    next_line = ""
    for line_start, line_end, raw in _line_spans(content or ""):
        text = raw.strip()
        if not text or _is_separator_row(raw):
            continue
        if _is_heading(raw):
            if line_end <= int(start_offset):
                heading = text.lstrip("#").strip()[:CONTEXT_HEADING_CHARS]
            continue
        if line_end <= int(start_offset):
            prev_line = text
        elif line_start >= int(end_offset) and not next_line:
            next_line = text
    parts = [
        part[:CONTEXT_NEIGHBOR_CHARS] if len(part) > CONTEXT_NEIGHBOR_CHARS else part
        for part in (heading, prev_line, next_line) if part
    ]
    if not parts:
        return ""
    joined = " / ".join(parts)
    return joined[:max_chars]


def segment_rows(subject: str, content: str) -> list[RowSegment]:
    """Segment memory content into sentence and table-row chunks.

    Offsets are source coordinates (same normalization-to-source map as the
    evidence units), so ``content[start:end]`` always contains the row text
    after whitespace collapse — EXCEPT the subject row (span (0,0), no
    content span; hit_spans drops it, mirroring the unit channel's subject
    convention).

    C3 (owner 2026-09-23, plan A+): a non-empty subject always lands a
    leading ``kind="subject"`` row — self-recall, default-bucket placement,
    and subject-only memories keep their strongest signal in the row world.
    A memory whose content yields no rows at all gets ONE fallback sentence
    row over the leading FALLBACK_ROW_CHARS slice, so every indexable memory
    carries at least one row.
    """
    rows: list[tuple[str, str, int, int]] = []
    spans = _line_spans(content or "")

    table_block: list[tuple[int, int, str]] = []
    prose_start: int | None = None
    prose_end = 0

    def flush_table() -> None:
        nonlocal prose_start, prose_end
        if not table_block:
            return
        if len(table_block) < 2:
            # A single stray pipe line is prose, not a table.
            if prose_start is None:
                prose_start = table_block[0][0]
            prose_end = table_block[-1][1]
        else:
            _emit_table_rows(table_block, rows)
        table_block.clear()

    def flush_prose() -> None:
        nonlocal prose_start
        if prose_start is None:
            return
        mapped = _normalize_with_map(content or "", prose_start, prose_end)
        prose_start = None
        if not mapped.text:
            return
        for sentence in _sentence_slices(mapped):
            if len(sentence.text) >= ROW_MIN_CHARS:
                rows.append(
                    (
                        "sentence",
                        sentence.text,
                        sentence.source_span[0],
                        sentence.source_span[1],
                    )
                )

    for start, end, raw in spans:
        if not raw.strip() or _is_heading(raw):
            # Blank lines and headings (R13: never indexed) are barriers —
            # they also terminate a pending table block so two tables
            # separated by a blank line never merge.
            flush_prose()
            flush_table()
            continue
        if _is_table_line(raw):
            flush_prose()
            table_block.append((start, end, raw))
            continue
        if table_block:
            flush_table()
        if prose_start is None:
            prose_start = start
        prose_end = end
    flush_table()
    flush_prose()

    # C3 fallback: nothing splittable from the content — one sentence row
    # over the leading slice keeps every indexable memory in vector recall
    # (「很短。」/header-only/short-clause memories). The span IS the slice,
    # so content[start:end] == row text (never truncated-text/full-span).
    if not rows and content:
        fallback_text = content[:FALLBACK_ROW_CHARS]
        if fallback_text.strip():
            rows.append(("sentence", fallback_text, 0, len(fallback_text)))

    rows.sort(key=lambda item: (item[2], item[3]))
    if subject and subject.strip():
        # C3 A+ (owner 2026-09-23): the subject row is the leading row. Span
        # (0,0) marks "no content span" — hit_spans drops it, the same
        # convention the unit channel used for subject units. R13's
        # "subject not indexed" is retired; subject_tags_vec keeps its own
        # sorting duty, this row serves evidence recall / self-recall /
        # default-bucket placement.
        rows.insert(0, ("subject", subject.strip(), 0, 0))
    return [
        RowSegment(
            kind=kind, text=text, start_offset=start, end_offset=end, row_index=index
        )
        for index, (kind, text, start, end) in enumerate(rows)
    ]
