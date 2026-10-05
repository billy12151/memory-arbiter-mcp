"""命中呈现纯函数层（从 read.py 搬出，拆分批 ④ 纯移动）。

常量+warning 构造+outline/hit_spans/preview 项构造，全部模块级纯函数；
read.py re-export 保活（tests/test_find_enhancements 与 queue_protocol 函数内
import 均从 memory_arbiter.pipeline.read 取名）。
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..evidence import local_text_units

if TYPE_CHECKING:
    pass


# Verified/formally-recorded conflict signal sources that ring the loud
# attention flag. conflict_group is the conflict-groups producer; the retired
# names are kept so legacy payloads still resolve.
_STRONG_CONFLICT_SOURCES = ("open_table", "conflict_guidance", "conflict_group")

# v0.15.4 find index-page preview: bounded outline per result item.
_OUTLINE_MAX_SEGMENTS = 8
_OUTLINE_HEAD_CHARS = 40

# v0.15.10 content_mode="hits": merged hit spans covering >= this share of the
# full content upgrade the item to full text (hit_spans stays as an annotation).
# No truncation anywhere — the owner's ruling is that server-side picking of
# "the important hits" is a systematic bias (legal RAG loses provisos).
_HIT_SPANS_FULL_COVERAGE = 0.5

# 0.17.0 hits window (plan docs/plan-2026-09-23-hits-window.zh-CN.md): a
# hit_window=N request extends each hit with the ±N neighbouring complete
# rows (same memory, same version, subject rows excluded). Cap keeps the
# >=50% coverage upgrade from turning every hits page into a full-text page.
_HIT_WINDOW_MAX = 5

_CONTENT_MODES = ("preview", "hits", "full")




def vec_disabled_warning(reason: str) -> str:
    """升级待办强制提示（owner 指令）：向量通道关闭必须带重建指引。"""
    return (
        f'vec_disabled={reason}: run memory_repair'
        "(task='rebuild_evidence') to restore vector recall"
    )

def _coerce_hit_window(value: Any, warnings: list[str]) -> int:
    """Normalise the caller's hit_window into [0, _HIT_WINDOW_MAX].

    Unparseable/negative → 0 (the byte-identical default); over the cap → the
    cap WITH a warning (a silent clamp would hide the caller's typo).
    Bools coerce as ints on purpose: a harmless display knob, not a span
    coordinate, so the strict-int rejection does not apply here.
    """
    try:
        window = int(value)
    except (TypeError, ValueError):
        return 0
    if window < 0:
        return 0
    if window > _HIT_WINDOW_MAX:
        warnings.append(
            f"hit_window clamped to {_HIT_WINDOW_MAX} (requested {window})"
        )
        return _HIT_WINDOW_MAX
    return window


def _stale_hit_spans_warning(memory_id: int, stale: dict[str, Any]) -> str:
    """Owner-pinned F1 wording: a dropped stale hit must tell the agent the
    index lags and a re-query is needed — never a silent wrong-region slice."""
    return (
        f"item #{memory_id}: hit spans dropped — the memory was likely edited "
        f"after your previous read (evidence index v{stale.get('evidence_version')} "
        f"vs memory v{stale.get('memory_version')}); re-query or re-read to get fresh spans"
    )


def _evidence_lag_warning(memory_id: int, context: str) -> str:
    """F1 companion for read/batch_read: the two silent fallbacks (hits → full
    record, full+span → legacy char slice) now say WHY the unit-aligned shape
    is missing. None is not always version lag — it also fires when the span
    overlaps no indexed unit or the row query fails — so the wording reports
    the missing rows and names version lag as the likely suspect, not a fact."""
    return (
        f"memory #{memory_id}: {context} — no current-version evidence rows matched this read "
        "(the evidence index may lag the memory version after an edit, or rows are not published yet); "
        "if you are reading by earlier offsets, re-query to confirm"
    )

# 0.16.0 batch read (plan §1.5): the full-content byte budget. preview/hits
# payloads are structurally bounded, so a count cap is enough; full content
# needs the byte budget + a structured over-long response (never a silent
# truncation — the agent re-reads items individually).


def _unit_aligned_hits(
    db: Any, memory: dict[str, Any], span: "dict[str, Any] | None",
    rows: "list[dict[str, Any]] | None" = None,
    window: int = 0,
) -> "tuple[list[dict[str, Any]], str | None] | None":
    """Unit-aligned hit spans for an id-driven hits read (plan §6⑨, four-round
    final form): the ``hits`` unit selector on id-driven calls is the per-id
    span coordinate, and mema's content atom is the evidence unit — returning
    complete units removes any half-sentence truncation risk by construction.

    Returns (hit_spans, upgraded_full_content). Each hit_spans entry carries
    the complete unit text plus offsets in read's span coordinate system. When
    covered units span >= 50% of the content the item upgrades to full text
    (same rule as find/batch_find hits). Returns None when the memory has no
    evidence rows for its current version (caller falls back to the char
    window / plain preview).

    ``rows`` (0.16.12 P1-T4): batch_read passes the page-prefetched unfiltered
    unit rows for THIS memory (empty list = prefetched and known empty); None
    keeps the per-id query. Span overlap is applied here in Python with the
    same predicate text_unit_rows applies in SQL.

    ``window`` (0.17.0 hit_window): each span-selected row is extended with
    the ±window neighbouring complete rows from the same version. Neighbours
    carry ``matched: false`` and the selected rows ``matched: true``; with no
    neighbour to add the output shape stays byte-identical (no matched keys,
    row order untouched). window>0 fetches/uses the FULL row set — the span
    selection then happens in Python with the identical predicate, so this is
    no extra cost on either branch (span=None full fetch is today's path).
    """
    full_rows: "list[dict[str, Any]] | None" = None
    if rows is None:
        try:
            fetch_full = window > 0
            rows = db.evidence.text_unit_rows(
                int(memory["id"]), int(memory.get("version") or 1),
                span_start=(int(span["start"]) if (span and not fetch_full) else None),
                span_end=(int(span["end"]) if (span and not fetch_full) else None),
            )
        except Exception:
            return None
        if window > 0:
            full_rows = rows
            if span is not None:
                start = int(span["start"])
                end = int(span["end"])
                rows = [
                    row for row in rows
                    if int(row["start_offset"]) < end and int(row["end_offset"]) > start
                ]
    else:
        full_rows = rows
        if span is not None:
            start = int(span["start"])
            end = int(span["end"])
            rows = [
                row for row in rows
                if int(row["start_offset"]) < end and int(row["end_offset"]) > start
            ]
    if not rows:
        return None
    content = str(memory.get("content") or "")
    hit_spans: list[dict[str, Any]] = [
        {
            # 疑似#6（owner 2026-10-04 拍板：text=原文切片）：find 侧一致口径
            # ——agent 按偏移回读原文得到的就是这段；行级化折叠文本（表格行
            # 折叠形态）不再作为 text 返回。
            "text": content[int(row["start_offset"]):int(row["end_offset"])],
            "start_offset": int(row["start_offset"]),
            "end_offset": int(row["end_offset"]),
            "unit_index": int(row["unit_index"]),
        }
        for row in rows
    ]
    neighbours_added = False
    if window > 0 and full_rows:
        selected_indexes = {int(row["unit_index"]) for row in rows}
        neighbour_indexes: set[int] = set()
        for row in rows:
            ridx = int(row["unit_index"])
            for idx in range(ridx - window, ridx + window + 1):
                if idx not in selected_indexes:
                    neighbour_indexes.add(idx)
        by_index = {int(row["unit_index"]): row for row in full_rows}
        neighbour_rows = [
            by_index[idx] for idx in sorted(neighbour_indexes) if idx in by_index
        ]
        if neighbour_rows:
            neighbours_added = True
            for entry in hit_spans:
                entry["matched"] = True
            for row in neighbour_rows:
                hit_spans.append({
                    # A7（0.17.1 修复批）：邻句同样用原文切片——命中行已改
                    # content[s:e]，邻句仍用 row["text"]（rowseg 折叠文本，
                    # 表格行/跨行句差异最大），同一响应内两种口径且违反
                    # "read span returns exactly that text" 的文档承诺。
                    "text": content[int(row["start_offset"]):int(row["end_offset"])],
                    "start_offset": int(row["start_offset"]),
                    "end_offset": int(row["end_offset"]),
                    "unit_index": int(row["unit_index"]),
                    "matched": False,
                })
            hit_spans.sort(key=lambda entry: (entry["start_offset"], entry["end_offset"]))
    upgraded: str | None = None
    if neighbours_added:
        # D3: coverage is computed on the extended set — merge first so
        # overlapping/adjacent units cannot double-count into an upgrade.
        merged: list[list[int]] = []
        for entry in hit_spans:
            s, e = entry["start_offset"], entry["end_offset"]
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        covered = sum(e - s for s, e in merged)
    else:
        covered = sum(item["end_offset"] - item["start_offset"] for item in hit_spans)
    if content and covered >= _HIT_SPANS_FULL_COVERAGE * len(content):
        upgraded = content
    return hit_spans, upgraded


def _outline_from_rows(rows: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    """Bounded outline from unit rows; ``total`` is the exact heading/text
    unit count (drives the "还有 N 段" marker). Shared by the single-id and
    batch-prefetched table paths."""
    outline: list[dict[str, Any]] = [
        {
            "head": (str(row["text"] or "").splitlines()[0] if row["text"] else "")[:_OUTLINE_HEAD_CHARS],
            "offset": int(row["start_offset"]),
        }
        for row in rows[:_OUTLINE_MAX_SEGMENTS]
    ]
    if total > _OUTLINE_MAX_SEGMENTS:
        outline.append({"head": f"…还有 {total - _OUTLINE_MAX_SEGMENTS} 段", "offset": None})
    return outline


def _outline_for_item(
    db: Any, memory_id: int, version: int, subject: str, content: str,
    rows: "list[dict[str, Any]] | None" = None,
) -> list[dict[str, Any]]:
    """P1-T5 table-first outline: serve from memory_row (0.17.0 C4: rows are
    the outline source — the unit tables retired with the old parity audit)
    and fall back to reparsing exactly when the table cannot serve the
    CURRENT version (post-edit/pre-republish window, or vec-less test paths).

    ``rows``: the caller's batch-prefetched heading/text rows for THIS memory
    (None = the single-item query path). An empty list means the prefetch
    found nothing for the current version → reparse fallback."""
    if rows is None:
        try:
            packed = db.evidence.outline_rows(
                int(memory_id), int(version), limit=_OUTLINE_MAX_SEGMENTS + 1,
            )
        except Exception:
            packed = None
        if packed is None:
            return _content_outline(subject, content)
        return _outline_from_rows(packed["rows"], int(packed["total"]))
    if not rows:
        return _content_outline(subject, content)
    return _outline_from_rows(rows, len(rows))


def _content_outline(subject: str, content: str) -> list[dict[str, Any]]:
    """Bounded table-of-contents for a find preview item.

    Segments reuse local_text_units (heading/text kinds only; the subject is
    excluded) so offsets share read's span
    coordinate system — span={"start": offset, "end": offset + N} slices the
    exact source region. Over-long contents collapse into a trailing
    "…还有 N 段" marker with offset=None.
    """
    units = [
        unit for unit in local_text_units(subject, content)
        if unit.kind in ("heading", "text")
    ]
    outline: list[dict[str, Any]] = [
        {
            "head": (unit.text.splitlines()[0] if unit.text else "")[:_OUTLINE_HEAD_CHARS],
            "offset": unit.start_offset,
        }
        for unit in units[:_OUTLINE_MAX_SEGMENTS]
    ]
    if len(units) > _OUTLINE_MAX_SEGMENTS:
        outline.append({"head": f"…还有 {len(units) - _OUTLINE_MAX_SEGMENTS} 段", "offset": None})
    return outline


def _hit_spans(
    raw_hits: Any, content: str, *, window: int = 0,
    window_rows: "list[dict[str, Any]] | None" = None,
    memory_version: int | None = None,
) -> "tuple[list[dict[str, Any]] | None, bool, int | None]":
    """v0.15.10: build hit_spans from the evidence channel's unit-level hits.

    Returns (spans, dropped_stale, stale_evidence_version) — dropped_stale
    and the max dropped row version drive the caller's stale_hit_spans
    marker + re-query warning (F1, owner 2026-09-23).

    Mandatory cleanups (adversarial-review findings, plan §2.3):
    - subject-kind hits are dropped: the subject unit's offsets are (0,0) and
      carry no content-span meaning (same reason outline excludes it);
    - overlapping/adjacent intervals are merged BEFORE any length math: the
      long-text fallback slices with overlap=60, so naive summation would
      double-count coverage (premature full-text upgrades) and surface
      duplicated text;
    - F1 version alignment: a hit whose evidence row was published for a
      DIFFERENT memory version than the item's current one is dropped — the
      old row's legal offsets would silently slice the wrong region of the
      new content. Hits without a row_version cannot be judged and keep the
      legacy pass-through (synthetic/direct callers; the pipeline always
      attaches row_version since 0.17.0).

    Window (0.17.0 hit_window): each surviving hit is extended with the
    complete rows whose unit_index lies within ±window of the hit's row_index
    (window_rows = the caller's range-prefetched current-version rows). When
    neighbour spans join the list they carry ``matched: false`` and hit spans
    carry ``matched: true`` so the agent can tell retrieval hits from context;
    with no neighbours the output shape stays byte-identical to v0.15.10.

    ``text`` is sliced from the source content so the span and the text are
    strictly self-consistent: read span=[start_offset, end_offset] returns
    exactly this text. Returns (None, dropped_stale, ...) when no hit survives
    (FTS/phrase-only recall has no evidence hits — the item falls back to the
    preview shape).
    """
    intervals: list[tuple[int, int]] = []
    hit_row_indexes: list[int] = []
    dropped_stale = False
    stale_versions: list[int] = []
    for h in raw_hits or []:
        if not isinstance(h, dict) or str(h.get("kind") or "") == "subject":
            continue
        if memory_version is not None:
            raw_rv = h.get("row_version")
            if raw_rv is not None and not isinstance(raw_rv, bool):
                try:
                    rv = int(raw_rv)
                except (TypeError, ValueError):
                    rv = None
                if rv is not None and rv != int(memory_version):
                    dropped_stale = True
                    stale_versions.append(rv)
                    continue
        s_raw, e_raw = h.get("start_offset"), h.get("end_offset")
        # Strict ints (bools rejected — v0.14 span-validation lesson): these
        # come from evidence rows, never user input, but stay defensive.
        if (
            isinstance(s_raw, bool) or isinstance(e_raw, bool)
            or not isinstance(s_raw, int) or not isinstance(e_raw, int)
        ):
            continue
        s, e = s_raw, e_raw
        if 0 <= s < e <= len(content):
            intervals.append((s, e))
            raw_ri = h.get("row_index")
            hit_row_indexes.append(
                int(raw_ri)
                if isinstance(raw_ri, int) and not isinstance(raw_ri, bool)
                else -1
            )
    if not intervals:
        return None, dropped_stale, (max(stale_versions) if stale_versions else None)

    # Window extension: collect the ±N neighbour rows per surviving hit.
    # Subject rows never enter (kind filter here mirrors row_spans_for_ids).
    neighbour_intervals: list[tuple[int, int]] = []
    if window > 0 and window_rows and hit_row_indexes:
        by_index = {
            int(row["unit_index"]): row for row in window_rows
            if row.get("unit_index") is not None
            and str(row.get("kind") or "") != "subject"
        }
        seen: set[int] = set()
        for (s, e), ridx in zip(intervals, hit_row_indexes):
            if ridx < 0:
                continue
            for idx in range(ridx - window, ridx + window + 1):
                if idx == ridx or idx in seen:
                    continue
                row = by_index.get(idx)
                if row is None:
                    continue
                ns_raw, ne_raw = row.get("start_offset"), row.get("end_offset")
                if (
                    isinstance(ns_raw, bool) or isinstance(ne_raw, bool)
                    or not isinstance(ns_raw, int) or not isinstance(ne_raw, int)
                ):
                    continue
                ns, ne = ns_raw, ne_raw
                if 0 <= ns < ne <= len(content):
                    seen.add(idx)
                    neighbour_intervals.append((ns, ne))

    if neighbour_intervals:
        # D4 marking: neighbours matched=false, hits matched=true. Merging
        # runs over the combined set; a merged run is matched=true when it
        # contains (or overlaps) any retrieval hit.
        combined: list[tuple[int, int, bool]] = [
            (s, e, True) for s, e in intervals
        ] + [(s, e, False) for s, e in sorted(set(neighbour_intervals))]
        combined.sort(key=lambda item: (item[0], item[1]))
        merged: list[list[Any]] = []
        for s, e, m in combined:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
                merged[-1][2] = merged[-1][2] or m
            else:
                merged.append([s, e, m])
        return (
            [
                {
                    "text": content[s:e], "start_offset": s, "end_offset": e,
                    "matched": bool(m),
                }
                for s, e, m in merged
            ],
            dropped_stale,
            max(stale_versions) if stale_versions else None,
        )

    intervals.sort()
    merged_plain: list[tuple[int, int]] = []
    for s, e in intervals:
        if merged_plain and s <= merged_plain[-1][1]:
            merged_plain[-1] = (merged_plain[-1][0], max(merged_plain[-1][1], e))
        else:
            merged_plain.append((s, e))
    return (
        [
            {"text": content[s:e], "start_offset": s, "end_offset": e}
            for s, e in merged_plain
        ],
        dropped_stale,
        max(stale_versions) if stale_versions else None,
    )


def _preview_item(
    item: dict[str, Any], *, content_mode: str = "preview", db: Any = None,
    outline_rows: "list[dict[str, Any]] | None" = None,
    hit_window: int = 0,
    window_rows: "list[dict[str, Any]] | None" = None,
) -> dict[str, Any]:
    """Build one find index-page item: metadata + content_chars + outline.

    v0.15.10 content_mode (single-choice enum, replaces the removed
    include_content boolean):
    - "preview" (default): index page, no content — unchanged contract;
    - "hits": + hit_spans (vector-hit unit text + offsets in read's span
      coordinate system). No truncation: when merged hits cover >= 50% of the
      content the item upgrades to full text and hit_spans stays as an
      annotation — the server never picks "the important hits" for the agent;
    - "full": + full content (the old include_content=true escape hatch).

    0.17.0 hit_window: in "hits" mode the spans extend ±hit_window complete
    rows around each hit (window_rows = the caller's one batched prefetch);
    dropped stale-version hits surface as a stale_hit_spans marker (F1).

    Internal underscore debug fields are passed through untouched — the
    debug_ranking=true page contract exposes them, and search_memories strips
    them otherwise. The single exception is _evidence_hits: in "hits" mode it
    is consumed here (and removed) to build hit_spans.
    """
    content = str(item.get("content") or "")
    preview = dict(item)
    preview["content_chars"] = len(content)
    if db is not None and item.get("id") is not None:
        # P1-T5: table-first outline (falls back to reparse inside)
        preview["outline"] = _outline_for_item(
            db, int(item["id"]), int(item.get("version") or 1),
            str(item.get("subject") or ""), content,
            rows=outline_rows if outline_rows is not None else None,
        )
    else:
        preview["outline"] = _content_outline(str(item.get("subject") or ""), content)
    keep_content = content_mode == "full"
    if content_mode == "hits":
        spans, dropped_stale, stale_evidence_version = _hit_spans(
            item.get("_evidence_hits"), content,
            window=hit_window, window_rows=window_rows,
            memory_version=int(item.get("version") or 1),
        )
        if spans:
            covered = sum(e - s for s, e in (
                (sp["start_offset"], sp["end_offset"]) for sp in spans
            ))
            if covered >= _HIT_SPANS_FULL_COVERAGE * len(content):
                keep_content = True
            preview["hit_spans"] = spans
        # spans is None → vector channel contributed nothing on this item
        # (FTS/phrase-only recall, or the embedder is down): the item keeps
        # the plain preview shape.
        if dropped_stale:
            # F1 (owner 2026-09-23): stale-version hits are dropped, never
            # silently sliced against the new content. The caller lifts this
            # marker into a response-level re-query warning.
            preview["stale_hit_spans"] = {
                "evidence_version": stale_evidence_version,
                "memory_version": int(item.get("version") or 1),
            }
        preview.pop("_evidence_hits", None)
    if not keep_content:
        preview.pop("content", None)
    return preview
