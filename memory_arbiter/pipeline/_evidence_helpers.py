"""evidence 共享纯函数层（从 evidence.py 搬出，拆分批 ⑤ 纯移动）。

技术性豁免/退休 gate 槽位/通知 payload/工期公平期限/巨表豁免/分段豁免过滤，
全部模块级纯函数与常量；evidence.py re-export 保活测试与 tools/scan_pipeline
的函数内 import 面。SEMANTIC_JOB_TIMEOUT_MS 读取点随 _job_fair_deadline 迁入
本模块（patch 缝路径变更见方案 §6-2）。
"""
from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from ..constants import SEMANTIC_JOB_TIMEOUT_MS
from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..evidence import evidence_content_hash
from ..text import canon_entity, canon_scope

if TYPE_CHECKING:
    from ..workers import SemanticConflictWorker


# 技术性降级原因清单（测试钉面 test_write_time_notice_e2e 的成员断言用；
# 运行时分类在 _evidence_land 的四分支映射，无直接消费者——两个清单改任一
# 处须同步另一处）。judge_invalid_output 已删：判定形状违约在合一后由
# _make_judge_fn 的 _bad() 降级为 error verdict，分类为 judge_backend_error，
# 该 reason 无产生点。
_TECHNICAL_REASONS = {
    "judge_timeout", "judge_unavailable", "judge_backend_error",
    "judge_budget_exhausted", "notice_budget_exhausted",
    "rows_capped", "pairs_examined_capped",
    "notice_write_failed",
}


def _retired_gate_slot_key(workspace: Any, attribute: Any, subject: Any) -> dict[str, str]:
    """Gate-v2 G3 slot identity without metadata provenance (owner 拍板):
    {"entity": workspace 名, "attribute": 抽取属性, "scope": subject 前 32 字}.
    Identity key only — historical groups keep their old slot_keys untouched,
    new groups speak the new dialect. canon_* stay for storage-side parity
    (both sides of the B-C4 comparison canonicalise identically)."""
    return {
        "entity": canon_entity(str(workspace or "")),
        "attribute": str(attribute or ""),
        "scope": canon_scope(str(subject or "")[:32]),
    }


def _conflict_notice_payload(
    *,
    reason: str,
    attribute: str,
    slot_key: "dict[str, str] | None" = None,
    left_id: int = 0, left_version: int = 1,
    left_value_norm: str = "", left_display: str = "", left_quote: Any = "",
    right_id: int = 0, right_version: int = 1,
    right_value_norm: str = "", right_display: str = "", right_quote: Any = "",
    left_content: str = "", right_content: str = "",
    attr_cos: "float | None" = None,
    left_evidence_extra: "dict[str, Any] | None" = None,
    right_evidence_extra: "dict[str, Any] | None" = None,
    left_member_extra: "dict[str, Any] | None" = None,
    right_member_extra: "dict[str, Any] | None" = None,
    extra: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    """Shared payload assembler for all notice sites (0.17.0 review R2
    r2s-01): the five per-site copies had already drifted — content
    fingerprints existed ONLY on the A-cross leg, so four channels'
    notices had no basis for staleness invalidation. Owner 拍板（2026-09-26
    补齐）：left/right content hashes ride EVERY payload now (same
    evidence_content_hash 口径 as the A-cross leg). A hash is emitted only
    when the site had that side's content in hand — absence means unknown
    (never a hash of the empty string)."""
    left_evidence: dict[str, Any] = {"text": left_quote}
    if left_evidence_extra:
        left_evidence.update(left_evidence_extra)
    right_evidence: dict[str, Any] = {"text": right_quote}
    if right_evidence_extra:
        right_evidence.update(right_evidence_extra)
    payload: dict[str, Any] = {
        "route": "notice_ready",
        "reason": reason,
        "slot_key": slot_key,
        "slot_provenance": {
            "entity": "workspace", "scope": "subject", "attribute": attribute,
        },
        "member_versions": [
            {"memory_id": left_id, "version": left_version,
             "value": left_value_norm,
             "evidence": {"quote": left_quote, **(left_member_extra or {})}},
            {"memory_id": right_id, "version": right_version,
             "value": right_value_norm,
             "evidence": {"quote": right_quote, **(right_member_extra or {})}},
        ],
        "value_groups": [
            {"normalized_value": left_value_norm, "display_value": left_display,
             "members": [f"{left_id}@{left_version}"]},
            {"normalized_value": right_value_norm, "display_value": right_display,
             "members": [f"{right_id}@{right_version}"]},
        ],
        "candidate_key": {
            "detector_version": CONFLICT_DETECTOR_VERSION,
            "members": sorted([f"{left_id}@{left_version}", f"{right_id}@{right_version}"]),
            "evidence": [],
        },
        "left_evidence": left_evidence,
        "right_evidence": right_evidence,
    }
    if left_content:
        payload["left_content_hash"] = evidence_content_hash(left_content)
    if right_content:
        payload["right_content_hash"] = evidence_content_hash(right_content)
    if attr_cos is not None:
        payload["attr_cos"] = round(float(attr_cos), 4)
    if extra:
        payload.update(extra)
    return payload


def _job_fair_deadline(semantic_worker: "SemanticConflictWorker", publish_done_at: "list[float]") -> "float | None":
    """0.17.0 Q1 (相分裂): the fairness deadline was a job closure, now a
    module function so the internal/cross dispatch phases share one
    wall-clock semantics. Detection-
    phase deadline = max(fairness wall, this job's own budget counted from
    publish completion); identical logic to the former closure."""
    value = semantic_worker.pending_job_deadline(
        SEMANTIC_JOB_TIMEOUT_MS / 1000.0,
    )
    if value is None:
        # Idle queue: the old contract stands — no wall, no cap; an
        # in-flight Qwen pair runs to completion.
        return None
    wall = float(value)
    if publish_done_at:
        own = publish_done_at[0] + SEMANTIC_JOB_TIMEOUT_MS / 1000.0
        if wall <= time.monotonic():
            # The fairness wall has ALREADY blown (the oldest queued
            # job waited past its budget) — truncation wins over the
            # own-anchor extension; never let one slow job park the
            # whole queue behind max(wall, own).
            return wall
        # Busy queue, wall still ahead: the embed phase must not eat
        # the detection budget, so count this job's budget from
        # publish completion — but never SHORTEN the wall other
        # queued jobs already rely on.
        return max(wall, own)
    return wall



def _giant_table_indexes(segments: "list[Any]") -> "set[int]":
    """B3（owner 2026-10-03 拍板，方案 B3）：超长表格段的行索引集合。

    段=kind=table_row 且 row_index 连续（rowseg 的 table_block 语义：夹散文
    即断段、各数各的）；段行数 > SEMANTIC_TABLE_ROW_EXEMPT → 该段整体豁免
    （不建行向量/不嵌入/不发布，不参与内外冲突检测）。subject 行不受影响。
    供检测相（deterministic phase 的 streaming/batch 过滤点）与 index-only
    路径（on_write=off / replay postprocess / conflict-apply edits）共用。
    """
    from ..constants import SEMANTIC_TABLE_ROW_EXEMPT

    exempted: set[int] = set()
    run: list[int] = []
    prev_index: int | None = None
    for seg in segments:
        kind = str(getattr(seg, "kind", "sentence"))
        idx = int(getattr(seg, "row_index", getattr(seg, "unit_index", 0)) or 0)
        if kind != "table_row":
            if len(run) > SEMANTIC_TABLE_ROW_EXEMPT:
                exempted.update(run)
            run = []
            continue
        if run and prev_index is not None and idx == prev_index + 1:
            run.append(idx)
        else:
            if len(run) > SEMANTIC_TABLE_ROW_EXEMPT:
                exempted.update(run)
            run = [idx]
        prev_index = idx
    if len(run) > SEMANTIC_TABLE_ROW_EXEMPT:
        exempted.update(run)
    return exempted


def _filter_exempted_segments(segments: "list[Any]") -> "tuple[list[Any], int]":
    indexes = _giant_table_indexes(segments)
    if not indexes:
        return list(segments), 0
    kept = [
        seg for seg in segments
        if int(getattr(seg, "row_index", getattr(seg, "unit_index", 0)) or 0) not in indexes
    ]
    return kept, len(indexes)


def filter_exempted_scan_rows(
    rows: "list[dict[str, Any]]",
) -> "tuple[list[dict[str, Any]], int]":
    """A3（0.17.1 修复批）：扫描腿的 B3 豁免适配。

    ``scan_rows`` 返回 DB dict（键 kind / unit_index），而 ``_giant_table_indexes``
    期望带 .kind/.row_index 的 segment 对象——用轻量 shim 喂同一 helper，
    避免复制聚段逻辑（scan_rows 的 unit_index 就是 row_index，
    evidence_store.py 注释明示）。返回 (保留行, 豁免行数)。
    """
    from collections import namedtuple

    if not rows:
        return [], 0
    _ScanSeg = namedtuple("_ScanSeg", "kind row_index")
    shim = [
        _ScanSeg(
            str(row.get("kind") or "sentence"),
            int(row.get("unit_index") or 0),
        )
        for row in rows
    ]
    indexes = _giant_table_indexes(shim)
    if not indexes:
        return list(rows), 0
    kept = [
        row for row in rows
        if int(row.get("unit_index") or 0) not in indexes
    ]
    return kept, len(indexes)
