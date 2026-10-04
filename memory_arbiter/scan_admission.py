"""scan 配对准入层（从 scan_pipeline.py 搬出，拆分批 ⑤ 纯移动）。

模块级 5 纯函数（_candidate_pair_member/spans_overlap/genuine_numeric_pair/
internal_pair_admission/_workspace_identity）+ 类内配对四方法
（_examine_internal/_pair_identity/_pair_members/_enqueue_pair）。
internal_pair_admission 的 evidence L1160+ 与测试直取经 scan_pipeline re-export
保活。CONFLICT_DETECTOR_VERSION 读取点随迁（e2e patch 面：db_generation 源模块
补丁覆盖，见方案 §6）。
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .constants import SEMANTIC_MAX_ROWS
from .difference_classifier import classify_pair, internal_noise_pair
from .semantic_conflict import decide_evidence

if TYPE_CHECKING:
    from .db import MemoryDB
    from .tools import MemoryTools


class _ScanPairsMixin:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"


    def _examine_internal(
        self, memory_id: int, version: int, workspace: str,
        units: list[dict[str, Any]],
    ) -> int:
        from . import scan_pipeline as _sp  # 拆分批：CDV 经 scan_pipeline 命名空间调用期读（patch 缝=R2-A4 同型）
        """Same-memory unit×unit contradictions (E10 ①, §6⑳).

        0.16.4 §0.5/§2: the whole filter sequence is ONE shared gate —
        ``internal_pair_admission`` below — called identically by the
        write-time side; the callers differ only in what an admitted pair
        means. Here: admitted shapes (check AND notify) land pending for
        agent judgment — no Qwen on the scan side (E11①); a write-time Qwen
        veto row survives via exists() and is never resurrected.

        0.17.0 R2: the per-memory candidate cap aligns with the WRITE side's
        row cap — both bound the examined rows with SEMANTIC_MAX_ROWS
        (constants.py, P2-3.1; the write side applies it in
        _conflicts_deterministic_collect). The scan side previously ran the
        O(n²) pair loop over the full row list unbounded.
        """
        landed = 0
        # 0.17.0 R2 上限对齐：单记忆候选行帽与写入侧同一常量——写入侧
        # _conflicts_deterministic_collect 以 SEMANTIC_MAX_ROWS（constants.py，
        # P2-3.1，值锚定排序后截断）界定参检行，scan 侧此前对全量行做无上限
        # O(n²) 两两检查。对齐为同一常量引用（非硬编码），来源即写入侧行帽。
        units = units[:max(1, SEMANTIC_MAX_ROWS)]
        count = len(units)
        for i in range(count):
            for j in range(i + 1, count):
                a, b = units[i], units[j]
                if not a.get("text") or not b.get("text"):
                    continue
                decision = decide_evidence(str(a["text"]), str(b["text"]))
                if not internal_pair_admission(
                    str(a["text"]), str(b["text"]),
                    (int(a["start_offset"]), int(a["end_offset"])),
                    (int(b["start_offset"]), int(b["end_offset"])),
                    decision,
                    exists_probe=lambda: self.db.internal_conflicts.exists(
                        memory_id, version, int(a["unit_index"]), int(b["unit_index"]),
                    ),
                ):
                    continue
                created = self.db.internal_conflicts.create(
                    memory_id=memory_id, memory_version=version,
                    unit_a=int(a["unit_index"]), unit_b=int(b["unit_index"]),
                    quote_a=str(a["text"]), quote_b=str(b["text"]),
                    span_a=[int(a["start_offset"]), int(a["end_offset"])],
                    span_b=[int(b["start_offset"]), int(b["end_offset"])],
                    reason=decision.reason,
                    detector_version=_sp.CONFLICT_DETECTOR_VERSION,
                )
                if created:
                    landed += 1
        return landed

    def _pair_identity(
        self, memory_id: int, version: int, unit: dict[str, Any],
        peer_id: int, hit: dict[str, Any],
    ) -> tuple[frozenset[str], dict[str, Any], str]:
        """Candidate identity shared with the legacy scan path — the exact
        ``_unit_pair_identity`` contract, so suppression recorded by either
        producer (or record_conflict) suppresses both."""
        from .db.evidence_store import EvidenceStore

        unit_view = {
            "memory_version": version,
            "eid": int(unit["eid"]),
            "start_offset": int(unit["start_offset"]),
            "end_offset": int(unit["end_offset"]),
            "content_hash": str(unit["content_hash"] or ""),
        }
        return EvidenceStore._unit_pair_identity(memory_id, unit_view, peer_id, hit)

    def _pair_members(
        self, memory_id: int, version: int, unit: dict[str, Any],
        peer_id: int, hit: dict[str, Any],
    ) -> list[dict[str, Any]]:
        from . import scan_pipeline as _sp  # 拆分批：CDV 经 scan_pipeline 命名空间调用期读（patch 缝=R2-A4 同型）
        def member(mid: int, ver: int, quote: str, span: list[int], unit_eid: int, content_hash: str) -> dict[str, Any]:
            return {
                "memory_id": mid, "version": ver,
                "attribute_raw": None, "value_raw": None,
                "normalized_attribute": None, "normalized_value": None,
                "evidence_quote": quote, "evidence_span": span,
                "content_hash": content_hash, "evidence_unit": unit_eid,
                "direction": "deterministic", "prompt_version": None,
                "detector_version": _sp.CONFLICT_DETECTOR_VERSION,
            }

        peer_version = int(hit.get("memory_version") or hit.get("memory_row_version") or 1)
        if peer_id < memory_id:
            return [
                member(peer_id, peer_version, str(hit.get("text") or ""),
                       [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)],
                       int(hit.get("id") or 0), str(hit.get("content_hash") or "")),
                member(memory_id, version, str(unit["text"]),
                       [int(unit["start_offset"]), int(unit["end_offset"])],
                       int(unit["eid"]), str(unit["content_hash"] or "")),
            ]
        return [
            member(memory_id, version, str(unit["text"]),
                   [int(unit["start_offset"]), int(unit["end_offset"])],
                   int(unit["eid"]), str(unit["content_hash"] or "")),
            member(peer_id, peer_version, str(hit.get("text") or ""),
                   [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)],
                   int(hit.get("id") or 0), str(hit.get("content_hash") or "")),
        ]

    def _enqueue_pair(
        self, workspace: str, memory_id: int, version: int, unit: dict[str, Any],
        peer_id: int, hit: dict[str, Any], *, decision: Any,
        candidate_key: dict[str, Any], candidate_hash: str,
        pair_cos: "float | None" = None,
    ) -> bool:
        # 0.17.0 review R2：入队时按 compute_pair_score 同式盖章 priority
        # （无 cos 缺 0.40 带项），判定页窗口内按组最高分降序展示——预算
        # 消费顺序信号在此生成一次，不改变任何判定。
        priority = 0.0
        if pair_cos is not None:
            from .pipeline.gates import compute_pair_score
            priority = compute_pair_score(
                decision, float(pair_cos),
                str(unit.get("text") or ""), str(hit.get("text") or ""),
            )
        members = self._pair_members(memory_id, version, unit, peer_id, hit)
        evidence = [
            {
                "memory_id": int(member["memory_id"]),
                "version": int(member["version"]),
                "evidence_quote": member["evidence_quote"],
                "evidence_span": member["evidence_span"],
                "evidence_unit": member["evidence_unit"],
            }
            for member in members
        ]
        outcome = self.db.scan_queue.enqueue(
            kind="conflict",
            workspace_canonical=workspace,
            candidate_key_hash=candidate_hash,
            member_versions=members,
            evidence=evidence,
            reason="; ".join([decision.reason]) if decision.reason else decision.action,
            # 0.16.4 §1: notify pairs no longer enqueue (evolution domain),
            # so the severity split lost its high branch — one value.
            severity="normal",
            source="scan_pipeline",
            priority=priority,
            detail={
                "action": decision.action,
                "distance": float(hit.get("distance") or 0),
                "candidate_key": candidate_key,
            },
        )
        return outcome.get("outcome") in {"queued"}


def _candidate_pair_member(
    memory_id: int, *,
    is_anchor: bool,
    unit: "dict[str, Any]", hit: "dict[str, Any]",
    anchor_text: str, peer_text: str,
) -> dict[str, Any]:
    from . import scan_pipeline as _sp  # 拆分批：CDV 经 scan_pipeline 命名空间调用期读（patch 缝=R2-A4 同型）
    """One ``members`` entry of a scan_rule_candidates pair (0.17.0 R2 收编).

    The candidates store and the duplicates_pool previously assembled this
    13-field member dict in FOUR byte-identical copies (left/right ×
    real/pool); the key set and its order ARE the record/queue contract, so
    the copies collapse into this single producer. ``is_anchor`` selects the
    field source: the anchor's unit row (eid/memory_version/content_hash/
    offsets) vs the peer's hit row (id/memory_version|memory_row_version/...).
    """
    if is_anchor:
        version = int(unit["memory_version"] or 1)
        quote = anchor_text
        span = [int(unit["start_offset"] or 0), int(unit["end_offset"] or 0)]
        content_hash = str(unit["content_hash"] or "")
        evidence_unit = int(unit["eid"])
    else:
        version = int(hit.get("memory_version") or hit.get("memory_row_version") or 1)
        quote = peer_text
        span = [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)]
        content_hash = str(hit.get("content_hash") or "")
        evidence_unit = int(hit.get("id") or 0)
    return {
        "memory_id": memory_id, "version": version,
        "attribute_raw": None, "value_raw": None,
        "normalized_attribute": None, "normalized_value": None,
        "evidence_quote": quote, "evidence_span": span,
        "content_hash": content_hash, "evidence_unit": evidence_unit,
        "direction": "deterministic", "prompt_version": None,
        "detector_version": _sp.CONFLICT_DETECTOR_VERSION,
    }


def spans_overlap(a: "tuple[int, int]", b: "tuple[int, int]") -> bool:
    """True when two evidence spans intersect at all. The long-text fallback
    splitter emits OVERLAPPING windows of one memory, and a unit pair that
    shares source text is a splitter artifact, not a contradiction."""
    a1, a2 = a
    b1, b2 = b
    return a1 < b2 and b1 < a2


def genuine_numeric_pair(quote_a: str, quote_b: str) -> bool:
    """Same-sentence-different-value shape test for internal numeric pairs.

    decide_evidence flags ANY two numeric tokens as numeric_value_candidate;
    on real libraries that fires on enumerated list items ("1. 营销交付" vs
    "7. 复核终审") whose numbers are ordinals, not conflicting values. A
    GENUINE internal numeric contradiction ("超时 30 秒" vs "超时 60 秒")
    repeats the same non-numeric tokens around the differing value — the
    non-digit token Jaccard separates the two shapes (first-round evidence:
    11k enumeration misfires vs the intended handful)."""
    import re

    def tokens(text: str) -> set[str]:
        parts = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", str(text).casefold())
        return {p for p in parts if p}

    ta, tb = tokens(quote_a), tokens(quote_b)
    if not ta or not tb:
        return False
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) >= 0.4


def internal_pair_admission(
    text_a: str, text_b: str,
    span_a: "tuple[int, int]", span_b: "tuple[int, int]",
    decision: Any,
    exists_probe: "Any | None" = None,
) -> bool:
    """0.16.4 §0.5/§2: the SINGLE admission gate for same-memory internal
    pairs, shared verbatim by the scan side (``_examine_internal``) and the
    write-time side (``pipeline/evidence.py``).

    The full filter sequence lives HERE and only here — splitter-artifact
    overlap → ignore → structural noise (0.16.3) → difference clearance
    (sim route) → genuine numeric shape → already-decided identity. The two
    callers differ ONLY in what an admitted pair means downstream: scan
    lands it pending directly (no Qwen, E11①); write-time collects it for
    the Qwen final review — notify shapes INCLUDED since 0.16.4 §2 (an
    in-memory real self-contradiction has a recognition duty, and Qwen's
    verdict is the attribution/veto/fail-open triple). Editing the sequence
    here edits both paths at once; that is the point.

    ``exists_probe`` is a lazy callable (probed only after every semantic
    filter passed) so ignore/noise pairs never pay the identity query.
    """
    if spans_overlap(span_a, span_b):
        return False
    if decision.action == "ignore":
        return False
    # 0.16.3 structural noise shapes (table slices, note-meta lines) never
    # are contradictions — live-library calibrated, 27/280 rows, zero false
    # kills in sampling.
    if internal_noise_pair(text_a, text_b):
        return False
    if decision.action != "notify":
        if decision.reason != "numeric_value_candidate":
            # similarity route: same difference-based clearance as the
            # cross-memory route — no extractable value difference means
            # duplicates/evolution, not conflict
            if classify_pair(text_a, text_b, route=str(decision.reason or "")) == "clear":
                return False
        elif not genuine_numeric_pair(text_a, text_b):
            return False
    if exists_probe is not None and exists_probe():
        return False
    return True


def _workspace_identity(memory_id: int, version: int, suspected: str) -> str:
    import hashlib

    return hashlib.sha256(
        f"workspace:{memory_id}@{version}:{suspected}".encode("utf-8")
    ).hexdigest()
