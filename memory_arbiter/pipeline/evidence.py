"""Local-text evidence indexing and conflict candidate processing."""
from __future__ import annotations

import hashlib
import sqlite3
import json
import threading
import time
from typing import Any, TYPE_CHECKING, Iterator

from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..constants import (
    EMBED_PREFIX_STS,
    SEMANTIC_JOB_TIMEOUT_MS,
    SEMANTIC_MAX_EXAMINED_PAIRS,
    SEMANTIC_CROSS_KNN_WINDOW,
    SEMANTIC_MAX_ROWS,
    SEMANTIC_MIN_PAIR_BUDGET_MS,
    SEMANTIC_JUDGE_CONTEXT_BEFORE,
    SEMANTIC_JUDGE_CONTEXT_AFTER,
)
from ..difference_classifier import classify_pair
from ..evidence import evidence_content_hash
from ..models import TrustedApplyingContext
from ..embedder import ManagedEmbedder
from ..semantic_conflict import (
    _values_all_equivalent,
    SemanticBackend,
    decide_evidence,
    direct_value_verdict,
    is_cross_evolution,
    notice_dedupe_key,
    normalize_value,
)
from ..semantic_judge import PairVerdict, row_window
from ..text import canon_entity, canon_scope

if TYPE_CHECKING:
    from ..tools import MemoryTools
    from ..workers import SemanticConflictWorker

# Technical failures degrade the check route and keep the job incomplete.
# 0.17.1: qwen_* keys renamed judge_*; qwen_unverified is GONE (grounding
# belonged to the slot-extraction paradigm); qwen_budget_exhausted keeps its
# semantics under the judge_ prefix.
_TECHNICAL_REASONS = {
    "judge_timeout", "judge_unavailable", "judge_backend_error",
    "judge_invalid_output", "judge_budget_exhausted", "notice_budget_exhausted",
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
            prev_index = None
            continue
        if run and idx == prev_index + 1:
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


class _JudgeBatch:
    """0.17.1 (owner 拍板: 攒批进本版): collect judge inputs through a phase's
    gates first (closure / version drift / budget / deadline — the pre-judge
    funnel is unchanged), then fire ONE batched judge_pairs call and drain
    the verdicts. Chunks are fixed-size SEMANTIC_MDEBERTA_BATCH slices; the
    deadline is enforced per pair at collection time (pass 1), not between
    chunks.

    The queue never crosses jobs — each job builds its own instance (R2-3
    no-cross-job-state rule)."""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def add(self, **item: Any) -> None:
        self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def drain(self, judge_fn: "Any", *, stop_probe: "Any = None") -> list[dict[str, Any]]:
        """judge_fn(list[(a, b)]) -> list[PairVerdict]; returns the queued
        items with ``verdict`` attached, in queue order. 逐片契约（R1 实施
        review P2-2）：每片返回数与该片对数逐一校验，违约片连同其后的全部
        item 补 error verdict——此前各片的结果保留，zip 永不跨片错位。
        ``stop_probe``（0.17.1 重标合一）：片间让路探针，返回 True 即停发
        余片——余 item 补 error verdict（忙时公平性，对齐 INTEGRATION 的
        worker-yield 宣称）；已发片结果照常回填。"""
        out: list[dict[str, Any]] = list(self._items)
        self._items = []
        if not out:
            return out
        pairs = [(item.get("judge_text_a", item["text_a"]),
                  item.get("judge_text_b", item["text_b"])) for item in out]
        # 分块恒用常量批档（judge_fn 是闭包，backend.batch_size 探测恒落空
        # 的死分支已删）；设备分档 auto 在 config 解析时定型，子进程内部自
        # 会再按 batch_size 切，此处只做调用切片。
        from ..constants import SEMANTIC_MDEBERTA_BATCH
        chunk = max(1, int(SEMANTIC_MDEBERTA_BATCH))
        verdicts: list[Any] = []
        fail_start: int | None = None
        fail_reason = ""
        for start in range(0, len(pairs), chunk):
            # 0.17.1 重标合一：片间让路探针（方案 §2.2.5）——忙时 63 片不再
            # 一口气越墙，对齐 INTEGRATION「batch 返回后 worker 让路」宣称。
            if stop_probe is not None and start > 0 and stop_probe():
                fail_start, fail_reason = start, "job budget expired mid-batch"
                break
            part = judge_fn(pairs[start : start + chunk])
            if len(part) != len(pairs[start : start + chunk]):
                fail_start = start
                fail_reason = (
                    f"judge returned {len(part)} verdicts "
                    f"for {len(pairs[start:start + chunk])} pairs"
                )
                break
            verdicts.extend(part)
        if fail_start is not None:
            # 对齐保证：fail_start == len(verdicts)（此前各片逐片校验满额）。
            from ..semantic_judge import PairVerdict
            bad = PairVerdict("no_conflict", {}, None, "mdeberta:unavailable", error=fail_reason)
            verdicts.extend([bad] * (len(out) - fail_start))
        for item, verdict in zip(out, verdicts):
            item["verdict"] = verdict
        return out


def _judge_outcome(verdict: Any, min_prob: float) -> str:
    """§3.3 decision table (owner 2026-09-28): conflict ≥ min_prob → 'notice';
    conflict below → 'below_threshold'; possible → 'possible'; no_conflict →
    'clear'. Technical failures surface via verdict.error (the caller's
    fail-open path) and never reach this function."""
    if getattr(verdict, "error", None):
        return "error"
    if verdict.label == "conflict":
        return "notice" if float(verdict.probs.get("conflict", 0.0)) >= min_prob else "below_threshold"
    if verdict.label == "possible_conflict":
        return "possible"
    return "clear"


def _judge_pair_compat(backend: Any, text_a: str, text_b: str) -> Any:
    """Legacy-backend bridge: an extraction-shaped backend exposing only
    classify_pair(env_a, env_b) maps its candidate bool to a conflict /
    no_conflict verdict (test fixtures assert on notice outcomes, which this
    preserves). Returns None when the backend cannot serve."""
    try:
        signal = backend.classify_pair({"quote": text_a}, {"quote": text_b})
    except Exception:
        return None
    if getattr(signal, "candidate", False):
        return PairVerdict(
            "conflict", {"conflict": 1.0, "no_conflict": 0.0, "possible_conflict": 0.0},
            None, "test-backend",
        )
    return PairVerdict(
        "no_conflict", {"conflict": 0.0, "no_conflict": 1.0, "possible_conflict": 0.0},
        None, "test-backend",
    )


def _pair_diff_anchor(text_a: str, text_b: str) -> str:
    """§3.4 slot 差异锚 (owner 拍板): no extraction → a reproducible pair
    hash as the slot attribute. Same pair ⇒ same anchor; different
    oppositions under one subject never collide on a slot. NFKC + whitespace
    collapse so cosmetic reflow cannot fork the identity."""
    import unicodedata

    def canon(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text or "").split())

    digest = hashlib.sha256(
        (canon(text_a) + "\x1f" + canon(text_b)).encode("utf-8"),
    ).hexdigest()
    return digest[:12]


class _JudgePairView:
    """The notice-site payload view of one judged pair — what the judge saw
    and concluded, for model_signal in the notice payload."""

    @staticmethod
    def signal(verdict: Any) -> dict[str, Any]:
        return {
            "label": verdict.label,
            "probs": {k: round(float(v), 4) for k, v in verdict.probs.items()},
            "mechanism": verdict.mechanism,
            "model_version": verdict.model_version,
        }


class _JobJudgeBudget:
    """0.17.1 重标合一（owner 拍板 2026-10-03，方案 §2.3）：internal 帽撤销，
    单一 job 全局判定池——internal keepers 与 A-cross 按提交顺序消费同一个池，
    internal-first 排序保证 land-first（E10①）。receipt 分账按来源保留。

    半池守恒（harness 实测回归修复，2026-10-03）：internal 消费上限
    ⌈total/2⌉——无上限的 internal-first 在行密集语料（数值/表格记忆的
    O(n²) keeper）上会吃光全池，跨记忆对全部 capped→backlog（eval 冲突
    道实测：106 写 0 notice、internal 9390 行、backlog 顶帽驱逐）。跨记忆
    是召回主通道（0.372→0.163 前车之鉴），保底半池；internal 超出份额
    静默消失（池语义不变）。正常写入（internal 1~5 对）不受影响。

    前身 _JobQwenBudget（0.17.0 Q1 owner D1）：internal 保护帽 ≤3 + A-cross
    吃余量——帽的原始理由是 Qwen 一对 ~1.6s、internal keepers 曾吃光共享预算
    （recall 0.372→0.163）；mDeBERTa 批前向 ~0.1s/16 对后前提不成立，帽退役，
    land-first 语义由批量内排序继承。池仍 per-job 实例（R2-3：共享对象上
    不挂跨 job 可变状态）。"""

    def __init__(self, total: int = SEMANTIC_MAX_EXAMINED_PAIRS) -> None:
        self.total = max(1, int(total))
        self.internal_used = 0
        self.a_cross_used = 0

    @property
    def internal_share(self) -> int:
        """internal 可用份额 = ⌈total/2⌉（cross 同样保底 ⌊total/2⌋）。"""
        return max(1, self.total - self.total // 2)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.internal_used - self.a_cross_used)

    def spend(self, source: str) -> bool:
        """单一总池消费 + internal 半池份额。池/份额耗尽返回 False，收尾
        语义按来源分：internal 静默消失、A-cross 记 pairs_examined_capped
        进 backlog。"""
        if source == "internal":
            if self.internal_used >= self.internal_share or self.remaining <= 0:
                return False
            self.internal_used += 1
        else:
            if self.remaining <= 0:
                return False
            self.a_cross_used += 1
        return True

    def receipt_block(self) -> "dict[str, Any] | None":
        """§3.3 conditional receipt block — absent entirely when nothing was
        deducted (zero values never appear)."""
        block: dict[str, Any] = {}
        if self.internal_used:
            block["internal"] = self.internal_used
        if self.a_cross_used:
            block["a_cross"] = self.a_cross_used
        return block or None

    @property
    def pairs_examined(self) -> int:
        """Job-global pairs_examined (§3.3): internal + A-cross."""
        return self.internal_used + self.a_cross_used


class EvidencePipeline:
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools
        self.db = tools.db
        self.settings = tools.settings


    @property
    def _semantic_worker(self) -> "SemanticConflictWorker":
        return self._tools._semantic_worker

    def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_active_embedder()

    def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_embedder()

    def _ensure_semantic_backend(self) -> "SemanticBackend | None":
        return self._tools._ensure_semantic_backend()


    def drain_conflict_backlog(self, limit: int = 2) -> int:
        """0.17.0 P2-4.2: idle-worker consumption of the conflict backlog.

        Bounded per call (``limit`` entries); new writes always win because
        the caller only invokes this when the job queue is empty and rechecks
        between entries. A stored extraction replays through the deterministic
        gate without Qwen; a Qwen-less replay with no extraction lands
        nothing (the entry stays pending for a backend-bearing pass — never
        silently completed)."""
        processed = 0
        skipped: list[int] = []  # unprocessable this pass (no backend) — rotate past, never freeze
        judge_pool: list[dict[str, Any]] = []  # A-5 批量判定池（本轮收齐 pass2 一次判）
        pool_ids: list[int] = []  # 已入池条目必须进 take_next 排除表——条目 pass2
        # 前不 complete 且 processed 不增，不排除则 while 每轮重取同一头部
        # 条目无限循环（0.17.1 review P0）。
        skipped_ids: list[int] = []  # pass2 判定失败留队（不 complete）
        backend_holder: dict[str, Any] = {"backend": None}
        while processed < limit:
            entry = self.db.conflict_backlog.take_next(exclude_ids=skipped + pool_ids)
            if entry is None:
                break
            left_id = int(entry["left_memory_id"])
            right_id = int(entry["right_memory_id"])
            # Version drift re-check at consumption time (refresh_stale is the
            # bulk sweep; this is the per-entry guard).
            rows = self.db.get_memories_by_ids([left_id, right_id])
            left = rows.get(left_id)
            right = rows.get(right_id)
            if (
                not left or not right
                or str(left.get("status")) != "active" or str(right.get("status")) != "active"
                or int(left.get("version") or 1) != int(entry["left_version"])
                or int(right.get("version") or 1) != int(entry["right_version"])
            ):
                self.db.conflict_backlog.refresh_stale()
                continue
            extraction = entry.get("extraction") if isinstance(entry.get("extraction"), dict) else None
            left_text = str(entry["left_text"] or "")
            right_text = str(entry["right_text"] or "")
            decision = decide_evidence(left_text, right_text)
            if decision.action == "ignore" or is_cross_evolution(decision):
                self.db.conflict_backlog.complete(int(entry["id"]))
                processed += 1
                continue
            if extraction is None:
                backend = self._ensure_semantic_backend()
                backend_holder["backend"] = backend
                if backend is None:
                    # P2-4 livelock fix: skip-and-rotate instead of breaking —
                    # the entry stays pending for a backend-bearing pass while
                    # lower-scored entries (if any carry stored extraction)
                    # still drain.
                    skipped.append(int(entry["id"]))
                    # R2-W1：该分支零进展（processed 不增），不设上限时
                    # while processed<limit 只能靠 take_next 扫完全部 pending
                    # 才停（每次调用新开 sqlite 连接，积压 500 ⇒ 空闲 tick 每
                    # 5s 一轮上千次连接空转）。超界即停本轮；条目未 complete
                    # 仍 pending，后端出现后照常重试——契约不变。
                    if len(skipped) > 2 * limit:
                        break
                    continue
                embedder, _warnings = self._ensure_embedder()
                direct = direct_value_verdict(left_text, right_text, decision, embedder=embedder)
                if direct is not None:
                    self._record_backlog_notice(
                        left, right, left_text, right_text, decision,
                        str(direct[0]), str(direct[1]), str(direct[2]),
                        reason="deterministic_same_key_value_diff",
                    )
                    self.db.conflict_backlog.complete(int(entry["id"]))
                    processed += 1
                    continue
                # 0.17.1 (owner A-5)：本 drain 轮收集判定池，pass2 一次批前向
                # （先收后判，与 A-cross 同构）。判定映射：notice/possible →
                # notice；clear → 出队；error/technical → 留队（不 complete，
                # 与 backend 不可用分支同契约；owner 2026-09-28 翻转 0.17.0
                # 的技术失败出队丢弃）。
                entry["left"] = left
                entry["right"] = right
                entry["decision"] = decision
                judge_pool.append(entry)
                pool_ids.append(int(entry["id"]))
                continue
            # Stored extraction replays through the deterministic gate (never
            # re-spends Qwen — owner design #8).
            attr = str(extraction.get("attribute") or "")
            value_a = str(extraction.get("value_a") or "")
            value_b = str(extraction.get("value_b") or "")
            if attr and value_a and value_b and normalize_value(value_a) != normalize_value(value_b):
                self._record_backlog_notice(
                    left, right, left_text, right_text, decision,
                    attr, value_a, value_b, reason="backlog_stored_extraction",
                )
            self.db.conflict_backlog.complete(int(entry["id"]))
            processed += 1
        # ── pass 2: one batched judge call for the collected pool ───────────
        if judge_pool and backend_holder["backend"] is not None:
            from ..semantic_judge import PairVerdict as _PV, locate_row, row_window
            # 0.17.1 §2.2（评审 P1-1 修订版）：drain 判定输入上下文化——
            # 版本复核已通过的 fresh content 走 locate_row 定位窗口。注意：
            # backlog 存的是 rowseg 折叠文本（表格行/跨行句），不是裸
            # content 的逐字子串，locate 对这两类结构性失配 → 退化裸行
            # （无 subject）；逐字单行文本才真正上下文化。与 A-cross（直用
            # 存储 span）不同口径，根治需 backlog 落 span 列（待拍板）。
            def _judge_text(mem: "dict[str, Any]", bare: str) -> str:
                content = str(mem.get("content") or "")
                span = locate_row(content, bare)
                if span is None:
                    return bare
                return row_window(content, span[0], span[1],
                                  subject=str(mem.get("subject") or ""),
                                  before=SEMANTIC_JUDGE_CONTEXT_BEFORE,
                                  after=SEMANTIC_JUDGE_CONTEXT_AFTER)
            pairs = [(_judge_text(e["left"], e["left_text"]),
                      _judge_text(e["right"], e["right_text"])) for e in judge_pool]
            verdicts = backend_holder["backend"].judge_pairs(pairs)
            if len(verdicts) != len(pairs):
                verdicts = [_PV("no_conflict", {}, None, "mdeberta:unavailable",
                                error="judge verdict count mismatch")] * len(pairs)
            min_prob = float(self.settings.semantic_conflict_mdeberta_notice_min_prob)
            for entry, verdict in zip(judge_pool, verdicts):
                outcome = _judge_outcome(verdict, min_prob)
                if outcome == "error":
                    # 留队重试（owner A-5）：下个 backend-bearing pass 再来
                    skipped_ids.append(int(entry["id"]))
                    continue
                left, right = entry["left"], entry["right"]
                left_text, right_text = entry["left_text"], entry["right_text"]
                decision = entry["decision"]
                # 量纲口径与 A-cross 同一（评审 P3④）：decision 在手比
                # decision 值对，无值不触发守卫（notice 照常落）。
                _dv = (getattr(decision, "left_value", None),
                       getattr(decision, "right_value", None))
                if outcome in ("notice", "possible") and not (
                    _dv[0] and _dv[1] and _values_all_equivalent(str(_dv[0]), str(_dv[1]))
                ):
                    self._record_backlog_notice(
                        left, right, left_text, right_text, decision,
                        # 值=行文本（§3.4 快照原文形态，D1 双通道放行）；
                        # [:400] 与 A-cross 组装同构——超 _MAX_FIELD_CHARS
                        # 的长行会让 escalate 在 intake 重新堵死
                        left_text[:32], left_text[:400], right_text[:400],
                        reason=f"backlog_judged:{verdict.label}",
                        severity="normal" if outcome == "notice" else "info",
                        model_signal=_JudgePairView.signal(verdict),
                    )
                self.db.conflict_backlog.complete(int(entry["id"]))
                processed += 1
        return processed

    def _record_backlog_notice(
        self, left: dict[str, Any], right: dict[str, Any],
        left_text: str, right_text: str, decision: Any,
        attribute: str, value_a: str, value_b: str, *, reason: str,
        severity: str = "normal", model_signal: "dict[str, Any] | None" = None,
    ) -> None:
        """Record one notice for a backlog pair through the standard channel
        (dedupe/suppression identical to the write path)."""
        left_id = int(left.get("id") or 0)
        right_id = int(right.get("id") or 0)
        left_version = int(left.get("version") or 1)
        right_version = int(right.get("version") or 1)
        # Gate-v2 G3: the old soft-invisible return (metadata entity/scope
        # missing → drop) is retired WITH the provenance gate — keeping it
        # after the storage strip would have silenced EVERY backlog notice
        # forever (entity/scope are gone from all metadata). Slot identity
        # now rides workspace + subject (plan slot_key 连锁).
        slot_key = _retired_gate_slot_key(
            left.get("workspace_canonical") or left.get("workspace"),
            attribute, str(left.get("subject") or ""),
        )
        extra: dict[str, Any] = {
            "anchors": decision.anchors,
            "backlog": True,
        }
        if model_signal is not None:
            extra["model_signal"] = model_signal
        self.db.record_semantic_notice(
            memory_id=left_id, peer_id=right_id, severity=severity,
            notice_type="semantic_evidence",
            title=f"Possible memory change with #{right_id}",
            message=str(decision.reason or "backlog"),
            payload=_conflict_notice_payload(
                reason=reason,
                attribute="backlog",
                slot_key=slot_key,
                left_id=left_id, left_version=left_version,
                left_value_norm=value_a, left_display=value_a,
                left_quote=left_text,
                right_id=right_id, right_version=right_version,
                right_value_norm=value_b, right_display=value_b,
                right_quote=right_text,
                left_content=str(left.get("content") or ""),
                right_content=str(right.get("content") or ""),
                extra=extra,
            ),
            dedupe_key=notice_dedupe_key(
                left_id, right_id, left_version, right_version, "semantic_evidence",
            ),
            left_version=left_version, right_version=right_version,
            source="semantic_evidence",
        )

    def index_memory(self, memory_id: int, record: dict[str, Any] | None = None) -> dict[str, Any]:
        current = record or self.db.get_memory(int(memory_id))
        embedder, warnings = self._ensure_embedder()
        if current is None:
            return {"status": "skipped", "reason": "memory_not_found"}
        if embedder is None:
            return {"status": "skipped", "reason": "embedder_unavailable", "warnings": warnings}
        vec_state = self.db.get_vec_index_state()
        if vec_state.get("state") in {"mismatch", "failed"} and not (
            vec_state.get("state") == "mismatch"
            and vec_state.get("target_space_id") == embedder.embedding_space_id
            and vec_state.get("space_rebuild_active") is True
        ):
            return {
                "status": "skipped",
                "reason": "embedding_space_rebuild_required",
                "warnings": warnings,
            }
        # C5 (unit retirement): index_memory embeds ROWS ONLY (subject row
        # first, sentences/table rows, C3 fallback — see rowseg), batched via
        # embed_texts; the unit tables are no longer written. Remaining
        # callers: boot backfill / repair — the write path indexes inside the
        # semantic job (C2).
        from ..rowseg import segment_rows
        row_segments = segment_rows(
            str(current.get("subject") or ""), str(current.get("content") or ""),
        )
        row_embeddings: list[list[float]] = []
        ok = True
        for embed_result in embedder.embed_texts([seg.text for seg in row_segments], prefix=EMBED_PREFIX_STS):
            if not embed_result.embedding:
                ok = False
                break
            row_embeddings.append([float(x) for x in embed_result.embedding])
        if not ok:
            return {"status": "failed", "reason": "empty_embedding"}
        # 0.16.12 P2-T2: prefer the row's maintained content_sha column (set
        # at insert and every content edit) — one hash per write instead of
        # re-hashing here; NULL only on exotic legacy rows, hence the fallback.
        row_sha = str(current.get("content_sha") or "") or evidence_content_hash(
            str(current.get("content") or "")
        )
        published = self.db.evidence.publish_rows(
            int(memory_id), int(current.get("version") or 1), row_sha,
            row_segments, row_embeddings,
        )
        if published.get("published"):
            # Self-heal the embedding-space mismatch: once a rebuild has
            # republished every non-deleted memory in the target space, the
            # vec channel flips back to ready (spec §19 defers fancier
            # space-migration tooling; this unblocks the common recovery).
            vec_state = self.db.get_vec_index_state()
            if (
                vec_state.get("state") == "mismatch"
                and vec_state.get("target_space_id") == embedder.embedding_space_id
            ):
                self.db.maybe_complete_space_rebuild(embedder.embedding_space_id)
        return {
            "status": "indexed" if published.get("published") else "failed",
            **published,
        }

    @staticmethod
    def _streamed_pairs(
        ranked_segments: list[Any], embedder: Any, max_segments: int,
    ) -> "Iterator[tuple[Any, list[float]]]":
        """C7: lazy (segment, embedding) stream, one batch ahead.

        A daemon producer thread embeds SEMANTIC_STREAM_BATCH_ROWS at a time
        into a depth-1 queue; the main thread's KNN+gates consume the
        previous batch while the GPU works on the next (the llama call holds
        the embedder lock; KNN never touches it, so the overlap is real).
        Ranking happened BEFORE this call, so the cap/phase-timeout stops
        here always cut the lowest-value tail. Embed failures degrade to
        skipped segments (the memory stays pending for the backfill).
        """
        import queue as _queue
        from ..constants import (
            SEMANTIC_EMBED_PHASE_TIMEOUT_MS, SEMANTIC_STREAM_BATCH_ROWS,
        )

        out: "_queue.Queue[Any]" = _queue.Queue(maxsize=1)
        DONE = object()

        def produce() -> None:
            # P1-1 fix (adversarial review): the EMBED set is uncapped —
            # publish_rows needs every row; the detection cap lives in the
            # consumer. Only the phase wall-clock stops submissions here.
            # P1-2 fix: a failed item keeps its POSITION (yielded as None and
            # the stream ends) so the consumer's landed prefix can never
            # misalign segments with vectors.
            phase_started = time.monotonic()
            try:
                stop = False
                for start in range(0, len(ranked_segments), SEMANTIC_STREAM_BATCH_ROWS):
                    if stop or time.monotonic() - phase_started > SEMANTIC_EMBED_PHASE_TIMEOUT_MS / 1000.0:
                        break  # phase cap: stop submitting, tail is lowest-value
                    batch = ranked_segments[start:start + SEMANTIC_STREAM_BATCH_ROWS]
                    results = embedder.embed_texts([seg.text for seg in batch], prefix=EMBED_PREFIX_STS)
                    pairs: list[tuple[Any, Any]] = []
                    for seg, result in zip(batch, results):
                        if not result.embedding:
                            # A failed item ends the stream AT its position;
                            # publishing the prefix stays aligned, the rest
                            # waits for the backfill.
                            out.put(pairs)
                            out.put([(seg, None)])
                            stop = True
                            break
                        pairs.append((seg, [float(x) for x in result.embedding]))
                    if stop:
                        break
                    out.put(pairs)
            except Exception:
                pass  # degraded embedder: short stream, memory stays pending
            finally:
                out.put(DONE)

        threading.Thread(
            target=produce, name="mema-stream-embed", daemon=True,
        ).start()
        while True:
            item = out.get()
            if item is DONE:
                return
            yield from item

    def index_rows_in_job(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """C2 (0.17.0 worker merge): the job's index-only form.

        Segment + batch-embed + publish rows — the indexing duty that used to
        live on the local-text index worker. Reached for index_only snapshots
        (conflict-apply edits §15.3, replay postprocess) and for every job
        when semantic_conflict_on_write="off". Idempotent: current rows for
        the version win, nothing re-embeds.
        """
        from ..constants import SEMANTIC_EMBED_PHASE_TIMEOUT_MS

        record = self.db.get_memory(int(memory_id))
        if record and record.get("status") == "pending":
            return {"status": "skipped", "reason": "pending_workspace_activation",
                    "index_only": True, "notices_created": 0}
        # 0.17.0 修复（owner 2026-09-25 拍板，C6「非 deleted 保全」口径）：重建
        # 选集含全部非 deleted 状态（active/retired/superseded/expired），而本路径
        # 此前对非 active 一律 memory_not_active 拒绝 → 非活跃记忆的 pending 永远
        # 清不空 → 空间翻转永不触发 → 向量通道锁死（真库实测 106 superseded +
        # 117 retired 卡死翻转）。放行全部非 deleted；检测仍只对 active（上方
        # pending-activation skip 不变）。
        status_now = str((record or {}).get("status") or "")
        if not record or status_now == "deleted":
            return {"status": "incomplete", "reason": "memory_not_active",
                    "index_only": True, "notices_created": 0}
        version = int(record.get("version") or 1)
        if version != int(snapshot.get("version") or 1):
            return {"status": "incomplete", "reason": "stale_snapshot",
                    "index_only": True, "notices_created": 0}
        content = str(record.get("content") or "")
        row_sha = str(record.get("content_sha") or "") or evidence_content_hash(content)
        # _ensure_embedder (NOT the space-gated _ensure_active_embedder): a
        # mismatch rebuild drives this path precisely to WRITE into the new
        # space — the gate would return None and deadlock the flip.
        embedder, _ = self._ensure_embedder()
        if embedder is None:
            return {"status": "incomplete", "reason": "embedder_unavailable",
                    "index_only": True, "notices_created": 0}
        vec_state = self.db.get_vec_index_state()
        if vec_state.get("state") == "failed" or (
            vec_state.get("state") == "mismatch"
            and vec_state.get("target_space_id") != embedder.embedding_space_id
        ):
            return {"status": "incomplete", "reason": "embedding_space_rebuild_required",
                    "index_only": True, "notices_created": 0}
        if vec_state.get("state") in {"mismatch", "failed"}:
            # C2: a mismatch rebuild drives EVERY index_only job through this
            # path — existing rows live in the OLD space, so "already current"
            # would strand the flip. Republish unconditionally; the heal at
            # the tail settles ready once the whole index is in the target
            # space. (The mismatch guard for ordinary detection jobs stays in
            # process_conflicts.)
            pass
        elif self.db.evidence.current_row_vectors(int(memory_id), version, row_sha):
            return {"status": "indexed", "reason": "already_current",
                    "index_only": True, "notices_created": 0}
        from ..rowseg import segment_rows
        phase_started = time.monotonic()
        segments = segment_rows(str(record.get("subject") or ""), content)
        # B3：index-only 路径同样豁免超长表格段（与检测相共用 helper）——
        # on_write=off / replay postprocess / conflict-apply edits 不得成为
        # 巨表嵌入的后门。
        segments, _exempted = _filter_exempted_segments(segments)
        results = embedder.embed_texts([segment.text for segment in segments], prefix=EMBED_PREFIX_STS)
        if time.monotonic() - phase_started > SEMANTIC_EMBED_PHASE_TIMEOUT_MS / 1000.0:
            # The llama call itself cannot be interrupted mid-flight; the cap
            # marks the job incomplete (retry) and keeps the stall observable
            # instead of silently treating a wedged embedder as success.
            return {"status": "incomplete", "reason": "embed_phase_timeout",
                    "index_only": True, "notices_created": 0}
        vectors: list[list[float]] = []
        for embed_result in results:
            if not embed_result.embedding:
                return {"status": "incomplete", "reason": "empty_embedding",
                        "index_only": True, "notices_created": 0}
            vectors.append([float(x) for x in embed_result.embedding])
        published = self.db.evidence.publish_rows(
            int(memory_id), version, row_sha, segments, vectors,
        )
        if published.get("published"):
            # C2: the space-rebuild self-heal moved with the indexing duty —
            # once every non-deleted memory has rows in the target space, the
            # vec channel flips back to ready (spec §19).
            vec_state = self.db.get_vec_index_state()
            if (
                vec_state.get("state") == "mismatch"
                and vec_state.get("target_space_id") == embedder.embedding_space_id
            ):
                self.db.maybe_complete_space_rebuild(embedder.embedding_space_id)
            receipt = {"status": "indexed", "index_only": True,
                       "row_count": int(published.get("row_count") or len(segments)),
                       "notices_created": 0}
            if _exempted:
                # B3：豁免可见不静默（index-only 路径与检测相同口径）
                receipt["table_rows_exempted"] = int(_exempted)
            return receipt
        return {"status": "incomplete", "reason": f"publish_{published.get('outcome')}",
                "index_only": True, "notices_created": 0}

    def conflicts_deterministic_phase(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """0.17.0 Q1 相分裂 (owner plan §3.2) — phase 1 of 3: indexing publish,
        G5 clean list, candidate collection/ordering, internal keeper
        collection, truncation early-exit. ZERO Qwen. Returns the job context
        dict; ``ctx["terminal"]`` set means no further phase may run
        (skipped / stale / truncated — R1-2: the wrapper still rides B and C
        on a truncation terminal, matching the pre-split behavior)."""
        ctx = self._new_conflict_ctx(memory_id, snapshot)
        record = self.db.get_memory(int(memory_id))
        if record and record.get("status") == "pending":
            # A pending (workspace-activation) memory is not an incomplete
            # check: the conflict job is simply skipped until activation,
            # matching the "skipped" semantics used by index_memory above.
            ctx["terminal"] = {"status": "skipped", "reason": "pending_workspace_activation", "notices_created": 0}
            return ctx
        if not record or record.get("status") != "active":
            ctx["terminal"] = {"status": "incomplete", "reason": "memory_not_active", "notices_created": 0}
            return ctx
        if int(record.get("version") or 1) != int(snapshot.get("version") or 1):
            ctx["terminal"] = {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
            return ctx
        content = str(record.get("content") or "")
        # 0.16.12 P2-T2: the row's content_sha column IS sha256(content) (set
        # at insert, recomputed on every content edit) — compare against it
        # instead of re-hashing the content again (legacy NULL falls back).
        row_sha = str(record.get("content_sha") or "")
        if not row_sha:
            row_sha = hashlib.sha256(content.encode()).hexdigest()
        if row_sha != snapshot.get("content_hash"):
            ctx["terminal"] = {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
            return ctx
        # 0.16.12 P2-T6: the COLLECTION phase runs on ONE read-only connection
        # under an explicit read transaction — per-unit evidence_knn and the
        # peer probes all reuse it instead of opening one connection each.
        # Q1 相分裂 (review R1-5): the internal/dispatch phases open their own
        # snapshots — B/C notices now land BETWEEN deterministic collection
        # and dispatch, so dispatch must see the world as of dispatch time.
        phase_started = time.monotonic()
        with self.db.connection() as job_conn:
            job_conn.execute("BEGIN")
            try:
                self._conflicts_deterministic_collect(ctx, record, job_conn, content, row_sha)
            finally:
                try:
                    job_conn.rollback()
                except sqlite3.Error:
                    pass
        ctx["phase_ms"].append((time.monotonic() - phase_started) * 1000)
        return ctx

    def _new_conflict_ctx(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """0.17.0 Q1: the job context threading the three conflict phases.
        The Qwen budget pool lives here — owned by this job, passed
        explicitly, never on the EvidencePipeline instance (review R2-3)."""
        return {
            "memory_id": int(memory_id),
            "snapshot": snapshot,
            "record": None,          # set by the deterministic phase
            "content": "",
            "content_hash": "",
            "workspace": None,
            "internal_version": 1,
            "embedder": None,
            "publish_done_at": [],   # C2 anchor for the fairness deadline
            "budget": _JobJudgeBudget(),
            "min_budget": SEMANTIC_MIN_PAIR_BUDGET_MS / 1000.0,
            "applying_slots": set(),
            "applying_pairs": set(),
            "degradation_reasons": set(),
            "reasons_seen": [],
            "reached_pair": set(),
            "ordered": [],
            "internal_judge_pairs": [],
            "allowed_memory_ids": None,
            "units_examined": 0,
            "rows_mode": False,
            "no_difference_filtered": 0,
            "memory_pairs_excluded": 0,
            "prefiltered_rows": 0,
            "below_cos_floor": 0,
            "repeatability_skipped": 0,
            "internal_found": 0,
            # 历史命名（Qwen 时代），回执/eval 链消费方钉住故不改名——现役判定引擎是 mDeBERTa
            "internal_qwen_confirmed": 0,
            "surfaced": 0,
            "surfaced_peer_ids": set(),
            "dropped_unlocalizable": 0,
            # 0.17.1 §3.3 outcome counters (judge engine): clear/below-
            # threshold are OBSERVABILITY keys, never degradation.
            "model_clear": 0,
            "model_conflict_below_threshold": 0,
            "model_possible_count": 0,
            "model_notices_capped": 0,
            "backlogged": 0,
            "sweep_evicted": 0,
            "incomplete_reason": None,
            "direct_verdicts": 0,
            "terminal": None,
            "truncated": False,
            "phase_ms": [],
        }

    def _record_job_degradation(
        self, ctx: dict[str, Any], reason: str, sample: "str | None" = None,
    ) -> None:
        """Behaviour change (v3 hardening): each degradation reason is counted
        at most once per task — the pair loops can hit the same technical
        failure for many pairs, and counting every hit made
        _check_degradation_count grow with pair count rather than with
        distinct failure modes."""
        reasons: set[str] = ctx["degradation_reasons"]
        if reason in reasons:
            return
        reasons.add(reason)
        ctx["reasons_seen"].append(reason)
        self._tools._record_check_degradation(reason, sample)

    def _enqueue_backlog_entries(
        self, ctx: dict[str, Any],
        entries: "list[tuple[int, tuple[dict[str, Any], Any, Any, float]]]",
    ) -> tuple[int, int]:
        """0.17.0 P2-4.2: truncation leftovers land in conflict_backlog
        instead of vanishing. Identity = detector version + both
        members@version + row anchors (review A7: a detector bump or a
        member edit invalidates the frozen pair)."""
        enqueued = 0
        evicted_total = 0
        record = ctx["record"]
        memory_id = int(ctx["memory_id"])
        left_version = int(record.get("version") or 1)
        for peer_id, (hit, seg_view, decision, pair_cos) in entries:
            # Gate-v2 G6: the backlog priority uses the SAME score as the
            # live Qwen budget — a stale formula would starve high-band
            # pairs after a truncation.
            from .gates import compute_pair_score

            score = compute_pair_score(
                decision, pair_cos, seg_view.text, str(hit.get("text") or ""),
            )
            right_version = int(hit.get("version") or hit.get("memory_row_version") or 1)
            key_hash = hashlib.sha256(
                "|".join((
                    CONFLICT_DETECTOR_VERSION,
                    f"{memory_id}@{left_version}",
                    f"{peer_id}@{right_version}",
                    f"{seg_view.start_offset}-{seg_view.end_offset}",
                    f"{hit.get('start_offset')}-{hit.get('end_offset')}",
                )).encode("utf-8"),
            ).hexdigest()
            outcome = self.db.conflict_backlog.enqueue(
                candidate_key_hash=key_hash,
                left_memory_id=memory_id, left_version=left_version,
                right_memory_id=int(peer_id), right_version=right_version,
                left_text=str(seg_view.text), right_text=str(hit.get("text") or ""),
                pair_score=score,
            )
            if outcome.get("outcome") in {"queued", "duplicate"}:
                enqueued += 1
            evicted_total += int(outcome.get("evicted") or 0)
        return enqueued, evicted_total

    def _collect_applying_slots(self, ctx: dict[str, Any], snapshot: dict[str, Any]) -> None:
        """Deterministic phase step 1 (r2s-02 split): slot-scoped suppression
        keys for conflict groups currently under application.

        Spec §5/§15.3: while a conflict group is applying, versions produced
        by its apply plan must not re-notify THE SAME conflict. Suppression is
        therefore slot-scoped and applied only after the gate resolves the
        candidate's slot_key, so a genuinely different conflict between the
        same two memories is still examined and surfaced. Validation is
        server-side against the live conflict rows; the trusted context only
        names which row to revalidate."""
        memory_id = int(ctx["memory_id"])
        applying_slots: set[str] = ctx["applying_slots"]
        applying_groups: list[dict[str, Any]] = []
        trusted = TrustedApplyingContext.from_dict(snapshot.get("trusted_applying_context"))
        if trusted is not None:
            live = self.db.get_conflict(trusted.conflict_id)
            if live is not None and live.get("status") == "applying":
                plan = (live.get("apply_summary") or {}).get("plan") or []
                plan_ids = {int(item.get("memory_id") or 0) for item in plan}
                trusted_memory = trusted.memory_id
                trusted_revision = trusted.revision
                trusted_action = trusted.action
                # Spec §15.3 preconditions: applying status, exact revision,
                # target in the plan, and the exact action from that plan step.
                revision_ok = (
                    trusted_revision is not None
                    and int(trusted_revision) == int(live.get("revision") or 0)
                )
                step = next(
                    (item for item in plan if int(item.get("memory_id") or 0) == trusted_memory),
                    None,
                )
                action_ok = (
                    bool(trusted_action)
                    and step is not None
                    and str(step.get("action") or "") == trusted_action
                )
                target_ok = trusted_memory in plan_ids
                if revision_ok and action_ok and target_ok:
                    applying_groups.append(live)
        applying_groups.extend(
            group for group in self.db.list_open_conflicts_for_memory_ids(
                [int(memory_id)], include_applying=True,
            ) if group.get("status") == "applying"
        )
        for group in applying_groups:
            if group.get("slot_key"):
                applying_slots.add(json.dumps(
                    group["slot_key"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                ))
            # 0.17.1 owner A-3: pair-identity suppression set — the judged
            # path has no extracted attribute, so the same PAIR under
            # application suppresses by (memory, peer) identity.
            applying_pairs: set[tuple[int, int]] = ctx.setdefault("applying_pairs", set())
            ids = sorted(
                int(m.get("memory_id") or 0)
                for m in (group.get("member_versions") or [])
                if m.get("memory_id") is not None
            )
            if len(ids) >= 2:
                applying_pairs.add((ids[0], ids[-1]))

    def _collect_internal_pairs(
        self, ctx: dict[str, Any], job_conn: "sqlite3.Connection",
        views_with_vectors: "list[tuple[Any, Any]]",
    ) -> bool:
        """B2（owner 2026-10-03 拍板：内部自查一律走向量+漏斗门）：numpy 内存
        top-k 邻居配对，替换 O(n²) 双循环。

        - 批路径：已发布行向量（current_row_vectors）在收集点前就位，原地调用；
        - streaming 首写：流 drain + publish 后、truncation 判断前调用
          （landed 原生 RowSegment 经 getattr 双访问归一，消费方 .unit_index 不破；
          晚于 truncation 会拔掉 E10①「internal findings land despite cross
          truncation」的落地）；
        - numpy ImportError → 返回 False（调用方记 internal_skipped_no_numpy，
          fail-open 可见；无 numpy 的部署本就无判定引擎）。
        漏斗门（与 cross 同序，直接常量比较——内存匹配已有全对余弦，不复用
        candidate_cos_gate 的 dict 形态）：cos floor 出局 → decide_evidence 值
        提取同值 skip → at-ceiling 且值不同放行（同键异值恰是自查目标形态，
        值门比表面余弦准——与 cross 的 at_ceil→重复语义在此一处刻意分歧）→
        internal_pair_admission（噪音/exists 探针，历史行由前序 job 落库快照
        内可见）照旧 → 入池（internal-first/半池守恒/落地全部不变）。
        已知后果（如实）：rows cap 截断的行无向量不进内部配对（原 n² 对 cap 外
        行同样丢弃，无净损失）。"""
        memory_id = int(ctx["memory_id"])
        internal_version = int(ctx["internal_version"])
        from ..scan_pipeline import internal_pair_admission
        from ..constants import (
            SEMANTIC_CANDIDATE_COS_CEIL,
            SEMANTIC_CANDIDATE_COS_FLOOR,
            SEMANTIC_INTERNAL_MAX_ROWS,
            SEMANTIC_INTERNAL_SELF_KNN_K,
        )

        try:
            import numpy as _np
        except ImportError:
            return False

        internal_judge_pairs: list[tuple[Any, Any, Any]] = ctx["internal_judge_pairs"]
        originators = [
            (view, vec) for view, vec in views_with_vectors
            if str(view.kind) != "subject" and vec
        ][:SEMANTIC_INTERNAL_MAX_ROWS]
        if len(originators) < 2:
            return True
        matrix = _np.array([list(map(float, vec)) for _v, vec in originators], dtype=_np.float32)
        norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = matrix / norms
        sims = unit @ unit.T
        k = min(int(SEMANTIC_INTERNAL_SELF_KNN_K), len(originators) - 1)
        seen_pairs: set[frozenset[int]] = set()
        for i in range(len(originators)):
            # top-k 邻居（排除自身：先置 -1 再取前 k+1）
            row = sims[i].copy()
            row[i] = -1.0  # 自身在 -row 中成最大值，被 argpartition 前段天然排除
            top = _np.argpartition(-row, k)[:k]
            for j in top:
                j = int(j)
                if j == i:
                    continue
                pair_key = frozenset({i, j})
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                seg_a, seg_b = originators[i][0], originators[j][0]
                cos = float(sims[i, j])
                if cos < SEMANTIC_CANDIDATE_COS_FLOOR:
                    ctx["below_cos_floor"] = ctx.get("below_cos_floor", 0) + 1
                    continue
                internal_decision = decide_evidence(seg_a.text, seg_b.text)
                left_v = getattr(internal_decision, "left_value", None)
                right_v = getattr(internal_decision, "right_value", None)
                # 同值判定双通道：值对齐（近同文本各抽值）或 reason 判等
                # （完全同文本不抽值——decide_evidence 直接 ignore/equivalent_value）
                same_value = (
                    str(getattr(internal_decision, "reason", "") or "") == "equivalent_value"
                    or (
                        left_v is not None and right_v is not None
                        and _values_all_equivalent(str(left_v), str(right_v))
                    )
                )
                if cos >= SEMANTIC_CANDIDATE_COS_CEIL:
                    # at-ceiling 分歧点：值不同放行（同键异值=自查目标），值同=重复 skip
                    if same_value:
                        ctx["no_difference_filtered"] = ctx.get("no_difference_filtered", 0) + 1
                        continue
                elif same_value:
                    ctx["no_difference_filtered"] = ctx.get("no_difference_filtered", 0) + 1
                    continue
                admitted = internal_pair_admission(
                    seg_a.text, seg_b.text,
                    (seg_a.start_offset, seg_a.end_offset),
                    (seg_b.start_offset, seg_b.end_offset),
                    internal_decision,
                    exists_probe=lambda a=seg_a, b=seg_b: self.db.internal_conflicts.exists_on_conn(
                        job_conn, int(memory_id), internal_version,
                        a.unit_index, b.unit_index,
                    ),
                )
                if not admitted:
                    continue
                internal_judge_pairs.append((seg_a, seg_b, internal_decision))
        return True

    def _collect_neighbour_screen(
        self, ctx: dict[str, Any], record: dict[str, Any], paired: "list[Any]",
        embedder: "Any", job_conn: "sqlite3.Connection",
    ) -> "list[int] | None":
        """Deterministic phase step 3 (r2s-02 split): Gate-v2 G5 ②″ 记忆级
        一揽子筛选. ONE subject-row coarse KNN builds the neighbour list;
        memory_pair_excluded vets each neighbour on subject/tags alone; the
        sentence KNN then runs ONLY inside the clean list (rowid-IN
        restriction — window slots are not burned on unrelated or
        already-excluded memories). 宽不罚——窄才漏. Returns the effective
        clean list (ctx["allowed_memory_ids"] stays the shared contract —
        the A-cross dispatch loop restricts the cross KNN to this list)."""
        memory_id = int(ctx["memory_id"])
        workspace = ctx["workspace"]
        from .gates import memory_pair_excluded as _pair_excluded
        from ..constants import SEMANTIC_NEIGHBOR_SCREEN
        subject_vec = next(
            (embedding for seg_view, embedding in paired if seg_view.kind == "subject"),
            None,
        ) if paired else None
        if subject_vec is None and embedder is not None:
            # First-write streaming path: paired vectors are all None until
            # publish — embed the subject inline (one embed, milliseconds) so
            # the screen runs on the MAIN write path too, not just
            # re-detections (实施后对抗 review P0：粗筛+通道 C 首写从不执行).
            subject_embed = embedder.embed_text(
                prefix=EMBED_PREFIX_STS, body=str(record.get("subject") or ""),
            )
            subject_vec = subject_embed.embedding or None
        allowed_memory_ids: "list[int] | None" = ctx["allowed_memory_ids"]
        if subject_vec is not None:
            neighbours = self.db.row_knn(
                subject_vec, k=SEMANTIC_NEIGHBOR_SCREEN, workspace=workspace,
                exclude_memory_id=memory_id, conn=job_conn,
                include_subject_rows=True, subject_rows_only=True,
            )
            excluded_ids: set[int] = set()
            own_tags = record.get("tags") or []
            for neighbour in neighbours:
                peer_id_n = int(neighbour["memory_id"])
                if peer_id_n in excluded_ids:
                    continue
                tags_raw_n = neighbour.get("tags")
                peer_tags_n = (
                    json.loads(tags_raw_n) if isinstance(tags_raw_n, str) and tags_raw_n
                    else (tags_raw_n if isinstance(tags_raw_n, list) else [])
                )
                if _pair_excluded(
                    str(record.get("subject") or ""), own_tags,
                    str(neighbour.get("subject") or ""), peer_tags_n,
                ):
                    excluded_ids.add(peer_id_n)
            ctx["allowed_memory_ids"] = [
                int(n["memory_id"]) for n in neighbours
                if int(n["memory_id"]) not in excluded_ids
            ]
            allowed_memory_ids = ctx["allowed_memory_ids"]
            ctx["memory_pairs_excluded"] = len(excluded_ids)
            if not allowed_memory_ids:
                ctx["allowed_memory_ids"] = []  # everything screened out: no KNN at all
                allowed_memory_ids = []
        return allowed_memory_ids

    def _conflicts_deterministic_collect(
        self, ctx: dict[str, Any], record: dict[str, Any],
        job_conn: "sqlite3.Connection", content: str, row_sha: str,
    ) -> None:
        """Q1 相分裂 phase 1 body: indexing publish, G5 screen, candidate
        collection/ordering, internal keeper collection, truncation early-
        exit (E10①). Zero Qwen — the backend is fetched by the later phases.
        Terminal outcomes (embedder/vec-state/truncation) land in
        ctx["terminal"] and skip every later phase."""
        memory_id = int(ctx["memory_id"])
        snapshot = ctx["snapshot"]
        embedder, _ = self._ensure_active_embedder()
        if embedder is None:
            ctx["terminal"] = {"status": "incomplete", "reason": "embedder_unavailable", "notices_created": 0}
            return
        if self.db.get_vec_index_state().get("state") in {"mismatch", "failed"}:
            ctx["terminal"] = {
                "status": "incomplete",
                "reason": "embedding_space_rebuild_required",
                "notices_created": 0,
            }
            return
        ctx["record"] = record
        ctx["content"] = content
        ctx["content_hash"] = row_sha
        ctx["workspace"] = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        ctx["internal_version"] = int(record.get("version") or 1)
        ctx["embedder"] = embedder
        self._collect_applying_slots(ctx, snapshot)
        applying_slots: set[str] = ctx["applying_slots"]

        def backlog_deadline() -> "float | None":
            # C2: detection-phase deadline = max(fairness wall, this job's
            # own budget counted from publish completion) — the shared
            # implementation lives in _job_fair_deadline.
            return _job_fair_deadline(self._semantic_worker, ctx["publish_done_at"])

        max_rows = max(1, SEMANTIC_MAX_ROWS)
        workspace = ctx["workspace"]
        by_peer: dict[int, tuple[dict[str, Any], Any, Any, float]] = {}

        # P2-T2: same digest as the stale check above — the maintained
        # content_sha column (or its recompute fallback), never a fresh hash.
        content_hash = row_sha
        # C2: publish_done_at anchors this job's own detection budget AFTER
        # the index phase (embedding is index work, not conflict budget).
        # Q1 相分裂: the list lives in ctx so the later phases share one
        # anchor (append is in-place — the local alias stays a live view of
        # ctx state).
        publish_done_at: list[float] = ctx["publish_done_at"]
        # 0.17.0 C2: the index duty lives in the job. Read the published row
        # vectors; when they are missing, recover in-job with
        # segment+batch-embed+PUBLISH (invariant: publish precedes any Qwen
        # call — a Qwen stall must never cost the search vectors). The old
        # two-queue chain (evidence worker → semantic forward) is gone.
        row_vectors = self.db.evidence.current_row_vectors(
            int(memory_id), int(record.get("version") or 1), content_hash,
        )
        # C7 streaming: on the first-write path the job does NOT embed inline.
        # The cross-memory loop below consumes a lazy (segment, embedding)
        # stream produced one batch ahead on a single worker thread, so
        # KNN+gates overlap the GPU work; publish_rows lands AFTER collection
        # (still ahead of every Qwen call — invariant unchanged) and the
        # space-rebuild heal moved with it.
        pending_segments: list[Any] = []
        if not row_vectors and embedder is not None:
            from ..rowseg import segment_rows
            pending_segments = list(
                segment_rows(str(record.get("subject") or ""), content)
            )
        elif row_vectors:
            # Already-published rows: the collection phase is over the index,
            # anchor the detection budget now (the streaming path anchors
            # after its post-collection publish instead).
            publish_done_at.append(time.monotonic())
        # C5 (unit retirement): rows are the only candidate source. No rows
        # recoverable in-job (degraded embedder) → the memory stays pending
        # for the backfill; the detection phase sees an empty segment set
        # rather than falling back to units.
        rows_mode = bool(row_vectors) or bool(pending_segments)
        ctx["rows_mode"] = rows_mode
        # Normalized segment view: rows carry row_index, units carry
        # unit_index — the view exposes .unit_index for BOTH so every
        # downstream consumer (internal create, envelopes, member evidence)
        # stays unchanged (P2-3.1 keeps every gate a pure text-pair function;
        # only the input granularity changed).
        from collections import namedtuple
        # C7: kind rides the view so consumers can skip the subject row
        # (index participant, never a pair originator — C3 A+ guard).
        _SegView = namedtuple("_SegView", "text start_offset end_offset unit_index kind")
        # B3 表格段豁免（owner 2026-10-03 拍板，方案 B3）：单一过滤点盖两条
        # 路径——streaming 滤 pending_segments（不嵌入不发布不建 memory_row）；
        # batch 滤读回行（新规则前入库的老表不再参与检测）。公共 helper 与
        # index-only 路径共用（_giant_table_indexes）。
        _exempted_total = 0
        if pending_segments:
            pending_segments, _n = _filter_exempted_segments(pending_segments)
            _exempted_total += _n
        elif row_vectors:
            _kept_segs, _n = _filter_exempted_segments([seg for seg, _e in row_vectors])
            if _n:
                _kept_idx = {
                    int(getattr(k, "row_index", getattr(k, "unit_index", 0)) or 0)
                    for k in _kept_segs
                }
                row_vectors = [
                    (seg, emb) for seg, emb in row_vectors
                    if int(getattr(seg, "row_index", getattr(seg, "unit_index", 0)) or 0) in _kept_idx
                ]
                _exempted_total += _n
        if _exempted_total:
            ctx["table_rows_exempted"] = ctx.get("table_rows_exempted", 0) + _exempted_total
        paired: list[tuple[Any, Any]] = list(row_vectors) or [
            (seg, None) for seg in pending_segments
        ]
        # P2-3.1 值锚定行优先：rows carrying an extractable value lead the cap
        # order (12th round: value features are the conflict predictor; topic
        # similarity is not). Deterministic tiebreak by segment order. C7:
        # ranking precedes batching, so a deadline/cap hit stops later
        # batches and always cuts the lowest-value tail.
        from ..semantic_conflict import _VALUE_RE
        paired.sort(
            key=lambda pair: (
                0 if _VALUE_RE.search(pair[0].text) else 1,
                int(getattr(pair[0], "row_index", 0)),
            )
        )
        seg_views = [
            _SegView(
                seg.text, int(seg.start_offset), int(seg.end_offset),
                int(seg.row_index), str(getattr(seg, "kind", "sentence")),
            )
            for seg, _embedding in paired
        ]
        seg_embeddings = [embedding for _seg, embedding in paired]
        max_segments = max_rows
        segments_capped_reason = "rows_capped"
        # B2：批路径向量在手，就地收集；streaming 延后到流 drain+publish 后
        # （见 truncation 判断前的 landed 收集点）——job_conn 快照对同相位
        # 发布必然不可见，DB KNN 不可用，内存向量是唯一正确来源。
        if not pending_segments:
            ctx["_internal_knn_ok"] = self._collect_internal_pairs(
                ctx, job_conn, list(zip(seg_views, seg_embeddings)),
            )
        # 0.16.2 write-time pre-gates (owner, data-driven): two deterministic
        # filters run in the KNN collection loop, BEFORE the per-peer dedup —
        # a cleared representative would otherwise burn a peer slot that a
        # kept hit of the same peer could have taken (live-library
        # simulation: 86 slots recoverable).
        # Gate 0 (0.16.4 §1, evolution domain): cross-memory notify shapes
        # die above, before all of the following — timeline phenomena are
        # not conflicts.
        # Gate 1 (provenance) is RETIRED in gate-v2 G3 (owner 拍板 1): the
        # real library left entity/scope empty on both sides of true
        # conflicts (#50), so the gate made everyone mutually invisible.
        # Claims attribute alignment carries the same-subject signal now.
        # Gate 2 (difference classifier): no extractable value difference
        # means the pair can never satisfy Qwen's same-attribute-different-
        # value gate. (Counter lives in ctx — the finalize phase reads it.)
        # Spec §15.5: a bounded check that ran out of budget must not later
        # claim checked_no_notice. The two truncation causes report
        # distinctly (2026-09-10 #957/#959 diagnosis: the shared string cost
        # an extra investigation round): the per-memory row cap is
        # rows_capped; the fair job deadline stays notice_budget_exhausted.
        # The cap is checked first so a state where both hold attributes to
        # the more specific cause.
        truncation_reason: str | None = None
        # 0.17.0 P2-3.1: the cross loop walks the normalized segments —
        # row_knn in rows mode (candidates are clean short sentences or
        # header-folded table rows). Rows carry no 'text'-only kind filter
        # (table rows are first-class candidates).
        landed: list[tuple[Any, list[float]]] = []
        streaming = bool(pending_segments)
        ranked_pending: list[Any] = []
        if streaming:
            # C7: iterate the lazy stream (ranking already applied to
            # pending_segments via `paired`). The stream is ALWAYS drained:
            # the producer thread must exit (no leak) and publish needs every
            # embedding — the detection cap/deadline below stop DETECTION,
            # never the collection of vectors (P1/P2 fixes, adversarial
            # review: partial publishes and dead threads are both gone).
            ranked_pending = [seg for seg, _none in paired]
            pair_iter = self._streamed_pairs(
                ranked_pending, embedder, max_segments,
            )
        else:
            pair_iter = iter(zip(seg_views, seg_embeddings))
        # Gate-v2 G4: the sentence prefilter is an OPTIONAL layer — the
        # write path runs it (and the claims-coverage skip, owner 拍板), the
        # scan path never sees this code (gates.row_prefilter is one shared
        # implementation; the scan编排 simply does not call it). Filtered /
        # covered rows do NOT count against rows_examined — their counters
        # are their own receipt keys.
        from .gates import candidate_cos_gate, row_prefilter
        # 0.17.1（评审 P2）：claim-coverage 跳过删后 current_claims 无消费方，
        # 白开一次表查询——整行删除（claims 数据层已连表 DROP，检测线零读取）。
        # KEYED BY ROW INDEX, never object identity: the streaming path
        # yields the raw segments while seg_views are _SegView copies — the
        # same row under two Python objects. row_index is the stable key
        # across both.
        admissible_row_idx = {
            int(getattr(view, "unit_index", getattr(view, "row_index", 0)))
            for view in row_prefilter(seg_views)
        }
        allowed_memory_ids = self._collect_neighbour_screen(
            ctx, record, paired, embedder, job_conn,
        )
        try:
            for seg_view, embedding in pair_iter:
                if streaming:
                    if embedding is None:
                        # Producer signalled a failed item at this position:
                        # the landed prefix is aligned and complete; the rest
                        # waits for the backfill (no partial-with-holes
                        # publish).
                        break
                    landed.append((seg_view, embedding))
                if seg_view.kind == "subject":
                    continue  # C3 A+ guard: indexed, never a pair originator
                if int(getattr(seg_view, "unit_index", getattr(seg_view, "row_index", 0))) not in admissible_row_idx:
                    # Gate-v2 G4: the row failed the sentence prefilter — it
                    # never originates a KNN query and never spends budget.
                    # 0.17.1: claim 覆盖句跳过随 claim 对比通道退役（owner 拍板：
                    # 覆盖句重新从通道 A 发起，否则成检测死区）。
                    ctx["prefiltered_rows"] += 1
                    continue
                if ctx["units_examined"] >= max_segments:
                    truncation_reason = truncation_reason or segments_capped_reason
                    continue  # detection capped; keep draining for publish
                active_deadline = backlog_deadline()
                if active_deadline is not None and time.monotonic() >= active_deadline:
                    truncation_reason = truncation_reason or "notice_budget_exhausted"
                    continue  # budget gone; keep draining for publish
                ctx["units_examined"] += 1
                if allowed_memory_ids is not None and not allowed_memory_ids:
                    continue  # whole neighbourhood screened out
                knn_hits = self.db.row_knn(
                    embedding, k=SEMANTIC_CROSS_KNN_WINDOW, workspace=workspace,
                    exclude_memory_id=memory_id, conn=job_conn,
                    include_subject_rows=False,  # subject rows poison the window
                    include_memory_ids=allowed_memory_ids,
                )
                # Gate-v2 G4 余弦门: true cosine band on fetched vectors —
                # below floor is noise (保安一号), at/above ceil is a
                # duplicate that belongs to the similarity channel, never a
                # conflict report; the band split stays observable.
                hit_vectors = self.db.evidence.row_vectors_for_ids(
                    [int(hit["id"]) for hit in knn_hits], conn=job_conn,
                )
                gated, below_floor_pairs, at_ceil_pairs = candidate_cos_gate(
                    embedding, knn_hits, hit_vectors,
                )
                ctx["below_cos_floor"] += len(below_floor_pairs)
                ctx["repeatability_skipped"] += len(at_ceil_pairs)
                gated_cos = {id(hit): cos for hit, cos in gated}
                for hit in knn_hits:
                    if id(hit) not in gated_cos:
                        continue
                    decision = decide_evidence(seg_view.text, str(hit.get("text") or ""))
                    if decision.action == "ignore":
                        continue
                    # Gate-v2 G5: the whole-memory process-record veto moved
                    # into memory_pair_excluded (once per peer at the screen,
                    # not once per hit here).
                    # 0.16.4 §1: cross-memory evolution domain — the earliest
                    # kill. It happens BEFORE the provenance gate, so a notify
                    # shape never consumes provenance/classifier work, a peer
                    # slot, a sort position, or Qwen budget. Same predicate as
                    # the scan side (§0.5 single implementation).
                    if is_cross_evolution(decision):
                        continue
                    peer_id = int(hit["memory_id"])
                    if classify_pair(
                        seg_view.text, str(hit.get("text") or ""), route=str(decision.reason or ""),
                    ) == "clear":
                        ctx["no_difference_filtered"] += 1
                        continue
                    existing = by_peer.get(peer_id)
                    closer = existing is not None and float(hit.get("distance") or 9) < float(existing[0].get("distance") or 9)
                    # 0.16.4 §1: only check shapes reach here now, so the
                    # notify-priority protection lost its subject — the closer
                    # neighbour of the same peer wins outright. pair_cos rides
                    # the representative for the gate-v2 G6 ranking (band
                    # membership), computed already by the cosine gate.
                    if existing is None or closer:
                        by_peer[peer_id] = (hit, seg_view, decision, gated_cos[id(hit)])
        except Exception:
            # A mid-collection failure must not strand the producer thread
            # (P2 leak fix): closing the generator wakes its queue wait; the
            # daemon thread's remaining put is unbounded-safe (queue depth 1
            # drains via GC'd consumer... belt: producer puts are followed by
            # a final DONE put that may block — the daemon flag keeps the
            # process free to exit regardless).
            try:
                close = getattr(pair_iter, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            raise
        if streaming:
            # C7 (P1 fix, adversarial review): publish ONLY the complete,
            # position-aligned set. A short stream (phase timeout / failed
            # embed) publishes NOTHING — current_row_vectors stays empty, the
            # next job or backfill re-embeds whole; a partial-with-holes or
            # misaligned publish would poison the row store until the next
            # version bump.
            publishable = (
                landed if len(landed) == len(ranked_pending) else []
            )
            if publishable:
                published = self.db.evidence.publish_rows(
                    int(memory_id), int(record.get("version") or 1), content_hash,
                    [seg for seg, _embedding in publishable],
                    [embedding for _seg, embedding in publishable],
                )
                if published.get("published"):
                    vec_state = self.db.get_vec_index_state()
                    if (
                        vec_state.get("state") == "mismatch"
                        and vec_state.get("target_space_id") == embedder.embedding_space_id
                    ):
                        self.db.maybe_complete_space_rebuild(embedder.embedding_space_id)
                # stale_snapshot et al: in-memory vectors served this run; the
                # next job lands the new version's rows.
            elif ranked_pending:
                truncation_reason = truncation_reason or "embed_phase_incomplete"
            publish_done_at.append(time.monotonic())
            # B2 streaming 插点：流 drain 完成后、truncation 判断前——内部
            # keeper 收集必须在 truncation 消费（E10① unannotated 落地）之前。
            # landed 是原生 RowSegment（无 .unit_index）——getattr 双访问归一
            # 为 _SegView 形态（:1305 同款先例），两个消费方读 .unit_index 不破。
            from collections import namedtuple as _nt
            _LandedView = _nt("_LandedView", "text start_offset end_offset unit_index kind")
            _landed_views = [
                (_LandedView(
                    seg.text, int(seg.start_offset), int(seg.end_offset),
                    int(getattr(seg, "unit_index", getattr(seg, "row_index", 0))),
                    str(getattr(seg, "kind", "sentence")),
                ), embedding)
                for seg, embedding in landed
            ]
            ctx["_internal_knn_ok"] = self._collect_internal_pairs(
                ctx, job_conn, _landed_views,
            )
        if truncation_reason:
            # E10① order guarantee: internal findings land BEFORE the cross
            # loop and survive its truncation. The internal Qwen pass has not
            # run yet at this point (it needs the backend fetched by its own
            # phase), so the collected keepers land unannotated here —
            # fail-open, never lost. (Adversarial self-review: without this,
            # a row cap or budget exhaustion mid-collection silently dropped
            # every internal keep pair of this run.)
            for unit_a, unit_b, internal_decision in ctx["internal_judge_pairs"]:
                if self.db.internal_conflicts.create(
                    memory_id=int(memory_id), memory_version=int(ctx["internal_version"]),
                    unit_a=unit_a.unit_index, unit_b=unit_b.unit_index,
                    quote_a=unit_a.text, quote_b=unit_b.text,
                    span_a=[unit_a.start_offset, unit_a.end_offset],
                    span_b=[unit_b.start_offset, unit_b.end_offset],
                    reason=str(internal_decision.reason or ""),
                    detector_version=CONFLICT_DETECTOR_VERSION,
                ):
                    ctx["internal_found"] += 1
            self._record_job_degradation(ctx, truncation_reason)
            early_result: dict[str, Any] = {
                "status": "incomplete", "reason": truncation_reason,
                "notices_created": 0, "reasons_seen": ctx["reasons_seen"],
                # Internal-truncation exits before the cross-memory loop even
                # starts, so no pairs were examined yet (counter not yet live).
                "pairs_examined": 0,
            }
            if ctx["internal_found"]:
                early_result["internal_conflicts"] = ctx["internal_found"]
            # 0.17.0 P2-4.2: collected-but-unexamined candidates go to the
            # backlog (owner design #8) — the receipt says how many.
            early_backlogged, early_evicted = self._enqueue_backlog_entries(ctx, list(by_peer.items()))
            if early_backlogged:
                early_result["backlogged"] = early_backlogged
            if early_evicted:
                early_result["backlog_evicted"] = early_evicted
            # Q1 R1-2: truncation freezes the whole dispatch side (no internal
            # Qwen, no A-cross) — the wrapper still rides B/C on this terminal,
            # matching the pre-split behavior exactly.
            ctx["terminal"] = early_result
            ctx["truncated"] = True
            return

        self._collect_order_candidates(ctx, by_peer)

    def _collect_order_candidates(
        self, ctx: dict[str, Any], by_peer: dict[int, "tuple[dict[str, Any], Any, Any, float]"],
    ) -> None:
        """Deterministic phase step 4 (r2s-02 split): candidate ordering.

        (Q1 相分裂: the backend fetch moved into the internal/dispatch
        phases — the deterministic phase stays Qwen-free.)
        C4 soft ordering (⑦ 定案): rank same-level pairs by subject+tags
        overlap before distance — the Qwen budget should spend on pairs the
        owner's signals (subject/tag) already flag as related. Zero-overlap
        pairs are only ordered later, never excluded. (0.16.4 §1: the
        notify-first key lost its subject — only check shapes remain.)
        0.17.0 P2-3.4: pair_score orders the Qwen budget — value features
        lead (routed numeric + both-sides-extractable), C4 subject/tags
        overlap is the base, row distance the tiebreak. Order-only: a
        single pair's verdict never changes (owner-approved boundary).
        Weights initial; P2-3.2 recalibrates on the noisy corpus.
        0.15.14 (A5): the former surfaced>=max_notice_pairs early stop is
        gone — notices are recorded per-pair inside the loop (write-on-
        discovery), so an early stop only saved Qwen time, which the
        examined-pairs cap now bounds deterministically.
        Q1 (owner D1): SEMANTIC_MAX_EXAMINED_PAIRS is the job-global judge
        pool living in ctx["budget"] (internal + A-cross); the
        per-phase caps live in _JobJudgeBudget."""
        memory_id = int(ctx["memory_id"])
        from ..semantic_conflict import vector_cosine

        hint_vectors = self.db.memories.subject_tags_vectors(
            [memory_id, *[peer_id for peer_id in by_peer]],
        )
        own_vector = hint_vectors.get(int(memory_id))
        if own_vector is None:
            overlap_rank: dict[int, float] = {peer_id: 0.0 for peer_id in by_peer}
        else:
            overlap_rank = {
                peer_id: vector_cosine(own_vector, hint_vectors.get(peer_id))
                for peer_id in by_peer
            }

        def _pair_score(
            peer_id: int, triple: "tuple[dict[str, Any], Any, Any, float]",
        ) -> float:
            from .gates import compute_pair_score

            _hit, _seg, decision, pair_cos = triple
            return compute_pair_score(decision, pair_cos, _seg.text, str(_hit.get("text") or ""))

        def _value_gap(triple: "tuple[dict[str, Any], Any, Any, float]") -> float:
            _hit, _seg, decision, _pair_cos = triple
            if not (decision.left_value and decision.right_value):
                return 0.0
            try:
                left_num = float(normalize_value(decision.left_value).rstrip("ms条次%") or 0)
                right_num = float(normalize_value(decision.right_value).rstrip("ms条次%") or 0)
                return abs(left_num - right_num)
            except (TypeError, ValueError):
                return 0.0

        ctx["ordered"] = sorted(
            by_peer.items(),
            key=lambda item: (
                -_pair_score(item[0], item[1]),
                -_value_gap(item[1]),
                -float(overlap_rank.get(item[0]) or 0.0),
                float(item[1][0].get("distance") or 9),
            ),
        )

    def conflicts_judge_phase(self, ctx: dict[str, Any]) -> None:
        """0.17.1 重标合一（owner 拍板 2026-10-03，方案 §2.2）：internal 与
        A-cross 两个判定消费方合并为单相位、一批发出。internal keepers 在
        提交列表前部（E10① land-first 改为批量内排序保证）；单一总池
        （internal 帽撤销——Qwen 一对 ~1.6s 的前提随 mDeBERTa 批前向
        ~0.1s/16 对不成立）。

        三段形状：pass 1 收集（internal 先、cross 后，逐候选保留全部前置
        检查；R1-5 的跨对读快照事务包裹收集与跨对落地）→ 一批 drain
        （judge_fn 统一、逐片异常免疫、片间让路探针）→ 按来源分账落地
        （各自现役连接纪律）。池耗尽的 internal 尾对静默消失（现状帽
        break 语义的池化等价，R2 对抗轮修正对照物）；判定缺席/让路窗口
        触发的 internal 对 unannotated 立即落地（fail-open，消费池保上限
        ——帽撤销后唯一的有界保证）。0.17.1（owner 拍板 2026-09-28，不变）：
        judge 写时永不 dismiss——conflict ≥ min_prob → 标注落地；其余
        （no_conflict 任意置信、possible、below_threshold、error）unannotated
        或带否定意见落地，扫描侧强模型拥有否定裁决权。"""
        phase_started = time.monotonic()
        backend = self._ensure_semantic_backend()
        budget: _JobJudgeBudget = ctx["budget"]
        min_budget: float = ctx["min_budget"]
        min_prob = float(self.settings.semantic_conflict_mdeberta_notice_min_prob)
        content = str(ctx["content"])
        record = ctx["record"]
        own_subject_text = str((record or {}).get("subject") or "")

        def _drain_stop_probe() -> bool:
            deadline = _job_fair_deadline(self._semantic_worker, ctx["publish_done_at"])
            return deadline is not None and deadline - time.monotonic() < min_budget * 2

        with self.db.connection() as dispatch_conn:
            dispatch_conn.execute("BEGIN")
            try:
                batch = _JudgeBatch()
                internal_entries: list[tuple[Any, Any, str]] = []
                # ── pass 1a：internal keepers 优先入批（land-first 排序）──
                for unit_a, unit_b, internal_decision in ctx["internal_judge_pairs"]:
                    if not budget.spend("internal"):
                        break
                    reason_text = str(internal_decision.reason or "")
                    if backend is None or _drain_stop_probe():
                        self._land_internal_conflict(ctx, unit_a, unit_b, reason_text)
                        continue
                    ja = row_window(
                        content, int(unit_a.start_offset or 0), int(unit_a.end_offset or 0),
                        subject=own_subject_text,
                        before=SEMANTIC_JUDGE_CONTEXT_BEFORE, after=SEMANTIC_JUDGE_CONTEXT_AFTER)
                    jb = row_window(
                        content, int(unit_b.start_offset or 0), int(unit_b.end_offset or 0),
                        subject=own_subject_text,
                        before=SEMANTIC_JUDGE_CONTEXT_BEFORE, after=SEMANTIC_JUDGE_CONTEXT_AFTER)
                    batch.add(
                        peer_id=int(ctx["memory_id"]), hit=None, unit=None,
                        decision=None, peer=None, left_version=1, right_version=1,
                        text_a=unit_a.text, text_b=unit_b.text, decision_values=None,
                        judge_text_a=ja, judge_text_b=jb,
                    )
                    internal_entries.append((unit_a, unit_b, reason_text))
                # ── pass 1b：跨记忆对收集（六道前置 + 直出 + 池消费）──
                peer_rows = self.conflicts_cross_collect(ctx, dispatch_conn, backend, batch)
                # ── pass 2：一批判定（judge_fn 统一；片间让路）──
                judged_started = time.monotonic()
                judged = batch.drain(self._make_judge_fn(backend), stop_probe=_drain_stop_probe)
                judged_ms = (time.monotonic() - judged_started) * 1000
                per_pair_ms = int(judged_ms / max(1, len(judged)))
                internal_count = len(internal_entries)
                # internal 分账落地：判成→标注；clear→否定意见；error/
                # below_threshold→unannotated fail-open——池内全部落地。
                for item, (unit_a, unit_b, reason_text) in zip(judged[:internal_count], internal_entries):
                    verdict = item["verdict"]
                    outcome = _judge_outcome(verdict, min_prob)
                    if outcome in ("notice", "possible"):
                        if verdict.error is None:
                            reason_text = (
                                f"{reason_text} | mdeberta:{verdict.label}"
                                f" P={verdict.probs.get('conflict', 0.0):.2f}"
                            )
                        if outcome == "notice":
                            ctx["internal_qwen_confirmed"] += 1
                    elif outcome == "clear":
                        reason_text = (
                            f"{reason_text} | mdeberta:no_conflict"
                            f" P={verdict.probs.get('no_conflict', 0.0):.2f}"
                        )
                    self._land_internal_conflict(ctx, unit_a, unit_b, reason_text)
                # cross 分账落地（现役 drain 循环体原样）
                for item in judged[internal_count:]:
                    verdict = item["verdict"]
                    self._tools._record_pair_sample(pair_ms=per_pair_ms)
                    outcome = _judge_outcome(verdict, min_prob)
                    if outcome == "error":
                        error_text = str(verdict.error or "")
                        reason = (
                            "judge_timeout" if "timeout" in error_text.lower()
                            else "judge_unavailable" if "disabled" in error_text.lower()
                            # drain 片间让路停发的 error 归让路（与收集侧同一面
                            # 墙的 judge_budget_exhausted 同口径），非后端故障
                            else "judge_budget_exhausted" if "budget" in error_text.lower()
                            else "judge_backend_error"
                        )
                        self._record_job_degradation(ctx, reason)
                        ctx["incomplete_reason"] = ctx["incomplete_reason"] or reason
                        # 未判成的对必须回 backlog（与 drain 自己的失败留队契约同构，
                        # 实施对抗 review P1）：pass1 已标 settled，回滚让 sweep 收走。
                        ctx["reached_pair"].discard(int(item["peer_id"]))
                        continue
                    if outcome == "clear":
                        ctx["model_clear"] = ctx.get("model_clear", 0) + 1
                        continue
                    if outcome == "below_threshold":
                        ctx["model_conflict_below_threshold"] = (
                            ctx.get("model_conflict_below_threshold", 0) + 1
                        )
                        continue
                    # 等值守卫（owner 2026-09-28，对抗轮收窄）：只比较确定性层抽取的
                    # 值对——decision_values 存在且归一相等 → 同值不同面 clear。
                    # 全行值集相等但含无量纲数字（1000条 vs 2000条）不再误杀。
                    dv = item.get("decision_values")
                    if outcome in ("notice", "possible") and dv is not None and _values_all_equivalent(dv[0], dv[1]):
                        ctx["model_unit_equivalent"] = ctx.get("model_unit_equivalent", 0) + 1
                        continue
                    try:
                        self._land_dispatch_notice(
                            ctx, dispatch_conn, peer_rows, int(item["peer_id"]), item["hit"],
                            item["unit"], item["decision"], content,
                            int(item["left_version"]), int(item["right_version"]),
                            slot_attribute=(
                                # 0.17.1 §3.4: no extraction → a reproducible pair-hash
                                # difference anchor; suppression pairs see owner A-3 note
                                # in _suppressed_by_applying.
                                _pair_diff_anchor(str(item["text_a"]), str(item["text_b"]))
                            ),
                            # §3.4 owner 口径：判定 notice 的两侧值 = 两侧行文本（quote
                            # 即值——无抽取物可放，判断页直接可读）。快照原文形态
                            # 由 D1 双通道放行（db/conflicts.py _normalize_members）。
                            value_a=str(item["text_a"])[:400], value_b=str(item["text_b"])[:400],
                            reason=f"model_classified_{verdict.label}",
                            applying_slots=ctx["applying_slots"],
                            applying_pairs=ctx.get("applying_pairs", set()),
                            surfaced_peer_ids=ctx["surfaced_peer_ids"],
                            severity="normal" if outcome == "notice" else "info",
                            model_signal=_JudgePairView.signal(verdict),
                        )
                    except sqlite3.Error:
                        # R2 对抗轮 P2：db 层写失败（磁盘满/锁超时）不掀翻相位
                        # ——与 error verdict 分支同构（notice_write_failed 归因
                        # + 回 backlog），已落地 notice 不受影响。
                        self._record_job_degradation(ctx, "notice_write_failed")
                        ctx["incomplete_reason"] = ctx["incomplete_reason"] or "notice_write_failed"
                        ctx["reached_pair"].discard(int(item["peer_id"]))
            finally:
                try:
                    dispatch_conn.rollback()
                except sqlite3.Error:
                    pass
        # 0.17.0 P2-4.2: budget/cap leftovers land in the backlog — bounded,
        # visible, never silently dropped (owner design #8). Stale/duplicate
        # keys report as enqueued here; eviction counts ride the store.
        leftover_entries = [
            item for item in ctx["ordered"] if item[0] not in ctx["reached_pair"]
        ]
        ctx["sweep_evicted"] = 0
        if leftover_entries:
            ctx["backlogged"], ctx["sweep_evicted"] = self._enqueue_backlog_entries(
                ctx, leftover_entries,
            )
        ctx["phase_ms"].append((time.monotonic() - phase_started) * 1000)

    def _land_internal_conflict(
        self, ctx: dict[str, Any], unit_a: Any, unit_b: Any, reason_text: str,
    ) -> None:
        """internal keeper 落 internal_conflicts（现役 create 语义：UNIQUE
        去重、fail-open；判定意见已拼进 reason_text 时为标注态）。"""
        if self.db.internal_conflicts.create(
            memory_id=int(ctx["memory_id"]), memory_version=int(ctx["internal_version"]),
            unit_a=unit_a.unit_index, unit_b=unit_b.unit_index,
            quote_a=unit_a.text, quote_b=unit_b.text,
            span_a=[unit_a.start_offset, unit_a.end_offset],
            span_b=[unit_b.start_offset, unit_b.end_offset],
            reason=reason_text, detector_version=CONFLICT_DETECTOR_VERSION,
        ):
            ctx["internal_found"] += 1

    @staticmethod
    def _make_judge_fn(backend: "SemanticBackend | None") -> "Any":
        """合一判定注入（方案 §2.2.2）。异常/形状不符统一降级为 error
        verdict（逐片免疫——R2 对抗轮 P1：internal 旧路径全 try 单对免疫、
        cross 旧路径异常掀翻整相，两形态不可共存；统一取免疫形态，单后端
        故障不再 worker_error，error verdict 由各源分账落地）。
        classify_pair-only 后端统一走 _judge_pair_compat 的 candidate 布尔
        映射——旧 cross 侧 conflict-1.0 直通语义退役（R1 抓出的两相位适配
        语义冲突）。"""
        from ..semantic_judge import PairVerdict as _PV

        def _bad(reason: str, count: int) -> list[Any]:
            return [_PV("no_conflict", {}, None, "mdeberta:unavailable", error=reason)
                    for _ in range(count)]

        def _run_judge(judge_pairs_in: list[tuple[str, str]]) -> list[Any]:
            if backend is None:
                return []
            try:
                if hasattr(backend, "judge_pairs"):
                    verdicts = backend.judge_pairs(judge_pairs_in)
                elif hasattr(backend, "judge_pair"):
                    # single-pair judge interface (ErrBackend fixtures)
                    verdicts = [backend.judge_pair(a, b) for a, b in judge_pairs_in]
                else:
                    # Test/legacy backends exposing classify_pair(env_a, env_b).
                    verdicts = []
                    for a, b in judge_pairs_in:
                        compat = _judge_pair_compat(backend, a, b)
                        verdicts.append(compat if compat is not None else _PV(
                            "no_conflict", {}, None, "mdeberta:unavailable",
                            error="backend cannot serve"))
            except Exception as exc:  # 降级为 error verdict，不掀翻相位（R2 P1）
                return _bad(str(exc), len(judge_pairs_in))
            if len(verdicts) != len(judge_pairs_in):
                return _bad(
                    f"judge returned {len(verdicts)} verdicts for {len(judge_pairs_in)} pairs",
                    len(judge_pairs_in),
                )
            return verdicts

        return _run_judge

    def conflicts_cross_collect(
        self, ctx: dict[str, Any], dispatch_conn: "sqlite3.Connection",
        backend: "SemanticBackend | None", batch: _JudgeBatch,
    ) -> "dict[int, dict[str, Any]]":
        """跨记忆腿收集（原 conflicts_dispatch_loop 的 pass 1；判定与落地
        上移合并相位）。六道前置逐候选保留：active/closed-pair/版本漂移/
        确定性直出（不耗池、backend 缺席照常落地）/deadline 探针/池消费。
        R1-5：探测读走调用方的 dispatch_conn 读快照事务。饱和不终止确定性
        检查（Q1 D2）：池耗尽 continue 不 break，未派发对留给 backlog
        sweep；直出对在 backend 缺席时照样落地。"""
        memory_id = int(ctx["memory_id"])
        record = ctx["record"]
        content = str(ctx["content"])
        embedder = ctx["embedder"]
        budget: _JobJudgeBudget = ctx["budget"]
        min_budget: float = ctx["min_budget"]
        reached_pair: set[int] = ctx["reached_pair"]
        applying_slots: set[str] = ctx["applying_slots"]
        applying_pairs: set[tuple[int, int]] = ctx.get("applying_pairs", set())
        surfaced_peer_ids: set[int] = ctx["surfaced_peer_ids"]
        # P2-T6: batch-prefetch every candidate peer in ONE id-IN query
        # instead of one get_memory connection per pair. Returned for the
        # drain-landing loop (same read snapshot, R1-5).
        peer_rows = self.db.get_memories_by_ids(
            [int(pid) for pid, _triple in ctx["ordered"]], conn=dispatch_conn,
        )
        for peer_id, (hit, unit, decision, _pair_cos) in ctx["ordered"]:
            peer = peer_rows.get(int(peer_id))
            if not peer or peer.get("status") != "active":
                reached_pair.add(peer_id)  # settled (inactive) — not backlog
                continue
            record_row: dict[str, Any] = record or {}
            peer_row: dict[str, Any] = peer or {}
            left_version = int(record.get("version") or 1)
            right_version = int(peer.get("version") or 1)
            if self.db.semantic_notices.is_semantic_pair_closed_on_conn(
                dispatch_conn, memory_id, peer_id, left_version, right_version,
            ):
                reached_pair.add(peer_id)  # settled (closed) — not backlog
                continue
            # 对抗 review 修复（Q1 相分裂 R1-5 的后果）：hit 证据来自确定性相
            # 快照，peer 行来自派发相新快照——两相之间 peer 被编辑时
            # memory_row_version（KNN 行自带）与 fresh version 不再一致，
            # 证据/版本错位的 notice 不可落库。视为 settled（本 job 跳过、
            # 不进 backlog——冻结对身份已过期，下次写会重收集）。
            if int(hit.get("memory_row_version") or 1) != right_version:
                reached_pair.add(peer_id)  # settled (stale hit) — not backlog
                continue
            # NOT reached yet: budget/cap skips below leave the pair
            # unmarked so the post-loop backlog sweep picks it up.
            # Deterministic direct path (2026-09-16, owner-approved): same
            # value-stripped key + canonical value difference IS the
            # same-attribute-different-value shape — land the notice without
            # spending the judge, whose budget is reserved for pairs only
            # judgment can settle. Runs BEFORE the backend/budget checks:
            # a direct pair consumes no judge budget and works even while the
            # backend is unavailable.
            direct = direct_value_verdict(
                unit.text, str(hit.get("text") or ""), decision, embedder=embedder,
            )
            if direct is not None:
                reached_pair.add(peer_id)  # deterministic verdict — settled
                # Q1 §3.3: deterministic 直出 counter — the stage-2
                # comprehensive-recall channel attribution reads it.
                ctx["direct_verdicts"] += 1
                self._land_dispatch_notice(
                    ctx, dispatch_conn, peer_rows, peer_id, hit, unit, decision,
                    content, left_version, right_version,
                    # direct[0] IS the real extracted attribute (kept through
                    # 0.17.1 — the deterministic path keeps its true slot).
                    slot_attribute=str(direct[0]),
                    value_a=str(direct[1]), value_b=str(direct[2]),
                    reason="deterministic_same_key_value_diff",
                    applying_slots=applying_slots, applying_pairs=applying_pairs,
                    surfaced_peer_ids=surfaced_peer_ids,
                )
                continue
            if backend is None:
                self._record_job_degradation(ctx, "judge_unavailable")
                ctx["incomplete_reason"] = ctx["incomplete_reason"] or "judge_unavailable"
                continue
            active_deadline = _job_fair_deadline(self._semantic_worker, ctx["publish_done_at"])
            if active_deadline is not None and active_deadline - time.monotonic() < min_budget * 2:
                self._record_job_degradation(ctx, "judge_budget_exhausted")
                ctx["incomplete_reason"] = ctx["incomplete_reason"] or "judge_budget_exhausted"
                continue
            # Q1 (owner D1/D2)：池耗尽 continue 不 break（饱和不终止确定性
            # 检查）；未派发对留给 backlog sweep。internal 腿先行消费池
            # （land-first），cross 吃余量。
            if not budget.spend("a_cross"):
                self._record_job_degradation(ctx, "pairs_examined_capped")
                ctx["incomplete_reason"] = ctx["incomplete_reason"] or "pairs_examined_capped"
                continue
            reached_pair.add(peer_id)  # queued for the judge — settled
            # 0.17.1 owner ②：判定输入= subject+对立行+前后句（窗口）；裸行
            # 保留给守卫/锚/notice 值（实施对抗 review P1-3 分离设计）。
            own_subject_text = str(record_row.get("subject") or "")
            peer_subject_text = str(peer_row.get("subject") or "")
            batch.add(
                peer_id=int(peer_id), hit=hit, unit=unit, decision=decision,
                peer=peer, left_version=left_version, right_version=right_version,
                text_a=unit.text, text_b=str(hit.get("text") or ""),
                decision_values=(str(decision.left_value or ""), str(decision.right_value or ""))
                if (decision.left_value or decision.right_value) else None,
                judge_text_a=row_window(
                    content, int(unit.start_offset or 0), int(unit.end_offset or 0),
                    subject=own_subject_text,
                    before=SEMANTIC_JUDGE_CONTEXT_BEFORE, after=SEMANTIC_JUDGE_CONTEXT_AFTER),
                judge_text_b=row_window(
                    str(peer_row.get("content") or ""),
                    int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0),
                    subject=peer_subject_text,
                    before=SEMANTIC_JUDGE_CONTEXT_BEFORE, after=SEMANTIC_JUDGE_CONTEXT_AFTER),
            )
        return peer_rows

    @staticmethod
    def _suppressed_by_applying(
        slot_key: dict[str, str], applying_slots: set[str],
        memory_id: int = 0, peer_id: int = 0, applying_pairs: "set[tuple[int, int]] | None" = None,
    ) -> bool:
        """spec §15.3 applying suppression, 0.17.1 form (owner A-3): the
        exact slot match is kept (deterministic path carries the real
        attribute); the judged path has no extractable attribute so the
        exact match never fires there — by design. The judged path's
        re-notify risk is covered by (a) the model recognising superseded
        shapes (owner: 主防线) and (b) dedupe/closed-pair version pins. A
        DIFFERENT slot on the same applying group must always land
        (test_applying_reentry_does_not_suppress_different_slot pins this) —
        so the judged path does NOT suppress on pair identity either.
        applying_pairs is retained in the signature for call-site
        stability and future trusted-context scoping."""
        exact = json.dumps(slot_key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return exact in applying_slots

    def _land_dispatch_notice(
        self, ctx: dict[str, Any], dispatch_conn: "sqlite3.Connection",
        peer_rows: dict[int, dict[str, Any]], peer_id: int, hit: dict[str, Any],
        unit: Any, decision: Any, content: str,
        left_version: int, right_version: int, *,
        slot_attribute: str, value_a: str, value_b: str, reason: str,
        applying_slots: set[str], applying_pairs: "set[tuple[int, int]] | None" = None,
        surfaced_peer_ids: "set[int] | None" = None,
        severity: str = "normal", model_signal: "dict[str, Any] | None" = None,
    ) -> None:
        """Land one A-cross notice (deterministic or judged) through the
        §15.3 applying suppression and the shared payload assembler."""
        record_row: dict[str, Any] = ctx["record"] or {}
        peer = peer_rows.get(int(peer_id)) or {}
        workspace = ctx["workspace"]
        slot_key = _retired_gate_slot_key(
            workspace or record_row.get("workspace_canonical") or record_row.get("workspace"),
            slot_attribute, str(record_row.get("subject") or ""),
        )
        if surfaced_peer_ids is None:
            surfaced_peer_ids = ctx["surfaced_peer_ids"]
        if self._suppressed_by_applying(slot_key, applying_slots, int(ctx["memory_id"]), int(peer_id), applying_pairs):
            # Suppression (widened, owner A-3): a pair landing on a slot
            # currently under application — scan review only (spec §15.3),
            # no new notice.
            return
        extra: dict[str, Any] = {
            "anchors": decision.anchors,
        }
        if model_signal is not None:
            extra["model_signal"] = model_signal
        outcome = self.db.record_semantic_notice(
            memory_id=int(ctx["memory_id"]), peer_id=peer_id,
            severity=severity,
            notice_type="semantic_evidence",
            title=f"Possible memory change with #{peer_id}", message=decision.reason,
            payload=_conflict_notice_payload(
                reason=reason,
                attribute=slot_attribute,
                slot_key=slot_key,
                left_id=int(ctx["memory_id"]), left_version=left_version,
                left_value_norm=value_a,
                left_display=value_a,
                left_quote=unit.text,
                left_member_extra={"start": unit.start_offset, "end": unit.end_offset},
                left_evidence_extra={
                    "start_offset": unit.start_offset, "end_offset": unit.end_offset,
                },
                right_id=int(peer_id), right_version=right_version,
                right_value_norm=value_b,
                right_display=value_b,
                right_quote=hit.get("text"),
                right_member_extra={
                    "start": hit.get("start_offset"), "end": hit.get("end_offset"),
                },
                right_evidence_extra={
                    "start_offset": hit.get("start_offset"),
                    "end_offset": hit.get("end_offset"),
                },
                left_content=str(content or ""),
                right_content=str(peer.get("content") or ""),
                extra=extra,
            ),
            dedupe_key=notice_dedupe_key(
                int(ctx["memory_id"]), peer_id, left_version, right_version, "semantic_evidence",
            ),
            left_version=left_version, right_version=right_version,
            source="semantic_evidence",
        )
        if outcome.get("outcome") == "created":
            ctx["surfaced"] += 1
            surfaced_peer_ids.add(int(peer_id))
            if severity == "info":
                ctx["model_possible_count"] = ctx.get("model_possible_count", 0) + 1
        elif outcome.get("outcome") not in {"deduped"}:
            # Second-round review: a ready pair whose notice could not be
            # persisted (workspace_mismatch / invalid_snapshot / unavailable
            # / error) must not vanish silently — without this the run could
            # report checked_no_notice while a real conflict was found and
            # lost.
            self._record_job_degradation(ctx, "notice_write_failed")
            ctx["incomplete_reason"] = ctx["incomplete_reason"] or "notice_write_failed"

    def conflicts_job_level_notice_cap(self, ctx: dict[str, Any]) -> int:
        """0.17.1 owner A-4: the write-job notice cap is JOB-LEVEL TOP-5 over
        the JUDGED pool only — conflict(normal) and possible(info) notices
        carrying a model_signal compete in one pool ranked by suspicion
        (P(conflict)+P(possible), conflict-class ties win); the rest are
        DEMOTED (severity → info, no_deliver flag) instead of deleted — the
        signal survives in the conflicts table. Deterministic direct notices
        carry no model_signal: they never enter this pool and are never
        demoted, so a write mixing direct + judged notices can exceed 5 in
        the agent feed. Returns how many were demoted. (The pre-0.17.1
        per-channel CLAIMS cap kept its number; its scope was write-total —
        the same 5, now enforced here in one place.)
        Called by the wrapper AFTER all channels landed their notices."""
        from ..constants import CLAIMS_MAX_NOTICES_PER_WRITE

        memory_id = int(ctx["memory_id"])
        rows = self.db.recent_semantic_notices_for_memory(memory_id, limit=64)
        judged = [
            row for row in rows
            if row.get("source") == "semantic_evidence"
            and row.get("status") == "open"  # pending/delivered 解码态；"candidate" 从不存在（对抗 P0：帽曾永空转）
            and isinstance(row.get("payload"), dict)
            and isinstance((row["payload"].get("model_signal") or {}), dict)
            and row["payload"].get("model_signal")
        ]
        if len(judged) <= CLAIMS_MAX_NOTICES_PER_WRITE:
            return 0
        def _suspicion(row: dict[str, Any]) -> float:
            probs = row["payload"]["model_signal"].get("probs") or {}
            tie = 1.0 if (row["payload"]["model_signal"].get("label") == "conflict") else 0.0
            return float(probs.get("conflict", 0.0)) + float(probs.get("possible_conflict", 0.0)) + tie
        ranked = sorted(judged, key=_suspicion, reverse=True)
        demoted = 0
        for row in ranked[CLAIMS_MAX_NOTICES_PER_WRITE:]:
            # 只计真降级：尾部已是 info 的行（possible 判定或前次 job 已降）
            # 不动也不计数——否则回执 model_notices_capped 随未消化积压
            # 逐 job 重复膨胀。
            if str(row.get("notice_severity")) != "info":
                self.db.demote_semantic_notice_to_info(int(row["id"]))
                demoted += 1
        if demoted:
            ctx["model_notices_capped"] = ctx.get("model_notices_capped", 0) + demoted
        return demoted

    def conflicts_finalize_receipt(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Q1 相分裂: merge the three phases' ctx state into the ONE job
        receipt (shape-compatible with the pre-split contract — additive
        keys only, and the zero case emits nothing new)."""
        if ctx["surfaced"]:
            result: dict[str, Any] = {
                "status": "completed", "outcome": "notices_created", "notices_created": ctx["surfaced"],
            }
            if ctx["internal_found"]:
                result["internal_conflicts"] = ctx["internal_found"]
            if ctx["incomplete_reason"]:
                # Notices went out, but later pairs hit a truncation/degradation
                # — surface it instead of a bare completed (second-round
                # review): the caller would otherwise read a bounded, partial
                # check as a full one.
                result["truncated"] = True
        elif ctx["incomplete_reason"]:
            result = {"status": "incomplete", "reason": ctx["incomplete_reason"], "notices_created": 0}
        else:
            result = {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0}
        if ctx["internal_found"] and "internal_conflicts" not in result:
            # Internal findings survive a truncated cross-memory loop: they
            # were landed BEFORE the loop ran (E10① order guarantee).
            result["internal_conflicts"] = ctx["internal_found"]
        # 0.16.2 write-time pre-gate visibility (conditional — the unfiltered
        # zero case keeps the exact-shape response contract unchanged):
        # what the deterministic filters killed this run, and what the
        # unified internal Qwen flow confirmed/vetoed.
        filter_summary: dict[str, int] = {}
        if ctx["no_difference_filtered"]:
            filter_summary["no_difference_skipped"] = ctx["no_difference_filtered"]
        # Gate-v2 G4 observability: the prefilter/coverage/band split (the
        # unfiltered zero case keeps the exact-shape response contract).
        gate_rows: dict[str, int] = {}
        if ctx["memory_pairs_excluded"]:
            gate_rows["memory_pairs_excluded"] = ctx["memory_pairs_excluded"]
        if ctx["prefiltered_rows"]:
            gate_rows["prefiltered_rows"] = ctx["prefiltered_rows"]
        if ctx["below_cos_floor"]:
            gate_rows["below_cos_floor"] = ctx["below_cos_floor"]
        if ctx["repeatability_skipped"]:
            gate_rows["repeatability_skipped"] = ctx["repeatability_skipped"]
        if gate_rows:
            result["candidate_gates"] = gate_rows
        if ctx["internal_qwen_confirmed"]:
            filter_summary["internal_qwen_confirmed"] = ctx["internal_qwen_confirmed"]
        if ctx["dropped_unlocalizable"]:
            # 0.17.0 P2-3.5 (owner ruling #9): dropped unlocalizable pairs are
            # never silent — the counter rides every completed receipt.
            filter_summary["dropped_unlocalizable"] = ctx["dropped_unlocalizable"]
        if filter_summary:
            result["deterministic_filter"] = filter_summary
        if ctx["backlogged"]:
            # 0.17.0 P2-4.2: truncation leftovers went to the conflict
            # backlog instead of vanishing.
            result["backlogged"] = ctx["backlogged"]
        if ctx["sweep_evicted"]:
            # P2-4.3: cap evictions are visible, never silent.
            result["backlog_evicted"] = ctx["sweep_evicted"]
        if ctx["rows_mode"]:
            result["rows_mode"] = True
            result["rows_examined"] = int(ctx["units_examined"])
        if ctx.get("table_rows_exempted"):
            # B3 豁免可见不静默
            result["table_rows_exempted"] = int(ctx["table_rows_exempted"])
        if ctx.get("_internal_knn_ok") is False:
            # B2 fail-open 可见：无 numpy → 内部自查整体跳过（非静默）
            result["internal_skipped_no_numpy"] = True
        if ctx["reasons_seen"]:
            # Degradations may also occur on pairs before a later pair surfaces
            # a notice, so the list is attached to completed outcomes too.
            result["reasons_seen"] = ctx["reasons_seen"]
        if ctx["direct_verdicts"]:
            result["direct_verdicts"] = int(ctx["direct_verdicts"])
        return result

    def conflicts_receipt_tail(self, ctx: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """r2s-08: ONE place stamps the receipt tail — judge_budget /
        pairs_examined / elapsed_ms. The finalize path, the wrapper's
        truncation-terminal branch, and process_conflicts all ride it (the
        tail was previously stamped three ways and had already drifted: the
        terminal branch forgot elapsed, finalize stamped pairs_examined: 0
        unconditionally).

        0.17.1 重标合一: the key is ``judge_budget`` — the one-release
        ``qwen_budget`` compat echo is dropped now that harness runner/score
        read the new key (owner 拍板 2026-10-03，方案 §2.4)."""
        if ctx["phase_ms"]:
            result["elapsed_ms"] = round(sum(ctx["phase_ms"]), 1)
        if ctx["budget"].pairs_examined:
            result["pairs_examined"] = int(ctx["budget"].pairs_examined)
        judge_budget = ctx["budget"].receipt_block()
        if judge_budget is not None:
            # Q1 §3.3 additive observability — absent entirely when nothing
            # was deducted and nothing was skipped.
            result["judge_budget"] = judge_budget
        # 0.17.1 §3.3 outcome counters — observability, zero values absent.
        for key in ("model_clear", "model_conflict_below_threshold",
                    "model_possible_count", "model_notices_capped"):
            if ctx.get(key):
                result[key] = int(ctx[key])
        return result

