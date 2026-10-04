"""evidence 判定相+落地+回执 mixin（从 evidence.py 搬出，拆分批 ⑤ 纯移动）。

conflicts_judge_phase/_land_internal_conflict/_make_judge_fn/conflicts_cross_collect/
_suppressed_by_applying/_land_dispatch_notice/notice_cap/finalize_receipt/receipt_tail。
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, TYPE_CHECKING

from ..constants import (
    SEMANTIC_JUDGE_CONTEXT_BEFORE,
    SEMANTIC_JUDGE_CONTEXT_AFTER,
)
from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..semantic_conflict import (
    SemanticBackend,
    _values_all_equivalent,
    direct_value_verdict,
    notice_dedupe_key,
)
from ..semantic_judge import row_window
from ._evidence_helpers import (
    _conflict_notice_payload,
    _job_fair_deadline,
    _retired_gate_slot_key,
)
from ._evidence_judge import (
    _JudgeBatch,
    _JobJudgeBudget,
    _JudgePairView,
    _judge_outcome,
    _judge_pair_compat,
    _pair_diff_anchor,
)

if TYPE_CHECKING:
    from ..config import Settings
    from ..db import MemoryDB
    from ..tools import MemoryTools
    from ..workers import SemanticConflictWorker
    from ..embedder import ManagedEmbedder


class _EvidenceLand:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"
        settings: "Settings"

        # 主类留守成员的 mypy strict 声明
        @property
        def _semantic_worker(self) -> "SemanticConflictWorker": ...
        def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _ensure_semantic_backend(self) -> "SemanticBackend | None": ...
        def _record_job_degradation(self, *args: Any, **kwargs: Any) -> Any: ...
        def _enqueue_backlog_entries(self, *args: Any, **kwargs: Any) -> Any: ...

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
