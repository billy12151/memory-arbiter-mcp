"""Server-orchestrated conflict-scan pipeline (0.16.0 plan §1/E/E10).

The agent-side scheduled task KICKS the pipeline; the server decides
full-vs-incremental, walks pending memories with per-memory watermarks, and
lands every suspected item in the independent ``scan_queue`` judgment queue.
The agent then clears the queue page by page (page protocol in commit 5).
No resident walker, no Qwen gate in this loop (E11 ①: the agent is the only
semantic judge here), no human in the loop.

Per-memory processing (E10 final form):
1. same-memory internal unit×unit examination (deterministic rule only);
2. cross-memory same-bucket KNN pairing by RELATIVE RANK (absolute distance
   bands are falsified on production data — #971 E10), rule-routed;
3. check-route noise pairs are machine-cleared by the difference classifier
   (counted as machine_cleared, never landed) since 0.16.2; value-difference
   pairs and everything else suspicious land in ``scan_queue`` for agent
   judgment (the numeric auto-reject cap died with that route);
4. a kick refuses to run while the workspace-normalization queue holds
   pending rows (0.16.10: C3b pairs within one bucket, so scanning before
   moves settle would pair on the wrong base — owner 2026-09-19).
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from typing import Any, TYPE_CHECKING

from .constants import (
    SCAN_MACHINE_ROUTE_TOP_K,
    SCAN_POISON_MAX_FAILURES,
    SCAN_SLOW_LANE_PER_KICK,
)
from .db_generation import CONFLICT_DETECTOR_VERSION as CONFLICT_DETECTOR_VERSION  # noqa: F401（显式 re-export：scan_admission 调用期读+测试 patch 缝）
from .difference_classifier import classify_pair
from .pipeline.evidence import filter_exempted_scan_rows
from .semantic_conflict import decide_evidence, is_cross_evolution
from .normalize_gate import compute_summary_votes, normalize_gate
from .scan_admission import (  # noqa: F401
    _candidate_pair_member as _candidate_pair_member,
    spans_overlap as spans_overlap,
    genuine_numeric_pair as genuine_numeric_pair,
    internal_pair_admission as internal_pair_admission,
    _workspace_identity as _workspace_identity,
)
from .scan_admission import _ScanPairsMixin
from .scan_enumerate import _ScanEnumerate

if TYPE_CHECKING:
    from .tools import MemoryTools

from .constants import PROTECTED_WORKSPACES as PROTECTED

DEFAULT_TIME_BUDGET_S = 45.0
DEFAULT_MAX_MEMORIES = 400
DEFAULT_NEIGHBOR_K = 10


class ScanPipeline(_ScanPairsMixin, _ScanEnumerate):
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools
        self.db = tools.db
        self._kick_lock = __import__("threading").Lock()

    # ── public API ──────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        import sqlite3 as _sqlite3

        try:
            pending = self.db.pending_scan_memory_count()
            queue_counts: dict[str, int] = self.db.scan_queue_counts()
            queue_backlog = self.db.scan_queue_backlog()
        except _sqlite3.Error as exc:
            # Additive completion may have been skipped (read-only file): the
            # 0.16.0 surfaces degrade with a structured response, never a raw
            # sqlite error through the tool boundary.
            return {
                "ok": False,
                "error": "scan_structures_unavailable",
                "detail": str(exc),
            }
        state = self.db.meta.scan_pipeline_state() or {}
        return {
            "ok": True,
            "detector_version": CONFLICT_DETECTOR_VERSION,
            "pending_memories": pending,
            "queue": queue_counts,
            "queue_backlog": queue_backlog,
            "pipeline": {
                "round_id": state.get("round_id"),
                "mode": state.get("mode"),
                "complete": bool(state.get("complete")),
                "processed": int(state.get("processed") or 0),
                "auto_rejected": int(state.get("auto_rejected") or 0),
                "machine_cleared": int(state.get("machine_cleared") or 0),
                "started_at": state.get("started_at"),
                "updated_at": state.get("updated_at"),
            },
            "conflict_scan_required": self.db.conflict_scan_state().get("required"),
        }

    def kick(
        self,
        *,
        max_memories: int = DEFAULT_MAX_MEMORIES,
        time_budget_s: float = DEFAULT_TIME_BUDGET_S,
        neighbor_k: int = DEFAULT_NEIGHBOR_K,
        slow_lane: bool = True,
    ) -> dict[str, Any]:
        if not self.db.db_available:
            return {"ok": False, "error": "database_unavailable"}
        if not self.db.state.sqlite_writable:
            return {"ok": False, "error": "database_not_writable"}
        if not self._kick_lock.acquire(blocking=False):
            # Re-entrancy guard (adversarial review P3): a concurrent kick
            # would reprocess the same batch with last-writer-wins counters.
            return {"ok": False, "error": "kick_in_progress"}
        try:
            return self._kick_locked(
                max_memories=max_memories,
                time_budget_s=time_budget_s,
                neighbor_k=neighbor_k,
                slow_lane=slow_lane,
            )
        finally:
            self._kick_lock.release()

    def _kick_locked(
        self,
        *,
        max_memories: int,
        time_budget_s: float,
        neighbor_k: int,
        slow_lane: bool = True,
    ) -> dict[str, Any]:
        vec_state = self.db.get_vec_index_state()
        if vec_state.get("state") in {"mismatch", "failed"}:
            return {"ok": False, "error": "embedding_space_rebuild_required"}
        self._expire_confirmed_pair_pending()
        pending_ws = self._pending_workspace_items()
        if pending_ws:
            # 0.16.10 §九 (owner 2026-09-19): workspace 归一判定清完才扫冲突——
            # C3b 同桶配对，桶归属未治理完时扫描基数是错的。门禁放在 round
            # 状态读写之前：被挡时零副作用（不建 round、不动水位、不写日志；
            # 唯一的例外是上面的 _expire_confirmed_pair_pending 自愈——其副作用
            # 正是清退这些拦路的双确认行，幂等）。
            return {
                "ok": False,
                "error": "workspace_backlog_pending",
                "pending_workspace_items": pending_ws,
                "hint": (
                    "workspace 归一判定未清完：先经 memory_repair(task='scan_queue', "
                    "action='page'/'submit') 处理完 kind='workspace' 的 pending 行再 kick"
                ),
            }
        max_memories = max(1, min(int(max_memories), 2000))
        # A10（0.17.1 修复批）：float("nan") 会让 min/max 链静默产出 nan，
        # 下游 `remaining_budget <= 0.5` 恒 False（nan 比较全假）→ 预算失效。
        # 非有限值按默认 45s 处理（与缺省一致），不静默变 1.0s。
        budget_value = float(time_budget_s)
        if not math.isfinite(budget_value):
            budget_value = 45.0
        time_budget_s = max(1.0, min(budget_value, 300.0))
        neighbor_k = max(1, min(int(neighbor_k), 20))

        state = self.db.meta.scan_pipeline_state()
        now = self._now()
        if state is not None and str(state.get("detector_version") or "") != CONFLICT_DETECTOR_VERSION:
            # Detector identity changed mid-round (epoch re-arm): the round's
            # suppression/identity semantics are stale — restart.
            state = None
        if state is None or state.get("complete"):
            prior_complete = bool(state and state.get("complete"))
            scan_required = bool(self.db.conflict_scan_state().get("required"))
            state = {
                "round_id": uuid.uuid4().hex,
                "detector_version": CONFLICT_DETECTOR_VERSION,
                # First-ever round (and epoch-armed recoveries) are full by
                # construction: every watermark is NULL. Everything after is
                # incremental — the watermark predicate IS the difference.
                "mode": "full" if (not prior_complete or scan_required) else "incremental",
                "last_id": 0,
                "processed": 0,
                "queued": 0,
                "auto_rejected": 0,
                "internal_found": 0,
                "machine_cleared": 0,
                "complete": False,
                "started_at": now,
                "updated_at": now,
            }
        state["updated_at"] = now
        self.db.meta.record_scan_pipeline_state(state)
        self._expire_stale_internal()

        suppression = self._load_suppression()
        fresh_round = int(state.get("processed") or 0) == 0
        started = time.monotonic()
        budget = time_budget_s
        last_id = int(state.get("last_id") or 0)
        round_processed = int(state.get("processed") or 0)
        processed = 0  # per-kick counter: bounds THIS call's work — the round
        # total lives in state["processed"]; comparing the cumulative number
        # against the per-call cap would jam every resumed kick at zero work.
        queued = int(state.get("queued") or 0)
        auto_rejected = int(state.get("auto_rejected") or 0)
        internal_found = int(state.get("internal_found") or 0)
        machine_cleared = int(state.get("machine_cleared") or 0)
        # B3（2026-10-05 审查接线）：扫描腿豁免计数进轮级账本——此前
        # _process_memory 写进 outcome 后全链无消费，kick 回执不可见。
        table_rows_exempted = int(state.get("table_rows_exempted") or 0)
        anchor_buckets: dict[str, int] = {}
        processed_ids: list[int] = []
        # A4：跨 kick 的失败计数（随轮状态落盘；不用 migration_state 的
        # per-id 键——那会让 memory_status 的全量回显无界增长）。
        poison_failures: dict[str, Any] = dict(state.get("poison_failures") or {})

        batch = 50
        # A4（0.17.1 修复批）：两个推进机制修正。
        # ①回卷：pending 是"全量"口径，而取数按 id>last_id —— 编辑过的旧
        #   记忆（id<last_id、watermark 落后）只有重开 round（last_id=0）才
        #   扫得到。此前的机制是"空页 → complete=True → 下一 kick 重开
        #   round"，靠一次谎报完成来自愈；而谎报会经 _complete_round 清掉
        #   conflict_scan_required 门（实测：门被清而毒记忆从未被扫描）。
        #   改为本轮内显式回卷一次：空页且仍有 pending 时 last_id=0 再取，
        #   补扫完再判完成——既自愈又不谎报。
        # ②失败围栏：本轮失败条不得被反复取回（尾位毒记忆此前单 kick 热
        #   重试 253 次吃光墙钟）。失败条排除在本轮取数之外，跨 kick 由
        #   poison_failures 计数（随轮状态落盘），达上界后回执可见。
        wrapped = False
        failed_this_kick: set[int] = set()
        while processed < max_memories:
            remaining_budget = budget - (time.monotonic() - started)
            if remaining_budget <= 0.5:
                break
            ids = [
                memory_id
                for memory_id in self.db.pending_scan_memory_ids(after_id=last_id, limit=batch)
                if memory_id not in failed_this_kick
            ]
            if not ids:
                if (
                    not wrapped
                    and last_id != 0
                    and self.db.pending_scan_memory_count() > 0
                ):
                    wrapped = True
                    last_id = 0
                    continue
                state["complete"] = True
                break
            for memory_id in ids:
                if time.monotonic() - started > budget or processed >= max_memories:
                    break
                # 疑似#11（owner 2026-10-04 拍板）：单条隔离——一条稳定抛
                # 异常的"毒记忆"不得永久卡死扫描循环（无它则每轮 kick 都
                # 死在同一 id，后面的记忆永远扫不到）。失败条不 mark_scanned
                # （水位不动，下轮重试）、不 processed（不抢预算）、日志可见。
                try:
                    outcome = self._process_memory(
                        memory_id,
                        suppression=suppression,
                        neighbor_k=neighbor_k,
                    )
                except Exception:
                    import logging

                    logging.getLogger(__name__).exception(
                        "scan kick: memory %s failed; skipping (watermark not advanced)",
                        memory_id,
                    )
                    failed_this_kick.add(memory_id)
                    poison_failures[str(memory_id)] = (
                        int(poison_failures.get(str(memory_id)) or 0) + 1
                    )
                    continue
                version = outcome["version"]
                if version is not None:
                    self.db.mark_scanned(memory_id, version)
                bucket = outcome.get("workspace") or ""
                if bucket:
                    anchor_buckets[bucket] = anchor_buckets.get(bucket, 0) + 1
                queued += outcome["queued"]
                auto_rejected += outcome["auto_rejected"]
                internal_found += outcome["internal"]
                machine_cleared += int(outcome.get("machine_cleared") or 0)
                table_rows_exempted += int(outcome.get("table_rows_exempted") or 0)
                last_id = max(last_id, memory_id)
                processed_ids.append(memory_id)
                processed += 1
            else:
                continue
            break

        normalized = self._enqueue_workspace_suspects(processed_ids)
        state.update({
            "last_id": last_id,
            "processed": round_processed + processed,
            "queued": queued,
            "auto_rejected": auto_rejected,
            "internal_found": internal_found,
            "machine_cleared": machine_cleared,
            "normalize_suspects": normalized,
            "updated_at": self._now(),
        })
        pending_left = self.db.pending_scan_memory_count()
        # A4（0.17.1 修复批）：complete 必须由"pending 清零"支撑——此前
        # 空页即置 True 且不复位，毒记忆（或任何失败条）被跳过时轮次仍
        # 宣称完成并清掉 conflict_scan_required 门（实测：门被清而该记忆
        # 从未被扫描，完整性宣称断裂）。回卷已尽力补扫，pending_left>0
        # 只可能是稳定失败的毒记忆 → 不宣称完成（门不清=覆盖不完整）。
        state["complete"] = pending_left == 0
        complete = bool(state.get("complete"))
        poison_skipped = sorted(
            int(mid)
            for mid, count in poison_failures.items()
            if int(count) >= SCAN_POISON_MAX_FAILURES
            # A4 口径（2026-10-05 审查修正）：只报「本 kick 仍失败」的条目
            # ——修复后的恢复 kick（水位已推进、覆盖完整）不得继续把
            # 已扫成的记忆报成 poison，与 complete=True 自相矛盾。
            and int(mid) in failed_this_kick
        )
        # 0.17.0 P2-6.2 slow lane: with the fast lane settled (or its batch
        # exhausted for this kick), spend a small leftover budget rotating
        # through the LEAST-recently-scanned watermark-current memories —
        # full coverage by wall time, no detector-version bump required.
        # Best-effort: budget exhaustion and errors just defer to next kick.
        slow_lane_done = 0
        if slow_lane and (complete or processed < 5):
            try:
                remaining = budget - (time.monotonic() - started)
                if remaining > 2.0:
                    slow_ids = self.db.least_recently_scanned_ids(
                        limit=SCAN_SLOW_LANE_PER_KICK, exclude_ids=processed_ids,
                    )
                    for slow_id in slow_ids:
                        if time.monotonic() - started > budget:
                            break
                        slow_outcome = self._process_memory(
                            slow_id, suppression=suppression, neighbor_k=neighbor_k,
                        )
                        if slow_outcome.get("version") is not None:
                            self.db.mark_scanned(slow_id, int(slow_outcome["version"]))
                        queued += slow_outcome["queued"]
                        internal_found += slow_outcome["internal"]
                        # 0.17.0 review R2：慢车道复制快车道记账时漏了
                        # machine_cleared——轮级「机判清除」计数少账，观测口径
                        # 与快车道不一致。
                        machine_cleared += int(slow_outcome.get("machine_cleared") or 0)
                        table_rows_exempted += int(
                            slow_outcome.get("table_rows_exempted") or 0
                        )
                        slow_lane_done += 1
            except Exception:
                pass
        # 0.17.1 P2 #14: the slow lane's increments used to land only in the
        # local counters (visible in this kick's receipt but never in the
        # persisted state table) — merge them before the single record call
        # below so the state table's round counters include slow-lane work.
        state.update({
            "queued": queued,
            "auto_rejected": auto_rejected,
            "internal_found": internal_found,
            "machine_cleared": machine_cleared,
            "table_rows_exempted": table_rows_exempted,
            "poison_failures": poison_failures,
        })
        self.db.meta.record_scan_pipeline_state(state)
        # C5 pacing record + audit line: the same doctor faces the legacy
        # scan path uses (broken-chain alarm, scan_required/scan_stale).
        self.db.record_scan_page_progress(
            after_memory_id=0 if fresh_round else last_id,
            next_anchor_memory_id=None if complete else last_id,
            anchor_buckets=[
                {"workspace": ws, "anchors_scanned": count, "last_anchor": 0}
                for ws, count in sorted(anchor_buckets.items())
            ],
            client=None,
        )
        if complete:
            self._complete_round(state)
        receipt: dict[str, Any] = {
            "ok": True,
            "round_id": state.get("round_id"),
            "mode": state.get("mode"),
            "processed_this_kick": processed,
            "slow_lane_processed": slow_lane_done,
            "processed_round_total": round_processed + processed,
            "queued_total": queued,
            "auto_rejected_total": auto_rejected,
            "machine_cleared_total": machine_cleared,
            "table_rows_exempted_total": table_rows_exempted,
            "internal_found_total": internal_found,
            "normalize_suspects_total": state.get("normalize_suspects") or 0,
            "pending_memories": pending_left,
            "complete": complete,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
        if poison_skipped:
            # A4：达上界的毒记忆——可见而非静默（水位未推进=覆盖不完整，
            # 由用户决定处置：修数据 / 显式忽略）。
            receipt["poison_skipped"] = poison_skipped
        return receipt

    def _pending_workspace_items(self) -> int:
        """0.16.10 §九: conflict scan requires the workspace-normalization
        judgment queue drained first — C3b pairs only within one bucket, so
        scanning before moves settle would pair on the wrong base. Read-only
        count; fails open so a broken queue read never stalls the scan
        safety net (the rest of the pipeline degrades the same way)."""
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM scan_queue "
                    "WHERE kind='workspace' AND status='pending'"
                ).fetchone()
            return int(row["c"]) if row is not None else 0
        except sqlite3.Error:
            return 0

    def _expire_confirmed_pair_pending(self) -> None:
        """Self-heal (0.17.x prompt suppression): retire pending kind='workspace'
        rows whose two buckets are both in the confirmed snapshot — they can
        never be legitimately judged 'move' anymore, and §九 would otherwise
        wedge the kick on rows nobody needs to see. Covers the windows the
        confirm-time sweep (memory_confirm_workspaces) cannot: crash between
        snapshot write and sweep, and pre-upgrade stock rows. Idempotent;
        runs BEFORE the §九 backlog gate. Any failure degrades to the original
        kick behaviour (the gate still sees the rows)."""
        try:
            from .doctor import load_confirmed_workspaces

            confirmed = load_confirmed_workspaces(self.db.settings)
            if not confirmed:
                return
            with self.db.write_transaction() as conn:
                pend = conn.execute(
                    """SELECT id, detail, workspace_canonical FROM scan_queue
                       WHERE kind='workspace' AND status='pending'"""
                ).fetchall()
                stale: list[int] = []
                for row in pend:
                    try:
                        detail = json.loads(str(row["detail"] or "{}"))
                    except json.JSONDecodeError:
                        detail = {}
                    if not isinstance(detail, dict):
                        detail = {}
                    own = str(detail.get("current_workspace") or row["workspace_canonical"] or "")
                    top = str(detail.get("suspected_workspace") or "")
                    if own and top and own in confirmed and top in confirmed:
                        stale.append(int(row["id"]))
                if stale:
                    now = self._now()
                    conn.executemany(
                        """UPDATE scan_queue SET status='expired',
                             decided_reason='confirmed pair suppressed (kick self-heal)',
                             decided_at=?, updated_at=?
                           WHERE id=? AND status='pending'""",  # CAS（B10 纪律）
                        [(now, now, rid) for rid in stale],
                    )
        except Exception:
            pass  # 自愈失败不影响 kick 原有行为

    # ── internals ───────────────────────────────────────────────────────────

    def _process_memory(
        self,
        memory_id: int,
        *,
        suppression: dict[str, Any],
        neighbor_k: int,
    ) -> dict[str, Any]:
        outcome: dict[str, Any] = {
            "version": None, "workspace": None,
            "queued": 0, "auto_rejected": 0, "internal": 0,
            "machine_cleared": 0,
        }
        # 0.17.0 R2 单连接收编：下方只读探针（get_memory / KNN / 向量批取 /
        # claims 向量 SELECT）此前每步各自开/关连接（conn churn），改为穿过
        # 同一条读连接（先例 P2-T6）。写路径（internal create、scan_queue 入队、
        # claims 桥接）保持各自事务——WAL 下空闲读连接不挡写。db_available 在
        # 此显式把关：connection() 直接抛错，而 get_memory 原本降级为 None。
        if not self.db.db_available:
            return outcome
        with self.db.connection() as conn:
            record = self.db.memories.get_memory(memory_id, conn=conn)
            if not record or record.get("status") != "active":
                if record is not None:
                    outcome["version"] = int(record.get("version") or 1)
                return outcome
            version = int(record.get("version") or 1)
            workspace = str(
                record.get("workspace_canonical") or record.get("workspace") or ""
            ).strip()
            outcome["version"] = version
            outcome["workspace"] = workspace
            # C2/C5 (0.17.0 worker merge): the scan source is rows, full stop —
            # the job no longer publishes unit vectors, so the old `if not units`
            # gate and the `is not units` identity probe are gone. No rows yet
            # (mid-backfill) means nothing scannable this round; the slow lane
            # re-picks the memory later.
            internal_source = [
                row for row in self.db.evidence.scan_rows(memory_id, version)
                # C3 A+ guard (adversarial review P2): subject rows are index
                # participants, never scan originators — same discipline as the
                # write-side loops and the diagnostic channel's anchor SQL.
                if str(row.get("kind") or "") != "subject"
            ]
            # A3（0.17.1 修复批）：B3 超长表格段豁免——扫描腿此前直读
            # scan_rows 无过滤，存量巨表仍做 O(n²) 内部检查（实测 120 行表
            # 一次 kick 落地 7021 条 internal_conflicts）。豁免计数必须在
            # 早退（if not internal_source）之前累加，否则丢账。
            internal_source, _exempted_rows = filter_exempted_scan_rows(internal_source)
            if _exempted_rows:
                outcome["table_rows_exempted"] = _exempted_rows
            if not internal_source:
                return outcome
            internal = self._examine_internal(memory_id, version, workspace, internal_source)
            outcome["internal"] = internal
            # Gate-v2 G3: the metadata.entity clear leg is retired with the
            # provenance gate — classify_pair runs on text evidence alone (the
            # entity params stay on the classifier for external callers, but the
            # detection chain no longer reads metadata.entity).
            # 0.17.0 P2-6.1: cross-memory candidates run on ROW vectors — same
            # identity discipline as the write side (eid is the memory_row.id).
            cross_units = internal_source
            def cross_knn(embedding: list[float], **kw: Any) -> list[dict[str, Any]]:
                # Detection window: exclude the peers' subject rows (same
                # discipline as the write side — they crowd out body rows).
                return self.db.row_knn(
                embedding, include_subject_rows=False, conn=conn, **kw
            )
            # 2) cross-memory same-bucket rank pairing.
            # Gate-v2 G4: the scan orchestration SKIPS the sentence prefilter by
            # owner decision (Agent judges prose oppositions) but runs the SAME
            # cosine band gate — one shared implementation (pipeline.gates).
            from .pipeline.gates import candidate_cos_gate, memory_pair_excluded
            below_cos_floor = 0
            repeatability_skipped = 0
            own_subject = str(record.get("subject") or "")
            own_tags = record.get("tags") or []
            memory_pairs_excluded = 0
            screened_peers: set[int] = set()
            excluded_peers: set[int] = set()
            a_enqueued_peers: set[int] = set()
            for unit in cross_units:
                if unit.get("embedding") is None:
                    continue
                hits = cross_knn(
                    unit["embedding"], k=neighbor_k + 1,
                    workspace=workspace or None,
                    exclude_memory_id=memory_id,
                )
                hit_vectors = self.db.evidence.row_vectors_for_ids(
                    [int(hit["id"]) for hit in hits], conn=conn,
                )
                _passed, below_pairs, at_ceil_pairs = candidate_cos_gate(unit["embedding"], hits, hit_vectors)
                hits = [hit for hit, _cos in _passed]
                cos_by_row_id = {int(h["id"]): float(c) for h, c in _passed}
                below_cos_floor += len(below_pairs)
                repeatability_skipped += len(at_ceil_pairs)
                # Rank counts TEXT hits only — non-text units the KNN interleaves
                # must not consume a top-3 slot (0.16.2 §1.5 ranks neighbours,
                # not raw row positions).
                text_rank = 0
                for hit in hits:
                    peer_id = int(hit["memory_id"])
                    if peer_id == memory_id:
                        continue
                    # Gate-v2 G5 scan 同构: the SAME memory-level screen, called
                    # once per peer (first hit wins the verdict; later hits of
                    # the same peer reuse it).
                    if peer_id not in screened_peers:
                        screened_peers.add(peer_id)
                        peer_tags = hit.get("tags")
                        if isinstance(peer_tags, str) and peer_tags:
                            try:
                                peer_tags = json.loads(peer_tags)
                            except (TypeError, ValueError):
                                peer_tags = []
                        if memory_pair_excluded(
                            own_subject, own_tags,
                            str(hit.get("subject") or ""), peer_tags or [],
                        ):
                            excluded_peers.add(peer_id)
                            memory_pairs_excluded += 1
                            continue
                    if peer_id in excluded_peers:
                        continue
                    text_rank += 1
                    peer_bucket = str(
                        hit.get("workspace_canonical") or hit.get("workspace") or ""
                    ).strip()
                    if peer_bucket and workspace and peer_bucket != workspace:
                        continue  # C3b: same-bucket pairing only
                    decision = decide_evidence(str(unit["text"]), str(hit.get("text") or ""))
                    if decision.action == "ignore":
                        continue
                    # 0.16.4 §1: cross-memory evolution domain (todo/polarity
                    # snapshots) is excluded BEFORE any machine route — it never
                    # reaches the rank gate, the classifier, or the queue. The
                    # todo-closure reminder keeps its dedicated channel
                    # (linked_open_items); the same predicate guards the
                    # write-time KNN loop (§0.5 single implementation).
                    if is_cross_evolution(decision):
                        continue
                    # 0.16.2 §1.5: machine-decidable check routes generate only
                    # within the top-3 neighbour ranks (notify kept top-10 until
                    # 0.16.4 excluded it here — only check shapes remain).
                    if text_rank > SCAN_MACHINE_ROUTE_TOP_K:
                        continue
                    # 0.16.2 §1.4: difference-based clearance — check-route
                    # pairs must carry an extractable value difference or they
                    # are duplicates/evolution noise. Cleared pairs are
                    # counted, never enqueued, never landed in conflicts.
                    verdict = classify_pair(
                        str(unit["text"]), str(hit.get("text") or ""),
                        route=str(decision.reason or ""),
                    )
                    if verdict == "clear":
                        outcome["machine_cleared"] += 1
                        continue
                    refs, candidate_key, candidate_hash = self._pair_identity(
                        memory_id, version, unit, peer_id, hit,
                    )
                    if self._suppressed(refs, candidate_hash, suppression):
                        continue
                    # E11③ live retirement (0.16.2 plan §6④/§7): numeric pairs
                    # that survive the difference classifier are same-sentence
                    # two-value candidates — they enqueue for agent judgment;
                    # the noise the old auto-reject consumed is cleared above
                    # without conflicts rows. No new scan_numeric_autoreject
                    # rows are created (existing ones stay as audit history,
                    # excluded from suppression per §1.8).
                    enqueued = self._enqueue_pair(
                        workspace, memory_id, version, unit, peer_id, hit,
                        decision=decision, candidate_key=candidate_key,
                        candidate_hash=candidate_hash,
                        pair_cos=cos_by_row_id.get(int(hit["id"])),
                    )
                    if enqueued:
                        outcome["queued"] += 1
                        a_enqueued_peers.add(peer_id)
            # Gate-v2 G4/G5 observability (conditional, additive receipt keys).
            if memory_pairs_excluded:
                outcome["memory_pairs_excluded"] = memory_pairs_excluded
            if below_cos_floor:
                outcome["below_cos_floor"] = below_cos_floor
            if repeatability_skipped:
                outcome["repeatability_skipped"] = repeatability_skipped
            return outcome

    def _load_suppression(self) -> dict[str, Any]:
        """Round-level suppression maps (same contract as scan_rule_candidates)."""
        recorded: dict[str, str] = {}
        active_groups: list[frozenset[str]] = []
        dismissed_groups: list[frozenset[str]] = []
        if not self.db.db_available:
            return {"hashes": recorded, "active": active_groups, "dismissed": dismissed_groups}
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT status,candidate_key_hash,member_versions FROM conflicts "
                    "WHERE status IN ('open','applying','not_a_conflict') "
                    # 0.16.2 §1.8 (owner ⑩): machine-exercised numeric
                    # auto-reject rows are audit history, NOT a suppression
                    # source — their refs subset-matched 91/121 real notify
                    # pairs into silence.
                    "AND COALESCE(source,'') != 'scan_numeric_autoreject'"
                ).fetchall()
        except Exception:
            rows = []
        for row in rows:
            status = str(row["status"])
            candidate_hash = str(row["candidate_key_hash"] or "")
            if candidate_hash:
                recorded[candidate_hash] = status
            try:
                members = json.loads(str(row["member_versions"] or "[]"))
                refs = frozenset(
                    f"{int(member['memory_id'])}@{int(member['version'])}"
                    for member in members
                )
            except Exception:
                continue
            if refs and status in {"open", "applying"}:
                active_groups.append(refs)
            elif refs and status == "not_a_conflict":
                dismissed_groups.append(refs)
        return {"hashes": recorded, "active": active_groups, "dismissed": dismissed_groups}

    def _suppressed(self, refs: frozenset[str], candidate_hash: str, suppression: dict[str, Any]) -> bool:
        recorded = suppression["hashes"].get(candidate_hash)
        if recorded is None and any(refs <= group for group in suppression["active"]):
            recorded = "open"
        if recorded is None and any(refs <= group for group in suppression["dismissed"]):
            recorded = "not_a_conflict"
        return recorded is not None

    def _enqueue_workspace_suspects(self, processed_ids: list[int]) -> int:
        """Vector-vote workspace suspects for THIS round's processed memories.

        Same summary-vector vote as the C3a anomaly check (top-10 neighbours,
        proportional normalize_gate: top foreign bucket >=4 votes AND >=60% of
        foreign votes → suspected), but scoped to memories the
        pipeline just processed (incremental by watermark, E10) and landing in
        the judgment queue (kind='workspace') instead of notices. The agent
        second-judges (E5: preview/outline input suffices); only a confirmed
        judgment that ALSO passes the server-side gate physically moves.
        """
        from .constants import NORMALIZE_VOTE_MIN_FOREIGN

        vectors = self.db.memories.all_summary_vectors()
        if not vectors:
            return 0
        # P2 #19: the membership set is built once — the per-id `in` check
        # below runs over the whole sorted vector index.
        processed = set(processed_ids)
        ids = [mid for mid in sorted(vectors) if mid in processed]
        if not ids:
            return 0
        # The vote matrix still spans the WHOLE library: a mis-placed memory
        # must be judged against its true neighbours, wherever they live.
        # Early-out equivalent to the old k<MIN_FOREIGN check (k is
        # min(NEIGHBORS, n-1), so k<MIN_FOREIGN iff n-1<MIN_FOREIGN — holds
        # while NORMALIZE_VOTE_NEIGHBORS >= NORMALIZE_VOTE_MIN_FOREIGN, the
        # current 10>=4; lowering NEIGHBORS below MIN_FOREIGN breaks the
        # equivalence and must revisit this gate).
        if len(vectors) - 1 < NORMALIZE_VOTE_MIN_FOREIGN:
            return 0
        # path="single": this caller counted votes by gemv before the
        # extraction; gemv and the block gemm are not bitwise-identical, so
        # the stable-sort tie discipline demands the formula never changes
        # for a given consumer (0.16.10 review finding).
        votes_by_id = compute_summary_votes(vectors, ids, path="single")
        from .doctor import load_confirmed_workspaces

        confirmed = load_confirmed_workspaces(self.db.settings)
        dismissed = self.db.scan_queue.load_workspace_dismissals()
        landed = 0
        for mid in ids:
            vote = votes_by_id.get(mid)
            if vote is None:
                continue
            votes = vote["votes"]
            own = vote["own"]
            k = vote["k"]
            # One shared gate for every consumer (0.16.2 §1.1): generation
            # here, decision-time re-vote, share check, audit payload, and
            # the weekly backstop all judge through normalize_gate.
            passed, evidence = normalize_gate(votes, own)
            if not passed:
                continue
            best_bucket = str(evidence["top_bucket"])
            best_votes = int(evidence["top_votes"])
            record = self.db.get_memory(mid)
            if not record or record.get("status") != "active":
                continue
            version = int(record.get("version") or 1)
            if own in confirmed and best_bucket in confirmed:
                continue  # owner-confirmed pair: no proposal, no queue row
            if (version, best_bucket) in dismissed.get(mid, ()):
                continue  # durable dismissal (version-pinned): no proposal
            detail = {
                "suspected_workspace": best_bucket,
                "current_workspace": own,
                "votes": votes,
                "neighbours_checked": k,
                "protected_involved": bool(
                    own in PROTECTED or best_bucket in PROTECTED
                ),
            }
            identity = _workspace_identity(mid, version, best_bucket)
            outcome = self.db.scan_queue.enqueue(
                kind="workspace",
                workspace_canonical=own,
                candidate_key_hash=identity,
                member_versions=[{"memory_id": mid, "version": version}],
                evidence=[],
                reason=f"vector vote {best_votes}/{k} -> {best_bucket!r}",
                severity="normal",
                source="scan_pipeline",
                detail=detail,
            )
            if outcome.get("outcome") == "queued":
                landed += 1
        return landed

    def _expire_stale_internal(self) -> int:
        """Opportunistic sweep (plan review P2-10): internal rows pinned to a
        version that has been edited away can never be judged — mark them
        stale so counts() stops over-reporting."""
        from .models import utc_now_iso

        try:
            with self.db.write_transaction() as conn:
                cur = conn.execute(
                    """UPDATE internal_conflicts SET status='stale', updated_at=?
                       WHERE status='pending'
                         AND EXISTS(SELECT 1 FROM memories m WHERE m.id=internal_conflicts.memory_id
                                    AND (m.version != internal_conflicts.memory_version
                                         OR m.status != 'active'))""",
                    (utc_now_iso(),),
                )
                return int(cur.rowcount or 0)
        except Exception:
            return 0

    def _complete_round(self, state: dict[str, Any]) -> None:
        """Round completion bookkeeping: audit line + legacy-gate clearing.

        The watermark predicate guarantees every active memory at round start
        was processed, so the legacy ``conflict_scan_required`` gate (armed by
        schema migrations) can be cleared without the page-CAS dance.
        """
        try:
            self.db.log_scan(duration_sec=0.0)
        except Exception:
            pass
        try:
            with self.db.write_transaction() as conn:
                conn.execute(
                    "UPDATE migration_state SET value='false', updated_at=CURRENT_TIMESTAMP "
                    "WHERE key='conflict_scan_required' AND value='true'"
                )
        except Exception:
            pass
        # §6⑩ drift detection anchor: a completed round proves a v2-contract
        # task exists on the host. Doctor compares this stamp against the
        # current SCHEDULED_TASKS_SPEC_VERSION to flag v1-era tasks.
        try:
            from .scan_tasks import SCHEDULED_TASKS_SPEC_VERSION

            with self.db.write_transaction() as conn:
                conn.execute(
                    """INSERT INTO migration_state(key,value,updated_at)
                       VALUES('scheduled_tasks_spec_confirmed',?,CURRENT_TIMESTAMP)
                       ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP""",
                    (str(SCHEDULED_TASKS_SPEC_VERSION),),
                )
        except Exception:
            pass

    @staticmethod
    def _now() -> str:
        from .models import utc_now_iso

        return utc_now_iso()

    QUOTE_LIGHT_CHARS = 60
