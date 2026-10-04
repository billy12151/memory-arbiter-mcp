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

import itertools
import json
import math
import sqlite3
import time
import uuid
from typing import Any, TYPE_CHECKING

from .acl import scope_names, workspace_scope_sql
from .constants import (
    SCAN_MACHINE_ROUTE_TOP_K,
    SCAN_POISON_MAX_FAILURES,
    SCAN_SLOW_LANE_PER_KICK,
    SEMANTIC_MAX_ROWS,
)
from .db_generation import CONFLICT_DETECTOR_VERSION
from .difference_classifier import classify_pair, internal_noise_pair, is_garbage
from .semantic_conflict import decide_evidence, is_cross_evolution
from .normalize_gate import compute_summary_votes, normalize_gate

if TYPE_CHECKING:
    from .acl import WorkspaceScope
    from .tools import MemoryTools

from .constants import PROTECTED_WORKSPACES as PROTECTED

DEFAULT_TIME_BUDGET_S = 45.0
DEFAULT_MAX_MEMORIES = 400
DEFAULT_NEIGHBOR_K = 10


class ScanPipeline:
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
            "machine_cleared": 0, "cleared_garbage": 0,
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
                        if is_garbage(str(unit["text"])) or is_garbage(str(hit.get("text") or "")):
                            outcome["cleared_garbage"] += 1
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

    def _examine_internal(
        self, memory_id: int, version: int, workspace: str,
        units: list[dict[str, Any]],
    ) -> int:
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
                    detector_version=CONFLICT_DETECTOR_VERSION,
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
        def member(mid: int, ver: int, quote: str, span: list[int], unit_eid: int, content_hash: str) -> dict[str, Any]:
            return {
                "memory_id": mid, "version": ver,
                "attribute_raw": None, "value_raw": None,
                "normalized_attribute": None, "normalized_value": None,
                "evidence_quote": quote, "evidence_span": span,
                "content_hash": content_hash, "evidence_unit": unit_eid,
                "direction": "deterministic", "prompt_version": None,
                "detector_version": CONFLICT_DETECTOR_VERSION,
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




    @staticmethod
    def _scan_envelope(memory: dict[str, Any], quote: str) -> dict[str, Any]:
        metadata_value = memory.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, dict) else {}
        return {
            "quote": str(quote)[:1000], "subject": str(memory.get("subject") or "")[:200],
            "tags": list(memory.get("tags") or [])[:20],
            "workspace_canonical": memory.get("workspace_canonical") or memory.get("workspace"),
            "memory_id": int(memory.get("id") or 0), "version": int(memory.get("version") or 1),
            "event_time": memory.get("event_time"),
            "metadata": {key: metadata.get(key) for key in ("entity", "scope") if metadata.get(key)},
        }

    QUOTE_LIGHT_CHARS = 60

    def _lightweight_scan_candidate(self, item: dict[str, Any]) -> dict[str, Any]:
        """C1 lightweight projection of one scan candidate for the default page.

        The full candidate payload (full quotes/spans/members/slot payloads)
        was calibrated for batch=2 reads and explodes the response at the
        spec's batch sizes (12MB pages). The default page keeps only the
        triage identity — pair ids, workspace, reasons, route/state and a
        short quote per side — while include_quotes=true restores the full
        envelope (whose members/slot_key/value_groups record_conflict needs).
        The full payload is computed first and projected last so enhancement
        order and suppression counting are unaffected.
        """
        members = item.get("members")
        if not isinstance(members, list):
            members = []

        def member_quote(index: int) -> str:
            if 0 <= index < len(members):
                quote = str((members[index] or {}).get("evidence_quote") or "")
                if quote:
                    return quote[:self.QUOTE_LIGHT_CHARS]
            return str(item.get("left_snippet") or item.get("right_snippet") or "")[:self.QUOTE_LIGHT_CHARS]

        workspace = item.get("workspace")
        if not workspace and members:
            left_mem = self.db.get_memory(int((members[0] or {}).get("memory_id") or 0))
            if left_mem:
                workspace = (
                    left_mem.get("workspace_canonical")
                    or left_mem.get("workspace")
                )
        light: dict[str, Any] = {
            "left_id": item.get("left_id"),
            "right_id": item.get("right_id"),
            "workspace": workspace,
            "state": item.get("state"),
            "route": item.get("route"),
            "reasons": list(item.get("reasons") or []),
            "distance": item.get("distance"),
            "left_quote": member_quote(0),
            "right_quote": member_quote(1),
        }
        # 0.17.1: judged notices carry model_signal; legacy qwen_signal rows
        # keep rendering through the old shape (one-release compat read).
        model_signal = item.get("model_signal") if isinstance(item.get("model_signal"), dict) else None
        if model_signal:
            light["model_signal"] = {
                key: model_signal.get(key) for key in ("label", "probs", "mechanism", "model_version")
            }
        qwen_signal = item.get("qwen_signal") if isinstance(item.get("qwen_signal"), dict) else None
        if qwen_signal:
            light["qwen_signal"] = {
                key: qwen_signal.get(key) for key in ("state", "reason", "prompt_version")
            }
        return light

    def _lightweight_scan_candidates(self, result: dict[str, Any]) -> dict[str, Any]:
        """Apply the C1 lightweight projection to a finished scan page.

        Candidates carry pair ids/workspace/state/reasons and a short quote
        per side; the full quotes/spans/members/value_groups envelope comes
        back only with include_quotes=true (record_conflict needs it).
        similarity_pool/duplicates_pool pairs get the same treatment via the
        shared per-item projection. slot_groups stay untouched: they are the
        grouping evidence for triage, not per-pair payload bloat.
        """
        for key in ("candidates", "similarity_pool", "duplicates_pool"):
            items = result.get(key)
            if isinstance(items, list):
                result[key] = [
                    (self._lightweight_scan_candidate(item) if isinstance(item, dict) else item)
                    for item in items
                ]
        return result

    def scan_rule_candidates(
        self,
        *,
        after_memory_id: int = 0,
        anchor_batch: int = 50,
        neighbor_k: int = 10,
        include_check: bool = False,
        max_distance: float | None = None,
        workspace: "WorkspaceScope" = None,
        similarity_pool_limit: int = 0,
        include_duplicates: bool = False,
        suspected_anomalies: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Enumerate conflict-candidate pairs for an external scan loop.

        Scheduled LLM review cannot load the whole library into a session,
        so the server enumerates the clues: for every active memory's
        current evidence units, KNN neighbours (rank-based, like the
        write-time notice path but with a wider window) pass through the
        deterministic decide_evidence rule. By default only rule-level
        notify routes (numeric/polarity/todo change) are returned —
        similarity-only check pairs are legion in topic-clustered
        libraries and are opt-in via include_check. Each pair carries the
        triggering unit snippets so the agent can triage without reading
        full memories. Pairs with an open conflict, or a version-pinned
        not_a_conflict dismissal, are filtered out.

        include_duplicates additionally exposes same-value near-duplicate
        pairs (ignore/equivalent_value|compatible_evidence) as a bounded
        duplicates_pool for governance merge; recorded pairs are suppressed
        with the same candidate-hash contract.

        C3b (0.15.13): pairing is workspace-grouped. Each anchor's KNN is
        scoped to the anchor's OWN bucket, so cross-bucket pairs are never
        generated (they cannot satisfy record_conflict's single-bucket
        group identity and used to loop weekly without landing).
        suspected_anomalies ({memory_id: suspected_bucket} from the active
        workspace_review notices) additionally sweeps each suspected
        misplaced memory against its SUSPECTED bucket: those hits are
        cross-bucket by construction and surface in a separate
        cross_bucket_references list for the running agent — authoritative
        disposition is the workspace_review notice (move), never
        record_conflict.

        Calibrated on a real 474-memory production copy: absolute vector
        distance has no discrimination there (random same-workspace pairs
        overlap notice pairs), so ranking + rules do the work and
        max_distance stays an optional extra gate.

        0.17.0 R2: this pairing orchestration moved here from the db layer
        (EvidenceStore) — rule/candidate policy belongs to the pipeline; the
        store keeps only the KNN/vector primitives. SQL and semantics are
        carried over verbatim.
        """
        db = self.db
        if not db.state.sqlite_vec_available:
            return {"error": "sqlite_vec_unavailable"}
        workspace_anchor_sql = ""
        anchor_params: list[Any] = []
        workspace_names = scope_names(workspace)
        echo_workspace = workspace_names[0] if workspace_names else None
        if workspace is not None:
            # Strict callers must not anchor on — or leak snippets from —
            # memories outside their admitted workspace set.
            anchor_scope_sql, anchor_scope_params = workspace_scope_sql(
                "COALESCE(NULLIF(workspace_canonical,''),workspace)", workspace,
            )
            if anchor_scope_sql:
                workspace_anchor_sql = f"AND {anchor_scope_sql} "
                anchor_params.extend(anchor_scope_params)
        with db.connection() as conn:
            anchors = [
                int(row["id"]) for row in conn.execute(
                    "SELECT id FROM memories WHERE status='active' AND id > ? "
                    + workspace_anchor_sql
                    + "ORDER BY id LIMIT ?",
                    (int(after_memory_id), *anchor_params, max(1, int(anchor_batch))),
                )
            ]
            if not anchors:
                return {
                    "anchors_scanned": 0, "next_anchor_memory_id": None,
                    "candidates": [], "counts": {"knn_pairs": 0, "rule_pass": 0,
                                                 "filtered_open": 0, "filtered_dismissed": 0,
                                                 "duplicates": 0},
                    "duplicates_pool": [], "duplicates_truncated": False,
                    "cross_bucket_references": [], "anchor_buckets": [],
                }
            # The group schema has no left/right columns. Suppression is tied
            # to the exact candidate snapshot successfully persisted by
            # record_conflict, not merely to a memory pair. That keeps an
            # unrecorded external review repeatable and allows changed member
            # versions/evidence to be reconsidered.
            recorded_candidate_statuses: dict[str, str] = {}
            active_group_members: list[frozenset[str]] = []
            dismissed_group_members: list[frozenset[str]] = []
            for row in conn.execute(
                "SELECT status,candidate_key_hash,member_versions FROM conflicts "
                "WHERE status IN ('open','applying','not_a_conflict')"
            ):
                candidate_hash = str(row["candidate_key_hash"] or "")
                status = str(row["status"])
                if candidate_hash:
                    recorded_candidate_statuses[candidate_hash] = status
                try:
                    members = json.loads(str(row["member_versions"] or "[]"))
                    refs = frozenset(
                        f"{int(member['memory_id'])}@{int(member['version'])}"
                        for member in members
                    )
                except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                    continue
                # C2 (0.15.13): a dismissed not_a_conflict pair now suppresses
                # by MEMORY-PAIR @version, not only by its exact evidence
                # snapshot. The same pair re-enumerated through different unit
                # slices (new hashes) used to resurface forever. Version
                # pinning stays: an edited memory lifts the suppression and
                # the pair is reconsidered. open/applying stay a separate
                # set so a pair with BOTH an open group and a dismissal
                # keeps the open group's precedence.
                if refs and status in {"open", "applying"}:
                    active_group_members.append(refs)
                elif refs and status == "not_a_conflict":
                    dismissed_group_members.append(refs)
            candidates: dict[tuple[int, int], dict[str, Any]] = {}
            # Spec §7.1 wide gate: similarity-only pairs dropped from the
            # default candidate set stay available as a bounded pool for the
            # caller's Qwen union instead of vanishing outright.
            similarity_pool: dict[tuple[int, int], dict[str, Any]] = {}
            # Near-duplicate (ignore/equivalent_value|compatible_evidence)
            # pairs, exposed for governance merge only when include_duplicates
            # is set. Same suppression contract as real candidates: pairs
            # already recorded (not_a_conflict/open/applying) are not
            # re-enumerated — the candidate_hash lookup runs inside the ignore
            # branch, ahead of the historical silent drop.
            duplicates_pool: dict[tuple[int, int], dict[str, Any]] = {}
            duplicates_truncated = False
            duplicates_cap = 2 * max(1, int(anchor_batch))
            pool_limit = max(0, int(similarity_pool_limit))
            knn_pair_count = 0
            stale_anchors = 0
            filtered_open = 0
            filtered_dismissed = 0
            cross_bucket_refs: dict[tuple[int, int], dict[str, Any]] = {}
            anchor_buckets: dict[str, dict[str, int]] = {}
            # C3b: suspected misplaced memories (from ACTIVE workspace_review
            # notices) are ALSO paired against their suspected bucket. Those
            # pairs are cross-bucket, cannot land in record_conflict, and go
            # to a reference list only.
            suspected = {
                int(mid): str(bucket)
                for mid, bucket in (suspected_anomalies or {}).items()
            }
            for anchor_id in anchors:
                anchor_row = conn.execute(
                    "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=?",
                    (anchor_id,),
                ).fetchone()
                # C3b grouping: pair only within the anchor's own bucket. The
                # strict caller scope (workspace) already bounds the whole
                # page; the anchor bucket narrows pairing further.
                anchor_bucket = (
                    str(anchor_row["workspace"] or "").strip() if anchor_row else ""
                )
                # C5 per-group page accounting: doctor's broken-chain alarm
                # needs to know which bucket the walk was last inside.
                if anchor_bucket:
                    entry = anchor_buckets.setdefault(
                        anchor_bucket, {"count": 0, "last_anchor": 0},
                    )
                    entry["count"] += 1
                    entry["last_anchor"] = max(entry["last_anchor"], anchor_id)
                # C2/C5: rows are the scan source — the job no longer
                # publishes unit vectors. rowseg emits no heading rows and
                # (pre-C3) no subject rows, so the old kind='text' filter's
                # intent (subjects/headings excluded) holds by construction.
                # C3 A+ guard: subject rows never ORIGINATE a scan pair
                # (subject version progression is timeline evolution, not a
                # numeric clue) — the row-channel counterpart of the unit
                # channel's kind='text' anchor filter.
                units = conn.execute(
                    """SELECT r.id AS eid, r.text AS text, v.embedding AS embedding,
                              r.memory_version AS memory_version, r.content_hash AS content_hash,
                              r.start_offset AS start_offset, r.end_offset AS end_offset
                       FROM memory_row r
                       JOIN memory_row_vec v ON v.id=r.id
                       WHERE r.memory_id=? AND r.memory_version=(
                           SELECT version FROM memories WHERE id=?)
                         AND r.kind != 'subject'
                       ORDER BY r.id""",
                    (anchor_id, anchor_id),
                )
                first_unit = units.fetchone()
                if first_unit is None:
                    # Async republish window or a permanently failed publish:
                    # surface it instead of silently skipping forever.
                    stale_anchors += 1
                anchor_content_row = conn.execute(
                    "SELECT content FROM memories WHERE id=?", (anchor_id,),
                ).fetchone()
                anchor_content = str(anchor_content_row["content"]) if anchor_content_row else ""
                peer_content_cache: dict[int, str] = {}

                def peer_content(peer_mid: int) -> str:
                    if peer_mid not in peer_content_cache:
                        row = conn.execute(
                            "SELECT content FROM memories WHERE id=?", (peer_mid,),
                        ).fetchone()
                        peer_content_cache[peer_mid] = str(row["content"]) if row else ""
                    return peer_content_cache[peer_mid]

                def locate_span(content: str, unit_text: str, hint_start: int, hint_end: int) -> dict[str, int] | None:
                    """Validate an exact evidence span and pad it for review.

                    Evidence pipeline v2 guarantees that cleaning the source
                    slice equals the unit text. Do not search for the text:
                    repeated phrases make search ambiguous and can silently
                    choose the wrong occurrence. A failed invariant drops the
                    span and falls back to a full read.
                    """
                    from .evidence import _clean

                    start, end = int(hint_start), int(hint_end)
                    if not (content and unit_text and 0 <= start < end <= len(content)):
                        return None
                    if _clean(content[start:end]) != unit_text:
                        return None
                    return {
                        "start": max(0, start - 128),
                        "end": min(len(content), end + 128),
                    }

                def _pool_near_duplicate(
                    peer: int, hit: "dict[str, Any]", anchor: int,
                    unit_row: "dict[str, Any]", text_a: str, reason: str,
                ) -> None:
                    """Gate-v2 G4: shared duplicates_pool admission for BOTH
                    near-duplicate sources — the deterministic ignore routes
                    and the at-ceil cosine pairs. Same suppression contract,
                    same free-dict-replace cap accounting."""
                    nonlocal duplicates_truncated
                    pair_key = (min(anchor, peer), max(anchor, peer))
                    member_refs, _key, candidate_hash = self.db.evidence._unit_pair_identity(
                        anchor, unit_row, peer, hit,
                    )
                    recorded = recorded_candidate_statuses.get(candidate_hash)
                    if recorded is not None or any(
                        member_refs <= group_members for group_members in active_group_members
                    ) or any(
                        member_refs <= group_members for group_members in dismissed_group_members
                    ):
                        return
                    hit_text = str(hit.get("text") or "")
                    # Re-hitting an already-pooled pair is a free dict
                    # replace, not pool growth — it must not count against
                    # the cap or flag truncation that never happened.
                    if pair_key in duplicates_pool or len(duplicates_pool) < duplicates_cap:
                        duplicates_pool[pair_key] = {
                            "left_id": pair_key[0], "right_id": pair_key[1],
                            "reason": reason,
                            "distance": float(hit.get("distance") or 0),
                            "candidate_key_hash": candidate_hash,
                            "left_snippet": text_a[:200] if pair_key[0] == anchor else hit_text[:200],
                            "right_snippet": hit_text[:200] if pair_key[1] == peer else text_a[:200],
                            "members": [
                                _candidate_pair_member(
                                    pair_key[0], is_anchor=(pair_key[0] == anchor),
                                    unit=unit_row, hit=hit,
                                    anchor_text=text_a, peer_text=hit_text,
                                ),
                                _candidate_pair_member(
                                    pair_key[1], is_anchor=(pair_key[1] == anchor),
                                    unit=unit_row, hit=hit,
                                    anchor_text=text_a, peer_text=hit_text,
                                ),
                            ],
                        }
                    else:
                        duplicates_truncated = True


                # C3b: same-bucket pairing. Under a strict caller scope the
                # anchor bucket must stay inside the admitted set.
                admitted_names = set(workspace_names) if workspace_names else set()
                pairing_scope = anchor_bucket if (
                    anchor_bucket and (workspace is None or anchor_bucket in admitted_names)
                ) else workspace
                suspect_bucket = suspected.get(anchor_id)
                from .pipeline.gates import candidate_cos_gate
                for unit in (() if first_unit is None else itertools.chain((first_unit,), units)):
                    text = str(unit["text"] or "")
                    if not text:
                        continue
                    unit_vector = self.db.evidence._blob_to_vector(bytes(unit["embedding"]))
                    hits = self.db.row_knn(
                        unit_vector,
                        k=max(1, int(neighbor_k)) + 1,
                        workspace=pairing_scope,
                        exclude_memory_id=anchor_id,
                        include_subject_rows=False,
                    )
                    # Gate-v2 G4 余弦门 (diagnostic-channel leg): below-floor
                    # pairs are noise; AT/ABOVE-ceil pairs are near-duplicates
                    # — they route into duplicates_pool below instead of being
                    # dropped (that pool IS their governance consumer). The
                    # suspect-bucket sweep stays OUTSIDE the gate: those hits
                    # are cross-bucket references, not conflict candidates.
                    gate_vectors = self.db.evidence.row_vectors_for_ids(
                        [int(hit["id"]) for hit in hits], conn=conn,
                    )
                    _passed, _below, at_ceil_pairs = candidate_cos_gate(unit_vector, hits, gate_vectors)
                    hits = [hit for hit, _cos in _passed]
                    if include_duplicates:
                        for dup_hit, _dup_cos in at_ceil_pairs:
                            _pool_near_duplicate(
                                int(dup_hit["memory_id"]), dup_hit, anchor_id, unit,
                                text, "near_duplicate_cosine",
                            )
                    # C3b: the suspected-bucket sweep for misplaced memories.
                    suspect_hits: list[dict[str, Any]] = []
                    if suspect_bucket and suspect_bucket != anchor_bucket:
                        suspect_hits = self.db.row_knn(
                            unit_vector,
                            k=max(1, int(neighbor_k)) + 1, include_subject_rows=False,
                            workspace=suspect_bucket,
                            exclude_memory_id=anchor_id,
                        )
                    for hit in itertools.chain(hits, suspect_hits):
                        # C2/C5: rows carry no 'text' kind — the old unit-
                        # channel filter's intent (exclude subject/heading
                        # units) holds by construction (rowseg emits neither).
                        peer_id = int(hit["memory_id"])
                        if peer_id == anchor_id:
                            continue
                        peer_bucket = str(hit.get("workspace_canonical") or hit.get("workspace") or "").strip()
                        if peer_bucket and anchor_bucket and peer_bucket != anchor_bucket:
                            # C3b: cross-bucket hits exist only in the
                            # suspected-bucket sweep (regular pairing is
                            # bucket-scoped). Reference-only: they can never
                            # satisfy record_conflict's single-bucket group
                            # identity. Authority is the workspace_review
                            # notice (move), never a conflict group.
                            pair_key = (min(anchor_id, peer_id), max(anchor_id, peer_id))
                            decision = decide_evidence(text, str(hit.get("text") or ""))
                            ref = cross_bucket_refs.setdefault(pair_key, {
                                "left_id": pair_key[0], "right_id": pair_key[1],
                                "workspace": anchor_bucket,
                                "suspected_workspace": suspect_bucket,
                                "reasons": set(), "distance": float(hit.get("distance") or 0),
                                "left_snippet": text[:200], "right_snippet": str(hit.get("text") or "")[:200],
                                "note": "cross-bucket reference only; disposition via the workspace_review notice (move), not record_conflict",
                            })
                            ref["reasons"].add(decision.reason)
                            ref["distance"] = min(ref["distance"], float(hit.get("distance") or 0))
                            continue
                        knn_pair_count += 1
                        # Every unit pair is judged: an earlier equivalent
                        # match (e.g. identical subjects) must not blacklist
                        # the peer, or a later numeric-change unit on the
                        # same pair would be lost.
                        decision = decide_evidence(text, str(hit.get("text") or ""))
                        # 0.16.4 §1/§0.5: the diagnostic channel routes
                        # through the SAME shared predicate as the scan
                        # pipeline and the write-time KNN loop — evolution-
                        # domain pairs must not surface here either, or the
                        # retroactive void's "re-enqueue and surface the gap"
                        # design would leak them back as notice_ready.
                        if is_cross_evolution(decision):
                            continue
                        if decision.action == "ignore":
                            if include_duplicates and decision.reason in {"equivalent_value", "compatible_evidence"}:
                                _pool_near_duplicate(
                                    peer_id, hit, anchor_id, unit, text, decision.reason,
                                )
                            continue
                        # Numeric deltas remain a deterministic scan baseline
                        # candidate even though they can no longer directly
                        # produce a write-time notice.
                        similarity_only = (
                            decision.action == "check"
                            and decision.reason != "numeric_value_candidate"
                            and not include_check
                        )
                        if similarity_only and pool_limit <= 0:
                            continue
                        if max_distance is not None and float(hit.get("distance") or 0) > float(max_distance):
                            continue
                        pair = (min(anchor_id, peer_id), max(anchor_id, peer_id))
                        distance = float(hit.get("distance") or 0)
                        store = similarity_pool if similarity_only else candidates
                        existing = store.get(pair)
                        hit_text = str(hit.get("text") or "")
                        anchor_span = locate_span(
                            anchor_content, text,
                            int(unit["start_offset"] or 0), int(unit["end_offset"] or 0),
                        )
                        peer_span = locate_span(
                            peer_content(peer_id), hit_text,
                            int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0),
                        )
                        member_refs, candidate_key, candidate_hash = self.db.evidence._unit_pair_identity(
                            anchor_id, unit, peer_id, hit,
                        )
                        recorded_status = recorded_candidate_statuses.get(candidate_hash)
                        if recorded_status is None and any(
                            member_refs <= group_members for group_members in active_group_members
                        ):
                            recorded_status = "open"
                        if recorded_status is None and any(
                            member_refs <= group_members for group_members in dismissed_group_members
                        ):
                            # C2: pair@version dismissal. Counted as dismissed,
                            # never silently folded into filtered_open — the
                            # counter is the convergence observability face.
                            recorded_status = "not_a_conflict"
                        if recorded_status is not None:
                            if recorded_status == "not_a_conflict":
                                filtered_dismissed += 1
                            else:
                                filtered_open += 1
                            continue
                        if existing is None:
                            state = "notice_ready" if decision.action == "notify" else "review_candidate"
                            store[pair] = {
                                "left_id": pair[0], "right_id": pair[1],
                                "state": state, "route": state,
                                "reasons": {decision.reason}, "distance": distance,
                                "candidate_key": candidate_key,
                                "candidate_key_hash": candidate_hash,
                                "members": [
                                    _candidate_pair_member(
                                        pair[0], is_anchor=(pair[0] == anchor_id),
                                        unit=unit, hit=hit,
                                        anchor_text=text, peer_text=hit_text,
                                    ),
                                    _candidate_pair_member(
                                        pair[1], is_anchor=(pair[1] == anchor_id),
                                        unit=unit, hit=hit,
                                        anchor_text=text, peer_text=hit_text,
                                    ),
                                ],
                                "value_groups": [], "slot_key": None, "slot_provenance": None,
                                "left_snippet": text[:200] if pair[0] == anchor_id else hit_text[:200],
                                "right_snippet": hit_text[:200] if pair[0] == anchor_id else text[:200],
                                # Pre-built deep-read calls: reading just the
                                # triggering region (plus context) instead of
                                # the full text keeps triage token cost low.
                                "deep_read": {
                                    "left": {
                                        "memory_id": pair[0],
                                        "span": anchor_span if pair[0] == anchor_id else peer_span,
                                        **({"workspace": echo_workspace} if echo_workspace else {}),
                                    },
                                    "right": {
                                        "memory_id": pair[1],
                                        "span": peer_span if pair[0] == anchor_id else anchor_span,
                                        **({"workspace": echo_workspace} if echo_workspace else {}),
                                    },
                                },
                            }
                        else:
                            # notice_ready outranks review_candidate when
                            # different unit pairs on the same memory pair
                            # disagree. Snippets and spans track the strongest
                            # signal, not the first discovery.
                            numeric_upgrade = (
                                decision.reason == "numeric_value_candidate"
                                and "numeric_value_candidate" not in existing["reasons"]
                            )
                            if not similarity_only and (
                                (existing["state"] == "review_candidate" and decision.action == "notify")
                                or numeric_upgrade
                            ):
                                # 0.17.1 owner ③（窗口 32）：同 memory pair 的
                                # numeric discovery 到来时必须接管 deep_read/
                                # snippets——先到的 similarity-only discovery
                                #（如 filler 区同文对）不能永久占住 span，否则
                                # 判定页把 Agent 带到无冲突证据的文本区。
                                if numeric_upgrade and existing["state"] == "review_candidate":
                                    existing["left_snippet"] = text[:200] if pair[0] == anchor_id else hit_text[:200]
                                    existing["right_snippet"] = hit_text[:200] if pair[0] == anchor_id else text[:200]
                                    existing["deep_read"] = {
                                        "left": {
                                            "memory_id": pair[0],
                                            "span": anchor_span if pair[0] == anchor_id else peer_span,
                                            **({"workspace": echo_workspace} if echo_workspace else {}),
                                        },
                                        "right": {
                                            "memory_id": pair[1],
                                            "span": peer_span if pair[0] == anchor_id else anchor_span,
                                            **({"workspace": echo_workspace} if echo_workspace else {}),
                                        },
                                    }
                                if existing["state"] == "review_candidate" and decision.action == "notify":
                                    existing["state"] = "notice_ready"
                                    existing["route"] = "notice_ready"
                                    existing["left_snippet"] = text[:200] if pair[0] == anchor_id else hit_text[:200]
                                    existing["right_snippet"] = hit_text[:200] if pair[0] == anchor_id else text[:200]
                                    existing["deep_read"] = {
                                        "left": {
                                            "memory_id": pair[0],
                                            "span": anchor_span if pair[0] == anchor_id else peer_span,
                                            **({"workspace": echo_workspace} if echo_workspace else {}),
                                        },
                                        "right": {
                                            "memory_id": pair[1],
                                            "span": peer_span if pair[0] == anchor_id else anchor_span,
                                            **({"workspace": echo_workspace} if echo_workspace else {}),
                                        },
                                    }
                            existing["reasons"].add(decision.reason)
                            existing["distance"] = min(existing["distance"], distance)
            ordered = [candidates[pair] for pair in sorted(candidates)]
            for item in ordered:
                item["reasons"] = sorted(item["reasons"])
            similarity_ordered = sorted(
                similarity_pool.values(), key=lambda item: float(item.get("distance") or 9),
            )[:pool_limit]
            for item in similarity_ordered:
                item["reasons"] = sorted(item["reasons"])
            next_anchor = anchors[-1]
            with db.connection() as conn:
                more = conn.execute(
                    "SELECT 1 FROM memories WHERE status='active' AND id > ? "
                    + workspace_anchor_sql
                    + "LIMIT 1",
                    (next_anchor, *anchor_params),
                ).fetchone()
            duplicates_ordered = [
                duplicates_pool[pair] for pair in sorted(duplicates_pool)
            ]
            cross_refs_ordered = [
                {**cross_bucket_refs[pair], "reasons": sorted(cross_bucket_refs[pair]["reasons"])}
                for pair in sorted(cross_bucket_refs)
            ]
            return {
                "anchors_scanned": len(anchors),
                "next_anchor_memory_id": int(next_anchor) if more else None,
                "candidates": ordered,
                "similarity_pool": similarity_ordered,
                "duplicates_pool": duplicates_ordered,
                "duplicates_truncated": duplicates_truncated,
                "cross_bucket_references": cross_refs_ordered,
                "anchor_buckets": [
                    {"workspace": ws, "anchors_scanned": entry["count"],
                     "last_anchor": entry["last_anchor"]}
                    for ws, entry in sorted(anchor_buckets.items())
                ],
                "counts": {
                    "knn_pairs": knn_pair_count,
                    "rule_pass": len(ordered),
                    "similarity_pool": len(similarity_ordered),
                    "duplicates": len(duplicates_ordered),
                    "filtered_open": filtered_open,
                    "filtered_dismissed": filtered_dismissed,
                    "stale_anchors": stale_anchors,
                },
            }

    def memory_scan_workspace_anomalies(self, **_: Any) -> dict[str, Any]:
        """C3a workspace anomaly check: single-pass matmul over all summary vectors.

        One SELECT reads every active memory's summary vector; numpy computes
        the N×N cosine in row-blocks (bounded memory); each row votes over its
        top-10 neighbours through the shared proportional normalize_gate. A
        memory whose neighbourhood passes the gate is a suspected misplacement:
        one kind='workspace' scan_queue row per memory (same queue, same gate
        as the pipeline's incremental suspects), capped at 10 new rows per run.
        Zero Qwen, milliseconds. numpy absence degrades with a structured
        outcome (it is not a declared dependency — llama-cpp-python normally
        brings it).
        """
        try:
            import numpy  # noqa: F401  # presence probe only: the structured error below must distinguish numpy-missing from an empty library
        except ImportError:
            return self.db.state.response({
                "error": "numpy_unavailable", "detail": (
                    "workspace anomaly check needs numpy (bundled with the "
                    "semantic-local extra); install numpy to run it"
                ),
            }, ok=False)
        # Fresh-boot coverage: the first run after an upgrade (before any
        # write) has no summary vectors yet — the write-path publish and the
        # startup backfill both ride the first embedder load. Ensure that load
        # happens here so the weekly task never no-ops its first round.
        if self.db.missing_summary_vec_rows():
            embedder, _warnings = self._tools._ensure_embedder()
            if embedder is not None:
                try:
                    self._tools._backfill_memory_summary_vectors(embedder)
                except Exception:
                    pass
        vectors = self.db.all_summary_vectors()
        if not vectors:
            return self.db.state.response({
                "status": "ok", "checked": 0, "suspected": 0, "queued": 0,
                "note": "no summary vectors yet (backfill pending or empty library)",
            })
        ids = sorted(vectors)
        n = len(ids)
        # The STABLE-sorting discipline lives in compute_summary_votes: equal
        # similarities (FakeEmbedder's binary vectors, duplicated content)
        # must pick the same neighbours on every machine/numpy version —
        # np.argpartition left ties arbitrary and CI once selected one beta
        # neighbour where the local run selected nine.
        votes_by_id = compute_summary_votes(vectors, ids)
        from .doctor import load_confirmed_workspaces

        confirmed = load_confirmed_workspaces(self.db.settings)
        dismissed = self.db.scan_queue.load_workspace_dismissals()
        suspected: list[dict[str, Any]] = []
        for row_mid in ids:
            vote = votes_by_id.get(row_mid)
            if vote is None:
                continue
            votes = vote["votes"]
            own = vote["own"]
            own_best = vote["own_best"]
            foreign_best = vote["foreign_best"]
            # Shared proportional gate (0.16.2 §1.1): the weekly
            # backstop judges by the SAME function as the pipeline's
            # suspect generation and the decision-time re-vote.
            passed, gate_evidence = normalize_gate(votes, own)
            if passed:
                top = str(gate_evidence["top_bucket"])
                if own in confirmed and top in confirmed:
                    continue  # owner-confirmed pair: never suspected
                entries = dismissed.get(row_mid)
                if entries:
                    record = self.db.get_memory(row_mid)
                    if record is not None:
                        top_ws = str(gate_evidence["top_bucket"])
                        if (int(record.get("version") or 1), top_ws) in entries:
                            continue  # durable dismissal: never suspected
                suspected.append({
                    "memory_id": row_mid,
                    "workspace": own,
                    "suspected_workspace": gate_evidence["top_bucket"],
                    "foreign_votes": gate_evidence["top_votes"],
                    "neighbours_checked": vote["k"],
                    "foreign_neighbour_id": foreign_best[1],
                    "own_neighbour_id": own_best[1],
                    "votes": dict(votes),
                })
        suspected.sort(key=lambda item: (-item["foreign_votes"], item["memory_id"]))
        # Lazy staleness (the notice channel's heir), BEFORE selecting the
        # cap: a pending suspect row whose subject already left its pinned
        # bucket is resolved — retire it even when the memory no longer shows
        # up in this sweep's findings (that is precisely why it is stale).
        self._tools._expire_relocated_workspace_rows()
        capped = suspected[:10]
        queued = 0
        # 0.16.2 §1.3: findings land in the judgment queue (same queue, same
        # gate as the pipeline's incremental suspects) — the workspace_review
        # notice channel no longer produces new findings. Identity reuses
        # _workspace_identity so a pipeline suspect and the weekly suspect
        # for the same memory@version+suspicion share one row (INSERT OR
        # IGNORE dedupes re-runs; dismissal keeps it dismissed).
        from .constants import PROTECTED_WORKSPACES
        from .scan_pipeline import _workspace_identity

        for item in capped:
            memory_id = int(item["memory_id"])
            record = self.db.get_memory(memory_id)
            if record is None or str(record.get("status") or "") != "active":
                continue
            version = int(record.get("version") or 1)
            own = str(item["workspace"])
            best_bucket = str(item["suspected_workspace"])
            detail = {
                "suspected_workspace": best_bucket,
                "current_workspace": own,
                "votes": item.get("votes") or {},
                "neighbours_checked": item["neighbours_checked"],
                "protected_involved": bool(
                    own in PROTECTED_WORKSPACES or best_bucket in PROTECTED_WORKSPACES
                ),
                "channel": "weekly_backstop",
            }
            outcome = self.db.scan_queue.enqueue(
                kind="workspace",
                workspace_canonical=own,
                candidate_key_hash=_workspace_identity(memory_id, version, best_bucket),
                member_versions=[{"memory_id": memory_id, "version": version}],
                evidence=[],
                reason=(
                    f"weekly vote {item['foreign_votes']}/{item['neighbours_checked']}"
                    f" -> {best_bucket!r}"
                ),
                severity="normal",
                source="workspace_anomaly_scan",
                detail=detail,
            )
            if str(outcome.get("outcome") or "") == "queued":
                queued += 1
        return self.db.state.response({
            "status": "ok",
            "checked": n,
            "suspected": len(suspected),
            "returned": len(capped),
            "queued": queued,
            "cap": 10,
            **({"capped": True} if len(suspected) > len(capped) else {}),
            "findings": [
                {
                    "memory_id": item["memory_id"],
                    "workspace": item["workspace"],
                    "suspected_workspace": item["suspected_workspace"],
                    "votes": f"{item['foreign_votes']}/{item['neighbours_checked']}",
                }
                for item in capped
            ],
        })

def _candidate_pair_member(
    memory_id: int, *,
    is_anchor: bool,
    unit: "dict[str, Any]", hit: "dict[str, Any]",
    anchor_text: str, peer_text: str,
) -> dict[str, Any]:
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
        "detector_version": CONFLICT_DETECTOR_VERSION,
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
