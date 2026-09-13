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
3. ``numeric_value_candidate`` pairs are auto-rejected with an audit trail
   and a per-round cap (E11 ③: machine-exercised not_a_conflict);
4. everything else suspicious lands in ``scan_queue`` as a pair row.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, TYPE_CHECKING

from .constants import SCAN_MACHINE_ROUTE_TOP_K
from .db_generation import CONFLICT_DETECTOR_VERSION
from .difference_classifier import classify_pair, is_garbage
from .semantic_conflict import decide_evidence

if TYPE_CHECKING:
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
            )
        finally:
            self._kick_lock.release()

    def _kick_locked(
        self,
        *,
        max_memories: int,
        time_budget_s: float,
        neighbor_k: int,
    ) -> dict[str, Any]:
        vec_state = self.db.get_vec_index_state()
        if vec_state.get("state") in {"mismatch", "failed"}:
            return {"ok": False, "error": "embedding_space_rebuild_required"}
        max_memories = max(1, min(int(max_memories), 2000))
        time_budget_s = max(1.0, min(float(time_budget_s), 300.0))
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

        batch = 50
        while processed < max_memories:
            remaining_budget = budget - (time.monotonic() - started)
            if remaining_budget <= 0.5:
                break
            ids = self.db.pending_scan_memory_ids(after_id=last_id, limit=batch)
            if not ids:
                state["complete"] = True
                break
            for memory_id in ids:
                if time.monotonic() - started > budget or processed >= max_memories:
                    break
                outcome = self._process_memory(
                    memory_id,
                    suppression=suppression,
                    neighbor_k=neighbor_k,
                )
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
        if pending_left == 0:
            state["complete"] = True
        complete = bool(state.get("complete"))
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
        return {
            "ok": True,
            "round_id": state.get("round_id"),
            "mode": state.get("mode"),
            "processed_this_kick": processed,
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

    # ── internals ───────────────────────────────────────────────────────────

    def _process_memory(
        self,
        memory_id: int,
        *,
        suppression: dict[str, Any],
        neighbor_k: int,
    ) -> dict[str, Any]:
        outcome = {
            "version": None, "workspace": None,
            "queued": 0, "auto_rejected": 0, "internal": 0,
            "machine_cleared": 0, "cleared_garbage": 0,
        }
        record = self.db.get_memory(memory_id)
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
        units = self.db.evidence.scan_units(memory_id, version)
        if not units:
            return outcome
        # 1) same-memory internal examination (no KNN needed; rule-only).
        internal = self._examine_internal(memory_id, version, workspace, units)
        outcome["internal"] = internal
        entity_a = self._entity_of(record)
        peer_entities: dict[int, "str | None"] = {}
        # 2) cross-memory same-bucket rank pairing.
        for unit in units:
            if unit.get("embedding") is None:
                continue
            hits = self.db.evidence.knn(
                unit["embedding"], k=neighbor_k + 1,
                workspace=workspace or None,
                exclude_memory_id=memory_id,
            )
            # Rank counts TEXT hits only — non-text units the KNN interleaves
            # must not consume a top-3 slot (0.16.2 §1.5 ranks neighbours,
            # not raw row positions).
            text_rank = 0
            for hit in hits:
                if hit.get("kind") != "text":
                    continue
                peer_id = int(hit["memory_id"])
                if peer_id == memory_id:
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
                # 0.16.2 §1.5: machine-decidable check routes generate only
                # within the top-3 neighbour ranks; notify keeps top-10.
                if decision.action != "notify" and text_rank > SCAN_MACHINE_ROUTE_TOP_K:
                    continue
                if decision.action == "check":
                    # 0.16.2 §1.4: difference-based clearance — check-route
                    # pairs must carry an extractable value difference or they
                    # are duplicates/evolution noise. Cleared pairs are
                    # counted, never enqueued, never landed in conflicts.
                    if peer_id not in peer_entities:
                        peer_record = self.db.get_memory(peer_id)
                        peer_entities[peer_id] = (
                            self._entity_of(peer_record) if peer_record else None
                        )
                    verdict = classify_pair(
                        str(unit["text"]), str(hit.get("text") or ""),
                        route=str(decision.reason or ""),
                        entity_a=entity_a, entity_b=peer_entities[peer_id],
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
                )
                if enqueued:
                    outcome["queued"] += 1
        return outcome

    @staticmethod
    def _entity_of(record: dict[str, Any]) -> "str | None":
        """metadata.entity of a memory row (0.16.2 §1.6 subject layer)."""
        raw = record.get("metadata")
        if isinstance(raw, dict):
            value = raw.get("entity")
            return str(value).strip() or None if value is not None else None
        if isinstance(raw, str) and raw:
            try:
                value = json.loads(raw).get("entity")
            except (TypeError, ValueError):
                return None
            return str(value).strip() or None if value is not None else None
        return None

    def _examine_internal(
        self, memory_id: int, version: int, workspace: str,
        units: list[dict[str, Any]],
    ) -> int:
        """Same-memory unit×unit contradictions (E10 ①, §6⑳).

        First-round gate (real-library calibration): overlapping/nested span
        pairs are splitter artifacts; `check` pairs land only in the
        genuine same-sentence-different-value shape — enumeration ordinals
        ("1. 营销交付" vs "7. 复核终审") are series structure, not
        contradictions. Deterministic notify pairs always land.
        """
        landed = 0
        count = len(units)
        for i in range(count):
            for j in range(i + 1, count):
                a, b = units[i], units[j]
                if not a.get("text") or not b.get("text"):
                    continue
                if spans_overlap(
                    (int(a["start_offset"]), int(a["end_offset"])),
                    (int(b["start_offset"]), int(b["end_offset"])),
                ):
                    continue
                decision = decide_evidence(str(a["text"]), str(b["text"]))
                if decision.action == "ignore":
                    continue
                if decision.action != "notify":
                    if decision.reason != "numeric_value_candidate":
                        continue
                    if not genuine_numeric_pair(str(a["text"]), str(b["text"])):
                        continue
                if self.db.internal_conflicts.exists(memory_id, version, int(a["unit_index"]), int(b["unit_index"])):
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
    ) -> bool:
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
            severity="high" if decision.action == "notify" else "normal",
            source="scan_pipeline",
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
        try:
            import numpy as np
        except ImportError:
            return 0
        from .constants import NORMALIZE_VOTE_NEIGHBORS, NORMALIZE_VOTE_MIN_FOREIGN
        from .normalize_gate import normalize_gate

        vectors = self.db.memories.all_summary_vectors()
        if not vectors:
            return 0
        ids = [mid for mid in sorted(vectors) if mid in set(processed_ids)]
        if not ids:
            return 0
        # The vote matrix still spans the WHOLE library: a mis-placed memory
        # must be judged against its true neighbours, wherever they live.
        all_ids = sorted(vectors)
        workspaces = {mid: str(vectors[mid][0] or "") for mid in all_ids}
        matrix = np.array([vectors[mid][1] for mid in all_ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        unit = matrix / norms[:, None]
        index_of = {mid: i for i, mid in enumerate(all_ids)}
        k = min(NORMALIZE_VOTE_NEIGHBORS, len(all_ids) - 1)
        if k < NORMALIZE_VOTE_MIN_FOREIGN:
            return 0
        landed = 0
        for mid in ids:
            row = index_of[mid]
            sims = unit @ unit[row]
            sims[row] = -1.0
            order = np.argsort(-sims, kind="stable")[:k]
            votes: dict[str, int] = {}
            for col in order:
                bucket = workspaces[all_ids[int(col)]]
                votes[bucket] = votes.get(bucket, 0) + 1
            own = workspaces[mid]
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

    def tokens(text: str) -> set:
        parts = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", str(text).casefold())
        return {p for p in parts if p}

    ta, tb = tokens(quote_a), tokens(quote_b)
    if not ta or not tb:
        return False
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) >= 0.4


def _workspace_identity(memory_id: int, version: int, suspected: str) -> str:
    import hashlib

    return hashlib.sha256(
        f"workspace:{memory_id}@{version}:{suspected}".encode("utf-8")
    ).hexdigest()
