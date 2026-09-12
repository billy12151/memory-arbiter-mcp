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

from .constants import SCAN_PIPELINE_AUTO_REJECT_CAP
from .db_generation import CONFLICT_DETECTOR_VERSION
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

    # ── public API ──────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        pending = self.db.pending_scan_memory_count()
        state = self.db.meta.scan_pipeline_state() or {}
        return {
            "ok": True,
            "detector_version": CONFLICT_DETECTOR_VERSION,
            "pending_memories": pending,
            "queue": self.db.scan_queue_counts(),
            "queue_backlog": self.db.scan_queue_backlog(),
            "pipeline": {
                "round_id": state.get("round_id"),
                "mode": state.get("mode"),
                "complete": bool(state.get("complete")),
                "processed": int(state.get("processed") or 0),
                "auto_rejected": int(state.get("auto_rejected") or 0),
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
                "complete": False,
                "started_at": now,
                "updated_at": now,
            }
        state["updated_at"] = now
        self.db.meta.record_scan_pipeline_state(state)

        suppression = self._load_suppression()
        fresh_round = int(state.get("processed") or 0) == 0
        started = time.monotonic()
        budget = time_budget_s
        last_id = int(state.get("last_id") or 0)
        processed = int(state.get("processed") or 0)
        queued = int(state.get("queued") or 0)
        auto_rejected = int(state.get("auto_rejected") or 0)
        internal_found = int(state.get("internal_found") or 0)
        auto_reject_remaining = max(0, SCAN_PIPELINE_AUTO_REJECT_CAP - auto_rejected)
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
                    auto_reject_remaining=auto_reject_remaining,
                )
                version = outcome["version"]
                if version is not None:
                    self.db.mark_scanned(memory_id, version)
                bucket = outcome.get("workspace") or ""
                if bucket:
                    anchor_buckets[bucket] = anchor_buckets.get(bucket, 0) + 1
                queued += outcome["queued"]
                auto_rejected += outcome["auto_rejected"]
                auto_reject_remaining = max(0, auto_reject_remaining - outcome["auto_rejected"])
                internal_found += outcome["internal"]
                last_id = max(last_id, memory_id)
                processed_ids.append(memory_id)
                processed += 1
            else:
                continue
            break

        normalized = self._enqueue_workspace_suspects(processed_ids)
        state.update({
            "last_id": last_id,
            "processed": processed,
            "queued": queued,
            "auto_rejected": auto_rejected,
            "internal_found": internal_found,
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
            "queued_total": queued,
            "auto_rejected_total": auto_rejected,
            "internal_found_total": internal_found,
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
        auto_reject_remaining: int,
    ) -> dict[str, Any]:
        outcome = {
            "version": None, "workspace": None,
            "queued": 0, "auto_rejected": 0, "internal": 0,
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
        # 2) cross-memory same-bucket rank pairing.
        for unit in units:
            if unit.get("embedding") is None:
                continue
            hits = self.db.evidence.knn(
                unit["embedding"], k=neighbor_k + 1,
                workspace=workspace or None,
                exclude_memory_id=memory_id,
            )
            for hit in hits:
                if hit.get("kind") != "text":
                    continue
                peer_id = int(hit["memory_id"])
                if peer_id == memory_id:
                    continue
                peer_bucket = str(
                    hit.get("workspace_canonical") or hit.get("workspace") or ""
                ).strip()
                if peer_bucket and workspace and peer_bucket != workspace:
                    continue  # C3b: same-bucket pairing only
                decision = decide_evidence(str(unit["text"]), str(hit.get("text") or ""))
                if decision.action == "ignore":
                    continue
                refs, candidate_key, candidate_hash = self._pair_identity(
                    memory_id, version, unit, peer_id, hit,
                )
                if self._suppressed(refs, candidate_hash, suppression):
                    continue
                if decision.reason == "numeric_value_candidate" and auto_reject_remaining > 0:
                    rejected = self._auto_reject_numeric(
                        workspace, member_versions=self._pair_members(memory_id, version, unit, peer_id, hit),
                        candidate_key=candidate_key,
                        detection_reason=f"numeric_value_candidate auto-rejected: {decision.reason}",
                    )
                    if rejected:
                        outcome["auto_rejected"] += 1
                        auto_reject_remaining -= 1
                        continue
                enqueued = self._enqueue_pair(
                    workspace, memory_id, version, unit, peer_id, hit,
                    decision=decision, candidate_key=candidate_key,
                    candidate_hash=candidate_hash,
                )
                if enqueued:
                    outcome["queued"] += 1
        return outcome

    def _examine_internal(
        self, memory_id: int, version: int, workspace: str,
        units: list[dict[str, Any]],
    ) -> int:
        """Same-memory unit×unit contradictions (E10 ①, §6⑳)."""
        landed = 0
        count = len(units)
        for i in range(count):
            for j in range(i + 1, count):
                a, b = units[i], units[j]
                if not a.get("text") or not b.get("text"):
                    continue
                decision = decide_evidence(str(a["text"]), str(b["text"]))
                if decision.action == "ignore":
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

    def _auto_reject_numeric(
        self, workspace: str, *, member_versions: list[dict[str, Any]],
        candidate_key: dict[str, Any], detection_reason: str,
    ) -> bool:
        """Machine-exercised not_a_conflict (E11 ③): audit row in ``conflicts``
        doubles as the pair@version suppression source."""
        result = self.db.record_conflict_group(
            workspace_canonical=workspace,
            slot_key=None,
            members=member_versions,
            value_groups=[],
            candidate_key=candidate_key,
            status="not_a_conflict",
            detector_version=CONFLICT_DETECTOR_VERSION,
            source="scan_numeric_autoreject",
            detection_reason=detection_reason,
        )
        return result.get("outcome") in {"inserted", "deduped"}

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
                    "WHERE status IN ('open','applying','not_a_conflict')"
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
        ≥8/10 in one foreign bucket → suspected), but scoped to memories the
        pipeline just processed (incremental by watermark, E10) and landing in
        the judgment queue (kind='workspace') instead of notices. The agent
        second-judges (E5: preview/outline input suffices); only a confirmed
        judgment that ALSO passes the server-side gate physically moves.
        """
        try:
            import numpy as np
        except ImportError:
            return 0
        from .constants import (
            NORMALIZE_VOTE_NEIGHBORS, NORMALIZE_VOTE_SHARE_MIN,
        )

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
        if k < NORMALIZE_VOTE_SHARE_MIN:
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
            best_bucket, best_votes = max(
                ((b, c) for b, c in votes.items() if b != own),
                key=lambda item: item[1],
                default=("", 0),
            )
            if best_votes < NORMALIZE_VOTE_SHARE_MIN or not best_bucket:
                continue
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

    @staticmethod
    def _now() -> str:
        from .models import utc_now_iso

        return utc_now_iso()




def _workspace_identity(memory_id: int, version: int, suspected: str) -> str:
    import hashlib

    return hashlib.sha256(
        f"workspace:{memory_id}@{version}:{suspected}".encode("utf-8")
    ).hexdigest()
