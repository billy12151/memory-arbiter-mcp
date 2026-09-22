"""Local-text evidence indexing and conflict candidate processing."""
from __future__ import annotations

import hashlib
import sqlite3
import json
import time
from typing import Any, TYPE_CHECKING

from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..constants import (
    SEMANTIC_JOB_TIMEOUT_MS,
    SEMANTIC_MAX_EVIDENCE_UNITS,
    SEMANTIC_MAX_EXAMINED_PAIRS,
    SEMANTIC_MAX_ROWS,
    SEMANTIC_MIN_PAIR_BUDGET_MS,
)
from ..difference_classifier import classify_pair
from ..evidence import evidence_content_hash, local_text_units
from ..models import TrustedApplyingContext
from ..embedder import ManagedEmbedder
from ..semantic_conflict import (
    PAIR_PROMPT_VERSION,
    PairGateResult,
    SemanticBackend,
    decide_evidence,
    direct_value_verdict,
    evaluate_single_direction_extraction,
    is_cross_evolution,
    notice_dedupe_key,
    normalize_value,
    signal_extraction,
)
from ..text import canon_entity, canon_scope

if TYPE_CHECKING:
    from ..tools import MemoryTools
    from ..workers import SemanticConflictWorker

# Technical failures degrade the check route and keep the job incomplete.
_TECHNICAL_REASONS = {
    "qwen_timeout", "qwen_unavailable", "qwen_backend_error",
    "qwen_invalid_output", "qwen_budget_exhausted", "notice_budget_exhausted",
    "evidence_units_capped", "rows_capped", "pairs_examined_capped",
    "notice_write_failed",
}


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

    def check_claims_conflicts(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """0.17.0 P2-5.3: the zero-Qwen claims channel (owner decision #6).

        Every claim of the freshly-written memory searches the claim-vector
        KNN (same workspace, active, current version); attr_cos ≥
        CLAIM_ATTR_TAU + value_norm difference + coexistence veto (A4) +
        provenance SOFT gate (owner 2026-09-22: both sides filled AND
        unequal → veto; anything else passes — the attr slot itself anchors
        the comparison) + evidence-channel pair-closure dedup → notice.
        Bounded by CLAIMS_MAX_NOTICES_PER_WRITE (review A3); overflow is
        counted, never silent."""
        from ..constants import CLAIMS_MAX_NOTICES_PER_WRITE, CLAIM_ATTR_TAU
        from ..semantic_conflict import vector_cosine

        record = snapshot if snapshot.get("content") is not None else (
            self.db.get_memory(int(memory_id)) or {}
        )
        version = int(record.get("version") or 1)
        workspace = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        own_claims = self.db.claims.current_claims(int(memory_id))
        if not own_claims:
            return {"claims_checked": 0, "notices": 0}
        if not self.db.state.sqlite_vec_available:
            return {"claims_checked": len(own_claims), "notices": 0, "reason": "vec_unavailable"}
        # Own claim vectors (published on the write path).
        own_rows: dict[int, list[float]] = {}
        with self.db.connection() as conn:
            for row in conn.execute(
                """SELECT c.id AS cid, v.embedding FROM memory_claims c
                   LEFT JOIN memory_claim_vec v ON v.id=c.id
                   WHERE c.memory_id=? AND c.memory_version=?""",
                (int(memory_id), version),
            ).fetchall():
                if row["embedding"] is not None:
                    own_rows[int(row["cid"])] = self.db.evidence._blob_to_vector(bytes(row["embedding"]))
        if not own_rows:
            return {"claims_checked": len(own_claims), "notices": 0, "reason": "vectors_pending"}

        from ..acl import workspace_scope_sql
        workspace_sql, workspace_params = workspace_scope_sql(
            "COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)", workspace,
        )
        eligible = "m.status='active' AND c.memory_version = m.version AND c.memory_id != ?"
        eligible_params: list[Any] = [int(memory_id)]
        if workspace_sql:
            eligible += f" AND {workspace_sql}"
            eligible_params.extend(workspace_params)
        id_constraint = f"c.id IN (SELECT c.id FROM memory_claims c JOIN memories m ON m.id=c.memory_id WHERE {eligible})"

        raw_meta = record.get("metadata")
        own_metadata = raw_meta if isinstance(raw_meta, dict) else {}
        own_entity = str(own_metadata.get("entity") or "").strip()
        own_scope = str(own_metadata.get("scope") or "").strip()

        notices = 0
        capped = False
        checked = 0
        with self.db.connection() as conn:
            for claim in own_claims:
                own_vector = own_rows.get(int(claim["id"]))
                if not own_vector:
                    continue
                checked += 1
                hits = conn.execute(
                    f"""SELECT c.*, v.distance AS distance,
                               m.subject, m.metadata, m.workspace, m.workspace_canonical
                        FROM memory_claim_vec v
                        JOIN memory_claims c ON c.id=v.id
                        JOIN memories m ON m.id=c.memory_id
                        WHERE v.embedding MATCH ? AND k=10 AND {id_constraint}
                        ORDER BY v.distance""",
                    [json.dumps(own_vector), *eligible_params],
                ).fetchall()
                for hit in hits:
                    hit_vector = None
                    vec_row = conn.execute(
                        "SELECT embedding FROM memory_claim_vec WHERE id=?", (int(hit["id"]),)
                    ).fetchone()
                    if vec_row is not None and vec_row["embedding"] is not None:
                        hit_vector = self.db.evidence._blob_to_vector(bytes(vec_row["embedding"]))
                    attr_cos = vector_cosine(own_vector, hit_vector)
                    same_exact = str(hit["attr_norm"]) == str(claim["attr_norm"])
                    if not same_exact and (attr_cos is None or attr_cos < CLAIM_ATTR_TAU):
                        continue
                    if str(hit["value_norm"]) == str(claim["value_norm"]):
                        continue
                    peer_id = int(hit["memory_id"])
                    # A4 coexistence: either side declaring multiple values for
                    # the attr is a self-coexistence, not an opposing claim.
                    if len(self.db.claims.coexisting_values(int(memory_id), str(claim["attr_norm"]))) > 1:
                        continue
                    if len(self.db.claims.coexisting_values(peer_id, str(hit["attr_norm"]))) > 1:
                        continue
                    # Soft provenance gate (owner 2026-09-22 拍板；单侧未填=
                    # 不挡——review 推荐①，随 P2-5 review 收口)。
                    raw_hit_meta = hit["metadata"] if "metadata" in hit.keys() else None
                    hit_metadata = raw_hit_meta if isinstance(raw_hit_meta, dict) else {}
                    if isinstance(raw_hit_meta, str) and raw_hit_meta:
                        try:
                            hit_metadata = json.loads(raw_hit_meta)
                        except (TypeError, ValueError):
                            hit_metadata = {}
                    hit_entity = str(hit_metadata.get("entity") or "").strip()
                    hit_scope = str(hit_metadata.get("scope") or "").strip()
                    if (
                        own_entity and hit_entity and own_entity != hit_entity
                        and own_scope and hit_scope and own_scope != hit_scope
                    ):
                        continue
                    # Cross-channel dedup (appendix C 7): a pair the evidence
                    # channel already settled this version never double-fires.
                    if self.db.semantic_notices.is_semantic_pair_closed_on_conn(
                        conn, int(memory_id), peer_id, version, int(hit["memory_version"] or 1),
                    ):
                        continue
                    if notices >= CLAIMS_MAX_NOTICES_PER_WRITE:
                        capped = True
                        break
                    entity = own_entity if own_entity == hit_entity else (own_entity or hit_entity or "")
                    scope = own_scope if own_scope == hit_scope else (own_scope or hit_scope or "")
                    if not entity or not scope:
                        continue
                    from ..text import canon_entity, canon_scope
                    slot_key = {
                        "entity": canon_entity(entity), "attribute": str(hit["attr_norm"]),
                        "scope": canon_scope(scope),
                    }
                    outcome = self.db.record_semantic_notice(
                        memory_id=int(memory_id), peer_id=peer_id, severity="normal",
                        notice_type="claim_conflict",
                        title=f"Claim conflict with #{peer_id}",
                        message=f"claims attr {claim['attr']} value differs",
                        payload={
                            "route": "notice_ready",
                            "reason": "claim_attr_vector_gate",
                            "source": "claim_conflict",
                            "slot_key": slot_key,
                            "slot_provenance": {
                                "entity": "metadata", "scope": "metadata",
                                "attribute": "claims_channel",
                            },
                            "attr_cos": round(float(attr_cos or 1.0), 4),
                            "member_versions": [
                                {"memory_id": int(memory_id), "version": version,
                                 "value": str(claim["value_norm"]),
                                 "evidence": {"quote": str(claim["value"])}},
                                {"memory_id": peer_id, "version": int(hit["memory_version"] or 1),
                                 "value": str(hit["value_norm"]),
                                 "evidence": {"quote": str(hit["value"])}},
                            ],
                            "value_groups": [
                                {"normalized_value": str(claim["value_norm"]),
                                 "display_value": str(claim["value"]),
                                 "members": [f"{memory_id}@{version}"]},
                                {"normalized_value": str(hit["value_norm"]),
                                 "display_value": str(hit["value"]),
                                 "members": [f"{peer_id}@{int(hit['memory_version'] or 1)}"]},
                            ],
                            "candidate_key": {
                                "detector_version": CONFLICT_DETECTOR_VERSION,
                                "members": sorted([
                                    f"{memory_id}@{version}",
                                    f"{peer_id}@{int(hit['memory_version'] or 1)}",
                                ]),
                                "evidence": [],
                            },
                            "left_evidence": {"text": str(claim["value"])},
                            "right_evidence": {"text": str(hit["value"])},
                            "claims_channel": True,
                        },
                        dedupe_key=notice_dedupe_key(
                            int(memory_id), peer_id, version, int(hit["memory_version"] or 1),
                            "claim_conflict",
                        ),
                        left_version=version, right_version=int(hit["memory_version"] or 1),
                        source="claim_conflict",
                    )
                    if outcome.get("outcome") == "created":
                        notices += 1
                if capped:
                    break
        result: dict[str, Any] = {"claims_checked": checked, "notices": notices}
        if capped:
            result["claims_notices_capped"] = True
        return result

    def drain_conflict_backlog(self, limit: int = 2) -> int:
        """0.17.0 P2-4.2: idle-worker consumption of the conflict backlog.

        Bounded per call (``limit`` entries); new writes always win because
        the caller only invokes this when the job queue is empty and rechecks
        between entries. A stored extraction replays through the deterministic
        gate without Qwen; a Qwen-less replay with no extraction lands
        nothing (the entry stays pending for a backend-bearing pass — never
        silently completed)."""
        processed = 0
        while processed < limit:
            entry = self.db.conflict_backlog.take_next()
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
                if backend is None:
                    break  # keep pending; a later backend-bearing pass retries
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
                def _env(memory: dict[str, Any], quote: str) -> dict[str, Any]:
                    metadata_value = memory.get("metadata")
                    metadata = metadata_value if isinstance(metadata_value, dict) else {}
                    return {
                        "quote": quote[:1000], "subject": str(memory.get("subject") or "")[:200],
                        "tags": list(memory.get("tags") or [])[:20],
                        "workspace_canonical": memory.get("workspace_canonical") or memory.get("workspace"),
                        "memory_id": int(memory.get("id") or 0),
                        "version": int(memory.get("version") or 1),
                        "event_time": memory.get("event_time"),
                        "metadata": {k: metadata.get(k) for k in ("entity", "scope") if metadata.get(k)},
                    }
                try:
                    forward = backend.classify_pair(
                        _env(left, left_text), _env(right, right_text),
                        retry_allowed=not self._semantic_worker.has_pending_jobs(),
                    )
                except TypeError:
                    forward = backend.classify_pair(_env(left, left_text), _env(right, right_text))
                gate = evaluate_single_direction_extraction(
                    signal_extraction(forward), _env(left, left_text), _env(right, right_text),
                )
                if gate.state == "notice_ready":
                    self._record_backlog_notice(
                        left, right, left_text, right_text, decision,
                        str(gate.attribute), str(gate.value_a), str(gate.value_b),
                        reason=str(gate.reason),
                    )
                self.db.conflict_backlog.complete(int(entry["id"]))
                processed += 1
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
        return processed

    def _record_backlog_notice(
        self, left: dict[str, Any], right: dict[str, Any],
        left_text: str, right_text: str, decision: Any,
        attribute: str, value_a: str, value_b: str, *, reason: str,
    ) -> None:
        """Record one notice for a backlog pair through the standard channel
        (dedupe/suppression identical to the write path)."""
        left_id = int(left.get("id") or 0)
        right_id = int(right.get("id") or 0)
        left_version = int(left.get("version") or 1)
        right_version = int(right.get("version") or 1)
        raw_meta = left.get("metadata")
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        raw_peer_meta = right.get("metadata")
        peer_metadata = raw_peer_meta if isinstance(raw_peer_meta, dict) else {}
        entity = metadata.get("entity") if metadata.get("entity") == peer_metadata.get("entity") else None
        scope = metadata.get("scope") if metadata.get("scope") == peer_metadata.get("scope") else None
        if not entity or not scope:
            return  # soft-invisible: slot provenance missing, dedupe will keep the pair calm
        slot_key = {
            "entity": canon_entity(entity), "attribute": attribute, "scope": canon_scope(scope),
        }
        slot_json = json.dumps(slot_key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.db.record_semantic_notice(
            memory_id=left_id, peer_id=right_id, severity="normal",
            notice_type="semantic_evidence",
            title=f"Possible memory change with #{right_id}",
            message=str(decision.reason or "backlog"),
            payload={
                "route": "notice_ready", "reason": reason,
                "prompt_version": PAIR_PROMPT_VERSION,
                "anchors": decision.anchors,
                "slot_key": slot_key,
                "slot_provenance": {"entity": "metadata", "scope": "metadata", "attribute": "backlog"},
                "member_versions": [
                    {"memory_id": left_id, "version": left_version, "value": value_a,
                     "evidence": {"quote": left_text}},
                    {"memory_id": right_id, "version": right_version, "value": value_b,
                     "evidence": {"quote": right_text}},
                ],
                "value_groups": [
                    {"normalized_value": value_a, "display_value": value_a, "members": [f"{left_id}@{left_version}"]},
                    {"normalized_value": value_b, "display_value": value_b, "members": [f"{right_id}@{right_version}"]},
                ],
                "candidate_key": {
                    "detector_version": CONFLICT_DETECTOR_VERSION,
                    "members": sorted([f"{left_id}@{left_version}", f"{right_id}@{right_version}"]),
                    "evidence": [],
                },
                "left_evidence": {"text": left_text},
                "right_evidence": {"text": right_text},
                "backlog": True,
            },
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
        units = local_text_units(
            str(current.get("subject") or ""), str(current.get("content") or ""),
        )
        embeddings: list[list[float]] = []
        for unit in units:
            result = embedder.embed_text(prefix="", body=unit.text)
            if not result.embedding:
                return {"status": "failed", "reason": "empty_embedding"}
            embeddings.append(list(result.embedding))
        # 0.17.0 P2-2.3: row-level embed for the conflict channel — outside
        # the publish transaction (same discipline as units), landing in the
        # same atomic snapshot. ~10→~35 embeds per memory on the GPU worker
        # thread; the write path itself stays async (P0 evidence queue).
        from ..rowseg import segment_rows
        row_segments = segment_rows(
            str(current.get("subject") or ""), str(current.get("content") or ""),
        )
        row_embeddings: list[list[float]] = []
        for segment in row_segments:
            result = embedder.embed_text(prefix="", body=segment.text)
            if not result.embedding:
                return {"status": "failed", "reason": "empty_embedding"}
            row_embeddings.append(list(result.embedding))
        # 0.16.12 P2-T2: prefer the row's maintained content_sha column (set
        # at insert and every content edit) — one hash per write instead of
        # re-hashing here; NULL only on exotic legacy rows, hence the fallback.
        row_sha = str(current.get("content_sha") or "") or evidence_content_hash(
            str(current.get("content") or "")
        )
        published = self.db.evidence.publish(
            int(memory_id), int(current.get("version") or 1), row_sha, units, embeddings,
            rows=row_segments, row_embeddings=row_embeddings,
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

    def process_conflicts(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Run the bounded notice gate and report completion to sync callers."""
        record = self.db.get_memory(int(memory_id))
        if record and record.get("status") == "pending":
            # A pending (workspace-activation) memory is not an incomplete
            # check: the conflict job is simply skipped until activation,
            # matching the "skipped" semantics used by index_memory above.
            return {"status": "skipped", "reason": "pending_workspace_activation", "notices_created": 0}
        if not record or record.get("status") != "active":
            return {"status": "incomplete", "reason": "memory_not_active", "notices_created": 0}
        if int(record.get("version") or 1) != int(snapshot.get("version") or 1):
            return {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
        content = str(record.get("content") or "")
        # 0.16.12 P2-T2: the row's content_sha column IS sha256(content) (set
        # at insert, recomputed on every content edit) — compare against it
        # instead of re-hashing the content again (legacy NULL falls back).
        row_sha = str(record.get("content_sha") or "")
        if not row_sha:
            row_sha = hashlib.sha256(content.encode()).hexdigest()
        if row_sha != snapshot.get("content_hash"):
            return {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
        # 0.16.12 P2-T6: the whole job body runs on ONE read-only connection
        # under an explicit read transaction — per-unit evidence_knn, per-pair
        # exists/closed probes and the peer prefetch all reuse it instead of
        # opening one connection each. The BEGIN gives the job a single WAL
        # snapshot (the old open-per-read behaviour saw a different
        # point-in-time on every read); the job's own notice writes keep their
        # own per-notice write transactions and never touch this connection.
        with self.db.connection() as job_conn:
            job_conn.execute("BEGIN")
            job_started = time.monotonic()
            try:
                result = self._process_conflicts_job(
                    memory_id, snapshot, record, job_conn, content, row_sha,
                )
            finally:
                try:
                    job_conn.rollback()
                except sqlite3.Error:
                    pass
        # 0.16.12 eval contract: ACTUAL job execution time (embedding +
        # KNN candidate collection + Qwen pairs + notice writes), excluding
        # the caller's notice_sync_wait window — additive receipt key so the
        # harness can report processing cost without the wait-window noise.
        result["elapsed_ms"] = round((time.monotonic() - job_started) * 1000, 1)
        return result

    def _process_conflicts_job(
        self, memory_id: int, snapshot: dict[str, Any],
        record: dict[str, Any], job_conn: "sqlite3.Connection",
        content: str, row_sha: str,
    ) -> dict[str, Any]:
        embedder, _ = self._ensure_active_embedder()
        if embedder is None:
            return {"status": "incomplete", "reason": "embedder_unavailable", "notices_created": 0}
        if self.db.get_vec_index_state().get("state") in {"mismatch", "failed"}:
            return {
                "status": "incomplete",
                "reason": "embedding_space_rebuild_required",
                "notices_created": 0,
            }
        # Spec §5/§15.3: while a conflict group is applying, versions produced
        # by its apply plan must not re-notify THE SAME conflict. Suppression is
        # therefore slot-scoped and applied only after the gate resolves the
        # candidate's slot_key, so a genuinely different conflict between the
        # same two memories is still examined and surfaced. Validation is
        # server-side against the live conflict rows; the trusted context only
        # names which row to revalidate.
        applying_slots: set[str] = set()
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
        min_budget = SEMANTIC_MIN_PAIR_BUDGET_MS / 1000.0
        # Behaviour change (v3 hardening): each degradation reason is counted
        # at most once per task. The pair loop can hit the same technical
        # failure for many pairs, and counting every hit made
        # _check_degradation_count grow with pair count rather than with
        # distinct failure modes. reasons_seen keeps the per-task list that
        # _check_degradation_reason (last write wins) would otherwise lose.
        degradation_reasons: set[str] = set()
        reasons_seen: list[str] = []

        def record_degradation(reason: str, sample: str | None = None) -> None:
            if reason in degradation_reasons:
                return
            degradation_reasons.add(reason)
            reasons_seen.append(reason)
            self._tools._record_check_degradation(reason, sample)

        def backlog_deadline() -> float | None:
            value = self._semantic_worker.pending_job_deadline(
                SEMANTIC_JOB_TIMEOUT_MS / 1000.0,
            )
            return float(value) if value is not None else None

        max_units = max(1, SEMANTIC_MAX_EVIDENCE_UNITS)
        max_rows = max(1, SEMANTIC_MAX_ROWS)
        workspace = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        by_peer: dict[int, tuple[dict[str, Any], Any, Any]] = {}
        reached_pair: set[int] = set()

        def _enqueue_backlog(entries: list[tuple[int, tuple[dict[str, Any], Any, Any]]]) -> int:
            """0.17.0 P2-4.2: truncation leftovers land in conflict_backlog
            instead of vanishing. Identity = detector version + both
            members@version + row anchors (review A7: a detector bump or a
            member edit invalidates the frozen pair)."""
            enqueued = 0
            left_version = int(record.get("version") or 1)
            for peer_id, (hit, seg_view, decision) in entries:
                from ..constants import (
                    PAIR_SCORE_W_BOTH_VALUES,
                    PAIR_SCORE_W_NUMERIC_ROUTE,
                )
                score = 0.0
                if str(decision.reason or "") == "numeric_value_candidate":
                    score += PAIR_SCORE_W_NUMERIC_ROUTE
                if decision.left_value and decision.right_value:
                    score += PAIR_SCORE_W_BOTH_VALUES
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
                    left_memory_id=int(memory_id), left_version=left_version,
                    right_memory_id=int(peer_id), right_version=right_version,
                    left_text=str(seg_view.text), right_text=str(hit.get("text") or ""),
                    pair_score=score,
                )
                if outcome.get("outcome") in {"queued", "duplicate"}:
                    enqueued += 1
            return enqueued

        # P2-T2: same digest as the stale check above — the maintained
        # content_sha column (or its recompute fallback), never a fresh hash.
        content_hash = row_sha
        # 0.17.0 P2-3.1: the conflict channel is ROW-level first. A1 timing
        # bridge (review): read the published row vectors; when the evidence
        # worker has not landed the publish yet, recover in-job with
        # rowseg+embed — the same read-then-recover contract the unit path
        # always had, so the two worker queues need no ordering guarantee.
        row_vectors = self.db.evidence.current_row_vectors(
            int(memory_id), int(record.get("version") or 1), content_hash,
        )
        if not row_vectors and embedder is not None:
            from ..rowseg import segment_rows
            row_vectors = []
            for segment in segment_rows(str(record.get("subject") or ""), content):
                embedded = embedder.embed_text(prefix="", body=segment.text)
                if embedded.embedding:
                    row_vectors.append((segment, list(embedded.embedding)))
        # Graceful degradation (mid-backfill / degraded embedder): no rows
        # anywhere → the pre-0.17.0 unit path stays the candidate source, so
        # detection never goes BLIND while row coverage catches up.
        rows_mode = bool(row_vectors)
        unit_vectors: list[tuple[Any, list[float]]] = []
        if not rows_mode:
            unit_vectors = self.db.evidence.current_text_vectors(
                int(memory_id), int(record.get("version") or 1), content_hash,
            )
            if not unit_vectors:
                # Recovery fallback for an incomplete/legacy evidence publish. The
                # normal write path has just published these vectors, so avoid a
                # second GGUF embedding pass in the common synchronous-notice path.
                unit_vectors = []
                for unit in local_text_units(str(record.get("subject") or ""), content):
                    if unit.kind != "text":
                        continue
                    embedded = embedder.embed_text(prefix="", body=unit.text)
                    if embedded.embedding:
                        unit_vectors.append((unit, list(embedded.embedding)))
        # Normalized segment view: rows carry row_index, units carry
        # unit_index — the view exposes .unit_index for BOTH so every
        # downstream consumer (internal create, envelopes, member evidence)
        # stays unchanged (P2-3.1 keeps every gate a pure text-pair function;
        # only the input granularity changed).
        from collections import namedtuple
        _SegView = namedtuple("_SegView", "text start_offset end_offset unit_index")
        paired = list(row_vectors if rows_mode else unit_vectors)
        seg_views = [
            _SegView(
                seg.text, int(seg.start_offset), int(seg.end_offset),
                int(seg.row_index if rows_mode else seg.unit_index),
            )
            for seg, _embedding in paired
        ]
        seg_embeddings = [embedding for _seg, embedding in paired]
        if rows_mode:
            # P2-3.1 值锚定行优先：rows carrying an extractable value lead the
            # cap order (12th round: value features are the conflict
            # predictor; topic similarity is not). Deterministic tiebreak by
            # segment order.
            from ..semantic_conflict import _VALUE_RE
            order = sorted(
                range(len(seg_views)),
                key=lambda idx: (0 if _VALUE_RE.search(seg_views[idx].text) else 1, seg_views[idx].unit_index),
            )
            seg_views = [seg_views[idx] for idx in order]
            seg_embeddings = [seg_embeddings[idx] for idx in order]
        max_segments = max_rows if rows_mode else max_units
        segments_capped_reason = "rows_capped" if rows_mode else "evidence_units_capped"
        # 0.16.0 E10① (§6⑳): same-memory internal examination comes FIRST —
        # units are in hand (no KNN), the rule is deterministic, and the
        # finding lands in the dedicated internal_conflicts structure (the
        # conflicts table's pair invariants reject a single memory@version
        # twice). It consumes no Qwen budget; the cross-memory loop below is
        # untouched in shape, and a truncated cross loop still reports the
        # internal findings already landed this run.
        # 0.16.2 (owner, unified flow): the internal check uses the SAME
        # filter-plus-slot-extraction logic as the cross-memory route. A
        # contradiction a reader would flag in one document must not become
        # invisible just because it was written inside ONE memory.
        # 0.16.4 §0.5/§2: the whole filter sequence is ONE shared gate —
        # internal_pair_admission (scan_pipeline) — called identically by
        # the scan side; the callers differ only in what an admitted pair
        # means. Here: EVERY admitted shape (check AND notify) collects for
        # the Qwen final review — a real in-memory self-contradiction has a
        # recognition duty, and Qwen's verdict is the triple
        # ready→pending+attribution / definitive negative→dismissed veto /
        # technical failure→pending unannotated (fail-open).
        internal_found = 0
        internal_version = int(record.get("version") or 1)
        # 0.17.0 P2-3.1: internal (same-memory) pairs are row segments in
        # rows mode (row_index lands in internal_conflicts' unit_a/unit_b —
        # the 0.17.0 detector bump separates the index semantics cleanly).
        from ..scan_pipeline import internal_pair_admission

        internal_qwen_pairs: list[tuple[Any, Any, Any]] = []
        for i in range(len(seg_views)):
            for j in range(i + 1, len(seg_views)):
                seg_a, seg_b = seg_views[i], seg_views[j]
                internal_decision = decide_evidence(seg_a.text, seg_b.text)
                admitted = internal_pair_admission(
                    seg_a.text, seg_b.text,
                    (seg_a.start_offset, seg_a.end_offset),
                    (seg_b.start_offset, seg_b.end_offset),
                    internal_decision,
                    exists_probe=lambda: self.db.internal_conflicts.exists_on_conn(
                        job_conn, int(memory_id), internal_version,
                        seg_a.unit_index, seg_b.unit_index,
                    ),
                )
                if not admitted:
                    continue
                internal_qwen_pairs.append((seg_a, seg_b, internal_decision))
        units_examined = 0
        # 0.16.2 write-time pre-gates (owner, data-driven): two deterministic
        # filters run in the KNN collection loop, BEFORE the per-peer dedup —
        # a cleared representative would otherwise burn a peer slot that a
        # kept hit of the same peer could have taken (live-library
        # simulation: 86 slots recoverable).
        # Gate 0 (0.16.4 §1, evolution domain): cross-memory notify shapes
        # die above, before all of the following — timeline phenomena are
        # not conflicts.
        # Gate 1 (provenance, zero loss): a notice needs BOTH sides' entity
        # AND scope metadata present and equal (the post-Qwen slot builder
        # drops anything else with slot_provenance_insufficient) — skipping
        # early saves the two Qwen inferences per pair. Cheapest and biggest
        # kill (~95% of representatives), so it runs FIRST.
        # Gate 2 (difference classifier): no extractable value difference
        # means the pair can never satisfy Qwen's same-attribute-different-
        # value gate.
        provenance_filtered = 0
        no_difference_filtered = 0
        raw_own_meta = record.get("metadata")
        own_metadata = raw_own_meta if isinstance(raw_own_meta, dict) else {}
        own_entity = str(own_metadata.get("entity") or "").strip()
        own_scope = str(own_metadata.get("scope") or "").strip()
        # Spec §15.5: a bounded check that ran out of budget must not later
        # claim checked_no_notice. The two truncation causes report
        # distinctly (2026-09-10 #957/#959 diagnosis: the shared string cost
        # an extra investigation round): the per-memory evidence-unit cap is
        # evidence_units_capped; the fair job deadline stays
        # notice_budget_exhausted. The cap is checked first so a state where
        # both hold attributes to the more specific cause.
        truncation_reason: str | None = None
        # 0.17.0 P2-3.1: the cross loop walks the normalized segments —
        # row_knn in rows mode (candidates are clean short sentences or
        # header-folded table rows), evidence_knn otherwise. Rows carry no
        # 'text'-only kind filter (table rows are first-class candidates).
        for seg_view, embedding in zip(seg_views, seg_embeddings):
            active_deadline = backlog_deadline()
            if units_examined >= max_segments:
                truncation_reason = segments_capped_reason
                break
            if active_deadline is not None and time.monotonic() >= active_deadline:
                truncation_reason = "notice_budget_exhausted"
                break
            units_examined += 1
            knn_hits = (
                self.db.row_knn(
                    embedding, k=5, workspace=workspace,
                    exclude_memory_id=memory_id, conn=job_conn,
                )
                if rows_mode else
                self.db.evidence_knn(
                    embedding, k=5, workspace=workspace,
                    exclude_memory_id=memory_id, conn=job_conn,
                )
            )
            for hit in knn_hits:
                if not rows_mode and hit.get("kind") != "text":
                    continue
                decision = decide_evidence(seg_view.text, str(hit.get("text") or ""))
                if decision.action == "ignore":
                    continue
                # 0.16.4 §1: cross-memory evolution domain — the earliest
                # kill. It happens BEFORE the provenance gate, so a notify
                # shape never consumes provenance/classifier work, a peer
                # slot, a sort position, or Qwen budget. Same predicate as
                # the scan side (§0.5 single implementation).
                if is_cross_evolution(decision):
                    continue
                peer_id = int(hit["memory_id"])
                raw_hit_meta = hit.get("metadata")
                if isinstance(raw_hit_meta, str) and raw_hit_meta:
                    try:
                        raw_hit_meta = json.loads(raw_hit_meta)
                    except (TypeError, ValueError):
                        raw_hit_meta = {}
                hit_metadata = raw_hit_meta if isinstance(raw_hit_meta, dict) else {}
                hit_entity = str(hit_metadata.get("entity") or "").strip()
                hit_scope = str(hit_metadata.get("scope") or "").strip()
                if not (
                    own_entity and hit_entity and own_entity == hit_entity
                    and own_scope and hit_scope and own_scope == hit_scope
                ):
                    provenance_filtered += 1
                    continue
                if classify_pair(
                    seg_view.text, str(hit.get("text") or ""), route=str(decision.reason or ""),
                ) == "clear":
                    no_difference_filtered += 1
                    continue
                existing = by_peer.get(peer_id)
                closer = existing is not None and float(hit.get("distance") or 9) < float(existing[0].get("distance") or 9)
                # 0.16.4 §1: only check shapes reach here now, so the
                # notify-priority protection lost its subject — the closer
                # neighbour of the same peer wins outright.
                if existing is None or closer:
                    by_peer[peer_id] = (hit, seg_view, decision)
        if truncation_reason:
            # E10① order guarantee: internal findings land BEFORE the cross
            # loop and survive its truncation. The internal Qwen pass has not
            # run yet at this point (it needs the backend fetched below), so
            # the collected keepers land unannotated here — fail-open, never
            # lost. (Adversarial self-review: without this, an evidence-unit
            # cap or budget exhaustion mid-collection silently dropped every
            # internal keep pair of this run.)
            for unit_a, unit_b, internal_decision in internal_qwen_pairs:
                if self.db.internal_conflicts.create(
                    memory_id=int(memory_id), memory_version=internal_version,
                    unit_a=unit_a.unit_index, unit_b=unit_b.unit_index,
                    quote_a=unit_a.text, quote_b=unit_b.text,
                    span_a=[unit_a.start_offset, unit_a.end_offset],
                    span_b=[unit_b.start_offset, unit_b.end_offset],
                    reason=str(internal_decision.reason or ""),
                    detector_version=CONFLICT_DETECTOR_VERSION,
                ):
                    internal_found += 1
            record_degradation(truncation_reason)
            early_result: dict[str, Any] = {
                "status": "incomplete", "reason": truncation_reason,
                "notices_created": 0, "reasons_seen": reasons_seen,
                # Internal-truncation exits before the cross-memory loop even
                # starts, so no pairs were examined yet (counter not yet live).
                "pairs_examined": 0,
            }
            if internal_found:
                early_result["internal_conflicts"] = internal_found
            # 0.17.0 P2-4.2: collected-but-unexamined candidates go to the
            # backlog (owner design #8) — the receipt says how many.
            early_backlogged = _enqueue_backlog(list(by_peer.items()))
            if early_backlogged:
                early_result["backlogged"] = early_backlogged
            return early_result

        backend = self._ensure_semantic_backend()
        # C4 soft ordering (⑦ 定案): rank same-level pairs by subject+tags
        # overlap before distance — the Qwen budget should spend on pairs the
        # owner's signals (subject/tag) already flag as related. Zero-overlap
        # pairs are only ordered later, never excluded. (0.16.4 §1: the
        # notify-first key lost its subject — only check shapes remain.)
        from ..semantic_conflict import vector_cosine

        hint_vectors = self.db.memories.subject_tags_vectors(
            [memory_id, *[peer_id for peer_id in by_peer]],
        )
        own_vector = hint_vectors.get(int(memory_id))
        if own_vector is None:
            overlap_rank = {peer_id: 0.0 for peer_id in by_peer}
        else:
            overlap_rank = {
                peer_id: vector_cosine(own_vector, hint_vectors.get(peer_id))
                for peer_id in by_peer
            }
        # 0.17.0 P2-3.4: pair_score orders the Qwen budget — value features
        # lead (routed numeric + both-sides-extractable), C4 subject/tags
        # overlap is the base, row distance the tiebreak. Order-only: a
        # single pair's verdict never changes (owner-approved boundary).
        # Weights initial; P2-3.2 recalibrates on the noisy corpus.
        from ..constants import (
            PAIR_SCORE_W_BOTH_VALUES,
            PAIR_SCORE_W_NUMERIC_ROUTE,
            PAIR_SCORE_W_OVERLAP,
        )

        def _pair_score(peer_id: int, triple: tuple[dict[str, Any], Any, Any]) -> float:
            _hit, _seg, decision = triple
            score = PAIR_SCORE_W_OVERLAP * float(overlap_rank.get(peer_id) or 0.0)
            if str(decision.reason or "") == "numeric_value_candidate":
                score += PAIR_SCORE_W_NUMERIC_ROUTE
            if decision.left_value and decision.right_value:
                score += PAIR_SCORE_W_BOTH_VALUES
            return score

        ordered = sorted(
            by_peer.items(),
            key=lambda item: (
                -_pair_score(item[0], item[1]),
                float(item[1][0].get("distance") or 9),
            ),
        )
        # 0.15.14 (A5): the former surfaced>=max_notice_pairs early stop is
        # gone — notices are recorded per-pair inside the loop (write-on-
        # discovery), so an early stop only saved Qwen time, which the
        # examined-pairs cap now bounds deterministically. The check examines
        # up to SEMANTIC_MAX_EXAMINED_PAIRS pairs (fair deadline first, cap
        # second) and surfaces every notice it finds; the notice count is
        # therefore bounded by the pairs cap.
        max_examined_pairs = max(1, SEMANTIC_MAX_EXAMINED_PAIRS)
        pairs_examined = 0
        surfaced = 0
        dropped_unlocalizable = 0
        backlogged = 0
        incomplete_reason: str | None = None

        def envelope(memory: dict[str, Any], quote: str) -> dict[str, Any]:
            metadata_value = memory.get("metadata")
            metadata = metadata_value if isinstance(metadata_value, dict) else {}
            return {
                "quote": quote[:1000], "subject": str(memory.get("subject") or "")[:200],
                "tags": list(memory.get("tags") or [])[:20],
                "workspace_canonical": memory.get("workspace_canonical") or memory.get("workspace"),
                "memory_id": int(memory.get("id") or 0), "version": int(memory.get("version") or 1),
                "event_time": memory.get("event_time"),
                "metadata": {key: metadata.get(key) for key in ("entity", "scope") if metadata.get(key)},
            }

        def classify(left_env: dict[str, Any], right_env: dict[str, Any]) -> Any:
            if backend is None:
                # The per-pair guard below never lets a None reach the call;
                # keep the narrowed local so the closure type-checks.
                raise RuntimeError("semantic backend unavailable mid-pair")
            try:
                # Once a pair starts, only the inference hard timeout may stop
                # it. The job budget is a fairness gate between pairs; the
                # retry gate (A6) is the same fairness idea one level down:
                # with another job queued, a protocol-invalid output fails
                # fast instead of doubling its own latency.
                return backend.classify_pair(
                    left_env, right_env, deadline_monotonic=None,
                    retry_allowed=not self._semantic_worker.has_pending_jobs(),
                )
            except TypeError:
                # Test/legacy backends implementing the original two-arg protocol.
                return backend.classify_pair(left_env, right_env)

        # 0.16.2 unified flow: internal check-keepers go through the SAME Qwen
        # slot extraction as cross-memory pairs, BEFORE the cross loop (E10①
        # order guarantee: internal findings land first and survive a
        # truncated cross loop). Shared pairs_examined budget and deadline —
        # internal keepers are naturally few (filter-calibrated). Verdicts:
        #   notice_ready  → land pending, reason annotated with the extracted
        #                    attribute/values
        #   definitive negative (not the same attribute with different
        #   values, unknown_field) → land DISMISSED — the veto must outlive
        #   the write or the scan-side re-examination would resurrect the pair
        #   (internal_conflicts.exists() blocks any-status-non-stale rows)
        #   technical failure / no backend → land pending unannotated
        #   (fail-open: an advisory rule signal must not be lost to an
        #   unavailable model)
        internal_qwen_confirmed = 0
        internal_qwen_vetoed = 0
        for unit_a, unit_b, internal_decision in internal_qwen_pairs:
            reason_text = str(internal_decision.reason or "")
            if backend is not None:
                active_deadline = backlog_deadline()
                budget_ok = not (
                    active_deadline is not None
                    and active_deadline - time.monotonic() < min_budget * 2
                ) and pairs_examined < max_examined_pairs
                if budget_ok:
                    pairs_examined += 1
                    env_a = envelope(record, unit_a.text)
                    env_b = envelope(record, unit_b.text)
                    if internal_decision.left_value and internal_decision.right_value:
                        env_a["rule_value"] = internal_decision.left_value
                        env_b["rule_value"] = internal_decision.right_value
                    # Single-direction judging (owner 2026-09-17): all three
                    # paths share the one-forward-extraction gate — the
                    # bidirectional mirror was falsified on the eval line.
                    forward = classify(env_a, env_b)
                    gate = evaluate_single_direction_extraction(
                        signal_extraction(forward), env_a, env_b,
                    )
                    if gate.state == "notice_ready":
                        reason_text = (
                            f"{reason_text} | qwen:{gate.attribute}="
                            f"{gate.value_a}|{gate.value_b}"
                        )
                        internal_qwen_confirmed += 1
                    else:
                        technical = (
                            (forward.error and "timeout" in str(forward.error).lower())
                            or forward.candidate_type == "backend_unavailable"
                            or forward.candidate_type == "backend_error"
                            or forward.candidate_type in {"invalid_json", "invalid_schema"}
                        )
                        if not technical and gate.reason != "qwen_unverified":
                            internal_qwen_vetoed += 1
                            self.db.internal_conflicts.create(
                                memory_id=int(memory_id), memory_version=internal_version,
                                unit_a=unit_a.unit_index, unit_b=unit_b.unit_index,
                                quote_a=unit_a.text, quote_b=unit_b.text,
                                span_a=[unit_a.start_offset, unit_a.end_offset],
                                span_b=[unit_b.start_offset, unit_b.end_offset],
                                reason=reason_text, detector_version=CONFLICT_DETECTOR_VERSION,
                                status="dismissed",
                                decided_reason=f"qwen veto: {gate.reason}",
                            )
                            continue
            if self.db.internal_conflicts.create(
                memory_id=int(memory_id), memory_version=internal_version,
                unit_a=unit_a.unit_index, unit_b=unit_b.unit_index,
                quote_a=unit_a.text, quote_b=unit_b.text,
                span_a=[unit_a.start_offset, unit_a.end_offset],
                span_b=[unit_b.start_offset, unit_b.end_offset],
                reason=reason_text, detector_version=CONFLICT_DETECTOR_VERSION,
            ):
                internal_found += 1

        # P2-T6: batch-prefetch every candidate peer in ONE id-IN query
        # instead of one get_memory connection per pair.
        peer_rows = self.db.get_memories_by_ids(
            [int(pid) for pid, _triple in ordered], conn=job_conn,
        )
        for peer_id, (hit, unit, decision) in ordered:
            peer = peer_rows.get(int(peer_id))
            if not peer or peer.get("status") != "active":
                reached_pair.add(peer_id)  # settled (inactive) — not backlog
                continue
            record_row: dict[str, Any] = record or {}
            peer_row: dict[str, Any] = peer or {}
            left_version = int(record.get("version") or 1)
            right_version = int(peer.get("version") or 1)
            if self.db.semantic_notices.is_semantic_pair_closed_on_conn(
                job_conn, memory_id, peer_id, left_version, right_version,
            ):
                reached_pair.add(peer_id)  # settled (closed) — not backlog
                continue
            # NOT reached yet: budget/cap skips below leave the pair
            # unmarked so the post-loop backlog sweep picks it up.
            # Deterministic direct path (2026-09-16, owner-approved): same
            # value-stripped key + canonical value difference IS the
            # same-attribute-different-value shape — land the notice without
            # spending Qwen, whose budget is reserved for pairs only
            # judgment can settle. Runs BEFORE the backend/budget checks:
            # a direct pair consumes no Qwen budget and works even while the
            # backend is unavailable.
            direct = direct_value_verdict(
                unit.text, str(hit.get("text") or ""), decision, embedder=embedder,
            )
            if direct is not None:
                reached_pair.add(peer_id)  # deterministic verdict — settled
            if direct is not None:
                gate = PairGateResult(
                    "notice_ready", "deterministic_same_key_value_diff",
                    direct[0], direct[1], direct[2], True,
                )
                qwen = {
                    "status": "bypassed", "reason": gate.reason,
                    "forward_type": "deterministic", "reverse_type": "deterministic",
                }
                forward_signal = None
            else:
                if backend is None:
                    record_degradation("qwen_unavailable")
                    incomplete_reason = "qwen_unavailable"
                    continue
                active_deadline = backlog_deadline()
                if active_deadline is not None and active_deadline - time.monotonic() < min_budget * 2:
                    record_degradation("qwen_budget_exhausted")
                    incomplete_reason = "qwen_budget_exhausted"
                    continue
                # A5 deterministic cap: only pairs that actually reach Qwen count;
                # skipped (closed/inactive) pairs never consume the budget.
                if pairs_examined >= max_examined_pairs:
                    record_degradation("pairs_examined_capped")
                    incomplete_reason = "pairs_examined_capped"
                    break
                pairs_examined += 1
                reached_pair.add(peer_id)  # Qwen examined — settled
                left_env = envelope(record_row, unit.text)
                right_env = envelope(peer_row, str(hit.get("text") or ""))
                # pair-v7: hand Qwen the rule layer's extracted value difference
                # (numeric check route only) as a locating hint — see _pair_text.
                if decision.left_value and decision.right_value:
                    left_env["rule_value"] = decision.left_value
                    right_env["rule_value"] = decision.right_value
                started = time.monotonic()
                # Single-direction gate (owner 2026-09-17): the reverse
                # extraction was the side-attribution hedge the 0.5B needed;
                # Qwen3-0.6B doesn't commit that error and the bidirectional
                # cross-mapping kept killing real conflicts at reverse
                # attribute drift (eval: 9/12 bidirectional vs 11/12 single,
                # hard false positives 0/43 on the product chain). One clean
                # extraction + grounding + veto lands the notice; scan and
                # internal-conflict paths keep the bidirectional gate.
                forward_signal = classify(left_env, right_env)
                reverse_signal = None
                self._tools._record_pair_sample(
                    pair_ms=int((time.monotonic() - started) * 1000),
                    forward=forward_signal,
                    reverse=None,
                )
                gate = evaluate_single_direction_extraction(
                    signal_extraction(forward_signal), left_env, right_env,
                )
                qwen = {
                    "status": gate.state, "reason": gate.reason,
                    "forward_type": forward_signal.candidate_type,
                    "reverse_type": "single_direction",
                }
            if gate.state != "notice_ready":
                signals = tuple(
                    signal for signal in (forward_signal, reverse_signal) if signal is not None
                )
                if any(signal.error and "timeout" in str(signal.error).lower() for signal in signals):
                    reason = "qwen_timeout"
                elif any(signal.candidate_type == "backend_unavailable" for signal in signals):
                    reason = "qwen_unavailable"
                elif any(signal.candidate_type == "backend_error" for signal in signals):
                    reason = "qwen_backend_error"
                elif any(signal.candidate_type in {"invalid_json", "invalid_schema"} for signal in signals):
                    reason = "qwen_invalid_output"
                elif any(signal.candidate_type == "unknown_field" for signal in signals):
                    # The model explicitly reported an unextractable field: a
                    # completed negative decision (fail-closed for notice), not
                    # a technical failure (spec §8 diagnostics distinction).
                    dropped_unlocalizable += 1
                    continue
                else:
                    reason = gate.reason
                if reason == "qwen_unverified":
                    # Grounding failed: uncertain — fail-closed for notices and
                    # the pair remains a scan review candidate. Owner ruling
                    # #9: unlocalizable candidates are DROPPED, counted loud.
                    record_degradation(reason)
                    dropped_unlocalizable += 1
                    incomplete_reason = reason
                    continue
                if reason in _TECHNICAL_REASONS:
                    sample = None
                    if reason == "qwen_invalid_output":
                        # Preserve the offending raw output so the failure mode
                        # (truncation / prose / malformed) is visible in status.
                        sample = next(
                            (
                                str(signal.raw) for signal in signals
                                if signal.candidate_type in {"invalid_json", "invalid_schema"} and signal.raw
                            ),
                            None,
                        )
                    record_degradation(reason, sample)
                    incomplete_reason = reason
                    continue
                # Definitive strict-gate negatives (not_same_attribute_different_value,
                # coexist_*, direction_invalid, bidirectional_*): the pair was
                # examined and decided. The check stays complete and no
                # degradation counter fires (spec §9/§15.5/§8).
                continue
            raw_meta = record_row.get("metadata")
            metadata = raw_meta if isinstance(raw_meta, dict) else {}
            raw_peer_meta = peer_row.get("metadata")
            peer_metadata = raw_peer_meta if isinstance(raw_peer_meta, dict) else {}
            entity = metadata.get("entity") if metadata.get("entity") == peer_metadata.get("entity") else None
            scope = metadata.get("scope") if metadata.get("scope") == peer_metadata.get("scope") else None
            if not entity or not scope:
                incomplete_reason = "slot_provenance_insufficient"
                continue
            # B-C4: slot keys are built with canonicalised entity/scope (the
            # comparison-side counterpart of the storage-side canon in
            # db/conflicts.py _normalize_slot) so lexical variants like
            # "MyProject"/"myproject" address the same slot.
            slot_key = {
                "entity": canon_entity(entity), "attribute": gate.attribute,
                "scope": canon_scope(scope),
            }
            slot_json = json.dumps(slot_key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            raw_slot_json = json.dumps(
                {"entity": entity, "attribute": gate.attribute, "scope": scope},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            if slot_json in applying_slots or raw_slot_json in applying_slots:
                # Suppression matches either form: conflict groups stored before
                # storage-side canonicalisation may still carry the raw
                # (unnormalised) slot_key. A third-party fact landing on a slot
                # currently under application: scan review only (spec §15.3),
                # no new notice.
                continue
            member_versions = [
                {"memory_id": memory_id, "version": left_version, "value": gate.value_a,
                 "evidence": {"quote": unit.text, "start": unit.start_offset, "end": unit.end_offset}},
                {"memory_id": peer_id, "version": right_version, "value": gate.value_b,
                 "evidence": {"quote": hit.get("text"), "start": hit.get("start_offset"), "end": hit.get("end_offset")}},
            ]
            # Model output may omit keys (or parsed may not be a dict at all):
            # fall back to the gate's normalised value instead of raising
            # KeyError (mirrors the scan-path defence in tools.py).
            forward_parsed = (
                forward_signal.parsed
                if forward_signal is not None and isinstance(forward_signal.parsed, dict)
                else {}
            )
            value_groups = [
                {"normalized_value": gate.value_a, "display_value": forward_parsed.get("value_a") or gate.value_a,
                 "members": [f"{memory_id}@{left_version}"]},
                {"normalized_value": gate.value_b, "display_value": forward_parsed.get("value_b") or gate.value_b,
                 "members": [f"{peer_id}@{right_version}"]},
            ]
            outcome = self.db.record_semantic_notice(
                memory_id=memory_id, peer_id=peer_id,
                # 0.16.4 §1: cross-memory notify is excluded at collection,
                # so the write-time notice severity is uniformly normal.
                severity="normal",
                notice_type="semantic_evidence",
                title=f"Possible memory change with #{peer_id}", message=decision.reason,
                payload={
                    "route": "notice_ready", "reason": gate.reason,
                    "prompt_version": PAIR_PROMPT_VERSION,
                    "anchors": decision.anchors,
                    "slot_key": slot_key,
                    "slot_provenance": {
                        "entity": "metadata", "scope": "metadata",
                        "attribute": (
                            "deterministic_skeleton" if direct is not None
                            else "single_direction_extraction"
                        ),
                    },
                    "member_versions": member_versions,
                    "value_groups": value_groups,
                    "candidate_key": {
                        "detector_version": CONFLICT_DETECTOR_VERSION,
                        "members": sorted([f"{memory_id}@{left_version}", f"{peer_id}@{right_version}"]),
                        "evidence": [],
                    },
                    "left_evidence": {
                        "text": unit.text, "start_offset": unit.start_offset,
                        "end_offset": unit.end_offset,
                    },
                    "right_evidence": {
                        "text": hit.get("text"), "start_offset": hit.get("start_offset"),
                        "end_offset": hit.get("end_offset"),
                    },
                    "left_content_hash": evidence_content_hash(content),
                    "right_content_hash": evidence_content_hash(str(peer.get("content") or "")),
                    "qwen_signal": qwen,
                },
                dedupe_key=notice_dedupe_key(
                    memory_id, peer_id, left_version, right_version, "semantic_evidence",
                ),
                left_version=left_version, right_version=right_version,
                source="semantic_evidence",
            )
            # surfaced counts notices actually created (deduped pairs were
            # already surfaced) — since A5 it is a result summary, not a gate.
            if outcome.get("outcome") == "created":
                surfaced += 1
            elif outcome.get("outcome") not in {"deduped"}:
                # Second-round review: a ready pair whose notice could not be
                # persisted (workspace_mismatch / invalid_snapshot /
                # unavailable / error) must not vanish silently — without this
                # the run could report checked_no_notice while a real conflict
                # was found and lost.
                record_degradation("notice_write_failed")
                incomplete_reason = "notice_write_failed"
        # 0.17.0 P2-4.2: budget/cap leftovers land in the backlog — bounded,
        # visible, never silently dropped (owner design #8). Stale/duplicate
        # keys report as enqueued here; eviction counts ride the store.
        leftover_entries = [item for item in ordered if item[0] not in reached_pair]
        if leftover_entries:
            backlogged = _enqueue_backlog(leftover_entries)
        if surfaced:
            result: dict[str, Any] = {
                "status": "completed", "outcome": "notices_created", "notices_created": surfaced,
            }
            if internal_found:
                result["internal_conflicts"] = internal_found
            if incomplete_reason:
                # Notices went out, but later pairs hit a truncation/degradation
                # — surface it instead of a bare completed (second-round
                # review): the caller would otherwise read a bounded, partial
                # check as a full one.
                result["truncated"] = True
        elif incomplete_reason:
            result = {"status": "incomplete", "reason": incomplete_reason, "notices_created": 0}
        else:
            result = {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0}
        if internal_found and "internal_conflicts" not in result:
            # Internal findings survive a truncated cross-memory loop: they
            # were landed BEFORE the loop ran (E10① order guarantee).
            result["internal_conflicts"] = internal_found
        # 0.16.2 write-time pre-gate visibility (conditional — the unfiltered
        # zero case keeps the exact-shape response contract unchanged):
        # what the deterministic filters killed this run, and what the
        # unified internal Qwen flow confirmed/vetoed.
        filter_summary: dict[str, int] = {}
        if provenance_filtered:
            filter_summary["provenance_skipped"] = provenance_filtered
        if no_difference_filtered:
            filter_summary["no_difference_skipped"] = no_difference_filtered
        if internal_qwen_confirmed:
            filter_summary["internal_qwen_confirmed"] = internal_qwen_confirmed
        if internal_qwen_vetoed:
            filter_summary["internal_qwen_vetoed"] = internal_qwen_vetoed
        if dropped_unlocalizable:
            # 0.17.0 P2-3.5 (owner ruling #9): dropped unlocalizable pairs are
            # never silent — the counter rides every completed receipt.
            filter_summary["dropped_unlocalizable"] = dropped_unlocalizable
        if filter_summary:
            result["deterministic_filter"] = filter_summary
        if backlogged:
            # 0.17.0 P2-4.2: truncation leftovers went to the conflict
            # backlog instead of vanishing.
            result["backlogged"] = backlogged
        if rows_mode:
            result["rows_mode"] = True
            result["rows_examined"] = int(units_examined)
        if reasons_seen:
            # Degradations may also occur on pairs before a later pair surfaces
            # a notice, so the list is attached to completed outcomes too.
            result["reasons_seen"] = reasons_seen
        # 0.16.12 perf baseline: the examination budget actually consumed this
        # run (additive receipt key; also visible on completed write receipts).
        result["pairs_examined"] = int(pairs_examined)
        return result
