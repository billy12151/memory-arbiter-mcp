"""Persistence and KNN operations for vNext local-text evidence."""
from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
import struct
from typing import Any, TYPE_CHECKING

from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..acl import WorkspaceScope, scope_names, workspace_scope_sql
from ..evidence import EvidenceUnit, has_indexable_text, INDEXABLE_PREFILTER_SQL
from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


def indexable_coverage_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Single eligibility definition shared by coverage, doctor, and the
    vNext migration gate: a memory is eligible when the indexer would
    actually publish units for it. Zero-indexable-text rows (blank /
    whitespace-only legacy artifacts) are counted as non_indexable instead
    of staying a permanent coverage gap."""
    total = int(
        conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status!='deleted'"
        ).fetchone()[0]
    )
    covered = int(
        conn.execute(
            """SELECT COUNT(DISTINCT m.id) FROM memories m
               WHERE m.status!='deleted'
                 AND EXISTS(SELECT 1 FROM memory_evidence e WHERE e.memory_id=m.id)"""
        ).fetchone()[0]
    )
    uncovered = 0
    for row in conn.execute(
        f"""SELECT m.subject AS subject, m.content AS content FROM memories m
            WHERE m.status!='deleted' AND {INDEXABLE_PREFILTER_SQL}
              AND NOT EXISTS(SELECT 1 FROM memory_evidence e WHERE e.memory_id=m.id)"""
    ):
        if has_indexable_text(str(row["subject"] or ""), str(row["content"] or "")):
            uncovered += 1
    return {
        "total_memories": total,
        "eligible_memories": covered + uncovered,
        "indexed_memories": covered,
        "non_indexable_memories": total - covered - uncovered,
    }


class EvidenceStore:
    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    def publish(
        self,
        memory_id: int,
        memory_version: int,
        content_hash: str,
        units: list[EvidenceUnit],
        embeddings: list[list[float]],
    ) -> dict[str, Any]:
        if len(units) != len(embeddings) or any(not value for value in embeddings):
            return {"outcome": "invalid_embeddings", "published": False}
        try:
            with self._db.write_transaction() as conn:
                current = conn.execute(
                    "SELECT version, content FROM memories WHERE id=?", (int(memory_id),)
                ).fetchone()
                if current is None or int(current["version"] or 1) != int(memory_version):
                    return {"outcome": "stale_snapshot", "published": False}
                from ..evidence import evidence_content_hash
                if evidence_content_hash(str(current["content"] or "")) != content_hash:
                    return {"outcome": "stale_snapshot", "published": False}

                old_ids = [
                    int(row["id"]) for row in conn.execute(
                        "SELECT id FROM memory_evidence WHERE memory_id=?", (int(memory_id),)
                    ).fetchall()
                ]
                if old_ids:
                    placeholders = ",".join("?" for _ in old_ids)
                    conn.execute(f"DELETE FROM memory_evidence_vec WHERE id IN ({placeholders})", old_ids)
                conn.execute("DELETE FROM memory_evidence WHERE memory_id=?", (int(memory_id),))

                status_row = conn.execute("SELECT status FROM memories WHERE id=?", (int(memory_id),)).fetchone()
                parent_status = str(status_row["status"] if status_row else "deleted")
                for unit, embedding in zip(units, embeddings):
                    cur = conn.execute(
                        """INSERT INTO memory_evidence(
                             memory_id,memory_version,content_hash,unit_index,kind,text,
                             start_offset,end_offset,created_at
                           ) VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            int(memory_id), int(memory_version), content_hash,
                            int(unit.unit_index), unit.kind, unit.text,
                            int(unit.start_offset), int(unit.end_offset), utc_now_iso(),
                        ),
                    )
                    if cur.lastrowid is None:
                        raise sqlite3.Error("evidence insert did not return an id")
                    conn.execute(
                        "INSERT INTO memory_evidence_vec(id,parent_status,embedding) VALUES (?,?,?)",
                        (int(cur.lastrowid), parent_status, json.dumps(embedding)),
                    )
            return {"outcome": "published", "published": True, "unit_count": len(units)}
        except sqlite3.Error as exc:
            return {"outcome": "error", "published": False, "error": str(exc)}

    def delete_for_memory(self, memory_id: int) -> None:
        with self._db.write_transaction() as conn:
            ids = [int(row["id"]) for row in conn.execute(
                "SELECT id FROM memory_evidence WHERE memory_id=?", (int(memory_id),)
            )]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(f"DELETE FROM memory_evidence_vec WHERE id IN ({placeholders})", ids)
            conn.execute("DELETE FROM memory_evidence WHERE memory_id=?", (int(memory_id),))

    def coverage(self) -> dict[str, int]:
        with self._db.connection() as conn:
            counts = indexable_coverage_counts(conn)
            units = int(conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0])
            try:
                vectors = int(conn.execute("SELECT COUNT(*) FROM memory_evidence_vec").fetchone()[0])
            except sqlite3.Error:
                vectors = 0
        return {
            "eligible_memories": counts["eligible_memories"],
            "non_indexable_memories": counts["non_indexable_memories"],
            "indexed_memories": counts["indexed_memories"],
            "units": units,
            "vectors": vectors,
        }

    def current_text_vectors(
        self, memory_id: int, memory_version: int, content_hash: str,
    ) -> list[tuple[EvidenceUnit, list[float]]]:
        """Return the exact current text units and their already-published vectors."""
        if not self._db.state.sqlite_vec_available:
            return []
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    """SELECT e.unit_index,e.kind,e.text,e.start_offset,e.end_offset,
                              v.embedding
                       FROM memory_evidence e
                       JOIN memory_evidence_vec v ON v.id=e.id
                       WHERE e.memory_id=? AND e.memory_version=? AND e.content_hash=?
                         AND e.kind='text'
                       ORDER BY e.unit_index""",
                    (int(memory_id), int(memory_version), str(content_hash)),
                ).fetchall()
            return [
                (
                    EvidenceUnit(
                        kind=str(row["kind"]), text=str(row["text"]),
                        start_offset=int(row["start_offset"]),
                        end_offset=int(row["end_offset"]),
                        unit_index=int(row["unit_index"]),
                    ),
                    self._blob_to_vector(bytes(row["embedding"])),
                )
                for row in rows
                if row["embedding"] is not None
            ]
        except (sqlite3.Error, TypeError, ValueError, struct.error):
            return []

    def knn(
        self,
        query_embedding: list[float],
        *,
        k: int = 100,
        parent_status_filter: str = "active",
        workspace: "WorkspaceScope" = None,
        exclude_memory_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """KNN over evidence vectors, filtered by parent lifecycle state.

        sqlite-vec pushes only the vec0 auxiliary column (``parent_status``)
        into the KNN itself; the workspace and exclude predicates filter
        joined tables *after* the global top-k. To keep per-workspace recall
        from collapsing when other workspaces own the globally nearest rows,
        callers with those filters grow the global KNN window until enough
        scoped rows are found or the complete lifecycle-eligible vector domain
        has been considered. A true pre-filter would need a workspace partition
        key on the vec0 table (deferred, spec §19).

        ``workspace`` may be an admitted canonical set (one name or the
        strict in-radius neighbourhood); the same over-fetch loop compensates
        for the post-KNN membership filter.
        """
        if not self._db.state.sqlite_vec_available or not query_embedding:
            return []
        if parent_status_filter == "expired":
            status_sql = "v.parent_status NOT IN ('active','deleted')"
            memory_status_sql = "m.status NOT IN ('active','deleted')"
        elif parent_status_filter == "all":
            status_sql = "v.parent_status != 'deleted'"
            memory_status_sql = "m.status != 'deleted'"
        else:
            status_sql = "v.parent_status='active'"
            memory_status_sql = "m.status='active'"
        # parent_status is best-effort (a vec-disabled process can skip its
        # update), so the authoritative memories.status is enforced too.
        clauses = [status_sql, memory_status_sql]
        requested_k = max(1, int(k))
        workspace_sql, workspace_params = workspace_scope_sql(
            "COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)", workspace,
        )
        filtered = bool(workspace_sql or exclude_memory_id is not None)
        filter_params: list[Any] = []
        if workspace_sql:
            clauses.append(workspace_sql)
            filter_params.extend(workspace_params)
        if exclude_memory_id is not None:
            clauses.append("e.memory_id!=?")
            filter_params.append(int(exclude_memory_id))
        try:
            with self._db.connection() as conn:
                # Workspace/exclusion predicates are evaluated after vec0 picks
                # its global KNN window. Grow that window until recall is
                # satisfied or every bounded candidate has been considered.
                candidate_count = int(
                    conn.execute(
                        f"""SELECT COUNT(*) FROM memory_evidence_vec v
                            JOIN memory_evidence e ON e.id=v.id
                            JOIN memories m ON m.id=e.memory_id
                            WHERE {status_sql} AND {memory_status_sql}"""
                    ).fetchone()[0]
                )
                # Grow to the complete candidate domain if necessary. A fixed
                # global cap can permanently starve a sparse admitted workspace
                # when closer rows belong to other workspaces. The loop still
                # stops as soon as requested_k scoped rows are found.
                max_fetch = max(1, candidate_count)
                fetch_k = min(max_fetch, requested_k * 4) if filtered else requested_k
                rows: list[Any] = []
                while fetch_k > 0:
                    params: list[Any] = [json.dumps(query_embedding), fetch_k, *filter_params]
                    rows = conn.execute(
                        f"""SELECT e.*, v.distance AS distance, m.status, m.subject, m.tags,
                                   m.workspace, m.workspace_canonical, m.source_type,
                                   m.confidence, m.protection_level, m.event_time,
                                   m.ingest_time, m.metadata, m.content,
                                   m.version AS memory_row_version, m.agent_id,
                                   m.source_ref, m.created_at AS memory_created_at
                            FROM memory_evidence_vec v
                            JOIN memory_evidence e ON e.id=v.id
                            JOIN memories m ON m.id=e.memory_id
                            WHERE v.embedding MATCH ? AND k=? AND {' AND '.join(clauses)}
                            ORDER BY v.distance""",
                        params,
                    ).fetchall()
                    if not filtered or len(rows) >= requested_k or fetch_k >= max_fetch:
                        break
                    fetch_k = min(max_fetch, fetch_k * 2)
            return [dict(row) for row in rows[:requested_k]]
        except sqlite3.Error:
            return []

    @staticmethod
    def _unit_pair_identity(
        anchor_id: int, unit: Any, peer_id: int, hit: dict[str, Any],
    ) -> tuple[frozenset[str], dict[str, Any], str]:
        """Build the record_conflict-compatible identity for one unit pair.

        Single source of truth for the candidate_key hash: the real-candidate
        path and the duplicates_pool path must derive suppression lookups from
        byte-identical keys or already-recorded pairs would re-surface.
        """
        anchor_ref = f"{anchor_id}@{int(unit['memory_version'] or 1)}"
        peer_ref = f"{peer_id}@{int(hit.get('memory_version') or hit.get('memory_row_version') or 1)}"
        member_refs = frozenset([anchor_ref, peer_ref])
        evidence_by_ref = {
            anchor_ref: {
                "member": anchor_ref,
                "unit": int(unit["eid"]),
                "span": [int(unit["start_offset"] or 0), int(unit["end_offset"] or 0)],
                "hash": str(unit["content_hash"] or ""),
            },
            peer_ref: {
                "member": peer_ref,
                "unit": int(hit.get("id") or 0),
                "span": [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)],
                "hash": str(hit.get("content_hash") or ""),
            },
        }
        sorted_refs = sorted(member_refs, key=lambda ref: tuple(int(value) for value in ref.split("@", 1)))
        candidate_key = {
            "detector_version": CONFLICT_DETECTOR_VERSION,
            "members": sorted_refs,
            "evidence": [evidence_by_ref[ref] for ref in sorted_refs],
        }
        candidate_hash = hashlib.sha256(
            json.dumps(
                candidate_key, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return member_refs, candidate_key, candidate_hash

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

        Calibrated on a real 474-memory production copy: absolute vector
        distance has no discrimination there (random same-workspace pairs
        overlap notice pairs), so ranking + rules do the work and
        max_distance stays an optional extra gate.
        """
        from ..semantic_conflict import decide_evidence

        db = self._db
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
                }
            # The group schema has no left/right columns. Suppression is tied
            # to the exact candidate snapshot successfully persisted by
            # record_conflict, not merely to a memory pair. That keeps an
            # unrecorded external review repeatable and allows changed member
            # versions/evidence to be reconsidered.
            recorded_candidate_statuses: dict[str, str] = {}
            active_group_members: list[frozenset[str]] = []
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
                if refs and status in {"open", "applying"}:
                    active_group_members.append(refs)
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
            for anchor_id in anchors:
                # Only body-text units participate, matching the write-time
                # notice path: subjects/headings are version-progression
                # heavy and fire numeric_value_changed on their own.
                units = conn.execute(
                    """SELECT e.id AS eid, e.text AS text, v.embedding AS embedding,
                              e.memory_version AS memory_version, e.content_hash AS content_hash,
                              e.start_offset AS start_offset, e.end_offset AS end_offset
                       FROM memory_evidence e
                       JOIN memory_evidence_vec v ON v.id=e.id
                       WHERE e.memory_id=? AND e.memory_version=(
                           SELECT version FROM memories WHERE id=?)
                         AND e.kind='text'
                       ORDER BY e.id""",
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
                    from ..evidence import _clean

                    start, end = int(hint_start), int(hint_end)
                    if not (content and unit_text and 0 <= start < end <= len(content)):
                        return None
                    if _clean(content[start:end]) != unit_text:
                        return None
                    return {
                        "start": max(0, start - 128),
                        "end": min(len(content), end + 128),
                    }

                for unit in (() if first_unit is None else itertools.chain((first_unit,), units)):
                    text = str(unit["text"] or "")
                    if not text:
                        continue
                    hits = self.knn(
                        self._blob_to_vector(bytes(unit["embedding"])),
                        k=max(1, int(neighbor_k)) + 1,
                        workspace=workspace,
                        exclude_memory_id=anchor_id,
                    )
                    for hit in hits:
                        if hit.get("kind") != "text":
                            continue
                        peer_id = int(hit["memory_id"])
                        if peer_id == anchor_id:
                            continue
                        knn_pair_count += 1
                        # Every unit pair is judged: an earlier equivalent
                        # match (e.g. identical subjects) must not blacklist
                        # the peer, or a later numeric-change unit on the
                        # same pair would be lost.
                        decision = decide_evidence(text, str(hit.get("text") or ""))
                        if decision.action == "ignore":
                            if include_duplicates and decision.reason in {"equivalent_value", "compatible_evidence"}:
                                pair_key = (min(anchor_id, peer_id), max(anchor_id, peer_id))
                                member_refs, _key, candidate_hash = self._unit_pair_identity(
                                    anchor_id, unit, peer_id, hit,
                                )
                                recorded = recorded_candidate_statuses.get(candidate_hash)
                                if recorded is None and not any(
                                    member_refs <= group_members for group_members in active_group_members
                                ):
                                    hit_text_ignored = str(hit.get("text") or "")
                                    # Re-hitting an already-pooled pair is a
                                    # free dict replace, not pool growth — it
                                    # must not count against the cap or flag
                                    # truncation that never happened.
                                    if pair_key in duplicates_pool or len(duplicates_pool) < duplicates_cap:
                                        # Members mirror the record_conflict
                                        # contract so an agent can dismiss a
                                        # false positive without re-deriving
                                        # the evidence identity.
                                        duplicates_pool[pair_key] = {
                                            "left_id": pair_key[0], "right_id": pair_key[1],
                                            "reason": decision.reason,
                                            "distance": float(hit.get("distance") or 0),
                                            "candidate_key_hash": candidate_hash,
                                            "left_snippet": text[:200] if pair_key[0] == anchor_id else hit_text_ignored[:200],
                                            "right_snippet": hit_text_ignored[:200] if pair_key[1] == peer_id else text[:200],
                                            "members": [
                                                {
                                                    "memory_id": pair_key[0],
                                                    "version": int(unit["memory_version"] or 1) if pair_key[0] == anchor_id else int(hit.get("memory_version") or hit.get("memory_row_version") or 1),
                                                    "attribute_raw": None, "value_raw": None,
                                                    "normalized_attribute": None, "normalized_value": None,
                                                    "evidence_quote": text if pair_key[0] == anchor_id else hit_text_ignored,
                                                    "evidence_span": (
                                                        [int(unit["start_offset"] or 0), int(unit["end_offset"] or 0)]
                                                        if pair_key[0] == anchor_id else
                                                        [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)]
                                                    ),
                                                    "content_hash": str(unit["content_hash"] or "") if pair_key[0] == anchor_id else str(hit.get("content_hash") or ""),
                                                    "evidence_unit": int(unit["eid"]) if pair_key[0] == anchor_id else int(hit.get("id") or 0),
                                                    "direction": "deterministic", "prompt_version": None,
                                                    "detector_version": CONFLICT_DETECTOR_VERSION,
                                                },
                                                {
                                                    "memory_id": pair_key[1],
                                                    "version": int(hit.get("memory_version") or hit.get("memory_row_version") or 1) if pair_key[1] == peer_id else int(unit["memory_version"] or 1),
                                                    "attribute_raw": None, "value_raw": None,
                                                    "normalized_attribute": None, "normalized_value": None,
                                                    "evidence_quote": hit_text_ignored if pair_key[1] == peer_id else text,
                                                    "evidence_span": (
                                                        [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)]
                                                        if pair_key[1] == peer_id else
                                                        [int(unit["start_offset"] or 0), int(unit["end_offset"] or 0)]
                                                    ),
                                                    "content_hash": str(hit.get("content_hash") or "") if pair_key[1] == peer_id else str(unit["content_hash"] or ""),
                                                    "evidence_unit": int(hit.get("id") or 0) if pair_key[1] == peer_id else int(unit["eid"]),
                                                    "direction": "deterministic", "prompt_version": None,
                                                    "detector_version": CONFLICT_DETECTOR_VERSION,
                                                },
                                            ],
                                        }
                                    else:
                                        duplicates_truncated = True
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
                        member_refs, candidate_key, candidate_hash = self._unit_pair_identity(
                            anchor_id, unit, peer_id, hit,
                        )
                        recorded_status = recorded_candidate_statuses.get(candidate_hash)
                        if recorded_status is None and any(
                            member_refs <= group_members for group_members in active_group_members
                        ):
                            recorded_status = "open"
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
                                    {
                                        "memory_id": pair[0],
                                        "version": int(hit.get("memory_version") or hit.get("memory_row_version") or 1) if pair[0] == peer_id else int(unit["memory_version"] or 1),
                                        "attribute_raw": None, "value_raw": None,
                                        "normalized_attribute": None, "normalized_value": None,
                                        "evidence_quote": text if pair[0] == anchor_id else hit_text,
                                        "evidence_span": (
                                            [int(unit["start_offset"] or 0), int(unit["end_offset"] or 0)]
                                            if pair[0] == anchor_id else
                                            [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)]
                                        ),
                                        "content_hash": str(unit["content_hash"] or "") if pair[0] == anchor_id else str(hit.get("content_hash") or ""),
                                        "evidence_unit": int(unit["eid"]) if pair[0] == anchor_id else int(hit.get("id") or 0),
                                        "direction": "deterministic", "prompt_version": None,
                                        "detector_version": CONFLICT_DETECTOR_VERSION,
                                    },
                                    {
                                        "memory_id": pair[1],
                                        "version": int(hit.get("memory_version") or hit.get("memory_row_version") or 1) if pair[1] == peer_id else int(unit["memory_version"] or 1),
                                        "attribute_raw": None, "value_raw": None,
                                        "normalized_attribute": None, "normalized_value": None,
                                        "evidence_quote": hit_text if pair[1] == peer_id else text,
                                        "evidence_span": (
                                            [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)]
                                            if pair[1] == peer_id else
                                            [int(unit["start_offset"] or 0), int(unit["end_offset"] or 0)]
                                        ),
                                        "content_hash": str(hit.get("content_hash") or "") if pair[1] == peer_id else str(unit["content_hash"] or ""),
                                        "evidence_unit": int(hit.get("id") or 0) if pair[1] == peer_id else int(unit["eid"]),
                                        "direction": "deterministic", "prompt_version": None,
                                        "detector_version": CONFLICT_DETECTOR_VERSION,
                                    },
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
                            if not similarity_only and existing["state"] == "review_candidate" and decision.action == "notify":
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
            return {
                "anchors_scanned": len(anchors),
                "next_anchor_memory_id": int(next_anchor) if more else None,
                "candidates": ordered,
                "similarity_pool": similarity_ordered,
                "duplicates_pool": duplicates_ordered,
                "duplicates_truncated": duplicates_truncated,
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

    @staticmethod
    def _blob_to_vector(blob: bytes) -> list[float]:
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))
