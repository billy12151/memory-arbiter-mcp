"""Persistence operations for the v0.9 structured-claim derived index.

The store is deliberately composed into MemoryDB rather than owning a SQLite
connection.  MemoryDB remains the single connection/transaction authority;
this module owns only claim/entity persistence and collision lookup.
"""
from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any, Optional

from .claims import canon_scope, canon_token
from .models import utc_now_iso

if TYPE_CHECKING:
    from .db import MemoryDB


def is_protected_memory(memory: dict[str, Any]) -> bool:
    return (
        memory.get("protection_level") == "locked"
        or memory.get("source_type") == "user_confirmed"
    )


def _same_time_window(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Conservative contradiction/evolution split using a one-second window."""
    left_ts = left.get("event_time") or left.get("ingest_time")
    right_ts = right.get("event_time") or right.get("ingest_time")
    if not left_ts and not right_ts:
        return True
    if not left_ts or not right_ts:
        left_ts = left_ts or left.get("ingest_time")
        right_ts = right_ts or right.get("ingest_time")
    if not left_ts or not right_ts:
        return True
    return str(left_ts)[:19] == str(right_ts)[:19]


def _decode_conflict_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    raw_details = data.get("structured_details")
    if isinstance(raw_details, str):
        try:
            data["structured_details"] = json.loads(raw_details)
        except json.JSONDecodeError:
            pass
    return data


class StructuredClaimStore:
    """Claim/entity persistence backed by MemoryDB's transaction factory."""

    def __init__(self, db: "MemoryDB"):
        self._db = db

    def publish_memory_claims(
        self,
        memory_id: int,
        claims: list[dict[str, Any]],
        ambiguous_count: int = 0,
        expected_claim_revision: Optional[int] = None,
    ) -> dict[str, Any]:
        """Atomically replace one memory's derived claims at its current revision."""
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable", "memory_id": int(memory_id)}
        try:
            with db.write_transaction() as conn:
                memory = db._fetch_memory(conn, int(memory_id))
                if not memory:
                    return {"outcome": "not_found", "memory_id": int(memory_id)}
                revision = int(memory.get("claim_revision") or 1)
                if expected_claim_revision is not None and revision != int(expected_claim_revision):
                    return {
                        "outcome": "stale_snapshot", "memory_id": int(memory_id),
                        "expected_claim_revision": int(expected_claim_revision),
                        "current_claim_revision": revision,
                    }
                if memory.get("status") != "active":
                    conn.execute("DELETE FROM memory_claims WHERE memory_id=?", (int(memory_id),))
                    conn.execute(
                        "UPDATE memories SET claims_indexed_revision=?, "
                        "claims_reconciled_revision=?, claim_ambiguous_count=? WHERE id=?",
                        (revision, revision, int(ambiguous_count), int(memory_id)),
                    )
                    return {
                        "outcome": "skipped_inactive", "memory_id": int(memory_id),
                        "claim_revision": revision, "claim_count": 0,
                    }
                conn.execute("DELETE FROM memory_claims WHERE memory_id=?", (int(memory_id),))
                now = utc_now_iso()
                for claim in claims:
                    conn.execute(
                        """
                        INSERT INTO memory_claims(
                            memory_id, entity, attribute, scope, value, raw_value,
                            value_type, extractor_rule, evidence, start_offset,
                            end_offset, claim_revision, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(memory_id), str(claim["entity"]), str(claim["attribute"]),
                            str(claim.get("scope") or ""), str(claim["value"]),
                            claim.get("raw_value"), claim.get("value_type"),
                            claim.get("extractor_rule"), claim.get("evidence"),
                            claim.get("start_offset"), claim.get("end_offset"),
                            revision, now,
                        ),
                    )
                conn.execute(
                    "UPDATE memories SET claims_indexed_revision=?, "
                    "claims_reconciled_revision=NULL, claim_ambiguous_count=? WHERE id=?",
                    (revision, int(ambiguous_count), int(memory_id)),
                )
                return {
                    "outcome": "indexed", "memory_id": int(memory_id),
                    "claim_revision": revision, "claim_count": len(claims),
                    "ambiguous_key_count": int(ambiguous_count),
                }
        except (sqlite3.Error, KeyError, TypeError, ValueError) as exc:
            return {"outcome": "error", "memory_id": int(memory_id), "error": str(exc)}

    def mark_claim_index_failed(
        self, memory_id: int, expected_claim_revision: Optional[int] = None,
    ) -> None:
        """Make derived-index failure persistent and doctor-visible."""
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return
        try:
            with db.connection() as conn:
                if expected_claim_revision is None:
                    conn.execute(
                        "UPDATE memories SET claims_indexed_revision=NULL, "
                        "claims_reconciled_revision=NULL WHERE id=?",
                        (int(memory_id),),
                    )
                else:
                    conn.execute(
                        "UPDATE memories SET claims_indexed_revision=NULL, "
                        "claims_reconciled_revision=NULL "
                        "WHERE id=? AND claim_revision=?",
                        (int(memory_id), int(expected_claim_revision)),
                    )
                conn.commit()
        except sqlite3.Error:
            pass

    def mark_claim_reconciled(
        self,
        memory_id: int,
        expected_claim_revision: int,
        enrich_ms: float,
        candidate_count: int,
    ) -> dict[str, Any]:
        """Advance reconciliation only after every pair transition succeeds."""
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable", "memory_id": int(memory_id)}
        try:
            with db.write_transaction() as conn:
                cur = conn.execute(
                    "UPDATE memories SET claims_reconciled_revision=?, "
                    "structured_enrich_ms=?, structured_candidate_count=? "
                    "WHERE id=? AND status='active' AND claim_revision=? "
                    "AND claims_indexed_revision=?",
                    (
                        int(expected_claim_revision), float(enrich_ms),
                        int(candidate_count), int(memory_id),
                        int(expected_claim_revision), int(expected_claim_revision),
                    ),
                )
                if cur.rowcount == 0:
                    return {
                        "outcome": "stale_snapshot",
                        "memory_id": int(memory_id),
                        "expected_claim_revision": int(expected_claim_revision),
                    }
                return {
                    "outcome": "reconciled",
                    "memory_id": int(memory_id),
                    "claim_revision": int(expected_claim_revision),
                }
        except sqlite3.Error as exc:
            return {
                "outcome": "error", "memory_id": int(memory_id), "error": str(exc),
            }

    def list_memory_claims(
        self, memory_id: int, current_only: bool = True,
    ) -> list[dict[str, Any]]:
        db = self._db
        if not db._db_available:
            return []
        try:
            with db.connection() as conn:
                sql = (
                    "SELECT mc.* FROM memory_claims mc "
                    "JOIN memories m ON m.id=mc.memory_id WHERE mc.memory_id=?"
                )
                if current_only:
                    sql += (
                        " AND mc.claim_revision=m.claim_revision "
                        "AND m.claims_indexed_revision=m.claim_revision"
                    )
                sql += " ORDER BY mc.start_offset, mc.id"
                rows = conn.execute(sql, (int(memory_id),)).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def find_structured_claim_pairs(self, memory_id: int) -> dict[str, Any]:
        """Return current contradiction pairs for one memory, aggregated by peer."""
        db = self._db
        if not db._db_available:
            return {"pairs": [], "evolution_pairs": 0}
        try:
            with db.connection() as conn:
                target = db._fetch_memory(conn, int(memory_id))
                if not target or target.get("status") != "active":
                    return {"pairs": [], "evolution_pairs": 0}
                if target.get("claims_indexed_revision") != target.get("claim_revision"):
                    return {"pairs": [], "evolution_pairs": 0, "stale_index": True}
                rows = conn.execute(
                    """
                    SELECT
                      mine.entity, mine.attribute, mine.scope AS mine_scope,
                      mine.value AS mine_value, mine.raw_value AS mine_raw_value,
                      mine.evidence AS mine_evidence, mine.start_offset AS mine_start_offset,
                      mine.end_offset AS mine_end_offset, mine.extractor_rule,
                      peer.memory_id AS peer_id, peer.scope AS peer_scope,
                      peer.value AS peer_value, peer.raw_value AS peer_raw_value,
                      peer.evidence AS peer_evidence, peer.start_offset AS peer_start_offset,
                      peer.end_offset AS peer_end_offset,
                      pm.version AS peer_version, pm.claim_revision AS peer_claim_revision,
                      pm.source_type AS peer_source_type,
                      pm.protection_level AS peer_protection_level,
                      pm.event_time AS peer_event_time, pm.ingest_time AS peer_ingest_time
                    FROM memory_claims mine
                    JOIN memory_claims peer
                      ON peer.entity=mine.entity AND peer.attribute=mine.attribute
                     AND peer.memory_id<>mine.memory_id
                    JOIN memories pm ON pm.id=peer.memory_id
                    WHERE mine.memory_id=?
                      AND mine.claim_revision=?
                      AND mine.value<>peer.value
                      AND pm.status='active'
                      AND peer.claim_revision=pm.claim_revision
                      AND pm.claims_indexed_revision=pm.claim_revision
                    ORDER BY peer.memory_id, mine.attribute
                    """,
                    (int(memory_id), int(target.get("claim_revision") or 1)),
                ).fetchall()
        except sqlite3.Error:
            return {"pairs": [], "evolution_pairs": 0, "error": True}

        grouped: dict[int, dict[str, Any]] = {}
        evolution_peers: set[int] = set()
        for row in rows:
            mine_scope = str(row["mine_scope"] or "")
            peer_scope = str(row["peer_scope"] or "")
            if mine_scope and peer_scope and mine_scope != peer_scope:
                continue
            peer = {
                "id": int(row["peer_id"]),
                "version": int(row["peer_version"] or 1),
                "claim_revision": int(row["peer_claim_revision"] or 1),
                "source_type": row["peer_source_type"],
                "protection_level": row["peer_protection_level"],
                "event_time": row["peer_event_time"],
                "ingest_time": row["peer_ingest_time"],
            }
            protected = is_protected_memory(target) or is_protected_memory(peer)
            contradiction = protected or _same_time_window(target, peer)
            if not contradiction:
                evolution_peers.add(peer["id"])
                continue
            entry = grouped.setdefault(peer["id"], {"peer": peer, "claims": []})
            entry["claims"].append({
                "entity": row["entity"], "attribute": row["attribute"],
                "scope": mine_scope or peer_scope,
                "left_memory_id": int(memory_id), "left_value": row["mine_value"],
                "left_raw_value": row["mine_raw_value"], "left_evidence": row["mine_evidence"],
                "left_start_offset": row["mine_start_offset"], "left_end_offset": row["mine_end_offset"],
                "right_memory_id": peer["id"], "right_value": row["peer_value"],
                "right_raw_value": row["peer_raw_value"], "right_evidence": row["peer_evidence"],
                "right_start_offset": row["peer_start_offset"], "right_end_offset": row["peer_end_offset"],
                "extractor_rule": row["extractor_rule"],
                "protected_review": protected and not _same_time_window(target, peer),
            })
        pairs: list[dict[str, Any]] = []
        for peer_id, entry in grouped.items():
            a, b = sorted((int(memory_id), peer_id))
            claims = entry["claims"]
            if a != int(memory_id):
                for claim in claims:
                    for suffix in (
                        "value", "raw_value", "evidence", "start_offset",
                        "end_offset", "memory_id",
                    ):
                        left_key, right_key = f"left_{suffix}", f"right_{suffix}"
                        claim[left_key], claim[right_key] = claim[right_key], claim[left_key]
            pairs.append({
                "left_id": a, "right_id": b,
                "left_version": (
                    int(target.get("version") or 1)
                    if a == int(memory_id) else entry["peer"]["version"]
                ),
                "right_version": (
                    entry["peer"]["version"]
                    if b == peer_id else int(target.get("version") or 1)
                ),
                "left_claim_revision": (
                    int(target.get("claim_revision") or 1)
                    if a == int(memory_id) else entry["peer"]["claim_revision"]
                ),
                "right_claim_revision": (
                    entry["peer"]["claim_revision"]
                    if b == peer_id else int(target.get("claim_revision") or 1)
                ),
                "claims": claims,
            })
        return {"pairs": pairs, "evolution_pairs": len(evolution_peers)}

    def list_structured_open_conflicts_for_memory(
        self, memory_id: int,
    ) -> list[dict[str, Any]]:
        return self.read_structured_open_conflicts_for_memory(memory_id).get("rows", [])

    def read_structured_open_conflicts_for_memory(
        self, memory_id: int,
    ) -> dict[str, Any]:
        """Read open structured rows without collapsing read failure into empty."""
        db = self._db
        if not db._db_available:
            return {"rows": [], "error": "database unavailable"}
        try:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM conflicts WHERE status='open' "
                    "AND left_claim_revision IS NOT NULL "
                    "AND (left_id=? OR right_id=?)",
                    (int(memory_id), int(memory_id)),
                ).fetchall()
                return {"rows": [_decode_conflict_row(row) for row in rows]}
        except sqlite3.Error as exc:
            return {"rows": [], "error": str(exc)}

    def structured_pair_gate_states(
        self,
        memory_id: int,
        pairs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Batch dismissal/terminal-snapshot checks for one reconciliation pass."""
        if not pairs:
            return {"dismissed": set(), "closed": set()}
        db = self._db
        if not db._db_available:
            return {"dismissed": set(), "closed": set(), "error": True}
        snapshots = {
            (int(pair["left_id"]), int(pair["right_id"])): (
                int(pair["left_version"]), int(pair["right_version"]),
                int(pair["left_claim_revision"]), int(pair["right_claim_revision"]),
            )
            for pair in pairs
        }
        dismissed: set[tuple[int, int]] = set()
        closed: set[tuple[int, int]] = set()
        try:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT left_id, right_id, status, left_version, right_version, "
                    "left_claim_revision, right_claim_revision FROM conflicts "
                    "WHERE status IN ('resolved','not_a_conflict') "
                    "AND (left_id=? OR right_id=?)",
                    (int(memory_id), int(memory_id)),
                ).fetchall()
        except sqlite3.Error:
            return {"dismissed": set(), "closed": set(), "error": True}
        for row in rows:
            key = (int(row["left_id"]), int(row["right_id"]))
            snapshot = snapshots.get(key)
            if snapshot is None:
                continue
            lv, rv, lcr, rcr = snapshot
            exact = (
                row["left_version"] == lv
                and row["right_version"] == rv
                and row["left_claim_revision"] == lcr
                and row["right_claim_revision"] == rcr
            )
            if exact:
                closed.add(key)
            if row["status"] == "not_a_conflict" and (
                row["left_version"] in (None, lv)
                and row["right_version"] in (None, rv)
                and row["left_claim_revision"] in (None, lcr)
                and row["right_claim_revision"] in (None, rcr)
            ):
                dismissed.add(key)
        return {"dismissed": dismissed, "closed": closed}

    def update_metadata_fields_low_side_effect(
        self,
        memory_id: int,
        set_fields: Optional[dict] = None,
        clear_fields: Optional[list] = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """Patch claim bookkeeping metadata without content-history side effects."""
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable", "memory_id": memory_id}
        try:
            with db.write_transaction() as conn:
                current = db._fetch_memory(conn, memory_id)
                if not current:
                    return {"outcome": "not_found", "memory_id": memory_id}
                status = current.get("status")
                if status != "active":
                    return {"outcome": "not_active", "memory_id": memory_id, "status": status}

                protection = current.get("protection_level")
                source_type = current.get("source_type")
                if (protection == "locked" or source_type == "user_confirmed") and not authorized:
                    return {
                        "outcome": "forbidden",
                        "memory_id": memory_id,
                        "protection_level": protection,
                        "source_type": source_type,
                    }

                raw_md = current.get("metadata")
                if isinstance(raw_md, dict):
                    md = dict(raw_md)
                elif isinstance(raw_md, str):
                    try:
                        parsed = json.loads(raw_md) if raw_md else {}
                        md = parsed if isinstance(parsed, dict) else {}
                    except (json.JSONDecodeError, ValueError):
                        md = {}
                else:
                    md = {}

                new_md = dict(md)
                for key, value in (set_fields or {}).items():
                    if value is None or value == "":
                        new_md.pop(key, None)
                    else:
                        new_md[key] = value
                for key in (clear_fields or []):
                    new_md.pop(key, None)

                if new_md == md:
                    return {"outcome": "no_change", "memory_id": memory_id, "metadata": md}

                old_entity = canon_token(md.get("entity"))
                new_entity = canon_token(new_md.get("entity"))
                old_scope = canon_scope(md.get("scope"))
                new_scope = canon_scope(new_md.get("scope"))
                claim_semantics_changed = old_entity != new_entity or old_scope != new_scope
                conn.execute(
                    "UPDATE memories SET metadata=?, claim_revision=claim_revision+? WHERE id=?",
                    (
                        json.dumps(new_md, ensure_ascii=False),
                        1 if claim_semantics_changed else 0,
                        memory_id,
                    ),
                )
                return {
                    "outcome": "updated", "memory_id": memory_id, "metadata": new_md,
                    "claim_semantics_changed": claim_semantics_changed,
                }
        except sqlite3.Error:
            return {"outcome": "error", "memory_id": memory_id}

    def list_entities(
        self,
        limit: int = 50,
        include_unassigned: bool = True,
    ) -> dict[str, Any]:
        """Aggregate explicit metadata.entity values across active memories."""
        db = self._db
        if not db._db_available:
            return {
                "entities": [], "distinct_entities": 0, "assigned_count": 0,
                "total_active": 0, "unassigned_count": 0, "unassigned_ids": [],
            }
        counts: dict[str, int] = {}
        sample: dict[str, int] = {}
        unassigned: list[int] = []
        unassigned_count = 0
        total = 0
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT id, metadata FROM memories WHERE status='active' ORDER BY id"
            ).fetchall()
        for row in rows:
            total += 1
            raw_md = row["metadata"]
            if isinstance(raw_md, str):
                try:
                    parsed = json.loads(raw_md) if raw_md else {}
                    md = parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, ValueError):
                    md = {}
            else:
                md = raw_md if isinstance(raw_md, dict) else {}
            entity = canon_token(md.get("entity"))
            if entity:
                counts[entity] = counts.get(entity, 0) + 1
                sample.setdefault(entity, int(row["id"]))
            else:
                unassigned_count += 1
                if include_unassigned:
                    unassigned.append(int(row["id"]))
        entities = [
            {"entity": entity, "count": counts[entity], "sample_memory_id": sample[entity]}
            for entity in sorted(counts, key=lambda key: (-counts[key], key))[:limit]
        ]
        return {
            "entities": entities,
            "distinct_entities": len(counts),
            "assigned_count": sum(counts.values()),
            "total_active": total,
            "unassigned_count": unassigned_count,
            "unassigned_ids": unassigned[:limit],
        }
