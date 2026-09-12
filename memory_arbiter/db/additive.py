"""Idempotent additive structure completion (0.16.0 plan §6⑲).

Historical gap: an existing database ran ZERO DDL at startup
(``initialize_schema`` is true only for missing/empty files), so new columns
and tables had no creation point without the heavyweight vnext migration.
This module is the sanctioned additive channel: check-then-create, idempotent
under concurrency (every statement is IF NOT EXISTS / column-probed ALTER),
never touches ``schema_generation``, and never triggers the vnext gate.

Covered structures (0.16.0):
- ``memories.scan_watermark`` — per-memory version watermark for the
  incremental conflict-scan pipeline (NULL = never scanned / invalidated by
  edit or move).
- ``scan_queue`` — the independent judgment queue for scan-suspected items
  (plan §6㉑): never enters ``conflicts``, never enters notices, invisible to
  the user-facing conflict lists.
- ``internal_conflicts`` — same-memory internal contradictions (plan §6⑳);
  the conflicts table's pair/slot invariants stay untouched.
- ``normalize_audit`` — auto-move audit trail for the workspace-normalization
  gate (plan §6⑫), the rollback anchor for memory_govern rollback.
- one-shot migration of historical scan-produced ``candidate`` rows out of
  ``conflicts`` (plan §6㉑⑥), including the ``notice_type IS NULL`` ghost
  notifications (P0-2).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from ..models import utc_now_iso

# migration_state guard key: the candidate-row migration runs exactly once.
_MIGRATION_KEY = "scan_queue_candidate_migration_v1"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def voided_identity_hash(base: str, row_id: int) -> str:
    """Deterministic 64-char replacement identity for a voided conflicts row.

    Shared by the candidate migration and ConflictStore.void_conflicts_*:
    rewriting candidate_key_hash/member_fingerprint releases the cross-status
    UNIQUE indexes (§6⑯③④) while keeping a deterministic, auditable value.
    """
    if base:
        return _sha(f"{base}:voided:{row_id}")
    return _sha(f"voided:{row_id}")


def scan_queue_ddl() -> str:
    return """
    CREATE TABLE IF NOT EXISTS scan_queue (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      kind TEXT NOT NULL DEFAULT 'conflict' CHECK(kind IN ('conflict','internal','workspace')),
      workspace_canonical TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','in_review','confirmed','dismissed','voided','expired')),
      candidate_key_hash TEXT NOT NULL UNIQUE CHECK(length(candidate_key_hash)=64),
      member_versions TEXT NOT NULL CHECK(json_valid(member_versions) AND json_type(member_versions)='array' AND length(member_versions) <= 262144),
      evidence TEXT CHECK(evidence IS NULL OR (json_valid(evidence) AND length(evidence) <= 131072)),
      reason TEXT NOT NULL DEFAULT '',
      severity TEXT,
      source TEXT NOT NULL DEFAULT 'scan_pipeline',
      detail TEXT CHECK(detail IS NULL OR (json_valid(detail) AND json_type(detail)='object' AND length(detail) <= 32768)),
      decided_ref TEXT,
      decided_reason TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      decided_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_scan_queue_pending ON scan_queue(status, id);
    CREATE INDEX IF NOT EXISTS idx_scan_queue_ws ON scan_queue(workspace_canonical, status);
    """


def internal_conflicts_ddl() -> str:
    return """
    CREATE TABLE IF NOT EXISTS internal_conflicts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
      memory_version INTEGER NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','dismissed','resolved','stale')),
      unit_a INTEGER NOT NULL,
      unit_b INTEGER NOT NULL,
      quote_a TEXT NOT NULL,
      quote_b TEXT NOT NULL,
      span_a TEXT NOT NULL CHECK(json_valid(span_a)),
      span_b TEXT NOT NULL CHECK(json_valid(span_b)),
      reason TEXT NOT NULL DEFAULT '',
      detector_version TEXT NOT NULL,
      decided_reason TEXT,
      decided_at TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(memory_id, memory_version, unit_a, unit_b)
    );
    CREATE INDEX IF NOT EXISTS idx_internal_conflicts_open
      ON internal_conflicts(memory_id, status);
    """


def normalize_audit_ddl() -> str:
    return """
    CREATE TABLE IF NOT EXISTS normalize_audit (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
      from_workspace TEXT NOT NULL,
      to_workspace TEXT NOT NULL,
      gate TEXT NOT NULL DEFAULT '{}',
      status TEXT NOT NULL DEFAULT 'applied'
        CHECK(status IN ('applied','rolled_back','manual_move')),
      rolled_back_at TEXT,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_normalize_audit_memory
      ON normalize_audit(memory_id, created_at);
    """


def has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(
        str(row[1]) == column for row in conn.execute(f"PRAGMA table_info({table})")
    )


def ensure_additive_structures(conn: sqlite3.Connection) -> list[str]:
    """Create any missing 0.16.0 structures; returns the applied change list.

    Idempotent and safe under concurrent boot: probes before ALTER, IF NOT
    EXISTS for tables/indexes, and the one-shot candidate migration is guarded
    by a ``migration_state`` key written in the same transaction. The caller
    owns the connection; changes are committed here so partial startup state
    never persists.
    """
    applied: list[str] = []
    if not has_column(conn, "memories", "scan_watermark"):
        conn.execute("ALTER TABLE memories ADD COLUMN scan_watermark INTEGER")
        applied.append("memories.scan_watermark")
    scan_queue_existed = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scan_queue'"
    ).fetchone())
    conn.executescript(scan_queue_ddl())
    if not scan_queue_existed:
        applied.append("scan_queue")
    internal_existed = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='internal_conflicts'"
    ).fetchone())
    conn.executescript(internal_conflicts_ddl())
    if not internal_existed:
        applied.append("internal_conflicts")
    audit_existed = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='normalize_audit'"
    ).fetchone())
    conn.executescript(normalize_audit_ddl())
    if not audit_existed:
        applied.append("normalize_audit")
    migrated = _migrate_legacy_candidates(conn)
    if migrated:
        applied.append(f"candidate_rows_migrated({migrated})")
    conn.commit()
    return applied


def _migrate_legacy_candidates(conn: sqlite3.Connection) -> int:
    """One-shot: move scan-produced ``candidate`` rows into ``scan_queue``.

    Conflicts rows with ``status='candidate'`` and ``notice_type IS NULL`` are
    scan-side queue material (plan §6㉑⑥) — including the ``delivered`` ghost
    notifications that P0-2 proved would otherwise hijack the notice channel.
    Write-time notices (``notice_type IS NOT NULL``) stay in ``conflicts``:
    the write-time Qwen path keeps its notice semantics (E11 ②).

    Each migrated row is voided in ``conflicts`` — status ``resolved`` (a
    terminal that is NOT in the suppression loader's status set, so re-queue
    and re-record stay possible), with candidate_key_hash and
    member_fingerprint rewritten to release the cross-status UNIQUE
    identities (§6⑯③④).
    """
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_MIGRATION_KEY,)
    ).fetchone()
    if guard is not None:
        return 0
    now = utc_now_iso()
    rows = conn.execute(
        "SELECT id,candidate_key_hash,member_fingerprint,workspace_canonical,"
        "member_versions,value_groups,detection_reason,source,notice_delivery_status "
        "FROM conflicts WHERE status='candidate' AND notice_type IS NULL"
    ).fetchall()
    migrated = 0
    for row in rows:
        row_id = int(row["id"])
        members_raw = row["member_versions"]
        try:
            members = json.loads(str(members_raw or "[]"))
        except (TypeError, ValueError):
            members = []
        if not members:
            # Unusable envelope: void without queueing.
            _void_row(conn, row_id, str(row["candidate_key_hash"] or ""),
                      str(row["member_fingerprint"] or ""), "migrated: unusable envelope", now)
            migrated += 1
            continue
        try:
            groups = json.loads(str(row["value_groups"] or "[]"))
        except (TypeError, ValueError):
            groups = []
        evidence = [
            {
                "memory_id": int(member["memory_id"]),
                "version": int(member.get("version") or 1),
                "evidence_quote": member.get("evidence_quote"),
                "evidence_span": member.get("evidence_span"),
                "evidence_unit": member.get("evidence_unit"),
                "value_raw": member.get("value_raw"),
                "normalized_value": member.get("normalized_value"),
            }
            for member in members
        ]
        try:
            legacy_row = conn.execute(
                "SELECT candidate_key FROM conflicts WHERE id=?", (row_id,)
            ).fetchone()
            legacy_candidate_key = json.loads(str(legacy_row[0] or "null")) if legacy_row else None
        except (TypeError, ValueError, sqlite3.Error):
            legacy_candidate_key = None
        conn.execute(
            """INSERT OR IGNORE INTO scan_queue(
                 kind,workspace_canonical,status,candidate_key_hash,member_versions,
                 evidence,reason,severity,source,detail,created_at,updated_at)
               VALUES('conflict',?, 'pending', ?, ?, ?, ?, 'normal', ?, ?, ?, ?)""",
            (
                str(row["workspace_canonical"] or ""),
                str(row["candidate_key_hash"]),
                str(members_raw),
                json.dumps({"evidence": evidence, "value_groups": groups}, ensure_ascii=False),
                str(row["detection_reason"] or ""),
                str(row["source"] or "scan_pipeline"),
                json.dumps({"candidate_key": legacy_candidate_key}, ensure_ascii=False),
                now, now,
            ),
        )
        _void_row(conn, row_id, str(row["candidate_key_hash"] or ""),
                  str(row["member_fingerprint"] or ""), "migrated_to_scan_queue", now)
        migrated += 1
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_MIGRATION_KEY, f"migrated={migrated}"),
    )
    return migrated


def _void_row(
    conn: sqlite3.Connection, row_id: int, candidate_hash: str,
    fingerprint: str, reason: str, now: str,
) -> None:
    """Terminal-void one conflicts row, releasing every cross-status identity.

    Status ``resolved`` keeps the row out of the suppression loader
    (``open``/``applying``/``not_a_conflict`` — §6⑯②), out of the active-slot
    index (open/applying only), while the hash rewrites free the candidate
    identity (§6⑯③) and the event-snapshot fingerprint (§6⑯④).
    """
    voided_hash = voided_identity_hash(candidate_hash, row_id)
    voided_fingerprint = voided_identity_hash(fingerprint or f"fp-missing:{row_id}", row_id)
    conn.execute(
        """UPDATE conflicts SET status='resolved',candidate_key_hash=?,member_fingerprint=?,
           decided_by='agent',decision_reason=?,decided_at=?,resolved_at=?,
           notice_delivery_status=CASE WHEN notice_delivery_status IN ('pending','delivered')
             THEN 'stale' ELSE notice_delivery_status END,
           revision=revision+1,refreshed_at=?
         WHERE id=? AND status IN ('open','applying','candidate')""",
        (voided_hash, voided_fingerprint, f"voided: {reason}", now, now, now, row_id),
    )
