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
    cleaned = _cleanup_first_round_artifacts(conn)
    if cleaned:
        applied.append(cleaned)
    swept = _sweep_queued_numeric_rows(conn)
    if swept:
        applied.append(swept)
    rerouted = _migrate_twin_bucket_residents(conn)
    if rerouted:
        applied.append(rerouted)
    cleared = _clearance_migrate_check_route_pairs(conn)
    if cleared:
        applied.append(cleared)
    purged = _purge_terminal_queue_rows(conn)
    if purged:
        applied.append(purged)
    quieted = _dismiss_internal_noise_rows(conn)
    if quieted:
        applied.append(quieted)
    evolution_voided = _void_evolution_queue_rows(conn)
    if evolution_voided:
        applied.append(evolution_voided)
    conn.commit()
    return applied


_NUMERIC_SWEEP_KEY = "scan_pipeline_numeric_sweep_v1"


def _sweep_queued_numeric_rows(conn: sqlite3.Connection) -> str:
    """One-shot: queued numeric-route pairs → voided (identity released) and
    watermarks reset, so the calibrated per-kick auto-reject cap (5000)
    classifies them into the audit trail on the re-run instead of burning
    agent judgment. Runs once; steady-state libraries see a no-op."""
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_NUMERIC_SWEEP_KEY,)
    ).fetchone()
    if guard is not None:
        return ""
    from ..models import utc_now_iso as _now

    now = _now()
    rows = conn.execute(
        "SELECT id, candidate_key_hash FROM scan_queue "
        "WHERE kind='conflict' AND status='pending' AND reason LIKE '%numeric_value_candidate%'"
    ).fetchall()
    for row in rows:
        conn.execute(
            "UPDATE scan_queue SET status='voided', candidate_key_hash=?, "
            "decided_reason='numeric sweep: reclassified to auto-reject', decided_at=?, updated_at=? "
            "WHERE id=?",
            (voided_identity_hash(str(row["candidate_key_hash"]), int(row["id"])), now, now, int(row["id"])),
        )
    conn.execute("DELETE FROM migration_state WHERE key='scan_pipeline_state'")
    conn.execute("UPDATE memories SET scan_watermark=NULL WHERE status='active'")
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_NUMERIC_SWEEP_KEY, f"voided={len(rows)}"),
    )
    return f"numeric_sweep(voided={len(rows)}, watermarks_reset)"


_CLEANUP_KEY = "scan_pipeline_gate_v2_cleanup_v1"


def _cleanup_first_round_artifacts(conn: sqlite3.Connection) -> str:
    """One-shot reset of the first-round classification (gate change).

    The first real kick ran before the internal precision gates (overlap
    skip, genuine-numeric shape) and with the mis-scoped round-level
    auto-reject cap: internal_conflicts flooded with splitter artifacts and
    numeric noise pairs landed in the queue. This guarded cleanup purges the
    internal table, voids queued numeric-route pairs (identity RELEASED so
    re-detection re-classifies them), resets the round state and watermarks,
    so the first full round re-runs under the corrected gates. Guarded by a
    migration_state key — never runs twice.
    """
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_CLEANUP_KEY,)
    ).fetchone()
    if guard is not None:
        return ""
    from ..models import utc_now_iso as _now

    now = _now()
    conn.execute("DELETE FROM internal_conflicts")
    rows = conn.execute(
        "SELECT id, candidate_key_hash FROM scan_queue "
        "WHERE kind='conflict' AND status='pending' AND reason LIKE '%numeric_value_candidate%'"
    ).fetchall()
    from .additive import voided_identity_hash as _vh  # module-local

    for row in rows:
        conn.execute(
            "UPDATE scan_queue SET status='voided', candidate_key_hash=?, "
            "decided_reason='gate v2: numeric route moved to auto-reject', decided_at=?, updated_at=? "
            "WHERE id=?",
            (_vh(str(row["candidate_key_hash"]), int(row["id"])), now, now, int(row["id"])),
        )
    conn.execute("DELETE FROM migration_state WHERE key='scan_pipeline_state'")
    conn.execute("UPDATE memories SET scan_watermark=NULL WHERE status='active'")
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_CLEANUP_KEY, "done"),
    )
    return f"gate_v2_cleanup(internal_purged, queue_numeric_voided={len(rows)}, watermarks_reset)"


_TWIN_REDIRECT_KEY = "twin_write_redirect_migration_v1"


def _migrate_twin_bucket_residents(conn: sqlite3.Connection) -> str:
    """One-shot: mema-twin rows written by non-twin agents move to
    mema-twin-dev (0.16.2 §1.2 stock, owner rule #976).

    The write-path redirect only covers new writes; this migrates existing
    violations. Real library: exactly 1 row (jingleAI-default). The twin's
    own rows (agent_id='mema-twin') are untouched. Rows moved here get their
    scan watermark cleared (move-as-edit) so the pipeline re-pairs them in
    the new bucket, and pending kind='workspace' suspects pinning the old
    bucket are expired in the same transaction — the move companion void in
    workspaces.move_memory_workspace_on_conn has no boot-time equivalent.
    """
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_TWIN_REDIRECT_KEY,)
    ).fetchone()
    if guard is not None:
        return ""
    now = utc_now_iso()
    rows = conn.execute(
        """SELECT id FROM memories
           WHERE status='active'
             AND COALESCE(NULLIF(workspace_canonical,''),workspace)='mema-twin'
             AND COALESCE(agent_id,'') != 'mema-twin'"""
    ).fetchall()
    moved = 0
    for row in rows:
        memory_id = int(row["id"])
        conn.execute(
            """UPDATE memories SET workspace='mema-twin-dev',
                 workspace_canonical='mema-twin-dev', scan_watermark=NULL
               WHERE id=?""",
            (memory_id,),
        )
        conn.execute(
            """UPDATE scan_queue SET status='expired',
                 decided_reason='redirect migration: subject moved to mema-twin-dev',
                 decided_at=?, updated_at=?
               WHERE kind='workspace' AND status='pending'
                 AND EXISTS(SELECT 1 FROM json_each(scan_queue.member_versions) AS m
                            WHERE CAST(json_extract(m.value,'$.memory_id') AS INTEGER)=?)""",
            (now, now, memory_id),
        )
        moved += 1
    if moved:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES ('mema-twin-dev', ?)",
            (now,),
        )
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_TWIN_REDIRECT_KEY, f"moved={moved}"),
    )
    return f"twin_redirect_migration(moved={moved})"


_CLEARANCE_KEY = "scan_queue_difference_clearance_v1"


def _clearance_migrate_check_route_pairs(conn: sqlite3.Connection) -> str:
    """One-shot: difference-based clearance of queued check-route pairs
    (0.16.2 §1.4, owner ④).

    The first full round enqueued every suspicious pair; the difference
    classifier now decides at enqueue time. This brings the STOCK to the
    same standard with the SAME implementation: notify pairs
    (severity='high') always survive, every check pair is classified from
    its evidence quotes — keepers stay pending, the rest are voided
    (identity released, so an edited pair can re-detect and re-classify).
    Cleared rows never land in ``conflicts``; counts go to
    ``migration_state`` as the audit trail.
    """
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_CLEARANCE_KEY,)
    ).fetchone()
    if guard is not None:
        return ""
    from ..difference_classifier import classify_pair, is_garbage

    now = utc_now_iso()
    rows = conn.execute(
        "SELECT id,severity,reason,candidate_key_hash,evidence FROM scan_queue "
        "WHERE kind='conflict' AND status='pending'"
    ).fetchall()
    cleared_sim = cleared_num = garbage_labeled = kept = 0
    for row in rows:
        route = str(row["reason"] or "")
        # Notify protection keys on BOTH severity and the route reason —
        # a severity-NULL legacy row that names a notify route must never
        # fall through to the classifier (defense in depth; the live
        # library has zero such rows, other deployments may not).
        if str(row["severity"] or "") == "high" or route.startswith(
            ("polarity_changed", "todo_resolved")
        ):
            kept += 1  # notify route: real-signal recall has no threshold
            continue
        try:
            evidence = json.loads(str(row["evidence"] or "[]"))
        except (TypeError, ValueError):
            evidence = []
        # Legacy candidate rows migrated from conflicts (0.16.0 §6⑲⑥) store
        # an envelope object; the scan pipeline stores a bare array. Unwrap
        # both — misreading the envelope would clear real evidence pairs.
        if isinstance(evidence, dict):
            evidence = evidence.get("evidence") or []
        quotes = [
            str(item.get("evidence_quote")) if isinstance(item, dict) and item.get("evidence_quote") else None
            for item in (evidence or [])
        ]
        while len(quotes) < 2:
            quotes.append(None)
        verdict = classify_pair(quotes[0], quotes[1], route=route)
        if verdict == "keep":
            kept += 1
            continue
        if is_garbage(quotes[0]) or is_garbage(quotes[1]):
            garbage_labeled += 1
        conn.execute(
            "UPDATE scan_queue SET status='voided', candidate_key_hash=?, "
            "decided_reason='difference clearance: no extractable value difference', "
            "decided_at=?, updated_at=? WHERE id=?",
            (voided_identity_hash(str(row["candidate_key_hash"] or ""), int(row["id"])),
             now, now, int(row["id"])),
        )
        if "numeric_value_candidate" in route:
            cleared_num += 1
        else:
            cleared_sim += 1
    summary = (
        f"cleared={cleared_sim + cleared_num} (sim={cleared_sim},numeric={cleared_num},"
        f"garbage={garbage_labeled}), kept={kept}"
    )
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_CLEARANCE_KEY, summary),
    )
    return f"difference_clearance({summary})"


_PURGE_QUEUE_KEY = "scan_queue_terminal_purge_v1"


def _purge_terminal_queue_rows(conn: sqlite3.Connection) -> str:
    """Standing boot hygiene (owner rule, 0.16.2): DELETE terminal scan_queue
    rows on every boot.

    The queue is a workbench, not an archive: every decision's durable
    outcome lives elsewhere (dismissal suppression in ``conflicts``, confirms
    in conflicts/memories), so voided/expired/dismissed/confirmed rows carry
    no operational value and only bloat the table. Deleting them also
    releases their candidate_key_hash identities. Pending/in_review rows are
    untouched — that is unfinished work.
    """
    counts: dict[str, int] = {}
    for status in ("voided", "expired", "dismissed", "confirmed"):
        cur = conn.execute("DELETE FROM scan_queue WHERE status=?", (status,))
        counts[status] = int(cur.rowcount or 0)
    total = sum(counts.values())
    if not total:
        return ""
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_PURGE_QUEUE_KEY, json.dumps(counts, sort_keys=True)),
    )
    return f"queue_purge(total={total}, {counts})"


_INTERNAL_NOISE_KEY = "internal_noise_rule_v1"


def _dismiss_internal_noise_rows(conn: sqlite3.Connection) -> str:
    """One-shot: dismiss pending internal rows matching the 0.16.3 structural
    noise shapes (table slices, note-meta lines) so the agent never sees the
    stock the new gate would not have produced. Guard-keyed, runs once; new
    writes/scans are filtered upstream by internal_noise_pair.
    """
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_INTERNAL_NOISE_KEY,)
    ).fetchone()
    if guard is not None:
        return ""
    from ..difference_classifier import internal_noise_pair

    rows = conn.execute(
        "SELECT id, quote_a, quote_b FROM internal_conflicts WHERE status='pending'"
    ).fetchall()
    dismissed = 0
    for row in rows:
        if internal_noise_pair(str(row["quote_a"] or ""), str(row["quote_b"] or "")):
            conn.execute(
                "UPDATE internal_conflicts SET status='dismissed', "
                "decided_reason='structural noise (table/meta-line rule v1)', "
                "decided_at=?, updated_at=? WHERE id=?",
                (utc_now_iso(), utc_now_iso(), int(row["id"])),
            )
            dismissed += 1
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_INTERNAL_NOISE_KEY, f"dismissed={dismissed}"),
    )
    return f"internal_noise_dismissal({dismissed})" if dismissed else ""


_EVOLUTION_VOID_KEY = "scan_pipeline_evolution_void_v1"


def _void_evolution_queue_rows(conn: sqlite3.Connection) -> str:
    """One-shot (0.16.4 §1): pending cross-memory notify queue rows → voided.

    The evolution-domain exclusion (is_cross_evolution) means the pipeline
    no longer produces todo_resolved/polarity_changed rows at all — the
    pending stock the old semantics queued is cleared so the agent never
    judges it. ``voided`` (not not_a_conflict): the identity is RELEASED on
    purpose — if the exclusion ever misses a path, the pair re-enqueues and
    surfaces the gap instead of being suppressed silent. Guard-keyed, runs
    once; the standing boot purge (_purge_terminal_queue_rows) deletes the
    voided rows on a later boot.
    """
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_EVOLUTION_VOID_KEY,)
    ).fetchone()
    if guard is not None:
        return ""
    now = utc_now_iso()
    cur = conn.execute(
        "UPDATE scan_queue SET status='voided', "
        "decided_reason='0.16.4 evolution-domain exclusion (retroactive clear)', "
        "decided_at=?, updated_at=? "
        "WHERE status='pending' AND kind='conflict' "
        "AND (reason LIKE '%todo_resolved%' OR reason LIKE '%polarity_changed%')",
        (now, now),
    )
    voided = int(cur.rowcount or 0)
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_EVOLUTION_VOID_KEY, f"voided={voided}"),
    )
    return f"evolution_void({voided})" if voided else ""


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
