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
from .conflict_backlog import conflict_backlog_ddl

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


def memory_row_ddl() -> str:
    # 0.17.0 P2-2.2: row-level store. C3 adds the leading subject row
    # (kind='subject', span (0,0)) — plan A+, owner 2026-09-23.
    # Vectors live in the memory_row_vec vec0 table (schema.py, dim from the
    # embedder); lifecycle mirrors the retired evidence_vec (parent_status
    # flip on status change, delete+rebuild on publish).
    return """
    CREATE TABLE IF NOT EXISTS memory_row (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
      memory_version INTEGER NOT NULL,
      content_hash TEXT,
      row_index INTEGER NOT NULL,
      kind TEXT NOT NULL CHECK(kind IN ('subject','sentence','table_row')),
      text TEXT NOT NULL,
      start_offset INTEGER NOT NULL,
      end_offset INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(memory_id, row_index)
    );
    CREATE INDEX IF NOT EXISTS memory_row_memory_idx ON memory_row(memory_id);
    """


def rebuild_memory_row_for_subject_kind(conn: sqlite3.Connection) -> str:
    """C3: existing 0.17.0-dev databases carry the two-value CHECK
    (sentence/table_row) — SQLite cannot ALTER a CHECK, so rebuild the table
    with the three-value DDL. Rows themselves are DROPPED: the subject row is
    a new leading row for every memory, so the whole store re-embeds via the
    boot backfill anyway (plan C3 存量迁移 — idempotent, one-time)."""
    existing_sql = ""
    for row in conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_row'"
    ):
        existing_sql = str(row["sql"] or "")
    if "'subject'" in existing_sql:
        return "memory_row_subject_kind(already)"
    conn.execute("DROP TABLE IF EXISTS memory_row_migration")
    # memory_row_ddl() carries TWO statements (table + index) — sqlite3's
    # execute() takes one at a time; split them explicitly (found live at
    # boot: the additive chain aborted with "only one statement at a time").
    conn.execute(
        memory_row_ddl()
        .replace("CREATE TABLE IF NOT EXISTS memory_row (",
                 "CREATE TABLE memory_row_migration (", 1)
        .split("CREATE INDEX")[0]
        .strip()
    )
    conn.execute("DROP TABLE memory_row")
    conn.execute("ALTER TABLE memory_row_migration RENAME TO memory_row")
    conn.execute("CREATE INDEX IF NOT EXISTS memory_row_memory_idx ON memory_row(memory_id)")
    # P3 fix (adversarial review): ids restart at 1 — a stale rebuild epoch
    # (old MAX id) would mark every re-embedded row as pre-epoch and block
    # the mismatch->ready flip forever. Clear it; a rebuild in flight re-arms
    # on its next execute.
    try:
        conn.execute(
            "DELETE FROM _vec_index_meta WHERE key='space_rebuild_evidence_id'"
        )
    except sqlite3.Error:
        pass
    return "memory_row_subject_kind(rebuilt)"


def scan_queue_ddl() -> str:
    return """
    CREATE TABLE IF NOT EXISTS scan_queue (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      kind TEXT NOT NULL DEFAULT 'conflict' CHECK(kind IN ('conflict','internal','workspace')),
      workspace_canonical TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','confirmed','dismissed','voided','expired')),
      -- 0.16.6 dropped the phantom in_review state (zero writers ever).
      -- Databases created before 0.16.6 keep their 6-state CHECK — harmless:
      -- new code never writes in_review, and a legacy row in that state
      -- would simply read as already_terminal at submit time.
      candidate_key_hash TEXT NOT NULL UNIQUE CHECK(length(candidate_key_hash)=64),
      member_versions TEXT NOT NULL CHECK(json_valid(member_versions) AND json_type(member_versions)='array' AND length(member_versions) <= 262144),
      evidence TEXT CHECK(evidence IS NULL OR (json_valid(evidence) AND length(evidence) <= 131072)),
      reason TEXT NOT NULL DEFAULT '',
      severity TEXT,
      source TEXT NOT NULL DEFAULT 'scan_pipeline',
      -- 0.17.0 review R2：判定页窗口内按 pair 分数降序展示（compute_pair_score
      -- 同式打分，入队时盖章；旧库由下方 has_column 迁移补列，默认 0=按 id）。
      priority REAL NOT NULL DEFAULT 0,
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


def workspace_dismissals_ddl() -> str:
    return """
    CREATE TABLE IF NOT EXISTS workspace_dismissals(
      memory_id INTEGER NOT NULL,
      version INTEGER NOT NULL,
      suspected_workspace TEXT NOT NULL,
      reason TEXT,
      decided_at TEXT NOT NULL,
      PRIMARY KEY(memory_id, version, suspected_workspace)
    );
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
    # 0.17.0 P2-6.2: slow-lane rotation clock — "least recently scanned"
    # picks anchors by wall time, independent of version bumps.
    if not has_column(conn, "memories", "last_scanned_at"):
        conn.execute("ALTER TABLE memories ADD COLUMN last_scanned_at TEXT")
        applied.append("memories.last_scanned_at")
    scan_queue_existed = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scan_queue'"
    ).fetchone())
    conn.executescript(scan_queue_ddl())
    if not has_column(conn, "scan_queue", "priority"):
        conn.execute("ALTER TABLE scan_queue ADD COLUMN priority REAL NOT NULL DEFAULT 0")
        applied.append("scan_queue.priority")
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
    # 0.17.0 Part 2: row-level conflict store + write-time backlog + claims.
    for name, ddl in (
        ("memory_row", memory_row_ddl()),
        ("conflict_backlog", conflict_backlog_ddl()),
    ):
        existed = bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone())
        conn.executescript(ddl)
        if not existed:
            applied.append(name)
    # 0.17.1（owner 2026-09-29 拍板）：claims 数据层全退——表连带 DROP。
    # 幂等：不存在的库无操作；存量库连数据一并清除（检测线已零读取，
    # 数据无消费方）。sqlite-vec 0.1.x DROP 虚拟主表不连带清影子表，
    # memory_claim_vec_% 需显式 sweep（对齐 rebuild_vec_tables 口径），
    # 否则跑过 claims 通道的存量库升级后影子表成永久孤儿。
    _retire_claims_tables(conn, applied)

    # 0.17.1 workspace dismiss 持久化：决策记录独立于 scan_queue 工作台——
    # 启动 purge / 检测器换代整表 DELETE 释放行身份后，dismiss 仍然存活。
    dismissals_existed = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workspace_dismissals'"
    ).fetchone())
    conn.executescript(workspace_dismissals_ddl())
    if not dismissals_existed:
        applied.append("workspace_dismissals")
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
    sha_dedupe = _add_content_sha_dedupe(conn)
    if sha_dedupe:
        applied.append(sha_dedupe)
    status_ingest_idx = _add_memories_status_ingest_index(conn)
    if status_ingest_idx:
        applied.append(status_ingest_idx)
    overflow_retired = _retire_conflicts_overflow(conn)
    if overflow_retired:
        applied.append(overflow_retired)
    notice_keys = _backfill_notice_dedupe_keys(conn)
    if notice_keys:
        applied.append(notice_keys)
    subject_rows = _rebuild_memory_row_store(conn)
    if subject_rows:
        applied.append(subject_rows)
    retired = _retire_unit_tables(conn)
    if retired:
        applied.append(retired)
    metadata_purged = _purge_retired_metadata_keys(conn)
    if metadata_purged:
        applied.append(metadata_purged)
    conn.commit()
    return applied


_UNIT_RETIREMENT_KEY = "unit_vector_tables_retired_v1"


def _retire_unit_tables(conn: sqlite3.Connection) -> str:
    """C6: guarded DROP of memory_evidence + memory_evidence_vec.

    The guard is ONE SQL answer (never two coverage numbers subtracted in
    Python): retirement may proceed only when no non-deleted, indexable
    memory still holds unit rows without row rows. Until the row backfill
    covers the library the migration skips and retries on the next boot —
    never delete-then-backfill. migration_state key prevents re-entry after
    the tables are gone.
    """
    try:
        already = conn.execute(
            "SELECT value FROM migration_state WHERE key=?", (_UNIT_RETIREMENT_KEY,)
        ).fetchone()
    except sqlite3.Error:
        return ""
    if already is not None:
        return ""
    tables = {
        str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('memory_evidence','memory_evidence_vec')"
        )
    }
    if not tables:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO migration_state(key,value) VALUES (?,?)",
                (_UNIT_RETIREMENT_KEY, "absent_at_boot"),
            )
        except sqlite3.Error:
            pass
        return ""
    if "memory_evidence" in tables:
        blocking = conn.execute(
            """SELECT 1 FROM memories m
               WHERE m.status!='deleted'
                 AND (COALESCE(m.subject,'')!='' OR TRIM(COALESCE(m.content,''))!='')
                 AND EXISTS(SELECT 1 FROM memory_evidence e WHERE e.memory_id=m.id)
                 AND NOT EXISTS(SELECT 1 FROM memory_row r WHERE r.memory_id=m.id)
               LIMIT 1"""
        ).fetchone()
        if blocking is not None:
            return ""  # row coverage incomplete — retry next boot
    if "memory_evidence_vec" in tables:
        conn.execute("DROP TABLE memory_evidence_vec")
    if "memory_evidence" in tables:
        conn.execute("DROP TABLE memory_evidence")
    try:
        conn.execute(
            "INSERT OR REPLACE INTO migration_state(key,value) VALUES (?,?)",
            (_UNIT_RETIREMENT_KEY, "dropped"),
        )
    except sqlite3.Error:
        pass
    return "unit_tables(dropped)"


_SUBJECT_ROW_KEY = "memory_row_subject_kind_v1"


_METADATA_PURGE_KEY = "metadata_entity_scope_purged_v1"


def _purge_retired_metadata_keys(conn: sqlite3.Connection) -> str:
    """Gate-v2 G3 keyed migration (owner: 不要了就清理干净): strip the retired
    metadata.entity/scope keys from every stored memory in ONE atomic SQL.
    json_valid guards the whole statement — a single malformed-JSON metadata
    row would otherwise raise inside json_type and abort boot (json_remove
    on a NULL column is a no-op and json_valid(NULL) is NULL, so both are
    excluded by the WHERE). Re-run is a no-op via the migration_state key;
    the write-path strip in db/memories.py is the second, permanent half of
    the double lock."""
    try:
        already = conn.execute(
            "SELECT value FROM migration_state WHERE key=?", (_METADATA_PURGE_KEY,)
        ).fetchone()
    except sqlite3.Error:
        return ""
    if already is not None:
        return ""
    cursor = conn.execute(
        "UPDATE memories SET metadata = json_remove(metadata, '$.entity', '$.scope') "
        "WHERE json_valid(metadata) "
        "AND (json_type(metadata,'$.entity') IS NOT NULL "
        "  OR json_type(metadata,'$.scope')  IS NOT NULL)"
    )
    try:
        conn.execute(
            "INSERT OR REPLACE INTO migration_state(key,value) VALUES (?,?)",
            (_METADATA_PURGE_KEY, f"purged({cursor.rowcount})"),
        )
    except sqlite3.Error:
        pass
    return f"metadata_entity_scope_purged({cursor.rowcount})"


def _rebuild_memory_row_store(conn: sqlite3.Connection) -> str:
    """C3 keyed migration: rebuild memory_row with the three-value kind CHECK
    and EMPTY it (subject rows are new for every memory — the boot backfill
    re-embeds the whole store, idempotent). The vec table's stale ids are
    cleared when the table exists; vec-less environments skip silently."""
    try:
        already = conn.execute(
            "SELECT value FROM migration_state WHERE key=?", (_SUBJECT_ROW_KEY,)
        ).fetchone()
    except sqlite3.Error:
        return ""
    if already is not None:
        return ""
    result = rebuild_memory_row_for_subject_kind(conn)
    try:
        conn.execute("DELETE FROM memory_row_vec")
    except sqlite3.Error:
        pass  # vec table not present in vec-less environments
    try:
        conn.execute(
            "INSERT OR REPLACE INTO migration_state(key,value) VALUES (?,?)",
            (_SUBJECT_ROW_KEY, result),
        )
    except sqlite3.Error:
        return ""
    return result


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
    releases their candidate_key_hash identities. Pending rows are
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



def _retire_claims_tables(conn: sqlite3.Connection, applied: list[str]) -> None:
    """0.17.1 claims 数据层退役（P1-1 修复，2026-10-03 方案 A1）。

    DROP 循环里 vec0 虚拟表在模块未注册的连接上报 "no such module" → 记
    deferred 跳过。影子表 sweep **仅当没有 vec 表 deferred 时执行**——
    deferred 轮若把影子表清光，恢复轮（模块就位）的 DROP 会因 xDestroy
    找不到影子表报 "SQL logic error"（不含 no-such-module）→ re-raise →
    additive 收尾对该库每次启动失败回滚，未来加列通道永久死亡。
    deferred 轮影子表保留，恢复轮 DROP 成功后 sweep 照常清。
    """
    dropped: list[str] = []
    vec_drop_pending = False
    for table in ("memory_claim_vec", "memory_claims"):
        existed = bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone())
        if existed:
            try:
                conn.execute(f"DROP TABLE {table}")
            except sqlite3.OperationalError as exc:
                # vec0 虚拟表的 DROP 需要模块注册：未装 sqlite-vec extra 的
                # 库上报 "no such module: vec0"。本轮跳过 vec 表与影子表
                # sweep（普通表照清），不阻塞 additive 收尾；装 vec 后的
                # 首次启动再清（影子表保留是恢复轮 DROP 可重试的前提）。
                if "no such module" not in str(exc):
                    raise
                applied.append(f"claims_vec_drop_deferred({table})")
                vec_drop_pending = True
                continue
            dropped.append(table)
    if vec_drop_pending:
        return
    claim_shadows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name LIKE 'memory_claim_vec_%'"
    ).fetchall()
    for row in claim_shadows:
        conn.execute(f'DROP TABLE IF EXISTS "{str(row[0])}"')
        dropped.append(str(row[0]))
    if dropped:
        applied.append("claims_tables_dropped(" + ",".join(dropped) + ")")


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


# ── 0.16.6 content dedup gate (owner spec 2026-09-14: partial unique over
# ACTIVE rows only — "只管活的") ────────────────────────────────────────────
_CONTENT_SHA_KEY = "content_sha_dedupe_v1"


def _add_content_sha_dedupe(conn: sqlite3.Connection) -> str:
    """One-shot: content_sha column + active-only unique index.

    sha256 over the raw UTF-8 content bytes, never normalised (any
    normalisation would fold distinct memories onto one hash). The unique
    index is PARTIAL on status='active': retired/pending rows exit the
    index on status change, so governance flows (merge, supersede) that
    legitimately produce an active + superseded same-content pair keep
    working (m986/m987 is exactly that shape). Non-active rows keep their
    sha for observability but never hold a slot.

    Idempotent: re-runs re-check the column, re-backfill NULL shas and
    re-run the duplicate self-check before the index is (re-)created.
    """
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_CONTENT_SHA_KEY,)
    ).fetchone()
    if guard is not None:
        return ""
    if not has_column(conn, "memories", "content_sha"):
        conn.execute("ALTER TABLE memories ADD COLUMN content_sha TEXT")
    # NULL/'' canonical rows would sit outside the index (NULL never
    # conflicts); normalise them to the workspace column first.
    conn.execute(
        "UPDATE memories SET workspace_canonical=workspace "
        "WHERE workspace_canonical IS NULL OR workspace_canonical=''"
    )
    rows = conn.execute(
        "SELECT id, content, status, workspace_canonical, content_sha FROM memories"
    ).fetchall()
    pending = [(_sha(row["content"] or ""), int(row["id"])) for row in rows
               if row["content_sha"] is None]
    # Self-check BEFORE any write: the effective sha of every active row
    # (stored or about-to-be-backfilled) must already be unique per
    # workspace — writing first would violate an existing index itself.
    seen: dict[tuple[object, str], list[int]] = {}
    for row in rows:
        if row["status"] != "active":
            continue
        effective = row["content_sha"] if row["content_sha"] is not None else _sha(row["content"] or "")
        seen.setdefault((row["workspace_canonical"], effective), []).append(int(row["id"]))
    dupes = {key: ids for key, ids in seen.items() if len(ids) > 1}
    if dupes:
        # Nothing has been written and the guard stays unwritten: after
        # governance retires one of each pair, the next boot retries whole.
        detail = "; ".join(
            f"ws={key[0]!r} sha={str(key[1])[:8]}… ids={ids}"
            for key, ids in list(dupes.items())[:20]
        )
        ids = sorted({i for pair_ids in dupes.values() for i in pair_ids})
        raise RuntimeError(
            "content_sha dedup migration aborted: active duplicate pairs exist. "
            "Governance must retire/merge one row of each pair before the unique "
            f"index can be created. Pairs: {detail} "
            f"Recovery (this error aborts service start by design): retire one row "
            f"of each pair directly, e.g. "
            f"sqlite3 <db> 'UPDATE memories SET status='superseded' WHERE id IN ({','.join(str(i) for i in ids)})' "
            "then restart — or boot the previous release, govern via the product "
            "tools, and upgrade again."
        )
    if pending:
        conn.executemany(
            "UPDATE memories SET content_sha=? WHERE id=? AND content_sha IS NULL",
            pending,
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_content_sha "
        "ON memories(workspace_canonical, content_sha) "
        "WHERE status='active' AND content_sha IS NOT NULL"
    )
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_CONTENT_SHA_KEY, f"backfilled={len(pending)}"),
    )
    return f"content_sha_dedupe(backfilled={len(pending)})"


_STATUS_INGEST_IDX_KEY = "memories_status_ingest_idx_v1"


def _add_memories_status_ingest_index(conn: sqlite3.Connection) -> str | None:
    """0.16.12 P1-T7: (status, ingest_time) index for the browse/recent paths
    (_recent_fallback's COUNT and recall_by_filters' ORDER BY ingest_time
    DESC). Pure index addition — no query logic changes. Idempotent: the
    migration_state key only records that the statement ran."""
    done = conn.execute(
        "SELECT 1 FROM migration_state WHERE key=?", (_STATUS_INGEST_IDX_KEY,),
    ).fetchone()
    if done:
        return None
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_status_ingest "
        "ON memories(status, ingest_time)"
    )
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_STATUS_INGEST_IDX_KEY, "created"),
    )
    return "memories_status_ingest_idx(created)"


_OVERFLOW_RETIRED_KEY = "conflicts_overflow_retired_v1"


def _retire_conflicts_overflow(conn: sqlite3.Connection) -> str:
    """One-shot: drop the write-only conflicts.overflow column.

    The column was a materialised flag for outcome="overflow"; every
    consumer reads the outcome, nobody reads the column (0.16.6 audit A5).
    Writers stopped setting it in the same release, keeping the revision
    CAS in the same statement intact.
    """
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_OVERFLOW_RETIRED_KEY,)
    ).fetchone()
    if guard is not None:
        return ""
    note = "dropped"
    if has_column(conn, "conflicts", "overflow"):
        try:
            conn.execute("ALTER TABLE conflicts DROP COLUMN overflow")
        except sqlite3.OperationalError as exc:  # pre-3.35 SQLite
            if "drop" not in str(exc).lower():
                raise  # transient (e.g. locked) — retry next boot, don't pin
            note = f"kept ({exc})"
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_OVERFLOW_RETIRED_KEY, note),
    )
    return f"conflicts_overflow_{note}"


_NOTICE_KEY_BACKFILL = "semantic_notice_dedupe_backfill_v1"


def _backfill_notice_dedupe_keys(conn: sqlite3.Connection) -> str:
    """One-shot: derive notice_dedupe_key for legacy decided semantic notices.

    is_semantic_pair_closed reads the idx_conflicts_notice_dedupe index
    since 0.16.6; rows decided before the key existed would look open and
    be re-detected (one extra notice per pair — self-healing on the next
    dismissal). Malformed member JSON is skipped, not fatal.
    """
    guard = conn.execute(
        "SELECT value FROM migration_state WHERE key=?", (_NOTICE_KEY_BACKFILL,)
    ).fetchone()
    if guard is not None:
        return ""
    import json as _json

    from ..semantic_conflict import notice_dedupe_key

    rows = conn.execute(
        "SELECT id, member_versions, notice_type FROM conflicts "
        "WHERE notice_dedupe_key IS NULL AND notice_type IS NOT NULL "
        "AND notice_delivery_status IN ('dismissed','resolved')"
    ).fetchall()
    keyed = skipped = 0
    for row in rows:
        try:
            members = _json.loads(row["member_versions"] or "[]")
            left, right = members[0], members[1]
            key = notice_dedupe_key(
                int(left["memory_id"]), int(right["memory_id"]),
                int(left["version"]), int(right["version"]),
                str(row["notice_type"]),
            )
        except (ValueError, TypeError, KeyError, IndexError):
            skipped += 1
            continue
        try:
            conn.execute(
                "UPDATE conflicts SET notice_dedupe_key=? WHERE id=? "
                "AND notice_dedupe_key IS NULL",
                (key, int(row["id"])),
            )
            keyed += 1
        except sqlite3.IntegrityError:
            # Another already-keyed row owns this pair; leave this one NULL.
            skipped += 1
    conn.execute(
        "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
        (_NOTICE_KEY_BACKFILL, f"keyed={keyed},skipped={skipped}"),
    )
    return f"notice_dedupe_backfill(keyed={keyed},skipped={skipped})"
