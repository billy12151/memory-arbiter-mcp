"""加性通道的纯 DDL 文案与结构探测 helper（从 additive.py 搬出，拆分批 ③ 纯移动）。

本模块零迁移逻辑：DDL 字符串构造、CHECK 收窄的表重建、列探测。
additive.py re-export 全部符号保住历史 import 面。
"""
from __future__ import annotations

import hashlib
import sqlite3


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


def governance_audit_ddl() -> str:
    """0.17.1 P2 #7: bucket-level governance audit trail.

    Deliberately NOT normalize_audit: these rows have no memory_id and no
    status, so the rollback_auto_move / doctor consumers of normalize_audit
    never see them (schema-incompatible by design — the R1 review's fix for
    the original normalize_audit-append proposal).
    """
    return """
    CREATE TABLE IF NOT EXISTS governance_audit (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL,
      action TEXT NOT NULL,
      reason TEXT,
      detail_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_governance_audit_created
      ON governance_audit(created_at, id);
    """


def has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(
        str(row[1]) == column for row in conn.execute(f"PRAGMA table_info({table})")
    )
