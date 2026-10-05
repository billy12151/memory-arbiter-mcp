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

import json
import sqlite3
from typing import TYPE_CHECKING

from ..models import utc_now_iso
from .conflict_backlog import conflict_backlog_ddl

# 拆分批 ③ 纯移动 re-export（历史 import 面不变；guard-key 常量随函数走）。
from .additive_ddl import (  # noqa: F401
    _sha as _sha,
    has_column as has_column,
    memory_row_ddl as memory_row_ddl,
    rebuild_memory_row_for_subject_kind as rebuild_memory_row_for_subject_kind,
    scan_queue_ddl as scan_queue_ddl,
    internal_conflicts_ddl as internal_conflicts_ddl,
    normalize_audit_ddl as normalize_audit_ddl,
    workspace_dismissals_ddl as workspace_dismissals_ddl,
    governance_audit_ddl as governance_audit_ddl,
    voided_identity_hash as voided_identity_hash,
)
from .additive_migrations import (  # noqa: F401
    _UNIT_RETIREMENT_KEY as _UNIT_RETIREMENT_KEY,
    _VEC_DEFERRED_KEY as _VEC_DEFERRED_KEY,
    _SUBJECT_ROW_KEY as _SUBJECT_ROW_KEY,
    _METADATA_PURGE_KEY as _METADATA_PURGE_KEY,
    _NUMERIC_SWEEP_KEY as _NUMERIC_SWEEP_KEY,
    _CLEANUP_KEY as _CLEANUP_KEY,
    _TWIN_REDIRECT_KEY as _TWIN_REDIRECT_KEY,
    _CLEARANCE_KEY as _CLEARANCE_KEY,
    _INTERNAL_NOISE_KEY as _INTERNAL_NOISE_KEY,
    _EVOLUTION_VOID_KEY as _EVOLUTION_VOID_KEY,
    _CONTENT_SHA_KEY as _CONTENT_SHA_KEY,
    _STATUS_INGEST_IDX_KEY as _STATUS_INGEST_IDX_KEY,
    _OVERFLOW_RETIRED_KEY as _OVERFLOW_RETIRED_KEY,
    _NOTICE_KEY_BACKFILL as _NOTICE_KEY_BACKFILL,
    _drop_vec_table_deferred as _drop_vec_table_deferred,
    _retire_unit_tables as _retire_unit_tables,
    _purge_retired_metadata_keys as _purge_retired_metadata_keys,
    _rebuild_memory_row_store as _rebuild_memory_row_store,
    _sweep_queued_numeric_rows as _sweep_queued_numeric_rows,
    _cleanup_first_round_artifacts as _cleanup_first_round_artifacts,
    _migrate_twin_bucket_residents as _migrate_twin_bucket_residents,
    _clearance_migrate_check_route_pairs as _clearance_migrate_check_route_pairs,
    _dismiss_internal_noise_rows as _dismiss_internal_noise_rows,
    _void_evolution_queue_rows as _void_evolution_queue_rows,
    _retire_claims_tables as _retire_claims_tables,
    _add_content_sha_dedupe as _add_content_sha_dedupe,
    _add_memories_status_ingest_index as _add_memories_status_ingest_index,
    _retire_conflicts_overflow as _retire_conflicts_overflow,
    _backfill_notice_dedupe_keys as _backfill_notice_dedupe_keys,
)


if TYPE_CHECKING:
    from typing import Callable

# migration_state guard key: the candidate-row migration runs exactly once.
_MIGRATION_KEY = "scan_queue_candidate_migration_v1"




def _run_additive_segment(
    conn: sqlite3.Connection, label: str, applied: list[str],
    fn: "Callable[[], None]",
) -> bool:
    """B2-2（0.17.1 修复批）：段级隔离——单段失败回滚该段并继续后续段。

    此前任一步 sqlite3.Error 逃出 ensure_additive_structures → core 吞掉
    整个 additive（其后所有步骤每次启动永久跳过，实测 unit 通道即此形态）。

    实现用 **SAVEPOINT**（而非 conn.rollback()——那会把先前段尚未提交的
    DML 一并回滚：Python sqlite3 默认 isolation_level='' 下 DDL 自动提交
    而 DML 挂在隐式事务上，实测先前段落的 deferred 账本会被后续段的
    rollback 抹掉）。SAVEPOINT 只回滚本段：失败段内的 DML 与其
    migration_state guard 键同段同事务，回滚后不留"半完成 + 已标 guard"
    状态（R2 对抗审查的核心约束），下轮可安全重试。

    DDL 在 SQLite 中同样受 SAVEPOINT 约束（无独立 DDL 事务），故本段
    建表失败会被回滚；全部 DDL 皆 IF NOT EXISTS，重跑幂等。
    """
    savepoint = "additive_seg"
    try:
        conn.execute(f"SAVEPOINT {savepoint}")
    except sqlite3.Error:
        # 连接不支持 SAVEPOINT（极旧 SQLite）：退化为直接执行（无隔离）
        try:
            fn()
            return True
        except sqlite3.Error as exc:
            applied.append(f"{label}:failed({exc})")
            return False
    try:
        fn()
        conn.execute(f"RELEASE {savepoint}")
        return True
    except sqlite3.Error as exc:
        try:
            conn.execute(f"ROLLBACK TO {savepoint}")
            conn.execute(f"RELEASE {savepoint}")
        except sqlite3.Error:
            pass
        applied.append(f"{label}:failed({exc})")
        return False


def ensure_additive_structures(conn: sqlite3.Connection) -> list[str]:
    """Create any missing 0.16.0 structures; returns the applied change list.

    Idempotent and safe under concurrent boot: probes before ALTER, IF NOT
    EXISTS for tables/indexes, and the one-shot candidate migration is guarded
    by a ``migration_state`` key written in the same transaction. The caller
    owns the connection; changes are committed here so partial startup state
    never persists.

    B2-2（0.17.1 修复批）：段级隔离——每段独立 try/except（见
    ``_run_additive_segment``），单段失败只回滚该段并记
    ``{label}:failed(...)``，后续段照常执行（此前一步失败会让其后所有
    步骤永久跳过）。段划分按真实执行顺序。
    """
    applied: list[str] = []

    def _columns() -> None:
        if not has_column(conn, "memories", "scan_watermark"):
            conn.execute("ALTER TABLE memories ADD COLUMN scan_watermark INTEGER")
            applied.append("memories.scan_watermark")
        # 0.17.0 P2-6.2: slow-lane rotation clock — "least recently scanned"
        # picks anchors by wall time, independent of version bumps.
        if not has_column(conn, "memories", "last_scanned_at"):
            conn.execute("ALTER TABLE memories ADD COLUMN last_scanned_at TEXT")
            applied.append("memories.last_scanned_at")

    _run_additive_segment(conn, "columns", applied, _columns)

    # 0.17.1（owner 2026-09-29 拍板）：claims 数据层全退——表连带 DROP。
    # 幂等：不存在的库无操作；存量库连数据一并清除（检测线已零读取，
    # 数据无消费方）。sqlite-vec 0.1.x DROP 虚拟主表不连带清影子表，
    # memory_claim_vec_% 需显式 sweep（对齐 rebuild_vec_tables 口径），
    # 否则跑过 claims 通道的存量库升级后影子表成永久孤儿。
    _run_additive_segment(
        conn, "claims_retirement", applied,
        lambda: _retire_claims_tables(conn, applied),
    )

    def _tables() -> None:
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

        # 0.17.1 workspace dismiss 持久化：决策记录独立于 scan_queue 工作台——
        # 启动 purge / 检测器换代整表 DELETE 释放行身份后，dismiss 仍然存活。
        dismissals_existed = bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workspace_dismissals'"
        ).fetchone())
        conn.executescript(workspace_dismissals_ddl())
        if not dismissals_existed:
            applied.append("workspace_dismissals")

        # 0.17.1 P2 #7: governance_audit — bucket-level action trail (rename/
        # migrate/confirm_pending reasons). No memory_id/status columns, so the
        # normalize_audit consumers (rollback_auto_move, doctor) are untouched.
        governance_existed = bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='governance_audit'"
        ).fetchone())
        conn.executescript(governance_audit_ddl())
        if not governance_existed:
            applied.append("governance_audit")

    # B2-2 修订（R3/R4 对抗双轮实证）：本段**不进 SAVEPOINT**——
    # executescript() 会先 COMMIT（隐式），把外层 savepoint 释放掉：实测
    # RELEASE / ROLLBACK TO 均报 "no such savepoint"，于是每次启动都记出
    # 假失败 ddl:failed(no such savepoint)，并把真实错误信息掩盖成症状
    # （R3/R4 独立复现）。诚实的形态：本段无隔离，但全部 DDL 皆
    # IF NOT EXISTS，重跑幂等；异常仍收敛为单段失败标记（后续段照常）。
    try:
        _tables()
    except sqlite3.Error as exc:
        applied.append(f"ddl:failed({exc})")

    def _migrations_a() -> None:
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

    _run_additive_segment(conn, "migrations_a", applied, _migrations_a)

    def _migrations_b() -> None:
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

    _run_additive_segment(conn, "migrations_b", applied, _migrations_b)

    def _unit_retirement() -> None:
        retired = _retire_unit_tables(conn, applied)
        if retired:
            applied.append(retired)

    _run_additive_segment(conn, "unit_retirement", applied, _unit_retirement)

    def _purge() -> None:
        metadata_purged = _purge_retired_metadata_keys(conn)
        if metadata_purged:
            applied.append(metadata_purged)

    _run_additive_segment(conn, "metadata_purge", applied, _purge)

    conn.commit()
    return applied




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
