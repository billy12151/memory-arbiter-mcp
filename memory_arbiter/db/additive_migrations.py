"""加性通道的一次性 guard-key 迁移（从 additive.py 搬出，拆分批 ③ 纯移动）。

每函数一个 migration_state guard key，幂等可重试；段编排
（ensure_additive_structures）与两个被测 patch 的函数
（_purge_terminal_queue_rows/_migrate_legacy_candidates）留守 additive.py。
guard-key 常量随函数走；additive.py re-export 保住历史 import 面。
"""
from __future__ import annotations

import json
import sqlite3

from ..models import utc_now_iso
from .additive_ddl import (
    _sha,
    has_column,
    rebuild_memory_row_for_subject_kind,
    voided_identity_hash,
)


_UNIT_RETIREMENT_KEY = "unit_vector_tables_retired_v1"
# B2-3（0.17.1 修复批）：vec 虚表 DROP 的 deferred 账本（单键 JSON 字典，
# 值 = {table: first_deferred_at}）。doctor 的 additive.deferred_drops 读它。
_VEC_DEFERRED_KEY = "vec_drop_deferred"


def _drop_vec_table_deferred(
    conn: sqlite3.Connection, table: str, applied: list[str],
    *, label: str | None = None,
) -> bool:
    """B2-1（0.17.1 修复批）：DROP 一个 vec0 虚表；模块未注册 → deferred。

    返回 True 表示"本轮已 deferred"（调用方必须跳过其影子 sweep 与 guard
    键写入，下轮重试）。口径与 claims 退役（_retire_claims_tables）一致，
    两处共用本 helper。``label`` 用于回执文案（claims 通道保留历史前缀
    ``claims_vec_drop_deferred``，unit 通道用 ``vec_drop_deferred``）。

    "no such module" = sqlite-vec 未加载（未装 extra / 未配模型）：本轮
    DROP 必失败，且影子表必须保留——否则恢复轮（模块就位）的 DROP 会因
    xDestroy 找不到影子表报 "SQL logic error"（不含 no-such-module，不被
    此处接住）→ 永久跳过。其他 sqlite3.Error 照旧 re-raise（真实故障不
    得被吞）。
    """
    existed = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone())
    if not existed:
        return False
    try:
        conn.execute(f"DROP TABLE {table}")
    except sqlite3.OperationalError as exc:
        if "no such module" not in str(exc):
            raise
        applied.append(f"{label or 'vec_drop_deferred'}({table})")
        # B2-3（0.17.1 修复批）：deferred 持久化——单个 JSON 字典键（不用
        # per-table 键：memory_status 全量回显 migration_state，无界增长）。
        # 值 = {table: first_deferred_at}；恢复轮 DROP 成功后移除对应表项。
        try:
            row = conn.execute(
                "SELECT value FROM migration_state WHERE key=?", (_VEC_DEFERRED_KEY,)
            ).fetchone()
            current = json.loads(row[0]) if row and row[0] else {}
            if not isinstance(current, dict):
                current = {}
            current.setdefault(table, utc_now_iso())
            conn.execute(
                "INSERT OR REPLACE INTO migration_state(key,value) VALUES (?,?)",
                (_VEC_DEFERRED_KEY, json.dumps(current, ensure_ascii=False)),
            )
        except sqlite3.Error:
            pass
        return True
    # 成功 DROP：清掉该表的 deferred 记录（若存在）
    try:
        row = conn.execute(
            "SELECT value FROM migration_state WHERE key=?", (_VEC_DEFERRED_KEY,)
        ).fetchone()
        current = json.loads(row[0]) if row and row[0] else {}
        if isinstance(current, dict) and table in current:
            current.pop(table, None)
            if current:
                conn.execute(
                    "INSERT OR REPLACE INTO migration_state(key,value) VALUES (?,?)",
                    (_VEC_DEFERRED_KEY, json.dumps(current, ensure_ascii=False)),
                )
            else:
                conn.execute("DELETE FROM migration_state WHERE key=?", (_VEC_DEFERRED_KEY,))
    except sqlite3.Error:
        pass
    return False


def _retire_unit_tables(
    conn: sqlite3.Connection, applied: "list[str] | None" = None,
) -> str:
    """C6: guarded DROP of memory_evidence + memory_evidence_vec.

    The guard is ONE SQL answer (never two coverage numbers subtracted in
    Python): retirement may proceed only when no non-deleted, indexable
    memory still holds unit rows without row rows. Until the row backfill
    covers the library the migration skips and retries on the next boot —
    never delete-then-backfill. migration_state key prevents re-entry after
    the tables are gone.

    B2-1（0.17.1 修复批）：vec 虚表 DROP 走 _drop_vec_table_deferred——
    deferred 时本函数返回 ""（不写 guard 键，下轮重试）且影子表保留；
    applied 用于回执（缺省 None 时内部临时列表，行为等价）。
    """
    applied = applied if applied is not None else []
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
        # B2-1（0.17.1 修复批）：unit vec 虚表同 claims 口径——模块未注册时
        # DROP 报 "no such module" → 本轮 deferred（跳过 DROP 与 guard 键写入，
        # 下轮重试）。此前的裸 DROP 在"影子表已被 rebuild_vec_tables sweep、
        # 虚表本体残留"的形态下报 "SQL logic error"（不含 no-such-module）
        # → 逃出本函数 → core 吞掉整个 additive → 其后所有步骤每次启动
        # 永久跳过（实测三步沙盒闭环）。
        if _drop_vec_table_deferred(conn, "memory_evidence_vec", applied):
            return ""  # 本轮不写 guard 键：装 vec 后的首次启动继续退役
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
    deferred 跳过（B2-1：与 unit 通道共用 _drop_vec_table_deferred）。影子表
    sweep **仅当没有 vec 表 deferred 时执行**——deferred 轮若把影子表清光，
    恢复轮（模块就位）的 DROP 会因 xDestroy 找不到影子表报 "SQL logic
    error"（不含 no-such-module）→ re-raise → additive 收尾对该库每次启动
    失败回滚，未来加列通道永久死亡。deferred 轮影子表保留，恢复轮 DROP
    成功后 sweep 照常清。
    """
    dropped: list[str] = []
    vec_drop_pending = False
    for table in ("memory_claim_vec", "memory_claims"):
        existed = bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone())
        if _drop_vec_table_deferred(
            conn, table, applied, label="claims_vec_drop_deferred",
        ):
            vec_drop_pending = True
            continue
        if existed:
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
