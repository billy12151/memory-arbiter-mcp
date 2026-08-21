"""Build and verify a clean side-by-side local-text evidence database."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .db import MemoryDB
from .db_generation import (
    CONFLICT_DETECTOR_VERSION,
    CURRENT_SCHEMA_GENERATION,
    PREVIOUS_SCHEMA_GENERATIONS,
    detect_database_generation,
)
from .evidence import local_text_units
from .db.meta import active_scan_boundary_on_connection, canonical_scan_boundary
from .tools import MemoryTools


PRESERVED_TABLES = (
    "memories", "memory_history", "memory_evidence",
    "workspace_canonicals", "workspace_aliases", "workspace_alias_events",
    "backup_replay_log",
)
DESTRUCTIVELY_REBUILT_TABLES = (
    "conflicts", "conflict_judgments", "semantic_notices",
)
_BUILDING_SCHEMA_GENERATION = f"{CURRENT_SCHEMA_GENERATION}:building"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def is_previous_evidence_generation(path: Path) -> bool:
    """Return whether path uses the immediately previous evidence schema."""
    try:
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            if not _table_exists(conn, "migration_state"):
                return False
            row = conn.execute(
                "SELECT value FROM migration_state WHERE key='schema_generation'"
            ).fetchone()
            return bool(row and str(row[0]) in PREVIOUS_SCHEMA_GENERATIONS)
    except sqlite3.Error:
        return False


def _counts_on_connection(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        if _table_exists(conn, table) else 0
        for table in PRESERVED_TABLES
    }


def _counts(path: Path) -> dict[str, int]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return _counts_on_connection(conn)


def _destructive_counts(path: Path) -> dict[str, int]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            if _table_exists(conn, table) else 0
            for table in DESTRUCTIVELY_REBUILT_TABLES
        }


def _fingerprint_on_connection(conn: sqlite3.Connection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for table, order_by in (
        ("memories", "id"),
        ("memory_history", "id"),
        ("memory_evidence", "id"),
        ("workspace_canonicals", "id"),
        ("workspace_aliases", "alias_workspace"),
        ("workspace_alias_events", "id"),
        ("backup_replay_log", "replay_key"),
    ):
        digest = hashlib.sha256()
        count = 0
        if _table_exists(conn, table):
            if table == "memory_evidence":
                rows = conn.execute(
                    """SELECT memory_id,memory_version,content_hash,unit_index,kind,
                              text,start_offset,end_offset
                       FROM memory_evidence ORDER BY memory_id,unit_index"""
                )
            else:
                rows = conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}")
            for row in rows:
                count += 1
                digest.update(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True).encode()
                )
                digest.update(b"\n")
        result[f"{table}_count"] = count
        result[f"{table}_digest"] = digest.hexdigest()
    return result


def _fingerprint(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return _fingerprint_on_connection(conn)
    finally:
        conn.close()


def inspect(source: Path, target: Path, settings: Settings | None = None) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        required_columns = {
            "id", "version", "status", "content", "subject", "tags",
            "workspace", "workspace_canonical",
        }
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(memories)")}
        if not required_columns.issubset(columns):
            return {
                "ok": False, "error": "unsupported_source_schema",
                "source": str(source),
                "missing_columns": sorted(required_columns - columns),
            }
        units = sum(
            len(local_text_units(str(row["subject"] or ""), str(row["content"] or "")))
            for row in conn.execute("SELECT subject,content FROM memories WHERE status!='deleted'")
        )
    finally:
        conn.close()
    conflict_only = is_previous_evidence_generation(source)
    vector_bytes = (
        0 if conflict_only
        else units * int((settings.vec_dim if settings is not None else Settings.vec_dim)) * 4
    )
    # build()/final_sync() create the parent directory themselves; inspect
    # may run first (dry run) against a not-yet-existing path.
    target.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target.parent).free
    required = source.stat().st_size + vector_bytes * 2 + 64 * 1024 * 1024
    return {
        "source": str(source), "target": str(target),
        "upgrade_mode": "conflict_only" if conflict_only else "full_evidence_rebuild",
        "source_bytes": source.stat().st_size, "counts": _counts(source),
        "destructive_history_loss": list(DESTRUCTIVELY_REBUILT_TABLES),
        "destructive_history_counts": _destructive_counts(source),
        "estimated_evidence_units": units,
        "estimated_vector_bytes": vector_bytes,
        "free_bytes": free_bytes, "required_bytes": required,
        "disk_ok": free_bytes >= required, "target_exists": target.exists(),
    }


def _current_conflict_schema(settings: Settings) -> list[str]:
    """Generate current conflict DDL from the authoritative schema definition."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        from .db.schema import SchemaStore

        fake_db = type("SchemaTemplateDB", (), {
            "settings": replace(settings, enable_sqlite_vec=False),
            "state": type("State", (), {"warn": lambda *_args: None})(),
            "_sqlite_vec_loadable": False,
        })()
        SchemaStore(fake_db)._init_schema(conn)
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE tbl_name='conflicts' AND sql IS NOT NULL "
            "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END,name"
        ).fetchall()
        return [str(row["sql"]) for row in rows]
    finally:
        conn.close()


def build_conflict_only(source: Path, target: Path, settings: Settings) -> dict[str, Any]:
    """Clone a dev1 evidence DB and transactionally replace only conflict state."""
    plan = inspect(source, target, settings)
    if target.exists():
        return {"ok": False, "error": "target_exists_use_new_path", "plan": plan}
    target.parent.mkdir(parents=True, exist_ok=True)
    source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    target_conn = sqlite3.connect(target)
    try:
        source_conn.backup(target_conn)
    finally:
        source_conn.close()
        target_conn.close()
    os.chmod(target, 0o600)

    source_fp = _fingerprint(source)
    epoch = uuid.uuid4().hex
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TABLE IF EXISTS semantic_notices")
        conn.execute("DROP TABLE IF EXISTS conflict_judgments")
        conn.execute("DROP TABLE IF EXISTS conflicts")
        for statement in _current_conflict_schema(settings):
            conn.execute(statement)
        boundary = canonical_scan_boundary(active_scan_boundary_on_connection(conn))
        state = {
            "schema_generation": CURRENT_SCHEMA_GENERATION,
            "phase": "ready",
            "source_path": str(source),
            "conflict_scan_required": "true",
            "conflict_scan_epoch": epoch,
            "conflict_scan_detector_version": CONFLICT_DETECTOR_VERSION,
            "conflict_scan_boundary": boundary,
        }
        for key, value in state.items():
            conn.execute(
                "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
                (key, value),
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

    target_fp = _fingerprint(target)
    preserved = all(source_fp.get(key) == target_fp.get(key) for key in source_fp)
    destructive_empty = all(value == 0 for value in _destructive_counts(target).values())
    switch_ready = preserved and destructive_empty and _checkpoint(target)
    if switch_ready:
        _remove_sidecars(target)
    return {
        "ok": switch_ready,
        "target": str(target),
        "upgrade_mode": "conflict_only",
        "indexed": 0,
        "evidence_reused": True,
        "source_stable": preserved,
        "source_fingerprint": source_fp,
        "target_fingerprint": target_fp,
        "destructive_tables_empty": destructive_empty,
        "conflict_scan": state,
        "switch_ready": switch_ready,
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _copy_preserved_tables(source: Path, db: MemoryDB, *, chunk_size: int = 500) -> None:
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    try:
        for table in PRESERVED_TABLES:
            if not _table_exists(src, table):
                continue
            with db.connection() as dst_probe:
                common = [
                    name for name in _columns(dst_probe, table)
                    if name in _columns(src, table)
                ]
            if not common:
                continue
            quoted = ",".join(f'"{name}"' for name in common)
            placeholders = ",".join("?" for _ in common)
            cursor = src.execute(f"SELECT {quoted} FROM {table}")
            while True:
                rows = cursor.fetchmany(max(1, int(chunk_size)))
                if not rows:
                    break
                with db.write_transaction() as dst:
                    dst.execute("PRAGMA defer_foreign_keys=ON")
                    dst.executemany(
                        f"INSERT INTO {table}({quoted}) VALUES({placeholders})",
                        (tuple(row[name] for name in common) for row in rows),
                    )
    finally:
        src.close()


def _set_state(db: MemoryDB, values: dict[str, str]) -> None:
    with db.write_transaction() as conn:
        for key, value in values.items():
            conn.execute(
                "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
                (key, value),
            )


def _active_scan_boundary(db: MemoryDB) -> str:
    with db.connection() as conn:
        return canonical_scan_boundary(active_scan_boundary_on_connection(conn))


def _mark_conflict_rebuild_ready(db: MemoryDB) -> dict[str, str]:
    epoch = uuid.uuid4().hex
    boundary = _active_scan_boundary(db)
    values = {
        "schema_generation": CURRENT_SCHEMA_GENERATION,
        "conflict_scan_required": "true",
        "conflict_scan_epoch": epoch,
        "conflict_scan_detector_version": CONFLICT_DETECTOR_VERSION,
        "conflict_scan_boundary": boundary,
    }
    _set_state(db, values)
    return values


def _checkpoint(path: Path) -> bool:
    def connect() -> sqlite3.Connection:
        conn = sqlite3.connect(path)
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (ImportError, sqlite3.Error):
            pass
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    try:
        with connect() as conn:
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            conn.commit()
        # SQLite returns (busy, log, checkpointed). A zero busy count proves all
        # committed WAL frames are in the main database file.
        return result is not None and int(result[0]) == 0
    except sqlite3.Error:
        return False


def _remove_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _reset_phase_to_failed(target: Path) -> None:
    """Best-effort: put a resumed target back into the refused state."""
    try:
        with contextlib.closing(sqlite3.connect(target)) as conn:
            conn.execute(
                "INSERT INTO migration_state(key, value) VALUES ('phase', 'failed') "
                "ON CONFLICT(key) DO UPDATE SET value='failed'"
            )
            conn.commit()
    except sqlite3.Error:
        pass


def _target_owned_by_source(target: Path, source: Path) -> bool:
    """Whether an existing migration artifact was built from this source."""
    if not target.exists():
        return True
    try:
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM migration_state WHERE key='source_path'"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    if row is None:
        return False
    try:
        recorded = Path(str(row[0])).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return recorded == source.expanduser().resolve()


def build(source: Path, target: Path, settings: Settings, *, resume: bool = False, progress: bool = True) -> dict[str, Any]:
    if is_previous_evidence_generation(source):
        if resume:
            return {"ok": False, "error": "conflict_only_upgrade_is_not_resumable"}
        return build_conflict_only(source, target, settings)
    plan = inspect(source, target, settings)
    if plan.get("ok") is False:
        return dict(plan)
    if not plan["disk_ok"]:
        return {"ok": False, "error": "insufficient_disk_space", "plan": plan}
    if target.exists() and not resume:
        return {"ok": False, "error": "target_exists_use_resume_or_new_path", "plan": plan}
    target.parent.mkdir(parents=True, exist_ok=True)
    target_settings = replace(settings, db_path=target, semantic_conflict_on_write="off")
    if not target.exists():
        db = MemoryDB(target_settings)
        os.chmod(target, 0o600)
        _copy_preserved_tables(source, db)
        with db.connection() as conn:
            db.schema._rebuild_fts(conn)
    else:
        # Classify by content, not by detect(): a crashed vnext target
        # (current schema, phase failed/backfill/resuming) is exactly what
        # --resume repairs, while detect() rightly classifies it "unknown"
        # so the MCP server cannot open it. Empty/new files take the fresh
        # path; anything else fails with a clean error.
        generation = detect_database_generation(target)
        if generation == "empty":
            db = MemoryDB(target_settings)
        else:
            try:
                with contextlib.closing(sqlite3.connect(target)) as conn:
                    state = {
                        str(row[0]): str(row[1])
                        for row in conn.execute("SELECT key,value FROM migration_state")
                    }
            except sqlite3.Error as exc:
                return {"ok": False, "error": f"target_not_a_vnext_database: {exc}"}
            if not (
                state.get("schema_generation") in {
                    CURRENT_SCHEMA_GENERATION, _BUILDING_SCHEMA_GENERATION,
                }
                and (
                    state.get("phase") in {"failed", "backfill", "resuming"}
                    or generation == "current"
                )
            ):
                return {"ok": False, "error": "target_not_a_vnext_database", "generation": generation}
            try:
                with contextlib.closing(sqlite3.connect(target)) as conn:
                    conn.execute(
                        "INSERT INTO migration_state(key, value) VALUES ('phase', 'resuming') "
                        "ON CONFLICT(key) DO UPDATE SET value='resuming'"
                    )
                    conn.commit()
            except sqlite3.Error as exc:
                return {"ok": False, "error": f"resume_unavailable: {exc}"}
            # 'resuming' is refused by the normal guard (a kill -9 in this
            # window must not leave the incomplete DB openable), so reopen
            # with the explicit incomplete allowance; every non-success exit
            # resets the phase so the target never stays openable.
            try:
                db = MemoryDB(target_settings, allow_incomplete=True)
            except Exception:
                _reset_phase_to_failed(target)
                raise
    tools = MemoryTools(settings=target_settings, db=db)
    try:
        _set_state(db, {
            "schema_generation": _BUILDING_SCHEMA_GENERATION,
            "phase": "backfill",
            "source_path": str(source),
        })
    except sqlite3.Error as exc:
        _reset_phase_to_failed(target)
        return {"ok": False, "error": f"resume_state_write_failed: {exc}"}
    with db.connection() as conn:
        cursor_row = conn.execute("SELECT value FROM migration_state WHERE key='cursor_memory_id'").fetchone()
        cursor = int(cursor_row["value"]) if cursor_row else 0
        remaining = int(conn.execute(
            "SELECT COUNT(*) FROM memories WHERE id>? AND status!='deleted'", (cursor,)
        ).fetchone()[0])
    failed: list[dict[str, Any]] = []
    indexed = 0
    page_size = 100
    while not failed:
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE id>? AND status!='deleted' ORDER BY id LIMIT ?",
                (cursor, page_size),
            ).fetchall()
        if not rows:
            break
        for row in rows:
            memory = dict(row)
            result = tools._index_local_text_evidence(int(memory["id"]), memory)
            if result.get("status") != "indexed":
                failed.append({"memory_id": int(memory["id"]), "result": result})
                break
            cursor = int(memory["id"])
            indexed += 1
            _set_state(db, {"cursor_memory_id": str(cursor)})
            if progress and indexed % 25 == 0:
                print(
                    json.dumps({"indexed": indexed, "remaining": max(0, remaining - indexed)}),
                    file=sys.stderr,
                    flush=True,
                )

    coverage = db.evidence.coverage()
    source_counts, target_counts = _counts(source), _counts(target)
    # Evidence may be newly generated for a pre-evidence source, so its row
    # count is validated by coverage below rather than equality. When source
    # evidence exists its logical fingerprint (excluding ids/timestamps) must
    # remain identical after vector republish.
    row_counts_match = all(
        source_counts.get(table, 0) == target_counts.get(table, 0)
        for table in PRESERVED_TABLES if table != "memory_evidence"
    )
    source_fp, target_fp = _fingerprint(source), _fingerprint(target)
    stable_keys = [
        key for key in source_fp
        if not key.startswith("memory_evidence_")
    ]
    if source_fp.get("memory_evidence_count", 0):
        stable_keys.extend(["memory_evidence_count", "memory_evidence_digest"])
    source_stable = all(source_fp.get(key) == target_fp.get(key) for key in stable_keys)
    destructive_tables_empty = all(
        target_counts.get(table, 0) == 0 for table in DESTRUCTIVELY_REBUILT_TABLES
    )
    complete = (
        not failed
        and coverage["indexed_memories"] == coverage["eligible_memories"]
        and row_counts_match
        and source_stable
        and destructive_tables_empty
    )
    scan_state: dict[str, str] = {}
    if complete:
        scan_state = _mark_conflict_rebuild_ready(db)
    _set_state(db, {
        "phase": "ready" if complete else "failed",
        "row_counts_match": str(row_counts_match).lower(),
        "evidence_coverage": f"{coverage['indexed_memories']}/{coverage['eligible_memories']}",
        "failed_count": str(len(failed)), "source_stable": str(source_stable).lower(),
        "destructive_tables_empty": str(destructive_tables_empty).lower(),
    })
    switch_ready = False
    if complete:
        del tools
        del db
        gc.collect()
        switch_ready = _checkpoint(target)
        if switch_ready:
            _remove_sidecars(target)
        os.chmod(target, 0o600)
    return {
        "ok": complete, "target": str(target), "indexed": indexed,
        "coverage": coverage, "row_counts_match": row_counts_match,
        "source_stable": source_stable, "source_fingerprint": source_fp,
        "target_fingerprint": target_fp, "failed": failed,
        "destructive_tables_empty": destructive_tables_empty,
        "conflict_scan": scan_state,
        "switch_ready": switch_ready,
        "next_step": "freeze writes and run --final-sync before switching db_path" if complete else "fix failures and rerun with --resume",
    }


def final_sync(
    source: Path,
    target: Path,
    settings: Settings,
    *,
    progress: bool = True,
    publish_callback: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build verified staging and atomically replace the side-by-side target."""
    if target.exists() and not _target_owned_by_source(target, source):
        return {
            "ok": False,
            "error": "existing_target_not_owned_by_source",
            "source": str(source),
            "target": str(target),
        }
    staging = target.with_name(target.name + ".finalizing")
    if staging.exists():
        if not _target_owned_by_source(staging, source):
            return {
                "ok": False,
                "error": "existing_staging_not_owned_by_source",
                "source": str(source),
                "staging": str(staging),
            }
        staging.unlink()
    _remove_sidecars(staging)
    result = build(source, staging, settings, progress=progress)
    if not result.get("ok"):
        result["staging"] = str(staging)
        return result
    if not result.get("switch_ready"):
        result.update({
            "ok": False,
            "error": "staging_not_switch_ready",
            "staging": str(staging),
        })
        return result
    # Hold an EXCLUSIVE transaction on the source from final verification
    # through target publication and, when supplied by the upgrade wrapper,
    # the config switch. This deterministically excludes an already-open old
    # writer from landing a commit in the verification-to-switch gap.
    source_lock = sqlite3.connect(source, timeout=5)
    source_lock.row_factory = sqlite3.Row
    try:
        source_lock.execute("PRAGMA busy_timeout=5000")
        source_lock.execute("BEGIN EXCLUSIVE")
        locked_fingerprint = _fingerprint_on_connection(source_lock)
        if locked_fingerprint != result.get("source_fingerprint"):
            source_lock.execute("ROLLBACK")
            result.update({
                "ok": False,
                "error": "source_changed_during_final_sync",
                "staging": str(staging),
                "next_step": (
                    "stop all writers and rerun the final sync; the fully built "
                    "staging database is kept at the staging path and will be "
                    "rebuilt on the next run (it roughly doubles disk usage "
                    "until then)"
                ),
            })
            return result
        _remove_sidecars(target)
        os.replace(staging, target)
        _remove_sidecars(staging)
        if publish_callback is not None:
            publish_result = publish_callback()
            result["config"] = publish_result
            if not publish_result.get("switched"):
                source_lock.execute("ROLLBACK")
                result.update({
                    "ok": False,
                    "error": "migration_complete_but_config_switch_failed",
                })
                return result
        source_lock.execute("COMMIT")
    except BaseException:
        if source_lock.in_transaction:
            source_lock.execute("ROLLBACK")
        raise
    finally:
        source_lock.close()
    # The staging build's startup-lock sidecar outlives the rename; the
    # switched-in database does not need one.
    staging_lock = staging.with_name(staging.name + ".startup.lock")
    if staging_lock.exists():
        staging_lock.unlink()
    result.update({
        "target": str(target),
        "final_sync": True,
        "next_step": (
            "verify the target in normal use, then switch db_path; "
            "keep the source for rollback"
        ),
    })
    return result


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a clean side-by-side Memory Arbiter database.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--final-sync", action="store_true", help="Stop writers first; rebuild staging and atomically replace target.")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    source = (args.source or settings.db_path).expanduser().resolve()
    target = (args.target or source.with_name(f"{source.stem}.vnext{source.suffix}")).expanduser().resolve()
    if not source.exists():
        print(json.dumps({"ok": False, "error": "source_not_found", "source": str(source)}))
        return 2
    if not args.execute:
        plan = inspect(source, target, settings)
        ok = plan.get("ok") is not False
        print(json.dumps({"ok": ok, "dry_run": True, "plan": plan}, ensure_ascii=False, indent=2))
        return 0 if ok else 2
    if args.final_sync:
        result = final_sync(source, target, settings)
    else:
        result = build(source, target, settings, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # A build that completed but could not be checkpointed (switch_ready=False)
    # is not a success: the target is not sealed for switching.
    return 0 if result.get("ok") and result.get("switch_ready") else 2
