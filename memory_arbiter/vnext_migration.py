"""Build and verify a clean side-by-side local-text evidence database."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import sqlite3
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import Settings
from .constants import (
    EMBED_PREFIX_STS,
    EMBEDDING_DEFAULT_DIM,
    is_default_workspace_term,
)
from .db import MemoryDB
from .db_generation import (
    CONFLICT_DETECTOR_VERSION,
    CURRENT_SCHEMA_GENERATION,
    SCHEMA_MIGRATIONS,
    detect_database_generation,
)
from .db.meta import ACTIVE_DIM_META_KEY, active_dim_on_connection
from .evidence import local_text_units
from .db.meta import active_scan_boundary_on_connection, canonical_scan_boundary
from .tools import MemoryTools


# 0.17.0 C6: memory_evidence is no longer preserved (unit channel retired);
# memory_row carries the derived rows instead and rebuilds like units did.
PRESERVED_TABLES = (
    "memories", "memory_history", "memory_row",
    "workspace_canonicals", "workspace_aliases", "backup_replay_log",
)
FULL_REBUILD_COPY_TABLES = tuple(
    table for table in PRESERVED_TABLES if table != "memory_row"
)
DESTRUCTIVELY_REBUILT_TABLES = (
    "conflicts", "conflict_judgments", "semantic_notices", "workspace_alias_events",
)
from .vnext_probe import (  # noqa: F401
    _table_exists as _table_exists,
    _configured_embedding_space_id as _configured_embedding_space_id,
    _source_vec_state as _source_vec_state,
    _source_active_dim as _source_active_dim,
    _counts_on_connection as _counts_on_connection,
    _counts as _counts,
    _destructive_counts as _destructive_counts,
    _fingerprint_on_connection as _fingerprint_on_connection,
    _fingerprint as _fingerprint,
)
from .vnext_final import final_sync as final_sync

_BUILDING_SCHEMA_GENERATION = f"{CURRENT_SCHEMA_GENERATION}:building"
_OBSOLETE_SUCCESS_RECEIPTS = (
    "row_counts_match", "evidence_coverage", "failed_count", "source_stable",
    "workspace_vector_failures", "destructive_tables_empty",
    "target_space_ready", "cursor_memory_id", "evidence_rebuild_space_id",
)


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
    source_generation = _source_schema_generation(source)
    migration = SCHEMA_MIGRATIONS.get(source_generation or "")
    vector_effect = migration.vector_effect if migration is not None else "rebuild"
    conflict_only = migration is not None and vector_effect == "preserve"
    vec_state = _source_vec_state(source)
    active_dim = _source_active_dim(source)
    configured_space = _configured_embedding_space_id(settings, active_dim)
    active_space = vec_state.get("active_space_id")
    if vec_state.get("state") == "ready" and configured_space and active_space == configured_space:
        vector_compatibility = "ready"
    elif configured_space and active_space and active_space != configured_space:
        vector_compatibility = "mismatch"
    else:
        vector_compatibility = vec_state.get("state", "unmanaged")
    # Disk estimation is the one sanctioned EMBEDDING_DEFAULT_DIM fallback:
    # before any model loads, only a rough per-vector byte figure is needed.
    estimate_dim = active_dim if active_dim is not None else EMBEDDING_DEFAULT_DIM
    vector_bytes = 0 if conflict_only else units * estimate_dim * 4
    # build()/final_sync() create the parent directory themselves; inspect
    # may run first (dry run) against a not-yet-existing path.
    target.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target.parent).free
    required = source.stat().st_size + vector_bytes * 2 + 64 * 1024 * 1024
    return {
        "source": str(source), "target": str(target),
        "upgrade_mode": "conflict_only" if conflict_only else "full_evidence_rebuild",
        "schema_migration": {
            "source_generation": source_generation,
            "target_generation": CURRENT_SCHEMA_GENERATION,
            "vector_effect": vector_effect,
        },
        "vector_compatibility": vector_compatibility,
        "active_space_id": active_space,
        "configured_space_id": configured_space,
        "active_dim": active_dim,
        "estimated_vector_dim": estimate_dim,
        "estimated_dim_is_default_fallback": active_dim is None,
        "evidence_reuse_reason": (
            "schema_migration_preserves_vectors" if conflict_only
            else "schema_migration_requires_rebuild"
        ),
        "source_bytes": source.stat().st_size, "counts": _counts(source),
        "destructive_history_loss": list(DESTRUCTIVELY_REBUILT_TABLES),
        "destructive_history_counts": _destructive_counts(source),
        "estimated_evidence_units": units,
        "estimated_vector_bytes": vector_bytes,
        "free_bytes": free_bytes, "required_bytes": required,
        "disk_ok": free_bytes >= required, "target_exists": target.exists(),
    }


def _source_schema_generation(path: Path) -> str | None:
    try:
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            row = conn.execute(
                "SELECT value FROM migration_state WHERE key='schema_generation'"
            ).fetchone()
            return str(row[0]) if row is not None else None
    except sqlite3.Error:
        return None


def _complete_migration_on_connection(
    conn: sqlite3.Connection,
    *,
    extra_state: dict[str, str] | None = None,
) -> None:
    values = dict(extra_state or {})
    values["schema_generation"] = CURRENT_SCHEMA_GENERATION
    values["migration_completed_at"] = conn.execute(
        "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')"
    ).fetchone()[0]
    for key, value in values.items():
        conn.execute(
            "INSERT INTO migration_state(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
            (key, value),
        )
    conn.execute("DELETE FROM migration_state WHERE key='phase'")
    placeholders = ",".join("?" for _ in _OBSOLETE_SUCCESS_RECEIPTS)
    conn.execute(
        f"DELETE FROM migration_state WHERE key IN ({placeholders})",
        _OBSOLETE_SUCCESS_RECEIPTS,
    )


def _set_preserved_vector_compatibility(
    conn: sqlite3.Connection,
    settings: Settings,
) -> dict[str, str | None]:
    meta = {
        str(row[0]): str(row[1])
        for row in conn.execute("SELECT key,value FROM _vec_index_meta")
    }
    configured = _configured_embedding_space_id(
        settings, active_dim_on_connection(conn),
    )
    active = meta.get("active_space_id")
    vector_rows = 0
    if configured and not active:
        try:
            # 0.17.0 C6 (review finding): count the ROW store — the unit vec
            # table is empty since the worker merge, so the old probe saw
            # vector_rows=0 and silently blessed foreign-space row vectors.
            vector_rows = int(
                conn.execute("SELECT COUNT(*) FROM memory_row_vec").fetchone()[0]
            ) + int(
                conn.execute("SELECT COUNT(*) FROM workspace_canonicals_vec").fetchone()[0]
            )
        except sqlite3.Error:
            vector_rows = 1
    if configured and ((active and configured != active) or (not active and vector_rows)):
        conn.execute(
            "INSERT INTO _vec_index_meta(key,value) VALUES('state','mismatch') "
            "ON CONFLICT(key) DO UPDATE SET value='mismatch'"
        )
        conn.execute(
            "INSERT INTO _vec_index_meta(key,value) VALUES('target_space_id',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (configured,),
        )
        state = "mismatch"
    elif configured and not active:
        conn.execute(
            "INSERT INTO _vec_index_meta(key,value) VALUES('state','ready') "
            "ON CONFLICT(key) DO UPDATE SET value='ready'"
        )
        conn.execute(
            "INSERT INTO _vec_index_meta(key,value) VALUES('active_space_id',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (configured,),
        )
        state = "ready"
        active = configured
    else:
        state = meta.get("state", "unmanaged")
    return {
        "state": state,
        "active_space_id": active,
        "target_space_id": configured if state == "mismatch" else meta.get("target_space_id"),
    }


def _current_conflict_schema(settings: Settings) -> list[str]:
    """Generate current conflict DDL from the authoritative schema definition."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        from .db.schema import SchemaStore

        fake_db = type("SchemaTemplateDB", (), {
            "settings": settings,
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


def build_conflict_only(
    source: Path,
    target: Path,
    settings: Settings,
    *,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Clone a dev1 evidence DB and transactionally replace only conflict state."""
    plan = plan or inspect(source, target, settings)
    if plan.get("upgrade_mode") != "conflict_only":
        return {
            "ok": False,
            "error": "evidence_space_not_reusable",
            "reason": plan.get("evidence_reuse_reason"),
            "plan": plan,
        }
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
        from .db.schema import SchemaStore
        SchemaStore._normalize_workspace_alias_schema(conn)
        conn.execute("DROP TABLE IF EXISTS semantic_notices")
        conn.execute("DROP TABLE IF EXISTS conflict_judgments")
        conn.execute("DROP TABLE IF EXISTS conflicts")
        for statement in _current_conflict_schema(settings):
            conn.execute(statement)
        boundary = canonical_scan_boundary(active_scan_boundary_on_connection(conn))
        state = {
            "phase": "building",
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
        # A progress row copied from the source (e.g. a completed scan) belongs
        # to a superseded epoch and must not wedge the fresh one.
        conn.execute("DELETE FROM migration_state WHERE key='conflict_scan_progress'")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

    target_fp = _fingerprint(target)
    preserved = all(source_fp.get(key) == target_fp.get(key) for key in source_fp)
    destructive_empty = all(value == 0 for value in _destructive_counts(target).values())
    switch_ready = preserved and destructive_empty
    vector_state: dict[str, str | None] = {}
    if switch_ready:
        try:
            with contextlib.closing(sqlite3.connect(target)) as final_conn:
                final_conn.execute("BEGIN IMMEDIATE")
                vector_state = _set_preserved_vector_compatibility(final_conn, settings)
                _complete_migration_on_connection(final_conn, extra_state={
                    key: value for key, value in state.items() if key != "phase"
                })
                final_conn.commit()
            switch_ready = _checkpoint(target)
        except (sqlite3.Error, OSError):
            switch_ready = False
    if switch_ready:
        _remove_sidecars(target)
    else:
        # Mirror the full-rebuild path: a validation-failed target must not sit
        # marked phase=ready/current — detect_database_generation would treat a
        # manually adopted artifact as a good current database.
        try:
            failed_conn = sqlite3.connect(target)
            try:
                failed_conn.execute("BEGIN IMMEDIATE")
                failed_conn.execute(
                    "INSERT INTO migration_state(key,value,updated_at) VALUES('phase','failed',CURRENT_TIMESTAMP) "
                    "ON CONFLICT(key) DO UPDATE SET value='failed',updated_at=CURRENT_TIMESTAMP"
                )
                failed_conn.execute(
                    "DELETE FROM migration_state WHERE key='migration_completed_at'"
                )
                failed_conn.commit()
            finally:
                failed_conn.close()
        except sqlite3.Error:
            pass
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
        "vector_effect": "preserve",
        "vec_index_state": vector_state,
        "conflict_scan": state,
        "switch_ready": switch_ready,
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _copy_preserved_tables(source: Path, db: MemoryDB, *, chunk_size: int = 500) -> None:
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    try:
        for table in FULL_REBUILD_COPY_TABLES:
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
            if table == "memories":
                # Gate-v2 G3: a 0.16-era source may carry metadata.entity/scope
                # — the copy is column-generic (no per-row hook), so the same
                # storage-level strip runs once over the copied table, in its
                # own transaction, before anything reads the rows back.
                with db.write_transaction() as dst:
                    dst.execute(
                        "UPDATE memories SET metadata = json_remove(metadata, '$.entity', '$.scope') "
                        "WHERE json_valid(metadata) "
                        "AND (json_type(metadata,'$.entity') IS NOT NULL "
                        "  OR json_type(metadata,'$.scope')  IS NOT NULL)"
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
        "conflict_scan_required": "true",
        "conflict_scan_epoch": epoch,
        "conflict_scan_detector_version": CONFLICT_DETECTOR_VERSION,
        "conflict_scan_boundary": boundary,
    }
    with db.write_transaction() as conn:
        _complete_migration_on_connection(conn, extra_state=values)
        conn.execute("DELETE FROM migration_state WHERE key='conflict_scan_progress'")
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
            conn.execute(
                "DELETE FROM migration_state WHERE key='migration_completed_at'"
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
    plan = inspect(source, target, settings)
    if plan.get("upgrade_mode") == "conflict_only":
        if resume:
            return {"ok": False, "error": "conflict_only_upgrade_is_not_resumable"}
        return build_conflict_only(source, target, settings, plan=plan)
    if plan.get("ok") is False:
        return dict(plan)
    if not plan["disk_ok"]:
        return {"ok": False, "error": "insufficient_disk_space", "plan": plan}
    if target.exists() and not resume:
        return {"ok": False, "error": "target_exists_use_resume_or_new_path", "plan": plan}
    target.parent.mkdir(parents=True, exist_ok=True)
    target_settings = replace(settings, db_path=target, semantic_conflict_on_write="off")
    if not target.exists():
        # Seed an incomplete generation before constructing MemoryDB. Schema
        # initialization preserves the :building marker, so a crash during
        # table copy is never classified as a current/openable database.
        with contextlib.closing(sqlite3.connect(target)) as seed:
            seed.execute(
                "CREATE TABLE migration_state("
                "key TEXT PRIMARY KEY,value TEXT NOT NULL,"
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            seed.execute(
                "INSERT INTO migration_state(key,value) VALUES('schema_generation',?)",
                (_BUILDING_SCHEMA_GENERATION,),
            )
            seed.execute(
                "INSERT INTO migration_state(key,value) VALUES('phase','building')"
            )
            seed.commit()
        try:
            db = MemoryDB(target_settings, allow_incomplete=True)
            os.chmod(target, 0o600)
            _copy_preserved_tables(source, db)
            with db.connection() as conn:
                db.schema._rebuild_fts(conn)
        except BaseException:
            _reset_phase_to_failed(target)
            raise
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
    rebuild_embedder, rebuild_warnings = tools._ensure_embedder()
    expected_space_id = (
        rebuild_embedder.embedding_space_id if rebuild_embedder is not None else None
    )
    if rebuild_embedder is not None:
        try:
            with db.write_transaction() as conn:
                row = conn.execute(
                    "SELECT value FROM migration_state WHERE key='evidence_rebuild_space_id'"
                ).fetchone()
                prior_rebuild_space = str(row["value"]) if row is not None else None
                cursor_row = conn.execute(
                    "SELECT value FROM migration_state WHERE key='cursor_memory_id'"
                ).fetchone()
                cursor_started = cursor_row is not None and int(cursor_row["value"] or 0) > 0
                if (
                    (prior_rebuild_space is not None and prior_rebuild_space != expected_space_id)
                    or (prior_rebuild_space is None and cursor_started)
                ):
                    conn.execute("DELETE FROM memory_row_vec")
                    conn.execute("DELETE FROM memory_row")
                    conn.execute("DELETE FROM workspace_canonicals_vec")
                    # Hint vectors from an aborted rebuild live in a foreign
                    # embedding space; the table may not exist on a fresh
                    # clone yet (created lazily at the first embedder load).
                    try:
                        conn.execute("DELETE FROM subject_tags_vec")
                    except sqlite3.Error:
                        pass
                    conn.execute("DELETE FROM migration_state WHERE key='cursor_memory_id'")
                    conn.execute("DELETE FROM _vec_index_meta")
                    conn.executemany(
                        "INSERT INTO _vec_index_meta(key,value) VALUES(?,?)",
                        (
                            ("state", "ready"),
                            ("active_space_id", expected_space_id),
                            (ACTIVE_DIM_META_KEY, str(rebuild_embedder.dim)),
                        ),
                    )
                conn.execute(
                    "INSERT INTO migration_state(key,value,updated_at) "
                    "VALUES('evidence_rebuild_space_id',?,CURRENT_TIMESTAMP) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                    "updated_at=CURRENT_TIMESTAMP",
                    (expected_space_id,),
                )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            _reset_phase_to_failed(target)
            return {"ok": False, "error": f"rebuild_space_state_failed: {exc}"}
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
    workspace_vector_failures: list[dict[str, Any]] = []
    if rebuild_embedder is not None:
        with db.connection() as conn:
            canonical_names = [
                str(row["name"])
                for row in conn.execute("SELECT name FROM workspace_canonicals ORDER BY id")
                if not is_default_workspace_term(str(row["name"] or ""))
            ]
        for canonical in canonical_names:
            workspace_embedding = rebuild_embedder.embed_text(prefix=EMBED_PREFIX_STS, body=canonical)
            warnings = db.workspaces.publish_workspace_canonical_vector(
                canonical,
                list(workspace_embedding.embedding) if workspace_embedding.embedding else None,
            )
            if warnings or not workspace_embedding.embedding:
                workspace_vector_failures.append({
                    "canonical": canonical,
                    "warnings": warnings or ["empty_embedding"],
                })
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
            index_result = tools._index_local_text_evidence(int(memory["id"]), memory)
            if index_result.get("status") != "indexed":
                failed.append({"memory_id": int(memory["id"]), "result": index_result})
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
    # Evidence is never copied by the full-rebuild path. Derived rows are
    # validated by coverage and target-space identity instead of source/target
    # equality.
    row_counts_match = all(
        source_counts.get(table, 0) == target_counts.get(table, 0)
        for table in FULL_REBUILD_COPY_TABLES
    )
    source_fp, target_fp = _fingerprint(source), _fingerprint(target)
    stable_keys = [
        key for key in source_fp
        if not key.startswith("memory_row_") and not key.startswith("memory_evidence_")
    ]
    # A full rebuild may intentionally change logical evidence text/offsets
    # when the embedding pipeline version changes. Core source data remains
    # fingerprint-stable; derived evidence is validated by coverage and the
    # target-space gate below.
    source_stable = all(source_fp.get(key) == target_fp.get(key) for key in stable_keys)
    destructive_counts = _destructive_counts(target)
    destructive_tables_empty = all(
        destructive_counts.get(table, 0) == 0 for table in DESTRUCTIVELY_REBUILT_TABLES
    )
    vec_state = db.get_vec_index_state()
    with db.connection() as conn:
        expected_workspace_vectors = sum(
            1 for row in conn.execute("SELECT name FROM workspace_canonicals")
            if not is_default_workspace_term(str(row["name"] or ""))
        )
        try:
            workspace_vectors = int(
                conn.execute("SELECT COUNT(*) FROM workspace_canonicals_vec").fetchone()[0]
            )
        except sqlite3.Error:
            workspace_vectors = 0
    target_space_ready = bool(
        expected_space_id
        and vec_state.get("state") == "ready"
        and vec_state.get("active_space_id") == expected_space_id
    )
    complete = (
        not failed
        and not workspace_vector_failures
        and coverage["indexed_memories"] == coverage["eligible_memories"]
        and coverage["vectors"] == coverage["units"]
        and workspace_vectors == expected_workspace_vectors
        and row_counts_match
        and source_stable
        and destructive_tables_empty
        and target_space_ready
    )
    scan_state: dict[str, str] = {}
    if complete:
        scan_state = _mark_conflict_rebuild_ready(db)
    if not complete:
        _set_state(db, {"phase": "failed"})
    switch_ready = False
    if complete:
        del tools
        del db
        gc.collect()
        switch_ready = _checkpoint(target)
        if switch_ready:
            _remove_sidecars(target)
        else:
            _reset_phase_to_failed(target)
            complete = False
        os.chmod(target, 0o600)
    return {
        "ok": complete, "target": str(target), "indexed": indexed,
        "upgrade_mode": "full_evidence_rebuild",
        "coverage": coverage, "row_counts_match": row_counts_match,
        "source_stable": source_stable, "source_fingerprint": source_fp,
        "target_fingerprint": target_fp, "failed": failed,
        "workspace_vector_failures": workspace_vector_failures,
        "workspace_vector_coverage": {
            "expected": expected_workspace_vectors,
            "vectors": workspace_vectors,
        },
        "destructive_tables_empty": destructive_tables_empty,
        "target_space_ready": target_space_ready,
        "expected_space_id": expected_space_id,
        "vec_index_state": vec_state,
        "embedding_warnings": rebuild_warnings,
        "conflict_scan": scan_state,
        "switch_ready": switch_ready,
        "next_step": "freeze writes and run --final-sync before switching db_path" if complete else "fix failures and rerun with --resume",
    }


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
