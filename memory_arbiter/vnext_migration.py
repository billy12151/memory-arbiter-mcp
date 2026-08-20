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
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import Settings
from .db import MemoryDB
from .db_generation import CURRENT_SCHEMA_GENERATION
from .evidence import local_text_units
from .tools import MemoryTools


PRESERVED_TABLES = (
    "memories", "memory_history", "conflicts", "conflict_judgments",
    "semantic_notices", "workspace_canonicals", "workspace_aliases",
    "workspace_alias_events", "backup_replay_log",
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _counts_on_connection(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        if _table_exists(conn, table) else 0
        for table in PRESERVED_TABLES
    }


def _counts(path: Path) -> dict[str, int]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return _counts_on_connection(conn)


def _fingerprint(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        digest = hashlib.sha256()
        count = 0
        for row in conn.execute(
            "SELECT id,version,status,content,subject,tags,workspace,workspace_canonical "
            "FROM memories ORDER BY id"
        ):
            count += 1
            digest.update(json.dumps(dict(row), ensure_ascii=False, sort_keys=True).encode())
            digest.update(b"\n")
        return {"memory_count": count, "memory_digest": digest.hexdigest()}
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
    vector_bytes = units * int((settings.vec_dim if settings is not None else Settings.vec_dim)) * 4
    # build()/final_sync() create the parent directory themselves; inspect
    # may run first (dry run) against a not-yet-existing path.
    target.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target.parent).free
    required = source.stat().st_size + vector_bytes * 2 + 64 * 1024 * 1024
    return {
        "source": str(source), "target": str(target),
        "source_bytes": source.stat().st_size, "counts": _counts(source),
        "estimated_evidence_units": units,
        "estimated_vector_bytes": vector_bytes,
        "free_bytes": free_bytes, "required_bytes": required,
        "disk_ok": free_bytes >= required, "target_exists": target.exists(),
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _copy_preserved_tables(source: Path, db: MemoryDB) -> None:
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    try:
        with db.write_transaction() as dst:
            dst.execute("PRAGMA defer_foreign_keys=ON")
            for table in PRESERVED_TABLES:
                if not _table_exists(src, table):
                    continue
                common = [name for name in _columns(dst, table) if name in _columns(src, table)]
                if not common:
                    continue
                quoted = ",".join(f'"{name}"' for name in common)
                rows = src.execute(f"SELECT {quoted} FROM {table}").fetchall()
                if not rows:
                    continue
                placeholders = ",".join("?" for _ in common)
                dst.executemany(
                    f"INSERT INTO {table}({quoted}) VALUES({placeholders})",
                    [tuple(row[name] for name in common) for row in rows],
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
        # A crashed build leaves phase in {failed, backfill}, which the
        # startup generation guard refuses to open as a current database —
        # the exact state --resume exists for. Clear the phase via a raw
        # connection first so MemoryDB can start; build() re-marks it below.
        with contextlib.closing(sqlite3.connect(target)) as conn:
            phase_row = conn.execute(
                "SELECT value FROM migration_state WHERE key='phase'"
            ).fetchone()
            if phase_row is not None and str(phase_row[0]) in {"failed", "backfill"}:
                conn.execute(
                    "UPDATE migration_state SET value='resuming' WHERE key='phase'"
                )
                conn.commit()
        db = MemoryDB(target_settings)
    tools = MemoryTools(settings=target_settings, db=db)
    _set_state(db, {"phase": "backfill", "source_path": str(source)})
    with db.connection() as conn:
        cursor_row = conn.execute("SELECT value FROM migration_state WHERE key='cursor_memory_id'").fetchone()
        cursor = int(cursor_row["value"]) if cursor_row else 0
        rows = conn.execute("SELECT * FROM memories WHERE id>? AND status!='deleted' ORDER BY id", (cursor,)).fetchall()
    failed: list[dict[str, Any]] = []
    indexed = 0
    for row in rows:
        memory = dict(row)
        result = tools._index_local_text_evidence(int(memory["id"]), memory)
        if result.get("status") != "indexed":
            failed.append({"memory_id": int(memory["id"]), "result": result})
            break
        indexed += 1
        _set_state(db, {"cursor_memory_id": str(memory["id"])})
        if progress and indexed % 25 == 0:
            print(json.dumps({"indexed": indexed, "remaining": len(rows) - indexed}), file=sys.stderr, flush=True)

    coverage = db.evidence.coverage()
    source_counts, target_counts = _counts(source), _counts(target)
    row_counts_match = source_counts == target_counts
    source_fp, target_fp = _fingerprint(source), _fingerprint(target)
    source_stable = source_fp == target_fp
    complete = not failed and coverage["indexed_memories"] == coverage["eligible_memories"] and row_counts_match and source_stable
    _set_state(db, {
        "schema_generation": CURRENT_SCHEMA_GENERATION,
        "phase": "ready" if complete else "failed",
        "row_counts_match": str(row_counts_match).lower(),
        "evidence_coverage": f"{coverage['indexed_memories']}/{coverage['eligible_memories']}",
        "failed_count": str(len(failed)), "source_stable": str(source_stable).lower(),
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
        "switch_ready": switch_ready,
        "next_step": "freeze writes and run --final-sync before switching db_path" if complete else "fix failures and rerun with --resume",
    }


def final_sync(
    source: Path,
    target: Path,
    settings: Settings,
    *,
    progress: bool = True,
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
    # The build verified the source before checkpoint/replace; a write that
    # landed since then would be silently absent from the target. Re-verify
    # the fingerprint immediately before the atomic replace and refuse to
    # switch on any drift (the caller documents that writers must be stopped).
    if _fingerprint(source) != result.get("source_fingerprint"):
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
