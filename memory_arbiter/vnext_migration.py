"""Side-by-side vNext database builder and verifier."""
from __future__ import annotations

import argparse
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
from .evidence import local_text_units
from .tools import MemoryTools


def _fingerprint(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        digest = hashlib.sha256()
        memory_count = 0
        for row in conn.execute(
            "SELECT id,version,status,content,subject,tags,workspace,workspace_canonical "
            "FROM memories ORDER BY id"
        ):
            memory_count += 1
            digest.update(json.dumps(dict(row), ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
        table_counts = _counts_on_connection(conn)
        return {"memory_count": memory_count, "memory_digest": digest.hexdigest(), "counts": table_counts}
    finally:
        conn.close()


def _counts_on_connection(conn: sqlite3.Connection) -> dict[str, int]:
    result: dict[str, int] = {}
    for table in (
        "memories", "memory_history", "conflicts", "conflict_judgments",
        "semantic_notices", "workspace_canonicals", "workspace_aliases",
        "workspace_alias_events", "backup_replay_log", "memory_sections",
    ):
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        result[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) if exists else 0
    return result


def _counts(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return _counts_on_connection(conn)
    finally:
        conn.close()


def inspect(source: Path, target: Path) -> dict[str, Any]:
    counts = _counts(source)
    conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT subject,content FROM memories WHERE status!='deleted'"
        ).fetchall()
        evidence_units = sum(
            len(local_text_units(str(row["subject"] or ""), str(row["content"] or "")))
            for row in rows
        )
    finally:
        conn.close()
    estimated_vector_bytes = evidence_units * 768 * 4
    free_bytes = shutil.disk_usage(target.parent).free
    required_bytes = source.stat().st_size + estimated_vector_bytes * 2 + 64 * 1024 * 1024
    return {
        "source": str(source),
        "target": str(target),
        "source_bytes": source.stat().st_size,
        "counts": counts,
        "estimated_evidence_units": evidence_units,
        "estimated_vector_bytes": estimated_vector_bytes,
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
        "disk_ok": free_bytes >= required_bytes,
        "target_exists": target.exists(),
    }


def _snapshot(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst, pages=256)
        dst.commit()
    finally:
        dst.close()
        src.close()


def _set_migration_state(db: MemoryDB, values: dict[str, str]) -> None:
    with db.write_transaction() as conn:
        for key, value in values.items():
            conn.execute(
                "INSERT INTO migration_state(key,value,updated_at) VALUES (?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
                (key, value),
            )


def _checkpoint_for_switch(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.commit()
    finally:
        conn.close()


def _remove_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            sidecar.unlink()


def _drop_retired_vector_tables(db: MemoryDB) -> None:
    with db.write_transaction() as conn:
        conn.execute("DROP TABLE IF EXISTS memories_vec")
        conn.execute("DROP TABLE IF EXISTS memory_sections_vec")


def build(
    source: Path,
    target: Path,
    settings: Settings,
    *,
    resume: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    plan = inspect(source, target)
    if not plan["disk_ok"]:
        return {"ok": False, "error": "insufficient_disk_space", "plan": plan}
    if target.exists() and not resume:
        return {"ok": False, "error": "target_exists_use_resume_or_new_path", "plan": plan}
    if not target.exists():
        _snapshot(source, target)

    target_settings = replace(
        settings,
        db_path=target,
        storage_profile="vnext",
        structured_claim_mode="off",
        semantic_conflict_on_write="off",
    )
    db = MemoryDB(target_settings)
    tools = MemoryTools(settings=target_settings, db=db)
    _set_migration_state(db, {
        "phase": "backfill",
        "source_path": str(source),
        "source_size": str(source.stat().st_size),
    })
    with db.connection() as conn:
        cursor_row = conn.execute(
            "SELECT value FROM migration_state WHERE key='cursor_memory_id'"
        ).fetchone()
        cursor = int(cursor_row["value"]) if cursor_row else 0
        rows = conn.execute(
            "SELECT * FROM memories WHERE id>? AND status!='deleted' ORDER BY id", (cursor,)
        ).fetchall()
    failed: list[dict[str, Any]] = []
    indexed = 0
    for row in rows:
        memory = dict(row)
        result = tools._index_local_text_evidence(int(memory["id"]), memory)
        if result.get("status") == "indexed":
            indexed += 1
            _set_migration_state(db, {"cursor_memory_id": str(memory["id"])})
        else:
            failed.append({"memory_id": int(memory["id"]), "result": result})
        if progress and (indexed % 25 == 0 or failed):
            print(
                json.dumps({"indexed": indexed, "remaining": len(rows) - indexed, "failed": len(failed)}),
                file=sys.stderr,
                flush=True,
            )
        if failed:
            break

    coverage = db.evidence.coverage()
    source_counts = _counts(source)
    target_counts = _counts(target)
    source_fingerprint = _fingerprint(source)
    target_fingerprint = _fingerprint(target)
    row_counts_match = all(
        source_counts.get(table, 0) == target_counts.get(table, 0)
        for table in (
            "memories", "memory_history", "conflicts", "conflict_judgments",
            "semantic_notices", "workspace_canonicals", "workspace_aliases",
            "workspace_alias_events", "backup_replay_log", "memory_sections",
        )
    )
    source_stable = source_fingerprint == target_fingerprint
    complete = (
        not failed
        and coverage["indexed_memories"] == coverage["eligible_memories"]
        and row_counts_match
        and source_stable
    )
    if complete:
        _drop_retired_vector_tables(db)
    _set_migration_state(db, {
        "phase": "ready" if complete else "failed",
        "row_counts_match": "true" if row_counts_match else "false",
        "evidence_coverage": f"{coverage['indexed_memories']}/{coverage['eligible_memories']}",
        "failed_count": str(len(failed)),
        "source_stable": "true" if source_stable else "false",
    })
    return {
        "ok": complete,
        "target": str(target),
        "indexed": indexed,
        "coverage": coverage,
        "row_counts_match": row_counts_match,
        "source_stable": source_stable,
        "source_fingerprint": source_fingerprint,
        "target_fingerprint": target_fingerprint,
        "failed": failed,
        "next_step": (
            "freeze writes, rerun with --resume for final sync, verify, then switch db_path and storage_profile=vnext"
            if complete else "fix failures and rerun with --resume"
        ),
    }


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a side-by-side vNext Memory Arbiter database.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--execute", action="store_true", help="Build the target; default is dry-run.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--final-sync", action="store_true",
        help="Build from a fresh source snapshot into staging and atomically replace the target. Stop writers first.",
    )
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    source = (args.source or settings.db_path).expanduser().resolve()
    target = (args.target or source.with_name(f"{source.stem}.vnext{source.suffix}")).expanduser().resolve()
    if not source.exists():
        print(json.dumps({"ok": False, "error": "source_not_found", "source": str(source)}))
        return 2
    plan = inspect(source, target)
    if not args.execute:
        print(json.dumps({"ok": True, "dry_run": True, "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    if args.final_sync:
        staging = target.with_name(target.name + ".finalizing")
        if staging.exists():
            staging.unlink()
        result = build(source, staging, settings, resume=False)
        if result.get("ok"):
            _checkpoint_for_switch(staging)
            _remove_sidecars(target)
            os.replace(staging, target)
            _remove_sidecars(staging)
            result["target"] = str(target)
            result["final_sync"] = True
        else:
            result["staging"] = str(staging)
    else:
        result = build(source, target, settings, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 2
