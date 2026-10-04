"""vnext 探针/计数/指纹层（从 vnext_migration.py 搬出，拆分批 ⑦ 纯移动）。
_configured_embedding_space_id 被 doctor 懒 import 与测试直取——经 vnext_migration re-export 保活。"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import Settings
from .constants import (
    EMBEDDING_MAX_SECTION_CHARS,
    EMBEDDING_N_CTX,
    EMBEDDING_RESERVED_TOKENS,
)
from .db.meta import active_dim_on_connection
from .embedder import (
    EMBEDDING_PIPELINE_VERSION,
    compute_embedding_space_id,
    compute_model_digest,
)


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


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _configured_embedding_space_id(settings: Settings | None, active_dim: int | None) -> str | None:
    """Compute the configured space identity without loading the GGUF runtime.

    The embedding dimension is a per-library fact (``active_dim`` from the
    source database). Without it there is nothing trustworthy to derive an
    identity from — return None rather than guessing with a default, so a
    missing dim can never write a wrong space_id into the target.
    """
    if settings is None or active_dim is None or settings.embedding_model_path is None:
        return None
    model_path = settings.embedding_model_path.expanduser()
    if not model_path.is_file():
        return None
    try:
        digest = compute_model_digest(str(model_path))
    except OSError:
        return None
    return compute_embedding_space_id(
        digest,
        active_dim,
        EMBEDDING_PIPELINE_VERSION,
        {
            "n_ctx": EMBEDDING_N_CTX,
            "reserved_tokens": EMBEDDING_RESERVED_TOKENS,
            "max_section_chars": EMBEDDING_MAX_SECTION_CHARS,
        },
    )


def _source_vec_state(path: Path) -> dict[str, str]:
    try:
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            if not _table_exists(conn, "_vec_index_meta"):
                return {}
            return {
                str(row[0]): str(row[1])
                for row in conn.execute("SELECT key,value FROM _vec_index_meta")
            }
    except sqlite3.Error:
        return {}


def _source_active_dim(path: Path) -> int | None:
    """Read-only active dim of the source library (meta key, else vec0 SQL)."""
    try:
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            return active_dim_on_connection(conn)
    except sqlite3.Error:
        return None


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
        ("memory_row", "id"),
        ("workspace_canonicals", "id"),
        ("workspace_aliases", "alias_workspace,canonical"),
        ("backup_replay_log", "replay_key"),
    ):
        digest = hashlib.sha256()
        count = 0
        if _table_exists(conn, table):
            if table == "memory_row":
                # 0.17.0 C6: the derived-store fingerprint follows the rows
                # (the old unit fingerprint watched a table that is now
                # empty — a coverage blind spot found in review).
                rows = conn.execute(
                    """SELECT memory_id,memory_version,content_hash,row_index,kind,
                              text,start_offset,end_offset
                       FROM memory_row ORDER BY memory_id,row_index"""
                )
            elif table == "workspace_aliases":
                rows = conn.execute(
                    "SELECT alias_workspace,canonical,status,updated_at "
                    "FROM workspace_aliases ORDER BY alias_workspace,canonical"
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
