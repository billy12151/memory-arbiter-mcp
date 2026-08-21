"""Read-only database generation detection used before runtime startup."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

try:
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows lacks fcntl; startup stays best-effort
    _HAVE_FCNTL = False


DatabaseGeneration = Literal["missing", "empty", "current", "legacy", "unknown"]
CURRENT_SCHEMA_GENERATION = "conflict_groups_v2"
PREVIOUS_SCHEMA_GENERATIONS = frozenset({"local_text_evidence_v1"})
# The single identity of the running conflict-detection pipeline (deterministic
# rules + Qwen pair extraction). The scan-clearing gate compares the PERSISTED
# requirement against this running constant, and scan candidate keys stamp it;
# a future detector change bumps this once and old-detector scans can no longer
# clear conflict_scan_required.
CONFLICT_DETECTOR_VERSION = "attribute-value-v1"
LEGACY_DERIVED_TABLES = {
    "memory_claims", "memories_vec", "memory_sections_vec",
}


class LegacyDatabaseError(RuntimeError):
    """Raised before current code can initialize or mutate a legacy database."""


def detect_database_generation(path: Path) -> DatabaseGeneration:
    """Classify a database without creating or modifying it."""
    path = Path(path).expanduser()
    if not path.exists():
        return "missing"
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            conn.close()
    except sqlite3.Error:
        return "unknown"
    if not tables:
        return "empty"
    if "memories" not in tables:
        return "unknown"
    # Old releases may have created partial vNext tables while continuing to
    # use the legacy stores. Legacy ownership wins over mere table presence.
    if tables & LEGACY_DERIVED_TABLES:
        return "legacy"
    if not {"memory_evidence", "migration_state"}.issubset(tables):
        return "legacy"
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            state = {
                str(row[0]): str(row[1])
                for row in conn.execute("SELECT key,value FROM migration_state")
            }
        finally:
            conn.close()
    except sqlite3.Error:
        return "unknown"
    if state.get("schema_generation") in PREVIOUS_SCHEMA_GENERATIONS:
        return "legacy"
    if state.get("schema_generation") == CURRENT_SCHEMA_GENERATION:
        # A side-by-side build whose backfill failed (or crashed mid-backfill)
        # must never start as "current": the data is incomplete even though
        # the schema is new. 'resuming' is the brief window in which
        # migrate-vnext --resume has unblocked a failed target; a kill -9 in
        # that window must not leave the incomplete DB openable.
        if state.get("phase") in {"failed", "backfill", "resuming"}:
            return "unknown"
        return "current"
    return "unknown"


def legacy_database_message(path: Path) -> str:
    return (
        f"Detected a legacy Memory Arbiter database at {Path(path).expanduser()}.\n"
        "This release requires a one-time structural migration. MCP will not "
        "start or modify the old database.\n"
        "Stop every process that can write to this database, then run `mema upgrade`. "
        "Use `mema upgrade --dry-run` first to inspect prerequisites and the side-by-side "
        "target. The old database is kept for rollback, but old conflict, decision, and "
        "semantic-notice history is not copied. The immediately previous evidence schema "
        "uses a fast conflict-only path; older schemas additionally require sqlite-vec, "
        "a local GGUF embedding model, and llama-cpp-python for reindexing."
    )


@contextmanager
def database_startup_lock(db_path: Path) -> Iterator[None]:
    """Serialize generation detection against concurrent first-start schema init.

    A database being created by another thread/process exposes an intermediate
    table set (memories exists, memory_evidence/migration_state not yet) that
    is indistinguishable from a legacy database, and a reader can also hit
    SQLITE_BUSY mid-creation. Hold an advisory flock across the whole
    detect-then-init sequence so concurrent startups wait for the initializing
    writer instead of misclassifying the half-built file. The lock file is a
    tiny persistent ``<db>.startup.lock`` sidecar; the kernel releases it if
    the holder dies.
    """
    if not _HAVE_FCNTL:  # pragma: no cover - non-POSIX fallback
        yield
        return
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".startup.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def require_current_or_new_database(path: Path) -> DatabaseGeneration:
    generation = detect_database_generation(path)
    if generation == "legacy":
        raise LegacyDatabaseError(legacy_database_message(path))
    if generation == "unknown":
        raise RuntimeError(
            f"Cannot identify the Memory Arbiter database at {Path(path).expanduser()}. "
            "Run `mema doctor --json` before opening it."
        )
    return generation
