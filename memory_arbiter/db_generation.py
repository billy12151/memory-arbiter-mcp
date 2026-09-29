"""Read-only database generation detection used before runtime startup."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

try:
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows lacks fcntl; startup stays best-effort
    _HAVE_FCNTL = False


DatabaseGeneration = Literal["missing", "empty", "current", "legacy", "unknown"]
CURRENT_SCHEMA_GENERATION = "workspace_state_v1"
PREVIOUS_SCHEMA_GENERATIONS = frozenset({
    "conflict_groups_v2", "local_text_evidence_v1",
})


@dataclass(frozen=True)
class SchemaMigrationDefinition:
    """Compatibility metadata for one explicit structural migration."""

    source_generation: str
    target_generation: str
    vector_effect: Literal["preserve", "rebuild"] = "preserve"


SCHEMA_MIGRATIONS = {
    generation: SchemaMigrationDefinition(
        source_generation=generation,
        target_generation=CURRENT_SCHEMA_GENERATION,
        vector_effect="preserve",
    )
    for generation in PREVIOUS_SCHEMA_GENERATIONS
}
# The single identity of the running conflict-detection pipeline (deterministic
# rules + Qwen pair extraction). The scan-clearing gate compares the PERSISTED
# requirement against this running constant, and scan candidate keys stamp it;
# a future detector change bumps this once and old-detector scans can no longer
# clear conflict_scan_required.
# v2 (0.16.0): the scan pipeline switched to server-orchestrated rank pairing
# with same-memory internal examination and queue-landed suspects (plan #971
# E10/E11) — a real detection-semantic change, so the epoch re-arms a full
# round (watermark-NULL arm + this stamped identity).
# v3 (0.16.2): difference-based clearance at enqueue time, top-3 rank gate for
# machine-decidable routes, proportional normalize gate, and the write-time
# pre-gates with unified internal Qwen slot extraction (plan #977 §1.1-§1.9)
# — same treatment: the epoch re-arms one full round under the new semantics.
# v4 (0.17.0): row-level conflict channel + attr-vector gate + process-record
# subject guard + lineage veto — a detection-semantic change (plan §10: the
# unified bump retires unit-keyed scan rows and re-arms one full round,
# co-collecting the P2-1 误杀对 revival).
# v6 (owner 2026-09-25): row-context envelope — A-cross and channel-C peer
# envelopes carry heading+neighbour context for Qwen attribute recovery
# (pair-v10); prompt/detection semantics changed, so the detector version
# re-arms one full scan round per the standing release contract.
# v2: backlog drain 判定输入上下文化（方案补充⑤）
# v3: 722eaee row_window 判定输入定版（H1 序/邻行 80 帽复原/每侧预算
#     165→300）——判定输入形态变化必须换代（d13280d 自立的换代硬约束），
#     旧扫描判定结果按旧输入口径作出，re-arm 一轮全量重扫按新口径重判。
CONFLICT_DETECTOR_VERSION = "mdeberta-v4m-v3"
# 0.17.1（owner 2026-09-29 拍板）：memory_claims/memory_claim_vec 已全退——
# additive.py 启动幂等 DROP（0.17.0 时代的 LIVE 表注释随之作废）。若有
# legacy tombstone 命中同名，按 0.17.0 附录 C-11 裁定语义作废。
LEGACY_DERIVED_TABLES = {
    "memories_vec", "memory_sections_vec",
}


class LegacyDatabaseError(RuntimeError):
    """Raised before current code can initialize or mutate a legacy database."""


def detect_database_generation(path: Path) -> DatabaseGeneration:
    """Classify a database with one constant-size migration-state query."""
    path = Path(path).expanduser()
    if not path.exists():
        return "missing"
    try:
        if path.stat().st_size == 0:
            return "empty"
    except OSError:
        return "unknown"
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            state = {
                str(row[0]): str(row[1])
                for row in conn.execute(
                    "SELECT key,value FROM migration_state "
                    "WHERE key IN ('schema_generation','phase')"
                )
            }
        finally:
            conn.close()
    except sqlite3.Error:
        return "unknown"
    phase = state.get("phase")
    if phase is not None and phase != "ready":
        return "unknown"
    if state.get("schema_generation") in PREVIOUS_SCHEMA_GENERATIONS:
        return "legacy"
    if state.get("schema_generation") == CURRENT_SCHEMA_GENERATION:
        # phase=ready is an accepted legacy success receipt. New migrations
        # remove phase when they atomically publish the generation marker.
        return "current"
    return "unknown"


def detect_upgrade_source_generation(path: Path) -> DatabaseGeneration:
    """Classify old pre-generation databases for the low-frequency CLI path."""
    generation = detect_database_generation(path)
    if generation in {"missing", "empty", "legacy"}:
        return generation
    try:
        with sqlite3.connect(f"file:{Path(path).expanduser()}?mode=ro", uri=True) as conn:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('memories','migration_state',"
                    "'memories_vec','memory_sections_vec')"
                )
            }
    except sqlite3.Error:
        return "unknown"
    if tables & LEGACY_DERIVED_TABLES:
        return "legacy"
    if generation == "unknown" and "migration_state" not in tables and "memories" in tables:
        return "legacy"
    return generation


def legacy_database_message(path: Path) -> str:
    return (
        f"Detected a legacy Memory Arbiter database at {Path(path).expanduser()}.\n"
        "This release requires a one-time structural migration. MCP will not "
        "start or modify the old database.\n"
        "Stop every process that can write to this database, then run `mema upgrade`. "
        "Use `mema upgrade --dry-run` first to inspect prerequisites and the side-by-side "
        "target. The old database is kept for rollback, but old conflict, decision, and "
        "semantic-notice history is not copied. Each schema migration declares whether "
        "vectors are preserved or rebuilt. A preserved but incompatible vector space is "
        "disabled and repaired separately; it does not block the structural migration."
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
