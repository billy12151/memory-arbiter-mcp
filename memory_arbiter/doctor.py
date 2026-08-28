"""Read-only health checks for the local-text evidence architecture."""
from __future__ import annotations
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .config import Settings
from .constants import is_default_workspace_term
from .db_generation import detect_database_generation, detect_upgrade_source_generation
from .degrade import DegradeState
from .models import utc_now_iso


class Severity(str, Enum):
    INFO = "info"; WARNING = "warning"; CRITICAL = "critical"


WORKSPACE_REVIEW_SIDECAR = "workspace_review.json"


def _workspace_review_finding(conn: sqlite3.Connection, settings: Settings) -> Finding:
    """workspace.review — full-registry confirmation diff.

    Diffs workspace_canonicals (minus reserved default terms) against the
    workspace_review.json sidecar that ONLY the authorized
    memory_govern(action='confirm_workspaces') action writes. One-way diff:
    new canonicals surface for review; names that disappeared (merged away,
    renamed) are silently ignored. Missing or corrupt sidecar = empty
    snapshot = first full review. Read-only — a doctor run must never refresh
    the snapshot, or an unattended routine run would silently mark unreviewed
    workspaces confirmed. The finding is WARNING (never critical); the CLI
    exits 1 while this finding is active. After a full-registry confirmation
    this check passes; unrelated warnings may still keep the overall exit at 1.
    """
    sidecar = Path(settings.db_path).parent / WORKSPACE_REVIEW_SIDECAR
    current: list[str] = []
    try:
        rows = conn.execute(
            "SELECT name FROM workspace_canonicals ORDER BY name"
        ).fetchall()
        current = [
            str(row["name"]) for row in rows
            if not is_default_workspace_term(str(row["name"]))
        ]
    except sqlite3.Error:
        # Registry unreadable (legacy shape): report pass so a read hiccup
        # can't mask the real findings or wedge the exit code.
        return _finding("workspace.review", True, "workspace registry unavailable; review skipped")
    confirmed: list[str] = []
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            data = None
        raw_confirmed = data.get("confirmed_workspaces") if isinstance(data, dict) else None
        if isinstance(raw_confirmed, list):
            confirmed = [
                str(name) for name in raw_confirmed
                if isinstance(name, str) and not is_default_workspace_term(name)
            ]
    new_items = sorted(set(current) - set(confirmed))
    evidence = {
        "confirmed": len(confirmed),
        "current": current,
        "new": new_items,
        "sidecar": str(sidecar),
    }
    if not new_items:
        return _finding(
            "workspace.review", True,
            f"{len(current)} workspace(s) confirmed (registry matches snapshot)",
            evidence=evidence,
        )
    return _finding(
        "workspace.review", False,
        f"{len(new_items)} unconfirmed workspace(s): {', '.join(new_items)}. "
        f"Full registry ({len(current)}): {', '.join(current)}. Merge duplicates via "
        "memory_govern(action='rename_workspace_canonical'), then record the reviewed "
        "set with memory_govern(action='confirm_workspaces', authorized=true). A name "
        "reappearing here after a rename has an existing keep-separate decision "
        "that blocked old-name forwarding; merge it deliberately or confirm it "
        "as its own workspace.",
        evidence=evidence,
    )


@dataclass
class Finding:
    check_id: str; dimension: str; severity: Severity; status: str; title: str
    detail: str = ""; evidence: dict[str, Any] | None = None; fix_hint: str | None = None


@dataclass
class OverviewReport:
    snapshot_ts: str; overall: Severity; findings: list[Finding]; summary: dict[str, Any]


def _finding(check_id: str, ok: bool, detail: str, *, critical: bool = False, evidence: dict[str, Any] | None = None) -> Finding:
    severity = Severity.INFO if ok else (Severity.CRITICAL if critical else Severity.WARNING)
    return Finding(check_id, check_id.split(".")[0], severity, "pass" if ok else "warn", check_id, detail, evidence or {})


def _safe_count(conn: sqlite3.Connection, source: str, where: str | None = None) -> int | None:
    """Count an optional vec0 source, preserving unavailable as unknown."""
    try:
        suffix = f" WHERE {where}" if where else ""
        return int(conn.execute(f"SELECT COUNT(*) FROM {source}{suffix}").fetchone()[0])
    except sqlite3.Error:
        return None


def _vec_table_dimension(conn: sqlite3.Connection, table: str) -> int | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if row is None:
        return None
    match = re.search(r"embedding\s+float\[(\d+)]", str(row[0] or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def run_all_checks(conn: sqlite3.Connection, settings: Settings, deep: bool = False, runtime_state: Optional[DegradeState] = None, embedder_probe: Optional[Callable[[], tuple[Any, list[str]]]] = None) -> OverviewReport:
    findings: list[Finding] = []
    from .db.evidence_store import indexable_coverage_counts

    total = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    counts = indexable_coverage_counts(conn)
    eligible = counts["eligible_memories"]
    indexed = counts["indexed_memories"]
    units = int(conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0])
    stale = int(conn.execute("SELECT COUNT(*) FROM memory_evidence e JOIN memories m ON m.id=e.memory_id WHERE e.memory_version<>m.version").fetchone()[0])
    orphan = int(conn.execute("SELECT COUNT(*) FROM memory_evidence e WHERE NOT EXISTS(SELECT 1 FROM memories m WHERE m.id=e.memory_id)").fetchone()[0])
    open_notices = int(conn.execute(
        "SELECT COUNT(*) FROM conflicts WHERE notice_type IS NOT NULL "
        "AND notice_delivery_status IN ('pending','delivered')"
    ).fetchone()[0])
    open_conflicts = int(conn.execute("SELECT COUNT(*) FROM conflicts WHERE status='open'").fetchone()[0])
    applying_rows = conn.execute(
        "SELECT id,refreshed_at FROM conflicts WHERE status='applying' ORDER BY refreshed_at,id"
    ).fetchall()
    scan_required_row = conn.execute(
        "SELECT value FROM migration_state WHERE key='conflict_scan_required'"
    ).fetchone()
    conflict_scan_required = bool(scan_required_row and str(scan_required_row[0]) == "true")
    findings.append(_finding("config.writable", runtime_state is None or runtime_state.sqlite_writable, "SQLite writable", critical=True))
    findings.append(_finding("evidence.coverage", indexed == eligible or not settings.embedding_auto_write, f"{indexed}/{eligible} memories indexed", evidence={"indexed": indexed, "eligible": eligible, "non_indexable": counts["non_indexable_memories"], "units": units}))
    findings.append(_finding("evidence.freshness", stale == 0, f"{stale} stale evidence rows", evidence={"stale": stale}))
    findings.append(_finding("evidence.orphans", orphan == 0, f"{orphan} orphan evidence rows", evidence={"orphan": orphan}))
    unresolved_conflicts = open_conflicts + len(applying_rows)
    findings.append(_finding(
        "conflicts.backlog", unresolved_conflicts < 100,
        f"{unresolved_conflicts} unresolved conflicts ({open_conflicts} open, {len(applying_rows)} applying)",
        evidence={"open": open_conflicts, "applying": len(applying_rows)},
    ))
    findings.append(_finding(
        "conflicts.scan_required", not conflict_scan_required,
        "complete a full matching-detector conflict scan" if conflict_scan_required
        else "conflict rebuild scan complete",
    ))
    # Applying is a transient execution state: a healthy apply completes in
    # minutes, so any group still applying at doctor time is either mid-flight
    # or wedged. Flag every one with id/idle-days evidence (replaces the
    # removed stale-applying list surfacing); agent steers replan or resolve.
    now = datetime.now(timezone.utc)
    applying_groups: list[dict[str, Any]] = []
    for row in applying_rows[:10]:
        refreshed = str(row[1] or "")
        idle_days: Optional[int] = None
        try:
            refreshed_at = datetime.fromisoformat(refreshed)
            if refreshed_at.tzinfo is None:
                refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
            idle_days = max(0, (now - refreshed_at).days)
        except ValueError:
            pass
        applying_groups.append({"id": int(row[0]), "refreshed_at": refreshed, "idle_days": idle_days})
    findings.append(_finding(
        "conflicts.applying", not applying_rows,
        f"{len(applying_rows)} applying conflict group(s) awaiting completion",
        evidence={"groups": applying_groups},
    ))
    findings.append(_finding("notices.backlog", open_notices < 100, f"{open_notices} open notices"))
    # v0.8.8 observability (restored): searches that ring conflict signals
    # append to attention_log.jsonl; surfacing the volume keeps advisory
    # flooding visible instead of growing an unread log forever.
    attention_lines = 0
    attention_recent = 0
    attention_path = Path(settings.db_path).parent / "attention_log.jsonl"
    if attention_path.exists():
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            with open(attention_path, encoding="utf-8") as fh:
                for line in fh:
                    attention_lines += 1
                    try:
                        if str(json.loads(line).get("ts", "")) >= cutoff:
                            attention_recent += 1
                    except json.JSONDecodeError:
                        continue
        except OSError:
            attention_lines = 0
    findings.append(_finding(
        "capacity.attention_volume", attention_recent < 200,
        f"{attention_recent} attention events in 7 days ({attention_lines} total)",
        evidence={"recent_7d": attention_recent, "total_lines": attention_lines},
    ))
    # full-registry workspace confirmation (read-only; the snapshot
    # is only ever written by the authorized confirm_workspaces action).
    findings.append(_workspace_review_finding(conn, settings))
    if settings.config_warnings:
        findings.append(_finding("config.warnings", False, "; ".join(settings.config_warnings)))
    if deep:
        quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        findings.append(_finding(
            "database.quick_check", quick_check == "ok", quick_check,
            critical=quick_check != "ok",
        ))
        migration = {
            str(row[0]): str(row[1])
            for row in conn.execute(
                "SELECT key,value FROM migration_state "
                "WHERE key IN ('schema_generation','phase','migration_completed_at')"
            )
        }
        generation_ok = (
            migration.get("schema_generation") == "workspace_state_v1"
            and migration.get("phase") not in {"building", "backfill", "resuming", "failed"}
        )
        findings.append(_finding(
            "database.schema_generation", generation_ok,
            f"generation={migration.get('schema_generation') or 'missing'}, "
            f"phase={migration.get('phase') or 'complete'}",
            critical=not generation_ok,
            evidence=migration,
        ))
        vec_meta = {
            str(row[0]): str(row[1])
            for row in conn.execute("SELECT key,value FROM _vec_index_meta")
        }
        active_space = vec_meta.get("active_space_id")
        configured_space: str | None = None
        try:
            from .vnext_migration import _configured_embedding_space_id
            configured_space = _configured_embedding_space_id(settings)
        except (OSError, ValueError):
            configured_space = None
        space_ok = (
            configured_space is None
            or (vec_meta.get("state") == "ready" and active_space == configured_space)
        )
        findings.append(_finding(
            "vector.space", space_ok,
            f"state={vec_meta.get('state') or 'unmanaged'}, "
            f"active={active_space or 'none'}, configured={configured_space or 'none'}",
            evidence={
                "state": vec_meta.get("state", "unmanaged"),
                "active_space_id": active_space,
                "configured_space_id": configured_space,
            },
        ))
        evidence_dim = _vec_table_dimension(conn, "memory_evidence_vec")
        workspace_dim = _vec_table_dimension(conn, "workspace_canonicals_vec")
        dimensions_ok = (
            not settings.enable_sqlite_vec
            or (evidence_dim == int(settings.vec_dim) and workspace_dim == int(settings.vec_dim))
        )
        findings.append(_finding(
            "vector.table_dimension", dimensions_ok,
            f"evidence={evidence_dim}, workspace={workspace_dim}, configured={int(settings.vec_dim)}",
            evidence={"evidence": evidence_dim, "workspace": workspace_dim,
                      "configured": int(settings.vec_dim)},
        ))
        evidence_vectors = _safe_count(conn, "memory_evidence_vec")
        orphan_vectors = _safe_count(
            conn, "memory_evidence_vec v LEFT JOIN memory_evidence e ON e.id=v.id",
            "e.id IS NULL",
        )
        missing_vectors = _safe_count(
            conn, "memory_evidence e LEFT JOIN memory_evidence_vec v ON v.id=e.id",
            "v.id IS NULL",
        )
        evidence_rows_ok = (
            not settings.enable_sqlite_vec
            or (
                evidence_vectors is not None
                and orphan_vectors == 0
                and missing_vectors == 0
            )
        )
        findings.append(_finding(
            "vector.evidence_rows",
            evidence_rows_ok,
            f"{units} evidence rows, {evidence_vectors} vectors, "
            f"{orphan_vectors} orphan vectors, {missing_vectors} missing vectors",
            evidence={"evidence": units, "vectors": evidence_vectors,
                      "orphan_vectors": orphan_vectors, "missing_vectors": missing_vectors},
        ))
        canonical_count = int(conn.execute(
            "SELECT COUNT(*) FROM workspace_canonicals WHERE lower(trim(name)) "
            "NOT IN ('','default','none','null','unknown') AND trim(name) NOT IN ('默认','未知')"
        ).fetchone()[0])
        canonical_vectors = _safe_count(conn, "workspace_canonicals_vec")
        workspace_rows_ok = (
            not settings.enable_sqlite_vec
            or (canonical_vectors is not None and canonical_vectors == canonical_count)
        )
        findings.append(_finding(
            "vector.workspace_rows",
            workspace_rows_ok,
            f"{canonical_count} non-default canonicals, {canonical_vectors} vectors",
            evidence={"canonicals": canonical_count, "vectors": canonical_vectors},
        ))
        # --deep / memory_review(deep=true): actually run the embedder and
        # compare the live dimension against vec.dim (seconds-level cost).
        embedding_configured = (
            settings.embedding_provider == "gguf"
            and settings.embedding_model_path is not None
            and settings.enable_sqlite_vec
        )
        if embedder_probe is None:
            findings.append(_finding("vector.dimension_probe", False, "deep probe requested but no embedder resolver was provided"))
        else:
            try:
                embedder, _probe_warnings = embedder_probe()
            except Exception as exc:
                findings.append(_finding("vector.dimension_probe", False, f"embedder probe failed: {exc}"))
            else:
                if embedder is None:
                    if embedding_configured:
                        findings.append(_finding("vector.dimension_probe", False, "embedding configured but the embedder is currently unavailable"))
                    else:
                        # Asking for a probe without embedding configured is a
                        # no-op, not a health problem — must not force exit 1.
                        findings.append(_finding("vector.dimension_probe", True, "deep probe skipped: embedding not configured"))
                else:
                    try:
                        er = embedder.embed_text(prefix="", body="dimension probe")
                        dim = len(er.embedding or [])
                    except Exception as exc:
                        findings.append(_finding("vector.dimension_probe", False, f"embedding failed: {exc}"))
                    else:
                        findings.append(_finding(
                            "vector.dimension_probe", dim == int(settings.vec_dim),
                            f"embedding dim {dim} vs config vec.dim {int(settings.vec_dim)}",
                        ))
    overall = max((f.severity for f in findings), key=lambda s: {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}[s])
    return OverviewReport(utc_now_iso(), overall, findings, {"mode": runtime_state.mode if runtime_state else "sqlite", "total_memories": total, "evidence_indexed": indexed, "evidence_units": units})


def report_to_dict(report: OverviewReport) -> dict[str, Any]:
    data = asdict(report); data["overall"] = report.overall.value
    for item in data["findings"]: item["severity"] = item["severity"].value if isinstance(item["severity"], Severity) else item["severity"]
    return data


def doctor_overview_mcp(db: Any, settings: Settings, deep: bool = False, **kwargs: Any) -> OverviewReport:
    try:
        with db.diagnostic_connection() as conn:
            return run_all_checks(conn, settings, deep, kwargs.get("runtime_state"), kwargs.get("embedder_probe"))
    except Exception as exc:
        return OverviewReport(utc_now_iso(), Severity.CRITICAL, [Finding("database.open", "database", Severity.CRITICAL, "error", "database.open", str(exc))], {"mode": "unavailable", "total_memories": 0})


@contextmanager
def open_ro_connection(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True); conn.row_factory = sqlite3.Row
    try: yield conn
    finally: conn.close()


def doctor_overview_cli(settings: Settings, deep: bool = False) -> OverviewReport:
    generation = detect_upgrade_source_generation(settings.db_path)
    if generation == "legacy":
        return OverviewReport(
            utc_now_iso(),
            Severity.CRITICAL,
            [Finding(
                "database.upgrade_required",
                "database",
                Severity.CRITICAL,
                "error",
                "database.upgrade_required",
                "legacy database generation; current code will not open or modify it",
                {"generation": generation, "path": str(settings.db_path)},
                "Stop all writers, then run `mema upgrade --dry-run` before `mema upgrade`.",
            )],
            {"mode": "upgrade_required", "total_memories": 0},
        )

    def _cli_embedder_probe() -> tuple[Any, list[str]]:
        # The CLI ambulance path has no MemoryTools; build the embedder
        # directly from settings so --deep actually probes the model its
        # help text promises (read-only; no DB access needed).
        if (
            settings.embedding_provider != "gguf"
            or settings.embedding_model_path is None
            or not settings.enable_sqlite_vec
        ):
            return None, []
        try:
            from .embedder import build_embedder
            return build_embedder(
                str(settings.embedding_model_path),
                settings.vec_dim,
                n_ctx=settings.embedding_n_ctx,
                reserved_tokens=settings.embedding_reserved_tokens,
                max_section_chars=settings.max_section_chars,
            )
        except Exception:
            return None, []

    try:
        with open_ro_connection(settings.db_path) as conn:
            if settings.enable_sqlite_vec:
                try:
                    import sqlite_vec

                    conn.enable_load_extension(True)
                    sqlite_vec.load(conn)
                    conn.enable_load_extension(False)
                except (ImportError, sqlite3.Error):
                    pass
            return run_all_checks(conn, settings, deep, embedder_probe=_cli_embedder_probe if deep else None)
    except Exception as exc:
        return OverviewReport(utc_now_iso(), Severity.CRITICAL, [Finding("database.open", "database", Severity.CRITICAL, "error", "database.open", str(exc))], {"mode": "unavailable", "total_memories": 0})


def build_unopenable_report(settings: Settings, exc: Exception) -> OverviewReport:
    return OverviewReport(utc_now_iso(), Severity.CRITICAL, [Finding("database.open", "database", Severity.CRITICAL, "error", "database.open", str(exc), {"exists": os.path.exists(settings.db_path)})], {"mode": "unavailable", "total_memories": 0})
