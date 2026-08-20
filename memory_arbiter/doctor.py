"""Read-only health checks for the local-text evidence architecture."""
from __future__ import annotations
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .config import Settings
from .degrade import DegradeState
from .models import utc_now_iso


class Severity(str, Enum):
    INFO = "info"; WARNING = "warning"; CRITICAL = "critical"


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
    open_notices = int(conn.execute("SELECT COUNT(*) FROM semantic_notices WHERE status='open'").fetchone()[0])
    open_conflicts = int(conn.execute("SELECT COUNT(*) FROM conflicts WHERE status='open'").fetchone()[0])
    findings.append(_finding("config.writable", runtime_state is None or runtime_state.sqlite_writable, "SQLite writable", critical=True))
    findings.append(_finding("evidence.coverage", indexed == eligible or not settings.embedding_auto_write, f"{indexed}/{eligible} memories indexed", evidence={"indexed": indexed, "eligible": eligible, "non_indexable": counts["non_indexable_memories"], "units": units}))
    findings.append(_finding("evidence.freshness", stale == 0, f"{stale} stale evidence rows", evidence={"stale": stale}))
    findings.append(_finding("evidence.orphans", orphan == 0, f"{orphan} orphan evidence rows", evidence={"orphan": orphan}))
    findings.append(_finding("conflicts.backlog", open_conflicts < 100, f"{open_conflicts} open conflicts"))
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
    if settings.config_warnings:
        findings.append(_finding("config.warnings", False, "; ".join(settings.config_warnings)))
    if deep:
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
            return run_all_checks(conn, settings, deep, embedder_probe=_cli_embedder_probe if deep else None)
    except Exception as exc:
        return OverviewReport(utc_now_iso(), Severity.CRITICAL, [Finding("database.open", "database", Severity.CRITICAL, "error", "database.open", str(exc))], {"mode": "unavailable", "total_memories": 0})


def build_unopenable_report(settings: Settings, exc: Exception) -> OverviewReport:
    return OverviewReport(utc_now_iso(), Severity.CRITICAL, [Finding("database.open", "database", Severity.CRITICAL, "error", "database.open", str(exc), {"exists": os.path.exists(settings.db_path)})], {"mode": "unavailable", "total_memories": 0})
