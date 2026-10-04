"""doctor 共享类型与 helper（从 doctor.py 搬出，拆分批 ⑦ 纯移动，断环叶子）。"""
from __future__ import annotations
import json
import sqlite3
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..constants import (
    is_default_workspace_term,
)
from ..degrade import DegradeState


def load_confirmed_workspaces(settings: Settings) -> frozenset[str]:
    """Confirmed-registry names from the workspace_review sidecar.

    Single parse shared by workspace.review and the scan-side workspace-suspect
    generators (0.17.x prompt suppression): a move proposal whose two buckets
    are BOTH confirmed is never generated. Missing/corrupt sidecar -> empty
    set (fail-open: proposals flow exactly as before the first confirm).
    Reserved default terms are never returned, so a proposal with default as
    either endpoint is never suppressed here.
    """
    sidecar = Path(settings.db_path).parent / WORKSPACE_REVIEW_SIDECAR
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return frozenset()
    raw = data.get("confirmed_workspaces") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(
        str(name) for name in raw
        if isinstance(name, str) and not is_default_workspace_term(str(name))
    )


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
    confirmed = sorted(load_confirmed_workspaces(settings))
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


def _last_completed_scan(settings: Settings) -> dict[str, Any] | None:
    """Newest completed entry of scan_log.jsonl (file-level, doctor-local).

    Mirrors AuditStore.scan_log_last_completed's tail-read selection so the
    notice trigger and the doctor finding agree on what counts as scan activity.
    """
    path = Path(settings.db_path).parent / "scan_log.jsonl"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            # Tail-read the last one or two lines so a trailing newline does
            # not hide the final record.
            last_lines = deque(fh, maxlen=2)
        for line in reversed(last_lines):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                # A torn trailing line (crash mid-append) must not hide
                # the valid completed record before it.
                continue
            if isinstance(record, dict) and record.get("status") == "completed":
                return record
    except OSError:
        return None
    return None


@dataclass
class _DoctorCtx:
    """Shared inputs plus the base metrics every run needs.

    ``collect`` runs exactly the DB queries the pre-split implementation ran up
    front, in the same order; the one deliberate reordering is that the
    scan_log.jsonl file read now happens after the has_scan_progress query
    instead of before it (both are pure reads, so nothing observable changes).
    Everything else stays inside the check that uses it: hoisting a
    single-consumer query would change nothing except when it executes, and
    the probe below is the standing proof that query timing here is
    observable.
    """

    conn: sqlite3.Connection
    settings: Settings
    deep: bool
    runtime_state: DegradeState | None
    embedder_probe: Callable[[], tuple[Any, list[str]]] | None
    total: int
    counts: dict[str, int]
    eligible: int
    indexed: int
    units: int
    # Named apart from the boolean `stale` inside conflicts.scan_stale: the
    # pre-split code reused one name for two unrelated values and only avoided
    # a bug because of statement order.
    evidence_stale: int
    orphan: int
    open_notices: int
    open_conflicts: int
    applying_rows: list[Any]
    conflict_scan_required: bool
    last_scan: dict[str, Any] | None
    has_scan_progress: bool

    @classmethod
    def collect(
        cls,
        conn: sqlite3.Connection,
        settings: Settings,
        deep: bool,
        runtime_state: DegradeState | None,
        embedder_probe: Callable[[], tuple[Any, list[str]]] | None,
    ) -> "_DoctorCtx":
        # Function-local like the original: importing at module scope would
        # change import ordering for a module this one is only loosely tied to.
        from ..db.evidence_store import indexable_coverage_counts

        total = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        counts = indexable_coverage_counts(conn)
        # C5: these counters read the row store; a pre-additive database (no
        # memory_row yet) reports zeros instead of crashing the whole check.
        try:
            units = int(conn.execute("SELECT COUNT(*) FROM memory_row").fetchone()[0])
            evidence_stale = int(conn.execute("SELECT COUNT(*) FROM memory_row e JOIN memories m ON m.id=e.memory_id WHERE e.memory_version<>m.version").fetchone()[0])
            orphan = int(conn.execute("SELECT COUNT(*) FROM memory_row e WHERE NOT EXISTS(SELECT 1 FROM memories m WHERE m.id=e.memory_id)").fetchone()[0])
        except sqlite3.OperationalError:
            units = evidence_stale = orphan = 0
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
        has_scan_progress = conn.execute(
            "SELECT 1 FROM migration_state WHERE key='conflict_scan_progress' AND value != ''"
        ).fetchone() is not None
        return cls(
            conn=conn, settings=settings, deep=deep, runtime_state=runtime_state,
            embedder_probe=embedder_probe,
            total=total, counts=counts,
            eligible=counts["eligible_memories"], indexed=counts["indexed_memories"],
            units=units, evidence_stale=evidence_stale, orphan=orphan,
            open_notices=open_notices, open_conflicts=open_conflicts,
            applying_rows=list(applying_rows),
            conflict_scan_required=bool(scan_required_row and str(scan_required_row[0]) == "true"),
            last_scan=_last_completed_scan(settings),
            has_scan_progress=has_scan_progress,
        )


# A check returns the finding it produced, or None when it has nothing to
# say. Several are genuinely conditional -- scan_epoch, scan_stale, scan_chain,
# spec_drift and config.warnings can all emit nothing on a healthy library.
