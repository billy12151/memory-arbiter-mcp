"""Shared doctor data model and helper functions.

Dependency-only module: imports no doctor check implementations. Both
``memory_arbiter.doctor`` (facade/orchestration) and ``doctor_checks.all_checks``
import from here so the extracted checks module can be imported directly without
circularly importing the facade.
"""
# mypy: disable-error-code="type-arg,no-untyped-def"
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

from ..models import utc_now_iso


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


@dataclass
class Finding:
    check_id: str
    dimension: str
    severity: Severity
    status: str
    title: str
    detail: str
    evidence: dict = field(default_factory=dict)
    fix_hint: str = ""
    doc_link: str = ""


@dataclass
class OverviewReport:
    snapshot_ts: str
    overall: Severity
    findings: list[Finding]
    summary: dict


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _scalar(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
    row = conn.execute(sql, tuple(params)).fetchone()
    if row is None:
        return None
    return row[0]


def _max_severity(findings: list[Finding]) -> Severity:
    if not findings:
        return Severity.INFO
    return max((f.severity for f in findings), key=lambda s: _SEVERITY_RANK[s])


def _na(check_id: str, dimension: str, reason: str) -> Finding:
    return Finding(
        check_id=check_id, dimension=dimension, severity=Severity.INFO,
        status="n/a", title=f"{check_id}: 不适用",
        detail=reason, evidence={},
    )


def _read_scan_log_last_completed(path) -> Optional[dict]:
    import json as _json
    try:
        if not path.exists():
            return None
        last_completed: Optional[dict] = None
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(rec, dict) and rec.get("status") == "completed":
                    last_completed = rec
        return last_completed
    except OSError:
        return None


def _days_since_iso(iso_ts: str) -> Optional[int]:
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - dt).days)
    except (ValueError, TypeError):
        return None


def _read_attention_log_counts(path, since_days: int = 7) -> dict:
    by_source: dict[str, int] = {}
    by_trigger: dict[str, int] = {}
    by_source_trigger: dict[str, dict[str, int]] = {}
    total = 0
    p = Path(path)
    if not p.exists():
        return {"total": 0, "window_days": since_days, "by_source": {},
                "by_trigger": {}, "by_source_trigger": {}}
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(rec, dict):
                    continue
                ts = rec.get("ts")
                days = _days_since_iso(ts) if isinstance(ts, str) else None
                if days is None or days > since_days:
                    continue
                src = str(rec.get("source", "unknown"))
                trig = str(rec.get("trigger", "unknown"))
                by_source[src] = by_source.get(src, 0) + 1
                by_trigger[trig] = by_trigger.get(trig, 0) + 1
                sub = by_source_trigger.setdefault(src, {})
                sub[trig] = sub.get(trig, 0) + 1
                total += 1
    except OSError:
        pass
    return {"total": total, "window_days": since_days, "by_source": by_source,
            "by_trigger": by_trigger, "by_source_trigger": by_source_trigger}
