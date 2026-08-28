"""Shared ISO-8601 time helpers — single implementation, many import sites.

Phase 1 (v0.12.4) consolidation. The three pre-existing Optional-returning
parsers had subtly different tolerance; they are unified here as two faithful
helpers rather than one forced merger (R2: ``arbitration.parse_time`` is
deliberately NON-Optional and is NOT folded in).

  * ``parse_iso8601`` — the ``update_monitor._parse_time`` / ``search._parse_time``
    semantics: returns None on falsy/unparseable; naive → UTC (via replace, no
    astimezone rebase); does NOT rewrite the trailing ``Z``.
  * ``parse_iso8601_utc`` — the ``search._parse_ingest_time`` semantics: also
    rewrites ``Z`` → ``+00:00`` and rebases to UTC via ``astimezone``.

Both return None (never raise) for bad input. Original locations keep re-export
aliases so existing imports keep working.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (microseconds stripped).

    Single source; ``models.utc_now_iso`` re-exports this.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso8601(value: Any) -> datetime | None:
    """Parse an ISO-8601 string; None on falsy/unparseable. Naive → UTC.

    Mirrors the lenient ``search._parse_time`` / ``update_monitor._parse_time``
    contract: accepts non-string input via ``str()``; never raises. Does NOT
    rewrite a trailing ``Z`` (the stdlib fromisoformat handles it on 3.11+);
    naive datetimes are pinned to UTC by ``replace`` (no astimezone rebase).
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value) if isinstance(value, str) else datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_iso8601_utc(value: Any) -> datetime | None:
    """Parse an ISO-8601 string to an aware UTC datetime; None on bad input.

    Mirrors ``search._parse_ingest_time``: rewrites a trailing ``Z`` and rebases
    the result to UTC via ``astimezone`` (so +05:00 input becomes the equivalent
    UTC instant, unlike :func:`parse_iso8601` which keeps the original offset).
    """
    raw = value or ""
    if not raw:
        return None
    try:
        normalized = str(raw).replace("Z", "+00:00")
        ts = datetime.fromisoformat(normalized)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None
