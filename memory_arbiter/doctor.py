"""Read-only health checks for the local-text evidence architecture."""
from __future__ import annotations
import functools
import json
import os
import sqlite3
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import Settings
from .constants import (
    EMBEDDING_MAX_SECTION_CHARS,
    EMBEDDING_N_CTX,
    EMBEDDING_RESERVED_TOKENS,
    SCAN_CHAIN_STALE_HOURS,
    SCAN_TASK_STALE_DAYS,
    is_default_workspace_term,
)
from .db.meta import active_dim_on_connection, vec_table_dimension
from .db_generation import detect_upgrade_source_generation
from .degrade import DegradeState
from .models import utc_now_iso
from .timeutil import parse_iso8601_utc


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
        from .db.evidence_store import indexable_coverage_counts

        total = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        counts = indexable_coverage_counts(conn)
        units = int(conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0])
        evidence_stale = int(conn.execute("SELECT COUNT(*) FROM memory_evidence e JOIN memories m ON m.id=e.memory_id WHERE e.memory_version<>m.version").fetchone()[0])
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
_Check = Callable[[_DoctorCtx], "Finding | None"]


def _c_config_writable(ctx: _DoctorCtx) -> Finding:
    return _finding("config.writable", ctx.runtime_state is None or ctx.runtime_state.sqlite_writable, "SQLite writable", critical=True)


def _c_evidence_coverage(ctx: _DoctorCtx) -> Finding:
    return _finding("evidence.coverage", ctx.indexed == ctx.eligible or not ctx.settings.embedding_auto_write, f"{ctx.indexed}/{ctx.eligible} memories indexed", evidence={"indexed": ctx.indexed, "eligible": ctx.eligible, "non_indexable": ctx.counts["non_indexable_memories"], "units": ctx.units})


def _c_evidence_freshness(ctx: _DoctorCtx) -> Finding:
    return _finding("evidence.freshness", ctx.evidence_stale == 0, f"{ctx.evidence_stale} stale evidence rows", evidence={"stale": ctx.evidence_stale})


def _c_evidence_orphans(ctx: _DoctorCtx) -> Finding:
    return _finding("evidence.orphans", ctx.orphan == 0, f"{ctx.orphan} orphan evidence rows", evidence={"orphan": ctx.orphan})


def _c_conflicts_backlog(ctx: _DoctorCtx) -> Finding:
    unresolved_conflicts = ctx.open_conflicts + len(ctx.applying_rows)
    # C4 (0.15.13): triage counters double as the C2 pair@version
    # suppression's convergence observability — sustained weekly growth means
    # memories keep churning versions or suppression is failing.
    dismissed_total = int(ctx.conn.execute(
        "SELECT COUNT(*) FROM conflicts WHERE status='not_a_conflict'"
    ).fetchone()[0])
    week_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    dismissed_week = int(ctx.conn.execute(
        "SELECT COUNT(*) FROM conflicts WHERE status='not_a_conflict' "
        "AND created_at >= ?", (week_cutoff,),
    ).fetchone()[0])
    dismissed_latest = ctx.conn.execute(
        "SELECT refreshed_at FROM conflicts WHERE status='not_a_conflict' "
        "ORDER BY refreshed_at DESC LIMIT 1"
    ).fetchone()
    dismissed_latest_at = str(dismissed_latest[0]) if dismissed_latest else None
    return _finding(
        "conflicts.backlog", unresolved_conflicts < 100,
        f"{unresolved_conflicts} unresolved conflicts ({ctx.open_conflicts} open, {len(ctx.applying_rows)} applying); "
        f"triage dismissed {dismissed_total} total, {dismissed_week} this week",
        evidence={
            "open": ctx.open_conflicts, "applying": len(ctx.applying_rows),
            "not_a_conflict_total": dismissed_total,
            "not_a_conflict_this_week": dismissed_week,
            "latest_triage_at": dismissed_latest_at,
        },
    )


def _c_conflicts_scan_required(ctx: _DoctorCtx) -> Finding:
    return _finding(
        "conflicts.scan_required", not ctx.conflict_scan_required,
        "complete a full matching-detector conflict scan" if ctx.conflict_scan_required
        else (
            "conflict rebuild scan complete"
            if (ctx.last_scan is not None or ctx.has_scan_progress)
            else "no completed conflict scan on record — scheduled scan tasks are not set up; see memory_repair help (topic: scheduled_tasks)"
        ),
    )


def _c_conflicts_scan_epoch(ctx: _DoctorCtx) -> Finding | None:
    # 0.16.0 §6⑪: detector-epoch arm visibility — report the identity change
    # WITH its reason (E9 ②) so an upcoming full round is explained, not
    # mysterious. A stale arm (persisted 'to' != running detector) means the
    # arm ran under an older binary; the boot path re-arms anyway.
    epoch_row = ctx.conn.execute(
        "SELECT value FROM migration_state WHERE key='scan_epoch_armed'"
    ).fetchone() if ctx.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='migration_state'"
    ).fetchone() else None
    if epoch_row is None:
        return None
    try:
        epoch_arm = json.loads(str(epoch_row[0]))
    except (TypeError, ValueError):
        epoch_arm = None
    if not isinstance(epoch_arm, dict):
        return None
    from .db_generation import CONFLICT_DETECTOR_VERSION

    armed_to = str(epoch_arm.get("to") or "")
    fresh = armed_to == CONFLICT_DETECTOR_VERSION
    return _finding(
        "conflicts.scan_epoch", fresh,
        (
            f"detector epoch armed: {epoch_arm.get('from') or 'none'} → {armed_to} "
            f"at {epoch_arm.get('at')} ({epoch_arm.get('reason')}); "
            "first full scan round pending"
        )
        if fresh
        else f"epoch arm stale (persisted to={armed_to}, running={CONFLICT_DETECTOR_VERSION})",
        evidence=epoch_arm,
    )


def _c_conflicts_scan_stale(ctx: _DoctorCtx) -> Finding | None:
    if ctx.last_scan is None:
        return None
    scanned_at = parse_iso8601_utc(ctx.last_scan.get("scan_time"))
    if scanned_at is None:
        return None
    # Same comparison basis as the guidance notice (exact seconds vs
    # the 14-day threshold, not floored .days), so tier and finding
    # cannot disagree inside day 14.
    age = datetime.now(timezone.utc) - scanned_at
    stale = age > timedelta(days=SCAN_TASK_STALE_DAYS)
    age_days = max(0, age.days)
    return _finding(
        "conflicts.scan_stale", not stale,
        f"last completed conflict scan {age_days} day(s) ago; "
        f"activity beyond {SCAN_TASK_STALE_DAYS} days means the scheduled task is not running",
        evidence={"last_scan_time": ctx.last_scan.get("scan_time"), "age_days": age_days},
    )


def _c_conflicts_scan_chain(ctx: _DoctorCtx) -> Finding | None:
    # C5 (0.15.13): broken-chain alarm over the routine scan's page-progress
    # kv. A whole chain normally completes in 15-20 minutes; an incomplete
    # record older than an hour with no completion line after it means the
    # round was interrupted mid-walk. scan_log.jsonl keeps its
    # completed-lines-only audit semantics (#825) — pacing lives in the kv.
    page_progress_row = ctx.conn.execute(
        "SELECT value FROM migration_state WHERE key='scan_page_progress'"
    ).fetchone()
    if page_progress_row is None:
        return None
    try:
        page_progress = json.loads(str(page_progress_row[0]))
    except (TypeError, ValueError):
        page_progress = None
    if not isinstance(page_progress, dict) or page_progress.get("complete"):
        return None
    progress_at = parse_iso8601_utc(str(page_progress.get("at") or ""))
    if progress_at is None:
        return None
    progress_age = datetime.now(timezone.utc) - progress_at
    completed_after = bool(
        ctx.last_scan is not None
        and str(ctx.last_scan.get("scan_time") or "") > str(page_progress.get("at") or "")
    )
    if progress_age <= timedelta(hours=SCAN_CHAIN_STALE_HOURS) or completed_after:
        return None
    return _finding(
        "conflicts.scan_chain", False,
        f"conflict scan interrupted at anchor {page_progress.get('next_anchor')} "
        f"({str(page_progress.get('at') or '')}); a full chain normally completes "
        f"in 15-20 minutes — resume paging from that anchor",
        evidence={
            "after": page_progress.get("after"),
            "next_anchor": page_progress.get("next_anchor"),
            "at": page_progress.get("at"),
            "client": page_progress.get("client"),
            "groups": page_progress.get("groups"),
        },
    )


def _c_conflicts_spec_drift(ctx: _DoctorCtx) -> Finding | None:
    # 0.16.0 §6⑩: scheduled-task spec drift. Any completed scan activity
    # proves A task exists; the spec stamp says WHICH contract it runs. Scan
    # activity without a current-version stamp = a v1 page-driven task that
    # should be rebuilt (the pipeline never got a kick).
    spec_stamp_row = ctx.conn.execute(
        "SELECT value FROM migration_state WHERE key='scheduled_tasks_spec_confirmed'"
    ).fetchone()
    if not ((ctx.last_scan is not None or ctx.has_scan_progress) and not ctx.conflict_scan_required):
        return None
    from .scan_tasks import SCHEDULED_TASKS_SPEC_VERSION

    try:
        spec_stamp = int(str(spec_stamp_row[0])) if spec_stamp_row is not None else None
    except (TypeError, ValueError):
        spec_stamp = None
    if spec_stamp == SCHEDULED_TASKS_SPEC_VERSION:
        return None
    return _finding(
        "conflicts.spec_drift", False,
        (
            f"scheduled task appears to follow spec v{spec_stamp or 1} while the "
            f"server serves v{SCHEDULED_TASKS_SPEC_VERSION}: rebuild the "
            "conflict_scan task from memory(action='help', "
            "data={'topic': 'scheduled_tasks'}) — v2 tasks kick "
            "memory_repair(task='scan_pipeline') and clear the scan_queue "
            "judgment queue instead of paging scan_candidates"
        ),
        evidence={"served_spec_version": SCHEDULED_TASKS_SPEC_VERSION,
                  "task_spec_version": spec_stamp},
    )


def _c_conflicts_applying(ctx: _DoctorCtx) -> Finding:
    # Applying is a transient execution state: a healthy apply completes in
    # minutes, so any group still applying at doctor time is either mid-flight
    # or wedged. Flag every one with id/idle-days evidence (replaces the
    # removed stale-applying list surfacing); agent steers replan or resolve.
    now = datetime.now(timezone.utc)
    applying_groups: list[dict[str, Any]] = []
    for row in ctx.applying_rows[:10]:
        refreshed = str(row[1] or "")
        idle_days: int | None = None
        try:
            refreshed_at = datetime.fromisoformat(refreshed)
            if refreshed_at.tzinfo is None:
                refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
            idle_days = max(0, (now - refreshed_at).days)
        except ValueError:
            pass
        applying_groups.append({"id": int(row[0]), "refreshed_at": refreshed, "idle_days": idle_days})
    return _finding(
        "conflicts.applying", not ctx.applying_rows,
        f"{len(ctx.applying_rows)} applying conflict group(s) awaiting completion",
        evidence={"groups": applying_groups},
    )


def _c_notices_backlog(ctx: _DoctorCtx) -> Finding:
    return _finding("notices.backlog", ctx.open_notices < 100, f"{ctx.open_notices} open notices")


def _c_conflicts_scan_queue_backlog(ctx: _DoctorCtx) -> Finding:
    # 0.16.0 §6㉑⑦: the scan judgment queue is agent-facing only — the user
    # surface is this backlog COUNT plus a hint, never the items themselves
    # (suspected items are mostly noise). Console overview reads the same
    # numbers via doctor.
    try:
        queue_rows = ctx.conn.execute(
            "SELECT status, COUNT(*) AS c FROM scan_queue GROUP BY status"
        ).fetchall()
        queue_counts = {str(row["status"]): int(row["c"]) for row in queue_rows}
    except sqlite3.Error:
        queue_counts = {}
    queue_backlog = queue_counts.get("pending", 0)
    # 0.16.2: internal contradictions live in their OWN table but count
    # toward the same agent workload — scan_queue page's queue_backlog merges
    # both, doctor must too or it under-reports (a live 954-item workload
    # read as 666).
    try:
        internal_pending = int(ctx.conn.execute(
            "SELECT COUNT(*) FROM internal_conflicts WHERE status='pending'"
        ).fetchone()[0])
    except sqlite3.Error:
        internal_pending = 0
    queue_backlog += internal_pending
    return _finding(
        "conflicts.scan_queue_backlog", queue_backlog < 100,
        (
            f"{queue_backlog} suspected item(s) awaiting agent judgment "
            f"({queue_counts.get('pending', 0)} pending, "
            f"{internal_pending} internal contradictions); "
            "when convenient, let the agent read the queue "
            "(memory_repair task='scan_queue', action='page')"
        ),
        evidence={**queue_counts, "backlog": queue_backlog, "internal_pending": internal_pending},
    )


def _c_normalize_autonomy(ctx: _DoctorCtx) -> Finding:
    # 0.16.0 §6⑫: autonomous normalization audit board — applied auto-moves,
    # rollbacks, and (never-moved) protected-bucket hints, so the owner sees
    # what the pipeline did on its own and what needs a human instead.
    try:
        applied = int(ctx.conn.execute(
            "SELECT COUNT(*) FROM normalize_audit WHERE status='applied'"
        ).fetchone()[0])
        rolled_back = int(ctx.conn.execute(
            "SELECT COUNT(*) FROM normalize_audit WHERE status='rolled_back'"
        ).fetchone()[0])
        # 0.16.3 default-fallback landings are audited per move; a growing
        # count is the owner's signal that agents are leaning on the escape
        # hatch instead of finding real buckets. 0.16.4: the count covers
        # BOTH entrances — memory_govern (status manual_move) and the
        # judgment-queue channel (status applied with the same gate flag),
        # which is the channel the spec steers agents to.
        default_fallback = int(ctx.conn.execute(
            "SELECT COUNT(*) FROM normalize_audit "
            "WHERE json_extract(gate, '$.default_fallback') = 1 "
            "AND status IN ('manual_move', 'applied')"
        ).fetchone()[0])
    except sqlite3.Error:
        applied = rolled_back = default_fallback = 0
    fallback_note = (
        f", {default_fallback} parked in default (fallback — re-home via "
        "memory_govern(action='move_memories_workspace'))"
        if default_fallback else ""
    )
    return _finding(
        "normalize.autonomy", True,
        f"{applied} autonomous move(s) applied, {rolled_back} rolled back{fallback_note}; "
        "rollback via memory_govern(action='rollback_auto_move', data={audit_id, authorized=true})",
        evidence={
            "applied": applied, "rolled_back": rolled_back,
            "default_fallback": default_fallback,
        },
    )


def _c_tags_over_limit(ctx: _DoctorCtx) -> Finding:
    # 0.16.0 §6⑮: tag-discipline backlog — pre-0.16.0 rows over the total cap
    # are NOT retro-truncated (reads unaffected); doctor lists them for a
    # manual cleanup pass (remove_tags) instead.
    from .constants import MAX_MEMORY_TOTAL_TAGS as _TAG_CAP

    try:
        over: list[dict[str, int]] = []
        for row in ctx.conn.execute(
            "SELECT id, tags FROM memories WHERE status!='deleted' AND tags != '[]'"
        ).fetchall():
            try:
                parsed = json.loads(str(row["tags"]))
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, list) and len(parsed) > _TAG_CAP:
                over.append({"id": int(row["id"]), "count": len(parsed)})
    except sqlite3.Error:
        over = []
    return _finding(
        "tags.over_limit", not over,
        (
            f"{len(over)} memory(ies) exceed the {_TAG_CAP}-tag cap: "
            + ", ".join(f"#{item['id']}({item['count']})" for item in over[:10])
            + (" …" if len(over) > 10 else "")
            + " — tags are a retrieval dimension, not an event log; trim with update remove_tags"
            if over else f"all memories within the {_TAG_CAP}-tag cap"
        ),
        evidence={"over_limit": over, "cap": _TAG_CAP},
    )


def _c_capacity_attention_volume(ctx: _DoctorCtx) -> Finding:
    # v0.8.8 observability (restored): searches that ring conflict signals
    # append to attention_log.jsonl; surfacing the volume keeps advisory
    # flooding visible instead of growing an unread log forever.
    attention_lines = 0
    attention_recent = 0
    attention_path = Path(ctx.settings.db_path).parent / "attention_log.jsonl"
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
    return _finding(
        "capacity.attention_volume", attention_recent < 200,
        f"{attention_recent} attention events in 7 days ({attention_lines} total)",
        evidence={"recent_7d": attention_recent, "total_lines": attention_lines},
    )


def _c_recall_blacklist(ctx: _DoctorCtx) -> Finding:
    # v0.15.5: recall blacklist visibility — default prefill vs customized
    # file, entry count, and any parse warnings (bad lines degrade the file).
    try:
        from .recall_blacklist import blacklist_path, load_blacklist
        _bl_path = blacklist_path(ctx.settings.db_path)
        _bl_names, _bl_warnings = load_blacklist(_bl_path)
        _bl_source = "file" if _bl_path.exists() else "default"
        return _finding(
            "recall.blacklist", not _bl_warnings,
            f"unscoped find excludes {len(_bl_names)} workspace(s) via recall blacklist "
            f"({_bl_source}: {', '.join(sorted(_bl_names)) or 'none'})",
            evidence={"source": _bl_source, "workspaces": sorted(_bl_names),
                      "warnings": list(_bl_warnings)},
        )
    except Exception as exc:  # degrade loudly: a silently missing check hides regressions
        return _finding("recall.blacklist", False, f"recall blacklist check failed: {exc}")


def _c_workspace_review(ctx: _DoctorCtx) -> Finding:
    # full-registry workspace confirmation (read-only; the snapshot
    # is only ever written by the authorized confirm_workspaces action).
    return _workspace_review_finding(ctx.conn, ctx.settings)


def _c_config_warnings(ctx: _DoctorCtx) -> Finding | None:
    if not ctx.settings.config_warnings:
        return None
    return _finding("config.warnings", False, "; ".join(ctx.settings.config_warnings))


# Order is observable twice over: console_static renders findings.slice(0, 6),
# and a check that reads state a previous one could have changed depends on
# staying where it is. Reordering this tuple changes product output.
_CHECKS: tuple[_Check, ...] = (
    _c_config_writable,
    _c_evidence_coverage,
    _c_evidence_freshness,
    _c_evidence_orphans,
    _c_conflicts_backlog,
    _c_conflicts_scan_required,
    _c_conflicts_scan_epoch,
    _c_conflicts_scan_stale,
    _c_conflicts_scan_chain,
    _c_conflicts_spec_drift,
    _c_conflicts_applying,
    _c_notices_backlog,
    _c_conflicts_scan_queue_backlog,
    _c_normalize_autonomy,
    _c_tags_over_limit,
    _c_capacity_attention_volume,
    _c_recall_blacklist,
    _c_workspace_review,
    _c_config_warnings,
)


@dataclass
class _ProbeOutcome:
    """How resolving the embedder went -- and nothing about what it produced.

    Deliberately excludes the embed_text result. When the probe hands back a
    handle but embedding fails, vector.device and evidence.unit_budget are
    still reported: the GPU-degradation signal is exactly what an operator
    needs in that situation. Folding the embed outcome in here would make
    ``has_embedder`` false and drop both findings silently.
    """

    kind: str  # "no_resolver" | "raised" | "none" | "ok"
    embedder: Any = None
    error: BaseException | None = None

    @property
    def has_embedder(self) -> bool:
        return self.kind == "ok"


@dataclass  # never slots=True: cached_property needs __dict__
class _DeepCtx:
    """Deep-mode state. Only genuinely read-only values are collected up front.

    ``vec_meta`` and ``active_dim`` are pure reads, so pulling them ahead of
    the quick_check and migration_state reads they now precede changes nothing.
    ``probe`` is not pure and therefore must not be: on the MCP path it is
    MemoryTools._ensure_embedder, whose first success creates the vec0 tables
    and writes _vec_index_meta. Resolving it during collect would change what
    vector.space, vector.table_dimension, vector.evidence_rows and
    vector.workspace_rows see, so it stays lazy and the first reader is
    _d_vector_dimension_probe -- asserted in tests/test_golden_doctor.py.
    """

    base: _DoctorCtx
    vec_meta: dict[str, str]
    active_dim: int | None

    @classmethod
    def collect(cls, base: _DoctorCtx) -> "_DeepCtx":
        vec_meta = {
            str(row[0]): str(row[1])
            for row in base.conn.execute("SELECT key,value FROM _vec_index_meta")
        }
        # The embedding dimension is a per-library fact (meta key, else the
        # vec0 tables' own CREATE SQL) — there is no configured vec.dim.
        return cls(base=base, vec_meta=vec_meta, active_dim=active_dim_on_connection(base.conn))

    @functools.cached_property
    def probe(self) -> _ProbeOutcome:
        if self.base.embedder_probe is None:
            return _ProbeOutcome("no_resolver")
        try:
            embedder, _probe_warnings = self.base.embedder_probe()
        except Exception as exc:
            return _ProbeOutcome("raised", error=exc)
        if embedder is None:
            return _ProbeOutcome("none")
        return _ProbeOutcome("ok", embedder=embedder)


_DeepCheck = Callable[[_DeepCtx], "Finding | None"]


def _d_database_quick_check(ctx: _DeepCtx) -> Finding:
    quick_check = str(ctx.base.conn.execute("PRAGMA quick_check").fetchone()[0])
    return _finding(
        "database.quick_check", quick_check == "ok", quick_check,
        critical=quick_check != "ok",
    )


def _d_database_schema_generation(ctx: _DeepCtx) -> Finding:
    migration = {
        str(row[0]): str(row[1])
        for row in ctx.base.conn.execute(
            "SELECT key,value FROM migration_state "
            "WHERE key IN ('schema_generation','phase','migration_completed_at')"
        )
    }
    generation_ok = (
        migration.get("schema_generation") == "workspace_state_v1"
        and migration.get("phase") not in {"building", "backfill", "resuming", "failed"}
    )
    return _finding(
        "database.schema_generation", generation_ok,
        f"generation={migration.get('schema_generation') or 'missing'}, "
        f"phase={migration.get('phase') or 'complete'}",
        critical=not generation_ok,
        evidence=migration,
    )


def _d_vector_space(ctx: _DeepCtx) -> Finding:
    active_space = ctx.vec_meta.get("active_space_id")
    configured_space: str | None = None
    try:
        from .vnext_migration import _configured_embedding_space_id
        configured_space = _configured_embedding_space_id(ctx.base.settings, ctx.active_dim)
    except (OSError, ValueError):
        configured_space = None
    space_ok = (
        configured_space is None
        or (ctx.vec_meta.get("state") == "ready" and active_space == configured_space)
    )
    return _finding(
        "vector.space", space_ok,
        f"state={ctx.vec_meta.get('state') or 'unmanaged'}, "
        f"active={active_space or 'none'}, configured={configured_space or 'none'}",
        evidence={
            "state": ctx.vec_meta.get("state", "unmanaged"),
            "active_space_id": active_space,
            "configured_space_id": configured_space,
        },
    )


def _d_vector_table_dimension(ctx: _DeepCtx) -> Finding:
    evidence_dim = vec_table_dimension(ctx.base.conn, "memory_evidence_vec")
    workspace_dim = vec_table_dimension(ctx.base.conn, "workspace_canonicals_vec")
    # Lazy table creation means absent tables are only a problem once the
    # index should be live; existing tables must agree with the active dim.
    dimensions_ok = (
        ctx.base.settings.embedding_model_path is None
        or (evidence_dim is None and workspace_dim is None)
        or (evidence_dim is not None and evidence_dim == workspace_dim == ctx.active_dim)
    )
    return _finding(
        "vector.table_dimension", dimensions_ok,
        f"evidence={evidence_dim}, workspace={workspace_dim}, active={ctx.active_dim}",
        evidence={"evidence": evidence_dim, "workspace": workspace_dim,
                  "active": ctx.active_dim},
    )


def _d_vector_evidence_rows(ctx: _DeepCtx) -> Finding:
    conn = ctx.base.conn
    evidence_vectors = _safe_count(conn, "memory_evidence_vec")
    orphan_vectors = _safe_count(
        conn, "memory_evidence_vec v LEFT JOIN memory_evidence e ON e.id=v.id",
        "e.id IS NULL",
    )
    missing_vectors = _safe_count(
        conn, "memory_evidence e LEFT JOIN memory_evidence_vec v ON v.id=e.id",
        "v.id IS NULL",
    )
    # Absent tables are expected before the first embedder build (lazy
    # creation); they count against the index only once a dim is active.
    evidence_rows_ok = (
        ctx.base.settings.embedding_model_path is None
        or (evidence_vectors is None and ctx.active_dim is None)
        or (
            evidence_vectors is not None
            and orphan_vectors == 0
            and missing_vectors == 0
        )
    )
    return _finding(
        "vector.evidence_rows",
        evidence_rows_ok,
        f"{ctx.base.units} evidence rows, {evidence_vectors} vectors, "
        f"{orphan_vectors} orphan vectors, {missing_vectors} missing vectors",
        evidence={"evidence": ctx.base.units, "vectors": evidence_vectors,
                  "orphan_vectors": orphan_vectors, "missing_vectors": missing_vectors},
    )


def _d_vector_workspace_rows(ctx: _DeepCtx) -> Finding:
    canonical_count = int(ctx.base.conn.execute(
        "SELECT COUNT(*) FROM workspace_canonicals WHERE lower(trim(name)) "
        "NOT IN ('','default','none','null','unknown') AND trim(name) NOT IN ('默认','未知')"
    ).fetchone()[0])
    canonical_vectors = _safe_count(ctx.base.conn, "workspace_canonicals_vec")
    workspace_rows_ok = (
        ctx.base.settings.embedding_model_path is None
        or (canonical_vectors is None and ctx.active_dim is None)
        or (canonical_vectors is not None and canonical_vectors == canonical_count)
    )
    return _finding(
        "vector.workspace_rows",
        workspace_rows_ok,
        f"{canonical_count} non-default canonicals, {canonical_vectors} vectors",
        evidence={"canonicals": canonical_count, "vectors": canonical_vectors},
    )


def _d_vector_dimension_probe(ctx: _DeepCtx) -> Finding:
    # --deep / memory_review(deep=true): actually run the embedder and
    # compare the live dimension against the library's active dim
    # (seconds-level cost). First reader of ctx.probe -- see _DeepCtx.
    embedding_configured = ctx.base.settings.embedding_model_path is not None
    probe = ctx.probe
    if probe.kind == "no_resolver":
        return _finding("vector.dimension_probe", False, "deep probe requested but no embedder resolver was provided")
    if probe.kind == "raised":
        return _finding("vector.dimension_probe", False, f"embedder probe failed: {probe.error}")
    if probe.kind == "none":
        if embedding_configured:
            return _finding("vector.dimension_probe", False, "embedding configured but the embedder is currently unavailable")
        # Asking for a probe without embedding configured is a
        # no-op, not a health problem — must not force exit 1.
        return _finding("vector.dimension_probe", True, "deep probe skipped: embedding not configured")
    try:
        er = probe.embedder.embed_text(prefix="", body="dimension probe")
        dim = len(er.embedding or [])
    except Exception as exc:
        return _finding("vector.dimension_probe", False, f"embedding failed: {exc}")
    if ctx.active_dim is None:
        # No stored dim yet (fresh library before its first
        # embed): nothing to compare against, report the
        # model's own dim.
        return _finding(
            "vector.dimension_probe", True,
            f"embedding dim {dim} (no stored active dim yet)",
        )
    return _finding(
        "vector.dimension_probe", dim == ctx.active_dim,
        f"embedding dim {dim} vs active dim {ctx.active_dim}",
    )


def _d_vector_device(ctx: _DeepCtx) -> Finding | None:
    # Device visibility: a runtime GPU→CPU self-heal keeps the
    # embedder alive but should not go unnoticed — surface it
    # as a warning so an operator restarts to re-probe the GPU.
    # Emitted whenever a handle exists, including when embedding itself
    # failed: that is precisely when an operator needs to see the device.
    if not ctx.probe.has_embedder:
        return None
    embedder = ctx.probe.embedder
    degraded = bool(getattr(embedder, "device_degraded", False))
    gpu_backed = bool(getattr(embedder, "gpu_backed", False))
    degraded_at = getattr(embedder, "device_degraded_at", None)
    if degraded and degraded_at is None:
        # Latch closed but the CPU rebuild itself failed: the
        # embedder is still pointed at the broken GPU instance
        # and every embed returns the sentinel — that is DOWN,
        # not merely slow.
        device_detail = (
            "GPU failed and the CPU fallback rebuild also failed — "
            "embedding unavailable until restart"
        )
    elif degraded:
        device_detail = (
            f"GPU failed at {degraded_at}; degraded to CPU inference — "
            "restart to re-probe"
        )
    elif gpu_backed:
        device_detail = "embedding on GPU"
    else:
        device_detail = "embedding on CPU"
    return _finding("vector.device", not degraded, device_detail)


def _d_evidence_unit_budget(ctx: _DeepCtx) -> Finding | None:
    # Budget tripwire: units beyond the token budget lose their
    # tail at embed time. The splitter caps text units at 400
    # chars (≈≤402 tokens, under the 512 default), so any hit
    # means an uncapped subject unit or a splitter change.
    # tokenize_locked serialises against embed_text on the
    # shared live instance; the whole scan is guarded because
    # run_all_checks must never raise.
    if not ctx.probe.has_embedder:
        return None
    embedder = ctx.probe.embedder
    tokenize_locked = getattr(embedder, "tokenize_locked", None)
    budget = embedder.token_budget() if hasattr(embedder, "token_budget") else None
    if tokenize_locked is None or budget is None:
        return _finding(
            "evidence.unit_budget", True,
            "skipped: embedder does not expose tokenize_locked/token_budget",
        )
    # Char prefilter scaled to the worst tokenizer density
    # (byte-fallback expansion ≈4 tokens/char): below
    # budget/4 characters an input cannot exceed budget.
    prefilter = max(1, int(budget) // 4)
    try:
        over_budget = 0
        checked_units = 0
        worst_tokens = 0
        for row in ctx.base.conn.execute(
            "SELECT text FROM memory_evidence WHERE length(text) > ?",
            (prefilter,),
        ).fetchall():
            checked_units += 1
            tokens = len(tokenize_locked(str(row[0])))
            worst_tokens = max(worst_tokens, tokens)
            if tokens > budget:
                over_budget += 1
        return _finding(
            "evidence.unit_budget", over_budget == 0,
            f"{over_budget} of {checked_units} oversized units exceed "
            f"the {budget}-token embed budget (worst {worst_tokens} tokens); "
            "tails beyond the budget are not indexed",
            evidence={
                "over_budget": over_budget,
                "checked_units": checked_units,
                "worst_tokens": worst_tokens,
                "budget": budget,
            },
        )
    except Exception as exc:
        return _finding(
            "evidence.unit_budget", True,
            f"skipped: unit budget scan failed: {exc}",
        )


_DEEP_CHECKS: tuple[_DeepCheck, ...] = (
    _d_database_quick_check,
    _d_database_schema_generation,
    _d_vector_space,
    _d_vector_table_dimension,
    _d_vector_evidence_rows,
    _d_vector_workspace_rows,
    _d_vector_dimension_probe,
    _d_vector_device,
    _d_evidence_unit_budget,
)


def run_all_checks(conn: sqlite3.Connection, settings: Settings, deep: bool = False, runtime_state: DegradeState | None = None, embedder_probe: Callable[[], tuple[Any, list[str]]] | None = None) -> OverviewReport:
    ctx = _DoctorCtx.collect(conn, settings, deep, runtime_state, embedder_probe)
    findings: list[Finding] = []
    # No blanket try/except around the loop: a check raising unexpectedly must
    # keep crashing rather than degrade into one silently missing finding. The
    # per-check guards that already exist stay where they are. A missing table
    # still raises out of collect, which doctor_overview_* turns into a
    # database.open critical report -- unchanged behaviour.
    for check in _CHECKS:
        produced = check(ctx)
        if produced is not None:
            findings.append(produced)
    if deep:
        deep_ctx = _DeepCtx.collect(ctx)
        for deep_check in _DEEP_CHECKS:
            produced = deep_check(deep_ctx)
            if produced is not None:
                findings.append(produced)
    overall = max((f.severity for f in findings), key=lambda s: {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}[s])
    return OverviewReport(utc_now_iso(), overall, findings, {"mode": runtime_state.mode if runtime_state else "sqlite", "total_memories": ctx.total, "evidence_indexed": ctx.indexed, "evidence_units": ctx.units})


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
        if settings.embedding_model_path is None:
            return None, []
        try:
            from .embedder import build_embedder
            return build_embedder(
                str(settings.embedding_model_path),
                n_ctx=EMBEDDING_N_CTX,
                reserved_tokens=EMBEDDING_RESERVED_TOKENS,
                max_section_chars=EMBEDDING_MAX_SECTION_CHARS,
            )
        except Exception:
            return None, []

    try:
        with open_ro_connection(settings.db_path) as conn:
            if settings.embedding_model_path is not None:
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
