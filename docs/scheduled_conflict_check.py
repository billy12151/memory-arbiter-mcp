"""Scheduled conflict audit — runs the Pattern B workflow from INTEGRATION.md.

This is a standalone, cron-friendly script. It scans the memory store for
potential conflicts (recent memories that might contradict older ones), uses
memory_compare to get rule-based verdicts, optionally arbitrates them, and
prints / writes a report. No model tokens spent on scanning — the heavy
lifting is SQL aggregation + deterministic comparison.

Usage:
    # Dry run — print a report, change nothing.
    python docs/scheduled_conflict_check.py

    # Apply arbitration for clear cases (auto-supersede the loser when the
    # verdict is unambiguous and neither side is user_confirmed/locked).
    python docs/scheduled_conflict_check.py --apply

    # Restrict to one workspace and write a markdown report to disk.
    python docs/scheduled_conflict_check.py --workspace default --report ~/audit.md

Cron example (daily at 09:00, dry-run, append to a log):
    0 9 * * * cd /path/to/memory-arbiter-mcp && \\
        /path/to/python docs/scheduled_conflict_check.py >> /var/log/mem-audit.log 2>&1

How it picks "suspicious pairs":
  1. Pull recent memories (last N days, configurable via --recent-days).
  2. For each recent memory, find older memories in the same workspace that
     share subject or tag overlap — those are the only pairs worth comparing.
     Blindly comparing every pair is N² and mostly noise.
  3. memory_compare returns a verdict; we surface the ones that are NOT
     "equivalent" and NOT manual_review — those are actionable conflicts.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.tools import MemoryTools  # noqa: E402


def _parse_tags(raw) -> list[str]:
    import json
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            return list(json.loads(raw))
        except Exception:
            return []
    return list(raw)


def find_suspicious_pairs(db: MemoryDB, workspace: str | None, recent_days: int) -> list[tuple[dict, dict]]:
    """Return (recent, older) pairs that share subject/tag overlap within a workspace."""
    if db.conn is None:
        return []
    import json
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=recent_days)).isoformat()

    ws_clause = "AND workspace = ?" if workspace else ""
    ws_params = [workspace] if workspace else []

    recent_rows = db.conn.execute(
        f"SELECT * FROM memories WHERE status = 'active' AND ingest_time >= ? {ws_clause} ORDER BY ingest_time DESC",
        [cutoff, *ws_params],
    ).fetchall()
    recent = [dict(r) for r in recent_rows]
    if not recent:
        return []

    all_active = db.conn.execute(
        f"SELECT * FROM memories WHERE status = 'active' {ws_clause} ORDER BY ingest_time DESC",
        ws_params,
    ).fetchall()
    all_rows = [dict(r) for r in all_active]

    pairs: list[tuple[dict, dict]] = []
    seen: set[tuple[int, int]] = set()
    for r in recent:
        r_tags = set(_parse_tags(r.get("tags")))
        r_subject = (r.get("subject") or "").lower()
        for other in all_rows:
            if other["id"] == r["id"]:
                continue
            # Only pair recent with older (avoid duplicates + self-comparison).
            if other["ingest_time"] >= r["ingest_time"]:
                continue
            key = (r["id"], other["id"])
            if key in seen:
                continue
            o_tags = set(_parse_tags(other.get("tags")))
            o_subject = (other.get("subject") or "").lower()
            # Overlap criterion: shared tag OR shared subject substring.
            tag_overlap = bool(r_tags & o_tags)
            subject_overlap = bool(r_subject and o_subject and (r_subject in o_subject or o_subject in r_subject))
            if tag_overlap or subject_overlap:
                pairs.append((r, other))
                seen.add(key)
    return pairs


def run_audit(tools: MemoryTools, workspace: str | None, recent_days: int, apply: bool) -> dict:
    db = tools.db
    summary = tools.memory_audit_summary()
    open_conflicts = tools.memory_list_conflicts(status="open")
    pairs = find_suspicious_pairs(db, workspace, recent_days)

    actionable = []  # pairs where compare says NOT equivalent and NOT manual_review
    manual_review = []
    clean = []
    for recent, older in pairs:
        cmp = tools.memory_compare(left_id=recent["id"], right_id=older["id"])
        verdict = cmp.get("data", {}).get("comparison", {})
        side = verdict.get("winner_side")
        if side in ("manual_review", "tie"):
            manual_review.append((recent, older, verdict))
        elif side in ("left", "right"):
            winner = recent if side == "left" else older
            loser = older if side == "left" else recent
            actionable.append((winner, loser, verdict))
            if apply:
                # Only auto-apply when the loser is not user-protected.
                loser_protected = (
                    loser.get("source_type") == "user_confirmed"
                    or loser.get("protection_level") in ("locked", "protected")
                )
                if not loser_protected:
                    tools.memory_arbitrate(left_id=recent["id"], right_id=older["id"],
                                           mark_conflict=True, apply=True)
        else:
            clean.append((recent, older))
    return {
        "summary": summary,
        "open_conflicts_before": open_conflicts,
        "pairs_scanned": len(pairs),
        "actionable_conflicts": actionable,
        "needs_manual_review": manual_review,
        "clean_pairs": len(clean),
        "applied": apply,
    }


def render_report(result: dict) -> str:
    lines = []
    lines.append(f"# memory-arbiter scheduled audit — {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    summ = result["summary"].get("data", {})
    lines.append("## Audit summary (per workspace)")
    for ws, stats in (summ.get("workspaces") or {}).items():
        lines.append(f"- **{ws}**: {stats.get('count', '?')} memories, {stats.get('open_conflicts', '?')} open conflicts")
    lines.append("")
    lines.append(f"- Suspicious pairs scanned: **{result['pairs_scanned']}**")
    lines.append(f"- Clear conflicts (rule-resolved): **{len(result['actionable_conflicts'])}**"
                 + (f"  — auto-arbitrated" if result["applied"] else "  — NOT applied (dry run)"))
    lines.append(f"- Need manual review: **{len(result['needs_manual_review'])}**")
    lines.append("")

    if result["actionable_conflicts"]:
        lines.append("## Conflicts detected")
        for winner, loser, verdict in result["actionable_conflicts"]:
            reasons = "; ".join(verdict.get("reasons") or [])
            lines.append(f"- WINNER id={winner['id']} `{(winner.get('subject') or '')[:50]}`")
            lines.append(f"  LOSER  id={loser['id']} `{(loser.get('subject') or '')[:50]}`")
            lines.append(f"  reason: {reasons}")
            lines.append("")

    if result["needs_manual_review"]:
        lines.append("## Needs your decision (manual review)")
        for a, b, verdict in result["needs_manual_review"]:
            reasons = "; ".join(verdict.get("reasons") or [])
            lines.append(f"- id={a['id']} `{(a.get('subject') or '')[:40]}`  vs  id={b['id']} `{(b.get('subject') or '')[:40]}`")
            lines.append(f"  reason: {reasons}")
            lines.append(f"  → call memory_arbitrate(left_id={a['id']}, right_id={b['id']}, apply=true) once you decide")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="memory-arbiter scheduled conflict audit")
    parser.add_argument("--workspace", default=None, help="restrict to one workspace")
    parser.add_argument("--recent-days", type=int, default=7, help="treat memories ingested in the last N days as 'recent' (default 7)")
    parser.add_argument("--apply", action="store_true", help="auto-arbitrate clear conflicts (supersede the loser). Dry run by default.")
    parser.add_argument("--report", default=None, help="write a markdown report to this path")
    args = parser.parse_args()

    settings = Settings.from_env()
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    result = run_audit(tools, args.workspace, args.recent_days, args.apply)
    report = render_report(result)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"Report written to {args.report}", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
