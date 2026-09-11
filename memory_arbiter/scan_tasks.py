"""Scheduled-task specification shared by the guidance notice and help topic.

Single source of truth for the three tasks mema asks agents to schedule
(hourly conflict scan, weekly workspace anomaly check, daily governance
reminder). The notice payload and the
``memory(action="help", data={"topic": "scheduled_tasks"})`` document
both render from this module so they cannot drift apart.

Platform-agnostic by design (owner decision 2026-09-02): the server never
guesses the caller's platform from X-Mema-Client; the agent knows how to
create scheduled tasks on its own host.
"""
from __future__ import annotations

from typing import Any

SCHEDULED_TASKS_TOPIC = "scheduled_tasks"

AGENT_INSTRUCTION = (
    "Tell the user: mema needs three scheduled tasks (hourly conflict scan, weekly "
    "workspace anomaly check, daily governance reminder) to discover conflicts "
    "automatically. Ask whether to set them up now; on consent, create the "
    "equivalent tasks on your own platform from setup.tasks. The notice disappears "
    "by itself once the tasks run — no report-back needed."
)

SCHEDULED_TASKS_SPEC: dict[str, Any] = {
    "tasks": [
        {
            "name": "conflict_scan",
            "purpose": "Page through scan_candidates and triage candidates (record_conflict open/not_a_conflict).",
            "cadence": "hourly",
            "calls": [
                {
                    "tool": "memory_repair", "task": "scan_candidates",
                    "data": {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True},
                },
                {
                    "tool": "memory_repair", "task": "record_conflict",
                    "data": {
                        "slot_key": {"entity": "project-x", "attribute": "database", "scope": "production"},
                        "members": [
                            {
                                "memory_id": 12, "version": 1, "attribute_raw": "database",
                                "value_raw": "MySQL", "normalized_attribute": "database",
                                "normalized_value": "mysql", "evidence_quote": "database is MySQL",
                                "evidence_span": [0, 17],
                                "content_hash": "0000000000000000000000000000000000000000000000000000000000000000",
                                "direction": "a_to_b", "prompt_version": "p1", "detector_version": "d1",
                            },
                            {
                                "memory_id": 34, "version": 1, "attribute_raw": "database",
                                "value_raw": "SQLite", "normalized_attribute": "database",
                                "normalized_value": "sqlite", "evidence_quote": "database is SQLite",
                                "evidence_span": [0, 18],
                                "content_hash": "1111111111111111111111111111111111111111111111111111111111111111",
                                "direction": "b_to_a", "prompt_version": "p1", "detector_version": "d1",
                            },
                        ],
                        "value_groups": [
                            {"normalized_value": "mysql", "display_value": "MySQL", "members": ["12@1"]},
                            {"normalized_value": "sqlite", "display_value": "SQLite", "members": ["34@1"]},
                        ],
                        "status": "open",
                        "detector_version": "d1",
                        "source": "scheduled_scan",
                        "reason": "Reviewed conflicting values from the scan page.",
                    },
                },
                {
                    "note": (
                        "Pairing is workspace-grouped (0.15.13): candidates only ever pair memories "
                        "within one workspace bucket, and a page may also carry cross_bucket_references "
                        "— pairs a suspected-misplaced memory forms with its likely home bucket. Those "
                        "references cannot be recorded as conflicts; handle them through the "
                        "workspace_review notice flow (confirm with the user, then "
                        "memory_govern(action='move_memories_workspace')). "
                        "Start at anchor_memory_id=0; use each page's next_anchor_memory_id as the "
                        "next anchor_memory_id until it returns null. The response is lightweight by "
                        "default (pair ids, workspace, reasons, short quotes); the sample call passes "
                        "include_quotes=true so each candidate carries the full members/slot envelope "
                        "record_conflict needs — keep it when triaging, drop it only when a scan "
                        "returns nothing. After each page returns, "
                        "immediately triage that page's candidates: a real conflict -> record_conflict "
                        "with status='open' (shape as the sample call above, values taken from the "
                        "page's slot_groups/candidates); not a conflict -> record_conflict with "
                        "status='not_a_conflict' (same shape; requires authorized=true because it "
                        "suppresses future detection of the same candidate). Both dispositions are "
                        "recorded — do not batch them to the end of the round: an interrupted run "
                        "must not lose the triage of already-fetched pages. After a process restart, "
                        "run memory_repair(task='rebuild_evidence') before resuming to catch up on "
                        "evidence indexing. If you receive a queue_full response, the semantic worker "
                        "is saturated — back off and retry the scan page later. Hourly is the "
                        "recommended cadence; a weekly rhythm also works (a full pass over ~550 "
                        "memories takes about 15 minutes)."
                    ),
                },
            ],
        },
        {
            "name": "workspace_anomaly_check",
            "purpose": (
                "One-call workspace health check: flags memories whose nearest-content "
                "neighbours overwhelmingly sit in another workspace (misplacement), before "
                "the conflict scan round it precedes."
            ),
            "cadence": "weekly",
            "calls": [
                {
                    "tool": "memory_repair", "task": "scan_workspace_anomalies",
                    "data": {},
                    "note": (
                        "Runs BEFORE the week's conflict scan rounds. Each flagged memory gets one "
                        "workspace_review notice (max 10 per run; the rest surface in later weeks). "
                        "Read each notice (memory_repair task='notice' action='read'), verify with "
                        "the user, then move confirmed memories via "
                        "memory_govern(action='move_memories_workspace') or dismiss false alarms. "
                        "Same-week conflict scans automatically sweep suspected memories against "
                        "their likely home bucket (cross_bucket_references)."
                    ),
                },
            ],
        },
        {
            "name": "governance_reminder",
            "purpose": "Check semantic notices and the unresolved conflict backlog, remind the user to govern.",
            "cadence": "daily",
            "calls": [{"tool": "memory_review", "view": "doctor"}],
        },
    ],
}


def scheduled_tasks_help() -> dict[str, Any]:
    """Full self-serve document for the scheduled_tasks help topic."""
    return {
        "description": (
            "The three scheduled tasks mema relies on for automated conflict discovery, "
            "workspace placement health, and governance follow-up. Platform-agnostic: "
            "create the equivalent tasks on whatever scheduler your host provides."
        ),
        "topic": SCHEDULED_TASKS_TOPIC,
        "setup": SCHEDULED_TASKS_SPEC,
        "self_closing": (
            "A completed full-scan boundary (a scan_candidates page whose next_anchor_memory_id "
            "returns null) appends one line to scan_log.jsonl; once that line appears, the "
            "scan_never_run/scan_stale guidance notice stops appearing and doctor's "
            "conflicts.scan_required / conflicts.scan_stale findings turn green."
        ),
    }
