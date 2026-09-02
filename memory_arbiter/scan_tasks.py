"""Scheduled-task specification shared by the guidance notice and help topic.

Single source of truth for the two tasks mema asks agents to schedule
(hourly conflict scan, daily governance reminder). The notice payload and
the ``memory(action="help", data={"topic": "scheduled_tasks"})`` document
both render from this module so they cannot drift apart.

Platform-agnostic by design (owner decision 2026-09-02): the server never
guesses the caller's platform from X-Mema-Client; the agent knows how to
create scheduled tasks on its own host.
"""
from __future__ import annotations

from typing import Any

SCHEDULED_TASKS_TOPIC = "scheduled_tasks"

AGENT_INSTRUCTION = (
    "Tell the user: mema needs two scheduled tasks (hourly conflict scan, daily governance "
    "reminder) to discover conflicts automatically. Ask whether to set them up now; on consent, "
    "create the equivalent tasks on your own platform from setup.tasks. The notice disappears "
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
                    "data": {"anchor_memory_id": 0, "batch": 200, "k": 10},
                },
                {
                    "note": (
                        "Start at anchor_memory_id=0; use each page's next_anchor_memory_id as the "
                        "next anchor_memory_id until it returns null. After a process restart, run "
                        "memory_repair(task='rebuild_evidence') before resuming to catch up on "
                        "evidence indexing. If you receive a queue_full response, the semantic worker "
                        "is saturated — back off and retry the scan page later."
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
            "The two scheduled tasks mema relies on for automated conflict discovery and "
            "governance follow-up. Platform-agnostic: create the equivalent tasks on whatever "
            "scheduler your host provides."
        ),
        "topic": SCHEDULED_TASKS_TOPIC,
        "setup": SCHEDULED_TASKS_SPEC,
        "self_closing": (
            "Every completed scan_candidates run appends to scan_log.jsonl; once a task runs, "
            "the scan_never_run/scan_stale guidance notice stops appearing and doctor's "
            "conflicts.scan_required / conflicts.scan_stale findings turn green."
        ),
    }
