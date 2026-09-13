"""Scheduled-task specification shared by the guidance notice and help topic.

Single source of truth for the three tasks mema asks agents to schedule
(0.16.0 spec v2: kick the server-orchestrated conflict-scan pipeline and clear
the judgment queue; weekly workspace anomaly check; daily governance
reminder). The notice payload and the
``memory(action="help", data={"topic": "scheduled_tasks"})`` document
both render from this module so they cannot drift apart.

``spec_version`` drives drift detection: the pipeline stamps the version it
served on every completed round, and doctor flags a library whose scan
activity predates the current spec (§6⑩) — the host task needs rebuilding.

Platform-agnostic by design (owner decision 2026-09-02): the server never
guesses the caller's platform from X-Mema-Client; the agent knows how to
create scheduled tasks on its own host.
"""
from __future__ import annotations

from typing import Any

SCHEDULED_TASKS_TOPIC = "scheduled_tasks"

# 0.16.0 §6⑩: bump on any change to task responsibilities or call shapes;
# drift detection keys off this number (v1 = page-driven scan_candidates
# triage, retired by the server-orchestrated pipeline; v2 → v3 = weekly
# anomaly findings moved from workspace_review notices into the judgment
# queue — "verify with the user" is gone, the agent judges via the gate).
SCHEDULED_TASKS_SPEC_VERSION = 3

AGENT_INSTRUCTION = (
    "Tell the user: mema needs three scheduled tasks (a conflict-scan pipeline "
    "kick + judgment-queue cleanup, a weekly workspace anomaly check, and a "
    "daily governance reminder) to discover conflicts automatically. Ask "
    "whether to set them up now; on consent, create the equivalent tasks on "
    "your own platform from setup.tasks. The notice disappears by itself once "
    "the tasks run — no report-back needed."
)

SCHEDULED_TASKS_SPEC: dict[str, Any] = {
    "spec_version": SCHEDULED_TASKS_SPEC_VERSION,
    "tasks": [
        {
            "name": "conflict_scan",
            "purpose": (
                "Kick the server-orchestrated conflict-scan pipeline until it reports "
                "complete=true, then clear the judgment queue page by page. The server "
                "decides full-vs-incremental and resumes from breakpoints on its own."
            ),
            "cadence": "hourly",
            "calls": [
                {
                    "tool": "memory_repair", "task": "scan_pipeline",
                    "data": {"action": "kick"},
                    "note": (
                        "Each kick is a bounded synchronous batch (default 45s / 400 "
                        "memories). Repeat the kick in the same run until the response "
                        "says complete=true — the first round after an upgrade is a full "
                        "scan and may need several kicks; steady-state rounds finish in "
                        "one and usually queue nothing."
                    ),
                },
                {
                    "tool": "memory_repair", "task": "scan_queue",
                    "data": {"action": "page"},
                    "note": (
                        "After complete=true, drain pending queue items: judge each item "
                        "from its evidence quotes (upgrade to batch_read hits windows or "
                        "full reads when uncertain; read full texts before any "
                        "confirm-driven edit). Submit dispositions with "
                        "memory_repair(task='scan_queue', action='submit'): per-pair "
                        "{candidate_key_hash, status: confirmed|dismissed, reason} — "
                        "confirms add slot_key + value_groups (display values per member); "
                        "a whole noise group can be dismissed with one group_token entry. "
                        "Dispositions land server-side (pre-authorized by design: the "
                        "queue row is the audit trail, the agent is the only semantic "
                        "judge, and a dismissal lands the not_a_conflict suppression "
                        "source). Workspace "
                        "suspects: confirm with target_workspace + conf — the server "
                        "re-runs the vector vote and only moves when both signals agree "
                        "(protected buckets are never moved autonomously). If the queue "
                        "is large, keep paging in later runs — page boundaries are "
                        "breakpoints and nothing is lost between runs."
                    ),
                },
                {
                    "note": (
                        "v3 contract (spec_version=3): check-route noise pairs are "
                        "machine-cleared by the difference-based classifier, so the "
                        "queue only holds real signals (notify pairs, kept "
                        "value-difference pairs, workspace suspects, internal "
                        "contradictions via internal_conflicts). If your current task "
                        "still expects scan_candidates pages or workspace_review "
                        "notices, rebuild it from this spec (doctor's "
                        "conflicts.spec_drift finding says so too)."
                    ),
                },
            ],
        },
        {
            "name": "workspace_anomaly_check",
            "purpose": (
                "One-call full-library workspace health sweep: flags memories whose "
                "nearest-content neighbours overwhelmingly sit in another workspace. "
                "Complements the pipeline's incremental per-memory vote by catching "
                "neighbourhood drift between scans."
            ),
            "cadence": "weekly",
            "calls": [
                {
                    "tool": "memory_repair", "task": "scan_workspace_anomalies",
                    "data": {},
                    "note": (
                        "Each flagged memory becomes one kind='workspace' judgment-queue "
                        "row (max 10 new rows per run; the rest surface in later weeks "
                        "once earlier rows are judged). The 0.16.0 pipeline also lands "
                        "its own vector-vote suspects in the same queue — this weekly "
                        "sweep is the full-library backstop, sharing the identical "
                        "proportional gate. Judge each queue row by its vote evidence "
                        "and submit via memory_repair(task='scan_queue', "
                        "action='submit'): confirmed with target_workspace + conf lets "
                        "the server re-run the gate and move autonomously; dismissed "
                        "retires the row. No user verification step — you are the "
                        "semantic judge (protected buckets never move; they surface as "
                        "a protected_bucket_hint for the user instead)."
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
        "spec_version": SCHEDULED_TASKS_SPEC_VERSION,
        "setup": SCHEDULED_TASKS_SPEC,
        "self_closing": (
            "A completed pipeline round (a scan_pipeline kick reporting complete=true) "
            "records the served spec_version; once that marker exists at the current "
            "version, the scan_never_run/scan_stale guidance notice stops appearing and "
            "doctor's conflicts.scan_required / conflicts.scan_stale / conflicts.spec_drift "
            "findings turn green."
        ),
    }
