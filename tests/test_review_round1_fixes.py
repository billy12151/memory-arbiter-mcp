"""Regression tests for the 2026-08-21 adversarial review round on 0.14.0.dev2.

Covers the conflict-store CAS paths (judge version pinning, escalate
append/link, duplicate_event, combined candidate identity, not_a_conflict
dispositions, single-member append, overflow), the apply attack vectors and
bookkeeping (spec items 2/17/18/19/20), and the surface-level fix contracts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.tools import MemoryTools


def _tools(tmp_path: Path, *, isolation: str = "none") -> tuple[MemoryTools, MemoryDB]:
    settings = Settings(
        db_path=tmp_path / "review.db", backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=False, isolation=isolation, workspace="default",
    )
    db = MemoryDB(settings)
    return MemoryTools(settings, db), db


def _memory(db: MemoryDB, content: str) -> int:
    memory_id, _ = db.insert_memory(
        MemoryRecord(content=content, subject="database", agent_id="test", workspace="default"),
        "default",
    )
    assert memory_id is not None
    return memory_id


def _member(memory_id: int, version: int, value: str, *, detector: str = "d1") -> dict:
    quote = f"database is {value}"
    return ConflictMember(
        memory_id=memory_id, version=version, attribute_raw="database", value_raw=value,
        normalized_attribute="database", normalized_value=value.casefold(), evidence_quote=quote,
        evidence_span=(0, len(quote)), content_hash=(f"{memory_id}@{version}" * 64)[:64],
        direction="a_to_b", prompt_version="p1", detector_version=detector,
    ).to_dict()


_SLOT_UNSET = object()


def _record_payload(left: int, right: int, *, detector: str = "d1", status: str = "open",
                    slot: object = _SLOT_UNSET) -> dict:
    if slot is _SLOT_UNSET:
        slot = {"entity": "project", "attribute": "database", "scope": "global"}
    members = [_member(left, 1, "mysql", detector=detector), _member(right, 1, "sqlite", detector=detector)]
    return {
        "slot_key": slot,
        "members": members,
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{left}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{right}@1",)).to_dict(),
        ],
        "detector_version": detector, "prompt_version": "p1", "source": "scheduled_scan",
        "reason": "different database values", "status": status,
    }


def _judge(tools: MemoryTools, conflict_id: int, plan: list[dict], revision: int = 1,
           chosen: str = "sqlite", resolution: int | None = None) -> dict:
    return tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": revision, "chosen_value": chosen,
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": plan, "resolution_memory_id": resolution,
    })


# ── judge pins the stored member matching the memory's current version ──────

def test_judge_pins_current_member_version_after_edit(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    # External edit moves A to v2; a later scan appends the re-detected A@2.
    db.edit_memory_intent(a, new_content="database is postgres", reason="user edit")
    appended = tools.memory_repair("record_conflict", {
        **_record_payload(a, b),
        "members": [_member(a, 2, "postgres"), _member(b, 1, "sqlite")],
        "value_groups": [
            ConflictValueGroup("postgres", "Postgres", (f"{a}@2",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{b}@1",)).to_dict(),
        ],
        "expected_revision": 1,
    })
    assert appended["data"]["outcome"] == "appended"
    # judge must plan against the CURRENT version instead of permanent stale_member.
    judged = _judge(tools, conflict_id, [{"memory_id": a, "action": "update_current_claim"},
                                         {"memory_id": b, "action": "use_as_resolution"}],
                    revision=2, chosen="postgres", resolution=b)
    assert judged["ok"] is True, judged
    plan = judged["data"]["apply_summary"]["plan"]
    pinned = next(step for step in plan if step["memory_id"] == a)
    assert pinned["expected_version"] == 2


def test_judge_stale_member_for_never_recorded_version(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    db.edit_memory_intent(a, new_content="database is postgres", reason="user edit")
    judged = _judge(tools, conflict_id, [{"memory_id": a, "action": "update_current_claim"}])
    assert judged["ok"] is False
    assert judged["data"]["outcome"] == "stale_member"
    assert "re-read" in judged["data"]["note"]


# ── escalate appends/links into an existing open group instead of raising ───

def _structured_notice(db: MemoryDB, tools: MemoryTools, a: int, b: int) -> int:
    import hashlib
    from memory_arbiter.semantic_conflict import notice_dedupe_key
    left, right = db.get_memory(a), db.get_memory(b)
    payload = {
        "route": "notice_ready", "reason": "same_attribute_different_grounded_value",
        "slot_key": {"entity": "project", "attribute": "database", "scope": "global"},
        "member_versions": [
            {"memory_id": a, "version": 1, "value": "mysql",
             "evidence": {"quote": "database is mysql", "start": 0, "end": 17}},
            {"memory_id": b, "version": 1, "value": "sqlite",
             "evidence": {"quote": "database is sqlite", "start": 0, "end": 18}},
        ],
        "value_groups": [
            {"normalized_value": "mysql", "display_value": "MySQL", "members": [f"{a}@1"]},
            {"normalized_value": "sqlite", "display_value": "SQLite", "members": [f"{b}@1"]},
        ],
        "candidate_key": {
            "detector_version": "d1",
            "members": sorted([f"{a}@1", f"{b}@1"]),
            "evidence": [],
        },
        "left_evidence": {"text": "database is mysql", "start_offset": 0, "end_offset": 17},
        "right_evidence": {"text": "database is sqlite", "start_offset": 0, "end_offset": 18},
        "left_content_hash": hashlib.sha256(left["content"].encode()).hexdigest(),
        "right_content_hash": hashlib.sha256(right["content"].encode()).hexdigest(),
    }
    created = db.record_semantic_notice(
        memory_id=a, peer_id=b, severity="high", notice_type="semantic_evidence",
        title="Possible memory change", message="different database values",
        payload=payload,
        dedupe_key=notice_dedupe_key(a, b, 1, 1, "semantic_evidence"),
        left_version=1, right_version=1, source="semantic_evidence",
    )
    assert created["outcome"] == "created"
    return int(created["notice_id"])


def test_escalate_appends_into_existing_open_group(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b, c = _memory(db, "database is mysql"), _memory(db, "database is sqlite"), _memory(db, "database is postgres")
    notice_id = _structured_notice(db, tools, a, b)
    # A superset formal group already owns the slot before the escalation.
    conflict_id = tools.memory_repair("record_conflict", {
        **_record_payload(a, c),
        "members": [_member(a, 1, "mysql"), _member(c, 1, "postgres")],
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{a}@1",)).to_dict(),
            ConflictValueGroup("postgres", "Postgres", (f"{c}@1",)).to_dict(),
        ],
    })["data"]["conflict_id"]
    escalated = tools.memory_repair("notice", {
        "action": "escalate", "notice_id": notice_id, "reason": "real contradiction",
    })
    assert escalated["ok"] is True, escalated
    assert escalated["data"]["conflict_outcome"] == "appended"
    assert escalated["data"]["conflict_id"] == conflict_id
    group = db.get_conflict(conflict_id)
    member_ids = {member["memory_id"] for member in group["member_versions"]}
    assert member_ids == {a, b, c}
    notice_row = db.get_conflict(notice_id)
    assert notice_row["notice_delivery_status"] == "resolved"


def test_escalate_links_when_members_already_covered(tmp_path: Path) -> None:
    """When the open group already covers every notice member, escalate links
    the notice (no new members) and resolves it."""
    tools, db = _tools(tmp_path)
    a, b, c = _memory(db, "database is mysql"), _memory(db, "database is sqlite"), _memory(db, "database is postgres")
    notice_id = _structured_notice(db, tools, a, b)
    # A superset open group at the same slot already contains a AND b.
    conflict_id = tools.memory_repair("record_conflict", {
        **_record_payload(a, b),
        "members": [_member(a, 1, "mysql"), _member(b, 1, "sqlite"), _member(c, 1, "postgres")],
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{a}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{b}@1",)).to_dict(),
            ConflictValueGroup("postgres", "Postgres", (f"{c}@1",)).to_dict(),
        ],
    })["data"]["conflict_id"]
    escalated = tools.memory_repair("notice", {"action": "escalate", "notice_id": notice_id, "reason": "verified"})
    assert escalated["ok"] is True
    assert escalated["data"]["conflict_outcome"] == "linked"
    assert escalated["data"]["conflict_id"] == conflict_id
    # No members were added; the notice row is resolved/linked.
    assert len(db.get_conflict(conflict_id)["member_versions"]) == 3
    assert db.get_conflict(notice_id)["notice_delivery_status"] == "resolved"


def test_escalate_reports_applying_group_exists(tmp_path: Path) -> None:
    """Escalating onto a slot whose group is already applying is refused
    structurally, not with a raw exception."""
    tools, db = _tools(tmp_path)
    a, b, c = _memory(db, "database is mysql"), _memory(db, "database is sqlite"), _memory(db, "database is postgres")
    notice_id = _structured_notice(db, tools, a, b)
    conflict_id = tools.memory_repair("record_conflict", {
        **_record_payload(a, c),
        "members": [_member(a, 1, "mysql"), _member(c, 1, "postgres")],
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{a}@1",)).to_dict(),
            ConflictValueGroup("postgres", "Postgres", (f"{c}@1",)).to_dict(),
        ],
    })["data"]["conflict_id"]
    _judge(tools, conflict_id, [{"memory_id": a, "action": "update_current_claim"},
                                {"memory_id": c, "action": "use_as_resolution"}],
           chosen="postgres", resolution=c)
    result = tools.memory_repair("notice", {"action": "escalate", "notice_id": notice_id, "reason": "x"})
    assert result["ok"] is False
    assert result["data"].get("outcome") == "applying_group_exists" or (
        result["data"].get("detail", {}).get("outcome") == "applying_group_exists"
    )


def test_record_conflict_duplicate_event_after_detector_change(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    # not_a_conflict recorded with slot+fingerprint under d1; re-recording the
    # same slot/members under d2 must return a structured outcome, not raise.
    recorded = tools.memory_repair("record_conflict", _record_payload(a, b, status="not_a_conflict"))
    assert recorded["ok"] is True
    re_recorded = tools.memory_repair("record_conflict", _record_payload(a, b, detector="d2"))
    assert re_recorded["data"]["outcome"] == "duplicate_event"
    assert re_recorded["data"]["conflict_id"] == recorded["data"]["conflict_id"]


def test_slotless_not_a_conflict_reevaluates_after_detector_change(tmp_path: Path) -> None:
    """Spec item 17: a candidate-only dismissal re-evaluates under a new detector."""
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    first = tools.memory_repair("record_conflict", _record_payload(a, b, status="not_a_conflict", slot=None))
    assert first["data"]["outcome"] == "inserted"
    second = tools.memory_repair("record_conflict", _record_payload(a, b, detector="d2", status="not_a_conflict", slot=None))
    assert second["data"]["outcome"] == "inserted"
    assert second["data"]["conflict_id"] != first["data"]["conflict_id"]


# ── candidate identity after append covers the combined snapshot ────────────

def test_append_candidate_key_covers_combined_snapshot(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b, c = _memory(db, "database is mysql"), _memory(db, "database is sqlite"), _memory(db, "database is postgres")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    appended = tools.memory_repair("record_conflict", {
        **_record_payload(a, c),
        "members": [_member(c, 1, "postgres")],
        "value_groups": [ConflictValueGroup("postgres", "Postgres", (f"{c}@1",)).to_dict()],
        "expected_revision": 1,
    })
    assert appended["data"]["outcome"] == "appended"
    row = db.get_conflict(conflict_id)
    assert row["candidate_key"]["members"] == sorted([f"{a}@1", f"{b}@1", f"{c}@1"])
    # Re-appending the same single member now dedupes against the combined
    # identity instead of colliding on the unique candidate index.
    duplicate = tools.memory_repair("record_conflict", {
        **_record_payload(a, c),
        "members": [_member(c, 1, "postgres")],
        "value_groups": [ConflictValueGroup("postgres", "Postgres", (f"{c}@1",)).to_dict()],
        "expected_revision": 2,
    })
    assert duplicate["data"]["outcome"] == "deduped"


def test_not_a_conflict_against_open_group_returns_open_group_exists(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    inverted = tools.memory_repair("record_conflict", {
        **_record_payload(a, b, status="not_a_conflict"), "expected_revision": 1,
    })
    assert inverted["ok"] is False
    assert inverted["data"]["outcome"] == "open_group_exists"
    assert inverted["data"]["conflict_id"] == conflict_id
    # The open group gained no members from the rejected disposition.
    assert len(db.get_conflict(conflict_id)["member_versions"]) == 2


def test_single_member_append_to_open_group(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b, c = _memory(db, "database is mysql"), _memory(db, "database is sqlite"), _memory(db, "database is postgres")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    appended = tools.memory_repair("record_conflict", {
        **_record_payload(a, c),
        "members": [_member(c, 1, "postgres")],
        "value_groups": [ConflictValueGroup("postgres", "Postgres", (f"{c}@1",)).to_dict()],
        "expected_revision": 1,
    })
    assert appended["ok"] is True
    assert appended["data"]["outcome"] == "appended"
    # Creation still requires two value groups.
    created = tools.memory_repair("record_conflict", {
        **_record_payload(a, b),
        "value_groups": [ConflictValueGroup("mysql", "MySQL", (f"{a}@1",)).to_dict()],
    })
    assert created["data"]["outcome"] == "invalid_input"


def test_slot_key_rejects_unknown_sentinel_values(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    for slot in (
        {"entity": "project", "attribute": "__unknown__", "scope": "global"},
        {"entity": "unknown", "attribute": "database", "scope": "global"},
    ):
        rejected = tools.memory_repair("record_conflict", _record_payload(a, b, slot=slot))
        assert rejected["ok"] is False
        assert rejected["data"]["outcome"] == "invalid_input"


# ── spec item 20: overflow returns manual review, never a second open group ──

def test_overflow_append_flags_manual_review(tmp_path: Path, monkeypatch) -> None:
    from memory_arbiter.db import conflicts as conflicts_module
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    before = db.get_conflict(conflict_id)
    monkeypatch.setattr(conflicts_module, "_MAX_MEMBERS", 2)
    c = _memory(db, "database is postgres")
    overflowed = tools.memory_repair("record_conflict", {
        **_record_payload(a, c),
        "members": [_member(c, 1, "postgres")],
        "value_groups": [ConflictValueGroup("postgres", "Postgres", (f"{c}@1",)).to_dict()],
        "expected_revision": 1,
    })
    assert overflowed["data"]["outcome"] == "overflow"
    after = db.get_conflict(conflict_id)
    assert after["overflow"] is True
    assert after["member_versions"] == before["member_versions"]
    # No second open group was created to bypass the unique constraint.
    assert len(db.list_conflicts(status="open", limit=100)) == 1


def test_overflow_at_insert_flags_manual_review(tmp_path: Path, monkeypatch) -> None:
    from memory_arbiter.db import conflicts as conflicts_module
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    monkeypatch.setattr(conflicts_module, "_MAX_MEMBERS", 1)
    initial = tools.memory_repair("record_conflict", _record_payload(a, b))
    assert initial["data"]["outcome"] == "overflow"
    assert db.list_conflicts(status="open", limit=100) == []


# ── spec item 2: post-resolution new value creates a new event ──────────────

def test_post_resolution_new_value_creates_new_event(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    assert _judge(tools, conflict_id, [{"memory_id": a, "action": "update_current_claim"},
                                       {"memory_id": b, "action": "use_as_resolution"}],
                  resolution=b)["ok"]
    tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": a,
        "action": "update_current_claim", "content": "database is sqlite",
        "authorized": True,
    })
    tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 3, "memory_id": b,
        "action": "use_as_resolution", "authorized": True,
    })
    assert tools.memory_govern("resolve_conflict", {
        "conflict_id": conflict_id, "expected_revision": 4, "authorized": True,
    })["ok"]
    # SQLite was later replaced by PostgreSQL: a NEW event on the same slot.
    c = _memory(db, "database is postgres")
    new_event = tools.memory_repair("record_conflict", {
        **_record_payload(b, c),
        "members": [_member(b, 1, "sqlite"), _member(c, 1, "postgres")],
        "value_groups": [
            ConflictValueGroup("sqlite", "SQLite", (f"{b}@1",)).to_dict(),
            ConflictValueGroup("postgres", "Postgres", (f"{c}@1",)).to_dict(),
        ],
    })
    assert new_event["data"]["outcome"] == "inserted"
    assert new_event["data"]["conflict_id"] != conflict_id
    assert db.get_conflict(conflict_id)["status"] == "resolved"


# ── spec item 18: apply attack vectors ──────────────────────────────────────

def _applying_two_member_group(tools: MemoryTools, db: MemoryDB, tmp_path: Path) -> tuple[int, int, int]:
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    assert _judge(tools, conflict_id, [{"memory_id": a, "action": "update_current_claim"},
                                       {"memory_id": b, "action": "use_as_resolution"}],
                  resolution=b)["ok"]
    return conflict_id, a, b


def test_apply_rejects_wrong_target_duplicate_and_stale_member(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    conflict_id, a, b = _applying_two_member_group(tools, db, tmp_path)
    outsider = _memory(db, "database is mariadb")
    summary_before = db.get_conflict(conflict_id)["apply_summary"]
    revision_before = db.get_conflict(conflict_id)["revision"]

    wrong = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": outsider,
        "action": "update_current_claim", "authorized": True,
    })
    assert wrong["ok"] is False and wrong["data"]["outcome"] == "invalid_action"

    applied = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": a,
        "action": "update_current_claim", "content": "database is sqlite",
        "authorized": True,
    })
    assert applied["ok"] is True

    duplicate = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 3, "memory_id": a,
        "action": "update_current_claim", "content": "database is sqlite",
        "authorized": True,
    })
    assert duplicate["ok"] is False and duplicate["data"]["outcome"] == "invalid_action"
    # The rejected duplicate must not mutate the persisted plan/summary.
    assert db.get_conflict(conflict_id)["apply_summary"]["plan"][0]["status"] == "completed"

    # stale_member: the remaining member is externally edited before its step.
    db.edit_memory_intent(b, new_content="database is sqlite v2", reason="user edit")
    stale = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 3, "memory_id": b,
        "action": "use_as_resolution", "authorized": True,
    })
    assert stale["ok"] is False and stale["data"]["outcome"] == "stale_member"
    row = db.get_conflict(conflict_id)
    assert row["status"] == "applying"
    assert row["apply_summary"]["plan"][0]["status"] == "completed"


def test_plain_update_cannot_forge_conflict_context(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    conflict_id, a, b = _applying_two_member_group(tools, db, tmp_path)
    summary_before = db.get_conflict(conflict_id)["apply_summary"]
    # An ordinary edit carrying a bogus conflict-context field: no such
    # parameter exists, so nothing can suppress or mutate the plan through it.
    edited = tools.memory("update", {
        "memory_id": a, "new_content": "database is sqlite",
        "reason": "honest edit", "applying_conflict_id": conflict_id,
    })
    assert edited["ok"] is True
    assert db.get_conflict(conflict_id)["apply_summary"] == summary_before
    # The forged field is reported as an ignored unknown field, not honored.
    assert any("applying_conflict_id" in str(item) for item in edited.get("warnings", []))


# ── spec item 19: result bookkeeping separate from the detection snapshot ───

def test_apply_result_version_and_hash_bookkeeping(tmp_path: Path) -> None:
    import hashlib
    tools, db = _tools(tmp_path)
    conflict_id, a, b = _applying_two_member_group(tools, db, tmp_path)
    snapshot_before = db.get_conflict(conflict_id)["member_versions"]
    applied = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": a,
        "action": "update_current_claim", "content": "database is sqlite",
        "authorized": True,
    })
    assert applied["ok"] is True
    row = db.get_conflict(conflict_id)
    step = next(item for item in row["apply_summary"]["plan"] if item["memory_id"] == a)
    updated = db.get_memory(a)
    assert step["result_version"] == updated["version"]
    assert step["result_hash"] == hashlib.sha256(updated["content"].encode("utf-8")).hexdigest()
    # The original detection snapshot was not rewritten.
    assert row["member_versions"] == snapshot_before
    assert next(m for m in row["member_versions"] if m["memory_id"] == a)["version"] == 1


def test_failed_step_suggests_replan_and_apply_keeps_history(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    # A single-step plan so the failed step leaves NO pending work: that is the
    # state where resolve_conflict would deterministically fail apply_incomplete.
    assert _judge(tools, conflict_id, [{"memory_id": a, "action": "update_current_claim"}],
                  resolution=b)["ok"]
    failed = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": a,
        "action": "update_current_claim", "content": "unrelated text entirely",
        "authorized": True,
    })
    assert failed["ok"] is False
    assert failed["data"]["action_required"] == "replan_conflict"
    # Detail surface's next_executable_call suggests replan, not a resolve that
    # would fail apply_incomplete.
    detail = tools.memory_review("conflict_detail", {"conflict_id": conflict_id})
    assert detail["data"]["next_executable_call"]["action"] == "replan_conflict"
    # List surface's per-conflict guidance agrees.
    listed = tools.memory_review("conflicts", {"status": "applying", "limit": 10})
    guided = next(c for c in listed["data"]["conflicts"] if c["id"] == conflict_id)
    assert guided["next_action"]["action"] == "replan_conflict"
    replanned = tools.memory_govern("replan_conflict", {
        "conflict_id": conflict_id, "expected_revision": 3,
        "apply_plan": [{"memory_id": a, "action": "update_current_claim"}],
        "authorized": True,
    })
    assert replanned["ok"] is True
    history = replanned["data"]["apply_summary"]["history"]
    assert history and history[0]["revision"] == 3
    re_applied = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 4, "memory_id": a,
        "action": "update_current_claim", "content": "database is sqlite",
        "authorized": True,
    })
    assert re_applied["ok"] is True
    assert db.get_conflict(conflict_id)["apply_summary"]["history"] == history


# ── legacy wrapper returns a structured error instead of recursing ──────────

def test_memory_record_conflict_wrapper_returns_structured_error(tmp_path: Path) -> None:
    tools, _ = _tools(tmp_path)
    result = tools.memory_record_conflict(1, 2, "legacy call")
    assert result["ok"] is False
    assert "memory_repair" in result["data"]["error"]


# ── 2026-08-21 review round 2: blocker + regression fixes ───────────────────

def test_scan_reverse_only_extraction_grounds_each_member_to_own_value(tmp_path: Path) -> None:
    """Round-2 blocker: a reverse-only (B->A) extraction must stamp each member
    with its OWN value, not its peer's."""
    import tests.test_vnext_evidence as tv
    from memory_arbiter.semantic_conflict import ModelSignal

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    a = tools.memory_write(content="生产数据库使用 mysql 方案", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(content="生产数据库使用 sqlite 方案", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)

    class RevOnly:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            if "mysql" in str(left["quote"]).casefold():  # forward A->B: invalid
                return ModelSignal(False, "invalid_schema", None, "", None, "bad")
            parsed = {"attribute_a": "数据库选型", "value_a": "sqlite",
                      "attribute_b": "数据库选型", "value_b": "mysql"}
            return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)
    tools._ensure_semantic_backend = lambda: RevOnly()

    result = tools.memory_repair("scan_candidates", {"batch": 50, "k": 10})
    enriched = [c for c in result["data"]["candidates"] if c.get("value_groups")]
    assert enriched
    by_mid = {m["memory_id"]: m["normalized_value"] for m in enriched[0]["members"]}
    assert by_mid[a["id"]] == "mysql"
    assert by_mid[b["id"]] == "sqlite"


def test_detector_version_wedge_is_recoverable_by_rearm(tmp_path: Path) -> None:
    """Round-2 blocker: a database carrying the old persisted detector string
    (pre-rename) must not wedge conflict_scan_required forever."""
    from memory_arbiter.config import Settings
    from memory_arbiter.db import MemoryDB
    from memory_arbiter.db_generation import CONFLICT_DETECTOR_VERSION
    from memory_arbiter.models import MemoryRecord
    from memory_arbiter.vnext_migration import _mark_conflict_rebuild_ready

    settings = Settings(db_path=tmp_path / "wedge.db", backup_jsonl=tmp_path / "b.jsonl")
    db = MemoryDB(settings)
    db.insert_memory(
        MemoryRecord.from_input({"content": "x", "subject": "s", "workspace": "w"}, settings.defaults()),
        "w",
    )
    _mark_conflict_rebuild_ready(db)
    # Simulate a dev2 db that persisted the OLD detector identity.
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='conflict_group_detector_v1' "
            "WHERE key='conflict_scan_detector_version'"
        )
    assert db.rearm_conflict_scan_if_drifted() is True
    state = db.conflict_scan_state()
    assert state["detector_version"] == CONFLICT_DETECTOR_VERSION
    assert db.record_conflict_scan_page(
        epoch=state["epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=state["boundary"], after_memory_id=0, next_anchor_memory_id=None,
        anchors_scanned=1, workspace=None,
    ) is True
    assert db.complete_conflict_scan(
        epoch=state["epoch"], detector_version=CONFLICT_DETECTOR_VERSION, boundary=state["boundary"],
    ) is True
    assert db.conflict_scan_state()["required"] is False


def test_needs_authorization_step_routes_to_replan_not_livelock(tmp_path: Path) -> None:
    """Round-2 high: a needs_authorization step must not wedge applying."""
    tools, db = _tools(tmp_path)
    a, b = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(a, b))["data"]["conflict_id"]
    _judge(tools, conflict_id, [{"memory_id": a, "action": "needs_authorization"}], resolution=b)
    result = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": a,
        "action": "needs_authorization", "authorized": True,
    })
    assert result["ok"] is False
    assert result["data"]["action_required"] == "replan_conflict"
    row = db.get_conflict(conflict_id)
    assert row["status"] == "applying"
    assert row["apply_summary"]["plan"][0]["status"] == "blocked"
    # Guidance surfaces route to replan, not back to the failing apply call.
    detail = tools.memory_review("conflict_detail", {"conflict_id": conflict_id})
    assert detail["data"]["next_executable_call"]["action"] == "replan_conflict"
    # Replan to an executable plan then finish.
    replanned = tools.memory_govern("replan_conflict", {
        "conflict_id": conflict_id, "expected_revision": row["revision"],
        "apply_plan": [{"memory_id": a, "action": "update_current_claim"}],
        "authorized": True,
    })
    assert replanned["ok"] is True


def test_none_explicit_filter_scopes_before_limit(tmp_path: Path) -> None:
    """Round-2 high: the none-mode explicit filter must scope in SQL so the
    limit is applied after scoping (no silent truncation to empty)."""
    import tests.test_vnext_evidence as tv
    tools = tv.make_tools(tmp_path)
    # Many higher-ranking projB memories, one projA memory.
    for i in range(8):
        tools.memory_write(content=f"marketing note {i}", workspace="projB", subject=f"b{i}", tags=[])
    tools.memory_write(content="marketing note target", workspace="projA", subject="a", tags=[])
    assert tools.wait_evidence_worker_drained(timeout=2)
    caller_canonical = tools._caller_workspace("projA").canonical
    res = tools.memory_search(query="marketing", workspace="projA", limit=3)
    results = res["data"]["results"]
    assert results, "explicit projA filter must not truncate to empty"
    # Every returned row is in the caller's canonical scope (FakeEmbedder maps
    # all workspace names to one canonical, so scoping is by canonical here).
    assert all(
        (r.get("workspace_canonical") or r.get("workspace")) == caller_canonical
        for r in results
    )
