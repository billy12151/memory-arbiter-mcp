"""Coverage for the v3 governance gates and migration/replay hardening slice.

B-C3: record_conflict requires authorized=true only for not_a_conflict
dispositions (ordinary open intake stays ungated).
B-E3: memory(action='judge') requires authorized=true when decided_by='user'.
B-C5 (superseded): the stale-applying list surfacing was removed by product
decision; doctor now reports all unresolved groups (conflicts.backlog counts
open+applying, conflicts.applying flags every mid-apply group).
B-C4: conflict slot entity/scope are stored in canon form.
B-D2: memory_replay_backup drains the evidence worker before responding.
B-D4: final_sync fails fast when the source has an active writer.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.tools import MemoryTools
from memory_arbiter.vnext_migration import final_sync


def _tools(tmp_path: Path) -> tuple[MemoryTools, MemoryDB]:
    settings = Settings(
        db_path=tmp_path / "gates.db", backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=False, isolation="none", workspace="default",
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


def _member(memory_id: int, version: int, value: str) -> dict:
    quote = f"database is {value}"
    return ConflictMember(
        memory_id=memory_id, version=version, attribute_raw="database", value_raw=value,
        normalized_attribute="database", normalized_value=value.casefold(), evidence_quote=quote,
        evidence_span=(0, len(quote)), content_hash=(f"{memory_id}@{version}" * 64)[:64],
        direction="a_to_b", prompt_version="p1", detector_version="d1",
    ).to_dict()


def _record_payload(left: int, right: int, *, status: str = "open",
                    slot: object = "default") -> dict:
    if slot == "default":
        slot = {"entity": "project", "attribute": "database", "scope": "global"}
    members = [_member(left, 1, "mysql"), _member(right, 1, "sqlite")]
    return {
        "slot_key": slot,
        "members": members,
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{left}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{right}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scheduled_scan",
        "reason": "different database values", "status": status,
    }


def _judge_payload(conflict_id: int, left: int, right: int, *, decided_by: str = "user") -> dict:
    return {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": decided_by, "ref": "chat", "reason": "confirmed",
        "apply_plan": [
            {"memory_id": left, "action": "update_current_claim"},
            {"memory_id": right, "action": "use_as_resolution"},
        ],
        "resolution_memory_id": right,
    }


def _applying_conflict(tools: MemoryTools, db: MemoryDB) -> tuple[int, int, int]:
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]
    judged = db.judge_conflict(
        conflict_id, expected_revision=1, chosen_value="sqlite",
        decided_by="agent", decided_ref=None, decision_reason="reviewed",
        apply_plan=[
            {"memory_id": left, "action": "update_current_claim"},
            {"memory_id": right, "action": "use_as_resolution"},
        ],
        resolution_memory_id=right,
    )
    assert judged["outcome"] == "applying", judged
    return conflict_id, left, right


# ── B-C3: record_conflict authorization gate ────────────────────────────────

def test_record_conflict_open_intake_needs_no_authorization(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    recorded = tools.memory_repair("record_conflict", _record_payload(left, right))
    assert recorded["ok"] is True, recorded
    assert recorded["data"]["outcome"] == "inserted"


def test_record_conflict_not_a_conflict_requires_authorization(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    denied = tools.memory_repair(
        "record_conflict", _record_payload(left, right, status="not_a_conflict", slot=None),
    )
    assert denied["ok"] is False
    assert denied["data"]["action_required"] == "ask_user_for_authorization"
    assert denied["data"]["authorized"] is False
    assert denied["data"]["retry"]["tool"] == "memory_repair"
    assert denied["data"]["retry"]["task"] == "record_conflict"

    allowed = tools.memory_repair("record_conflict", {
        **_record_payload(left, right, status="not_a_conflict", slot=None),
        "authorized": True,
    })
    assert allowed["ok"] is True, allowed
    assert allowed["data"]["outcome"] == "inserted"


# ── B-E3: judge authorization gate ──────────────────────────────────────────

def test_judge_decided_by_user_requires_authorization(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]

    denied = tools.memory("judge", _judge_payload(conflict_id, left, right))
    assert denied["ok"] is False
    assert denied["data"]["action_required"] == "ask_user_for_authorization"
    assert denied["data"]["governance_action"] == "judge"
    assert denied["data"]["retry"]["tool"] == "memory"
    # The denial is fail-fast: the conflict is untouched.
    assert db.get_conflict(conflict_id)["status"] == "open"

    allowed = tools.memory("judge", {
        **_judge_payload(conflict_id, left, right), "authorized": True,
    })
    assert allowed["ok"] is True, allowed
    assert allowed["data"]["outcome"] == "applying"


def test_judge_decided_by_agent_needs_no_authorization(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]
    judged = tools.memory("judge", _judge_payload(conflict_id, left, right, decided_by="agent"))
    assert judged["ok"] is True, judged
    assert judged["data"]["outcome"] == "applying"


# ── B-C5 (superseded): doctor reports unresolved groups; listing stays pure ──

def test_open_listing_excludes_applying_groups(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    conflict_id, _, _ = _applying_conflict(tools, db)
    listing = tools.memory_list_conflicts(status="open", limit=50)
    assert conflict_id not in [c["id"] for c in listing["data"]["conflicts"]]


def test_doctor_backlog_counts_open_and_applying(tmp_path: Path) -> None:
    from memory_arbiter.doctor import run_all_checks

    tools, db = _tools(tmp_path)
    left, right = _memory(db, "cache backend is redis"), _memory(db, "cache backend is memcached")
    # A different slot from _applying_conflict's default: recording a second
    # group on an occupied slot returns stale_conflict, not a new open group.
    tools.memory_repair("record_conflict", _record_payload(
        left, right, slot={"entity": "project", "attribute": "cache", "scope": "global"},
    ))
    _applying_conflict(tools, db)

    conn = sqlite3.connect(tools.settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        report = run_all_checks(conn, tools.settings)
    finally:
        conn.close()
    backlog = next(f for f in report.findings if f.check_id == "conflicts.backlog")
    assert "2 unresolved conflicts (1 open, 1 applying)" in backlog.detail
    assert backlog.evidence == {"open": 1, "applying": 1}


def test_doctor_flags_every_applying_group_with_idle_age(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from memory_arbiter.doctor import run_all_checks

    tools, db = _tools(tmp_path)
    conflict_id, _, _ = _applying_conflict(tools, db)
    stale_at = (datetime.now(timezone.utc) - timedelta(days=23)).replace(microsecond=0).isoformat()
    with db.write_transaction() as conn:
        conn.execute("UPDATE conflicts SET refreshed_at=? WHERE id=?", (stale_at, conflict_id))

    conn = sqlite3.connect(tools.settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        report = run_all_checks(conn, tools.settings)
    finally:
        conn.close()
    applying = next(f for f in report.findings if f.check_id == "conflicts.applying")
    assert applying.status == "warn"
    group = applying.evidence["groups"][0]
    assert group["id"] == conflict_id
    assert group["idle_days"] == 23


def test_doctor_applying_check_ok_when_none(tmp_path: Path) -> None:
    from memory_arbiter.doctor import run_all_checks

    tools, _ = _tools(tmp_path)
    conn = sqlite3.connect(tools.settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        report = run_all_checks(conn, tools.settings)
    finally:
        conn.close()
    applying = next(f for f in report.findings if f.check_id == "conflicts.applying")
    assert applying.status == "pass"
    assert applying.evidence == {"groups": []}


# ── B-C4: slot entity/scope stored in canon form ────────────────────────────

def test_normalize_slot_canonicalizes_entity_and_scope(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    payload = _record_payload(
        left, right,
        slot={"entity": "  MyProject ", "attribute": "Database", "scope": "PRODUCTION"},
    )
    recorded = tools.memory_repair("record_conflict", payload)
    assert recorded["ok"] is True, recorded
    slot = db.get_conflict(recorded["data"]["conflict_id"])["slot_key"]
    assert slot["entity"] == "myproject"
    assert slot["scope"] == "production"
    # Attribute is detector-owned and keeps its raw-stripped form.
    assert slot["attribute"] == "Database"


# ── B-D2: replay drains the evidence worker before responding ───────────────

def test_replay_backup_drains_evidence_worker(tmp_path: Path, monkeypatch) -> None:
    tools, db = _tools(tmp_path)
    calls: list[float] = []

    def _fake_drain(timeout: float = 30.0) -> bool:
        calls.append(timeout)
        return True

    monkeypatch.setattr(tools, "wait_evidence_worker_drained", _fake_drain)
    result = tools.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert result["ok"] is True, result
    assert calls, "replay must drain the evidence worker before responding"
    assert result["data"]["evidence_worker_drained"] is True

    calls.clear()
    preview = tools.memory_repair("replay_backup", {"dry_run": True})
    assert preview["ok"] is True
    assert calls == [], "dry-run inspection must not drain"


# ── B-D4: final_sync fails fast on an active source writer ──────────────────

def test_final_sync_fails_fast_with_active_source_writer(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    setup = sqlite3.connect(source)
    setup.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT)")
    setup.commit()
    setup.close()

    writer = sqlite3.connect(source)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO memories (content) VALUES ('in-flight write')")
    try:
        result = final_sync(
            source, target,
            Settings(db_path=target, backup_jsonl=tmp_path / "backup.jsonl"),
            progress=False,
        )
    finally:
        writer.close()
    assert result["ok"] is False
    assert result["error"] == "source_has_active_writer"
    assert "stop all writers" in result["next_step"]
    # Fail-fast: no staging rebuild was started.
    assert not target.with_name(target.name + ".finalizing").exists()


# ── Write-time prompt: editing a member of an unresolved conflict group ─────

def test_editing_open_group_member_prompts_unresolved_conflicts(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    tools.memory_repair("record_conflict", _record_payload(left, right))
    response = tools.memory_edit(left, new_content="database is mysql, per legacy notes")
    assert response["data"]["edited"] is True
    assert response["data"]["attention_required"] is True
    unresolved = response["data"]["unresolved_conflicts"]
    assert len(unresolved) == 1
    assert unresolved[0]["status"] == "open"
    assert response["data"]["action_required"] == "review_unresolved_conflicts"


def test_editing_applying_group_member_prompts_unresolved_conflicts(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    conflict_id, left, _ = _applying_conflict(tools, db)
    response = tools.memory_edit(left, new_content="database is mysql, legacy")
    assert response["data"]["edited"] is True
    unresolved = response["data"]["unresolved_conflicts"]
    assert [(g["conflict_id"], g["status"]) for g in unresolved] == [(conflict_id, "applying")]


def test_editing_nonmember_memory_stays_silent(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    tools.memory_repair("record_conflict", _record_payload(left, right))
    bystander = _memory(db, "unrelated fact about caching")
    response = tools.memory_edit(bystander, new_content="unrelated fact about caching, updated")
    assert response["data"]["edited"] is True
    assert "attention_required" not in response["data"]


def test_tags_only_edit_of_member_stays_silent(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    tools.memory_repair("record_conflict", _record_payload(left, right))
    response = tools.memory_edit(left, tags_only=True, add_tags=["legacy"])
    assert response["data"]["edited"] is True
    assert "attention_required" not in response["data"]


# ── #3: TrustedApplyingContext typed boundary ───────────────────────────────

def test_trusted_applying_context_roundtrip_and_malformed() -> None:
    from memory_arbiter.models import TrustedApplyingContext

    ctx = TrustedApplyingContext(
        conflict_id=7, revision=3, memory_id=42, action="update_current_claim",
        chosen_value="sqlite",
    )
    assert TrustedApplyingContext.from_dict(ctx.to_dict()) == ctx
    # Malformed snapshots (typos / missing keys / wrong types) degrade to
    # None instead of half-engaging the §15.3 suppression.
    assert TrustedApplyingContext.from_dict(None) is None
    assert TrustedApplyingContext.from_dict("not-a-dict") is None
    assert TrustedApplyingContext.from_dict({"conflict_id": 7}) is None
    assert TrustedApplyingContext.from_dict({
        "conflict_id": "x", "revision": 3, "memory_id": 42, "action": "a",
    }) is None
