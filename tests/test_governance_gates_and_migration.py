"""Coverage for the v3 governance gates and migration/replay hardening slice.

B-C3: record_conflict requires authorized=true only for not_a_conflict
dispositions (ordinary open intake stays ungated).
B-E3: memory(action='judge') requires authorized=true when decided_by='user'.
B-C5: the default open conflict listing also surfaces applying groups whose
refreshed_at is older than 7 days, flagged stale_applying with replan guidance.
B-C4: conflict slot entity/scope are stored in canon form.
B-D2: memory_replay_backup drains the evidence worker before responding.
B-D4: final_sync fails fast when the source has an active writer.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
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


# ── B-C5: stale applying surfacing in the default open listing ──────────────

def test_fresh_applying_group_stays_out_of_default_open_listing(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    conflict_id, _, _ = _applying_conflict(tools, db)
    listing = tools.memory_list_conflicts(status="open", limit=50)
    assert conflict_id not in [c["id"] for c in listing["data"]["conflicts"]]


def test_stale_applying_group_surfaces_with_replan_guidance(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    conflict_id, _, _ = _applying_conflict(tools, db)
    stale_at = (datetime.now(timezone.utc) - timedelta(days=8)).replace(microsecond=0).isoformat()
    with db.write_transaction() as conn:
        conn.execute("UPDATE conflicts SET refreshed_at=? WHERE id=?", (stale_at, conflict_id))

    listing = tools.memory_list_conflicts(status="open", limit=50)
    surfaced = next(c for c in listing["data"]["conflicts"] if c["id"] == conflict_id)
    assert surfaced["stale_applying"] is True
    assert surfaced["next_action"]["tool"] == "memory_govern"
    assert surfaced["next_action"]["action"] == "replan_conflict"
    assert surfaced["next_action"]["data"]["authorized"] is True
    # Read-only surfacing: the conflict itself is unchanged.
    assert db.get_conflict(conflict_id)["status"] == "applying"


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
