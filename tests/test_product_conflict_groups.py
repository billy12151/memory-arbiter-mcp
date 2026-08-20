from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.tools import MemoryTools


def _tools(tmp_path: Path, *, isolation: str = "none") -> tuple[MemoryTools, MemoryDB]:
    settings = Settings(
        db_path=tmp_path / "product.db", backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=False, isolation=isolation, workspace="default",
    )
    db = MemoryDB(settings)
    return MemoryTools(settings, db), db


def _memory(db: MemoryDB, content: str) -> int:
    memory_id, _ = db.insert_memory(MemoryRecord(content=content, subject="database", agent_id="test", workspace="default"), "default")
    assert memory_id is not None
    return memory_id


def _member(memory_id: int, value: str) -> dict:
    quote = f"database is {value}"
    return ConflictMember(
        memory_id=memory_id, version=1, attribute_raw="database", value_raw=value,
        normalized_attribute="database", normalized_value=value.casefold(), evidence_quote=quote,
        evidence_span=(0, len(quote)), content_hash=(str(memory_id) * 64)[:64], direction="a_to_b",
        prompt_version="p1", detector_version="d1",
    ).to_dict()


def _record_payload(left: int, right: int) -> dict:
    members = [_member(left, "mysql"), _member(right, "sqlite")]
    return {
        "slot_key": {"entity": "project", "attribute": "database", "scope": "global"},
        "members": members,
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{left}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{right}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scheduled_scan",
        "reason": "different database values", "status": "open",
    }


def test_product_record_judge_apply_and_resolve(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    mysql, sqlite = _memory(db, "database is mysql"), _memory(db, "database is sqlite")

    recorded = tools.memory_repair("record_conflict", _record_payload(mysql, sqlite))
    assert recorded["ok"] is True
    conflict_id = recorded["data"]["conflict_id"]

    judged = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [
            {"memory_id": mysql, "action": "update_current_claim"},
            {"memory_id": sqlite, "action": "use_as_resolution"},
        ],
        "resolution_memory_id": sqlite,
    })
    assert judged["ok"] is True
    assert judged["data"]["next_action"]["action"] == "apply_conflict_action"

    applied = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": mysql,
        "action": "update_current_claim", "content": "database is sqlite",
        "reason": "apply confirmed value", "authorized": True,
    })
    assert applied["ok"] is True
    assert db.get_memory(mysql)["content"] == "database is sqlite"
    assert db.get_conflict(conflict_id)["member_versions"][0]["version"] == 1

    preserved = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 3, "memory_id": sqlite,
        "action": "use_as_resolution", "authorized": True,
    })
    assert preserved["ok"] is True
    assert preserved["data"]["next_action"]["action"] == "resolve_conflict"

    resolved = tools.memory_govern("resolve_conflict", {
        "conflict_id": conflict_id, "expected_revision": 4,
        "reason": "plan complete", "authorized": True,
    })
    assert resolved["ok"] is True
    assert db.get_conflict(conflict_id)["status"] == "resolved"


def test_apply_conflict_action_requires_authorization_and_cas(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "agent", "ref": None, "reason": "reviewed",
        "apply_plan": [{"memory_id": left, "action": "update_current_claim"}],
        "resolution_memory_id": right,
    })
    payload = {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": left,
        "action": "update_current_claim", "content": "database is sqlite",
    }
    denied = tools.memory_govern("apply_conflict_action", payload)
    assert denied["ok"] is False and denied["data"]["action_required"] == "ask_user_for_authorization"
    stale = tools.memory_govern("apply_conflict_action", {**payload, "expected_revision": 1, "authorized": True})
    assert stale["ok"] is False and stale["data"]["outcome"] == "stale_conflict"
    assert db.get_memory(left)["content"] == "database is mysql"


def test_apply_rejects_ungrounded_value_then_authorized_replan_preserves_history(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]
    assert tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": left, "action": "update_current_claim"}],
        "resolution_memory_id": right,
    })["ok"] is True

    failed = tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": left,
        "action": "update_current_claim", "content": "database remains mysql",
        "authorized": True,
    })
    assert failed["ok"] is False
    assert failed["data"]["outcome"] == "apply_failed"
    assert failed["data"]["action_required"] == "replan_conflict"
    assert db.get_conflict(conflict_id)["status"] == "applying"

    replanned = tools.memory_govern("replan_conflict", {
        "conflict_id": conflict_id, "expected_revision": 3,
        "apply_plan": [{"memory_id": left, "action": "update_current_claim"}],
        "resolution_memory_id": right, "authorized": True,
    })
    assert replanned["ok"] is True
    summary = db.get_conflict(conflict_id)["apply_summary"]
    assert summary["plan"][0]["expected_version"] == 2
    assert summary["history"][0]["plan"][0]["status"] == "failed"


def test_cross_workspace_record_and_resolution_are_rejected_transactionally(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left = _memory(db, "database is mysql")
    foreign_id, _ = db.insert_memory(
        MemoryRecord(content="database is sqlite", subject="database", agent_id="test", workspace="foreign"),
        "foreign",
    )
    assert foreign_id is not None
    cross = tools.memory_repair("record_conflict", _record_payload(left, foreign_id))
    assert cross["ok"] is False
    assert cross["data"]["outcome"] == "workspace_mismatch"
    assert db.list_conflicts("open", 10) == []

    right = _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]
    judged = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": left, "action": "update_current_claim"}],
        "resolution_memory_id": foreign_id,
    })
    assert judged["ok"] is False
    assert judged["data"]["outcome"] == "invalid_resolution_memory"
    assert db.get_conflict(conflict_id)["status"] == "open"


def test_judge_rejects_unrelated_chosen_value_without_mutation(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]

    judged = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "postgres",
        "decided_by": "user", "ref": "chat", "reason": "unrelated value",
        "apply_plan": [
            {"memory_id": left, "action": "preserve_historical_record"},
            {"memory_id": right, "action": "preserve_historical_record"},
        ],
        "resolution_memory_id": right,
    })

    assert judged["ok"] is False
    assert judged["data"]["outcome"] == "invalid_chosen_value"
    conflict = db.get_conflict(conflict_id)
    assert conflict["status"] == "open"
    assert conflict["revision"] == 1
    assert conflict["chosen_value"] is None
    assert conflict["apply_summary"] == {"plan": []}


def test_strict_closure_revalidates_all_member_workspaces_transactionally(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path, isolation="strict")
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair(
        "record_conflict", {**_record_payload(left, right), "workspace": "default"},
    )["data"]["conflict_id"]

    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET workspace='foreign',workspace_canonical='foreign' WHERE id=?",
            (right,),
        )
    before = db.get_conflict(conflict_id)
    judged = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": left, "action": "preserve_historical_record"}],
        "resolution_memory_id": left, "workspace": "default",
    })
    assert judged["ok"] is False
    assert db.get_conflict(conflict_id) == before

    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET workspace='default',workspace_canonical='default' WHERE id=?",
            (right,),
        )
    judged = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": left, "action": "preserve_historical_record"}],
        "resolution_memory_id": right, "workspace": "default",
    })
    assert judged["ok"] is True

    for operation, payload in (
        ("apply_conflict_action", {"memory_id": left, "action": "preserve_historical_record"}),
        ("replan_conflict", {"apply_plan": [{"memory_id": left, "action": "preserve_historical_record"}]}),
        ("resolve_conflict", {}),
    ):
        conflict = db.get_conflict(conflict_id)
        with db.write_transaction() as conn:
            conn.execute(
                "UPDATE memories SET workspace='foreign',workspace_canonical='foreign' WHERE id=?",
                (right,),
            )
        denied = tools.memory_govern(operation, {
            "conflict_id": conflict_id, "expected_revision": conflict["revision"],
            "workspace": "default", "authorized": True, **payload,
        })
        assert denied["ok"] is False
        assert db.get_conflict(conflict_id) == conflict
        with db.write_transaction() as conn:
            conn.execute(
                "UPDATE memories SET workspace='default',workspace_canonical='default' WHERE id=?",
                (right,),
            )


def test_resolution_version_must_still_be_active_and_pinned(tmp_path: Path) -> None:
    tools, db = _tools(tmp_path)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": right, "action": "use_as_resolution"}],
        "resolution_memory_id": right,
    })
    tools.memory_govern("apply_conflict_action", {
        "conflict_id": conflict_id, "expected_revision": 2, "memory_id": right,
        "action": "use_as_resolution", "authorized": True,
    })
    db.edit_memory_intent(right, new_content="database is postgres", expected_version=1)
    resolved = tools.memory_govern("resolve_conflict", {
        "conflict_id": conflict_id, "expected_revision": 3, "authorized": True,
    })
    assert resolved["ok"] is False
    assert resolved["data"]["outcome"] == "stale_resolution_memory"


def test_removed_judgment_surface_and_pair_payload_are_rejected(tmp_path: Path) -> None:
    tools, _ = _tools(tmp_path)
    assert tools.memory_review("judgments", {"conflict_id": 1})["ok"] is False
    assert tools.memory_govern("correct_judgment", {"conflict_id": 1, "authorized": True})["ok"] is False
    pair = tools.memory_repair("record_conflict", {"left_id": 1, "right_id": 2, "reason": "legacy"})
    assert pair["ok"] is False
    assert any("left_id" in warning for warning in pair.get("warnings", []))
