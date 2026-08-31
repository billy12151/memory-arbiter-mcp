# ── from test_conflict_groups.py ──

from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.tools import MemoryTools


def _db(tmp_path: Path) -> MemoryDB:
    return MemoryDB(Settings(db_path=tmp_path / "memory.db", backup_jsonl=tmp_path / "backup.jsonl"))


def _memory(db: MemoryDB, content: str, workspace: str = "w") -> int:
    memory_id, _ = db.insert_memory(MemoryRecord(content=content, subject="database", agent_id="a", workspace=workspace), workspace)
    assert memory_id is not None
    return memory_id


def _member(memory_id: int, value: str, version: int = 1) -> ConflictMember:
    quote = f"database is {value}"
    return ConflictMember(
        memory_id=memory_id, version=version, attribute_raw="database", value_raw=value,
        normalized_attribute="database", normalized_value=value.casefold(), evidence_quote=quote,
        evidence_span=(0, len(quote)), content_hash=(str(memory_id) * 64)[:64], direction="a_to_b",
        prompt_version="p1", detector_version="d1",
    )


def _groups(*members: ConflictMember) -> list[ConflictValueGroup]:
    by_value: dict[str, list[str]] = {}
    for member in members:
        by_value.setdefault(member.normalized_value, []).append(f"{member.memory_id}@{member.version}")
    return [ConflictValueGroup(value, value.title(), tuple(refs)) for value, refs in by_value.items()]


def _record(db: MemoryDB, members: list[ConflictMember], *, expected_revision=None, status="open", slot=True):
    return db.record_conflict_group(
        workspace_canonical="w",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"} if slot else None,
        members=members, value_groups=_groups(*members), detection_reason="different database",
        source="scan", detector_version="d1", prompt_version="p1", conflict_point="database",
        status=status, expected_revision=expected_revision,
    )


def test_fresh_schema_has_single_conflict_table_and_group_indexes(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with db.connection() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflict_judgments'").fetchone() is None
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_notices'").fetchone() is None
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(conflicts)")}
        assert {
            "revision", "slot_key_hash", "candidate_key_hash", "member_versions",
            "value_groups", "apply_summary", "notice_delivery_status",
            "notice_task_id", "notice_dedupe_key", "notice_payload",
        } <= columns
        indexes = {row["name"] for row in conn.execute("PRAGMA index_list(conflicts)")}
        assert {
            "idx_conflicts_active_slot", "idx_conflicts_candidate_identity",
            "idx_conflicts_event_snapshot", "idx_conflicts_notice_dedupe",
        } <= indexes


def test_open_group_appends_members_with_cas_and_dedupes_member_versions(tmp_path: Path) -> None:
    db = _db(tmp_path)
    mysql_a, sqlite, mysql_b = (_memory(db, value) for value in ("mysql-a", "sqlite", "mysql-b"))
    first_members = [_member(mysql_a, "mysql"), _member(sqlite, "sqlite")]
    first = _record(db, first_members)
    assert first["outcome"] == "inserted" and first["revision"] == 1

    stale = _record(db, [_member(mysql_b, "mysql"), _member(sqlite, "sqlite")], expected_revision=0)
    assert stale["outcome"] == "stale_conflict"
    appended = _record(db, [_member(mysql_b, "mysql"), _member(sqlite, "sqlite")], expected_revision=1)
    assert appended["outcome"] == "appended" and appended["revision"] == 2
    row = db.get_conflict(first["conflict_id"])
    assert row is not None
    assert [f"{m['memory_id']}@{m['version']}" for m in row["member_versions"]].count(f"{sqlite}@1") == 1
    assert len(row["member_versions"]) == 3


def test_candidate_only_not_a_conflict_is_idempotent_without_fake_slot(tmp_path: Path) -> None:
    db = _db(tmp_path)
    a, b = _memory(db, "a"), _memory(db, "b")
    members = [_member(a, "mysql"), _member(b, "sqlite")]
    first = _record(db, members, status="not_a_conflict", slot=False)
    second = _record(db, members, status="not_a_conflict", slot=False)
    assert first["outcome"] == "inserted"
    assert second == {"outcome": "deduped", "conflict_id": first["conflict_id"], "revision": 1}
    row = db.get_conflict(first["conflict_id"])
    assert row is not None and row["slot_key"] is None and row["slot_key_hash"] is None
    changed = [_member(a, "mysql", version=2), _member(b, "sqlite")]
    assert _record(db, changed, status="not_a_conflict", slot=False)["outcome"] == "stale_snapshot"


def test_open_applying_apply_bookkeeping_and_resolve_only_completed(tmp_path: Path) -> None:
    db = _db(tmp_path)
    mysql, sqlite = _memory(db, "mysql"), _memory(db, "sqlite")
    created = _record(db, [_member(mysql, "mysql"), _member(sqlite, "sqlite")])
    judged = db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="sqlite", decided_by="user",
        decided_ref="chat", decision_reason="confirmed", resolution_memory_id=sqlite,
        apply_plan=[{"memory_id": mysql, "action": "update_current_claim"},
                    {"memory_id": sqlite, "action": "use_as_resolution"}],
    )
    assert judged["outcome"] == "applying" and judged["revision"] == 2
    assert db.resolve_conflict(created["conflict_id"], expected_revision=2)["outcome"] == "apply_incomplete"

    # Apply bookkeeping is owned by the atomic governance lifecycle; the DB no
    # longer exposes a second result-only facade that can claim an unapplied step.
    assert not hasattr(db, "record_conflict_apply_result")

    # Complete a fresh group through the lifecycle API to test resolution.
    other_mysql, other_sqlite = _memory(db, "mysql-new"), _memory(db, "sqlite-new")
    created2 = db.record_conflict_group(
        workspace_canonical="w", slot_key={"entity": "project2", "attribute": "database", "scope": "global"},
        members=[_member(other_mysql, "mysql"), _member(other_sqlite, "sqlite")],
        value_groups=_groups(_member(other_mysql, "mysql"), _member(other_sqlite, "sqlite")),
        detection_reason="different", source="scan", detector_version="d1",
    )
    judged2 = db.judge_conflict(
        created2["conflict_id"], expected_revision=1, chosen_value="sqlite", decided_by="agent",
        decided_ref=None, decision_reason="approved", apply_plan=[
            {"memory_id": other_mysql, "action": "preserve_historical_record"},
            {"memory_id": other_sqlite, "action": "use_as_resolution"},
        ], resolution_memory_id=other_sqlite,
    )
    tools = MemoryTools(db.settings, db)
    step1 = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created2["conflict_id"], "expected_revision": judged2["revision"],
        "memory_id": other_mysql, "action": "preserve_historical_record", "authorized": True,
    })["data"]
    stale = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created2["conflict_id"], "expected_revision": judged2["revision"],
        "memory_id": other_sqlite, "action": "use_as_resolution", "authorized": True,
    })
    assert stale["data"]["outcome"] == "stale_conflict"
    step2 = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created2["conflict_id"], "expected_revision": step1["revision"],
        "memory_id": other_sqlite, "action": "use_as_resolution", "authorized": True,
    })["data"]
    resolved = db.resolve_conflict(created2["conflict_id"], expected_revision=step2["revision"])
    assert resolved["outcome"] == "resolved"
    assert _record(db, [_member(other_mysql, "mysql"), _member(other_sqlite, "sqlite")], expected_revision=resolved["revision"])["outcome"] != "appended"


def test_judge_and_replan_reject_non_object_plan_entries(tmp_path: Path) -> None:
    db = _db(tmp_path)
    left, right = _memory(db, "mysql"), _memory(db, "sqlite")
    created = _record(db, [_member(left, "mysql"), _member(right, "sqlite")])

    judged = db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="sqlite",
        decided_by="user", decided_ref="chat", decision_reason="confirmed",
        apply_plan=["bad"],
    )
    assert judged["outcome"] == "invalid_plan"

    applying = db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="sqlite",
        decided_by="user", decided_ref="chat", decision_reason="confirmed",
        apply_plan=[{"memory_id": left, "action": "preserve_historical_record"}],
    )
    assert applying["outcome"] == "applying"
    replanned = db.conflicts.replan_conflict(
        created["conflict_id"], expected_revision=applying["revision"],
        apply_plan=["bad"],
    )
    assert replanned["outcome"] == "invalid_plan"


def test_conflict_detail_exposes_group_snapshot_resolution_and_next_call(tmp_path: Path) -> None:
    db = _db(tmp_path)
    tools = MemoryTools(db.settings, db)
    mysql, sqlite = _memory(db, "mysql"), _memory(db, "sqlite")
    created = _record(db, [_member(mysql, "mysql"), _member(sqlite, "sqlite")])

    detail = tools._conflict_detail_for_workspace(created["conflict_id"])
    assert detail is not None
    assert detail["revision"] == 1
    assert detail["slot"] == {"entity": "project", "attribute": "database", "scope": "global"}
    assert len(detail["member_versions"]) == 2
    assert len(detail["value_groups"]) == 2
    assert detail["resolution_memory"] is None
    assert detail["apply_summary"] == {"plan": []}
    assert detail["next_executable_call"] == {
        "tool": "memory", "action": "judge",
        "data": {"conflict_id": created["conflict_id"], "expected_revision": 1},
    }

    db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="sqlite",
        decided_by="user", decided_ref="chat", decision_reason="confirmed",
        apply_plan=[{"memory_id": mysql, "action": "update_current_claim"}],
        resolution_memory_id=sqlite,
    )
    applying = tools._conflict_detail_for_workspace(created["conflict_id"])
    assert applying is not None
    assert applying["revision"] == 2
    assert applying["resolution_memory"]["memory"]["id"] == sqlite
    assert applying["resolution_memory_version"] == 1
    assert applying["next_executable_call"]["action"] == "apply_conflict_action"
    assert applying["next_executable_call"]["data"]["expected_revision"] == 2


def test_strict_conflict_detail_requires_all_members_and_redacts_group_fields(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "strict.db", backup_jsonl=tmp_path / "strict.jsonl",
        isolation="strict", workspace="alpha",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings, db)
    visible = _memory(db, "visible mysql", "alpha")
    hidden = _memory(db, "HIDDEN sqlite secret", "alpha")
    created = db.record_conflict_group(
        workspace_canonical="alpha",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=[_member(visible, "mysql"), _member(hidden, "sqlite")],
        value_groups=_groups(_member(visible, "mysql"), _member(hidden, "sqlite")),
        detection_reason="HIDDEN conflict reason", source="scan", detector_version="d1",
        conflict_point="HIDDEN database point",
    )
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET workspace='beta',workspace_canonical='beta' WHERE id=?", (hidden,),
        )

    detail = tools._conflict_detail_for_workspace(created["conflict_id"], tools._caller_workspace("alpha"))
    assert detail is None
    reviewed = tools.memory_review(
        "conflict_detail", {"conflict_id": created["conflict_id"], "workspace": "alpha"},
    )
    assert reviewed["ok"] is False
    assert reviewed["data"]["error"] == "conflict id not found"
    assert "HIDDEN" not in str(reviewed)


def test_signals_link_open_and_applying_groups_by_members(tmp_path: Path) -> None:
    db = _db(tmp_path)
    tools = MemoryTools(db.settings, db)
    mysql, sqlite, postgres = (_memory(db, value) for value in ("mysql", "sqlite", "postgres"))
    created = _record(db, [_member(mysql, "mysql"), _member(sqlite, "sqlite"), _member(postgres, "postgres")])
    rows = [db.get_memory(mysql)]
    warnings: list[str] = []
    tools._attach_conflict_signals(rows, warnings)
    signal = rows[0]["conflict_signal"]
    assert warnings == []
    assert signal["conflict_source"] == "conflict_group"
    assert signal["member_count"] == 3
    assert signal["action_required"] == "judge_conflict"

    db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="sqlite", decided_by="agent",
        decided_ref=None, decision_reason="approved",
        apply_plan=[{"memory_id": mysql, "action": "update_current_claim"}],
    )
    rows = [db.get_memory(mysql)]
    tools._attach_conflict_signals(rows, warnings)
    assert rows[0]["conflict_signal"]["conflict_status"] == "applying"
    assert rows[0]["conflict_signal"]["action_required"] == "apply_conflict_action"


def test_candidate_key_and_value_groups_are_exactly_verified(tmp_path: Path) -> None:
    db = _db(tmp_path)
    left, right = _memory(db, "mysql"), _memory(db, "sqlite")
    members = [_member(left, "mysql"), _member(right, "sqlite")]
    valid_key = {
        "detector_version": "d1",
        "members": [f"{left}@1", f"{right}@1"],
        "evidence": [
            {"member": f"{left}@1", "unit": None, "span": [0, len("database is mysql")], "hash": (str(left) * 64)[:64]},
            {"member": f"{right}@1", "unit": None, "span": [0, len("database is sqlite")], "hash": (str(right) * 64)[:64]},
        ],
    }
    base = {
        "workspace_canonical": "w",
        "slot_key": {"entity": "project", "attribute": "database", "scope": "global"},
        "members": members, "value_groups": _groups(*members), "detection_reason": "different",
        "source": "scan", "detector_version": "d1", "candidate_key": valid_key,
    }
    assert db.record_conflict_group(**base)["outcome"] == "inserted"

    malformed = {**valid_key, "members": list(reversed(valid_key["members"]))}
    assert db.record_conflict_group(**{**base, "candidate_key": malformed})["outcome"] == "invalid_input"
    wrong_hash = {**valid_key, "evidence": [dict(valid_key["evidence"][0], hash="f" * 64), valid_key["evidence"][1]]}
    assert db.record_conflict_group(**{**base, "candidate_key": wrong_hash})["outcome"] == "invalid_input"

    missing = _groups(*members)
    missing[0] = ConflictValueGroup(missing[0].normalized_value, missing[0].display_value, ())
    assert db.record_conflict_group(**{**base, "value_groups": missing, "candidate_key": None})["outcome"] == "invalid_input"
    wrong_value = _groups(*members)
    wrong_value[0] = ConflictValueGroup("wrong", wrong_value[0].display_value, wrong_value[0].members)
    assert db.record_conflict_group(**{**base, "value_groups": wrong_value, "candidate_key": None})["outcome"] == "invalid_input"

    oversized_slot = {"entity": "x" * 4096, "attribute": "database", "scope": "global"}
    oversized = db.record_conflict_group(**{**base, "slot_key": oversized_slot, "candidate_key": None})
    assert oversized["outcome"] == "invalid_input"
    assert "slot_key exceeds" in oversized["error"]


def test_member_linkage_and_dismissal_have_no_result_caps(tmp_path: Path) -> None:
    db = _db(tmp_path)
    target, peer = _memory(db, "target"), _memory(db, "peer")
    members = [_member(target, "mysql"), _member(peer, "sqlite")]
    dismissed = _record(db, members, status="not_a_conflict", slot=False)
    assert dismissed["outcome"] == "inserted"
    with db.write_transaction() as conn:
        template = conn.execute("SELECT * FROM conflicts WHERE id=?", (dismissed["conflict_id"],)).fetchone()
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(conflicts)") if row["name"] != "id"]
        values = [template[column] for column in columns]
        for index in range(10_005):
            changed = list(values)
            changed[columns.index("candidate_key_hash")] = f"{index:064x}"
            changed[columns.index("member_fingerprint")] = f"{index + 20_000:064x}"
            conn.execute(
                f"INSERT INTO conflicts({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                changed,
            )
    assert db.is_pair_dismissed(target, peer) is True
    assert (target, peer) in db.dismissed_pairs_for([target])


# ── from test_product_conflict_groups.py ──
# helpers renamed: _product_memory, _product_member, _product_tools (collisions with test_conflict_groups.py)


from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.tools import MemoryTools


def _product_tools(tmp_path: Path, *, isolation: str = "none") -> tuple[MemoryTools, MemoryDB]:
    settings = Settings(
        db_path=tmp_path / "product.db", backup_jsonl=tmp_path / "backup.jsonl",
        isolation=isolation, workspace="default",
    )
    db = MemoryDB(settings)
    return MemoryTools(settings, db), db


def _product_memory(db: MemoryDB, content: str) -> int:
    memory_id, _ = db.insert_memory(MemoryRecord(content=content, subject="database", agent_id="test", workspace="default"), "default")
    assert memory_id is not None
    return memory_id


def _product_member(memory_id: int, value: str) -> dict:
    quote = f"database is {value}"
    return ConflictMember(
        memory_id=memory_id, version=1, attribute_raw="database", value_raw=value,
        normalized_attribute="database", normalized_value=value.casefold(), evidence_quote=quote,
        evidence_span=(0, len(quote)), content_hash=(str(memory_id) * 64)[:64], direction="a_to_b",
        prompt_version="p1", detector_version="d1",
    ).to_dict()


def _record_payload(left: int, right: int) -> dict:
    members = [_product_member(left, "mysql"), _product_member(right, "sqlite")]
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
    tools, db = _product_tools(tmp_path)
    mysql, sqlite = _product_memory(db, "database is mysql"), _product_memory(db, "database is sqlite")

    recorded = tools.memory_repair("record_conflict", _record_payload(mysql, sqlite))
    assert recorded["ok"] is True
    conflict_id = recorded["data"]["conflict_id"]

    judged = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed", "authorized": True,
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
    tools, db = _product_tools(tmp_path)
    left, right = _product_memory(db, "database is mysql"), _product_memory(db, "database is sqlite")
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
    tools, db = _product_tools(tmp_path)
    left, right = _product_memory(db, "database is mysql"), _product_memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]
    assert tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed", "authorized": True,
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
    tools, db = _product_tools(tmp_path)
    left = _product_memory(db, "database is mysql")
    foreign_id, _ = db.insert_memory(
        MemoryRecord(content="database is sqlite", subject="database", agent_id="test", workspace="foreign"),
        "foreign",
    )
    assert foreign_id is not None
    cross = tools.memory_repair("record_conflict", _record_payload(left, foreign_id))
    assert cross["ok"] is False
    assert cross["data"]["outcome"] == "workspace_mismatch"
    assert db.list_conflicts("open", 10) == []

    right = _product_memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]
    judged = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed", "authorized": True,
        "apply_plan": [{"memory_id": left, "action": "update_current_claim"}],
        "resolution_memory_id": foreign_id,
    })
    assert judged["ok"] is False
    assert judged["data"]["outcome"] == "invalid_resolution_memory"
    assert db.get_conflict(conflict_id)["status"] == "open"


def test_judge_rejects_unrelated_chosen_value_without_mutation(tmp_path: Path) -> None:
    tools, db = _product_tools(tmp_path)
    left, right = _product_memory(db, "database is mysql"), _product_memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]

    judged = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "postgres",
        "decided_by": "user", "ref": "chat", "reason": "unrelated value", "authorized": True,
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
    tools, db = _product_tools(tmp_path, isolation="strict")
    left, right = _product_memory(db, "database is mysql"), _product_memory(db, "database is sqlite")
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
        "decided_by": "user", "ref": "chat", "reason": "confirmed", "authorized": True,
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
        "decided_by": "user", "ref": "chat", "reason": "confirmed", "authorized": True,
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
    tools, db = _product_tools(tmp_path)
    left, right = _product_memory(db, "database is mysql"), _product_memory(db, "database is sqlite")
    conflict_id = tools.memory_repair("record_conflict", _record_payload(left, right))["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed", "authorized": True,
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
    tools, _ = _product_tools(tmp_path)
    assert tools.memory_review("judgments", {"conflict_id": 1})["ok"] is False
    assert tools.memory_govern("correct_judgment", {"conflict_id": 1, "authorized": True})["ok"] is False
    pair = tools.memory_repair("record_conflict", {"left_id": 1, "right_id": 2, "reason": "legacy"})
    assert pair["ok"] is False
    assert any("left_id" in warning for warning in pair.get("warnings", []))


# ── from test_conflict_notice_hardening.py ──

"""Regression tests for the v3 conflict-notice hardening slice.

Covers: model output with missing parsed keys no longer raising KeyError on
the notice path, per-task degradation counting deduplicated by reason,
pending-workspace memories reporting a skipped (not incomplete) conflict job,
and slot_key entity/scope canonicalisation with raw+canon double-form
suppression matching against applying conflict groups (B-C4).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.test_vnext_evidence as tv
from memory_arbiter.evidence import evidence_content_hash
from memory_arbiter.models import ConflictMember, ConflictValueGroup
from memory_arbiter.semantic_conflict import ModelSignal


def _write_pair(tools, meta: dict, left: str = "database is mysql",
                right: str = "database is sqlite") -> tuple[dict, dict]:
    peer = tools.memory_write(content=left, subject="a", tags=[], metadata=meta)["data"]
    new = tools.memory_write(content=right, subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    return peer, new


def _hit(peer_id: int, text: str, *, row_id: int = 1, distance: float = 0.1) -> dict:
    return {
        "memory_id": peer_id, "id": row_id, "kind": "text", "text": text,
        "start_offset": 0, "end_offset": len(text), "distance": distance,
    }


def test_notice_value_groups_tolerate_missing_parsed_keys(tmp_path: Path, monkeypatch) -> None:
    """A notice_ready gate with value keys missing from parsed must not KeyError."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "MyProject", "scope": "Production"}
    peer, new = _write_pair(tools, meta)
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: [_hit(peer["id"], "database is mysql")])

    class Backend:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            # A well-formed signal whose parsed dict omits the value keys.
            parsed = {"attribute_a": "数据库选型", "attribute_b": "数据库选型"}
            return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: Backend())
    gate = SimpleNamespace(
        state="notice_ready", reason="bidirectional_conflict",
        attribute="数据库选型", value_a="sqlite", value_b="mysql",
    )
    monkeypatch.setattr(
        "memory_arbiter.pipeline.evidence.evaluate_pair_extractions", lambda *a, **k: gate,
    )

    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert result["outcome"] == "notices_created"
    notice = tools.db.list_semantic_notices(status="open", limit=10)[0]
    groups = notice["payload"]["value_groups"]
    # Display values fall back to the gate's normalised values.
    assert [group["display_value"] for group in groups] == ["sqlite", "mysql"]
    # B-C4: the notice slot_key is stored in canonical entity/scope form.
    assert notice["payload"]["slot_key"] == {
        "entity": "myproject", "attribute": "数据库选型", "scope": "production",
    }


def test_same_reason_degradation_counted_once_per_task(tmp_path: Path, monkeypatch) -> None:
    """Two pairs failing with the same technical reason bump the counter once."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    peers = [
        tools.memory_write(content=f"database is {value}", subject=value, tags=[], metadata=meta)["data"]
        for value in ("mysql", "postgres")
    ]
    new = tools.memory_write(content="database is sqlite", subject="sqlite", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    hits = [
        _hit(peers[0]["id"], "database is mysql", row_id=1, distance=0.1),
        _hit(peers[1]["id"], "database is postgres", row_id=2, distance=0.2),
    ]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))

    class BadOutput:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            return ModelSignal(False, "invalid_schema", None, "", None, "bad output")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: BadOutput())

    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert result["status"] == "incomplete"
    assert result["reason"] == "qwen_invalid_output"
    assert result["reasons_seen"] == ["qwen_invalid_output"]
    assert tools._check_degradation_count == 1


def test_pending_memory_conflict_job_reports_skipped(tmp_path: Path) -> None:
    """A pending (workspace-activation) memory skips the job instead of incomplete."""
    tools = tv.make_tools(tmp_path)
    written = tools.memory_write(content="database is mysql", subject="a", tags=[], status="pending")
    assert written["ok"] is True
    memory_id = written["data"]["id"]

    result = tools._process_semantic_conflict_job(
        memory_id, {"memory_id": memory_id, "version": 1, "content_hash": "unused"},
    )

    assert result == {
        "status": "skipped", "reason": "pending_workspace_activation", "notices_created": 0,
    }


def _slot_member(memory_id: int, value: str) -> dict:
    quote = f"连接池上限为 {value}。"
    return ConflictMember(
        memory_id=memory_id, version=1, attribute_raw="连接池上限", value_raw=value,
        normalized_attribute="连接池上限", normalized_value=value, evidence_quote=quote,
        evidence_span=(0, len(quote)), content_hash=(str(memory_id) * 64)[:64],
        direction="a_to_b", prompt_version="p1", detector_version="d1",
    ).to_dict()


@pytest.mark.parametrize("stored_entity,stored_scope", [
    ("MyProject", "Production"),   # legacy raw (pre-canon) storage form
    ("myproject", "production"),   # canonical storage form
])
def test_applying_suppression_matches_raw_and_canon_slot_forms(
    tmp_path: Path, monkeypatch, stored_entity: str, stored_scope: str,
) -> None:
    """Suppression hits when either the canon or the raw slot form matches the
    stored applying group, regardless of metadata casing."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "MyProject", "scope": "Production"}
    a, b = _write_pair(tools, meta, left="连接池上限为 10。", right="连接池上限为 20。")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", tv._strict_pair_backend)

    recorded = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": stored_entity, "attribute": "连接池上限", "scope": stored_scope},
        "members": [_slot_member(a["id"], "10"), _slot_member(b["id"], "20")],
        "value_groups": [
            ConflictValueGroup("10", "10", (f"{a['id']}@1",)).to_dict(),
            ConflictValueGroup("20", "20", (f"{b['id']}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scan",
        "reason": "pool size conflict", "status": "open",
    })
    conflict_id = recorded["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "20",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"},
                       {"memory_id": b["id"], "action": "use_as_resolution"}],
        "resolution_memory_id": b["id"], "authorized": True,
    })
    # Storage canonises slot_key at record time; a group stored BEFORE that
    # canonicalisation existed may still carry the raw form — inject it
    # directly to simulate such a legacy row.
    import json as _json
    legacy_slot = {"entity": stored_entity, "attribute": "连接池上限", "scope": stored_scope}
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE conflicts SET slot_key=? WHERE id=?",
            (_json.dumps(legacy_slot, ensure_ascii=False, sort_keys=True), conflict_id),
        )

    tools.db.edit_memory_intent(a["id"], new_content="连接池上限为 30。", reason="apply")
    updated = tools.db.get_memory(a["id"])
    monkeypatch.setattr(
        tools.db, "evidence_knn", lambda *a_, **k: [_hit(b["id"], "连接池上限为 20。")],
    )
    snapshot = {
        "memory_id": a["id"], "version": updated["version"],
        "content_hash": evidence_content_hash(updated["content"]),
        "trusted_applying_context": {
            "conflict_id": conflict_id, "revision": 2, "memory_id": a["id"],
            "action": "update_current_claim", "chosen_value": "20",
        },
    }

    result = tools._process_semantic_conflict_job(a["id"], snapshot)

    # The pair was examined and slot-suppressed (not an error, no new notice).
    assert result["outcome"] == "checked_no_notice"
    fresh = [n for n in tools.db.list_semantic_notices(status="open")
             if {n.get("memory_id"), n.get("peer_id")} == {a["id"], b["id"]}]
    assert fresh == []


def test_scan_slot_key_uses_canonical_entity_scope(tmp_path: Path, monkeypatch) -> None:
    """Scan-path slot keys canonicalise entity/scope (B-C4 comparison side)."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "SVC", "scope": "Production"}
    tools.memory_write(content="生产数据库使用 mysql 方案", subject="a", tags=[], metadata=meta)
    tools.memory_write(content="生产数据库使用 sqlite 方案", subject="b", tags=[], metadata=meta)
    assert tools.wait_evidence_worker_drained(timeout=2)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", tv._grounded_db_backend)

    result = tools.memory_repair("scan_candidates", {"batch": 50, "k": 10})

    assert result["ok"] is True
    groups = result["data"].get("slot_groups") or []
    assert groups
    # The attribute's normalised form is owned by the extraction gate; the
    # canonicalisation under test here covers entity/scope.
    assert groups[0]["slot_key"]["entity"] == "svc"
    assert groups[0]["slot_key"]["scope"] == "production"


# ── from test_evidence_conflict.py ──
# helper _tools renamed: _evidence_tools (collision with test_product_conflict_groups.py)

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.semantic_conflict import (
    AttributeValueExtraction,
    coexistence_veto,
    decide_evidence,
    evaluate_pair_extractions,
    extraction_from_text,
    model_signal_from_text,
    notice_dedupe_key,
    value_is_grounded,
)
from memory_arbiter.tools import MemoryTools


def _evidence_tools(tmp_path: Path, **overrides) -> MemoryTools:
    values = {
        "db_path": tmp_path / "memory.db",
        "backup_jsonl": tmp_path / "backup.jsonl",
    }
    values.update(overrides)
    return MemoryTools(Settings(**values))


def test_deterministic_routes_are_explainable() -> None:
    numeric = decide_evidence("PostgreSQL port is 5432", "PostgreSQL port is 3306")
    assert numeric.action == "check"
    assert numeric.reason == "numeric_value_candidate"
    assert decide_evidence("pgsql", "PostgreSQL").action == "ignore"
    assert decide_evidence("database connection pool", "database connection policy").action == "check"


def test_qwen_protocol_is_strict_bounded_four_field_extraction() -> None:
    raw = '{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型","value_b":"SQLite"}'
    accepted = model_signal_from_text(raw)
    assert accepted.candidate is True
    extraction, error = extraction_from_text(raw)
    assert error is None and extraction is not None
    for invalid in (
        '{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型"}',
        '{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型","value_b":"SQLite","conflict":true}',
        '{"attribute_a":"数据库选型","value_a":3,"attribute_b":"数据库选型","value_b":"SQLite"}',
    ):
        parsed, parse_error = extraction_from_text(invalid)
        assert parsed is None and parse_error


def test_notice_dedupe_is_symmetric_and_version_pinned() -> None:
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_evidence") == notice_dedupe_key(
        2, 1, 4, 3, "semantic_evidence"
    )
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_evidence") != notice_dedupe_key(
        1, 2, 4, 4, "semantic_evidence"
    )


def test_check_degrades_to_no_notice_without_qwen(tmp_path: Path) -> None:
    tools = _evidence_tools(tmp_path)
    assert tools._ensure_semantic_backend() is None
    decision = decide_evidence("database connection pool", "database connection policy")
    assert decision.action == "check"


def test_notice_freshness_uses_only_memory_versions(tmp_path: Path) -> None:
    tools = _evidence_tools(tmp_path)
    left = tools.memory_write(content="left", subject="s", workspace="w")["data"]["id"]
    right = tools.memory_write(content="right", subject="s", workspace="w")["data"]["id"]
    created = tools.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal",
        notice_type="semantic_evidence", title="candidate", message="check",
        payload={}, left_version=1, right_version=1,
        dedupe_key=notice_dedupe_key(left, right, 1, 1, "semantic_evidence"),
    )
    notice = tools.db.read_semantic_notice(created["notice_id"])
    assert notice["freshness"]["fresh"] is True
    tools.memory_edit(left, new_content="changed", reason="new")
    notice = tools.db.read_semantic_notice(created["notice_id"])
    assert notice["freshness"]["fresh"] is False


def test_bidirectional_mapping_grounding_and_notice_gate() -> None:
    forward = AttributeValueExtraction("数据库引擎", "MySQL", "数据库选型", "SQLite")
    reverse = AttributeValueExtraction("数据库选型", "SQLite", "数据库引擎", "MySQL")
    result = evaluate_pair_extractions(
        forward, reverse,
        {"quote": "生产数据库使用 MySQL。"},
        {"quote": "生产数据库使用 SQLite。"},
        require_bidirectional=True,
    )
    assert result.state == "notice_ready"
    wrong_side = AttributeValueExtraction("数据库选型", "MySQL", "数据库引擎", "SQLite")
    rejected = evaluate_pair_extractions(
        forward, wrong_side,
        {"quote": "生产数据库使用 MySQL。"}, {"quote": "生产数据库使用 SQLite。"},
        require_bidirectional=True,
    )
    assert rejected.state == "review_candidate"
    assert rejected.reason == "bidirectional_mapping_mismatch"


def test_grounding_is_mechanical_and_coexistence_reasons_are_stable() -> None:
    assert value_is_grounded("5s", "接口超时为 5 秒。")
    assert value_is_grounded("PostgreSQL", "数据库采用 pgsql。")
    assert not value_is_grounded("关系数据库", "数据库采用 PostgreSQL。")
    assert coexistence_veto(
        {"quote": "测试环境数据库使用 MySQL"},
        {"quote": "生产环境数据库使用 SQLite"},
    ) == "coexist_environment_mismatch"
    assert coexistence_veto(
        {"quote": "v1 API timeout 5s"}, {"quote": "v2 API timeout 10s"},
    ) == "coexist_version_mismatch"


def test_single_direction_scan_survives_but_notice_fails_closed() -> None:
    extraction = AttributeValueExtraction("端口", "5432", "端口", "3306")
    scan = evaluate_pair_extractions(
        extraction, None, {"quote": "端口 5432"}, {"quote": "端口 3306"},
        require_bidirectional=False,
    )
    notice = evaluate_pair_extractions(
        extraction, None, {"quote": "端口 5432"}, {"quote": "端口 3306"},
        require_bidirectional=True,
    )
    assert scan.state == "review_candidate" and scan.reason == "single_direction_only"
    assert notice.state == "review_candidate" and notice.reason == "bidirectional_extraction_required"


# ── 2026-08-21 review round: semantic-layer fixes ───────────────────────────

def test_unknown_sentinel_is_extraction_failure() -> None:
    """The protocol-legal '__unknown__' marker never becomes a usable field."""
    raw = '{"attribute_a":"数据库","value_a":"__unknown__","attribute_b":"数据库","value_b":"SQLite"}'
    extraction, error = extraction_from_text(raw)
    assert extraction is None
    assert error == "unknown_field"
    signal = model_signal_from_text(raw)
    # Distinguished from a protocol violation so diagnostics separate model
    # output from technical failure.
    assert signal.candidate_type == "unknown_field"
    assert signal.candidate is False


def test_top_level_array_output_is_rejected() -> None:
    raw = '[{"attribute_a":"db","value_a":"MySQL","attribute_b":"db","value_b":"SQLite"}]'
    extraction, error = extraction_from_text(raw)
    assert extraction is None
    assert error is not None and error.startswith("invalid_schema")


def test_bare_agent_marker_does_not_trigger_evolution_veto() -> None:
    # "由" as a passive/agent marker with no replacement wording must not veto.
    assert coexistence_veto(
        {"quote": "新网关由运维分配"}, {"quote": "端口是 8080"},
    ) is None
    # Real replacement wording still vetoes.
    assert coexistence_veto(
        {"quote": "旧网关已迁移到新集群"}, {"quote": "当前使用新集群"},
    ) == "coexist_explicit_evolution"


def test_unit_spelling_variants_normalize_equal_at_post_gate() -> None:
    # 8GB vs 8G is a restated duplicate, not a conflict, once units compact.
    result = evaluate_pair_extractions(
        AttributeValueExtraction("内存", "8GB", "内存", "8G"),
        AttributeValueExtraction("内存", "8G", "内存", "8GB"),
        {"quote": "内存 8GB"}, {"quote": "内存 8G"},
        require_bidirectional=True,
    )
    assert result.state == "review_candidate"
    assert result.reason == "not_same_attribute_different_value"


# ── 2026-08-21 review round 2: normalization edge cases ─────────────────────

def test_decimal_point_preserved_in_normalization() -> None:
    from memory_arbiter.semantic_conflict import normalize_value
    assert normalize_value("1.5s") != normalize_value("15s")
    # Integer/fractional pair stays distinct.
    assert normalize_value("1.5") != normalize_value("15")


def test_unknown_sentinel_rejection_is_case_insensitive() -> None:
    raw = '{"attribute_a":"db","value_a":"__UNKNOWN__","attribute_b":"db","value_b":"SQLite"}'
    extraction, error = extraction_from_text(raw)
    assert extraction is None
    assert error == "unknown_field"


def test_prose_prefixed_array_is_rejected() -> None:
    raw = 'Result: [ {"attribute_a":"db","value_a":"MySQL","attribute_b":"db","value_b":"SQLite"} ]'
    extraction, error = extraction_from_text(raw)
    assert extraction is None
    assert error is not None and error.startswith("invalid_schema")


def test_evolution_veto_covers_bian_wei_family() -> None:
    assert coexistence_veto(
        {"quote": "网关由 A 变为 B"}, {"quote": "当前网关是 B"},
    ) == "coexist_explicit_evolution"
    assert coexistence_veto(
        {"quote": "旧配置调整为新值"}, {"quote": "现在使用新值"},
    ) == "coexist_explicit_evolution"


# ── from test_conflict_upgrade.py ──
# helper _memory renamed: _upgrade_memory (collision with test_conflict_groups.py)


import json
import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.db_generation import (
    CONFLICT_DETECTOR_VERSION,
    CURRENT_SCHEMA_GENERATION,
    detect_database_generation,
)
from memory_arbiter.evidence import evidence_content_hash, local_text_units
from memory_arbiter.models import MemoryRecord
from memory_arbiter.tools import MemoryTools
from memory_arbiter.vnext_migration import (
    _configured_embedding_space_id,
    build_conflict_only,
    _copy_preserved_tables,
    _fingerprint,
    _mark_conflict_rebuild_ready,
    inspect,
)


def _settings(path: Path, tmp_path: Path) -> Settings:
    return Settings(db_path=path, backup_jsonl=tmp_path / "backup.jsonl")


def _same_space_settings(path: Path, tmp_path: Path) -> Settings:
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"same-space-model")
    return Settings(
        db_path=path,
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
    )


def _mark_current_space(db: MemoryDB, settings: Settings) -> str:
    pytest.importorskip("sqlite_vec")
    # Lazy vec-table semantics (0.15.0): the vec0 tables exist only after the
    # first successful embedder build. Hand-injecting vectors below mirrors
    # that build: create the tables at dim 2, then record the same-dim space.
    assert db.ensure_vec_tables(2) == []
    space_id = _configured_embedding_space_id(settings, 2)
    assert space_id is not None
    with db.connection() as conn:
        memories = [dict(row) for row in conn.execute(
            "SELECT id,version,content,subject,status FROM memories WHERE status!='deleted'"
        )]
    for memory in memories:
        units = local_text_units(
            str(memory.get("subject") or ""), str(memory.get("content") or ""),
        )
        embeddings = [[0.0, 1.0] for _unit in units]
        published = db.evidence.publish(
            int(memory["id"]), int(memory.get("version") or 1),
            evidence_content_hash(str(memory.get("content") or "")), units, embeddings,
        )
        assert published.get("published") is True
    with db.connection() as conn:
        canonicals = [
            str(row["name"])
            for row in conn.execute("SELECT name FROM workspace_canonicals ORDER BY id")
            if str(row["name"] or "").casefold() != "default"
        ]
    for canonical in canonicals:
        assert db.workspaces.publish_workspace_canonical_vector(
            canonical, [0.0, 1.0],
        ) == []
    with db.write_transaction() as conn:
        conn.execute("DELETE FROM _vec_index_meta")
        conn.executemany(
            "INSERT INTO _vec_index_meta(key,value) VALUES(?,?)",
            (("state", "ready"), ("active_space_id", space_id)),
        )
    return space_id


def _upgrade_memory(defaults: dict[str, object]) -> MemoryRecord:
    return MemoryRecord.from_input(
        {"content": "数据库使用 SQLite。", "subject": "数据库选型", "workspace": "project"},
        defaults,
    )


def test_destructive_copy_preserves_core_data_but_not_conflict_history(tmp_path: Path) -> None:
    source_path = tmp_path / "source.sqlite3"
    target_path = tmp_path / "target.sqlite3"
    source = MemoryDB(_settings(source_path, tmp_path))
    memory_id, _ = source.insert_memory(_upgrade_memory(_settings(source_path, tmp_path).defaults()), "project")
    assert memory_id is not None
    with source.write_transaction() as conn:
        conn.execute(
            "INSERT INTO memory_history(memory_id,content_snapshot,subject_snapshot,tags_snapshot,version,changed_at,reason) VALUES(?,?,?,?,?,?,?)",
            (memory_id, "old", "数据库选型", "[]", 1, "2026-08-20T00:00:00Z", "test"),
        )
        conn.execute(
            "INSERT INTO workspace_aliases(alias_workspace,canonical,status,updated_at) VALUES(?,?,?,?)",
            ("proj", "project", "confirmed", "2026-08-20T00:00:00Z"),
        )
        # Removed tables may exist in a previous-generation source even though
        # current runtime schema intentionally does not create them.
        conn.execute("CREATE TABLE semantic_notices(id INTEGER PRIMARY KEY, message TEXT)")
        conn.execute("INSERT INTO semantic_notices VALUES(1,'legacy notice')")
        conn.execute("CREATE TABLE conflict_judgments(id INTEGER PRIMARY KEY, reason TEXT)")
        conn.execute("INSERT INTO conflict_judgments VALUES(1,'legacy judgment')")
        conn.execute(
            "INSERT INTO backup_replay_log("
            "replay_key,memory_id,payload_hash,replayed_at,postprocess_status,"
            "postprocess_stages,postprocess_error_code) VALUES(?,?,?,?,?,?,?)",
            (
                "receipt-1", memory_id, "payload-hash", "2026-08-20T00:00:00Z",
                "failed", '{"evidence":"complete","semantic":"failed"}', "model_timeout",
            ),
        )

    target = MemoryDB(_settings(target_path, tmp_path))
    _copy_preserved_tables(source_path, target)

    with target.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM workspace_aliases").fetchone()[0] == 1
        receipt = conn.execute(
            "SELECT replay_key,memory_id,payload_hash,replayed_at,postprocess_status,"
            "postprocess_stages,postprocess_error_code FROM backup_replay_log"
        ).fetchone()
        assert tuple(receipt) == (
            "receipt-1", memory_id, "payload-hash", "2026-08-20T00:00:00Z",
            "failed", '{"evidence":"complete","semantic":"failed"}', "model_timeout",
        )
        assert conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 0
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_notices'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflict_judgments'"
        ).fetchone() is None
    assert _fingerprint(source_path) == _fingerprint(target_path)


def test_current_startup_does_not_run_schema_migrations(tmp_path: Path) -> None:
    path = tmp_path / "legacy-alias.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_new")
        conn.execute(
            "CREATE TABLE workspace_aliases(alias_workspace TEXT NOT NULL UNIQUE, canonical TEXT NOT NULL, relation TEXT NOT NULL, status TEXT NOT NULL, source TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO workspace_aliases VALUES('raw','candidate-a','alias','rejected','user','2026-08-20T00:00:00Z')"
        )
        conn.execute("DROP TABLE workspace_aliases_new")
    reopened = MemoryDB(_settings(path, tmp_path))
    assert reopened.db_available is True
    with reopened.connection() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
    assert columns == [
        "alias_workspace", "canonical", "relation", "status", "source", "updated_at",
    ]


def test_startup_defers_partial_table_health_to_doctor_or_runtime(tmp_path: Path) -> None:
    path = tmp_path / "partial.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_good")
        conn.execute(
            "CREATE TABLE workspace_aliases(alias_workspace TEXT,canonical TEXT,updated_at TEXT)"
        )
        conn.execute("INSERT INTO workspace_aliases VALUES('raw','target','2026-01-01')")
        conn.execute("DROP TABLE workspace_aliases_good")
    reopened = MemoryDB(_settings(path, tmp_path))
    assert reopened.db_available is True
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspace_aliases").fetchone()[0] == 1
        columns = [row[1] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
    assert columns == ["alias_workspace", "canonical", "updated_at"]


def test_startup_defers_invalid_compact_rows_to_doctor_or_runtime(tmp_path: Path) -> None:
    path = tmp_path / "invalid.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_good")
        conn.execute(
            "CREATE TABLE workspace_aliases("
            "alias_workspace TEXT,canonical TEXT,status TEXT,updated_at TEXT,"
            "PRIMARY KEY(alias_workspace,canonical))"
        )
        conn.execute("INSERT INTO workspace_aliases VALUES('raw','target','bogus','2026-01-01')")
        conn.execute("DROP TABLE workspace_aliases_good")
    reopened = MemoryDB(_settings(path, tmp_path))
    assert reopened.db_available is True
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT status FROM workspace_aliases").fetchone()
    assert row[0] == "bogus"


def test_startup_does_not_repair_interrupted_table_migration(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    assert db.record_workspace_decision("raw", "target")[0]
    with db.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_legacy")
    reopened = MemoryDB(_settings(path, tmp_path))
    assert reopened.db_available is True
    assert reopened.resolve_workspace_canonical("raw", None, register_new=False)["canonical"] == "raw"
    with reopened.connection() as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='workspace_aliases_legacy'"
        ).fetchone() is not None


def test_full_build_copy_failure_never_leaves_current_target(
    tmp_path: Path, monkeypatch,
) -> None:
    from memory_arbiter import vnext_migration

    source_path = tmp_path / "source-current.sqlite3"
    target_path = tmp_path / "target-partial.sqlite3"
    MemoryDB(_settings(source_path, tmp_path))

    def fail_copy(*args, **kwargs):
        raise sqlite3.IntegrityError("injected copy failure")

    monkeypatch.setattr(vnext_migration, "_copy_preserved_tables", fail_copy)
    with pytest.raises(sqlite3.IntegrityError):
        vnext_migration.build(
            source_path, target_path, _settings(source_path, tmp_path), progress=False,
        )
    assert detect_database_generation(target_path) != "current"
    with sqlite3.connect(target_path) as conn:
        state = dict(conn.execute("SELECT key,value FROM migration_state"))
    assert state["phase"] == "failed"
    assert state["schema_generation"].endswith(":building")


def test_full_build_checkpoint_failure_marks_target_failed(
    tmp_path: Path, monkeypatch,
) -> None:
    from memory_arbiter import vnext_migration

    source_path = tmp_path / "source-empty.sqlite3"
    target_path = tmp_path / "target-checkpoint.sqlite3"
    MemoryDB(_settings(source_path, tmp_path))
    monkeypatch.setattr(vnext_migration, "_checkpoint", lambda _path: False)
    result = vnext_migration.build(
        source_path, target_path, _settings(source_path, tmp_path), progress=False,
    )
    assert result["ok"] is False
    assert result["switch_ready"] is False
    assert detect_database_generation(target_path) != "current"
    with sqlite3.connect(target_path) as conn:
        state = dict(conn.execute("SELECT key,value FROM migration_state"))
    assert state["phase"] == "failed"


def test_scan_gate_requires_matching_epoch_detector_boundary_and_live_set(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    settings = _settings(path, tmp_path)
    db = MemoryDB(settings)
    memory_id, _ = db.insert_memory(_upgrade_memory(settings.defaults()), "project")
    assert memory_id is not None
    state = _mark_conflict_rebuild_ready(db)
    visible = db.conflict_scan_state()
    assert visible["required"] is True
    assert visible["detector_version"] == CONFLICT_DETECTOR_VERSION

    # Completion is not a caller assertion: without persisted pages it fails,
    # even when epoch/detector/boundary are otherwise correct.
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is False
    assert db.complete_conflict_scan(
        epoch="wrong", detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is False
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version="old-detector",
        boundary=visible["boundary"],
    ) is False

    with db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status='superseded' WHERE id=?", (memory_id,))
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is False

    with db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status='active' WHERE id=?", (memory_id,))
    assert db.record_conflict_scan_page(
        epoch=state["conflict_scan_epoch"],
        detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
        after_memory_id=0,
        next_anchor_memory_id=None,
        anchors_scanned=1,
        workspace=None,
    ) is True
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is True
    assert db.conflict_scan_state()["required"] is False
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is False


def test_scan_candidates_pages_persist_progress_and_clear_gate(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    settings = _settings(path, tmp_path)
    db = MemoryDB(settings)
    for index in range(3):
        memory_id, _ = db.insert_memory(
            MemoryRecord.from_input(
                {"content": f"配置值为 {index}。", "subject": "配置", "workspace": "project"},
                settings.defaults(),
            ),
            "project",
        )
        assert memory_id is not None
    _mark_conflict_rebuild_ready(db)
    tools = MemoryTools(settings=settings, db=db)
    # The scan path is exercised without requiring sqlite-vec in this gate test.
    tools.db.state.sqlite_vec_available = True
    tools.db.scan_rule_candidates = lambda **kwargs: {
        "anchors_scanned": min(2, 3 - int(kwargs["after_memory_id"])),
        "next_anchor_memory_id": 2 if int(kwargs["after_memory_id"]) == 0 else None,
        "candidates": [],
        "counts": {},
    }

    first = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 2})
    assert first["data"]["conflict_scan_progress"]["complete"] is False
    assert db.conflict_scan_state()["required"] is True
    # Skipping the persisted cursor is rejected and cannot clear the gate.
    skipped = tools.memory_repair("scan_candidates", {"anchor_memory_id": 1, "batch": 2})
    assert skipped["data"]["conflict_scan_progress_rejected"] is True
    assert db.conflict_scan_state()["required"] is True

    final = tools.memory_repair("scan_candidates", {"anchor_memory_id": 2, "batch": 2})
    assert final["data"]["conflict_scan_completed"] is True
    assert db.conflict_scan_state()["required"] is False


def test_previous_generation_is_refused_until_destructive_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "previous.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' WHERE key='schema_generation'"
        )
    assert CURRENT_SCHEMA_GENERATION != "local_text_evidence_v1"
    assert detect_database_generation(path) == "legacy"


def test_previous_conflict_generation_is_refused_until_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "conflict-v2.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='conflict_groups_v2' "
            "WHERE key='schema_generation'"
        )
    assert detect_database_generation(path) == "legacy"


def test_conflict_only_upgrade_compacts_workspace_state_and_drops_events(tmp_path: Path) -> None:
    source_path = tmp_path / "conflict-v2.sqlite3"
    target_path = tmp_path / "workspace-v1.sqlite3"
    migration_settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(migration_settings)
    assert source.record_workspace_decision("raw", "target")[0]
    with source.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_compact")
        conn.execute(
            "CREATE TABLE workspace_aliases("
            "alias_workspace TEXT NOT NULL,canonical TEXT NOT NULL,relation TEXT NOT NULL,"
            "status TEXT NOT NULL,source TEXT NOT NULL,updated_at TEXT NOT NULL,"
            "PRIMARY KEY(alias_workspace,canonical))"
        )
        conn.execute(
            "INSERT INTO workspace_aliases SELECT alias_workspace,canonical,'alias',"
            "status,'user',updated_at FROM workspace_aliases_compact"
        )
        conn.execute("DROP TABLE workspace_aliases_compact")
        conn.execute(
            "CREATE TABLE workspace_alias_events("
            "id INTEGER PRIMARY KEY,alias_workspace TEXT,action TEXT)"
        )
        conn.execute("INSERT INTO workspace_alias_events VALUES(1,'raw','accept')")
        conn.execute(
            "UPDATE migration_state SET value='conflict_groups_v2' "
            "WHERE key='schema_generation'"
        )
    _mark_current_space(source, migration_settings)
    result = build_conflict_only(source_path, target_path, migration_settings)
    assert result["ok"] is True, result
    with sqlite3.connect(target_path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
        events = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='workspace_alias_events'"
        ).fetchone()
    assert columns == ["alias_workspace", "canonical", "status", "updated_at"]
    assert events is None
    assert detect_database_generation(target_path) == "current"


def test_previous_evidence_generation_rebuilds_only_conflict_domain(tmp_path: Path) -> None:
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    memory_id, _ = source.insert_memory(_upgrade_memory(settings.defaults()), "project")
    assert memory_id is not None
    _mark_current_space(source, settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )
    before = _fingerprint(source_path)

    result = build_conflict_only(source_path, target_path, settings)

    assert result["ok"] is True
    assert result["indexed"] == 0
    assert result["evidence_reused"] is True
    assert result["vector_effect"] == "preserve"
    assert result["vec_index_state"]["state"] == "ready"
    assert result["source_fingerprint"] == result["target_fingerprint"] == before
    assert detect_database_generation(target_path) == "current"
    with sqlite3.connect(target_path) as conn:
        state = dict(conn.execute("SELECT key,value FROM migration_state"))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(conflicts)")}
        assert conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 0
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_notices'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflict_judgments'"
        ).fetchone() is None
    assert {"revision", "member_versions", "value_groups", "notice_type"} <= columns
    assert state["schema_generation"] == CURRENT_SCHEMA_GENERATION
    assert state["migration_completed_at"]
    assert "phase" not in state
    assert state["source_path"] == str(source_path)
    assert state["conflict_scan_required"] == "true"


def test_previous_generation_with_old_embedding_space_preserves_then_marks_mismatch(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    source_settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(source_settings)
    memory_id, _ = source.insert_memory(_upgrade_memory(source_settings.defaults()), "project")
    assert memory_id is not None
    _mark_current_space(source, source_settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )
        conn.execute("DELETE FROM _vec_index_meta")
        conn.executemany(
            "INSERT INTO _vec_index_meta(key,value) VALUES(?,?)",
            (("state", "ready"), ("active_space_id", "old-pipeline-space")),
        )
        source_vectors = [
            tuple(row) for row in conn.execute(
                "SELECT id,parent_status,hex(embedding) FROM memory_evidence_vec ORDER BY id"
            ).fetchall()
        ]
    settings = source_settings

    plan = inspect(source_path, target_path, settings)
    migrated = build_conflict_only(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"
    assert plan["vector_compatibility"] == "mismatch"
    assert migrated["ok"] is True
    assert migrated["vec_index_state"]["state"] == "mismatch"
    with sqlite3.connect(target_path) as conn:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        state = dict(conn.execute("SELECT key,value FROM _vec_index_meta"))
        target_vectors = conn.execute(
            "SELECT id,parent_status,hex(embedding) FROM memory_evidence_vec ORDER BY id"
        ).fetchall()
    assert state["state"] == "mismatch"
    assert state["active_space_id"] == "old-pipeline-space"
    assert state["target_space_id"] == plan["configured_space_id"]
    assert target_vectors == source_vectors


def test_previous_generation_without_managed_space_still_preserves_vectors(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _settings(source_path, tmp_path)
    source = MemoryDB(settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    plan = inspect(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"
    assert plan["vector_compatibility"] == "unmanaged"


def test_previous_generation_with_incomplete_same_space_index_defers_to_doctor(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    memory_id, _ = source.insert_memory(_upgrade_memory(settings.defaults()), "project")
    assert memory_id is not None
    _mark_current_space(source, settings)
    with source.write_transaction() as conn:
        evidence_id = conn.execute(
            "SELECT id FROM memory_evidence WHERE memory_id=? ORDER BY id LIMIT 1",
            (memory_id,),
        ).fetchone()[0]
        conn.execute("DELETE FROM memory_evidence_vec WHERE id=?", (evidence_id,))
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    plan = inspect(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"


def test_previous_generation_with_stale_same_space_evidence_defers_to_doctor(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    memory_id, _ = source.insert_memory(_upgrade_memory(settings.defaults()), "project")
    assert memory_id is not None
    _mark_current_space(source, settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET content='changed without republish',version=version+1 WHERE id=?",
            (memory_id,),
        )
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    plan = inspect(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"


def test_previous_generation_with_wrong_vector_dimension_defers_to_deep_doctor(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    source_settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(source_settings)
    _mark_current_space(source, source_settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )
    mismatched_settings = Settings(
        db_path=source_path,
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=source_settings.embedding_model_path,
    )
    # Forged identity: the library's vectors are 2-dim (published above), the
    # recorded space id is computed as if the active dim were 3.
    forged_space = _configured_embedding_space_id(mismatched_settings, 3)
    assert forged_space is not None
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE _vec_index_meta SET value=? WHERE key='active_space_id'",
            (forged_space,),
        )

    plan = inspect(source_path, target_path, mismatched_settings)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"


def test_generation_switch_marker_is_atomic_with_scan_metadata(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    _mark_conflict_rebuild_ready(db)
    with sqlite3.connect(path) as conn:
        state = dict(conn.execute("SELECT key,value FROM migration_state"))
    assert state["schema_generation"] == CURRENT_SCHEMA_GENERATION
    assert state["migration_completed_at"]
    assert "phase" not in state
    assert state["conflict_scan_required"] == "true"
    assert state["conflict_scan_detector_version"] == CONFLICT_DETECTOR_VERSION
    boundary = json.loads(state["conflict_scan_boundary"])
    assert boundary["active_count"] == 0
    assert boundary["max_memory_id"] == 0
    assert len(boundary["active_set_digest"]) == 64


def test_status_and_doctor_expose_pending_rebuild_scan(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    settings = _settings(path, tmp_path)
    db = MemoryDB(settings)
    _mark_conflict_rebuild_ready(db)
    tools = MemoryTools(settings=settings, db=db)
    status = tools.memory_status()["data"]
    assert status["conflict_scan_required"] is True
    assert status["conflict_scan"]["detector_version"] == CONFLICT_DETECTOR_VERSION
    findings = tools.memory_doctor_overview()["data"]["findings"]
    scan = next(item for item in findings if item["check_id"] == "conflicts.scan_required")
    assert scan["status"] == "warn"


def test_write_between_upgrade_and_scan_rearms_instead_of_wedging(tmp_path: Path) -> None:
    """Spec §15.7/§15.8.24 recovery: a live-boundary drift re-arms the epoch so a
    fresh full scan can still clear conflict_scan_required, instead of wedging."""
    path = tmp_path / "db.sqlite3"
    settings = _settings(path, tmp_path)
    db = MemoryDB(settings)
    first_id, _ = db.insert_memory(_upgrade_memory(settings.defaults()), "project")
    assert first_id is not None
    original = _mark_conflict_rebuild_ready(db)
    original_boundary = db.conflict_scan_state()["boundary"]

    # A memory write after upgrade drifts the live active-set boundary: pages
    # recorded against the original boundary are now rejected.
    second_id, _ = db.insert_memory(
        MemoryRecord.from_input(
            {"content": "另一个配置。", "subject": "配置", "workspace": "project"},
            settings.defaults(),
        ),
        "project",
    )
    assert second_id is not None
    assert db.record_conflict_scan_page(
        epoch=original["conflict_scan_epoch"],
        detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=original_boundary,
        after_memory_id=0, next_anchor_memory_id=None, anchors_scanned=2,
        workspace=None,
    ) is False
    assert db.conflict_scan_state()["required"] is True

    # Re-arm against the current live set, then a full scan of the new boundary
    # clears the flag.
    assert db.rearm_conflict_scan_if_drifted() is True
    rearmed = db.conflict_scan_state()
    assert rearmed["required"] is True
    assert rearmed["epoch"] != original["conflict_scan_epoch"]
    assert rearmed["progress"] is None
    assert db.record_conflict_scan_page(
        epoch=rearmed["epoch"],
        detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=rearmed["boundary"],
        after_memory_id=0, next_anchor_memory_id=None, anchors_scanned=2,
        workspace=None,
    ) is True
    assert db.complete_conflict_scan(
        epoch=rearmed["epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=rearmed["boundary"],
    ) is True
    assert db.conflict_scan_state()["required"] is False
    # No re-arm happens when the boundary is already consistent.
    assert db.rearm_conflict_scan_if_drifted() is False


def test_conflict_only_validation_failure_marks_phase_failed(tmp_path: Path, monkeypatch) -> None:
    """A conflict-only rebuild whose post-transaction validation fails must not
    leave a phase=ready/current target that detect_database_generation trusts."""
    import memory_arbiter.vnext_migration as vm

    source = tmp_path / "source.sqlite3"
    settings = _same_space_settings(source, tmp_path)
    db = MemoryDB(settings)
    db.insert_memory(_upgrade_memory(settings.defaults()), "project")
    _mark_current_space(db, settings)
    # Move it to the previous generation so build_conflict_only is the path.
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO migration_state(key,value,updated_at) VALUES('schema_generation',?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
            ("local_text_evidence_v1",),
        )

    # Force validation to fail after the rebuild transaction commits.
    monkeypatch.setattr(vm, "_checkpoint", lambda *_a, **_k: False)
    target = tmp_path / "target.sqlite3"
    result = build_conflict_only(source, target, settings)
    assert result["ok"] is False
    # The stranded artifact is marked failed, so it is not classified current.
    assert detect_database_generation(target) != "current"
    with sqlite3.connect(target) as conn:
        conn.row_factory = sqlite3.Row
        phase = conn.execute("SELECT value FROM migration_state WHERE key='phase'").fetchone()
    assert phase is not None and phase["value"] == "failed"


def test_conflict_only_does_not_turn_post_publish_space_drift_into_schema_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    import memory_arbiter.vnext_migration as vm

    source_path = tmp_path / "source.sqlite3"
    target_path = tmp_path / "target.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    _mark_current_space(source, settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    original_checkpoint = vm._checkpoint

    def corrupt_after_checkpoint(path: Path) -> bool:
        checkpointed = original_checkpoint(path)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "UPDATE _vec_index_meta SET value='foreign-space' "
                "WHERE key='active_space_id'"
            )
        return checkpointed

    monkeypatch.setattr(vm, "_checkpoint", corrupt_after_checkpoint)

    result = build_conflict_only(source_path, target_path, settings)

    assert result["ok"] is True
    assert detect_database_generation(target_path) == "current"
    with sqlite3.connect(target_path) as conn:
        phase = conn.execute(
            "SELECT value FROM migration_state WHERE key='phase'"
        ).fetchone()
    assert phase is None


# ── from test_semantic_process.py ──


import os
import threading
import time
from pathlib import Path

from memory_arbiter.semantic_conflict import (
    IsolatedGGUFSemanticBackend,
    ModelSignal,
    WorkspaceCandidateSignal,
)


def _responsive_child(conn, config):
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "shutdown":
                return
            conn.send({
                "ok": True,
                "result": ModelSignal(True, "replacement", 0.9, "{}", {"pid": os.getpid()}),
            })
    except (EOFError, OSError):
        return


def _blocking_child(conn, config):
    try:
        request = conn.recv()
        if request.get("command") == "load":
            conn.send({"ok": True, "result": {"loaded": True}})
            conn.recv()
        time.sleep(10)
    except (EOFError, OSError):
        return


def _serial_probe_child(conn, config):
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "load":
                conn.send({"ok": True, "result": {"loaded": True}})
                continue
            time.sleep(0.08)
            conn.send({"ok": True, "result": ModelSignal(True, "replacement", 0.9, "{}", {"pid": os.getpid()})})
    except (EOFError, OSError):
        return


def _fair_probe_child(conn, config):
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "load":
                conn.send({"ok": True, "result": {"loaded": True}})
                continue
            time.sleep(0.025)
            if request.get("command") == "classify_pair":
                result = ModelSignal(True, "replacement", 0.9, "{}", {"kind": "notice"})
            else:
                result = WorkspaceCandidateSignal("canonical", "alias", 0.9, "same")
            conn.send({"ok": True, "result": result})
    except (EOFError, OSError):
        return


def _workspace_probe_child(conn, config):
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "load":
                conn.send({"ok": True, "result": {"loaded": True}})
                continue
            if request.get("command") == "classify_pair":
                time.sleep(10)
            conn.send({
                "ok": True,
                "result": WorkspaceCandidateSignal("canonical", "alias", 0.9, "same"),
            })
    except (EOFError, OSError):
        return


def _first_generation_blocks_child(conn, config):
    marker = Path(config["model_path"]).with_suffix(".started")
    first_generation = not marker.exists()
    marker.touch(exist_ok=True)
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "load":
                conn.send({"ok": True, "result": {"loaded": True}})
                continue
            if first_generation:
                time.sleep(10)
            conn.send({"ok": True, "result": ModelSignal(True, "replacement", 0.9, "{}", {"pid": os.getpid()})})
    except (EOFError, OSError):
        return


def _backend(tmp_path: Path, target, timeout=500) -> IsolatedGGUFSemanticBackend:
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    return IsolatedGGUFSemanticBackend(
        model, hard_timeout_ms=timeout, process_target=target,
    )


def test_process_backend_reuses_one_resident_child(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _responsive_child)
    first = backend.classify_pair({}, {})
    first_pid = first.parsed["pid"]
    second = backend.classify_pair({}, {})
    assert second.parsed["pid"] == first_pid
    assert backend.status()["max_concurrency"] == 1
    assert backend.unload(timeout=1)["ok"] is True


def test_process_backend_hard_timeout_terminates_child(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _blocking_child, timeout=100)
    result = backend.classify_pair({}, {})
    assert result.candidate is False
    assert "hard timeout" in (result.error or "")
    status = backend.status()
    assert status["model_state"] == "unloaded"
    assert status["timed_out_jobs"] == 1
    assert status["child_restarts"] == 1


def test_process_backend_concurrent_callers_are_serialized(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _serial_probe_child, timeout=1000)
    results = []
    started = time.monotonic()
    threads = [threading.Thread(target=lambda: results.append(backend.classify_pair({}, {}))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    elapsed = time.monotonic() - started
    assert len(results) == 2
    assert elapsed >= 0.14
    assert backend.status()["max_concurrency"] == 1
    backend.force_terminate()
    assert backend.status()["child_pid"] is None


def test_scheduler_discards_expired_workspace_and_preserves_notice_budget(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _fair_probe_child, timeout=1000)
    blocker = threading.Thread(target=lambda: backend.classify_pair({}, {}))
    blocker.start()
    deadline = time.monotonic() + 1
    while backend.status()["inflight"] != 1 and time.monotonic() < deadline:
        time.sleep(0.005)

    started = time.monotonic()
    expired = backend.suggest_workspace_candidate(
        "raw", {}, ["canonical"], deadline_monotonic=time.monotonic() + 0.005,
    )
    elapsed = time.monotonic() - started
    assert expired.candidate is None
    assert "deadline" in (expired.error or "")
    assert elapsed < 0.05

    notice = backend.classify_pair({}, {})
    assert notice.candidate is True
    blocker.join(1)
    assert not blocker.is_alive()
    backend.force_terminate()


def test_scheduler_is_fair_under_continuous_workspace_and_notice_load(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _fair_probe_child, timeout=1000)
    barrier = threading.Barrier(9)
    completed: list[str] = []
    lock = threading.Lock()

    def run(kind: str) -> None:
        barrier.wait()
        if kind == "workspace":
            result = backend.suggest_workspace_candidate(
                "raw", {}, ["canonical"], deadline_monotonic=time.monotonic() + 2,
            )
            ok = result.candidate == "canonical"
        else:
            ok = backend.classify_pair({}, {}).candidate
        if ok:
            with lock:
                completed.append(kind)

    threads = [threading.Thread(target=run, args=(kind,)) for kind in (["workspace", "notice"] * 4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(3)
    assert completed.count("workspace") == 4
    assert completed.count("notice") == 4
    assert max(
        max(i for i, kind in enumerate(completed) if kind == "workspace"),
        max(i for i, kind in enumerate(completed) if kind == "notice"),
    ) < 8
    backend.force_terminate()


def test_disable_closes_admission_before_unload_timeout(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _workspace_probe_child, timeout=5_000)
    started = threading.Event()

    def run_inference():
        started.set()
        backend.classify_pair({}, {})

    thread = threading.Thread(target=run_inference)
    thread.start()
    assert started.wait(1)
    deadline = time.monotonic() + 2
    while backend.status()["inflight"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)

    disabled = backend.unload(timeout=0.01, disable=True)
    assert disabled["timeout"] is True
    assert backend.status()["disabled"] is True

    suggest_started = time.monotonic()
    suggestion = backend.suggest_workspace_candidate("raw", {}, ["canonical"])
    assert time.monotonic() - suggest_started < 0.5
    assert suggestion.candidate is None
    assert suggestion.error == "semantic backend disabled"
    backend.force_terminate()
    thread.join(2)
    assert not thread.is_alive()


def test_process_backend_recovers_with_new_generation_after_timeout(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _first_generation_blocks_child, timeout=100)
    first = backend.classify_pair({}, {})
    assert first.candidate is False
    second = backend.classify_pair({}, {})
    assert second.candidate is True
    status = backend.status()
    assert status["generation"] == 2
    assert status["timed_out_jobs"] == 1
    backend.force_terminate()
