from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.tools import MemoryTools


def _db(tmp_path: Path) -> MemoryDB:
    return MemoryDB(Settings(db_path=tmp_path / "memory.db", backup_jsonl=tmp_path / "backup.jsonl", enable_sqlite_vec=False))


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
        enable_sqlite_vec=False, isolation="strict", workspace="alpha",
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
