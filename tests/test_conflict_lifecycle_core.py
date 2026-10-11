# ── from test_conflict_groups.py ──

from __future__ import annotations

import sqlite3

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.semantic_conflict import (
    decide_evidence, normalize_value, notice_dedupe_key,
)
from memory_arbiter.tools import MemoryTools
from memory_arbiter.db_generation import (
    CONFLICT_DETECTOR_VERSION,
    CURRENT_SCHEMA_GENERATION,
    detect_database_generation,
)
from memory_arbiter.evidence import evidence_content_hash
from memory_arbiter.rowseg import segment_rows
from memory_arbiter.vnext_migration import (
    _configured_embedding_space_id,
    build_conflict_only,
    _copy_preserved_tables,
    _fingerprint,
    _mark_conflict_rebuild_ready,
    inspect,
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


def test_long_phrase_use_as_resolution_completes_after_deadlock_fix(tmp_path: Path) -> None:
    """#970 regression: conflict #50's exact shape.

    A long English normalized_value chosen against Chinese member content
    used to fail apply-side grounding for use_as_resolution and dead-end the
    applying group. The action must complete mechanically now (D2) and the
    group must resolve (D1 keeps the paraphrase out at intake, so the stored
    group value here equals normalize_value(value_raw)).
    """
    db = _db(tmp_path)
    tools = MemoryTools(db.settings, db)
    left = _memory(db, "发版上传：twine upload dist/* 全部文件")
    right = _memory(db, "发版经验：上传要显式文件名，不用 dist/*")
    raw_left, raw_right = "upload dist/*", "upload explicit filenames"
    members = [_member(left, raw_left), _member(right, raw_right)]
    created = db.record_conflict_group(
        workspace_canonical="w",
        slot_key={"entity": "release", "attribute": "upload", "scope": "pypi"},
        members=members, value_groups=_groups(*members),
        detection_reason="different upload recipe", source="scan",
        detector_version="d1", prompt_version="p1", conflict_point="upload",
    )
    assert created["outcome"] == "inserted"
    chosen = normalize_value(raw_right)
    judged = db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value=chosen,
        decided_by="user", decided_ref="chat", decision_reason="explicit filenames win",
        apply_plan=[{"memory_id": left, "action": "preserve_historical_record"},
                    {"memory_id": right, "action": "use_as_resolution"}],
        resolution_memory_id=right,
    )
    assert judged["outcome"] == "applying"
    step1 = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created["conflict_id"], "expected_revision": judged["revision"],
        "memory_id": left, "action": "preserve_historical_record", "authorized": True,
    })["data"]
    assert step1["outcome"] == "completed"
    step2 = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created["conflict_id"], "expected_revision": step1["revision"],
        "memory_id": right, "action": "use_as_resolution", "authorized": True,
    })["data"]
    assert step2["outcome"] == "completed"
    resolved = db.resolve_conflict(created["conflict_id"], expected_revision=step2["revision"])
    assert resolved["outcome"] == "resolved"


def test_record_conflict_rejects_paraphrased_normalized_value(tmp_path: Path) -> None:
    """D1 (#970): a member whose normalized_value is an agent's paraphrase of
    value_raw (not its mechanical normalization) must be rejected at intake."""
    db = _db(tmp_path)
    left = _memory(db, "database is mysql")
    member = _member(left, "mysql")
    paraphrased = member.to_dict()
    paraphrased["normalized_value"] = "use mysql only"
    rejected = db.record_conflict_group(
        workspace_canonical="w",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=[paraphrased],
        value_groups=[{"normalized_value": "use mysql only", "display_value": "x",
                       "members": [f"{left}@1"]}],
        detection_reason="different", source="scan", detector_version="d1",
    )
    assert rejected["outcome"] == "invalid_input"
    assert "normalize_value" in str(rejected.get("error"))


def test_replan_updates_chosen_value_and_recovers_grounding_failure(tmp_path: Path) -> None:
    """D3 (#970): chosen_value is mutable via replan, and the update_current_claim
    grounding failure carries the recovery note."""
    db = _db(tmp_path)
    tools = MemoryTools(db.settings, db)
    left, right = _memory(db, "database is mysql"), _memory(db, "database is sqlite")
    members = [_member(left, "mysql"), _member(right, "sqlite")]
    created = db.record_conflict_group(
        workspace_canonical="w",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=members, value_groups=_groups(*members),
        detection_reason="different database", source="scan", detector_version="d1",
    )
    judged = db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="sqlite",
        decided_by="user", decided_ref="chat", decision_reason="confirmed",
        apply_plan=[{"memory_id": left, "action": "update_current_claim"}],
        resolution_memory_id=right,
    )
    assert judged["outcome"] == "applying"

    bad = db.conflicts.replan_conflict(
        created["conflict_id"], expected_revision=judged["revision"],
        apply_plan=[{"memory_id": left, "action": "update_current_claim"}],
        chosen_value="not a stored value",
    )
    assert bad["outcome"] == "invalid_chosen_value"

    # update_current_claim still grounds: an edit that does not contain the
    # chosen value fails, names the replan recovery, and stays recoverable.
    failed = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created["conflict_id"], "expected_revision": judged["revision"],
        "memory_id": left, "action": "update_current_claim", "authorized": True,
        "content": "数据库仍然使用 mysql，未定论。",
    })["data"]
    assert failed["outcome"] == "apply_failed"
    assert failed["apply_summary"]["plan"][0]["error"] == "chosen_value_not_grounded"
    assert failed["action_required"] == "replan_conflict"
    assert "chosen_value" in failed["note"]

    # Recovery A: rewrite the content so the chosen value is grounded.
    # (Near-duplicate wording: editing left into right's exact bytes is a
    # 0.16.6 duplicate_content refusal; grounding only needs "sqlite".)
    recovered = db.conflicts.replan_conflict(
        created["conflict_id"], expected_revision=failed["revision"],
        apply_plan=[{"memory_id": left, "action": "update_current_claim"}],
    )
    assert recovered["outcome"] == "replanned"
    rewritten = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created["conflict_id"], "expected_revision": recovered["revision"],
        "memory_id": left, "action": "update_current_claim", "authorized": True,
        "content": "database is sqlite, confirmed choice",
    })["data"]
    assert rewritten["outcome"] == "completed"
    resolved = db.resolve_conflict(created["conflict_id"], expected_revision=rewritten["revision"])
    assert resolved["outcome"] == "resolved"


def test_replan_chosen_value_update_changes_stored_choice(tmp_path: Path) -> None:
    """D3 (#970): a replan with chosen_value replaces the stored choice with
    the value_groups' own normalized form (canonical, not caller casing)."""
    db = _db(tmp_path)
    left, right = _memory(db, "mysql"), _memory(db, "sqlite")
    created = _record(db, [_member(left, "mysql"), _member(right, "sqlite")])
    judged = db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="mysql",
        decided_by="user", decided_ref="chat", decision_reason="first pick",
        apply_plan=[{"memory_id": left, "action": "use_as_resolution"}],
        resolution_memory_id=left,
    )
    assert judged["outcome"] == "applying"
    updated = db.conflicts.replan_conflict(
        created["conflict_id"], expected_revision=judged["revision"],
        apply_plan=[{"memory_id": right, "action": "use_as_resolution"}],
        chosen_value=" SQLITE ",
    )
    assert updated["outcome"] == "replanned"
    row = db.get_conflict(created["conflict_id"])
    assert row is not None
    assert row["chosen_value"] == "sqlite"
    # P1 (#970 adversarial): a chosen_value replacement moves the resolution
    # holder to a member of the new value group; keeping the old holder would
    # record the rejected value's owner as the surviving authority.
    assert row["resolution_memory_id"] == right


def test_use_as_resolution_rejects_non_chosen_value_holder(tmp_path: Path) -> None:
    """P1 (#970 adversarial): use_as_resolution must fail on a member whose
    own value group is not the chosen value — otherwise a plan could mark a
    mysql holder as the sqlite resolution and resolve the group, leaving the
    wrong data in place. The failure is recoverable via replan."""
    db = _db(tmp_path)
    tools = MemoryTools(db.settings, db)
    left, right = _memory(db, "db is mysql"), _memory(db, "db is sqlite")
    members = [_member(left, "mysql"), _member(right, "sqlite")]
    created = db.record_conflict_group(
        workspace_canonical="w",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=members, value_groups=_groups(*members),
        detection_reason="different database", source="scan", detector_version="d1",
    )
    judged = db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="sqlite",
        decided_by="user", decided_ref="chat", decision_reason="misplanned",
        apply_plan=[{"memory_id": right, "action": "preserve_historical_record"},
                    {"memory_id": left, "action": "use_as_resolution"}],
        resolution_memory_id=left,
    )
    assert judged["outcome"] == "applying"
    step1 = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created["conflict_id"], "expected_revision": judged["revision"],
        "memory_id": right, "action": "preserve_historical_record", "authorized": True,
    })["data"]
    assert step1["outcome"] == "completed"
    misplaced = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created["conflict_id"], "expected_revision": step1["revision"],
        "memory_id": left, "action": "use_as_resolution", "authorized": True,
    })["data"]
    assert misplaced["outcome"] == "apply_failed"
    assert misplaced["apply_summary"]["plan"][1]["error"] == "resolution_member_value_mismatch"
    assert misplaced["action_required"] == "replan_conflict"
    # Recovery: replan points the marker at the actual sqlite holder.
    recovered = db.conflicts.replan_conflict(
        created["conflict_id"], expected_revision=misplaced["revision"],
        apply_plan=[{"memory_id": right, "action": "use_as_resolution"}],
        resolution_memory_id=right,
    )
    assert recovered["outcome"] == "replanned"
    step2 = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created["conflict_id"], "expected_revision": recovered["revision"],
        "memory_id": right, "action": "use_as_resolution", "authorized": True,
    })["data"]
    assert step2["outcome"] == "completed"
    resolved = db.resolve_conflict(created["conflict_id"], expected_revision=step2["revision"])
    assert resolved["outcome"] == "resolved"


def test_use_as_resolution_survives_stale_conflict_not_grounded(tmp_path: Path) -> None:
    """D2 (#970): use_as_resolution no longer depends on content grounding;
    only the CAS revision still guards it."""
    db = _db(tmp_path)
    tools = MemoryTools(db.settings, db)
    left, right = _memory(db, "无英文短语的中文正文甲"), _memory(db, "无英文短语的中文正文乙")
    members = [_member(left, "上传 dist/*"), _member(right, "显式文件名上传")]
    created = db.record_conflict_group(
        workspace_canonical="w",
        slot_key={"entity": "release", "attribute": "upload", "scope": "pypi"},
        members=members, value_groups=_groups(*members),
        detection_reason="different", source="scan", detector_version="d1",
    )
    judged = db.judge_conflict(
        created["conflict_id"], expected_revision=1,
        chosen_value=normalize_value("显式文件名上传"),
        decided_by="user", decided_ref="chat", decision_reason="confirmed",
        apply_plan=[{"memory_id": left, "action": "preserve_historical_record"},
                    {"memory_id": right, "action": "use_as_resolution"}],
        resolution_memory_id=right,
    )
    assert judged["outcome"] == "applying"
    step1 = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created["conflict_id"], "expected_revision": judged["revision"],
        "memory_id": left, "action": "preserve_historical_record", "authorized": True,
    })["data"]
    assert step1["outcome"] == "completed"
    step2 = tools.memory_govern("apply_conflict_action", {
        "conflict_id": created["conflict_id"], "expected_revision": step1["revision"],
        "memory_id": right, "action": "use_as_resolution", "authorized": True,
    })["data"]
    assert step2["outcome"] == "completed"
    resolved = db.resolve_conflict(created["conflict_id"], expected_revision=step2["revision"])
    assert resolved["outcome"] == "resolved"


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


# ── from test_product_conflict_groups.py ──
# helpers renamed: _product_memory, _product_member, _product_tools (collisions with test_conflict_groups.py)


from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.semantic_conflict import normalize_value
from memory_arbiter.tools import MemoryTools
from tests._helpers import _db, _groups, _member, _memory, _record


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
        normalized_attribute="database", normalized_value=normalize_value(value), evidence_quote=quote,
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
        "action": "update_current_claim", "content": "database is sqlite, confirmed choice",
        "reason": "apply confirmed value", "authorized": True,
    })
    assert applied["ok"] is True
    # Near-duplicate wording: the 0.16.6 dedup gate refuses editing one
    # active row into another's byte-identical content, so the claim update
    # grounds "sqlite" without cloning the sqlite memory verbatim.
    assert db.get_memory(mysql)["content"] == "database is sqlite, confirmed choice"
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

