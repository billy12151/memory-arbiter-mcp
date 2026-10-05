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
        normalized_attribute="database", normalized_value=normalize_value(value), evidence_quote=quote,
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



def _pass_cos_gate(monkeypatch, cos: float = 0.85):
    """Gate-v2 G4: pass every hand-built hit through the cosine band
    (embedder-agnostic — patches the gates module, which the detection loop
    imports at call time)."""
    import memory_arbiter.pipeline.gates as _gates
    monkeypatch.setattr(
        _gates, "candidate_cos_gate",
        lambda own, hits, vecs: ([(h, cos) for h in hits], [], []),
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
        # 0.17.0 C5: rows are the evidence channel — publish_rows is the one
        # true publisher (unit tables are never written).
        rows = segment_rows(
            str(memory.get("subject") or ""), str(memory.get("content") or ""),
        )
        embeddings = [[0.0, 1.0] for _row in rows]
        published = db.evidence.publish_rows(
            int(memory["id"]), int(memory.get("version") or 1),
            evidence_content_hash(str(memory.get("content") or "")), rows, embeddings,
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

from memory_arbiter.db_generation import detect_database_generation

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


def _write_pair(tools, meta: dict, left: str = "database is mysql",
                right: str = "database is sqlite") -> tuple[dict, dict]:
    peer = tools.memory_write(content=left, subject="a", tags=[], metadata=meta)["data"]
    new = tools.memory_write(content=right, subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    return peer, new


def _hit(peer_id: int, text: str, *, row_id: int = 1, distance: float = 0.1,
         metadata: dict | None = None) -> dict:
    # 0.16.2 write-time provenance gate reads hit['metadata'] exactly like
    # the real knn row does — hand-built hits carry it explicitly.
    return {
        "memory_id": peer_id, "id": row_id, "kind": "text", "text": text,
        "start_offset": 0, "end_offset": len(text), "distance": distance,
        "metadata": metadata,
    }


def test_notice_value_groups_carry_quote_sides(tmp_path: Path, monkeypatch) -> None:
    """0.17.1: judged notices carry the two ROW TEXTS as the value sides —
    no extraction, so value_groups hold the quotes themselves (empty
    normalised value, quote as display). The old missing-parsed-keys
    tolerance died with the extraction paradigm (nothing parses)."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "MyProject", "scope": "Production"}
    peer, new = _write_pair(tools, meta, left="database mysql 8 and 16", right="database sqlite 3 and 4")
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: [_hit(peer["id"], "database mysql 8 and 16", metadata=dict(meta))])
    _pass_cos_gate(monkeypatch)

    class Backend:
        @staticmethod
        def judge_pair(text_a, text_b):
            from memory_arbiter.semantic_judge import PairVerdict
            return PairVerdict(
                "conflict", {"conflict": 0.95, "no_conflict": 0.03, "possible_conflict": 0.02},
                "numeric_value", "mdeberta-v4m:test",
            )
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: Backend())

    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert result["outcome"] == "notices_created"
    notice = tools.db.list_semantic_notices(status="open", limit=10)[0]
    groups = notice["payload"]["value_groups"]
    assert [g["display_value"] for g in groups] == [
        "database sqlite 3 and 4", "database mysql 8 and 16",
    ]
    payload = notice["payload"]
    assert payload["model_signal"]["label"] == "conflict"
    assert payload["model_signal"]["model_version"].startswith("mdeberta-v4m:")


def test_same_reason_degradation_counted_once_per_task(tmp_path: Path, monkeypatch) -> None:
    """Two pairs failing with the same technical reason bump the counter once."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    peers = [
        tools.memory_write(
            content=f"database {value} 8 and 16", subject=value, tags=[], metadata=meta,
        )["data"]
        for value in ("mysql", "postgres")
    ]
    new = tools.memory_write(
        content="database sqlite 3 and 4", subject="sqlite", tags=[], metadata=meta,
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    hits = [
        _hit(peers[0]["id"], "database mysql 8 and 16", row_id=1, distance=0.1, metadata=dict(meta)),
        _hit(peers[1]["id"], "database postgres 8 and 16", row_id=2, distance=0.2, metadata=dict(meta)),
    ]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: list(hits))
    _pass_cos_gate(monkeypatch)

    class BadOutput:
        @staticmethod
        def judge_pair(text_a, text_b):
            from memory_arbiter.semantic_judge import PairVerdict
            # both pairs fail with the SAME technical reason
            return PairVerdict("no_conflict", {}, None, "mdeberta:unavailable", error="bad output")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: BadOutput())

    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert result["status"] == "incomplete"
    assert result["reason"] == "judge_backend_error"
    assert result["reasons_seen"] == ["judge_backend_error"]
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
        normalized_attribute="连接池上限", normalized_value=normalize_value(value), evidence_quote=quote,
        evidence_span=(0, len(quote)), content_hash=(str(memory_id) * 64)[:64],
        direction="a_to_b", prompt_version="p1", detector_version="d1",
    ).to_dict()


@pytest.mark.parametrize("stored_entity,stored_scope", [
    # Gate-v2 G3: slot identity is workspace + own subject now — both sides
    # build the canon form directly, so the old raw-vs-canon mismatch
    # (pre-canon legacy rows) has no metadata source to diverge from.
    ("default", "a"),
])

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



def test_notice_dedupe_is_symmetric_and_version_pinned() -> None:
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_evidence") == notice_dedupe_key(
        2, 1, 4, 3, "semantic_evidence"
    )
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_evidence") != notice_dedupe_key(
        1, 2, 4, 4, "semantic_evidence"
    )


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


def test_decimal_point_preserved_in_normalization() -> None:
    from memory_arbiter.semantic_conflict import normalize_value
    assert normalize_value("1.5s") != normalize_value("15s")
    # Integer/fractional pair stays distinct.
    assert normalize_value("1.5") != normalize_value("15")


# ── 2026-09-16 (eval cf-oppo-10/12): direction-dependent fillers ──────────

def test_normalize_value_strips_approximator_prefix_and_measure_ge() -> None:
    from memory_arbiter.semantic_conflict import normalize_value
    assert normalize_value("近 90 天") == normalize_value("90天")
    assert normalize_value("约 500 元") == normalize_value("500元")
    assert normalize_value("大约 3 小时") == normalize_value("3小时")
    assert normalize_value("5个工作日") == normalize_value("5工作日")
    # Real value differences survive the filler stripping.
    assert normalize_value("近 90 天") != normalize_value("180天")
    assert normalize_value("5个工作日") != normalize_value("72小时")


def test_workday_calendar_day_units_convert() -> None:
    from memory_arbiter.semantic_conflict import canonical_unit_value as cuv
    assert cuv("5个工作日") == cuv("5days") == "432000000ms"
    assert cuv("3自然日") == cuv("3日") == "259200000ms"
    assert cuv("5businessdays") == cuv("5calendardays") == "432000000ms"
    assert cuv("5workdays") == cuv("5workingdays") == "432000000ms"
    # 同值方向的行为变化是拍板口径：工作日按自然天 24h，3 工作日 = 72 小时。
    assert cuv("3个工作日") == cuv("72小时") == "259200000ms"


def test_cf_oppo_12_direct_path_lands_after_workday_unit() -> None:
    from memory_arbiter.semantic_conflict import (
        decide_evidence, direct_value_verdict, _normalized_values, _numeric_stripped_skeleton,
    )
    left, right = "对美专线清关承诺 72 小时内完成。", "对美专线清关承诺 5 个工作日内完成。"
    decision = decide_evidence(left, right)
    assert decision.reason == "numeric_value_candidate"
    assert decision.left_value == "259200000ms" and decision.right_value == "432000000ms"
    assert _numeric_stripped_skeleton(left) == _numeric_stripped_skeleton(right)
    hit = direct_value_verdict(left, right, decision)
    assert hit is not None and hit[1] == "259200000ms" and hit[2] == "432000000ms"


def test_unit_canonical_conversion() -> None:
    from memory_arbiter.semantic_conflict import canonical_unit_value, normalize_value
    # Exact physical relations convert to the canonical base (ms / kb / 元).
    assert normalize_value("0.5s") == normalize_value("500ms")
    assert normalize_value("72小时") == normalize_value("3天")
    assert normalize_value("1GB") == normalize_value("1024mb")
    assert normalize_value("90秒") == normalize_value("1.5分钟")
    assert normalize_value("1.5 hours") == normalize_value("90 minutes")
    assert normalize_value("5万") == "50000"
    assert normalize_value("500块") == normalize_value("500元")
    # Deliberately NOT converted: 工作日 (domain assumption), 月/年
    # (variable length). % carries no physical conversion — and note the
    # pre-existing _mechanical_normalize drops the sign, so "0.3%" reads as
    # the bare number (unchanged by this change).
    assert normalize_value("5个工作日") != normalize_value("40小时")
    assert normalize_value("3个月") != normalize_value("90天")
    assert normalize_value("0.3%") == "0.3"
    # Non-numeric shapes are identity.
    assert normalize_value("mysql") == "mysql"
    assert canonical_unit_value("") == ""


def test_numeric_channel_unit_equivalence_is_duplicate() -> None:
    """"500ms" vs "0.5s" canonicalises equal — a duplicate, never a conflict
    candidate; "500ms" vs "5s" stays a candidate."""
    from memory_arbiter.semantic_conflict import decide_evidence
    decision = decide_evidence("接口超时时间为 500ms。", "接口超时时间为 0.5s。")
    assert decision.action == "ignore"
    decision = decide_evidence("接口超时时间为 500ms。", "接口超时时间为 5s。")
    assert decision.action == "check" and decision.reason == "numeric_value_candidate"


# ── 2026-09-16 (owner): deterministic direct path ─────────────────────────

def test_direct_value_verdict_threshold_and_guards() -> None:
    from memory_arbiter.semantic_conflict import (
        EvidenceDecision, decide_evidence, direct_value_verdict,
    )
    # Same skeleton + single canonical value difference → verdict.
    decision = decide_evidence("接口超时时间为 500ms。", "接口超时时间为 200ms。")
    verdict = direct_value_verdict(
        "接口超时时间为 500ms。", "接口超时时间为 200ms。", decision, key_cosine=1.0,
    )
    assert verdict is not None and verdict[1] == "500ms" and verdict[2] == "200ms"
    assert verdict[0]  # non-empty attribute from the skeleton
    # Below the key threshold → back to Qwen (tested with DIFFERING
    # skeletons; identical skeletons shortcut at cosine 1.0 by design).
    differing = EvidenceDecision("check", "numeric_value_candidate", [], "500ms", "200ms")
    assert direct_value_verdict(
        "接口超时时间为 500ms。", "接口超时上限为 200ms。", differing, key_cosine=0.90,
    ) is None
    assert direct_value_verdict(
        "接口超时时间为 500ms。", "接口超时上限为 200ms。", differing, key_cosine=0.97,
    ) is not None
    # Canonically EQUAL values never reach the direct path (not a candidate).
    equal = decide_evidence("接口超时时间为 500ms。", "接口超时时间为 0.5s。")
    assert direct_value_verdict(
        "接口超时时间为 500ms。", "接口超时时间为 0.5s。", equal, key_cosine=1.0,
    ) is None
    # Version ordinals: evolution must not become a direct conflict.
    versioned = EvidenceDecision("check", "numeric_value_candidate", [], "500ms", "200ms")
    assert direct_value_verdict(
        "方案 v1 超时上限为 500ms。", "方案 v2 超时上限为 200ms。", versioned, key_cosine=1.0,
    ) is None
    # Evolution wording veto still applies on the direct path.
    evolving = decide_evidence("超时配置不再采用 500ms。", "超时配置改为 200ms。")
    if evolving.action == "check":
        assert direct_value_verdict(
            "超时配置不再采用 500ms。", "超时配置改为 200ms。", evolving, key_cosine=1.0,
        ) is None
    # Multi-value sentences are pairwise-ambiguous.
    multi = decide_evidence("连接池上限为 10，队列长度为 3。", "连接池上限为 99，队列长度为 5。")
    assert multi.left_value and ", " in multi.left_value
    assert direct_value_verdict(
        "连接池上限为 10，队列长度为 3。", "连接池上限为 99，队列长度为 5。", multi, key_cosine=1.0,
    ) is None


def test_english_pairs_rule_layer() -> None:
    """English evidence: numeric channel, identifier value-opposition, and
    the direct path all work language-agnostically."""
    from memory_arbiter.difference_classifier import classify_pair
    from memory_arbiter.semantic_conflict import decide_evidence, direct_value_verdict
    # Numeric channel with English units.
    decision = decide_evidence(
        "Refund requests over 5000 CNY require finance review.",
        "Refund requests over 500 CNY require finance review.",
    )
    assert decision.action == "check" and decision.reason == "numeric_value_candidate"
    assert classify_pair(
        "Refund requests over 5000 CNY require finance review.",
        "Refund requests over 500 CNY require finance review.",
        route=decision.reason,
    ) == "keep"
    # Identifier opposition (MySQL/PostgreSQL) via the value-opposition keep.
    left = "The production database uses MySQL with a dual-primary setup."
    right = "The production database uses PostgreSQL with a single primary."
    decision = decide_evidence(left, right)
    assert decision.action == "check"
    assert classify_pair(left, right, route=decision.reason) == "keep"
    # English numeric pair with identical skeleton → direct verdict.
    decision = decide_evidence(
        "The connection pool limit is 10.", "The connection pool limit is 99.",
    )
    verdict = direct_value_verdict(
        "The connection pool limit is 10.", "The connection pool limit is 99.",
        decision, key_cosine=1.0,
    )
    assert verdict is not None and verdict[1] == "10" and verdict[2] == "99"


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
    assert reopened.resolve_workspace_canonical("raw", None)["canonical"] == "raw"
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
        assert conn.execute("SELECT COUNT(*) FROM memory_row").fetchone()[0] == 2
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
                "SELECT id,parent_status,hex(embedding) FROM memory_row_vec ORDER BY id"
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
            "SELECT id,parent_status,hex(embedding) FROM memory_row_vec ORDER BY id"
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
        row_id = conn.execute(
            "SELECT id FROM memory_row WHERE memory_id=? ORDER BY id LIMIT 1",
            (memory_id,),
        ).fetchone()[0]
        conn.execute("DELETE FROM memory_row_vec WHERE id=?", (row_id,))
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



def _blocking_child(conn, config):
    try:
        request = conn.recv()
        if request.get("command") == "load":
            conn.send({"ok": True, "result": {"loaded": True}})
            conn.recv()
        time.sleep(10)
    except (EOFError, OSError):
        return


def test_bare_candidate_promoted_when_rerecorded_as_open(tmp_path: Path) -> None:
    # mema 924 #2（Team 93d17fc 移植）：裸 candidate（无 notice、无 slot）撞上后续
    # 同事件 open 记录时必须就地提升，而不是返回 deduped 把事件永久卡死在 candidate。
    db = _db(tmp_path)
    m1, m2 = _memory(db, "database is sqlite"), _memory(db, "database is postgres")
    a, b = _member(m1, "sqlite"), _member(m2, "postgres")
    created = _record(db, [a, b])
    assert created["outcome"] == "inserted", created
    cid = int(created["conflict_id"])
    # 模拟任一生产者把该事件冻结为裸 candidate：status 降 candidate、slot/notice 清空。
    with db.connection() as conn:
        conn.execute(
            "UPDATE conflicts SET status='candidate', slot_key=NULL, slot_key_hash=NULL, notice_type=NULL WHERE id=?",
            (cid,),
        )
        conn.commit()
    with db.connection() as conn:
        row = conn.execute("SELECT status, slot_key, notice_type FROM conflicts WHERE id=?", (cid,)).fetchone()
    assert row["status"] == "candidate" and row["slot_key"] is None and row["notice_type"] is None

    op = _record(db, [a, b])
    assert op["outcome"] == "deduped", op
    assert int(op["conflict_id"]) == cid
    assert int(op["revision"]) == int(created["revision"]) + 1
    with db.connection() as conn:
        row2 = conn.execute("SELECT status, slot_key FROM conflicts WHERE id=?", (cid,)).fetchone()
    assert row2["status"] == "open", f"candidate row was not promoted: {dict(row2)}"
    assert row2["slot_key"] is not None


def test_pre_qwen_vetoes_ops_marker_and_decision_vs_observation() -> None:
    """2026-09-17 (owner): 进 Qwen 前的两个 ignore veto。
    ops_marker 拦运维标记（时间戳尾 slug / test marker 措辞）；
    decision_vs_observation 拦决策措辞×实测措辞的来源不对称
    （owner 原则：证据 vs 裁决不构成对立）。正样本与 owner 裁定为真
    冲突的 conflict-test 基准对必须保活。"""
    from memory_arbiter.semantic_conflict import decide_evidence
    # 目标命中
    assert decide_evidence(
        "JingleAI global bridge self-test marker jingleai-global-bridge-self-test-1783837684.",
        "JingleAI global bridge full-test marker jingleai-global-bridge-fulltest-1783837846.",
    ).reason == "ops_marker"
    assert decide_evidence(
        "用户进一步修正 weak 归一策略：Qwen 高置信的直接静默处理。",
        "2026-08-08 按用户最新原则测试 weak/strict 策略：样本扩展到 54 条。",
    ).reason == "decision_vs_observation"
    # owner 示例：实测结论 vs 用户确认
    assert decide_evidence("用户确认上限 5000 元。", "实测结论上限 500 元。").reason \
        == "decision_vs_observation"
    # 保活：conflict-test 基准对（owner 裁定为真冲突，marker 词面刻意不匹配）
    d = decide_evidence("conflict-test 的 export-format 取值为 json。",
                        "conflict-test 的 export-format 取值为 csv。")
    assert d.action == "check"
    # 保活：数值正样本不受 veto 影响
    assert decide_evidence("单笔退款超过 5000 元需要财务复核。",
                           "单笔退款超过 500 元就需要财务复核。").reason == "numeric_value_candidate"


def test_chinese_duration_folds_to_same_value() -> None:
    """0.17.0 P2-1.2：中文时长词预折——「半秒 vs 500毫秒」曾因归一化失败
    误报为冲突（基线 ny-neg-samevalue 活体证据）；第 1-5 轮 spike 铁证
    D1 型（值都抽对、全死在归一化）。"""
    from memory_arbiter.semantic_conflict import normalize_value
    for left, right in [
        ("半秒", "500毫秒"), ("半秒", "0.5s"), ("半秒", "500ms"),
        ("一刻钟", "15分钟"), ("一刻钟", "900000ms"), ("半小时", "30分钟"),
        ("两秒", "2秒"), ("两秒", "2000ms"), ("十点", "10点"),
        ("三小时", "180分钟"),
    ]:
        assert normalize_value(left) == normalize_value(right), (left, right)
    # 不等仍是冲突：十点 vs 两点
    assert normalize_value("十点") != normalize_value("两点")
    assert normalize_value("半秒") != normalize_value("三秒")
    # 散文不折：折算只作用于整值锚定形态
    from memory_arbiter.semantic_conflict import _fold_chinese_duration
    assert _fold_chinese_duration("大约半秒的延迟") == "大约半秒的延迟"



def test_pre_qwen_veto_process_records() -> None:
    """2026-09-17（owner 原则延伸）：过程记录（review 轮次/设计→发版/
    重启复验）在 Qwen 之前过滤——单向实测三个穿透形态的收口。业务
    对象 id 的值对立（任务 id=123 预算 5000 vs 500）必须保活。"""
    from memory_arbiter.semantic_conflict import _PROCESS_REVERIFY_RE, decide_evidence
    assert decide_evidence(
        "id=609 adversarial review findings after implementation: backend import",
        "Follow-up review for id=609 after attempted fixes: environment confirmed",
    ).reason == "process_record"
    assert decide_evidence(
        "## 设计文档\n最终设计：v0.7.3_design_v3_final.md，审核已经过 8 轮。",
        "# memory-arbiter v0.8.0 发版完成\n分支已合并到 main。",
    ).reason == "process_record"
    assert decide_evidence(
        "JingleAI 使用 OpenClaw venv 验证 sqlite_vec 与 GGUF embedding 链路。",
        "JingleAI 重启后再次验证 memory-arbiter：sqlite_vec、embedding、检索链路正常。",
    ).reason == "process_record"
    # 保活：业务对象 id 的真值对立（same-object-id 规则被刻意砍掉的原因）
    d = decide_evidence("任务 id=123 的预算上限 5000 元。", "任务 id=123 的预算上限 500 元。")
    assert d.reason == "numeric_value_candidate"
    assert decide_evidence("单笔退款超过 5000 元需要财务复核。",
                           "单笔退款超过 500 元就需要财务复核。").reason == "numeric_value_candidate"
    # 0.17.0 P2-1.1：裸「复审」是政策语句不是评审记录——第 11 轮生产误杀
    # （"每半年复审"配置对永远到不了 Qwen）；带语境的评审记录仍 veto。
    assert decide_evidence(
        "网关证书每半年复审一次，由安全组执行。",
        "网关证书每季度复审一次，由安全组执行。",
    ).reason != "process_record"
    assert decide_evidence(
        "方案复审结论：通过，可进入实施。",
        "方案复审记录归档于 id=77。",
    ).reason == "process_record"
    # 同组排查：裸「复验」与 "pre-verify" 不再误杀
    assert decide_evidence(
        "灰度复验通过后全量放开。",
        "灰度复验通过后按计划推进。",
    ).reason != "process_record"
    assert "pre-verify" not in _PROCESS_REVERIFY_RE.pattern  # 边界在位
    assert not _PROCESS_REVERIFY_RE.search("we pre-verify the checksum")


def test_lineage_version_evolution_veto_e1() -> None:
    """0.17.0 P2-6.3（E1）：双侧自报不同谱系版本=取代演进，不按冲突报；
    E2（产品版本差异）与无谱系对不受影响。"""
    from memory_arbiter.semantic_conflict import decide_evidence
    assert decide_evidence(
        "缓存 TTL 设计文档 v1.0：本地缓存 TTL 5 分钟，按业务域配置。",
        "缓存 TTL 设计文档 v2.0 取代 v1.0：TTL 调整为 10 分钟。",
    ).reason == "lineage_version_evolution"
    assert decide_evidence(
        "限流规则第一版：单机 100 QPS。",
        "限流规则第 2 版：单机 200 QPS。",
    ).reason == "lineage_version_evolution"
    # E2：同谱系语境下的产品版本差异仍是候选（不 veto）
    d = decide_evidence(
        "网关设计文档定稿：产品版本要求 1.0，协议走 gRPC。",
        "网关设计文档定稿：产品版本要求 2.0，协议走 REST。",
    )
    assert d.reason != "lineage_version_evolution"
    # 无谱系对：原行为不变
    assert decide_evidence("连接池上限为 10。", "连接池上限为 99。").reason == "numeric_value_candidate"
    # 对抗 review P1-1 反例：标记词+十进制数值是中文配置句式，不是版本号
    assert decide_evidence("熔断方案 1.5 秒超时。", "熔断方案 2.0 秒超时。").reason == "numeric_value_candidate"
    assert decide_evidence("上传规格 0.5mb 上限。", "上传规格 2.5mb 上限。").reason == "numeric_value_candidate"
    # v 前缀整数（无点）也生效
    assert decide_evidence(
        "方案 v1 上线后参数 X 为 5。", "方案 v2 上线后参数 X 为 9。",
    ).reason == "lineage_version_evolution"


