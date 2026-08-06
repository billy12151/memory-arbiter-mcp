from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from threading import Barrier

import pytest

from memory_arbiter.claims import extract_claims
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def _tools(tmp_path: Path, *, mode: str = "beta_all") -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "b.jsonl",
        client="test",
        agent_id="tester",
        workspace="ws",
        enable_sqlite_vec=False,
        structured_claim_mode=mode,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write_claim(
    tools: MemoryTools,
    value: str,
    *,
    entity: str = "api",
    attribute: str = "port",
    event_time: str = "2026-01-01T00:00:00Z",
    scope: str | None = None,
    source_type: str = "agent_generated",
    protection_level: str = "normal",
) -> dict:
    metadata = {"entity": entity}
    if scope is not None:
        metadata["scope"] = scope
    return tools.memory_write(
        content=f"{attribute}: {value}", subject="service config", tags=["config"],
        metadata=metadata, event_time=event_time, source_type=source_type,
        protection_level=protection_level,
    )


def test_v090_schema_migrates_legacy_database(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL,
          agent_id TEXT NOT NULL, workspace TEXT NOT NULL,
          tags TEXT NOT NULL DEFAULT '[]', source_type TEXT NOT NULL,
          source_ref TEXT, event_time TEXT NOT NULL, ingest_time TEXT NOT NULL,
          confidence REAL NOT NULL DEFAULT 0.5,
          protection_level TEXT NOT NULL DEFAULT 'normal',
          status TEXT NOT NULL DEFAULT 'active', subject TEXT,
          metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
        );
        CREATE TABLE conflicts (
          id INTEGER PRIMARY KEY AUTOINCREMENT, left_id INTEGER NOT NULL,
          right_id INTEGER NOT NULL, subject TEXT,
          status TEXT NOT NULL DEFAULT 'open', reason TEXT NOT NULL,
          winner_id INTEGER, created_at TEXT NOT NULL, resolved_at TEXT,
          FOREIGN KEY(left_id) REFERENCES memories(id),
          FOREIGN KEY(right_id) REFERENCES memories(id)
        );
        """
    )
    conn.close()
    settings = Settings(
        db_path=db_path, backup_jsonl=tmp_path / "legacy.jsonl",
        enable_sqlite_vec=False,
    )
    db = MemoryDB(settings)
    with db.connection() as migrated:
        memory_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(memories)")
        }
        conflict_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(conflicts)")
        }
        assert {
            "claim_revision", "claims_indexed_revision", "claims_reconciled_revision",
            "structured_enrich_ms", "structured_candidate_count",
        } <= memory_columns
        assert {
            "left_claim_revision", "right_claim_revision", "judgment_status",
            "active_judgment_id", "structured_details", "structured_detected_at",
            "scan_detected_at", "resolution_kind", "conflict_scope",
        } <= conflict_columns
        judgment_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(conflict_judgments)")
        }
        assert {"resolution_kind", "conflict_scope"} <= judgment_columns
        assert migrated.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_claims'"
        ).fetchone()
        assert migrated.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conflict_judgments'"
        ).fetchone()


def test_extract_all_phase1_rules_and_guards() -> None:
    record = {
        "content": (
            'timeout: 30s\n"port":"5432"\n数据库：MySQL\n原因：系统设计如此\n'
            '| Setting | Value |\n|---|---|\n| db_path | ./prod.db |\nQPS 1000\nv0.8.8\n'
            'id=123\ncommit: a1b2c3d\nurl: https://example.test\n'
            '更多信息 https://docs.example.test/setup\n'
            '数据库连接地址：MySQL\n'
            '```\nport: 9999\n```'
        ),
        "subject": "config", "tags": ["api"], "metadata": {"entity": "API"},
    }
    diagnostics: dict = {}
    claims = extract_claims(record, diagnostics)
    by_attr = {claim["attribute"]: claim for claim in claims}
    assert by_attr["timeout"]["value"] == "30 s"
    assert by_attr["port"]["value"] == "5432"
    assert by_attr["数据库"]["value"] == "mysql"
    assert by_attr["db_path"]["value"] == "./prod.db"
    assert "setting" not in by_attr
    assert by_attr["qps"]["value"] == "1000"
    assert by_attr["version"]["value"] == "0.8.8"
    assert "原因" not in by_attr
    assert "http" not in by_attr and "https" not in by_attr
    assert "连接地址" not in by_attr
    assert "id" not in by_attr and "commit" not in by_attr and "url" not in by_attr
    assert diagnostics["rejected_reference_count"] == 4
    assert all(claim["entity"] == "api" for claim in claims)
    assert all("evidence" in claim and "start_offset" in claim for claim in claims)


def test_same_key_distinct_values_are_ambiguous() -> None:
    diagnostics: dict = {}
    claims = extract_claims(
        {"content": "version: 0.8.8\nversion: 0.9.0", "subject": "s", "metadata": {"entity": "p"}},
        diagnostics,
    )
    assert claims == []
    assert diagnostics["ambiguous_key_count"] == 1


def test_entity_falls_back_to_default_after_generic_subject_and_tag() -> None:
    diagnostics: dict = {}
    claims = extract_claims(
        {"content": "port: 5432", "subject": "配置", "tags": ["notes", "api"], "metadata": {}},
        diagnostics,
    )
    assert claims[0]["entity"] == "default"
    assert diagnostics["entity_source"] == "default"


def test_schema_and_zero_claim_are_distinct_from_failure(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    result = tools.memory_write(content="plain prose without a structured value", subject="essay")
    memory = result["data"]["record"]
    assert result["data"]["claim_indexed"] is True
    assert result["data"]["realtime_conflict_check"]["claim_count"] == 0
    assert memory["claim_revision"] == memory["claims_indexed_revision"] == 1
    assert memory["claims_reconciled_revision"] == 1
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memory_claims").fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_claim_publish_failure_rolls_back_old_index_and_claims_cascade(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    memory_id = _write_claim(tools, "5432")["data"]["id"]
    original = tools.db.list_memory_claims(memory_id)
    duplicate = dict(original[0])
    duplicate.pop("id", None)
    outcome = tools.db.publish_memory_claims(
        memory_id, [duplicate, duplicate], expected_claim_revision=1,
    )
    assert outcome["outcome"] == "error"
    assert tools.db.list_memory_claims(memory_id)[0]["value"] == "5432"
    with tools.db.connection() as conn:
        conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        conn.commit()
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_claims WHERE memory_id=?", (memory_id,)
        ).fetchone()[0] == 0


def test_write_creates_pending_candidate_and_pair_aggregate(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    left = tools.memory_write(
        content="port: 5432\ntimeout: 30s", subject="cfg",
        metadata={"entity": "api"}, event_time="2026-01-01T00:00:00Z",
    )
    right = tools.memory_write(
        content="port: 3306\ntimeout: 60s", subject="cfg",
        metadata={"entity": "api"}, event_time="2026-01-01T00:00:00Z",
    )
    assert left["data"]["realtime_conflict_check"]["candidate_count"] == 0
    data = right["data"]
    assert data["attention_required"] is True
    assert data["action_required"] == "judge_conflict_before_use"
    assert data["verification_status"] == "pending_llm"
    assert len(data["conflict_judgment_requests"]) == 1
    request = data["conflict_judgment_requests"][0]
    assert len(request["claims"]) == 2
    conflicts = tools.memory_list_conflicts()["data"]["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["source"] == "structured_claim"
    assert conflicts[0]["judgment_status"] == "pending_llm"
    assert conflicts[0]["left_claim_revision"] == 1
    assert {row["attribute"] for row in conflicts[0]["structured_details"]} == {"port", "timeout"}


def test_conflict_canonicalization_swaps_all_side_pins(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    left = _write_claim(tools, "5432")["data"]["id"]
    right = _write_claim(tools, "5432", entity="other")["data"]["id"]
    result = tools.db.record_conflict_enriched(
        right, left, conflict_type="contradiction", conflict_point="api.port",
        reason="reverse-order regression", source="structured_claim",
        left_version=7, right_version=3,
        left_claim_revision=11, right_claim_revision=5,
        judgment_status="pending_llm",
        structured_details=[{
            "attribute": "port", "left_value": "3306", "right_value": "5432",
            "left_memory_id": right, "right_memory_id": left,
        }],
    )
    assert result["outcome"] == "inserted"
    row = tools.db.list_conflicts()[0]
    assert (row["left_id"], row["right_id"]) == (left, right)
    assert (row["left_version"], row["right_version"]) == (3, 7)
    assert (row["left_claim_revision"], row["right_claim_revision"]) == (5, 11)
    assert row["structured_details"][0]["left_value"] == "5432"
    assert row["structured_details"][0]["left_memory_id"] == left


def test_scope_and_evolution_do_not_ring(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432", scope="test")
    different_scope = _write_claim(tools, "3306", scope="prod")
    assert not different_scope["data"].get("attention_required")

    tools2 = _tools(tmp_path / "other")
    _write_claim(tools2, "5432", event_time="2026-01-01T00:00:00Z")
    later = _write_claim(tools2, "3306", event_time="2026-02-01T00:00:00Z")
    assert later["data"]["realtime_conflict_check"]["evolution_pair_count"] == 1
    assert not later["data"].get("attention_required")


def test_edit_reindexes_and_resolves_pair(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    right = _write_claim(tools, "3306")
    right_id = right["data"]["id"]
    edited = tools.memory_edit(memory_id=right_id, new_content="port: 5432")
    record = edited["data"]["record"]
    assert record["version"] == 2
    assert record["claim_revision"] == 2
    assert record["claims_indexed_revision"] == 2
    assert record["claims_reconciled_revision"] == 2
    assert tools.memory_list_conflicts()["data"]["count"] == 0


def test_partial_alignment_refreshes_multi_claim_pair(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    tools.memory_write(
        content="port: 5432\ntimeout: 30s", subject="cfg",
        metadata={"entity": "api"}, event_time="2026-01-01T00:00:00Z",
    )
    right = tools.memory_write(
        content="port: 3306\ntimeout: 60s", subject="cfg",
        metadata={"entity": "api"}, event_time="2026-01-01T00:00:00Z",
    )
    conflict_id = right["data"]["conflict_judgment_requests"][0]["conflict_id"]
    edited = tools.memory_edit(
        memory_id=right["data"]["id"],
        new_content="port: 5432\ntimeout: 60s",
    )
    assert edited["data"]["attention_required"] is True
    conflict = tools.memory_list_conflicts()["data"]["conflicts"][0]
    assert conflict["id"] == conflict_id
    assert "timeout" in conflict["conflict_point"]
    assert "port" not in conflict["conflict_point"]
    assert conflict["right_version"] == 2


def test_structured_dismissal_reopens_after_edit(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    right = _write_claim(tools, "3306")
    conflict_id = right["data"]["conflict_judgment_requests"][0]["conflict_id"]
    dismissed = tools.memory_resolve_conflict(conflict_id, status="not_a_conflict")
    assert dismissed["data"]["outcome"] == "not_a_conflict"
    edited = tools.memory_edit(memory_id=right["data"]["id"], new_content="port: 3307")
    assert edited["data"]["attention_required"] is True
    rows = tools.db.list_conflicts(status="not_a_conflict") + tools.db.list_conflicts(status="open")
    assert len(rows) == 2


def test_entity_update_keeps_content_version_and_bumps_claim_revision(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    memory_id = tools.memory_write(content="port: 5432", subject="release", tags=["cfg"])["data"]["id"]
    before = tools.db.get_memory(memory_id)
    updated = tools.memory_set_entity(memory_id, entity=" My Service ", scope=" PROD ")
    after = updated["data"]["record"]
    assert before["version"] == after["version"] == 1
    assert after["claim_revision"] == 2
    assert after["claims_indexed_revision"] == 2
    assert after["claims_reconciled_revision"] == 2
    assert after["metadata"] == {"entity": "my service", "scope": "prod"}
    entities = tools.memory_list_entities()["data"]
    assert entities["entities"][0]["entity"] == "my service"


def test_entity_update_is_idempotent_clearable_and_protected(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    memory_id = _write_claim(
        tools, "5432", source_type="user_confirmed", protection_level="locked",
    )["data"]["id"]
    denied = tools.memory_set_entity(memory_id, entity="worker")
    assert denied["ok"] is False and denied["data"]["outcome"] == "forbidden"
    changed = tools.memory_set_entity(memory_id, entity=" Worker ", authorized=True)
    revision = changed["data"]["record"]["claim_revision"]
    same = tools.memory_set_entity(memory_id, entity="worker", authorized=True)
    assert same["data"]["outcome"] == "no_change"
    assert same["data"]["record"]["claim_revision"] == revision
    cleared = tools.memory_set_entity(memory_id, clear=True, authorized=True)
    assert "entity" not in cleared["data"]["record"]["metadata"]
    assert tools.memory_history(memory_id)["data"]["count"] == 0


def test_entity_list_counts_unassigned_even_when_ids_are_hidden(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    tools.memory_write(content="ordinary prose", subject="release")
    _write_claim(tools, "5432")
    entities = tools.memory_list_entities(include_unassigned=False)["data"]
    assert entities["unassigned_count"] == 1
    assert entities["unassigned_ids"] == []


def test_llm_assessed_guidance_is_non_blocking_and_auditable(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    result = _write_claim(tools, "3306")
    request = result["data"]["conflict_judgment_requests"][0]
    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"],
        expected_left_version=request["left"]["version"],
        expected_right_version=request["right"]["version"],
        expected_left_claim_revision=request["left"]["claim_revision"],
        expected_right_claim_revision=request["right"]["claim_revision"],
        verdict="contradiction", recommended_use="right",
        suggested_winner=request["right"]["id"], confidence_hint="high",
        reason="the right record is the current answer source",
        affects_current_output=True, usage_context="answer",
    )
    assert judged["data"]["judgment_status"] == "llm_assessed"
    assert judged["data"]["disclosure_required"] is True
    searched = tools.memory_search(query="3306")
    assert not searched["data"].get("attention_required")
    signal = searched["data"]["results"][0]["conflict_signal"]
    assert signal["conflict_source"] == "conflict_guidance"
    assert signal["judgment"]["recommended_use"] == "right"
    docket = tools.memory_list_conflicts()["data"]["conflicts"][0]
    assert docket["judgment_recommended_use"] == "right"
    assert docket["judgment_judge_type"] == "llm"


def test_resolution_fields_classify_partial_update_without_supersede(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "4", attribute="state_count")
    result = _write_claim(tools, "5", attribute="state_count")
    request = result["data"]["conflict_judgment_requests"][0]

    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        verdict="evolution", recommended_use="merge", suggested_winner=None,
        confidence_hint="high", reason="only the state_count field evolved",
        affects_current_output=False, usage_context="unrelated",
        resolution_kind="partial_update", conflict_scope="field",
    )

    assert judged["data"]["judgment_status"] == "pending_user"
    assert judged["data"]["user_action_required"] is True
    assert judged["data"]["recommended_resolution_action"] == "update_or_merge"
    assert judged["data"]["supersede_candidate"] is False
    docket = tools.memory_list_conflicts()["data"]["conflicts"][0]
    assert docket["resolution_kind"] == "partial_update"
    assert docket["conflict_scope"] == "field"
    assert docket["active_judgment"]["recommended_resolution_action"] == "update_or_merge"


def test_resolution_fields_allow_full_replacement_suggestion(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "old", attribute="routing")
    result = _write_claim(tools, "new", attribute="routing")
    request = result["data"]["conflict_judgment_requests"][0]

    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        verdict="evolution", recommended_use="right",
        suggested_winner=request["right"]["id"], confidence_hint="high",
        reason="new memory fully replaces the old routing rule",
        affects_current_output=False, usage_context="unrelated",
        resolution_kind="full_replacement", conflict_scope="whole_memory",
    )

    assert judged["data"]["conflict_status"] == "open"
    assert judged["data"]["judgment_status"] == "pending_user"
    assert judged["data"]["user_action_required"] is True
    assert judged["data"]["recommended_resolution_action"] == "supersede_old_memory"
    assert judged["data"]["supersede_candidate"] is True
    searched = tools.memory_search(query="routing")
    signals = [r.get("conflict_signal") for r in searched["data"]["results"] if r.get("conflict_signal")]
    assert any(s["action_required"] == "ask_user" for s in signals)
    guidance = next(s for s in signals if s["action_required"] == "ask_user")
    assert guidance["conflict_status"] == "open"
    assert guidance["resolution_kind"] == "full_replacement"
    assert guidance["supersede_candidate"] is True


def test_stale_snapshot_clears_active_resolution_projection(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "old", attribute="routing")
    result = _write_claim(tools, "new", attribute="routing")
    request = result["data"]["conflict_judgment_requests"][0]
    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        verdict="contradiction", recommended_use="right",
        suggested_winner=request["right"]["id"], confidence_hint="high",
        reason="initial full replacement guidance",
        affects_current_output=False, usage_context="unrelated",
        resolution_kind="full_replacement", conflict_scope="whole_memory",
    )
    assert judged["data"]["supersede_candidate"] is True
    assert tools.db.edit_memory(request["right"]["id"], "routing: newer") is not None

    stale = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        verdict="contradiction", recommended_use="right",
        suggested_winner=request["right"]["id"], confidence_hint="high",
        reason="stale repeat",
        affects_current_output=False, usage_context="unrelated",
        resolution_kind="full_replacement", conflict_scope="whole_memory",
    )
    assert stale["data"]["outcome"] == "stale_snapshot"
    conflict = tools.memory_list_conflicts()["data"]["conflicts"][0]
    assert conflict["active_judgment_id"] is None
    assert conflict["resolution_kind"] is None
    assert conflict["supersede_candidate"] is False
    assert conflict["suggested_winner"] is None


def test_resolution_validation_rejects_inconsistent_combinations(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "4", attribute="state_count")
    result = _write_claim(tools, "5", attribute="state_count")
    request = result["data"]["conflict_judgment_requests"][0]

    partial_winner = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        verdict="evolution", recommended_use="right",
        suggested_winner=request["right"]["id"], confidence_hint="high",
        reason="invalid partial winner",
        affects_current_output=False, usage_context="unrelated",
        resolution_kind="partial_update", conflict_scope="field",
    )
    assert partial_winner["ok"] is False
    assert partial_winner["data"]["outcome"] == "invalid_recommendation"

    field_replacement = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        verdict="evolution", recommended_use="right",
        suggested_winner=request["right"]["id"], confidence_hint="high",
        reason="invalid field replacement",
        affects_current_output=False, usage_context="unrelated",
        resolution_kind="full_replacement", conflict_scope="field",
    )
    assert field_replacement["ok"] is False
    assert field_replacement["data"]["outcome"] == "invalid_resolution_scope"

    partial_none = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        verdict="evolution", recommended_use="none", suggested_winner=None,
        confidence_hint="high", reason="invalid partial none",
        affects_current_output=False, usage_context="unrelated",
        resolution_kind="partial_update", conflict_scope="field",
    )
    assert partial_none["ok"] is False
    assert partial_none["data"]["outcome"] == "invalid_recommendation"

    false_positive_winner = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        verdict="compatible", recommended_use="right",
        suggested_winner=request["right"]["id"], confidence_hint="high",
        reason="invalid not-a-conflict winner",
        affects_current_output=False, usage_context="unrelated",
        resolution_kind="not_a_conflict", conflict_scope="record",
    )
    assert false_positive_winner["ok"] is False
    assert false_positive_winner["data"]["outcome"] == "invalid_recommendation"


def test_pending_candidate_reappears_loudly_on_search(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    _write_claim(tools, "3306")
    searched = tools.memory_search(query="3306")
    assert searched["data"]["attention_required"] is True
    assert searched["data"]["action_required"] == "judge_conflict_before_use"
    signal = searched["data"]["results"][0]["conflict_signal"]
    assert signal["conflict_judgment_request"]["required_tool"] == "memory_submit_conflict_judgment"


def test_protected_policy_and_high_impact_escalation(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    protected = _write_claim(
        tools, "5432", source_type="user_confirmed", protection_level="locked"
    )["data"]["id"]
    result = _write_claim(tools, "3306")
    request = result["data"]["conflict_judgment_requests"][0]
    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"], expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="contradiction",
        recommended_use="right", suggested_winner=request["right"]["id"],
        confidence_hint="high", reason="model preferred the ordinary side",
        affects_current_output=True, usage_context="answer",
    )
    assert judged["data"]["suggested_winner"] == protected
    assert judged["data"]["user_action_required"] is False
    history = tools.memory_list_conflict_judgments(request["conflict_id"])["data"]["judgments"]
    assert [row["judge_type"] for row in history] == ["llm", "policy"]

    # A new pair whose value drives code is never auto-used.
    other = _write_claim(tools, "9999", entity="worker")
    _ = other
    high = _write_claim(tools, "8888", entity="worker")
    req2 = high["data"]["conflict_judgment_requests"][0]
    escalated = tools.memory_submit_conflict_judgment(
        conflict_id=req2["conflict_id"], expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="contradiction",
        recommended_use="right", suggested_winner=req2["right"]["id"],
        confidence_hint="high", reason="would drive generated code",
        affects_current_output=True, usage_context="code",
    )
    assert escalated["data"]["judgment_status"] == "pending_user"
    assert escalated["data"]["user_action_required"] is True


def test_human_correction_supersedes_llm_without_deleting_history(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    result = _write_claim(tools, "3306")
    request = result["data"]["conflict_judgment_requests"][0]
    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"], expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="contradiction",
        recommended_use="right", suggested_winner=request["right"]["id"],
        confidence_hint="medium", reason="initial model judgment",
        affects_current_output=False, usage_context="unrelated",
    )["data"]
    denied = tools.memory_correct_conflict_judgment(
        conflict_id=request["conflict_id"], verdict="contradiction",
        recommended_use="left", suggested_winner=request["left"]["id"],
        reason="human correction", expected_judgment_id=judged["judgment_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        authorized=False,
    )
    assert denied["ok"] is False
    corrected = tools.memory_correct_conflict_judgment(
        conflict_id=request["conflict_id"], verdict="contradiction",
        recommended_use="left", suggested_winner=request["left"]["id"],
        reason="human correction", expected_judgment_id=judged["judgment_id"],
        expected_left_version=1, expected_right_version=1,
        expected_left_claim_revision=1, expected_right_claim_revision=1,
        authorized=True,
    )
    assert corrected["data"]["judgment_status"] == "human_confirmed"
    rows = tools.memory_list_conflict_judgments(request["conflict_id"])["data"]["judgments"]
    assert len(rows) == 2
    assert rows[-1]["supersedes_judgment_id"] == judged["judgment_id"]

    overwritten = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"], expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="contradiction",
        recommended_use="right", suggested_winner=request["right"]["id"],
        confidence_hint="high", reason="attempted model overwrite",
        affects_current_output=False, usage_context="unrelated",
    )
    assert overwritten["ok"] is False
    assert overwritten["data"]["outcome"] == "higher_priority_judgment_active"


def test_stale_snapshot_is_rejected(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    result = _write_claim(tools, "3306")
    request = result["data"]["conflict_judgment_requests"][0]
    # Bypass tools' post-edit reconcile to simulate concurrent content change
    # between request preparation and verdict submission.
    assert tools.db.edit_memory(request["right"]["id"], "port: 3307") is not None
    stale = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"], expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="contradiction",
        recommended_use="right", suggested_winner=request["right"]["id"],
        confidence_hint="high", reason="stale result",
        affects_current_output=False, usage_context="unrelated",
    )
    assert stale["ok"] is False
    assert stale["data"]["outcome"] == "stale_snapshot"


def test_resolved_judgment_is_not_reopened_by_noop_rebuild(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    result = _write_claim(tools, "3306")
    request = result["data"]["conflict_judgment_requests"][0]
    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"], expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="compatible",
        recommended_use="contextual", suggested_winner=None,
        confidence_hint="high", reason="values apply to compatible subcontexts",
        affects_current_output=True, usage_context="answer",
    )
    assert judged["data"]["conflict_status"] == "resolved"
    assert judged["data"]["disclosure_required"] is True
    assert "contextual" in judged["data"]["disclosure"]
    rebuilt = tools.memory_rebuild_claims(
        memory_ids=[request["right"]["id"]], dry_run=False,
    )
    structured = rebuilt["data"]["results"][0]
    assert structured["diagnostic"]["pending_llm_count"] == 0
    assert tools.memory_list_conflicts()["data"]["count"] == 0


def test_confirm_invalidates_prior_guidance_and_reopens_gate(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    left_id = _write_claim(tools, "5432")["data"]["id"]
    result = _write_claim(tools, "3306")
    request = result["data"]["conflict_judgment_requests"][0]
    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"], expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="contradiction",
        recommended_use="right", suggested_winner=request["right"]["id"],
        confidence_hint="high", reason="initial low-risk guidance",
        affects_current_output=False, usage_context="unrelated",
    )
    assert judged["data"]["judgment_status"] == "llm_assessed"
    confirmed = tools.memory_confirm(left_id, authorized=True)
    assert confirmed["data"]["record"]["claim_revision"] == 2
    assert confirmed["data"]["verification_status"] == "pending_llm"
    new_request = confirmed["data"]["conflict_judgment_requests"][0]
    assert new_request["left"]["claim_revision"] == 2


def test_pending_user_rebuild_stays_user_gate(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    result = _write_claim(tools, "3306")
    request = result["data"]["conflict_judgment_requests"][0]
    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"], expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="uncertain",
        recommended_use="ask_user", suggested_winner=None,
        confidence_hint="low", reason="insufficient context",
        affects_current_output=True, usage_context="answer",
    )
    assert judged["data"]["judgment_status"] == "pending_user"
    # Reassigning the same canonical entity is a no-op, so use an explicit
    # rebuild of the exact snapshot to exercise gate routing.
    rebuilt = tools.memory_rebuild_claims(
        memory_ids=[request["right"]["id"]], dry_run=False,
    )["data"]["results"][0]
    assert rebuilt["diagnostic"]["pending_llm_count"] == 0
    assert rebuilt["diagnostic"]["pending_user_count"] == 1


def test_off_switch_and_rebuild_dry_run(tmp_path: Path) -> None:
    tools = _tools(tmp_path, mode="off")
    first = _write_claim(tools, "5432")
    second = _write_claim(tools, "3306")
    assert first["data"]["realtime_conflict_check"]["skipped_reason"] == "structured_claim_mode_off"
    assert not second["data"].get("attention_required")
    denied = tools.memory_rebuild_claims(dry_run=False)
    assert denied["ok"] is False
    plan = tools.memory_rebuild_claims(dry_run=True)
    assert plan["ok"] is True and plan["data"]["count"] == 2


def test_rebuild_without_ids_advances_through_stale_batches(tmp_path: Path) -> None:
    tools = _tools(tmp_path, mode="off")
    ids = [_write_claim(tools, str(5400 + i))["data"]["id"] for i in range(3)]
    tools.settings.structured_claim_mode = "beta_all"
    first = tools.memory_rebuild_claims(dry_run=False, batch_size=2)
    assert first["ok"] is True
    assert [row["memory_id"] for row in first["data"]["results"]] == ids[:2]
    second_plan = tools.memory_rebuild_claims(dry_run=True, batch_size=2)
    assert second_plan["data"]["memory_ids"] == ids[2:]
    second = tools.memory_rebuild_claims(dry_run=False, batch_size=2)
    assert second["ok"] is True and second["data"]["processed"] == 1
    assert tools.memory_rebuild_claims(dry_run=True)["data"]["count"] == 0


def test_claim_publish_retries_concurrent_revision_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _tools(tmp_path, mode="off")
    memory_id = _write_claim(tools, "5432")["data"]["id"]
    tools.settings.structured_claim_mode = "beta_all"
    original = tools.db.publish_memory_claims
    calls = 0

    def racing_publish(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert tools.db.edit_memory(memory_id, "port: 3306") is not None
        return original(*args, **kwargs)

    monkeypatch.setattr(tools.db, "publish_memory_claims", racing_publish)
    rebuilt = tools.memory_rebuild_claims(memory_ids=[memory_id], dry_run=False)
    assert rebuilt["ok"] is True and calls == 2
    claim = tools.db.list_memory_claims(memory_id)[0]
    assert claim["value"] == "3306"
    assert claim["claim_revision"] == 2


def test_claim_failure_is_fail_open_and_doctor_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _tools(tmp_path)

    def explode(*_args: object, **_kwargs: object) -> list[dict]:
        raise RuntimeError("extractor boom")

    monkeypatch.setattr("memory_arbiter.tools.extract_claims", explode)
    result = _write_claim(tools, "5432")
    assert result["ok"] is True
    assert result["data"]["claim_indexed"] is False
    assert not result["data"].get("attention_required")
    finding = next(
        row for row in tools.memory_doctor_overview()["data"]["findings"]
        if row["check_id"] == "consistency.structured_claims"
    )
    assert finding["status"] == "warn"
    assert finding["evidence"]["stale_memories"] == 1


def test_reconciliation_failure_remains_rebuildable_and_doctor_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    original_find = tools.db.find_structured_claim_pairs

    def fail_collision_query(_memory_id: int) -> dict:
        return {"pairs": [], "evolution_pairs": 0, "error": True}

    monkeypatch.setattr(tools.db, "find_structured_claim_pairs", fail_collision_query)
    second = _write_claim(tools, "3306")
    second_id = second["data"]["id"]
    assert second["data"]["claim_indexed"] is True
    assert second["data"]["claim_reconciled"] is False
    assert not second["data"].get("attention_required")
    monkeypatch.setattr(tools.db, "find_structured_claim_pairs", original_find)

    plan = tools.memory_rebuild_claims(dry_run=True)
    assert second_id in plan["data"]["memory_ids"]
    finding = next(
        row for row in tools.memory_doctor_overview()["data"]["findings"]
        if row["check_id"] == "consistency.structured_claims"
    )
    assert finding["evidence"]["stale_index_memories"] == 0
    assert finding["evidence"]["unreconciled_memories"] == 1

    repaired = tools.memory_rebuild_claims(memory_ids=[second_id], dry_run=False)
    assert repaired["ok"] is True
    assert repaired["data"]["results"][0]["diagnostic"]["claim_reconciled"] is True
    assert repaired["data"]["results"][0]["conflicts"]
    assert tools.db.get_memory(second_id)["claims_reconciled_revision"] == 1


def test_open_conflict_read_failure_does_not_advance_reconciled_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    original_read = tools.db.read_structured_open_conflicts_for_memory

    def fail_open_read(_memory_id: int) -> dict:
        return {"rows": [], "error": "injected open-row read failure"}

    monkeypatch.setattr(
        tools.db, "read_structured_open_conflicts_for_memory", fail_open_read,
    )
    second = _write_claim(tools, "3306")
    second_id = second["data"]["id"]
    assert second["data"]["claim_indexed"] is True
    assert second["data"]["claim_reconciled"] is False
    assert tools.db.get_memory(second_id)["claims_reconciled_revision"] is None

    monkeypatch.setattr(
        tools.db, "read_structured_open_conflicts_for_memory", original_read,
    )
    assert second_id in tools.memory_rebuild_claims(dry_run=True)["data"]["memory_ids"]
    repaired = tools.memory_rebuild_claims(memory_ids=[second_id], dry_run=False)
    assert repaired["ok"] is True
    assert tools.db.get_memory(second_id)["claims_reconciled_revision"] == 1


def test_unexpected_pair_write_failure_is_fail_open_and_rebuildable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    original_record = tools.db.record_conflict_enriched

    def fail_structured_record(*args: object, **kwargs: object) -> dict:
        if kwargs.get("source") == "structured_claim":
            raise sqlite3.OperationalError("injected pair write failure")
        return original_record(*args, **kwargs)

    monkeypatch.setattr(tools.db, "record_conflict_enriched", fail_structured_record)
    second = _write_claim(tools, "3306")
    second_id = second["data"]["id"]
    assert second["ok"] is True
    assert second["data"]["claim_indexed"] is True
    assert second["data"]["claim_reconciled"] is False
    assert second["data"]["realtime_conflict_check"]["skipped_reason"] == (
        "structured_enrichment_error"
    )
    assert second_id in tools.memory_rebuild_claims(dry_run=True)["data"]["memory_ids"]

    monkeypatch.setattr(tools.db, "record_conflict_enriched", original_record)
    repaired = tools.memory_rebuild_claims(memory_ids=[second_id], dry_run=False)
    assert repaired["ok"] is True
    assert repaired["data"]["results"][0]["conflicts"]


def test_concurrent_writes_surface_a_structured_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _tools(tmp_path)
    rendezvous = Barrier(2)
    original_publish = tools.db.publish_memory_claims

    def synchronized_publish(*args: object, **kwargs: object) -> dict:
        rendezvous.wait(timeout=5)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(tools.db, "publish_memory_claims", synchronized_publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_write_claim, tools, value)
            for value in ("5432", "3306")
        ]
        results = [future.result(timeout=10) for future in futures]

    assert all(result["ok"] for result in results)
    assert any(result["data"].get("attention_required") for result in results)
    conflicts = tools.memory_list_conflicts()["data"]["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["judgment_status"] == "pending_llm"
    assert conflicts[0]["left_claim_revision"] == 1
    assert conflicts[0]["right_claim_revision"] == 1


def test_scan_refresh_preserves_structured_snapshot_and_gate(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    left_id = _write_claim(tools, "5432")["data"]["id"]
    right = _write_claim(tools, "3306")
    right_id = right["data"]["id"]
    conflict_id = right["data"]["conflict_judgment_requests"][0]["conflict_id"]

    refreshed = tools.memory_record_conflict(
        left_id=left_id, right_id=right_id,
        reason="scheduled scan independently confirmed the contradiction",
        conflict_type="contradiction", conflict_point="api.port",
        source="llm_informed", refresh=True,
        left_version=1, right_version=1,
        scan_prompt_version="scan-v1", scan_model="test-model",
    )
    assert refreshed["data"]["outcome"] == "refreshed"
    row = tools.memory_list_conflicts()["data"]["conflicts"][0]
    assert row["id"] == conflict_id
    assert row["left_claim_revision"] == row["right_claim_revision"] == 1
    assert row["structured_detected_at"] is not None
    assert row["scan_detected_at"] is not None
    assert row["judgment_status"] == "pending_llm"

    searched = tools.memory_search(query="3306")
    assert searched["data"]["attention_required"] is True
    assert searched["data"]["action_required"] == "judge_conflict_before_use"
    finding = next(
        item for item in tools.memory_doctor_overview()["data"]["findings"]
        if item["check_id"] == "consistency.structured_claims"
    )
    assert finding["evidence"]["detection_channels"]["both"] == 1


def test_scan_refresh_does_not_overwrite_active_judgment_projection(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    left_id = _write_claim(tools, "5432")["data"]["id"]
    right = _write_claim(tools, "3306")
    request = right["data"]["conflict_judgment_requests"][0]
    right_id = right["data"]["id"]
    judged = tools.memory_submit_conflict_judgment(
        conflict_id=request["conflict_id"], expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="contradiction",
        recommended_use="right", suggested_winner=right_id,
        confidence_hint="high", reason="persisted host judgment",
        affects_current_output=False, usage_context="unrelated",
    )
    assert judged["data"]["judgment_status"] == "llm_assessed"
    before = tools.memory_list_conflicts()["data"]["conflicts"][0]

    tools.memory_record_conflict(
        left_id=left_id, right_id=right_id,
        reason="later scan disagreed with the persisted receipt",
        conflict_type="contradiction", conflict_point="api.port",
        suggested_winner=left_id, confidence_hint="low",
        source="llm_informed", refresh=True,
        left_version=1, right_version=1,
    )
    after = tools.memory_list_conflicts()["data"]["conflicts"][0]
    assert after["active_judgment_id"] == before["active_judgment_id"]
    assert after["judgment_status"] == "llm_assessed"
    assert after["suggested_winner"] == right_id
    assert after["reason"] == "persisted host judgment"
    assert after["confidence_hint"] == "high"
    assert after["scan_detected_at"] is not None


def test_scan_first_is_upgraded_to_structured_snapshot(tmp_path: Path) -> None:
    tools = _tools(tmp_path, mode="off")
    left_id = _write_claim(tools, "5432")["data"]["id"]
    right_id = _write_claim(tools, "3306")["data"]["id"]
    scanned = tools.memory_record_conflict(
        left_id=left_id, right_id=right_id, reason="scan found mismatch",
        conflict_type="contradiction", source="llm_informed",
        left_version=1, right_version=1,
    )
    conflict_id = scanned["data"]["conflict_id"]

    tools.settings.structured_claim_mode = "beta_all"
    rebuilt = tools.memory_rebuild_claims(
        memory_ids=[left_id, right_id], dry_run=False,
    )
    assert rebuilt["ok"] is True
    row = tools.memory_list_conflicts()["data"]["conflicts"][0]
    assert row["id"] == conflict_id
    assert row["left_claim_revision"] == row["right_claim_revision"] == 1
    assert row["judgment_status"] == "pending_llm"
    assert row["structured_detected_at"] is not None
    assert row["scan_detected_at"] is not None


def test_non_structured_conflict_cannot_enter_judgment_state_machine(tmp_path: Path) -> None:
    tools = _tools(tmp_path, mode="off")
    left_id = _write_claim(tools, "5432")["data"]["id"]
    right_id = _write_claim(tools, "3306")["data"]["id"]
    conflict_id = tools.memory_record_conflict(
        left_id=left_id, right_id=right_id, reason="scan-only mismatch",
        conflict_type="contradiction", source="llm_informed",
        left_version=1, right_version=1,
    )["data"]["conflict_id"]

    rejected = tools.memory_submit_conflict_judgment(
        conflict_id=conflict_id, expected_left_version=1,
        expected_right_version=1, expected_left_claim_revision=1,
        expected_right_claim_revision=1, verdict="contradiction",
        recommended_use="right", suggested_winner=right_id,
        confidence_hint="high", reason="must not accept missing claim pins",
        affects_current_output=True, usage_context="answer",
    )
    assert rejected["ok"] is False
    assert rejected["data"]["outcome"] == "invalid_structured_snapshot"


def test_inactive_write_does_not_publish_or_collide(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    pending = tools.memory_write(
        content="port: 3306", subject="cfg", metadata={"entity": "api"},
        event_time="2026-01-01T00:00:00Z", status="pending",
    )
    assert not pending["data"].get("attention_required")
    assert pending["data"]["realtime_conflict_check"]["skipped_reason"] == "inactive"
    assert tools.db.list_memory_claims(pending["data"]["id"]) == []


def test_structured_claim_mode_config_and_invalid_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("MEMORY_ARBITER_STRUCTURED_CLAIM_MODE", "off")
    assert Settings.from_env().structured_claim_mode == "off"
    monkeypatch.setenv("MEMORY_ARBITER_STRUCTURED_CLAIM_MODE", "surprise")
    settings = Settings.from_env()
    assert settings.structured_claim_mode == "beta_all"
    assert any("structured_claim_mode" in warning for warning in settings.config_warnings)


def test_doctor_reports_claim_state(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write_claim(tools, "5432")
    _write_claim(tools, "3306")
    report = tools.memory_doctor_overview()["data"]
    finding = next(f for f in report["findings"] if f["check_id"] == "consistency.structured_claims")
    assert finding["evidence"]["claims"] == 2
    assert finding["evidence"]["stale_memories"] == 0
    assert finding["evidence"]["reconciled_memories"] == 2
    assert finding["evidence"]["structured_latency_ms"]["count"] == 2
    assert finding["evidence"]["candidate_peer_count"]["max"] == 1.0
    assert finding["evidence"]["detection_channels"] == {
        "structured_only": 1, "scan_only": 0, "both": 0,
    }
    assert finding["evidence"]["pending"]["pending_llm"]["count"] == 1
    assert finding["evidence"]["outcomes_by_rule"]["p1_kv"]["pending_llm"] == 1
    assert finding["evidence"]["outcomes_by_attribute"]["port"]["pending_llm"] == 1
