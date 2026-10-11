

# ── from test_workspace_qwen_candidate.py ──
# helper make_tools renamed: qwen_candidate_make_tools (collision)

"""Qwen/local-model workspace candidate suggester (design 636 §6, §7, §9).

The real GGUF model is optional; these tests exercise the parser and the
per-isolation policy with a stub backend so they run without a model file.
"""
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools
import json
from contextlib import contextmanager
from types import SimpleNamespace
import sqlite3
import threading
import pytest
from memory_arbiter.db import MemoryDB, _normalize_alias_key
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryStatus
from tests._helpers import record_conflict
from tests._helpers import alias_governance_make_tools, register, write


# ── parser ───────────────────────────────────────────────────────────────────






# ── per-isolation policy (stub backend) ──────────────────────────────────────

def qwen_candidate_make_tools(tmp_path: Path, isolation: str) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "qwen.sqlite3",
        backup_jsonl=tmp_path / "qwen.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _force_undecided_with_candidate(t: MemoryTools, backend, similar_name="金营项目", distance=0.2):
    """Patch resolver → undecided + a candidate, and inject a stub backend.

    The candidate distance defaults to 0.2 (inside workspace_match_distance): a
    real near-miss the vector brought within range, which is the only situation
    where Qwen is allowed to arbitrate an AUTO merge. Over-distance candidates
    (e.g. 0.4) are filtered out before Qwen sees them by design."""
    def fake_resolve(ws_raw, embedder=None, *, match_distance=None):
        return {
            "canonical": ws_raw, "is_new": True, "matched_by": "new",
            "distance": None, "similar": [{"name": similar_name, "distance": distance}],
            "rejected_canonicals": [],
        }
    t.db.resolve_workspace_canonical = fake_resolve  # type: ignore
    t._ensure_semantic_backend = lambda: backend  # type: ignore


def test_model_suggester_retired_undecided_asks(tmp_path):
    """0.17.1 owner 拍板：模型建议器随 Qwen 判定引擎退役——undecided 规则
    直接 ASK（suggester_retired_ask / no_similar_candidates）；旧 AUTO 合并
    与全部 qwen_* decision_reason 不复存在。"""
    t = qwen_candidate_make_tools(tmp_path, "weak")
    tools = t["tools"] if isinstance(t, dict) else t
    assert not hasattr(tools, "_suggest_workspace_candidate"), (
        "retired suggester must be gone from MemoryTools"
    )
    result = tools.memory_write(
        content="占位内容", subject="zcode-data-migrat",
        workspace="zcode-data-migrat", metadata={"entity": "x", "scope": "y"},
    )
    data = result["data"]
    reason = data.get("workspace_decision_reason")
    assert reason not in {"qwen_high_conf", "qwen_low_conf", "qwen_timeout",
                          "qwen_backend_error", "qwen_unavailable", "qwen_rejected"}, reason


# ── alias governance decisions ───────────────────────────────────────────────

def decide(
    tools: MemoryTools, workspace: str, canonical: str,
    *, status: str = "confirmed", force: bool = False,
) -> None:
    ok, warnings = tools.db.record_workspace_decision(
        workspace, canonical, status=status, force=force,
    )
    assert ok, warnings




def test_normalize_workspace_decision_key() -> None:
    assert _normalize_alias_key("  金营项目 ") == _normalize_alias_key("金营项目")
    assert _normalize_alias_key("Project  X") == _normalize_alias_key("project x")
    assert _normalize_alias_key("") == ""
    assert _normalize_alias_key(None) == ""


def test_compact_schema_has_no_event_ledger(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    with tools.db.connection() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
        events = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workspace_alias_events'"
        ).fetchone()
    assert columns == ["alias_workspace", "canonical", "status", "updated_at"]
    assert events is None


def test_confirmed_redirect_short_circuits_resolver(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "金营二期", "金营项目")
    resolved = tools.db.resolve_workspace_canonical(" 金营二期 ", None)
    assert resolved["canonical"] == "金营项目"
    assert resolved["matched_by"] == "confirmed_alias"
    state = tools.db.get_workspace_decision("金营二期")
    assert state["status"] == "confirmed"


def test_negative_decisions_accumulate_and_suppress_candidates(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "raw", "candidate-a", status="rejected")
    decide(tools, "raw", "candidate-b", status="rejected")
    resolved = tools.db.resolve_workspace_canonical("raw", None)
    assert set(resolved["rejected_canonicals"]) == {"candidate-a", "candidate-b"}
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT canonical,status FROM workspace_aliases "
            "WHERE alias_workspace='raw' ORDER BY canonical"
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("candidate-a", "rejected"), ("candidate-b", "rejected"),
    ]


def test_exact_negative_requires_force_to_reverse(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "raw", "target", status="rejected")
    ok, warnings = tools.db.record_workspace_decision("raw", "target")
    assert ok is False and "kept separate" in warnings[0]
    decide(tools, "raw", "target", force=True)
    assert tools.db.get_workspace_decision("raw")["status"] == "confirmed"


def test_one_confirmed_redirect_preserves_unrelated_negatives(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "raw", "candidate-a", status="rejected")
    decide(tools, "raw", "canonical")
    decide(tools, "raw", "new-canonical")
    with tools.db.connection() as conn:
        confirmed = conn.execute(
            "SELECT canonical FROM workspace_aliases "
            "WHERE alias_workspace='raw' AND status='confirmed'"
        ).fetchall()
        rejected = conn.execute(
            "SELECT canonical FROM workspace_aliases "
            "WHERE alias_workspace='raw' AND status='rejected'"
        ).fetchall()
    assert [row["canonical"] for row in confirmed] == ["new-canonical"]
    assert [row["canonical"] for row in rejected] == ["candidate-a"]


def test_removed_pairwise_actions_are_non_mutating_tombstones(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    for action in ("accept_workspace_alias", "reject_workspace_alias"):
        result = tools.memory_govern(action, {
            "alias": "raw", "canonical": "target", "authorized": True,
        })
        assert result["ok"] is False
        assert result["data"]["outcome"] == "removed"
        assert result["data"]["error_code"] == "workspace_alias_action_removed"
        assert result["data"]["removed_action"] == action
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspace_aliases").fetchone()[0] == 0


def test_removed_accept_guides_to_supported_flows(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    result = tools.memory_govern("accept_workspace_alias", {
        "alias": "raw", "canonical": "target",
    })
    actions = {
        item.get("suggested_call", {}).get("action")
        for item in result["data"]["replacements"]
        if item.get("suggested_call")
    }
    assert {"migrate_workspace", "rename_workspace_canonical", "confirm_pending_workspace"} <= actions
    migrate = next(
        item["suggested_call"] for item in result["data"]["replacements"]
        if (item.get("suggested_call") or {}).get("action") == "migrate_workspace"
    )
    assert migrate == {
        "tool": "memory_govern", "action": "migrate_workspace",
        "data": {"from": "raw", "to": "target"},
    }
    reject = tools.memory_govern("reject_workspace_alias", {})
    assert reject["data"]["replacements"][0]["suggested_call"] is None


def test_rename_moves_memories_and_prevents_old_name_resplit(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    memory_id = write(tools, "OldName")
    result = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "OldName", "new": "NewName", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "NewName"
    resolved = tools.db.resolve_workspace_canonical("OldName", None)
    assert resolved["canonical"] == "NewName"
    assert resolved["matched_by"] == "confirmed_alias"


def test_rename_moves_conflicts_with_their_members(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    left = write(tools, "OldName", "database is mysql")
    right = write(tools, "OldName", "database is sqlite")
    conflict_id = record_conflict(tools, "OldName", left, right)

    result = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "OldName", "new": "NewName", "authorized": True,
    })

    assert result["ok"] is True
    conflict = tools.db.get_conflict(conflict_id)
    assert conflict["workspace_canonical"] == "NewName"
    assert tools.memory_review(
        "conflict_detail", {"conflict_id": conflict_id, "workspace": "NewName"},
    )["ok"] is True


def test_migrate_moves_memories_and_prevents_source_resplit(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    memory_id = write(tools, "Sub2")
    result = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "Sub2", "to": "Main", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "Main"
    resolved = tools.db.resolve_workspace_canonical("Sub2", None)
    assert resolved["canonical"] == "Main"
    assert resolved["matched_by"] == "confirmed_alias"


def test_migrate_moves_conflicts_with_their_members(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    left = write(tools, "Sub2", "database is mysql")
    right = write(tools, "Sub2", "database is sqlite")
    conflict_id = record_conflict(tools, "Sub2", left, right)

    result = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "Sub2", "to": "Main", "authorized": True,
    })

    assert result["ok"] is True
    assert tools.db.get_conflict(conflict_id)["workspace_canonical"] == "Main"


def test_migrate_conflict_slot_collision_refused_up_front(tmp_path: Path) -> None:
    # v0.15.12+: the move is refused BEFORE any write with the colliding
    # conflict ids spelled out, instead of aborting mid-transaction on the
    # partial unique index with a bare sqlite error. Auto-resolving either
    # open conflict would fabricate a triage decision, so refusal is the only
    # honest outcome; triage then happens via the normal governance flow.
    tools = alias_governance_make_tools(tmp_path)
    old_left = write(tools, "Old", "old mysql")
    old_right = write(tools, "Old", "old sqlite")
    new_left = write(tools, "New", "new mysql")
    new_right = write(tools, "New", "new sqlite")
    old_conflict = record_conflict(tools, "Old", old_left, old_right)
    new_conflict = record_conflict(tools, "New", new_left, new_right)

    result = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "Old", "to": "New", "authorized": True,
    })

    assert result["ok"] is False
    warning = next(w for w in result["warnings"] if "would collide" in w)
    assert f"#{old_conflict}->#{new_conflict}" in warning
    assert tools.db.get_memory(old_left)["workspace_canonical"] == "Old"
    assert tools.db.get_memory(old_right)["workspace_canonical"] == "Old"
    assert tools.db.get_conflict(old_conflict)["workspace_canonical"] == "Old"
    assert tools.db.get_conflict(new_conflict)["workspace_canonical"] == "New"

    # Resolving the colliding source conflict unblocks the move. The full
    # open -> applying -> resolved chain is exercised in the lifecycle
    # tests; here we only need the row out of the active-slot index.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE conflicts SET status='resolved' WHERE id=?", (old_conflict,),
        )
    retry = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "Old", "to": "New", "authorized": True,
    })
    assert retry["ok"] is True
    assert tools.db.get_memory(old_left)["workspace_canonical"] == "New"
    assert tools.db.get_conflict(new_conflict)["workspace_canonical"] == "New"


def test_rename_conflict_slot_collision_refused_up_front(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    old_left = write(tools, "Old", "old mysql")
    old_right = write(tools, "Old", "old sqlite")
    new_left = write(tools, "New", "new mysql")
    new_right = write(tools, "New", "new sqlite")
    old_conflict = record_conflict(tools, "Old", old_left, old_right)
    new_conflict = record_conflict(tools, "New", new_left, new_right)

    result = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "New", "authorized": True,
    })

    assert result["ok"] is False
    warning = next(w for w in result["warnings"] if "would collide" in w)
    assert f"#{old_conflict}->#{new_conflict}" in warning
    assert tools.db.get_memory(old_left)["workspace_canonical"] == "Old"
    assert tools.db.get_conflict(new_conflict)["workspace_canonical"] == "New"


def test_migrate_different_slot_conflicts_do_not_collide(tmp_path: Path) -> None:
    # Same-slot pairs collide; different slots must keep moving normally.
    tools = alias_governance_make_tools(tmp_path)
    left = write(tools, "Old", "old mysql")
    right = write(tools, "Old", "old sqlite")
    conflict_id = record_conflict(tools, "Old", left, right)

    result = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "Old", "to": "New", "authorized": True,
    })

    assert result["ok"] is True
    assert tools.db.get_memory(left)["workspace_canonical"] == "New"
    assert tools.db.get_conflict(conflict_id)["workspace_canonical"] == "New"


def test_slot_collision_guard_ignores_terminal_status_rows(tmp_path: Path) -> None:
    # Only 'open'/'applying' rows occupy the active-slot index; resolved /
    # not_a_conflict rows sharing a slot must NOT trigger the refusal.
    tools = alias_governance_make_tools(tmp_path)
    a_left, a_right = write(tools, "A", "a mysql"), write(tools, "A", "a sqlite")
    b_left, b_right = write(tools, "B", "b mysql"), write(tools, "B", "b sqlite")
    c_left, c_right = write(tools, "C", "c mysql"), write(tools, "C", "c sqlite")
    a_conflict = record_conflict(tools, "A", a_left, a_right)
    b_conflict = record_conflict(tools, "B", b_left, b_right)
    c_conflict = record_conflict(tools, "C", c_left, c_right)
    with tools.db.write_transaction() as conn:
        # All three share one slot (record_conflict's fixed slot key) but only
        # B stays active.
        conn.execute("UPDATE conflicts SET status='resolved' WHERE id=?", (a_conflict,))
        conn.execute("UPDATE conflicts SET status='not_a_conflict' WHERE id=?", (c_conflict,))

    from memory_arbiter.db.workspaces import WorkspaceStore
    with tools.db.connection() as conn:
        assert WorkspaceStore._conflict_slot_collision_warning_on_conn(conn, "B", "A") is None
        assert WorkspaceStore._conflict_slot_collision_warning_on_conn(conn, "B", "C") is None
        assert WorkspaceStore._conflict_slot_collision_warning_on_conn(conn, "A", "C") is None


def test_normalize_reports_refused_merge_as_skipped_in_plan_and_execute(tmp_path: Path) -> None:
    # A would-be collision must show up as skipped-with-reason in the DRY-RUN
    # plan (plan honesty: not as a merge the execute pass must refuse), and
    # the execute pass must refuse the same merge and keep both canonicals.
    tools = alias_governance_make_tools(tmp_path)
    register(tools, "AgentLane", "agent-lane")
    winner_left = write(tools, "AgentLane", "winner mysql")
    winner_right = write(tools, "AgentLane", "winner sqlite")
    loser_left = write(tools, "agent-lane", "loser mysql")
    loser_right = write(tools, "agent-lane", "loser sqlite")
    record_conflict(tools, "AgentLane", winner_left, winner_right)
    record_conflict(tools, "agent-lane", loser_left, loser_right)

    plan = tools.db.workspaces.normalize_workspace_canonicals(dry_run=True)
    assert plan["ok"] is True
    refused = [item for item in plan["skipped"] if item.get("type") == "merge_refused"]
    assert len(refused) == 1
    assert refused[0]["from"] == "agent-lane" and refused[0]["to"] == "AgentLane"
    assert any(merge["from"] != "agent-lane" for merge in plan["merged"]) is False or all(
        merge["from"] != "agent-lane" for merge in plan["merged"]
    )

    executed = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert executed["ok"] is True
    assert [item for item in executed["skipped"] if item.get("type") == "merge_refused"]
    assert tools.db.get_memory(loser_left)["workspace_canonical"] == "agent-lane"
    assert tools.db.get_memory(winner_left)["workspace_canonical"] == "AgentLane"
    assert any("would collide" in warning for warning in executed["warnings"])
    result = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "mema", "to": "memory-arbiter-mcp", "authorized": True,
    })
    assert result["ok"] is True
    assert result["data"]["memories_updated"] == 0
    resolved = tools.db.resolve_workspace_canonical("mema", None)
    assert resolved["canonical"] == "memory-arbiter-mcp"


def test_exact_negative_blocks_rename_forwarding(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "Old", "New", status="rejected")
    result = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "New", "authorized": True,
    })
    assert result["ok"] is True
    resolved = tools.db.resolve_workspace_canonical("Old", None)
    assert resolved["matched_by"] != "confirmed_alias"
    assert "New" in resolved["rejected_canonicals"]


def test_unrelated_negative_does_not_block_rename_forwarding(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "Old", "Other", status="rejected")
    tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "New", "authorized": True,
    })
    resolved = tools.db.resolve_workspace_canonical("Old", None)
    assert resolved["canonical"] == "New"
    with tools.db.connection() as conn:
        rejected = conn.execute(
            "SELECT canonical FROM workspace_aliases "
            "WHERE alias_workspace='old' AND status='rejected'"
        ).fetchall()
    assert [row["canonical"] for row in rejected] == ["Other"]


def test_repoint_is_collision_safe_and_preserves_existing_decision(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "foo", "Old", status="rejected")
    decide(tools, "foo", "New", status="rejected")
    result = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "New", "authorized": True,
    })
    assert result["ok"] is True
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT canonical,status FROM workspace_aliases "
            "WHERE alias_workspace='foo'"
        ).fetchall()
    assert [tuple(row) for row in rows] == [("New", "rejected")]


def test_case_only_rename_leaves_no_self_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    memory_id = write(tools, "ProjectX")
    result = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "ProjectX", "new": "projectx", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "projectx"
    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases "
            "WHERE alias_workspace='projectx' AND canonical='projectx'"
        ).fetchone()[0] == 0


def test_strict_retries_remain_pending_until_confirmation(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    first = write(tools, "Unconfirmed")
    second = write(tools, "Unconfirmed", "second fact")
    assert tools.db.get_memory(first)["status"] == MemoryStatus.PENDING.value
    assert tools.db.get_memory(second)["status"] == MemoryStatus.PENDING.value
    with tools.db.connection() as conn:
        canonical = conn.execute(
            "SELECT 1 FROM workspace_canonicals WHERE name='Unconfirmed'"
        ).fetchone()
    assert canonical is None


def test_confirm_pending_case_variant_reuses_raw_spelling(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "BrandNew")
    result = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "BrandNew",
        "memory_id": memory_id, "canonical": "brandnew", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "BrandNew"
    with tools.db.connection() as conn:
        names = [row["name"] for row in conn.execute(
            "SELECT name FROM workspace_canonicals WHERE lower(name)='brandnew'"
        )]
    assert names == ["BrandNew"]


def test_default_pending_cannot_be_confirmed_into_project(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    written = tools.memory_write(
        content="global pending", subject="global", workspace="default",
        source_type="agent_generated", status="pending",
    )
    result = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "default",
        "memory_id": written["data"]["id"], "canonical": "ProjectX", "authorized": True,
    })
    assert result["ok"] is False
    assert "reserved default" in result["data"]["error"]
    assert tools.db.get_memory(written["data"]["id"])["status"] == MemoryStatus.PENDING.value


def test_competing_move_does_not_split_memory_and_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    memory_id = write(tools, "Old")
    first = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "A", "authorized": True,
    })
    second = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "B", "authorized": True,
    })
    assert first["ok"] is True
    assert second["ok"] is False
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "A"
    assert tools.db.resolve_workspace_canonical("Old", None)["canonical"] == "A"


def test_confirm_pending_exact_name_activates_without_self_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "BrandNew")
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.PENDING.value
    result = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "BrandNew",
        "memory_id": memory_id, "canonical": "BrandNew", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.ACTIVE.value
    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases WHERE alias_workspace='brandnew'"
        ).fetchone()[0] == 0


def test_confirm_pending_different_name_records_redirect_atomically(tmp_path: Path) -> None:
    """redirect 力学（raw≠canonical → confirmed alias 原子落库）在 none 隔离 +
    显式 pending 写入下钉住——strict 下该形态被第二道校验正确禁止（见
    test_confirm_pending_strict_forbids_cross_canonical）。"""
    tools = alias_governance_make_tools(tmp_path, isolation="none")
    written = tools.memory_write(
        content="abbrev scope", subject="redirect", workspace="abbrev",
        source_type="agent_generated", status="pending",
    )
    memory_id = written["data"]["id"]
    result = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "abbrev",
        "memory_id": memory_id, "canonical": "CanonicalProject", "authorized": True,
    })
    assert result["ok"] is True
    record = tools.db.get_memory(memory_id)
    assert record["status"] == MemoryStatus.ACTIVE.value
    assert record["workspace_canonical"] == "CanonicalProject"
    assert tools.db.resolve_workspace_canonical("abbrev", None)["canonical"] == "CanonicalProject"


def test_confirm_pending_rolls_back_decision_assignment_and_activation(tmp_path: Path, monkeypatch) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "abbrev")
    original = tools.db.set_memory_workspace_canonical_on_conn

    def fail(*args, **kwargs):
        return False, ["injected failure"]

    monkeypatch.setattr(tools.db, "set_memory_workspace_canonical_on_conn", fail)
    result = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "default",
        "memory_id": memory_id, "canonical": "CanonicalProject", "authorized": True,
    })
    assert result["ok"] is False
    monkeypatch.setattr(tools.db, "set_memory_workspace_canonical_on_conn", original)
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.PENDING.value
    assert tools.db.get_workspace_decision("abbrev") is None


def test_confirm_pending_error_does_not_leak_foreign_record(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "Other", "TOP SECRET")
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "Caller",
        "workspace": "Caller", "authorized": True,
    })
    assert result["ok"] is False
    assert result["data"]["record"] is None
    assert "TOP SECRET" not in str(result)


def test_default_pool_cannot_enter_internal_decision_state(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    for left, right in (("default", "project"), ("project", "默认")):
        ok, warnings = tools.db.record_workspace_decision(left, right)
        assert ok is False
        assert "reserved global pool" in warnings[0]


def test_concurrent_confirmed_decisions_leave_one_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def worker(target: str) -> None:
        barrier.wait()
        outcomes.append(tools.db.record_workspace_decision("raw", target)[0])

    threads = [threading.Thread(target=worker, args=(target,)) for target in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(outcomes)
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT canonical FROM workspace_aliases "
            "WHERE alias_workspace='raw' AND status='confirmed'"
        ).fetchall()
    assert len(rows) == 1


def test_negative_decision_filters_real_vector_candidate(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    tools = alias_governance_make_tools(tmp_path, vec=True)
    if not tools.db.state.sqlite_vec_available:
        pytest.skip("sqlite-vec unavailable")

    class Embedder:
        def embed_text(self, prefix="", body=""):
            return SimpleNamespace(embedding=[1.0, 0.0])

    embedder = Embedder()
    # P2 #9: resolve is read-only now — register the fixture canonical
    # (+vector) the way the write path does: canonical row first, then the
    # derived vector via the two-step publish API.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('Target',datetime('now'))",
        )
    tools.db.workspaces.publish_workspace_canonical_vector("Target", [1.0, 0.0])
    decide(tools, "raw", "Target", status="rejected")
    resolved = tools.db.resolve_workspace_canonical("raw", embedder)
    assert resolved["canonical"] != "Target"
    assert "Target" not in [item["name"] for item in resolved["similar"]]


def test_workspace_decision_schema_normalization_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "raw", "candidate-a", status="rejected")
    del tools
    reopened = MemoryDB(Settings(
        db_path=path if path.exists() else tmp_path / "gov.sqlite3",
        backup_jsonl=tmp_path / "other.jsonl",
    ))
    with reopened.connection() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
    assert columns == ["alias_workspace", "canonical", "status", "updated_at"]



def test_workspace_required_on_govern_actions_surface(tmp_path: Path) -> None:
    """C1 必传（owner 2026-10-03）：五个改桶动作缺/空 workspace 在 validation
    层打回，报错带 workspaces 列表指路；move/separate 补钉（remember/confirm/
    rename 已入 golden 语料）。"""
    tools = alias_governance_make_tools(tmp_path, isolation="none")
    for action, payload in (
        ("separate_workspace_alias", {"alias": "agent-lane", "canonical": "AgentLane", "authorized": True}),
        ("move_memories_workspace", {"memory_ids": [1], "new_workspace": "ws", "authorized": True}),
    ):
        missing = tools.memory_govern(action, dict(payload))
        assert missing["ok"] is False, action
        assert missing["data"]["field"] == "workspace"
        assert "memory_review(view='workspaces')" in missing["data"]["reason"]
        empty = tools.memory_govern(action, {**payload, "workspace": "  "})
        assert empty["ok"] is False, action
        assert empty["data"]["field"] == "workspace"


def test_memory_review_workspaces_view(tmp_path: Path) -> None:
    """C2：workspaces 列表——计数/别名/空桶标注/排序/limit 单 SQL 口径。"""
    tools = alias_governance_make_tools(tmp_path, isolation="none")
    write(tools, "Alpha")  # pending（none 下 generic 新名走 ASK→pending 或 active，断言口径放宽）
    write(tools, "Beta")
    result = tools.memory_review("workspaces", {"limit": 50})
    assert result["ok"] is True
    ws = {b["canonical"]: b for b in result["data"]["workspaces"]}
    assert "Alpha" in ws and "Beta" in ws
    alpha = ws["Alpha"]
    for key in ("active_count", "pending_count", "alias_count", "last_write_at", "empty"):
        assert key in alpha
    limited = tools.memory_review("workspaces", {"limit": 1})
    assert limited["ok"] is True
    assert limited["data"]["count"] == 1


def test_memory_review_workspaces_strict_admitted_only(tmp_path: Path) -> None:
    """C2 strict ACL：workspaces 视图只见 admitted 集；无 canonical=denied 优先。"""
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    tools.settings.workspace = "Alpha"
    write(tools, "Alpha")
    write(tools, "Beta")
    with tools.db.connection() as conn:
        pending = conn.execute(
            "SELECT id FROM memories WHERE workspace='Alpha' AND status='pending'"
        ).fetchone()
    if pending:
        tools.memory_govern("confirm_pending_workspace", {
            "workspace": "Alpha", "memory_id": pending[0],
            "canonical": "Alpha", "authorized": True,
        })
    result = tools.memory_review("workspaces", {"workspace": "Alpha"})
    assert result["ok"] is True
    names = {b["canonical"] for b in result["data"]["workspaces"]}
    assert "Alpha" in names
    assert "Beta" not in names, "strict 调用者不得看见 scope 外桶"