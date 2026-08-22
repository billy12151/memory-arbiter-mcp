"""Internal workspace redirect and negative-decision state.

Pairwise alias actions are intentionally absent from the product surface. Tests
cover the behaviors that remain load-bearing: deterministic resolver redirects,
negative candidate suppression, rename/migrate forwarding, strict pending
confirmation, collision safety, compact persistence, and rollback.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import threading

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB, _normalize_alias_key
from memory_arbiter.models import MemoryStatus
from memory_arbiter.tools import MemoryTools


def make_tools(
    tmp_path: Path, isolation: str = "weak", *, vec: bool = False,
) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "gov.sqlite3",
        backup_jsonl=tmp_path / "gov.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        enable_sqlite_vec=vec, vec_dim=2, isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def decide(
    tools: MemoryTools, workspace: str, canonical: str,
    *, status: str = "confirmed", force: bool = False,
) -> None:
    ok, warnings = tools.db.record_workspace_decision(
        workspace, canonical, status=status, force=force,
    )
    assert ok, warnings


def write(tools: MemoryTools, workspace: str, content: str = "workspace fact") -> int:
    return int(tools.memory_write(
        content=content, subject="workspace", workspace=workspace,
        source_type="agent_generated",
    )["data"]["id"])


def test_normalize_workspace_decision_key() -> None:
    assert _normalize_alias_key("  金营项目 ") == _normalize_alias_key("金营项目")
    assert _normalize_alias_key("Project  X") == _normalize_alias_key("project x")
    assert _normalize_alias_key("") == ""
    assert _normalize_alias_key(None) == ""


def test_compact_schema_has_no_event_ledger(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    with tools.db.connection() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
        events = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workspace_alias_events'"
        ).fetchone()
    assert columns == ["alias_workspace", "canonical", "status", "updated_at"]
    assert events is None


def test_confirmed_redirect_short_circuits_resolver(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    decide(tools, "金营二期", "金营项目")
    resolved = tools.db.resolve_workspace_canonical(" 金营二期 ", None, register_new=False)
    assert resolved["canonical"] == "金营项目"
    assert resolved["matched_by"] == "confirmed_alias"
    state = tools.db.get_workspace_decision("金营二期")
    assert state["status"] == "confirmed"


def test_negative_decisions_accumulate_and_suppress_candidates(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    decide(tools, "raw", "candidate-a", status="rejected")
    decide(tools, "raw", "candidate-b", status="rejected")
    resolved = tools.db.resolve_workspace_canonical("raw", None, register_new=False)
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
    tools = make_tools(tmp_path)
    decide(tools, "raw", "target", status="rejected")
    ok, warnings = tools.db.record_workspace_decision("raw", "target")
    assert ok is False and "kept separate" in warnings[0]
    decide(tools, "raw", "target", force=True)
    assert tools.db.get_workspace_decision("raw")["status"] == "confirmed"


def test_one_confirmed_redirect_preserves_unrelated_negatives(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
    memory_id = write(tools, "OldName")
    result = tools.memory_govern("rename_workspace_canonical", {
        "old": "OldName", "new": "NewName", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "NewName"
    resolved = tools.db.resolve_workspace_canonical("OldName", None, register_new=False)
    assert resolved["canonical"] == "NewName"
    assert resolved["matched_by"] == "confirmed_alias"


def test_migrate_moves_memories_and_prevents_source_resplit(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "Sub2")
    result = tools.memory_govern("migrate_workspace", {
        "from": "Sub2", "to": "Main", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "Main"
    resolved = tools.db.resolve_workspace_canonical("Sub2", None, register_new=False)
    assert resolved["canonical"] == "Main"
    assert resolved["matched_by"] == "confirmed_alias"


def test_migrate_without_existing_rows_installs_redirect(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_govern("migrate_workspace", {
        "from": "mema", "to": "memory-arbiter-mcp", "authorized": True,
    })
    assert result["ok"] is True
    assert result["data"]["memories_updated"] == 0
    resolved = tools.db.resolve_workspace_canonical("mema", None, register_new=False)
    assert resolved["canonical"] == "memory-arbiter-mcp"


def test_exact_negative_blocks_rename_forwarding(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "Old", "New", status="rejected")
    result = tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "New", "authorized": True,
    })
    assert result["ok"] is True
    resolved = tools.db.resolve_workspace_canonical("Old", None, register_new=False)
    assert resolved["matched_by"] != "confirmed_alias"
    assert "New" in resolved["rejected_canonicals"]


def test_unrelated_negative_does_not_block_rename_forwarding(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "Old", "Other", status="rejected")
    tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "New", "authorized": True,
    })
    resolved = tools.db.resolve_workspace_canonical("Old", None, register_new=False)
    assert resolved["canonical"] == "New"
    with tools.db.connection() as conn:
        rejected = conn.execute(
            "SELECT canonical FROM workspace_aliases "
            "WHERE alias_workspace='old' AND status='rejected'"
        ).fetchall()
    assert [row["canonical"] for row in rejected] == ["Other"]


def test_repoint_is_collision_safe_and_preserves_existing_decision(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "foo", "Old", status="rejected")
    decide(tools, "foo", "New", status="rejected")
    result = tools.memory_govern("rename_workspace_canonical", {
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
    tools = make_tools(tmp_path)
    memory_id = write(tools, "ProjectX")
    result = tools.memory_govern("rename_workspace_canonical", {
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
    tools = make_tools(tmp_path, isolation="strict")
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
    tools = make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "BrandNew")
    result = tools.memory_govern("confirm_pending_workspace", {
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
    tools = make_tools(tmp_path, isolation="strict")
    written = tools.memory_write(
        content="global pending", subject="global", workspace="default",
        source_type="agent_generated", status="pending",
    )
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": written["data"]["id"], "canonical": "ProjectX", "authorized": True,
    })
    assert result["ok"] is False
    assert "reserved default" in result["data"]["error"]
    assert tools.db.get_memory(written["data"]["id"])["status"] == MemoryStatus.PENDING.value


def test_competing_move_does_not_split_memory_and_redirect(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "Old")
    first = tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "A", "authorized": True,
    })
    second = tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "B", "authorized": True,
    })
    assert first["ok"] is True
    assert second["ok"] is False
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "A"
    assert tools.db.resolve_workspace_canonical("Old", None, register_new=False)["canonical"] == "A"


def test_confirm_pending_exact_name_activates_without_self_redirect(tmp_path: Path) -> None:
    tools = make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "BrandNew")
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.PENDING.value
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "BrandNew", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.ACTIVE.value
    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases WHERE alias_workspace='brandnew'"
        ).fetchone()[0] == 0


def test_confirm_pending_different_name_records_redirect_atomically(tmp_path: Path) -> None:
    tools = make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "abbrev")
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "CanonicalProject", "authorized": True,
    })
    assert result["ok"] is True
    record = tools.db.get_memory(memory_id)
    assert record["status"] == MemoryStatus.ACTIVE.value
    assert record["workspace_canonical"] == "CanonicalProject"
    assert tools.db.resolve_workspace_canonical("abbrev", None, register_new=False)["canonical"] == "CanonicalProject"


def test_confirm_pending_rolls_back_decision_assignment_and_activation(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "abbrev")
    original = tools.db.set_memory_workspace_canonical_on_conn

    def fail(*args, **kwargs):
        return False, ["injected failure"]

    monkeypatch.setattr(tools.db, "set_memory_workspace_canonical_on_conn", fail)
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "CanonicalProject", "authorized": True,
    })
    assert result["ok"] is False
    monkeypatch.setattr(tools.db, "set_memory_workspace_canonical_on_conn", original)
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.PENDING.value
    assert tools.db.get_workspace_decision("abbrev") is None


def test_confirm_pending_error_does_not_leak_foreign_record(tmp_path: Path) -> None:
    tools = make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "Other", "TOP SECRET")
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "Caller",
        "workspace": "Caller", "authorized": True,
    })
    assert result["ok"] is False
    assert result["data"]["record"] is None
    assert "TOP SECRET" not in str(result)


def test_default_pool_cannot_enter_internal_decision_state(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for left, right in (("default", "project"), ("project", "默认")):
        ok, warnings = tools.db.record_workspace_decision(left, right)
        assert ok is False
        assert "reserved global pool" in warnings[0]


def test_concurrent_confirmed_decisions_leave_one_redirect(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path, vec=True)
    if not tools.db.state.sqlite_vec_available:
        pytest.skip("sqlite-vec unavailable")

    class Embedder:
        def embed_text(self, prefix="", body=""):
            return SimpleNamespace(embedding=[1.0, 0.0])

    embedder = Embedder()
    tools.db.resolve_workspace_canonical("Target", embedder, register_new=True)
    decide(tools, "raw", "Target", status="rejected")
    resolved = tools.db.resolve_workspace_canonical("raw", embedder, register_new=False)
    assert resolved["canonical"] != "Target"
    assert "Target" not in [item["name"] for item in resolved["similar"]]


def test_workspace_decision_schema_normalization_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    tools = make_tools(tmp_path)
    decide(tools, "raw", "candidate-a", status="rejected")
    del tools
    reopened = MemoryDB(Settings(
        db_path=path if path.exists() else tmp_path / "gov.sqlite3",
        backup_jsonl=tmp_path / "other.jsonl",
    ))
    with reopened.connection() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
    assert columns == ["alias_workspace", "canonical", "status", "updated_at"]
