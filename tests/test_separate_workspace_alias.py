"""mema #795: separate_workspace_alias — the MCP surface for undoing an
installed alias redirect / recording keep-separate. Naming is an owner
decision (2026-09-02): "separate", not "reject" (reject reads like refusing a
memory write). Reversing a recorded separation later requires force=true on
the confirm side.
"""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.surfaces import ProductSurfaces
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    return MemoryTools(settings, MemoryDB(settings))


def _alias_rows(tools: MemoryTools) -> list[tuple[str, str, str]]:
    with tools.db.connection() as conn:
        return [
            (str(r["alias_workspace"]), str(r["canonical"]), str(r["status"]))
            for r in conn.execute(
                "SELECT alias_workspace,canonical,status FROM workspace_aliases "
                "ORDER BY alias_workspace,canonical"
            )
        ]


def test_separate_requires_authorization_with_impact(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_govern("separate_workspace_alias", {
        "alias": "旧项目", "canonical": "新项目",
    })
    assert result["ok"] is False
    data = result["data"]
    assert data["action_required"] == "ask_user_for_authorization"
    assert data["impact"] == ProductSurfaces._GOVERNANCE_IMPACTS["separate_workspace_alias"]


def test_separate_overrides_installed_redirect(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # Install a confirmed redirect first (the state being undone).
    with tools.db.write_transaction() as conn:
        ok, errors = tools.db.workspaces.record_workspace_decision_on_conn(
            conn, "旧项目", "新项目", status="confirmed",
        )
    assert ok and not errors
    resolved = tools.db.resolve_workspace_canonical("旧项目", None, register_new=False)
    assert resolved["canonical"] == "新项目"

    result = tools.memory_govern("separate_workspace_alias", {
        "alias": "旧项目", "canonical": "新项目",
        "reason": "user says they are different projects", "authorized": True,
    })
    assert result["ok"] is True, result["data"]
    assert result["data"]["separated"] is True
    assert _alias_rows(tools) == [("旧项目", "新项目", "rejected")]
    # The redirect no longer resolves.
    resolved_after = tools.db.resolve_workspace_canonical("旧项目", None, register_new=False)
    assert resolved_after["canonical"] != "新项目"


def test_separate_guard_blocks_silent_reconfirm(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    separated = tools.memory_govern("separate_workspace_alias", {
        "alias": "a-ws", "canonical": "b-ws", "reason": "keep apart", "authorized": True,
    })
    assert separated["ok"] is True

    with tools.db.write_transaction() as conn:
        ok, errors = tools.db.workspaces.record_workspace_decision_on_conn(
            conn, "a-ws", "b-ws", status="confirmed",
        )
    assert not ok and any("kept separate" in error for error in errors)

    # force=true reverses the decision explicitly.
    with tools.db.write_transaction() as conn:
        ok, errors = tools.db.workspaces.record_workspace_decision_on_conn(
            conn, "a-ws", "b-ws", status="confirmed", force=True,
        )
    assert ok and not errors


def test_separate_rejects_default_pool(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_govern("separate_workspace_alias", {
        "alias": "default", "canonical": "b-ws", "authorized": True,
    })
    assert result["ok"] is False
    assert "reserved" in str(result["data"]["error"])
