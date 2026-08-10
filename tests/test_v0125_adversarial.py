from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import MemoryStatus
from memory_arbiter.tools import MemoryTools


def _tools(tmp_path: Path, *, isolation: str = "strict", structured: str = "off") -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "adv.sqlite3",
        backup_jsonl=tmp_path / "adv.jsonl",
        workspace="default",
        isolation=isolation,
        structured_claim_mode=structured,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _pending(tools: MemoryTools, workspace: str, content: str = "pending") -> int:
    return tools.memory_write(content=content, subject="pending", workspace=workspace)["data"]["id"]


def _active(tools: MemoryTools, workspace: str, content: str = "active", **kw) -> int:
    memory_id = tools.memory_write(content=content, subject=kw.pop("subject", "active"), workspace=workspace, **kw)["data"]["id"]
    activated = tools.memory_activate(memory_id=memory_id, authorized=True, workspace=workspace)
    assert activated["ok"] is True
    return memory_id


def test_strict_mutating_id_paths_reject_cross_workspace(tmp_path: Path):
    tools = _tools(tmp_path)
    a_id = _active(tools, "projA", "alpha")
    b_id = _active(tools, "projB", "beta")
    c_pending = _pending(tools, "projC", "gamma pending")

    assert tools.memory_confirm(b_id, authorized=True, workspace="projA")["ok"] is False
    assert tools.memory_activate(memory_id=c_pending, authorized=True, workspace="projA")["ok"] is False
    assert tools.memory_activate(memory_id=c_pending, authorized=True)["ok"] is True
    assert tools.memory_supersede(memory_id=b_id, reason="x", authorized=True, workspace="projA")["ok"] is False
    assert tools.memory_supersede(memory_id=a_id, reason="x", superseded_by=b_id, authorized=True, workspace="projA")["ok"] is False
    assert tools.memory_set_entity(memory_id=b_id, entity="svc", workspace="projA")["ok"] is False
    assert tools.memory_store_embedding(memory_id=b_id, embedding=[0.1, 0.2], workspace="projA")["ok"] is False
    assert tools.memory_cleanup_history(memory_id=b_id, workspace="projA")["ok"] is False

    unchanged = tools.memory_get(memory_id=b_id, workspace="projB")
    assert unchanged["data"]["memory"]["status"] == MemoryStatus.ACTIVE.value
    assert unchanged["data"]["memory"]["source_type"] != "user_confirmed"


def test_confirm_pending_workspace_is_caller_scoped_under_strict(tmp_path: Path):
    tools = _tools(tmp_path)
    pending_id = _pending(tools, "projB", "beta pending")

    denied = tools.memory_govern(
        "confirm_pending_workspace",
        {"memory_id": pending_id, "canonical": "projA", "workspace": "projA"},
    )
    ok = tools.memory_govern(
        "confirm_pending_workspace",
        {"memory_id": pending_id, "canonical": "projB", "workspace": "projB"},
    )

    assert denied["ok"] is False
    assert ok["ok"] is True
    assert ok["data"]["record"]["status"] == MemoryStatus.ACTIVE.value


def test_strict_structured_claim_pairs_do_not_cross_workspaces(tmp_path: Path):
    tools = _tools(tmp_path, structured="beta_all")
    a_id = _active(
        tools,
        "projA",
        "服务端口是 5432。",
        subject="svc port",
        metadata={"entity": "svc", "scope": "prod"},
    )
    b_id = _active(
        tools,
        "projB",
        "服务端口是 3306。",
        subject="svc port",
        metadata={"entity": "svc", "scope": "prod"},
    )

    a_rebuild = tools.memory_rebuild_claims(memory_ids=[a_id], dry_run=False, workspace="projA")
    b_rebuild = tools.memory_rebuild_claims(memory_ids=[b_id], dry_run=False, workspace="projB")
    conflicts = tools.memory_list_conflicts(status="open", workspace="projA")

    assert a_rebuild["ok"] is True
    assert b_rebuild["ok"] is True
    assert conflicts["data"]["conflicts"] == []


def test_activate_pending_reindexes_structured_claims(tmp_path: Path):
    tools = _tools(tmp_path, structured="beta_all")
    first = _active(
        tools,
        "projA",
        "服务端口是 5432。",
        subject="svc port",
        metadata={"entity": "svc", "scope": "prod"},
    )
    tools.memory_rebuild_claims(memory_ids=[first], dry_run=False, workspace="projA")
    pending = tools.memory_write(
        content="服务端口是 3306。",
        subject="svc port",
        workspace="projA",
        metadata={"entity": "svc", "scope": "prod"},
        status="pending",
    )["data"]["id"]

    activated = tools.memory_activate(memory_id=pending, authorized=True, workspace="projA")
    record = tools.memory_get(memory_id=pending, workspace="projA")["data"]["memory"]

    assert activated["ok"] is True
    assert activated["data"]["claim_indexed"] is True
    assert activated["data"]["claim_reconciled"] is True
    assert record["claims_indexed_revision"] == record["claim_revision"]
    assert tools.memory_list_conflicts(status="open", workspace="projA")["data"]["count"] == 1


def test_authorized_arbitrate_resolves_loser_conflicts(tmp_path: Path):
    tools = _tools(tmp_path, isolation="none")
    left = tools.memory_write(content="old value", subject="same")["data"]["id"]
    right = tools.memory_write(content="new value", subject="same", source_type="user_confirmed")["data"]["id"]

    result = tools.memory_arbitrate(left, right, mark_conflict=True, authorized=True)
    loser = result["data"]["comparison"]["loser_id"]
    open_rows = [
        c for c in tools.memory_list_conflicts(status="open")["data"]["conflicts"]
        if c["left_id"] == loser or c["right_id"] == loser
    ]

    assert result["data"]["applied"] is True
    assert result["data"]["linked_conflicts_resolved"] >= 1
    assert open_rows == []
