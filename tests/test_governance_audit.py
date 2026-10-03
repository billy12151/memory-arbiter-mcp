"""0.17.1 P2 batch: governance_audit trail (#7), tags_only field replaces
(#8), move-batch rollback sentinel hygiene (#10)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "gov.sqlite3",
        backup_jsonl=tmp_path / "gov.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write(tools: MemoryTools, subject: str, workspace: str, content: str | None = None) -> int:
    return int(tools.memory_write(
        content=content or f"content of {subject}", subject=subject, workspace=workspace,
    )["data"]["id"])


def _governance_rows(tools: MemoryTools) -> list[dict]:
    with tools.db.connection() as conn:
        return [
            {"action": row["action"], "reason": row["reason"], "detail": row["detail_json"]}
            for row in conn.execute(
                "SELECT action, reason, detail_json FROM governance_audit ORDER BY id"
            ).fetchall()
        ]


# ── #7: governance_audit ────────────────────────────────────────────────────

def test_rename_migrate_confirm_pending_write_governance_rows(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "s1", "old-bucket")

    renamed = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "old-bucket", "old": "old-bucket", "new": "renamed-bucket",
        "reason": "typo fix", "authorized": True,
    })
    assert renamed["ok"], renamed

    _write(tools, "s2", "merge-src")
    migrated = tools.memory_govern("migrate_workspace", {
        "workspace": "merge-src", "from": "merge-src", "to": "renamed-bucket",
        "reason": "same project", "authorized": True,
    })
    assert migrated["ok"], migrated

    tools.settings.isolation = "none"
    pending_id = tools.memory_write(
        content="pending body", subject="s3", workspace="abbrev", status="pending",
    )["data"]["id"]
    confirmed = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "abbrev", "memory_id": pending_id, "canonical": "full-name",
        "reason": "user picked the bucket", "authorized": True,
    })
    assert confirmed["ok"], confirmed

    rows = _governance_rows(tools)
    by_action = {row["action"]: row for row in rows}
    assert set(by_action) == {"rename_workspace", "migrate_workspace", "confirm_pending_workspace"}
    assert by_action["rename_workspace"]["reason"] == "typo fix"
    assert "old-bucket" in by_action["rename_workspace"]["detail"]
    assert "renamed-bucket" in by_action["rename_workspace"]["detail"]
    assert by_action["migrate_workspace"]["reason"] == "same project"
    assert "merge-src" in by_action["migrate_workspace"]["detail"]
    confirm = by_action["confirm_pending_workspace"]
    assert confirm["reason"] == "user picked the bucket"
    assert str(pending_id) in confirm["detail"]
    assert "full-name" in confirm["detail"]


def test_audit_view_carries_recent_governance_rows(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "s1", "old-bucket")
    renamed = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "old-bucket", "old": "old-bucket", "new": "renamed-bucket",
        "reason": "reviewed rename", "authorized": True,
    })
    assert renamed["ok"], renamed

    audit = tools.memory_review("audit", {})
    assert audit["ok"], audit
    governance = audit["data"].get("governance_audit")
    assert governance, audit["data"]
    newest = governance[0]
    assert newest["action"] == "rename_workspace"
    assert newest["reason"] == "reviewed rename"
    assert newest["detail"]["old_canonical"] == "old-bucket"
    assert newest["detail"]["new_canonical"] == "renamed-bucket"
    # newest first
    assert governance[0]["id"] >= governance[-1]["id"]


def test_rollback_auto_move_ignores_governance_rows(tmp_path: Path) -> None:
    """governance_audit 与 normalize_audit 消费面隔离：rollback 只认
    normalize_audit 的 applied 行，governance 行不可被当作回滚锚点。"""
    tools = make_tools(tmp_path)
    _write(tools, "s1", "old-bucket")
    renamed = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "old-bucket", "old": "old-bucket", "new": "renamed-bucket",
        "reason": "typo fix", "authorized": True,
    })
    assert renamed["ok"], renamed
    with tools.db.connection() as conn:
        gov_id = conn.execute(
            "SELECT id FROM governance_audit ORDER BY id LIMIT 1"
        ).fetchone()["id"]
    assert gov_id > 0

    outcome = tools.memory_govern("rollback_auto_move", {
        "workspace": "renamed-bucket", "audit_id": int(gov_id),
        "reason": "mistake", "authorized": True,
    })
    assert not outcome["ok"]
    assert "not found or not applied" in str(outcome["data"].get("error", ""))


# ── #8: tags_only field replaces ────────────────────────────────────────────

def test_tags_only_rejects_new_tags_and_new_subject(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "s1", "w")
    for payload in ({"new_tags": ["x"]}, {"new_subject": "renamed"}):
        result = tools.memory_edit(
            memory_id=mid, tags_only=True, add_tags=["keep"], **payload,
        )
        assert result["ok"] is False, payload
        assert result["data"]["edited"] is False
        error = str(result["data"].get("error", ""))
        assert "tags_only=true cannot be combined" in error
        assert "add_tags/remove_tags" in error  # the guidance sentence
    # the clean tags-only call still works
    ok = tools.memory_edit(memory_id=mid, tags_only=True, add_tags=["clean"])
    assert ok["ok"], ok


# ── #10: aborted-response sentinel hygiene ──────────────────────────────────

def test_move_aborted_response_strips_voided_ticket_sentinel(
    tmp_path: Path, monkeypatch,
) -> None:
    tools = make_tools(tmp_path)
    id_a = _write(tools, "sa", "default", content="fact a")
    id_b = _write(tools, "sb", "default", content="fact b")
    store = tools.db.workspaces

    def fake_move(
        conn, memory_id, workspace, *, precomputed_embedding=None, allow_default=False,
        current_bucket=None, sha_collision=None, queue_rows=None,
    ):
        if int(memory_id) == id_a:
            return True, ["voided_conflict_tickets:2"]
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "move_memory_workspace_on_conn", fake_move)
    monkeypatch.setattr(
        store, "prepare_missing_workspace_canonical_embedding", lambda c, e: None,
    )

    outcome = tools.memory_govern("move_memories_workspace", {
        "workspace": "default", "memory_ids": [id_a, id_b],
        "new_workspace": "proj-x", "reason": "relocate", "authorized": True,
    })
    assert not outcome["ok"]
    assert outcome["data"]["moved_ids"] == []
    # the sentinel describes voids the rollback undid — it must not leak as a
    # raw warning nor as the structured success-path counter
    assert not any(
        str(w).startswith("voided_conflict_tickets:") for w in outcome.get("warnings") or []
    )
    assert "conflict_tickets_voided" not in outcome["data"]


def test_move_batch_same_request_sha_siblings_refuse_sequentially(tmp_path: Path) -> None:
    """P2 #10 批化等价：同请求内两条同 content_sha 的 ACTIVE 行依次移动——
    第一条落桶后，第二条必须像逐 id 流程一样吃 dedup gate 拒绝。"""
    tools = make_tools(tmp_path)
    dup_a = _write(tools, "sa", "ws-a", content="byte-identical fact")
    dup_b = _write(tools, "sb", "ws-b", content="byte-identical fact")
    other = _write(tools, "sc", "ws-c", content="unique fact")

    outcome = tools.memory_govern("move_memories_workspace", {
        "workspace": "default", "memory_ids": [dup_a, dup_b, other],
        "new_workspace": "proj-x", "reason": "relocate", "authorized": True,
    })
    assert not outcome["ok"]
    assert outcome["data"]["moved_ids"] == [dup_a, other]
    reasons = {e["memory_id"]: str(e.get("reason", "")) for e in outcome["data"]["errors"]}
    assert "content duplicate collision" in reasons[dup_b]
    assert other not in reasons
    # the refused sibling never moved: its row stays in its original bucket
    assert tools.db.get_memory(dup_b)["workspace"] == "ws-b"
    assert tools.db.get_memory(dup_a)["workspace"] == "proj-x"
