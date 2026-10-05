"""0.17.1 修复批 A5：治理审计无条件落库 + committed 三态.

背景（产品面实测）：operations.py 以 ``not warnings`` 判定落审计与
renamed/ok。但 rename 在 repoint 警告形态下**已提交**（UPDATE + 别名重装），
旧口径会同时报失败且零审计（实测：库内已改名、响应 renamed=False、审计
0 行；agent 会重试）。对抗 review 补充：只改 renamed 不改 ok 会产出
ok=False + renamed=True 的自相矛盾响应，agent 的第一判据是 ok。

钉死契约：
- committed 由"事务是否越过前置守卫并提交"决定（含 no-op/空源桶）；
- 普通 rename/migrate → ok=True/renamed=True + 审计 1 行（不回归）；
- repoint 警告形态 → ok=True/renamed=True + 审计 1 行且 detail 含 warnings（核心钉）；
- 前置拒绝（sha 冲突/slot 冲突）→ ok=False + 审计 0 行；
- no-op（old==new）→ ok=True + 审计 1 行（保住今天行为）；
- 审计 detail 可被 memory_review(view="audit") 读到。
"""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path, workspace: str = "Old") -> MemoryTools:
    return MemoryTools(Settings(db_path=tmp_path / "m.db", backup_jsonl=tmp_path / "b.jsonl",
                                workspace=workspace))


def _audit_rows(tools: MemoryTools, action: str) -> list[dict]:
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT action, reason, detail_json FROM governance_audit WHERE action=? ORDER BY id",
            (action,),
        ).fetchall()
    import json as _json
    return [{**dict(r), "detail": _json.loads(r["detail_json"] or "{}")} for r in rows]


def test_plain_rename_ok_and_audited(tmp_path: Path) -> None:
    """普通 rename：ok=True + 审计 1 行（既有行为不回归）。"""
    tools = make_tools(tmp_path)
    tools.memory_write(content="a", subject="s", workspace="Old", source_type="agent_generated")
    r = tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "New", "authorized": True, "reason": "user", "workspace": "Old",
    })
    assert r.get("ok") is True
    assert (r.get("data") or {}).get("renamed") is True
    assert len(_audit_rows(tools, "rename_workspace")) == 1


def test_repoint_warning_rename_is_committed_and_audited(tmp_path: Path) -> None:
    """核心钉：repoint 警告形态下 ok=True/renamed=True + 审计 1 行（此前 ok=False/0 行）。"""
    tools = make_tools(tmp_path)
    tools.memory_write(content="a", subject="s", workspace="Old", source_type="agent_generated")
    # 播种 rejected 孪生行：alias='new-x' 与目标 'New_x' 机械等价
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO workspace_aliases(alias_workspace, canonical, status, updated_at) "
            "VALUES(?,?,?,?)", ("new-x", "Old", "rejected", "2026-01-01T00:00:00+00:00"),
        )
    r = tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "New_x", "authorized": True, "reason": "user", "workspace": "Old",
    })
    data = r.get("data") or {}
    assert r.get("ok") is True, f"已提交的治理动作不得报失败: {r.get('ok')}"
    assert data.get("renamed") is True
    assert data.get("memories_updated") == 1
    rows = _audit_rows(tools, "rename_workspace")
    assert len(rows) == 1, "已提交必须有审计行"
    assert rows[0]["detail"].get("warnings"), "警告一并入审计 detail"
    # 库内事实与响应一致
    with tools.db.connection() as conn:
        canon = conn.execute("SELECT workspace_canonical FROM memories").fetchone()[0]
    assert canon == "New_x"


def test_refused_rename_not_audited(tmp_path: Path) -> None:
    """前置拒绝（sha 冲突）：ok=False + 审计 0 行（未提交不落）。"""
    tools = make_tools(tmp_path, workspace="A")
    tools.memory_write(content="same", subject="sA", workspace="A", source_type="agent_generated")
    tools.memory_write(content="same", subject="sB", workspace="B", source_type="agent_generated")
    r = tools.memory_govern("rename_workspace_canonical", {
        "old": "A", "new": "B", "authorized": True, "reason": "x", "workspace": "A",
    })
    assert r.get("ok") is False
    assert (r.get("data") or {}).get("renamed") is False
    assert _audit_rows(tools, "rename_workspace") == []


def test_noop_rename_still_audited(tmp_path: Path) -> None:
    """no-op（old==new）：ok=True + 审计 1 行（保住今天的既有行为）。"""
    tools = make_tools(tmp_path, workspace="Same")
    tools.memory_write(content="a", subject="s", workspace="Same", source_type="agent_generated")
    r = tools.memory_govern("rename_workspace_canonical", {
        "old": "Same", "new": "Same", "authorized": True, "reason": "noop", "workspace": "Same",
    })
    assert r.get("ok") is True
    assert (r.get("data") or {}).get("renamed") is True
    assert len(_audit_rows(tools, "rename_workspace")) == 1


def test_migrate_four_states(tmp_path: Path) -> None:
    """migrate 四态：普通 / repoint 警告 / sha 拒绝 / 空源桶合法提交。"""
    tools = make_tools(tmp_path, workspace="Src")
    tools.memory_write(content="v1", subject="s", workspace="Src", source_type="agent_generated")
    r = tools.memory_govern("migrate_workspace", {
        "from": "Src", "to": "Dst", "authorized": True, "reason": "r", "workspace": "Src",
    })
    assert r.get("ok") is True
    assert (r.get("data") or {}).get("migrated") is True
    assert len(_audit_rows(tools, "migrate_workspace")) == 1

    # sha 冲突拒绝（Dst 已有同内容 active 行）
    tools2 = make_tools(tmp_path / "t2", workspace="A")
    tools2.memory_write(content="same", subject="sA", workspace="A", source_type="agent_generated")
    tools2.memory_write(content="same", subject="sB", workspace="B", source_type="agent_generated")
    r2 = tools2.memory_govern("migrate_workspace", {
        "from": "A", "to": "B", "authorized": True, "reason": "x", "workspace": "A",
    })
    assert r2.get("ok") is False
    assert _audit_rows(tools2, "migrate_workspace") == []

    # 空源桶：合法提交（删 canonical 行 + 装 redirect）
    tools3 = make_tools(tmp_path / "t3", workspace="Ghost")
    r3 = tools3.memory_govern("migrate_workspace", {
        "from": "Ghost", "to": "Real", "authorized": True, "reason": "r", "workspace": "Ghost",
    })
    assert r3.get("ok") is True, "空源桶 migrate 是合法提交"
    assert len(_audit_rows(tools3, "migrate_workspace")) == 1


def test_audit_detail_readable_via_review(tmp_path: Path) -> None:
    """审计 detail 可被 memory_review(view='audit') 读到（消费面回归钉）。"""
    tools = make_tools(tmp_path)
    tools.memory_write(content="a", subject="s", workspace="Old", source_type="agent_generated")
    tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "New", "authorized": True, "reason": "why", "workspace": "Old",
    })
    review = tools.memory_review("audit", {})
    payload = str(review.get("data") or {})
    assert "rename_workspace" in payload
