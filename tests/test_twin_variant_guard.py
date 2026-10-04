"""0.17.1 修复批 A9：twin 保护桶机械变体劫持（判定键统一 + 注册守卫）.

背景（实测）：twin_redirect.py 的判定只做 casefold+strip，而解析器的 1b
折叠还去 _-\\s（workspaces.py._mechanical_ws_key）。攻击者先写 mema_twin
完成注册 → twin 本体写 mema-twin 被折到 mema_twin（攻击者桶）→ 攻击者
strict 可读出 TWIN-PERSONA-SECRET。治理路径（rename→mematwin）同样能
注册变体并劫持。

钉死契约：
- 判定键：mema_twin / Mema-Twin / "mema twin" 都触发改道；
- 写路径注册拒绝变体（攻击者首写不成功）；
- 治理路径 rename / migrate 拒绝变体目标（核心新钉）；
- 别名确认拒绝变体；
- twin 本体写 mema-twin 落 mema-twin（不被劫持）；
- mema-twin-dev 同族保护不回归；非保护桶变体不受影响；
- 两处机械键实现等价（防漂移钉）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools
from memory_arbiter.twin_redirect import (
    _mechanical_key,
    protected_bucket_variant,
    twin_redirect_target,
)


def make_tools(tmp_path: Path, **kw) -> MemoryTools:
    return MemoryTools(Settings(db_path=tmp_path / "m.db", backup_jsonl=tmp_path / "b.jsonl", **kw))


# ── 判定键 ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ["mema_twin", "mematwin", "Mema-Twin", "mema twin", "MEMA_TWIN"])
def test_twin_redirect_covers_mechanical_variants(variant: str) -> None:
    """核心钉：分隔符/大小写变体一律改道到 mema-twin-dev。"""
    assert twin_redirect_target(variant, client="attacker", agent_id="attacker") == "mema-twin-dev"


def test_twin_identity_unaffected() -> None:
    assert twin_redirect_target("mema-twin", client="mema-twin", agent_id="mema-twin") is None
    assert twin_redirect_target("mema_twin", client="mema-twin", agent_id="mema-twin") is None


def test_non_protected_buckets_unaffected() -> None:
    """非保护桶的机械变体不受影响（守卫只针对 PROTECTED_WORKSPACES）。"""
    assert twin_redirect_target("agent_lane", client="c", agent_id="a") is None
    assert protected_bucket_variant("agent_lane") is False
    assert protected_bucket_variant("AgentLane") is False


def test_mechanical_key_equivalence() -> None:
    """防漂移钉：twin_redirect 的 _mechanical_key 与 db/workspaces 实现等价。"""
    from memory_arbiter.db.workspaces import _mechanical_ws_key

    for sample in ("mema-twin", "mema_twin", "Mema Twin", "agent-lane", "ＡgentLane", ""):
        assert _mechanical_key(sample) == _mechanical_ws_key(sample), sample


def test_protected_bucket_variant_predicate() -> None:
    assert protected_bucket_variant("mema_twin") is True
    assert protected_bucket_variant("mematwin") is True
    assert protected_bucket_variant("mema-twin") is False, "原名不是变体"
    assert protected_bucket_variant("mema-twin-dev") is False, "同族原名不是变体"


# ── 写路径 ──────────────────────────────────────────────────────────────────

def test_write_path_refuses_variant_registration(tmp_path: Path) -> None:
    """核心钉：变体名写入不得把变体注册进 canonical 表（此前成功注册 → 劫持 twin）。

    写路径的 twin 改道（twin_redirect_target 机械键）先把变体名折到
    mema-twin-dev，故注册守卫是第二道；本测试断言"变体名不落库"这一
    结果本身。
    """
    tools = make_tools(tmp_path)
    res = tools.memory_write(content="atk", subject="s", workspace="mema_twin",
                             source_type="agent_generated")
    with tools.db.connection() as conn:
        names = [r["name"] for r in conn.execute("SELECT name FROM workspace_canonicals")]
    assert "mema_twin" not in names, f"变体名不得注册: {names}"


def test_twin_write_not_hijacked_after_attack_attempt(tmp_path: Path) -> None:
    """核心钉：攻击尝试后 twin 本体写入仍落 mema-twin（不被劫持到攻击者桶）。"""
    from memory_arbiter.request_identity import RequestIdentity, request_identity_scope

    tools = make_tools(tmp_path)
    tools.memory_write(content="atk", subject="s", workspace="mema_twin", source_type="agent_generated")
    with request_identity_scope(RequestIdentity(client="mema-twin", agent_id="mema-twin")):
        res = tools.memory_write(content="TWIN-PERSONA-SECRET", subject="p", workspace="mema-twin",
                                 source_type="user_confirmed")
    assert res.get("ok"), res
    rec = tools.db.get_memory(int(res["data"]["id"]))
    assert rec["workspace_canonical"] == "mema-twin", "twin 写入不得被折进变体桶"
    # 反向证明：攻击者读不到（strict 下用变体桶名看不到 twin 正文）
    with tools.db.connection() as conn:
        canon = conn.execute(
            "SELECT workspace_canonical FROM memories WHERE id=?", (int(res["data"]["id"]),)
        ).fetchone()[0]
    assert canon != "mema_twin"


# ── 治理路径 ────────────────────────────────────────────────────────────────

def test_rename_refuses_variant_target(tmp_path: Path) -> None:
    """核心钉（R2 洞）：rename 到变体名被拒（此前成功并劫持）。"""
    tools = make_tools(tmp_path, workspace="attacker-lane")
    tools.memory_write(content="a", subject="s", workspace="attacker-lane", source_type="agent_generated")
    r = tools.memory_govern("rename_workspace_canonical", {
        "old": "attacker-lane", "new": "mematwin", "authorized": True,
        "reason": "x", "workspace": "attacker-lane",
    })
    assert r.get("ok") is False
    with tools.db.connection() as conn:
        names = [x["name"] for x in conn.execute("SELECT name FROM workspace_canonicals")]
    assert "mematwin" not in names


def test_migrate_refuses_variant_target(tmp_path: Path) -> None:
    """核心钉：migrate 到变体名被拒。"""
    tools = make_tools(tmp_path, workspace="src")
    tools.memory_write(content="a", subject="s", workspace="src", source_type="agent_generated")
    r = tools.memory_govern("migrate_workspace", {
        "from": "src", "to": "mema_twin", "authorized": True, "reason": "x", "workspace": "src",
    })
    assert r.get("ok") is False
    with tools.db.connection() as conn:
        names = [x["name"] for x in conn.execute("SELECT name FROM workspace_canonicals")]
    assert "mema_twin" not in names


def test_confirm_alias_refuses_variant(tmp_path: Path) -> None:
    """别名确认路径拒绝变体（P1-3 同族毒行）。"""
    tools = make_tools(tmp_path)
    ok, warnings = tools.db.workspaces.record_workspace_decision_on_conn(
        tools.db.connection().__enter__(), "proj-a", "mema_twin", status="confirmed",
    ) if False else (None, None)
    # 走产品面：separate/confirm 路径由 governance 测试覆盖；此处直接调
    # _apply_alias_decision_on_conn 的语义（record_workspace_decision_on_conn）
    with tools.db.write_transaction() as conn:
        from memory_arbiter.db.workspaces import WorkspaceStore
        ok, warnings = WorkspaceStore.record_workspace_decision_on_conn(
            tools.db.workspaces, conn, "proj-a", "mema_twin", status="confirmed",
        )
    assert ok is False
    assert any("protected-bucket" in w for w in warnings)
