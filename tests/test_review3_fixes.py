"""R3/R4 实施后 review 的修复钉（2026-10-10）.

R3（对码）与 R4（对抗）独立复现的缺陷及其修复：

- **B2 P1**：`executescript` 隐式 COMMIT 摧毁外层 SAVEPOINT → 每次启动
  记假失败 `ddl:failed(no such savepoint)`（R3/R4 双向独立复现），且真实
  错误信息被掩盖成症状。修复：ddl 段移出 SAVEPOINT 包装并诚实标注。
- **A9 P1**：搬运路径（set canonical / move by id / 向量发布）无保护桶守卫
  → 以 twin 身份走 move 即可把变体注册进 canonical 表（R3 实测 ok=True）。
- **A4 P2**：doctor 缺 poison finding（方案 §A4-b/T5 未实现）→ 达界毒记忆
  只有 kick 回执可见，无主动巡检面。
- **B5 P2**：jsonl_backup_last_used_at 只在一处置位，另两个置位点
  （core.py 启动降级 / schema.py 只读探测失败）不同步。

钉死契约见各测试 docstring。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import SCAN_POISON_MAX_FAILURES
from memory_arbiter.tools import MemoryTools

pytest.importorskip("sqlite_vec")

import tests.test_vnext_evidence as tv  # noqa: E402
from memory_arbiter.request_identity import RequestIdentity, request_identity_scope  # noqa: E402


def make_tools(tmp_path: Path, **kw) -> MemoryTools:
    return MemoryTools(Settings(db_path=tmp_path / "m.db", backup_jsonl=tmp_path / "b.jsonl", **kw))


def _additive_warnings(tools: MemoryTools) -> list[str]:
    return [w for w in (tools.memory_status().get("warnings") or []) if "additive" in w]


def test_ddl_segment_emits_no_spurious_savepoint_failure(tmp_path: Path) -> None:
    """核心钉（R3/R4 P1）：正常启动不得出现 ddl:failed(no such savepoint)。"""
    tools = make_tools(tmp_path)
    for line in _additive_warnings(tools):
        assert "no such savepoint" not in line, line
        assert ":failed(" not in line, f"正常库不得有段失败标记: {line}"


def test_repeated_boots_stay_clean(tmp_path: Path) -> None:
    """核心钉：同库连续多次启动（幂等路径）也不产生假失败。"""
    for _ in range(3):
        tools = make_tools(tmp_path)
        for line in _additive_warnings(tools):
            assert "no such savepoint" not in line, line
            assert ":failed(" not in line, line


def test_real_segment_failure_still_reported(tmp_path: Path, monkeypatch) -> None:
    """真实段失败仍可见（修复不得吞掉真故障）。"""
    from memory_arbiter.db import additive as additive_mod

    tools = make_tools(tmp_path)
    tools.memory_write(content="x", subject="s", workspace="w", source_type="agent_generated")
    with tools.db.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS internal_conflicts")
    monkeypatch.setattr(
        additive_mod, "_migrate_legacy_candidates",
        lambda conn: (_ for _ in ()).throw(__import__("sqlite3").OperationalError("boom")),
    )
    with tools.db.connection() as conn:
        applied = additive_mod.ensure_additive_structures(conn)
    assert any("migrations_a:failed" in a for a in applied), applied


def test_move_to_protected_variant_refused(tmp_path: Path) -> None:
    """核心钉（R3 P1）：move 到保护桶变体被拒且不注册（R3 曾实测 ok=True）。"""
    tools = make_tools(tmp_path, workspace="w", client="mema-twin", agent_id="mema-twin")
    mid = int(tools.memory_write(content="p", subject="s", workspace="w",
                                source_type="agent_generated")["data"]["id"])
    with request_identity_scope(RequestIdentity(client="mema-twin", agent_id="mema-twin")):
        mv = tools.memory_govern("move_memories_workspace", {
            "memory_ids": [mid], "new_workspace": "mema_twin",
            "authorized": True, "reason": "x", "workspace": "w",
        })
    assert mv.get("ok") is False, mv
    with tools.db.connection() as conn:
        names = [r["name"] for r in conn.execute("SELECT name FROM workspace_canonicals")]
    assert "mema_twin" not in names, f"变体不得注册: {names}"


def test_set_canonical_to_protected_variant_refused(tmp_path: Path) -> None:
    """set canonical 路径同样拒变体。"""
    tools = make_tools(tmp_path, workspace="w")
    mid = int(tools.memory_write(content="p", subject="s", workspace="w",
                                source_type="agent_generated")["data"]["id"])
    with tools.db.write_transaction() as conn:
        ok, warnings = tools.db.workspaces.set_memory_workspace_canonical_on_conn(
            conn, mid, "mematwin",
        )
    assert ok is False
    assert any("protected-bucket" in w for w in warnings)


def test_doctor_reports_poison_after_threshold(tmp_path: Path) -> None:
    """核心钉（R3 P2）：达上界的毒记忆在 doctor 可见。"""
    tools = make_tools(tmp_path)
    ids = [int(tools.memory_write(content=f"c{i} 债务", subject=f"s{i}", workspace="ws",
                                 source_type="agent_generated")["data"]["id"]) for i in range(2)]
    sp = tools._scan_pipeline
    orig = sp._process_memory

    def boom(memory_id: int, **kw):
        if memory_id == ids[0]:
            raise RuntimeError("poison")
        return orig(memory_id, **kw)

    sp._process_memory = boom
    for _ in range(SCAN_POISON_MAX_FAILURES):
        tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 50,
                                             "time_budget_s": 3})
    doc = tools.memory_doctor_overview(deep=False)
    payload = json.dumps(doc, ensure_ascii=False)
    assert "scan.poison" in payload, payload[-500:]


def test_doctor_clean_scan_state_has_no_poison_finding(tmp_path: Path) -> None:
    """无失败时不新增噪声。"""
    tools = make_tools(tmp_path)
    tools.memory_write(content="x", subject="s", workspace="w", source_type="agent_generated")
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10,
                                         "time_budget_s": 5})
    doc = tools.memory_doctor_overview(deep=False)
    assert "scan.poison" not in json.dumps(doc, ensure_ascii=False)


def test_jsonl_obs_covers_readonly_degrade(tmp_path: Path) -> None:
    """B5 补齐：只读探测失败降级也记时间戳（与 jsonl_backup_active 同步）。"""
    tools = make_tools(tmp_path)
    tools.memory_write(content="x", subject="s", workspace="w", source_type="agent_generated")
    db_path = Path(tools.settings.db_path)
    import os
    os.chmod(db_path, 0o444)
    try:
        tools2 = MemoryTools(Settings(db_path=db_path, backup_jsonl=tmp_path / "b2.jsonl"))
        if tools2.db.state.jsonl_backup_active:
            assert tools2.db.state.jsonl_backup_last_used_at, (
                "jsonl_backup_active=True 时 last_used_at 不得为 None"
            )
    finally:
        os.chmod(db_path, 0o600)
