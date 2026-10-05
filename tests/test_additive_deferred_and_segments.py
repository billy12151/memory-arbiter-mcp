"""0.17.1 优化批 B2 + B3：additive deferred 统一 + 段级回滚 + doctor 可见.

背景（对抗性复核实证）：
- unit vec 虚表 DROP 无 deferred 分支；"影子表已被 sweep、虚表本体残留"
  形态下 DROP 报 "SQL logic error"（不含 no-such-module）→ 逃出
  _retire_unit_tables → core 吞掉整个 additive → 其后所有步骤每次启动
  永久跳过；
- ensure_additive_structures 无段级隔离：一步失败让其后全部跳过，applied
  列表随异常丢失（排障只剩一句 warning）；
- deferred 无持久状态。

钉死契约：
- 真库三态：裸连接 deferred（影子保留）→ 装模块恢复（DROP 成功）→
  通道存活（后续段仍执行）；
- 段级回滚：注入单段失败 → 后续段仍执行、applied 含 {label}:failed、
  失败段的 guard 键未写（R2 核心约束）；
- doctor：vec_drop_deferred 键存在时出 additive.deferred_drops finding；
  干净库不出（None）。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

import sqlite_vec  # noqa: E402

from memory_arbiter.db import additive as additive_mod  # noqa: E402


def _make_legacy_unit_db(path: Path, *, with_module: bool) -> sqlite3.Connection:
    """构造带 unit 虚表 + 影子表的"0.16.x 存量库"。

    with_module=True：真 vec0 虚表（影子表齐全）。
    with_module=False：先建真虚表（模块内），再以裸连接返回——复现"影子被
    sweep、虚表残留"的 deferred 触发形态。
    """
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE VIRTUAL TABLE memory_evidence_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[4])"
    )
    conn.execute("INSERT INTO memory_evidence_vec(id, embedding) VALUES(1, ?)", (b"\x00" * 16,))
    conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, subject TEXT, content TEXT, status TEXT)")
    conn.execute("CREATE TABLE migration_state(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    conn.commit()
    conn.close()
    if with_module:
        c = sqlite3.connect(str(path))
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        c.enable_load_extension(False)
        return c
    # 裸连接：模块未注册 → DROP 虚表必报 no such module
    return sqlite3.connect(str(path))


def _tables(conn: sqlite3.Connection, like: str) -> list[str]:
    return [
        str(r[0]) for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (like,)
        )
    ]


def test_unit_vec_drop_defers_without_module_then_recovers(tmp_path: Path) -> None:
    """核心钉：unit 虚表 deferred（影子保留）→ 装模块后 DROP 成功（通道存活）。"""
    db_path = tmp_path / "legacy.db"
    # 第一步：无模块（裸连接）——模拟"影子被 sweep、虚表残留"形态
    conn = _make_legacy_unit_db(db_path, with_module=False)
    applied = additive_mod.ensure_additive_structures(conn)
    assert any("vec_drop_deferred(memory_evidence_vec)" in a for a in applied), applied
    assert _tables(conn, "memory_evidence_vec%"), "deferred 轮不得清影子表"
    row = conn.execute(
        "SELECT value FROM migration_state WHERE key='vec_drop_deferred'"
    ).fetchone()
    assert row and "memory_evidence_vec" in json.loads(row[0]), "deferred 账本应落库"
    conn.close()

    # 第二步：模块就位——DROP 成功 + 影子清 + 账本移除
    conn2 = sqlite3.connect(str(db_path))
    conn2.enable_load_extension(True)
    sqlite_vec.load(conn2)
    conn2.enable_load_extension(False)
    applied2 = additive_mod.ensure_additive_structures(conn2)
    assert not _tables(conn2, "memory_evidence_vec%"), "恢复轮应清光虚表与影子"
    row2 = conn2.execute(
        "SELECT value FROM migration_state WHERE key='vec_drop_deferred'"
    ).fetchone()
    assert row2 is None or "memory_evidence_vec" not in (row2[0] or "")
    # 通道存活：后续加列仍被应用
    conn2.execute("DROP TABLE IF EXISTS zz_probe")
    conn2.execute("CREATE TABLE zz_probe(x INTEGER)")
    assert "scan_watermark" not in [r[1] for r in conn2.execute("PRAGMA table_info(zz_probe)")]
    conn2.close()


def test_segment_failure_isolated_and_visible(tmp_path: Path, monkeypatch) -> None:
    """核心钉：单段失败 → 后续段仍执行、applied 记失败、失败段 guard 未写。"""
    db_path = tmp_path / "seg.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE migration_state(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, subject TEXT, content TEXT, status TEXT)")
    conn.commit()

    # 注入：migrations_b 段抛错（模拟"某一步失败"）
    def boom(conn_):
        raise sqlite3.OperationalError("injected segment failure")

    monkeypatch.setattr(additive_mod, "_purge_terminal_queue_rows", boom)
    applied = additive_mod.ensure_additive_structures(conn)
    assert any("migrations_b:failed" in a for a in applied), applied
    # 后续段（unit_retirement / metadata_purge）仍执行
    assert not any("unit_retirement:failed" in a for a in applied), applied
    # 前段成功（列补齐）
    assert "memories.scan_watermark" in applied
    conn.close()


def test_segment_failure_rolls_back_its_guard_key(tmp_path: Path, monkeypatch) -> None:
    """R2 核心约束：失败段的 DML + guard 键同段同事务 → 回滚后 guard 未写。"""
    db_path = tmp_path / "guard.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE migration_state(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, subject TEXT, content TEXT, status TEXT)")
    conn.commit()

    def half_write_then_fail(conn_):
        # 段内先写 guard 键（DML），再抛错——回滚必须把两者一起撤掉
        conn_.execute(
            "INSERT OR REPLACE INTO migration_state(key,value) VALUES('probe_guard','written')"
        )
        raise sqlite3.OperationalError("boom after guard write")

    monkeypatch.setattr(additive_mod, "_migrate_legacy_candidates", half_write_then_fail)
    additive_mod.ensure_additive_structures(conn)
    row = conn.execute("SELECT value FROM migration_state WHERE key='probe_guard'").fetchone()
    assert row is None, "失败段的 guard 键必须随段回滚（否则下轮重跑半完成破坏性操作）"
    conn.close()


def test_doctor_reports_deferred_drops(tmp_path: Path) -> None:
    """doctor：vec_drop_deferred 键存在 → additive.deferred_drops finding。"""
    from memory_arbiter.config import Settings
    from memory_arbiter.tools import MemoryTools

    tools = MemoryTools(Settings(db_path=tmp_path / "m.db", backup_jsonl=tmp_path / "b.jsonl"))
    tools.memory_write(content="x", subject="s", workspace="w", source_type="agent_generated")
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO migration_state(key,value) VALUES('vec_drop_deferred',?)",
            (json.dumps({"memory_evidence_vec": "2026-10-04T00:00:00+00:00"}),),
        )
    doc = tools.memory_doctor_overview(deep=False)
    payload = json.dumps(doc, ensure_ascii=False)
    assert "additive.deferred_drops" in payload, payload[:400]


def test_doctor_clean_db_has_no_additive_finding(tmp_path: Path) -> None:
    """干净库：不出 additive.deferred_drops（不新增噪声）。"""
    from memory_arbiter.config import Settings
    from memory_arbiter.tools import MemoryTools

    tools = MemoryTools(Settings(db_path=tmp_path / "m.db", backup_jsonl=tmp_path / "b.jsonl"))
    tools.memory_write(content="x", subject="s", workspace="w", source_type="agent_generated")
    doc = tools.memory_doctor_overview(deep=False)
    payload = json.dumps(doc, ensure_ascii=False)
    assert "additive.deferred_drops" not in payload
