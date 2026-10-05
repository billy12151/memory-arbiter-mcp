"""0.17.1 claims 数据层全退回归钉（owner 2026-09-29 拍板连表删）。

三层保证：
1. 新库不再建 memory_claims / memory_claim_vec；
2. 存量库（0.17.0 时代带表带数据）启动 additive completion 幂等 DROP，
   数据一并清除（检测线已零读取，无消费方）；
3. 全链无残留引用：write/update 的 claims 参数出 schema（未知键软着陆：
   警告+忽略）、_persist_claims* 与 ClaimsStore 消亡。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def _table_exists(db_path: Path, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_fresh_db_has_no_claims_tables(tmp_path: Path) -> None:
    tools = _make_tools(tmp_path)
    tools.db  # 触发建库 + additive completion
    db_path = tmp_path / "memory.sqlite3"
    assert not _table_exists(db_path, "memory_claims")
    assert not _table_exists(db_path, "memory_claim_vec")


def test_legacy_db_claims_tables_dropped_on_startup(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    tools = _make_tools(tmp_path)
    tools.memory_write(content="第一版口径。", subject="v1", tags=[])
    # 模拟 0.17.0 存量库：在合法库上补建两张表并塞数据（新连接写完即关，
    # 下一个 MemoryDB 实例重开时触发 additive completion）
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE memory_claims (
                id INTEGER PRIMARY KEY,
                memory_id INTEGER NOT NULL,
                memory_version INTEGER NOT NULL,
                attr TEXT, attr_norm TEXT,
                value TEXT, value_norm TEXT,
                source TEXT DEFAULT 'agent',
                created_at TEXT,
                UNIQUE(memory_id, memory_version, attr_norm, value_norm)
            );
            CREATE INDEX memory_claims_attr_norm_idx ON memory_claims(attr_norm);
            INSERT INTO memory_claims(memory_id, memory_version, attr, attr_norm,
                                      value, value_norm, created_at)
                VALUES (1, 1, '数据库', '数据库', 'MySQL', 'mysql', '2026-09-01');
            """
        )
        conn.commit()
    finally:
        conn.close()
    # 重开（新的 MemoryDB 触发 additive completion）→ 幂等 DROP
    tools2 = _make_tools(tmp_path)
    tools2.memory("status", {})
    assert not _table_exists(db_path, "memory_claims"), "存量表必须被 DROP"
    assert not _table_exists(db_path, "memory_claim_vec")


def test_claims_param_is_ignored_with_warning(tmp_path: Path) -> None:
    """claims 出 schema 后走未知键软着陆：警告忽略、零回执键。"""
    tools = _make_tools(tmp_path)
    result = tools.memory_write(
        content="部署端口=8000。",
        subject="端口",
        tags=[],
        claims=[{"attr": "端口", "value": "8000"}],
    )
    assert result["ok"] is True
    data = result["data"]
    assert "claims_written" not in data and "claims_rejected" not in data
    warnings = " ".join(str(w) for w in result.get("warnings", []))
    assert "unknown field ignored: claims" in warnings


def test_no_persist_claims_code_path(tmp_path: Path) -> None:
    tools = _make_tools(tmp_path)
    assert not hasattr(tools.db, "claims"), "ClaimsStore 必须从 db 上摘除"
    import memory_arbiter.pipeline.write as write_mod
    assert not hasattr(write_mod.WritePipeline, "_persist_claims")
    assert not hasattr(write_mod.WritePipeline, "_persist_claims_for_version")
