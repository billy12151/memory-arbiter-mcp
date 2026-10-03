"""P1-1（方案 A1，2026-10-03）：claims vec 虚表 deferred-DROP 与影子表 sweep 顺序.

钉死的契约：
- deferred 轮（连接未注册 vec0 模块）：vtab DROP 记 deferred 跳过，影子表
  sweep **不执行**（影子表保留是恢复轮 DROP 可重试的前提——先清影子表
  会让恢复轮 xDestroy 报 SQL logic error，additive 收尾对该库永久失败）；
- 恢复轮（模块就位、DROP 成功）：vtab DROP + 影子表 sweep 都执行；
- 真库集成：sqlite_vec 建真 vec0 虚表库 → 裸连接（不 load 模块）跑
  deferred → 加载模块连接跑恢复（模块加载是连接级选择，core.py 实证）。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.db.additive import _retire_claims_tables

pytest.importorskip("sqlite_vec")


class _StubConn:
    """记录调用序列；对 vtab DROP 抛可配置异常的假连接。"""

    def __init__(self, *, vtab_drop_error: str | None, has_vtab: bool = True,
                 has_claims: bool = True, shadows: list[str] | None = None) -> None:
        self.calls: list[str] = []
        self._vtab_drop_error = vtab_drop_error
        self._has_vtab = has_vtab
        self._has_claims = has_claims
        self._shadows = shadows or ["memory_claim_vec_chunks", "memory_claim_vec_rowids"]

    def execute(self, sql: str, params=()):
        self.calls.append(sql)
        lowered = sql.lower()
        if lowered.startswith("select 1 from sqlite_master"):
            name = str(params[0]) if params else ""
            return _Rows([[0]] if (name == "memory_claim_vec" and self._has_vtab)
                         or (name == "memory_claims" and self._has_claims) else [])
        if "like 'memory_claim_vec_%'" in lowered:
            return _Rows([[name] for name in self._shadows])
        if lowered.startswith("drop table memory_claim_vec") and self._vtab_drop_error:
            raise sqlite3.OperationalError(self._vtab_drop_error)
        return _Rows([])


class _Rows:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


def test_deferred_round_keeps_shadow_tables() -> None:
    conn = _StubConn(vtab_drop_error="no such module: vec0")
    applied: list[str] = []
    _retire_claims_tables(conn, applied)  # type: ignore[arg-type]
    assert "claims_vec_drop_deferred(memory_claim_vec)" in applied
    drop_calls = [c for c in conn.calls if c.lower().startswith("drop table")]
    # vtab DROP 尝试后抛错（调用已入序列）、普通表照清、影子表 sweep 不执行
    assert "DROP TABLE memory_claim_vec" in drop_calls
    assert "DROP TABLE memory_claims" in drop_calls
    assert not any(c.lower().startswith('drop table if exists') for c in drop_calls), "deferred 轮不得清影子表"


def test_recovery_round_drops_vtab_and_sweeps_shadows() -> None:
    conn = _StubConn(vtab_drop_error=None)
    applied: list[str] = []
    _retire_claims_tables(conn, applied)  # type: ignore[arg-type]
    drop_calls = [c for c in conn.calls if c.lower().startswith("drop table")]
    assert "DROP TABLE memory_claim_vec" in drop_calls
    assert any('DROP TABLE IF EXISTS "memory_claim_vec_chunks"' == c for c in drop_calls)
    assert any("claims_tables_dropped(memory_claim_vec" in a for a in applied)


def test_unrelated_operational_error_reraises() -> None:
    conn = _StubConn(vtab_drop_error="database is locked")
    with pytest.raises(sqlite3.OperationalError):
        _retire_claims_tables(conn, [])  # type: ignore[arg-type]


def test_real_db_deferred_then_recovery(tmp_path: Path) -> None:
    """真库两轮：裸连接 deferred（影子表保留）→ 加载模块恢复（全清）。"""
    import sqlite_vec

    db_path = tmp_path / "claims.sqlite3"
    loaded = sqlite3.connect(db_path)
    loaded.enable_load_extension(True)
    sqlite_vec.load(loaded)
    loaded.enable_load_extension(False)
    loaded.execute("CREATE VIRTUAL TABLE memory_claim_vec USING vec0(embedding float[2])")
    loaded.execute("CREATE TABLE memory_claims (id INTEGER PRIMARY KEY)")
    loaded.commit()
    # 影子表随 vec0 建表产生
    shadows = [r[0] for r in loaded.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'memory_claim_vec_%'")]
    assert shadows, "vec0 建表应产生影子表"
    loaded.close()

    # 第一轮：裸连接（不 load 模块）→ deferred，影子表保留
    bare = sqlite3.connect(db_path)
    applied: list[str] = []
    _retire_claims_tables(bare, applied)
    bare.commit()
    assert "claims_vec_drop_deferred(memory_claim_vec)" in applied
    assert bare.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'memory_claim_vec_%'"
    ).fetchone()[0] == len(shadows), "deferred 轮影子表必须保留"
    bare.close()

    # 第二轮：加载模块 → DROP 成功 + 影子表全清
    recovered = sqlite3.connect(db_path)
    recovered.enable_load_extension(True)
    sqlite_vec.load(recovered)
    recovered.enable_load_extension(False)
    applied2: list[str] = []
    _retire_claims_tables(recovered, applied2)
    recovered.commit()
    assert any("claims_tables_dropped(memory_claim_vec" in a for a in applied2)
    assert recovered.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'memory_claim_vec_%'"
    ).fetchone()[0] == 0
    recovered.close()
