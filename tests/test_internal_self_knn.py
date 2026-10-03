"""B2 内部向量自查（owner 2026-10-03 拍板：所有内部自查都要用向量，且过漏斗门）.

钉死的契约：
- 配对=向量 top-k：相似行（FakeEmbedder 同关键词→同向）成对、不相似不成对；
  k=SEMANTIC_INTERNAL_SELF_KNN_K 封顶（6 个同向邻居行只出 5 对/行去重后更少）；
- 漏斗门序：cos floor 出局（below_cos_floor 计数）、同值 no-difference skip、
  at-ceiling 且值不同放行（同键异值=自查目标，与 cross 的 at_ceil→重复语义
  刻意分歧——B2 核心分歧点回归钉）、噪音/exists 探针照旧；
- 256 行大表的 KNN 计算 numpy 路径耗时上限（防实现退化成纯 Python 全对）；
- numpy 缺席 → internal_skipped_no_numpy receipt 标记（fail-open 可见）。
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import memory_arbiter.pipeline.evidence as ev
from memory_arbiter.constants import SEMANTIC_INTERNAL_SELF_KNN_K
import tests.test_vnext_evidence as tv


def _drain_internal(tools, memory_id: int) -> tuple[list, dict]:
    assert tools.wait_semantic_worker_drained(timeout=5)
    receipt = tools._process_semantic_conflict_job(memory_id, tv._job_snapshot(tools, memory_id))
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT unit_a, unit_b, quote_a, quote_b, reason FROM internal_conflicts "
            "WHERE memory_id=? ORDER BY unit_a, unit_b", (memory_id,),
        ).fetchall()
    return rows, receipt


def test_similar_rows_pair_dissimilar_do_not(tmp_path: Path) -> None:
    """同关键词行（同向量→cos=1≥ceiling）值不同→放行；异主题行（cos≈0<
    floor）不成对。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    content = "\n".join([
        "生产数据库使用 PostgreSQL 16。",
        "生产数据库使用 PostgreSQL 17。",   # 同向(1,0) 同键异值 → 内部对
        "压测脚本入口在 scripts/load.py。",  # (0,1) 异主题 → floor 出局
    ])
    new = tools.memory_write(content=content, subject="db", tags=[], workspace="default")["data"]
    rows, receipt = _drain_internal(tools, new["id"])
    quotes = {(r["quote_a"], r["quote_b"]) for r in rows}
    assert any(
        "PostgreSQL 16" in a and "PostgreSQL 17" in b
        for a, b in quotes
    ), f"同键异值对必须入内部队: {quotes}"
    assert not any("load.py" in a or "load.py" in b for a, b in quotes), \
        "异主题行不得进入内部配对（cos floor）"
    assert receipt.get("internal_conflicts", 0) >= 1


def test_knn_k_cap(tmp_path: Path) -> None:
    """k 封顶=出度上限：每行只取 k 个最相似邻居，唯一对 ≤ n*k（稠密团里
    入度不受限——n*k/2 的度约束只对互选图成立，此处不适用）。20 行同向：
    ≤100 对 vs n²/2=190，显著小于全连即证明封顶生效。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    n = 20
    content = "\n".join(f"生产数据库使用 PostgreSQL {v}。" for v in range(10, 10 + n))
    new = tools.memory_write(content=content, subject="db", tags=[], workspace="default")["data"]
    rows, _receipt = _drain_internal(tools, new["id"])
    assert len(rows) <= n * SEMANTIC_INTERNAL_SELF_KNN_K, \
        f"k 封顶失效: {len(rows)} 对（n*k={n * SEMANTIC_INTERNAL_SELF_KNN_K}）"
    assert len(rows) < n * (n - 1) // 2, "必须显著小于全连（封顶生效）"
    assert len(rows) >= 1


def test_at_ceiling_different_value_passes(tmp_path: Path) -> None:
    """B2 核心分歧点回归钉：表面近同（FakeEmbedder 同向=cos 1.0≥ceiling）
    但值不同（5000ms vs 3000ms）→ 放行判定，不被 ceiling 当重复杀。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    content = "压测超时阈值为 5000ms。\n压测超时阈值为 3000ms。"  # 「超时」→(0.8,0.2) 同向
    new = tools.memory_write(content=content, subject="timeout", tags=[], workspace="default")["data"]
    rows, _receipt = _drain_internal(tools, new["id"])
    assert len(rows) == 1, f"同键异值对必须恰好成对: {[(r['quote_a'], r['quote_b']) for r in rows]}"


def test_same_value_ceiling_skipped(tmp_path: Path) -> None:
    """同向且同值（真重复）→ no-difference skip，不成内部对。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    content = "生产数据库使用 PostgreSQL 16。\n生产数据库使用 PostgreSQL 16。"
    new = tools.memory_write(content=content, subject="db", tags=[], workspace="default")["data"]
    rows, receipt = _drain_internal(tools, new["id"])
    assert len(rows) == 0, "同值重复不得成内部对"
    assert receipt.get("deterministic_filter", {}).get("no_difference_skipped", 0) >= 1


def test_256_row_timing(tmp_path: Path) -> None:
    """256 行表的 KNN 计算 numpy 路径耗时上限（纯 Python 全对≈13-33s，
    numpy 应 <5s——防实现退化）。FakeEmbedder 全同向最快路径。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    content = "\n".join(f"压测超时阈值为 {5000 + i}ms。" for i in range(256))
    new = tools.memory_write(content=content, subject="storm", tags=[], workspace="default")["data"]
    assert tools.wait_semantic_worker_drained(timeout=30)
    t0 = time.monotonic()
    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"256 行 KNN 计算退化: {elapsed:.1f}s（numpy 应毫秒级）"
    assert receipt.get("internal_conflicts", 0) >= 1


def test_no_numpy_marks_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy ImportError → 内部自查跳过 + receipt 标记（fail-open 可见）。"""
    import builtins

    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    new = tools.memory_write(
        content="生产数据库使用 PostgreSQL 16。\n生产数据库使用 PostgreSQL 17。",
        subject="db", tags=[], workspace="default",
    )["data"]

    real_import = builtins.__import__

    def _no_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("no numpy (test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_numpy)
    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))
    assert receipt.get("internal_skipped_no_numpy") is True
    assert receipt.get("internal_conflicts", 0) == 0
