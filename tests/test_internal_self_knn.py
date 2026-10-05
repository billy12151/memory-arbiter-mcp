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
from tests.test_vnext_evidence import FakeEmbedder


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


def test_asymmetric_neighbour_pair_orientation(tmp_path: Path) -> None:
    """2026-10-05 审查修复：KNN 配对方向与扫描腿约定对齐（unit_index 升序）。

    构造非对称邻域：L0 的 top-5 被 5 个近邻（-5° 同族）占满（L6 不在），
    L6 的 top-5 含 L0（45°，cos≈0.707 过 floor）。修复前该对由 L6 发现、
    以 (L6,L0) 方向落库，而扫描腿 _examine_internal 恒以 i<j 探测——
    exists 必 miss，同一矛盾下一轮 kick 再落 (L0,L6) 一行（UNIQUE 有序
    键不拦）。修复后写侧归一 unit_a < unit_b，两侧共享同一身份。
    """
    import math

    class _AsymEmbedder(FakeEmbedder):
        near = (math.cos(math.radians(-5)), math.sin(math.radians(-5)))
        vectors = {}

        @classmethod
        def embed_text(cls, prefix, body, max_body_chars=None):
            from memory_arbiter.embedder import EmbedResult
            v = cls.vectors.get(body.strip()) or [0.0, 1.0]
            return EmbedResult(list(v), False, len(body), len(body))

    lines = [f"网关超时阈值为 {5000 + i}ms。" for i in range(7)]
    # L0 独占 0°；L1..L5 同在 -5°（与 L0 cos≈0.996、互相 cos=1）；L6 在 45°
    _AsymEmbedder.vectors = {
        lines[0]: (1.0, 0.0),
        **{lines[i]: _AsymEmbedder.near for i in range(1, 6)},
        lines[6]: (math.cos(math.radians(45)), math.sin(math.radians(45))),
    }
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools._embedder = _AsymEmbedder()
    tools.settings.semantic_conflict_on_write = "off"
    new = tools.memory_write(
        content="\n".join(lines), subject="asym", tags=[], workspace="default",
    )["data"]
    rows, _receipt = _drain_internal(tools, new["id"])
    assert rows, "非对称邻域至少 L0×L6 一对必须落地"
    # 核心方向钉：写侧落的每个内部对都与扫描腿 i<j 约定同身份
    bad = [(r["unit_a"], r["unit_b"]) for r in rows if int(r["unit_a"]) > int(r["unit_b"])]
    assert not bad, f"KNN 发现方向未归一（扫描腿将以反向再落一行）: {bad}"
    # L0×L6 对确实存在（且其 quote 方向与身份一致）
    pair = [
        r for r in rows
        if ("5000ms" in (r["quote_a"] or "") and "5006ms" in (r["quote_b"] or ""))
        or ("5000ms" in (r["quote_b"] or "") and "5006ms" in (r["quote_a"] or ""))
    ]
    assert pair, f"L0×L6 对缺失: {[(r['quote_a'], r['quote_b']) for r in rows]}"
    # 扫描腿口径探测（i<j 有序）必须命中写侧已落的行
    with tools.db.connection() as conn:
        hit = tools.db.internal_conflicts.exists_on_conn(
            conn, int(new["id"]), 1, pair[0]["unit_a"], pair[0]["unit_b"],
        )
    assert hit, "扫描腿 exists 探针（i<j 有序键）必须命中写侧落库的身份"


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


def _table_content(rows: int) -> str:
    lines = ["| 配置项 | 数值 |", "| --- | --- |"]
    lines += [f"| 超时阈值{i} | {1000 + i}ms |" for i in range(rows)]
    return "\n".join(lines)


def _memory_row_count(tools, memory_id: int) -> int:
    with tools.db.connection() as conn:
        return conn.execute(
            "SELECT count(*) FROM memory_row WHERE memory_id=?", (memory_id,)
        ).fetchone()[0]


def test_giant_table_exempted_from_row_index(tmp_path: Path) -> None:
    """B3 行为钉：>100 行表格段不建行向量（仅 subject 行保留）。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    new = tools.memory_write(
        content=_table_content(120), subject="big table", tags=[], workspace="default",
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=10)
    assert _memory_row_count(tools, new["id"]) == 1, "超 100 行表格不得建行向量（仅 subject 保留）"


def test_mixed_memory_each_table_counts_own(tmp_path: Path) -> None:
    """混合记忆行为钉：120 行表豁免、散文与小表照常索引（各数各的）。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    content = (
        _table_content(120)
        + "\n压测超时阈值为 5000ms。\n压测超时阈值为 3000ms。\n散文三。\n散文四。\n散文五。"
        + "\n" + _table_content(10)
    )
    new = tools.memory_write(content=content, subject="mixed", tags=[], workspace="default")["data"]
    assert tools.wait_semantic_worker_drained(timeout=10)
    n = _memory_row_count(tools, new["id"])
    # subject + 5 散文 + 小表残余（短单元格可能被 ROW_MIN_CHARS 折掉部分）——
    # 豁免大表后基数 6~15，且绝无 120 大行
    assert 6 <= n <= 15, f"散文与小表行必须照常索引: {n}"


def test_small_table_still_indexed(tmp_path: Path) -> None:
    """≤100 行表格段照常嵌入（豁免不误伤）。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    new = tools.memory_write(
        content=_table_content(100), subject="ok table", tags=[], workspace="default",
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=10)
    assert _memory_row_count(tools, new["id"]) >= 90, "≤100 表格行必须照常索引（短单元格折损余量）"


def test_giant_table_filter_pure() -> None:
    """B3 纯函数钉：段聚合按 row_index 连续、各数各的、阈值边界。"""
    from memory_arbiter.pipeline.evidence import _filter_exempted_segments
    from memory_arbiter.rowseg import segment_rows

    # 120 行表 → 120 豁免（subject 保留）
    segs = segment_rows("s", _table_content(120))
    kept, n = _filter_exempted_segments(segs)
    assert n == 120
    assert all(str(getattr(k, "kind", "")) != "table_row" for k in kept)

    # 100 行表 → 零豁免
    _k, n100 = _filter_exempted_segments(segment_rows("s", _table_content(100)))
    assert n100 == 0

    # 夹散文断段各数各的：120 表 + 一句散文 + 80 表 → 只豁免 120 段
    content = _table_content(120) + "\n中间一句散文注释。\n" + _table_content(80)
    _k, n_split = _filter_exempted_segments(segment_rows("s", content))
    assert n_split == 120, f"断段后 80 行段不得豁免: {n_split}"


def test_upgrade_boot_backfills_row_vectors(tmp_path: Path) -> None:
    """升级路径钉（owner 2026-10-05 要求确保）：0.16 老库（无语句级向量）
    升级后 boot 回填自动写入 memory_row/memory_row_vec——直调守护线程
    执行体（生产 boot 后台线程同一段代码）。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    ids = [tools.memory_write(
        content=f"升级路径探针 {i}：超时阈值 {1000 + i}ms。",
        subject="upgrade-probe", tags=[], workspace="default",
    )["data"]["id"] for i in range(3)]
    assert tools.wait_semantic_worker_drained(timeout=10)
    with tools.db.connection() as conn:
        conn.execute("DELETE FROM memory_row")
        conn.execute("DELETE FROM memory_row_vec")
        conn.commit()
        assert conn.execute("SELECT count(*) FROM memory_row").fetchone()[0] == 0

    from memory_arbiter.db import MemoryDB
    from memory_arbiter.tools import MemoryTools

    db2 = MemoryDB(tools.settings)
    tools2 = MemoryTools(settings=tools.settings, db=db2)
    db2.ensure_vec_tables(tv.FakeEmbedder.dim)
    db2.init_vec_index_state("fake-vnext-space", True, active_dim=tv.FakeEmbedder.dim)
    tools2._run_boot_backfills(tv.FakeEmbedder)  # 守护线程执行体，同步直调
    with db2.connection() as conn:
        rows = conn.execute("SELECT count(*) FROM memory_row").fetchone()[0]
        vecs = conn.execute("SELECT count(*) FROM memory_row_vec").fetchone()[0]
        covered = conn.execute("SELECT count(DISTINCT memory_id) FROM memory_row").fetchone()[0]
    assert covered == 3, f"全部记忆应被回填覆盖: {covered}"
    assert rows >= 3 and vecs >= 3
