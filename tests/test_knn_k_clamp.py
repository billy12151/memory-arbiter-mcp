"""0.17.1 修复批 A1：vec0 k 硬上限 clamp（深 offset 向量通道静默消失的修复）.

背景（对抗性复核实证）：recall 的 pool_cap=(offset+limit+1)，evidence 通道
k=pool_cap*16。k>4096 时 sqlite-vec 报 "k value in knn query too large"，
被 row_knn 的 except sqlite3.Error 吞成空结果 —— limit=100 时 offset=155
返 100 条、offset=156 返 0 条（语义通道静默消失，无 warning）。

钉死契约：
- row_knn / summary KNN 的 k 在 VEC0_MAX_K 处 clamp（超限不报错、返非空）；
- k=VEC0_MAX_K 与 k=极大值返回同一集合（clamp 后等价）；
- 产品面 find(offset=200, limit=100) 不再全灭；
- 常量引用断言：VEC0_MAX_K 漂移即失败（防常量被改而测试仍绿）。
"""
from __future__ import annotations

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import VEC0_MAX_K
from memory_arbiter.tools import MemoryTools

pytest.importorskip("sqlite_vec")

import tests.test_vnext_evidence as tv  # noqa: E402
from tests.test_vnext_evidence import FakeEmbedder  # noqa: E402


def make_tools(tmp_path, **kw) -> MemoryTools:
    return tv.make_tools(tmp_path, **kw)


def _seed(tools: MemoryTools, count: int, workspace: str = "w") -> list[int]:
    ids = []
    for i in range(count):
        res = tools.memory_write(
            content=f"债务转移 相关条目编号 {i}",
            subject=f"债务{i}",
            workspace=workspace,
            tags=["x"],
        )
        assert res.get("ok"), res
        ids.append(int(res["data"]["id"]))
    assert tools.wait_semantic_worker_drained(timeout=120)
    return ids


def test_vec0_max_k_constant_is_the_real_ceiling(tmp_path) -> None:
    """常量引用钉：VEC0_MAX_K 就是 sqlite-vec 实测上限（漂移即红）。"""
    assert VEC0_MAX_K == 4096


def test_row_knn_clamps_over_limit_k(tmp_path) -> None:
    """k=5000（超限）在 400 行库上必须返回非空 —— clamp 生效。"""
    tools = make_tools(tmp_path)
    _seed(tools, 400)
    q = FakeEmbedder.embed_text(prefix="", body="债务转移").embedding
    rows = tools.db.row_knn(q, k=5000, workspace="w")
    assert rows, "k=5000 必须被 clamp 到 4096 并返回结果（此前整条查询失败返 []）"


def test_row_knn_clamp_boundary_equivalence(tmp_path) -> None:
    """k=VEC0_MAX_K 与 k=100000 返回同一集合（clamp 后等价，边界钉）。"""
    tools = make_tools(tmp_path)
    _seed(tools, 300)
    q = FakeEmbedder.embed_text(prefix="", body="债务转移").embedding
    at_limit = tools.db.row_knn(q, k=VEC0_MAX_K, workspace="w")
    over_limit = tools.db.row_knn(q, k=100_000, workspace="w")
    assert at_limit, "k=4096 应返回结果"
    assert [r["id"] for r in at_limit] == [r["id"] for r in over_limit]


def test_find_deep_offset_no_longer_goes_dark(tmp_path) -> None:
    """产品面：limit=100, offset=200（k=4816>4096）此前返 0 条，现返非空。"""
    tools = make_tools(tmp_path)
    _seed(tools, 400)
    res = tools.memory("find", {"query": "债务转移", "limit": 100, "offset": 200, "workspace": "w"})
    assert res.get("ok"), res
    results = res["data"]["results"]
    assert len(results) > 0, "深 offset 不得再全灭（clamp 保通道）"


def test_summary_knn_clamps_over_limit_k(tmp_path) -> None:
    """第二入口（summary KNN）同 clamp —— API 层防御。"""
    tools = make_tools(tmp_path)
    _seed(tools, 3)
    q = FakeEmbedder.embed_text(prefix="", body="债务").embedding
    rows = tools.db.memory_summary_knn(
        q, k=100_000, exclude_memory_id=0, workspace_canonical="w",
    )
    assert rows, "summary KNN 超限 k 必须被 clamp 且返回非空"


def test_row_knn_stderr_notes_the_clamp(tmp_path, capsys) -> None:
    """clamp 发生时 stderr 留一行（与 missing_row_vector_rows 排障先例同族）。"""
    tools = make_tools(tmp_path)
    _seed(tools, 5)
    q = FakeEmbedder.embed_text(prefix="", body="债务").embedding
    tools.db.row_knn(q, k=9000, workspace="w")
    err = capsys.readouterr().err
    assert "clamp" in err and "9000" in err
