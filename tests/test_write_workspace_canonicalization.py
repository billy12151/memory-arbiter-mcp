"""写入时 workspace 自动归一（canonical 归拢）——0.16.12 补钉（owner 要求）.

此前覆盖缺口：alias 折叠只有 move 路径测试、default 同义词只有 fallback 提交
路径测试，写入路径（remember/memory_write）的 canonical 归一没有直接钉住。
本文件按行为分层钉：

  1. 机械变体折叠：Proj-A / PROJA 写入折到已注册 canonical projA；
  2. default 同义词折叠：默认/Main 等写入折到 canonical default；
  3. raw workspace 原样保留：canonical 折叠不动 raw 列（COALESCE 回退语义）；
  4. 完全不同名不折叠（无 embedder 时无相似度归一）；
  5. embedder 在位时的相似名 admission 折叠（近邻名折到首个 canonical）。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.tools import MemoryTools  # noqa: E402


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    return MemoryTools(settings, db=MemoryDB(settings))


def _rows(tools: MemoryTools) -> list[tuple]:
    with tools.db.connection() as conn:
        return [
            (str(r["workspace"]), str(r["workspace_canonical"]))
            for r in conn.execute(
                "SELECT workspace, workspace_canonical FROM memories ORDER BY id"
            )
        ]


def test_write_folds_mechanical_variants_to_registered_canonical(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for ws in ("projA", "Proj-A", "PROJA"):
        result = tools.memory("remember", {"content": f"内容 {ws}", "subject": f"s-{ws}", "workspace": ws})
        assert result["ok"], result
    rows = _rows(tools)
    # 机械变体折到已注册 canonical；raw 列原样保留（第 2/3 条 raw 不变）
    assert [canonical for _, canonical in rows] == ["projA", "projA", "projA"]
    assert [raw for raw, _ in rows] == ["projA", "Proj-A", "PROJA"]


def test_write_folds_default_synonyms_to_canonical_default(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for ws in ("默认", "default", "Default"):
        result = tools.memory("remember", {"content": f"内容 {ws}", "subject": f"s-{ws}", "workspace": ws})
        assert result["ok"], result
    rows = _rows(tools)
    assert [canonical for _, canonical in rows] == ["default", "default", "default"], rows


def test_write_keeps_distinct_workspaces_unfolded_without_embedder(tmp_path: Path) -> None:
    """无 embedder 时没有相似度 admission——完全不同名必须各自成桶。"""
    tools = make_tools(tmp_path)
    for ws in ("projA", "billing", "oncall"):
        result = tools.memory("remember", {"content": f"内容 {ws}", "subject": f"s-{ws}", "workspace": ws})
        assert result["ok"], result
    assert [canonical for _, canonical in _rows(tools)] == ["projA", "billing", "oncall"]


def test_write_admission_folds_near_neighbor_workspace_names(tmp_path: Path) -> None:
    """embedder 在位时，近邻 workspace 名（字符直方图近似）由 admission 折到
    首个已注册 canonical——单桶归拢语义（0.16.12 P3 期间实证并在此钉住）。"""
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sqlite-vec not installed")

    from memory_arbiter.embedder import EmbedResult

    class CharHistogramEmbedder:
        embedding_space_id = "char-histogram-space"
        dim = 32
        last_encode_error = None

        @staticmethod
        def embed_text(prefix: str, body: str, max_body_chars=None) -> EmbedResult:
            vector = [0.0] * 32
            text = body.casefold()  # 直方图只对 body（prefix 不参与分箱）
            for ch in text:
                vector[ord(ch) % 32] += 1.0
            return EmbedResult(vector, False, len(text), len(text))

    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "v.sqlite3", backup_jsonl=tmp_path / "vb.jsonl",
        embedding_model_path=model,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = CharHistogramEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(CharHistogramEmbedder.dim) == []
    db.init_vec_index_state(
        CharHistogramEmbedder.embedding_space_id, True, active_dim=CharHistogramEmbedder.dim,
    )
    # wsA 先注册；wsA2 与 wsA 直方图近邻（多一字符，cos≈0.866/距离≈0.134
    # < 0.25，body-only 直方图口径）→ 折到 wsA
    for ws in ("wsA", "wsA2"):
        result = tools.memory_write(
            content=f"内容 {ws}", subject=f"s-{ws}", workspace=ws,
            source_type="agent_generated", tags=[],
        )
        assert result["ok"], result
    rows = _rows(tools)
    assert rows[0][1] == "wsA"
    assert rows[1][1] == "wsA", f"近邻名应被 admission 归一: {rows}"
    assert rows[1][0] == "wsA2"  # raw 保留
    # 完全不同的名字不折叠
    result = tools.memory_write(
        content="内容 billing", subject="s-billing", workspace="billing",
        source_type="agent_generated", tags=[],
    )
    assert result["ok"], result
    assert _rows(tools)[2] == ("billing", "billing")
