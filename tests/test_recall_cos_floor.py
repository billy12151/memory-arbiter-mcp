"""检索线 K2：向量准入线 COS_RECALL_FLOOR=0.52（方案 §3c，owner 拍板 9）。

evidence-only 候选（无词法席位的纯向量行）best 行真余弦 < 0.52 整条
不进结果；词法候选豁免（排名+8.25 双保险维持现状）；真余弦缺失
fail-open；expired 审计路径豁免（宁滥勿缺，沿 8.25 门既有口径）。

标定依据：0.58 实测删 A04(0.567)/A12(0.543) 两条已在 top10 内的
relevant（R@10 0.889 击穿 ≥0.93 门）；0.52 保三条贴线 relevant 全员。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import RECALL_POOL_CAP
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import MemoryRecord
from memory_arbiter.search import _wide_recall
from memory_arbiter.tools import MemoryTools

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import math


def _unit(vector: "list[float]") -> "list[float]":
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


class _FloorEmbedder:
    """东向查询（alpha） vs 「新潮」行 cos 0.4（线内删除样本）vs 北向杂讯."""

    embedding_space_id = "retrieval-k2-floor-space"
    dim = 2
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars: "int | None" = None) -> EmbedResult:
        text = f"{prefix} {body}".casefold()
        if "贴线" in text:
            # 0.52 整值在 float64 归一后落 0.51998（线下），取 0.5205
            # 线上样本与 0.4 线下样本夹出开边界（R2-P2-7）
            vector = _unit([0.5205, 0.8538])   # 真余弦 ≈0.5205 > 0.52
        elif "新潮" in text:
            vector = _unit([0.4, 0.9165])   # 真余弦 ≈0.4 < 0.52
        elif "alpha" in text:
            vector = [1.0, 0.0]
        else:
            vector = [0.0, 1.0]
        return EmbedResult(vector, False, 1, 1)

    @classmethod
    def embed_texts(cls, bodies: "list[str]", max_body_chars: "int | None" = None) -> list[EmbedResult]:
        return [cls.embed_text("", body) for body in bodies]


def _make_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake-k2.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "k2.sqlite3",
        backup_jsonl=tmp_path / "k2-backup.jsonl",
        embedding_model_path=model,
        client="c", agent_id="a",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = _FloorEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(_FloorEmbedder.dim) == []
    db.init_vec_index_state(_FloorEmbedder.embedding_space_id, True, _FloorEmbedder.dim)
    return tools


def _write_and_index(tools: MemoryTools, subject: str, content: str) -> int:
    mid, _warnings = tools.db.insert_memory(MemoryRecord(
        content=content, agent_id="a", workspace="main", tags=[],
        source_type="agent_generated", subject=subject,
    ))
    assert mid is not None
    outcome = tools._evidence.index_memory(int(mid))
    assert outcome.get("status") == "indexed", outcome
    return int(mid)


def _pool(tools: MemoryTools, query: str, *, status_filter: str = "active",
          query_embedding: "list[float] | None" = None) -> "dict[int, dict]":
    if status_filter == "active":
        status_clause = "m.status = 'active'"
        like_clause = "status = 'active'"
    else:
        status_clause = "m.status NOT IN ('active','deleted')"
        like_clause = "status NOT IN ('active','deleted')"
    if query_embedding is None:
        query_embedding = tools._read_pipeline._auto_embed(query, None, [])
    return {
        int(row["id"]): row
        for row in _wide_recall(
            tools.db, query, "main", None, status_clause, like_clause,
            status_filter=status_filter, query_embedding=query_embedding,
            pool_cap=RECALL_POOL_CAP,
        )
    }


def test_floor_just_above_line_is_kept(tmp_path: Path) -> None:
    """线上（≈0.5205）保留、线下（0.4）删除——开边界被夹在两者之间
    （0.52 整值 float64 归一后落 0.51998，精确贴线不可稳定表示）。"""
    tools = _make_tools(tmp_path)
    boundary = _write_and_index(tools, "贴线主题记录", "贴线方案说明")
    pool = _pool(tools, "alpha query")
    assert boundary in pool


def test_subfloor_evidence_only_row_is_dropped(tmp_path: Path) -> None:
    tools = _make_tools(tmp_path)
    subfloor = _write_and_index(tools, "无关旧主题", "新潮方案说明")
    pool = _pool(tools, "alpha query")
    assert subfloor not in pool


def test_subfloor_row_with_lexical_seat_is_exempt(tmp_path: Path) -> None:
    tools = _make_tools(tmp_path)
    kept = _write_and_index(tools, "alpha 方案汇总", "新潮方案二")
    pool = _pool(tools, "alpha query")
    assert kept in pool
    assert pool[kept].get("_lexical_rank") is not None


def test_missing_true_cosine_fails_open(tmp_path: Path) -> None:
    """向量拉取失败（行向量缺席）→ 无真余弦 → 不判入线，行保留。"""
    tools = _make_tools(tmp_path)
    subfloor = _write_and_index(tools, "无关旧主题", "新潮方案说明")
    tools.db.evidence.row_vectors_for_ids = lambda row_ids, **_: {}
    pool = _pool(tools, "alpha query")
    assert subfloor in pool


def test_expired_audit_path_is_exempt(tmp_path: Path) -> None:
    """expired 审计是宁滥勿缺的遍历语义——准入线不生效（沿 8.25 口径）。"""
    tools = _make_tools(tmp_path)
    subfloor = _write_and_index(tools, "无关旧主题", "新潮方案说明")
    conn = tools.db._new_connection()
    try:
        conn.execute("UPDATE memories SET status = 'expired' WHERE id = ?", (subfloor,))
        conn.commit()
    finally:
        conn.close()
    # row_knn 过滤的是发布时快照 v.parent_status：状态变更后重索引刷新
    outcome = tools._evidence.index_memory(subfloor)
    assert outcome.get("status") == "indexed", outcome
    pool = _pool(tools, "alpha query", status_filter="expired")
    assert subfloor in pool


def test_end_to_end_page_drops_subfloor_but_keeps_lexical(tmp_path: Path) -> None:
    tools = _make_tools(tmp_path)
    subfloor = _write_and_index(tools, "无关旧主题", "新潮方案说明")
    kept = _write_and_index(tools, "alpha 方案汇总", "新潮方案二")
    result = tools.memory_search(query="alpha query", debug_ranking=True)
    ids = [int(row["id"]) for row in result["data"]["results"]]
    assert subfloor not in ids
    assert kept in ids          # subject 强命中越过 8.25 页门
