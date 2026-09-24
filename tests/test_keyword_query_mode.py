"""检索线 K1：关键词模式判定（is_keyword_query）+ 中间带救济（方案 §3a/§3b）。

判据（owner 2026-09-24 拍板 1/8）：空格分隔、2~8 个 token、全部 token
1~4 字纯 CJK。救济带 [COS_RECALL_FLOOR, COS_MIDBAND_CEIL)、仅
evidence-only 行（无 _lexical_rank）、content/subject 含任一关键词；
boost 落在 _fusion_score（×300 → final +3.0），先于 _soft_rerank。

端到端用定向 2 维 embedder（手册→0.6/0.8，桥接查询→东，alpha→东，
beta→北）验证救济把中间带 evidence-only 行抬过 8.25 页门，而非关键词
查询（单长 token）同一行留在页外。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import KEYWORD_RESCUE_BOOST
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import MemoryRecord
from memory_arbiter.search import (
    _apply_keyword_rescue,
    _soft_rerank,
    is_keyword_query,
)
from memory_arbiter.tools import MemoryTools

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# §3a 真值表（纯函数）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "向量 唯一键 冲突",
        "金营 智能配券 场景",       # 2/4/2 —— 拍板 8：4 字 token 放宽
        "操作纪律 桥接脚本",         # 4/4 —— B07 原题形态
        "催收 辱骂 侮辱",
        "甲 乙",                    # 1 字 token 允许
    ],
)
def test_keyword_query_true(query: str) -> None:
    assert is_keyword_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "",                          # 空
        "单",                        # 单 token
        "甲乙",                      # 单 token（无空格）
        "网站备案手续办完了吗",        # 单长 token
        "deploy pipeline is green",  # 非 CJK
        "alpha 冲突",                # ASCII token 混入
        "发版 之前要跑哪些检查",       # 短词+长句混合
        "催收 侮辱，",                # 标点 token（is_pure_cjk_token 漏、逐字判定拦）
        "向量 唯一键 冲突，",          # 标点 token
        "桥 接 自 检 合 并 发 版 检 查",  # 9 token > 上限 8
    ],
)
def test_keyword_query_false(query: str) -> None:
    assert is_keyword_query(query) is False


# ---------------------------------------------------------------------------
# §3b 救济矩阵（纯内存，in-place）
# ---------------------------------------------------------------------------

def _row(
    mid: int,
    cos: "float | None",
    *,
    lexical: "int | None" = None,
    content: str = "",
    subject: str = "t",
    fusion: float = 0.0164,
) -> dict:
    row: dict = {
        "id": mid,
        "subject": subject,
        "content": content,
        "_fusion_score": fusion,
    }
    if cos is not None:
        row["_evidence_best_score"] = cos
    if lexical is not None:
        row["_lexical_rank"] = lexical
    return row


def test_rescue_fires_on_midband_evidence_only_row_with_keyword() -> None:
    pool = [_row(1, 0.65, content="这里讲桥接协议")]
    _apply_keyword_rescue("桥接 自检", pool)
    assert pool[0].get("_keyword_rescued") is True
    assert pool[0]["_fusion_score"] == pytest.approx(0.0164 + KEYWORD_RESCUE_BOOST)


def test_rescue_matches_subject_and_any_keyword() -> None:
    pool = [_row(1, 0.60, content="无关内容", subject="自检清单汇总")]
    _apply_keyword_rescue("桥接 自检", pool)
    assert pool[0].get("_keyword_rescued") is True


def test_rescue_skips_without_keyword_hit() -> None:
    pool = [_row(1, 0.65, content="完全无关的正文")]
    _apply_keyword_rescue("桥接 自检", pool)
    assert "_keyword_rescued" not in pool[0]
    assert pool[0]["_fusion_score"] == pytest.approx(0.0164)


def test_rescue_skips_lexical_rows() -> None:
    pool = [_row(1, 0.65, lexical=1, content="含桥接的词法行")]
    _apply_keyword_rescue("桥接 自检", pool)
    assert "_keyword_rescued" not in pool[0]


@pytest.mark.parametrize("cos", [0.51, 0.75, 0.90, None])
def test_rescue_skips_out_of_band(cos: "float | None") -> None:
    pool = [_row(1, cos, content="含桥接的行")]
    _apply_keyword_rescue("桥接 自检", pool)
    assert "_keyword_rescued" not in pool[0]


def test_rescue_band_is_half_open_on_floor() -> None:
    pool = [_row(1, 0.52, content="贴线但含桥接")]
    _apply_keyword_rescue("桥接 自检", pool)
    assert pool[0].get("_keyword_rescued") is True


def test_non_keyword_query_never_rescues() -> None:
    pool = [_row(1, 0.65, content="含桥接自检的行")]
    _apply_keyword_rescue("桥接自检说明", pool)  # 单 token：非关键词模式
    assert "_keyword_rescued" not in pool[0]


def test_four_char_token_matches_whole_only() -> None:
    """owner 2026-09-24 追加拍板：4 字 token 整词匹配、不切词——查不到
    说明查询关键词不对，不强行匹配。目标内容只有「桥接」没有连写的
    「桥接脚本」时不救（B07 原查询形态即此结果）；内容含整词才救。"""
    pool = [
        _row(1, 0.65, content="飞天小虾私有桥接通道自检"),   # 只有「桥接」无「桥接脚本」
        _row(2, 0.65, content="私有桥接脚本自检说明"),       # 含整词「桥接脚本」
    ]
    _apply_keyword_rescue("操作纪律 桥接脚本", pool)
    assert "_keyword_rescued" not in pool[0]
    assert pool[1].get("_keyword_rescued") is True


def test_rescued_row_floats_above_same_fusion_peer_in_soft_rerank() -> None:
    """BOOST 生效方向：同融合分的两行，被救济者 final 分高 3.0。"""
    rescued = _row(1, 0.65, content="含桥接", fusion=0.0164, subject="记录甲")
    _apply_keyword_rescue("桥接 自检", [rescued])
    peer = _row(2, cos=None, subject="记录乙", fusion=0.0164)
    ranked = _soft_rerank("桥接 自检", [rescued, peer])
    scores = {int(r["id"]): float(r["_final_score"]) for r in ranked}
    assert scores[1] - scores[2] == pytest.approx(KEYWORD_RESCUE_BOOST * 300.0)


# ---------------------------------------------------------------------------
# 端到端：救济把中间带 evidence-only 行抬过 8.25 页门
# ---------------------------------------------------------------------------

class _BandEmbedder:
    embedding_space_id = "retrieval-k1-band-space"
    dim = 2
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars: "int | None" = None) -> EmbedResult:
        text = f"{prefix} {body}".casefold()
        if "手册" in text:
            vector = [0.6, 0.8]        # 与东向查询真余弦 0.6（中间带）
        elif "桥接" in text:
            vector = [1.0, 0.0]        # 查询方向
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
    model = tmp_path / "fake-k1.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "k1.sqlite3",
        backup_jsonl=tmp_path / "k1-backup.jsonl",
        embedding_model_path=model,
        client="c", agent_id="a",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = _BandEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(_BandEmbedder.dim) == []
    db.init_vec_index_state(_BandEmbedder.embedding_space_id, True, _BandEmbedder.dim)
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


def _seed_library(tmp_path: Path) -> "tuple[MemoryTools, int, int]":
    tools = _make_tools(tmp_path)
    exact = _write_and_index(tools, "桥接 自检", "桥接自检操作手册")       # subject 命中查询 → 精确置顶
    rescued = _write_and_index(
        tools, "飞天小虾私有通道记录", "私有桥接自检说明 手册",          # 中间带 + content 含关键词
    )
    _write_and_index(tools, "另一条手册记录", "私有通道说明 手册")        # 同余弦、无关键词（对照）
    _write_and_index(tools, "自检须知汇总", "自检须知汇总文档")           # surface 词法弱命中（对照）
    return tools, exact, rescued


def test_rescue_carries_midband_row_past_relevance_floor(tmp_path: Path) -> None:
    tools, exact, rescued = _seed_library(tmp_path)
    result = tools.memory_search(query="桥接 自检", debug_ranking=True)
    results = result["data"]["results"]
    ids = [int(row["id"]) for row in results]
    by_id = {int(row["id"]): row for row in results}
    # 精确置顶第一；被救济行（中间带 evidence-only + 关键词命中）越过
    # 8.25 页门进页并带内部旗标；同余弦、无关键词命中的对照行（除救济
    # 外逐位同条件）留在页外——差值就是 KEYWORD_RESCUE_BOOST。
    assert ids and ids[0] == exact
    assert rescued in ids
    assert by_id[exact].get("_exact_match") is True
    assert by_id[rescued].get("_keyword_rescued") is True
    assert any("keyword rescue" in note for note in by_id[rescued]["_ranking_notes"])
    assert 3 not in ids  # 另一条手册记录：同余弦、无关键词（对照）
