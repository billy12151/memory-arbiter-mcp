"""0.15.9 recall-quality tests (mema 923 §4): page floor F=8.1, CJK phrase
channel, channel-3 per-token surface recall without the pool-full short-circuit."""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import MemoryRecord
from memory_arbiter.search import _sanitize_fts_query, search_memories


def _db(tmp_path: Path) -> MemoryDB:
    return MemoryDB(Settings(db_path=tmp_path / "memory.db", backup_jsonl=tmp_path / "backup.jsonl"))


def _mem(db: MemoryDB, subject: str, content: str, tags: tuple[str, ...] = ()) -> int:
    mid, _ = db.insert_memory(
        MemoryRecord(content=content, subject=subject, agent_id="a", workspace="w", tags=list(tags)), "w",
    )
    assert mid is not None
    return int(mid)


def test_sanitize_fts_query_quotes_cjk_phrase_first() -> None:
    expr = _sanitize_fts_query("债务转移 债权人同意")
    assert '"债务转移" OR 债务转 OR 务转移' in expr
    assert '"债权人同意" OR 债权人 OR 权人同 OR 人同意' in expr
    # ASCII tokens keep the plain quoted-phrase form (unchanged since v0.3).
    assert _sanitize_fts_query("sqlite vec0") == '"sqlite" AND "vec0"'
    # 2-char CJK tokens still yield no trigram and no phrase (LIKE channels own them).
    assert _sanitize_fts_query("催收 辱骂") == ""


def test_page_floor_cuts_weak_but_keeps_strong(tmp_path: Path) -> None:
    db = _db(tmp_path)
    strong = _mem(db, "债务转移的最新规则", "债务转移 债权人同意 的完整条文记录。")
    weak = _mem(db, "周报", "本周杂谈，顺带提了一句债务转移，别的没了。")
    out = search_memories(db, "债务转移 债权人同意", limit=10)
    ids = [int(r["id"]) for r in out.results]
    assert strong in ids
    assert weak not in ids
    assert out.retrieval_mode == "direct"


def test_page_floor_all_below_reports_empty_with_warning(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _mem(db, "周报", "本周杂谈，顺带提了一句债务转移，别的没了。")
    out = search_memories(db, "债务转移 债权人同意", limit=10)
    assert out.results == []
    assert out.retrieval_mode == "direct"
    assert any("relevance floor" in w for w in out.warnings)


def test_page_floor_exempt_for_expired_audit(tmp_path: Path) -> None:
    db = _db(tmp_path)
    mid = _mem(db, "周报", "本周杂谈，顺带提了一句债务转移，别的没了。")
    with db.connection() as conn:
        conn.execute("UPDATE memories SET status='superseded' WHERE id=?", (mid,))
        conn.commit()
    out = search_memories(db, "债务转移 债权人同意", limit=10, status_filter="expired")
    assert mid in [int(r["id"]) for r in out.results]


def test_channel3_per_token_surface_recall_two_char_token(tmp_path: Path) -> None:
    db = _db(tmp_path)
    target = _mem(db, "催收行为规范", "禁止辱骂债务人，催收话术必须合规。")
    out = search_memories(db, "催收 辱骂", limit=10)
    assert target in [int(r["id"]) for r in out.results]


def test_channel3_not_short_circuited_by_full_pool(tmp_path: Path) -> None:
    db = _db(tmp_path)
    for i in range(60):
        _mem(db, f"杂记 {i}", f"债务转移相关备注第 {i} 条，内容各不相同，编号 {i}")
    # surface 命中必须同时覆盖 query 主词（anchors strong）才能过线；
    # 部分 token 覆盖的 surface 命中进池但被线切属预期（宁缺毋滥）。
    target = _mem(db, "催收与债务转移行为规范", "禁止辱骂债务人，催收话术必须合规。")
    out = search_memories(db, "债务转移 催收", limit=10)
    ids = [int(r["id"]) for r in out.results]
    assert target in ids


def test_channel3_stopwords_do_not_manufacture_surface_hits(tmp_path: Path) -> None:
    db = _db(tmp_path)
    filler = _mem(db, "活动总结", "记录一条日常安排")
    target = _mem(db, "催收规范", "规范催收流程与话术")
    out = search_memories(db, "可以 催收", limit=10)
    ids = [int(r["id"]) for r in out.results]
    assert target in ids
    assert filler not in ids


def test_query_floor_layered_vec_only_cosine_exemption() -> None:
    """0.17.0 分层门槛：复合线管词法锚定候选，evidence-only 行走余弦线。

    数据锚：recall-v3-len en→zh 15 个贴线 gold 复合 7.51-8.22、余弦
    0.509-0.681（xlang-floor-policies.json 豁免政策与全取消召回逐项相等）。
    """
    from memory_arbiter.search import _passes_query_recall_floor

    # 词法锚定（FTS/surface）：复合分线照旧，豁免不适用
    assert _passes_query_recall_floor({"_final_score": 8.3, "_lexical_rank": 2}) is True
    assert _passes_query_recall_floor({"_final_score": 8.0, "_lexical_rank": 2}) is False
    # evidence-only：余弦线接管（0.48 = COS_RECALL_FLOOR）——但仅
    # alloglottic（非 CJK 主导）查询开闸，见 gate 测试
    assert _passes_query_recall_floor({
        "_final_score": 8.0, "_lexical_rank": None, "_evidence_best_score": 0.5,
    }, alloglottic=True) is True
    assert _passes_query_recall_floor({
        "_final_score": 8.0, "_lexical_rank": None, "_evidence_best_score": 0.47,
    }, alloglottic=True) is False
    # 余弦不可得（K2 fail-open 同款）：保守走复合线，不凭空放行
    assert _passes_query_recall_floor({
        "_final_score": 8.0, "_lexical_rank": None, "_evidence_best_score": None,
    }, alloglottic=True) is False
    # 同一行在 CJK 查询下（闸关）：复合线照拦
    assert _passes_query_recall_floor({
        "_final_score": 8.0, "_lexical_rank": None, "_evidence_best_score": 0.5,
    }) is False
    # 字段全缺（极端防御）：不过线
    assert _passes_query_recall_floor({"_final_score": 7.0}) is False


def test_query_floor_boundary_equality_pins_comparators() -> None:
    """恰等边界钉 >= 比较符：与 K2 准入门（< COS_RECALL_FLOOR 拒）无缝。"""
    from memory_arbiter.constants import COS_RECALL_FLOOR, QUERY_RECALL_SCORE_FLOOR
    from memory_arbiter.search import _passes_query_recall_floor

    assert _passes_query_recall_floor({"_final_score": QUERY_RECALL_SCORE_FLOOR}) is True
    assert _passes_query_recall_floor({
        "_final_score": 0.0, "_lexical_rank": None,
        "_evidence_best_score": COS_RECALL_FLOOR,
    }, alloglottic=True) is True
    assert _passes_query_recall_floor({
        "_final_score": 0.0, "_lexical_rank": None,
        "_evidence_best_score": COS_RECALL_FLOOR - 0.001,
    }, alloglottic=True) is False


def test_query_floor_exemption_gated_by_alloglottic_query() -> None:
    """豁免闸（owner「无关召回涨了就修」）：非 CJK 主导查询才开豁免。

    数据锚：主考卷 zh 查询豁免 51 行 0 gold 全噪音（C01/C02 legal 探针
    28 条误召回），en→zh 豁免 15 gold——闸把两者分开。
    """
    from memory_arbiter.search import _query_non_cjk_dominant, _passes_query_recall_floor

    row = {"_final_score": 8.0, "_lexical_rank": None, "_evidence_best_score": 0.6}
    # zh 查询：豁免关闭，复合线照拦（C01 legal 防线）
    assert _passes_query_recall_floor(row, alloglottic=False) is False
    # en 查询：豁免开启，余弦线放行
    assert _passes_query_recall_floor(row, alloglottic=True) is True
    # 词法锚定行任何查询下都走复合线
    lex_row = {"_final_score": 8.0, "_lexical_rank": 2, "_evidence_best_score": 0.6}
    assert _passes_query_recall_floor(lex_row, alloglottic=True) is False

    assert _query_non_cjk_dominant("催收 辱骂 侮辱") is False
    assert _query_non_cjk_dominant("Which system handles bank marketing renewals?") is True
    assert _query_non_cjk_dominant("llama.cpp n_batch truncation") is True
    assert _query_non_cjk_dominant("migrate_workspace UNIQUE 冲突") is True   # 技术名词混排
    assert _query_non_cjk_dominant("修复 FTS5 查询 bug") is True              # 少量 CJK
    assert _query_non_cjk_dominant("12345 !!!") is False                      # 无字母信息→保守关闭
    assert _query_non_cjk_dominant("") is False
