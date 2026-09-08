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
