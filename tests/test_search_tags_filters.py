from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import types
from pathlib import Path
from typing import Optional

import pytest

from memory_arbiter.arbitration import compare_memories
from memory_arbiter.config import Settings, parse_bool
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import SourceType
from memory_arbiter.tools import MemoryTools
from memory_arbiter.search import (
    _TAGS_SCORE_CAP,
    _TAGS_STRONG_WEIGHT,
    _TAGS_MEDIUM_WEIGHT,
    _TAGS_WEAK_WEIGHT,
    _cjk_substring_match,
    _is_pure_cjk_token,
    _normalize_token_for_tag_match,
    _score_tags_surface,
)


class _MockManagedEmbedder:
    """Minimal mock for ManagedEmbedder — wraps a plain encode function.

    Mirrors the production Never-raises contract: if _encode raises, the
    exception is caught, last_encode_error is set, and an empty EmbedResult
    is returned so callers must check er.embedding.
    """

    def __init__(self, encode_fn):
        self._encode = encode_fn
        self.embedding_space_id = "mock_space_id"
        self.last_encode_error = None

    def embed_text(self, prefix="", body="", max_body_chars=None):
        # Mirror the production separator so the prefix's trailing token and the
        # body's leading token are not merged (e.g. "alpha"+"alpha x" → "alphaalpha").
        sep = "\n" if prefix and body else ""
        text = (prefix + sep + body).strip()
        try:
            emb = self._encode(text)
        except Exception as exc:
            self.last_encode_error = str(exc)
            return EmbedResult(embedding=[], truncated=True, original_tokens=0, used_tokens=0)
        return EmbedResult(embedding=emb, truncated=False, original_tokens=0, used_tokens=0)


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=False,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def make_vec_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    settings = Settings(
        db_path=tmp_path / "memory-vec.sqlite3",
        backup_jsonl=tmp_path / "backup-vec.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=2,
        split_threshold=1,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def clear_config_env(monkeypatch) -> None:
    for key in (
        "MEMORY_ARBITER_CONFIG",
        "MEMORY_ARBITER_DB_PATH",
        "MEMORY_ARBITER_BACKUP_JSONL",
        "MEMORY_ARBITER_POLICY",
        "MEMORY_ARBITER_CLIENT",
        "MEMORY_ARBITER_AGENT_ID",
        "MEMORY_ARBITER_WORKSPACE",
        "MEMORY_ARBITER_ENABLE_SQLITE_VEC",
        "MEMORY_ARBITER_VEC_DIM",
        "MEMORY_ARBITER_RECALL_POOL_CAP",
        "MEMORY_ARBITER_CONTENT_LIKE_CAP",
        "MEMORY_ARBITER_EMBEDDING_PROVIDER",
        "MEMORY_ARBITER_EMBEDDING_MODEL_PATH",
        "MEMORY_ARBITER_EMBEDDING_AUTO_QUERY",
        "MEMORY_ARBITER_EMBEDDING_AUTO_WRITE",
        "MEMORY_ARBITER_GGUF",
        "MEMORY_ARBITER_TOOL_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)



def _tags_score(query: str, tags: list[str]) -> tuple[float, str]:
    """Convenience wrapper returning (score, level)."""
    return _score_tags_surface(
        query, tags,
        _TAGS_STRONG_WEIGHT, _TAGS_MEDIUM_WEIGHT, _TAGS_WEAK_WEIGHT, _TAGS_SCORE_CAP,
    )[:2]


def test_tag_all_tokens_match_strong() -> None:
    # id=206 修复目标：query 两个 token 都是精确 tag → strong
    score, level = _tags_score("v0.7.2 发版", ["v0.7.2", "发版"])
    assert level == "strong"
    assert score == _TAGS_STRONG_WEIGHT


def test_tag_long_cjk_query() -> None:
    # 长 CJK query 单 token，tags 前缀+后缀都命中 → strong（修 v1 CJK bug）
    score, level = _tags_score("发版历史", ["发版", "历史"])
    assert level == "strong"
    assert score == _TAGS_STRONG_WEIGHT


def test_tag_mixed_long_query() -> None:
    score, level = _tags_score("v0.7.2 发版历史", ["v0.7.2", "发版", "历史"])
    assert level == "strong"
    assert score == _TAGS_STRONG_WEIGHT


def test_tag_half_match_medium() -> None:
    score, level = _tags_score("v0.7.2 发版", ["v0.7.2", "技术参考"])
    assert level == "medium"
    assert score == _TAGS_MEDIUM_WEIGHT


def test_tag_one_token_match_weak() -> None:
    # 3 个 query token，只命中 1 个 → ratio 1/3 < 0.5 → weak
    score, level = _tags_score("v0.7.2 发版 历史", ["发版", "其它"])
    assert level == "weak"
    assert score == _TAGS_WEAK_WEIGHT


def test_tag_no_match_none() -> None:
    score, level = _tags_score("v0.7.2 发版", ["doctor", "bug"])
    assert level == "none"
    assert score == 0.0


def test_tag_version_word_boundary() -> None:
    # query v0.7.2 vs tag v0.7.0 → ASCII equality 不命中（防伪召回）
    score, level = _tags_score("v0.7.2", ["v0.7.0"])
    assert level == "none"
    assert score == 0.0


def test_tag_version_normalization_bidirectional() -> None:
    # query 0.7.2 vs tag V0.7.2 → 双向归一化后都成 0.7.2 → 命中
    score, level = _tags_score("0.7.2", ["V0.7.2"])
    assert level == "strong"
    assert score == _TAGS_STRONG_WEIGHT


def test_tag_v_not_stripped_for_words() -> None:
    # query vue 不剥 v（不跟数字）→ 不匹配 tag ue
    score, level = _tags_score("vue", ["ue"])
    assert level == "none"


def test_tag_cjk_prefix_substring() -> None:
    # tag 发版 是 query token 发版历史 的前缀 → 命中
    score, level = _tags_score("发版历史", ["发版"])
    assert level == "strong"


def test_tag_cjk_suffix_substring() -> None:
    # tag 历史 是 query token 发版历史 的后缀 → 命中
    score, level = _tags_score("发版历史", ["历史"])
    assert level == "strong"


def test_tag_cjk_middle_substring_excluded() -> None:
    # tag 版历 是 query token 发版历史 的中间子串 → 不命中（review_2 漏洞 1）
    score, level = _tags_score("发版历史", ["版历"])
    assert level == "none"


def test_tag_ascii_no_substring() -> None:
    # tag memory vs query memory-arbiter → ASCII 不 substring → none
    score, level = _tags_score("memory-arbiter", ["memory"])
    assert level == "none"


def test_tag_empty_tags_list() -> None:
    score, level = _tags_score("v0.7.2 发版", [])
    assert level == "none"
    assert score == 0.0


def test_tag_empty_query() -> None:
    score, level = _tags_score("", ["v0.7.2", "发版"])
    assert level == "none"
    assert score == 0.0


def test_tag_subject_unchanged() -> None:
    # subject 仍走原 _score_surface：整串 substring 命中仍判 strong，不受 tag 改动影响
    from memory_arbiter.search import _score_surface, extract_anchors
    q = "v0.7.2 发版"
    subject = "v0.7.2 发版记录"
    score, level = _score_surface(
        extract_anchors(q), subject,
        10.0, 6.0, 2.0, 10.0, q.lower(),
    )
    assert level == "strong"


def test_tag_mixed_ascii_cjk_no_space_none() -> None:
    # 第五轮 S1 / 第八轮 E2：无空格混合 token 走 equality → 不命中（已知盲区）
    score, level = _tags_score("v0.7.2发版", ["v0.7.2", "发版"])
    assert level == "none"


def test_tag_mixed_with_space_strong() -> None:
    # 对照：有空格的混合 query → strong（推荐写法）
    score, level = _tags_score("v0.7.2 发版", ["v0.7.2", "发版"])
    assert level == "strong"


def test_tag_debug_fields_populated() -> None:
    # debug dict 字段齐全（用于 _soft_rerank 写 _tag_query_tokens 等）
    from memory_arbiter.search import _score_tags_surface
    _, _, debug = _score_tags_surface(
        "v0.7.2 发版", ["v0.7.2", "发版"],
        _TAGS_STRONG_WEIGHT, _TAGS_MEDIUM_WEIGHT, _TAGS_WEAK_WEIGHT, _TAGS_SCORE_CAP,
    )
    assert debug == {"total": 2, "matched": 2, "ratio": 1.0}


# ---- _is_pure_cjk_token / _cjk_substring_match / _normalize direct -------
# 设计 §2.3 E2 明确：_is_pure_cjk_token 不能用 token.isascii() 反向判定，
# 否则混合 token "0.7.2发版"（含 ASCII 数字）会被归入 CJK 类走 substring。
# 这些单元测试钉死判定函数的行为契约。


def test_is_pure_cjk_token_contract() -> None:
    assert _is_pure_cjk_token("发版") is True
    assert _is_pure_cjk_token("发版历史") is True
    assert _is_pure_cjk_token("v0.7.2") is False   # 纯 ASCII
    assert _is_pure_cjk_token("0.7.2发版") is False  # 混合（含 ASCII 数字）—— 关键
    assert _is_pure_cjk_token("memory") is False
    assert _is_pure_cjk_token("") is True           # 无 ASCII alnum → 视为 pure（空 query 已在调用方拦截）


def test_normalize_token_for_tag_match_contract() -> None:
    assert _normalize_token_for_tag_match("v0.7.2") == "0.7.2"
    assert _normalize_token_for_tag_match("0.7.2") == "0.7.2"
    assert _normalize_token_for_tag_match("V0.7.2") == "0.7.2"
    assert _normalize_token_for_tag_match("vue") == "vue"           # v 不跟数字，不剥
    assert _normalize_token_for_tag_match("  Abc  ") == "abc"        # strip + lower
    assert _normalize_token_for_tag_match("发版") == "发版"


def test_cjk_substring_match_contract() -> None:
    assert _cjk_substring_match("发版", "发版历史") is True   # prefix
    assert _cjk_substring_match("历史", "发版历史") is True   # suffix
    assert _cjk_substring_match("发版历史", "发版历史") is True  # equal
    assert _cjk_substring_match("版历", "发版历史") is False  # middle（bigram 伪 tag）
    assert _cjk_substring_match("发", "发版") is False       # 单字 tag 长度门槛
    # _cjk_substring_match 本身是纯字符串 prefix/suffix 判定，ASCII 串按同样的规则：
    assert _cjk_substring_match("xyz", "abcxyz") is True    # suffix 命中（调用方 _is_pure_cjk_token 保证 ASCII 串不进这条路径）
    assert _cjk_substring_match("abc", "abcxyz") is True    # prefix 命中
    assert _cjk_substring_match("bcxy", "abcxyz") is False  # middle 不命中


# ---- v0.7.3 commit 5: subject classify_match_level coverage threshold ----
# 守护 anchors.classify_match_level 的 specific_coverage 阈值（0.4→0.6，
# id=210/id=211 dogfooding 真根因）。这是 commit 5 三大修正之一，但此前
# anchors.py 没有任何测试 import，改阈值不会触发测试失败。这一组测试
# 用直接构造的 Anchor/AnchorMatch 钉死阈值数值，防止手滑改回 0.4/0.5
# 让 id=105 场景（subject 偶然含一字）静默升回 medium(6.0)，重新挤掉
# tag 双命中的 id=206。
#
# 合成数据证据见 scripts/tune_tag_weights.py（n=2000×5 seed）：
#   coverage 0.5 无效（A>B=0.5），0.6 是临界点（A>B=1.000），0.7+ 无额外收益。

from memory_arbiter.anchors import (
    Anchor as _Anchor,
    AnchorMatch as _AnchorMatch,
    classify_match_level as _classify_match_level,
)


def _classify(
    specific_hits: int, generic_hits: int, query_specific_count: int
) -> str:
    """直接构造 matches dict + query_anchors，绕开 extract_anchors 的 bigram
    干扰，精确控制 specific_coverage = specific_hits / query_specific_count。

    query_specific_count = query 里非 generic 的 anchor 数（分母）。
    构造的 query_anchors 全部 is_generic=False，所以 query_specific_count
    等于传入值；total_hits 由 specific+generic 决定（走 summary）。
    """
    query_anchors = [_Anchor(text=f"q{i}", is_generic=False)
                     for i in range(query_specific_count)]
    matches = {
        "_summary": _AnchorMatch(
            hit=True, kind="summary",
            specific_hits=specific_hits,
            generic_hits=generic_hits,
            total_hits=specific_hits + generic_hits,
        ),
    }
    return _classify_match_level(query_anchors, matches)


def test_classify_coverage_half_is_weak_not_medium() -> None:
    # id=210 dogfooding 核心 bug：subject 偶然含 query 一半 anchor
    # （specific=1, query_specific=2 → coverage=0.500）。
    # 0.6 阈值下落 weak(2.0)，旧 0.4 阈值会误升 medium(6.0)。
    # 改回 0.4/0.5 这个断言会失败 —— 这就是回归守门员。
    assert _classify(specific_hits=1, generic_hits=0, query_specific_count=2) == "weak"


def test_classify_coverage_full_is_medium() -> None:
    # 对照：query 两个 specific anchor 全命中（coverage=1.000）→ medium。
    # 这是 id=206 真正讲主题时该拿的 level。
    assert _classify(specific_hits=2, generic_hits=0, query_specific_count=2) == "medium"


def test_classify_coverage_threshold_boundary_0_6() -> None:
    # 阈值数值本身：3/5 = 0.600 刚好 >= 0.6 → medium。
    # 构造必须让第一条 medium 规则（specific>=1 AND total>=2）不触发，
    # 才能真正走到 coverage 判断：这里 specific=3 但 total 也=3 会先命中
    # 第一条规则——所以用 specific=3, total=1 是不可能的（total>=specific）。
    # 改用 1/2=0.5（守 weak）+ 2/2=1.0（守 medium）这对边界，见上下两条。
    # 本条留作"第一规则优先于 coverage"的文档性断言：3/5 走 medium 是因为
    # specific>=1 AND total>=2，不是因为 coverage。
    assert _classify(specific_hits=3, generic_hits=0, query_specific_count=5) == "medium"


def test_classify_coverage_just_below_threshold_is_weak() -> None:
    # 守"低于 0.6 阈值（且不触发第一规则）必须落 weak"。
    # 构造 specific=1, total=1（避开第一规则 specific>=1 AND total>=2），
    # query_specific=3 → coverage=0.333 < 0.6 → weak。
    # 这条钉死 coverage 规则的阈值：若有人改回 0.4 阈值，0.333 仍是 weak
    # （因为 0.333 < 0.4），所以它守的是"阈值不能低于 0.333"；真正守"0.5
    # 边界"的是上面那条 coverage_half 测试。
    assert _classify(specific_hits=1, generic_hits=0, query_specific_count=3) == "weak"


def test_classify_medium_via_specific_plus_total_rule() -> None:
    # medium 的第一条规则：specific_hits>=1 AND total_hits>=2（与 coverage 无关）。
    # 1 specific + 1 generic = total 2，query_specific=3 → coverage 0.333 < 0.6，
    # 但靠 specific+total 规则仍升 medium。
    assert _classify(specific_hits=1, generic_hits=1, query_specific_count=3) == "medium"


def test_classify_only_generic_is_weak() -> None:
    # 只有 generic 命中、specific=0：coverage=0，不满足 medium 两条规则，
    # 但 total>=1 → weak（不是 none）。
    assert _classify(specific_hits=0, generic_hits=2, query_specific_count=3) == "weak"


def test_classify_no_hits_is_none() -> None:
    # 一个都没命中 → none。
    assert _classify(specific_hits=0, generic_hits=0, query_specific_count=3) == "none"


def test_classify_single_specific_anchor_hit_is_medium() -> None:
    # coverage 规则的独占触发区：query 只有 1 个 specific anchor（如裸 query
    # "发版"），它命中时 specific=1 total=1，第一规则（total>=2）不满足，
    # 靠 coverage=1.0>=0.6 升 medium。
    # 这条守"coverage 规则不能被废掉"：把阈值改到 >1.0（如 2.0）会让
    # 单 token query 永远拿不到 medium，subject 命中只剩 weak(2.0)。
    # 上一轮把阈值改 2.0 跑全量 184 全绿，就是漏了这个场景。
    assert _classify(specific_hits=1, generic_hits=0, query_specific_count=1) == "medium"


def test_classify_id105_regression_via_real_pipeline() -> None:
    # 端到端回归：用真实 extract_anchors 复现 id=105 bug 场景。
    # query "v0.7.2 发版" 的两个 specific anchor 中，subject 只命中"发版"
    # （v0.4.0 ≠ v0.7.2）→ coverage=0.500 → 必须落 weak，不能 medium。
    # 这条覆盖"extract_anchors → score_anchor_overlap → classify_match_level"
    # 完整链路，守 subject 路径在真实 bigram 切分下的阈值行为。
    from memory_arbiter.anchors import (
        extract_anchors, score_anchor_overlap, classify_match_level,
    )
    query = "v0.7.2 发版"
    subject_id105 = "[已完成] README v0.4.0 发版"  # 含"发版"但不含 v0.7.2
    qa = extract_anchors(query)
    sa = extract_anchors(subject_id105)
    matches = score_anchor_overlap(qa, sa)
    assert classify_match_level(qa, matches) == "weak"
    # 同时验证 subject 语义全命中时能正确升 medium（id=206 的 subject 路径）
    subject_full = "v0.7.2 发版记录"
    matches_full = score_anchor_overlap(qa, extract_anchors(subject_full))
    assert classify_match_level(qa, matches_full) == "medium"


# ---- v0.7.3 change 2: search enhancement (tags_filter / time /
# source_type / has_more / 4-tuple / shortcut) end-to-end tests ---------
#
# 这些测试走 MemoryTools.memory_search 端到端（真实 sqlite + 真实 search_memories），
# 覆盖设计 §5.2 测试矩阵的核心场景。每次测试用一个全新的 tmp_path 库，写入已知
# 数据，断言行为。

import datetime as _dt


def _write_mem(tools: MemoryTools, *, content: str, subject: str, tags: list[str],
               source_type: str = "agent_generated", ingest_time: str | None = None,
               workspace: str = "ws") -> int:
    """Helper: write one memory via tools.memory_write, return its id."""
    payload = {
        "content": content, "subject": subject, "tags": tags,
        "source_type": source_type, "workspace": workspace,
        "agent_id": "tester",
    }
    if ingest_time is not None:
        payload["ingest_time"] = ingest_time
    res = tools.memory_write(**payload)
    assert res["ok"], f"write failed: {res}"
    return res["data"]["id"]


def test_tags_filter_exact_match(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="v0.7.2 发版记录", subject="发版", tags=["v0.7.2", "发版"])
    _write_mem(tools, content="其他无关", subject="其他", tags=["其它"])
    res = tools.memory_search(query="发版", tags_filter=["发版"])
    ids = [r["id"] for r in res["data"]["results"]]
    assert len(ids) == 1, f"tags_filter should exact-match only the 发版 tag, got {ids}"


def test_tags_filter_no_substring_false_positive(tmp_path: Path) -> None:
    # tags_filter=["v0.7"] 不应该命中 tags=["v0.7.0"]（精确匹配，防 LIKE 误命中）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=["v0.7.0"])
    res = tools.memory_search(query="v0.7.0", tags_filter=["v0.7"])
    assert res["data"]["results"] == [], "tags_filter must not substring-match"


def test_tags_filter_and_semantics(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="both", subject="s", tags=["发版", "v0.7.2"])
    _write_mem(tools, content="only one", subject="s2", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版", "v0.7.2"])
    ids = [r["id"] for r in res["data"]["results"]]
    assert len(ids) == 1, f"AND semantics: only the memory with both tags, got {ids}"


def test_tags_filter_empty_result_no_fallback(tmp_path: Path) -> None:
    # 匹配不到时不走 fallback（fallback 会返回不符合过滤条件的记忆）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="recent1", subject="r1", tags=["other"])
    res = tools.memory_search(query="发版", tags_filter=["发版"])
    assert res["data"]["results"] == []
    # 不应有 fallback warning（fallback 才会附加 "No direct memory match"）
    assert not any("No direct memory match" in w for w in res["warnings"])


def test_tags_filter_empty_list_treated_as_none(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="c", subject="s", tags=["发版"])
    # tags_filter=[] 等同不传 → 不过滤 → 命中（走 fallback 或正常召回）
    res = tools.memory_search(query="不存在的词", tags_filter=[])
    # 空 query 路径才走 fallback；这里 query 非空无匹配 → fallback
    assert "count" in res["data"]


def test_tags_filter_duplicates_deduped(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="c", subject="s", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版", "发版"])
    assert len(res["data"]["results"]) == 1


def test_tags_filter_empty_string_ignored(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="c", subject="s", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版", ""])
    assert len(res["data"]["results"]) == 1


def test_after_time_filter(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="old", subject="old", tags=["t"], ingest_time="2026-01-15T00:00:00+00:00")
    _write_mem(tools, content="new", subject="new", tags=["t"], ingest_time="2026-07-15T00:00:00+00:00")
    res = tools.memory_search(query="t", after_time="2026-06-01")
    subjects = [r["subject"] for r in res["data"]["results"]]
    assert "new" in subjects and "old" not in subjects


def test_after_time_with_timezone(tmp_path: Path) -> None:
    # after_time=2026-06-01T00:00:00+08:00 == 2026-05-31T16:00:00 UTC
    tools = make_tools(tmp_path)
    _write_mem(tools, content="before", subject="before", tags=["t"], ingest_time="2026-05-31T15:00:00+00:00")
    _write_mem(tools, content="after", subject="after", tags=["t"], ingest_time="2026-05-31T17:00:00+00:00")
    res = tools.memory_search(query="t", after_time="2026-06-01T00:00:00+08:00")
    subjects = [r["subject"] for r in res["data"]["results"]]
    assert "after" in subjects and "before" not in subjects


def test_after_time_invalid_format(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=["t"])
    res = tools.memory_search(query="t", after_time="xyz")
    assert any("invalid ISO 8601" in w for w in res["warnings"])
    # 无效 after_time 被忽略 → 正常返回
    assert res["ok"]


def test_before_time_filter(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="old", subject="old", tags=["t"], ingest_time="2026-01-15T00:00:00+00:00")
    _write_mem(tools, content="new", subject="new", tags=["t"], ingest_time="2026-07-15T00:00:00+00:00")
    res = tools.memory_search(query="t", before_time="2026-06-01")
    subjects = [r["subject"] for r in res["data"]["results"]]
    assert "old" in subjects and "new" not in subjects


def test_filters_disable_fallback(tmp_path: Path) -> None:
    # 有过滤但 pool 空 → 返回空 + 精准 warning，不走 recent_fallback
    tools = make_tools(tmp_path)
    _write_mem(tools, content="recent", subject="recent", tags=["other"])
    res = tools.memory_search(query="发版", tags_filter=["发版"])
    assert res["data"]["results"] == []
    assert not any("No direct memory match" in w for w in res["warnings"])


def test_source_type_filter(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="uc", subject="uc", tags=["t"], source_type="user_confirmed")
    _write_mem(tools, content="ag", subject="ag", tags=["t"], source_type="agent_generated")
    res = tools.memory_search(query="t", source_type="user_confirmed")
    subjects = [r["subject"] for r in res["data"]["results"]]
    assert subjects == ["uc"]


def test_has_more_when_more_exist(tmp_path: Path) -> None:
    # 库里 > limit 条匹配 tags_filter → has_more=True
    tools = make_tools(tmp_path)
    for i in range(15):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    assert res["data"]["has_more"] is True
    assert res["data"]["total_estimate"] == 15
    assert len(res["data"]["results"]) == 10


def test_has_more_false_when_exact(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for i in range(10):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    assert res["data"]["has_more"] is False
    assert res["data"]["total_estimate"] == 10


def test_has_more_false_when_fewer(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for i in range(7):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    assert res["data"]["has_more"] is False
    assert res["data"]["total_estimate"] == 7


def test_has_more_false_when_query_matches_few_no_filter(tmp_path: Path) -> None:
    # E1：无过滤场景，query 只匹配少数，全库更大 → has_more=False（修 count_active 误报）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="alpha beta", subject="alpha beta", tags=[])
    _write_mem(tools, content="alpha beta", subject="alpha beta", tags=[])
    _write_mem(tools, content="alpha beta", subject="alpha beta", tags=[])
    # 写一堆不匹配 query 的记忆
    for i in range(20):
        _write_mem(tools, content=f"noise{i}", subject=f"noise{i}", tags=[])
    res = tools.memory_search(query="alpha beta", limit=10)
    assert res["data"]["has_more"] is False, (
        f"E1: 无过滤场景 total_estimate 应=len(pool)=3，不是全库 23；got has_more={res['data']['has_more']}, total={res['data']['total_estimate']}"
    )
    assert res["data"]["total_estimate"] == 3


def test_no_filters_backward_compat(tmp_path: Path) -> None:
    # 不传任何新参数 → 完全同 v0.7.2（含 fallback 行为）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="hello world", subject="hello", tags=[])
    res = tools.memory_search(query="hello")
    assert res["ok"]
    assert len(res["data"]["results"]) >= 1
    # 返回结构应含新字段 has_more/total_estimate（即使不用过滤）
    assert "has_more" in res["data"]
    assert "total_estimate" in res["data"]


def test_search_returns_search_outcome_at_search_memories_level(tmp_path: Path) -> None:
    # v0.7.4 (M2): search_memories now returns a SearchOutcome dataclass, not a tuple.
    from memory_arbiter.search import search_memories, SearchOutcome
    tools = make_tools(tmp_path)
    _write_mem(tools, content="hello", subject="hello", tags=[])
    result = search_memories(tools.db, "hello")
    assert isinstance(result, SearchOutcome), f"search_memories must return SearchOutcome, got {type(result)}"
    assert isinstance(result.results, list)
    assert isinstance(result.warnings, list)
    assert len(result.results) >= 1
    assert result.retrieval_mode == "direct"
    assert isinstance(result.has_more, bool)
    assert isinstance(result.total_estimate, int)


def test_tools_layer_exposes_has_more(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=[])
    res = tools.memory_search(query="x")
    assert "has_more" in res["data"]
    assert "total_estimate" in res["data"]


def test_invalid_after_time_falls_back_to_none(tmp_path: Path) -> None:
    # after_time 无效 → warning + 视为 None；如果同时有其他 filter，has_filters 仍 True
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=["发版"])
    res = tools.memory_search(query="发版", after_time="not-a-date", tags_filter=["发版"])
    assert any("invalid ISO 8601" in w for w in res["warnings"])
    # tags_filter 仍生效
    assert len(res["data"]["results"]) == 1


def test_after_gt_before_warns(tmp_path: Path) -> None:
    # D4：after > before 矛盾 → warning + 两者都忽略
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=["t"], ingest_time="2026-06-15T00:00:00+00:00")
    res = tools.memory_search(query="t", after_time="2026-07-01", before_time="2026-06-01")
    assert any("after_time" in w and "before_time" in w and "empty" in w for w in res["warnings"]), (
        f"D4: after>before should warn; got {res['warnings']}"
    )


def test_empty_query_tags_filter_returns_matches(tmp_path: Path) -> None:
    # G6 (v0.8.5): empty query + tags_filter now does filter-driven recall
    # (was a dead path returning [] with a "query required" warning).
    tools = make_tools(tmp_path)
    matched = [_write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"]) for i in range(5)]
    _write_mem(tools, content="noise", subject="noise", tags=["决策"])  # different tag, must be excluded
    res = tools.memory_search(query="", tags_filter=["发版"])
    assert res["data"]["retrieval_mode"] == "direct"
    assert {r["id"] for r in res["data"]["results"]} == set(matched)
    assert res["data"]["total_estimate"] == 5
    assert res["data"]["has_more"] is False
    # The old "query required" warning is gone.
    assert not any("query required" in w for w in res["warnings"])


def test_empty_query_tags_filter_and_semantics(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    both = _write_mem(tools, content="both", subject="both", tags=["发版", "v0.8"])
    _write_mem(tools, content="one", subject="one", tags=["发版"])  # only one of the two tags
    res = tools.memory_search(query="", tags_filter=["发版", "v0.8"])
    assert [r["id"] for r in res["data"]["results"]] == [both]


def test_empty_query_after_time_filter(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="old", subject="old", tags=["t"], ingest_time="2026-01-01T00:00:00Z")
    new_id = _write_mem(tools, content="new", subject="new", tags=["t"], ingest_time="2026-07-01T00:00:00Z")
    res = tools.memory_search(query="", tags_filter=["t"], after_time="2026-06-01T00:00:00Z")
    assert [r["id"] for r in res["data"]["results"]] == [new_id]


def test_empty_query_source_type_filter(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    uc = _write_mem(tools, content="uc", subject="uc", tags=["t"], source_type="user_confirmed")
    _write_mem(tools, content="ag", subject="ag", tags=["t"], source_type="agent_generated")
    res = tools.memory_search(query="", source_type="user_confirmed")
    assert [r["id"] for r in res["data"]["results"]] == [uc]


def test_empty_query_has_more_and_total(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for i in range(15):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="", tags_filter=["发版"], limit=10)
    assert len(res["data"]["results"]) == 10
    assert res["data"]["total_estimate"] == 15
    assert res["data"]["has_more"] is True


def test_empty_query_pool_cap_total_is_full_count(tmp_path: Path) -> None:
    # total_estimate comes from count_filtered_memories (full-table count), not pool-capped.
    tools = make_tools(tmp_path)
    for i in range(60):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="", tags_filter=["发版"], limit=10)
    assert len(res["data"]["results"]) == 10
    assert res["data"]["total_estimate"] == 60
    assert res["data"]["has_more"] is True


def test_empty_query_no_filters_goes_fallback(tmp_path: Path) -> None:
    # 短路改造后，空 query + 无过滤仍应走 fallback（保留 v0.7.2 行为）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="r1", subject="r1", tags=[])
    res = tools.memory_search(query="")
    # fallback 路径：返回 recent memories + fallback warning
    assert any("No direct memory match" in w for w in res["warnings"]) or len(res["data"]["results"]) > 0


def test_bm25_mode_applies_filter_params(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    keep = _write_mem(tools, content="hello keep", subject="hello", tags=["keep"])
    _write_mem(tools, content="hello drop", subject="hello", tags=["drop"])
    monkeypatch.setenv("MEMORY_ARBITER_RANKING_MODE", "bm25")
    try:
        res = tools.memory_search(query="hello", tags_filter=["keep"])
        ids = {r["id"] for r in res["data"]["results"]}
        assert ids == {keep}
        assert any("bm25 mode falls back to hybrid" in w for w in res["warnings"]), res["warnings"]
        assert res["data"]["has_more"] is False
        assert res["data"]["total_estimate"] == 1
    finally:
        monkeypatch.delenv("MEMORY_ARBITER_RANKING_MODE", raising=False)


def test_bm25_mode_expired_search_honors_offset(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    ids = []
    for i in range(5):
        mid = _write_mem(tools, content=f"release expired {i}", subject="release", tags=[])
        ids.append(mid)
        assert tools.db.update_memory(mid, {"status": "superseded"}) is True
    monkeypatch.setenv("MEMORY_ARBITER_RANKING_MODE", "bm25")
    try:
        page1 = tools.memory_search_expired(query="release", limit=2, offset=0)
        page2 = tools.memory_search_expired(query="release", limit=2, offset=2)
        p1_ids = [r["id"] for r in page1["data"]["results"]]
        p2_ids = [r["id"] for r in page2["data"]["results"]]
        assert len(p1_ids) == 2
        assert len(p2_ids) == 2
        assert set(p1_ids).isdisjoint(p2_ids)
        assert page1["data"]["has_more"] is True
        assert page2["data"]["offset"] == 2
        assert page2["data"]["next_offset"] == 4
        assert set(p1_ids + p2_ids).issubset(set(ids))
    finally:
        monkeypatch.delenv("MEMORY_ARBITER_RANKING_MODE", raising=False)


def test_passes_filters_unit() -> None:
    # B3：_passes_filters 直接单元测试
    from memory_arbiter.search import _passes_filters
    from datetime import datetime, timezone

    def mk(ingest_time: str | None = "2026-06-15T00:00:00+00:00", tags: list = None, source_type: str = "agent_generated"):
        rec = {"tags": json.dumps(tags or []), "ingest_time": ingest_time, "source_type": source_type}
        return rec

    after = datetime(2026, 6, 1, tzinfo=timezone.utc)
    before = datetime(2026, 7, 1, tzinfo=timezone.utc)

    # tags AND
    assert _passes_filters(mk(tags=["发版", "v0.7.2"]), ["发版", "v0.7.2"], None, None, None) is True
    assert _passes_filters(mk(tags=["发版"]), ["发版", "v0.7.2"], None, None, None) is False

    # time bounds
    assert _passes_filters(mk(ingest_time="2026-06-15T00:00:00+00:00"), None, after, before, None) is True
    assert _passes_filters(mk(ingest_time="2026-05-15T00:00:00+00:00"), None, after, None, None) is False
    assert _passes_filters(mk(ingest_time="2026-08-15T00:00:00+00:00"), None, None, before, None) is False

    # time 无效 → 过滤掉
    assert _passes_filters(mk(ingest_time="not-a-date"), None, after, None, None) is False

    # source_type 等值
    assert _passes_filters(mk(source_type="user_confirmed"), None, None, None, "user_confirmed") is True
    assert _passes_filters(mk(source_type="agent_generated"), None, None, None, "user_confirmed") is False

    # JSON parse 失败 → 空集 → 不命中
    rec_bad = {"tags": "{bad json", "ingest_time": "2026-06-15T00:00:00+00:00", "source_type": "x"}
    assert _passes_filters(rec_bad, ["any"], None, None, None) is False


def test_pool_cap_truncates_results(tmp_path: Path) -> None:
    # B1/T1：库里很多匹配 tags_filter，pool_cap 截断后 reranked ≤ limit，has_more=True，total=全库匹配数
    tools = make_tools(tmp_path)
    for i in range(60):  # 超过 pool_cap=50
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    assert len(res["data"]["results"]) <= 10
    assert res["data"]["has_more"] is True
    # total_estimate 走 count_filtered（SQL 全表），=60
    assert res["data"]["total_estimate"] == 60


def test_count_matches_post_filter(tmp_path: Path) -> None:
    # T2：count_filtered_memories 返回值 == Python post-filter 后、切片前的 pool 长度
    tools = make_tools(tmp_path)
    for i in range(8):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    # 用 tags_filter 让 has_filters=True，count_filtered 会算全表匹配数
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    # 8 条全匹配，pool 召回 8，reranked=8，total=8
    assert res["data"]["total_estimate"] == 8
    assert len(res["data"]["results"]) == 8
    assert res["data"]["has_more"] is False


