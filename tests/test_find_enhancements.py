"""find enhancements: index-page preview (content_chars/outline,
include_content escape hatch), size metering of the actually-returned page,
page-hit unresolved_conflict_count, and the unfiltered total_estimate=None
semantics (v0.15.2 size block + v0.15.4 preview revision).
"""
from __future__ import annotations

import json
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.pipeline.read import _preview_item
from memory_arbiter.tokens import TOKEN_ESTIMATE_BASIS, estimate_tokens
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    return MemoryTools(settings, MemoryDB(settings))


def test_estimate_tokens_bucket_logic() -> None:
    assert estimate_tokens("") == 0
    # Pure CJK: 0.77 per char.
    assert estimate_tokens("配置只认配置文件" * 10) == round(0.77 * 80)
    # CJK punctuation bucket.
    assert estimate_tokens("，。：；" * 5) == round(0.85 * 20)
    # Digits: 1.15 per char.
    assert estimate_tokens("0123456789") == round(1.15 * 10)
    # Newlines/spaces.
    assert estimate_tokens("\n" * 10) == 10
    assert estimate_tokens(" " * 100) == 15
    # English words: 1.15 per word + spaces.
    assert estimate_tokens("alpha beta") == round(1.15 * 2 + 0.15)
    # Markdown chars at 0.9, ASCII punctuation at 0.6.
    assert estimate_tokens("##**") == round(0.9 * 4)
    assert estimate_tokens("....") == round(0.6 * 4)
    assert TOKEN_ESTIMATE_BASIS.startswith("heuristic_v1")


# ── v0.15.4: index-page preview ──────────────────────────────────────────


def test_find_preview_default_drops_content(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="alpha deployment note with details", subject="s", tags=[])
    result = tools.memory_search(query="deployment", limit=10)
    assert result["ok"] is True
    item = result["data"]["results"][0]
    assert "content" not in item
    assert item["content_chars"] == len("alpha deployment note with details")
    assert item["outline"] == [
        {"head": "alpha deployment note with details", "offset": 0},
    ]
    # Metadata is retained on the index page.
    for key in ("id", "subject", "tags", "event_time", "score"):
        assert key in item, key


def test_find_include_content_restores_full_text(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    body = "alpha deployment note with details"
    tools.memory_write(content=body, subject="s", tags=[])
    result = tools.memory_search(query="deployment", limit=10, include_content=True)
    item = result["data"]["results"][0]
    assert item["content"] == body
    # content_chars/outline stay either way (uniform contract).
    assert item["content_chars"] == len(body)
    assert item["outline"]


def _long_paragraph(index: int) -> str:
    return f"段落{index} 这是一段足够长的内容不会因为太短而被合并掉"


def test_find_outline_offsets_match_source(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    paragraphs = [_long_paragraph(i) for i in range(5)]
    content = "\n\n".join(paragraphs)
    tools.memory_write(content=content, subject="s", tags=["outline-probe"])
    result = tools.memory_search(query="outline-probe", tags_filter=["outline-probe"], limit=10)
    outline = result["data"]["results"][0]["outline"]
    assert len(outline) == 5
    for index, segment in enumerate(outline):
        assert segment["head"] == paragraphs[index][:40]
        assert segment["offset"] == content.index(paragraphs[index])


def test_find_outline_offset_interops_with_read_span(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = "第一段主要内容足够长不会被合并掉"
    second = "第二段目标内容在这里等待切片读取"
    content = f"{first}\n\n{second}"
    memory_id = tools.memory_write(content=content, subject="s", tags=[])["data"]["id"]
    found = tools.memory_search(query="第二段", limit=10)
    outline = found["data"]["results"][0]["outline"]
    assert outline[1]["offset"] == content.index(second)
    read = tools.memory_get(
        memory_id=memory_id,
        span={"start": outline[1]["offset"], "end": outline[1]["offset"] + len(second)},
    )
    assert read["ok"] is True
    assert read["data"]["memory"]["content"] == second


def test_find_outline_exactly_eight_segments_has_no_overflow(tmp_path: Path) -> None:
    content = "\n\n".join(_long_paragraph(i) for i in range(8))
    preview = _preview_item({"subject": "s", "content": content}, include_content=False)
    assert len(preview["outline"]) == 8
    assert all(segment["offset"] is not None for segment in preview["outline"])


def test_find_outline_overflow_marker_beyond_eight() -> None:
    content = "\n\n".join(_long_paragraph(i) for i in range(11))
    preview = _preview_item({"subject": "s", "content": content}, include_content=False)
    outline = preview["outline"]
    assert len(outline) == 9
    assert outline[-1] == {"head": "…还有 3 段", "offset": None}


def test_find_outline_single_long_line_splits_into_parts() -> None:
    content = "x" * 1000
    preview = _preview_item({"subject": "s", "content": content}, include_content=False)
    outline = preview["outline"]
    # No newline/heading structure: the overlap fallback still bounds the
    # outline, and offsets stay ascending source coordinates.
    assert 1 < len(outline) <= 8
    assert outline[0]["offset"] == 0
    offsets = [segment["offset"] for segment in outline]
    assert offsets == sorted(offsets)
    assert all(len(segment["head"]) <= 40 for segment in outline)


def test_preview_item_empty_content() -> None:
    preview = _preview_item({"subject": "s", "content": ""}, include_content=False)
    assert preview["content_chars"] == 0
    assert preview["outline"] == []
    assert "content" not in preview


def test_preview_item_include_content_keeps_content() -> None:
    preview = _preview_item({"subject": "s", "content": "body"}, include_content=True)
    assert preview["content"] == "body"
    assert preview["content_chars"] == 4


# ── v0.15.4: size block — meters the page as actually returned ────────────


def test_find_size_block_default_on_and_opt_out(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="alpha deployment note with details", subject="s", tags=[])
    result = tools.memory_search(query="deployment", limit=10)
    assert result["ok"] is True
    item = result["data"]["results"][0]
    size = result["data"]["size"]
    assert size["returned_count"] == 1
    expected_chars = len(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))
    assert size["returned_chars"] == expected_chars
    assert size["tokens_estimate"] == estimate_tokens(
        json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
    )
    # v0.15.4: the beyond-limit ghost fields are gone.
    assert "matched_beyond_limit_count" not in size
    assert "matched_beyond_limit_chars" not in size
    assert "index page" in size["display_hint"]
    assert str(size["tokens_estimate"]) in size["display_hint"]

    # v0.15.6: the per-call opt-out is gone — the global config key
    # include_size (default true) governs; a passed flag is ignored with a
    # pointer warning. The config-off case lives in
    # tests/test_size_metering_unified.py.
    off = tools.memory_search(query="deployment", limit=10, include_size=False)
    assert any("global config key" in w for w in off["warnings"])
    assert "size" in off["data"]


def test_find_size_has_no_beyond_limit_fields_when_more_match(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for index in range(3):
        tools.memory_write(content=f"deployment note number {index}", subject=f"s{index}", tags=[])
    result = tools.memory_search(query="deployment", limit=1)
    assert result["ok"] is True
    size = result["data"]["size"]
    assert size["returned_count"] == 1
    assert "matched_beyond_limit_count" not in size
    assert "matched_beyond_limit_chars" not in size
    assert "not returned" not in (size["display_hint"] or "")


def test_find_size_empty_result_has_no_display_hint(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_search(query="does-not-exist", limit=10)
    assert result["ok"] is True
    size = result["data"]["size"]
    assert size["returned_count"] == 0
    assert size.get("display_hint") is None


def test_find_index_page_is_far_cheaper_than_full_content(tmp_path: Path) -> None:
    """Absolute-magnitude pin (not the self-referential formula): the whole
    point of the index page is that a long memory's preview costs a small
    fraction of its full text."""
    tools = make_tools(tmp_path)
    body = "这是一条很长的记忆正文，用来放大预览与全文的成本差距。" * 500
    tools.memory_write(content=body, subject="big-memory", tags=[])
    preview = tools.memory_search(query="big-memory", limit=10)
    full = tools.memory_search(query="big-memory", limit=10, include_content=True)
    preview_tokens = preview["data"]["size"]["tokens_estimate"]
    full_tokens = full["data"]["size"]["tokens_estimate"]
    assert preview_tokens * 10 < full_tokens, (
        f"index page should cost a small fraction of full text: "
        f"preview={preview_tokens} full={full_tokens}"
    )
    # display_hint matches the page kind.
    assert "index page" in preview["data"]["size"]["display_hint"]
    assert "full-content page" in full["data"]["size"]["display_hint"]


def test_find_browse_page_hint_does_not_forbid_paging(tmp_path: Path) -> None:
    """Empty-query browse pages carry an exact total — paging is the intended
    use there, so the hint must not say 'reword instead of deep paging'."""
    tools = make_tools(tmp_path)
    for index in range(3):
        tools.memory_write(content=f"browse note {index}", subject=f"b{index}", tags=[])
    result = tools.memory_search(query="", limit=2)
    assert result["ok"] is True
    assert result["data"]["retrieval_mode"] == "recent_browse"
    assert result["data"]["has_more"] is True
    hint = result["data"]["size"]["display_hint"]
    assert "deep paging" not in hint
    assert "browse page" in hint


def test_find_filtered_page_hint_allows_paging(tmp_path: Path) -> None:
    """Empty-query + tags_filter recall is retrieval_mode=direct but carries an
    exact SQL count — the 'reword instead of deep paging' guidance would
    contradict its own signal."""
    tools = make_tools(tmp_path)
    for index in range(5):
        tools.memory_write(content=f"release note {index}", subject=f"r{index}", tags=["release"])
    result = tools.memory_search(query="", tags_filter=["release"], limit=2)
    assert result["ok"] is True
    assert result["data"]["retrieval_mode"] == "direct"
    assert result["data"]["total_estimate"] == 5
    assert result["data"]["has_more"] is True
    hint = result["data"]["size"]["display_hint"]
    assert "deep paging" not in hint
    assert "index page" in hint
    assert "exact" in hint


# ── v0.15.4: total_estimate=None on unfiltered query-recall ──────────────


def test_find_unfiltered_query_reports_none_total_and_no_more(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for index in range(3):
        tools.memory_write(content="alpha beta", subject=f"ab{index}", tags=[])
    for index in range(20):
        tools.memory_write(content=f"noise{index}", subject=f"noise{index}", tags=[])
    result = tools.memory_search(query="alpha beta", limit=10)
    assert result["ok"] is True
    assert result["data"]["total_estimate"] is None
    assert result["data"]["has_more"] is False
    # Filtered recall keeps the exact SQL count.
    filtered = tools.memory_search(query="alpha beta", tags_filter=["none-match"], limit=10)
    assert filtered["data"]["total_estimate"] == 0


# ── v0.15.4: unresolved_conflict_count — page-hit only ────────────────────


def _member(memory_id: int, version: int, value: str) -> dict:
    quote = f"database is {value}"
    return {
        "memory_id": memory_id, "version": version, "attribute_raw": "database",
        "value_raw": value, "normalized_attribute": "database",
        "normalized_value": value.casefold(), "evidence_quote": quote,
        "evidence_span": [0, len(quote)], "content_hash": (str(memory_id) * 64)[:64],
        "direction": "a_to_b", "prompt_version": "p1", "detector_version": "d1",
    }


def test_find_unresolved_conflict_count_counts_page_hits(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="database is mysql", subject="s", tags=[])["data"]["id"]
    right = tools.memory_write(content="database is sqlite", subject="s2", tags=[])["data"]["id"]

    created = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "p", "attribute": "db", "scope": "g"},
        "members": [_member(left, 1, "mysql"), _member(right, 1, "sqlite")],
        "value_groups": [
            {"normalized_value": "mysql", "display_value": "mysql", "members": [f"{left}@1"]},
            {"normalized_value": "sqlite", "display_value": "sqlite", "members": [f"{right}@1"]},
        ],
        "status": "open", "detector_version": "d1", "prompt_version": "p1",
        "source": "scan", "reason": "diff", "authorized": True,
    })
    assert created["ok"] is True, created["data"]
    result = tools.memory_search(query="database", limit=10)
    # Both conflict members are on the page → the value counts page items hit,
    # not groups (one group, two page hits).
    assert result["data"]["unresolved_conflict_count"] == 2
    # The conflict_group signal (with next_executable_call) still attaches.
    signals = [r.get("conflict_signal") for r in result["data"]["results"] if r.get("conflict_signal")]
    assert signals, "conflict signal must still attach"
    assert all(sig.get("next_executable_call") for sig in signals)


def test_find_unresolved_conflict_count_absent_without_page_hits(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="database is mysql", subject="s", tags=[])["data"]["id"]
    right = tools.memory_write(content="database is sqlite", subject="s2", tags=[])["data"]["id"]
    created = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "p", "attribute": "db", "scope": "g"},
        "members": [_member(left, 1, "mysql"), _member(right, 1, "sqlite")],
        "value_groups": [
            {"normalized_value": "mysql", "display_value": "mysql", "members": [f"{left}@1"]},
            {"normalized_value": "sqlite", "display_value": "sqlite", "members": [f"{right}@1"]},
        ],
        "status": "open", "detector_version": "d1", "prompt_version": "p1",
        "source": "scan", "reason": "diff", "authorized": True,
    })
    assert created["ok"] is True, created["data"]
    # An open conflict exists in scope, but the page hits an unrelated memory
    # (a direct hit, so no recent-fallback pulls the members onto the page) →
    # the whole field stays out of the response.
    tools.memory_write(content="unique zebra token", subject="zebra", tags=[])
    result = tools.memory_search(query="zebra", limit=10)
    assert result["ok"] is True
    assert result["data"]["results"], "expected a direct hit page"
    assert "unresolved_conflict_count" not in result["data"]


def make_strict_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "strict.sqlite3",
        backup_jsonl=tmp_path / "strict.jsonl",
        workspace="projA",
        isolation="strict",
    )
    return MemoryTools(settings, MemoryDB(settings))


def _confirm_pending(tools: MemoryTools, memory_id: int) -> None:
    record = tools.db.get_memory(memory_id)
    if record["status"] != "pending":
        return
    confirmed = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id,
        "canonical": record["workspace_canonical"] or record["workspace"],
        "authorized": True,
    })
    assert confirmed["ok"] is True, confirmed


def test_find_unresolved_conflict_count_strict_scope(tmp_path: Path) -> None:
    """Strict callers still get the page-hit count when page items conflict."""
    tools = make_strict_tools(tmp_path)
    left = tools.memory_write(
        content="database is mysql", subject="s", tags=[], workspace="projA",
    )["data"]["id"]
    right = tools.memory_write(
        content="database is sqlite", subject="s2", tags=[], workspace="projA",
    )["data"]["id"]
    _confirm_pending(tools, left)
    _confirm_pending(tools, right)
    left_version = int(tools.db.get_memory(left)["version"])
    right_version = int(tools.db.get_memory(right)["version"])

    created = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "p", "attribute": "db", "scope": "g"},
        "members": [_member(left, left_version, "mysql"), _member(right, right_version, "sqlite")],
        "value_groups": [
            {"normalized_value": "mysql", "display_value": "mysql", "members": [f"{left}@{left_version}"]},
            {"normalized_value": "sqlite", "display_value": "sqlite", "members": [f"{right}@{right_version}"]},
        ],
        "status": "open", "detector_version": "d1", "prompt_version": "p1",
        "source": "scan", "reason": "diff", "workspace": "projA", "authorized": True,
    })
    assert created["ok"] is True, created["data"]
    result = tools.memory_search(query="database", workspace="projA", limit=10)
    assert result["ok"] is True, result
    assert result["data"]["unresolved_conflict_count"] == 2
    assert not any(
        "unresolved_conflict_count" in str(warning) for warning in result["warnings"]
    )
