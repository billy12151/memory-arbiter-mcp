"""find enhancements: index-page preview (content_chars/outline,
content_mode enum), size metering of the actually-returned page,
page-hit unresolved_conflict_count, and the unfiltered total_estimate=None
semantics (v0.15.2 size block + v0.15.4 preview revision + v0.15.10
content_mode/hit_spans).
"""
from __future__ import annotations

import pytest

import json
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import MemoryRecord
from memory_arbiter.pipeline.read import _preview_item
from memory_arbiter.tokens import TOKEN_ESTIMATE_BASIS, estimate_tokens
from memory_arbiter.tools import MemoryTools



@pytest.fixture(autouse=True)
def _relevance_floor_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0.15.9: the tests in this file exercise scoping/preview/metering with
    # content-only fixtures; the query-recall relevance floor is orthogonal to
    # what they pin (its behavior lives in tests/test_recall_quality_0159.py).
    # Disable it here so weak fixtures stay recallable.
    monkeypatch.setattr("memory_arbiter.search.QUERY_RECALL_SCORE_FLOOR", -1.0)


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


def test_find_content_mode_full_restores_full_text(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    body = "alpha deployment note with details"
    tools.memory_write(content=body, subject="s", tags=[])
    result = tools.memory_search(query="deployment", limit=10, content_mode="full")
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
    preview = _preview_item({"subject": "s", "content": content}, content_mode="preview")
    assert len(preview["outline"]) == 8
    assert all(segment["offset"] is not None for segment in preview["outline"])


def test_find_outline_overflow_marker_beyond_eight() -> None:
    content = "\n\n".join(_long_paragraph(i) for i in range(11))
    preview = _preview_item({"subject": "s", "content": content}, content_mode="preview")
    outline = preview["outline"]
    assert len(outline) == 9
    assert outline[-1] == {"head": "…还有 3 段", "offset": None}


def test_find_outline_single_long_line_splits_into_parts() -> None:
    content = "x" * 1000
    preview = _preview_item({"subject": "s", "content": content}, content_mode="preview")
    outline = preview["outline"]
    # No newline/heading structure: the overlap fallback still bounds the
    # outline, and offsets stay ascending source coordinates.
    assert 1 < len(outline) <= 8
    assert outline[0]["offset"] == 0
    offsets = [segment["offset"] for segment in outline]
    assert offsets == sorted(offsets)
    assert all(len(segment["head"]) <= 40 for segment in outline)


def test_preview_item_empty_content() -> None:
    preview = _preview_item({"subject": "s", "content": ""}, content_mode="preview")
    assert preview["content_chars"] == 0
    assert preview["outline"] == []
    assert "content" not in preview


def test_preview_item_content_mode_full_keeps_content() -> None:
    preview = _preview_item({"subject": "s", "content": "body"}, content_mode="full")
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

    # include_size flag warning is pinned in test_size_metering_unified.py.


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
    full = tools.memory_search(query="big-memory", limit=10, content_mode="full")
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
        "workspace": record.get("workspace_canonical") or record.get("workspace") or "default",
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


# ── v0.15.10: content_mode / hit_spans ─────────────────────────────────────


def _evidence_hit(kind: str, start: int, end: int, score: float = 0.9) -> dict:
    return {
        "kind": kind, "text": "raw unit text is irrelevant — spans slice source",
        "start_offset": start, "end_offset": end, "score": score,
    }


def _cyclic_content(length: int) -> str:
    return "".join(str(i % 10) for i in range(length))


def test_content_mode_invalid_value_rejected(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_search(query="x", content_mode="snippets")
    assert result["ok"] is False
    assert 'content_mode must be one of' in str(result["data"].get("error"))
    batch = tools.memory_batch_find(
        queries=[{"query": "x"}], content_mode="snippets",
    )
    assert batch["ok"] is False
    assert 'content_mode must be one of' in str(batch["data"].get("error"))


def test_include_content_removed_with_migration_pointer(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_search(query="x", include_content=True)
    assert result["ok"] is False
    assert "include_content was removed in v0.15.10" in str(result["data"].get("error"))
    assert 'content_mode="full"' in str(result["data"].get("error"))
    batch = tools.memory_batch_find(
        queries=[{"query": "x"}], include_content=True,
    )
    assert batch["ok"] is False
    assert "include_content was removed in v0.15.10" in str(batch["data"].get("error"))


def test_hits_mode_low_coverage_keeps_spans_drops_content() -> None:
    content = _cyclic_content(100)
    hits = [_evidence_hit("text", 60, 75), _evidence_hit("text", 10, 30)]
    preview = _preview_item(
        {"subject": "s", "content": content, "_evidence_hits": hits},
        content_mode="hits",
    )
    assert "content" not in preview
    spans = preview["hit_spans"]
    # Merged intervals in source order; text is sliced from the source so the
    # span and text are strictly self-consistent (read span returns exactly it).
    assert [(sp["start_offset"], sp["end_offset"]) for sp in spans] == [
        (10, 30), (60, 75),
    ]
    for sp in spans:
        assert sp["text"] == content[sp["start_offset"]:sp["end_offset"]]
    # Internal evidence field never leaks.
    assert "_evidence_hits" not in preview


def test_hits_mode_overlapping_intervals_merge() -> None:
    content = _cyclic_content(200)
    # Overlap of 10 chars (the long-text fallback slices with overlap=60):
    # naive summation would double-count; merging must collapse them.
    hits = [_evidence_hit("text", 50, 110), _evidence_hit("text", 0, 60)]
    preview = _preview_item(
        {"subject": "s", "content": content, "_evidence_hits": hits},
        content_mode="hits",
    )
    spans = preview["hit_spans"]
    assert [(sp["start_offset"], sp["end_offset"]) for sp in spans] == [(0, 110)]
    assert spans[0]["text"] == content[0:110]


def test_hits_mode_subject_kind_hit_filtered() -> None:
    content = _cyclic_content(50)
    hits = [_evidence_hit("subject", 0, 0), _evidence_hit("text", 5, 20)]
    preview = _preview_item(
        {"subject": "s", "content": content, "_evidence_hits": hits},
        content_mode="hits",
    )
    assert [(sp["start_offset"], sp["end_offset"]) for sp in preview["hit_spans"]] == [(5, 20)]


def test_hits_mode_high_coverage_upgrades_to_full_text() -> None:
    content = _cyclic_content(100)
    hits = [_evidence_hit("text", 0, 60)]  # 60% >= 50% threshold
    preview = _preview_item(
        {"subject": "s", "content": content, "_evidence_hits": hits},
        content_mode="hits",
    )
    assert preview["content"] == content
    # hit_spans stays as an annotation.
    assert [(sp["start_offset"], sp["end_offset"]) for sp in preview["hit_spans"]] == [(0, 60)]


def test_hits_mode_coverage_exactly_half_upgrades() -> None:
    content = _cyclic_content(100)
    hits = [_evidence_hit("text", 0, 50)]  # exactly 50% — >= threshold upgrades
    preview = _preview_item(
        {"subject": "s", "content": content, "_evidence_hits": hits},
        content_mode="hits",
    )
    assert preview["content"] == content


def test_hits_mode_no_evidence_hits_keeps_plain_preview() -> None:
    preview = _preview_item({"subject": "s", "content": "body"}, content_mode="hits")
    assert "hit_spans" not in preview
    assert "content" not in preview
    assert preview["content_chars"] == 4


def test_hits_mode_pipeline_without_embedder_stays_preview(tmp_path: Path) -> None:
    """No embedder → no evidence channel → content_mode="hits" degrades to the
    plain preview shape (FTS/phrase-only items carry no hit_spans)."""
    tools = make_tools(tmp_path)
    tools.memory_write(content="alpha deployment note with details", subject="s", tags=[])
    result = tools.memory_search(query="deployment", content_mode="hits")
    assert result["ok"] is True
    item = result["data"]["results"][0]
    assert "content" not in item
    assert "hit_spans" not in item
    assert "hit-spans page" in result["data"]["size"]["display_hint"]


def test_batch_find_content_mode_full_and_hits(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="alpha deployment note with details", subject="s", tags=[])
    full = tools.memory_batch_find(
        queries=[{"query": "deployment"}], content_mode="full",
    )
    assert full["ok"] is True, full["data"]
    item = full["data"]["results"][0]
    assert item["content"] == "alpha deployment note with details"
    assert "full texts" in full["data"]["size"]["display_hint"]
    hits = tools.memory_batch_find(
        queries=[{"query": "deployment"}], content_mode="hits",
    )
    assert hits["ok"] is True, hits["data"]
    assert "vector-hit spans" in str(hits["data"]["size"]["display_hint"])


# ── 0.16.10 wide-recall channel split ────────────────────────────────────────

class _FakeEmbedder:
    """sqlite-vec-capable stand-in (pattern: tests/test_scan_pipeline.py).
    dim 2 so the all-zero query vector matches the vec table dimension."""

    embedding_space_id = "fake-find-recall-space"
    dim = 2
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars: "int | None" = None) -> EmbedResult:
        return EmbedResult([0.0, 1.0], False, 1, 1)


def _make_vec_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake-find.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "vec.sqlite3",
        backup_jsonl=tmp_path / "vec-backup.jsonl",
        embedding_model_path=model,
        # The query_embedding=None branch of the regression test must stay on
        # the vectorless degradation path — auto-embedding would silently arm
        # the semantic channel and defeat the comparison.
        embedding_auto_query=False,
        client="c", agent_id="a",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = _FakeEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(_FakeEmbedder.dim) == []
    db.init_vec_index_state(_FakeEmbedder.embedding_space_id, True, _FakeEmbedder.dim)
    return tools


def test_wide_recall_content_like_channel_skipped_when_vector_available(tmp_path: Path) -> None:
    """0.16.10: the content-LIKE wide-recall channel is a no-vector
    degradation channel only (`... and not vector_available`). A memory
    reachable ONLY through a content substring (subject/tags clean of the
    query words) stays out of the recalled page when a query embedding is
    supplied; the same call without the embedding recalls it via LIKE.

    Fixture constraints that keep this a pure channel test:
    - the query is two 2-char words — the FTS5 trigram tokenizer cannot form
      a trigram from them, so the FTS channels cannot smuggle the memory into
      the pool (a ≥3-char word would FTS-match the content and mask the gate);
    - the memory is inserted directly (no evidence worker), so the
      zero-vector evidence KNN scans an empty vec table — an empty result
      there is expected and deliberately not relied on;
    - the relevance floor stays disabled via this file's autouse fixture, or
      the content-only match (~1.0) would be dropped from the page either way.
    """
    tools = _make_vec_tools(tmp_path)
    mid, _warnings = tools.db.insert_memory(MemoryRecord(
        content="流水线用 ci cd 做发布",
        agent_id="a",
        workspace="ws",
        subject="发布说明",
        tags=["deploy"],
    ))
    assert mid is not None

    with_vec = tools.memory_search(query="ci cd", limit=10, query_embedding=[0.0, 0.0])
    assert with_vec["ok"] is True
    assert mid not in [item["id"] for item in with_vec["data"]["results"]]

    without_vec = tools.memory_search(query="ci cd", limit=10, query_embedding=None)
    assert without_vec["ok"] is True
    assert mid in [item["id"] for item in without_vec["data"]["results"]]


def test_find_conflict_group_query_runs_once_per_search(tmp_path: Path, monkeypatch) -> None:
    """0.16.12 P1-T3：signal 挂接与 unresolved_conflict_count 共享同一次组查询。"""
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="database is mysql", subject="s", tags=[])["data"]["id"]
    right = tools.memory_write(content="database is sqlite", subject="s2", tags=[])["data"]["id"]
    tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "p", "attribute": "db", "scope": "g"},
        "members": [_member(left, 1, "mysql"), _member(right, 1, "sqlite")],
        "value_groups": [
            {"normalized_value": "mysql", "display_value": "mysql", "members": [f"{left}@1"]},
            {"normalized_value": "sqlite", "display_value": "sqlite", "members": [f"{right}@1"]},
        ],
        "status": "open", "detector_version": "d1", "prompt_version": "p1",
        "source": "scan", "reason": "diff", "authorized": True,
    })
    calls = []
    original = tools.db.conflicts.list_open_conflicts_for_memory_ids

    def counted(ids, *, include_applying=False):
        calls.append(list(ids))
        return original(ids, include_applying=include_applying)

    monkeypatch.setattr(tools.db.conflicts, "list_open_conflicts_for_memory_ids", counted)
    result = tools.memory_search(query="database", limit=10)
    assert result["data"]["unresolved_conflict_count"] == 2
    assert result["data"]["results"]
    assert any(r.get("conflict_signal") for r in result["data"]["results"])
    assert len(calls) == 1  # 两个消费点共享一次查询


def test_outline_table_path_matches_reparse(tmp_path: Path) -> None:
    """0.16.12 P1-T5 双路一致性：同一篇文档，查表 outline 与重解析 outline
    逐段一致（head 截断 + offset + 还有 N 段）。表未发布当前版本时回退
    重解析（编辑后异步重建窗口零行为差）。"""
    from memory_arbiter.pipeline.read import _content_outline, _outline_for_item

    tools = make_tools(tmp_path)
    subject = "双路一致性文档"
    content = "\n\n".join(f"第 {i} 段：审计一致性检查内容 {i}" for i in range(12))
    mid = tools.memory_write(content=content, subject=subject, tags=[])["data"]["id"]
    version = int(tools.db.get_memory(mid)["version"])
    # 表尚无当前版本行（无 embedder 的测试路径不发布 evidence）→ 回退重解析
    fallback = _outline_for_item(tools.db, mid, version, subject, content)
    assert fallback == _content_outline(subject, content)
    # 直接向 evidence 表发布当前版本单元 → 查表路径
    import hashlib
    from memory_arbiter.models import utc_now_iso
    with tools.db.write_transaction() as conn:
        for index, part in enumerate(content.split("\n\n")):
            start = sum(len(p) + 2 for p in content.split("\n\n")[:index])
            conn.execute(
                "INSERT INTO memory_row(memory_id, memory_version, content_hash,"
                " row_index, kind, text, start_offset, end_offset, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (mid, version, "h", index, "sentence", part, start, start + len(part), utc_now_iso()),
            )
    via_table = _outline_for_item(tools.db, mid, version, subject, content)
    assert via_table == _content_outline(subject, content)
    # 12 段 > 8 段上限：两条路都带「还有 4 段」尾标
    assert via_table[-1]["head"] == "…还有 4 段" and via_table[-1]["offset"] is None


# ── 0.17.0: hit_window 邻句窗口 + F1 旧版本命中丢弃 ──────────────────────────


def _window_hit(kind: str, start: int, end: int, *, row_index: int | None = None,
                row_version: int | None = None) -> dict:
    h = {"kind": kind, "text": "raw unit text irrelevant — spans slice source",
         "start_offset": start, "end_offset": end, "score": 0.9}
    if row_index is not None:
        h["row_index"] = row_index
    if row_version is not None:
        h["row_version"] = row_version
    return h


def _window_rows(content: str, sentences: list[str], *, subject_row: bool = False) -> list[dict]:
    spans = _sent_spans(content, sentences)
    rows: list[dict] = []
    start_index = 0
    if subject_row:
        rows.append({"unit_index": 0, "kind": "subject", "start_offset": 0, "end_offset": 0})
        start_index = 1
    for offset, (s, e) in enumerate(spans):
        rows.append({"unit_index": offset + start_index, "kind": "sentence",
                     "start_offset": s, "end_offset": e})
    return rows


def _sent_spans(content: str, sentences: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for sentence in sentences:
        start = content.index(sentence, pos)
        spans.append((start, start + len(sentence)))
        pos = start + len(sentence)
    return spans


def test_hits_window_default_zero_byte_identical() -> None:
    """不传 hit_window：span 条目逐键与 v0.15.10 一致——无 matched、无
    row_index 外发、无 stale_hit_spans（行为门 §4.1）。"""
    content = _cyclic_content(100)
    hits = [_window_hit("text", 60, 75, row_index=3, row_version=1),
            _window_hit("text", 10, 30)]
    preview = _preview_item(
        {"subject": "s", "content": content, "version": 1, "_evidence_hits": hits},
        content_mode="hits",
    )
    spans = preview["hit_spans"]
    assert [(sp["start_offset"], sp["end_offset"]) for sp in spans] == [(10, 30), (60, 75)]
    for sp in spans:
        assert set(sp.keys()) == {"text", "start_offset", "end_offset"}
    assert "stale_hit_spans" not in preview
    assert "content" not in preview


def test_hits_window_one_includes_neighbors() -> None:
    sentences = ["首句介绍背景情况。", "次句给出核心论断甲。", "三句展开论断甲细节。",
                 "四句补充边界条件乙。", "末句总结全部内容。"]
    content = "\n".join(sentences)
    all_spans = _sent_spans(content, sentences)
    preview = _preview_item(
        {"subject": "s", "content": content, "version": 1,
         "_evidence_hits": [_window_hit("text", *all_spans[2], row_index=2, row_version=1)]},
        content_mode="hits", hit_window=1,
        window_rows=_window_rows(content, sentences),
    )
    spans = preview["hit_spans"]
    assert [(sp["start_offset"], sp["end_offset"]) for sp in spans] == all_spans[1:4]
    assert [sp["matched"] for sp in spans] == [False, True, False]
    for sp in spans:
        assert sp["text"] == content[sp["start_offset"]:sp["end_offset"]]
        assert "\n" not in sp["text"]  # 完整句，非字符截断


def test_hits_window_never_pulls_subject() -> None:
    sentences = ["甲句讲规则一。", "乙句讲规则二。", "丙句讲规则三。"]
    content = "\n".join(sentences)
    all_spans = _sent_spans(content, sentences)
    # subject 行在窗口范围内（row_index=0，哨兵 (0,0)）：命中 row_index=1，window=1。
    rows = _window_rows(content, sentences, subject_row=True)
    preview = _preview_item(
        {"subject": "subject-哨兵文本", "content": content, "version": 1,
         "_evidence_hits": [_window_hit("text", *all_spans[0], row_index=1, row_version=1)]},
        content_mode="hits", hit_window=1, window_rows=rows,
    )
    spans = preview["hit_spans"]
    for sp in spans:
        assert (sp["start_offset"], sp["end_offset"]) != (0, 0)
        assert "subject-哨兵文本" not in sp["text"]


def test_hits_window_coverage_upgrade() -> None:
    sentences = ["前半句话讲催收规范要求。", "后半句话记录债务转移。"]
    content = "\n".join(sentences)
    all_spans = _sent_spans(content, sentences)
    preview = _preview_item(
        {"subject": "s", "content": content, "version": 1,
         "_evidence_hits": [_window_hit("text", *all_spans[0], row_index=0, row_version=1)]},
        content_mode="hits", hit_window=1,
        window_rows=_window_rows(content, sentences),
    )
    # 窗口扩展后覆盖 100% ≥ 50% → 升级全文，hit_spans 保留作标注。
    assert preview["content"] == content
    assert len(preview["hit_spans"]) == 2
    assert {sp["matched"] for sp in preview["hit_spans"]} == {True, False}


def test_hits_window_clamped_and_invalid(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="clamp 验证正文内容讲发布流程", subject="clamp", tags=[])
    clamped = tools.memory_search(query="clamp", content_mode="hits", hit_window=99)
    assert clamped["ok"] is True
    assert any("hit_window clamped to 5" in w for w in clamped["warnings"])
    for bad in ("x", -1):
        res = tools.memory_search(query="clamp", content_mode="hits", hit_window=bad)
        assert res["ok"] is True
        assert not any("hit_window" in w for w in res["warnings"])
    # 非 hits 档同传：静默忽略（无 clamp warning）。
    preview_res = tools.memory_search(query="clamp", content_mode="preview", hit_window=99)
    assert preview_res["ok"] is True
    assert not any("hit_window" in w for w in preview_res["warnings"])


def test_hits_window_adjacent_spans_stay_separate() -> None:
    # 命中句与邻句之间有分隔空白 → 不强并，各 span 精确到句。
    sentences = ["甲句讲规则一。", "乙句讲规则二。", "丙句讲规则三。"]
    content = "　".join(sentences)  # 全角空格分隔
    all_spans = _sent_spans(content, sentences)
    assert all_spans[1][0] > all_spans[0][1]  # 确有间隔
    preview = _preview_item(
        {"subject": "s", "content": content, "version": 1,
         "_evidence_hits": [_window_hit("text", *all_spans[1], row_index=1, row_version=1)]},
        content_mode="hits", hit_window=1,
        window_rows=_window_rows(content, sentences),
    )
    spans = preview["hit_spans"]
    assert [(sp["start_offset"], sp["end_offset"]) for sp in spans] == all_spans
    assert [sp["text"] for sp in spans] == sentences


def test_hits_window_directly_adjacent_merge_carries_hit_flag() -> None:
    # 无分隔直接相连 → 合并规则照旧作用于扩展后区间集；合并段含命中 → matched=true。
    content = "甲句内容。乙句内容。丙句内容。"
    bounds = [(0, 5), (5, 10), (10, 15)]
    preview = _preview_item(
        {"subject": "s", "content": content, "version": 1,
         "_evidence_hits": [_window_hit("text", 5, 10, row_index=1, row_version=1)]},
        content_mode="hits", hit_window=1,
        window_rows=[{"unit_index": i, "kind": "sentence", "start_offset": s, "end_offset": e}
                     for i, (s, e) in enumerate(bounds)],
    )
    spans = preview["hit_spans"]
    assert [(sp["start_offset"], sp["end_offset"]) for sp in spans] == [(0, 15)]
    assert spans[0]["matched"] is True
    assert spans[0]["text"] == content[0:15]


def test_hits_window_edge_cases() -> None:
    # 空 content：区间校验恒拒 → 无 hit_spans（钉死防回归）。
    empty = _preview_item(
        {"subject": "s", "content": "", "version": 1,
         "_evidence_hits": [_window_hit("text", 0, 5, row_index=0, row_version=1)]},
        content_mode="hits", hit_window=2,
        window_rows=[{"unit_index": 1, "kind": "sentence", "start_offset": 0, "end_offset": 5}],
    )
    assert "hit_spans" not in empty
    # 单行 memory：window 退化为自身 → 无邻句 → 无 matched 键（形状不变）。
    content = "唯一一句完整的话。"
    single = _preview_item(
        {"subject": "s", "content": content, "version": 1,
         "_evidence_hits": [_window_hit("text", 0, len(content), row_index=0, row_version=1)]},
        content_mode="hits", hit_window=3,
        window_rows=[{"unit_index": 0, "kind": "sentence", "start_offset": 0, "end_offset": len(content)}],
    )
    spans = single["hit_spans"]
    assert [(sp["start_offset"], sp["end_offset"]) for sp in spans] == [(0, len(content))]
    assert all("matched" not in sp for sp in spans)
    # 单行覆盖 100% → 升级全文照常发生。
    assert single["content"] == content


def test_hits_stale_version_dropped_with_signal(tmp_path: Path) -> None:
    """F1（owner 拍板）：行版本 ≠ 记忆版本的命中被丢弃（绝不静默切错区域），
    条目带 stale_hit_spans，响应 warnings 含重新查询提示。"""
    pytest.importorskip("sqlite_vec")
    tools = _make_vec_tools(tmp_path)
    mid, _ = tools.db.insert_memory(MemoryRecord(
        content="旧版本正文讲负债重组流程安排。", agent_id="a", workspace="ws",
        subject="负债重组", tags=[],
    ))
    import json as _json
    from memory_arbiter.models import utc_now_iso
    with tools.db.write_transaction() as conn:
        cur = conn.execute(
            "INSERT INTO memory_row(memory_id, memory_version, content_hash,"
            " row_index, kind, text, start_offset, end_offset, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, 1, "h", 0, "sentence", "旧版本正文讲负债重组流程安排。", 0, 14, utc_now_iso()),
        )
        conn.execute(
            "INSERT INTO memory_row_vec(id, parent_status, embedding) VALUES (?,?,?)",
            (cur.lastrowid, "active", _json.dumps([0.0, 1.0])),
        )
    # 模拟编辑 bump version 但行未重发布（异步窗口期）。
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET version=2, content=? WHERE id=?",
            ("新版本正文完全不同的话题内容。", mid),
        )
    res = tools.memory_search(
        query="负债重组", query_embedding=[0.0, 1.0],
        content_mode="hits", hit_window=2,
    )
    assert res["ok"] is True, res["data"]
    item = res["data"]["results"][0]
    assert "hit_spans" not in item  # 丢弃，绝不切错区域
    assert item["stale_hit_spans"] == {"evidence_version": 1, "memory_version": 2}
    assert any("re-query" in w for w in res["warnings"])


def test_hits_partial_stale_window_survives(tmp_path: Path) -> None:
    """部分命中过期部分新鲜：新鲜命中照常出 span（含窗口扩展），stale 如实上报。"""
    pytest.importorskip("sqlite_vec")
    tools = _make_vec_tools(tmp_path)
    sentences = ["过期句讲旧协议版本一。", "新鲜句讲新协议版本二。", "邻近句补充细节说明。"]
    content = "\n".join(sentences)
    mid, _ = tools.db.insert_memory(MemoryRecord(
        content=content, agent_id="a", workspace="ws", subject="协议版本", tags=[],
    ))
    import json as _json
    from memory_arbiter.models import utc_now_iso
    all_spans = _sent_spans(content, sentences)
    with tools.db.write_transaction() as conn:
        # row_index=0 @v1（过期，占住 index 0）；row_index=1,2 @v2（当前）。
        stale = conn.execute(
            "INSERT INTO memory_row(memory_id, memory_version, content_hash,"
            " row_index, kind, text, start_offset, end_offset, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, 1, "h", 0, "sentence", sentences[0], *all_spans[0], utc_now_iso()),
        )
        conn.execute(
            "INSERT INTO memory_row_vec(id, parent_status, embedding) VALUES (?,?,?)",
            (stale.lastrowid, "active", _json.dumps([0.0, 1.0])),
        )
        # row_index=1 @v2 带向量 = 新鲜命中；row_index=2 只入 memory_row 不入
        # vec：row_knn 返回 k 近邻不筛距离，带向量的行永远是命中——无向量的
        # 行只能经窗口通道作为邻句带出。
        fresh = conn.execute(
            "INSERT INTO memory_row(memory_id, memory_version, content_hash,"
            " row_index, kind, text, start_offset, end_offset, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, 2, "h2", 1, "sentence", sentences[1], *all_spans[1], utc_now_iso()),
        )
        conn.execute(
            "INSERT INTO memory_row_vec(id, parent_status, embedding) VALUES (?,?,?)",
            (fresh.lastrowid, "active", _json.dumps([0.0, 1.0])),
        )
        neighbour = conn.execute(
            "INSERT INTO memory_row(memory_id, memory_version, content_hash,"
            " row_index, kind, text, start_offset, end_offset, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, 2, "h2", 2, "sentence", sentences[2], *all_spans[2], utc_now_iso()),
        )
        assert neighbour.lastrowid is not None
        conn.execute("UPDATE memories SET version=2 WHERE id=?", (mid,))
    res = tools.memory_search(
        query="协议版本", query_embedding=[0.0, 1.0],
        content_mode="hits", hit_window=1,
    )
    assert res["ok"] is True, res["data"]
    item = res["data"]["results"][0]
    spans = item["hit_spans"]
    # 新鲜命中 row_index=1 扩展 → 邻 row_index=2 带出；过期 row_index=0 不参与。
    assert [(sp["start_offset"], sp["end_offset"]) for sp in spans] == all_spans[1:3]
    assert [sp["matched"] for sp in spans] == [True, False]
    assert item["stale_hit_spans"] == {"evidence_version": 1, "memory_version": 2}
