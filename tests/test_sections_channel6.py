from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from memory_arbiter.arbitration import compare_memories
from memory_arbiter.config import Settings, parse_bool
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import SourceType
from memory_arbiter.tools import MemoryTools


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



def _keyword_embedding(text: str) -> list[float]:
    """Deterministic 2D embedding keyed off the first token in text.

    Maps the first run of word-ish chars to one of a fixed set of orthogonal /
    diametrically-opposed unit vectors so cosine distances are predictable:
    two distinct known tokens are always > 0.7 apart, while identical tokens
    are at distance 0.  Unknown tokens map to a sentinel direction.
    """
    # Ordered fixed directions around the unit circle (90° apart → cosine dist 1.0
    # between neighbours, 2.0 between opposites).  The key's hash picks one index.
    m = re.search(r"[A-Za-z0-9\u4e00-\u9fff]+", text or "")
    key = m.group(0) if m else "x"
    table = {
        "alpha": (1.0, 0.0),
        "beta": (0.0, 1.0),
        "gamma": (-1.0, 0.0),
        "delta": (0.0, -1.0),
    }
    return [table[key][0], table[key][1]] if key in table else [-0.7071, -0.7071]


def _keyword_embedder(space_id: str = "mock_space_id"):
    """A ManagedEmbedder-like mock whose embedding is keyed off text content."""
    return _MockManagedEmbedder(lambda text: _keyword_embedding(text))


def _set_vec_ready(tools: MemoryTools, space_id: str = "mock_space_id") -> None:
    with tools.db.write_transaction() as conn:
        MemoryDB._set_meta(conn, "state", "ready")
        MemoryDB._set_meta(conn, "active_space_id", space_id)


def _publish_two_sections(tools: MemoryTools, memory_id: int, content: str,
                          first_anchor_token: str, second_anchor_token: str) -> dict:
    """Helper: publish a 2-section split whose anchors genuinely exist in content."""
    mem = tools.db.get_memory(memory_id)
    return tools.memory_split(
        memory_id=memory_id,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": first_anchor_token},
            {"title": second_anchor_token, "anchor_text": second_anchor_token, "occurrence_index": 0},
        ],
    )


def test_split_publish_success_then_search_returns_matched_sections(tmp_path: Path) -> None:
    """Happy path: valid anchors → offsets resolve → publish → search returns matched_sections."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    # Content: two distinct anchors so each section embeds to a different vector.
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)

    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True, published
    assert published["data"]["split_active"] is True
    assert published["data"]["section_count"] == 2

    mem = tools.db.get_memory(memory_id)
    assert mem["split_status"] == "active"
    # Sections + section vecs written
    sections = tools.db.get_sections_by_memory(memory_id)
    assert len(sections) == 2
    with tools.db.connection() as conn:
        vec_ids = MemoryDB._get_section_vec_ids(conn, memory_id)
    assert len(vec_ids) == 2

    # Search with a query whose embedding matches one section's keyword.
    result = tools.memory_search(query="beta", query_embedding=_keyword_embedding("beta"))
    assert result["ok"] is True
    hit = next(r for r in result["data"]["results"] if r["id"] == memory_id)
    # 1/2 matched → partial branch → content_scope=matched_sections, full
    # section bodies present in matched_sections.
    assert hit["content_scope"] == "matched_sections"
    assert hit["section_enhancement_applied"] is True
    assert hit.get("matched_sections")
    assert hit["matched_sections"][0]["title"] == "beta"
    # v0.8: matched_sections carry the full section content.
    assert hit["matched_sections"][0].get("content")


def test_split_publish_success_zero_hit_returns_full_memory(tmp_path: Path) -> None:
    """Zero section matches → return the FULL memory (design §6.3), no preview."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)
    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True

    # Query with a token that maps to neither section's vector.
    result = tools.memory_search(query="zzz", query_embedding=_keyword_embedding("zzz"))
    hit = next(r for r in result["data"]["results"] if r["id"] == memory_id)
    # v0.8: zero-match returns the full memory, not a bounded preview.
    assert hit["content_scope"] == "full_memory"
    assert hit["content"] == content
    assert hit.get("content_truncated") is None  # removed in v0.8
    assert "content_omitted" not in hit           # removed in v0.8
    assert hit["section_enhancement_applied"] is True


def test_split_publish_success_fulltext_fallback(tmp_path: Path) -> None:
    """When the matched fraction ≥ section_fulltext_threshold → return full text."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    # Lower fulltext threshold to 0.0 so any match counts as "most matched".
    tools.settings.section_fulltext_threshold = 0.0
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)
    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True

    result = tools.memory_search(query="alpha", query_embedding=_keyword_embedding("alpha"))
    hit = next(r for r in result["data"]["results"] if r["id"] == memory_id)
    assert hit["content_scope"] == "full_memory"
    assert hit["section_enhancement_applied"] is True
    assert hit.get("matched_sections")  # reference list still present
    assert hit.get("content")  # full text returned


def test_edit_clears_sections_and_bumps_revision(tmp_path: Path) -> None:
    """After a successful publish, editing content clears sections + bumps split_revision."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)
    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True
    assert tools.db.get_memory(memory_id)["split_revision"] == 1

    edited = tools.memory_edit(memory_id=memory_id, new_content=content + "\n appended")
    assert edited["ok"] is True

    after = tools.db.get_memory(memory_id)
    assert after["split_status"] is None
    assert after["split_revision"] == 2
    assert tools.db.get_sections_by_memory(memory_id) == []
    with tools.db.connection() as conn:
        assert MemoryDB._get_section_vec_ids(conn, memory_id) == set()


def test_attach_sections_invariant_missing_section_vec(tmp_path: Path) -> None:
    """Manually deleting a section vec → invariant reported + full text returned."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)
    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True

    # Sabotage: delete one section vector.
    sections = tools.db.get_sections_by_memory(memory_id)
    victim = sections[0]["id"]
    with tools.db.connection() as conn:
        conn.execute("DELETE FROM memory_sections_vec WHERE id = ?", (victim,))
        conn.commit()

    result = tools.memory_search(query="alpha", query_embedding=_keyword_embedding("alpha"))
    hit = next(r for r in result["data"]["results"] if r["id"] == memory_id)
    assert "split_invariant_broken_missing_section_vec" in hit.get("warnings", [])
    assert hit["content_scope"] == "full_memory"
    assert hit["section_enhancement_applied"] is False


def test_split_publish_rejects_when_vec_space_changed(tmp_path: Path) -> None:
    """active_space_id != embedder.embedding_space_id at publish → vec_space_changed, no write."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    # ready but active_space_id is a DIFFERENT space than the embedder.
    _set_vec_ready(tools, space_id="some_other_space")

    mem = tools.db.get_memory(memory_id)
    rejected = tools.memory_split(
        memory_id=memory_id,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": "alpha"},
            {"title": "beta", "anchor_text": "beta", "occurrence_index": 0},
        ],
    )
    assert rejected["ok"] is False
    assert rejected["data"]["error"] == "vec_space_changed"
    # Nothing written.
    assert tools.db.get_sections_by_memory(memory_id) == []
    assert tools.db.get_memory(memory_id)["split_status"] is None


def test_split_single_batch_ignores_legacy_batch_params(tmp_path: Path) -> None:
    """Regression: the dropped prepare_batch_index/llm_batch_chars kwargs are absorbed by **_, not errors."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)

    # Prepare call passing the now-removed params — must not raise.
    prepared = tools.memory_split(
        memory_id=memory_id,
        prepare_batch_index=0,  # legacy, now ignored via **_
        llm_batch_chars=9999,   # legacy, now ignored via **_
    )
    assert prepared["ok"] is True
    assert "content" in prepared["data"]
    # New single-batch response has no batch_count / llm_batch_chars fields.
    assert "batch_count" not in prepared["data"]
    assert "llm_batch_chars" not in prepared["data"]


def test_empty_embedding_not_stored_on_write_and_search(tmp_path: Path) -> None:
    """Never-raises contract: when embed_text returns an empty embedding (encode
    failure), memory_write must not store it and memory_search must not open the
    vec gate.  Does not require sqlite-vec.
    """
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
        embedding_auto_write=True,
        embedding_auto_query=True,
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))

    def bad_encode(_text: str) -> list[float]:
        raise RuntimeError("model crashed")

    failing = _MockManagedEmbedder(bad_encode)
    tools._ensure_embedder = lambda: (failing, [])  # type: ignore[method-assign]

    # ---- memory_write: empty embedding must not be stored ----
    written = tools.memory_write(content="body text", subject="subject")
    assert written["ok"] is True
    assert written["data"].get("embedding_stored") is not True
    assert any("auto-embedding write failed" in w for w in written["warnings"])

    # ---- memory_search: empty query embedding must not be used ----
    result = tools.memory_search(query="anything", query_embedding=None)
    assert result["ok"] is True
    assert any("auto-embedding query failed" in w for w in result["warnings"])


def test_empty_embedding_rejects_split_publish(tmp_path: Path) -> None:
    """Never-raises contract: when section embed_text returns empty, the split
    publish must be rejected and no sections/vecs written.
    """
    pytest.importorskip("sqlite_vec")
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)

    # Prepare with a working embedder (no embedding needed for prepare).
    mem = tools.db.get_memory(memory_id)
    prepared = tools.memory_split(memory_id=memory_id)
    assert prepared["ok"] is True

    # Switch to a failing embedder for the publish step.
    def bad_encode(_text: str) -> list[float]:
        raise RuntimeError("model crashed")

    failing = _MockManagedEmbedder(bad_encode)
    tools._ensure_embedder = lambda: (failing, [])  # type: ignore[method-assign]

    rejected = tools.memory_split(
        memory_id=memory_id,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": "alpha"},
            {"title": "beta", "anchor_text": "beta", "occurrence_index": 0},
        ],
    )
    assert rejected["ok"] is False
    assert "section embedding failed" in rejected["data"]["error"]
    # No sections or section vecs written.
    assert tools.db.get_sections_by_memory(memory_id) == []
    with tools.db.connection() as conn:
        vec_ids = MemoryDB._get_section_vec_ids(conn, memory_id)
    assert len(vec_ids) == 0


# ===========================================================================
# v0.6.1 Channel 6 (section-vec KNN) tests — T1 through T14.
# See docs/v0.6.1_detailed_design_channel6.md §6.2 for the test matrix.
# ===========================================================================


def _make_channel6_tools(tmp_path: Path, pool_cap: int = 50) -> MemoryTools:
    """Vec-enabled tools with split on + a small pool cap (for saturation tests)."""
    pytest.importorskip("sqlite_vec")
    settings = Settings(
        db_path=tmp_path / "ch6.sqlite3",
        backup_jsonl=tmp_path / "ch6-backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=2,
        split_threshold=1,
        recall_pool_cap=pool_cap,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_v061_t1_channel6_recalls_what_channel5_misses(tmp_path: Path) -> None:
    """T1: Channel 6 can surface a memory that Channel 5 cannot. We verify the
    mechanism directly: section_vec_knn returns the target (via its section vec)
    while vec_knn returns nothing (no memory-level vec stored). Per §6.2 the mock
    embedder can't simulate true dilution, so we test the mechanism, not the
    end-to-end KNN ranking."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # Target memory: NO memory-level vec stored (Channel 5 can't recall it).
    target_content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    target_id = tools.memory_write(content=target_content, subject="target")["data"]["id"]
    published = _publish_two_sections(tools, target_id, target_content, "alpha", "beta")
    assert published["ok"] is True

    # Channel 5 (memory-level vec KNN) finds nothing — no vec stored.
    ch5_rows = tools.db.vec_knn(_keyword_embedding("beta"), k=5)
    assert target_id not in [r["id"] for r in ch5_rows], (
        "Channel 5 should NOT recall a memory with no memory-level vec"
    )
    # Channel 6 (section-level vec KNN) DOES find it — the "beta" section vec matches.
    ch6_rows = tools.db.section_vec_knn(_keyword_embedding("beta"), k=5)
    recalled_ids = {r["memory_id"] for r in ch6_rows}
    assert target_id in recalled_ids, (
        "Channel 6 (section_vec_knn) should recall the target via its section vec"
    )


def test_v061_t2_dedup_one_memory_multiple_sections(tmp_path: Path) -> None:
    """T2: one memory with 3 sections all near the query enters the pool once."""
    tools = _make_channel6_tools(tmp_path, pool_cap=10)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # Three sections, each starting with "alpha" so each section vector = (1,0).
    content = "alpha " + ("x" * 60) + "\nalpha " + ("y" * 60) + "\nalpha " + ("z" * 60)
    mid = tools.memory_write(content=content, subject="dedup")["data"]["id"]
    mem = tools.db.get_memory(mid)
    published = tools.memory_split(
        memory_id=mid,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": "alpha"},
            {"title": "alpha", "anchor_text": "alpha", "occurrence_index": 1},
            {"title": "alpha", "anchor_text": "alpha", "occurrence_index": 2},
        ],
    )
    assert published["ok"] is True

    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    # The memory should appear exactly once in debug ranking (dedup held).
    target_entries = [r for r in result["data"]["results"] if r["id"] == mid]
    assert len(target_entries) == 1


def test_v061_t3_split_active_exempt_from_long_content_penalty(tmp_path: Path) -> None:
    """T3: a split-active long doc recalled by FTS does NOT get long-content penalty."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    long_content = "alpha " + ("x" * 3000) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=long_content, subject="long-content-split", tags=None)["data"]["id"]
    _publish_two_sections(tools, mid, long_content, "alpha", "beta")

    # Query "alpha" hits content lexically (FTS); subject set but tags None (weak tag signal).
    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    assert mid in debug_map, "target memory must appear in results for the test to be meaningful"
    notes = debug_map[mid].get("_ranking_notes", [])
    assert "long content penalty applied" not in notes, (
        "split-active long doc should be exempt from long-content penalty"
    )


def test_v061_t4_non_split_still_gets_long_content_penalty(tmp_path: Path) -> None:
    """T4: regression — a non-split long memory still incurs long-content penalty."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    long_content = "alpha " + ("x" * 3000)
    mid = tools.memory_write(content=long_content, subject="long-content-nosplit", tags=None)["data"]["id"]
    # NOT split — so the penalty should still apply.

    result = tools.memory_search(query="alpha", debug_ranking=True)
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    assert mid in debug_map
    notes = debug_map[mid].get("_ranking_notes", [])
    assert "long content penalty applied" in notes


def test_v061_t5_channel6_skipped_when_vec_disabled(tmp_path: Path) -> None:
    """T5: with the vec gate closed, no Channel 6 candidates appear."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    # Deliberately do NOT call _set_vec_ready — gate stays closed.
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="gate")["data"]["id"]
    _publish_two_sections(tools, mid, content, "alpha", "beta")

    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    for r in result["data"].get("_debug_ranking", []):
        assert not r.get("_section_vec_candidate"), (
            "Channel 6 must not fire when the vec gate is closed"
        )


def test_v061_t6_pool_cap_not_exceeded(tmp_path: Path) -> None:
    """T6: Channel 6 does not push the pool beyond pool_cap."""
    tools = _make_channel6_tools(tmp_path, pool_cap=4)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    for i in range(6):
        c = f"alpha {i} " + ("x" * 60) + "\nbeta " + ("y" * 60)
        mid = tools.memory_write(content=c, subject=f"cap-{i}")["data"]["id"]
        _publish_two_sections(tools, mid, c, "alpha", "beta")

    result = tools.memory_search(query="alpha", query_embedding=_keyword_embedding("alpha"))
    # The number of unique results never exceeds pool_cap.
    assert len(result["data"]["results"]) <= 4


def test_v061_t7_channel5_candidate_has_split_status(tmp_path: Path) -> None:
    """T7: vec_knn (Channel 5) now returns split_status (§2.1前置 checkpoint)."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="t7")["data"]["id"]
    # memory_write doesn't auto-store a memory-level vec under the mock embedder
    # (embedding_provider != "gguf"), so store one explicitly.
    tools.memory_store_embedding(mid, _keyword_embedding("alpha"))
    _publish_two_sections(tools, mid, content, "alpha", "beta")

    rows = tools.db.vec_knn(_keyword_embedding("alpha"), k=5)
    target_row = next((r for r in rows if r["id"] == mid), None)
    assert target_row is not None, "target not in Channel 5 KNN results"
    assert "split_status" in target_row, "vec_knn must return split_status (§2.1)"
    assert target_row["split_status"] == "active"


def test_v080_long_content_zero_match_returns_full_memory(tmp_path: Path) -> None:
    """v0.8 §6.3: a long split-active doc under zero-match returns the FULL
    memory, not a bounded preview. Supersedes the v0.6.1 preview behaviour."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # Two compliant sections (each ≤ max_section_chars=3600), total > preview.
    chunk_a = "alpha " + ("q" * 3000)
    chunk_b = "beta " + ("r" * 3000)
    big = chunk_a + "\n" + chunk_b
    mid = tools.memory_write(content=big, subject="bigdoc")["data"]["id"]
    published = _publish_two_sections(tools, mid, big, "alpha", "beta")
    assert published["ok"] is True

    result = tools.memory_search(query="zzz", query_embedding=_keyword_embedding("zzz"))
    hit = next((r for r in result["data"]["results"] if r["id"] == mid), None)
    assert hit is not None, "big doc must appear in zero-match results for the test to be meaningful"
    # v0.8: full memory returned, no truncation.
    assert hit["content_scope"] == "full_memory"
    assert hit["content"] == big
    assert len(hit["content"]) > 2000, "full text must exceed the legacy preview bound"
    assert "content_truncated" not in hit


def test_v061_t9_debug_ranking_channel6_fields(tmp_path: Path) -> None:
    """T9: a Channel 6 candidate carries section-vec debug fields + note. We feed
    a synthetic Channel 6 candidate through _soft_rerank directly — in a real
    end-to-end search the candidate may also be recalled by FTS first, masking
    the Channel 6 flag, so testing the scorer in isolation is more reliable."""
    from memory_arbiter.search import _soft_rerank

    candidate = {
        "id": 999,
        "subject": "t9",
        "tags": "[]",
        "content": "",               # Channel 6 omits content (A3)
        "split_status": "active",
        "status": "active",
        "source_type": "unknown",
        "protection_level": "normal",
        "ingest_time": "2020-01-01T00:00:00Z",
        "_vec_candidate": True,
        "_section_vec_candidate": True,
        "_section_vec_distance": 0.3,
        "_section_vec_section_id": 42,
    }
    reranked = _soft_rerank("anything", [candidate])
    rec = reranked[0]
    assert rec.get("_section_vec_candidate") is True
    assert rec.get("_section_vec_distance") == 0.3
    assert rec.get("_section_vec_section_id") == 42
    notes = rec.get("_ranking_notes", [])
    assert "section-vec recall candidate (Channel 6)" in notes


def test_v061_t10_penalty_baseline_c9(tmp_path: Path) -> None:
    """T10/C9 (blocking): non-split long memory incurs BOTH content_only and
    long-content penalties — the baseline T3/T4 exemption logic regresses against."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    long_content = "alpha " + ("x" * 3000)
    mid = tools.memory_write(content=long_content, subject="long-content-penalty", tags=None)["data"]["id"]
    result = tools.memory_search(query="alpha", debug_ranking=True)
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    assert mid in debug_map
    rec = debug_map[mid]
    notes = rec.get("_ranking_notes", [])
    assert "content_only_match" in str(rec.get("_match_reason", "")) or any(
        "matched content but not subject/tags" in n for n in notes
    )
    assert "long content penalty applied" in notes
    # relevance = 3.0 - 2.0 - 1.5 = -0.5; trust=0 (default source), recency ∈ [0, 0.8]
    assert -0.5 <= rec["_final_score"] < 0.5


def test_v061_t11_pool_saturation_skips_channel6_c2(tmp_path: Path) -> None:
    """T11/C2 (blocking): when Channels 1-5 fill the pool, Channel 6 is skipped."""
    tools = _make_channel6_tools(tmp_path, pool_cap=2)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # Two FTS-recallable memories fill the small pool (cap=2) before Channel 6.
    tools.memory_write(content="alpha match one " + "x" * 60, subject="fill-1")
    tools.memory_write(content="alpha match two " + "y" * 60, subject="fill-2")
    # A split-active memory that only Channel 6 could surface.
    target_content = "gamma " + ("x" * 60) + "\nbeta " + ("y" * 60)
    target_id = tools.memory_write(content=target_content, subject="late")["data"]["id"]
    _publish_two_sections(tools, target_id, target_content, "gamma", "beta")

    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    # Pool was saturated by FTS hits → Channel 6 never ran → no section-vec candidates.
    assert all(
        not r.get("_section_vec_candidate") for r in result["data"]["results"]
    ), "Channel 6 should be skipped when the pool is already full"


def test_v061_t12_content_only_penalty_still_applies_to_split_active(tmp_path: Path) -> None:
    """T12 (A5 regression): content_only_penalty still hits split-active, but
    long-content penalty is exempted. Score uses a range (not exact) per §6.2."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    long_content = "alpha " + ("x" * 3000) + "\nbeta " + ("y" * 60)
    mid = tools.memory_write(content=long_content, subject="long-content-section-embed", tags=None)["data"]["id"]
    _publish_two_sections(tools, mid, long_content, "alpha", "beta")

    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    assert mid in debug_map, "target memory must appear in results for the test to be meaningful"
    rec = debug_map[mid]
    notes = rec.get("_ranking_notes", [])
    # The core assertion: split-active exempts long-content penalty regardless
    # of which channel recalled the memory.
    assert "long content penalty applied" not in notes, (
        "split-active long doc should be exempt from long-content penalty"
    )
    # content_only_penalty still applies on the FTS path (A5: NOT exempted).
    # Only check this when the memory was NOT recalled by vec (vec floor would
    # mask the content_only signal).
    if not rec.get("_vec_candidate"):
        assert any("matched content but not subject/tags" in n for n in notes) or (
            rec.get("_match_reason") == "content_only_match"
        )
        # relevance = 3.0 - 2.0 = 1.0; with recency the final lands in [1.0, 2.0).
        assert 1.0 <= rec["_final_score"] < 2.0


def test_v061_t13_fulltext_branch_channel6_not_empty_content(tmp_path: Path) -> None:
    """T13 (blocking, second-round Bug regression): a Channel 6-only candidate
    entering the fulltext branch must NOT return empty content. §4.2 归一化 fixes
    this. We unit-test _attach_sections directly with a content='' candidate."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    full_text = "alpha " + ("x" * 60) + "\nbeta " + ("y" * 60)
    mid = tools.memory_write(content=full_text, subject="t13")["data"]["id"]
    _publish_two_sections(tools, mid, full_text, "alpha", "beta")
    # Lower fulltext threshold so ≥1 match counts as "most matched".
    tools.settings.section_fulltext_threshold = 0.0

    # Simulate a Channel 6 candidate: content="" but the memory is split-active.
    # _attach_sections must normalize content from current_mem_map.
    fake_candidate = {
        "id": mid,
        "content": "",          # Channel 6 deliberately omits content (A3)
        "split_status": "active",
        "_vec_candidate": True,
        "_section_vec_candidate": True,
        "subject": "t13",
    }
    normalized = tools._attach_sections(
        [fake_candidate], _keyword_embedding("alpha"), []
    )
    hit = normalized[0]
    # v0.8: fulltext branch returns the full memory for Channel 6 candidates
    # (content normalized from current_mem_map).
    assert hit.get("content_scope") == "full_memory"
    assert hit.get("content"), (
        "fulltext branch must return non-empty content for Channel 6 candidates "
        "(second-round Bug regression)"
    )


def test_v061_t14_partial_branch_does_not_leak_full_content(tmp_path: Path) -> None:
    """T14 (§4.2 interaction): a Channel 6 candidate in the partial branch must
    have content=None (normalization's full text is correctly discarded)."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    full_text = "alpha " + ("x" * 60) + "\nbeta " + ("y" * 60)
    mid = tools.memory_write(content=full_text, subject="t14")["data"]["id"]
    _publish_two_sections(tools, mid, full_text, "alpha", "beta")

    # Query "alpha" → section "alpha" matches (1/2 = partial, below fulltext 0.8).
    fake_candidate = {
        "id": mid,
        "content": "",
        "split_status": "active",
        "_vec_candidate": True,
        "_section_vec_candidate": True,
        "subject": "t14",
    }
    normalized = tools._attach_sections(
        [fake_candidate], _keyword_embedding("alpha"), []
    )
    hit = normalized[0]
    # v0.8: partial branch returns only the matched section's full text (joined),
    # NOT the full memory — the unmatched "beta" section body must not leak.
    assert hit.get("content_scope") == "matched_sections"
    assert hit.get("content")  # non-empty: the matched section's text
    # The matched alpha section's content is present...
    assert any(ms.get("content") for ms in hit.get("matched_sections", []))
    # ...but the unmatched beta section body must NOT appear in the partial content.
    beta_body = full_text.split("\nbeta ", 1)[-1] if "\nbeta " in full_text else "y" * 60
    assert beta_body not in hit["content"]


# ===========================================================================
# v0.6.3 provenance tests — section source attribution (parser vs agent).
# ===========================================================================


def test_v080_provenance_is_explicit_not_inferred(tmp_path: Path) -> None:
    """v0.8 (§6.2): provenance is an explicit caller argument, not inferred
    from anchor text. Two paths, two values:
      * rules path (memory_write auto-split)         → provenance='parser'
      * agent path (memory_split publish)            → provenance='agent'
    """
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # --- rules path: Markdown headings → auto-split → provenance='parser' ---
    md_content = (
        "# alpha\n" + ("x" * 60) + "\n\n"
        "## beta\n" + ("y" * 60)
    )
    md_mid = tools.memory_write(content=md_content, subject="md-doc")["data"]["id"]
    md_sections = tools.db.get_sections_by_memory(md_mid)
    assert len(md_sections) == 2, "write should auto-split the heading doc"
    assert all(s["provenance"] == "parser" for s in md_sections)

    # --- agent path: plain text (no headings) → write returns a split_request,
    #     Agent publishes via memory_split → provenance='agent' ---
    plain = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    plain_mid = tools.memory_write(content=plain, subject="plain-doc")["data"]["id"]
    mem = tools.db.get_memory(plain_mid)
    assert mem["split_status"] is None   # write did not split (no headings)
    published = tools.memory_split(
        memory_id=plain_mid,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": "alpha"},
            {"title": "beta", "anchor_text": "beta", "occurrence_index": 0},
        ],
    )
    assert published["ok"] is True
    agent_sections = tools.db.get_sections_by_memory(plain_mid)
    assert all(s["provenance"] == "agent" for s in agent_sections)


def test_v061_r1_channel6_recall_superseded_with_expired_search(tmp_path: Path) -> None:
    """R1: memory_search_expired recalls a superseded split-active memory.

    Channel 6's post-filter uses the active/expired query split. The
    memory_search_expired path queries parent_status_filter="expired" so
    superseded memories remain reachable via their section vectors.
    """
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # A split-active memory (no memory-level vec → only Channel 6 can recall).
    target_content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    target_id = tools.memory_write(content=target_content, subject="stale")["data"]["id"]
    _publish_two_sections(tools, target_id, target_content, "alpha", "beta")

    # Supersede it (requires authorization).
    replacement = tools.memory_write(content="replacement active", subject="stale")
    tools.memory_supersede(
        memory_id=target_id, reason="replaced",
        superseded_by=replacement["data"]["id"], authorized=True,
    )

    # Default search excludes superseded → target should be absent.
    default_result = tools.memory_search(
        query="beta", query_embedding=_keyword_embedding("beta"), debug_ranking=True
    )
    default_ids = {r["id"] for r in default_result["data"]["results"]}
    assert target_id not in default_ids, "superseded memory leaked into default search"

    # memory_search_expired should recall it via section vec.
    expired_result = tools.memory_search_expired(
        query="beta", query_embedding=_keyword_embedding("beta"),
        debug_ranking=True,
    )
    expired_map = {r["id"]: r for r in expired_result["data"]["results"]}
    assert target_id in expired_map, (
        "Channel 6 should recall superseded split-active memory via "
        "memory_search_expired"
    )


def test_knn_prefilters_inactive_parent_before_top_k(tmp_path: Path) -> None:
    """A closer superseded memory/section cannot consume the default KNN slot.

    v0.9.4: supersede now marks vectors as 'superseded' (not cascade-deletes).
    The metadata predicate ``AND v.parent_status='active'`` pre-filters at
    KNN scan time. The superseded memory's vectors are still there, but
    they are only visible via ``memory_search_expired`` which uses
    ``parent_status_filter="expired"`` to query non-active non-deleted
    vectors."""
    pytest.importorskip("sqlite_vec")
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    stale_content = "alpha " + ("x" * 60) + "\nbeta " + ("y" * 60)
    stale_id = tools.memory_write(content=stale_content, subject="alpha")["data"]["id"]
    assert _publish_two_sections(
        tools, stale_id, stale_content, "alpha", "beta"
    )["ok"] is True

    active_content = "gamma " + ("x" * 60) + "\ndelta " + ("y" * 60)
    active_id = tools.memory_write(content=active_content, subject="gamma")["data"]["id"]
    assert _publish_two_sections(
        tools, active_id, active_content, "gamma", "delta"
    )["ok"] is True
    assert tools.db.store_embedding(
        stale_id, _keyword_embedding("alpha")
    )[0] is True
    assert tools.db.store_embedding(
        active_id, _keyword_embedding("gamma")
    )[0] is True

    assert tools.memory_supersede(
        memory_id=stale_id,
        reason="replaced",
        authorized=True,
    )["ok"] is True

    # Default KNN (active only): stale's vector is marked 'superseded',
    # so the metadata predicate v.parent_status='active' excludes it.
    default_memory_rows = tools.db.vec_knn(_keyword_embedding("alpha"), k=1)
    assert len(default_memory_rows) == 1
    assert default_memory_rows[0]["id"] == active_id

    default_rows = tools.db.section_vec_knn(
        _keyword_embedding("alpha"), k=1
    )
    assert len(default_rows) == 1
    assert default_rows[0]["memory_id"] == active_id

    # memory_search_expired: parent_status_filter="expired" queries non-active
    # non-deleted vectors, so the stale memory IS returned.
    expired_memory_rows = tools.db.vec_knn(
        _keyword_embedding("alpha"), k=5, parent_status_filter="expired"
    )
    assert any(r["id"] == stale_id for r in expired_memory_rows), (
        "superseded memory should be recalled via vec_knn(parent_status_filter=\"expired\")"
    )
    expired_rows = tools.db.section_vec_knn(
        _keyword_embedding("alpha"), k=5, parent_status_filter="expired"
    )
    assert any(r["memory_id"] == stale_id for r in expired_rows), (
        "superseded section should be recalled via section_vec_knn(parent_status_filter=\"expired\")"
    )


# ---- v0.7.3 change 1: tag scoring unit tests (design §2.6 matrix) ------
# These exercise _score_tags_surface / _cjk_substring_match /
# _normalize_token_for_tag_match directly, not the full search pipeline.

from memory_arbiter.search import (
    _score_tags_surface,
    _cjk_substring_match,
    _normalize_token_for_tag_match,
    _is_pure_cjk_token,
    _TAGS_STRONG_WEIGHT,
    _TAGS_MEDIUM_WEIGHT,
    _TAGS_WEAK_WEIGHT,
    _TAGS_SCORE_CAP,
    SearchOutcome,
)



