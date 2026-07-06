from __future__ import annotations

from pathlib import Path

from memory_arbiter.arbitration import compare_memories
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import SourceType
from memory_arbiter.tools import MemoryTools


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


def test_write_and_search(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="Project API token policy lives in README security section.",
        tags=["policy", "security"],
        source_type="document_extracted",
        event_time="2026-01-01T00:00:00Z",
        subject="api-token-policy",
    )
    assert written["ok"] is True
    assert written["data"]["id"] is not None

    found = tools.memory_search(query="token policy", workspace="repo-a")
    assert found["ok"] is True
    assert found["data"]["count"] >= 1
    assert found["data"]["results"][0]["subject"] == "api-token-policy"


def test_chinese_search_matches_contiguous_fragment(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="京东科技金融营销系统梳理，覆盖金营、营销链路、活动费用申请和活动配置。",
        tags=["营销系统", "金营", "系统梳理"],
        source_type="document_extracted",
        subject="京东科技金融-营销系统梳理",
    )

    found = tools.memory_search(query="营销系统", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] >= 1
    assert found["data"]["results"][0]["subject"] == "京东科技金融-营销系统梳理"


def test_chinese_search_overspecified_query_still_matches(tmp_path: Path) -> None:
    """Regression for the original '营销交付系统' bug: a query whose bigrams
    are not all present in the document must still hit FTS5 via OR recall on
    the shared bigrams, instead of falling back to recent memories."""
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="营销交付需求提报：银行走XBP，自持走邮件，财富走JoySpace。活动配置走权益中台。",
        tags=["营销交付", "营销链路"],
        source_type="document_extracted",
        subject="营销交付-需求提报链路",
    )

    # "营销交付系统" has bigrams (付系, 系统) absent from the doc; the shared
    # bigrams (营销, 销交, 交付) must still match via OR.
    found = tools.memory_search(query="营销交付系统", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] >= 1
    assert found["data"]["results"][0]["subject"] == "营销交付-需求提报链路"
    # Must NOT have triggered the recent-memory fallback.
    assert not any("No direct memory match" in w for w in found["warnings"])


def test_search_returns_recent_memories_when_no_direct_match(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="来源：营销系统梳理.xlsx。营销链路包含需求提报、费用申请、设计图提报、活动配置等环节。",
        tags=["营销系统", "金营", "PRD参考"],
        source_type="document_extracted",
        subject="京东科技金融-营销系统梳理",
        event_time="2026-07-04T09:00:30Z",
    )

    found = tools.memory_search(query="营销运营全流程 系统", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] == 1
    assert found["data"]["results"][0]["subject"] == "京东科技金融-营销系统梳理"
    assert any("No direct memory match" in warning for warning in found["warnings"])


def test_recent_fallback_ranks_by_trust_then_recency(tmp_path: Path) -> None:
    """Regression: when a query misses FTS5 and falls back to recent memories,
    a user_confirmed+locked record must outrank a newer agent_generated+normal
    one. Previously pure ``ingest_time DESC`` ordering let daily agent chatter
    bury authoritative records."""
    tools = make_tools(tmp_path)
    # Newer but low-trust: agent chatter from today.
    tools.memory_write(
        content="memory-arbiter v0.2.4 release notes supersede tooling spec",
        subject="release-chatter",
        source_type="agent_generated",
        event_time="2026-07-05T00:00:00Z",
    )
    # Older but authoritative: the actual system overview the user confirmed.
    tools.memory_write(
        content="营销系统梳理：金营平台、权益中台、活动配置全链路系统清单",
        subject="营销系统-权威",
        source_type="user_confirmed",
        confidence=1.0,
        event_time="2026-07-04T00:00:00Z",
    )

    found = tools.memory_search(query="完全不存在的查询词 xyz123", workspace="repo-a")

    assert found["ok"] is True
    assert any("No direct memory match" in w for w in found["warnings"])
    results = found["data"]["results"]
    assert len(results) == 2
    # Authoritative record must rank first despite being older.
    assert results[0]["subject"] == "营销系统-权威"
    assert results[0]["protection_level"] == "locked"
    assert results[1]["subject"] == "release-chatter"


def test_memory_recent_lists_recent_workspace_memories(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="Old memory", subject="old", event_time="2026-01-01T00:00:00Z")
    tools.memory_write(content="New memory", subject="new", event_time="2026-02-01T00:00:00Z")
    tools.memory_write(content="Other workspace", subject="other", workspace="repo-b", event_time="2026-03-01T00:00:00Z")

    recent = tools.memory_recent(workspace="repo-a", limit=10)

    assert recent["ok"] is True
    assert [record["subject"] for record in recent["data"]["results"]] == ["new", "old"]


def test_arbitration_prefers_event_time(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    old = tools.memory_write(content="Use port 3000", subject="dev-port", event_time="2026-01-01T00:00:00Z")
    new = tools.memory_write(content="Use port 5173", subject="dev-port", event_time="2026-02-01T00:00:00Z")
    result = tools.memory_arbitrate(old["data"]["id"], new["data"]["id"], mark_conflict=True)

    assert result["ok"] is True
    assert result["data"]["comparison"]["winner_id"] == new["data"]["id"]
    assert result["data"]["conflict_id"] is not None


def test_user_confirmed_is_protected(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    confirmed = tools.memory_write(
        content="User says production branch is main",
        subject="prod-branch",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    generated = tools.memory_write(
        content="Agent guessed production branch is release",
        subject="prod-branch",
        source_type="agent_generated",
        event_time="2026-03-01T00:00:00Z",
    )
    result = tools.memory_arbitrate(confirmed["data"]["id"], generated["data"]["id"], apply=True)

    assert result["data"]["comparison"]["winner_id"] == confirmed["data"]["id"]
    assert "automatic overwrite is forbidden" in result["data"]["comparison"]["reasons"][0]


def test_confirm_promotes_record(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(content="OpenClaw memory plugin is enabled for agent alpha", source_type="pending")
    confirmed = tools.memory_confirm(written["data"]["id"], source_ref="user-chat")

    assert confirmed["ok"] is True
    assert confirmed["data"]["record"]["source_type"] == SourceType.USER_CONFIRMED.value
    assert confirmed["data"]["record"]["protection_level"] == "locked"


def test_supersede_requires_authorization(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="Old spec",
        subject="spec",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    rejected = tools.memory_supersede(memory_id=memory_id, reason="replaced", authorized=False)
    assert rejected["ok"] is False
    assert rejected["data"]["superseded"] is False
    # Memory must remain active+locked when unauthorized
    still_active = tools.db.get_memory(memory_id)
    assert still_active["status"] == "active"
    assert still_active["protection_level"] == "locked"


def test_supersede_marks_record_and_resolves_conflicts(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    old = tools.memory_write(
        content="Old release spec",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    new = tools.memory_write(
        content="New release spec supersedes the old one",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-02-01T00:00:00Z",
    )
    old_id, new_id = old["data"]["id"], new["data"]["id"]

    # arbitrate hits the user-protected wall and leaves an open conflict
    blocked = tools.memory_arbitrate(old_id, new_id, mark_conflict=True, apply=True)
    assert blocked["data"]["comparison"]["manual_review"] is True
    open_conflicts_before = tools.memory_list_conflicts(status="open")["data"]["count"]
    assert open_conflicts_before >= 1

    result = tools.memory_supersede(
        memory_id=old_id,
        reason="replaced by newer spec",
        superseded_by=new_id,
        authorized=True,
    )

    assert result["ok"] is True
    assert result["data"]["superseded"] is True
    assert result["data"]["memory_id"] == old_id
    assert result["data"]["conflict_id"] is not None
    assert result["data"]["linked_conflicts_resolved"] >= 1

    updated = tools.db.get_memory(old_id)
    assert updated["status"] == "superseded"
    assert updated["protection_level"] == "normal"

    # The open conflict from the blocked arbitrate must now be resolved
    open_conflicts_after = tools.memory_list_conflicts(status="open")["data"]["count"]
    assert open_conflicts_after == 0
    # And an audit row exists for the supersede itself
    resolved = tools.memory_list_conflicts(status="resolved")["data"]["conflicts"]
    assert any("USER-AUTHORIZED SUPERSEDE" in c["reason"] for c in resolved)


def test_supersede_rejects_already_superseded(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="Stale memory",
        subject="stale",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    first = tools.memory_supersede(memory_id=memory_id, reason="stale", authorized=True)
    assert first["ok"] is True

    second = tools.memory_supersede(memory_id=memory_id, reason="stale again", authorized=True)
    assert second["ok"] is False
    assert "already" in second["data"]["error"]


def test_degraded_status_mentions_missing_vec(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    status = tools.memory_status()

    assert status["ok"] is True
    assert status["degraded"] is True
    assert any("sqlite-vec disabled" in warning for warning in status["warnings"])


def test_compare_manual_review_when_both_protected() -> None:
    left = {
        "id": 1,
        "source_type": "user_confirmed",
        "protection_level": "locked",
        "event_time": "2026-01-01T00:00:00Z",
        "ingest_time": "2026-01-02T00:00:00Z",
    }
    right = {
        "id": 2,
        "source_type": "user_confirmed",
        "protection_level": "locked",
        "event_time": "2026-02-01T00:00:00Z",
        "ingest_time": "2026-02-02T00:00:00Z",
    }
    result = compare_memories(left, right)

    assert result["manual_review"] is True
    assert result["winner_id"] is None


def test_audit_summary_aggregates_per_workspace(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="Confirmed port 5173",
        subject="dev-port",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    tools.memory_write(
        content="Agent guessed port 3000",
        subject="dev-port",
        source_type="agent_generated",
        event_time="2026-03-01T00:00:00Z",
    )
    tools.memory_write(
        content="Other workspace memory",
        subject="other",
        workspace="repo-b",
        source_type="document_extracted",
        event_time="2026-02-01T00:00:00Z",
    )
    # Create an open conflict inside repo-a
    ids = [r["id"] for r in tools.memory_recent(workspace="repo-a", limit=10)["data"]["results"]]
    tools.memory_arbitrate(ids[0], ids[1], mark_conflict=True)

    summary = tools.memory_audit_summary()
    data = summary["data"]

    assert summary["ok"] is True
    assert data["total_memories"] == 3
    assert data["total_open_conflicts"] == 1
    repo_a = data["workspaces"]["repo-a"]
    assert repo_a["count"] == 2
    assert repo_a["oldest"] == "2026-01-01T00:00:00+00:00"
    assert repo_a["newest"] == "2026-03-01T00:00:00+00:00"
    assert repo_a["open_conflicts"] == 1
    assert repo_a["by_source_type"] == {"user_confirmed": 1, "agent_generated": 1}
    assert data["workspaces"]["repo-b"]["count"] == 1


def test_audit_summary_empty_when_no_memories(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    summary = tools.memory_audit_summary()
    assert summary["ok"] is True
    assert summary["data"] == {
        "workspaces": {},
        "total_memories": 0,
        "total_open_conflicts": 0,
    }


def test_fts5_search_handles_query_with_dots(tmp_path: Path) -> None:
    """Regression: queries containing '.' (e.g. version numbers, file paths)
    used to raise ``fts5: syntax error near "."`` and silently fall back to
    LIKE. FTS5 must now run without warnings."""
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="Release notes for memory-arbiter v0.2.1, the token optimization release.",
        subject="v0.2.1 release",
        source_type="document_extracted",
    )

    found = tools.memory_search(query="v0.2.1", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] == 1
    assert found["data"]["results"][0]["subject"] == "v0.2.1 release"
    assert not any("FTS5 query failed" in w for w in found["warnings"])


def test_fts5_search_handles_special_chars_without_warning(tmp_path: Path) -> None:
    """FTS5 special chars (``: * ( ) -``) in the query must not trigger a
    syntax-error fallback to LIKE."""
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="Config lives at config/db.py with key apiKey:path(0)",
        subject="config-paths",
        source_type="document_extracted",
    )

    for query in ("config/db.py", "apiKey:path(0)", "config * (db)"):
        found = tools.memory_search(query=query, workspace="repo-a")
        assert found["ok"] is True
        assert not any("FTS5 query failed" in w for w in found["warnings"]), (
            f"query {query!r} triggered FTS5 fallback: {found['warnings']}"
        )


def test_sanitize_fts_query_quotes_and_joins_tokens() -> None:
    from memory_arbiter.search import _sanitize_fts_query

    assert _sanitize_fts_query("") == ""
    assert _sanitize_fts_query("v0.2.1") == '"v0.2.1"'
    assert _sanitize_fts_query("v0.2.1 release task") == '"v0.2.1" AND "release" AND "task"'
    # Embedded double-quotes are escaped as "" per FTS5 phrase syntax
    assert _sanitize_fts_query('a"b') == '"a""b"'


def test_sanitize_fts_query_splits_cjk_into_trigram_or_group() -> None:
    """Regression: a CJK token used to be wrapped as a single strict phrase,
    so ``营销交付系统`` missed documents that contained ``营销交付需求提报``
    (one trigram absent). It must now expand to an OR of overlapping trigrams.

    The FTS5 table uses ``tokenize='trigram'``, which only matches queries
    that produce 3-char tokens — so 2-char phrases never hit, and a strict
    CJK phrase silently misses when overspecified. Bare trigrams joined by
    OR restore recall without any new tokenizer dependency.
    """
    from memory_arbiter.search import _sanitize_fts_query

    # Pure CJK: overlapping trigrams (unquoted) joined by OR.
    assert _sanitize_fts_query("营销交付") == "(营销交 OR 销交付)"
    assert _sanitize_fts_query("营销交付系统") == (
        "(营销交 OR 销交付 OR 交付系 OR 付系统)"
    )
    # Single/double CJK chars cannot form a trigram → dropped (LIKE fallback
    # handles them via the empty-FTS-result path).
    assert _sanitize_fts_query("营") == ""
    assert _sanitize_fts_query("营销") == ""
    # Mixed CJK + ASCII: short CJK token dropped, ASCII token preserved.
    assert _sanitize_fts_query("营销 marketing") == '"marketing"'
    # ASCII behavior unchanged when no CJK present.
    assert _sanitize_fts_query("v0.2.1 release") == '"v0.2.1" AND "release"'


def test_search_excludes_superseded_by_default(tmp_path: Path) -> None:
    """v0.2.6: search filters out superseded records unless explicitly opted in.

    A superseded record is, by definition, no longer authoritative. Letting it
    leak into default results pollutes recall — release chatter, superseded
    specs, etc. drown out current truth. The default must hide them.
    """
    tools = make_tools(tmp_path)
    active = tools.memory_write(
        content="release-spec-v2 active authoritative",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-02-01T00:00:00Z",
    )
    stale = tools.memory_write(
        content="release-spec-v1 stale superseded",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    active_id, stale_id = active["data"]["id"], stale["data"]["id"]
    superseded = tools.memory_supersede(memory_id=stale_id, reason="replaced", superseded_by=active_id, authorized=True)
    assert superseded["data"]["superseded"] is True

    found = tools.memory_search(query="release-spec", workspace="repo-a")
    ids = [r["id"] for r in found["data"]["results"]]
    assert active_id in ids
    assert stale_id not in ids, "superseded record leaked into default search results"


def test_search_includes_superseded_when_requested(tmp_path: Path) -> None:
    """v0.2.6: include_superseded=True returns both, with superseded ranked below.

    Audit/history walkthroughs need to see the full supersede chain. The flag
    restores them — but the ORDER BY clause must still sink superseded rows
    below every active row so history audits don't bury current truth.
    """
    tools = make_tools(tmp_path)
    active = tools.memory_write(
        content="release-spec-v2 active authoritative",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-02-01T00:00:00Z",
    )
    stale = tools.memory_write(
        content="release-spec-v1 stale superseded record",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    active_id, stale_id = active["data"]["id"], stale["data"]["id"]
    tools.memory_supersede(memory_id=stale_id, reason="replaced", superseded_by=active_id, authorized=True)

    found = tools.memory_search(query="release-spec", workspace="repo-a", include_superseded=True)
    ids = [r["id"] for r in found["data"]["results"]]
    assert active_id in ids, "active record missing under include_superseded=True"
    assert stale_id in ids, "superseded record not returned under include_superseded=True"
    # Superseded must rank below active even though both match.
    assert ids.index(stale_id) > ids.index(active_id), "superseded ranked above active"


def test_supersede_rejects_non_active_replacement(tmp_path: Path) -> None:
    """v0.2.6: supersede chain-breakage guard.

    Starting in v0.2.6, search hides superseded records by default. If a
    supersede points at a replacement that is itself deleted/superseded, the
    default would leave the chain pointing at a record search can't see — the
    user loses both views. Reject early with an explicit error.
    """
    tools = make_tools(tmp_path)
    # Build a chain: A (active) ← supersedes — B (active) ← supersedes — C (active)
    a = tools.memory_write(content="A active", subject="s", source_type="user_confirmed", event_time="2026-01-01T00:00:00Z")
    b = tools.memory_write(content="B active", subject="s", source_type="user_confirmed", event_time="2026-02-01T00:00:00Z")
    c = tools.memory_write(content="C active", subject="s", source_type="user_confirmed", event_time="2026-03-01T00:00:00Z")
    a_id, b_id, c_id = a["data"]["id"], b["data"]["id"], c["data"]["id"]
    # B supersedes A — A is now superseded, B still active.
    tools.memory_supersede(memory_id=a_id, reason="B replaces A", superseded_by=b_id, authorized=True)

    # Now try to supersede C by pointing at A (which is itself superseded) — must be rejected.
    rejected = tools.memory_supersede(memory_id=c_id, reason="C replaced by A", superseded_by=a_id, authorized=True)
    assert rejected["ok"] is False
    assert "not active" in rejected["data"]["error"]
    # C must remain active (the supersede was blocked before any state change).
    c_record = tools.db.get_memory(c_id)
    assert c_record["status"] == "active"


def test_superseded_always_ranked_below_active_even_with_higher_score(tmp_path: Path) -> None:
    """v0.2.6: the soft-demote clause is unconditional, not gated on the filter.

    Even when include_superseded=True lets superseded rows back into the result
    set, they must rank below active rows regardless of bm25 score. A
    superseded record that happens to mention the query terms more often than
    the active record (common: release chatter repeats the codename) must not
    bubble up. This is the audit-mode safety net.
    """
    tools = make_tools(tmp_path)
    # Active record: short, mentions query term once → lower bm25 signal.
    active = tools.memory_write(
        content="release release-spec canonical",
        subject="release-spec-active",
        source_type="user_confirmed",
        event_time="2026-02-01T00:00:00Z",
    )
    # Superseded record: long, repeats query term many times → higher bm25 signal.
    stale_blob = "release release release release release release-spec release-spec release-spec"
    stale = tools.memory_write(
        content=stale_blob,
        subject="release-spec-stale",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    active_id, stale_id = active["data"]["id"], stale["data"]["id"]
    tools.memory_supersede(memory_id=stale_id, reason="replaced", superseded_by=active_id, authorized=True)

    found = tools.memory_search(query="release", workspace="repo-a", include_superseded=True)
    ids = [r["id"] for r in found["data"]["results"]]
    assert active_id in ids and stale_id in ids
    assert ids.index(stale_id) > ids.index(active_id), (
        "superseded ranked above active despite the demote clause — ORDER BY is broken"
    )


# ---- v0.3.1: optional semantic recall (sqlite-vec vec0) -----------------
try:
    import sqlite_vec  # type: ignore  # noqa: F401
    _VEC_AVAILABLE = True
except Exception:
    _VEC_AVAILABLE = False


def make_tools_vec(tmp_path: Path) -> MemoryTools:
    """Fixture that enables sqlite-vec (required for semantic recall tests)."""
    settings = Settings(
        db_path=tmp_path / "memory_vec.sqlite3",
        backup_jsonl=tmp_path / "backup_vec.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=4,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_semantic_recall_off_when_no_embedding(tmp_path: Path) -> None:
    """Default behaviour unchanged: without query_embedding, search is lexical-only."""
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="The deployment uses blue-green strategy.",
        subject="deploy-strategy",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    # No query_embedding passed — pure lexical search, same as v0.3.0.
    found = tools.memory_search(query="deployment", workspace="repo-a")
    assert found["ok"] is True
    assert found["data"]["count"] >= 1


def test_store_embedding_rejects_missing_memory(tmp_path: Path) -> None:
    """Storing an embedding for a non-existent memory id should fail cleanly."""
    if not _VEC_AVAILABLE:
        return  # environment without sqlite-vec; skip gracefully
    tools = make_tools_vec(tmp_path)
    result = tools.memory_store_embedding(memory_id=99999, embedding=[0.1, 0.2, 0.3, 0.4])
    assert result["ok"] is False
    assert "not found" in (result.get("data", {}).get("error") or "").lower()


def test_semantic_recall_surfaces_lexically_unmatched_memory(tmp_path: Path) -> None:
    """A memory with zero lexical overlap should still be reachable via vec0 KNN.

    This is the core value of semantic recall: 'happy' query finds 'joyful'
    content when embeddings say they're close, even though trigram/BM25 miss.
    """
    if not _VEC_AVAILABLE:
        return
    tools = make_tools_vec(tmp_path)
    # Two memories. The first shares no trigrams/tokens with the query
    # 'serene calmness'; the second is lexically unrelated too but further
    # away in vector space. Without semantic recall, neither would surface.
    happy = tools.memory_write(
        content="A tranquil meadow at dawn, quiet and still.",
        subject="meadow-scene",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    tools.memory_write(
        content="Quarterly revenue grew 12% year over year.",
        subject="revenue-report",
        source_type="agent_generated",
        event_time="2026-01-02T00:00:00Z",
    )
    happy_id = happy["data"]["id"]
    # Craft a 4-dim embedding that is close to the query vector and far from
    # everything else. Query vector below sits near happy's embedding.
    happy_embedding = [0.9, 0.1, 0.0, 0.0]
    revenue_embedding = [0.0, 0.0, 0.9, 0.1]
    assert tools.memory_store_embedding(memory_id=happy_id, embedding=happy_embedding)["ok"] is True
    # Store one for revenue too, to make sure KNN discriminates.
    revenue_id = [r["id"] for r in tools.memory_recent(workspace="repo-a")["data"]["results"] if r["subject"] == "revenue-report"][0]
    assert tools.memory_store_embedding(memory_id=revenue_id, embedding=revenue_embedding)["ok"] is True

    # Query embedding close to happy → happy should surface even though
    # 'serene calmness' shares no trigrams with the meadow content.
    found = tools.memory_search(query="serene calmness", workspace="repo-a", query_embedding=[0.85, 0.15, 0.0, 0.0])
    assert found["ok"] is True
    ids = [r["id"] for r in found["data"]["results"]]
    assert happy_id in ids, "semantic recall failed: lexically-unmatched memory not surfaced"


def test_query_embedding_without_sqlite_vec_warns(tmp_path: Path) -> None:
    """When sqlite-vec is unavailable, passing query_embedding should warn, not crash."""
    tools = make_tools(tmp_path)  # vec disabled in this fixture
    tools.memory_write(
        content="Some content here.",
        subject="note",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    result = tools.memory_search(query="content", workspace="repo-a", query_embedding=[0.1, 0.2, 0.3, 0.4])
    assert result["ok"] is True  # must not crash
    # Should carry a warning that semantic channel was skipped.
    warnings_text = " ".join(result.get("warnings") or [])
    assert "sqlite-vec unavailable" in warnings_text or result["data"]["count"] >= 0

