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


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
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



def test_edit_full_replacement_stores_history_and_updates_fts(tmp_path: Path) -> None:
    """Full content replace: version bumps, old content archived, FTS re-synced."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="version one content here",
        subject="s1",
        tags=["t"],
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    result = tools.memory_edit(memory_id=memory_id, new_content="version two content here", reason="refresh")
    assert result["ok"] is True
    assert result["data"]["edited"] is True
    assert result["data"]["new_version"] == 2
    assert result["data"]["history_id"] is not None

    updated = tools.db.get_memory(memory_id)
    assert updated["content"] == "version two content here"
    assert updated["version"] == 2

    # History archived the old snapshot at the old version
    history = tools.memory_history(memory_id=memory_id)["data"]["history"]
    assert len(history) == 1
    assert history[0]["content_snapshot"] == "version one content here"
    assert history[0]["version"] == 1
    assert history[0]["reason"] == "refresh"

    # FTS must reflect the new content, not the old. Query the FTS index
    # directly — memory_search falls back to "recent" when a token matches
    # nothing, which would mask a stale-FTS bug. Use a token unique to the old
    # body (kangaroo) vs unique to the new body (platypus).
    written2 = tools.memory_write(
        content="alpha kangaroo draft",
        subject="s2",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    mid2 = written2["data"]["id"]
    tools.memory_edit(memory_id=mid2, new_content="alpha platypus final", reason="swap")

    with tools.db.connection() as fts_conn:
        fts_new = [r["rowid"] for r in fts_conn.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'platypus'"
        ).fetchall()]
        fts_old = [r["rowid"] for r in fts_conn.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'kangaroo'"
        ).fetchall()]
    assert mid2 in fts_new, f"new token not in FTS: {fts_new}"
    assert mid2 not in fts_old, f"old token still in FTS (stale index): {fts_old}"
    # And the live memories row carries the new text (sanity)
    assert tools.db.get_memory(mid2)["content"] == "alpha platypus final"


def test_edit_partial_replacement(tmp_path: Path) -> None:
    """old_text+new_text does an exact substring substitution, not full replace."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="the api returns status 200 on success",
        subject="s",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    result = tools.memory_edit(memory_id=memory_id, old_text="200", new_text="404", reason="typo")
    assert result["ok"] is True
    assert result["data"]["record"]["content"] == "the api returns status 404 on success"
    assert result["data"]["new_version"] == 2

    # old_text not present -> explicit error, no mutation
    bad = tools.memory_edit(memory_id=memory_id, old_text="nonexistent", new_text="x")
    assert bad["ok"] is False
    assert "old_text not found" in bad["data"]["error"]
    # version unchanged after the failed edit
    assert tools.db.get_memory(memory_id)["version"] == 2

def test_edit_partial_replacement_stale_version_does_not_overwrite(tmp_path: Path) -> None:
    """Partial edit validates intent against the current row inside the txn."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="alpha beta gamma",
        subject="s",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    first = tools.memory_edit(memory_id=memory_id, old_text="beta", new_text="delta")
    assert first["ok"] is True

    stale = tools.memory_edit(
        memory_id=memory_id,
        old_text="beta",
        new_text="epsilon",
        expected_version=1,
    )

    assert stale["ok"] is False
    assert "stale_edit" in stale["data"]["error"] or "old_text not found" in stale["data"]["error"]
    current = tools.db.get_memory(memory_id)
    assert current["content"] == "alpha delta gamma"
    assert current["version"] == 2


def test_edit_full_path_rechecks_protection_without_prefetch(monkeypatch, tmp_path: Path) -> None:
    """Full edit must not depend on a transaction-external memory snapshot."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="normal fact",
        subject="s",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    assert tools.db.update_memory(memory_id, {"protection_level": "locked"}) is True

    def forbidden_prefetch(_memory_id):
        raise AssertionError("content edit path must not prefetch get_memory outside edit transaction")

    monkeypatch.setattr(tools.db, "get_memory", forbidden_prefetch)
    rejected = tools.memory_edit(memory_id=memory_id, new_content="tampered", authorized=False)

    assert rejected["ok"] is False
    assert "authorized" in rejected["data"]["error"]


    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="confirmed fact",
        subject="s",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    rejected = tools.memory_edit(memory_id=memory_id, new_content="tampered", reason="try", authorized=False)
    assert rejected["ok"] is False
    assert rejected["data"]["edited"] is False
    assert "authorized" in rejected["data"]["error"]
    # content untouched
    assert tools.db.get_memory(memory_id)["content"] == "confirmed fact"

    allowed = tools.memory_edit(memory_id=memory_id, new_content="corrected fact", reason="auth", authorized=True)
    assert allowed["ok"] is True
    assert allowed["data"]["edited"] is True
    assert tools.db.get_memory(memory_id)["content"] == "corrected fact"


def test_edit_normal_memory_no_auth_needed(tmp_path: Path) -> None:
    """agent_generated/normal records can be edited without authorized flag."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="draft note",
        subject="s",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    assert tools.db.get_memory(memory_id)["protection_level"] == "normal"

    result = tools.memory_edit(memory_id=memory_id, new_content="revised note", reason="cleanup")
    assert result["ok"] is True
    assert result["data"]["edited"] is True


def test_edit_rejects_empty_new_content(tmp_path: Path) -> None:
    """An empty new_content must be rejected rather than silently wiping the memory."""
    tools = make_tools(tmp_path)
    memory_id = tools.memory_write(content="keep this", subject="s", source_type="agent_generated")["data"]["id"]

    rejected = tools.memory_edit(memory_id=memory_id, new_content="   ")
    assert rejected["ok"] is False
    assert rejected["data"]["edited"] is False
    # Content is untouched.
    assert tools.db.get_memory(memory_id)["content"] == "keep this"


def test_edit_rejects_superseded(tmp_path: Path) -> None:
    """Editing a superseded record is refused (idempotency / terminal-state gate)."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="doomed",
        subject="s",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    tools.memory_supersede(memory_id=memory_id, reason="retired", authorized=True)

    rejected = tools.memory_edit(memory_id=memory_id, new_content="revived", reason="try", authorized=True)
    assert rejected["ok"] is False
    assert "already" in rejected["data"]["error"]
    assert rejected["data"]["edited"] is False


def test_history_returns_version_chain(tmp_path: Path) -> None:
    """Two edits produce two history rows; history is newest-version-first."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="rev one",
        subject="s",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    tools.memory_edit(memory_id=memory_id, new_content="rev two", reason="second")
    tools.memory_edit(memory_id=memory_id, new_content="rev three", reason="third")

    result = tools.memory_history(memory_id=memory_id)
    assert result["ok"] is True
    assert result["data"]["current_version"] == 3
    assert result["data"]["count"] == 2
    versions = [h["version"] for h in result["data"]["history"]]
    assert versions == [2, 1]  # newest version snapshot first
    assert result["data"]["history"][0]["content_snapshot"] == "rev two"


def test_memory_edit_content_plus_add_remove_tags(tmp_path: Path) -> None:
    """Content edit path honors add_tags/remove_tags (was silently ignored).

    add/remove overlay on current tags, order-preserving + deduped (mirrors
    the tags-only path). new_tags acts as the base when both are passed.
    """
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="original body", subject="s", tags=["a", "b"],
        source_type="agent_generated", event_time="2026-01-01T00:00:00Z",
    )
    mid = written["data"]["id"]

    # add + remove against current tags ["a","b"] -> ["b","c"]
    edited = tools.memory_edit(
        memory_id=mid, new_content="new body", add_tags=["c"], remove_tags=["a"], reason="retag",
    )
    assert edited["ok"] is True
    updated = tools.memory_get(memory_id=mid)["data"]["memory"]
    assert updated["content"] == "new body"
    assert updated["tags"] == ["b", "c"]  # removed "a", appended "c", order preserved
    assert updated["version"] == 2
    assert tools.memory_history(memory_id=mid)["data"]["count"] == 1  # content edit wrote history

    # new_tags as base + add overlay -> ["x","y"] (new_tags wins over current tags)
    written2 = tools.memory_write(
        content="second", subject="s2", tags=["a", "b"],
        source_type="agent_generated", event_time="2026-01-01T00:00:00Z",
    )
    mid2 = written2["data"]["id"]
    edited2 = tools.memory_edit(
        memory_id=mid2, new_content="second v2", new_tags=["x"], add_tags=["y"], reason="base",
    )
    assert edited2["ok"] is True
    assert tools.memory_get(memory_id=mid2)["data"]["memory"]["tags"] == ["x", "y"]


def test_cleanup_history_full_requires_authorization(tmp_path: Path) -> None:
    """Full history cleanup needs authorized=True; per-memory cleanup does not.
    And under no arguments must the memories table lose zero rows."""
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="a one", subject="a", source_type="agent_generated", event_time="2026-01-01T00:00:00Z")
    b = tools.memory_write(content="b one", subject="b", source_type="agent_generated", event_time="2026-01-01T00:00:00Z")
    a_id, b_id = a["data"]["id"], b["data"]["id"]
    tools.memory_edit(memory_id=a_id, new_content="a two", reason="e")
    tools.memory_edit(memory_id=b_id, new_content="b two", reason="e")
    # 2 history rows now

    # Full cleanup without auth -> rejected, nothing removed
    rejected = tools.memory_cleanup_history()
    assert rejected["ok"] is False
    assert rejected["data"]["cleaned"] == 0
    assert "authorized" in rejected["data"]["error"]
    assert tools.memory_history(memory_id=a_id)["data"]["count"] == 1  # history intact

    # Per-memory cleanup needs no auth
    single = tools.memory_cleanup_history(memory_id=a_id)
    assert single["ok"] is True
    assert single["data"]["cleaned"] == 1
    assert tools.memory_history(memory_id=a_id)["data"]["count"] == 0
    assert tools.memory_history(memory_id=b_id)["data"]["count"] == 1  # b untouched

    negative_age = tools.memory_cleanup_history(older_than_days=-1)
    assert negative_age["ok"] is False
    assert negative_age["data"]["cleaned"] == 0
    assert tools.memory_history(memory_id=b_id)["data"]["count"] == 1

    # Full cleanup WITH auth clears the rest
    full = tools.memory_cleanup_history(authorized=True)
    assert full["ok"] is True
    assert full["data"]["cleaned"] == 1  # b's remaining row

    # SAFETY RED LINE: memories table must be fully intact despite "full cleanup"
    assert tools.db.get_memory(a_id)["content"] == "a two"
    assert tools.db.get_memory(b_id)["content"] == "b two"



def test_insert_memory_rejects_empty_subject(tmp_path: Path) -> None:
    # B1: subject is now required at the DB layer, mirroring the content check.
    tools = make_tools(tmp_path)
    db = tools.db
    from memory_arbiter.models import MemoryRecord

    record = MemoryRecord(
        content="content without subject",
        agent_id="agent-a",
        workspace="repo-a",
        subject=None,
    )
    with pytest.raises(ValueError, match="subject is required"):
        db.insert_memory(record, "repo-a")

    # 空串 / 纯空白同样拒绝
    for empty in ("", "   "):
        record_empty = MemoryRecord(
            content="c", agent_id="agent-a", workspace="repo-a", subject=empty
        )
        with pytest.raises(ValueError, match="subject is required"):
            db.insert_memory(record_empty, "repo-a")

    # 正常带 subject 仍可写入
    ok = tools.memory_write(content="has subject", subject="s1", source_type="agent_generated")
    assert ok["ok"] is True


def test_memory_write_rejects_empty_subject_contract(tmp_path: Path) -> None:
    """T2: memory_write (the MCP agent-facing path) must return ok=False with a
    clear error when subject is missing or empty — BEFORE any workspace
    canonical side effect. This is the contract agents rely on."""
    tools = make_tools(tmp_path)

    # missing subject entirely
    res = tools.memory_write(content="no subject", source_type="agent_generated")
    assert res["ok"] is False
    assert res["data"]["written"] is False
    assert "subject is required" in res["data"]["error"]

    # empty string
    res2 = tools.memory_write(content="empty subject", subject="", source_type="agent_generated")
    assert res2["ok"] is False
    assert "subject is required" in res2["data"]["error"]

    # whitespace-only
    res3 = tools.memory_write(content="ws subject", subject="   ", source_type="agent_generated")
    assert res3["ok"] is False
    assert "subject is required" in res3["data"]["error"]

    # strict isolation + missing workspace + missing subject: subject error
    # must surface FIRST (before workspace error), so the workspace canonical
    # registration side effect never runs for a rejected write.
    strict_tools = make_tools(tmp_path)
    strict_tools.settings.isolation = "strict"
    res4 = strict_tools.memory_write(content="x", workspace="")  # no subject, no ws
    assert res4["ok"] is False
    # subject check precedes workspace check in memory_write
    assert "subject is required" in res4["data"]["error"]


def test_memory_edit_rejects_empty_new_subject(tmp_path: Path) -> None:
    # B2: passing new_subject="" (empty string) used to wipe subject via
    # edit_memory's `new_subject if new_subject is not None else old_subject`
    # branch. It must now be refused at the service layer (pass None to keep).
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="original content", subject="keep-me", source_type="agent_generated"
    )
    mid = written["data"]["id"]

    res = tools.memory_edit(memory_id=mid, new_content="original content", new_subject="")
    assert res["ok"] is False
    assert "new_subject is empty" in res["data"]["error"]
    # subject 未被清空
    got = tools.memory_get(memory_id=mid)["data"]["memory"]
    assert got["subject"] == "keep-me"

    # 纯空白同样拒绝
    res2 = tools.memory_edit(memory_id=mid, new_content="original content", new_subject="   ")
    assert res2["ok"] is False
    assert "new_subject is empty" in res2["data"]["error"]

    # None 保持原 subject（合法）
    res3 = tools.memory_edit(memory_id=mid, new_content="original content", new_subject=None, reason="noop edit")
    assert res3["ok"] is True
    got3 = tools.memory_get(memory_id=mid)["data"]["memory"]
    assert got3["subject"] == "keep-me"


def test_tags_only_edit_unaffected_by_subject_rule(tmp_path: Path) -> None:

    # B 回归: tags-only 路径走 update_tags_low_side_effect，完全不碰 subject，
    # 因此不受 subject 必填/空值拦截影响。用一条 subject 为空的历史记录验证
    # （历史空 subject 记录必须仍能 tags-only 编辑，否则回填前就无法打标）。
    tools = make_tools(tmp_path)
    db = tools.db
    from memory_arbiter.models import MemoryRecord

    # 直接写一条 subject 为空的记录（绕过 memory_write，模拟历史数据）
    record = MemoryRecord(
        content="legacy no-subject memory",
        agent_id="agent-a",
        workspace="repo-a",
        subject=None,
    )
    # 直接 INSERT 绕过新的 subject 校验（构造历史空 subject 数据）
    with db.connection() as conn:
        cur = conn.execute(
            "INSERT INTO memories (content, agent_id, workspace, tags, source_type, "
            "event_time, ingest_time, confidence, protection_level, status, subject, "
            "metadata, version, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record.content, record.agent_id, "repo-a", "[]", "agent_generated",
             "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", 0.5, "normal",
             "active", None, "{}", 1, "2026-01-01T00:00:00Z"),
        )
        mid = int(cur.lastrowid)
        if db.state.fts5_available:
            conn.execute(
                "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                (mid, record.content, "", ""),
            )
        conn.commit()

    # tags-only 编辑必须成功（subject 仍为空也不拦）
    res = tools.memory_edit(memory_id=mid, tags_only=True, add_tags=["backfill"], authorized=False)
    assert res["ok"] is True
    assert res["data"]["tags_only"] is True
    got = tools.memory_get(memory_id=mid)["data"]["memory"]
    assert "backfill" in got["tags"]
    # subject 未被改动（仍为空，待回填脚本处理）
    assert got["subject"] in (None, "")
