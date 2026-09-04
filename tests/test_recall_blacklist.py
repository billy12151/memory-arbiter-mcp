"""Recall blacklist (v0.15.5): unscoped find excludes blacklisted workspaces.

Design (memory 861):
- default prefill ("mema-twin") applies when the file is absent;
- explicit workspace filter — including a blacklisted one — is always honored
  (twin/compile/governance reach the bucket on purpose);
- filter-driven recall (empty query + tags_filter) and expired-audit paths
  are explicit queries and never consult the blacklist;
- edits are live on the next find (mtime cache), empty file opts back in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.recall_blacklist import (
    DEFAULT_BLACKLIST,
    blacklist_path,
    load_blacklist,
    reset_cache,
)
from memory_arbiter.search import search_memories
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "bl.sqlite3",
        backup_jsonl=tmp_path / "bl.jsonl",
        client="zcode",
        agent_id="agent-a",
        workspace="default",
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write(tools: MemoryTools, content: str, workspace: str, subject: str,
           tags: list[str] | None = None) -> None:
    r = tools.memory_write(
        content=content, workspace=workspace, tags=tags,
        source_type="agent_generated", subject=subject,
    )
    assert r.get("ok"), r


_DEFAULT_BL = frozenset({"mema-twin"})  # what the pipeline layer passes down


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_cache()
    yield
    reset_cache()


# ── loader unit ─────────────────────────────────────────────────────────────

def test_default_when_file_absent(tmp_path):
    p = tmp_path / "recall_blacklist.jsonl"
    names, warnings = load_blacklist(p)
    assert names == frozenset(DEFAULT_BLACKLIST) == frozenset({"mema-twin"})
    assert warnings == []


def test_file_overrides_default(tmp_path):
    p = tmp_path / "recall_blacklist.jsonl"
    p.write_text("# internal buckets\narchive-bucket\n\nnotes-pool\n", encoding="utf-8")
    names, warnings = load_blacklist(p)
    assert names == frozenset({"archive-bucket", "notes-pool"})
    assert warnings == []
    assert "mema-twin" not in names  # file replaces the default entirely


def test_empty_file_opts_back_in(tmp_path):
    p = tmp_path / "recall_blacklist.jsonl"
    p.write_text("# nothing blacklisted\n\n", encoding="utf-8")
    names, _ = load_blacklist(p)
    assert names == frozenset()


def test_invalid_lines_warned_and_skipped(tmp_path):
    p = tmp_path / "recall_blacklist.jsonl"
    p.write_text("good-one\na/b\n" + "x" * 300 + "\n", encoding="utf-8")
    names, warnings = load_blacklist(p)
    assert names == frozenset({"good-one"})
    assert len(warnings) == 2


def test_mtime_cache_and_live_reload(tmp_path):
    p = tmp_path / "recall_blacklist.jsonl"
    p.write_text("alpha\n", encoding="utf-8")
    assert load_blacklist(p)[0] == frozenset({"alpha"})
    assert load_blacklist(p)[0] == frozenset({"alpha"})  # cached path
    p.write_text("alpha\nbeta\n", encoding="utf-8")
    import os
    os.utime(p, ns=(0, 0))  # force a stat change even within fs timestamp granularity
    p.write_text("alpha\nbeta\n", encoding="utf-8")
    assert load_blacklist(p)[0] == frozenset({"alpha", "beta"})


def test_blacklist_path_next_to_db():
    assert blacklist_path("/data/x/memory.sqlite3") == Path("/data/x/recall_blacklist.jsonl")


# ── recall semantics ────────────────────────────────────────────────────────

def test_unscoped_find_excludes_blacklisted(tmp_path):
    tools = make_tools(tmp_path)
    _write(tools, "项目甲的部署流程要点", "proj-a", "部署流程")
    _write(tools, "周报偏好：结论先行", "mema-twin", "周报偏好")
    # no file → default blacklist (mema-twin)
    out = search_memories(tools.db, "部署流程", exclude_workspaces=_DEFAULT_BL)
    hits = [r["workspace"] for r in out.results]
    assert hits and all(w != "mema-twin" for w in hits)

    out2 = search_memories(tools.db, "偏好", exclude_workspaces=_DEFAULT_BL)
    assert all(r["workspace"] != "mema-twin" for r in out2.results)


def test_explicit_blacklisted_workspace_is_honored(tmp_path):
    """The exemption that matters most: twin/compile/governance pass an
    explicit workspace and MUST still reach the bucket."""
    tools = make_tools(tmp_path)
    _write(tools, "周报偏好：结论先行", "mema-twin", "周报偏好")
    # pipeline passes hard_scope together with the resolved ws_canonical
    out = search_memories(tools.db, "偏好", workspace="mema-twin",
                          ws_canonical="mema-twin", hard_scope=True,
                          exclude_workspaces=_DEFAULT_BL)  # scope wins over blacklist
    assert [r["workspace"] for r in out.results] == ["mema-twin"]


def test_empty_query_recent_browse_excludes_blacklisted(tmp_path):
    tools = make_tools(tmp_path)
    _write(tools, "普通记忆一条", "proj-a", "普通")
    _write(tools, "偏好记忆一条", "mema-twin", "偏好")
    out = search_memories(tools.db, "", exclude_workspaces=_DEFAULT_BL)  # recent-browse
    assert all(r["workspace"] != "mema-twin" for r in out.results)
    assert any(r["workspace"] == "proj-a" for r in out.results)


def test_filter_driven_recall_not_filtered(tmp_path):
    """Empty query + tags_filter is an explicit, cursor-paginated query — the
    blacklist must not silently empty it."""
    tools = make_tools(tmp_path)
    _write(tools, "周报偏好：结论先行", "mema-twin", "周报偏好",
           tags=["twin-preference"])
    # filter-driven recall is explicit: even with the blacklist handed in it
    # must not be emptied (the G6 branch never applies it)
    out = search_memories(tools.db, "", tags_filter=["twin-preference"],
                          exclude_workspaces=_DEFAULT_BL)
    assert any(r["workspace"] == "mema-twin" for r in out.results)


def test_canonical_drift_still_excluded(tmp_path):
    """Rows written workspace=mema-twin but canonicalized elsewhere (the
    twin-dryrun era) must be caught by the raw-workspace arm of the filter."""
    tools = make_tools(tmp_path)
    _write(tools, "干跑期偏好", "mema-twin", "干跑偏好")
    conn = tools.db._new_connection()
    conn.execute("UPDATE memories SET workspace_canonical='twin-dryrun' WHERE workspace='mema-twin'")
    conn.commit()
    conn.close()
    out = search_memories(tools.db, "偏好", exclude_workspaces=_DEFAULT_BL)
    assert all(
        (r.get("workspace_canonical") or "") != "mema-twin"
        and r["workspace"] != "mema-twin"
        for r in out.results
    )


def test_file_edit_is_live_on_next_find(tmp_path):
    tools = make_tools(tmp_path)
    _write(tools, "归档桶里的旧方案", "archive-pool", "旧方案")
    _write(tools, "在线项目的方案要点", "proj-live", "在线方案")

    def _ws(query: str) -> list[str]:
        r = tools.memory_search(query=query)
        return [x["workspace"] for x in (r.get("data") or {}).get("results") or []]

    assert "archive-pool" in _ws("方案")  # not blacklisted yet
    bl = blacklist_path(tools.db.settings.db_path)
    bl.write_text("archive-pool\n", encoding="utf-8")
    assert "archive-pool" not in _ws("方案")  # live on the very next find
    assert _ws("方案")  # other workspaces unaffected


def test_pipeline_memory_search_surface(tmp_path):
    """End-to-end via the find tool surface: unscoped excludes, explicit honors."""
    tools = make_tools(tmp_path)
    _write(tools, "周报偏好：结论先行", "mema-twin", "周报偏好")
    _write(tools, "部署流程要点", "proj-a", "部署")
    r1 = tools.memory_search(query="偏好")
    ws1 = [x["workspace"] for x in (r1.get("data") or {}).get("results") or []]
    assert all(w != "mema-twin" for w in ws1)
    r2 = tools.memory_search(query="偏好", workspace="mema-twin")
    ws2 = [x["workspace"] for x in (r2.get("data") or {}).get("results") or []]
    assert ws2 == ["mema-twin"]


# ── round-1 review regressions ─────────────────────────────────────────────

def test_bom_first_entry_not_phantom(tmp_path):
    p = tmp_path / "recall_blacklist.jsonl"
    p.write_bytes(b"\xef\xbb\xbfmema-twin\n")
    names, warnings = load_blacklist(p)
    assert names == frozenset({"mema-twin"})
    assert warnings == []


def test_entry_cap_truncates_with_warning(tmp_path):
    p = tmp_path / "recall_blacklist.jsonl"
    p.write_text("\n".join(f"ws-{i}" for i in range(600)) + "\n", encoding="utf-8")
    names, warnings = load_blacklist(p)
    assert len(names) == 490
    assert any("truncated" in w for w in warnings)
    # exact-cap file parses clean (no spurious truncation warning)
    p2 = tmp_path / "recall_blacklist2.jsonl"
    p2.write_text("\n".join(f"w-{i}" for i in range(490)) + "\n", encoding="utf-8")
    names2, warnings2 = load_blacklist(p2)
    assert len(names2) == 490 and warnings2 == []


def test_linked_open_items_respect_blacklist(tmp_path):
    """Round1#2: blacklisted-bucket todos must not ride the tag-overlap
    attachment back into an unscoped find payload."""
    tools = make_tools(tmp_path)
    _write(tools, "部署流程要点", "proj-a", "部署", tags=["deploy"])
    _write(tools, "偏好待办", "mema-twin", "偏好待办",
           tags=["deploy", "todo"])
    r = tools.memory_search(query="部署")
    results = (r.get("data") or {}).get("results") or []
    linked = (r.get("data") or {}).get("linked_open_items") or []
    assert results  # proj-a still found
    assert all(
        item.get("subject") != "偏好待办" for item in linked
    ), linked


def test_vec_evidence_channel_excludes_blacklisted(tmp_path):
    """Round1#3: the evidence-KNN channel (where the historical over-bind
    bug lived) must exclude blacklisted workspaces when sqlite-vec is live."""
    pytest.importorskip("sqlite_vec")
    from memory_arbiter.search import _wide_recall
    settings = Settings(
        db_path=tmp_path / "vec.sqlite3", backup_jsonl=tmp_path / "vec.jsonl",
        client="zcode", agent_id="a", workspace="default",
        embedding_model_path=tmp_path / "fake.gguf",
    )
    (tmp_path / "fake.gguf").write_bytes(b"fake")
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    assert tools.db.ensure_vec_tables(2) == []
    # two memories with distinct 2-dim embeddings in different workspaces
    import sqlite3
    for ws, ev in (("proj-a", (0.9, 0.1)), ("mema-twin", (0.1, 0.9))):
        rr = tools.memory_write(content=f"{ws} 的语义内容", workspace=ws,
                                source_type="agent_generated", subject=f"{ws} 语义")
        assert rr.get("ok"), rr
        mid = (rr.get("data") or {}).get("id")
        conn = tools.db._new_connection()
        import hashlib
        conn.execute(
            "INSERT INTO memory_evidence(memory_id, memory_version, content_hash,"
            " unit_index, kind, text, start_offset, end_offset, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (mid, 1, hashlib.sha1(f"{ws}-ev".encode()).hexdigest(),
             0, "body", f"{ws} evidence text", 0, 10, "2026-09-04T00:00:00Z"),
        )
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO memory_evidence_vec(id, parent_status, embedding) VALUES(?,?,?)",
            (eid, "active", __import__("json").dumps(list(ev))),
        )
        conn.commit()
        conn.close()
    pool = _wide_recall(
        tools.db, "语义", None, None, "m.status = 'active'", "status = 'active'",
        query_embedding=[0.9, 0.1],
        exclude_workspaces=frozenset({"mema-twin"}),
    )
    assert pool, "vec channel should still recall proj-a"
    assert all(r["workspace"] != "mema-twin" for r in pool), [
        (r["workspace"], r.get("_vec_candidate")) for r in pool
    ]
    pool2 = _wide_recall(
        tools.db, "语义", None, None, "m.status = 'active'", "status = 'active'",
        query_embedding=[0.1, 0.9],
        exclude_workspaces=frozenset({"mema-twin"}),
    )
    # twin bucket is blacklisted even when it owns the globally nearest vector
    assert all(r["workspace"] != "mema-twin" for r in pool2)


def test_settings_homed_caller_keeps_home_bucket(tmp_path):
    """对抗#1：settings.workspace 落在黑名单桶的调用者，家桶不被排除。"""
    settings = Settings(
        db_path=tmp_path / "home.sqlite3", backup_jsonl=tmp_path / "home.jsonl",
        client="zcode", agent_id="twin-host", workspace="mema-twin",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    _write(tools, "周报偏好：结论先行", "mema-twin", "周报偏好")
    r = tools.memory_search(query="偏好")  # no explicit workspace param
    ws = [x["workspace"] for x in (r.get("data") or {}).get("results") or []]
    assert ws == ["mema-twin"]  # home bucket reachable despite blacklist


def test_g6_linked_attachments_follow_exemption(tmp_path):
    """对抗#3：G6（空 query+filters）是显式查询——linked 附件同结果一样不滤。"""
    tools = make_tools(tmp_path)
    # result row (proj-a, matched by the filter) + a twin-bucket todo that is
    # NOT itself in the results (linked only attaches non-result todos)
    _write(tools, "部署流程要点", "proj-a", "部署",
           tags=["deploy", "twin-preference"])
    _write(tools, "偏好待办", "mema-twin", "偏好待办",
           tags=["deploy", "todo"])
    r = tools.memory_search(query="", tags_filter=["twin-preference"])
    results = (r.get("data") or {}).get("results") or []
    assert any(x["workspace"] == "proj-a" for x in results)
    linked = (r.get("data") or {}).get("linked_open_items") or []
    assert any(item.get("subject") == "偏好待办" for item in linked), linked
