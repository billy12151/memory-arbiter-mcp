"""v0.8.0 protocol tests (T0): lock the target protocol before implementing.

These tests pin the v0.8 contract described in the final design doc
(memory-arbiter-v0.8.0-dev-design-20260723-revised.md). Per T0:

  * Registry / status-enum / content_scope / get-parameter-matrix assertions
    lock the *target* shape and FAIL until the corresponding task lands.
    They are wrapped in ``pytest.mark.xfail(reason=...)`` so the 246-test
    baseline stays green; each xfail is removed when its task is implemented.
  * "Core does NOT do X" negative assertions (no LLM call, no pending/
    fallback_active, no mechanical fallback) describe behaviour that must
    ALREADY hold or must never be introduced — these run directly.

As tasks T1–T6 land, flip each xfail to a hard assert (remove the decorator).
The final gate (§16) requires zero xfails remaining.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.tools import MemoryTools


# ------------------------------------------------------------------
#  Shared harness (mirrors test_memory_arbiter.py conventions)
# ------------------------------------------------------------------

class _MockManagedEmbedder:
    def __init__(self, encode_fn):
        self._encode = encode_fn
        self.embedding_space_id = "mock_space_id"
        self.last_encode_error = None

    def embed_text(self, prefix="", body="", max_body_chars=None):
        sep = "\n" if prefix and body else ""
        text = (prefix + sep + body).strip()
        try:
            emb = self._encode(text)
        except Exception as exc:
            self.last_encode_error = str(exc)
            return EmbedResult(embedding=[], truncated=True, original_tokens=0, used_tokens=0)
        return EmbedResult(embedding=emb, truncated=False, original_tokens=0, used_tokens=0)


def _keyword_embedding(text: str) -> list[float]:
    table = {
        "alpha": (1.0, 0.0),
        "beta": (0.0, 1.0),
        "gamma": (-1.0, 0.0),
        "delta": (0.0, -1.0),
    }
    low = text.lower()
    for key, (a, b) in table.items():
        if key in low:
            return [a, b]
    return [-0.7071, -0.7071]


def _keyword_embedder(space_id: str = "mock_space_id"):
    return _MockManagedEmbedder(lambda text: _keyword_embedding(text))


def make_vec_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    settings = Settings(
        db_path=tmp_path / "v08-vec.sqlite3",
        backup_jsonl=tmp_path / "v08-backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=2,
        split_threshold=1,   # so test content always exceeds threshold
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def make_plain_tools(tmp_path: Path) -> MemoryTools:
    """vec OFF tools — used to assert split capability is unavailable."""
    settings = Settings(
        db_path=tmp_path / "v08-plain.sqlite3",
        backup_jsonl=tmp_path / "v08-plain-backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=False,
        split_threshold=1,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _set_vec_ready(tools: MemoryTools, space_id: str = "mock_space_id") -> None:
    with tools.db.write_transaction() as conn:
        MemoryDB._set_meta(conn, "state", "ready")
        MemoryDB._set_meta(conn, "active_space_id", space_id)


def _content_with_two_headings() -> str:
    """Markdown with 2 fenced-code-safe headings → should be rules-publishable."""
    return (
        "# 第一章\n" + ("alpha 内容 " * 40) + "\n\n"
        "# 第二章\n" + ("beta 内容 " * 40)
    )


def _content_with_fenced_fake_heading() -> str:
    """A ``` block contains a line that looks like a heading — must be ignored."""
    return (
        "# 真标题\n" + ("正文 " * 40) + "\n\n"
        "```\n## 代码块里的假标题\n```\n\n"
        + ("更多正文 " * 40)
    )


# ==================================================================
#  §2.1 / §6.5 — Registry: which tools exist
# ==================================================================

@pytest.mark.xfail(reason="T6: get_sections/memory_split_status wrappers not removed yet")
def test_registry_keeps_memory_split_removes_status_and_get_sections() -> None:
    """§2.1(1): registry keeps memory_split; drops memory_split_status + get_sections."""
    from memory_arbiter.server import build_server
    app = build_server()
    names = _registered_tool_names(app)
    assert "memory_split" in names, "memory_split must be retained (Agent continuation/repair)"
    assert "memory_write" in names
    assert "memory_search" in names
    assert "memory_get" in names
    assert "memory_split_status" not in names, "memory_split_status must be removed (merged into get/doctor)"
    assert "get_sections" not in names, "get_sections must be removed (merged into search/get)"


def _registered_tool_names(app) -> set[str]:
    """Best-effort extraction of registered tool names from a FastMCP app."""
    # FastMCP stores tools on a ToolManager; exact attribute varies by version.
    tm = getattr(app, "_tool_manager", None) or getattr(app, "tool_manager", None)
    if tm is not None:
        tools_map = getattr(tm, "_tools", None) or getattr(tm, "tools", None)
        if isinstance(tools_map, dict):
            return set(tools_map.keys())
        if isinstance(tools_map, (list, tuple)):
            return {getattr(t, "name", None) or t.get("name") for t in tools_map}
    # Fallback: FastMCP registers functions decorated with @app.tool(); the
    # decorated wrappers live on the app object. This is brittle but only used
    # if the above internal paths change.
    return {n for n in dir(app) if not n.startswith("_")}


# ==================================================================
#  §5.2 — State enum: only NULL / active / failed are written by new flow
# ==================================================================

def test_new_flow_never_writes_pending(tmp_path: Path) -> None:
    """§5.2 / reverse test: writing a long unstructured doc must NOT set pending."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    # No headings → not rules-publishable → split_request expected, status NULL.
    long_plain = "无标题的纯文本内容 " * 400
    r = tools.memory_write(content=long_plain, subject="plain")
    assert r["ok"] is True
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] in (None, "failed"), (
        f"new flow must not write pending; got {mem['split_status']!r}"
    )
    assert mem["split_status"] != "pending"


def test_new_flow_never_writes_fallback_active(tmp_path: Path) -> None:
    """§5.2: no mechanical fallback → fallback_active must never appear."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    long_plain = "无标题纯文本 " * 400
    r = tools.memory_write(content=long_plain, subject="plain")
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] != "fallback_active"


@pytest.mark.xfail(reason="T5: status split block still uses split_enabled, not split_capability")
def test_no_pending_timeout_or_fallback_semantics_in_status_dict(tmp_path: Path) -> None:
    """§6.5: memory_status split info must not surface pending/fallback_active.

    Checks the split-related surface specifically (not a whole-blob scan, which
    would false-positive on unrelated substrings like "backup"). After T5 lands
    this becomes a direct assertion on split_capability.
    """
    tools = make_plain_tools(tmp_path)
    status = tools.memory_status()["data"]
    # The only split-status-shaped values that may appear must come from the
    # allowed enum. pending / fallback_active must never be written or reported.
    sp = status.get("split_capability")
    if sp is None:
        pytest.xfail("split_capability not yet present (T5)")
    assert "pending" not in repr(sp).lower()
    assert "fallback" not in repr(sp).lower()


# ==================================================================
#  §6.1 — memory_write split return object shape
# ==================================================================

@pytest.mark.xfail(reason="T4: memory_write rules auto-split not implemented yet")
def test_write_rules_auto_split_for_two_headings(tmp_path: Path) -> None:
    """§2.2(2): 2 safe headings + vec ready → rules publish, split.mode=rules applied."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = _content_with_two_headings()
    r = tools.memory_write(content=content, subject="doc")
    assert r["ok"] is True
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] == "active"
    sp = r["data"]["split"]
    assert sp["required"] is True
    assert sp["applied"] is True
    assert sp["mode"] == "rules"
    assert sp["status"] == "active"
    # sections actually published
    assert len(tools.db.get_sections_by_memory(mem["id"])) == 2


@pytest.mark.xfail(reason="T4: split_request not returned yet")
def test_write_returns_split_request_for_unstructured_long_doc(tmp_path: Path) -> None:
    """§2.2(3): no safe heading plan → full split_request, content already stored, status NULL."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    long_plain = "无标题纯文本内容 " * 400
    r = tools.memory_write(content=long_plain, subject="plain")
    assert r["ok"] is True
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] is None   # NOT failed, NOT pending
    sp = r["data"]["split"]
    assert sp["required"] is True
    assert sp["applied"] is False
    assert sp["mode"] == "agent_semantic"
    assert sp["status"] is None
    assert sp["action_required"] == "memory_split"
    sr = r["data"]["split_request"]
    assert sr["content"] == long_plain            # full content, no truncation
    assert sr["content_hash"] == hashlib.sha256(long_plain.encode("utf-8")).hexdigest()
    assert sr["memory_version"] == mem["version"]
    assert sr["split_revision"] == mem["split_revision"]
    assert "split_schema" in sr


@pytest.mark.xfail(reason="T4: fenced-code-aware parser not wired into write")
def test_write_rules_split_ignores_fenced_code_headings(tmp_path: Path) -> None:
    """§7.1: a heading-looking line inside ``` must not become a section boundary.

    Requires that the rules plan actually produced sections (2, from the two
    real headings), so an empty section list does NOT silently pass.
    """
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = _content_with_fenced_fake_heading()
    r = tools.memory_write(content=content, subject="doc")
    assert r["ok"] is True
    sections = tools.db.get_sections_by_memory(r["data"]["id"])
    assert len(sections) == 2, f"rules split must yield 2 real headings, got {len(sections)}"
    titles = [s["title"] for s in sections]
    assert "代码块里的假标题" not in titles


@pytest.mark.xfail(reason="T4: single heading must yield split_request, not failed")
def test_write_single_heading_yields_split_request(tmp_path: Path) -> None:
    """§7.3: only one heading → split_request, not failed."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "# 唯一标题\n" + ("正文内容 " * 400)
    r = tools.memory_write(content=content, subject="doc")
    assert r["ok"] is True
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] is None
    assert r["data"]["split"]["action_required"] == "memory_split"


def test_write_short_content_no_split_required(tmp_path: Path) -> None:
    """§6.1: below threshold → split.required=false, status NULL."""
    settings = Settings(
        db_path=tmp_path / "short.sqlite3",
        backup_jsonl=tmp_path / "short.jsonl",
        enable_sqlite_vec=True, vec_dim=2,
        split_threshold=4000,  # default
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    r = tools.memory_write(content="短内容", subject="s")
    assert r["ok"] is True
    # split block absent or required=false — either is acceptable pre-T4.
    sp = r["data"].get("split", {"required": False})
    assert sp.get("required") is False


def test_write_vec_not_ready_does_not_enter_split(tmp_path: Path) -> None:
    """§2.2(9): vec not ready → no split attempt, no split failure recorded."""
    tools = make_plain_tools(tmp_path)   # vec OFF
    long = "长内容 " * 1000
    r = tools.memory_write(content=long, subject="doc")
    assert r["ok"] is True
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] is None
    # no sections created
    assert tools.db.get_sections_by_memory(mem["id"]) == []
    # status should report capability unavailable
    status = tools.memory_status()["data"]
    assert "split_capability" in status or status.get("split_enabled") is False


# ==================================================================
#  §6.2 — memory_split as Agent continuation entry
# ==================================================================

@pytest.mark.xfail(reason="T4: prepare no longer sets requires_user_confirmation")
def test_split_prepare_does_not_require_user_confirmation(tmp_path: Path) -> None:
    """§6.2: prepare returns full snapshot + schema, no user-confirmation gate."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "无标题纯文本 " * 400
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    prep = tools.memory_split(memory_id=mid)
    assert prep["ok"] is True
    assert prep["data"].get("requires_user_confirmation") is not True
    assert prep["data"]["content"] == content
    for k in ("content_hash", "memory_version", "split_status", "split_revision"):
        assert k in prep["data"]


def test_split_publish_agent_metadata_succeeds(tmp_path: Path) -> None:
    """§6.2 publish: Agent-supplied metadata publishes via the retained tool."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    r = tools.memory_split(
        memory_id=mid,
        split_decision="split",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": "alpha"},
            {"title": "beta", "anchor_text": "beta", "occurrence_index": 0},
        ],
    )
    assert r["ok"] is True
    assert r["data"]["split_active"] is True


def test_split_publish_rejects_oversized_section(tmp_path: Path) -> None:
    """§6.2: a section slice exceeding max_section_chars → section_too_large."""
    tools = make_vec_tools(tmp_path)
    tools.settings.max_section_chars = 10
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 200) + "\n" + "beta " + ("y" * 200)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    r = tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[{"title": "alpha"}, {"title": "beta", "anchor_text": "beta", "occurrence_index": 0}],
    )
    assert r["ok"] is False
    assert "section_too_large" in str(r["data"].get("error", ""))


def test_split_failure_does_not_lose_original(tmp_path: Path) -> None:
    """§2.2(6): real publish failure → content still readable, status failed."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    # anchor not in content → offset computation fails
    r = tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[{"title": "a"}, {"title": "b", "anchor_text": "NONEXISTENT", "occurrence_index": 0}],
    )
    assert r["ok"] is False
    # original content intact
    assert tools.db.get_memory(mid)["content"] == content


def test_rebuild_failure_preserves_old_active(tmp_path: Path) -> None:
    """§5.3: rebuild failure must keep the old active sections intact."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    # first publish
    tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[{"title": "alpha"}, {"title": "beta", "anchor_text": "beta", "occurrence_index": 0}],
    )
    active_mem = tools.db.get_memory(mid)
    assert active_mem["split_status"] == "active"
    old_sections = tools.db.get_sections_by_memory(mid)
    # rebuild with a bad anchor → must fail but keep old active
    r = tools.memory_split(
        memory_id=mid, split_decision="rebuild",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=active_mem["version"],
        decision_split_status="active",
        decision_split_revision=active_mem["split_revision"],
        sections=[{"title": "a"}, {"title": "b", "anchor_text": "NONEXISTENT", "occurrence_index": 0}],
    )
    assert r["ok"] is False
    after = tools.db.get_memory(mid)
    assert after["split_status"] == "active"   # preserved
    assert tools.db.get_sections_by_memory(mid) == old_sections


# ==================================================================
#  §6.3 — memory_search content_scope
# ==================================================================

def test_search_partial_returns_full_section_content(tmp_path: Path) -> None:
    """§6.3: partial hit → matched_sections carry full section content."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[{"title": "alpha"}, {"title": "beta", "anchor_text": "beta", "occurrence_index": 0}],
    )
    r = tools.memory_search(query="beta", query_embedding=_keyword_embedding("beta"))
    hit = next(x for x in r["data"]["results"] if x["id"] == mid)
    assert hit["content_scope"] == "matched_sections"
    for ms in hit["matched_sections"]:
        assert ms.get("content")   # full section body present


def test_search_zero_match_returns_full_memory(tmp_path: Path) -> None:
    """§6.3: zero section match → full memory, content_scope=full_memory."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[{"title": "alpha"}, {"title": "beta", "anchor_text": "beta", "occurrence_index": 0}],
    )
    r = tools.memory_search(query="zzz", query_embedding=_keyword_embedding("zzz"))
    hit = next(x for x in r["data"]["results"] if x["id"] == mid)
    assert hit.get("content_scope") == "full_memory"
    assert hit.get("content") == content          # full text, not preview
    assert "content_truncated" not in hit or hit.get("content_truncated") is False
    assert "content_omitted" not in hit           # removed in v0.8


def test_search_matched_sections_no_embedding_diagnostics(tmp_path: Path) -> None:
    """§6.3: ordinary matched_sections omit embedding_truncated/original/used_tokens."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[{"title": "alpha"}, {"title": "beta", "anchor_text": "beta", "occurrence_index": 0}],
    )
    r = tools.memory_search(query="beta", query_embedding=_keyword_embedding("beta"))
    hit = next(x for x in r["data"]["results"] if x["id"] == mid)
    for ms in hit.get("matched_sections", []):
        assert "embedding_truncated" not in ms
        assert "embedding_original_tokens" not in ms
        assert "embedding_used_tokens" not in ms


# ==================================================================
#  §6.4 — memory_get parameter matrix
# ==================================================================

def test_get_sections_none_catalog_all_and_section_ids(tmp_path: Path) -> None:
    """§6.4: none/catalog/all + section_ids matrix."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[{"title": "alpha"}, {"title": "beta", "anchor_text": "beta", "occurrence_index": 0}],
    )
    secs = tools.db.get_sections_by_memory(mid)

    none_r = tools.memory_get(memory_id=mid, sections="none")["data"]
    assert "content" in none_r["memory"]
    assert none_r.get("section_catalog") is None
    assert none_r.get("sections") is None

    cat_r = tools.memory_get(memory_id=mid, sections="catalog")["data"]
    assert cat_r.get("section_catalog") is not None
    assert len(cat_r["section_catalog"]) == 2

    all_r = tools.memory_get(memory_id=mid, sections="all")["data"]
    assert len(all_r["sections"]) == 2
    assert all_r["sections"][0]["content"]   # body present

    # section_ids: pick the second; missing id routed to missing_section_ids
    by_id = tools.memory_get(memory_id=mid, section_ids=[secs[1]["id"], 999999])["data"]
    assert len(by_id["sections"]) == 1
    assert 999999 in by_id.get("missing_section_ids", [])


def test_get_rejects_matched_mode(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    mid = tools.memory_write(content="x", subject="s")["data"]["id"]
    r = tools.memory_get(memory_id=mid, sections="matched")
    assert r["ok"] is False


def test_get_returns_split_subobject(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[{"title": "alpha"}, {"title": "beta", "anchor_text": "beta", "occurrence_index": 0}],
    )
    r = tools.memory_get(memory_id=mid)["data"]
    sp = r["split"]
    assert sp["status"] == "active"
    assert "revision" in sp and "section_count" in sp and "content_hash" in sp


# ==================================================================
#  §6.5 — status & doctor: split_capability
# ==================================================================

@pytest.mark.xfail(reason="T5: status still exposes split_enabled instead of split_capability")
def test_status_exposes_split_capability(tmp_path: Path) -> None:
    tools = make_plain_tools(tmp_path)
    status = tools.memory_status()["data"]
    assert "split_capability" in status
    sc = status["split_capability"]
    assert sc["available"] is False
    assert sc["reason"] in ("vec_not_ready", "embedder_unavailable")


@pytest.mark.xfail(reason="T5: doctor split checks not expanded")
def test_doctor_reports_backlog_and_capability(tmp_path: Path) -> None:
    """§6.5: doctor surfaces split.capability + long_unsplit_backlog."""
    tools = make_plain_tools(tmp_path)
    report = tools.memory_doctor_overview()["data"]
    check_ids = {f["check_id"] for f in report["findings"]}
    assert "split.capability" in check_ids
    assert "split.long_unsplit_backlog" in check_ids


# ==================================================================
#  §10 — config: forbidden new options absent
# ==================================================================

def test_config_has_no_split_llm_fallback() -> None:
    """§1.3: no split_llm_fallback / pending / fallback_active config."""
    fields = Settings.__dataclass_fields__
    assert "split_llm_fallback" not in fields
    assert "llm_provider" not in fields
    assert "llm_model" not in fields
