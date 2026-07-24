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
    """Two real headings outside fences + a heading-looking line inside a fence.

    The fenced ``## 假标题`` must NOT become a section boundary.
    """
    return (
        "# 真标题一\n" + ("正文一 " * 40) + "\n\n"
        "# 真标题二\n" + ("正文二 " * 40) + "\n\n"
        "```\n## 代码块里的假标题\n代码内容\n```\n\n"
        + ("更多正文 " * 40)
    )


def _content_with_colliding_heading_in_fence() -> str:
    """A real heading whose raw_line ALSO appears inside an EARLIER fence.

    Regression guard for the silent mis-segmentation fixed in v0.8.0: the rules
    parser is fence-aware, but offset re-location was a raw substring search
    (not fence-aware), so it silently picked the fenced occurrence as the
    boundary. The boundary must sit at the REAL heading.
    """
    return (
        "# 真标题一\n" + ("alpha 正文 " * 30) + "\n\n"
        "```python\n## 重复标题\nfoo = 1\nbar = 2\n```\n\n"
        "## 重复标题\n" + ("beta 正文 " * 30)
    )


def _content_with_colliding_heading_in_body() -> str:
    """A real heading whose raw_line ALSO appears earlier as a body substring.

    Same regression class as the fence variant: the parser ignores the body
    mention (the line does not start with ``#``), but a raw substring search
    would locate it first.
    """
    return (
        "# 真标题一\n"
        "注意：下方 see ## 重复标题 那段是正文引用，不是标题。\n"
        + ("alpha 正文 " * 30) + "\n\n"
        "## 重复标题\n" + ("beta 正文 " * 30)
    )


# ==================================================================
#  §2.1 / §6.5 — Registry: which tools exist
# ==================================================================

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


def test_no_pending_timeout_or_fallback_semantics_in_status_dict(tmp_path: Path) -> None:
    """§6.5: memory_status split_capability must not surface pending/fallback."""
    tools = make_plain_tools(tmp_path)
    status = tools.memory_status()["data"]
    sp = status["split_capability"]
    assert "pending" not in repr(sp).lower()
    assert "fallback" not in repr(sp).lower()


# ==================================================================
#  §6.1 — memory_write split return object shape
# ==================================================================

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
    # Compare against the DB-normalised content (trailing whitespace stripped).
    stored = mem["content"]
    assert sr["content"] == stored                 # full content, no truncation
    assert sr["content_hash"] == hashlib.sha256(stored.encode("utf-8")).hexdigest()
    assert sr["memory_version"] == mem["version"]
    assert sr["split_revision"] == mem["split_revision"]
    assert "split_schema" in sr


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


def test_write_rules_split_boundary_at_real_heading_not_fence(tmp_path: Path) -> None:
    """§7.1/§7.3 regression: a heading raw_line that ALSO appears inside an
    earlier fence must still split at the REAL heading.

    Before the v0.8.0 fix the parser was fence-aware but offset re-location was
    a raw substring search, so it silently picked the fenced occurrence — the
    continuity/coverage check still passed, producing a section whose title did
    not match its body. This test pins the boundary to the real heading.
    """
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = _content_with_colliding_heading_in_fence()
    r = tools.memory_write(content=content, subject="doc")
    assert r["ok"] is True
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] == "active"
    sections = tools.db.get_sections_by_memory(mem["id"])
    assert len(sections) == 2
    stored = mem["content"]
    fence_off = stored.find("## 重复标题\nfoo = 1")
    real_off = stored.find("## 重复标题\nbeta 正文")
    assert fence_off != -1 and real_off != -1 and fence_off < real_off
    # The boundary MUST be the real heading, never the fenced occurrence.
    assert sections[1]["start_offset"] == real_off
    assert sections[1]["start_offset"] != fence_off
    sec0 = stored[sections[0]["start_offset"]:sections[0]["end_offset"]]
    sec1 = stored[sections[1]["start_offset"]:sections[1]["end_offset"]]
    assert "beta 正文" not in sec0          # real heading's body not leaked into sec 0
    assert sec1.startswith("## 重复标题")    # sec 1 starts at the real heading
    assert "beta 正文" in sec1
    assert "foo = 1" not in sec1             # fence content stays in sec 0


def test_write_rules_split_boundary_at_real_heading_not_body(tmp_path: Path) -> None:
    """Same regression class as the fence variant, but the colliding text is a
    body substring (a line that does not start with ``#``). The body mention
    must not become the boundary either."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = _content_with_colliding_heading_in_body()
    r = tools.memory_write(content=content, subject="doc")
    assert r["ok"] is True
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] == "active"
    sections = tools.db.get_sections_by_memory(mem["id"])
    assert len(sections) == 2
    stored = mem["content"]
    body_off = stored.find("see ## 重复标题")
    real_off = stored.find("## 重复标题\nbeta 正文")
    assert body_off != -1 and real_off != -1 and body_off < real_off
    assert sections[1]["start_offset"] == real_off
    assert sections[1]["start_offset"] != body_off
    sec1 = stored[sections[1]["start_offset"]:sections[1]["end_offset"]]
    assert sec1.startswith("## 重复标题")
    assert "beta 正文" in sec1


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
    assert status["split_capability"]["available"] is False


# ==================================================================
#  §6.2 — memory_split as Agent continuation entry
# ==================================================================

def test_split_prepare_does_not_require_user_confirmation(tmp_path: Path) -> None:
    """§6.2: prepare returns full snapshot + schema, no user-confirmation gate."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "无标题纯文本 " * 400
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    prep = tools.memory_split(memory_id=mid)
    assert prep["ok"] is True
    assert prep["data"].get("requires_user_confirmation") is not True
    # Compare against the DB-stored content (normalised), not the raw input.
    assert prep["data"]["content"] == mem["content"]
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

def test_status_exposes_split_capability(tmp_path: Path) -> None:
    tools = make_plain_tools(tmp_path)
    status = tools.memory_status()["data"]
    assert "split_capability" in status
    sc = status["split_capability"]
    assert sc["available"] is False
    assert sc["reason"] in ("vec_not_ready", "embedder_unavailable")


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


# ==================================================================
#  Coverage backfill — closes review gaps G3–G12 (non-blocking debt).
#  Each block is labelled with its gap id from the v0.8.0 review report.
# ==================================================================

def _vec_tools(tmp_path: Path) -> MemoryTools:
    """vec-ready tools with the keyword embedder wired in (common setup)."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    return tools


def _publish_agent_sections(tools: MemoryTools, mid: int, sections: list[dict]) -> dict:
    """Publish agent-authored sections against the current memory snapshot."""
    mem = tools.db.get_memory(mid)
    ch = hashlib.sha256((mem["content"] or "").encode("utf-8")).hexdigest()
    return tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=ch, decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"], decision_split_revision=mem["split_revision"],
        sections=sections,
    )


def _two_heading_doc() -> str:
    return "# 第一章\n" + ("alpha 内容 " * 40) + "\n# 第二章\n" + ("beta 内容 " * 40)


# ---- G3: write-time count / size gates → split_request ----------------------

def test_write_exceeds_max_sections_returns_split_request(tmp_path: Path) -> None:
    """G3: > max_sections headings → split_request with rule_section_count_out_of_range."""
    tools = _vec_tools(tmp_path)
    tools.settings.max_sections = 3
    content = "\n\n".join(f"# 标题 {i}\n正文 {i}" for i in range(5))  # 5 headings
    r = tools.memory_write(content=content, subject="doc")
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] is None                      # NOT failed, NOT pending
    sp = r["data"]["split"]
    assert sp["applied"] is False and sp["mode"] == "agent_semantic"
    assert sp["reason"] == "rule_section_count_out_of_range"
    assert r["data"]["split_request"]["reason"] == "rule_section_count_out_of_range"


def test_write_candidate_section_too_large_returns_split_request(tmp_path: Path) -> None:
    """G3: a section slice > max_section_chars → split_request with rule_section_too_large."""
    tools = _vec_tools(tmp_path)
    tools.settings.max_section_chars = 20
    content = "# 标题一\n" + ("alpha正文 " * 30) + "\n# 标题二\nbeta 正文"
    r = tools.memory_write(content=content, subject="doc")
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] is None
    sp = r["data"]["split"]
    assert sp["applied"] is False
    assert sp["reason"] == "rule_section_too_large"
    assert tools.db.get_sections_by_memory(mem["id"]) == []  # nothing published


# ---- G4: content edit re-runs the split decision ----------------------------

def test_edit_content_re_splits_via_rules(tmp_path: Path) -> None:
    """G4: editing content to a rules-publishable doc re-splits synchronously."""
    tools = _vec_tools(tmp_path)
    mid = tools.memory_write(content="短初始内容", subject="doc")["data"]["id"]
    assert tools.db.get_memory(mid)["split_status"] is None
    r = tools.memory_edit(memory_id=mid, new_content=_two_heading_doc())
    assert r["ok"] is True
    mem = tools.db.get_memory(mid)
    assert mem["split_status"] == "active"
    assert len(tools.db.get_sections_by_memory(mid)) == 2
    assert r["data"]["split"]["mode"] == "rules" and r["data"]["split"]["applied"] is True


def test_edit_content_to_unstructured_returns_split_request(tmp_path: Path) -> None:
    """G4: editing an active-split doc to unstructured long → clears sections, split_request."""
    tools = _vec_tools(tmp_path)
    mid = tools.memory_write(content=_two_heading_doc(), subject="doc")["data"]["id"]
    assert tools.db.get_memory(mid)["split_status"] == "active"
    assert len(tools.db.get_sections_by_memory(mid)) == 2
    r = tools.memory_edit(memory_id=mid, new_content="无标题纯文本 " * 400)
    assert r["ok"] is True
    mem = tools.db.get_memory(mid)
    assert mem["split_status"] is None
    assert tools.db.get_sections_by_memory(mid) == []
    assert r["data"]["split"]["action_required"] == "memory_split"
    assert "split_request" in r["data"]


# ---- G5: tags-only edit on an active-split memory ---------------------------

def test_tags_only_edit_preserves_active_split_index(tmp_path: Path) -> None:
    """G5: tags-only edit on an already-split memory leaves sections/revision untouched."""
    tools = _vec_tools(tmp_path)
    mid = tools.memory_write(content=_two_heading_doc(), subject="doc")["data"]["id"]
    before = tools.db.get_memory(mid)
    before_rev = before["split_revision"]
    before_secs = tools.db.get_sections_by_memory(mid)
    assert before["split_status"] == "active"
    r = tools.memory_edit(memory_id=mid, tags_only=True, add_tags=["newtag"])
    assert r["ok"] is True
    after = tools.db.get_memory(mid)
    assert after["split_status"] == "active"
    assert after["split_revision"] == before_rev                 # revision unchanged
    assert tools.db.get_sections_by_memory(mid) == before_secs   # same sections
    assert "split_request" not in r["data"]                      # no Agent request
    assert r["data"].get("split", {}).get("action_required") is None
    assert "newtag" in r["data"]["tags"]


# ---- G6: CAS rejects when any single snapshot field is stale -----------------

@pytest.mark.parametrize("field,value,expect", [
    ("decision_content_hash", "0" * 64, "memory_changed"),
    ("decision_memory_version", 99999, "memory_changed"),
    ("decision_split_status", "failed", "split_revision_conflict"),
    ("decision_split_revision", 99999, "split_revision_conflict"),
])
def test_split_publish_rejects_each_stale_snapshot_field(
    tmp_path: Path, field, value, expect,
) -> None:
    """G6: changing any one of the four CAS fields rejects the publish."""
    tools = _vec_tools(tmp_path)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    ch = hashlib.sha256((mem["content"] or "").encode("utf-8")).hexdigest()
    kwargs = dict(
        memory_id=mid, split_decision="split",
        decision_content_hash=ch, decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"], decision_split_revision=mem["split_revision"],
        sections=[{"title": "a"}, {"title": "b", "anchor_text": "beta", "occurrence_index": 0}],
    )
    kwargs[field] = value
    r = tools.memory_split(**kwargs)
    assert r["ok"] is False
    assert expect in str(r["data"].get("error", ""))


# ---- G7: partial search joins multiple matched sections ---------------------

def test_search_partial_joins_multiple_matched_sections(tmp_path: Path) -> None:
    """G7: when ≥2 sections match, content is their full bodies joined by blank line."""
    tools = _vec_tools(tmp_path)
    content = ("alpha 内容一 " * 20) + "\n" + ("alpha 内容二 " * 20) + "\n" + ("beta 内容三 " * 20)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    pub = _publish_agent_sections(tools, mid, [
        {"title": "一"},
        {"title": "二", "anchor_text": "alpha 内容二", "occurrence_index": 0},
        {"title": "三", "anchor_text": "beta 内容三", "occurrence_index": 0},
    ])
    assert pub["ok"] is True and pub["data"]["section_count"] == 3
    r = tools.memory_search(query="alpha", query_embedding=_keyword_embedding("alpha"))
    hit = next(x for x in r["data"]["results"] if x["id"] == mid)
    assert hit["content_scope"] == "matched_sections"
    assert hit["matched_section_count"] == 2
    assert hit["total_section_count"] == 3
    ms = hit["matched_sections"]
    assert [m["section_index"] for m in ms] == [0, 1]           # joined in index order
    assert hit["content"] == ms[0]["content"] + "\n\n" + ms[1]["content"]


# ---- G8: Core defers all LLM work on the split_request path -----------------

class _SpyEmbedder:
    """Wraps an embedder and counts embed_text calls."""
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.embedding_space_id = inner.embedding_space_id
        self.last_encode_error = None

    def embed_text(self, prefix="", body="", max_body_chars=None):
        self.calls += 1
        return self._inner.embed_text(prefix=prefix, body=body, max_body_chars=max_body_chars)


def test_split_request_path_makes_no_embedder_call(tmp_path: Path) -> None:
    """G8: the split_request (unstructured) path defers ALL section/LLM work to the
    agent. Core must not attempt any — a spy on the only external component records
    zero calls, and the response explicitly flags extra_llm_call_required."""
    tools = _vec_tools(tmp_path)
    spy = _SpyEmbedder(_keyword_embedder())
    tools._embedder = spy
    r = tools.memory_write(content="无标题纯文本内容 " * 400, subject="plain")
    assert r["ok"] is True
    sp = r["data"]["split"]
    assert sp["action_required"] == "memory_split"
    assert sp["extra_llm_call_required"] is True
    assert spy.calls == 0          # Core did no section / LLM work itself


# ---- G9: every publish-failure mode leaves the original content readable -----

def test_split_missing_anchor_keeps_original_content(tmp_path: Path) -> None:
    """G9: a non-first section with no anchor → offset fail, original intact."""
    tools = _vec_tools(tmp_path)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    r = tools.memory_split(
        memory_id=mid, split_decision="split",
        decision_content_hash=hashlib.sha256((mem["content"] or "").encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"], decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[{"title": "a"}, {"title": "b"}],   # b has no anchor_text
    )
    assert r["ok"] is False
    assert tools.db.get_memory(mid)["content"] == content       # original intact
    assert tools.db.get_sections_by_memory(mid) == []           # nothing published


def test_split_oversized_section_keeps_original_content(tmp_path: Path) -> None:
    """G9: section_too_large rejection leaves the original content readable."""
    tools = _vec_tools(tmp_path)
    tools.settings.max_section_chars = 10
    content = "alpha " + ("x" * 200) + "\n" + "beta " + ("y" * 200)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    r = _publish_agent_sections(tools, mid, [
        {"title": "a"}, {"title": "b", "anchor_text": "beta", "occurrence_index": 0},
    ])
    assert r["ok"] is False and "section_too_large" in str(r["data"].get("error", ""))
    assert tools.db.get_memory(mid)["content"] == content
    assert tools.db.get_sections_by_memory(mid) == []


def test_split_embedding_failure_keeps_original_content(tmp_path: Path) -> None:
    """G9: an embedder that returns empty → embedding failure, original intact."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _MockManagedEmbedder(lambda text: [])   # always empty embedding
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    r = _publish_agent_sections(tools, mid, [
        {"title": "a"}, {"title": "b", "anchor_text": "beta", "occurrence_index": 0},
    ])
    assert r["ok"] is False and "embedding" in str(r["data"].get("error", ""))
    assert tools.db.get_memory(mid)["content"] == content
    assert tools.db.get_sections_by_memory(mid) == []


# ---- G10: catalog + sections=all expose embedding diagnostics ---------------

def test_get_catalog_and_all_expose_embedding_diagnostics(tmp_path: Path) -> None:
    """G10: the diagnostic surfaces (catalog/get all/doctor) DO carry embedding
    budget fields, unlike ordinary search matched_sections."""
    tools = _vec_tools(tmp_path)
    mid = tools.memory_write(content=_two_heading_doc(), subject="doc")["data"]["id"]
    assert tools.db.get_memory(mid)["split_status"] == "active"
    diag_keys = ("embedding_truncated", "embedding_original_tokens", "embedding_used_tokens")
    cat = tools.memory_get(memory_id=mid, sections="catalog")["data"]["section_catalog"]
    for e in cat:
        for k in diag_keys:
            assert k in e
    allr = tools.memory_get(memory_id=mid, sections="all")["data"]["sections"]
    for s in allr:
        for k in diag_keys:
            assert k in s


# ---- G11: legacy split.enabled / preview_chars warn + ignore ----------------

def test_legacy_split_config_keys_warned_and_ignored(tmp_path: Path, monkeypatch) -> None:
    """G11: an old config with split.enabled + section_zero_match_preview_chars
    starts up, ignores both keys, and emits exactly two deprecation warnings."""
    import json as _json
    cfg = {
        "db_path": str(tmp_path / "legacy.sqlite3"),
        "backup_jsonl": str(tmp_path / "legacy.jsonl"),
        "split": {
            "enabled": True, "threshold": 4000,
            "section_zero_match_preview_chars": 2000, "max_section_chars": 3600,
        },
    }
    cfg_path = tmp_path / "legacy.json"
    cfg_path.write_text(_json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))
    s = Settings.from_env()
    assert not hasattr(s, "split_enabled")
    assert s.split_threshold == 4000 and s.max_section_chars == 3600   # other keys still parsed
    warns = [w for w in s.config_warnings if "removed in v0.8" in w]
    assert len(warns) == 2
    assert any("split.enabled" in w for w in warns)
    assert any("section_zero_match_preview_chars" in w for w in warns)


# ---- G12: parser edge cases (preamble / title_path / CRLF / empty title) ----

def test_rules_split_preamble_folded_into_first_section(tmp_path: Path) -> None:
    """G12: text before the first heading is part of section 0 (start_offset 0)."""
    tools = _vec_tools(tmp_path)
    content = "这是前言 preamble\n# 第一章\n" + ("alpha 内容 " * 30) + "\n# 第二章\n" + ("beta 内容 " * 30)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    secs = tools.db.get_sections_by_memory(mid)
    assert len(secs) == 2
    assert secs[0]["start_offset"] == 0
    stored = tools.db.get_memory(mid)["content"]
    assert stored[secs[0]["start_offset"]:secs[0]["end_offset"]].startswith("这是前言")


def test_rules_split_title_path_for_nested_headings(tmp_path: Path) -> None:
    """G12: a nested heading carries a title_path of its ancestor chain."""
    tools = _vec_tools(tmp_path)
    content = "# 父级\n" + ("alpha 内容 " * 30) + "\n## 子级\n" + ("beta 内容 " * 30)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    secs = tools.db.get_sections_by_memory(mid)
    assert len(secs) == 2
    # section 1 is the nested ## 子级 under # 父级
    assert secs[1]["title_path"] == "父级 / 子级"
    assert secs[0]["title_path"] is None      # top-level heading has no path


def test_rules_split_handles_crlf_line_endings(tmp_path: Path) -> None:
    """G12: CRLF (\\r\\n) line endings still parse into 2 sections."""
    tools = _vec_tools(tmp_path)
    content = "# 第一章\r\n" + ("alpha 内容 " * 30) + "\r\n# 第二章\r\n" + ("beta 内容 " * 30)
    mid = tools.memory_write(content=content, subject="doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    assert mem["split_status"] == "active"
    secs = tools.db.get_sections_by_memory(mid)
    assert len(secs) == 2
    # full coverage with CRLF preserved
    stored = mem["content"]
    assert stored[secs[0]["start_offset"]:secs[0]["end_offset"]] + stored[secs[1]["start_offset"]:secs[1]["end_offset"]] == stored


def test_empty_title_heading_is_not_a_section(tmp_path: Path) -> None:
    """G12: a '## ' line with no title text is not a heading → with only one real
    heading left, the write returns a split_request rather than crashing."""
    tools = _vec_tools(tmp_path)
    content = "## \n正文\n# 唯一真标题\n" + ("alpha 内容 " * 30)
    r = tools.memory_write(content=content, subject="doc")
    mem = tools.db.get_memory(r["data"]["id"])
    assert mem["split_status"] is None                 # only 1 valid heading → no publish
    assert r["data"]["split"]["action_required"] == "memory_split"
    assert tools.db.get_sections_by_memory(mem["id"]) == []
