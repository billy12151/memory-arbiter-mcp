"""0.18.0 console knowledge workbench — kb_member data layer + API tests.

Grows per commit: c1 migration/primitives, c4 tree/candidates/membership API,
c5 llm config, c6 llm ask/save. See docs/mema-console-knowledge-plan-2026-10-10.md
(in ZCodeProject) Part B for the spec under test.
"""
from __future__ import annotations

import sqlite3

import pytest

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.console_api import ConsoleAPI
from memory_arbiter.tools import MemoryTools


def _tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "kb.sqlite3",
        backup_jsonl=tmp_path / "kb.jsonl",
        client="pytest",
        agent_id="kb-test",
        workspace="kb-ws",
    )
    return MemoryTools(settings)


def _write(tools: MemoryTools, subject: str, *, workspace: str = "kb-ws", tags: list[str] | None = None, source_type: str = "agent_generated") -> int:
    resp = tools.memory_write(
        content=f"content of {subject}",
        subject=subject,
        tags=tags or [],
        source_type=source_type,
        workspace=workspace,
    )
    assert isinstance(resp, dict) and resp.get("ok"), resp
    data = resp.get("data")
    assert isinstance(data, dict) and data.get("id"), resp
    return int(data["id"])


# ---- c1: migration + primitives -------------------------------------------


def test_kb_member_column_and_index_exist(tmp_path: Path) -> None:
    tools = _tools(tmp_path)  # boot creates schema + additive structures
    with tools.db.connection() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert "kb_member" in cols
        default = [row for row in conn.execute("PRAGMA table_info(memories)").fetchall()
                   if row["name"] == "kb_member"][0]["dflt_value"]
        assert default == "0"
        idx = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_kb_active'"
        ).fetchone()
        assert idx is not None
        row = conn.execute("SELECT id, kb_member FROM memories LIMIT 1").fetchone()
        if row is not None:
            assert row["kb_member"] in (0, 1)


def test_kb_migration_idempotent_on_legacy_db(tmp_path: Path) -> None:
    """Simulate a pre-0.18.0 database: create the table WITHOUT kb_member, boot
    twice, assert the column appears once and repeated boots are no-ops."""
    tools = _tools(tmp_path)
    db_path = tools.settings.db_path
    with sqlite3.connect(db_path) as conn:
        # the partial index depends on kb_member, so drop it before the column
        conn.execute("DROP INDEX IF EXISTS idx_memories_kb_active")
        conn.execute("ALTER TABLE memories DROP COLUMN kb_member")
    for _ in range(2):
        Settings.from_env  # noqa: B018 (documented no-op; keep boot explicit)
        MemoryTools(Settings(
            db_path=db_path,
            backup_jsonl=tools.settings.backup_jsonl,
            client="pytest",
            agent_id="kb-test",
            workspace="kb-ws",
        ))
    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert "kb_member" in cols


def test_set_kb_member_roundtrip_and_listing(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    mid = _write(tools, "kb member roundtrip", tags=["vop"])
    other = _write(tools, "not a member", workspace="other-ws")

    assert tools.db.memories.set_kb_member(mid, True) == 1
    # idempotent: same value again reports 0 changed rows but still fine
    assert tools.db.memories.set_kb_member(mid, True) == 0

    members = tools.db.memories.list_kb_members()
    ids = {m["id"] for m in members}
    assert mid in ids and other not in ids
    entry = next(m for m in members if m["id"] == mid)
    assert entry["tags"] == ["vop"]  # parsed out of the stored JSON string
    assert entry["content_chars"] == len(f"content of kb member roundtrip")
    assert entry["workspace_canonical"] == "kb-ws"

    # workspace filter slices to one bucket
    assert {m["id"] for m in tools.db.memories.list_kb_members("kb-ws")} == {mid}

    assert tools.db.memories.set_kb_member(mid, False) == 1
    assert tools.db.memories.list_kb_members() == []


def test_set_kb_member_rejects_non_active(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    mid = _write(tools, "to be expired")
    tools.memory_supersede(memory_id=mid, reason="test", authorized=True)
    assert tools.db.memories.set_kb_member(mid, True) == 0
    assert tools.db.memories.set_kb_member(999999, True) == 0


def test_kb_membership_audited(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    mid = _write(tools, "audited member")
    tools.db.memories.set_kb_member(mid, True, actor="console")
    rows = tools.db.audit.recent_governance_actions(limit=10)
    match = [r for r in rows if r["action"] == "kb_membership_set"]
    assert match, rows
    assert match[0]["detail"]["memory_id"] == mid
    assert match[0]["detail"]["actor"] == "console"


def test_new_tag_replace_does_not_drop_kb_member(tmp_path: Path) -> None:
    """The design rationale pin: update's new_tags full-replace must not touch
    the governance bit (the reason kb_member is a column, not a tag)."""
    tools = _tools(tmp_path)
    mid = _write(tools, "tag replace keeps kb", tags=["old"])
    assert tools.db.memories.set_kb_member(mid, True) == 1
    tools.memory_edit(
        memory_id=mid, tags_only=True,
        add_tags=["brand", "new"], remove_tags=["old"], reason="retag",
    )
    row = tools.db.get_memory(mid)
    assert row["kb_member"] == 1
    assert row["tags"] == ["brand", "new"]


# ---- c3: renderMd escaping/rendering ---------------------------------------

_NODE_JS_PROBE = """
const html = renderMd(input);
console.log(JSON.stringify(html));
"""


def _render_md_via_node(text: str) -> str:
    import json
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available for renderMd JS checks")
    from memory_arbiter.console_static import INDEX_HTML

    script = re.search(r"<script>(.*)</script>", INDEX_HTML, re.S)
    assert script is not None
    js = script.group(1)
    prefix = js.split("const I18N")[0]
    esc_fn = "function esc(v){ return String(v ?? '').replace(/[&<>\"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[s])); }"
    inline_fn = re.search(r"function inlineMd.*?\n}", js, re.S)
    md_fn = re.search(r"function renderMd.*?\n}", js, re.S)
    assert inline_fn and md_fn, "renderMd/inlineMd missing from INDEX_HTML"
    body = (
        prefix + esc_fn + "\n"
        + inline_fn.group(0) + "\n"
        + md_fn.group(0) + "\n"
        + f"const input = {json.dumps(text)};\n" + _NODE_JS_PROBE
    )
    proc = subprocess.run([node, "-e", body], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_render_md_escapes_html_before_rendering() -> None:
    out = _render_md_via_node('<script>alert(1)</script> **bold**')
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "<strong>bold</strong>" in out


def test_render_md_attribute_injection_stays_inert() -> None:
    """No content-derived attributes: img/onerror payloads must come out as
    escaped text, never as tags or attributes (CSP allows unsafe-inline, so
    the esc-first pipeline is the only line of defense)."""
    out = _render_md_via_node('<img src=x onerror=alert(1)>')
    assert "<img" not in out
    assert "&lt;img" in out


def test_render_md_structures() -> None:
    out = _render_md_via_node(
        "# Title\n\n- a\n- b\n\n> quote\n\n| h1 | h2 |\n| --- | --- |\n| 1 | 2 |\n\n```\ncode\n```"
    )
    for fragment in ("<h1>", "<ul>", "<li>", "<blockquote>", "<table>", "<th>", "<pre>"):
        assert fragment in out, (fragment, out)


def test_render_md_links_render_as_text() -> None:
    out = _render_md_via_node("see https://evil.example x")
    assert "<a " not in out and "href" not in out
    assert "https://evil.example" in out


def test_render_md_fold_marker_for_long_content() -> None:
    """memoryBodyHtml keeps long (>4000 chars) bodies behind the fold marker."""
    from memory_arbiter.console_static import INDEX_HTML

    assert 'id="memFold"' in INDEX_HTML
    assert "function renderMd(" in INDEX_HTML
    assert "toggleMemFold" in INDEX_HTML
