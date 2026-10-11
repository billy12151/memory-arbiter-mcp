"""0.18.0 console knowledge workbench — kb_member data layer + API tests.

Grows per commit: c1 migration/primitives, c4 tree/candidates/membership API,
c5 llm config, c6 llm ask/save. See docs/mema-console-knowledge-plan-2026-10-10.md
(in ZCodeProject) Part B for the spec under test.
"""
from __future__ import annotations

import json
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


# ---- c4: knowledge tree / candidates / membership --------------------------

import threading
import urllib.error
import urllib.request

from memory_arbiter.console_api import ConsoleAPI
from memory_arbiter.console_server import build_http_server


def _kb_api(tmp_path: Path) -> ConsoleAPI:
    return ConsoleAPI(_tools(tmp_path))


def test_knowledge_tree_groups_by_workspace_and_tags(tmp_path: Path) -> None:
    api = _kb_api(tmp_path)
    tools = api.tools
    a = _write(tools, "vop basics", tags=["vop", "pm"])
    b = _write(tools, "vop advanced", tags=["vop"])
    c = _write(tools, "untagged one", tags=[])
    d = _write(tools, "other bucket doc", workspace="other-ws", tags=["doc"])
    for mid in (a, b, c, d):
        tools.db.memories.set_kb_member(mid, True)

    tree = api.knowledge_tree()
    assert tree["total"] == 4
    by_name = {w["workspace"]: w for w in tree["workspaces"]}
    assert set(by_name) == {"kb-ws", "other-ws"}
    kb = by_name["kb-ws"]
    assert kb["total"] == 3
    tags = {g["tag"]: g for g in kb["tags"]}
    assert tags["vop"]["count"] == 2  # multi-tag memory counted in both groups
    assert tags["pm"]["memory_ids"] == [a]
    assert tags["(untagged)"]["memory_ids"] == [c]

    sliced = api.knowledge_tree(workspace="other-ws")
    assert sliced["total"] == 1
    assert sliced["workspaces"][0]["workspace"] == "other-ws"


def test_knowledge_candidates_excludes_members(tmp_path: Path) -> None:
    api = _kb_api(tmp_path)
    tools = api.tools
    doc = _write(tools, "extracted doc", source_type="document_extracted")
    doc2 = _write(tools, "another doc", source_type="document_extracted", workspace="other-ws")
    _write(tools, "agent note", source_type="agent_generated")

    items = api.knowledge_candidates()["items"]
    ids = {i["id"] for i in items}
    assert doc in ids and doc2 in ids  # only document_extracted actives

    tools.db.memories.set_kb_member(doc, True)
    ids2 = {i["id"] for i in api.knowledge_candidates()["items"]}
    assert doc not in ids2 and doc2 in ids2

    sliced = {i["id"] for i in api.knowledge_candidates(workspace="other-ws")["items"]}
    assert sliced == {doc2}


def test_knowledge_membership_roundtrip_and_errors(tmp_path: Path) -> None:
    api = _kb_api(tmp_path)
    tools = api.tools
    mid = _write(tools, "membership target")

    res = api.knowledge_membership(mid, "add")
    assert res == {"memory_id": mid, "member": True, "changed": True}
    assert api.knowledge_membership(mid, "add")["changed"] is False  # idempotent
    assert api.knowledge_membership(mid, "remove")["member"] is False

    assert api.knowledge_membership(mid, "bogus")["error"]
    assert api.knowledge_membership(424242, "add")["_http_status"] == 404


def test_knowledge_membership_strict_equality_gate(tmp_path: Path) -> None:
    """strict: only the selected single canonical bucket is operable — the
    vector-admitted neighbor set is NOT a console boundary (plan R2 P1-7)."""
    settings = Settings(
        db_path=tmp_path / "strict.sqlite3", backup_jsonl=tmp_path / "strict.jsonl",
        client="pytest", agent_id="kb-test", workspace="kb-ws", isolation="strict",
    )
    tools = MemoryTools(settings)
    api = ConsoleAPI(tools)
    mine = _write(tools, "in bucket")  # first write to a new bucket lands pending under strict
    other = _write(tools, "other bucket", workspace="other-ws")
    # activate both pending buckets through the governance surface
    tools.memory_confirm_pending_workspace(
        memory_id=mine, canonical="kb-ws", reason="test", authorized=True, workspace="kb-ws",
    )
    tools.memory_confirm_pending_workspace(
        memory_id=other, canonical="other-ws", reason="test", authorized=True, workspace="other-ws",
    )
    assert tools.db.get_memory(mine)["status"] == "active"

    assert api.knowledge_membership(mine, "add", workspace="kb-ws")["member"] is True
    denied = api.knowledge_membership(other, "add", workspace="kb-ws")
    assert denied["_http_status"] == 403
    missing = api.knowledge_membership(mine, "add")
    assert missing["_http_status"] == 400  # strict requires explicit workspace
    # tree/candidates slice under strict too
    assert api.knowledge_tree(workspace="kb-ws")["total"] == 1


def _http_json(
    port: int, path: str, body: dict | None = None,
    headers: dict[str, str] | None = None, method: str | None = None,
):
    url = f"http://127.0.0.1:{port}{path}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Console-Request", "1")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    return urllib.request.urlopen(req, timeout=5)


def test_post_defense_and_membership_route(tmp_path: Path) -> None:
    api = _kb_api(tmp_path)
    tools = api.tools
    mid = _write(tools, "http route target")
    server = build_http_server("127.0.0.1", 0, api=api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _, port = server.server_address

        # missing custom header → 403
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/knowledge/membership",
            data=json.dumps({"memory_id": mid, "op": "add"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected 403")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403

        # wrong content-type → 415
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/knowledge/membership",
            data=b"memory_id=1", headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Console-Request": "1",
            }, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected 415")
        except urllib.error.HTTPError as exc:
            assert exc.code == 415

        # malformed json → 400
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/knowledge/membership",
            data=b"{not json", headers={
                "Content-Type": "application/json", "X-Console-Request": "1",
            }, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400

        # oversized content-length → 413 without reading the body
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("POST", "/api/knowledge/membership")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("X-Console-Request", "1")
        conn.putheader("Content-Length", str(3 * 1024 * 1024))
        conn.endheaders()
        try:
            assert conn.getresponse().status == 413
        finally:
            conn.close()

        # OPTIONS never carries CORS headers (R2 P1-1 regression pin)
        opt = urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/api/knowledge/tree", method="OPTIONS",
            ), timeout=5,
        )
        assert opt.status == 204
        assert not any(k.lower().startswith("access-control-") for k in opt.headers.keys())

        # happy path over HTTP
        resp = _http_json(port, "/api/knowledge/membership", {"memory_id": mid, "op": "add"})
        payload = json.loads(resp.read().decode("utf-8"))
        assert payload["member"] is True

        # GET endpoints over HTTP
        tree = json.loads(_http_json(port, "/api/knowledge/tree").read().decode("utf-8"))
        assert tree["total"] == 1
    finally:
        server.shutdown()
        server.server_close()


# ---- c5: llm config ---------------------------------------------------------

import os

from memory_arbiter.config import Settings as CfgSettings, write_llm_config


def test_write_llm_config_roundtrip_and_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "config.json"
    applied, path, normalized, _warnings = write_llm_config(
        {
            "base_url": "https://api.example.com/v4",
            "api_key": "sk-test-1234",
            "model": "glm-test",
            "max_context_tokens": "64000",
        },
        config_path=cfg_path,
        identity=CfgSettings(
            db_path=tmp_path / "x.sqlite3", backup_jsonl=tmp_path / "x.jsonl",
            client="pytest", agent_id="kb-test",
        ),
    )
    assert applied == ["api_key", "base_url", "max_context_tokens", "model"]
    assert normalized["max_context_tokens"] == 64000
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["client"] == "pytest" and on_disk["agent_id"] == "kb-test"  # skeleton identity
    assert on_disk["llm"]["base_url"] == "https://api.example.com/v4"

    # restart path: from_env parses the llm section back
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))
    s = CfgSettings.from_env()
    assert s.llm_base_url == "https://api.example.com/v4"
    assert s.llm_api_key == "sk-test-1234"
    assert s.llm_model == "glm-test"
    assert s.llm_max_context_tokens == 64000
    assert s.llm_max_output_tokens == 8192  # default preserved

    # unknown keys rejected; other config sections survive a write
    with pytest.raises(ValueError):
        write_llm_config({"evil": "1"}, config_path=cfg_path)
    cfg_path.write_text(json.dumps({"workspace": "keepme", "llm": {"model": "m1"}}))
    write_llm_config({"model": "m2"}, config_path=cfg_path)
    after = json.loads(cfg_path.read_text())
    assert after["workspace"] == "keepme" and after["llm"]["model"] == "m2"


def test_llm_config_save_endpoint_mask_and_sentinel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "config.json"
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))
    api = ConsoleAPI(MemoryTools(Settings(
        db_path=tmp_path / "c.sqlite3", backup_jsonl=tmp_path / "c.jsonl",
        client="pytest", agent_id="kb-test", workspace="kb-ws",
    )))
    # settings_view masks the key and marks only llm editable
    api.llm_config_save({"base_url": "https://x.example", "api_key": "sk-secret-99"})
    view = api.settings_view()
    llm_items = [i for g in view["groups"] for i in g["items"] if g["key"] == "llm"]
    key_item = next(i for i in llm_items if i["path"] == "llm.api_key")
    assert key_item["current"] == "***t-99" and "sk-secret-99" not in str(view)
    assert all(i["editable"] for i in llm_items)
    ro_items = [i for g in view["groups"] for i in g["items"] if g["key"] != "llm"]
    assert ro_items and all(not i["editable"] for i in ro_items)

    # sentinel keeps the stored key; hot reload applies immediately
    res = api.llm_config_save({"api_key": "__unchanged__", "model": "glm-4.7"})
    assert "api_key" not in res["saved"]
    assert api.settings.llm_model == "glm-4.7"
    assert api.settings.llm_api_key == "sk-secret-99"
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["llm"]["api_key"] == "sk-secret-99"

    # bad input surfaces as 400 payload
    bad = api.llm_config_save({"nope": 1})
    assert bad["_http_status"] == 400


def test_setup_merges_unknown_sections(tmp_path: Path) -> None:
    """setup must not wipe the llm section when regenerating config (R2 P1-9)."""
    from memory_arbiter import setup_cli

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "client": "old", "workspace": "keepme",
        "llm": {"base_url": "https://api.example", "model": "glm-x"},
    }))
    fresh = setup_cli._default_config_dict(tmp_path / "m.gguf", tmp_path / "db.sqlite3", tmp_path / "b.jsonl")
    merged = setup_cli._merge_unknown_sections(dict(fresh), cfg_path)
    assert merged["llm"]["model"] == "glm-x"  # unknown section survives
    assert merged["workspace"] == fresh["workspace"]  # known keys stay fresh
    # corrupt old config: merge is a no-op, setup proceeds
    cfg_path.write_text("{not json")
    assert setup_cli._merge_unknown_sections(dict(fresh), cfg_path) == fresh


def test_knowledge_tree_special_chars_in_workspace(tmp_path: Path) -> None:
    """Workspace names with quotes/unicode must survive the canonical
    resolution and parameterized SQL of the tree/membership path."""
    api = _kb_api(tmp_path)
    tools = api.tools
    weird = "桶'名\"引号"
    mid = _write(tools, "weird bucket doc", workspace=weird)
    tools.db.memories.set_kb_member(mid, True)
    tree = api.knowledge_tree(workspace=weird)
    assert tree["total"] == 1
    assert tree["workspaces"][0]["workspace"] == weird
    assert api.knowledge_membership(mid, "remove")["member"] is False
    assert api.knowledge_membership(mid, "add")["member"] is True


# ---- c6: llm ask / test / save ----------------------------------------------

import http.server
import threading as _threading

from memory_arbiter.llm_client import ContextMaterial, LLMError, assemble_context, validate_output

_FAKE_STATE: dict = {"responses": [], "requests": []}


class _FakeOpenAIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        _FAKE_STATE["requests"].append(body)
        if self.path == "/v1/chat/completions" or self.path == "/chat/completions":
            self._reply(200, _FAKE_STATE["responses"].pop(0) if _FAKE_STATE["responses"] else {"ok": True})
        else:
            self._reply(404, {"error": "not found"})

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _fake_llm_settings(tmp_path: Path, port: int) -> Settings:
    return Settings(
        db_path=tmp_path / "llm.sqlite3", backup_jsonl=tmp_path / "llm.jsonl",
        client="pytest", agent_id="kb-test", workspace="kb-ws",
        llm_base_url=f"http://127.0.0.1:{port}", llm_api_key="sk-fake",
        llm_model="fake-model", llm_max_context_tokens=32000,
    )


def _fake_completion(content: str, reasoning: str | None = None) -> dict:
    message: dict = {"content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


@pytest.fixture()
def fake_llm():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    _threading.Thread(target=server.serve_forever, daemon=True).start()
    _FAKE_STATE["responses"] = []
    _FAKE_STATE["requests"] = []
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


GOOD_SUMMARY = (
    "总括：这两条记忆说明了测试主题的核心事实，材料之间互相印证，"
    "给出了从基础到扩展的完整图景，适合作为复习的起点材料来看待。[101]\n"
    "- 要点一：第一条记忆确认了基础事实，即测试主题的出发点与边界条件 [101]\n"
    "- 要点二：第二条记忆给出了扩展方向，与要点一互相印证而非互相矛盾 [102]"
)


def test_llm_ask_happy_path_with_reasoning(tmp_path: Path, fake_llm) -> None:
    _FAKE_STATE["responses"] = [_fake_completion(GOOD_SUMMARY, reasoning="我需要先归纳材料")]
    port = fake_llm.server_address[1]
    api = ConsoleAPI(MemoryTools(_fake_llm_settings(tmp_path, port)))
    mid = _write(api.tools, "topic one", tags=["t"])
    mid2 = _write(api.tools, "topic two", tags=["t"])
    api.knowledge_membership(mid, "add")
    api.knowledge_membership(mid2, "add")

    res = api.llm_ask([mid, mid2], "summarize")
    assert "error" not in res, res
    assert res["degraded"] is False and res["attempt"] == 1
    assert res["reasoning"] == "我需要先归纳材料"
    assert res["source_memory_ids"] == [mid, mid2]
    sent = _FAKE_STATE["requests"][0]
    assert sent["model"] == "fake-model" and sent["max_tokens"] == 8192
    assert f"[{mid}]" in sent["messages"][1]["content"]


def test_llm_ask_retry_then_degraded(tmp_path: Path, fake_llm) -> None:
    # attempt 1 fails the gate (too short), attempt 2 fails again → degraded
    _FAKE_STATE["responses"] = [
        _fake_completion("short"),
        _fake_completion("still short"),
    ]
    port = fake_llm.server_address[1]
    api = ConsoleAPI(MemoryTools(_fake_llm_settings(tmp_path, port)))
    mid = _write(api.tools, "m1")
    res = api.llm_ask([mid], "summarize")
    assert res["degraded"] is True and res["attempt"] == 2
    assert res["warnings"] and "100" in res["warnings"][0]
    assert len(_FAKE_STATE["requests"]) == 2  # exactly one corrective retry

    # retry path passes on the second attempt
    _FAKE_STATE["responses"] = [_fake_completion("short"), _fake_completion(GOOD_SUMMARY)]
    res2 = api.llm_ask([mid], "summarize")
    assert res2["degraded"] is False and res2["attempt"] == 2
    corrective = _FAKE_STATE["requests"][1]["messages"]
    assert corrective[-1]["role"] == "user" and "结构校验" in corrective[-1]["content"]


def test_llm_ask_v1_fallback_and_cache_keyed_by_url(tmp_path: Path, fake_llm) -> None:
    """404 on /chat/completions falls back to /v1/chat/completions; the probe
    is cached per base_url (R2 P2-2)."""
    from memory_arbiter import llm_client

    port = fake_llm.server_address[1]
    llm_client._V1_PROBE_CACHE.clear()
    settings = _fake_llm_settings(tmp_path, port)
    # first response is a 404 (plain path missing), then a good v1 response
    _FAKE_STATE["responses"] = [{"error": {"message": "no route"}}]

    class _First404Handler(_FakeOpenAIHandler):
        def do_POST(self):  # noqa: N802
            if self.path == "/chat/completions":
                _FAKE_STATE["requests"].append({"path": self.path})
                self._reply(404, {"error": "not found"})
                return
            super().do_POST()

    server2 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _First404Handler)
    _threading.Thread(target=server2.serve_forever, daemon=True).start()
    try:
        port2 = server2.server_address[1]
        settings2 = Settings(
            db_path=tmp_path / "llm2.sqlite3", backup_jsonl=tmp_path / "llm2.jsonl",
            client="pytest", agent_id="kb-test", workspace="kb-ws",
            llm_base_url=f"http://127.0.0.1:{port2}", llm_api_key="sk-fake",
            llm_model="fake-model", llm_max_context_tokens=32000,
        )
        _FAKE_STATE["responses"] = [_fake_completion(GOOD_SUMMARY)]
        result = llm_client.chat_completion(settings2, [{"role": "user", "content": "hi"}])
        assert result["content"] == GOOD_SUMMARY
        assert llm_client._V1_PROBE_CACHE[f"http://127.0.0.1:{port2}"] == "v1"
        # second call: the cached variant serves directly (no 404 round trip;
        # the fake consumed exactly one response — a plain-path probe would
        # have eaten the 404 first and left the good response unconsumed)
        _FAKE_STATE["requests"] = []
        _FAKE_STATE["responses"] = [_fake_completion(GOOD_SUMMARY)]
        again = llm_client.chat_completion(settings2, [{"role": "user", "content": "hi"}])
        assert again["content"] == GOOD_SUMMARY
        assert llm_client._V1_PROBE_CACHE[f"http://127.0.0.1:{port2}"] == "v1"
    finally:
        server2.shutdown()
        server2.server_close()


def test_llm_ask_budget_and_config_errors(tmp_path: Path, fake_llm) -> None:
    port = fake_llm.server_address[1]
    settings = _fake_llm_settings(tmp_path, port)
    settings.llm_max_context_tokens = 5000  # budget after reserves ≈ 104 tokens
    api = ConsoleAPI(MemoryTools(settings))
    huge = api.tools.memory_write(
        content="词" * 2000, subject="huge memory", tags=[], workspace="kb-ws",
    )["data"]["id"]
    # one huge memory → single-over-budget error with guidance (no LLM call made)
    res = api.llm_ask([huge], "summarize")
    assert res["_http_status"] == 400 and "budget" in res["error"]
    assert _FAKE_STATE["requests"] == []  # fails before transport

    # unconfigured llm → guided error
    bare = Settings(
        db_path=tmp_path / "bare.sqlite3", backup_jsonl=tmp_path / "bare.jsonl",
        client="t", agent_id="t", workspace="kb-ws",
    )
    api2 = ConsoleAPI(MemoryTools(bare))
    own = _write(api2.tools, "own memory")
    res2 = api2.llm_ask([own], "summarize")
    assert res2["_http_status"] == 400 and "Settings" in res2["error"]

    # bad action / empty ids / oversize ids
    assert api.llm_ask([huge], "nope")["_http_status"] == 400
    assert api.llm_ask([], "summarize")["_http_status"] == 400
    assert api.llm_ask(list(range(51)), "summarize")["_http_status"] == 400


def test_llm_timeline_fabricated_year_rejected(tmp_path: Path, fake_llm) -> None:
    port = fake_llm.server_address[1]
    api = ConsoleAPI(MemoryTools(_fake_llm_settings(tmp_path, port)))
    mid = api.tools.memory_write(
        content="in 2015 something happened", subject="policy memory",
        tags=[], workspace="kb-ws", event_time="2015-06-01",
    )["data"]["id"]
    _FAKE_STATE["responses"] = [
        _fake_completion("- 2015 — 事件A [101]\n- 2039 — 我编造的事件 [101]"),
        _fake_completion("- 2015 — 事件A [101]\n- 2015 — 事件B [101]"),
    ]
    api.tools.db.memories.set_kb_member(mid, True)
    res = api.llm_ask([mid], "timeline")
    assert res["attempt"] == 2 and res["degraded"] is False
    assert "2039" in _FAKE_STATE["requests"][1]["messages"][-1]["content"]


def test_validate_output_table() -> None:
    mats = [ContextMaterial(1, "s", "2015-01-01", "in 2015 something happened; 2019 too")]
    assert validate_output("timeline", "- 2015 — a [1]", mats) == []
    errs = validate_output("timeline", "- 2039 — fabricated [1]", mats)
    assert errs and "2039" in errs[0]
    assert validate_output("quiz_me", "一?二?三?", []) == []
    assert validate_output("quiz_me", "一?", [])  # too few questions
    assert validate_output("outline", "# only one", [])  # fewer than two ## headings
    assert validate_output("custom", "ok", [])  # under 10 chars → violation
    assert validate_output("custom", "ok-ish answer", []) == []


def test_llm_test_probe(tmp_path: Path, fake_llm) -> None:
    port = fake_llm.server_address[1]
    api = ConsoleAPI(MemoryTools(_fake_llm_settings(tmp_path, port)))
    _FAKE_STATE["responses"] = [_fake_completion("ok")]
    res = api.llm_test()
    assert res["reply"] == "ok" and res["model"] == "fake-model"


def test_knowledge_save_two_step_and_attribution(tmp_path: Path) -> None:
    api = _kb_api(tmp_path)
    res = api.knowledge_save(
        subject="总结 · 测试主题",
        content="# 总结\n\n要点一 [101]",
        tags=["llm-output", "llm-summary"],
        workspace="kb-ws",
    )
    assert res["saved"] is True and res["kb_member"] is True
    row = api.tools.db.get_memory(res["memory_id"])
    assert row["kb_member"] == 1
    assert row["agent_id"] == "console"  # request-identity attribution (R2 P0-1)
    assert row["tags"] == ["llm-output", "llm-summary"]
    assert row["status"] == "active"

    # empty save rejected
    assert api.knowledge_save(subject="", content="x", tags=[], workspace=None)["_http_status"] == 400


def test_knowledge_save_strict_pending_branch(tmp_path: Path) -> None:
    """strict + unconfirmed bucket: write lands pending; save reports the
    governance path instead of failing silently (plan R2 P1-10)."""
    settings = Settings(
        db_path=tmp_path / "sp.sqlite3", backup_jsonl=tmp_path / "sp.jsonl",
        client="t", agent_id="t", workspace="kb-ws", isolation="strict",
    )
    tools = MemoryTools(settings)
    api = ConsoleAPI(tools)
    res = api.knowledge_save(subject="s", content="c", tags=[], workspace="brand-new-ws")
    assert res["saved"] is True and res["pending_workspace"] is True and res["kb_member"] is False
    row = tools.db.get_memory(res["memory_id"])
    assert row["status"] == "pending"


# ---- c6+: console edit / retire (owner 2026-10-10 mid-run instruction) ------

def test_knowledge_edit_and_retire_in_process(tmp_path: Path) -> None:
    """Console edit/retire go through the same in-process pipeline
    (memory_edit / memory_supersede) — no MCP service, no cross-process
    write; version bump + history + kb_member survival included."""
    api = _kb_api(tmp_path)
    tools = api.tools
    mid = _write(tools, "editable knowledge", tags=["knowledge-ish"])
    tools.db.memories.set_kb_member(mid, True)

    res = api.knowledge_edit(mid, old_text="editable", new_text="edited", reason="typo")
    assert res["edited"] is True
    row = tools.db.get_memory(mid)
    assert row["content"] == "content of edited knowledge"
    assert int(row["version"]) >= 2  # edit bumped the content version
    assert row["kb_member"] == 1  # governance bit untouched by the edit

    # tags-only edit also works; bogus arg combos are rejected
    res2 = api.knowledge_edit(mid, add_tags=["curated"], reason="tag")
    assert res2["edited"] is True
    bad = api.knowledge_edit(mid)  # nothing to do
    assert bad["_http_status"] == 400
    half = api.knowledge_edit(mid, old_text="edited")  # old without new
    assert half["_http_status"] == 400

    # retire always requires the explicit authorization bit (the console
    # confirm dialog is that authorization); unauthenticated first, then approved
    denied = api.knowledge_retire(mid, reason="outdated")
    assert denied.get("needs_authorization") is True
    ret = api.knowledge_retire(mid, reason="outdated", authorized=True)
    assert ret["retired"] is True
    assert tools.db.get_memory(mid)["status"] == "superseded"


def test_knowledge_retire_locked_needs_authorization(tmp_path: Path) -> None:
    api = _kb_api(tmp_path)
    tools = api.tools
    mid = _write(tools, "user confirmed fact", source_type="user_confirmed")
    denied = api.knowledge_retire(mid, reason="try without auth")
    assert denied.get("needs_authorization") is True
    assert tools.db.get_memory(mid)["status"] == "active"
    approved = api.knowledge_retire(mid, reason="owner approved", authorized=True)
    assert approved["retired"] is True
    assert tools.db.get_memory(mid)["status"] == "superseded"


def test_knowledge_edit_strict_gate(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "se.sqlite3", backup_jsonl=tmp_path / "se.jsonl",
        client="t", agent_id="t", workspace="kb-ws", isolation="strict",
    )
    tools = MemoryTools(settings)
    api = ConsoleAPI(tools)
    mine = _write(tools, "mine")
    other = _write(tools, "other", workspace="other-ws")
    tools.memory_confirm_pending_workspace(
        memory_id=mine, canonical="kb-ws", reason="t", authorized=True, workspace="kb-ws",
    )
    tools.memory_confirm_pending_workspace(
        memory_id=other, canonical="other-ws", reason="t", authorized=True, workspace="other-ws",
    )
    ok = api.knowledge_edit(mine, old_text="mine", new_text="ours", workspace="kb-ws")
    assert ok["edited"] is True
    denied = api.knowledge_edit(other, old_text="other", new_text="x", workspace="kb-ws")
    assert denied["_http_status"] == 403


# ---- adversarial review R2 regression pins ----------------------------------

def test_v1_suffixed_base_url_not_poisoned(fake_llm, tmp_path: Path) -> None:
    """P0-1 pin: base_url ending in /v1 must work on EVERY call — the
    variant cache must never send it to /v1/v1/..."""
    from memory_arbiter import llm_client

    llm_client._V1_PROBE_CACHE.clear()
    port = fake_llm.server_address[1]

    class _OnlyV1Style(http.server.BaseHTTPRequestHandler):
        hits: list[str] = []

        def log_message(self, *args): pass

        def do_POST(self):  # noqa: N802
            _OnlyV1Style.hits.append(self.path)
            if self.path == "/v1/chat/completions":  # what base+"/chat/completions" resolves to
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                payload = _fake_completion("ok")
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _OnlyV1Style)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        settings = Settings(
            db_path=tmp_path / "v1s.sqlite3", backup_jsonl=tmp_path / "v1s.jsonl",
            client="t", agent_id="t", workspace="kb-ws",
            llm_base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            llm_api_key="sk-fake", llm_model="m", llm_max_context_tokens=32000,
        )
        for _ in range(3):  # every subsequent call must keep working
            assert llm_client.chat_completion(settings, [{"role": "user", "content": "hi"}])["content"] == "ok"
        assert all(h == "/v1/chat/completions" for h in _OnlyV1Style.hits)
    finally:
        server.shutdown()
        server.server_close()


def test_tree_markup_no_inline_js_interpolation() -> None:
    """P0-2 pin: knowledge-tree buttons must never interpolate library
    values inside an inline JS string — data-* attributes only."""
    from memory_arbiter.console_static import INDEX_HTML

    assert "state.kb.filter={ws:'${" not in INDEX_HTML
    assert "state.kb.filter={ws:'${esc" not in INDEX_HTML
    assert 'data-ws="${esc(w.workspace)}"' in INDEX_HTML
    assert 'data-tag="${esc(g.tag)}"' in INDEX_HTML
    assert "kbFilterWs(this)" in INDEX_HTML and "kbFilterTag(this)" in INDEX_HTML


def test_knowledge_save_duplicate_replay_passthrough(tmp_path: Path) -> None:
    """P1-5 pin: identical content re-save replays the old id, reuses it for
    the knowledge base, and says so."""
    api = _kb_api(tmp_path)
    first = api.knowledge_save(subject="note A", content="same body", tags=[], workspace="kb-ws")
    second = api.knowledge_save(subject="note B", content="same body", tags=[], workspace="kb-ws")
    assert second["duplicate_replay"] is True
    assert second["memory_id"] == first["memory_id"]
    assert second["kb_member"] is True
    assert api.tools.db.get_memory(first["memory_id"])["kb_member"] == 1


def test_timeline_accepts_material_event_time_day() -> None:
    """P1-1 pin: full-ISO stored event_time ("...T05:39:47+00:00") — the day
    part must count as a material time point, not 'fabricated'."""
    mats = [ContextMaterial(1, "s", "2026-10-10T05:39:47+00:00", "body without dates")]
    assert validate_output("timeline", "- 2026-10-10 — 事件A [1]", mats) == []


def test_quiz_validator_counts_questions_only() -> None:
    """P1-2 pin: a compliant 8-question quiz WITH numbered answers must pass
    (the old regex counted answers+headings too and false-flagged >20)."""
    questions = "\n\n".join(f"### 第{i}题\n\n题目{i}？出处 [10{i}]" for i in range(1, 9))
    answers = "\n".join(f"{i}. 答案{i}。" for i in range(1, 9))
    text = f"# 考卷\n\n{questions}\n\n## 答案\n\n{answers}"
    assert validate_output("quiz", text, []) == []


def test_kb_membership_noop_flip_not_audited(tmp_path: Path) -> None:
    """P2-2 pin: same-value flips must not add governance audit rows."""
    api = _kb_api(tmp_path)
    mid = _write(api.tools, "audit target")
    api.tools.db.memories.set_kb_member(mid, True)
    before = len(api.tools.db.audit.recent_governance_actions(limit=50))
    api.tools.db.memories.set_kb_member(mid, True)  # no-op
    api.tools.db.memories.set_kb_member(mid, True)  # no-op
    after = len(api.tools.db.audit.recent_governance_actions(limit=50))
    assert before == after


def test_deep_json_body_gets_400_not_thread_death(tmp_path: Path) -> None:
    """P2-1 pin: 100k-deep nested JSON must answer 400, not crash the
    handler thread with RecursionError."""
    api = _kb_api(tmp_path)
    server = build_http_server("127.0.0.1", 0, api=api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _, port = server.server_address
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        body = ("[" * 100000) + ("]" * 100000)
        conn.putrequest("POST", "/api/knowledge/membership")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("X-Console-Request", "1")
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders()
        conn.send(body.encode())
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


# ---- R3 盘点修复回归钉（2026-10-11 fix batch）--------------------------------


def test_knowledge_edit_retire_strict_explicit_ws_not_settings_fallback(tmp_path: Path) -> None:
    """P1 pin: with isolation=strict the inner memory_edit / memory_supersede
    ACL must receive the console-selected workspace. Before the fix the
    missing workspace fell back to settings.workspace, so an edit the shared
    equality gate had already approved was rejected with
    forbidden_strict_workspace whenever the settings bucket differed."""
    settings = Settings(
        db_path=tmp_path / "se.sqlite3", backup_jsonl=tmp_path / "se.jsonl",
        client="t", agent_id="t", workspace="other-ws", isolation="strict",
    )
    tools = MemoryTools(settings)
    api = ConsoleAPI(tools)
    mine = _write(tools, "mine")
    tools.memory_confirm_pending_workspace(
        memory_id=mine, canonical="kb-ws", reason="t", authorized=True, workspace="kb-ws",
    )
    ok = api.knowledge_edit(mine, old_text="mine", new_text="ours", workspace="kb-ws")
    assert ok["edited"] is True
    # supersede requires authorized unconditionally (two-click confirm flow)
    retired = api.knowledge_retire(mine, reason="done", workspace="kb-ws", authorized=True)
    assert retired.get("retired") is True


def test_knowledge_save_reports_membership_flip_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§B.1.4 branch pin: write success + flip failure must report
    kb_member=false with a retry hint instead of a membership that is not
    there (the old response hard-coded kb_member=True)."""
    api = _kb_api(tmp_path)
    from memory_arbiter.db.memories import MemoriesStore

    monkeypatch.setattr(MemoriesStore, "set_kb_member", lambda self, mid, member, **kw: 0)
    res = api.knowledge_save(subject="s", content="c", tags=None, workspace="kb-ws")
    assert res["saved"] is True
    assert res["kb_member"] is False
    assert "knowledge base" in res["hint_en"]
    assert "知识库" in res["hint_zh"]


def test_write_llm_config_clamps_like_from_env_and_locks_file(tmp_path: Path) -> None:
    """R3 盘点 P2 pin: the write path applies the same clamp bounds as
    from_env (one config file, one answer across reloads) and enforces 0600
    on a file that can carry an api_key."""
    cfg_path = tmp_path / "cfg.json"
    _applied, _path, normalized, warnings = write_llm_config(
        {"max_context_tokens": 10**12, "max_output_tokens": 0, "timeout_seconds": 99999},
        config_path=cfg_path,
    )
    assert normalized["max_context_tokens"] == 10_000_000
    assert normalized["max_output_tokens"] == 1
    assert normalized["timeout_seconds"] == 3600
    assert sum("clamped" in w for w in warnings) == 3
    assert (cfg_path.stat().st_mode & 0o777) == 0o600


def test_timeline_content_full_datetime_not_fabricated() -> None:
    """R3 盘点 P2 pin: a full ISO datetime inside content counts as its date —
    _ISO_RE's trailing \\b stops before the 'T', so without the explicit
    datetime pass a real date reads as fabricated and burns a retry."""
    mats = [ContextMaterial(1, "s", "", "上线时间 2026-03-01T09:30:00 完成")]
    assert validate_output("timeline", "- 2026-03-01 — 上线 [1]", mats) == []
    # the year-month prefix claim still passes (either form is grounded)
    assert validate_output("timeline", "- 2026-03 — 上线 [1]", mats) == []
    assert validate_output("timeline", "- 2030-03 — 编造 [1]", mats)


def test_console_i18n_literal_keys_all_defined() -> None:
    """R3 盘点 P2 pin: every literal t('key') / t("key") / data-i18n="key" on
    the console page must exist in the I18N dict — a missing key renders the
    key name itself to the user (the saveBtn/llmSaved class of bug)."""
    import re as _re

    from memory_arbiter.console_static import INDEX_HTML

    used = (
        set(_re.findall(r"\bt\('([^']+)'\)", INDEX_HTML))
        | set(_re.findall(r'\bt\("([^"]+)"\)', INDEX_HTML))
        | set(_re.findall(r'data-i18n="([^"]+)"', INDEX_HTML))
    )
    block_m = _re.search(r"const I18N = \{(.*?)\n\};", INDEX_HTML, _re.S)
    assert block_m, "I18N dict not found in INDEX_HTML"
    defined = set(
        _re.findall(r"(?:^|[{,\s])([A-Za-z_][A-Za-z0-9_]*)\s*:\s*'", block_m.group(1), _re.M)
    )
    missing = sorted(k for k in used if k not in defined)
    assert not missing, f"i18n keys used but not defined: {missing}"


def test_timeline_claim_side_full_datetime_day_checked() -> None:
    """R2 对抗轮 pin: the claim side must mirror the material side's full
    datetime pass — a fabricated DAY inside a T-form datetime must be caught
    (the bare-ISO form already was; without this the T-form day forgery
    passed validation unseen)."""
    mats = [ContextMaterial(1, "s", "", "上线时间 2026-03-01T09:30:00 完成")]
    assert validate_output("timeline", "- 2026-03-15T09:30:00 — 伪 [1]", mats)
    assert validate_output("timeline", "- 2026-03-01T09:30:00 — 上线 [1]", mats) == []
