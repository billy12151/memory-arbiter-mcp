# ── from test_console_api.py ──

from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.console_api import ConsoleAPI
from memory_arbiter.models import ConflictMember, ConflictValueGroup
from memory_arbiter.tools import MemoryTools


def _api(tmp_path: Path) -> ConsoleAPI:
    settings = Settings(
        db_path=tmp_path / "console.sqlite3",
        backup_jsonl=tmp_path / "console.jsonl",
        client="pytest",
        agent_id="console-test",
        workspace="console-ws",
    )
    return ConsoleAPI(MemoryTools(settings))


def test_overview_returns_counts_and_brand(tmp_path: Path) -> None:
    api = _api(tmp_path)
    api.tools.memory_write(
        content="confirmed fact",
        subject="Confirmed",
        tags=["console"],
        source_type="user_confirmed",
        workspace="console-ws",
        agent_id="test",
    )
    overview = api.overview()
    assert overview["brand"]["en"] == "mema"
    assert overview["brand"]["zh"] == "迷码"
    assert overview["counts"]["total"] == 1
    assert overview["counts"]["active"] == 1
    assert overview["support"]["repo_url"] == "https://github.com/billy12151/memory-arbiter-mcp"
    assert overview["support"]["new_issue_url"].endswith("/issues/new")


def _record_group(api: ConsoleAPI, left: int, right: int, *, point: str = "scope", extra: int | None = None) -> dict:
    workspace = api.tools.db.get_memory(left)["workspace_canonical"]
    members = [
        ConflictMember(left, 1, point, "old", point, "old", "old", (0, 3), "a" * 64, "a_to_b", "p1", "d1"),
        ConflictMember(right, 1, point, "new", point, "new", "new", (0, 3), "b" * 64, "b_to_a", "p1", "d1"),
    ]
    if extra is not None:
        members.append(ConflictMember(extra, 1, point, "new", point, "new", "new", (0, 3), "c" * 64, "a_to_b", "p1", "d1"))
    return api.tools.db.record_conflict_group(
        workspace_canonical=workspace,
        slot_key={"entity": "console", "attribute": point, "scope": "global"},
        members=members,
        value_groups=[
            ConflictValueGroup("old", "old", (f"{left}@1",)),
            ConflictValueGroup("new", "new", tuple([f"{right}@1"] + ([f"{extra}@1"] if extra is not None else []))),
        ],
        detection_reason="values differ", source="scan", detector_version="d1",
        prompt_version="p1", conflict_point=point,
    )


def test_conflict_detail_returns_group_members(tmp_path: Path) -> None:
    api = _api(tmp_path)
    left = api.tools.memory_write(
        content="old scope",
        subject="Old",
        tags=["console"],
        source_type="agent_generated",
        workspace="console-ws",
        agent_id="test",
    )["data"]["id"]
    right = api.tools.memory_write(
        content="new scope",
        subject="New",
        tags=["console"],
        source_type="user_confirmed",
        workspace="console-ws",
        agent_id="test",
    )["data"]["id"]
    conflict = _record_group(api, left, right, point="Console MVP scope changed")
    detail = api.conflict_detail(conflict["conflict_id"])
    assert detail["conflict"]["conflict_point"] == "Console MVP scope changed"
    assert [member["memory"]["id"] for member in detail["members"]] == [left, right]
    assert detail["revision"] == 1
    assert len(detail["value_groups"]) == 2


def test_conflict_detail_frontend_contract_supports_three_members(tmp_path: Path) -> None:
    api = _api(tmp_path)
    ids = [
        api.tools.memory_write(content=f"value {index}", subject=f"M{index}", workspace="console-ws")["data"]["id"]
        for index in range(3)
    ]
    conflict = _record_group(api, ids[0], ids[1], extra=ids[2])
    detail = api.conflict_detail(conflict["conflict_id"])
    assert [member["memory"]["id"] for member in detail["members"]] == ids
    assert len(detail["member_versions"]) == 3
    new_group = next(group for group in detail["value_groups"] if group["normalized_value"] == "new")
    assert new_group["members"] == [f"{ids[1]}@1", f"{ids[2]}@1"]
    assert detail["revision"] == 1
    assert detail["apply_summary"] == {"plan": []}
    assert "left" not in detail and "right" not in detail and "winner_side" not in detail


def test_conflict_detail_exposes_unified_resolution_state(tmp_path: Path) -> None:
    api = _api(tmp_path)
    left = api.tools.memory_write(
        content="old full rule", subject="Old", workspace="console-ws",
    )["data"]["id"]
    right = api.tools.memory_write(
        content="new full rule", subject="New", workspace="console-ws",
    )["data"]["id"]
    conflict = _record_group(api, left, right, point="rule")
    judged = api.tools.db.judge_conflict(
        conflict["conflict_id"], expected_revision=1, chosen_value="new",
        decided_by="user", decided_ref="console", decision_reason="confirmed",
        resolution_memory_id=right,
        apply_plan=[
            {"memory_id": left, "action": "update_current_claim"},
            {"memory_id": right, "action": "use_as_resolution"},
        ],
    )

    detail = api.conflict_detail(conflict["conflict_id"])
    assert detail["conflict"]["status"] == "applying"
    assert detail["revision"] == judged["revision"]
    assert detail["resolution_memory"]["memory"]["id"] == right
    assert detail["next_executable_call"]["action"] == "apply_conflict_action"


def test_memories_expired_invalid_offset_defaults_to_zero(tmp_path: Path) -> None:
    api = _api(tmp_path)
    result = api.memories(status="expired", offset="abc")
    assert result["status"] == "expired"
    assert result["items"] == []


def test_memories_strict_without_query_requires_explicit_workspace(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "console.sqlite3",
        backup_jsonl=tmp_path / "console.jsonl",
        client="pytest",
        agent_id="console-test",
        workspace="console-ws",
        isolation="strict",
    )
    api = ConsoleAPI(MemoryTools(settings))
    result = api.memories(status="active")
    assert "error" in result
    assert result["_http_status"] == 400

    ok = api.memories(status="active", workspace="console-ws")
    assert "error" not in ok
    assert ok["workspace_source"] == "explicit"


def test_settings_view_handles_missing_config_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.delenv("MEMORY_ARBITER_CONFIG", raising=False)
    api = _api(tmp_path)
    view = api.settings_view()
    assert view["config_file"]["path"] is None
    assert view["config_file"]["exists"] is False


def test_memories_rejects_unknown_status(tmp_path: Path) -> None:
    api = _api(tmp_path)
    result = api.memories(status="deleted")
    assert result["error"] == "status must be active or expired"
    assert result["_http_status"] == 400


def test_memories_supports_offset_for_empty_query_browse(tmp_path: Path) -> None:
    """Empty query + offset now paginates the recency browse (was a 400 error
    when browse went through memory_search, which didn't support offset)."""
    api = _api(tmp_path)
    # write enough memories to have a second page
    for i in range(5):
        api.tools.memory_write(
            content=f"memory {i}", workspace="w", source_type="agent_generated", subject=f"test-{i}")
    page1 = api.memories(status="active", limit=2, offset=0)
    assert "error" not in page1
    assert len(page1["items"]) == 2
    page2 = api.memories(status="active", limit=2, offset=2)
    assert "error" not in page2
    assert len(page2["items"]) == 2
    # pages don't overlap
    assert {m["id"] for m in page1["items"]}.isdisjoint({m["id"] for m in page2["items"]})


def test_memories_supports_offset_for_active_search(tmp_path: Path) -> None:
    """Active search with a query accepts offset as a best-effort window:
    pages are non-overlapping, but v0.15.4 removed the total/has_more signal
    on unfiltered query-recall — total falls back to the page size and
    has_more is always False (reword the query instead of deep paging)."""
    api = _api(tmp_path)
    for i in range(5):
        api.tools.memory_write(
            content=f"alpha memory {i}", workspace="w",
            source_type="agent_generated", subject=f"search-{i}")
    page1 = api.memories(query="alpha", status="active", limit=2, offset=0)
    assert "error" not in page1
    assert len(page1["items"]) == 2
    assert page1["has_more"] is False
    # total_estimate is None on this path → console shows the page size and
    # never marks it precise.
    assert page1["total"] == 2
    assert page1["total_precise"] is False
    page2 = api.memories(query="alpha", status="active", limit=2, offset=2)
    assert "error" not in page2
    assert {m["id"] for m in page1["items"]}.isdisjoint({m["id"] for m in page2["items"]})
    # last window (offset=4, limit=2, 5 matches): 1 item
    page3 = api.memories(query="alpha", status="active", limit=2, offset=4)
    assert "error" not in page3
    assert len(page3["items"]) == 1
    assert page3["has_more"] is False
    # Console is the human-facing channel: items keep full content even though
    # find defaults to the index-page preview.
    assert page1["items"][0].get("content")


def test_active_search_unfiltered_reports_page_size_total(tmp_path: Path) -> None:
    """v0.15.4: unfiltered query-recall no longer reports a drifting
    pool-based total. total_estimate is None, the console falls back to the
    page size, and has_more stays False no matter how deep the offset goes.
    Pins the semantic so a future engine change is caught explicitly."""
    api = _api(tmp_path)
    for i in range(60):
        api.tools.memory_write(
            content=f"drift memory {i}", workspace="w",
            source_type="agent_generated", subject=f"drift-{i}")
    p1 = api.memories(query="drift", status="active", limit=10, offset=0)
    assert "error" not in p1
    assert p1["total"] == 10  # page-size fallback, not a pool count
    assert p1["total_precise"] is False
    assert p1["has_more"] is False
    # deep window still returns items (offset paging itself still works)
    p_deep = api.memories(query="drift", status="active", limit=10, offset=50)
    assert "error" not in p_deep
    assert len(p_deep["items"]) == 10
    assert p_deep["total"] == 10
    assert p_deep["has_more"] is False


def test_recent_browse_count_is_total_not_page_size(tmp_path: Path) -> None:
    """_recent_browse (empty query) returns count=total, not len(items). The
    console UI drives paging off has_more, but count being total is what makes
    'X total memories' correct. A prior bug returned len(items) (commit 6332766
    fixed it for browse); this locks it in. D1 also exposes `total` + `total_precise`
    so the UI can show 共 N 条 and a jump-to-page input."""
    api = _api(tmp_path)
    for i in range(7):
        api.tools.memory_write(
            content=f"memory {i}", workspace="w",
            source_type="agent_generated", subject=f"cnt-{i}")
    page = api.memories(status="active", limit=3, offset=0)
    assert "error" not in page
    assert len(page["items"]) == 3
    assert page["count"] == 7  # total, not 3
    assert page["total"] == 7
    assert page["total_precise"] is True  # empty-query browse is exact
    assert page["has_more"] is True
    # last page
    last = api.memories(status="active", limit=3, offset=6)
    assert len(last["items"]) == 1
    assert last["count"] == 7
    assert last["total"] == 7
    assert last["total_precise"] is True
    assert last["has_more"] is False


def test_recent_browse_explicit_workspace_filters_in_none_mode(tmp_path: Path) -> None:
    api = _api(tmp_path)
    alpha = api.tools.memory_write(
        content="alpha", subject="alpha", workspace="alpha",
    )["data"]["id"]
    api.tools.memory_write(content="beta", subject="beta", workspace="beta")

    result = api.memories(status="active", workspace="alpha")

    assert [item["id"] for item in result["items"]] == [alpha]
    assert result["total"] == 1


def test_overview_explicit_workspace_filters_counts_in_none_mode(tmp_path: Path) -> None:
    api = _api(tmp_path)
    api.tools.memory_write(content="alpha", subject="alpha", workspace="alpha")
    api.tools.memory_write(content="beta", subject="beta", workspace="beta")

    result = api.overview(workspace="alpha")

    assert result["counts"]["total"] == 1
    assert result["counts"]["active"] == 1
    assert result["by_workspace"] == {"alpha": 1}


def test_explicit_workspace_remains_soft_in_weak_console_mode(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "console.sqlite3",
        backup_jsonl=tmp_path / "console.jsonl",
        client="pytest", agent_id="console-test", workspace="default",
        isolation="weak",
    )
    api = ConsoleAPI(MemoryTools(settings))
    api.tools.memory_write(content="alpha", subject="alpha", workspace="alpha")
    api.tools.memory_write(content="beta", subject="beta", workspace="beta")

    browse = api.memories(status="active", workspace="alpha")
    overview = api.overview(workspace="alpha")

    assert browse["total"] == 2
    assert overview["counts"]["total"] == 2
    assert set(overview["by_workspace"]) == {"alpha", "beta"}


def test_console_conflicts_respect_explicit_workspace_in_none_mode(tmp_path: Path) -> None:
    api = _api(tmp_path)
    alpha_ids = [
        api.tools.memory_write(content=f"alpha {i}", subject=f"a{i}", workspace="alpha")["data"]["id"]
        for i in range(2)
    ]
    beta_ids = [
        api.tools.memory_write(content=f"beta {i}", subject=f"b{i}", workspace="beta")["data"]["id"]
        for i in range(2)
    ]
    alpha_conflict = _record_group(api, *alpha_ids, point="alpha")
    beta_conflict = _record_group(api, *beta_ids, point="beta")

    listed = api.conflicts(workspace="alpha")

    assert [item["id"] for item in listed["items"]] == [alpha_conflict["conflict_id"]]
    assert api.conflict_detail(alpha_conflict["conflict_id"], workspace="alpha")["conflict"]["id"] == alpha_conflict["conflict_id"]
    hidden = api.conflict_detail(beta_conflict["conflict_id"], workspace="alpha")
    assert hidden["_http_status"] == 404


def test_expired_search_without_query_is_precise_total(tmp_path: Path) -> None:
    """Expired search without a query is exact (SQL COUNT) → total_precise True;
    with a query it is best-effort → total_precise False. Drives the UI's 共/约 label."""
    api = _api(tmp_path)
    # No need to populate expired rows — the precision flag is set by the
    # engine's pagination_precision field, not by result count. Empty expired
    # browse still reports the field shape the UI relies on.
    page = api.memories(status="expired", limit=10, offset=0)
    assert "error" not in page
    assert "total" in page
    assert "total_precise" in page
    # no query → pagination_precision="exact" → precise
    assert page["total_precise"] is True


def test_expired_search_with_query_is_best_effort_total(tmp_path: Path) -> None:
    """T4: expired search WITH a query goes through query-recall →
    pagination_precision="best_effort" → total_precise False. Pins the branch
    that test_expired_search_without_query_is_precise_total deliberately skips."""
    api = _api(tmp_path)
    # fixture: write then supersede so there's an expired row with content
    w = api.tools.memory_write(
        content="expired content about foo", workspace="w",
        source_type="agent_generated", subject="exp-foo")
    mid = w["data"]["id"]
    api.tools.memory_supersede(memory_id=mid, authorized=True, reason="test")
    page = api.memories(query="foo", status="expired", limit=10, offset=0)
    assert "error" not in page
    assert page["total_precise"] is False
    assert "total" in page


def test_settings_view_exposes_isolation(tmp_path: Path) -> None:
    api = _api(tmp_path)
    view = api.settings_view()
    isolation = next(
        item for group in view["groups"] for item in group["items"] if item["path"] == "isolation"
    )
    assert isolation["current"] == "none"
    assert isolation["label_zh"] == "工作区隔离等级"


def test_settings_view_is_read_only_and_bilingual(tmp_path: Path) -> None:
    api = _api(tmp_path)
    view = api.settings_view()
    assert view["read_only"] is True
    items = [item for group in view["groups"] for item in group["items"]]
    assert items
    assert all(item["editable"] is False for item in items)
    assert all(item["label_en"] and item["label_zh"] for item in items)


def test_overview_counts_and_doctor_agree_on_unresolved_conflicts(tmp_path: Path) -> None:
    from memory_arbiter.doctor import run_all_checks

    api = _api(tmp_path)

    def _memory(content: str) -> int:
        return api.tools.memory_write(
            content=content, subject="console-conflicts", tags=["console"],
            source_type="agent_generated", workspace="console-ws", agent_id="test",
        )["data"]["id"]

    open_left, open_right = _memory("cache backend is redis"), _memory("cache backend is memcached")
    open_group = _record_group(api, open_left, open_right, point="cache backend")
    assert open_group["conflict_id"]

    apply_left, apply_right = _memory("console port is 8080"), _memory("console port is 9090")
    applying_group = _record_group(api, apply_left, apply_right, point="console port")
    judged = api.tools.db.judge_conflict(
        applying_group["conflict_id"], expected_revision=1, chosen_value="new",
        decided_by="agent", decided_ref=None, decision_reason="reviewed",
        apply_plan=[
            {"memory_id": apply_left, "action": "update_current_claim"},
            {"memory_id": apply_right, "action": "use_as_resolution"},
        ],
        resolution_memory_id=apply_right,
    )
    assert judged["outcome"] == "applying", judged

    # Console 口径 == doctor 口径: unresolved = open + applying.
    counts = api._status_counts()
    assert counts["open_conflicts"] == 2
    assert counts["applying_conflicts"] == 1
    overview = api.overview()
    assert overview["counts"]["open_conflicts"] == 2
    assert overview["counts"]["applying_conflicts"] == 1

    with api.tools.db.connection() as conn:
        report = run_all_checks(conn, api.tools.settings)
    backlog = next(f for f in report.findings if f.check_id == "conflicts.backlog")
    assert backlog.evidence == {"open": 1, "applying": 1}


# ── from test_console_server.py ──


import json
import threading
import urllib.error
import urllib.request

from memory_arbiter.console_server import build_http_server


class DummyAPI:
    def health(self):
        return {"ok": True}

    def conflict_detail(self, conflict_id: int):
        return {"error": f"conflict id {conflict_id} not found", "_http_status": 404}

    def memory_detail(self, memory_id: int, sections: str = "catalog"):
        return {"memory": {"id": memory_id}, "sections": sections}

    def memories(self, **kwargs):
        return {"error": "strict isolation requires workspace", "_http_status": 400}


class ExplodingAPI(DummyAPI):
    def health(self):
        raise RuntimeError("secret /Users/example/private.sqlite3")


def test_console_server_rejects_non_localhost() -> None:
    try:
        build_http_server("0.0.0.0", 8766)
    except ValueError as exc:
        assert "local-only" in str(exc)
    else:
        raise AssertionError("expected local-only host rejection")


def test_console_server_rejects_ipv6_until_supported() -> None:
    try:
        build_http_server("::1", 8766)
    except ValueError as exc:
        assert "local-only" in str(exc)
    else:
        raise AssertionError("expected unsupported IPv6 host rejection")


def test_console_server_builds_on_localhost(tmp_path) -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    try:
        host, port = server.server_address
        assert host == "127.0.0.1"
        assert isinstance(port, int)
    finally:
        server.server_close()


def test_console_server_rejects_untrusted_host_header() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/health",
            headers={"Host": f"evil.example:{port}"},
        )
        try:
            urllib.request.urlopen(request, timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "forbidden host"
        else:
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown()
        server.server_close()


def test_console_server_maps_list_endpoint_errors_to_400() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/memories", timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "strict isolation requires workspace"
        else:
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown()
        server.server_close()


def test_console_server_returns_400_for_bad_path_id() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/conflicts/not-an-id", timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "conflict id must be an integer"
        else:
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown()
        server.server_close()


def test_console_server_returns_404_for_missing_conflict() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/conflicts/123", timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "conflict id 123 not found"
        else:
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown()
        server.server_close()


def test_console_server_handles_head_and_options() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        head = urllib.request.Request(f"http://127.0.0.1:{port}/api/health", method="HEAD")
        with urllib.request.urlopen(head, timeout=2) as resp:
            assert resp.code == 200
            assert int(resp.headers.get("Content-Length", "0")) > 0
            assert resp.read() == b""
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
            assert resp.headers["X-Frame-Options"] == "DENY"
            assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]
        options = urllib.request.Request(f"http://127.0.0.1:{port}/api/health", method="OPTIONS")
        with urllib.request.urlopen(options, timeout=2) as resp:
            assert resp.code == 204
            assert "GET" in (resp.headers.get("Allow") or "")
    finally:
        server.shutdown()
        server.server_close()


def test_console_server_does_not_echo_internal_exception_details() -> None:
    server = build_http_server("127.0.0.1", 0, api=ExplodingAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 500
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload == {"error": "internal server error"}
            assert "private.sqlite3" not in json.dumps(payload)
        else:
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown()
        server.server_close()


# ── from test_console_static.py ──


import shutil
import subprocess

from memory_arbiter.console_static import INDEX_HTML


def test_console_static_has_sidebar_language_and_branding() -> None:
    assert "sidebarNav" in INDEX_HTML
    assert "langZh" in INDEX_HTML
    assert "langEn" in INDEX_HTML
    assert "mema Console" in INDEX_HTML
    assert "迷码" in INDEX_HTML
    assert "#/settings" in INDEX_HTML
    assert "loadMemories().catch" in INDEX_HTML
    assert "catch(e)" in INDEX_HTML


def test_console_static_conflict_detail_uses_one_to_many_contract() -> None:
    for field in ("d.members", "d.member_versions", "d.value_groups", "d.revision", "d.apply_summary"):
        assert field in INDEX_HTML
    assert "conflictDecision(c)" in INDEX_HTML
    assert "conflictMemberCards(d)" in INDEX_HTML
    assert "d.left" not in INDEX_HTML
    assert "d.right" not in INDEX_HTML
    assert "winner_side" not in INDEX_HTML


def test_console_static_shows_resolution_guidance_without_retired_fields() -> None:
    assert "resolutionActionText" in INDEX_HTML
    assert "user authorization is still required" in INDEX_HTML
    assert "resolution_kind" not in INDEX_HTML
    assert "judgment_" not in INDEX_HTML


def test_console_static_shows_support_panel_without_github_api() -> None:
    assert "Support mema" in INDEX_HTML
    assert "支持迷码" in INDEX_HTML
    assert "Star on GitHub" in INDEX_HTML
    assert "Request feature" in INDEX_HTML
    assert "Report bug" in INDEX_HTML
    assert "UX feedback" in INDEX_HTML
    assert "体验反馈" in INDEX_HTML
    assert "openFeedback('ux_feedback')" in INDEX_HTML
    assert "['bug','feature','ux_feedback'].includes(type)" in INDEX_HTML
    assert "buildIssueUrl" in INDEX_HTML
    assert "encodeURIComponent" in INDEX_HTML or "URLSearchParams" in INDEX_HTML
    assert "/issues/new" in INDEX_HTML
    assert "Console does not upload your memory automatically" in INDEX_HTML
    forbidden = ["github_token", "oauth", "device flow", "api.github.com", "fetch(supportUrls"]
    lower = INDEX_HTML.lower()
    for word in forbidden:
        assert word.lower() not in lower
    assert "fetch(supporturls" not in lower
    assert "fetch(buildissueurl" not in lower


def test_console_static_javascript_parses(tmp_path) -> None:
    node = shutil.which("node")
    if not node:
        return
    start = INDEX_HTML.index("<script>") + len("<script>")
    end = INDEX_HTML.index("</script>")
    script = INDEX_HTML[start:end]
    script_path = tmp_path / "console.js"
    script_path.write_text(script, encoding="utf-8")
    subprocess.run([node, "--check", str(script_path)], check=True)


def test_console_static_does_not_offer_write_actions() -> None:
    forbidden = ["memory_write", "memory_supersede", "memory_confirm", "memory_resolve_conflict"]
    lower = INDEX_HTML.lower()
    for word in forbidden:
        assert word not in lower


def test_pagination_functions_exist_and_bind_correctly() -> None:
    """T3: pagination JS functions and event bindings must exist in the served
    HTML. This locks the wiring that a node --check syntax parse cannot catch —
    e.g. a misspelled handler name or a broken onkeydown attribute would parse
    fine as a string but silently break the UI."""
    # functions defined at top level
    for fn in ("function memPrev(", "function memNext(", "function memJump(",
               "function commitFilters("):
        assert fn in INDEX_HTML, f"missing: {fn}"
    # memJump Enter binding
    assert 'onkeydown="if(event.key===' in INDEX_HTML
    assert "memJump(parseInt(this.value,10)||1)" in INDEX_HTML
    # jump input disabled when totalPages<=1 (jumpDisabled flag drives it)
    assert "jumpDisabled" in INDEX_HTML
    assert "totalPages<=1" in INDEX_HTML
    # pagination state initialized with the default page size constant
    assert "DEFAULT_PAGE_SIZE" in INDEX_HTML
    assert "memPage: {" in INDEX_HTML
    # request sequence guard against stale responses (M3 race fix)
    assert "memReqSeq" in INDEX_HTML
