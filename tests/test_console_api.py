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
    members = [
        ConflictMember(left, 1, point, "old", point, "old", "old", (0, 3), "a" * 64, "a_to_b", "p1", "d1"),
        ConflictMember(right, 1, point, "new", point, "new", "new", (0, 3), "b" * 64, "b_to_a", "p1", "d1"),
    ]
    if extra is not None:
        members.append(ConflictMember(extra, 1, point, "new", point, "new", "new", (0, 3), "c" * 64, "a_to_b", "p1", "d1"))
    return api.tools.db.record_conflict_group(
        workspace_canonical="console-ws",
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
    """Active search with a query now accepts offset (was a 400 error pre-A2).
    query-recall offset is best-effort, but for a small fixture the pages must
    be non-overlapping and has_more must flip on the last page."""
    api = _api(tmp_path)
    for i in range(5):
        api.tools.memory_write(
            content=f"alpha memory {i}", workspace="w",
            source_type="agent_generated", subject=f"search-{i}")
    page1 = api.memories(query="alpha", status="active", limit=2, offset=0)
    assert "error" not in page1
    assert len(page1["items"]) == 2
    assert page1["has_more"] is True
    # active search exposes total (from total_estimate) and marks it imprecise
    # (query-recall). The UI shows "约 N 条" instead of "共 N 条".
    assert page1["total"] >= 5
    assert page1["total_precise"] is False
    page2 = api.memories(query="alpha", status="active", limit=2, offset=2)
    assert "error" not in page2
    assert {m["id"] for m in page1["items"]}.isdisjoint({m["id"] for m in page2["items"]})
    # last page (offset=4, limit=2, 5 total): 1 item, has_more False
    page3 = api.memories(query="alpha", status="active", limit=2, offset=4)
    assert "error" not in page3
    assert len(page3["items"]) == 1
    assert page3["has_more"] is False


def test_active_search_total_drifts_across_pages_beyond_pool_cap(tmp_path: Path) -> None:
    """T1: active search's total is a best-effort estimate (query-recall pool).
    When the match count exceeds the pool cap (~50), total grows as offset
    deepens because pool_cap = max(50, offset+limit+1). This test pins that
    semantic so a future engine change (e.g. exact total for query-recall) is
    caught explicitly, not silently."""
    api = _api(tmp_path)
    # 60 matching memories — beyond the default pool cap of 50
    for i in range(60):
        api.tools.memory_write(
            content=f"drift memory {i}", workspace="w",
            source_type="agent_generated", subject=f"drift-{i}")
    # page 1: pool capped at 50 → total ~50
    p1 = api.memories(query="drift", status="active", limit=10, offset=0)
    assert "error" not in p1
    total_p1 = p1["total"]
    assert total_p1 <= 51  # pool cap + 1
    # deep page: offset=50 → pool_cap = 50+10+1 = 61 → total grows to 60
    p_deep = api.memories(query="drift", status="active", limit=10, offset=50)
    assert "error" not in p_deep
    total_deep = p_deep["total"]
    # total must have grown past the initial pool-capped estimate
    assert total_deep > total_p1
    assert total_deep == 60  # all matches now fit the widened pool
    # deep page has items (offset 50 < 60 matches)
    assert len(p_deep["items"]) == 10
    assert p_deep["has_more"] is False  # offset 50 + 10 = 60 = total


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
