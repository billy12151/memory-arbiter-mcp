"""0.16.0 commit 9 tests (§6⑮): the 32-tag persisted-total cap on remember
and update, remove+add legality in one call, unrestricted query inputs, and
the doctor tags.over_limit finding for pre-cap stock."""
from __future__ import annotations

from pathlib import Path

from test_scan_pipeline import make_tools, _write


def test_remember_rejects_more_than_32_tags(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    res = tools.memory_write(
        content="超限内容", subject="超限主题", tags=[f"t{i}" for i in range(33)],
        source_type="agent_generated",
    )
    assert not res.get("ok"), res
    assert "32" in str(res["data"])
    # exactly 32 is fine
    res = tools.memory_write(
        content="达标内容", subject="达标主题", tags=[f"t{i}" for i in range(32)],
        source_type="agent_generated",
    )
    assert res.get("ok"), res


def test_update_add_tags_over_cap_rejected_whole_call(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "追加主题", "追加正文", tags=[f"t{i}" for i in range(30)])
    # +3 would hit 33 → the whole update is refused (content edit included)
    res = tools.memory("update", {
        "memory_id": mid, "tags_only": True, "add_tags": ["a1", "a2", "a3"], "reason": "over",
    })
    assert not res.get("ok"), res
    assert "remove_tags" in str(res["data"])
    assert res["data"].get("current_total") == 33
    record = tools.db.get_memory(mid)
    assert len(record["tags"]) == 30, "拒绝必须整笔生效，不得部分落库"
    # remove+add in ONE call is legal (cap applies to the merged result)
    res2 = tools.memory("update", {
        "memory_id": mid,
        "tags_only": True,
        "remove_tags": ["t0", "t1", "t2"],
        "add_tags": ["a1", "a2", "a3"],
        "reason": "swap",
    })
    assert res2.get("ok"), res2
    assert len(tools.db.get_memory(mid)["tags"]) == 30


def test_tags_only_path_over_cap_rejected(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "tags主题", "tags正文", tags=[f"t{i}" for i in range(32)])
    res = tools.memory("update", {
        "memory_id": mid, "tags_only": True, "add_tags": ["one-more"], "reason": "over",
    })
    assert not res.get("ok"), res
    assert res["data"].get("cap") == 32


def test_tags_filter_query_inputs_unrestricted(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "查询主题", "查询正文", tags=["目标标签"])
    # 100-tag query input stays legal (call-level cap unchanged)
    res = tools.memory("find", {"tags_filter": [f"q{i}" for i in range(100)]})
    assert res.get("ok"), res
    res = tools.memory("find", {"tags_filter": [f"q{i}" for i in range(101)]})
    assert not res.get("ok"), "单次调用 MAX_TAGS=100 上限保留"


def test_doctor_reports_over_cap_stock(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "存量超限", "存量正文", tags=[f"t{i}" for i in range(32)])
    # Simulate pre-0.16.0 stock (40 tags) directly — writes are now capped.
    import json as _json

    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET tags=? WHERE id=?",
            (_json.dumps([f"t{i}" for i in range(40)]), mid),
        )
        # The out-of-band UPDATE simulates a pre-0.16.0 legacy row; rebuild
        # FTS so the simulated row is internally consistent (real legacy
        # rows always were — their writes went through the synced API).
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
    report = tools.memory_doctor_overview(deep=False)
    payload = report.get("data") or report
    findings = {f["check_id"]: f for f in payload["findings"]}
    finding = findings.get("tags.over_limit")
    assert finding is not None
    assert finding["status"] == "warn"
    assert any(item["id"] == mid and item["count"] == 40 for item in finding["evidence"]["over_limit"])
    # Stock reads stay unaffected (no retro truncation)
    assert len(tools.db.get_memory(mid)["tags"]) == 40
    # over-cap stock stays TRIMMABLE: a pure remove passes even though the
    # merged total (39) is still over the cap
    trim = tools.memory("update", {
        "memory_id": mid, "tags_only": True, "remove_tags": ["t0"], "reason": "trim stock",
    })
    assert trim.get("ok"), trim
    assert len(tools.db.get_memory(mid)["tags"]) == 39
