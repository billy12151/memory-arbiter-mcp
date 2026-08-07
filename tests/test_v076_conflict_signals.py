"""v0.7.6 tests: conflict signals (search), write_hints, record_conflict refresh.

Covers:
  A. search conflict_signal — open_table attachment + runtime_metadata_hint
  B. write_hints — possible_duplicate / possible_evolution_of
  C. record_conflict refresh mode
  D. evolution conflict_type field pass-through
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def _tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "b.jsonl",
        client="test",
        agent_id="tester",
        workspace="ws",
        enable_sqlite_vec=False,
        update_check_enabled=False,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write(
    tools: MemoryTools,
    *,
    content: str,
    subject: str,
    tags: list[str] | None = None,
    workspace: str = "ws",
    source_type: str = "agent_generated",
    protection_level: str = "normal",
    ingest_time: str | None = None,
) -> int:
    payload: dict[str, Any] = {
        "content": content,
        "subject": subject,
        "tags": tags or [],
        "workspace": workspace,
        "source_type": source_type,
        "protection_level": protection_level,
        "agent_id": "tester",
    }
    if ingest_time is not None:
        payload["ingest_time"] = ingest_time
    res = tools.memory_write(**payload)
    assert res["ok"], f"write failed: {res}"
    return res["data"]["id"]


# ──────────────────────────────────────────────────────────────────────────
#  A. search conflict_signal
# ──────────────────────────────────────────────────────────────────────────

def test_search_attaches_open_table_conflict_signal(tmp_path: Path) -> None:
    """A search result that is a side of an open conflict gets conflict_signal."""
    tools = _tools(tmp_path)
    a = _write(tools, content="revenue is 20%", subject="revenue", tags=["finance"])
    b = _write(tools, content="revenue is 30%", subject="revenue-v2", tags=["finance"])
    # Record a conflict a vs b, with b as the suggested winner.
    tools.memory_record_conflict(
        left_id=a, right_id=b, reason="contradiction",
        conflict_type="contradiction", conflict_point="20% vs 30%",
        suggested_winner=b, confidence_hint="high", source="llm_informed",
    )
    res = tools.memory_search(query="revenue")
    results = res["data"]["results"]
    result_a = next(r for r in results if r["id"] == a)
    assert "conflict_signal" in result_a
    sig = result_a["conflict_signal"]
    assert sig["conflict_source"] == "open_table"
    assert sig["conflict_point"] == "20% vs 30%"
    assert sig["suggested_winner"] == b
    assert sig["confidence_hint"] == "high"
    assert sig["conflict_peer"]["id"] == b


def test_search_conflict_signal_peer_outside_limit(tmp_path: Path) -> None:
    """limit=1 returns only one result, but the peer summary is still attached."""
    tools = _tools(tmp_path)
    a = _write(tools, content="alpha detail about config", subject="config", tags=["cfg"])
    b = _write(tools, content="beta detail about config", subject="config-v2", tags=["cfg"])
    tools.memory_record_conflict(
        left_id=a, right_id=b, reason="conflict",
        conflict_type="contradiction", conflict_point="config diff",
        suggested_winner=b, confidence_hint="medium", source="llm_informed",
    )
    res = tools.memory_search(query="config", limit=1)
    results = res["data"]["results"]
    assert len(results) == 1
    # The single result should have a conflict_signal with a peer summary.
    r = results[0]
    assert "conflict_signal" in r
    assert r["conflict_signal"]["conflict_peer"] is not None
    assert "id" in r["conflict_signal"]["conflict_peer"]


def test_search_attention_flag_when_conflict_hits(tmp_path: Path) -> None:
    """v0.8.7: a direct hit carrying conflict_signal promotes a top-level
    attention_required flag + summary (mirrors the write path)."""
    tools = _tools(tmp_path)
    a = _write(tools, content="revenue is 20%", subject="revenue", tags=["finance"])
    b = _write(tools, content="revenue is 30%", subject="revenue-v2", tags=["finance"])
    tools.memory_record_conflict(
        left_id=a, right_id=b, reason="contradiction",
        conflict_type="contradiction", conflict_point="20% vs 30%",
        suggested_winner=b, confidence_hint="high", source="llm_informed",
    )
    res = tools.memory_search(query="revenue")
    assert res["data"].get("attention_required") is True
    summary = res["data"].get("attention_summary")
    assert isinstance(summary, str)
    assert "signal" in summary
    # Summary names the first hit and its peer.
    assert f"#{a}" in summary or f"#{b}" in summary

    # A no-conflict search on the same store must NOT set the flag.
    res2 = tools.memory_search(query="finance")
    if not any("conflict_signal" in r for r in res2["data"]["results"]):
        assert not res2["data"].get("attention_required")


def test_search_no_attention_flag_when_no_conflict(tmp_path: Path) -> None:
    """v0.8.7: a search with no conflict_signal on any hit has no flag."""
    tools = _tools(tmp_path)
    _write(tools, content="harmless note about cats", subject="cats", tags=["animal"])
    _write(tools, content="another note about dogs", subject="dogs", tags=["animal"])
    res = tools.memory_search(query="cats")
    # No conflicts recorded → no signal → no flag.
    assert not any("conflict_signal" in r for r in res["data"]["results"])
    assert "attention_required" not in res["data"]
    assert "attention_summary" not in res["data"]


def test_attention_events_logged_and_doctor_counts(tmp_path: Path) -> None:
    """v0.8.8: firing attention_required (write + search paths) appends to
    attention_log.jsonl, and the doctor volume check reads it back."""
    import json as _json
    from memory_arbiter.doctor import _check_attention_volume

    tools = _tools(tmp_path)
    log_path = tools.settings.db_path.parent / "attention_log.jsonl"

    # Write path: a near-duplicate fires attention_required.
    _write(tools, content="api token policy v1",
           subject="api-token-policy", tags=["policy", "security"])
    tools.memory_write(
        content="api token policy v2", subject="api-token-policy",
        tags=["policy", "security"], workspace="ws",
        source_type="agent_generated", agent_id="tester",
    )

    # Search path: a recorded conflict + search fires attention_required.
    a = _write(tools, content="revenue is 20%", subject="revenue", tags=["finance"])
    b = _write(tools, content="revenue is 30%", subject="revenue-v2", tags=["finance"])
    tools.memory_record_conflict(
        left_id=a, right_id=b, reason="contradiction",
        conflict_type="contradiction", conflict_point="20% vs 30%",
        suggested_winner=b, confidence_hint="high", source="llm_informed",
    )
    tools.memory_search(query="revenue")

    # Both event types should be in the log.
    events = [_json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    triggers = {e["trigger"] for e in events}
    assert "write" in triggers
    assert "search" in triggers
    search_ev = next(e for e in events if e["trigger"] == "search")
    assert search_ev["source"] == "open_table"

    # Doctor reports the volume by source.
    finding = _check_attention_volume(tools.settings)
    assert finding.check_id == "capacity.attention_volume"
    assert finding.evidence["total"] >= 2
    assert "open_table" in finding.evidence["by_source"]
    # v0.8.8 review: trigger dimension is reported (write event vs search re-hit).
    assert finding.evidence["by_trigger"]["write"] >= 1
    assert finding.evidence["by_trigger"]["search"] >= 1
    assert finding.evidence["by_source_trigger"]["open_table"]["search"] >= 1


def test_search_conflict_signal_disabled(tmp_path: Path) -> None:
    """include_conflict_signal=False suppresses the field entirely."""
    tools = _tools(tmp_path)
    a = _write(tools, content="alpha", subject="subj", tags=["tag1"])
    b = _write(tools, content="beta", subject="subj-v2", tags=["tag1"])
    tools.memory_record_conflict(left_id=a, right_id=b, reason="r")
    res = tools.memory_search(query="subj", include_conflict_signal=False)
    for r in res["data"]["results"]:
        assert "conflict_signal" not in r
    # v0.8.8: disabling conflict_signal also suppresses the attention log.
    assert not (tools.settings.db_path.parent / "attention_log.jsonl").exists()


def test_search_conflict_signal_not_on_fallback(tmp_path: Path) -> None:
    """conflict_signal only fires on direct mode, not recent_fallback."""
    tools = _tools(tmp_path)
    a = _write(tools, content="alpha", subject="subj", tags=["tag1"])
    b = _write(tools, content="beta", subject="subj-v2", tags=["tag1"])
    tools.memory_record_conflict(left_id=a, right_id=b, reason="r")
    # Query that matches nothing → recent_fallback, no conflict_signal.
    res = tools.memory_search(query="zzz_no_match_xyz")
    assert res["data"]["retrieval_mode"] == "recent_fallback"
    for r in res["data"]["results"]:
        assert "conflict_signal" not in r


def test_search_runtime_metadata_hint_is_distinct_and_advisory(tmp_path: Path) -> None:
    """v0.8.8: high metadata overlap yields an advisory runtime_metadata_hint.

    With write落表, the overlapping write records a metadata_write_hint row,
    which surfaces as runtime_metadata_hint (advisory, NOT open_table) — and,
    unlike the old transient computed hint, CARRIES conflict_id (dismissable).
    """
    tools = _tools(tmp_path)
    low_id = _write(
        tools,
        content="runtime overlap draft implementation notes",
        subject="runtime overlap design",
        tags=["runtime-hint", "overlap", "review"],
        source_type="agent_generated",
        protection_level="normal",
    )
    high_id = _write(
        tools,
        content="runtime overlap authoritative notes",
        subject="runtime overlap design confirmed",
        tags=["runtime-hint", "overlap", "review"],
        source_type="user_confirmed",
        protection_level="locked",
    )

    res = tools.memory_search(query="runtime overlap", include_conflict_signal=True)
    assert res["data"]["retrieval_mode"] == "direct"
    results = res["data"]["results"]
    hinted = [r for r in results if r.get("conflict_signal", {}).get("conflict_source") == "runtime_metadata_hint"]
    assert hinted, results
    sig = hinted[0]["conflict_signal"]
    assert sig["confidence_hint"] == "low"
    assert sig["conflict_type"] == "metadata_overlap"
    assert "conflict_id" in sig  # v0.8.8: write-hint row is dismissable
    assert sig.get("source") == "metadata_write_hint"
    assert sig["conflict_peer"]["id"] in {low_id, high_id}
    _opens = tools.memory_list_conflicts(status="open")["data"]["conflicts"]
    assert len(_opens) == 1 and _opens[0]["source"] == "metadata_write_hint"
    # v0.8.8: advisory runtime hints do NOT ring the loud must-surface flag
    # (they'd nag on every retrieval) — the agent judges them by content. But
    # they ARE logged for the doctor volume stat.
    assert not res["data"].get("attention_required")
    import json as _j
    _log = tools.settings.db_path.parent / "attention_log.jsonl"
    _ev = [_j.loads(l) for l in _log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert any(e["trigger"] == "search" and e["source"] == "runtime_metadata_hint" for e in _ev)


# ──────────────────────────────────────────────────────────────────────────
#  B. write_hints
# ──────────────────────────────────────────────────────────────────────────

def test_write_hints_duplicate(tmp_path: Path) -> None:
    """Writing a memory with same subject/tags as an existing one returns write_hints."""
    tools = _tools(tmp_path)
    _write(
        tools, content="original content about api tokens",
        subject="api-token-policy", tags=["policy", "security"],
    )
    res = tools.memory_write(
        content="new content about api tokens",
        subject="api-token-policy", tags=["policy", "security"],
        workspace="ws", source_type="agent_generated", agent_id="tester",
    )
    assert res["ok"]
    hints = res["data"].get("write_hints")
    assert hints is not None
    targets = hints["possible_supersede_targets"]
    assert len(targets) >= 1
    assert targets[0]["hint_type"] == "possible_duplicate"


def test_write_hints_attention_flag(tmp_path: Path) -> None:
    """v0.8.7: a firing write_hint also promotes a loud attention_required flag."""
    tools = _tools(tmp_path)
    _write(
        tools, content="original content about api tokens",
        subject="api-token-policy", tags=["policy", "security"],
    )
    res = tools.memory_write(
        content="new content about api tokens",
        subject="api-token-policy", tags=["policy", "security"],
        workspace="ws", source_type="agent_generated", agent_id="tester",
    )
    assert res["ok"]
    data = res["data"]
    assert data.get("attention_required") is True
    summary = data.get("attention_summary")
    assert isinstance(summary, str) and summary.startswith("Possible duplicate/evolution of memory #")


def test_write_hints_evolution(tmp_path: Path) -> None:
    """Writing significantly longer content with same tags hints possible_evolution_of."""
    tools = _tools(tmp_path)
    _write(
        tools, content="short original",
        subject="release-notes-v1", tags=["release", "changelog"],
    )
    res = tools.memory_write(
        content="x" * 200,  # significantly longer than "short original"
        subject="release-notes-v1", tags=["release", "changelog"],
        workspace="ws", source_type="agent_generated", agent_id="tester",
    )
    assert res["ok"]
    hints = res["data"].get("write_hints")
    assert hints is not None
    targets = hints["possible_supersede_targets"]
    assert len(targets) >= 1
    assert targets[0]["hint_type"] == "possible_evolution_of"


def test_write_hints_records_dismissable_row(tmp_path: Path) -> None:
    """v0.8.8: write_hints now records a dismissable metadata_write_hint row
    (reverses id=264 / the old 'never writes to conflicts' contract)."""
    tools = _tools(tmp_path)
    _write(tools, content="first", subject="dup", tags=["tag-a", "tag-b"])
    tools.memory_write(
        content="second", subject="dup", tags=["tag-a", "tag-b"],
        workspace="ws", source_type="agent_generated", agent_id="tester",
    )
    conflicts = tools.memory_list_conflicts(status="open")["data"]["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["source"] == "metadata_write_hint"


def test_write_hints_no_candidates(tmp_path: Path) -> None:
    """When no overlap candidates exist, write_hints field is absent."""
    tools = _tools(tmp_path)
    _write(tools, content="unrelated", subject="xyz", tags=["zzz-unique"])
    res = tools.memory_write(
        content="new stuff", subject="abc", tags=["aaa-unique"],
        workspace="ws", source_type="agent_generated", agent_id="tester",
    )
    assert res["ok"]
    assert "write_hints" not in res["data"]
    assert not res["data"].get("attention_required")  # v0.8.7: no flag when no hint


# ──────────────────────────────────────────────────────────────────────────
#  C. record_conflict refresh mode
# ──────────────────────────────────────────────────────────────────────────

def test_record_conflict_refresh_existing(tmp_path: Path) -> None:
    """refresh=True updates an existing open conflict's enrichment fields."""
    tools = _tools(tmp_path)
    a = _write(tools, content="a", subject="sa")
    b = _write(tools, content="b", subject="sb")
    # Insert initial conflict.
    r1 = tools.memory_record_conflict(
        left_id=a, right_id=b, reason="initial",
        conflict_type="contradiction", conflict_point="v1",
        suggested_winner=a, confidence_hint="low", source="llm_informed",
    )
    assert r1["data"]["outcome"] == "inserted"
    cid = r1["data"]["conflict_id"]
    # Deduped without refresh.
    r2 = tools.memory_record_conflict(left_id=a, right_id=b, reason="dedup test")
    assert r2["data"]["outcome"] == "deduped"
    assert r2["data"]["conflict_id"] == cid
    # Refresh updates fields.
    r3 = tools.memory_record_conflict(
        left_id=a, right_id=b, reason="updated reasoning",
        conflict_type="contradiction", conflict_point="v2 refined",
        suggested_winner=b, confidence_hint="high", source="llm_informed",
        refresh=True, left_version=1, right_version=1,
        scan_model="test-model-v2",
    )
    assert r3["data"]["outcome"] == "refreshed"
    assert r3["data"]["conflict_id"] == cid
    # Verify the fields were actually updated.
    conflicts = tools.memory_list_conflicts(status="open")["data"]["conflicts"]
    c = next(c for c in conflicts if c["id"] == cid)
    assert c["conflict_point"] == "v2 refined"
    assert c["suggested_winner"] == b
    assert c["confidence_hint"] == "high"
    assert c["scan_model"] == "test-model-v2"


def test_record_conflict_dedup_without_refresh_keeps_existing(tmp_path: Path) -> None:
    """Default (refresh=False) does not modify the existing row."""
    tools = _tools(tmp_path)
    a = _write(tools, content="a", subject="sa")
    b = _write(tools, content="b", subject="sb")
    tools.memory_record_conflict(
        left_id=a, right_id=b, reason="original",
        conflict_type="contradiction", conflict_point="keep this",
        confidence_hint="high",
    )
    # Second call without refresh — even with different args, existing row stays.
    tools.memory_record_conflict(
        left_id=a, right_id=b, reason="different reason",
        conflict_point="should not overwrite",
    )
    conflicts = tools.memory_list_conflicts(status="open")["data"]["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["conflict_point"] == "keep this"


# ──────────────────────────────────────────────────────────────────────────
#  D. evolution conflict_type
# ──────────────────────────────────────────────────────────────────────────

def test_record_conflict_evolution_type(tmp_path: Path) -> None:
    """conflict_type=evolution stores and returns correctly."""
    tools = _tools(tmp_path)
    a = _write(tools, content="old version of spec", subject="spec-v1", tags=["spec"])
    b = _write(tools, content="new version of spec", subject="spec-v2", tags=["spec"])
    tools.memory_record_conflict(
        left_id=a, right_id=b, reason="stale_active_memory: spec-v1 superseded by spec-v2",
        conflict_type="evolution",
        conflict_point="spec v1 should be superseded but both still active",
        suggested_winner=b, confidence_hint="high", source="llm_informed",
    )
    conflicts = tools.memory_list_conflicts(status="open")["data"]["conflicts"]
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c["conflict_type"] == "evolution"
    assert "stale_active_memory" in c["reason"]
    assert c["suggested_winner"] == b

# ──────────────────────────────────────────────────────────────────────────
#  E. server wrapper pass-through for v0.7.6 parameters
# ──────────────────────────────────────────────────────────────────────────

def test_server_wrapper_passes_v076_parameters(tmp_path: Path) -> None:
    """FastMCP wrapper must expose and pass through v0.7.6 parameters."""
    import inspect
    from unittest.mock import patch

    from memory_arbiter import server as srv

    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "b.jsonl",
        client="test",
        agent_id="tester",
        workspace="ws",
        enable_sqlite_vec=False,
        update_check_enabled=False,
        tool_profile="legacy_full",
    )
    with patch("memory_arbiter.server.Settings.from_env", return_value=settings):
        app = srv.build_server()
    tools_reg = app._tool_manager._tools  # type: ignore[attr-defined]

    search_fn = tools_reg["memory_search"].fn
    edit_fn = tools_reg["memory_edit"].fn
    record_fn = tools_reg["memory_record_conflict"].fn
    assert "include_conflict_signal" in str(inspect.signature(search_fn))
    assert "tags_only" in str(inspect.signature(edit_fn))
    assert "add_tags" in str(inspect.signature(edit_fn))
    assert "remove_tags" in str(inspect.signature(edit_fn))
    assert "refresh" in str(inspect.signature(record_fn))
    assert "left_version" in str(inspect.signature(record_fn))
    assert "right_version" in str(inspect.signature(record_fn))
    assert "scan_model" in str(inspect.signature(record_fn))

    my_tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    a = my_tools.memory_write(
        content="wrapper alpha conflict", subject="wrapper-alpha",
        tags=["wrapper"], workspace="ws", source_type="agent_generated", agent_id="tester",
    )["data"]["id"]
    b = my_tools.memory_write(
        content="wrapper beta conflict", subject="wrapper-beta",
        tags=["wrapper"], workspace="ws", source_type="agent_generated", agent_id="tester",
    )["data"]["id"]

    search_fn(query="wrapper", include_conflict_signal=False)
    edit_res = edit_fn(memory_id=a, tags_only=True, add_tags=["checked"])
    assert edit_res["ok"]
    assert "checked" in edit_res["data"]["tags"]

    inserted = record_fn(left_id=a, right_id=b, reason="initial", conflict_point="old")
    assert inserted["data"]["outcome"] == "inserted"
    refreshed = record_fn(
        left_id=a, right_id=b, reason="refined", conflict_point="new",
        refresh=True, left_version=1, right_version=1, scan_model="wrapper-model",
    )
    assert refreshed["data"]["outcome"] == "refreshed"
    conflict = my_tools.memory_list_conflicts(status="open")["data"]["conflicts"][0]
    assert conflict["conflict_point"] == "new"
    assert conflict["scan_model"] == "wrapper-model"


# ──────────────────────────────────────────────────────────────────────────
#  D. v0.8.8 advisory 双层:write 落表 + dismiss(not_a_conflict) + version CAS
# ──────────────────────────────────────────────────────────────────────────

def test_is_pair_dismissed_and_version_cas(tmp_path: Path) -> None:
    """v0.8.8: is_pair_dismissed tracks not_a_conflict + version CAS."""
    tools = _tools(tmp_path)
    a = _write(tools, content="api policy v1", subject="api-token-policy", tags=["policy", "security"])
    b = _write(tools, content="api policy v2", subject="api-token-policy", tags=["policy", "security"])
    row = tools.memory_list_conflicts(status="open")["data"]["conflicts"][0]
    assert tools.db.is_pair_dismissed(a, b) is False
    tools.memory_resolve_conflict(conflict_id=row["id"], status="not_a_conflict", reason="not a conflict")
    assert tools.db.is_pair_dismissed(a, b) is True
    # Editing one side bumps its version → CAS invalidates the dismissal.
    tools.memory_edit(memory_id=a, new_content="api policy v1 revised")
    assert tools.db.is_pair_dismissed(a, b) is False


def test_dismissed_pair_not_surfaced_on_search(tmp_path: Path) -> None:
    """v0.8.8 Layer 0: once a pair is dismissed, search no longer attaches a
    conflict_signal for it (the nag is permanently stopped until a version change)."""
    tools = _tools(tmp_path)
    _write(tools, content="api policy v1", subject="api-token-policy", tags=["policy", "security"])
    tools.memory_write(content="api policy v2", subject="api-token-policy",
                       tags=["policy", "security"], workspace="ws",
                       source_type="agent_generated", agent_id="tester")
    # Before dismiss: the pair surfaces as an advisory runtime_metadata_hint.
    res1 = tools.memory_search(query="api token")
    assert any("conflict_signal" in r for r in res1["data"]["results"])
    # Dismiss it.
    row = tools.memory_list_conflicts(status="open")["data"]["conflicts"][0]
    tools.memory_resolve_conflict(conflict_id=row["id"], status="not_a_conflict")
    # After dismiss: no conflict_signal on any result.
    res2 = tools.memory_search(query="api token")
    assert not any("conflict_signal" in r for r in res2["data"]["results"])


def test_resolve_conflict_status_validation(tmp_path: Path) -> None:
    """v0.8.8: resolve_conflict accepts 'not_a_conflict'; rejects invalid status."""
    tools = _tools(tmp_path)
    _write(tools, content="x", subject="s", tags=["t1", "t2"])
    _write(tools, content="y", subject="s", tags=["t1", "t2"])
    row = tools.memory_list_conflicts(status="open")["data"]["conflicts"][0]
    bad = tools.memory_resolve_conflict(conflict_id=row["id"], status="bogus")
    assert bad["data"]["outcome"] == "invalid_status"
    assert len(tools.memory_list_conflicts(status="open")["data"]["conflicts"]) == 1  # unchanged
    good = tools.memory_resolve_conflict(conflict_id=row["id"], status="not_a_conflict")
    assert good["data"]["outcome"] == "not_a_conflict"
    assert len(tools.memory_list_conflicts(status="open")["data"]["conflicts"]) == 0
    assert len(tools.memory_list_conflicts(status="not_a_conflict")["data"]["conflicts"]) == 1


def test_supersede_cleans_dismissed_rows(tmp_path: Path) -> None:
    """v0.8.8: superseding a memory clears not_a_conflict rows touching it
    (the dismissal has no referent once the memory is superseded)."""
    tools = _tools(tmp_path)
    a = _write(tools, content="x", subject="s", tags=["t1", "t2"])
    _write(tools, content="y", subject="s", tags=["t1", "t2"])
    row = tools.memory_list_conflicts(status="open")["data"]["conflicts"][0]
    tools.memory_resolve_conflict(conflict_id=row["id"], status="not_a_conflict")
    assert len(tools.memory_list_conflicts(status="not_a_conflict")["data"]["conflicts"]) == 1
    tools.memory_supersede(memory_id=a, authorized=True, reason="gone")
    assert len(tools.memory_list_conflicts(status="not_a_conflict")["data"]["conflicts"]) == 0


def test_list_conflicts_source_filter(tmp_path: Path) -> None:
    """v0.8.8: list_conflicts returns source and supports source filtering."""
    tools = _tools(tmp_path)
    _write(tools, content="x", subject="s", tags=["t1", "t2"])
    _write(tools, content="y", subject="s", tags=["t1", "t2"])
    opens = tools.memory_list_conflicts(status="open")["data"]["conflicts"]
    assert opens and opens[0]["source"] == "metadata_write_hint"
    assert tools.memory_list_conflicts(status="open", source="llm_informed")["data"]["conflicts"] == []
    assert len(tools.memory_list_conflicts(status="open", source="metadata_write_hint")["data"]["conflicts"]) == 1


def test_doctor_open_conflicts_by_source(tmp_path: Path) -> None:
    """v0.8.8: doctor helper groups open conflicts by source (漏洞#4)."""
    from memory_arbiter.doctor import _open_conflicts_by_source
    tools = _tools(tmp_path)
    _write(tools, content="x", subject="s", tags=["t1", "t2"])
    _write(tools, content="y", subject="s", tags=["t1", "t2"])
    with tools.db.connection() as conn:
        bs = _open_conflicts_by_source(conn)
    assert bs.get("metadata_write_hint") == 1


def test_write_fail_open_when_dismiss_check_throws(tmp_path: Path, monkeypatch) -> None:
    """v0.8.8 T13b: if the dismiss check (is_pair_dismissed) raises, the write
    must STILL ring attention + store the memory (fail-open). A diagnostic-layer
    failure must never silently suppress the duplicate prompt or lose the write."""
    tools = _tools(tmp_path)
    _write(tools, content="api policy v1", subject="api-token-policy", tags=["policy", "security"])

    def _boom(a, b):
        raise RuntimeError("db down")
    monkeypatch.setattr(tools.db, "is_pair_dismissed", _boom)

    res = tools.memory_write(
        content="api policy v2", subject="api-token-policy", tags=["policy", "security"],
        workspace="ws", source_type="agent_generated", agent_id="tester",
    )
    # Memory stored AND attention still rang despite the dismiss-check crash.
    assert res["ok"]
    assert res["data"]["id"] is not None
    assert res["data"].get("attention_required") is True


def test_dismiss_survives_null_version_pin(tmp_path: Path) -> None:
    """v0.8.8 (ZCode review): a not_a_conflict row with a NULL version pin must
    STILL count as dismissed — otherwise a race/legacy NULL silently re-enables
    the nag. Defends `NULL = x` → NULL (not true) in the version CAS."""
    tools = _tools(tmp_path)
    a = _write(tools, content="x", subject="s", tags=["t1", "t2"])
    b = _write(tools, content="y", subject="s", tags=["t1", "t2"])
    row = tools.memory_list_conflicts(status="open")["data"]["conflicts"][0]
    tools.memory_resolve_conflict(conflict_id=row["id"], status="not_a_conflict")
    assert tools.db.is_pair_dismissed(a, b) is True
    # Corrupt the pin to NULL (simulate the race / legacy-NULL case).
    with tools.db.connection() as conn:
        conn.execute("UPDATE conflicts SET left_version=NULL, right_version=NULL WHERE id=?", (row["id"],))
        conn.commit()
    # Still dismissed — NULL pin is treated as a valid dismissal, not a re-nag.
    assert tools.db.is_pair_dismissed(a, b) is True

