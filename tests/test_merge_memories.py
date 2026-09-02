"""Governed near-duplicate merge (memory_govern action=merge_memories) tests.

Covers the v0.15.2 PR1 contract: loser supersede with a persisted merged_into
pointer (read-modify-write over wholesale metadata replacement), survivor
content merge via the edit primitive, per-id rejection of conflict-group
losers, idempotent re-merge, authorization gate, and the scan_candidates
duplicates_pool discovery outlet with its suppression contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember
from memory_arbiter.surfaces import ProductSurfaces
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    return MemoryTools(settings, MemoryDB(settings))


def _memory(tools: MemoryTools, content: str, workspace: str = "w", **meta: object) -> int:
    result = tools.memory_write(
        content=content, subject="note", tags=[], workspace=workspace,
        metadata=dict(meta) if meta else None,
    )
    memory_id = int(result["data"]["id"])
    assert memory_id
    return memory_id


def _merge(tools: MemoryTools, survivor: int, losers: list[int], **extra: object) -> dict:
    return tools.memory_govern("merge_memories", {
        "survivor_id": survivor, "loser_ids": losers,
        "reason": "test merge", "authorized": True, **extra,
    })


def _record_group(db: MemoryDB, left: int, right: int, *, status: str = "open") -> int:
    def member(memory_id: int, value: str) -> ConflictMember:
        quote = f"database is {value}"
        return ConflictMember(
            memory_id=memory_id, version=1, attribute_raw="database", value_raw=value,
            normalized_attribute="database", normalized_value=value.casefold(),
            evidence_quote=quote, evidence_span=(0, len(quote)),
            content_hash=(str(memory_id) * 64)[:64], direction="a_to_b",
            prompt_version="p1", detector_version="d1",
        )

    left_member, right_member = member(left, "mysql"), member(right, "sqlite")
    created = db.record_conflict_group(
        workspace_canonical="w",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=[left_member, right_member],
        value_groups=[
            {"normalized_value": "mysql", "display_value": "mysql", "members": [f"{left}@1"]},
            {"normalized_value": "sqlite", "display_value": "sqlite", "members": [f"{right}@1"]},
        ],
        detection_reason="different database", source="scan",
        detector_version="d1", prompt_version="p1", conflict_point="database",
        status=status,
    )
    return int(created["conflict_id"])


def test_merge_requires_authorization_with_impact(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor, loser = _memory(tools, "keep me"), _memory(tools, "dup")
    result = tools.memory_govern("merge_memories", {
        "survivor_id": survivor, "loser_ids": [loser], "reason": "why",
    })
    assert result["ok"] is False
    data = result["data"]
    assert data["action_required"] == "ask_user_for_authorization"
    assert data["authorized"] is False
    assert data["retry"]["action"] == "merge_memories"
    assert "merge_memories" in ProductSurfaces._GOVERNANCE_IMPACTS
    assert data["impact"] == ProductSurfaces._GOVERNANCE_IMPACTS["merge_memories"]


def test_merge_supersedes_losers_with_pointer_and_keeps_metadata(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor = _memory(tools, "canonical statement")
    loser = _memory(tools, "canonical statement", origin_note="keep me")
    result = _merge(tools, survivor, [loser])
    assert result["ok"] is True, result
    data = result["data"]
    assert data["merged"] is True
    assert data["merged_ids"] == [loser]
    assert data["deduped_ids"] == []
    assert data["failed_ids"] == []
    assert data["survivor_edited"] is False
    assert data["post_commit"] == {"status": "skipped", "reason": "no_survivor_change"}
    superseded = tools.memory_get(loser)["data"]["memory"]
    assert superseded["status"] == "superseded"
    metadata = superseded["metadata"]
    assert metadata["merged_into"] == survivor
    assert metadata["merge_reason"] == "test merge"
    assert metadata["origin_note"] == "keep me"  # read-modify-write, not wholesale drop
    assert superseded["version"] == 2  # status change bumps version (retire contract)


def test_merged_loser_leaves_active_search_and_stays_expired_visible(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor = _memory(tools, "unique topic alpha")
    loser = _memory(tools, "unique topic alpha")
    _merge(tools, survivor, [loser])
    found = tools.memory_search(query="unique topic alpha", limit=10)
    active_ids = [int(item.get("id") or item.get("memory_id")) for item in found["data"]["results"]]
    assert loser not in active_ids
    assert survivor in active_ids
    expired = tools.memory_search_expired(query="unique topic alpha")
    expired_ids = [int(item.get("id") or item.get("memory_id")) for item in expired["data"]["results"]]
    assert loser in expired_ids


def test_merge_with_merged_content_edits_survivor_and_rechecks(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor = _memory(tools, "partial one")
    loser = _memory(tools, "partial two")
    result = _merge(tools, survivor, [loser], merged_content="partial one plus partial two")
    assert result["ok"] is True, result
    data = result["data"]
    assert data["survivor_edited"] is True
    assert data["history_id"]
    assert data["post_commit"] == {"status": "recheck_conflicts"}
    updated = tools.memory_get(survivor)["data"]["memory"]
    assert updated["content"] == "partial one plus partial two"
    assert updated["version"] == 2
    history = tools.memory_history(survivor)["data"]
    snapshots = history.get("history") or history.get("entries") or []
    assert snapshots and snapshots[0]["content_snapshot"] == "partial one"


def test_merge_idempotent_same_and_conflicting_targets(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor, other, loser = _memory(tools, "s"), _memory(tools, "o"), _memory(tools, "d")
    _merge(tools, survivor, [loser])
    again = _merge(tools, survivor, [loser])
    assert again["ok"] is True
    assert again["data"]["deduped_ids"] == [loser]
    assert again["data"]["merged_ids"] == []
    # Re-merging the same loser into a different survivor is refused per-id.
    conflict = _merge(tools, other, [loser])
    assert conflict["ok"] is False
    assert conflict["data"]["failed_ids"][0]["error"] == "already_merged"


def test_merge_validates_request_shape(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor = _memory(tools, "s")
    for payload, fragment in (
        ({"loser_ids": []}, "loser_ids"),
        ({"loser_ids": [survivor]}, "survivor_id must not appear"),
        ({"loser_ids": list(range(1, 52)) + [survivor]}, "loser_ids"),
    ):
        result = tools.memory_govern("merge_memories", {
            "survivor_id": survivor, "reason": "r", "authorized": True, **payload,
        })
        assert result["ok"] is False, payload
        assert fragment in str(result["data"])
    missing_reason = tools.memory_govern("merge_memories", {
        "survivor_id": survivor, "loser_ids": [1], "authorized": True,
    })
    assert missing_reason["ok"] is False


def test_merge_rejects_losers_in_open_conflict_groups(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor = _memory(tools, "survivor statement")
    left, right = _memory(tools, "database is mysql"), _memory(tools, "database is sqlite")
    conflict_id = _record_group(tools.db, left, right)
    result = _merge(tools, survivor, [left])
    assert result["ok"] is False
    failure = result["data"]["failed_ids"][0]
    assert failure["memory_id"] == left
    assert failure["error"] == "conflict_member"
    assert failure["attention_required"] is True
    assert failure["conflicts"][0]["conflict_id"] == conflict_id
    # The loser is untouched: the group stays intact for the conflict flow.
    assert tools.memory_get(left)["data"]["memory"]["status"] == "active"


def test_merge_partial_success_commits_good_losers_despite_per_id_failures(tmp_path: Path) -> None:
    """The core per-id contract: one failing loser must not block the others
    in the same transaction, and the good losers must actually be superseded."""
    tools = make_tools(tmp_path)
    survivor = _memory(tools, "canonical statement")
    good_loser = _memory(tools, "canonical statement")
    conflict_left, conflict_right = _memory(tools, "port is 8080"), _memory(tools, "port is 8081")
    _record_group(tools.db, conflict_left, conflict_right)
    cross_ws = _memory(tools, "canonical statement", workspace="other")

    result = _merge(tools, survivor, [good_loser, conflict_left, cross_ws])
    assert result["ok"] is True, result["data"]
    data = result["data"]
    assert data["merged_ids"] == [good_loser]
    errors = {failure["memory_id"]: failure["error"] for failure in data["failed_ids"]}
    assert errors[conflict_left] == "conflict_member"
    assert errors[cross_ws] == "workspace_mismatch"
    # Partial commit really landed.
    assert tools.memory_get(good_loser)["data"]["memory"]["status"] == "superseded"
    assert tools.memory_get(good_loser)["data"]["memory"]["metadata"]["merged_into"] == survivor
    assert tools.memory_get(conflict_left)["data"]["memory"]["status"] == "active"
    assert tools.memory_get(cross_ws)["data"]["memory"]["status"] == "active"


def test_merge_blank_merged_content_is_rejected_not_silently_skipped(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor, loser = _memory(tools, "s"), _memory(tools, "d")
    result = _merge(tools, survivor, [loser], merged_content="   ")
    assert result["ok"] is False
    assert "non-empty" in str(result["data"]["error"])
    # Nothing changed — the call failed before the transaction.
    assert tools.memory_get(loser)["data"]["memory"]["status"] == "active"


def test_merge_survivor_in_group_without_edit_passes(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor = _memory(tools, "database is mysql")
    loser = _memory(tools, "database is mysql")
    peer = _memory(tools, "database is sqlite")
    _record_group(tools.db, survivor, peer)
    result = _merge(tools, survivor, [loser])
    assert result["ok"] is True, result
    assert "attention_required" not in result["data"]


def test_merge_survivor_in_group_with_edit_flags_attention(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor = _memory(tools, "database is mysql")
    loser = _memory(tools, "database is mysql")
    peer = _memory(tools, "database is sqlite")
    conflict_id = _record_group(tools.db, survivor, peer)
    result = _merge(tools, survivor, [loser], merged_content="database is mysql, verified")
    assert result["ok"] is True, result
    data = result["data"]
    assert data["attention_required"] is True
    assert data["action_required"] == "review_unresolved_conflicts"
    assert data["unresolved_conflicts"][0]["conflict_id"] == conflict_id


def test_merge_rejects_cross_workspace_losers(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor = _memory(tools, "same fact", workspace="w")
    loser = _memory(tools, "same fact", workspace="other")
    result = _merge(tools, survivor, [loser])
    assert result["ok"] is False
    assert result["data"]["failed_ids"][0]["error"] == "workspace_mismatch"
    assert tools.memory_get(loser)["data"]["memory"]["status"] == "active"


def test_merge_survivor_must_be_active(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    survivor, loser = _memory(tools, "s"), _memory(tools, "d")
    tools.memory_supersede(memory_id=survivor, reason="gone", authorized=True)
    result = _merge(tools, survivor, [loser])
    assert result["ok"] is False
    assert "not active" in str(result["data"]["error"])


@pytest.fixture()
def vec_tools(tmp_path: Path):
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    yield tools


def _write_for_scan(tools: MemoryTools, content: str) -> int:
    result = tools.memory_write(content=content, subject="scan", tags=[], workspace="w")
    return int(result["data"]["id"])


def test_scan_duplicates_pool_default_off_and_opt_in(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _write_for_scan(tools, "release version 1.2.3 is shipped")
    _write_for_scan(tools, "release version 1.2.3 is shipped")
    assert tools.wait_evidence_worker_drained(timeout=5)

    baseline = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10})
    assert baseline["ok"] is True, baseline
    assert baseline["data"]["duplicates_pool"] == []
    assert baseline["data"]["duplicates_truncated"] is False
    assert baseline["data"]["counts"]["duplicates"] == 0

    with_pool = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10, "include_duplicates": True,
    })
    assert with_pool["ok"] is True, with_pool
    pool = with_pool["data"]["duplicates_pool"]
    assert len(pool) == 1
    entry = pool[0]
    assert entry["reason"] in {"equivalent_value", "compatible_evidence"}
    assert {entry["left_id"], entry["right_id"]} == {
        int(m) for m in [entry["left_id"], entry["right_id"]]
    }
    # The pair must not leak into the real candidate set.
    assert with_pool["data"]["candidates"] == []


def test_scan_duplicates_pool_respects_recorded_suppression(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _write_for_scan(tools, "port 8080 is the default api port")
    _write_for_scan(tools, "port 8080 is the default api port")
    assert tools.wait_evidence_worker_drained(timeout=5)

    scan = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10, "include_duplicates": True,
    })
    assert scan["ok"] is True, scan
    pool = scan["data"]["duplicates_pool"]
    assert len(pool) == 1
    entry = pool[0]
    assert entry["members"], "pool entries must carry record_conflict-compatible members"

    # Dismissing the pair as not_a_conflict through the product surface must
    # suppress re-enumeration in the duplicates pool (suppression moved ahead
    # of the historical ignore-drop; hashes are derived from identical
    # evidence identities).
    recorded = tools.memory_repair("record_conflict", {
        "slot_key": None,
        "members": entry["members"],
        # Rule-gated duplicates carry no extracted value, so both members sit
        # in one group under the string form of None (str(None)).
        "value_groups": [
            {"normalized_value": "None", "display_value": "same value",
             "members": [f"{entry['members'][0]['memory_id']}@{entry['members'][0]['version']}",
                         f"{entry['members'][1]['memory_id']}@{entry['members'][1]['version']}"]},
        ],
        "status": "not_a_conflict",
        "detector_version": entry["members"][0]["detector_version"],
        "prompt_version": None,
        "source": "scan",
        "reason": "same value, not a conflict",
        "workspace": "w",
        "authorized": True,
    })
    assert recorded["ok"] is True, recorded["data"]
    after = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10, "include_duplicates": True,
    })
    assert after["ok"] is True
    assert after["data"]["duplicates_pool"] == [], (
        "recorded not_a_conflict pairs must not re-enter the duplicates pool"
    )


def test_scan_duplicates_pool_cap_and_truncation(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    for index in range(4):
        _write_for_scan(tools, f"cap probe identical statement number {index % 2}")
    assert tools.wait_evidence_worker_drained(timeout=5)
    result = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 1, "k": 10, "include_duplicates": True,
    })
    assert result["ok"] is True, result
    assert len(result["data"]["duplicates_pool"]) <= 2 * 1
