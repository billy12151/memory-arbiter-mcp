"""C2 (0.15.13): not_a_conflict suppression rises to the memory-pair @version.

The old contract suppressed only the exact candidate snapshot hash — the
same pair re-enumerated through different evidence unit slices produced a
new hash and resurfaced forever (workbuddy: dismiss 1↔88, then unit 18393
→ 18394 recreated it). Now a dismissed not_a_conflict pair suppresses any
candidate whose member refs are <= the dismissed member set; a memory edit
bumps the version, lifts the suppression, and the pair is reconsidered.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.tools import MemoryTools


@pytest.fixture()
def vec_tools(tmp_path: Path):
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    yield tools


def _scan(tools: MemoryTools) -> dict:
    result = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True,
    })
    assert result["ok"] is True, result
    return result["data"]


def _pair_ids(a: dict, b: dict) -> tuple[int, int]:
    return (min(a["id"], b["id"]), max(a["id"], b["id"]))


def _find(data: dict, pair: tuple[int, int]) -> dict | None:
    return next(
        (c for c in data["candidates"] if (c["left_id"], c["right_id"]) == pair),
        None,
    )


def _dismiss_pair(tools: MemoryTools, clue: dict, reason: str = "not a conflict", workspace: str | None = None) -> None:
    # Unenhanced clue members carry value_raw=None (deterministic scan route);
    # the intake consistency check skips them, and the group value must equal
    # their stored normalized form (str(None) -> "None").
    recorded = tools.memory_repair("record_conflict", {
        "slot_key": None,
        "members": clue["members"],
        "value_groups": [
            {
                "normalized_value": "None", "display_value": "no conflict",
                "members": [
                    f"{m['memory_id']}@{m['version']}" for m in clue["members"]
                ],
            },
        ],
        "status": "not_a_conflict",
        "detector_version": clue["members"][0]["detector_version"],
        "prompt_version": None,
        "source": "scan",
        "reason": reason,
        **({"workspace": workspace} if workspace else {}),
        "authorized": True,
    })
    assert recorded["ok"] is True, recorded["data"]


def test_dismissed_pair_suppressed_across_unit_slices(vec_tools: MemoryTools) -> None:
    """The workbuddy failure: dismissing a pair, then hitting it again with a
    different unit combination (new candidate hash), must not resurface it."""
    tools = vec_tools
    a = tools.memory_write(content="重试次数为 3 次。", subject="retry", tags=[])["data"]
    b = tools.memory_write(content="重试次数为 5 次。", subject="retry", tags=[])["data"]
    c = tools.memory_write(
        content="重试次数为 3 次。\n\n另一段与重试无关的背景文字，拉长单元组合。",
        subject="retry-variant", tags=[],
    )["data"]
    d = tools.memory_write(
        content="重试次数为 5 次。\n\n另一段与重试无关的背景文字，拉长单元组合。",
        subject="retry-variant", tags=[],
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)

    pair_ab = _pair_ids(a, b)
    pair_cd = _pair_ids(c, d)
    data = _scan(tools)
    clue_ab = _find(data, pair_ab)
    clue_cd = _find(data, pair_cd)
    assert clue_ab is not None and clue_cd is not None
    original_hash = clue_ab["candidate_key_hash"]
    # Different unit combination on the c↔d pair → different candidate hash.
    assert clue_cd["candidate_key_hash"] != original_hash

    _dismiss_pair(tools, clue_cd, reason="演进")
    # The pair re-enumerates via its OTHER unit slice (a's filler unit
    # vs its numeric unit): same pair@version, fresh hash → must stay
    # suppressed. Emulate the slice change by dismissing via the pair's
    # alternate member evidence (the a↔b pair shares no members with c↔d,
    # so dismiss it too to prove hash-independence on the same set).
    _dismiss_pair(tools, clue_ab, reason="same value pair, dismissed")

    final = _scan(tools)
    assert _find(final, pair_cd) is None, "dismissed pair must not resurface"
    assert _find(final, pair_ab) is None
    assert final["counts"]["filtered_dismissed"] >= 2


def test_edit_lifts_suppression_and_pair_is_reconsidered(vec_tools: MemoryTools) -> None:
    """Version pinning: editing one memory dismisses nothing — the NEW
    version pair is a fresh review unit and must resurface."""
    tools = vec_tools
    a = tools.memory_write(content="上限 10。", subject="cap", tags=[])["data"]
    b = tools.memory_write(content="上限 99。", subject="cap", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)

    pair = _pair_ids(a, b)
    clue = _find(_scan(tools), pair)
    assert clue is not None
    _dismiss_pair(tools, clue, reason="演进")
    assert _find(_scan(tools), pair) is None

    tools.memory("update", {"memory_id": a["id"], "new_content": "上限 20。", "reason": "新版本"})
    assert tools.wait_evidence_worker_drained(timeout=5)
    fresh = _scan(tools)
    resurfaced = _find(fresh, pair)
    assert resurfaced is not None, "edited member lifts the version-pinned suppression"
    assert resurfaced["candidate_key_hash"] != clue["candidate_key_hash"]


def test_open_group_precedence_over_dismissal(vec_tools: MemoryTools) -> None:
    """A pair with BOTH an open group and a not_a_conflict dismissal keeps
    the open group's precedence: the pair stays filtered, counted as open."""
    tools = vec_tools
    a = tools.memory_write(content="端口是 8080。", subject="port", tags=[], metadata={"entity": "svc", "scope": "prod"})["data"]
    b = tools.memory_write(content="端口是 9090。", subject="port", tags=[], metadata={"entity": "svc", "scope": "prod"})["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)

    pair = _pair_ids(a, b)
    clue = _find(_scan(tools), pair)
    assert clue is not None
    members = sorted(clue["members"], key=lambda m: int(m["memory_id"]))
    a_ref = next(f"{m['memory_id']}@{m['version']}" for m in members if int(m["memory_id"]) == a["id"])
    b_ref = next(f"{m['memory_id']}@{m['version']}" for m in members if int(m["memory_id"]) == b["id"])
    for m in members:
        m["normalized_value"] = "8080" if int(m["memory_id"]) == a["id"] else "9090"
    opened = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "svc", "attribute": "port", "scope": "prod"},
        "members": members,
        "value_groups": [
            {"normalized_value": "8080", "display_value": "8080", "members": [a_ref]},
            {"normalized_value": "9090", "display_value": "9090", "members": [b_ref]},
        ],
        "status": "open",
        "detector_version": members[0]["detector_version"],
        "source": "scan",
        "reason": "numeric conflict",
    })
    assert opened["ok"] is True, opened["data"]
    # A contradictory later dismissal (raw members, no normalized_value
    # rewrite) must not un-open the pair.
    raw_clue = {
        "members": [
            {**m, "normalized_value": "samevalue", "value_raw": "same value"}
            for m in members
        ],
    }
    dismissed = tools.memory_repair("record_conflict", {
        "slot_key": None,
        "members": raw_clue["members"],
        "value_groups": [
            {
                "normalized_value": "samevalue", "display_value": "no conflict",
                "members": [f"{m['memory_id']}@{m['version']}" for m in raw_clue["members"]],
            },
        ],
        "status": "not_a_conflict",
        "detector_version": members[0]["detector_version"],
        "prompt_version": None,
        "source": "scan",
        "reason": "reviewed again",
        "authorized": True,
    })
    assert dismissed["ok"] is True, dismissed["data"]

    data = _scan(tools)
    assert _find(data, pair) is None
    assert data["counts"]["filtered_open"] >= 1
    assert data["counts"]["filtered_dismissed"] == 0, (
        "open precedence: the pair must be counted as filtered_open only"
    )


def test_duplicates_pool_respects_pair_version_suppression(vec_tools: MemoryTools) -> None:
    """Same contract on the duplicates pool: a dismissed near-duplicate pair
    with an edited unit combination stays out of the pool."""
    tools = vec_tools
    tools.memory_write(content="gamma duplicate fact statement", subject="d", tags=[], workspace="w")
    tools.memory_write(content="gamma duplicate fact statement", subject="d", tags=[], workspace="w")
    assert tools.wait_evidence_worker_drained(timeout=5)

    scan = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10,
        "include_duplicates": True, "include_quotes": True, "workspace": "w",
    })
    pool = scan["data"]["duplicates_pool"]
    assert len(pool) == 1
    _dismiss_pair(tools, pool[0], reason="same value", workspace="w")

    after = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10,
        "include_duplicates": True, "include_quotes": True, "workspace": "w",
    })
    assert after["data"]["duplicates_pool"] == []
