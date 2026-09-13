"""0.16.2 normalize gate unit tests (plan §1.1): the shared proportional gate
and its five consumers must agree — 4-vote floor, 60% foreign share, own-bucket
exclusion, small-bucket full vote."""
from __future__ import annotations

from memory_arbiter.constants import NORMALIZE_FOREIGN_SHARE_MIN, NORMALIZE_VOTE_MIN_FOREIGN
from memory_arbiter.normalize_gate import normalize_gate

import pytest


def test_floor_boundary_three_refused_four_pass() -> None:
    votes = {"proja": 3, "projb": 1}
    passed, ev = normalize_gate(votes, "own")
    assert not passed, "3 foreign votes below the floor"
    assert ev["top_votes"] == 3 and ev["min_foreign"] == NORMALIZE_VOTE_MIN_FOREIGN

    passed4, ev4 = normalize_gate({"proja": 4, "projb": 1}, "own")
    assert passed4, "4/5 = 80% >= 60% passes the floor and share"
    assert ev4["top_bucket"] == "proja" and ev4["share"] == 0.8


def test_share_boundary_dominant_mix() -> None:
    # 4/7 ≈ 57% < 60% → refused despite passing the floor.
    passed, ev = normalize_gate({"proja": 4, "projb": 3}, "own")
    assert not passed
    assert ev["top_votes"] == 4 and ev["total_foreign"] == 7
    assert ev["share"] == pytest.approx(4 / 7, abs=1e-3)
    assert ev["share_min"] == NORMALIZE_FOREIGN_SHARE_MIN

    # 6/10 = 60% exactly → passes.
    passed6, _ = normalize_gate({"proja": 6, "projb": 4}, "own")
    assert passed6


def test_no_foreign_votes_refused() -> None:
    # Neighbours all in the own bucket → no foreign vote at all.
    passed, ev = normalize_gate({"own": 10}, "own")
    assert not passed
    assert ev["top_bucket"] == "" and ev["total_foreign"] == 0 and ev["share"] == 0.0
    passed_empty, _ = normalize_gate({}, "own")
    assert not passed_empty


def test_small_bucket_full_vote_passes() -> None:
    # The 930 case: 4 votes for the twin family, 6 elsewhere — 40% share.
    passed, ev = normalize_gate({"mema-twin-dev": 4, "proja": 3, "projb": 3}, "own")
    assert not passed, "4/10 = 40% below the 60% share"

    # All 4 foreign neighbours agree → 4/4 = 100% → the case the old absolute
    # gate blocked and owner re-adjudicated as a true move.
    passed_all, ev_all = normalize_gate({"mema-twin-dev": 4}, "own")
    assert passed_all
    assert ev_all["top_bucket"] == "mema-twin-dev" and ev_all["share"] == 1.0


def test_two_vote_majority_blocked_by_floor() -> None:
    passed, _ = normalize_gate({"proja": 2, "projb": 0}, "own")
    assert not passed, "2 votes at 100% share still below the 4-vote floor"


def test_evidence_carries_numbers_not_constant_names() -> None:
    _, ev = normalize_gate({"proja": 5, "projb": 2}, "own")
    # The audit payload must stay valid if constants are renamed: numbers only.
    assert set(ev) == {"top_bucket", "top_votes", "total_foreign", "share", "min_foreign", "share_min"}
    assert isinstance(ev["top_votes"], int) and isinstance(ev["share"], float)
