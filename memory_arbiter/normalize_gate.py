"""Shared workspace-normalization gate (0.16.2 plan §1.1).

Five consumers judge through this one function — pipeline suspect
generation, decision-time vote, the decision share check, the auto-move
audit payload, and the weekly full-library backstop. "Same queue, same
gate" breaks the moment any consumer hardcodes its own threshold; the
0.16.0 gate did exactly that (absolute >=8/10 votes, calibrated against
the 930/950 cases owner later re-adjudicated as true moves) and each
consumer re-derived it by hand.

0.16.2 proportional gate: the top foreign bucket needs >=4 votes AND
>=60% of all foreign votes (own bucket excluded — it is never a move
candidate). Small buckets whose members all sit in one foreign
neighbourhood now pass (4/4 = 100%); 2-vote coincidences stay blocked by
the floor.
"""
from __future__ import annotations

from typing import Any


def normalize_gate(votes: dict[str, int], own: str) -> "tuple[bool, dict[str, Any]]":
    """Judge one vector vote.

    ``votes`` maps bucket -> neighbour count and MAY include ``own``; the
    own bucket is excluded before judging. Returns ``(passed, evidence)`` —
    ``evidence`` is written to the ``normalize_audit`` gate payload verbatim.
    It carries the resolved numbers, not constant names: a renamed constant
    must never break auto-moves again (0.16.2 review P1-3 consumer ④).
    """
    from .constants import NORMALIZE_FOREIGN_SHARE_MIN, NORMALIZE_VOTE_MIN_FOREIGN

    foreign = {bucket: count for bucket, count in votes.items() if bucket != own}
    total_foreign = sum(foreign.values())
    top_bucket, top_votes = max(
        foreign.items(), key=lambda item: item[1], default=("", 0)
    )
    share = (top_votes / total_foreign) if total_foreign else 0.0
    evidence: dict[str, Any] = {
        "top_bucket": top_bucket,
        "top_votes": int(top_votes),
        "total_foreign": int(total_foreign),
        "share": round(float(share), 4),
        "min_foreign": NORMALIZE_VOTE_MIN_FOREIGN,
        "share_min": NORMALIZE_FOREIGN_SHARE_MIN,
    }
    passed = bool(top_bucket) and top_votes >= NORMALIZE_VOTE_MIN_FOREIGN and share >= NORMALIZE_FOREIGN_SHARE_MIN
    return passed, evidence
