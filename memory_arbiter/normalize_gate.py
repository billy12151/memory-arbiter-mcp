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

from typing import Any, Iterable


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


def compute_summary_votes(
    vectors: "dict[int, tuple[str, list[float]]]",
    target_ids: Iterable[int],
    *,
    path: str = "auto",
) -> "dict[int, dict[str, Any]]":
    """Shared summary-vector vote computation (0.16.10 shared-recall layer).

    Suspect generation (pipeline incremental), the weekly full-library
    backstop, and the decision-time re-vote ALL count votes through this one
    function, so the three can never drift apart — the 0.16.2 §1.1 "same
    queue, same gate" principle extended from the gate to the counting. Per
    target it returns ``votes`` (bucket -> neighbour count, own bucket
    included), ``own``, ``k``, and the ``own_best``/``foreign_best``
    (sim, id) neighbour probes the weekly audit payload needs.

    ``path``: ``"auto"`` picks the blocked matmul when targets cover most of
    the library (the weekly backstop's shape), ``"single"`` forces the
    row-by-row gemv every produce/decision-time caller used before this
    extraction. The two formulas are NOT bitwise-identical (1-48 ulp on
    gemv vs gemm), and the stable-sort tie discipline only fixes ordering
    WITHIN one sims array — so a caller that used to see gemv must keep
    gemv or ties can flip at the rank boundary (0.16.10 review finding).

    numpy absent -> {} (every caller already degrades to its no-vote path);
    ``k <= 0`` or no target in the library -> {} too.
    """
    try:
        import numpy as np
    except ImportError:
        return {}
    from .constants import NORMALIZE_VOTE_NEIGHBORS

    ids = sorted(vectors)
    n = len(ids)
    k = min(NORMALIZE_VOTE_NEIGHBORS, n - 1)
    if k <= 0:
        return {}
    wanted = set(target_ids) & set(ids)
    if not wanted:
        return {}
    workspaces = {mid: str(vectors[mid][0] or "") for mid in ids}
    matrix = np.array([vectors[mid][1] for mid in ids], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    unit = matrix / norms[:, None]

    def _vote_from_sims(sims: Any, row_mid: int) -> dict[str, Any]:
        order = np.argsort(-sims, kind="stable")[:k]
        votes: dict[str, int] = {}
        own_best = (-2.0, -1)
        foreign_best = (-2.0, -1)
        own = workspaces[row_mid]
        for col in order:
            col = int(col)
            sim = float(sims[col])
            bucket = workspaces[ids[col]]
            votes[bucket] = votes.get(bucket, 0) + 1
            if bucket == own:
                if sim > own_best[0]:
                    own_best = (sim, ids[col])
            elif sim > foreign_best[0]:
                foreign_best = (sim, ids[col])
        return {
            "votes": votes,
            "own": own,
            "k": k,
            "own_best": own_best,
            "foreign_best": foreign_best,
        }

    out: dict[int, dict[str, Any]] = {}
    if path == "block" or (path == "auto" and len(wanted) * 4 >= n):
        # Majority-target path: one blocked matmul over the whole library,
        # self-exclusion on the diagonal before the sort — the weekly
        # backstop's shape, reused whenever targets cover most rows anyway.
        block = 512
        for start in range(0, n, block):
            sims = unit[start:start + block] @ unit.T
            sims[np.arange(sims.shape[0]), np.arange(start, start + sims.shape[0])] = -1.0
            for local_row in range(sims.shape[0]):
                row_mid = ids[start + local_row]
                if row_mid in wanted:
                    out[row_mid] = _vote_from_sims(sims[local_row], row_mid)
    else:
        index_of = {mid: i for i, mid in enumerate(ids)}
        for mid in sorted(wanted):
            sims = unit @ unit[index_of[mid]]
            sims[index_of[mid]] = -1.0
            out[mid] = _vote_from_sims(sims, mid)
    return out
