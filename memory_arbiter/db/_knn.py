"""Shared vec0 KNN global-window growth loop (0.16.10 shared-recall layer).

sqlite-vec evaluates workspace/exclusion predicates AFTER the vec0 auxiliary
columns pick their global top-k window, so filtered callers must grow that
window until enough scoped rows are found or the whole lifecycle-eligible
vector domain has been considered. Two stores (``EvidenceStore.knn`` over
``memory_evidence_vec`` and ``MemoriesStore.subject_tags_knn`` over
``subject_tags_vec``) ran byte-identical copies of this loop; it now lives in
one place so a window-strategy change can never drift between them.

Private module: not re-exported through ``db/__init__`` — stores import the
helper directly.
"""
from __future__ import annotations

import sqlite3
from typing import Any


def knn_window_loop(
    conn: sqlite3.Connection,
    *,
    count_sql: str,
    query_sql: str,
    query_json: str,
    requested_k: int,
    params: list[Any],
    filtered: bool,
) -> list[Any]:
    """Run the MATCH/k=? query, growing the global window until scoped recall
    is satisfied or the whole candidate domain has been considered.

    ``count_sql`` counts the UNFILTERED lifecycle-eligible domain — the growth
    ceiling must be the unscoped count, because a scoped ceiling would cap the
    window below foreign rows and silently starve recall. ``query_sql`` must
    contain the ``k=?`` placeholder in vec0 MATCH position; the params order
    is ``[query_json, fetch_k, *params]`` on every iteration. With
    ``filtered=False`` the loop degenerates to a single fetch of exactly
    ``requested_k`` (no COUNT-driven growth — the fast path unscoped callers
    used before this extraction).
    """
    candidate_count = int(conn.execute(count_sql).fetchone()[0])
    max_fetch = max(1, candidate_count)
    fetch_k = min(max_fetch, requested_k * 4) if filtered else requested_k
    rows: list[Any] = []
    while fetch_k > 0:
        rows = conn.execute(
            query_sql, [query_json, fetch_k, *params]
        ).fetchall()
        if not filtered or len(rows) >= requested_k or fetch_k >= max_fetch:
            break
        fetch_k = min(max_fetch, fetch_k * 2)
    return rows[:requested_k]
