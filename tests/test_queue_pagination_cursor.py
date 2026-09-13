"""0.16.2 pagination-cursor tests (live repro 2026-09-13): the weekly task
judged 67 of 125 workspace suspects, never reached a conflict pair, and
exited believing it was done — two protocol defects compounded:

1. next_page_token echo: only conflict groups advanced the cursor, so a
   workspace/internal page returned the CALLER's token unchanged (0 on the
   first page). A defensive agent reads a repeated token as a pagination
   loop and stops.
2. Head-blocking: workspace rows had no cursor, so rows the caller failed
   to judge (gate_failed stays pending) re-served as the same page head
   forever — pages made zero progress while has_more stayed true.

Fix: workspace rows join the id cursor; an empty page with remaining
backlog wraps to the head instead of echoing the token.
"""
from __future__ import annotations

import json
from pathlib import Path

from test_scan_pipeline import make_tools, _write


def _page(tools, **payload):
    return tools.memory_repair("scan_queue", {"action": "page", **payload})["data"]


def _seed_workspace_rows(tools, n: int) -> None:
    now = "2026-09-13T00:00:00+00:00"
    with tools.db.write_transaction() as conn:
        for i in range(n):
            conn.execute(
                """INSERT INTO scan_queue(kind,workspace_canonical,status,candidate_key_hash,
                     member_versions,evidence,reason,severity,source,detail,created_at,updated_at)
                   VALUES('workspace','ws','pending',?,?,?,?, 'normal','scan_pipeline',?, ?, ?)""",
                (
                    f"{i+1:064d}",
                    json.dumps([{"memory_id": 9000 + i, "version": 1}]),
                    "[]",
                    "vector vote 4/10 -> 'proja'",
                    json.dumps({"suspected_workspace": "proja", "current_workspace": "ws"}),
                    now, now,
                ),
            )


def test_workspace_pages_advance_the_cursor(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _seed_workspace_rows(tools, 25)
    seen: list[int] = []
    tokens: list[int] = []
    token = 0
    for _ in range(3):
        page = _page(tools, page_token=token)
        ws = [i for i in page["items"] if i["kind"] == "workspace"]
        assert ws, "seeded workspace rows must be served"
        seen.extend(i["memory_id"] for i in ws)
        assert page["next_page_token"] > token, "workspace pages must advance the cursor"
        tokens.append(page["next_page_token"])
        token = page["next_page_token"]
    assert len(set(seen)) == len(seen), "no row may repeat across cursor pages"
    assert len(tokens) == len(set(tokens)), "no token echo"


def test_unjudged_head_rows_do_not_block_pagination(tmp_path: Path) -> None:
    """gate_failed rows stay pending — the NEXT page must move past them,
    not re-serve the same head forever."""
    tools = make_tools(tmp_path)
    _seed_workspace_rows(tools, 25)
    first = _page(tools, page_token=0)
    head_ids = [i["memory_id"] for i in first["items"] if i["kind"] == "workspace"]
    # Judge NOTHING (the failure mode): page 2 must still show NEW rows.
    second = _page(tools, page_token=first["next_page_token"])
    second_ids = [i["memory_id"] for i in second["items"] if i["kind"] == "workspace"]
    assert second_ids and not (set(head_ids) & set(second_ids)), (
        "an unjudged head must not be re-served on the next cursor page"
    )


def test_empty_tail_wraps_to_head_instead_of_echoing(tmp_path: Path) -> None:
    """Cursor past every row while backlog remains: the page wraps to the
    head (fresh pass over survivors), never returns an empty page with the
    same token (the pagination-loop signal that ended the live run)."""
    tools = make_tools(tmp_path)
    _seed_workspace_rows(tools, 5)
    # Cursor far beyond every row id.
    page = _page(tools, page_token=10**6)
    assert page["count"] > 0, "wrap-around must re-serve remaining rows, not an empty page"
    ids = [i["memory_id"] for i in page["items"]]
    assert 9000 in ids, "wrapped page restarts from the head"
    if page["has_more"]:
        assert page["next_page_token"] != 10**6, "no token echo after wrap"
