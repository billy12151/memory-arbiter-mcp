"""检索扩展层：_coerce_tags/_linked_open_items_for_search/_recent_fallback（从 search.py 搬出，拆分批 ⑥ 纯移动）。
search.py re-export 保活；_recent_fallback 由 search_memories 经 re-export 调用。"""
from __future__ import annotations

import sqlite3
from typing import Any, TYPE_CHECKING

from .acl import WorkspaceScope, workspace_exclusion_sql, workspace_scope_sql
from .db import MemoryDB, row_to_dict

if TYPE_CHECKING:
    pass


def _coerce_tags(raw: Any) -> list[str]:
    """v0.7.4: normalise a memory's ``tags`` field into a deduped ``list[str]``.

    Implementation lives in text.coerce_tags (Phase 1 single source); thin re-export
    here so existing imports keep working. Never raises — bad shapes yield [].
    """
    from .text import coerce_tags
    return coerce_tags(raw)


def _linked_open_items_for_search(
    db: MemoryDB,
    results: list[dict[str, Any]],
    warnings: list[str],
    max_items: int = 5,
    ws_canonical: "WorkspaceScope" = None,
    exclude_workspaces: "list[str] | set[str] | frozenset[str] | None" = None,
) -> list[dict[str, Any]]:
    """v0.7.4: attach up to ``max_items`` active todo memories that share
    meaningful tags with the current result set (linked_open_items).

    Pure read-only enhancement — never writes. Never raises: on any DB error
    returns [] and appends a degradation warning to ``warnings``.

    Three-layer short-circuit (design 性能设计):
      L0 (memory): bail without touching DB if results carry no meaningful tag
           (after stripping ``todo`` and single-char tags).
      L1 (DB): EXISTS check for any active memory tagged ``todo``; bail if none.
      L2 (DB): multiple SELECTs on one connection compute ``active_count``,
           per-tag ``df``, todo candidates, apply the M1 stoplist, score, sort,
           truncate. Note: this is a best-effort read, NOT a transactional
           snapshot — the bare SELECTs don't share a read snapshot under WAL,
           so concurrent writes can in principle make the count/df/candidates
           slightly inconsistent. Acceptable for an advisory side-hint; if
           consistency ever matters here, wrap the SELECTs in a read txn.

    Stoplist (M1 — uniform, independent of todo count):
      tag == 'todo' | len(tag) <= 1 | df >= 3 AND df/active_count >= 0.20

    ``json_valid(tags)`` is applied in SQL so malformed-tag rows are silently
    filtered (M4-A) — this does NOT produce a warning. Only a real DB failure
    produces a warning (M4-B).
    """
    if not results:
        return []

    # --- L0: collect meaningful tags from results (strip todo / single-char) ---
    result_id_to_tags: dict[int, set[str]] = {}
    all_meaningful: set[str] = set()
    for rec in results:
        rid = rec.get("id")
        tags = _coerce_tags(rec.get("tags"))
        meaningful = {t for t in tags if t != "todo" and len(t) > 1}
        if rid is not None and meaningful:
            result_id_to_tags[int(rid)] = meaningful
            all_meaningful |= meaningful
    if not all_meaningful or not db.db_available:
        return []

    result_ids = list(result_id_to_tags.keys())

    def _is_stoplisted(tag: str, df: int, active_count: int) -> bool:
        # M1: uniform stoplist — no todo-count branching.
        if tag == "todo":
            return True
        if len(tag) <= 1:
            return True
        if df >= 3 and active_count > 0 and df / active_count >= 0.20:
            return True
        return False

    try:
        conn = db._new_connection()
        scope_sql, scope_params = workspace_scope_sql(
            "COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace)", ws_canonical,
        )
        # v0.15.5: the blacklist applies to this whole-DB todo-attachment
        # channel too — otherwise blacklisted-bucket todos leak back into the
        # same unscoped find payload the recall pool just excluded them from.
        excl_m_sql, _, excl_params = workspace_exclusion_sql(exclude_workspaces)
        workspace_clause = (
            (f"AND {scope_sql} " if scope_sql else "") + (f"AND {excl_m_sql} " if excl_m_sql else "")
        )
        workspace_params: list[Any] = list(scope_params) + list(excl_params)
        try:
            # --- L1: EXISTS check for active+todo memories ---
            todo_exists = conn.execute(
                "SELECT EXISTS ("
                " SELECT 1 FROM memories m"
                " WHERE m.status='active' " + workspace_clause +
                "   AND EXISTS ("
                "     SELECT 1 FROM json_each("
                "       CASE WHEN json_valid(m.tags) THEN m.tags ELSE '[]' END"
                "     ) WHERE json_each.value='todo' AND json_each.type='text'"
                "   )"
                ") AS e",
                workspace_params,
            ).fetchone()["e"]
            if not todo_exists:
                return []

            # --- L2: active_count ---
            active_count = int(conn.execute(
                "SELECT COUNT(*) AS c FROM memories m WHERE m.status='active' " + workspace_clause,
                workspace_params,
            ).fetchone()["c"])
            if active_count <= 0:
                return []

            # todo candidates: active + tagged 'todo', excluding result IDs.
            ph = ",".join("?" * len(result_ids)) if result_ids else ""
            exclude_clause = f"AND m.id NOT IN ({ph})" if result_ids else ""
            cand_rows = conn.execute(
                f"""
                SELECT m.id, m.subject, m.tags, m.ingest_time
                FROM memories m
                WHERE m.status='active' {exclude_clause} {workspace_clause}
                  AND EXISTS (
                    SELECT 1 FROM json_each(
                      CASE WHEN json_valid(m.tags) THEN m.tags ELSE '[]' END
                    ) WHERE json_each.value='todo' AND json_each.type='text'
                  )
                """,
                list(result_ids) + workspace_params,
            ).fetchall()
            if not cand_rows:
                return []

            # per-tag df across the active set (json_valid guard ⇒ M4-A silence).
            # 0.16.12 P1-T6: cached per scope behind the global fingerprint
            # (COUNT(active), SUM(version)) — the L1/L2 queries above stay
            # live (they gate whether df is needed at all), only the GROUP BY
            # json_each full scan is memoised.
            tag_df: dict[str, int] = {}
            scope_key = tuple(str(p) for p in workspace_params)
            fp_row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(version),0) FROM memories WHERE status='active'"
            ).fetchone()
            fingerprint = (int(fp_row[0]), int(fp_row[1]))
            with db._linked_df_cache_lock:
                cached = db._linked_df_cache.get(scope_key)
            if cached is not None and cached[0] == fingerprint:
                tag_df = dict(cached[1])
            else:
                df_rows = conn.execute(
                    f"""
                    SELECT tag.value AS t, COUNT(DISTINCT m.id) AS df
                    FROM memories m, json_each(
                      CASE WHEN json_valid(m.tags) THEN m.tags ELSE '[]' END
                    ) AS tag
                    WHERE m.status='active' {workspace_clause} AND tag.type='text'
                    GROUP BY tag.value
                    """,
                    workspace_params,
                ).fetchall()
                for r in df_rows:
                    tag_df[r["t"]] = int(r["df"])
                with db._linked_df_cache_lock:
                    db._linked_df_cache[scope_key] = (fingerprint, dict(tag_df))

            scored: list[dict[str, Any]] = []
            for row in cand_rows:
                cand_id = int(row["id"])
                cand_tags = _coerce_tags(row["tags"])
                cand_subject = row["subject"] or f"memory #{cand_id}"
                cand_ingest = row["ingest_time"] or ""
                matched_meaningful: set[str] = set()
                matched_result_ids: set[int] = set()
                for tag in cand_tags:
                    if tag not in all_meaningful:
                        continue
                    if _is_stoplisted(tag, tag_df.get(tag, 0), active_count):
                        continue
                    matched_meaningful.add(tag)
                    for rid, rtags in result_id_to_tags.items():
                        if tag in rtags:
                            matched_result_ids.add(rid)
                # score: 2 per matched meaningful tag (≥ 2 ⇒ ≥ 1 overlap).
                score = 2 * len(matched_meaningful)
                if score < 2:
                    continue
                scored.append({
                    "id": cand_id,
                    "subject": cand_subject,
                    "tags": cand_tags,
                    "ingest_time": cand_ingest,
                    "reason": "tag_overlap: " + ", ".join(sorted(matched_meaningful)),
                    "matched_result_ids": sorted(matched_result_ids),
                    "_score": score,
                })
            if not scored:
                return []
            # score DESC → ingest_time DESC → id DESC.
            scored.sort(
                key=lambda x: (x["_score"], x["ingest_time"], x["id"]),
                reverse=True,
            )
            out: list[dict[str, Any]] = []
            for item in scored[:max_items]:
                item.pop("_score", None)
                out.append(item)
            return out
        finally:
            conn.close()
    except sqlite3.Error as exc:
        warnings.append(f"linked_open_items lookup failed: {exc}; returned [].")
        return []


def _recent_fallback(
    db: MemoryDB,
    workspace: str | None,
    tags: list[str] | None,
    limit: int,
    like_status_clause: str,
    warnings: list[str],
    offset: int = 0,
    ws_canonical: "WorkspaceScope" = None,
    exclude_workspaces: "list[str] | set[str] | frozenset[str] | None" = None,
) -> tuple[list[dict[str, Any]], list[str], bool, int]:
    """Recent-memory fallback when no direct match found (r4 §4.2 safety net)."""
    clauses = [like_status_clause]
    params: list[Any] = []
    for tag in tags or []:
        clauses.append("tags LIKE ?")
        params.append(f"%{tag}%")
    # strict isolation: filter to the caller's admitted workspace set INSIDE the
    # SQL so COUNT and the paginated window agree — a Python post-filter on an
    # already-paginated page reports a wrong total and can under-fill the page.
    # Match canonical with a raw fallback for rows written before the column
    # existed (workspace_canonical NULL → compare against raw workspace).
    scope_sql, scope_params = workspace_scope_sql(
        "COALESCE(NULLIF(workspace_canonical, ''), workspace)", ws_canonical,
    )
    if scope_sql:
        clauses.append(scope_sql)
        params.extend(scope_params)
    _, excl_sql, excl_params = workspace_exclusion_sql(exclude_workspaces)
    if excl_sql:
        clauses.append(excl_sql)
        params.extend(excl_params)
    conn = db._new_connection()
    try:
        count_row = conn.execute(
            f"SELECT COUNT(*) AS c FROM memories WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        total_estimate = int(count_row["c"] or 0) if count_row else 0
        rows = conn.execute(
            f"""SELECT *, 0 AS score FROM memories
                WHERE {' AND '.join(clauses)}
                ORDER BY
                  CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
                  CASE protection_level
                    WHEN 'locked' THEN 0
                    WHEN 'protected' THEN 1
                    ELSE 2
                  END,
                  CASE source_type
                    WHEN 'user_confirmed' THEN 0
                    WHEN 'document_extracted' THEN 1
                    ELSE 2
                  END,
                  confidence DESC,
                  ingest_time DESC,
                  event_time DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()
    finally:
        conn.close()
    # v0.15.9: the no-direct-match warning is gone with the query fallback —
    # the only remaining caller is explicit empty-query browsing, where
    # "no match" wording would be noise.
    has_more = total_estimate > offset + len(rows)
    return [row_to_dict(row) for row in rows], warnings, has_more, total_estimate
