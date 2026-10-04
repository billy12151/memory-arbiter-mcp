from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Literal

from .anchors import (
    STOP_ANCHORS,
    extract_anchors,
)
from .acl import (
    WorkspaceScope,
    scope_names,
    workspace_exclusion_sql,
    workspace_scope_sql,
)
from .db import MemoryDB, row_to_dict

# v0.7.4 (M2): retrieval_mode classifies how the returned rows were produced.
# linked_open_items only triggers on "direct" (a real query hit) — the other
# modes return browse/fallback/empty rows where injecting todos would be noise.
RetrievalMode = Literal[
    "direct",            # FTS/LIKE/evidence channels genuinely matched the query
    "recent_browse",     # empty query, no filters — caller is browsing recent
    "empty",             # filters yielded nothing, pool empty, or everything
                         # below the relevance floor (v0.15.9: recent-fallback
                         # removed — empty is honest, no recency stuffing)
    "unavailable",       # SQLite not available
]

from .constants import CONTENT_LIKE_CAP, COS_EXACT_BOOST, COS_RECALL_FLOOR, Isolation, QUERY_RECALL_SCORE_FLOOR, RECALL_POOL_CAP, SURFACE_ADMISSION_QUOTA, WORKSPACE_MIN_NAME_LEN, WORKSPACE_WEAK_VECTOR_WEIGHT, is_default_workspace_term
from .text import CJK_RE_SEARCH as _CJK_RE  # noqa: F401
from .search_text import (  # noqa: F401
    _subject_key as _subject_key,
    _is_cjk_token as _is_cjk_token,
    _split_cjk_token as _split_cjk_token,
    _quote_phrase as _quote_phrase,
    _sanitize_fts_query as _sanitize_fts_query,
    _normalize_token_for_tag_match as _normalize_token_for_tag_match,
    _cjk_substring_match as _cjk_substring_match,
    _is_pure_cjk_token as _is_pure_cjk_token,
    _is_short_cjk_keyword as _is_short_cjk_keyword,
    _sanitize_fts_query_or as _sanitize_fts_query_or,
    _parse_time as _parse_time,
    _sanitize_tags_filter as _sanitize_tags_filter,
    _passes_filters as _passes_filters,
    _query_non_cjk_dominant as _query_non_cjk_dominant,
)
from .search_scoring import (  # noqa: F401
    _trust_bonus as _trust_bonus,
    _parse_ingest_time as _parse_ingest_time,
    _ingest_sort_key as _ingest_sort_key,
    _recency_bonus as _recency_bonus,
    _workspace_bonus as _workspace_bonus,
    _score_surface as _score_surface,
    _score_tags_surface as _score_tags_surface,
    _RRF_K as _RRF_K,
    _RRF_SCORE_WEIGHT as _RRF_SCORE_WEIGHT,
    _SUBJECT_SCORE_CAP as _SUBJECT_SCORE_CAP,
    _TAGS_SCORE_CAP as _TAGS_SCORE_CAP,
    _CONTENT_SCORE_CAP as _CONTENT_SCORE_CAP,
    _TRUST_BONUS_USER_CONFIRMED as _TRUST_BONUS_USER_CONFIRMED,
    _TRUST_BONUS_DOCUMENT_EXTRACTED as _TRUST_BONUS_DOCUMENT_EXTRACTED,
    _TRUST_BONUS_DEFAULT as _TRUST_BONUS_DEFAULT,
    _LONG_CONTENT_PENALTY as _LONG_CONTENT_PENALTY,
    _CONTENT_ONLY_PENALTY as _CONTENT_ONLY_PENALTY,
    _VEC_FLOOR_SCORE as _VEC_FLOOR_SCORE,
    _SUBJECT_STRONG_WEIGHT as _SUBJECT_STRONG_WEIGHT,
    _SUBJECT_MEDIUM_WEIGHT as _SUBJECT_MEDIUM_WEIGHT,
    _SUBJECT_WEAK_WEIGHT as _SUBJECT_WEAK_WEIGHT,
    _TAGS_STRONG_WEIGHT as _TAGS_STRONG_WEIGHT,
    _TAGS_MEDIUM_WEIGHT as _TAGS_MEDIUM_WEIGHT,
    _TAGS_WEAK_WEIGHT as _TAGS_WEAK_WEIGHT,
    _RECENCY_BONUS_DEFAULT as _RECENCY_BONUS_DEFAULT,
    _RECENCY_THRESHOLDS as _RECENCY_THRESHOLDS,
    _WS_BONUS_SAME as _WS_BONUS_SAME,
    _WS_PENALTY_CROSS as _WS_PENALTY_CROSS,
    _soft_rerank as _soft_rerank,
    is_keyword_query as is_keyword_query,
    _apply_keyword_rescue as _apply_keyword_rescue,
)
from .search_extras import (  # noqa: F401
    _coerce_tags as _coerce_tags,
    _linked_open_items_for_search as _linked_open_items_for_search,
    _recent_fallback as _recent_fallback,
)


@dataclass
class SearchOutcome:
    """Structured search result with rows, warnings, pagination, and retrieval mode."""

    results: list[dict[str, Any]]
    warnings: list[str]
    has_more: bool
    # v0.15.4: None on the unfiltered query-recall path (no exact total
    # exists there; len(pool) was a recall count, not a total). Filtered
    # paths keep the exact SQL count.
    total_estimate: int | None
    retrieval_mode: RetrievalMode


# Single source: text.CJK_RE_SEARCH (Phase 1). Re-exported here for back-compat.
from .semantic_conflict import vector_cosine
# shared vector-admission helpers + the weak weighting curve.




def _wide_recall(
    db: MemoryDB,
    query: str,
    workspace: str | None,
    tags: list[str] | None,
    status_clause_m: str,
    like_status_clause: str,
    status_filter: str = "active",  # "active", "expired", "all"
    pool_cap: int = 50,
    content_like_fallback: bool = True,
    query_embedding: list[float] | None = None,
    content_like_cap: int = 30,
    ws_canonical: "WorkspaceScope" = None,
    exclude_workspaces: "list[str] | set[str] | frozenset[str] | None" = None,
) -> list[dict[str, Any]]:
    """v0.3.0 wide recall: merge multiple retrieval channels into a candidate pool.

    Channels (per r4 §6, revised 0.16.10 by vector availability):
      1. FTS top N (main, tokens AND'd)
      2. FTS OR-query top N (loosened — tokens OR'd; only if pool not yet full)
      3. subject/tags LIKE (precise surface recall; always runs)
      4. content LIKE — DEGRADED-ONLY since 0.16.10: runs only when vectors are
         unavailable (no query_embedding or no sqlite-vec), pool not yet full,
         with ≥2 anchor hits, capped. Ablated 2026-09-19 (recall-ch1235): zero
         relevant-target contribution on corpus recall-v1 while vectors work.
      5. evidence-vector KNN over `memory_evidence_vec` — optional, only when
         query_embedding provided and sqlite-vec available. Catches semantically similar but lexically
         dissimilar memories. Candidates are flagged so soft-rerank can give
         them a floor score (the query text didn't literally match anything).

    Returns dedup'd candidate pool (list of dict rows). Each row already has
    its raw fields; soft-rerank will add scoring fields.

    ``ws_canonical`` may be an admitted canonical set (one name or the
    strict in-radius neighbourhood). Every channel scopes in SQL — never a
    Python post-filter — so COUNT/pagination/df stay consistent and the
    admission genuinely widens recall rather than being a no-op over a pool the
    SQL already locked to one canonical.
    """
    if not db.db_available or not query:
        return []
    vector_available = bool(query_embedding) and bool(db.state.sqlite_vec_available)
    pool: dict[int, dict[str, Any]] = {}
    scope_m_sql, scope_params = workspace_scope_sql("COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace)", ws_canonical)
    scope_plain_sql, scope_plain_params = workspace_scope_sql("COALESCE(NULLIF(workspace_canonical, ''), workspace)", ws_canonical)
    # v0.15.5 recall blacklist: exclusion applies with or without a positive
    # scope (unscoped find is the headline case). Each channel embeds exactly
    # ONE cond (m-aliased for FTS joins, plain for the LIKE scans) and extends
    # its params once — params are per-cond, never a shared list.
    excl_m_sql, excl_plain_sql, excl_params = workspace_exclusion_sql(exclude_workspaces)
    workspace_clause_m = (f" AND {scope_m_sql}" if scope_m_sql else "") + (f" AND {excl_m_sql}" if excl_m_sql else "")
    workspace_params: list[Any] = list(scope_params) + list(excl_params)  # FTS: scope_m + excl_m
    conn = db._new_connection()
    try:
        # Channel 1+2: FTS main + OR. _sanitize_fts_query already OR-joins CJK
        # trigrams; for the OR channel we additionally try a loosened query that
        # only requires any single trigram/token to hit.
        if db.state.fts5_available:
            per_channel_cap = max(pool_cap, 30)
            # Main FTS query (AND across token groups).
            fts_main = _sanitize_fts_query(query)
            if fts_main:
                sql = f"""
                    SELECT m.*, bm25(memories_fts) AS score
                    FROM memories_fts
                    JOIN memories m ON memories_fts.rowid = m.id
                    WHERE memories_fts MATCH ? AND {status_clause_m}{workspace_clause_m}
                """
                params: list[Any] = [fts_main, *workspace_params]
                sql += f" ORDER BY CASE m.status WHEN 'superseded' THEN 1 ELSE 0 END, score LIMIT ?"
                params.append(per_channel_cap)
                try:
                    for row in conn.execute(sql, params).fetchall():
                        d = row_to_dict(row)
                        pool[d["id"]] = d
                except sqlite3.Error:
                    pass
            # OR channel: only if main didn't fill the pool. This catches the
            # "query was overspecified" case where AND'd trigrams miss.
            if len(pool) < pool_cap:
                fts_or = _sanitize_fts_query_or(query)
                if fts_or and fts_or != fts_main:
                    sql = f"""
                        SELECT m.*, bm25(memories_fts) AS score
                        FROM memories_fts
                        JOIN memories m ON memories_fts.rowid = m.id
                        WHERE memories_fts MATCH ? AND {status_clause_m}{workspace_clause_m}
                    """
                    params = [fts_or, *workspace_params]
                    sql += f" ORDER BY CASE m.status WHEN 'superseded' THEN 1 ELSE 0 END, score LIMIT ?"
                    params.append(per_channel_cap)
                    try:
                        for row in conn.execute(sql, params).fetchall():
                            d = row_to_dict(row)
                            if d["id"] not in pool:
                                pool[d["id"]] = d
                    except sqlite3.Error:
                        pass

        # Channel 3: subject/tags LIKE — precise surface recall. v0.15.9: this
        # channel ALWAYS runs (the old pool-full short-circuit let OR-noise
        # crowd out exact surface hits) and matches per-token in addition to
        # the whole query string — a multi-word query's full string can never
        # substring-match a subject, but its 法规名 token can. Function-word
        # tokens are excluded via STOP_ANCHORS so filler never manufactures
        # surface hits; strength separation stays with soft-rerank anchors.
        surface_toks = [t for t in query.split() if len(t) >= 2 and t not in STOP_ANCHORS]
        like_clauses: list[str] = []
        like_params: list[Any] = []
        if query:
            like_clauses.append("(subject LIKE ? OR tags LIKE ?)")
            like_params.extend([f"%{query}%", f"%{query}%"])
        for tok in surface_toks:
            if tok == query:
                continue
            like_clauses.append("(subject LIKE ? OR tags LIKE ?)")
            like_params.extend([f"%{tok}%", f"%{tok}%"])
        if like_clauses:
            clauses = [like_status_clause, "(" + " OR ".join(like_clauses) + ")"]
            params = list(like_params)
            for tag in tags or []:
                clauses.append("tags LIKE ?")
                params.append(f"%{tag}%")
            if scope_plain_sql:
                clauses.append(scope_plain_sql)
                params.extend(scope_plain_params)
            if excl_plain_sql:
                clauses.append(excl_plain_sql)
                params.extend(excl_params)
            params.append(pool_cap)
            sql = f"""SELECT *, 0 AS score FROM memories
                      WHERE {' AND '.join(clauses)}
                      ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
                               ingest_time DESC LIMIT ?"""
            try:
                for row in conn.execute(sql, params).fetchall():
                    d = row_to_dict(row)
                    if d["id"] not in pool:
                        # Surface hits enter the pool last (worst lexical
                        # ranks); the trim below reserves bounded seats so
                        # exact matches are never starved by fusion order.
                        d["_surface_candidate"] = True
                        pool[d["id"]] = d
            except sqlite3.Error:
                pass

        # Channel 4: content LIKE — a limited gap-filler. Requires ≥2 query anchors hit
        # (r4 §6.1) and is capped at 5-10 to avoid noise explosion. 0.16.10: only a
        # no-vector degradation channel — with vectors up it contributed zero
        # relevant-target recall on corpus recall-v1 (recall-ch1235 ablation).
        if content_like_fallback and len(pool) < pool_cap and not vector_available:
            # Only run if query has at least 2 anchors — otherwise the ≥2-anchor
            # gate can never be satisfied and we save the scan.
            q_anchors = extract_anchors(query)
            if len(q_anchors) >= 2:
                like_q = f"%{query}%"
                clauses = [like_status_clause, "content LIKE ?"]
                params = [like_q]
                for tag in tags or []:
                    clauses.append("tags LIKE ?")
                    params.append(f"%{tag}%")
                if scope_plain_sql:
                    clauses.append(scope_plain_sql)
                    params.extend(scope_plain_params)
                if excl_plain_sql:
                    clauses.append(excl_plain_sql)
                    params.extend(excl_params)
                params.append(content_like_cap)  # cap content-LIKE gap-fill (frozen constant CONTENT_LIKE_CAP)
                sql = f"""SELECT *, 0 AS score FROM memories
                          WHERE {' AND '.join(clauses)}
                          ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
                                   ingest_time DESC LIMIT ?"""
                try:
                    added = 0
                    for row in conn.execute(sql, params).fetchall():
                        d = row_to_dict(row)
                        if d["id"] not in pool:
                            # Mark as content_only candidate for soft-rerank awareness.
                            d["_content_only_candidate"] = True
                            pool[d["id"]] = d
                            added += 1
                            if added >= content_like_cap or len(pool) >= pool_cap:
                                break
                except sqlite3.Error:
                    pass
    finally:
        conn.close()

    # Freeze lexical rank before adding evidence. Evidence is an independent
    # bounded channel: it must run even when FTS/LIKE already filled the pool.
    lexical_rank = {int(memory_id): rank for rank, memory_id in enumerate(pool, 1)}

    # Local-text evidence KNN aggregates multiple evidence hits into one memory
    # candidate so long documents cannot occupy many result slots. The
    # ``query_embedding is not None`` conjunct looks redundant after
    # vector_available but is what lets mypy narrow the Optional here (a bare
    # bool alias carries no narrowing).
    if vector_available and query_embedding is not None:
        evidence_memory_cap = max(pool_cap, 10)
        # C4: the evidence channel rides ROW vectors (unit tables retired).
        # k widened cap*8→cap*16: rows are ~3.4x the units for the same
        # memory, so the same memory-level recall needs a deeper row window
        # (calibrated by R@10 + noise-recall gates, plan §3).
        evidence_rows = db.row_knn(
            query_embedding,
            k=evidence_memory_cap * 16,
            parent_status_filter=status_filter,
            workspace=ws_canonical,
            exclude_workspaces=exclude_workspaces,
        )
        by_memory: dict[int, dict[str, Any]] = {}
        for row in evidence_rows:
            rid = row.get("memory_id")
            if rid is None:
                continue
            mid = int(rid)
            entry = by_memory.setdefault(mid, {"row": row, "hits": []})
            distance = row.get("distance")
            try:
                score = 1.0 - float(distance) if isinstance(distance, (int, float)) else 0.0
            except (TypeError, ValueError):
                score = 0.0
            entry["hits"].append({
                "evidence_id": row.get("id"),
                "kind": row.get("kind"),
                "text": row.get("text"),
                "start_offset": row.get("start_offset"),
                "end_offset": row.get("end_offset"),
                # 0.17.0 hit_window + F1: the hit's row_index drives the ±N
                # window expansion, and the row's memory_version lets the
                # preview layer drop stale-version hits instead of slicing
                # the new content with old offsets (owner 2026-09-23). Both
                # ride the row_knn row (r.*); debug pages expose them
                # additively (established _evidence_hits contract).
                "row_index": row.get("row_index"),
                "row_version": row.get("memory_version"),
                "distance": distance,
                "score": score,
            })
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        best_hit_ids: dict[int, int] = {}
        for mid, entry in by_memory.items():
            hits = sorted(entry["hits"], key=lambda h: float(h.get("score") or 0.0), reverse=True)
            best = float(hits[0].get("score") or 0.0) if hits else 0.0
            best_evidence_id = hits[0].get("evidence_id") if hits else None
            if best_evidence_id is not None:
                best_hit_ids[mid] = int(best_evidence_id)
            support = 0.0
            seen_kinds: set[str] = set()
            for hit in hits[1:4]:
                kind = str(hit.get("kind") or "")
                if kind in seen_kinds:
                    continue
                seen_kinds.add(kind)
                support += max(0.0, float(hit.get("score") or 0.0) - 0.45)
            ranked.append((best + 0.08 * support, mid, entry["row"]))
        ranked.sort(reverse=True, key=lambda item: item[0])
        # Gate-v2 G2: true cosine for each memory's best row (the legacy
        # `1.0 - L2 distance` score goes negative on non-unit vectors and
        # #91's display read -67.5). One batched IN fetch for the per-memory
        # best rows only — the full window (cap*16 rows) would pull ~800
        # blobs per query for a value only the best row feeds.
        best_cosines: dict[int, float] = {}
        if best_hit_ids and query_embedding:
            best_vectors = db.evidence.row_vectors_for_ids(
                list(best_hit_ids.values()),
            )
            for mid, row_id in best_hit_ids.items():
                vector = best_vectors.get(row_id)
                if vector:
                    best_cosines[mid] = vector_cosine(list(query_embedding), vector)
        evidence_only: list[int] = []
        for evidence_rank, (_score, mid, row) in enumerate(
            ranked[:evidence_memory_cap], 1,
        ):
            lexical_row = pool.get(mid)
            d = dict(lexical_row or row)
            d["id"] = mid
            d["_vec_candidate"] = True
            d["_evidence_vec_candidate"] = True
            d["_evidence_rank"] = evidence_rank
            d["_evidence_hits"] = sorted(
                by_memory[mid]["hits"],
                key=lambda h: float(h.get("score") or 0.0),
                reverse=True,
            )
            best_cosine = best_cosines.get(mid)
            if best_cosine is not None:
                d["_evidence_best_score"] = best_cosine
            if lexical_row is None:
                evidence_only.append(mid)
            pool[mid] = d
        if evidence_only:
            # Evidence-only candidates must look like every other result row:
            # real memories columns (version, agent_id, source_ref, ...) with
            # evidence details confined to _evidence_hits — not raw KNN join
            # rows carrying evidence fields at the top level.
            try:
                conn = db._new_connection()
                try:
                    placeholders = ",".join("?" for _ in evidence_only)
                    mem_rows = {
                        int(r["id"]): row_to_dict(r)
                        for r in conn.execute(
                            f"SELECT * FROM memories WHERE id IN ({placeholders})",
                            evidence_only,
                        ).fetchall()
                    }
                finally:
                    conn.close()
            except sqlite3.Error:
                mem_rows = {}
            for mid in evidence_only:
                mem_row = mem_rows.get(mid)
                if mem_row is None:
                    pool.pop(mid, None)
                    continue
                preserved: dict[str, Any] = {
                    key: pool[mid][key]
                    for key in (
                        "_vec_candidate", "_evidence_vec_candidate",
                        "_evidence_rank", "_evidence_hits", "id",
                        # K1: 真余弦必须活过重建——G2 的余弦精确席与检索线
                        # 的中间带救济都消费它，而 evidence-only 行（两者
                        # 的目标人群）此前在这里被整体洗掉（先在缺陷：
                        # 既有 G2 余弦用例碰巧是双通道行才一直绿着）。
                        "_evidence_best_score",
                    )
                    if key in pool[mid]
                }
                pool[mid] = {**mem_row, **preserved}

    # Fuse channel ranks, then restore the original bounded pool size. A memory
    # present in both channels naturally receives more support than one present
    # in only one channel. Trust/recency remain later, lightweight adjustments.
    query_key = _subject_key(query)
    for memory_id, row in pool.items():
        lexical = lexical_rank.get(int(memory_id))
        evidence = row.get("_evidence_rank")
        fusion = 0.0
        if lexical is not None:
            fusion += 1.0 / (_RRF_K + lexical)
            row["_lexical_rank"] = lexical
            # Gate-v2 G2 transparency: the FTS channel's raw rank on the item
            # (the fusion arithmetic flattens rank differences into ~1/60
            # steps — #91 read evidence cosine 1.0 through lexical rank 15).
            row["lexical_rank"] = lexical
        if evidence is not None:
            fusion += 1.0 / (_RRF_K + int(evidence))
        # Gate-v2 G2 exact-hit guarantee (owner 拍板 6): a memory whose
        # subject IS the query (normalized: casefold + strip whitespace), or
        # whose best evidence row's TRUE cosine clears COS_EXACT_BOOST, is
        # what the user asked for — +1.0 fusion dwarfs every rank
        # computation (~300 final points) instead of competing with them.
        best_cosine = row.get("_evidence_best_score")
        if (
            (query_key and query_key == _subject_key(row.get("subject")))
            or (best_cosine is not None and best_cosine >= COS_EXACT_BOOST)
        ):
            fusion += 1.0
            row["_exact_match"] = True
            if best_cosine is not None:
                # Transparency: the evidence channel's real cosine on the item.
                row["evidence_best_score"] = round(float(best_cosine), 4)
        row["_fusion_score"] = fusion

    # 检索线 K2：向量准入线（方案 §3c，owner 2026-09-24 拍板 9）——
    # evidence-only 候选（无词法席位的纯向量行）的 best 行真余弦低于
    # COS_RECALL_FLOOR 的整条不进结果，词法候选豁免（维持 §1c 的
    # 排名+8.25 双保险）。仅 active 查询路径生效：expired 审计是
    # 宁滥勿缺的遍历语义，沿 8.25 门的既有豁免口径（R2-P1-10）。
    # 真余弦缺失（向量未发布/拉取失败）fail-open——与 G2 对 None 的
    # 处理一致，不误杀（R2-P1-4）。池内字典操作，零新增 SQL。
    if status_filter == "active":
        dropped = [
            memory_id
            for memory_id, row in pool.items()
            if memory_id not in lexical_rank
            and row.get("_evidence_best_score") is not None
            and float(row["_evidence_best_score"]) < COS_RECALL_FLOOR
        ]
        for memory_id in dropped:
            del pool[memory_id]

    fused = sorted(
        pool.values(),
        key=lambda row: (
            float(row.get("_fusion_score") or 0.0),
            -int(row.get("_lexical_rank") or 10**9),
        ),
        reverse=True,
    )
    if not lexical_rank or not any(row.get("_evidence_rank") for row in fused):
        return fused[:pool_cap]

    # Reserve bounded admission for both channels before the final soft rerank.
    # RRF alone gives a lexical-only and evidence-only candidate at the same
    # rank the same score, so deterministic tie-breaking could still starve one
    # channel when the other is full. The quotas guarantee representation while
    # keeping the candidate count exactly at pool_cap.
    lexical_quota = (pool_cap + 1) // 2
    evidence_quota = pool_cap - lexical_quota
    lexical_candidates = sorted(
        (row for row in fused if row.get("_lexical_rank") is not None),
        key=lambda row: int(row["_lexical_rank"]),
    )
    evidence_candidates = sorted(
        (row for row in fused if row.get("_evidence_rank") is not None),
        key=lambda row: int(row["_evidence_rank"]),
    )
    selected: dict[int, dict[str, Any]] = {}
    for row in lexical_candidates[:lexical_quota]:
        selected[int(row["id"])] = row
    for row in evidence_candidates[:evidence_quota]:
        selected[int(row["id"])] = row
    # Gate-v2 G2: exact hits are exempt from the quota arithmetic — both
    # quotas admit by ORIGINAL channel rank, and an exact match entering the
    # pool through a late channel (surface/LIKE) carries the worst ranks, so
    # without this seat it would be trimmed before its +1.0 fusion ever gets
    # to rerank (adversarial review P1: the fixture pool was too small to
    # catch this; the real library would have starved it).
    for row in fused:
        if row.get("_exact_match") and int(row["id"]) not in selected:
            selected[int(row["id"])] = row
    # v0.15.9: bounded reserved seats for channel-3 surface hits — without
    # this, exact subject/tags matches starve in the fusion-order trim because
    # they always carry the worst lexical ranks (they enter the pool last).
    surface_candidates = sorted(
        (row for row in fused
         if row.get("_surface_candidate") and int(row["id"]) not in selected),
        key=lambda row: int(row.get("_lexical_rank") or 10**9),
    )
    for row in surface_candidates[:SURFACE_ADMISSION_QUOTA]:
        selected[int(row["id"])] = row
    if len(selected) < pool_cap:
        for row in fused:
            selected.setdefault(int(row["id"]), row)
            if len(selected) >= pool_cap:
                break
    return list(selected.values())

def _passes_query_recall_floor(row: dict[str, Any], alloglottic: bool = False) -> bool:
    """0.17.0 分层门槛（owner 2026-09-26 拍板）：复合分线只管词法锚定候选，
    evidence-only 纯向量行改由余弦线把守。

    背景：8.25 复合线系中文个人库标定，而跨语言命中（en 查询→中文记忆）与
    改述查询天然 evidence-only——跨语种字面零重叠使词法通道零贡献、复合分
    系统性偏低（recall-v3-len en→zh 15 个 gold 复合 7.51-8.22 贴线被误杀，
    余弦 0.509-0.681 全在 COS_RECALL_FLOOR 之上）。evidence-only 行的向量
    质量已由 K2 向量准入线（同值 COS_RECALL_FLOOR）把守：进池与放行共用
    一把余弦尺，不再被复合线二道惩罚；词法锚定候选（FTS/surface，
    ``_lexical_rank`` 非 None）维持复合线——legal-form 噪音防线不变。
    豁免仅在查询非 CJK 主导（alloglottic，由 _query_non_cjk_dominant 判定）
    时启用——主考卷实测 zh 查询豁免 0 gold/51 噪音（legal 防线必须保留），
    en 查询豁免 15 gold；双语料数据锚：eval/results/xlang-floor-policies.json
    与 eval/results/recall-lf-fields.json 席位扫描；LOCOMO 同类反事实见
    WorkBuddy mema-vs-mem0 消融报告。
    """
    if float(row.get("_final_score") or 0.0) >= QUERY_RECALL_SCORE_FLOOR:
        return True
    if not alloglottic:
        return False
    if row.get("_lexical_rank") is not None:
        return False
    ev = row.get("_evidence_best_score")
    return ev is not None and float(ev) >= COS_RECALL_FLOOR


def search_memories(
    db: MemoryDB,
    query: str,
    workspace: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
    status_filter: str = "active",  # "active", "expired", "all" ("superseded" → "expired")
    debug_ranking: bool = False,
    query_embedding: list[float] | None = None,
    # v0.7.3 additions (design §3.1) — all optional, omit == v0.7.2 behaviour
    tags_filter: list[str] | None = None,
    after_time: str | None = None,
    before_time: str | None = None,
    source_type: str | None = None,
    offset: int = 0,
    ws_canonical: str | None = None,
    isolation: str = "none",
    hard_scope: bool = False,
    ws_scope: "WorkspaceScope" = None,
    exclude_workspaces: "list[str] | set[str] | frozenset[str] | None" = None,
    keep_evidence_hits: bool = False,
) -> SearchOutcome:
    """v0.9.4: returns a SearchOutcome with retrieval_mode.

    has_more/total_estimate give the caller a way to tell exhaustive queries
    ("all release notes") from complete ones. retrieval_mode drives
    linked_open_items triggering (only "direct" is eligible). v0.15.4: on the
    unfiltered active (find) query-recall path total_estimate is None and
    has_more is False — len(pool) was a capped recall count, not a total, and
    deep paging is discouraged in favour of rewording the query or adding
    filters. The expired audit path keeps the best-effort pool count and
    has_more inference: cursor pagination is its intended use.

    v0.9.4: ``offset`` enables cursor pagination. On the empty-query+filters
    path it maps to SQL OFFSET (exact, backed by count_filtered_memories). On
    the query-recall path it widens the candidate pool to cover the offset
    window — best-effort, since relevance-ranked recall has no exact total and
    the pool cap bounds the reachable depth (deep pages may return empty).

    ``ws_scope`` is the strict caller's admitted canonical set (its own
    plus in-radius neighbours). It replaces the single-canonical strict scope in
    the recall SQL; with vector admission off it is just ``(ws_canonical,)`` so
    the SQL is byte-identical to v0.12.5. ``ws_canonical`` stays the single
    center used for weak weighting and the empty-scope fallback.
    """
    warnings: list[str] = []
    if not db.db_available:
        return SearchOutcome([], ["SQLite unavailable; search cannot read JSONL backup in MVP."], False, 0, "unavailable")
    limit = max(1, min(int(limit), 100))
    offset = max(0, min(int(offset), 10000))
    query = (query or "").strip()
    # An explicit none-mode workspace filter (hard_scope) scopes recall in the
    # SQL itself so the limit is applied AFTER scoping; strict always scopes.
    # under strict the scope is the admitted canonical set (falls back
    # to the single canonical when no set was supplied / admission is off).
    if hard_scope and ws_canonical:
        scope_ws: "WorkspaceScope" = ws_canonical
    elif isolation == Isolation.STRICT and ws_canonical:
        scope_ws = ws_scope if ws_scope else ws_canonical
    else:
        scope_ws = None
    # v0.15.5 recall blacklist: only the UNSCOPED ambient pool is filtered —
    # an explicit workspace (incl. a blacklisted one) or a strict admitted
    # set overrides it; filter-driven recall below is explicit by construction.
    if scope_ws is not None:
        exclude_workspaces = None
    # v0.3.1: when a query_embedding is supplied but sqlite-vec is not active,
    # warn so the caller knows the semantic channel was silently skipped.
    if query_embedding and not db.state.sqlite_vec_available:
        warnings.append("query_embedding provided but sqlite-vec unavailable; semantic recall skipped.")
    # Status filter: active (default), expired (non-active non-deleted), or all.
    # "superseded" is accepted as a back-compat alias for "expired" (v0.9.4
    # widened the expired domain from superseded-only to all non-active
    # non-deleted, covering conflicted/pending for audit recall).
    if status_filter == "superseded":
        status_filter = "expired"
    if status_filter == "active":
        status_clause = "m.status = 'active'"
        like_status_clause = "status = 'active'"
    elif status_filter == "expired":
        status_clause = "m.status NOT IN ('active','deleted')"
        like_status_clause = "status NOT IN ('active','deleted')"
    else:  # "all"
        status_clause = "m.status != 'deleted'"
        like_status_clause = "status != 'deleted'"

    # === v0.7.3: parse + sanitize filter params (design §3.5 第一步) ===
    after_dt = _parse_time(after_time) if after_time else None
    if after_time and after_dt is None:
        warnings.append(f"after_time={after_time!r} invalid ISO 8601; ignored")
        after_time = None
    before_dt = _parse_time(before_time) if before_time else None
    if before_time and before_dt is None:
        warnings.append(f"before_time={before_time!r} invalid ISO 8601; ignored")
        before_time = None
    # D4: after > before 矛盾检查（严格 >；== 是单点区间，合法）
    if after_dt and before_dt and after_dt > before_dt:
        warnings.append(
            f"after_time ({after_dt.replace(microsecond=0).isoformat()}) > before_time "
            f"({before_dt.replace(microsecond=0).isoformat()}); interval is empty; both ignored"
        )
        after_dt = None
        before_dt = None
        after_time = None
        before_time = None
    tags_filter = _sanitize_tags_filter(tags_filter)
    has_filters = bool(tags_filter or after_time or before_time or source_type)

    # === v0.7.3 F1/C1: search.py empty-query shortcut ===
    # 现状是 `if not query: return _recent_fallback(...)`，会让 query 为空时
    # 无条件走 fallback——即使 has_filters=True 也跳过 post-filter，返回未
    # 过滤的最近记忆。改成 query 空 且 无过滤 才短路；query 空 + 有过滤
    # 继续往下（wide_recall 内部仍会因 not query 返 []，post-filter 后仍空，
    # 最终走第二步的 "query required for filter-aware recall" 精准 warning）。
    if not query and not has_filters:
        fb_ws = scope_ws
        fb_rows, fb_warnings, fb_hm, fb_te = _recent_fallback(
            db, workspace, tags, limit, like_status_clause, warnings, offset=offset, ws_canonical=fb_ws,
            exclude_workspaces=exclude_workspaces,
        )
        return SearchOutcome(fb_rows, fb_warnings, fb_hm, fb_te, "recent_browse")

    # === G6 (v0.8.5): empty query + filters → filter-driven recall ===
    # query 为空但带了 tags_filter / 时间 / source_type：不再走 wide_recall
    # （它会因 not query 返 []），而是直接按 filter 召回、ingest_time 倒序。
    # 解锁 list-by-tag / by-source_type / by-time。v0.9.4 adds SQL OFFSET for
    # expired audit pagination.
    if not query and has_filters:
        strict_ws_filter = scope_ws
        rows = db.recall_by_filters(
            like_status_clause, tags_filter, after_dt, before_dt, source_type, limit, offset,
            ws_canonical=strict_ws_filter,
        )
        total_estimate = db.count_filtered_memories(
            like_status_clause, tags_filter, after_dt, before_dt, source_type,
            ws_canonical=strict_ws_filter,
        )
        if not rows:
            warning = "offset beyond result set" if total_estimate > 0 and offset >= total_estimate else "no memories match the given filters"
            return SearchOutcome([], warnings + [warning], False, total_estimate, "empty")
        results = rows[:limit]
        has_more = total_estimate > offset + len(results)
        # No _soft_rerank on this branch (empty query) and recall_by_filters returns
        # bare SELECT * rows, so there are no _-prefixed debug fields to strip.
        return SearchOutcome(results, warnings, has_more, total_estimate, "direct")

    # === v0.7.3: pool 组装 + post-filter（design §3.5 第三步） ===
    # v0.9.4: when paginating (offset > 0), widen the recall pool past the
    # requested window by one row so the offset window stays reachable as a
    # best-effort page (v0.15.4: this path reports has_more=False /
    # total_estimate=None, so the widening now serves the window itself, not
    # a has_more inference). At offset=0 the original
    # pool_cap is preserved (keeps the pool-saturation / Channel-6 skip
    # semantics intact). Query-recall still has no exact total; the empty-query
    # SQL paths above are the precise pagination paths.
    base_pool_cap = RECALL_POOL_CAP
    pool_cap = max(base_pool_cap, offset + limit + 1) if offset > 0 else base_pool_cap
    pool = _wide_recall(db, query, workspace, tags, status_clause, like_status_clause,
                        status_filter=status_filter, query_embedding=query_embedding,
                        pool_cap=pool_cap,
                        content_like_cap=CONTENT_LIKE_CAP,
                        ws_canonical=scope_ws,
                        exclude_workspaces=exclude_workspaces)

    # v0.9.7: strict isolation — hard-filter the candidate pool to the query's
    # workspace. weak does NOT filter (it only nudges ranking in _soft_rerank);
    # none ignores workspace entirely.
    # the SQL above already scoped every channel to the admitted set,
    # so this is a defense-in-depth membership check over the SAME set — never
    # the narrower single canonical, which would silently undo the admission.
    if isolation == Isolation.STRICT and ws_canonical:
        admitted_set = set(scope_names(scope_ws)) or {ws_canonical}
        pool = [
            r for r in pool
            if (r.get("workspace_canonical") or r.get("workspace")) in admitted_set
        ]

    if has_filters:
        pool = [r for r in pool if _passes_filters(r, tags_filter, after_dt, before_dt, source_type)]
        if not pool:
            # 有过滤但召回空：返回空结果，不走 fallback（fallback 会返回不
            # 符合过滤条件的记忆，违反语义）。区分两种空因给出精准 warning。
            if not query:
                empty_reason = (
                    "query required for filter-aware recall; tags_filter/after_time/"
                    "before_time/source_type only post-filter query-recalled candidates "
                    "(see §8 risk 9)"
                )
            else:
                empty_reason = "filters too restrictive or no matches; pool was empty after post-filter"
            return SearchOutcome([], warnings + [empty_reason], False, 0, "empty")
    else:
        # v0.15.9: recent-fallback removed (mema 923 §5). A non-empty query
        # that recalls nothing returns an honest empty result with an
        # actionable hint — recency-stuffed results poisoned agent answers
        # (downstream legal queries got unrelated recent memories). strict
        # keeps its own, more specific message; empty-query browsing
        # (recent_browse) is untouched — it is explicit intent, not a fallback.
        if not pool:
            if isolation == "strict" and ws_canonical:
                return SearchOutcome(
                    [], warnings + ["no same-workspace match; strict isolation does not fall back to recent memories"],
                    False, 0, "empty",
                )
            return SearchOutcome(
                [], warnings + [
                    "no memories match the query; reword the query or add tags_filter "
                    "(empty-query find still browses recent memories)",
                ],
                False, 0, "empty",
            )

    # precompute the canonical distance map once, after the pool
    # is assembled, so the weak-isolation rerank can weight on real vector
    # distance without any scoring leaf touching the DB. Read-only; a default
    # query canonical never enters the vector system; degradation returns an
    # empty map (scoring falls back to the binary step per record).
    weak_distance_map: dict[str, float] | None = None
    if (
        isolation == "weak"
        and ws_canonical
        and not is_default_workspace_term(ws_canonical)
        and WORKSPACE_WEAK_VECTOR_WEIGHT
        and pool
    ):
        pool_canonicals = {
            str(r.get("workspace_canonical") or r.get("workspace") or "").strip()
            for r in pool
        }
        pool_canonicals.discard("")
        if pool_canonicals:
            weak_distance_map = db.workspaces.canonical_distance_map(
                ws_canonical, pool_canonicals,
            )

    # 检索线 K1：关键词模式查询的中间带救济（方案 §3b）——池内内存
    # 操作，先于 _soft_rerank 生效；非关键词查询零成本直返。
    _apply_keyword_rescue(query, pool)
    reranked = _soft_rerank(
        query, pool,
        ws_canonical=ws_canonical, isolation=isolation,
        distance_map=weak_distance_map,
        ws_min_name_len=WORKSPACE_MIN_NAME_LEN,
    )
    # v0.15.9 page floor: on the ACTIVE query-recall path, below-floor
    # candidates never reach the page (宁缺毋滥). Expired audit recall keeps
    # everything — its purpose is exhaustive review, not relevance.
    # 0.17.0 分层门槛：词法锚定候选走复合分线，evidence-only 纯向量行走
    # COS_RECALL_FLOOR 余弦线（与 K2 准入线同值——见
    # _passes_query_recall_floor docstring）。
    if status_filter == "active":
        pre_floor_count = len(reranked)
        alloglottic = _query_non_cjk_dominant(query) if query else False
        reranked = [r for r in reranked if _passes_query_recall_floor(r, alloglottic)]
        if pre_floor_count and not reranked:
            warnings.append(
                f"no candidates reached the relevance floor ({QUERY_RECALL_SCORE_FLOOR:g}); "
                "reword the query or add tags_filter"
            )
    # Slice to the requested page window.
    page = reranked[offset:offset + limit]

    # === v0.7.3: has_more / total_estimate（design §3.6 E1）；v0.15.4 修订 ===
    # 有过滤场景走 count_filtered_memories（SQL 全表按过滤计数，精确）。
    # v0.15.4: 无过滤 active（find）场景 total_estimate 报 None —— len(pool)
    # 只是 query 召回数（受 pool_cap 截断），报成"总数"语义误导；has_more 同
    # 步置 false，top 页未命中应换词/加 tags_filter，而不是深翻页。expired
    # 审计路径不套此语义：它的用途就是带 next_offset 光标的审计遍历，保留
    # 原 best-effort pool 计数与 has_more 推断。
    total: int | None
    if has_filters:
        total = db.count_filtered_memories(
            like_status_clause, tags_filter, after_dt, before_dt, source_type,
            ws_canonical=scope_ws,
        )
        # K1: has_more = total > offset + len(page). 修了原公式 len==limit and total>limit
        # 在 pool 召回不足时漏报（reranked<limit 但 total>reranked 应判 True）。
        has_more = total > offset + len(page)
    elif status_filter == "active":
        total = None
        has_more = False
    else:
        total = len(pool)
        has_more = total > offset + len(page)

    # hybrid mode: strip debug fields unless explicitly requested.
    # v0.15.10: keep_evidence_hits exempts _evidence_hits from the strip so the
    # find preview layer can build hit_spans (content_mode="hits"); the preview
    # builder consumes and removes it before the response leaves the pipeline,
    # so the internal field never reaches the wire.
    if not debug_ranking:
        for r in page:
            for k in list(r.keys()):
                if k.startswith("_") and not (keep_evidence_hits and k == "_evidence_hits"):
                    r.pop(k, None)
    return SearchOutcome(page, warnings, has_more, total, "direct")
