"""检索域 mixin（memory_search/_auto_embed/_search_scope_context/memory_batch_find，
从 read.py 搬出，拆分批 ④ 纯移动）。memory_search_expired 留守 read.py（读域同族）。
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..acl import CallerWorkspace
from ..embedder import ManagedEmbedder
from ..constants import EMBED_PREFIX_SEARCH, EMBEDDING_MAX_SECTION_CHARS
from ..tokens import meter_payloads
from ..constants import BATCH_READ_FULL_BUDGET_BYTES
from ._read_hits import (
    _CONTENT_MODES,
    _coerce_hit_window,
    _preview_item,
    _stale_hit_spans_warning,
    vec_disabled_warning,
    _STRONG_CONFLICT_SOURCES,
)

if TYPE_CHECKING:
    from ..config import Settings
    from ..db import MemoryDB
    from ..tools import MemoryTools


class _ReadSearch:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"
        settings: "Settings"
        _embedder_warnings: list[str]

        # 主类委托薄层/留守成员的 mypy strict 声明
        def _attach_conflict_signals(self, *args: Any, **kwargs: Any) -> Any: ...
        def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace": ...
        def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _get_memory_visible(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _strict_acl_unavailable(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _vector_lag(self) -> dict[str, int]: ...
        @staticmethod
        def _search_memories(*args: Any, **kwargs: Any) -> Any: ...
        @staticmethod
        def _compare_memories(*args: Any, **kwargs: Any) -> Any: ...
        @staticmethod
        def _linked_open_items_for_search(*args: Any, **kwargs: Any) -> Any: ...

    def memory_search(self, query: str = "", workspace: str | None = None, tags: list[str] | None = None, limit: int = 10, offset: int = 0, debug_ranking: bool = False, query_embedding: list[float] | None = None, tags_filter: list[str] | None = None, after_time: str | None = None, before_time: str | None = None, source_type: str | None = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True, include_size: bool | None = None, content_mode: str = "preview", hit_window: int = 0, **_: Any) -> dict[str, Any]:
        extra_warnings = list(self._embedder_warnings)
        if include_size is not None:
            # v0.15.6: the size block is one global config key covering every
            # recall surface; the old per-call flag is accepted (registry
            # compatibility) but only earns a pointer to the knob.
            extra_warnings.append(
                "include_size is a global config key since v0.15.6 (default true) "
                "governing find/read/expired/history together; per-call value ignored"
            )
        if "include_content" in _:
            # v0.15.10 breaking: the boolean was replaced by the content_mode
            # enum. Direct callers get the migration pointer here; the
            # validation boundary rejects it for product calls first.
            return self.db.state.response(
                {
                    "error": 'include_content was removed in v0.15.10; use content_mode="full" for full text or content_mode="hits" for vector-hit spans instead',
                    "results": [],
                    "count": 0,
                },
                ok=False,
            )
        if content_mode not in _CONTENT_MODES:
            return self.db.state.response(
                {
                    "error": 'content_mode must be one of "preview" | "hits" | "full" (default "preview")',
                    "results": [],
                    "count": 0,
                },
                ok=False,
            )
        if "include_superseded" in _:
            return self.db.state.response(
                {
                    "error": "include_superseded was removed in v0.9.4; memory_search is active-only. Use memory_search_expired for expired history/audit recall (non-active non-deleted: superseded, conflicted, pending). The old mixed active+superseded mode is gone.",
                    "results": [],
                    "count": 0,
                },
                ok=False,
            )
        # 0.17.0 hit_window: hits-mode-only knob (silently ignored beside the
        # other modes, matching the limit_per_query convention). Placed after
        # the early-error returns so a clamp warning survives into the response.
        hit_window_value = (
            _coerce_hit_window(hit_window, extra_warnings)
            if content_mode == "hits" else 0
        )
        query_embedding = self._auto_embed(query, query_embedding, extra_warnings)
        ctx = self._search_scope_context(workspace, extra_warnings)
        isolation = ctx["isolation"]
        caller = ctx["caller"]
        ws_canonical = ctx["ws_canonical"]
        workspace = ctx["workspace"]
        hard_scope = ctx["hard_scope"]
        ws_scope = ctx["ws_scope"]
        exclude_ws = ctx["exclude_ws"]
        if isolation == "strict" and not ws_canonical:
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                data = denied.get("data") or {}
                data.update({"results": [], "count": 0})
                return denied
        # v0.9.4: search_memories now uses status_filter instead of include_superseded
        outcome = self._search_memories(
            self.db, query, workspace, tags, limit,
            status_filter="active",  # Default: active only
            offset=offset,
            debug_ranking=debug_ranking,
            query_embedding=query_embedding,
            tags_filter=tags_filter,
            after_time=after_time,
            before_time=before_time,
            source_type=source_type,
            ws_canonical=ws_canonical,
            isolation=isolation,
            hard_scope=hard_scope,
            ws_scope=ws_scope,
            exclude_workspaces=exclude_ws,
            keep_evidence_hits=(content_mode == "hits" and not debug_ranking),
        )
        results = outcome.results
        warnings = outcome.warnings
        has_more = outcome.has_more
        total_estimate = outcome.total_estimate
        retrieval_mode = outcome.retrieval_mode
        # 0.16.12 P1-T3: ONE open-conflict-group query for the page feeds both
        # the signal attachment below and the unresolved_conflict_count segment
        # further down (previously two identical calls). Computed only on a
        # healthy DB so the down-DB warning semantics of each consumer stay
        # byte-identical (both fall back to their own query on None).
        shared_groups: "list[dict[str, Any]] | None" = None
        if results and self.db.db_available:
            _page_ids = sorted({int(r["id"]) for r in results if r.get("id") is not None})
            if _page_ids:
                try:
                    shared_groups = self.db.conflicts.list_open_conflicts_for_memory_ids(
                        _page_ids, include_applying=True,
                    )
                except Exception:
                    # Both consumers re-query inside their own try/except when
                    # handed None, so their per-site failure warnings stay
                    # byte-identical to the pre-dedup behaviour.
                    shared_groups = None
        # v0.7.6: attach conflict signals (open_table / conflict_guidance
        # sources), only on genuine query hits (direct mode). Failures degrade
        # silently.
        if include_conflict_signal and retrieval_mode == "direct" and results:
            results = self._attach_conflict_signals(
                results, extra_warnings, precomputed_groups=shared_groups,
            )
        # v0.8.7: promote conflict_signal to a loud top-level flag (mirrors the
        # write path's attention_required). If any direct hit carries a
        # conflict_signal, surface a one-line summary at data top level so the
        # calling agent notices it on a quick scan instead of having to inspect
        # each result's nested conflict_signal.
        attention_required = False
        attention_summary: str | None = None
        if include_conflict_signal and retrieval_mode == "direct" and results:
            # Distinct conflict_signal sources on these hits (source -> first
            # result carrying it): each source is logged once, and the loud
            # flag can be gated by source.
            seen_sources: dict[str, dict[str, Any]] = {}
            for r in results:
                sig = r.get("conflict_signal")
                if not sig:
                    continue
                seen_sources.setdefault(str(sig.get("conflict_source", "conflict")), r)
            # v0.8.8: log every source that appeared (doctor reports volume by
            # source, so advisory flooding stays visible even when it doesn't
            # ring the loud flag below).
            for src, r in seen_sources.items():
                sig = r.get("conflict_signal") or {}
                peer = sig.get("conflict_peer") or {}
                ids = [int(r["id"])] if r.get("id") is not None else []
                if isinstance(peer, dict) and peer.get("id") is not None:
                    ids.append(int(peer["id"]))
                self.db.log_attention(trigger="search", source=src, memory_ids=ids)
            # v0.8.8: the loud must-surface flag fires ONLY for verified
            # open_table / conflict_guidance signals (formally recorded
            # conflicts). A loud flag on weaker sources would nag, so those
            # stay a per-result signal for the calling agent to judge by
            # content: surface only if the two genuinely contradict, else
            # silently proceed.
            ot = next((seen_sources.get(source) for source in _STRONG_CONFLICT_SOURCES if seen_sources.get(source)), None)
            if ot is not None:
                attention_required = True
                ot_sig = ot.get("conflict_signal") or {}
                head = f"Search hit #{ot.get('id')}"
                if ot.get("subject"):
                    head += f" ({ot['subject']})"
                source_label = ot_sig.get("conflict_source") or "open_table"
                head += f" carries a {source_label} signal"
                peer = ot_sig.get("conflict_peer") or {}
                if isinstance(peer, dict) and peer.get("id") is not None:
                    peer_txt = f"#{peer['id']}"
                    if peer.get("subject"):
                        peer_txt += f" ({peer['subject']})"
                    head += f" vs {peer_txt}"
                n = sum(1 for x in results if (
                    (x.get("conflict_signal") or {}).get("conflict_source") in
                    _STRONG_CONFLICT_SOURCES
                ))
                if n > 1:
                    head += f" and {n - 1} more"
                attention_summary = head
        # v0.7.4: linked_open_items — only on genuine query hits (direct mode),
        # never on browse/empty. sqlite failures degrade to [] + warning.
        linked: list[dict[str, Any]] = []
        if include_linked_open_items and retrieval_mode == "direct" and results:
            # G6 (empty query + filters) is an explicit, curated query — its
            # linked attachments follow the same exemption as its results.
            _explicit_filter_path = not query and bool(
                tags_filter or after_time or before_time or source_type)
            linked = self._linked_open_items_for_search(
                self.db, results, extra_warnings,
                ws_canonical=ws_scope,
                exclude_workspaces=None if _explicit_filter_path else exclude_ws,
            )
        # v0.15.4: find is an index page. Every item carries content_chars +
        # a bounded outline (offsets share read's span coordinate system);
        # v0.15.10: content_mode picks the content depth — preview (default),
        # hits (vector-hit spans, full-text upgrade at >=50% coverage), full.
        # P1-T5: one batched outline prefetch for the whole page (preview /
        # hits modes) keeps the per-item outline off the per-item connection.
        _outline_map: dict[int, list[dict[str, Any]]] = {}
        if content_mode in {"preview", "hits"} and results:
            _outline_map = self.db.evidence.outline_rows_for_ids([
                (int(r["id"]), int(r.get("version") or 1))
                for r in results if r.get("id") is not None
            ])
        # 0.17.0 hit_window: ONE range-limited batch prefetch for the whole
        # page (the ±N neighbour rows around each item's evidence hits), at
        # the same point as the outline prefetch — never per item.
        _window_map: dict[int, list[dict[str, Any]]] = {}
        if content_mode == "hits" and hit_window_value > 0 and results:
            _window_entries: list[tuple[int, int, int, int]] = []
            for r in results:
                indexes = [
                    int(h["row_index"]) for h in (r.get("_evidence_hits") or [])
                    if isinstance(h, dict)
                    and isinstance(h.get("row_index"), int)
                    and not isinstance(h.get("row_index"), bool)
                ]
                if indexes and r.get("id") is not None:
                    _window_entries.append((
                        int(r["id"]), int(r.get("version") or 1),
                        min(indexes) - hit_window_value,
                        max(indexes) + hit_window_value,
                    ))
            if _window_entries:
                _window_map = self.db.evidence.row_spans_for_ids(_window_entries)
        results = [
            _preview_item(
                r, content_mode=content_mode, db=self.db,
                outline_rows=_outline_map.get(int(r["id"]), []),
                hit_window=hit_window_value,
                window_rows=_window_map.get(int(r["id"])),
            ) for r in results
        ]
        if content_mode == "hits":
            # F1 (owner 2026-09-23): every dropped stale-version hit must tell
            # the agent the evidence index lags and a re-query is needed.
            for r in results:
                stale = r.get("stale_hit_spans")
                if isinstance(stale, dict) and r.get("id") is not None:
                    extra_warnings.append(_stale_hit_spans_warning(int(r["id"]), stale))
        response_data = {
            "results": results,
            "count": len(results),
            # v0.7.3: exhaustive-query support (design §3.6)
            "has_more": has_more,
            "total_estimate": total_estimate,
            # v0.7.4 (M2): expose retrieval_mode so callers know how rows were produced.
            "retrieval_mode": retrieval_mode,
            # v0.7.4: related active todos, separated from the ranking engine.
            "linked_open_items": linked,
            "query_domain": "active",
            # vNext §13.1: async evidence index lag, never pretend strong consistency.
            "vector_lag": self._vector_lag(),
        }
        if self.settings.include_size:
            # v0.15.4: size metering measures the page as actually returned
            # (post-preview), so an index page reads as small and a
            # content_mode="full" page reads as the full text it carries.
            # v0.15.6: gated by the global config key (find/read/expired/
            # history share one switch), not a per-call flag.
            size_block = meter_payloads(results)
            returned_tokens = size_block["tokens_estimate"]
            display_hint = None
            if results:
                page_cost = (
                    f"~{returned_tokens} tokens returned "
                    f"for {len(results)} item{'s' if len(results) != 1 else ''}"
                )
                if retrieval_mode != "direct":
                    # Browse pages carry an exact total and has_more —
                    # paging through them is the intended use, so the
                    # "don't deep-page" guidance would contradict the signal.
                    display_hint = (
                        f"find browse page ({page_cost}): items are index-page "
                        "previews (content_chars + outline); has_more/"
                        "total_estimate are exact on this path."
                    )
                elif content_mode == "full":
                    display_hint = (
                        f'find full-content page ({page_cost}): content_mode="full" '
                        "returned full texts; default find is an index page "
                        "(content_chars + outline) that costs far less."
                    )
                elif content_mode == "hits":
                    display_hint = (
                        f'find hit-spans page ({page_cost}): content_mode="hits" '
                        "returned vector-hit spans per item (hit_spans[].text + "
                        "start/end offsets share read's span coordinates; "
                        "hit_window=N extends each hit with +/-N neighbouring "
                        "complete sentences, neighbours marked matched=false); items "
                        "whose hits cover >=50% of the content upgraded to full "
                        "text — hit_spans never truncates. Items without vector "
                        "hits keep the plain preview shape. hit_spans appears only "
                        "on query-recall pages (browse/filter pages carry none); "
                        "stale-version hits are dropped with a stale_hit_spans "
                        "marker and a re-query warning."
                    )
                elif total_estimate is None:
                    # Unfiltered active query-recall: no exact total exists,
                    # so deep paging is genuinely discouraged here.
                    display_hint = (
                        f"find is an index page ({page_cost}): "
                        "content_chars shows each item's read cost and outline.offset "
                        "slices exactly via read span; if the top page misses, reword "
                        "the query or add tags_filter instead of deep paging."
                    )
                else:
                    # Filtered query / filter-driven recall: still an index
                    # page, but the exact count makes paging legitimate.
                    display_hint = (
                        f"find is an index page ({page_cost}): "
                        "content_chars shows each item's read cost and outline.offset "
                        "slices exactly via read span; has_more/total_estimate are "
                        "exact on this filtered path."
                    )
            response_data["size"] = {**size_block, "display_hint": display_hint}
        # v0.15.4: the field only appears when page items directly hit an
        # open/applying conflict group, and the value is the number of page
        # items hit — no longer an unconditional caller-scope group count.
        try:
            # list_open_conflicts_for_memory_ids degrades to [] on a down DB,
            # so check availability explicitly — otherwise "no page hits" and
            # "count query failed" would be indistinguishable.
            if not self.db.db_available:
                raise RuntimeError("conflicts DB unavailable")
            page_ids = {int(r["id"]) for r in results if r.get("id") is not None}
            hit_ids: set[int] = set()
            if page_ids:
                groups = (
                    shared_groups
                    if shared_groups is not None
                    else self.db.conflicts.list_open_conflicts_for_memory_ids(
                        sorted(page_ids), include_applying=True,
                    )
                )
                for group in groups:
                    for member in group.get("member_versions") or []:
                        member_id = member.get("memory_id") if isinstance(member, dict) else None
                        if member_id is not None and int(member_id) in page_ids:
                            hit_ids.add(int(member_id))
            if hit_ids:
                response_data["unresolved_conflict_count"] = len(hit_ids)
        except Exception as exc:
            # Never drop the field silently: a strict caller cannot tell "no
            # open conflicts" from "count query failed" without a trace.
            extra_warnings.append(f"unresolved_conflict_count failed: {exc}")
        if attention_required:
            response_data["attention_required"] = True
            response_data["attention_summary"] = attention_summary
            strong_signal = next(
                (
                    r.get("conflict_signal") for r in results
                    if (r.get("conflict_signal") or {}).get("conflict_source")
                    in _STRONG_CONFLICT_SOURCES
                ),
                None,
            )
            if strong_signal and strong_signal.get("action_required"):
                response_data["action_required"] = strong_signal.get("action_required")
                response_data["verification_status"] = strong_signal.get("verification_status")
        if caller.isolation == "strict":
            response_data.update(caller.response_fields())
        return self.db.state.response(
            response_data,
            extra_warnings=extra_warnings + warnings + list(caller.warnings),
        )


    def _auto_embed(
        self, query: str, query_embedding: "list[float] | None",
        extra_warnings: list[str],
    ) -> "list[float] | None":
        """v0.15.9: vec-state check + auto-embedding for ONE query.

        Extracted verbatim from memory_search so batch_find can embed each
        query through the identical path (failures degrade to shared
        warnings; the vec index is never assumed healthy).
        """
        vec_state = self.db.get_vec_index_state()
        vec_disabled = vec_state.get("state") in {"mismatch", "failed"}
        if vec_disabled and (query_embedding is not None or (query and self.settings.embedding_auto_query)):
            disabled_reason = (
                "embedding_space_mismatch"
                if vec_state.get("state") == "mismatch"
                else "embedding_migration_failed"
            )
            extra_warnings.append(
                f"vec_disabled={disabled_reason}: run memory_repair(task='rebuild_evidence') to restore vector recall"
            )
            return None
        if query_embedding is not None or not (query and self.settings.embedding_auto_query):
            return query_embedding
        embedder, ensure_warnings = self._ensure_embedder()
        extra_warnings.extend(ensure_warnings)
        if embedder is None:
            return None
        # 0.16.12 P1-T1: identical query within the same (space, lineage,
        # epoch) skips the synchronous embed — the first vec-state check above
        # already gated this path, and a hit means the stored vector is the
        # byte-identical output of a previous embed under this lineage.
        cache_key = self._tools._query_embed_cache_key(
            embedder, query, vec_state.get("active_space_id"),
        )
        cached = self._tools._query_embed_cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            # Char-level pre-trim for pathological pastes; the token
            # budget inside embed_text still makes the final cut.
            er = embedder.embed_text(
                prefix=EMBED_PREFIX_SEARCH, body=query,
                max_body_chars=max(EMBEDDING_MAX_SECTION_CHARS, 2048),
            )
            if er.embedding:
                refreshed_state = self.db.get_vec_index_state()
                if refreshed_state.get("state") in {"mismatch", "failed"}:
                    reason = (
                        "embedding_space_mismatch"
                        if refreshed_state.get("state") == "mismatch"
                        else "embedding_migration_failed"
                    )
                    extra_warnings.append(vec_disabled_warning(reason))
                    return None
                self._tools._query_embed_cache_put(cache_key, er.embedding)
                return er.embedding
            extra_warnings.append(
                f"auto-embedding query failed: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
            )
        except Exception as exc:
            extra_warnings.append(f"auto-embedding query failed: {exc}")
        return None

    def _search_scope_context(
        self, workspace: "str | None", extra_warnings: list[str],
    ) -> dict[str, Any]:
        """v0.15.9: per-call scope preamble shared by find and batch_find.

        Resolves isolation, caller workspace, strict admitted set, hard
        scoping and the recall blacklist ONCE per call — every query in a
        batch shares the same caller context by construction.
        """
        # v0.9.7/v0.12.5: workspace isolation on the read path.
        isolation = self.settings.isolation
        caller = self._caller_workspace(workspace)
        # Spec §15.6: an explicit workspace filter is canonicalized then applied
        # in every isolation mode. In none this honors the caller's explicit
        # filter only — never an ACL: omitted workspace still spans all
        # workspaces and the settings fallback never filters.
        explicit_filter = isolation != "none" or caller.source == "explicit"
        ws_canonical = caller.canonical if explicit_filter else None
        workspace = caller.workspace if explicit_filter else workspace
        # An explicit filter goes through SQL hard_scope ONLY under none
        # isolation, so the limit applies AFTER workspace scoping — never a
        # post-page truncation. weak NEVER hard-filters (soft rerank only;
        # pinned by test_weak_recall_never_filters).
        hard_scope = isolation == "none" and caller.source == "explicit" and bool(caller.canonical)
        # strict recall/ACL scope is the admitted canonical set (own +
        # in-radius neighbours). None/weak never hard-scope by it.
        ws_scope = caller.scope_canonicals() if isolation == "strict" and ws_canonical else None
        # v0.15.5 recall blacklist: an UNSCOPED find (no explicit workspace,
        # non-strict) excludes blacklisted workspaces (default: the mema-twin
        # preference bucket) from the ambient pool. An explicit workspace
        # filter — including a blacklisted one — is honored as-is, and strict
        # scoping bypasses the blacklist via its admitted set.
        exclude_ws: "frozenset[str] | None" = None
        if caller.source != "explicit" and isolation != "strict":
            from ..recall_blacklist import blacklist_path, load_blacklist
            exclude_ws, bl_warnings = load_blacklist(blacklist_path(self.db.settings.db_path))
            extra_warnings.extend(bl_warnings)
            # A caller HOMED in a blacklisted bucket (settings.workspace) is
            # effectively explicit about that bucket — un-exclude its home
            # only, never drop the whole blacklist.
            if exclude_ws and caller.canonical and caller.canonical in exclude_ws:
                exclude_ws = exclude_ws - {caller.canonical}
        return {
            "isolation": isolation, "caller": caller, "ws_canonical": ws_canonical,
            "workspace": workspace, "hard_scope": hard_scope,
            "ws_scope": ws_scope, "exclude_ws": exclude_ws,
        }


    def memory_batch_find(
        self,
        queries: "list[dict[str, Any]] | None" = None,
        workspace: str | None = None,
        tags_filter: list[str] | None = None,
        after_time: str | None = None,
        before_time: str | None = None,
        source_type: str | None = None,
        limit_per_query: int = 3,
        content_mode: str = "preview",
        hit_window: int = 0,
        deduplicate: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        """v0.15.9 batch_find (mema 923 §6): one call, N queries, merged page.

        Contract highlights (pinned in tests/test_batch_find.py):
        - fail-fast: the validation boundary rejects malformed batches whole;
          there is no per-query runtime error channel (shared degradations —
          embedder/vec down — surface as shared warnings);
        - no fallback: a query that recalls nothing reports count=0 /
          retrieval_mode="empty" — batch inherits find's honest-empty
          semantics, never recent memories;
        - deduplicate=true merges by memory_id across queries; each item
          carries matched_query_ids (all hitting query ids, submission
          order) and best_query_id (the query whose page scored it
          highest); global order = best final score desc, then first-hit
          query order, then memory_id asc (deterministic);
        - per query, the page is sliced to limit_per_query BEFORE merging;
        - shared filters (workspace/tags_filter/source_type/time window)
          and the caller scope (isolation/ACL/recall blacklist) apply
          identically to every query in the batch.
        """
        from ..constants import BATCH_FIND_DEFAULT_LIMIT_PER_QUERY

        if "include_content" in _:
            return self.db.state.response(
                {
                    "error": 'include_content was removed in v0.15.10; use content_mode="full" for full text or content_mode="hits" for vector-hit spans instead',
                    "results": [],
                    "count": 0,
                    "per_query": [],
                },
                ok=False,
            )
        if content_mode not in _CONTENT_MODES:
            return self.db.state.response(
                {
                    "error": 'content_mode must be one of "preview" | "hits" | "full" (default "preview")',
                    "results": [],
                    "count": 0,
                    "per_query": [],
                },
                ok=False,
            )
        if not queries:
            return self.db.state.response(
                {"error": "queries must be a non-empty list of {id?, query} objects",
                 "results": [], "count": 0, "per_query": []},
                ok=False,
            )
        try:
            limit_per_query = int(limit_per_query)
        except (TypeError, ValueError):
            limit_per_query = BATCH_FIND_DEFAULT_LIMIT_PER_QUERY
        limit_per_query = max(1, min(limit_per_query, 20))
        deduplicate = True if deduplicate is None else bool(deduplicate)

        extra_warnings = list(self._embedder_warnings)
        # 0.17.0 hit_window: hits-mode-only knob (silently ignored beside the
        # other modes, matching the limit_per_query convention).
        hit_window_value = (
            _coerce_hit_window(hit_window, extra_warnings)
            if content_mode == "hits" else 0
        )
        ctx = self._search_scope_context(workspace, extra_warnings)
        # Parity with find (R1-F1): the caller-scope warnings belong to the
        # call, so every query in the batch shares them.
        extra_warnings.extend(list(ctx["caller"].warnings))
        if ctx["isolation"] == "strict" and not ctx["ws_canonical"]:
            denied = self._strict_acl_unavailable(ctx["caller"])
            if denied is not None:
                data = denied.get("data") or {}
                data.update({"results": [], "count": 0, "per_query": []})
                return denied

        per_query: list[dict[str, Any]] = []
        # (query_order, row_in_query_order, qid, row) — rows carry debug fields
        # for the merge; they are stripped before the preview is built.
        collected: list[tuple[int, int, str, dict[str, Any]]] = []
        attention_hits: list[dict[str, Any]] = []
        for order, item in enumerate(queries):
            qid = str(item.get("id") or item.get("query") or f"q{order}")
            query = str(item.get("query") or "")
            emb = self._auto_embed(query, None, extra_warnings)
            outcome = self._search_memories(
                self.db, query, ctx["workspace"], None, limit_per_query,
                status_filter="active", offset=0, debug_ranking=True,
                query_embedding=emb, tags_filter=tags_filter,
                after_time=after_time, before_time=before_time,
                source_type=source_type, ws_canonical=ctx["ws_canonical"],
                isolation=ctx["isolation"], hard_scope=ctx["hard_scope"],
                ws_scope=ctx["ws_scope"], exclude_workspaces=ctx["exclude_ws"],
            )
            rows = outcome.results
            if outcome.retrieval_mode == "direct" and rows:
                rows = self._attach_conflict_signals(rows, extra_warnings)
                for row in rows:
                    sig = row.get("conflict_signal") or {}
                    if sig.get("conflict_source") in _STRONG_CONFLICT_SOURCES:
                        attention_hits.append({"qid": qid, "row": row, "sig": sig})
            per_query.append({
                "id": qid,
                "count": len(rows),
                "has_more": bool(outcome.has_more),
                "retrieval_mode": outcome.retrieval_mode,
            })
            extra_warnings.extend(outcome.warnings)
            for row_idx, row in enumerate(rows):
                collected.append((order, row_idx, qid, row))

        def _final(row: dict[str, Any]) -> float:
            try:
                return float(row.get("_final_score") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        # 0.17.0 hit_window: the ±N neighbour-row prefetch happens ONCE,
        # after the query loop and BEFORE the merge loop (never inside
        # _preview — that would be one query per item). Multiple collected
        # rows of the same memory (dedup merge) contribute overlapping ranges;
        # row_spans_for_ids dedupes the entries internally.
        _window_map: dict[int, list[dict[str, Any]]] = {}
        if content_mode == "hits" and hit_window_value > 0 and collected:
            _window_entries: list[tuple[int, int, int, int]] = []
            for _order, _row_idx, _qid, row in collected:
                indexes = [
                    int(h["row_index"]) for h in (row.get("_evidence_hits") or [])
                    if isinstance(h, dict)
                    and isinstance(h.get("row_index"), int)
                    and not isinstance(h.get("row_index"), bool)
                ]
                if indexes:
                    _window_entries.append((
                        int(row["id"]), int(row.get("version") or 1),
                        min(indexes) - hit_window_value,
                        max(indexes) + hit_window_value,
                    ))
            if _window_entries:
                _window_map = self.db.evidence.row_spans_for_ids(_window_entries)

        def _preview(row: dict[str, Any]) -> dict[str, Any]:
            # v0.15.10: batch_find searches with debug_ranking=True so rows
            # still carry _evidence_hits; in "hits" mode it survives the clean
            # (and only it — the merged page keeps no other debug field) so
            # _preview_item can consume it into hit_spans.
            keep_hits = content_mode == "hits"
            clean = {
                k: v for k, v in row.items()
                if not k.startswith("_") or (keep_hits and k == "_evidence_hits")
            }
            return _preview_item(
                clean, content_mode=content_mode, db=self.db,
                hit_window=hit_window_value,
                window_rows=_window_map.get(int(row["id"])),
            )

        results: list[dict[str, Any]] = []
        if deduplicate:
            merged: dict[int, dict[str, Any]] = {}
            for order, row_idx, qid, row in collected:
                mid = int(row["id"])
                entry = merged.setdefault(mid, {
                    "row": row, "best_final": _final(row), "best_order": order,
                    "best_row_idx": row_idx, "best_qid": qid, "matched": [],
                })
                if qid not in entry["matched"]:
                    entry["matched"].append(qid)
                final = _final(row)
                # Deterministic best-pick: score desc, then earliest query,
                # then earliest position inside that query's page.
                if (final, -order, -row_idx) > (entry["best_final"], -entry["best_order"], -entry["best_row_idx"]):
                    entry.update({"row": row, "best_final": final, "best_order": order,
                                  "best_row_idx": row_idx, "best_qid": qid})
            ordered = sorted(
                merged.values(),
                key=lambda e: (-e["best_final"], e["best_order"], e["best_row_idx"], int(e["row"]["id"])),
            )
            for entry in ordered:
                preview = _preview(entry["row"])
                preview["matched_query_ids"] = entry["matched"]
                preview["best_query_id"] = entry["best_qid"]
                results.append(preview)
        else:
            for order, row_idx, qid, row in collected:
                preview = _preview(row)
                preview["matched_query_ids"] = [qid]
                preview["best_query_id"] = qid
                results.append(preview)

        if content_mode == "hits":
            # F1 (owner 2026-09-23): stale-version hits dropped by the
            # preview builder surface as per-item re-query warnings. One per
            # memory — deduplicate=false can carry the same memory N times.
            _stale_seen: set[int] = set()
            for entry in results:
                stale = entry.get("stale_hit_spans")
                if isinstance(stale, dict) and entry.get("id") is not None:
                    mid = int(entry["id"])
                    if mid in _stale_seen:
                        continue
                    _stale_seen.add(mid)
                    extra_warnings.append(_stale_hit_spans_warning(mid, stale))

        response_data: dict[str, Any] = {
            "results": results,
            "count": len(results),
            "per_query": per_query,
            "deduplicated": bool(deduplicate),
            "query_domain": "active",
            "vector_lag": self._vector_lag(),
        }

        # v0.8.7 loud attention flag, aggregated across the whole batch: one
        # summary naming the first strong hit and how many more exist.
        if attention_hits:
            first = attention_hits[0]
            first_row, first_sig = first["row"], first["sig"]
            head = f"batch_find[{first['qid']}] hit #{first_row.get('id')}"
            if first_row.get("subject"):
                head += f" ({first_row['subject']})"
            head += f" carries a {first_sig.get('conflict_source') or 'open_table'} signal"
            peer = first_sig.get("conflict_peer") or {}
            if isinstance(peer, dict) and peer.get("id") is not None:
                peer_txt = f"#{peer['id']}"
                if peer.get("subject"):
                    peer_txt += f" ({peer['subject']})"
                head += f" vs {peer_txt}"
            if len(attention_hits) > 1:
                head += f" and {len(attention_hits) - 1} more"
            seen_sources: dict[str, dict[str, Any]] = {}
            for hit in attention_hits:
                sig = hit["sig"]
                src = str(sig.get("conflict_source") or "conflict")
                row = hit["row"]
                if src in seen_sources:
                    continue
                seen_sources[src] = row
                ids = [int(row["id"])] if row.get("id") is not None else []
                peer = sig.get("conflict_peer") or {}
                if isinstance(peer, dict) and peer.get("id") is not None:
                    ids.append(int(peer["id"]))
                self.db.log_attention(trigger="search", source=src, memory_ids=ids)
            response_data["attention_required"] = True
            response_data["attention_summary"] = head

        if self.settings.include_size:
            size_block = meter_payloads(results)
            page_cost = (
                f"~{size_block['tokens_estimate']} tokens returned for "
                f"{len(results)} merged item{'s' if len(results) != 1 else ''} "
                f"across {len(per_query)} quer{'y' if len(per_query) == 1 else 'ies'}"
            )
            response_data["size"] = {
                **size_block,
                "display_hint": (
                    f"batch_find merged page ({page_cost}): items are index-page previews "
                    "(content_chars + outline) carrying matched_query_ids; per_query.count "
                    "is each query's recalled page size after the relevance floor. "
                    + ('content_mode="full" returned full texts — per-query limits multiply.'
                       if content_mode == "full" else
                       'content_mode="hits" returned vector-hit spans per item (>=50% '
                       'coverage items upgraded to full text).'
                       if content_mode == "hits" else
                       "Read specific items via memory(action='read') with outline offsets.")
                ),
            }
        # 疑似#7（owner 2026-10-04 拍板：要补）：batch_find 全文/hits 页补响应
        # 字节预算——与 batch_read 的 80KB 家族同款同常数、同结构化降级
        # （never silent truncation）：超限整页降元数据（保留 content_chars），
        # 指引 agent 逐条 read。preview 页本就无内容，不进此门。
        #
        # A6（0.17.1 修复批）：find 页条目是**扁平**形状（content 在顶层，
        # _preview_item 返回 dict(item)），此前的 entry["memory"] 取值恒空
        # → total_bytes 恒 0、门永不触发（实测 360KB 放行）；hits 页的
        # hit_spans[].text 也不在预算内（实测未升级全文的 hits 页 509KB
        # 放行）。两处一并计入。
        if content_mode in {"full", "hits"}:
            def _item_bytes(entry: dict[str, Any]) -> int:
                total = len(str(entry.get("content") or "").encode("utf-8"))
                for span in entry.get("hit_spans") or []:
                    if isinstance(span, dict):
                        total += len(str(span.get("text") or "").encode("utf-8"))
                return total

            total_bytes = sum(_item_bytes(entry) for entry in results)
            if total_bytes > BATCH_READ_FULL_BUDGET_BYTES:
                slim_results: list[dict[str, Any]] = []
                for entry in results:
                    slim = {key: value for key, value in entry.items() if key != "content"}
                    if entry.get("hit_spans"):
                        # hits 页：hit_spans 是命中的窗口坐标，降级保留 +
                        # 显式标记（绝不静默截断）。content（若因覆盖率升级
                        # 而存在）已剥离，content_chars 保留全文长度。
                        slim["hit_spans_truncated_by_budget"] = True
                    slim_results.append(slim)
                response_data["results"] = slim_results
                response_data["over_budget"] = True
                response_data["budget_bytes"] = BATCH_READ_FULL_BUDGET_BYTES
                response_data["total_bytes"] = total_bytes
                response_data["hint"] = (
                    "batch_find over the content byte budget: full texts were dropped "
                    "(content_chars kept; hit_spans, when present, are the hit windows) — "
                    "read items individually (memory action='read') or lower limit_per_query"
                )
                if self.settings.include_size and isinstance(response_data.get("size"), dict):
                    response_data["size"] = {
                        **meter_payloads(slim_results),
                        "display_hint": response_data["size"].get("display_hint"),
                    }
        return self.db.state.response(response_data, extra_warnings=extra_warnings)
