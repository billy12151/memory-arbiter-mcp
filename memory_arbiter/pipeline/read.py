"""Internal read, search, comparison, and conflict-signal operations."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..acl import CallerWorkspace
from ..embedder import ManagedEmbedder

from ..constants import (
    BATCH_READ_FULL_BUDGET_BYTES,
    BATCH_READ_FULL_BUDGET_MAX_BYTES,
    EMBED_PREFIX_SEARCH,
    EMBEDDING_MAX_SECTION_CHARS,
    SUPERSEDED_LIMIT,
)
from ..tokens import meter_payloads
from ._read_hits import (  # noqa: F401
    _STRONG_CONFLICT_SOURCES as _STRONG_CONFLICT_SOURCES,
    _CONTENT_MODES as _CONTENT_MODES,
    _OUTLINE_MAX_SEGMENTS as _OUTLINE_MAX_SEGMENTS,
    _OUTLINE_HEAD_CHARS as _OUTLINE_HEAD_CHARS,
    _HIT_SPANS_FULL_COVERAGE as _HIT_SPANS_FULL_COVERAGE,
    _HIT_WINDOW_MAX as _HIT_WINDOW_MAX,
    vec_disabled_warning as vec_disabled_warning,
    _coerce_hit_window as _coerce_hit_window,
    _stale_hit_spans_warning as _stale_hit_spans_warning,
    _evidence_lag_warning as _evidence_lag_warning,
    _unit_aligned_hits as _unit_aligned_hits,
    _outline_from_rows as _outline_from_rows,
    _outline_for_item as _outline_for_item,
    _content_outline as _content_outline,
    _hit_spans as _hit_spans,
    _preview_item as _preview_item,
)
from ._read_search import _ReadSearch

if TYPE_CHECKING:
    from ..tools import MemoryTools


class ReadPipeline(_ReadSearch):
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools
        self.db = tools.db
        self.settings = tools.settings
        # R2-S1：语义 worker 是唯一 worker（C2 合并后索引同队列），向量滞
        # 后观测跟着走。
        self._semantic_worker = tools._semantic_worker
        self._embedder_warnings = tools._embedder_warnings

    def _attach_conflict_signals(
        self, *args: Any, **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return self._tools._attach_conflict_signals(*args, **kwargs)

    def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace":
        return self._tools._caller_workspace(*args, **kwargs)

    def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_embedder()

    def _get_memory_visible(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._get_memory_visible(*args, **kwargs)

    def _strict_acl_unavailable(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._strict_acl_unavailable(*args, **kwargs)

    @staticmethod
    def _search_memories(*args: Any, **kwargs: Any) -> Any:
        # Preserve the legacy monkeypatch seam: tests and external diagnostics
        # patch memory_arbiter.tools.search_memories, so resolve that module
        # binding at call time rather than using this module's import cache (R4).
        from .. import tools as tools_mod
        return getattr(tools_mod, "search_memories")(*args, **kwargs)

    @staticmethod
    def _compare_memories(*args: Any, **kwargs: Any) -> Any:
        # Preserve legacy patch seam for memory_arbiter.tools.compare_memories.
        from .. import tools as tools_mod
        return getattr(tools_mod, "compare_memories")(*args, **kwargs)

    @staticmethod
    def _linked_open_items_for_search(*args: Any, **kwargs: Any) -> Any:
        # Preserve the legacy monkeypatch seam for
        # memory_arbiter.tools._linked_open_items_for_search.
        from .. import tools as tools_mod
        return getattr(tools_mod, "_linked_open_items_for_search")(*args, **kwargs)

    def _vector_lag(self) -> dict[str, int]:
        """Spec §13.1: search must not pretend the async evidence index is
        consistent with the write path — surface pending index work (the
        semantic queue is the only index queue since the C2 worker merge)."""
        try:
            worker = self._semantic_worker.status()
        except Exception:
            return {"pending_evidence_index": 0}
        pending = int(worker.get("queue_depth") or 0) + len(worker.get("inflight") or [])
        return {"pending_evidence_index": pending}

    def memory_search_expired(
        self,
        query: str = "",
        workspace: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        debug_ranking: bool = False,
        query_embedding: list[float] | None = None,
        tags_filter: list[str] | None = None,
        after_time: str | None = None,
        before_time: str | None = None,
        source_type: str | None = None,
        include_conflict_signal: bool = True,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        """v0.9.4: search expired (non-active non-deleted) memories with vec-hybrid recall.

        Searches ONLY non-active, non-deleted memories (superseded +
        conflicted + pending) for audit/history walkthroughs:
        - evidence channel: ``row_knn`` with the
          ``parent_status NOT IN ('active','deleted')`` predicate
        - FTS channel: ``search_memories(status_filter="expired")`` with
          ``status_clause = "m.status NOT IN ('active','deleted')"``

        ``limit`` controls the per-page cap (default 20, hard cap 50; the
        page cap is the frozen constant SUPERSEDED_LIMIT). ``offset``
        enables cursor pagination — exact on the empty-query+filters path
        (SQL OFFSET backed by a precise count), best-effort on the
        query-recall path (pool windowed to offset+limit).

        Active-query split (§3.5): ``memory_search`` (active only) and
        ``memory_search_expired`` (expired only) are two independent queries.
        """
        extra_warnings: list[str] = list(self._embedder_warnings)
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
            query_embedding = None
        elif query_embedding is None and query and self.settings.embedding_auto_query:
            embedder, ensure_warnings = self._ensure_embedder()
            extra_warnings.extend(ensure_warnings)
            if embedder is not None:
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
                        else:
                            query_embedding = er.embedding
                    else:
                        extra_warnings.append(
                            f"auto-embedding query failed: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
                        )
                except Exception as exc:
                    extra_warnings.append(f"auto-embedding query failed: {exc}")

        limit_requested = int(limit)
        offset_requested = int(offset)
        effective_offset = max(0, min(offset_requested, 10000))
        effective_limit = min(max(1, limit_requested), max(1, SUPERSEDED_LIMIT), 50)

        # v0.12.5: expired recall uses the shared caller-workspace resolver.
        isolation = self.settings.isolation
        caller = self._caller_workspace(workspace)
        # Same contract as active search: an explicit filter canonicalizes and
        # applies in every mode; none mode never filters without one.
        explicit_filter = isolation != "none" or caller.source == "explicit"
        ws_canonical = caller.canonical if explicit_filter else None
        workspace = caller.workspace if explicit_filter else workspace
        hard_scope = isolation == "none" and caller.source == "explicit" and bool(caller.canonical)
        if isolation == "strict" and not ws_canonical:
            return self.db.state.response(
                {
                    "error": "forbidden_strict_workspace",
                    "reason": "missing_caller_workspace",
                    "results": [],
                    "count": 0,
                    **caller.response_fields(),
                },
                ok=False,
                extra_warnings=extra_warnings + list(caller.warnings),
            )

        outcome = self._search_memories(
            self.db, query, workspace, tags, effective_limit,
            status_filter="expired",  # superseded + conflicted + pending (§3.5 split)
            debug_ranking=debug_ranking,
            query_embedding=query_embedding,
            tags_filter=tags_filter,
            after_time=after_time,
            before_time=before_time,
            source_type=source_type,
            offset=effective_offset,
            ws_canonical=ws_canonical,
            isolation=isolation,
            hard_scope=hard_scope,
            ws_scope=caller.scope_canonicals() if isolation == "strict" and ws_canonical else None,
        )
        results = outcome.results
        warnings = outcome.warnings
        has_more = outcome.has_more
        total_estimate = outcome.total_estimate
        retrieval_mode = outcome.retrieval_mode

        # v0.7.6: attach conflict signals (strict expired results are non-active
        # and may lack safe workspace summaries; fail closed by omitting signals).
        if include_conflict_signal and isolation != "strict" and retrieval_mode == "direct" and results:
            results = self._attach_conflict_signals(results, extra_warnings)

        attention_required = False
        attention_summary: str | None = None
        if include_conflict_signal and retrieval_mode == "direct" and results:
            seen_sources: dict[str, dict[str, Any]] = {}
            for r in results:
                sig = r.get("conflict_signal")
                if not sig:
                    continue
                seen_sources.setdefault(str(sig.get("conflict_source", "conflict")), r)
            ot = next((seen_sources.get(source) for source in _STRONG_CONFLICT_SOURCES if seen_sources.get(source)), None)
            if ot is not None:
                attention_required = True
                ot_sig = ot.get("conflict_signal") or {}
                head = f"Expired search hit #{ot.get('id')}"
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

        next_offset = effective_offset + len(results) if has_more else None
        response_data = {
            "results": results,
            "count": len(results),
            "has_more": has_more,
            "total_estimate": total_estimate,
            "retrieval_mode": retrieval_mode,
            "query_domain": "expired",
            "domain_statuses": "non-active non-deleted (superseded, conflicted, pending)",
            "offset": effective_offset,
            "limit_requested": limit_requested,
            "effective_limit": effective_limit,
            "next_offset": next_offset,
            "offset_clamped": effective_offset != offset_requested,
            "limit_capped": effective_limit != limit_requested,
            "pagination_precision": "exact" if not str(query or "").strip() else "best_effort",
            "vector_lag": self._vector_lag(),
        }
        if self.settings.include_size:
            # v0.15.6: the shared size block. Expired pages carry full texts
            # (no preview path), so the meter reads as the full-text page it
            # is; the display_hint carries the number as a report-this-cost
            # instruction, silent on empty pages (no cost to report).
            size_block = meter_payloads(results)
            display_hint = None
            if results:
                display_hint = (
                    f"expired recall (~{size_block['tokens_estimate']} tokens returned "
                    f"for {len(results)} full-text item{'s' if len(results) != 1 else ''}): "
                    "report this recall cost when citing it; narrow the query or add "
                    "tags_filter when pages run large."
                )
            response_data["size"] = {**size_block, "display_hint": display_hint}
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

    def memory_get(
        self,
        memory_id: int,
        sections: str = "none",
        section_ids: list[int] | None = None,
        span: dict[str, Any] | None = None,
        content_mode: str = "full",
        hit_window: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        """Return one full memory by id, a unit-aligned window of it, or its preview.

        0.16.0 four-call content_mode unification (plan §6⑨): find/batch_find/
        read/batch_read share preview/hits/full semantics. ``read`` defaults to
        full (backward compatible). ``span={"start", "end"}`` (``end`` optional
        since 0.17.1 — an omitted end reads through the end of the content,
        the same default as batch spans) selects complete evidence units
        overlapping the window — mema's content atom is the unit,
        so windowed reads never slice a half sentence (the legacy char-slice
        remains only as the fallback when no evidence rows exist yet).
        """
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer"}, ok=False)
        if sections not in ("none", None) or section_ids:
            return self.db.state.response(
                {"error": "section reads were removed; read the full memory content"},
                ok=False,
            )
        if content_mode not in _CONTENT_MODES:
            return self.db.state.response(
                {"error": 'content_mode must be one of "preview" | "hits" | "full" (default "full")'},
                ok=False,
            )
        # 0.17.0 hit_window: hits-mode-only knob + the F1 fallback warnings
        # share one per-call warning list (empty by default → byte-identical).
        read_warnings: list[str] = []
        hit_window_value = (
            _coerce_hit_window(hit_window, read_warnings)
            if content_mode == "hits" else 0
        )
        span_start: int | None = None
        span_end: int | None = None
        if span is not None:
            if not isinstance(span, dict):
                return self.db.state.response(
                    {"error": "span must be an object with start/end"},
                    ok=False, extra_warnings=read_warnings,
                )
            raw_start = span.get("start")
            raw_end = span.get("end")
            # P2 #12: a span may carry only `start` — `end` defaults to the
            # content length, resolved after the record is in hand (the
            # validation point cannot see the content yet).
            if (
                not isinstance(raw_start, int) or isinstance(raw_start, bool)
                or (
                    raw_end is not None
                    and (not isinstance(raw_end, int) or isinstance(raw_end, bool))
                )
            ):
                return self.db.state.response(
                    {"error": "span start/end must be integers"},
                    ok=False, extra_warnings=read_warnings,
                )
            span_start = raw_start
            span_end = raw_end
            if span_start < 0 or (span_end is not None and span_end <= span_start):
                return self.db.state.response(
                    {"error": "span requires 0 <= start < end"},
                    ok=False, extra_warnings=read_warnings,
                )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        memory = self._get_memory_visible(memory_id_int, caller)
        if not memory:
            error_data: dict[str, Any] = {"error": f"memory id {memory_id_int} not found"}
            if caller.isolation == "strict":
                error_data.update(caller.response_fields())
            return self.db.state.response(
                error_data, ok=False,
                extra_warnings=read_warnings + list(caller.warnings),
            )

        content = str(memory.get("content") or "")
        if span is not None and span_end is None:
            # P2 #12: {"start": N} without `end` reads through the end of the
            # content — resolved here (the validation point cannot see the
            # record) so every downstream consumer (full-mode slicing AND the
            # hits-mode unit alignment) sees a concrete end.
            span_end = len(content)
            span = {**span, "end": span_end}
        data: dict[str, Any]
        if content_mode == "preview" and span is None:
            preview = {
                key: value for key, value in memory.items() if key != "content"
            }
            preview["content_chars"] = len(content)
            preview["outline"] = _outline_for_item(
                self.db, int(memory["id"]), int(memory.get("version") or 1),
                str(memory.get("subject") or ""), content,
            )
            data = {"memory": preview}
        elif content_mode == "hits":
            unit_hits = _unit_aligned_hits(self.db, memory, span, window=hit_window_value)
            if unit_hits is not None:
                hit_spans, upgraded = unit_hits
                record = {
                    key: value for key, value in memory.items() if key != "content"
                }
                record["content_chars"] = len(content)
                record["outline"] = _outline_for_item(
                    self.db, int(memory["id"]), int(memory.get("version") or 1),
                    str(memory.get("subject") or ""), content,
                )
                record["hit_spans"] = hit_spans
                if upgraded is not None:
                    record["content"] = upgraded
                data = {"memory": record}
            else:
                # No evidence rows for the current version (fresh write before
                # the async index lands, or a down embedder): fall back to the
                # full record — an honest answer beats an empty hits page.
                # F1: the fallback is no longer silent — the agent must know
                # the unit index lags this version.
                read_warnings.append(_evidence_lag_warning(
                    memory_id_int, 'content_mode="hits" fell back to the full record',
                ))
                data = {"memory": memory}
        else:
            if span_start is not None and span_end is not None:
                if span_start >= len(content):
                    return self.db.state.response(
                        {"error": "span start is past the end of the content",
                         "total_chars": len(content)},
                        ok=False,
                    )
                clipped_end = min(span_end, len(content))
                rows = self.db.evidence.text_unit_rows(
                    memory_id_int, int(memory.get("version") or 1),
                    span_start=span_start, span_end=clipped_end,
                )
                if rows:
                    # Unit-aligned window: a contiguous slice of the source
                    # content spanning from the first to the last covered
                    # unit. Units may overlap (long-text fallback), so joining
                    # unit texts would duplicate text — slice the original
                    # instead and report the covered units in the metadata.
                    first_start = min(int(row["start_offset"]) for row in rows)
                    last_end = max(int(row["end_offset"]) for row in rows)
                    windowed = dict(memory)
                    windowed["content"] = content[first_start:last_end]
                    data = {
                        "memory": windowed,
                        "span": {
                            "start": span_start, "end": clipped_end,
                            "total_chars": len(content),
                            "unit_aligned": True, "units": len(rows),
                        },
                    }
                else:
                    # Legacy fallback while no evidence rows exist. F1: say so
                    # — the agent may be reading by pre-edit offsets.
                    read_warnings.append(_evidence_lag_warning(
                        memory_id_int, "span read used the legacy character slice",
                    ))
                    windowed = dict(memory)
                    windowed["content"] = content[span_start:clipped_end]
                    data = {
                        "memory": windowed,
                        "span": {"start": span_start, "end": clipped_end, "total_chars": len(content)},
                    }
            else:
                data = {"memory": memory}
        if self.settings.include_size:
            # v0.15.6: same size block as find, metering the record as
            # actually returned — a span read meters the windowed payload, so
            # the number is the true cost of this call.
            # The display_hint is an instruction, not paging guidance: agents
            # that cite a record should surface what the recall cost, and the
            # hint carries the number so they don't have to dig for it.
            size_block = meter_payloads([data["memory"]])
            tokens = size_block["tokens_estimate"]
            span_meta = data.get("span")
            if span_meta is not None:
                display_hint = (
                    f"read span (~{tokens} tokens returned; window "
                    f"{span_meta['start']}:{span_meta['end']} of "
                    f"{span_meta['total_chars']} chars): report this recall "
                    "cost when citing it."
                )
            else:
                display_hint = (
                    f"read (~{tokens} tokens returned for 1 record): "
                    "report this recall cost when citing it."
                )
            data["size"] = {**size_block, "display_hint": display_hint}
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=read_warnings + list(caller.warnings))

    def memory_batch_read(
        self,
        memory_ids: "list[int] | None" = None,
        content_mode: str = "preview",
        hit_window: int = 0,
        spans: "dict[str, Any] | None" = None,
        **_: Any,
    ) -> dict[str, Any]:
        """0.16.0 batch read: read = the single-item special case of this call.

        Contract (plan §1.5, owner-pinned caps): memory_ids[] + content_mode in
        {preview, hits, full}; caps preview 50 / hits 50 / full 10 ids; full
        adds an 80KB byte budget (100KB hard ceiling) — an over-budget batch
        returns a structured over-long prompt and the agent re-reads items
        individually, never a silent truncation. ``spans`` maps memory_id to
        {start, end} (``end`` optional since 0.17.1 — an omitted end reads
        through the end of that record's content, matching single-read spans)
        and is the ``hits`` unit selector for id-driven calls
        (complete evidence units, zero half-sentence truncation by
        construction). Every id passes the caller's ACL individually.
        """
        from ..constants import (
            BATCH_READ_MAX_FULL, BATCH_READ_MAX_HITS, BATCH_READ_MAX_PREVIEW,
        )

        if content_mode not in _CONTENT_MODES:
            return self.db.state.response(
                {"error": 'content_mode must be one of "preview" | "hits" | "full" (default "preview")',
                 "results": [], "count": 0},
                ok=False,
            )
        # 0.17.0 hit_window: hits-mode-only knob + the F1 fallback warnings
        # share one per-call warning list (empty by default → byte-identical).
        read_warnings: list[str] = []
        hit_window_value = (
            _coerce_hit_window(hit_window, read_warnings)
            if content_mode == "hits" else 0
        )
        cap = {"preview": BATCH_READ_MAX_PREVIEW, "hits": BATCH_READ_MAX_HITS, "full": BATCH_READ_MAX_FULL}[content_mode]
        wanted: list[int] = []
        seen: set[int] = set()
        for raw in memory_ids or []:
            try:
                mid = int(raw)
            except (TypeError, ValueError):
                return self.db.state.response(
                    {"error": "memory_ids must contain positive integer ids", "results": [], "count": 0},
                    ok=False, extra_warnings=read_warnings,
                )
            if mid <= 0:
                return self.db.state.response(
                    {"error": "memory_ids must contain positive integer ids", "results": [], "count": 0},
                    ok=False, extra_warnings=read_warnings,
                )
            if mid not in seen:
                seen.add(mid)
                wanted.append(mid)
        if not wanted:
            return self.db.state.response(
                {"error": "memory_ids must be a non-empty list", "results": [], "count": 0},
                ok=False, extra_warnings=read_warnings,
            )
        if len(wanted) > cap:
            return self.db.state.response(
                {
                    "error": f'content_mode="{content_mode}" accepts at most {cap} ids per call',
                    "cap": cap, "results": [], "count": 0,
                },
                ok=False, extra_warnings=read_warnings,
            )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied

        # P2 #12: end=None enters the map as-is — a spans entry carrying only
        # `start` resolves its end to the record's content length once the
        # visible records are prefetched (the construction point cannot see
        # content), matching the single-read default.
        span_map: dict[int, dict[str, int | None]] = {}
        for key, span in (spans or {}).items():
            try:
                mid = int(str(key))
            except (TypeError, ValueError):
                continue
            if isinstance(span, dict):
                try:
                    start = int(span.get("start", 0))
                    end_raw = span.get("end")
                    span_map[mid] = {
                        "start": start,
                        "end": int(end_raw) if end_raw is not None else None,
                    }
                except (TypeError, ValueError):
                    continue

        # 0.16.12 P1-T4: batch prefetch — ONE connection for the page's memory
        # rows (plus one for the evidence units hits/full need) replaces the
        # per-id get_memory/text_unit_rows connections. The per-id visibility
        # rule is the shared MemoryTools._memory_visible predicate (identical
        # to _get_memory_visible's), applied in Python on the prefetched rows.
        prefetched = self.db.get_memories_by_ids(wanted)
        visible_records: dict[int, dict[str, Any]] = {}
        unit_needed: list[tuple[int, int]] = []
        for mid in wanted:
            record = prefetched.get(mid)
            if record is None or not self._tools._memory_visible(record, caller):
                continue
            visible_records[mid] = record
            if content_mode == "hits":
                unit_needed.append((mid, int(record.get("version") or 1)))
            elif content_mode == "full":
                span = span_map.get(mid)
                if (
                    span is not None and (span["end"] is None or span["end"] > span["start"])
                    and span["start"] < len(str(record.get("content") or ""))
                ):
                    unit_needed.append((mid, int(record.get("version") or 1)))
        # P2 #12 resolution point: deferred span ends ({"start": N} entries)
        # resolve against the now-available content lengths. A start past the
        # end keeps end > start (start+1) so the explicit-span semantics hold
        # (empty window + span_past_end). Ids without a prefetched record
        # keep a >start end — inert (reported not_found, never reaching a
        # span consumer).
        for mid, span_entry in span_map.items():
            if span_entry["end"] is None:
                record = prefetched.get(mid)
                content_len = len(str((record or {}).get("content") or ""))
                span_entry["end"] = (
                    content_len if content_len > span_entry["start"] else span_entry["start"] + 1
                )
        unit_rows_map: dict[int, list[dict[str, Any]]] = (
            self.db.evidence.text_unit_rows_for_ids(unit_needed) if unit_needed else {}
        )
        outline_map: dict[int, list[dict[str, Any]]] = (
            self.db.evidence.outline_rows_for_ids([
                (mid, int(rec.get("version") or 1)) for mid, rec in visible_records.items()
            ]) if content_mode in {"preview", "hits"} else {}
        )

        results: list[dict[str, Any]] = []
        not_found: list[int] = []
        for mid in wanted:
            memory = visible_records.get(mid)
            if not memory:
                not_found.append(mid)
                results.append({"memory_id": mid, "found": False, "error": "not_found"})
                continue
            content = str(memory.get("content") or "")
            span = span_map.get(mid)
            item: dict[str, Any] = {"memory_id": mid, "found": True}
            if content_mode == "preview":
                record = {key: value for key, value in memory.items() if key != "content"}
                record["content_chars"] = len(content)
                record["outline"] = _outline_for_item(
                    self.db, mid, int(memory.get("version") or 1),
                    str(memory.get("subject") or ""), content,
                    rows=outline_map.get(mid, []),
                )
                item["memory"] = record
            elif content_mode == "hits":
                unit_hits = _unit_aligned_hits(
                    self.db, memory, span, rows=unit_rows_map.get(mid, []),
                    window=hit_window_value,
                )
                record = {key: value for key, value in memory.items() if key != "content"}
                record["content_chars"] = len(content)
                record["outline"] = _outline_for_item(
                    self.db, mid, int(memory.get("version") or 1),
                    str(memory.get("subject") or ""), content,
                    rows=outline_map.get(mid, []),
                )
                if unit_hits is not None:
                    hit_spans, upgraded = unit_hits
                    record["hit_spans"] = hit_spans
                    if upgraded is not None:
                        record["content"] = upgraded
                else:
                    # F1: the fallback is no longer silent. (Unlike single
                    # read, this fallback record carries no content — the
                    # wording below must not claim a full record.)
                    read_warnings.append(_evidence_lag_warning(
                        mid, 'content_mode="hits" fell back to the metadata-only record',
                    ))
                item["memory"] = record
            else:  # full
                record = dict(memory)
                if span is not None and span["end"] > span["start"]:
                    if span["start"] < len(content):
                        clipped_end = min(span["end"], len(content))
                        rows = [
                            row for row in unit_rows_map.get(mid, [])
                            if int(row["start_offset"]) < clipped_end
                            and int(row["end_offset"]) > span["start"]
                        ]
                        if rows:
                            # Unit-aligned: contiguous slice from the first to
                            # the last covered unit (units may overlap; joining
                            # them would duplicate text).
                            first_start = min(int(row["start_offset"]) for row in rows)
                            last_end = max(int(row["end_offset"]) for row in rows)
                            record["content"] = content[first_start:last_end]
                        else:
                            # Legacy fallback while no evidence rows exist (F1: say so).
                            read_warnings.append(_evidence_lag_warning(
                                mid, "span read used the legacy character slice",
                            ))
                            record["content"] = content[span["start"]:clipped_end]
                    else:
                        record["content"] = ""
                        item["span_past_end"] = True
                item["memory"] = record
            results.append(item)

        over_budget: list[dict[str, Any]] = []
        if content_mode in {"full", "hits"}:
            # F3 (0.17.0): the hit_window's >=50% coverage upgrade can put
            # full contents on a hits page, so the "hits payloads are
            # structurally bounded" premise no longer holds — any entry that
            # carries content (upgraded hits or full) counts against the same
            # byte budget, with the same structured over-long response.
            def _content_bytes(entry: dict[str, Any]) -> int:
                memory = entry.get("memory") or {}
                return len(str(memory.get("content") or "").encode("utf-8"))

            total_bytes = sum(_content_bytes(entry) for entry in results if entry.get("found"))
            for entry in results:
                if entry.get("found") and _content_bytes(entry) > BATCH_READ_FULL_BUDGET_MAX_BYTES:
                    over_budget.append({"memory_id": entry["memory_id"], "bytes": _content_bytes(entry)})
            if total_bytes > BATCH_READ_FULL_BUDGET_BYTES or over_budget:
                # Structured over-long response — never a silent truncation
                # (plan §6⑰). The page downgrades to metadata-only and tells
                # the agent to read items individually.
                slim_results: list[dict[str, Any]] = []
                for entry in results:
                    if not entry.get("found"):
                        slim_results.append(entry)
                        continue
                    memory = entry.get("memory") or {}
                    record = {key: value for key, value in memory.items() if key != "content"}
                    record["content_chars"] = len(str(memory.get("content") or ""))
                    slim_results.append({"memory_id": entry["memory_id"], "found": True, "memory": record})
                data: dict[str, Any] = {
                    "results": slim_results,
                    "count": len(slim_results),
                    "over_budget": True,
                    "budget_bytes": BATCH_READ_FULL_BUDGET_BYTES,
                    "total_bytes": total_bytes,
                    "hint": (
                        "batch read over the content byte budget; no contents were returned — "
                        "read the items individually (memory action='read') or narrow the batch"
                    ),
                }
                if self.settings.include_size:
                    data["size"] = meter_payloads(slim_results)
                if caller.isolation == "strict":
                    data.update(caller.response_fields())
                return self.db.state.response(
                    data, extra_warnings=read_warnings + list(caller.warnings),
                )

        data = {"results": results, "count": len(results)}
        if self.settings.include_size:
            meter_items = [entry["memory"] for entry in results if entry.get("found")]
            size_block = meter_payloads(meter_items)
            data["size"] = {
                **size_block,
                "display_hint": (
                    f"batch read (~{size_block['tokens_estimate']} tokens returned for "
                    f"{len(meter_items)} record(s), content_mode={content_mode}): report this "
                    "recall cost when citing it."
                ),
            }
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=read_warnings + list(caller.warnings))


    def memory_recent(self, workspace: str | None = None, limit: int = 20, **_: Any) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        caller = self._caller_workspace(workspace)
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation == "strict" and caller.canonical:
            results = self.db.list_memories_for_workspace(
                caller.canonical, limit=limit, admitted=caller.scope_canonicals(),
            )
        else:
            results = self.db.list_memories(limit=limit)
        data = {"results": results, "count": len(results)}
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

    def memory_compare(self, left_id: int | None = None, right_id: int | None = None, left: dict[str, Any] | None = None, right: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        left_record = left or (self._get_memory_visible(int(left_id), caller) if left_id is not None else None)
        right_record = right or (self._get_memory_visible(int(right_id), caller) if right_id is not None else None)
        if caller.isolation == "strict" and (left is not None or right is not None):
            # Caller-supplied records may be stale/untrusted. Require by-id ACL in strict.
            if left_id is None or right_id is None:
                return self.db.state.response({"error": "strict memory_compare requires left_id and right_id", **caller.response_fields()}, ok=False, extra_warnings=list(caller.warnings))
            left_record = self._get_memory_visible(int(left_id), caller)
            right_record = self._get_memory_visible(int(right_id), caller)
        if not left_record or not right_record:
            data = {"error": "left and right records are required"}
            if caller.isolation == "strict":
                data.update(caller.response_fields())
            return self.db.state.response(data, ok=False, extra_warnings=list(caller.warnings))
        compare_data: dict[str, Any] = {"comparison": self._compare_memories(left_record, right_record), "left": left_record, "right": right_record}
        if caller.isolation == "strict":
            compare_data.update(caller.response_fields())
        return self.db.state.response(compare_data, extra_warnings=list(caller.warnings))
