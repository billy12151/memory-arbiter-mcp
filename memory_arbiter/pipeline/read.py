"""Internal read, search, comparison, and conflict-signal operations."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..acl import CallerWorkspace
from ..embedder import ManagedEmbedder

from ..arbitration import compare_memories
from ..constants import EMBEDDING_MAX_SECTION_CHARS, SUPERSEDED_LIMIT, strict_ws
from ..models import MemoryStatus
from ..search import search_memories

if TYPE_CHECKING:
    from ..tools import MemoryTools


# Verified/formally-recorded conflict signal sources that ring the loud
# attention flag. conflict_group is the conflict-groups producer; the retired
# names are kept so legacy payloads still resolve.
_STRONG_CONFLICT_SOURCES = ("open_table", "conflict_guidance", "conflict_group")


class ReadPipeline:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools
        self.db = tools.db
        self.settings = tools.settings
        self._evidence_worker = tools._evidence_worker
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
        consistent with the write path — surface pending index work."""
        try:
            worker = self._evidence_worker.status()
        except Exception:
            return {"pending_evidence_index": 0}
        pending = int(worker.get("queue_depth") or 0) + len(worker.get("inflight") or [])
        return {"pending_evidence_index": pending}

    def memory_search(self, query: str = "", workspace: str | None = None, tags: list[str] | None = None, limit: int = 10, offset: int = 0, debug_ranking: bool = False, query_embedding: list[float] | None = None, tags_filter: list[str] | None = None, after_time: str | None = None, before_time: str | None = None, source_type: str | None = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True, include_size: bool = True, **_: Any) -> dict[str, Any]:
        if "include_superseded" in _:
            return self.db.state.response(
                {
                    "error": "include_superseded was removed in v0.9.4; memory_search is active-only. Use memory_search_expired for expired history/audit recall (non-active non-deleted: superseded, conflicted, pending). The old mixed active+superseded mode is gone.",
                    "results": [],
                    "count": 0,
                },
                ok=False,
            )
        extra_warnings = list(self._embedder_warnings)
        vec_state = self.db.get_vec_index_state()
        vec_disabled = vec_state.get("state") in {"mismatch", "failed"}
        if vec_disabled and (query_embedding is not None or (query and self.settings.embedding_auto_query)):
            disabled_reason = (
                "embedding_space_mismatch"
                if vec_state.get("state") == "mismatch"
                else "embedding_migration_failed"
            )
            extra_warnings.append(f"vec_disabled={disabled_reason}")
            query_embedding = None
        elif query_embedding is None and query and self.settings.embedding_auto_query:
            embedder, ensure_warnings = self._ensure_embedder()
            extra_warnings.extend(ensure_warnings)
            if embedder is not None:
                try:
                    # Char-level pre-trim for pathological pastes; the token
                    # budget inside embed_text still makes the final cut.
                    er = embedder.embed_text(
                        prefix="", body=query,
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
                            extra_warnings.append(f"vec_disabled={reason}")
                        else:
                            query_embedding = er.embedding
                    else:
                        extra_warnings.append(
                            f"auto-embedding query failed: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
                        )
                except Exception as exc:
                    extra_warnings.append(f"auto-embedding query failed: {exc}")
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
        # A none/weak explicit filter scopes recall in SQL (hard_scope) so the
        # limit is applied AFTER workspace scoping — never a post-page truncation.
        hard_scope = isolation == "none" and caller.source == "explicit" and bool(caller.canonical)
        if isolation == "strict" and not ws_canonical:
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                data = denied.get("data") or {}
                data.update({"results": [], "count": 0})
                return denied
        # strict recall/ACL scope is the admitted canonical set (own +
        # in-radius neighbours). None/weak never hard-scope by it.
        ws_scope = caller.scope_canonicals() if isolation == "strict" and ws_canonical else None
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
        )
        results = outcome.results
        warnings = outcome.warnings
        has_more = outcome.has_more
        total_estimate = outcome.total_estimate
        retrieval_mode = outcome.retrieval_mode
        # v0.7.6: attach conflict signals (open_table / conflict_guidance
        # sources), only on genuine query hits (direct mode). Failures degrade
        # silently.
        if include_conflict_signal and retrieval_mode == "direct" and results:
            results = self._attach_conflict_signals(results, extra_warnings)
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
        # never on browse/fallback/empty. Failures degrade to [] + warning.
        linked: list[dict[str, Any]] = []
        if include_linked_open_items and retrieval_mode == "direct" and results:
            linked = self._linked_open_items_for_search(
                self.db, results, extra_warnings,
                ws_canonical=ws_scope,
            )
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
        if include_size:
            # v0.15.2: one-shot size metering so callers can see what a
            # results page costs against pasting full texts; no feedback loop,
            # limit/offset semantics untouched.
            from ..tokens import estimate_tokens

            contents = [
                str(r.get("content") or "") for r in results if r.get("content") is not None
            ]
            returned_chars = sum(len(c) for c in contents)
            returned_tokens = sum(estimate_tokens(c) for c in contents)
            matched_total = int(total_estimate) if total_estimate is not None else len(results)
            beyond_count = max(0, matched_total - offset - len(results))
            avg_chars = returned_chars // max(1, len(contents)) if contents else 0
            display_hint = None
            if contents:
                display_hint = (
                    f"Surface the result-page cost to the user when presenting "
                    f"these results: ~{returned_tokens} tokens returned "
                    f"({len(contents)} item{'s' if len(contents) != 1 else ''})"
                    + (
                        f"; {beyond_count} more matched "
                        f"(~{beyond_count * avg_chars} chars) not returned."
                        if beyond_count
                        else "."
                    )
                )
            response_data["size"] = {
                "returned_chars": returned_chars,
                "returned_count": len(contents),
                "tokens_estimate": returned_tokens,
                "matched_beyond_limit_chars": beyond_count * avg_chars,
                "matched_beyond_limit_count": beyond_count,
                # The metering only pays off if the user actually sees results;
                # silence the hint on empty pages to avoid instructing the agent
                # about a cost that does not exist.
                "display_hint": display_hint,
            }
        try:
            response_data["unresolved_conflict_count"] = self.db.conflicts.count_open_conflicts(
                ws_scope,
            )
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
        - evidence channel: ``evidence_knn(parent_status_filter="expired")``
          with ``parent_status NOT IN ('active','deleted')`` predicate
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
            extra_warnings.append(f"vec_disabled={disabled_reason}")
            query_embedding = None
        elif query_embedding is None and query and self.settings.embedding_auto_query:
            embedder, ensure_warnings = self._ensure_embedder()
            extra_warnings.extend(ensure_warnings)
            if embedder is not None:
                try:
                    # Char-level pre-trim for pathological pastes; the token
                    # budget inside embed_text still makes the final cut.
                    er = embedder.embed_text(
                        prefix="", body=query,
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
                            extra_warnings.append(f"vec_disabled={reason}")
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
        **_: Any,
    ) -> dict[str, Any]:
        """Return one full memory by id, or a character window of it.

        ``span={"start": int, "end": int}`` returns the record with
        ``content`` sliced to that window (plus span metadata) so triage
        loops can deep-read only the region a clue pointed at instead of
        paying for the full text. Omitting ``span`` keeps the full read.
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
        span_start: int | None = None
        span_end: int | None = None
        if span is not None:
            if not isinstance(span, dict):
                return self.db.state.response({"error": "span must be an object with start/end"}, ok=False)
            raw_start = span.get("start")
            raw_end = span.get("end")
            if (
                not isinstance(raw_start, int) or isinstance(raw_start, bool)
                or not isinstance(raw_end, int) or isinstance(raw_end, bool)
            ):
                return self.db.state.response({"error": "span start/end must be integers"}, ok=False)
            span_start = raw_start
            span_end = raw_end
            if span_start < 0 or span_end <= span_start:
                return self.db.state.response({"error": "span requires 0 <= start < end"}, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        memory = self._get_memory_visible(memory_id_int, caller)
        if not memory:
            error_data: dict[str, Any] = {"error": f"memory id {memory_id_int} not found"}
            if caller.isolation == "strict":
                error_data.update(caller.response_fields())
            return self.db.state.response(error_data, ok=False, extra_warnings=list(caller.warnings))

        if span_start is not None and span_end is not None:
            content = str(memory.get("content") or "")
            if span_start >= len(content):
                return self.db.state.response(
                    {"error": "span start is past the end of the content",
                     "total_chars": len(content)},
                    ok=False,
                )
            clipped_end = min(span_end, len(content))
            windowed = dict(memory)
            windowed["content"] = content[span_start:clipped_end]
            data: dict[str, Any] = {
                "memory": windowed,
                "span": {"start": span_start, "end": clipped_end, "total_chars": len(content)},
            }
        else:
            data = {"memory": memory}
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

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
