"""Read/search operations for MemoryTools (Phase 4 extraction)."""
# mypy: disable-error-code=no-any-return
from __future__ import annotations

import hashlib
from typing import Any, Optional, TYPE_CHECKING

from ..arbitration import compare_memories
from ..constants import strict_ws
from ..models import MemoryStatus
from ..search import search_memories

if TYPE_CHECKING:
    from ..tools import MemoryTools


class ReadPipeline:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

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

    def memory_search(self, query: str = "", workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, offset: int = 0, debug_ranking: bool = False, query_embedding: Optional[list[float]] = None, tags_filter: Optional[list[str]] = None, after_time: Optional[str] = None, before_time: Optional[str] = None, source_type: Optional[str] = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True, **_: Any) -> dict[str, Any]:
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
                    er = embedder.embed_text(prefix="", body=query)
                    if er.embedding:
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
        ws_canonical = caller.canonical if isolation != "none" else None
        workspace = caller.workspace if isolation != "none" else workspace
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
        )
        results = outcome.results
        warnings = outcome.warnings
        has_more = outcome.has_more
        total_estimate = outcome.total_estimate
        retrieval_mode = outcome.retrieval_mode
        # v0.6.0: attach section enhancement to active-split results
        results = self._attach_sections(results, query_embedding, extra_warnings)
        # v0.7.6: attach conflict signals (open_table + runtime_metadata_hint),
        # only on genuine query hits (direct mode). Failures degrade silently.
        if include_conflict_signal and retrieval_mode == "direct" and results:
            results = self._attach_conflict_signals(results, extra_warnings)
        # v0.8.7: promote conflict_signal to a loud top-level flag (mirrors the
        # write path's attention_required). If any direct hit carries a
        # conflict_signal, surface a one-line summary at data top level so the
        # calling agent notices it on a quick scan instead of having to inspect
        # each result's nested conflict_signal.
        attention_required = False
        attention_summary: Optional[str] = None
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
            # open_table signals. runtime_metadata_hint is advisory (unverified,
            # and re-fires on every retrieval of an overlapping pair), so a
            # must-surface flag would nag. It stays a per-result signal for the
            # calling agent to judge by content (see memory_search docstring):
            # surface only if the two genuinely contradict, else silently proceed.
            ot = seen_sources.get("structured_claim_candidate") or seen_sources.get("open_table")
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
                    {"open_table", "structured_claim_candidate"}
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
                ws_canonical=strict_ws(isolation, ws_canonical),
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
        }
        if attention_required:
            response_data["attention_required"] = True
            response_data["attention_summary"] = attention_summary
            strong_signal = next(
                (
                    r.get("conflict_signal") for r in results
                    if (r.get("conflict_signal") or {}).get("conflict_source")
                    in {"structured_claim_candidate", "open_table"}
                ),
                None,
            )
            if strong_signal and strong_signal.get("conflict_source") == "structured_claim_candidate":
                response_data["action_required"] = "judge_conflict_before_use"
                response_data["verification_status"] = "pending_llm"
            elif strong_signal and strong_signal.get("action_required"):
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
        workspace: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 20,
        debug_ranking: bool = False,
        query_embedding: Optional[list[float]] = None,
        tags_filter: Optional[list[str]] = None,
        after_time: Optional[str] = None,
        before_time: Optional[str] = None,
        source_type: Optional[str] = None,
        include_conflict_signal: bool = True,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        """v0.9.4: search expired (non-active non-deleted) memories with vec-hybrid recall.

        Searches ONLY non-active, non-deleted memories (superseded +
        conflicted + pending) for audit/history walkthroughs:
        - vec channel: ``vec_knn(parent_status_filter="expired")`` with
          ``parent_status NOT IN ('active','deleted')`` predicate
        - FTS channel: ``search_memories(status_filter="expired")`` with
          ``status_clause = "m.status NOT IN ('active','deleted')"``

        ``limit`` controls the per-page cap (default 20, hard cap 50,
        configurable via ``MEMORY_ARBITER_SUPERSEDED_LIMIT``). ``offset``
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
                    er = embedder.embed_text(prefix="", body=query)
                    if er.embedding:
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
        # Apply superseded_limit cap
        superseded_limit_cap = self.settings.superseded_limit
        effective_limit = min(max(1, limit_requested), max(1, int(superseded_limit_cap)), 50)

        # v0.12.5: expired recall uses the shared caller-workspace resolver.
        isolation = self.settings.isolation
        caller = self._caller_workspace(workspace)
        ws_canonical = caller.canonical if isolation != "none" else None
        workspace = caller.workspace if isolation != "none" else workspace
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
        )
        results = outcome.results
        warnings = outcome.warnings
        has_more = outcome.has_more
        total_estimate = outcome.total_estimate
        retrieval_mode = outcome.retrieval_mode

        # v0.6.0: attach section enhancement to active-split results
        results = self._attach_sections(results, query_embedding, extra_warnings)

        # v0.7.6: attach conflict signals (strict expired results are non-active
        # and may lack safe workspace summaries; fail closed by omitting signals).
        if include_conflict_signal and isolation != "strict" and retrieval_mode == "direct" and results:
            results = self._attach_conflict_signals(results, extra_warnings)

        attention_required = False
        attention_summary: Optional[str] = None
        if include_conflict_signal and retrieval_mode == "direct" and results:
            seen_sources: dict[str, dict[str, Any]] = {}
            for r in results:
                sig = r.get("conflict_signal")
                if not sig:
                    continue
                seen_sources.setdefault(str(sig.get("conflict_source", "conflict")), r)
            ot = seen_sources.get("structured_claim_candidate") or seen_sources.get("open_table")
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
                    {"structured_claim_candidate", "open_table"}
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
        }
        if attention_required:
            response_data["attention_required"] = True
            response_data["attention_summary"] = attention_summary
            strong_signal = next(
                (
                    r.get("conflict_signal") for r in results
                    if (r.get("conflict_signal") or {}).get("conflict_source")
                    in {"structured_claim_candidate", "open_table"}
                ),
                None,
            )
            if strong_signal and strong_signal.get("conflict_source") == "structured_claim_candidate":
                response_data["action_required"] = "judge_conflict_before_use"
                response_data["verification_status"] = "pending_llm"
            elif strong_signal and strong_signal.get("action_required"):
                response_data["action_required"] = strong_signal.get("action_required")
                response_data["verification_status"] = strong_signal.get("verification_status")
        if caller.isolation == "strict":
            response_data.update(caller.response_fields())
        return self.db.state.response(
            response_data,
            extra_warnings=extra_warnings + warnings + list(caller.warnings),
        )

    # v0.8 split-status values the new flow may write. Anything else is a
    # legacy/unknown status surfaced read-only for repair (design §5.2).
    _V08_SPLIT_STATUSES = (None, "active", "failed", "declined")

    def memory_get(
        self,
        memory_id: int,
        sections: str = "catalog",
        section_ids: Optional[list[int]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """通过 ID 获取一条记忆的全文、分段目录或指定 section 原文（只读）。

        v0.8（§6.4）合并了原 get_sections / memory_split_status 的读取能力：
          sections: none | catalog | all（默认 catalog）。matched 非法（get
                    没有 search 上下文）。
          section_ids: 优先于 sections；不存在或不属于该 memory 的 ID 进入
                       missing_section_ids，不会因单个缺失让整个调用失败。
        返回 split 子对象（status / legacy_status / revision / section_count /
        content_hash）。全局 vec 状态留在 memory_status / doctor。
        """
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer"}, ok=False)
        if sections not in ("none", "catalog", "all"):
            return self.db.state.response(
                {"error": "sections must be one of none|catalog|all (matched is not valid without a search context)"},
                ok=False,
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
            return self.db.state.response(error_data, ok=False, extra_warnings=list(caller.warnings))

        content = memory.get("content") or ""
        data: dict[str, Any] = {"memory": memory}
        if caller.isolation == "strict":
            data.update(caller.response_fields())

        # ---- split sub-object ----
        raw_status = memory.get("split_status")
        legacy_status = raw_status if raw_status not in self._V08_SPLIT_STATUSES else None
        all_sections = self.db.get_sections_by_memory(memory_id_int)
        data["split"] = {
            "status": raw_status,
            "legacy_status": legacy_status,
            "revision": int(memory.get("split_revision") or 0),
            "section_count": len(all_sections),
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None,
        }

        # ---- section_ids takes precedence over the sections mode ----
        if section_ids:
            found, missing = self.db.get_sections_by_ids(memory_id_int, section_ids)
            data["sections"] = [
                {**s, "content": content[s["start_offset"]:s["end_offset"]]}
                for s in found
            ]
            data["missing_section_ids"] = missing
            # When explicit IDs are requested, also surface a catalog so the
            # caller sees the surrounding sections.
            data["section_catalog"] = [self._catalog_entry(s) for s in all_sections]
            return self.db.state.response(data)

        if sections == "none":
            return self.db.state.response(data)
        if sections == "catalog":
            data["section_catalog"] = [self._catalog_entry(s) for s in all_sections]
            return self.db.state.response(data)
        # sections == "all"
        data["section_catalog"] = [self._catalog_entry(s) for s in all_sections]
        data["sections"] = [
            {**s, "content": content[s["start_offset"]:s["end_offset"]]}
            for s in all_sections
        ]
        return self.db.state.response(data)

    def memory_store_embedding(self, memory_id: int, embedding: list[float], **_: Any) -> dict[str, Any]:
        """Store or replace an embedding for a memory (v0.3.1 semantic recall).

        The caller is responsible for generating the embedding with any model
        of matching dimension. memory-arbiter does not bundle an embedding
        model by design (local-first, zero cloud, no heavy deps). See
        docs/semantic_example.py for a backfill script using sentence-transformers.
        """
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer"}, ok=False)
        if not isinstance(embedding, list) or not embedding:
            return self.db.state.response({"error": "embedding must be a non-empty list of floats"}, ok=False)
        if not self.db.get_memory(memory_id_int):
            return self.db.state.response({"error": f"memory id {memory_id_int} not found"}, ok=False)
        ok, store_warnings = self.db.store_embedding(memory_id_int, embedding)
        return self.db.state.response({"stored": ok, "memory_id": memory_id_int, "dimensions": len(embedding)}, ok=ok, extra_warnings=store_warnings)

    def memory_recent(self, workspace: Optional[str] = None, limit: int = 20, **_: Any) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        caller = self._caller_workspace(workspace)
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation == "strict" and caller.canonical:
            results = self.db.list_memories_for_workspace(caller.canonical, limit=limit)
        else:
            results = self.db.list_memories(limit=limit)
        data = {"results": results, "count": len(results)}
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

    def memory_compare(self, left_id: Optional[int] = None, right_id: Optional[int] = None, left: Optional[dict[str, Any]] = None, right: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
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
