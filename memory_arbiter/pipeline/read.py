"""Internal read, search, comparison, and conflict-signal operations."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..acl import CallerWorkspace
from ..embedder import ManagedEmbedder

from ..arbitration import compare_memories
from ..constants import EMBEDDING_MAX_SECTION_CHARS, SUPERSEDED_LIMIT, strict_ws
from ..evidence import local_text_units
from ..models import MemoryStatus
from ..search import search_memories
from ..tokens import meter_payloads

if TYPE_CHECKING:
    from ..tools import MemoryTools


# Verified/formally-recorded conflict signal sources that ring the loud
# attention flag. conflict_group is the conflict-groups producer; the retired
# names are kept so legacy payloads still resolve.
_STRONG_CONFLICT_SOURCES = ("open_table", "conflict_guidance", "conflict_group")

# v0.15.4 find index-page preview: bounded outline per result item.
_OUTLINE_MAX_SEGMENTS = 8
_OUTLINE_HEAD_CHARS = 40

# v0.15.10 content_mode="hits": merged hit spans covering >= this share of the
# full content upgrade the item to full text (hit_spans stays as an annotation).
# No truncation anywhere — the owner's ruling is that server-side picking of
# "the important hits" is a systematic bias (legal RAG loses provisos).
_HIT_SPANS_FULL_COVERAGE = 0.5

_CONTENT_MODES = ("preview", "hits", "full")

# 0.16.0 batch read (plan §1.5): the full-content byte budget. preview/hits
# payloads are structurally bounded, so a count cap is enough; full content
# needs the byte budget + a structured over-long response (never a silent
# truncation — the agent re-reads items individually).
from ..constants import (  # noqa: E402
    BATCH_READ_FULL_BUDGET_BYTES,
    BATCH_READ_FULL_BUDGET_MAX_BYTES,
)


def _unit_aligned_hits(
    db: Any, memory: dict[str, Any], span: "dict[str, Any] | None",
) -> "tuple[list[dict[str, Any]], str | None] | None":
    """Unit-aligned hit spans for an id-driven hits read (plan §6⑨, four-round
    final form): the ``hits`` unit selector on id-driven calls is the per-id
    span coordinate, and mema's content atom is the evidence unit — returning
    complete units removes any half-sentence truncation risk by construction.

    Returns (hit_spans, upgraded_full_content). Each hit_spans entry carries
    the complete unit text plus offsets in read's span coordinate system. When
    covered units span >= 50% of the content the item upgrades to full text
    (same rule as find/batch_find hits). Returns None when the memory has no
    evidence rows for its current version (caller falls back to the char
    window / plain preview).
    """
    try:
        rows = db.evidence.text_unit_rows(
            int(memory["id"]), int(memory.get("version") or 1),
            span_start=(int(span["start"]) if span else None),
            span_end=(int(span["end"]) if span else None),
        )
    except Exception:
        return None
    if not rows:
        return None
    content = str(memory.get("content") or "")
    hit_spans = [
        {
            "text": str(row["text"]),
            "start_offset": int(row["start_offset"]),
            "end_offset": int(row["end_offset"]),
            "unit_index": int(row["unit_index"]),
        }
        for row in rows
    ]
    upgraded: str | None = None
    covered = sum(item["end_offset"] - item["start_offset"] for item in hit_spans)
    if content and covered >= _HIT_SPANS_FULL_COVERAGE * len(content):
        upgraded = content
    return hit_spans, upgraded



def _content_outline(subject: str, content: str) -> list[dict[str, Any]]:
    """Bounded table-of-contents for a find preview item.

    Segments reuse the evidence pipeline's local_text_units (heading/text
    kinds only; the subject unit is excluded) so offsets share read's span
    coordinate system — span={"start": offset, "end": offset + N} slices the
    exact source region. Over-long contents collapse into a trailing
    "…还有 N 段" marker with offset=None.
    """
    units = [
        unit for unit in local_text_units(subject, content)
        if unit.kind in ("heading", "text")
    ]
    outline: list[dict[str, Any]] = [
        {
            "head": (unit.text.splitlines()[0] if unit.text else "")[:_OUTLINE_HEAD_CHARS],
            "offset": unit.start_offset,
        }
        for unit in units[:_OUTLINE_MAX_SEGMENTS]
    ]
    if len(units) > _OUTLINE_MAX_SEGMENTS:
        outline.append({"head": f"…还有 {len(units) - _OUTLINE_MAX_SEGMENTS} 段", "offset": None})
    return outline


def _hit_spans(raw_hits: Any, content: str) -> "list[dict[str, Any]] | None":
    """v0.15.10: build hit_spans from the evidence channel's unit-level hits.

    Two mandatory cleanups (adversarial-review findings, plan §2.3):
    - subject-kind hits are dropped: the subject unit's offsets are (0,0) and
      carry no content-span meaning (same reason outline excludes it);
    - overlapping/adjacent intervals are merged BEFORE any length math: the
      long-text fallback slices with overlap=60, so naive summation would
      double-count coverage (premature full-text upgrades) and surface
      duplicated text.

    ``text`` is sliced from the source content so the span and the text are
    strictly self-consistent: read span=[start_offset, end_offset] returns
    exactly this text. Returns None when nothing survives (FTS/phrase-only
    recall has no evidence hits — the item falls back to the preview shape).
    """
    intervals: list[tuple[int, int]] = []
    for h in raw_hits or []:
        if not isinstance(h, dict) or str(h.get("kind") or "") == "subject":
            continue
        s_raw, e_raw = h.get("start_offset"), h.get("end_offset")
        # Strict ints (bools rejected — v0.14 span-validation lesson): these
        # come from evidence rows, never user input, but stay defensive.
        if (
            isinstance(s_raw, bool) or isinstance(e_raw, bool)
            or not isinstance(s_raw, int) or not isinstance(e_raw, int)
        ):
            continue
        s, e = s_raw, e_raw
        if 0 <= s < e <= len(content):
            intervals.append((s, e))
    if not intervals:
        return None
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [
        {"text": content[s:e], "start_offset": s, "end_offset": e}
        for s, e in merged
    ]


def _preview_item(item: dict[str, Any], *, content_mode: str = "preview") -> dict[str, Any]:
    """Build one find index-page item: metadata + content_chars + outline.

    v0.15.10 content_mode (single-choice enum, replaces the removed
    include_content boolean):
    - "preview" (default): index page, no content — unchanged contract;
    - "hits": + hit_spans (vector-hit unit text + offsets in read's span
      coordinate system). No truncation: when merged hits cover >= 50% of the
      content the item upgrades to full text and hit_spans stays as an
      annotation — the server never picks "the important hits" for the agent;
    - "full": + full content (the old include_content=true escape hatch).

    Internal underscore debug fields are passed through untouched — the
    debug_ranking=true page contract exposes them, and search_memories strips
    them otherwise. The single exception is _evidence_hits: in "hits" mode it
    is consumed here (and removed) to build hit_spans.
    """
    content = str(item.get("content") or "")
    preview = dict(item)
    preview["content_chars"] = len(content)
    preview["outline"] = _content_outline(str(item.get("subject") or ""), content)
    keep_content = content_mode == "full"
    if content_mode == "hits":
        spans = _hit_spans(item.get("_evidence_hits"), content)
        if spans:
            covered = sum(e - s for s, e in (
                (sp["start_offset"], sp["end_offset"]) for sp in spans
            ))
            if covered >= _HIT_SPANS_FULL_COVERAGE * len(content):
                keep_content = True
            preview["hit_spans"] = spans
        # spans is None → vector channel contributed nothing on this item
        # (FTS/phrase-only recall, or the embedder is down): the item keeps
        # the plain preview shape.
        preview.pop("_evidence_hits", None)
    if not keep_content:
        preview.pop("content", None)
    return preview


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

    def memory_search(self, query: str = "", workspace: str | None = None, tags: list[str] | None = None, limit: int = 10, offset: int = 0, debug_ranking: bool = False, query_embedding: list[float] | None = None, tags_filter: list[str] | None = None, after_time: str | None = None, before_time: str | None = None, source_type: str | None = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True, include_size: bool | None = None, content_mode: str = "preview", **_: Any) -> dict[str, Any]:
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
        # never on browse/empty. Failures degrade to [] + warning.
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
        results = [_preview_item(r, content_mode=content_mode) for r in results]
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
                        "start/end offsets share read's span coordinates); items "
                        "whose hits cover >=50% of the content upgraded to full "
                        "text — hit_spans never truncates. Items without vector "
                        "hits keep the plain preview shape."
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
                for group in self.db.conflicts.list_open_conflicts_for_memory_ids(
                    sorted(page_ids), include_applying=True,
                ):
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
            extra_warnings.append(f"vec_disabled={disabled_reason}")
            return None
        if query_embedding is not None or not (query and self.settings.embedding_auto_query):
            return query_embedding
        embedder, ensure_warnings = self._ensure_embedder()
        extra_warnings.extend(ensure_warnings)
        if embedder is None:
            return None
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
                    return None
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
        # A none/weak explicit filter scopes recall in SQL (hard_scope) so the
        # limit is applied AFTER workspace scoping — never a post-page truncation.
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
            return _preview_item(clean, content_mode=content_mode)

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
        return self.db.state.response(response_data, extra_warnings=extra_warnings)

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
        **_: Any,
    ) -> dict[str, Any]:
        """Return one full memory by id, a unit-aligned window of it, or its preview.

        0.16.0 four-call content_mode unification (plan §6⑨): find/batch_find/
        read/batch_read share preview/hits/full semantics. ``read`` defaults to
        full (backward compatible). ``span={"start", "end"}`` selects complete
        evidence units overlapping the window — mema's content atom is the unit,
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

        content = str(memory.get("content") or "")
        data: dict[str, Any]
        if content_mode == "preview" and span is None:
            preview = {
                key: value for key, value in memory.items() if key != "content"
            }
            preview["content_chars"] = len(content)
            preview["outline"] = _content_outline(str(memory.get("subject") or ""), content)
            data = {"memory": preview}
        elif content_mode == "hits":
            unit_hits = _unit_aligned_hits(self.db, memory, span)
            if unit_hits is not None:
                hit_spans, upgraded = unit_hits
                record = {
                    key: value for key, value in memory.items() if key != "content"
                }
                record["content_chars"] = len(content)
                record["outline"] = _content_outline(str(memory.get("subject") or ""), content)
                record["hit_spans"] = hit_spans
                if upgraded is not None:
                    record["content"] = upgraded
                data = {"memory": record}
            else:
                # No evidence rows for the current version (fresh write before
                # the async index lands, or a down embedder): fall back to the
                # full record — an honest answer beats an empty hits page.
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
                    # Legacy fallback while no evidence rows exist.
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
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

    def memory_batch_read(
        self,
        memory_ids: "list[int] | None" = None,
        content_mode: str = "preview",
        spans: "dict[str, Any] | None" = None,
        **_: Any,
    ) -> dict[str, Any]:
        """0.16.0 batch read: read = the single-item special case of this call.

        Contract (plan §1.5, owner-pinned caps): memory_ids[] + content_mode in
        {preview, hits, full}; caps preview 50 / hits 50 / full 10 ids; full
        adds an 80KB byte budget (100KB hard ceiling) — an over-budget batch
        returns a structured over-long prompt and the agent re-reads items
        individually, never a silent truncation. ``spans`` maps memory_id to
        {start, end} and is the ``hits`` unit selector for id-driven calls
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
        cap = {"preview": BATCH_READ_MAX_PREVIEW, "hits": BATCH_READ_MAX_HITS, "full": BATCH_READ_MAX_FULL}[content_mode]
        wanted: list[int] = []
        seen: set[int] = set()
        for raw in memory_ids or []:
            try:
                mid = int(raw)
            except (TypeError, ValueError):
                return self.db.state.response(
                    {"error": "memory_ids must contain positive integer ids", "results": [], "count": 0},
                    ok=False,
                )
            if mid <= 0:
                return self.db.state.response(
                    {"error": "memory_ids must contain positive integer ids", "results": [], "count": 0},
                    ok=False,
                )
            if mid not in seen:
                seen.add(mid)
                wanted.append(mid)
        if not wanted:
            return self.db.state.response(
                {"error": "memory_ids must be a non-empty list", "results": [], "count": 0},
                ok=False,
            )
        if len(wanted) > cap:
            return self.db.state.response(
                {
                    "error": f'content_mode="{content_mode}" accepts at most {cap} ids per call',
                    "cap": cap, "results": [], "count": 0,
                },
                ok=False,
            )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied

        span_map: dict[int, dict[str, int]] = {}
        for key, span in (spans or {}).items():
            try:
                mid = int(str(key))
            except (TypeError, ValueError):
                continue
            if isinstance(span, dict):
                try:
                    start = int(span.get("start", 0))
                    end_raw = span.get("end")
                    span_map[mid] = {"start": start, "end": int(end_raw) if end_raw is not None else 0}
                except (TypeError, ValueError):
                    continue

        results: list[dict[str, Any]] = []
        not_found: list[int] = []
        for mid in wanted:
            memory = self._get_memory_visible(mid, caller)
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
                record["outline"] = _content_outline(str(memory.get("subject") or ""), content)
                item["memory"] = record
            elif content_mode == "hits":
                unit_hits = _unit_aligned_hits(self.db, memory, span)
                record = {key: value for key, value in memory.items() if key != "content"}
                record["content_chars"] = len(content)
                record["outline"] = _content_outline(str(memory.get("subject") or ""), content)
                if unit_hits is not None:
                    hit_spans, upgraded = unit_hits
                    record["hit_spans"] = hit_spans
                    if upgraded is not None:
                        record["content"] = upgraded
                item["memory"] = record
            else:  # full
                record = dict(memory)
                if span is not None and span["end"] > span["start"]:
                    if span["start"] < len(content):
                        clipped_end = min(span["end"], len(content))
                        rows = self.db.evidence.text_unit_rows(
                            mid, int(memory.get("version") or 1),
                            span_start=span["start"], span_end=clipped_end,
                        )
                        if rows:
                            # Unit-aligned: contiguous slice from the first to
                            # the last covered unit (units may overlap; joining
                            # them would duplicate text).
                            first_start = min(int(row["start_offset"]) for row in rows)
                            last_end = max(int(row["end_offset"]) for row in rows)
                            record["content"] = content[first_start:last_end]
                        else:
                            record["content"] = content[span["start"]:clipped_end]
                    else:
                        record["content"] = ""
                        item["span_past_end"] = True
                item["memory"] = record
            results.append(item)

        over_budget: list[dict[str, Any]] = []
        if content_mode == "full":
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
                        "batch read over the full-content budget; no contents were returned — "
                        "read the items individually (memory action='read') or narrow the batch"
                    ),
                }
                if self.settings.include_size:
                    data["size"] = meter_payloads(slim_results)
                if caller.isolation == "strict":
                    data.update(caller.response_fields())
                return self.db.state.response(data, extra_warnings=list(caller.warnings))

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
