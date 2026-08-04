from __future__ import annotations

import sys
from typing import Any, Optional

from .config import Settings
from .tools import MemoryTools


def build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError(
            "MCP Python SDK is not installed. Install with `pip install -r requirements.txt` "
            "or `pip install mcp`, then run `memory-arbiter-mcp` again."
        ) from exc

    app = FastMCP("memory-arbiter-mcp")
    tools = MemoryTools(Settings.from_env())
    tools.start_update_monitor()
    tools.start_split_worker()

    @app.tool()
    def memory_write(
        content: str,
        agent_id: Optional[str] = None,
        workspace: Optional[str] = None,
        tags: Optional[list[str]] = None,
        source_type: str = "unknown",
        source_ref: Optional[str] = None,
        event_time: Optional[str] = None,
        ingest_time: Optional[str] = None,
        confidence: float = 0.5,
        protection_level: str = "normal",
        status: str = "active",
        subject: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Write one structured memory into the cross-tool shared store. content is required; subject/tags/source_type are strongly recommended.

tags are the primary ranking and filter signal (a tag exact-match outweighs content). Tag with "query-intent words" — terms a user might later search with. Examples:
  release note -> tags include "release" + version number
  technical decision -> tags include "decision" + topic word
  user preference -> tags include "preference" + preference type
Avoid re-tagging words already in subject (redundant, no retrieval gain).

v0.5.0: with GGUF embedding + sqlite-vec configured, the vector is stored automatically on write; the response only echoes embedding_stored when vectorization was actually attempted.

v0.9 conflict gate: ALWAYS inspect `action_required`, `verification_status`, and `conflict_judgment_requests`. `action_required=judge_conflict_before_use` means deterministic claims collided: before using either value, read both evidence sides and call `memory_submit_conflict_judgment` with the exact snapshot pins. If that returns `user_action_required=true`, ask the user. A submitted LLM judgment records reusable guidance but never edits or supersedes memory. Also inspect advisory `write_hints` for possible duplicate/evolution records; those are separately dismissable. Never silently ignore a top-level `attention_required`."""
        return tools.memory_write(
            content=content,
            agent_id=agent_id,
            workspace=workspace,
            tags=tags or [],
            source_type=source_type,
            source_ref=source_ref,
            event_time=event_time,
            ingest_time=ingest_time,
            confidence=confidence,
            protection_level=protection_level,
            status=status,
            subject=subject,
            metadata=metadata or {},
        )

    @app.tool()
    def memory_search(query: str = "", workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, debug_ranking: bool = False, query_embedding: Optional[list[float]] = None, tags_filter: Optional[list[str]] = None, after_time: Optional[str] = None, before_time: Optional[str] = None, source_type: Optional[str] = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True, include_superseded: Optional[bool] = None) -> dict[str, Any]:
        """Retrieve active memories by relevance. limit is page size (default 10), not a result cap. has_more=true means more unreturned active results may exist; memory_search has no offset cursor, so narrow with a more specific query, a larger limit (max 100), or tags_filter. For paginated expired/history recall use memory_search_expired.

v0.9.4: active-query split — ``memory_search`` returns ONLY ``status='active'`` memories. Superseded/deleted memories are excluded at the database level. For superseded history recall use ``memory_search_expired``.

For project knowledge, past decisions, preferences, or doc-summary questions, search memory before reading source files. Prefer 2-4 core keywords; if nothing hits, retry with synonyms/shorter terms; an empty query or memory_recent lists recent memories. debug_ranking=true returns ranking debug fields.

Parameters:
  query: search terms. An empty query still works: with no filters it returns recent memories via fallback; WITH filters (tags_filter / after_time / before_time / source_type) it runs filter-driven recall — e.g. "list all entries with tag X" — ordered by ingest_time (newest first). When mixing ASCII identifiers with CJK terms, separate with spaces (e.g. "v0.7.2 release" not "v0.7.2release"), otherwise mixed tokens take an equality path and may miss.
  tags_filter: strict AND filter — the memory's tags must contain every listed tag. Note: enabling tags_filter usually disables vector semantic recall (vector candidates' tags are unrelated to the literal query and get cut by the exact-match post-filter).
  after_time / before_time: ISO 8601 time window (filters on ingest_time; naive values treated as UTC). Invalid formats are ignored with a warning.
  source_type: filter by source type (user_confirmed / agent_generated / document_extracted, etc.).
  include_linked_open_items: default true. On a query hit, attaches up to 5 same-topic active todos (tags containing "todo") in linked_open_items. Pass false to suppress in noisy contexts. Fires only on a real hit (retrieval_mode=direct).

include_conflict_signal: default true. When a hit involves open conflicts, attaches a conflict_signal field (sources: open_table = structured conflicts recorded by scan/record; runtime_metadata_hint = runtime heuristic, not LLM-verified). Fires only on a real hit (retrieval_mode=direct).

Conflict surfacing (v0.8.8): a direct hit may carry a per-result `conflict_signal` (sources: `open_table` = scan/record-verified; `runtime_metadata_hint` = advisory, unverified). Handle by source:
- `open_table` (verified): sets a top-level `attention_required` flag + `attention_summary`. Surface it to the user — e.g. "⚠️ 命中的 #N 与 #M 存在已记录冲突,引用前请核实"; they can dismiss it with `memory_resolve_conflict(status='not_a_conflict')` — once dismissed, the pair carries no conflict_signal (Layer 0), so it stops surfacing until one side is edited.
- `runtime_metadata_hint` (advisory): does NOT set the top-level flag (it re-fires on every retrieval of an overlapping pair, so a must-surface flag would nag). It stays a per-result signal — JUDGE it by content before saying anything: compare the hit's content against `conflict_signal.conflict_peer`'s snippet (or `memory_get` the peer if the snippet is too thin). Only mention it to the user if the two genuinely contradict; if they are merely topically related, silently proceed. This keeps advisory's recall (catching pairs the verified paths missed) without nagging on false positives.

v0.9 structured routing: `structured_claim_candidate` + `pending_llm` is a hard host-LLM gate; run the included judgment request before using the value. `open_table` + `pending_user` requires the user. `conflict_guidance` means a current LLM/policy/human judgment already exists: follow its recommended use without re-prompting, and include its disclosure when `disclosure_required=true`. Any source version/claim-revision change invalidates that guidance and reopens the gate.

Note: tags_filter is AND semantics — every listed tag must be present. Suited to: finding the N most relevant entries, exhaustive queries with filters, and structured listing via empty query + filters.

v0.5.0: with GGUF embedding + sqlite-vec configured, the query is vectorized automatically even without query_embedding; an explicit query_embedding still takes precedence."""
        kwargs = dict(
            query=query, workspace=workspace, tags=tags or [], limit=limit,
            debug_ranking=debug_ranking, query_embedding=query_embedding,
            tags_filter=tags_filter, after_time=after_time, before_time=before_time,
            source_type=source_type, include_linked_open_items=include_linked_open_items,
            include_conflict_signal=include_conflict_signal,
        )
        if include_superseded is not None:
            kwargs["include_superseded"] = include_superseded
        return tools.memory_search(**kwargs)

    @app.tool()
    def memory_search_expired(
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
    ) -> dict[str, Any]:
        """v0.9.4: search expired (non-active non-deleted) memories with vec-hybrid recall.

        Searches ONLY non-active, non-deleted memories (superseded + conflicted
        + pending) for audit/history walkthroughs:
        - vec channel: ``vec_knn(parent_status_filter="expired")`` with
          ``parent_status NOT IN ('active','deleted')`` predicate
        - FTS channel: ``search_memories(status_filter="expired")`` with
          ``status_clause = "m.status NOT IN ('active','deleted')"``

        ``limit`` controls the per-page cap (default 20, hard cap 50,
        configurable via ``MEMORY_ARBITER_SUPERSEDED_LIMIT``). ``offset``
        enables cursor pagination: exact on the empty-query+filters path (SQL
        OFFSET backed by a precise count), best-effort on the query-recall
        path (candidate pool windowed to offset+limit; deep pages may return
        empty since relevance recall has no exact total).

        Active-query split (§3.5): ``memory_search`` (active only) and
        ``memory_search_expired`` (expired only) are two independent queries.
        """
        return tools.memory_search_expired(
            query=query, workspace=workspace, tags=tags or [], limit=limit,
            debug_ranking=debug_ranking, query_embedding=query_embedding,
            tags_filter=tags_filter, after_time=after_time, before_time=before_time,
            source_type=source_type, include_conflict_signal=include_conflict_signal,
            offset=offset,
        )

    @app.tool()
    def memory_resync_vec_parent_status(
        dry_run: bool = True,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """v0.9.4: repair vec.parent_status to match memories.status.

        Scans memories_vec and memory_sections_vec for rows where
        ``parent_status != COALESCE(memories.status, 'deleted')`` and updates
        them in-place. This fixes drift caused by direct DB edits, migration
        bugs, or failed transactions.

        ``dry_run=True`` (default) only reports how many rows would be updated.
        Set ``dry_run=False`` to apply the repair. Per design §3.5 N16 this is a
        non-destructive UPDATE (it only aligns ``parent_status`` to the existing
        ``memories.status``; no content rewritten, no vectors deleted), so it does
        NOT require ``authorized=True`` — that parameter is a no-op compatibility
        placeholder kept for signature parity with other repair tools.
        """
        return tools.memory_resync_vec_parent_status(
            dry_run=dry_run, authorized=authorized
        )

    @app.tool()
    def memory_get(
        memory_id: int,
        sections: str = "catalog",
        section_ids: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Get a single memory's full text, section catalog, or specified section text by ID (read-only).

v0.8.0 merged the former get_sections / memory_split_status read paths:
  sections: none | catalog | all (default catalog). "matched" is illegal (get
            has no search context).
  section_ids: takes precedence over sections; IDs that don't exist or don't
            belong to this memory go into missing_section_ids — a single missing
            ID never fails the whole call.
Returns a split sub-object (status / legacy_status / revision / section_count /
content_hash). Global vec state lives in memory_status / doctor."""
        return tools.memory_get(memory_id=memory_id, sections=sections, section_ids=section_ids)

    @app.tool()
    def memory_store_embedding(memory_id: int, embedding: list[float]) -> dict[str, Any]:
        """Manually store or replace the semantic vector for a memory. With v0.5.0 GGUF embedding configured, new writes and ordinary queries vectorize automatically; this tool still suits backfill, non-GGUF models, remote APIs, or custom vector pipelines. The vector dimension must match vec.dim."""
        return tools.memory_store_embedding(memory_id=memory_id, embedding=embedding)

    @app.tool()
    def memory_recent(workspace: Optional[str] = None, limit: int = 20) -> dict[str, Any]:
        """List recent memories (no keyword filtering). As of v0.7.4, workspace is reserved metadata — results span the whole DB and are no longer filtered by workspace; the parameter is kept only for interface stability. Use when keywords are uncertain, memory_search returns nothing, or you want to browse the store before deciding whether to read source files."""
        return tools.memory_recent(workspace=workspace, limit=limit)

    @app.tool()
    def memory_compare(left_id: int, right_id: int) -> dict[str, Any]:
        """Low-frequency diagnostic tool: compare two memories by rule priority (protection -> event_time -> source_type -> confidence -> ingest_time) and return an explainable comparison reason; it records no conflict. For daily conflict discovery, rely on memory_search's conflict_signal (open_table / runtime_metadata_hint) or the scan_conflict_candidates -> record_conflict workflow."""
        return tools.memory_compare(left_id=left_id, right_id=right_id)

    @app.tool()
    def memory_arbitrate(left_id: int, right_id: int, mark_conflict: bool = True, authorized: bool = False, apply: Optional[bool] = None) -> dict[str, Any]:
        """Legacy manual arbitration tool. mark_conflict=true uses the legacy record_conflict path (without v0.7.5 enrichment fields). With authorized=true, the non-protected loser is automatically marked superseded; authorized defaults to false, so only the comparison is returned unless a human has confirmed. The new conflict workflow (scan_conflict_candidates -> record_conflict -> list_conflicts -> supersede/resolve) is preferred for daily use; this tool is not the main entry point. The old `apply` parameter was renamed to `authorized` in v0.8.5; passing `apply` now returns an explicit migration error instead of being silently ignored."""
        return tools.memory_arbitrate(left_id=left_id, right_id=right_id, mark_conflict=mark_conflict, authorized=authorized, apply=apply)

    @app.tool()
    def memory_list_conflicts(status: str = "open", limit: int = 50) -> dict[str, Any]:
        """List memory conflict records; by default only open ones are returned."""
        return tools.memory_list_conflicts(status=status, limit=limit)

    @app.tool()
    def memory_scan_conflict_candidates(
        workspace: Optional[str] = None,
        top_k: int = 8,
        max_pairs: int = 200,
        max_distance: float = 12.0,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """Vector-recall candidate conflict pairs (no LLM). Incremental scan (only newly added + recently edited memories), pair dedup, same-workspace filter, distance cutoff. Each memory's embedding runs top-K nearest neighbors; pairs are canonicalized (left<right). When sqlite-vec is unavailable, returns scanned=False with a hint (a config state, not an error). After receiving candidate pairs, the agent should run an LLM comparison on each pair, then persist the verdict via memory_record_conflict. Note: this tool is designed for the agent-side scheduled/manual scan loop; it is not meant to be called from ordinary conversation."""
        return tools.memory_scan_conflict_candidates(
            workspace=workspace, top_k=top_k, max_pairs=max_pairs,
            max_distance=max_distance, incremental=incremental,
        )

    @app.tool()
    def memory_record_conflict(
        left_id: int,
        right_id: int,
        reason: str,
        conflict_type: Optional[str] = None,
        conflict_point: Optional[str] = None,
        suggested_winner: Optional[int] = None,
        confidence_hint: Optional[str] = None,
        source: Optional[str] = None,
        refresh: bool = False,
        left_version: Optional[int] = None,
        right_version: Optional[int] = None,
        scan_prompt_version: Optional[str] = None,
        scan_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record a conflict with scan-enrichment fields (conflict_type/conflict_point/suggested_winner/confidence_hint/source). Canonical pair (left<right) + idempotent (an existing open pair returns deduped=True without rewriting). refresh=true updates the enrichment fields of an existing row (returns refreshed), used by the scheduled scan task to rewrite after re-judging. source marks the suggestion origin (e.g. llm_informed). conflict_type may be contradiction/evolution/etc.; evolution specifically means same-topic evolution residue (stale_active_memory): the newer version should supersede the older, but both are still active. Note: this tool is designed for the agent-side scheduled/manual scan loop; it is not meant to be called from ordinary conversation."""
        return tools.memory_record_conflict(
            left_id=left_id, right_id=right_id, reason=reason,
            conflict_type=conflict_type, conflict_point=conflict_point,
            suggested_winner=suggested_winner, confidence_hint=confidence_hint,
            source=source, refresh=refresh,
            left_version=left_version, right_version=right_version,
            scan_prompt_version=scan_prompt_version, scan_model=scan_model,
        )

    @app.tool()
    def memory_resolve_conflict(conflict_id: int, reason: str = "", status: str = "resolved") -> dict[str, Any]:
        """Close a single open conflict by conflict_id. ``status``: 'resolved' (default) or 'not_a_conflict' (v0.8.8 — the pair was a false positive; write/search then skip it via Layer 0 until a version change). Unlike memory_supersede's resolve_conflicts_for (which closes all conflicts involving a memory), this tool closes only the specified one."""
        return tools.memory_resolve_conflict(conflict_id=conflict_id, reason=reason, status=status)

    @app.tool()
    def memory_submit_conflict_judgment(
        conflict_id: int,
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        confidence_hint: Optional[str],
        reason: str,
        affects_current_output: bool,
        usage_context: str,
        judge_ref: Optional[str] = None,
    ) -> dict[str, Any]:
        """Required v0.9 host-LLM receipt for a structured_claim_candidate. Submit only after reading both evidence sides. verdict: contradiction|evolution|compatible|uncertain; recommended_use: left|right|contextual|merge|ask_user|none; usage_context: answer|code|config|memory_write|external_action|unrelated|unknown. Snapshot versions and claim revisions are mandatory CAS pins: stale judgments are rejected. The result may require user action under protected/high-impact policy. This never edits or supersedes a memory."""
        return tools.memory_submit_conflict_judgment(
            conflict_id=conflict_id,
            expected_left_version=expected_left_version,
            expected_right_version=expected_right_version,
            expected_left_claim_revision=expected_left_claim_revision,
            expected_right_claim_revision=expected_right_claim_revision,
            verdict=verdict,
            recommended_use=recommended_use,
            suggested_winner=suggested_winner,
            confidence_hint=confidence_hint,
            reason=reason,
            affects_current_output=affects_current_output,
            usage_context=usage_context,
            judge_ref=judge_ref,
        )

    @app.tool()
    def memory_correct_conflict_judgment(
        conflict_id: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        reason: str,
        expected_judgment_id: int,
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
        authorized: bool = False,
        judge_ref: Optional[str] = None,
    ) -> dict[str, Any]:
        """Append an authorized human correction to a v0.9 conflict judgment. The old LLM/policy judgment remains in append-only history; the new human judgment becomes active. Requires authorized=true and exact active-judgment plus memory/claim snapshot CAS. Never edits either memory."""
        return tools.memory_correct_conflict_judgment(
            conflict_id=conflict_id, verdict=verdict,
            recommended_use=recommended_use, suggested_winner=suggested_winner,
            reason=reason, expected_judgment_id=expected_judgment_id,
            expected_left_version=expected_left_version,
            expected_right_version=expected_right_version,
            expected_left_claim_revision=expected_left_claim_revision,
            expected_right_claim_revision=expected_right_claim_revision,
            authorized=authorized, judge_ref=judge_ref,
        )

    @app.tool()
    def memory_list_conflict_judgments(conflict_id: int) -> dict[str, Any]:
        """List append-only LLM/policy/human judgments for one conflict, oldest first. Read-only."""
        return tools.memory_list_conflict_judgments(conflict_id=conflict_id)

    @app.tool()
    def memory_set_entity(
        memory_id: int,
        entity: Optional[str] = None,
        scope: Optional[str] = None,
        clear: bool = False,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """Set a memory's canonical metadata.entity and optional metadata.scope for v0.9 structured claims. Values are lexically normalized. This does not create content history or bump memory.version, but a semantic entity/scope change increments claim_revision and immediately rebuilds claims. locked/user_confirmed memories require authorized=true."""
        return tools.memory_set_entity(
            memory_id=memory_id, entity=entity, scope=scope,
            clear=clear, authorized=authorized,
        )

    @app.tool()
    def memory_list_entities(limit: int = 50, include_unassigned: bool = True) -> dict[str, Any]:
        """List canonical metadata.entity values across active memories with counts, samples, and a bounded list of unassigned memory ids for incremental backfill. Read-only."""
        return tools.memory_list_entities(limit=limit, include_unassigned=include_unassigned)

    @app.tool()
    def memory_rebuild_claims(
        memory_ids: Optional[list[int]] = None,
        dry_run: bool = True,
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """Idempotently rebuild v0.9 deterministic claims and reconcile structured conflicts. dry_run=true returns the bounded plan; execution is disabled while structured_claim_mode=off."""
        return tools.memory_rebuild_claims(
            memory_ids=memory_ids, dry_run=dry_run, batch_size=batch_size,
        )

    @app.tool()
    def memory_confirm(memory_id: int, source_ref: Optional[str] = None, confidence: float = 1.0, authorized: bool = False) -> dict[str, Any]:
        """Mark an active memory as user-confirmed, promoting it to source_type=user_confirmed + protection_level=locked so it cannot be overwritten automatically. Requires authorized=true — promotion to the highest trust/protection tier must be an explicit, human-confirmed action. Superseded/deleted memories cannot be confirmed/reactivated; write a new active memory instead."""
        return tools.memory_confirm(memory_id=memory_id, source_ref=source_ref, confidence=confidence, authorized=authorized)

    @app.tool()
    def memory_activate(memory_id: int, authorized: bool = False) -> dict[str, Any]:
        """Activate a memory held as pending by strict workspace isolation. When isolation=strict and a write introduces a brand-new workspace, the memory is written as status=pending (excluded from active recall) and the write response carries action_required=confirm_new_workspace. This tool flips it to active so it becomes recallable. Requires authorized=true. Unlike memory_confirm, this does NOT promote to user_confirmed/locked — it only clears the strict-isolation gate."""
        return tools.memory_activate(memory_id=memory_id, authorized=authorized)

    @app.tool()
    def memory_supersede(
        memory_id: int,
        reason: str,
        superseded_by: Optional[int] = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """Explicitly supersede a memory, bypassing user_confirmed/locked protection (use when memory_arbitrate is blocked). Requires authorized=true. Side effects: protection level is lowered and related open conflicts are marked resolved; an audit row is written to the conflicts table. Irreversible."""
        return tools.memory_supersede(
            memory_id=memory_id,
            reason=reason,
            superseded_by=superseded_by,
            authorized=authorized,
        )

    @app.tool()
    def memory_status() -> dict[str, Any]:
        """Show memory-arbiter runtime status: database path, degradation mode, client identity, policy config, config-parse warnings, and whether auto-embedding is configured."""
        return tools.memory_status()

    @app.tool()
    def memory_audit_summary() -> dict[str, Any]:
        """Return a per-workspace memory statistics overview: entry counts, oldest/newest entry times, open-conflict count, and source_type distribution. Pure SQL aggregation, no semantic judgment — use it to quickly decide whether a deeper review is needed."""
        return tools.memory_audit_summary()

    @app.tool()
    def memory_doctor_overview(deep: bool = False) -> dict[str, Any]:
        """Run a health check on memory-arbiter and return a graded diagnostic report (read-only). Covers config integrity, the vectorization enablement chain, section splitting, data consistency, and capacity buildup. Each finding carries a severity and a fix_hint tailored to the current config. With deep=true, it additionally loads the GGUF model for a dimension probe (seconds of overhead)."""
        return tools.memory_doctor_overview(deep=deep)

    @app.tool()
    def memory_edit(
        memory_id: int,
        new_content: Optional[str] = None,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        new_subject: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
        reason: str = "",
        authorized: bool = False,
        tags_only: bool = False,
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Edit a memory's content or tags in place. Three modes: new_content for full replacement, old_text+new_text for precise local replacement, or tags_only=true with add_tags/remove_tags to update tags only. tags-only mode writes no memory_history, increments no version, recomputes no embedding, and triggers no re-segmentation (FTS still syncs tags); locked/user_confirmed memories require authorized=true. As of v0.7.6, complete a todo with tags_only=true + remove_tags=["todo"]. v0.9: content edits rebuild structured claims. If the response says action_required=judge_conflict_before_use, complete the returned memory_submit_conflict_judgment request before using either claim; do not silently ignore attention_required."""
        return tools.memory_edit(
            memory_id=memory_id,
            new_content=new_content,
            old_text=old_text,
            new_text=new_text,
            new_subject=new_subject,
            new_tags=new_tags,
            reason=reason,
            authorized=authorized,
            tags_only=tags_only,
            add_tags=add_tags,
            remove_tags=remove_tags,
        )

    @app.tool()
    def memory_history(memory_id: int) -> dict[str, Any]:
        """View a memory's version history (snapshots from the memory_history table, ordered by version descending). Read-only, touches no tables. Pairs with memory_edit: every pre-edit snapshot is stored here and can be manually restored if needed."""
        return tools.memory_history(memory_id=memory_id)

    @app.tool()
    def memory_cleanup_history(
        memory_id: Optional[int] = None,
        older_than_days: Optional[int] = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """Trim history snapshots from the memory_history table (never touches active memories). Three granularities: pass memory_id to trim one memory's history; pass older_than_days to trim snapshots older than N days; pass neither for a full trim, which requires authorized=true as a confirmation gate. Absolutely safe: regardless of arguments, it only ever runs DELETE FROM memory_history — never touches a row in the memories table."""
        return tools.memory_cleanup_history(
            memory_id=memory_id,
            older_than_days=older_than_days,
            authorized=authorized,
        )

    @app.tool()
    def memory_cleanup_inactive_vectors(
        dry_run: bool = True,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """Purge orphan vec rows and optionally resync parent_status (v0.9.4). Superseded vectors are retained for memory_search_expired; this tool only deletes true orphan vectors whose parent memory/section row no longer exists. dry_run=true reports vec_parent_status_mismatches and orphan_vectors. For non-destructive drift repair only, use memory_resync_vec_parent_status(dry_run=false), which does not require authorized. Cleanup execution may purge orphan rows, so it requires dry_run=false AND authorized=true. Only touches memories_vec / memory_sections_vec; memory content and FTS are never modified."""
        return tools.memory_cleanup_inactive_vectors(
            dry_run=dry_run,
            authorized=authorized,
        )

    # ── v0.8.0: Section split (Agent continuation/repair entry) ──
    # For normal writes use memory_write (rule-based documents auto-split; unstructured
    #  long text returns a split_request for the agent to continue). This tool is only
    #  for: internal continuation on receiving a split_request, repair of historical
    #  NULL/failed/declined records, and active-memory rebuild. Do not pre-call for
    #  ordinary writes.

    @app.tool()
    def memory_split(
        memory_id: int,
        split_decision: Optional[str] = None,
        decision_content_hash: Optional[str] = None,
        decision_memory_version: Optional[int] = None,
        decision_split_status: Optional[str] = None,
        decision_split_revision: Optional[int] = None,
        sections: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Long-content section split — internal continuation/repair entry (repositioned in v0.8.0). Two-phase CAS protocol:

          split_decision=None (prepare) — returns the full text + content_hash + snapshot
            + section schema, for the agent's own LLM to produce only section metadata
            (title/summary/anchor_text/occurrence_index/title_path). An active memory
            returns allowed_decision="rebuild"; others return allowed_decision="split".
          split_decision="split" — validates content_hash/version snapshot + anchor/offset/
            section-size (<=max_section_chars) and atomically publishes sections + vectors.
            Only allowed for memories with split_status in {NULL, failed, declined}.
          split_decision="rebuild" — rebuilds the derived index of an already-active memory;
            on failure the old index is kept.
          split_decision="decline" — explicitly give up; sets split_status="declined".

The publish stage provenance is fixed to "agent" (the rules path runs internally via memory_write/edit, not this tool). For normal writes use memory_write (documents with rule headings auto-split; unstructured long text returns a split_request for the agent to continue). This tool is only called on receipt of a split_request, to repair historical NULL/failed/declined records, or to rebuild an active memory — do not pre-call it for ordinary writes."""
        return tools.memory_split(
            memory_id=memory_id,
            split_decision=split_decision,
            decision_content_hash=decision_content_hash,
            decision_memory_version=decision_memory_version,
            decision_split_status=decision_split_status,
            decision_split_revision=decision_split_revision,
            sections=sections,
        )

    @app.tool()
    def memory_rebuild_embeddings(
        memory_ids: Optional[list[int]] = None,
        dry_run: bool = True,
        batch_size: Optional[int] = 50,
    ) -> dict[str, Any]:
        """Rebuild vectors in bulk (v0.6.0). Used to migrate the vector layer after switching embedding models, or for localized repair when the index is ready. Needs no LLM call — it only recomputes vectors. dry_run=true returns the plan without executing."""
        return tools.memory_rebuild_embeddings(
            memory_ids=memory_ids,
            dry_run=dry_run,
            batch_size=batch_size,
        )

    return app


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from .doctor_cli import run_cli
        run_cli(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        from .setup_cli import run_cli as run_setup
        raise SystemExit(run_setup(sys.argv[2:]))
    try:
        build_server().run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
