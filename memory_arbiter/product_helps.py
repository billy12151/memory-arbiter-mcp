"""产品帮助纯数据层：_PRODUCT_HELPS/onboarding/value reference（从 surfaces.py 搬出，拆分批 ⑦ 纯移动）。
surfaces.py re-export 保活（_product_help 内的「dict() 浅拷贝+动态注入 judge 字段」契约不变）。"""
from __future__ import annotations

from importlib import resources
from typing import Any

from .models import MemoryStatus, ProtectionLevel, SourceType


AGENT_ONBOARDING_TOPIC = "agent_onboarding"


def _agent_onboarding_guide() -> str:
    try:
        return str(resources.files("memory_arbiter").joinpath("AGENT_ONBOARDING.md").read_text(encoding="utf-8"))
    except Exception:
        return (
            "mema / Memory Arbiter: use MCP tools for memory operations and governance. "
            "Use memory(action='remember'|'find'|'read'|'update'|'judge'), memory_review for read-only inspection, "
            "memory_govern only for explicit user-authorized governance, and memory_repair for maintenance. "
            "Do not infer conflicts away; if a response says attention_required or action_required "
            "(e.g. read_semantic_notice, ask_user_for_authorization, confirm_new_workspace), handle it before relying on the memory."
        )


def _memory_value_reference() -> dict[str, Any]:
    return {
        "source_type": [item.value for item in SourceType],
        "protection_level": [item.value for item in ProtectionLevel],
        "memory_status_note": (
            "Lifecycle values a record can carry (seen on reads). On remember, "
            "status accepts only 'active' (the default - omit it) or 'pending'; "
            "strict isolation sets pending internally until confirm_pending_workspace "
            "activates the memory. superseded/conflicted/deleted are rejected as write inputs."
        ),
        "memory_status": [item.value for item in MemoryStatus],
        "update_modes": {
            "replace_content": "memory_id plus new_content, optionally new_subject/new_tags/add_tags/remove_tags/reason.",
            "replace_text": "memory_id plus old_text and new_text, optionally new_subject/new_tags/add_tags/remove_tags/reason.",
            "patches": "memory_id plus patches: [{old_text, new_text}, ...] (1..8, applied in order, first occurrence, atomic all-or-nothing), optionally new_subject/new_tags/add_tags/remove_tags/reason.",
            "tags_only": "memory_id plus tags_only=true with add_tags and/or remove_tags; content is unchanged.",
        },
    }


# One shared help-document instance (#9): the literal's only dynamic input is
# the (now module-level) _memory_value_reference, so it is built once here
# instead of on every _product_help call. Every mutation path in
# _product_help copies via dict() before adding keys, so callers can never
# mutate this shared instance.
_PRODUCT_HELPS: dict[str, Any] = {
    "memory": {
        "description": "Daily memory operations: remember, find, read, update, judge, status. workspace is required on remember — discover buckets once with memory_review(view='workspaces') (session start) and reuse; re-query only when unsure or before creating a new bucket ('default' is the global pool).",
        "actions": ["remember", "find", "batch_find", "read", "batch_read", "update", "judge", "status", "help"],
        "examples": {
            "remember": {"action": "remember", "data": {"workspace": "memory-arbiter-mcp", "content": "Fact to remember", "subject": "Short subject", "tags": ["project"]}},
            "find": {"action": "find", "data": {"query": "project decision", "limit": 5}},
            "batch_find": {"action": "batch_find", "data": {"queries": [{"id": "collections", "query": "催收 辱骂 侮辱"}, {"id": "debt-transfer", "query": "债务转移 债权人同意"}], "limit_per_query": 3}},
            "read": {"action": "read", "data": {"memory_id": 123}},
            "batch_read": {"action": "batch_read", "data": {"memory_ids": [12, 34], "content_mode": "full"}},
            "update": {"action": "update", "data": {"memory_id": 123, "new_content": "Updated current fact", "reason": "User provided a newer source-of-truth."}},
            "update_patches": {"action": "update", "data": {"memory_id": 123, "patches": [{"old_text": "MySQL 5.7", "new_text": "MySQL 8.0"}, {"old_text": "us-east-1", "new_text": "us-west-2"}], "reason": "Two spotted corrections in one edit."}},
            "judge": {"action": "judge", "data": {"conflict_id": 1, "expected_revision": 1, "chosen_value": "SQLite", "decided_by": "user", "ref": "chat", "reason": "User confirmed the current database.", "apply_plan": [{"memory_id": 12, "action": "update_current_claim"}, {"memory_id": 34, "action": "use_as_resolution"}], "resolution_memory_id": 34}},
        },
        "source_of_truth_rule": "When a user says a new document replaces the current source of truth, find/read the existing current memory and update it; do not create a second active memory or retire the old one unless the user explicitly asks for whole-memory retirement.",
        "tag_discipline": (
            "Tags are a RETRIEVAL dimension, not an event log (0.16.0): one memory keeps at "
            "most 32 tags in total. remember with >32 tags is refused; update/add_tags that "
            "would push the persisted total over 32 is refused as a whole (error carries the "
            "current total and a remove_tags-first hint; remove+add in ONE call is legal). "
            "tags_filter and other query inputs stay capped per call (100), not per memory. "
            "One-off status or timestamps belong in metadata, never in tags. Pre-0.16.0 rows "
            "over the cap keep working (no retro truncation) — doctor tags.over_limit lists "
            "them for a manual trim."
        ),
        "write_duplicate_hint": (
            "Since 0.16.6 a DB-level dedup gate comes FIRST: byte-identical "
            "content in the same workspace returns an idempotent success "
            "(data.duplicate_replay=true with the existing id; nothing is "
            "written) and never reaches this hint — only ACTIVE rows hold a "
            "slot, so rewriting a retired memory's exact content lands "
            "normally. The notice below is therefore near-duplicate territory. "
            "remember/activation responses may carry a similar_active_memory notice when the "
            "new subject/tags closely match an existing active memory (subject ratio >=0.95 "
            "AND tag Jaccard >=0.8; empty tag sets on both sides count as Jaccard 1.0, so the "
            "subject alone decides; subjects differing only in digit runs are treated as "
            "series entries and stay quiet). Candidates are recalled by subject+tags vector "
            "KNN (top-k per workspace) when an embedding model is configured, with a capped "
            "same-workspace scan as fallback; ranking itself stays deterministic. "
            "Triage it silently: ignore deliberate series entries, prefer updating the "
            "original on true duplicates, and ask the user only when retiring/merging "
            "(governance) is needed. Pending->active transitions (confirm/activate) are "
            "re-checked with the same notice."
        ),
        "find_size_metering": (
            "find is an index page: by default (content_mode=\"preview\") each result "
            "carries metadata plus content_chars (full-text length, i.e. the read "
            "cost) and a bounded outline (<=8 segments of {head, offset}; "
            "heading/text units from the evidence pipeline, so outline.offset "
            "shares read's span coordinate system and span=[offset, offset+N] "
            "slices that exact segment). v0.15.10 replaces the removed "
            "include_content boolean with a single-choice content_mode enum: "
            "\"preview\" (default, no content) | \"hits\" (adds hit_spans — the "
            "vector-matched unit text with start_offset/end_offset in read's "
            "span coordinates; NO truncation ever: when merged hits cover >=50% "
            "of the content the item upgrades to full text with hit_spans kept "
            "as an annotation, so the server never picks 'the important hits' "
            "for you; items without vector hits keep the plain preview shape; "
            "hit_window=N (default 0) extends each hit with +/-N neighbouring "
            "complete sentences, neighbours marked matched=false; hit_spans "
            "appears only on query-recall pages — browse/filter pages carry "
            "none; hits whose evidence row lags the memory's current version "
            "are dropped with a stale_hit_spans marker plus a re-query "
            "warning) | "
            "\"full\" (adds the whole content — the old include_content=true). "
            "Score is only "
            "meaningful relative to other items on the same page. If the top page "
            "misses, reword the query or add tags_filter instead of deep paging: "
            "unfiltered query-recall reports total_estimate=null/has_more=false. "
            "The size block (returned_chars/returned_count/tokens_estimate) meters "
            "the page as actually returned. Since v0.15.6 every recall surface "
            "carries the same size block under one global config key include_size "
            "(default true): find and batch_find (the page as returned), read (the record as "
            "returned — a span read meters the window, so the number is the true "
            "cost of that call), memory_review expired and history (their result "
            "lists, full texts included). Each size block also carries a "
            "display_hint with the token number and a report-this-recall-cost "
            "instruction — surface it when citing the recalled records. "
            "include_size=false in config.json turns all of them off "
            "together; find's old per-call include_size parameter "
            "is ignored (warning only). "
            "tokens_estimate uses a deterministic bucket estimator "
            "(heuristic_v1, calibrated against a Qwen2.5 tokenizer at design time — no Qwen runtime ships since 0.17.1): pure Chinese prose runs ~30% high and pure English "
            "~17% high, and emoji/ZWJ sequences run systematically low (byte-level BPE means a "
            "single emoji is >=1 token) — the estimate and the estimated share one yardstick, so "
            "savings comparisons stay valid. "
            "unresolved_conflict_count appears only when page items directly hit an "
            "open/applying conflict group, and counts those page items."
        ),
        "batch_find_semantics": (
            "batch_find runs up to 8 queries in one call and returns one merged "
            "index page. Shared filters (workspace/tags_filter/source_type/"
            "after_time/before_time) and the caller scope apply to every query. "
            "id is optional and defaults to the query text; ids and queries must "
            "be unique inside a batch (fail-fast: malformed batches are rejected "
            "whole — there is no partial-success mode). limit_per_query (default "
            "3, max 20) slices each query's page BEFORE merging. "
            "deduplicate=true (default) merges by memory_id across queries: each "
            "item carries matched_query_ids (every hitting query) and "
            "best_query_id; the page is ordered by best final score, then first "
            "hit, then memory_id. per_query reports {id, count, has_more, "
            "retrieval_mode} per query — a query that recalls nothing reports "
            "count=0/empty; batch never falls back to recent memories and every "
            "item has already passed the relevance floor. content_mode "
            "(v0.15.10, same enum as find: preview default | hits | full) picks "
            "the per-item content depth; hits supports hit_window=N (default 0, "
            "±N neighbouring complete sentences around each hit, neighbours "
            "marked matched=false; hit_spans appears only on query-recall "
            "pages; stale-version hits are dropped with a re-query warning); "
            "prefer the preview and read specific spans."
        ),
        "value_reference": _memory_value_reference(),
    },
    "memory_review": {
        "description": "Read-only inspection. Never changes memory state. Start sessions with view='workspaces' to list bucket names (the required workspace input for remember and workspace-govern actions).",
        "views": ["overview", "doctor", "conflicts", "conflict_detail", "history", "expired", "audit", "entities", "workspaces", "help"],
        "examples": {
            "conflicts": {"view": "conflicts", "data": {"status": "open", "limit": 20}},
            "history": {"view": "history", "data": {"memory_id": 123}},
            "expired": {"view": "expired", "data": {"query": "old decision", "limit": 10}},
        },
    },
    "memory_govern": {
        "description": "Explicit user-authorized governance. Every state-changing action requires authorized=true after the user confirms that specific action. Workspace-mutating actions (confirm_pending_workspace, rename_workspace_canonical, migrate_workspace, separate_workspace_alias, move_memories_workspace) also require workspace — call memory_review(view='workspaces') first to list existing buckets. Do not use for ordinary source-of-truth updates; use memory(action='update') instead.",
        "actions": ["retire", "merge_memories", "apply_conflict_action", "replan_conflict", "resolve_conflict", "confirm", "rename_workspace_canonical", "migrate_workspace", "move_memories_workspace", "rollback_auto_move", "separate_workspace_alias", "confirm_pending_workspace", "confirm_workspaces", "help"],
        "examples": {
            "retire": {"action": "retire", "data": {"memory_id": 123, "superseded_by": 456, "reason": "User explicitly requested retiring the old whole memory.", "authorized": True}},
            "merge_memories": {"action": "merge_memories", "data": {"survivor_id": 456, "loser_ids": [123, 124], "reason": "Same fact recorded twice; keeping the newer record.", "authorized": True}},
            "merge_memories_with_content": {"action": "merge_memories", "data": {"survivor_id": 456, "loser_ids": [123], "merged_content": "Combined statement retaining every unique detail from both records.", "reason": "User confirmed the combined wording.", "authorized": True}},
            "separate_workspace_alias": {"action": "separate_workspace_alias", "data": {"alias": "旧项目名", "canonical": "新项目名", "workspace": "新项目名", "reason": "User confirmed the two workspaces must stay separate.", "authorized": True}},
            "apply_conflict_action": {"action": "apply_conflict_action", "data": {"conflict_id": 1, "expected_revision": 2, "memory_id": 12, "action": "update_current_claim", "content": "The database is SQLite.", "reason": "Apply the confirmed conflict decision.", "authorized": True}},
            "resolve_conflict": {"action": "resolve_conflict", "data": {"conflict_id": 1, "expected_revision": 4, "reason": "All planned member actions completed.", "authorized": True}},
            "rename_workspace_canonical": {"action": "rename_workspace_canonical", "data": {"old": "旧项目名", "new": "新项目名", "workspace": "旧项目名", "reason": "User confirmed the rename.", "authorized": True}},
            "migrate_workspace": {"action": "migrate_workspace", "data": {"from": "金营二期", "to": "金营项目", "workspace": "金营二期", "reason": "User confirmed the merge.", "authorized": True}},
            "move_memories_workspace": {"action": "move_memories_workspace", "data": {"memory_ids": [123, 124], "new_workspace": "金营项目", "workspace": "金营项目", "reason": "User confirmed these memories belong to the project bucket.", "authorized": True}},
            "rollback_auto_move": {"action": "rollback_auto_move", "data": {"audit_id": 1, "reason": "User says the auto-move was wrong.", "authorized": True}},
            "confirm_pending_workspace": {"action": "confirm_pending_workspace", "data": {"memory_id": 123, "canonical": "金营项目", "workspace": "金营项目", "authorized": True}},
            "confirm_workspaces": {"action": "confirm_workspaces", "data": {"reason": "Reviewed the registry after renaming duplicates; snapshots the current registry.", "authorized": True}},
        },
        "safety_note": "Set authorized=true only after the user explicitly confirms the specific governance action. Retire only whole memories; for partial updates or current-document replacement, update the existing memory instead.",
        "workspace_move_vs_migrate": (
            "migrate_workspace merges one whole canonical workspace into another by "
            "name and reroutes the old name through an alias; move_memories_workspace "
            "moves selected memories by id to another workspace bucket (both workspace "
            "columns) and leaves alias/normalization rules untouched. Moving does not "
            "change memory status: pending memories stay pending until activated via "
            "confirm_pending_workspace, and superseded/deleted rows keep their status "
            "(reported via moved_non_active)."
        ),
        "authorization_rule": "All state-changing actions require authorized=true. Without it, the response returns action_required=ask_user_for_authorization and an impact description.",
        "confirm_actions": {
            "confirm": "Promote one memory to user_confirmed and lock it against ordinary changes.",
            "confirm_pending_workspace": (
                "Confirm a new canonical workspace under strict isolation and activate its pending memory."
            ),
            "confirm_workspaces": (
                "Record the reviewed workspace snapshot after rename/merge cleanup. "
                "Omit workspaces to snapshot the current registry and clear workspace.review; "
                "an explicit subset confirms only those names, so other current names remain warnings."
            ),
        },
    },
    "memory_repair": {
        "description": "Maintenance and repair operations. Prefer dry_run first; cleanup, activation, and protected-memory metadata changes still require authorized=true when the underlying operation requires it. Table segments >100 rows are exempt from row vectors and pair detection (visible as table_rows_exempted in kick receipts); after an upgrade the first server start auto-backfills row vectors (watch doctor rows.coverage).",
        "tasks": ["rebuild_evidence", "scan_pipeline", "scan_queue", "scan_candidates", "scan_duplicates", "scan_workspace_anomalies", "cleanup_history", "set_entity", "activate_pending", "replay_backup", "normalize_workspaces", "semantic_control", "notice", "record_conflict", "help"],
        "examples": {
            "rebuild_evidence": {"task": "rebuild_evidence", "data": {"dry_run": True, "memory_ids": [123]}},
            "set_entity": {"task": "set_entity", "data": {"memory_id": 123, "entity": "project-x", "scope": "charter"}},
            "semantic_control": {"task": "semantic_control", "data": {"action": "status"}},
            "replay_backup": {"task": "replay_backup", "data": {"dry_run": True}},
            "scan_workspace_anomalies": {"task": "scan_workspace_anomalies", "data": {}},
            "normalize_workspaces": {"task": "normalize_workspaces", "data": {"dry_run": True}},
            "normalize_workspaces_apply": {"task": "normalize_workspaces", "data": {"dry_run": False, "authorized": True}},
            "record_conflict": {"task": "record_conflict", "data": {"slot_key": {"entity": "project-x", "attribute": "database", "scope": "production"}, "members": [{"memory_id": 12, "version": 1, "attribute_raw": "database", "value_raw": "MySQL", "normalized_attribute": "database", "normalized_value": "mysql", "evidence_quote": "database is MySQL", "evidence_span": [0, 17], "content_hash": "0000000000000000000000000000000000000000000000000000000000000000", "direction": "a_to_b", "prompt_version": "p1", "detector_version": "d1"}, {"memory_id": 34, "version": 1, "attribute_raw": "database", "value_raw": "SQLite", "normalized_attribute": "database", "normalized_value": "sqlite", "evidence_quote": "database is SQLite", "evidence_span": [0, 18], "content_hash": "1111111111111111111111111111111111111111111111111111111111111111", "direction": "b_to_a", "prompt_version": "p1", "detector_version": "d1"}], "value_groups": [{"normalized_value": "mysql", "display_value": "MySQL", "members": ["12@1"]}, {"normalized_value": "sqlite", "display_value": "SQLite", "members": ["34@1"]}], "status": "open", "detector_version": "d1", "prompt_version": "p1", "source": "scheduled_scan", "reason": "Reviewed conflicting values."}},
            "scan_pipeline": {"task": "scan_pipeline", "data": {"action": "kick"}},
            "scan_queue": {"task": "scan_queue", "data": {"action": "page"}},
            "scan_candidates": {"task": "scan_candidates", "data": {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_check": False}},
            "scan_candidates_quotes": {"task": "scan_candidates", "data": {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True}},
            "scan_candidates_duplicates": {"task": "scan_candidates", "data": {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_duplicates": True}},
            "scan_duplicates": {"task": "scan_duplicates", "data": {"include_quotes": True}},
            "notice": {"task": "notice", "data": {"action": "list", "status": "open", "limit": 5}},
            "notice_read": {"task": "notice", "data": {"action": "read", "notice_id": 1}},
            "notice_dismiss": {"task": "notice", "data": {"action": "dismiss", "notice_id": 1, "reason": "Reviewed; not actionable."}},
            "notice_resolve": {"task": "notice", "data": {"action": "resolve", "notice_id": 1, "reason": "Reviewed and handled."}},
            "notice_escalate": {"task": "notice", "data": {"action": "escalate", "notice_id": 1, "reason": "Verified against both memories: real contradiction needing governance."}},
        },
        "semantic_notice_delivery": "Notices progress pending -> delivered while open, then dismissed/resolved, or stale when any frozen member is no longer active at its pinned version. Read requires freshness.fresh=true and executing every read_calls entry for complete memories before triage; two-member notices also expose optional left/right aliases. Dismiss a false positive, resolve a handled one, or escalate a verified contradiction into a formal conflict. Escalate only files the case — it never edits memory content: escalate, then memory(action='judge') with apply_plan to land the correction and close the group (the judge records who decided what for audit); if a member was edited directly after escalating, judge reports stale_member — append the new version via record_conflict or resolve instead.",
        "checked_no_notice": "A completed semantic task with outcome=checked_no_notice examined its eligible candidates and emitted zero notices; it is not a claim that no conflict can exist outside that task snapshot or candidate budget.",
        "scan_duplicates": (
            "Full-library near-duplicate sweep in ONE bounded response: aggregates the "
            "same duplicates detection as scan_candidates across all pages under a "
            "global pair cap (200). Default entries are lightweight (left/right ids, "
            "subjects, workspace, reason, distance, candidate_key_hash); "
            "include_quotes=true adds the triggering evidence quotes. Same suppression "
            "contract as scan_candidates (recorded not_a_conflict/open/applying pairs "
            "are not re-listed). It does not advance conflict-scan progress or write "
            "scan_log. Use this for duplicate triage; scan_candidates with "
            "include_duplicates stays a single-page spot check (it returns full "
            "record_conflict-compatible members for one page)."
        ),
        "normalize_workspaces_scope": (
            "normalize_workspaces is a GLOBAL registry operation: it takes no workspace "
            "filter and folds spelling-variant canonicals across the whole registry. "
            "Under strict isolation it still requires a resolvable caller workspace "
            "(settings.workspace), the same ACL gate as scan_candidates/record_conflict."
        ),
        "semantic_control_actions": [
            "status", "pause", "resume", "enable", "unload", "disable",
        ],
        "semantic_control_note": (
            "status reports the mDeBERTa judge backend (0.17.1+: ckpt / "
            "ckpt_sha8 / model_dir / model_version / device / batch_size / "
            "max_len, breaker state via last_error / restarts / timed_out / "
            "disabled) and the effective notice_sync_wait_ms (config key "
            "semantic_conflict.notice_sync_wait_ms, default 3000, clamp "
            "0-5000; 0 = the write response never waits for the post-commit "
            "check, for batch ingestion). The legacy Qwen backend fields "
            "(n_ctx / prompt_version / pair-v*) were removed with the engine."
        ),
    },
}
