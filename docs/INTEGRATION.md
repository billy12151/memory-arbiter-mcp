# Integration Guide

## MCP Surface

Configure the command as `mema` and use the four product tools. Call `memory(action="help")` or the corresponding tool help to discover fields.

Writes should include `subject`, `source_type`, `event_time`, `workspace`, `source_ref`, and useful tags. Use `user_confirmed` only for facts explicitly verified by the user.

When a new source replaces an existing current source, find and read the existing memory, then update it. Do not create a second active copy.

## Evidence Recall

With sqlite-vec and a local embedding model configured, writes asynchronously publish local-text evidence. Search automatically embeds the query and aggregates multiple evidence hits into one memory result. Evidence hit offsets identify the relevant local text while the memory read API returns the complete source.

Use `memory_repair(task="rebuild_evidence", data={"dry_run":true})` to inspect missing or stale coverage, then queue a bounded rebuild.

## Conflict Review

Deterministic evidence routes pairs as:

- `notify`: high-confidence explicit change; create a notice even without Qwen.
- `check`: semantic similarity needs Qwen filtering; no Qwen means no notice.
- `ignore`: insufficient, equivalent, compatible, or explicitly different scope.

The Agent must read both complete memories before presenting a notice to the user. A notice does not create a formal conflict. Formal conflicts and append-only judgments are separate governance records pinned by left and right memory versions.

## Workspace Isolation

- `none`: workspace does not restrict recall.
- `weak`: workspace softly influences ranking and alias suggestions.
- `strict`: reads and conflict details require the caller workspace; new workspaces remain pending until explicitly confirmed.

## Backup And Migration

JSONL replay is previewable without authorization. Applying replay requires explicit user authorization and is idempotent by replay key and payload hash. Derived post-processing consists only of evidence indexing; successful publication automatically chains conflict candidate processing.

Use side-by-side migration for historical databases. Stop writers before `--final-sync`, verify the target, switch `db_path`, and keep the source for rollback.

## 中文摘要

集成时只使用四个产品工具。统一 evidence 向量同时承担语义召回和冲突候选召回；`notify` 不依赖 Qwen，`check` 在 Qwen 不可用时降级为忽略。notice 只是候选，Agent 必须读取两侧完整记忆后判断。历史数据库通过旁路迁移生成干净新库。
