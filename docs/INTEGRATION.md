# Integration Guide

## MCP Surface

Configure the command as `mema` and use the four product tools. Call `memory(action="help")` or the corresponding tool help to discover fields.

Writes should include `subject`, `source_type`, `event_time`, `workspace`, `source_ref`, and useful tags. Use `user_confirmed` only for facts explicitly verified by the user.

When a new source replaces an existing current source, find and read the existing memory, then update it. Do not create a second active copy.

## Evidence Recall

With sqlite-vec and a local embedding model configured, writes asynchronously publish local-text evidence. Search automatically embeds the query and aggregates multiple evidence hits into one memory result. Evidence hit offsets identify the relevant local text while the memory read API returns the complete source.

Use `memory_repair(task="rebuild_evidence", data={"dry_run":true})` to inspect missing or stale coverage, then queue a bounded rebuild.

Lexical and evidence recall are independent bounded channels. Results are aggregated by memory and merged with reciprocal-rank fusion before the existing trust, recency, filter, and workspace adjustments. This prevents a full lexical pool from suppressing semantically relevant local evidence while preserving exact lexical hits.

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

### Upgrade From An Existing Database

The current runtime expects the local-text evidence schema. An existing database is not upgraded in place because its derived claim, memory-vector, and section-vector state belongs to a different lifecycle.

1. Upgrade the package, but keep the existing MCP process and database available until migration is planned.
2. Preview the migration. The command reports source and target paths, row counts, estimated evidence units, vector storage, and free disk space:

   ```bash
   mema migrate-vnext --source old.sqlite3 --target memory.vnext.sqlite3
   ```

3. Build the side-by-side target. This copies retained records into a clean schema and rebuilds FTS and evidence vectors. The source database is read-only and remains untouched:

   ```bash
   mema migrate-vnext --source old.sqlite3 --target memory.vnext.sqlite3 --execute
   ```

4. Stop all MCP writers. Run final sync so the target is rebuilt from the last source snapshot, verified, checkpointed, and atomically replaced:

   ```bash
   mema migrate-vnext --source old.sqlite3 --target memory.vnext.sqlite3 --execute --final-sync
   ```

5. Require `ok=true`, `row_counts_match=true`, `source_stable=true`, full evidence coverage, no failed memories, and `switch_ready=true` before switching.
6. Back up the user configuration, change `db_path` to the new database, restart the MCP server, and run `mema doctor --json`.
7. Keep the old database. A direct rollback is lossless only while the new database has accepted no new writes; otherwise newer records must be exported or replayed first.

Qwen installation is not a migration prerequisite. The evidence index requires sqlite-vec and the configured embedding model; without Qwen, deterministic `notify` remains active and `check` candidates fail closed to `ignore`.

## 中文摘要

集成时只使用四个产品工具。统一 evidence 向量同时承担语义召回和冲突候选召回；`notify` 不依赖 Qwen，`check` 在 Qwen 不可用时降级为忽略。notice 只是候选，Agent 必须读取两侧完整记忆后判断。历史数据库通过旁路迁移生成干净新库。

老用户升级顺序：先执行 `mema migrate-vnext` dry-run；再用 `--execute` 旁路构建；停止所有写入后执行 `--final-sync`；确认行数、fingerprint、evidence coverage、失败列表和 `switch_ready` 全部通过；备份配置并切换 `db_path`；重启后运行 `mema doctor --json`。旧库默认保留。Qwen 不是迁移前提，没有 Qwen 时 `notify` 仍工作，`check` 降级为忽略。
