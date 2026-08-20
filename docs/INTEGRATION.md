# Integration Guide

## MCP Surface

Configure the command as `mema` and use the four product tools. Call `memory(action="help")` or the corresponding tool help to discover fields.

Writes should include `subject`, `source_type`, `event_time`, `workspace`, `source_ref`, and useful tags. Use `user_confirmed` only for facts explicitly verified by the user.

When a new source replaces an existing current source, find and read the existing memory, then update it. Do not create a second active copy.

## Evidence Recall

With sqlite-vec and a local embedding model configured, writes asynchronously publish local-text evidence. Search automatically embeds the query and aggregates multiple evidence hits into one memory result. Evidence hit offsets identify the relevant local text while the memory read API returns the complete source.

Use `memory_repair(task="rebuild_evidence", data={"dry_run":true})` to inspect missing or stale coverage, then queue a bounded rebuild. After an embedding-model change the vector index reports `state=mismatch` and the evidence channel is disabled; running `rebuild_evidence` (non-strict workspace, repeat per batch until the pending set is empty) republishes every memory in the new space and flips the index back to `ready` automatically. Memories with no indexable text at all (blank or whitespace-only subject and content, e.g. legacy imports) are excluded from rebuild pending sets and coverage counts by design; `memory_status` reports them under `local_text_evidence.non_indexable_memories`. Evidence KNN applies workspace/exclude filters after the vector search with a bounded over-fetch (up to 4x, capped at 2048 rows; the factor shrinks for k>512), so very small workspaces in very large shared libraries can still see reduced evidence-channel recall. Semantic status (`memory_repair(task="semantic_control", data={"action":"status"})`) surfaces `check_degradation.last_reason` (`qwen_unavailable` / `qwen_backend_error` / `qwen_timeout` / `qwen_invalid_output` / `qwen_budget_exhausted`) and `worker.dropped_queue_full` for overflow drops.

Lexical and evidence recall are independent bounded channels. Results are aggregated by memory and merged with reciprocal-rank fusion before the existing trust, recency, filter, and workspace adjustments. This prevents a full lexical pool from suppressing semantically relevant local evidence while preserving exact lexical hits.

## Conflict Review

Deterministic evidence routes pairs as:

- `notify`: high-confidence explicit change; create a notice even without Qwen.
- `check`: semantic similarity needs Qwen filtering; no Qwen means no notice.
- `ignore`: insufficient, equivalent, compatible, or explicitly different scope.

The Agent must read both complete memories before presenting a notice to the user. A notice does not create a formal conflict by itself. Formal conflicts enter the conflicts table through two product paths: `memory_repair(task="notice", data={"action":"escalate", ...})` turns a verified notice into a conflict row (and resolves the notice, backfilling its `conflict_id`), and `memory_repair(task="record_conflict", ...)` registers findings from a scheduled external scan loop (idempotent per open pair; `refresh=true` rewrites enrichment fields; the row pins both memory versions as CAS snapshots). Formal conflicts and append-only judgments are separate governance records pinned by left and right memory versions.

## Workspace Isolation

- `none`: workspace does not restrict recall.
- `weak`: workspace softly influences ranking and alias suggestions.
- `strict`: reads and conflict details require the caller workspace; new workspaces remain pending until explicitly confirmed.

## Backup And Migration

JSONL replay is previewable without authorization. Applying replay requires explicit user authorization and is idempotent by replay key and payload hash. Derived post-processing consists only of evidence indexing; successful publication automatically chains conflict candidate processing.

Use side-by-side migration for historical databases. Stop writers before `--final-sync`, verify the target, switch `db_path`, and keep the source for rollback.

### Upgrade From An Existing Database

The current runtime expects the local-text evidence schema. An existing database is not upgraded in place because its derived claim, memory-vector, and section-vector state belongs to a different lifecycle.

1. Upgrade the package. If the configured database is legacy, MCP refuses to start, does not modify it, and instructs the user to run `mema upgrade`.
2. Preview the migration. The command reports source and target paths, row counts, estimated evidence units, vector storage, and free disk space:

   ```bash
   mema upgrade --dry-run
   ```

3. Stop all mema MCP clients. Run the upgrade and confirm the displayed plan. Migration usually takes 1–5 minutes and no MCP service is provided during it:

   ```bash
   mema upgrade
   ```

4. `mema upgrade` performs the final snapshot build and requires `ok=true`, `row_counts_match=true`, `source_stable=true`, full evidence coverage, no failed memories, and `switch_ready=true`.
5. After verification, it backs up and atomically updates the selected standard JSON config. If `MEMORY_ARBITER_DB_PATH` overrides the path, or the JSON config does not point at the migrated source, the command does not edit configuration and prints the manual action instead.
6. Restart the MCP client and run `mema doctor --json`.
7. Keep the old database. A direct rollback is lossless only while the new database has accepted no new writes; otherwise newer records must be exported or replayed first.

Options: `--no-switch` builds and verifies without changing configuration; `--yes` accepts the plan non-interactively; `--source` and `--target` override paths. The lower-level `mema migrate-vnext` command remains available for diagnostics and advanced workflows.

Qwen installation is not a migration prerequisite. The upgrade preflight does require `llama-cpp-python` (the `semantic-local` extra) because the evidence embedding model runs on it, and the evidence index additionally requires sqlite-vec and the configured embedding model; without Qwen, deterministic `notify` remains active and `check` candidates fail closed to `ignore`.

## 中文摘要

集成时只使用四个产品工具。统一 evidence 向量同时承担语义召回和冲突候选召回；`notify` 不依赖 Qwen，`check` 在 Qwen 不可用时降级为忽略。notice 只是候选，Agent 必须读取两侧完整记忆后判断。历史数据库通过旁路迁移生成干净新库。

老用户升级顺序：升级包后，旧库会让 MCP 拒绝启动且不会被修改；先执行 `mema upgrade --dry-run` 查看预估；关闭所有 mema MCP 客户端后运行 `mema upgrade`，迁移期间 1–5 分钟不提供 MCP；命令验证行数、fingerprint、evidence coverage、失败列表和 `switch_ready`，成功后备份并原子切换标准 JSON 配置；环境变量覆盖路径时只输出手工操作；重启后运行 `mema doctor --json`。旧库默认保留。Qwen 不是迁移前提，没有 Qwen 时 `notify` 仍工作，`check` 降级为忽略。
