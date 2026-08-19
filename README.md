# Memory Arbiter MCP

Memory Arbiter is a local SQLite memory service for AI agents. It combines lexical search, local-text evidence vectors, explicit history, workspace isolation, advisory conflict notices, and user-authorized governance.

## Architecture

Every memory keeps its complete source text. A background worker builds one derived evidence index from:

- `subject`
- Markdown headings
- local sentence or paragraph groups
- overlapping windows for long text without useful boundaries

The indexer does not extract facts, infer entities, or call an Agent. The same evidence vectors power semantic recall and conflict candidate retrieval, so there is no second memory-level or section-level vector index.

Conflict candidates follow one route:

1. Evidence KNN finds semantically close local text.
2. Narrow deterministic evidence classifies the pair as `notify`, `check`, or `ignore`.
3. `notify` creates an advisory notice without Qwen veto.
4. `check` calls the optional local Qwen model on only the short pair. If Qwen is unavailable or rejects it, the pair is ignored.
5. The Agent reads both full memories and decides whether to surface, dismiss, resolve, or explicitly record a formal conflict.

Qwen never edits memory, confirms a conflict, or supersedes a record. Formal conflict judgments use only the two memory versions as CAS pins.

## Architecture Upgrade

This release moves recall and conflict discovery onto one local-text evidence architecture:

- Lexical and evidence channels now recall independently, merge by memory with reciprocal-rank fusion, and keep bounded representation from both channels. On a 17-query long-document benchmark, product recall improved from `R@10 52.9% / R@20 52.9%` to `R@10 94.1% / R@20 100%`; average query time increased from about 84ms to 109ms.
- Conflict discovery now uses evidence KNN followed by deterministic `notify/check/ignore`. On a balanced 24-pair benchmark, recall improved from 8.3% to 75% and F1 from 15.4% to 81.8%.
- Qwen is optional. Deterministic `notify` works without it; Qwen only filters or recovers `check` candidates. Without Qwen, writes, recall, notices from strong evidence, and governance continue normally.
- The derived schema is now one `memory_evidence` + `memory_evidence_vec` index. It replaces structured-claim derivation and separate memory/section vector indexes. Notice and formal-conflict freshness use memory version pins.

Existing databases must be rebuilt side by side; do not point the new runtime directly at an old database:

```bash
# Preview only.
mema upgrade --dry-run

# Stop every mema MCP client, then migrate and switch the standard JSON config.
mema upgrade

# Restart the MCP client and verify.
mema doctor --json
```

The command displays memory count, estimated evidence volume, disk requirement, source, and target before asking for confirmation. Migration usually takes 1–5 minutes and MCP remains unavailable during that window. The old database is never deleted. Standard JSON configuration is backed up and switched only after full verification; environment-variable `db_path` overrides are reported as a manual action. Use `--no-switch` to build and verify without editing configuration, or `--yes` for non-interactive execution. Keep the old database until the new one has run successfully in normal use. If the new database has accepted writes, do not switch back without first accounting for those newer records.

## Install

```bash
pip install memory-arbiter-mcp
pip install "memory-arbiter-mcp[vec]"            # sqlite-vec
pip install "memory-arbiter-mcp[semantic-local]" # local GGUF runtime
```

Run the MCP server:

```bash
mema
```

Available subcommands:

```bash
mema doctor --json
mema setup
mema console
mema upgrade
mema migrate-vnext --source old.sqlite3 --target memory.vnext.sqlite3 --execute
```

## Product Tools

Memory Arbiter exposes four MCP tools:

- `memory`: `remember`, `find`, `read`, `update`, `judge`, `status`, `help`
- `memory_review`: read-only health, conflicts, judgments, history, expired memory, audit, and entities
- `memory_govern`: explicitly authorized retirement, confirmation, conflict resolution, judgment correction, and workspace governance
- `memory_repair`: evidence rebuild, history cleanup, pending activation, backup replay, semantic runtime control, and notice lifecycle

State-changing governance requires `authorized=true` after the user confirms that specific action.

## Configuration

Configuration discovery order:

1. `MEMORY_ARBITER_CONFIG`
2. `~/.config/memory-arbiter/config.json`
3. environment variables and defaults

See [`examples/memory-arbiter.config.example.json`](examples/memory-arbiter.config.example.json).

Important settings:

| Setting | Purpose |
| --- | --- |
| `db_path` | Current SQLite database |
| `backup_jsonl` | Append-only fallback when SQLite cannot write |
| `vec.enabled` | Enables sqlite-vec evidence recall |
| `embedding.model_path` | Local GGUF embedding model |
| `embedding.max_unit_chars` | Safety bound for one evidence embedding call |
| `isolation` | `none`, `weak`, or `strict` workspace behavior |
| `semantic_conflict.model_path` | Optional local Qwen GGUF model for `check` pairs |
| `semantic_conflict.job_timeout_ms` | Between-pair job budget |
| `semantic_conflict.inference_timeout_ms` | Hard timeout for one Qwen call |

## Notices

A successful product response may include one compact semantic notice stub. Read it with:

```json
{"task":"notice","data":{"action":"read","notice_id":1}}
```

Then execute the returned left and right read calls. The notice is advisory, not a confirmed conflict. Dismiss false positives and resolve notices that have been handled. Editing either memory invalidates an undelivered notice through its version pins.

## Migration

Historical databases are migrated side by side. The migration command creates a clean current schema, copies only retained tables and shared columns, rebuilds FTS and evidence vectors, verifies row counts and memory fingerprints, and leaves the source database untouched.

Use `--final-sync` only after stopping writers. Keep the source database until the new database is verified in normal use.

## Degradation

- Without sqlite-vec or an embedding model, lexical recall and memory governance continue; evidence indexing is unavailable.
- Without Qwen, deterministic `notify` still surfaces and `check` becomes `ignore`.
- If SQLite is unavailable or unwritable, writes use the append-only JSONL envelope only when that write succeeds.

## Development

```bash
uv run pytest -q
python scripts/sync_version.py --check
```

## 中文说明

Memory Arbiter 是面向 AI Agent 的本地 SQLite 记忆服务。完整原文只存一份，后台从标题、Markdown heading 和局部语句生成统一 evidence 向量；语义召回和冲突候选都使用这一套索引。

冲突候选只有一条主线：evidence KNN -> `notify/check/ignore` -> `check` 才调用本地 Qwen -> Agent 阅读两侧完整记忆后终审。Qwen 不可用时，`notify` 继续提醒，`check` 降级为忽略，`ignore` 保持忽略。

历史数据库使用 `mema migrate-vnext` 旁路生成干净新库，不原地修改。治理操作只有在用户明确确认本次动作后才能传 `authorized=true`。

本次架构升级将词法召回与局部 evidence 向量独立召回后按 memory 融合。17 条真实长文查询中，产品搜索从 `R@10 52.9% / R@20 52.9%` 提升到 `R@10 94.1% / R@20 100%`，平均查询约从 84ms 增至 109ms。冲突发现改为 evidence KNN + `notify/check/ignore`，24 对平衡样本中 recall 从 8.3% 提升到 75%，F1 从 15.4% 提升到 81.8%。

Qwen 不是必需组件：`notify` 由确定性 evidence 独立触发，Qwen 只处理 `check` 灰区；没有 Qwen 时，写入、搜索、强证据 notice 和治理仍可正常工作。数据结构统一为 `memory_evidence` + `memory_evidence_vec`，不再维护 claims、memory vector 和 section vector 三套派生生命周期。

老用户升级时先运行 `mema upgrade --dry-run` 查看记忆数量、evidence 规模和磁盘需求；在方便中断服务 1–5 分钟时关闭所有 mema MCP 客户端，再运行 `mema upgrade`。命令会旁路构建并完整验证新库，成功后备份标准 JSON 配置并切换 `db_path`；环境变量覆盖路径时只输出手工修改指引。重启后运行 `mema doctor --json`。旧库始终保留；新库产生新写入后，不得直接切回旧库而忽略增量数据。
