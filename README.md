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
