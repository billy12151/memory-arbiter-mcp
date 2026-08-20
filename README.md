# Memory Arbiter MCP

Memory Arbiter is a trustworthy local fact layer for AI agents — not just shared memory, but shared facts that are current, trusted, traceable, and safe to use. It is a local SQLite service exposed over MCP: four product tools, evidence-based recall, advisory conflict notices, and user-authorized governance. Every fact is stored once in local SQLite and every model it can call runs locally.

## Why trust it

- **One complete source of truth.** Every memory keeps its full original text. Evidence vectors, full-text search, and rankings are all *derived* indexes — rebuildable, never the only copy.
- **Provenance on every write.** Each memory carries `source_type`, `source_ref`, `event_time`, and `ingest_time`, so you can tell a user-confirmed fact from an agent's guess.
- **Trust levels.** `user_confirmed` promotion and `normal`/`protected`/`locked` protection levels prevent an agent from silently overwriting facts you have verified.
- **Full version history.** Every edit appends to `memory_history` with a version bump, and supersede chains keep old facts traceable instead of silently replaced.
- **Advisory vs. formal conflicts.** Candidate conflicts surface as advisory notices the agent must read before acting; only an explicit user decision creates or resolves a formal conflict.
- **Authorized governance.** Every state-changing `memory_govern` action requires per-action `authorized=true` after the user confirms that specific action.
- **Local-only.** Embeddings run on a local GGUF model; the optional Qwen filter is a local GGUF too. The single outbound call is an optional PyPI update check, disabled with `update_check.enabled=false`.

## Install & quickstart

```bash
pip install memory-arbiter-mcp
pip install "memory-arbiter-mcp[vec]"            # sqlite-vec evidence recall
pip install "memory-arbiter-mcp[semantic-local]" # local GGUF runtime (embeddings + Qwen)
```

Run `mema setup` to write `~/.config/memory-arbiter/config.json` and self-check the environment (it never installs or downloads anything). Then wire your MCP client from `examples/*.mcp.json` and start the server with `mema`.

The daily loop is four calls — `remember` a reusable fact, `find` to recall, `read` for exact lookup, `update` when a newer source replaces an existing current memory (never create a second active copy of one source of truth). Point any agent at the packaged rule:

```json
{"action":"help","data":{"topic":"agent_onboarding"}}
```

## The four tools

- `memory`: `remember`, `find`, `read`, `update`, `judge`, `status`, `help`
- `memory_review`: read-only health, conflicts, judgments, history, expired memory, audit, and entities
- `memory_govern`: explicitly authorized retirement, confirmation, conflict resolution, judgment correction, and workspace governance
- `memory_repair`: evidence rebuild, history cleanup, entity assignment, pending activation, backup replay, semantic runtime control, and notice lifecycle

A successful product response may carry a top-level `notices` array and an `action_required` field telling the agent the next expected step.

## How recall works

Lexical and evidence channels recall independently and merge per memory with reciprocal-rank fusion, then trust, recency, filter, and workspace adjustments.

- **Lexical**: FTS5 over content plus subject/tags LIKE and a bounded content-LIKE anchor channel.
- **Evidence**: a background worker derives local-text evidence units from the `subject`, Markdown headings, sentence/paragraph groups, and overlapping windows for long text. The indexer never extracts facts, infers entities, or calls a model — it only slices the stored source. Evidence hits carry offsets into the original text; `read` returns the complete source.

## Conflicts: notices vs. formal conflicts

One route only: evidence KNN finds semantically close local text, a narrow deterministic rule classifies the pair as `notify`, `check`, or `ignore`.

- `notify` creates an advisory notice without any model veto.
- `check` sends only the short pair to the optional local Qwen model; without Qwen it degrades to `ignore`.
- The agent reads both complete memories and decides whether to surface, dismiss, or resolve the notice.

A notice has no formal `conflict_id`: never pass it to `judge` or `resolve_conflict`. Formal conflicts are judged with the two memory versions as CAS pins. Qwen never edits memory, confirms a conflict, or supersedes a record.

## Workspaces

`isolation` is `none`, `weak`, or `strict`. Under `strict`, a write that resolves to a new workspace stays `pending` until `memory_govern(confirm_pending_workspace)` activates it; all reads and governance are scoped to the caller's canonical workspace. Alias governance (`accept`/`reject`/`rename`/`migrate`) keeps workspace naming consistent across clients.

## Operating mema

- `mema doctor [--json|--deep]` — read-only health checks; `--deep` loads the GGUF model and probes the live embedding dimension.
- `mema console` — read-only local console on 127.0.0.1.
- `memory(action="status")` — surfaces `local_text_evidence` coverage, `vec_index_state`, the index worker, and `semantic_conflict` runtime including `check_degradation.last_reason`.
- Maintenance tasks on `memory_repair`: `rebuild_evidence` (dry-run then batched execute; after an embedding-model change the index reports `state=mismatch` and rebuild flips it back to `ready` automatically), `semantic_control` (`status/pause/resume/enable/unload/disable`), `replay_backup` (dry-run then authorized execute), `cleanup_history`, `set_entity`, `activate_pending`.

## Upgrading from an older database

This release replaces the structured-claims derivation and separate memory/section vector indexes with one `memory_evidence` + `memory_evidence_vec` index. Databases from before this architecture are refused at startup and must be rebuilt side by side — never point the new runtime directly at an old database.

```bash
# Preview only.
mema upgrade --dry-run

# Stop every mema MCP client, then migrate and switch the standard JSON config.
mema upgrade

# Restart the MCP client and verify.
mema doctor --json
```

The command displays memory count, estimated evidence volume and vector storage, free disk space, source, and target before asking for confirmation. Migration usually takes 1–5 minutes and MCP remains unavailable during that window. The old database is never deleted. Standard JSON configuration is backed up and switched only after full verification; environment-variable `db_path` overrides are reported as a manual action. Use `--no-switch` to build and verify without editing configuration, or `--yes` for non-interactive execution. The lower-level `mema migrate-vnext` command remains available for diagnostics. Keep the old database until the new one has run successfully in normal use; if the new database has accepted writes, do not switch back without first accounting for those newer records.

## Configuration

Configuration discovery order:

1. `MEMORY_ARBITER_CONFIG`
2. `~/.config/memory-arbiter/config.json`
3. environment variables and defaults

See [`examples/memory-arbiter.config.example.json`](examples/memory-arbiter.config.example.json) and [`.env.example`](.env.example).

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

## Degradation

- Without sqlite-vec or an embedding model, lexical recall and memory governance continue; evidence indexing is unavailable.
- Without Qwen, deterministic `notify` still surfaces and `check` becomes `ignore`.
- If SQLite is unavailable or unwritable, writes use the append-only JSONL envelope only when that write succeeds.

## Development

```bash
uv run pytest -q
python scripts/sync_version.py --check
```

## 中文摘要

Memory Arbiter（迷码）是面向 AI Agent 的本地可信事实层：每条事实只存一份完整原文，向量与检索均为可重建的派生索引；冲突候选经 evidence KNN → `notify/check/ignore` → 可选本地 Qwen 降噪 → Agent 读两侧完整记忆终审，Qwen 不编辑记忆、不确认冲突。正式冲突以左右 memory version 为 CAS 快照；治理操作须用户对本次动作明确授权（`authorized=true`）。升级老库用 `mema upgrade` 旁路构建并切换配置，旧库不删。详见 [INTRO.md](INTRO.md) 与 [docs/INTEGRATION.md](docs/INTEGRATION.md)。
