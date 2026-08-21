# Memory Arbiter MCP

Memory Arbiter is a trustworthy local fact layer for AI agents — not just shared memory, but shared facts that are current, trusted, traceable, and safe to use. It is a local SQLite service exposed over MCP: four product tools, evidence-based recall, advisory conflict notices, and user-authorized governance. Every fact is stored once in local SQLite and every model it can call runs locally.

> Development documentation: `0.14.0.dev2` (memory 715 baseline; destructive conflict-history upgrade described below).

## Why trust it

- **One complete source of truth.** Every memory keeps its full original text. Evidence vectors, full-text search, and rankings are all *derived* indexes — rebuildable, never the only copy.
- **Provenance on every write.** Each memory carries `source_type`, `source_ref`, `event_time`, and `ingest_time`. The `user_confirmed` label is reserved by convention for facts the user explicitly verified; technically enforced protection is what happens after labeling — a `user_confirmed` memory is locked against silent edits.
- **Trust levels.** `normal`/`protected`/`locked` protection levels prevent an agent from silently overwriting what is locked; `memory_govern(confirm)` promotes a memory to `user_confirmed` only with per-action user authorization.
- **Full version history.** Every edit appends to `memory_history` with a version bump, and supersede chains keep old facts traceable instead of silently replaced.
- **One conflict record per event.** A single `conflicts` table holds the immutable detection snapshot, value groups, decision, and application results for each one-to-many conflict event. Qwen proposes no winner and never edits memory.
- **Authorized governance.** Every state-changing `memory_govern` action requires per-action `authorized=true` after the user confirms that specific action.
- **Local-only.** Embeddings run on a local GGUF model; the optional Qwen filter is a local GGUF too. The single outbound call is an optional PyPI update check, disabled with `update_check.enabled=false`.

## Install & quickstart

```bash
pip install memory-arbiter-mcp
pip install "memory-arbiter-mcp[vec]"            # sqlite-vec evidence recall
pip install "memory-arbiter-mcp[semantic-local]" # local GGUF runtime (embeddings + Qwen)
```

Run `mema setup` to write `~/.config/memory-arbiter/config.json` and self-check the embedding environment (it never installs or downloads anything). Its packaged inline template is deliberately smaller than the reference example: it writes DB/backup, tool profile, workspace-isolation defaults (`isolation`, `workspace_match_distance`), vec, embedding, recall caps, and `update_check.enabled=true`; only the semantic-conflict keys are omitted. Add those from `examples/memory-arbiter.config.example.json` when needed. Then wire your MCP client from `examples/*.mcp.json` and start the server with `mema`.

The daily loop is four calls — `remember` a reusable fact, `find` to recall, `read` for exact lookup, `update` when a newer source replaces an existing current memory (never create a second active copy of one source of truth). Point any agent at the packaged rule:

```json
{"action":"help","data":{"topic":"agent_onboarding"}}
```

## The four tools

- `memory`: `remember`, `find`, `read`, `update`, `judge`, `status`, `help`
- `memory_review`: read-only health, conflict groups/details, history, expired memory, audit, and entities
- `memory_govern`: explicitly authorized retirement, conflict-plan application/resolution, confirmation, and workspace governance
- `memory_repair`: evidence rebuild, broad conflict scanning/recording, history cleanup, entity assignment, pending activation, backup replay, semantic runtime control, and notice lifecycle

Every product call returns the envelope `{ok, mode, warnings, degraded, data}`. Operation-specific `action_required`, `next_action`, `replan`, and records live under `data`; successful calls may additionally carry a top-level `notices` array. Each notice has its own `action_required` and machine-readable call under the notice object. Do not look for a generic top-level `action_required`.

## How recall works

Lexical and evidence channels recall independently and merge per memory with reciprocal-rank fusion, then trust, recency, filter, and workspace adjustments.

- **Lexical**: FTS5 over content plus subject/tags LIKE and a bounded content-LIKE anchor channel.
- **Evidence**: a background worker derives local-text evidence units from the `subject`, Markdown headings, sentence/paragraph groups, and overlapping windows for long text. The indexer never extracts facts, infers entities, or calls a model — it only slices the stored source. Evidence hits carry source offsets. `memory(action="read", data={"memory_id": 42, "span":{"start":120,"end":640}})` returns only that clipped source window plus `data.span.{start,end,total_chars}`; omit `span` to read the complete source. Span bounds are strict integers with `0 <= start < end`, and `end` clips at content length.

## Conflict groups and notices

Evidence KNN recalls sentence-level neighbours; it does not decide conflict truth. For each short pair, optional local Qwen runs in both directions (A→B and B→A) and may return exactly four fields:

```json
{"attribute_a":"database","value_a":"MySQL","attribute_b":"database","value_b":"SQLite"}
```

Code then validates the JSON, side mapping, mechanical attribute/value normalization, quote grounding, duplicate/compatibility rules, and entity/scope provenance. Qwen never chooses a winner, suppresses the scheduled scan, or edits memory.

There are deliberately two gates:

- **Scheduled scan is broad.** `memory_repair(task="scan_candidates")` retains deterministic KNN/rule candidates and — when `semantic_conflict.scan_enhance=true` (default) and the local Qwen backend is up — runs a bounded per-page Qwen enhancement: rule candidates gain extracted attribute/value fields and `value_groups`, similarity-only pairs (normally opt-in via `include_check`) that extract a valid same-attribute/different-value in either direction are unioned in, and verified candidates with matching entity/scope are aggregated into `slot_groups`. `scan_max_pairs`/`scan_budget_ms` bound the cost. Single-direction, weak-grounding, and incomplete entity/scope cases remain `review_candidate`; a model failure never shrinks the baseline candidate set. The external reviewer records every triaged candidate with `record_conflict(status="open"|"not_a_conflict")` to obtain snapshot dedupe.
- **Write-time notice is strict.** A user-visible notice requires two valid, mutually consistent four-field extractions, grounded differing values, a complete canonical `workspace + entity + attribute + scope`, and no deterministic coexistence veto. Anything less fails closed into later scan review. `notice_sync_wait_ms` only controls synchronous delivery; it does not change detection.

The single `conflicts` table stores one one-to-many event and its immutable member/value snapshot. Its public lifecycle is `open → applying → resolved`, with `not_a_conflict` as a terminal triage result. `memory(action="judge")` CAS-pins the conflict revision, records the chosen value and plan, and moves it to `applying`; execute each returned `memory_govern(action="apply_conflict_action")` sequentially with explicit authorization and the latest revision, then call authorized `resolve_conflict` only after every planned member action completes. Partial failures remain `applying`: when `data.action_required="replan_conflict"`, re-read the group/members and call authorized `memory_govern(action="replan_conflict")` with the current revision and replacement plan. Replanning preserves prior plan history; never retry stale precomputed steps.

## Workspaces

Workspace canonical normalization runs in every `isolation` mode and is separate from access control. `none` performs the same canonical normalization as `weak` but applies no workspace ACL; an unscoped read still spans all workspaces, while a workspace passed explicitly on a read is canonicalized and used to scope that query. `weak` additionally uses workspace as a soft ranking/hint signal. Under `strict`, Qwen never silently merges a near-match: a new workspace stays `pending` until authorized `memory_govern(confirm_pending_workspace)` activates it, and reads/governance are scoped to the canonical workspace. Automatic vector/Qwen normalization affects only the memory's `workspace_canonical`; only explicit governance creates confirmed/rejected aliases.

## Operating mema

- `mema doctor [--json|--deep]` — read-only health checks; `--deep` loads the GGUF model and probes the live embedding dimension.
- `mema console` — read-only local console on 127.0.0.1.
- `memory(action="status")` — surfaces `local_text_evidence` coverage, `vec_index_state`, the process-local index queue, and `semantic_conflict` runtime including queue drops/restarts and `check_degradation.last_reason`.
- Maintenance tasks on `memory_repair`: `rebuild_evidence` (dry-run then batched execute; after an embedding-model change the index reports `state=mismatch` and rebuild flips it back to `ready` automatically), `semantic_control` (`status/pause/resume/enable/unload/disable`), `replay_backup` (dry-run then authorized execute), `cleanup_history`, `set_entity`, `activate_pending`.

Evidence/semantic queues are process-local, so a crash or forced shutdown can lose queued work. Do not infer durable coverage from queue depth. After restart or any `queue_full`/discard/restart signal, inspect `local_text_evidence` coverage, run `rebuild_evidence` until its dry-run is empty and the vector state is `ready`, then run/paginate `scan_candidates` as the conflict-recovery backstop. Rebuilding evidence is idempotent derived-index repair; scanning is what recovers conflict candidates/notices that were never processed.

## Upgrading from an older database

**Development upgrade warning for 0.14.0.dev2:** current runtime startup accepts only schema generation `conflict_groups_v2`. Both the immediately previous `local_text_evidence_v1` database and older claim/memory-vector/section-vector databases are classified as legacy and refused without modification. Both use the public side-by-side `mema upgrade` command, but the work differs: `local_text_evidence_v1` takes the fast conflict-only path and reuses evidence/vector tables unchanged; older generations rebuild evidence and vectors.

The side-by-side copy retains memory content/history, backup replay receipts, workspace canonical/alias governance, audit, and logical `memory_evidence` source units. From `local_text_evidence_v1`, it clones existing FTS/evidence/vector state unchanged and transactionally rebuilds only the conflict domain. From older generations it rebuilds FTS and republishes evidence vectors in the configured embedding space. Both paths intentionally start with empty new `conflicts`/notice state and do not copy old `conflicts`, append-only `conflict_judgments`, or `semantic_notices` history. Current contradictions must be rediscovered by a scheduled full-library scan.

After rebuild, status/doctor reports `conflict_scan_required=true` with a persistent scan epoch. Only a successful full scan covering the upgrade-time active-memory set with the matching detector version may CAS-clear that flag; partial pages, failed scans, and older-detector scans do not. The target is published only after row/fingerprint checks, complete eligible evidence coverage, a successful `PRAGMA wal_checkpoint(TRUNCATE)`, and removal of target WAL/SHM sidecars; the source database is never deleted.

```bash
# Preview only.
mema upgrade --dry-run

# Stop every mema MCP client/worker. Make a WAL-safe rollback backup:
sqlite3 /absolute/path/to/memory.sqlite3 "PRAGMA wal_checkpoint(TRUNCATE);"
cp /absolute/path/to/memory.sqlite3 /absolute/path/to/memory.pre-0.14.sqlite3

# Migrate and switch the standard JSON config.
mema upgrade

# Restart the MCP client and verify.
mema doctor --json
```

The full evidence-rebuild path requires sqlite-vec, a configured/readable local GGUF embedding model, `llama-cpp-python` (install the `semantic-local` extra because it also runs GGUF embeddings), a writable target directory, and enough free disk. The fast `local_text_evidence_v1` conflict-only path does not load either model. The optional semantic-conflict Qwen model itself is never a migration prerequisite. The command reports its selected mode, memory count, estimated vector work, free disk space, source, and target before asking for confirmation.

The explicit checkpoint above matters because copying only the main `.sqlite3` file while live WAL frames exist is not a complete backup; alternatively use SQLite's online `.backup` command before stopping. Abort if `wal_checkpoint(TRUNCATE)` reports a non-zero busy count. `mema upgrade` also checkpoints/verifies the new target before switching, but it does not create the operator's rollback copy of the source.

The old database is never deleted. Standard JSON configuration is backed up and switched only after full verification; environment-variable `db_path` overrides are reported as a manual action. Use `--no-switch` to build and verify without editing configuration. `--yes` skips both the interactive confirmation **and** its acknowledgement that all writers/workers are stopped and old conflict/judgment/notice history will be permanently omitted; it does not stop processes, checkpoint the source, or create a backup. The lower-level `mema migrate-vnext` command remains available for diagnostics. Keep the old database until the new one has run successfully in normal use; if the new database has accepted writes, do not switch back without first accounting for those newer records.

## Configuration

Configuration discovery order:

1. `MEMORY_ARBITER_CONFIG` (points at a config file)
2. `~/.config/memory-arbiter/config.json`
3. environment variables and defaults

Within one scope, a value set in the config file wins over the corresponding environment variable (e.g. `db_path` in `config.json` overrides `MEMORY_ARBITER_DB_PATH`); environment variables apply when no config file sets the key.

See [`examples/memory-arbiter.config.example.json`](examples/memory-arbiter.config.example.json) and [`.env.example`](.env.example).

| Setting | Purpose |
| --- | --- |
| `db_path` | Current SQLite database |
| `backup_jsonl` | Append-only fallback when SQLite cannot write |
| `update_check.enabled` | Optional one-shot background PyPI discovery (default `true`); the only network call, with cached/suppressed notices and no auto-upgrade |
| `vec.enabled` | Enables sqlite-vec evidence recall |
| `embedding.model_path` | Local GGUF embedding model |
| `embedding.max_unit_chars` | Safety bound for one evidence embedding call |
| `isolation` | `none`, `weak`, or `strict` workspace behavior |
| `semantic_conflict.model_path` | Optional local Qwen2.5-0.5B GGUF for bidirectional four-field extraction |
| `semantic_conflict.notice_sync_wait_ms` | Write-time notice delivery wait (`0..5000`, default `3000` ms); timeout continues the same task asynchronously |
| `semantic_conflict.workspace_qwen_budget_ms` | Independent workspace near-match Qwen budget (`50..5000`, default `750` ms) |
| `semantic_conflict.job_timeout_ms` | Between-pair asynchronous job budget |
| `semantic_conflict.inference_timeout_ms` | Hard timeout for one Qwen call |

## Degradation

- Without sqlite-vec or an embedding model, lexical recall and memory governance continue; evidence indexing is unavailable.
- Without Qwen, strict write-time model-dependent notices fail closed; scheduled scan continues returning its deterministic KNN/rule baseline candidates.
- If SQLite is unavailable or unwritable, writes use the append-only JSONL envelope only when that write succeeds.

## Development

```bash
uv run pytest -q
python scripts/sync_version.py --check
```

During development, package/docs may describe an unreleased dev version while `server.json` intentionally remains at the last published registry release (`0.13.1`). The registry manifest is advanced only as part of release preparation; do not treat that deliberate lag as the runtime/database upgrade matrix.

## 中文摘要

Memory Arbiter（迷码）是面向 AI Agent 的本地可信事实层：每条事实只存一份完整原文，向量与检索均为可重建的派生索引。冲突 scan 走宽门召回，write-time notice 走双向四字段 Qwen 抽槽与严格 grounding；单一 `conflicts` 表保存一对多事件，生命周期为 `open → applying → resolved` 或 `not_a_conflict`。裁决后按 `judge → apply_conflict_action → resolve_conflict` 顺序治理。`none/weak/strict` 都做 workspace 归一，`none` 仅代表无 ACL。0.14.0.dev2 升级会清除旧 conflict/judgment/notice 历史并要求完成带 epoch 的全库 scan。详见 [INTRO.md](INTRO.md) 与 [docs/INTEGRATION.md](docs/INTEGRATION.md)。
