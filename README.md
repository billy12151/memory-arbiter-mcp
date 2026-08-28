# Memory Arbiter MCP

**English | [中文](README.zh-CN.md)**

Memory Arbiter is a trustworthy local fact layer for AI agents — not just shared memory, but shared facts that are current, trusted, traceable, and safe to use. It is a local SQLite service exposed over MCP: four product tools, evidence-based recall, advisory conflict notices, and user-authorized governance. Every fact is stored once in local SQLite and every model it can call runs locally.

> Current release: `0.14.8` (governance authorization gates, attribute-aligned coexistence veto, unresolved-conflict visibility in doctor/console plus a synchronous prompt when editing a conflicted memory, and migration/replay durability hardening).

## Why trust it

- **One complete source of truth.** Every memory keeps its full original text. Evidence vectors, full-text search, and rankings are all *derived* indexes — rebuildable, never the only copy.
- **Provenance on every write.** Each memory carries `source_type`, `source_ref`, `event_time`, and `ingest_time`. The `user_confirmed` label is reserved by convention for facts the user explicitly verified; technically enforced protection is what happens after labeling — a `user_confirmed` memory is locked against silent edits.
- **Trust levels.** `normal`/`protected`/`locked` protection levels prevent an agent from silently overwriting what is locked; `memory_govern(confirm)` promotes a memory to `user_confirmed` only with per-action user authorization.
- **Full version history.** Every edit appends to `memory_history` with a version bump, and supersede chains keep old facts traceable instead of silently replaced.
- **One conflict record per event.** A single `conflicts` table holds the immutable detection snapshot, value groups, decision, and application results for each one-to-many conflict event. Qwen proposes no winner and never edits memory.
- **Authorized governance.** Every state-changing `memory_govern` action requires per-action `authorized=true` after the user confirms that specific action.
- **Local-only.** Embeddings run on a local GGUF model; the optional Qwen filter is a local GGUF too. The single outbound call is an optional PyPI update check, disabled with `update_check.enabled=false`.

## Install & quickstart

### Install with your AI Agent

Paste this into Codex, Claude Code, Cursor, or another coding agent with terminal access:

```text
Read the latest README at https://github.com/billy12151/memory-arbiter-mcp.
Install and configure the latest mema release for my operating system and current AI client.
Preserve any existing config and database; do not overwrite or delete existing data.
Ask me before choosing between materially different install modes, changing existing config,
or performing any destructive or privileged action. When finished, run mema doctor and report
the install method, config path, database path, client integration, and verification result.
```

The agent should treat this README as the source of truth, inspect the local environment before choosing `uvx`, core, `vec`, or `semantic-local`, and stop for user input when a safe choice cannot be inferred. A successful install is not complete until `mema doctor` has run and any warning has been reported.

### Install manually

```bash
pip install memory-arbiter-mcp
pip install "memory-arbiter-mcp[vec]"            # sqlite-vec evidence recall
pip install "memory-arbiter-mcp[semantic-local]" # local GGUF runtime (embeddings + Qwen)
```

Run `mema setup` to write `~/.config/memory-arbiter/config.json` and self-check the embedding environment (it never installs or downloads anything). Its starter template includes DB/backup paths, stdio/localhost HTTP transport, core workspace controls (`isolation`, canonical matching, weak weighting, strict admission, cutoff/guard), vec, embedding, recall caps, and `update_check.enabled=true`. The reference `examples/memory-arbiter.config.example.json` additionally shows optional workspace-Qwen and semantic-conflict tuning. Then wire your MCP client from `examples/*.mcp.json` and start the server with `mema`.

The server requires an explicitly configured identity: set `client` and `agent_id` in config.json or `MEMORY_ARBITER_CLIENT`/`MEMORY_ARBITER_AGENT_ID` in the environment (the stdio `examples/*.mcp.json` entries do this via `env`). There are no built-in defaults — the server refuses to start when either is blank. Under stdio this configured identity is the process-level caller identity used for attribution and policy decisions; `memory(action="remember")` does not accept `agent_id`/`client` in `data`. streamable-http takes caller identity from the per-request headers described below.

stdio remains the default. For one local server shared by several clients, set `mcp.transport` to `streamable-http` (or `MEMORY_ARBITER_MCP_TRANSPORT=streamable-http`) and connect to `http://127.0.0.1:8000/mcp`. Each client's MCP server entry must set fixed `X-Mema-Client` and `X-Mema-Agent-Id` headers; see [`examples/streamable-http.mcp.json`](examples/streamable-http.mcp.json). The client sends them automatically on every HTTP MCP request—agents should not add identity to individual tool calls. Missing, empty, invalid, duplicated, or conflicting identity is rejected instead of falling back to defaults. Community HTTP mode binds only to localhost, and these headers are advisory provenance and policy input, **not authentication or multi-tenant isolation**.

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

Workspace canonical normalization runs in every `isolation` mode and is separate from access control. `none` applies no workspace ACL: an omitted workspace spans the library, while an explicitly supplied workspace is canonicalized and scopes that read. `weak` adds a soft ranking/hint signal; `workspace_weak_vector_weight=true` makes that nudge decay continuously with guarded canonical-vector distance. Under `strict`, Qwen never silently merges a near-match: a new workspace stays `pending` until authorized `memory_govern(confirm_pending_workspace)` activates it. Strict visibility is exact-canonical by default. When `workspace_recall_admission=true`, workspace-sensitive recall/read/repair operations, conflict/notice workflows, and console content/count views share one admitted set: the caller canonical plus every canonical at or below `workspace_recall_cutoff` (default `0.25`) after default-pool, short-name (`workspace_min_name_len`), and generic-substring guards. Process-global maintenance (for example semantic runtime control, backup replay, doctor, and settings) is not a workspace-scoped content view. Missing vectors or sqlite-vec degradation fall back to the exact caller canonical. The reserved `default` pool is insulated and is not visible from a strict project scope. Automatic vector/Qwen normalization affects only the memory's `workspace_canonical`; supported workspace governance uses rename, migrate, move-by-id (`move_memories_workspace`), pending confirmation, and full-registry confirmation. Internal redirect/negative-decision state prevents old names from re-splitting and suppressed candidates from reappearing, but is not a user-facing workflow.

The first successful write that registers a canonical workspace returns a non-blocking top-level `workspace_review` notice in `none`/`weak`, plus `data.write_hints.new_workspace_detected`. Review possible duplicates before running authorized `confirm_workspaces`. `strict` instead returns the existing blocking `action_required=confirm_new_workspace` flow and does not emit the duplicate non-blocking notice.

## Operating mema

- `mema doctor [--json|--deep]` — read-only health checks; `--deep` loads the GGUF model and probes the live embedding dimension. `workspace.review` warns (CLI exit 1) for canonicals missing from the reviewed snapshot. Rename/merge duplicates first, then call authorized `memory_govern(confirm_workspaces)` without an explicit list to snapshot the current registry and return this check to pass. The overall CLI exits 0 only when no other warning remains.
- `mema console` — read-only local console on 127.0.0.1.
- `memory(action="status")` — surfaces `local_text_evidence` coverage, `vec_index_state`, the process-local index queue, and `semantic_conflict` runtime including queue drops/restarts and `check_degradation.last_reason`.
- Maintenance tasks on `memory_repair`: `rebuild_evidence` (dry-run then batched execute; after an embedding-model change the index reports `state=mismatch` and rebuild flips it back to `ready` automatically), `semantic_control` (`status/pause/resume/enable/unload/disable`), `replay_backup` (dry-run then authorized execute), `cleanup_history`, `set_entity`, `activate_pending`.

Evidence/semantic queues are process-local, so a crash or forced shutdown can lose queued work. Do not infer durable coverage from queue depth. After restart or any `queue_full`/discard/restart signal, inspect `local_text_evidence` coverage, run `rebuild_evidence` until its dry-run is empty and the vector state is `ready`, then run/paginate `scan_candidates` as the conflict-recovery backstop. Rebuilding evidence is idempotent derived-index repair; scanning is what recovers conflict candidates/notices that were never processed.

## Upgrading from an older database

**Upgrade warning for 0.14.8:** current runtime startup accepts only schema generation `workspace_state_v1`. Both `conflict_groups_v2` and `local_text_evidence_v1`, plus older claim/memory-vector/section-vector databases, are refused without modification. Run the public side-by-side `mema upgrade`. Every schema migration declares `vector_effect=preserve|rebuild`; the migrations from the two previous evidence generations preserve vector payloads regardless of current model availability. Compatibility is evaluated separately: a different configured embedding space records `state=mismatch`, disables vector reads, and is repaired later with `memory_repair(rebuild_evidence)`. Both paths compact current workspace redirect/negative-decision state and discard the obsolete workspace decision event ledger.

The side-by-side copy retains memory content/history, backup replay receipts, workspace canonicals and current redirect/negative-decision state, and audit. The obsolete workspace decision event ledger is not copied. Preserve migrations clone FTS/evidence/vector payloads unchanged and transactionally rebuild only the conflict domain; vector health or space mismatch never changes the structural migration result. Rebuild migrations regenerate evidence and vectors. Both paths intentionally start with empty new `conflicts`/notice state and do not copy old `conflicts`, append-only `conflict_judgments`, or `semantic_notices` history. Current contradictions must be rediscovered by a scheduled full-library scan.

After rebuild, status/doctor reports `conflict_scan_required=true` with a persistent scan epoch. Only a successful full scan covering the upgrade-time active-memory set with the matching detector version may CAS-clear that flag; partial pages, failed scans, and older-detector scans do not. The target is published only after row/fingerprint checks, a successful `PRAGMA wal_checkpoint(TRUNCATE)`, and removal of target WAL/SHM sidecars — the full-rebuild path additionally requires complete eligible evidence coverage; the source database is never deleted.

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

The full evidence-rebuild path requires sqlite-vec, a configured/readable local GGUF embedding model, `llama-cpp-python` (install the `semantic-local` extra because it also runs GGUF embeddings), a writable target directory, and enough free disk. A preserve migration does not load either model and does not require vector completeness; it reports vector compatibility independently and marks incompatible preserved data `mismatch`. The optional semantic-conflict Qwen model itself is never a migration prerequisite. The command reports its selected mode, vector effect/compatibility, memory count, estimated vector work, free disk space, source, and target before asking for confirmation.

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
| `mcp.transport` | `stdio` (default) or opt-in `streamable-http` localhost server |
| `mcp.http.host` / `port` / `path` | Local HTTP endpoint; host is restricted to loopback, defaults to `127.0.0.1:8000/mcp` |
| `mcp.http.stateless` | Stateless Streamable HTTP (default `true`); set `false` only for clients that require server-side MCP sessions |
| `update_check.enabled` | Optional one-shot background PyPI discovery (default `true`); the only network call, with cached/suppressed notices and no auto-upgrade |
| `vec.enabled` | Enables sqlite-vec evidence recall |
| `embedding.model_path` | Local GGUF embedding model |
| `embedding.max_unit_chars` | Safety bound for one evidence embedding call |
| `isolation` | `none`, `weak`, or `strict` workspace behavior |
| `semantic_conflict.model_path` | Optional local Qwen2.5-0.5B GGUF for bidirectional four-field extraction |
| `semantic_conflict.notice_sync_wait_ms` | Write-time notice delivery wait (`0..5000`, default `5000` ms); timeout continues the same task asynchronously |
| `semantic_conflict.workspace_qwen_budget_ms` | Independent workspace near-match Qwen budget (`50..5000`, default `750` ms) |
| `semantic_conflict.job_timeout_ms` | Between-pair asynchronous job budget |
| `semantic_conflict.inference_timeout_ms` | Hard timeout for one Qwen call |

## HTTP mode: sharing one local server

stdio (the default) needs no background process: each MCP client launches `mema` as its own short-lived child process. Switch to `streamable-http` only when you want **one long-lived local server that several clients connect to**.

| | stdio (default) | streamable-http |
| --- | --- | --- |
| Who starts mema | each client spawns a child process | you run one persistent process; clients connect to it |
| Background process needed | **no** | **yes** — otherwise it dies when the terminal closes |
| Client config | command + args | url + two fixed request headers |
| Good for | one person, one client | several clients on one machine sharing one memory store |

Setting it up:

1. **Config**: set `mcp.transport` to `"streamable-http"` in `~/.config/memory-arbiter/config.json` (or `MEMORY_ARBITER_MCP_TRANSPORT=streamable-http`).
2. **Keep it running**: mema has **no built-in daemon** — use a process manager. On macOS, the launchd template at [`examples/com.memory-arbiter.mema.plist`](examples/com.memory-arbiter.mema.plist) runs it at load, restarts on crash, and logs to `/tmp/mema.{out,err}.log` (replace `__MEMA_BIN__` with the absolute path `which mema` prints; put it in `~/Library/LaunchAgents/` then `launchctl load`). For a quick try, `tmux new -d -s mema 'mema'` works.
3. **Client**: copy [`examples/streamable-http.mcp.json`](examples/streamable-http.mcp.json), filling in `X-Mema-Client` and `X-Mema-Agent-Id`.

Notes: HTTP defaults to stateless request handling because mema keeps memory and semantic-notice state in SQLite, not in an MCP session. A service restart therefore does not leave clients holding an expired server session. Semantic notices created asynchronously are claimed from SQLite and attached to a later successful tool response as before; only a worker job that has not yet persisted its notice can be interrupted by a process restart. Set `mcp.http.stateless=false` only when a client specifically requires server-side MCP sessions or server-initiated SSE messages.

The client sends the fixed headers automatically on every HTTP MCP request — agents must not add identity to individual tool `data`, or it is rejected. Missing/empty/duplicate/conflicting identity fails closed (400), never falling back to defaults. The service binds to loopback only; these headers are provenance, **not authentication**. Because launchd does not inherit your shell PATH or expand `~`, put absolute paths in `ProgramArguments` and for any GGUF `model_path` in config.json.

### Claude Desktop / Claude Code through localhost HTTP

Claude's local MCP configuration launches stdio commands. To reuse one running mema HTTP service instead of spawning another mema process, put this single entry under `mcpServers` in `~/.claude.json` (current Claude Desktop/Cowork and Claude Code installations may share this user-level file):

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "/opt/homebrew/bin/npx",
      "args": [
        "-y",
        "mcp-remote@0.1.43",
        "http://127.0.0.1:8000/mcp",
        "--allow-http",
        "--transport", "http-only",
        "--header", "X-Mema-Client:claude",
        "--header", "X-Mema-Agent-Id:claude",
        "--silent"
      ],
      "env": {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "NO_PROXY": "127.0.0.1,localhost"
      }
    }
  }
}
```

Use the absolute `npx` path from `which npx` on your machine. Remove any older `memory-arbiter` entry that directly launches `mema`/`memory-arbiter-mcp`, otherwise Claude may start a second server process. Fully quit and reopen Claude Desktop, and restart Claude Code sessions after changing the file. `mcp-remote` is a third-party bridge; the pinned version above is the configuration tested with mema. If your Claude installation uses a separate Desktop MCP file, place the same single entry there instead, but do not register both copies.

## Degradation

- Without sqlite-vec or an embedding model, lexical recall and memory governance continue; evidence indexing is unavailable.
- Without Qwen, strict write-time model-dependent notices fail closed; scheduled scan continues returning its deterministic KNN/rule baseline candidates.
- If SQLite is unavailable or unwritable, writes use the append-only JSONL envelope only when that write succeeds. JSONL contains memory records and their selected canonical, not internal redirects or negative decisions; preserve or upgrade the SQLite database to retain workspace decision state.

## Development

```bash
uv run pytest -q
python scripts/sync_version.py --check
```

During development, package/docs may describe an unreleased dev version while `server.json` intentionally remains at the last published registry release (`0.13.1`). The registry manifest is advanced only as part of release preparation; do not treat that deliberate lag as the runtime/database upgrade matrix.

## 中文摘要

Memory Arbiter（迷码）是面向 AI Agent 的本地可信事实层：每条事实只存一份完整原文，向量与检索均为可重建的派生索引。冲突 scan 走宽门召回，write-time notice 走双向四字段 Qwen 抽槽与严格 grounding；单一 `conflicts` 表保存一对多事件，生命周期为 `open → applying → resolved` 或 `not_a_conflict`。裁决后按 `judge → apply_conflict_action → resolve_conflict` 顺序治理。`none/weak/strict` 都做 workspace 归一；strict 可选 guarded vector admission，default 池不进入项目 scope。`workspace_state_v1` 升级会清除旧 conflict/judgment/notice 历史和旧 workspace decision event ledger，并要求完成带 epoch 的全库 scan。**完整中文文档见 [README.zh-CN.md](README.zh-CN.md)**；另见 [INTRO.md](INTRO.md) 与 [docs/INTEGRATION.zh-CN.md](docs/INTEGRATION.zh-CN.md)。
