<!-- mcp-name: io.github.billy12151/memory-arbiter-mcp -->
# Memory Arbiter MCP

**English | [中文](README.zh-CN.md)**

Memory Arbiter is a trustworthy local fact layer for AI agents — not just shared memory, but shared facts that are current, trusted, traceable, and safe to use. It is a local SQLite service exposed over MCP: four product tools, evidence-based recall, advisory conflict notices, and user-authorized governance. Every fact is stored once in local SQLite and every model it can call runs locally.

> Current release: `0.17.1` (mDeBERTa three-class judge with the claims channel retired; workspace required on remember + workspace-govern actions, `workspaces` view; automatic row-vector backfill after upgrade; full-project review fix wave).

## Why trust it

- **One complete source of truth.** Every memory keeps its full original text. Evidence vectors, full-text search, and rankings are all *derived* indexes — rebuildable, never the only copy.
- **Provenance on every write.** Each memory carries `source_type`, `source_ref`, `event_time`, and `ingest_time`. The `user_confirmed` label is reserved by convention for facts the user explicitly verified; technically enforced protection is what happens after labeling — a `user_confirmed` memory is locked against silent edits.
- **Trust levels.** `normal`/`protected`/`locked` protection levels prevent an agent from silently overwriting what is locked; `memory_govern(confirm)` promotes a memory to `user_confirmed` only with per-action user authorization.
- **Full version history.** Every edit appends to `memory_history` with a version bump, and supersede chains keep old facts traceable instead of silently replaced.
- **One conflict record per event.** A single `conflicts` table holds the immutable detection snapshot, value groups, decision, and application results for each one-to-many conflict event. The judge proposes no winner and never edits memory.
- **Authorized governance.** Every state-changing `memory_govern` action requires per-action `authorized=true` after the user confirms that specific action.
- **Local-only.** Embeddings run on a local GGUF model; the optional conflict judge is a local mDeBERTa checkpoint (CPU). The single outbound call is an optional PyPI update check, disabled with `update_check.enabled=false`.

## Benchmark: LoCoMo-Refined

Evaluated on [LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined)
(all 1,382 questions, official judge `Qwen3-14B`):

| Configuration | Overall | Text-only |
|---|---:|---:|
| **mema + writer-agent** (deepseek-v4.1-flash writer, public rules) | **61.00%** | **65.62%** |
| **mema verbatim** (zero-LLM write path, utterances + `event_time` as-is) | 43.13% | **49.48%** |

Context: Mem0's full pipeline scores 48.91% (text-only re-scored by the
benchmark authors) — **bare mema with no LLM in the write path matches it**;
with a writer-agent, mema sits second on the leaderboard's text-only basis.
The write path also detected **552 semantic conflicts** across the 20 replayed
conversations (self-trained mDeBERTa judge, running as shipped).

Full methodology, ablations (including answerer-independence and the 82%
information ceiling), reproduction steps and submission files:
[`evals/locomo/RESULTS.md`](./evals/locomo/RESULTS.md).

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
# One command: package + deps + embedding model + config.json (resumable downloads, ModelScope fallback)
curl -fsSL https://memarbiter.cn/install.sh | bash
```

Or step by step:

```bash
pip install "memory-arbiter-mcp[vec,semantic-local]"  # core + sqlite-vec + local GGUF runtime
mema setup --install   # downloads the embedding GGUF (~330MB, resumable); the mDeBERTa judge is a separate manual install
mema doctor            # verify
```

`mema setup` without `--install` stays guidance-only: it writes `~/.config/memory-arbiter/config.json` and self-checks the environment without touching pip or the network; `--install` is the execution mode (pip installs the extras, downloads the embedding GGUF model with resume/mirror fallback, and writes the finished config itself). Since 0.15.0 configuration is file-only and the whole user surface is 21 keys (see [Configuration](#configuration)): paths, identity, workspace/isolation, `update_check.enabled`, `include_size`, the embedding model, the optional semantic-conflict mDeBERTa judge (`mdeberta_ckpt`, see below), and MCP transport/host/port. The reference `examples/memory-arbiter.config.example.json` shows the same slim surface with per-key notes. Then wire your MCP client from `examples/*.mcp.json` and start the server with `mema`.

When a capability is missing (e.g. the models were never downloaded), every tool response carries a persistent degraded-mode banner with the `mema setup --install` remediation, and each agent's first call includes a capability health card — an incomplete install cannot pass for a complete one silently.

The server requires an explicitly configured identity: set `client` and `agent_id` in config.json or the `MEMORY_ARBITER_CLIENT`/`MEMORY_ARBITER_AGENT_ID` launch-context environment variables (the stdio `examples/*.mcp.json` entries do this via `env`). There are no built-in defaults — the server refuses to start when either is blank. Under stdio this configured identity is the process-level caller identity used for attribution; `memory(action="remember")` does not accept `agent_id`/`client` in `data`. streamable-http takes caller identity from the per-request headers described below.

stdio remains the default. For one local server shared by several clients, set `mcp.transport` to `streamable-http` (or `MEMORY_ARBITER_MCP_TRANSPORT=streamable-http`, one of the six retained launch-context variables) and connect to `http://127.0.0.1:8000/mcp`. Each client's MCP server entry must set fixed `X-Mema-Client` and `X-Mema-Agent-Id` headers; see [`examples/streamable-http.mcp.json`](examples/streamable-http.mcp.json). The client sends them automatically on every HTTP MCP request—agents should not add identity to individual tool calls. Missing, empty, invalid, duplicated, or conflicting identity is rejected instead of falling back to defaults. Community HTTP mode binds only to localhost, and these headers are advisory provenance, **not authentication or multi-tenant isolation**.

The daily loop is four calls — `remember` a reusable fact, `find` to recall, `read` for exact lookup, `update` when a newer source replaces an existing current memory (never create a second active copy of one source of truth). `remember` requires `workspace` (0.17.1): first call `memory_review(view="workspaces")` to list existing buckets, then pass an existing canonical name — `'default'` is the global pool; pass a new name only when deliberately creating a new bucket. Point any agent at the packaged rule:

```json
{"action":"help","data":{"topic":"agent_onboarding"}}
```

## The four tools

- `memory`: `remember`, `find`, `batch_find`, `read`, `batch_read`, `update`, `judge`, `status`, `help`
- `memory_review`: read-only health, conflict groups/details, history, expired memory, audit, and entities
- `memory_govern`: explicitly authorized retirement, conflict-plan application/resolution, confirmation, and workspace governance
- `memory_repair`: evidence rebuild, broad conflict scanning/recording, history cleanup, entity assignment, pending activation, backup replay, semantic runtime control, and notice lifecycle

Every product call returns the envelope `{ok, mode, warnings, degraded, data}`. Operation-specific `action_required`, `next_action`, `replan`, and records live under `data`; successful calls may additionally carry a top-level `notices` array. Each notice has its own `action_required` and machine-readable call under the notice object. Do not look for a generic top-level `action_required`.

`batch_find` runs up to 8 queries in one call and merges the pages (dedup by `memory_id`; each item carries `matched_query_ids`/`best_query_id`; `per_query` reports per-query stats) — for multi-topic tasks this replaces 5-10 tool round-trips with one. Since 0.15.9 find is also honest about emptiness: a query that recalls nothing returns an empty page with a reword hint (no recent-memory stuffing), and candidates below the calibrated relevance floor (8.1) never enter a query-recall page at all — fewer "looks related, isn't" citations.

`find` is an index page: by default (`content_mode="preview"`, 0.15.10) each result carries metadata plus `content_chars` (the full-text length — what a `read` would cost) and a bounded `outline` of up to 8 `{head, offset}` segments whose offsets share `read`'s span coordinate system, so `span=[offset, offset+N]` slices that exact segment. **Keyword query style (0.17.0):** a query of space-separated short CJK words (each ≤4 chars, e.g. `向量 唯一键 冲突`) is understood as keywords — memories semantically close to the query whose content/subject contains one of the keywords (whole-word substring) rank higher; topic-generic words that flood the candidate pool are ignored, and no forced matching happens when a keyword appears nowhere (reword instead). Content depth is a single-choice enum: `content_mode="hits"` adds `hit_spans` — the vector-matched units as `{text, start_offset, end_offset}` sliced from the source (read `span=[start,end]` returns exactly that text). Hit spans are never truncated: when merged hits cover ≥50% of the content the item upgrades to full text with `hit_spans` kept as an annotation; items without vector hits keep the plain preview shape. `hit_window=N` (default 0) extends each hit with ±N neighbouring complete sentences (neighbours marked `matched=false`); `hit_spans` appears only on query-recall pages — browse/filter pages carry none — and stale-version hits are dropped with a `stale_hit_spans` marker plus a re-query warning (the evidence index may lag right after an edit). `content_mode="full"` returns whole texts (the old `include_content=true`, removed in 0.15.10 — the call fails loudly with a migration pointer). Scores compare only within the page, and if the top page misses you should reword the query or add `tags_filter` rather than deep-page — unfiltered query-recall reports `total_estimate=null`/`has_more=false`, while filtered recall keeps the exact count. The `size` block meters the page as actually returned: `returned_chars`/`returned_count` and a `tokens_estimate` from a deterministic bucket-table estimator (`heuristic_v1`) calibrated against a Qwen2.5 tokenizer at design time (calibration reference only — no Qwen runtime ships since 0.17.1); it runs ~30% high on pure Chinese prose and ~17% high on pure English — the estimate and the estimated share one yardstick, so savings comparisons stay valid. Since 0.15.6 the same size block rides every recall surface — `read` (meters the record as returned, span windows included), `memory_review` `expired` and `history` (meter their result lists) — under one global config key `include_size` (default `true`); each block's `display_hint` repeats the token number with a report-this-recall-cost instruction, `include_size=false` turns all of them off together, and `find`'s old per-call `include_size` parameter is ignored with a warning. `unresolved_conflict_count` appears only when page items directly hit an open/applying conflict group, and counts those page items.

## How recall works

Lexical and evidence channels recall independently and merge per memory with reciprocal-rank fusion, then trust, recency, filter, and workspace adjustments.

- **Lexical**: FTS5 over content plus subject/tags LIKE; the bounded content-LIKE anchor channel runs only when vectors are unavailable (degradation path since 0.16.10).
- **Evidence**: the write-path job derives row segments from the `subject` (leading subject row), sentences, and table rows (0.17.0: the former unit tables retired; row vectors are the one evidence channel). The indexer never extracts facts, infers entities, or calls a model — it only slices the stored source. Evidence hits carry source offsets. `memory(action="read", data={"memory_id": 42, "span":{"start":120,"end":640}})` returns only that clipped source window plus `data.span.{start,end,total_chars}`; omit `span` to read the complete source. Span bounds are strict integers with `0 <= start < end`, and `end` clips at content length. Table segments longer than 100 rows are exempt as a whole (no row vectors, no pair detection — visible as `table_rows_exempted` in receipts).

## Conflict groups and notices

Evidence KNN recalls sentence-level neighbours; it does not decide conflict truth. Since 0.17.1, each candidate pair that survives the deterministic funnel (sentence prefilter, memory-level screen, cosine band, rule evidence) is judged by an optional local **mDeBERTa** encoder (three classes: `conflict` / `no_conflict` / `possible_conflict`, plus a mechanism head). The judge sees the two bare row texts — no metadata, no prompt engineering — and returns calibrated probabilities:

- `conflict` at P ≥ `semantic_conflict.mdeberta_notice_min_prob` (default 0.5) → a normal-severity notice;
- `possible_conflict` → an `info`-severity notice (grey-zone reference, no `action_required`);
- `no_conflict` → silent clear (counted in the receipt).

The judge never chooses a winner, suppresses the scheduled scan, or edits memory.

There are deliberately two gates:

- **Scheduled scan is broad.** `memory_repair(task="scan_candidates")` retains deterministic KNN/rule candidates; the scan path runs no model — the Agent reviews candidates there. `include_duplicates=true` stays a single-page spot check (full record_conflict-compatible members for that page); for a full duplicate sweep use the separate `memory_repair(task="scan_duplicates")` task, which aggregates every page server-side under one global 200-pair cap. The external reviewer records every triaged candidate with `record_conflict(status="open"|"not_a_conflict")` to obtain snapshot dedupe.
- **Write-time notice is judge-gated.** A user-visible notice requires the judge to say `conflict` above the configured probability floor (or the deterministic same-key/value-difference direct verdict). The slot identity rides `workspace + subject` plus a pair-hash difference anchor (the deterministic direct path keeps the real attribute). Anything less fails open into later scan review; internal (same-memory) findings always land pending — the scan-side strong model owns negative verdicts. The synchronous-delivery wait (configurable `semantic_conflict.notice_sync_wait_ms`, default `3000`, clamp `0–5000`) only gates when a notice is attached; it does not change detection.

**Set up the scheduled tasks.** mema does not ship an internal timer by design: the pipeline's value loop ends in agent-side judgment, so the external scheduler is what wakes the agent. 0.16.0 spec v2 replaces the v1 page-driven triage loop: an hourly task kicks `memory_repair(task="scan_pipeline", data={"action": "kick"})` repeatedly until `complete=true` (the server walks the library itself, decides full-vs-incremental from per-memory watermarks, and lands every suspected item in an agent-only judgment queue — never in notices or user-facing conflict lists), then drains it with `memory_repair(task="scan_queue", data={"action": "page"})` / `action="submit"` (server-side land-from-reference: dismiss → suppression source, confirm → the agent supplies only slot_key + display values). Keep a daily `memory_review(view="doctor")`. `scan_candidates` remains as a manual/diagnostic channel only; `conflicts.spec_drift` flags v1-era tasks for rebuilding. Each completed full-scan boundary (a page returning `next_anchor_memory_id=null` with anchors scanned) appends one lightweight audit line to `scan_log.jsonl`; until then agents receive a `scan_never_run`/`scan_stale` guidance notice, and doctor flags a still-owed rebuild (`conflicts.scan_required`) or a scan idle beyond 14 days (`conflicts.scan_stale`) — once the tasks run, both fall silent on their own. The full platform-agnostic spec is `memory(action="help", data={"topic": "scheduled_tasks"})`.

The single `conflicts` table stores one one-to-many event and its immutable member/value snapshot. Its public lifecycle is `open → applying → resolved`, with `not_a_conflict` as a terminal triage result. `memory(action="judge")` CAS-pins the conflict revision, records the chosen value and plan, and moves it to `applying`; execute each returned `memory_govern(action="apply_conflict_action")` sequentially with explicit authorization and the latest revision, then call authorized `resolve_conflict` only after every planned member action completes. `use_as_resolution` must land on a member that holds the chosen value group. The default judgment corrects the wrong data in memory — plan `update_current_claim` or `append_superseded_context` for members holding superseded claims, and use `preserve_historical_record` only when the user explicitly asks to keep the historical record. Partial failures remain `applying`: when `data.action_required="replan_conflict"`, re-read the group/members and call authorized `memory_govern(action="replan_conflict")` with the current revision and replacement plan — a grounding-failed `update_current_claim` is recovered either by re-editing so the chosen value appears verbatim in the member content (the stored machine-normalized form; for multi-word phrases that means the normalized string itself), or by passing a replacement `chosen_value` drawn from the group's `value_groups` (0.15.15; replacing `chosen_value` also moves the resolution holder to the new value group unless pinned explicitly). Replanning preserves prior plan history; never retry stale precomputed steps.

## Workspaces

Workspace canonical normalization runs in every `isolation` mode and is separate from access control. `none` applies no workspace ACL: an omitted workspace spans the library, while an explicitly supplied workspace is canonicalized and scopes that read. `weak` adds a soft ranking/hint signal (a fixed binary nudge — the continuous vector-distance weighting is no longer a knob). Under `strict`, the system never silently merges a near-match: a new workspace stays `pending` until authorized `memory_govern(confirm_pending_workspace)` activates it. Strict visibility uses guarded vector admission (always on since 0.15.0, a frozen constant): workspace-sensitive recall/read/repair operations, conflict/notice workflows, and console content/count views share one admitted set: the caller canonical plus every canonical at or below a 0.25 cosine cutoff after default-pool, short-name, and generic-substring guards. Process-global maintenance (for example semantic runtime control, backup replay, doctor, and settings) is not a workspace-scoped content view. Missing vectors or sqlite-vec degradation fall back to the exact caller canonical. The reserved `default` pool is insulated and is not visible from a strict project scope. Automatic vector normalization affects only the memory's `workspace_canonical`; supported workspace governance uses rename, migrate, move-by-id (`move_memories_workspace`), pending confirmation, and full-registry confirmation. Internal redirect/negative-decision state prevents old names from re-splitting and suppressed candidates from reappearing, but is not a user-facing workflow.

The first successful write that registers a canonical workspace returns a non-blocking top-level `workspace_review` notice in `none`/`weak`, plus `data.write_hints.new_workspace_detected`. Review possible duplicates before running authorized `confirm_workspaces`. `strict` instead returns the existing blocking `action_required=confirm_new_workspace` flow and does not emit the duplicate non-blocking notice.

### Recall blacklist (0.15.5)

Workspaces listed in `recall_blacklist.jsonl` (next to the database file) are excluded from **unscoped** `find` recall — the default ambient pool. One bare workspace name per line; blank lines and `#` comments are ignored; edits are live on the next find. No file → the built-in default applies (`mema-twin`, the mema-twin preference bucket); an empty file → nothing is excluded; a created file replaces the default entirely.

What is *not* filtered: an explicit `workspace` argument (even a blacklisted one), strict isolation, filter-driven recall (empty query + `tags_filter`/time/`source_type`), the expired-audit path, write-time dedup, conflict scans, and id-based reads. Exclusion matches both the canonical and the raw workspace column, so rows whose canonical drifted are still caught. `doctor` reports the effective list (`recall.blacklist`).

## Operating mema

- `mema doctor [--json|--deep]` — read-only health checks; `--deep` loads the GGUF model and probes the live embedding dimension. `workspace.review` warns (CLI exit 1) for canonicals missing from the reviewed snapshot. Rename/merge duplicates first, then call authorized `memory_govern(confirm_workspaces)` without an explicit list to snapshot the current registry and return this check to pass. The overall CLI exits 0 only when no other warning remains.
- `mema console` — read-only local console on 127.0.0.1.
- `memory(action="status")` — surfaces `local_text_evidence` coverage, `vec_index_state`, the process-local index queue, and `semantic_conflict` runtime including queue drops/restarts and `check_degradation.last_reason`.
- Maintenance tasks on `memory_repair`: `rebuild_evidence` (dry-run then batched execute; after an embedding-model change the index reports `state=mismatch` and rebuild flips it back to `ready` automatically), `semantic_control` (`status/pause/resume/enable/unload/disable`), `replay_backup` (dry-run then authorized execute), `cleanup_history`, `set_entity`, `activate_pending`, and `scan_duplicates` (a one-call full-library near-duplicate sweep bounded at 200 lightweight pairs; `include_quotes=true` adds the triggering evidence quotes).

Evidence/semantic queues are process-local, so a crash or forced shutdown can lose queued work. Do not infer durable coverage from queue depth. After a restart or an evidence-side `busy`/discard signal, inspect `local_text_evidence` coverage and run `rebuild_evidence` until its dry-run is empty and the vector state is `ready`; semantic-worker queue drops are recovered by the scheduled `scan_candidates` pass, not by `rebuild_evidence`. Rebuilding evidence is idempotent derived-index repair; scanning is what recovers conflict candidates/notices that were never processed.

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

After the upgrade, the first server start launches a daemon thread that backfills sentence row vectors automatically; watch progress in `mema doctor` (`rows.coverage`) or `memory(action="status")`. With no embedding model configured the backfill stays pending and resumes once the model is available — nothing to run by hand.
```

The full evidence-rebuild path requires sqlite-vec, a configured/readable local GGUF embedding model, `llama-cpp-python` (install the `semantic-local` extra because it also runs GGUF embeddings), a writable target directory, and enough free disk. A preserve migration does not load either model and does not require vector completeness; it reports vector compatibility independently and marks incompatible preserved data `mismatch`. The optional conflict-judge checkpoint itself is never a migration prerequisite. The command reports its selected mode, vector effect/compatibility, memory count, estimated vector work, free disk space, source, and target before asking for confirmation.

The explicit checkpoint above matters because copying only the main `.sqlite3` file while live WAL frames exist is not a complete backup; alternatively use SQLite's online `.backup` command before stopping. Abort if `wal_checkpoint(TRUNCATE)` reports a non-zero busy count. `mema upgrade` also checkpoints/verifies the new target before switching, but it does not create the operator's rollback copy of the source.

The old database is never deleted. Standard JSON configuration is backed up and switched only after full verification; environment-variable `db_path` overrides are reported as a manual action. Use `--no-switch` to build and verify without editing configuration. `--yes` skips both the interactive confirmation **and** its acknowledgement that all writers/workers are stopped and old conflict/judgment/notice history will be permanently omitted; it does not stop processes, checkpoint the source, or create a backup. The lower-level `mema migrate-vnext` command remains available for diagnostics. Keep the old database until the new one has run successfully in normal use; if the new database has accepted writes, do not switch back without first accounting for those newer records.

## Configuration

Configuration is file-only since 0.15.0. Everything tunable lives in `~/.config/memory-arbiter/config.json` (or the file the `MEMORY_ARBITER_CONFIG` launch-context variable points at; `mema setup` writes the starter template). Engine parameters, timeouts, thresholds, and caps are frozen constants (`memory_arbiter/constants.py`).

The complete user surface is 21 keys (0.15.14: added `semantic_conflict.n_gpu_layers`, removed `semantic_conflict.max_notice_pairs` and `policy_path`; 0.17.1: removed `claims.required`, `semantic_conflict.model_path` and `n_gpu_layers`, added the four `semantic_conflict.mdeberta_*` keys):

```json
{
  "db_path": "~/.local/share/memory-arbiter/memory.sqlite3",
  "backup_jsonl": "~/.local/share/memory-arbiter/memory.backup.jsonl",
  "client": "your-client",
  "agent_id": "your-agent-id",
  "workspace": "default",
  "isolation": "none",
  "update_check": { "enabled": true },
  "include_size": true,
  "embedding": {
    "model_path": "~/.local/share/memory-arbiter/models/embedding.gguf",
    "auto_query": true,
    "auto_write": true
  },
  "semantic_conflict": {
    "enabled": true,
    "mdeberta_ckpt": "~/.local/share/memory-arbiter/models/mdeberta-v52_ep3_fp16.pt",
    "on_write": "async",
    "notice_sync_wait_ms": 3000
  },
  "mcp": {
    "transport": "stdio",
    "http": { "host": "127.0.0.1", "port": 8000 }
  }
}
```

| Setting | Purpose |
| --- | --- |
| `db_path` | Current SQLite database |
| `backup_jsonl` | Append-only fallback when SQLite cannot write |
| `client` / `agent_id` | Required caller identity; no built-in defaults — the server refuses to start when either is blank |
| `workspace` / `isolation` | Default workspace and `none`/`weak`/`strict` workspace behavior |
| `update_check.enabled` | Optional one-shot background PyPI discovery (default `true`); the only network call, with cached/suppressed notices and no auto-upgrade |
| `include_size` | Global switch for the recall size block (v0.15.6): on = `find`/`read`/`expired`/`history` all attach `{returned_chars, returned_count, tokens_estimate}`; off = none of them do |
| `embedding.model_path` | Local GGUF embedding model — pointing at it is the sole intent to enable sqlite-vec evidence recall |
| `embedding.auto_query` / `auto_write` | Auto-embed at query/write time (default `true`) |
| `semantic_conflict.mdeberta_ckpt` | Local mDeBERTa checkpoint (v52_ep3_fp16, ~531MB, ships separately) for write-time conflict judging. Configured → auto-enabled, loaded at startup, kept resident (CPU fp32). Install the `mdeberta` extra first; see the model-download note below |
| `semantic_conflict.enabled` | Explicit off-switch; unset + `mdeberta_ckpt` means enabled, explicit `false` wins |
| `semantic_conflict.on_write` | Write-time detection: `async` (default) or `off` |
| `semantic_conflict.notice_sync_wait_ms` | How long the write response waits for the post-commit check so its result rides along (v0.15.8, default `3000`, clamp `0–5000`); `0` = never block the write response — batch ingestion still gets the check run asynchronously and notices deliver on a later response |
| `semantic_conflict.mdeberta_model_dir` | config.json + tokenizer directory; default = `<ckpt dir>/mdeberta-base` |
| `semantic_conflict.mdeberta_notice_min_prob` | P(conflict) floor for a normal-severity notice (default `0.5`); below it the pair is counted, not notified |
| `semantic_conflict.mdeberta_batch` | Judge batch size (default `0` = auto, device-tiered: 16 with a GPU — Apple Silicon / NVIDIA — 8 on CPU; an explicit value `>0` overrides the tier. No startup probe) |
| `mcp.transport` | `stdio` (default) or opt-in `streamable-http` localhost server |
| `mcp.http.host` / `port` | Local HTTP endpoint; host is restricted to loopback, defaults to `127.0.0.1:8000`; the endpoint path is fixed at `/mcp` |

See [`examples/memory-arbiter.config.example.json`](examples/memory-arbiter.config.example.json).

Semantics worth knowing: the embedding dimension comes from the model itself — the database records the active dimension, and switching to a model with a different dimension automatically drops and rebuilds the vector tables at the new dimension at startup (a one-time full evidence rebuild). Ranking is fixed hybrid (lexical + evidence fusion); there is no ranking-mode knob. HTTP request handling is stateless with a 4 MB request-body cap.

Six environment variables remain as launch context: `MEMORY_ARBITER_CONFIG`, `MEMORY_ARBITER_DB_PATH`, `MEMORY_ARBITER_BACKUP_JSONL`, `MEMORY_ARBITER_MCP_TRANSPORT`, `MEMORY_ARBITER_CLIENT`, `MEMORY_ARBITER_AGENT_ID`. They select process context (which config file, which DB, which transport, which identity), and a config-file value wins over the matching variable. Every other `MEMORY_ARBITER_*` variable is no longer read — a stale export surfaces a "no longer read" warning in `mema doctor`, the console settings page, and `memory(action="status")`. Removed file keys similarly warn "no longer configurable" and are ignored; [docs/INTEGRATION.md](docs/INTEGRATION.md) carries the 0.14 → 0.15 key-migration table.

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

Notes: HTTP request handling is stateless (a frozen constant since 0.15.0) because mema keeps memory and semantic-notice state in SQLite, not in an MCP session. A service restart therefore does not leave clients holding an expired server session. Semantic notices created asynchronously are claimed from SQLite and attached to a later successful tool response as before; only a worker job that has not yet persisted its notice can be interrupted by a process restart.

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

## Conflict judge model (mDeBERTa, optional)

The write-time conflict judge needs a one-time manual install (the 1.1GB checkpoint does not ship on PyPI):

1. `pip install memory-arbiter-mcp[mdeberta]` (torch CPU wheel ~200MB + transformers);
2. download `mdeberta-v52_ep3_fp16.pt` (~531MB, fp16 weights) plus the `mdeberta-base` directory (config.json + tokenizer, MB-sized) from the mini-clash release artifacts;
3. point `semantic_conflict.mdeberta_ckpt` at the `.pt` file in config.json (auto-enables + preloads at startup; `mdeberta_model_dir` defaults to a `mdeberta-base` folder next to the checkpoint).

`mema doctor` verifies the dependency, the checkpoint digest, and the judge's label contract. Unconfigured = write-time arbitration disabled (scan/Agent fallback unaffected). ONNX+INT8 (smaller, no torch) is planned for a later release.

## Degradation

- Without sqlite-vec or an embedding model, lexical recall and memory governance continue; evidence indexing is unavailable.
- Without the mDeBERTa judge (extra/checkpoint/config), write-time arbitration is disabled (notices pause); scheduled scan continues returning its deterministic KNN/rule baseline candidates.
- If SQLite is unavailable or unwritable, writes use the append-only JSONL envelope only when that write succeeds. JSONL contains memory records and their selected canonical, not internal redirects or negative decisions; preserve or upgrade the SQLite database to retain workspace decision state.

## Development

```bash
uv run pytest -q
python scripts/sync_version.py --check
```

During development, package/docs may describe an unreleased dev version while `server.json` (the MCP Registry manifest) stays at the last released version; `scripts/sync_version.py` advances it as part of release preparation, and `--check` keeps it from drifting.

## 中文摘要

Memory Arbiter（迷码）是面向 AI Agent 的本地可信事实层：每条事实只存一份完整原文，向量与检索均为可重建的派生索引。冲突 scan 走宽门召回，write-time notice 走 mDeBERTa 三分类判定（0.17.1 起，漏斗门全保留）；单一 `conflicts` 表保存一对多事件，生命周期为 `open → applying → resolved` 或 `not_a_conflict`。裁决后按 `judge → apply_conflict_action → resolve_conflict` 顺序治理。`none/weak/strict` 都做 workspace 归一；strict 使用 guarded vector admission，default 池不进入项目 scope。`workspace_state_v1` 升级会清除旧 conflict/judgment/notice 历史和旧 workspace decision event ledger，并要求完成带 epoch 的全库 scan。**完整中文文档见 [README.zh-CN.md](README.zh-CN.md)**；另见 [INTRO.md](INTRO.md) 与 [docs/INTEGRATION.zh-CN.md](docs/INTEGRATION.zh-CN.md)。
