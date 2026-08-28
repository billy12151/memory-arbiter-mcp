# Integration Guide

**English | [中文](INTEGRATION.zh-CN.md)**

This guide describes the `0.14.9` contract.

## MCP Surface

Configure the command as `mema` and use the four product tools. Call `memory(action="help")` or the corresponding tool help to discover current fields.

stdio is the default transport. To share one Community process among local clients, set `mcp.transport="streamable-http"` and connect each client to `http://127.0.0.1:8000/mcp`. HTTP request handling is stateless by default (`mcp.http.stateless=true`): memory and semantic notices live in SQLite, and a later successful tool response claims any pending semantic notice regardless of the MCP connection that initiated the asynchronous work. This also prevents a server restart from stranding clients on an expired in-memory session. Set it to `false` only for a client that requires server-side MCP sessions or server-initiated SSE messages. A process restart can still interrupt a worker job that has not persisted its notice yet.

The server requires an explicitly configured identity on every transport: set `client` and `agent_id` in config.json or `MEMORY_ARBITER_CLIENT`/`MEMORY_ARBITER_AGENT_ID` in the environment. There are no built-in defaults, and the server refuses to start when either is blank. Under stdio this configured identity is the process-level caller identity — attribution and policy decisions use it, and `agent_id`/`client` fields in tool `data` are never accepted as provenance. streamable-http takes the caller identity from the per-request headers below.

In each MCP server configuration, set fixed `X-Mema-Client` and `X-Mema-Agent-Id` request headers; the client then sends them automatically on initialize, tool discovery, and tool calls. Do not have the agent dynamically add identity to tool `data`. HTTP mode fails closed when either header is missing, empty, invalid, duplicated, or conflicts with tool data. It is restricted to loopback: the headers provide local provenance and policy input, not authentication, tenant isolation, or permission to expose the service publicly.

Writes require a non-empty `subject`; include `source_type`, `event_time`, `source_ref`, and useful tags when known. Pass the real project `workspace` for project facts. Use explicit `workspace="default"` only for facts intentionally stored in the global pool; do not rely on omission because client settings may supply a workspace. Strict isolation requires a workspace. Use `user_confirmed` only for facts explicitly verified by the user. When a new source replaces an existing current source, find/read the existing memory and update it instead of creating a second active copy.

## Evidence Recall

With sqlite-vec and a local embedding model configured, writes asynchronously publish sentence/paragraph evidence derived from the stored source. Lexical and evidence channels recall independently and merge per memory with reciprocal-rank fusion before trust, recency, filters, and workspace adjustments. Evidence offsets identify relevant source text.

`memory(action="read", data={"memory_id":42})` returns the complete source. Add `"span":{"start":120,"end":640}` to return only `data.memory.content[120:640]` plus `data.span.{start,end,total_chars}`. Bounds must be strict JSON integers with `0 <= start < end`; an oversized end clips to content length, while a start beyond content fails. `scan_candidates.deep_read` may provide ready-to-use spans; semantic-notice read calls are full-memory reads by design, so omit the span parameter there whenever full context is needed.

Use `memory_repair(task="rebuild_evidence", data={"dry_run":true})` to inspect missing/stale coverage, then queue bounded rebuild pages. Repeat execute/dry-run until no ids remain and status reports complete eligible coverage. A changed embedding space reports `state=mismatch` and disables the evidence channel until every eligible memory is republished. Evidence units and vectors are derived. Schema migrations explicitly declare `vector_effect=preserve|rebuild`; a preserve migration copies vectors unchanged, then evaluates compatibility separately without blocking the structural upgrade.

## Conflict Detection Contract

### Bidirectional four-field extraction

Evidence KNN provides bounded short-pair recall and ranking only. Optional local Qwen2.5-0.5B runs once as A→B and once as B→A. Each result must be a strict JSON object with exactly four bounded string fields:

```json
{"attribute_a":"database","value_a":"MySQL","attribute_b":"database","value_b":"SQLite"}
```

The model must not return a final conflict/coexistence decision, winner, or mutation. Code validates:

1. each direction has a concrete same-attribute/different-value extraction;
2. after swapping sides, both directions agree on normalized attributes and value-to-source mapping;
3. values are grounded in their corresponding evidence quote, except mechanical case/unit/number/confirmed-alias derivations;
4. normalized values actually differ;
5. deterministic duplicate, compatibility, environment/version/region/object, observation-time, historical/current, evolution, and measurement-scope vetoes do not apply;
6. a formal slot has sufficient `workspace_canonical + entity + attribute + scope` provenance.

Fuzzy attribute similarity cannot create a formal slot. Qwen failure or absence has no authority to veto deterministic scan candidates.

### Scheduled scan: broad gate

`memory_repair(task="scan_candidates")` enumerates bounded KNN/rule candidates without loading the whole library into the agent session. The scan keeps the deterministic baseline and, when `semantic_conflict.scan_enhance=true` (default) and the local Qwen backend is available, runs a bounded enhancement over the page: rule candidates are enriched with extracted `attribute/value` member fields and `value_groups`, and similarity-only pairs (normally opt-in via `include_check`) that extract a legal same-attribute/different-value in either direction are unioned into `candidates`. `semantic_conflict.scan_max_pairs` (default 8, 0 disables) caps Qwen pair evaluations per page and `semantic_conflict.scan_budget_ms` (default 60000) bounds the page deadline; verified candidates whose memories agree on metadata `entity/scope` are aggregated into `slot_groups`. Single-direction output, weak grounding, or missing entity/scope remains `review_candidate` for agent deep reading; Qwen absent/invalid/timeout/budget failure never reduces the baseline set and never removes a rule candidate.

Candidates carry member versions, evidence spans, candidate identity, and deep-read calls. `scan_candidates` does not itself persist triage. For every reviewed candidate, call `memory_repair(task="record_conflict")` with `status="open"` or `status="not_a_conflict"`; otherwise it may appear on a later scan. Pass `slot_key` only for `status="open"` — a `not_a_conflict` triage records through `candidate_key` alone and returns `open_group_exists` if an open group already owns the slot. Candidate-only `not_a_conflict` rows use `candidate_key` and do not invent `scope="unknown"`.

### Write-time notice: strict gate

A user-visible notice requires both valid directions, consistent side mapping, strict quote grounding, distinct normalized values, complete slot provenance, and no coexistence veto. Any failure closes the notice path and leaves the case for scheduled scan review. Notice snapshots freeze member versions, value groups, slot provenance, detector/prompt version, task id, and dedupe key.

After a successful write, the server waits at most `semantic_conflict.notice_sync_wait_ms` (default 5000, range 0–5000 ms) for the bounded notice task. `0` is fully asynchronous. If the wait expires, the write returns successfully and the same accepted task continues asynchronously; it is not cancelled or recomputed. A queue-full/rejected enqueue is different: there is no task to wait for. `checked_no_notice` means only that every candidate inside that bounded write-time task completed the strict gate; it is not a whole-library claim. Scheduled scan remains the durable recall backstop.

`semantic_conflict.job_timeout_ms` (default 5000) is a queue-fairness budget, not an inference timeout. It activates only when another semantic job is waiting and is checked between candidate pairs. An already-started Qwen request runs under `semantic_conflict.inference_timeout_ms` (default 30000) even if the job budget expires; after that pair finishes, the worker yields before starting another pair. With no backlog, the job budget is inactive.

## One Conflicts Table

There is no separate public judgment store. One `conflicts` row represents one one-to-many event and retains its immutable detection snapshot, `value_groups`, decision, and application results. Important identity/concurrency fields include `revision`, `workspace_canonical`, `slot_key`, `slot_key_hash`, `candidate_key`, `candidate_key_hash`, `member_versions`, and `member_fingerprint`.

Public lifecycle:

- `open`: confirmed as worth governing; scan may append new members for the same slot with `expected_revision` CAS.
- `applying`: a decision and apply plan exist; ordinary members cannot be appended and duplicate reminders are suppressed only for validated plan actions.
- `resolved`: every planned change completed and was reviewed; the event is closed and never expanded.
- `not_a_conflict`: this candidate snapshot was reviewed as compatible/false positive.

Only one `open`/`applying` row may exist for a canonical workspace+slot. A new value/version after resolution creates a new event.

## Resolution Workflow

1. Read `memory_review(view="conflict_detail")` and all relevant memories. Present the shared slot, every value group, and its members.
2. Call `memory(action="judge")` with the current `expected_revision`, chosen value, decision provenance/reason, resolution memory, and a per-member plan. This transitions `open → applying`.
3. Follow the returned machine call. Execute one authorized `memory_govern(action="apply_conflict_action")` at a time with the latest revision. Never run precomputed steps concurrently.
4. Re-read/replan on `stale_conflict` or `stale_member`. A failed apply returns `ok=false` with `data.action_required="replan_conflict"` and `data.replan`; partial failures remain `applying`. After re-reading the group and members, call authorized `memory_govern(action="replan_conflict")` with the current revision, replacement `apply_plan`, and optional replacement `resolution_memory_id`. Replanning preserves previous plan history and bumps revision.
5. After every current plan step succeeds, call authorized `memory_govern(action="resolve_conflict")` with the latest revision.

Prefer correcting an incorrect current fact, preserve historical/external/locked context appropriately, and reuse an existing correct active memory as `resolution_memory_id` when possible. Ordinary `memory(update)` cannot supply a conflict id as a notice-suppression switch.

## Workspace Normalization and ACL

Canonical normalization runs under every isolation mode and is independent from ACL:

- `none`: exact/confirmed/vector/rule/Qwen normalization behaves like weak mode, but no workspace ACL is applied. An omitted workspace filter returns all workspaces.
- `weak`: same normalization plus soft ranking/hints; no hard visibility filter.
- `strict`: exact/confirmed and safe mechanical rules may reuse a canonical; Qwen cannot silently merge. A new workspace remains pending until authorized `confirm_pending_workspace`. Visibility is exact-canonical by default. With `workspace_recall_admission=true`, workspace-sensitive recall/read/repair operations, conflict/notice workflows, and console content/count views share the caller canonical plus every canonical within `workspace_recall_cutoff` (default 0.25) that passes default-pool, `workspace_min_name_len`, and generic-substring guards. Process-global maintenance such as semantic runtime control, backup replay, doctor, and settings is not scoped this way. Missing vectors or sqlite-vec degradation falls back to exact-canonical scope; the insulated `default` pool is never admitted into a strict project scope.

In `none` and `weak`, the first write that registers a canonical returns a non-blocking top-level notice with `type=workspace_review`, `action_required=review_workspace_registry`, a doctor review call, and the authorized `confirm_workspaces` call to use only after review. Repeated writes to an existing canonical do not repeat it. `strict` uses its blocking pending-workspace response instead.

Resolution order is internal confirmed/negative workspace decisions, exact canonical, bounded vector candidates, deterministic `AUTO|KEEP|ASK`, then Qwen only for an undecided near-match. Qwen must choose from the supplied candidates and may suggest an `alias`/`typo`/`same_project` relationship, but automatic normalization writes only the memory's `workspace_canonical`; it does not create a persistent redirect. Negative decisions suppress repeated proposals. Product governance uses rename, migrate, pending confirmation, and full-registry review; internal decision rows are not a product workflow.

Workspace and conflict inference share a serial local worker. `semantic_conflict.workspace_qwen_budget_ms` (default 750, range 50–5000 ms) is an independent short budget: timeout/busy preserves the raw canonical and returns a review hint rather than blocking the write-time notice gate.

## Response Envelope

All four product tools return `{ok, mode, warnings, degraded, data}`. Operation results and required steps are under `data`, for example `data.action_required`, `data.next_action`, or `data.replan`. On a successful response only, delivery side channels may add a top-level `notices` array. A semantic notice stub's `action_required` and `read_call` live inside that notice item. Clients should therefore inspect both `response.data.action_required` and every `response.notices[*].action_required`; there is no generic top-level action-required field.

## Notice Lifecycle and Recovery

Notice delivery is an atomic best-effort database claim, not transport exactly-once. Internal `pending` and `delivered` both appear publicly as `open`; attaching a stub changes `pending → delivered` and sets `delivered_at`. Public terminal actions are `dismiss` (false positive, conflict becomes `not_a_conflict`) and `resolve` (already handled). If either pinned memory snapshot has changed before delivery, the server may internally mark the notice `stale`. Delivery itself does not edit memory, judge/resolve a formal conflict, or supersede either side. Note that read-classified surfaces still advance delivery state: notice list/read and `memory(action="status")` delivery claim pending notices (`pending → delivered`, or `stale` when a pinned snapshot changed), including for policy-denied HTTP identities, which are read-only by design.

The evidence and semantic queues are process-local. A crash, forced shutdown, queue-full drop, or model-subprocess restart can lose unprocessed indexing/classification work even though the memory write committed. Recovery is coverage-driven: inspect `memory(action="status")`, run `rebuild_evidence` repeatedly until dry-run is empty, eligible coverage is complete, and vector state is ready; then paginate `scan_candidates` and record each reviewed candidate. Do not assume restart reconstructs the old queue. `rebuild_evidence` restores source-derived units/vectors; the subsequent scan restores missed conflict discovery.

## Backup and Upgrade

JSONL replay is previewable without authorization. Applying replay requires explicit user authorization and is idempotent by replay key/payload hash. JSONL stores memory records and their selected canonical only; it does not restore internal redirects or negative decisions. Retain the SQLite source/upgrade artifact when workspace decision state must survive.

### Upgrade matrix

| Source database | Runtime behavior | Public upgrade path | Retained / rebuilt |
| --- | --- | --- | --- |
| New/empty or `workspace_state_v1` | Starts normally | None | Existing current data |
| `conflict_groups_v2` or `local_text_evidence_v1` | Refused without modification | Side-by-side `mema upgrade`; conflict-only only on an exact ready-space match | Core/public data retained; conflict history and old workspace decision events omitted; FTS/evidence/vector reused only when space-compatible, otherwise rebuilt |
| Older claim + memory/section-vector generations | Refused without modification | Same side-by-side `mema upgrade` | Core/public data retained; evidence units generated/retained and vectors rebuilt |
| Unknown, partial, failed/resuming target | Refused | Diagnose with `mema doctor --json`; repair/resume only with the lower-level migration workflow | Never open as current until verification succeeds |

There is no public in-place upgrade: both paths build and verify a side-by-side target. A full rebuild requires sqlite-vec, a readable configured GGUF embedding model, `llama-cpp-python` (the `semantic-local` extra also supplies the embedding runtime), a writable target directory, and enough free space. The conflict-only path is selected only when the source vector state is `ready` and its active space ID exactly matches the configured model and pipeline; it then reuses evidence/vector state without loading a model. The separate optional semantic-conflict Qwen model is not a migration prerequisite.

### WAL-safe procedure

1. Preview with `mema upgrade --dry-run`.
2. Stop every old-version writer/client and drain or terminate the semantic worker.
3. Create a rollback copy only after checkpointing the stopped source:

   ```bash
   sqlite3 /absolute/path/to/memory.sqlite3 "PRAGMA wal_checkpoint(TRUNCATE);"
   cp /absolute/path/to/memory.sqlite3 /absolute/path/to/memory.pre-0.14.sqlite3
   ```

   Abort if the checkpoint reports non-zero `busy`; copying only the main file while committed frames remain in `-wal` is incomplete. SQLite's online `.backup` command is also safe before shutdown.
4. Run `mema upgrade`; restart and verify with `mema doctor --json`.
5. Complete the epoch-pinned full `scan_candidates` pass shown by status/doctor.

The side-by-side copy retains memory content/history, backup replay receipts, workspace canonicals and current redirect/negative-decision state, and audit. The obsolete workspace decision event ledger is intentionally omitted. Preserve migrations clone FTS/evidence/vector payloads unchanged and replace only the conflict domain; compatibility is evaluated independently, and an incompatible configured space is recorded as `mismatch` so vector reads stay disabled until repair. Rebuild migrations regenerate evidence vectors. The target intentionally starts with empty conflict/notice state: old `conflicts`, `conflict_judgments`, and `semantic_notices` history are not migrated. Target publication requires fingerprint stability, empty destructive tables, an atomic generation/completion marker, and a successful target `wal_checkpoint(TRUNCATE)` before WAL/SHM removal and config switch; rebuild migrations additionally require complete eligible evidence coverage and `state=ready` in the expected space.

On success `conflict_scan_required=true` and a persistent epoch are recorded. Only a full scan covering the upgrade-time active set with the matching detector can CAS-clear it. A partial page, failed scan, or old detector cannot. `--yes` only skips the prompt acknowledging writer shutdown and permanent conflict/judgment/notice-history loss; it does not stop processes, checkpoint/back up the source, or relax verification. `--no-switch` leaves configuration untouched. Environment-overridden/mismatched config paths require the printed manual switch.

## 中文摘要

0.14.2 使用单一 `conflicts` 表保存一对多冲突事件、裁决和应用结果；生命周期为 `open → applying → resolved` 或 `not_a_conflict`。Qwen 只做 A→B/B→A 四字段 attribute/value 抽槽，不选正确值、不修改记忆。scheduled scan 宽门保召回，write-time notice 双向一致且严格 grounding 才提醒。治理顺序固定为 `judge → apply_conflict_action（逐条 CAS）→ resolve_conflict`。`none/weak/strict` 都做 workspace canonical normalization；strict 可选 guarded vector admission，default 池不进入项目 scope。升级到 `workspace_state_v1` 会丢弃旧 conflict/judgment/notice 历史和旧 workspace decision event ledger，并设置必须由匹配 detector 的完整全库 scan 清除的 epoch 标志。**完整中文集成指南见 [INTEGRATION.zh-CN.md](INTEGRATION.zh-CN.md)。**
