# Integration Guide

This guide describes the `0.14.0.dev2` development contract.

## MCP Surface

Configure the command as `mema` and use the four product tools. Call `memory(action="help")` or the corresponding tool help to discover current fields.

Writes should include `subject`, `source_type`, `event_time`, `workspace`, `source_ref`, and useful tags. Use `user_confirmed` only for facts explicitly verified by the user. When a new source replaces an existing current source, find/read the existing memory and update it instead of creating a second active copy.

## Evidence Recall

With sqlite-vec and a local embedding model configured, writes asynchronously publish sentence/paragraph evidence derived from the stored source. Lexical and evidence channels recall independently and merge per memory with reciprocal-rank fusion before trust, recency, filters, and workspace adjustments. Evidence offsets identify relevant source text.

`memory(action="read", data={"memory_id":42})` returns the complete source. Add `"span":{"start":120,"end":640}` to return only `data.memory.content[120:640]` plus `data.span.{start,end,total_chars}`. Bounds must be strict JSON integers with `0 <= start < end`; an oversized end clips to content length, while a start beyond content fails. `scan_candidates.deep_read` and semantic-notice read calls may provide ready-to-use spans; omit the span whenever full context is needed.

Use `memory_repair(task="rebuild_evidence", data={"dry_run":true})` to inspect missing/stale coverage, then queue bounded rebuild pages. Repeat execute/dry-run until no ids remain and status reports complete eligible coverage. A changed embedding space reports `state=mismatch` and disables the evidence channel until every eligible memory is republished. Evidence units and vectors are derived: a migration may retain the logical source units, but vectors are always rebuilt in the configured space.

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

`memory_repair(task="scan_candidates")` enumerates bounded KNN/rule candidates without loading the whole library into the agent session. The scan keeps the deterministic baseline and unions Qwen-enhanced candidates with it. Either valid extraction direction can support recall. Single-direction output, weak grounding, or missing entity/scope remains `review_candidate` for agent deep reading; Qwen absent/invalid/timeout/budget failure never reduces the baseline set.

Candidates carry member versions, evidence spans, candidate identity, and deep-read calls. `scan_candidates` does not itself persist triage. For every reviewed candidate, call `memory_repair(task="record_conflict")` with `status="open"` or `status="not_a_conflict"`; otherwise it may appear on a later scan. Candidate-only `not_a_conflict` rows use `candidate_key` and do not invent `scope="unknown"`.

### Write-time notice: strict gate

A user-visible notice requires both valid directions, consistent side mapping, strict quote grounding, distinct normalized values, complete slot provenance, and no coexistence veto. Any failure closes the notice path and leaves the case for scheduled scan review. Notice snapshots freeze member versions, value groups, slot provenance, detector/prompt version, task id, and dedupe key.

After a successful write, the server waits at most `semantic_conflict.notice_sync_wait_ms` (default 3000, range 0–5000 ms) for the bounded notice task. `0` is fully asynchronous. If the wait expires, the write returns successfully and the same accepted task continues asynchronously; it is not cancelled or recomputed. A queue-full/rejected enqueue is different: there is no task to wait for. `checked_no_notice` means only that every candidate inside that bounded write-time task completed the strict gate; it is not a whole-library claim. Scheduled scan remains the durable recall backstop.

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
- `strict`: exact/confirmed and safe mechanical rules may reuse a canonical; Qwen cannot silently merge. A new workspace remains pending until authorized `confirm_pending_workspace`.

Resolution order is confirmed/rejected alias, exact canonical, bounded vector candidates, deterministic `AUTO|KEEP|ASK`, then Qwen only for an undecided near-match. Qwen must choose from at most five supplied candidates and may suggest `alias|typo|same_project|same_family|related|unrelated|uncertain`. Automatic vector/Qwen normalization writes only the memory's `workspace_canonical`; it does not create a confirmed alias. A rejected alias is never re-proposed.

Workspace and conflict inference share a serial local worker. `semantic_conflict.workspace_qwen_budget_ms` (default 750, range 50–5000 ms) is an independent short budget: timeout/busy preserves the raw canonical and returns a review hint rather than blocking the write-time notice gate.

## Response Envelope

All four product tools return `{ok, mode, warnings, degraded, data}`. Operation results and required steps are under `data`, for example `data.action_required`, `data.next_action`, or `data.replan`. On a successful response only, delivery side channels may add a top-level `notices` array. A semantic notice stub's `action_required` and `read_call` live inside that notice item. Clients should therefore inspect both `response.data.action_required` and every `response.notices[*].action_required`; there is no generic top-level action-required field.

## Notice Lifecycle and Recovery

Notice delivery is an atomic best-effort database claim, not transport exactly-once. Internal `pending` and `delivered` both appear publicly as `open`; attaching a stub changes `pending → delivered` and sets `delivered_at`. Public terminal actions are `dismiss` (false positive, conflict becomes `not_a_conflict`) and `resolve` (already handled). If either pinned memory snapshot has changed before delivery, the server may internally mark the notice `stale`. Delivery itself does not edit memory, judge/resolve a formal conflict, or supersede either side.

The evidence and semantic queues are process-local. A crash, forced shutdown, queue-full drop, or model-subprocess restart can lose unprocessed indexing/classification work even though the memory write committed. Recovery is coverage-driven: inspect `memory(action="status")`, run `rebuild_evidence` repeatedly until dry-run is empty, eligible coverage is complete, and vector state is ready; then paginate `scan_candidates` and record each reviewed candidate. Do not assume restart reconstructs the old queue. `rebuild_evidence` restores source-derived units/vectors; the subsequent scan restores missed conflict discovery.

## Backup and Upgrade

JSONL replay is previewable without authorization. Applying replay requires explicit user authorization and is idempotent by replay key/payload hash.

### Upgrade matrix

| Source database | Runtime behavior | Public upgrade path | Retained / rebuilt |
| --- | --- | --- | --- |
| New/empty or `conflict_groups_v2` | Starts normally; current additive migrations may run | None | Existing current data |
| `local_text_evidence_v1` (immediately previous generation) | Refused without modification | Side-by-side `mema upgrade` | Core/public data retained; conflict history omitted; evidence vectors rebuilt |
| Older claim + memory/section-vector generations | Refused without modification | Same side-by-side `mema upgrade` | Core/public data retained; evidence units generated/retained and vectors rebuilt |
| Unknown, partial, failed/resuming target | Refused | Diagnose with `mema doctor --json`; repair/resume only with the lower-level migration workflow | Never open as current until verification succeeds |

There is no public in-place conflict-only upgrade. `mema upgrade` requires sqlite-vec, a readable configured GGUF embedding model, `llama-cpp-python` (the `semantic-local` extra also supplies the embedding runtime), a writable target directory, and enough free space. The separate optional semantic-conflict Qwen model is not a migration prerequisite.

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

The side-by-side copy retains memory content/history, backup replay receipts, workspace aliases/canonicals, audit, and logical evidence units. It rebuilds FTS and republishes vectors in the configured embedding space. It intentionally starts with empty new conflict/notice state: old `conflicts`, `conflict_judgments`, and `semantic_notices` history are not migrated. Target publication requires row/fingerprint stability, complete eligible evidence coverage, empty destructive tables, and a successful target `wal_checkpoint(TRUNCATE)` before WAL/SHM removal and config switch.

On success `conflict_scan_required=true` and a persistent epoch are recorded. Only a full scan covering the upgrade-time active set with the matching detector can CAS-clear it. A partial page, failed scan, or old detector cannot. `--yes` only skips the prompt acknowledging writer shutdown and permanent conflict/judgment/notice-history loss; it does not stop processes, checkpoint/back up the source, or relax verification. `--no-switch` leaves configuration untouched. Environment-overridden/mismatched config paths require the printed manual switch.

## 中文摘要

0.14.0.dev2 使用单一 `conflicts` 表保存一对多冲突事件、裁决和应用结果；生命周期为 `open → applying → resolved` 或 `not_a_conflict`。Qwen 只做 A→B/B→A 四字段 attribute/value 抽槽，不选正确值、不修改记忆。scheduled scan 宽门保召回，write-time notice 双向一致且严格 grounding 才提醒。治理顺序固定为 `judge → apply_conflict_action（逐条 CAS）→ resolve_conflict`。`none/weak/strict` 都做 workspace canonical normalization，`none` 只是不启用 ACL。升级会丢弃旧 conflict/judgment/notice 历史，并设置必须由匹配 detector 的完整全库 scan 清除的 epoch 标志。
