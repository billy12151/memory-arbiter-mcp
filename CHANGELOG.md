# Changelog

All notable changes to memory-arbiter-mcp are documented in this file.
Versions follow semantic versioning.

## [0.14.9] — 2026-08-29

Architecture-only release (plan id=775; behavior-preserving except where noted).

### Changed

- **Typed `TrustedApplyingContext` replaces the bare §15.3 suppression dict** — the apply flow constructs a frozen dataclass (`to_dict`/`from_dict` for the worker-snapshot transport), so a typo'd field is now a construction error instead of silently disengaging suppression. Malformed snapshots previously crashed the semantic job inside `int()` coercions; `from_dict` now degrades them to `None`, fail-open to the ordinary re-check (behavior improvement). `chosen_value` is carried for audit parity; the gate does not consume it.
- **Single `_post_commit` entry for all write-path post-commit work** — ten call sites state `recheck_conflicts` explicitly instead of hand-picking between the two enqueue helpers, closing the pairing-mistake class that let a pending memory enter the semantic queue. Writers that opt out return `semantic_conflict_check: {status: skipped, reason: recheck_disabled}` where the field used to be absent (response field is now always present — **breaking for consumers asserting its absence**).
- **No more `__getattr__` forwarding; strict mypy unexempted** — all nine forwarding shims (six pipelines, three db stores) are replaced by explicit attributes and typed delegates; the three file-level `no-any-return` exemptions are deleted and strict mypy passes across the package. Newly-reached latent typing gaps fixed: `lastrowid` null handling, a `sqlite3.Row` flowing where `dict` was assumed, closure narrowing of the semantic backend, and a next-step dict inferred as `dict[str, int]`.
- **PEP 585/604 annotations enforced** — 624 annotations migrate to `X | None` / built-in generics; `UP006/UP007/UP045` join the selected Ruff rules so the mixed style cannot return (requires Python >= 3.11, both forms runtime-safe).

### Verification

- Full tests (918), strict mypy with zero exemptions, and Ruff (incl. UP) pass.

## [0.14.8] — 2026-08-28

### Added

- **Doctor reports all unresolved conflict groups** — `conflicts.backlog` now counts `open` + `applying` (was `open` only) with a split in the detail, and a new `conflicts.applying` check flags every group still mid-apply with id/idle-days evidence, so wedged apply plans surface in health checks instead of being invisible to every counting path. Candidate notices keep their own `notices.backlog` check. The default conflict listing stays pure `status=open`.
- **Console and audit counters match the doctor's unresolved definition** — `ConsoleAPI._status_counts`, `memory_audit_summary`, and the db-level audit now count `open` + `applying` instead of `open` only (previously a wedged apply plan was invisible to every counting path, and the console number could disagree with the conflict list); console counts gain an `applying_conflicts` split and the English sidebar label reads "Unresolved conflicts".
- **Editing a memory inside an unresolved conflict group prompts synchronously** — `memory(action="update")` content edits now return `attention_required` with the unresolved group ids/revisions/statuses when the edited memory is a member: the version bump can stale an open group's pinned member snapshot or an applying group's plan, so the caller is pointed at `conflict_detail` and judge / replan / resolve. tags-only edits keep the version and stay silent; the apply flow's own governed writes are a separate path and unaffected.
- **`final_sync` pre-build writer check** — the source database is probed with an EXCLUSIVE lock before the staging build starts; an active writer fails fast with `source_has_active_writer` instead of being discovered only at the post-build fingerprint gate (which remains in place).
- **Backup replay drains the evidence worker before reporting completion** — receipts marked complete now mean the derived evidence index actually landed; a drain timeout appends a warning.

### Changed

- **Governance authorization gates on two high-risk sub-paths** — `memory_repair(task="record_conflict")` with `status="not_a_conflict"` now requires `authorized=true` (routine `open` recording stays ungated), closing the slot-occupancy suppression vector; `memory(action="judge")` with `decided_by="user"` now requires `authorized=true` (`decided_by="agent"` stays ungated).
- **Coexistence veto aligns with extracted attributes** — dimension marker words (e.g. 平均/峰值, 移动端/管理后台) count as a coexisting-dimension difference only when they appear in the extracted attribute text of the corresponding side; markers found only in the quote body no longer veto. The stricter gate lets true conflicts reach the notice path that were previously misread as dimension differences. Direct `coexistence_veto` callers without extractions keep the legacy substring behavior.
- **Check-degradation records dedupe per task and reason** — one task reports each distinct technical reason once and exposes `reasons_seen`; the counter grows more slowly, so alert thresholds based on absolute counts shift meaning.
- **Pending memories skip the semantic check explicitly** — a pending memory's semantic-conflict check now returns `{status: skipped, reason: pending_workspace_activation}` instead of `incomplete/memory_not_active`; evidence indexing is unchanged.
- **Conflict slot_keys store canonical entity/scope orthography** — comparison paths match both canon and raw forms so legacy stored groups stay suppressed.

### Fixed

- **Notice-path display values tolerate malformed model output** — non-dict or missing `parsed` keys fall back to the gate values instead of raising `KeyError` in the worker, matching the scan path's existing guard.
- Removed the dead `memory_record_conflict` stub (unreachable on the four-tool MCP surface) and replaced magic backup-notice signature tuples with named constants; product help payloads build once as a module-level constant.

### Verification

- Full tests (916), strict mypy, and Ruff pass.

## [0.14.7] — 2026-08-28

### Added

- **`memory_govern(action="move_memories_workspace")`** — moves selected memories by id to another workspace bucket, writing both the raw `workspace` column and `workspace_canonical` (complement to `migrate_workspace`, which merges whole canonicals by name). Per-id pre-validation plus a single atomic transaction under the same advisory flock as rename/migrate/normalize; illegal ids land in `failed_ids` with reasons. Divergent rows (canonical already pointing elsewhere, e.g. rows written through a confirmed alias) refuse unless authorized, then re-anchor with an explicit response note and reroute guidance. Strict callers may only move within their admitted workspace scope. The destination folds onto the registered spelling and follows exactly one confirmed-alias hop, matching what an ordinary write using the same name would do; a brand-new bucket registers its canonical, publishes its vector, and emits a `workspace_review` notice. Open/applying conflict scope keys are never rewritten; cross-scope members are reported for a scan.
- **`memory_repair(task="normalize_workspaces")`** — idempotent, dry-run-by-default bulk fold of registered spelling-variant canonicals (first-seen winner) across the whole registry, repointing aliases, rewriting memories/conflicts scope keys, and installing redirects, serialized with rename/migrate via the startup flock.

### Changed

- **Request identity is required at startup** — `client`/`agent_id` have no silent defaults; `build_runtime` refuses to start without them (config values count). The stdio transport bridges the configured identity per process; HTTP headers remain per-request. Attribution always comes from the trusted identity: payload `agent_id`/`client` are ignored on write, the server no longer injects them, and backup replay preserves the original attribution. `memory_status` policy echo is reduced to whether the current caller is allowed.
- **Workspace alias decisions share one primitive** — record/install paths go through `_apply_alias_decision_on_conn` with the union of both guard sets (default-term refused in both directions, status enum, non-empty, self-pair no-op). Rejected pairs match by mechanical key across spellings, so a rejected `A→B` also blocks variants; confirmed decisions register and store the canonical orthography. `final_sync` publish failures now return `target_ready`/`needs_config_switch`/`target` instead of a bare "config switch failed". The self-mapping `sqlitevec` alias dead entry is removed. Qwen semantic-conflict preload now defaults on (fail-open).

### Fixed

- **Full-width IME default synonyms no longer register a phantom default pool** — `is_default_workspace_term` NFKC-normalizes before the synonym comparison, so `ｄｅｆａｕｌｔ`/`ＮＵＬＬ` land in the global pool like every other reserved term.
- **`write_transaction` no longer masks the original error when `BEGIN` itself fails** — the cleanup rolls back only when a transaction is active, so a busy-timeout failure reports `database is locked` instead of `cannot rollback`.
- **Confirmed-alias lookups pick the same winner as the resolver** — `updated_at DESC, canonical ASC` ordering is now shared, so drift states with two confirmed rows cannot split move and write destinations.
- **Id fields reject values above the SQLite integer range** with a structured `invalid_input` instead of an `OverflowError`, and `new_workspace` is bounded like every other workspace field.

### Verification

- Full tests (875), strict mypy, Ruff, compileall, release build, isolated wheel smoke, and adversarial PoC re-runs pass; remote main CI verified green on the release SHA.

## [0.14.6] — 2026-08-24

### Fixed

- **Status-only memory updates keep evidence versions current** — when governance changes source/trust/status/entity semantics and increments the authoritative memory version without changing evidence text, existing evidence rows now advance to that version in the same transaction. Supersede/retire no longer leaves false stale-evidence warnings; vec0 `parent_status` continues to update as before.

### Verification

- Full tests, strict mypy, Ruff, compileall, dependency audit, release build, isolated wheel smoke, production smoke, and deep production database checks pass.

## [0.14.5] — 2026-08-24

### Fixed

- **New workspace registration is visible outside strict mode** — the first successful `none`/`weak` write that registers a canonical now returns a non-blocking top-level `workspace_review` notice plus the existing write hint. The notice points to doctor review and requires fresh user authorization before `confirm_workspaces`; repeated writes and strict pending-workspace flows do not duplicate it.
- **Local hidden state is excluded from source control and distributions** — repository ignore rules now cover local agent/editor directories and all non-example `.env.*` files, while the source manifest explicitly prunes those directories and OS metadata. Required project files such as `.github/workflows`, `.gitignore`, and the public `.env.example` remain tracked or packaged as appropriate.
- **Production smoke data uses one recognizable workspace** — `mema-production-smoke` now always writes temporary records to canonical workspace `测试`; version/run identity remains in subject, tags, and source reference. Smoke runs no longer create ambiguous release-specific workspace names.

### Verification

- Full tests, strict mypy, Ruff, compileall, dependency audit, sdist/wheel inspection, Twine checks, isolated wheel smoke, workspace-notice behavior, and production database health checks pass.

## [0.14.4] — 2026-08-24

### Changed

- **Schema and vector compatibility are independent** — every migration declares `vector_effect=preserve|rebuild`. Preserve migrations no longer load a model, validate full vector coverage, or switch to a full rebuild merely because the configured embedding space differs; preserved incompatible vectors are marked `mismatch` and remain disabled until the existing rebuild workflow repairs them.
- **Startup generation gate is constant-time** — current databases use one bounded `migration_state` lookup and skip schema DDL, table enumeration, vector scans, row counts, and integrity checks. Pre-generation table inspection remains available only to the low-frequency upgrade/doctor path.
- **Migration receipts are compact and atomic** — successful migrations publish `schema_generation` and `migration_completed_at` in the final transaction and remove temporary `phase`, cursor, coverage, count, and verification receipts. Deep doctor owns `quick_check`, schema-state, vector-space/dimension, coverage, and bidirectional orphan checks.
- **Restart-safe localhost HTTP default** — Streamable HTTP now defaults to stateless request handling because product state and semantic notices are SQLite-backed rather than MCP-session-backed. Explicit `mcp.http.stateless=false` remains available for clients that require server-side sessions or server-initiated SSE messages.

### Fixed

- **Vector-space mismatch is fail-closed across every normal path** — lazy model loading now validates the configured embedding space before query, evidence write, workspace-vector publication, placement hints, or conflict KNN. Ordinary calls cannot mix target-space vectors into an old index; only an explicit rebuild with a persisted rebuild epoch may republish them.
- **Derived-table loss has an executable recovery path** — deep doctor reports missing/unqueryable vec0 tables without treating errors as zero healthy rows, `rebuild_evidence` dry-run previews the missing tables, and authorized execution recreates only the derived tables before a full evidence/workspace-vector rebuild.
- **Migration publication and recovery stay crash-safe** — successful migrations atomically publish generation plus one completion timestamp, retain staging ownership, reject unknown phases, and leave failed/checkpoint-incomplete targets unopenable. Obsolete success receipts no longer accumulate.
- **Three independent review rounds** — a normal review, a function-oriented review, and an adversarial review fixed startup maintenance work, doctor false health, staging ownership cleanup, lazy-space write contamination, and explicit vector-table recovery.

### Verification

- 777 tests pass in the vec-equipped development environment before release metadata updates; strict mypy, Ruff, compileall, dependency audit, migration failure-injection tests, real stateless HTTP smoke, deep production-index checks, sdist/wheel build, and Twine checks pass.

## [0.14.3] — 2026-08-24

### Fixed

- **Upgrade evidence reuse is now space-safe** — side-by-side upgrade reuses a previous generation's evidence/vector index only when its state is `ready`, its active embedding-space ID and vec0 dimensions match the configured model/pipeline, and evidence plus workspace-canonical vector coverage is complete and current. Otherwise it performs a clean full rebuild, and publication requires the target to be `ready` in the actual runtime embedder space.
- **Space recovery rebuilds every derived vector family** — `rebuild_evidence` now atomically replaces workspace-canonical vectors for the target space, removes obsolete deleted-memory evidence, verifies one vector per evidence unit, and refuses the final `ready` transition while any derived index is incomplete. Resuming a failed full rebuild under a different embedding space clears partial derived state and starts again.
- **Claude localhost HTTP setup documented** — README examples show the tested pinned `mcp-remote` bridge for Claude Desktop/Cowork and Claude Code, including fixed identity headers and removal of duplicate direct-stdio server entries.

### Verification

- 769 tests pass locally in the vec-equipped environment; strict mypy, Ruff, compileall, pip-audit, version synchronization, build, and production-index recovery checks pass.

## [0.14.2] — 2026-08-23

Independent whole-project adversarial hardening release. No schema generation change is required; 0.14.1 databases remain current.

### Fixed

- **Lifecycle mutations are transactionally authoritative** — memory confirmation and pending activation now re-read status and strict workspace visibility under the same `BEGIN IMMEDIATE` transaction as their update, preventing a concurrent retirement or workspace move from being overwritten by a stale preflight snapshot. Strict content/tag edits, entity assignment, retirement, and per-memory history cleanup likewise re-check workspace access inside their write transaction.
- **Workspace moves retain conflict ownership** — canonical rename/migration now moves unified conflict and notice rows in the same transaction as their member memories. A target-workspace slot collision rejects and rolls back the entire move instead of leaving conflicts invisible under the old canonical.
- **HTTP body limits cover chunked requests** — Streamable HTTP now bounds the actual accumulated POST/PUT/PATCH body before entering FastMCP, so omitting `Content-Length` cannot bypass `mcp.http.max_request_body_size`; GET/SSE requests remain streaming and are not pre-read.
- **Structured conflict inputs are bounded and typed** — conflict members, value groups, candidate keys, slot keys, and judge/replan apply plans now have matching item/JSON-size bounds at the product and DB layers. Non-object entries return structured validation outcomes instead of raw Python or SQLite exceptions.
- **Direct-call validation parity** — the authoritative `memory_write` path now applies the same input contract as the product wrapper, and direct `memory_confirm` rejects boolean, non-finite, and out-of-range confidence values. Persisted conflict text/version fields and timestamps are bounded before storage.
- **Console workspace views stay coherent** — in `isolation=none`, an explicitly selected workspace now scopes recent browse, overview counters, audit buckets, conflict lists, and conflict detail to the same canonical workspace; `weak` remains whole-library soft ranking. Audit aggregation uses canonical rather than raw workspace names.
- **Retirement rejects self-replacement** — `superseded_by` must identify a different active memory, preventing a logically impossible self-replacement operation from being accepted.

### Verification

- 758 tests pass locally in the vec-equipped environment; strict mypy, Ruff, compileall, pip-audit, build, twine check, and isolated wheel smoke pass after the adversarial fixes.

## [0.14.1] — 2026-08-22

Stable release restoring the Chinese documentation surface, closing findings from a fresh three-track adversarial review of the 0.14.1.dev0 push, and correcting semantic timeout scheduling before release.

### Added

- **Full Chinese documentation** — new `README.zh-CN.md` (plain-language, with diagrams, aimed at non-developer readers) and `docs/INTEGRATION.zh-CN.md` (precise contract mirror) with top-of-file cross-links in both directions. The bilingual mirror that the evidence-pipeline refactor dropped is back as standalone files so each language can be maintained independently. `AGENT_ONBOARDING.md` intentionally stays single-file English: it is packaged and served verbatim to agents via `help(topic="agent_onboarding")` with no language negotiation, and duplicating it would double every injection's token cost.

### Fixed

- **Semantic timing semantics** — `job_timeout_ms` now activates only when another semantic job is waiting, starts from the oldest pending job's enqueue time, and is checked between candidate pairs as a queue-fairness budget. An inference already started is governed solely by `inference_timeout_ms`; `notice_sync_wait_ms` remains a delivery wait and its default increases from 3000 to 5000 ms so more normal low-load checks can return their notice synchronously. Semantic drain now waits for both queued and in-flight work, and semantic status exposes the effective sync-wait value.
- **Notice resolve created zombie open conflicts (blocker, regression from d3729ce)** — `update_semantic_notice_status(..., "resolved")` wrote the *decoded* status into `conflicts.status`; decoding maps delivery pending/delivered to `open`, so every resolved notice became a formal open conflict (occupying the per-slot unique index; a subsequent formal group on that slot crashed with a raw `sqlite3.IntegrityError`). Resolved notices now keep their raw terminal `candidate` status; the UPDATE is guarded by a structured `slot_occupied` outcome (defensive — unreachable once status no longer flips to open), and regression tests pin status-keeping plus "a later formal group on the same slot records without a raw IntegrityError."
- **Scan epoch wedge (medium)** — a completed `conflict_scan_progress` row surviving into a fresh epoch (side-by-side migration copies `migration_state`) made `record_conflict_scan_page` reject every page of the new epoch, wedging `conflict_scan_required` until the next memory write. Progress from a superseded epoch is now treated as absent, and both epoch-stamping migration paths delete the stale progress row.
- **Grounding-failed apply left a silent orphaned edit (medium)** — `apply_conflict_action(update_current_claim/use_as_resolution)` commits the content edit together with failure bookkeeping by design, but the old evidence rows were already deleted and no re-index was enqueued. The failed-but-committed path now re-enqueues untrusted evidence indexing and marks the step/response with `orphaned_edit: true` so a replan accounts for the changed member.
- **CI: vec-less Core jobs red on 0.14.1.dev0** — four vec-dependent tests in `tests/test_workspace_default_insulation.py` queried `workspace_canonicals_vec` without the `sqlite-vec not installed` skip guard used by sibling files; they now skip cleanly when sqlite-vec is absent.
- **HTTP identity fail-closed hardening** — `_identity_for_tool` no longer swallows an `IdentityHeaderError` from the current MCP request and falls back to the session ContextVar (a potentially stale initialize-request identity). Headers present but invalid now surface as an error; the ContextVar fallback only applies when no request/headers exist (stdio). Unreachable with mcp 1.29 today; closes the fail-open direction against future SDK divergence.
- **Strict isolation placement-hint leak** — the default-workspace placement suggestion ran an unscoped evidence KNN and returned a foreign workspace's canonical name and memory id to strict callers whose read ACL hides both. The hint is now skipped under `isolation="strict"` (it is a none/weak convenience).
- **Scan enhancement fail-open hardening** — a backend whose `classify_pair` raises (not merely returns an invalid signal) no longer escapes `_enhance_scan_candidates` and discards the deterministic baseline; per-pair failures degrade to `error` and the page continues.
- **Judge `stale_member` guidance** — the note recommended retrying judge "with a plan pinned to the members' current versions", which deterministically fails because judge pins versions from the recorded group snapshot. It now directs the caller to register the new member version via `record_conflict` first (or plan around the member).
- **Config edge** — `mcp.http.path="//"` passed validation then `rstrip("/")` collapsed to an unservable empty route; an all-slash path now warns and resets to `/mcp`.
- **Documentation corrections** — README's upgrade publication gate no longer claims complete eligible evidence coverage for both paths (only the full-rebuild path requires it; conflict-only clones evidence unchanged); INTRO.md's workspace-governance list now includes full-registry confirmation (`confirm_workspaces`); the Integration Guide uses the exact `alias|typo|same_project` enum and notes that read-classified notice surfaces (list/read, `memory(status)` delivery) still advance `pending → delivered`/`stale` state, including for policy-denied HTTP identities.
- **Dead code** — removed the uncalled duplicate `_agent_onboarding_guide`/`AGENT_ONBOARDING_TOPIC` from `tools.py` (the live copy is `surfaces.py`).

### Reviewed, no change

- `confirm_pending_workspace` under strict without an explicit `workspace` skips caller binding by design: the action is user-authorized per call, its documented purpose is assigning (possibly remapping) the pending memory's canonical with an internal redirect, and pre-existing strict tests pin deliberate remap success. Caller binding applies when a workspace is asserted.
- Deep-verify that a resolved conflict's final member states all match the chosen value remains a known design gap (orphaned edits are now re-indexed and flagged, but `resolve_conflict` does not re-ground members); deferred as a design decision, not a silent fix.

### Verification

- 737 tests pass locally in the vec-equipped environment; strict mypy, Ruff, compileall, and `git diff --check` pass; `scripts/sync_version.py --check` passes.

## [0.14.1.dev0] — 2026-08-22

Development self-test build for one localhost Streamable HTTP MCP server shared by multiple local AI clients. It retains stdio as the default and treats request headers as local provenance/policy input, not authentication or tenant isolation.

### Added

- **Local Streamable HTTP MCP transport** — opt-in `mcp.transport=streamable-http` serves the four product tools at `http://127.0.0.1:8000/mcp`; host, port, path, body bound, stateful/stateless mode, and JSON-response mode are configurable. Community mode refuses non-loopback binds and hides non-MCP routes.
- **Per-request advisory identity** — every HTTP MCP request requires fixed `X-Mema-Client` and `X-Mema-Agent-Id` headers. Missing, empty, duplicate, malformed, overlong, or tool-data-conflicting identities fail closed; stdio keeps its existing Settings/env identity behavior.
- **HTTP client example and integration contract** — setup/config templates, registry descriptors, `.env.example`, README, Integration Guide, and `examples/streamable-http.mcp.json` document fixed client headers and the Community-vs-Team trust boundary.

### Fixed

- **Stateful-session identity isolation** — each tool call reads the current MCP request headers before the inherited ContextVar and re-enters a request identity scope, so an initialization request cannot pin later writes, status, notices, or policy checks to stale identity.
- **Mutation policy coverage** — denied HTTP identities cannot mutate through update/judge, governance, repair, or persistent conflict-scan progress. Read-only status/review, notice list/read, semantic status, and evidence/backup dry-runs remain available.
- **HTTP boundary hardening** — validates loopback peer and Host, exact MCP path, duplicate headers, configured request-body size, and current Header/body identity consistency. `memory(status)` and onboarding notice delivery use the effective request identity.
- **MCP SDK compatibility** — raises and locks the supported SDK floor to `mcp>=1.29.0,<2`, the tested line used by the HTTP request context and body-limit implementation.
- **Two adversarial review rounds** — fixed stale stateful identity, mutation-policy fail-open paths, SDK-range/runtime mismatch, over-broad repair denial, persistent `scan_candidates` policy bypass, and dry-run default/string normalization. Final re-review reported no confirmed findings.

### Verification

- Locked MCP 1.29 environment: 729 tests passed; strict mypy, Ruff, compileall, and `git diff --check` passed.
- Wheel build and Twine metadata check passed.
- Real Streamable HTTP MCP initialize/list-tools/call-tool smoke passed; missing headers returned 400 and valid headers produced the expected write/status identity.

## [0.14.0.dev5] — 2026-08-22

Implements phases 0–3 of the workspace recall + doctor full-confirmation plan (mema 721). Strict vector admission is opt-in and disabled by default.

### Added

- **Workspace decision-state simplification** — pairwise `accept_workspace_alias` / `reject_workspace_alias` are removed from product help and mutation dispatch; old calls receive non-mutating, machine-readable tombstones with rename/migrate/pending-confirmation guidance. Durable old-name redirects and negative candidate suppression remain internal. `workspace_aliases` is compacted to `(alias_workspace, canonical, status, updated_at)` and the unused `workspace_alias_events` ledger is removed. The new `workspace_state_v1` generation forces side-by-side upgrade from `conflict_groups_v2`, preventing old binaries from opening the compact schema as current.
- **Strict vector admission (default off)** — `MEMORY_ARBITER_WORKSPACE_RECALL_ADMISSION=true` widens strict isolation from an exact same-canonical filter to an admitted canonical set: the caller's own canonical plus any within `workspace_recall_cutoff` (default 0.25), computed once per call by `admitted_canonicals` through the same shared guards as the weak curve (default-pool insulation, `workspace_min_name_len`, substring/generic-only proximity). Recall and the read ACL always consume the SAME set — searchable implies readable, with no "searchable but unreadable" state. Scoping happens in SQL at every site (all five wide-recall channels, the bm25 path, recent fallback, filter recall/COUNT, evidence KNN and its over-fetch loop, linked-open-items df, memory-by-id, recent list, entity/audit aggregates, console status/browse), so totals, pagination, and document frequencies stay consistent instead of being post-filtered after a page was already cut. Conflict/notice authorization widens on the caller side only: the group's own `workspace_canonical` still binds all of its members. The admitted set always contains the caller's canonical, so with the flag off every scope collapses to a single-element `= ?` clause — byte-identical to v0.12.5.
- **doctor check `workspace.review` (721 期2)** — diffs `workspace_canonicals` (reserved default terms excluded) against the `workspace_review.json` sidecar in a one-way diff (new canonicals surface; disappeared names are ignored). Missing/corrupt sidecar degrades to a first full review without raising; WARNING severity only, never critical. Any unconfirmed workspace makes this finding a warning (doctor CLI exit 1 unless a more severe finding exists). A full-registry `confirm_workspaces` call returns this check to pass; unrelated warnings may still keep the overall CLI at exit 1. The first run on an existing library reports the whole registry unconfirmed. Read-only: doctor never refreshes the snapshot. Migration note for libraries written by ≤0.14.0.dev4: default synonyms (`默认`/`none`/`null`/`unknown`/`未知`) previously registered as their own phantom canonicals; after this change reads scoped to such a synonym resolve to `default`, and no batch migration is shipped in this dev cycle — rewrite affected memories (read them unscoped, then re-write under the intended workspace) if your library has such pools.
- **Governance action `confirm_workspaces` (721 期2)** — authorized-only action that writes the `workspace_review.json` snapshot (`{"confirmed_workspaces": [...], "confirmed_at": ..., "version": 1}`) atomically (tmp + rename). Defaults to snapshotting the current registry (run it after rename merges); an explicit `workspaces` list overrides (bounded: ≤100 items × 2000 chars). Wired through all eight product-surface points (`_GOVERNANCE_IMPACTS`, help actions/examples/confirm_actions, dispatch branch, `PRODUCT_FIELD_REGISTRY`, pipeline implementation, tools forwarder).
- **Weak-isolation continuous vector weighting (721 期1, default off)** — `MEMORY_ARBITER_WORKSPACE_WEAK_VECTOR_WEIGHT=true` makes the weak workspace nudge continuous: full +0.30 inside cosine 0.15, linear decay to 0 at 0.30, 0 beyond (a known-far workspace no longer eats the -0.15 penalty). Distances come from a new read-only `canonical_distance_map` precomputed once per search pool (721 §2b) — scoring leaves stay pure. Off = exact v0.9.7 binary behaviour.
- **Shared vector-admission helpers (721 §2a)** — `workspace_vector_distance`/`workspace_admit`/`weak_workspace_vector_weight` in `workspace_rules`, with the short-name guard (`MEMORY_ARBITER_WORKSPACE_MIN_NAME_LEN`, default 3) and the substring/generic-token proximity guard (`main`↔`openclaw-main` at 0.132 no longer admits).

### Fixed

- **default pool double insulation (721 期0)** — the reserved default synonyms (`""`, `default`, `默认`, `none`, `null`, `unknown`, `未知`; single source `constants.DEFAULT_TERMS`) are now bidirectionally insulated from workspace matching: the resolver folds every synonym to the one `default` canonical; KNN excludes them; canonical-vector publication skips them; and rename/migrate/internal decision state refuses any pair touching default.
- `memory_govern(rename_workspace_canonical)` now reports `renamed: false` (it previously echoed `renamed: true` alongside a failing `ok=false`).
- **Round-1 adversarial review fixes** — `canonical_distance_map` no longer crashes the search path when sqlite-vec returns a NULL cosine distance for a degenerate vector (all-zero/NaN): the canonical is treated as vectorless and its records fall back to the binary weighting step; `confirm_pending_workspace` no longer dead-ends on a pending memory whose raw workspace is a reserved default synonym (the synonym is already the global pool, so no alias is recorded); the `confirm_workspaces` payload is bounded (≤100 items × 2000 chars, enforced in validation and the pipeline) and the snapshot is written atomically (tmp + rename) so a concurrent doctor read cannot observe a torn file.
- **Round-2 adversarial review fixes** — `confirm_pending_workspace` folds a default-term `canonical` argument to the canonical `default` spelling (a synonym could previously be re-persisted as a phantom canonical that scoped reads never match); `confirm_workspaces` uses a unique tmp name per write (concurrent calls on a shared library can no longer collide on the fixed `.tmp` path), persists the accepted `reason` into the snapshot, and refuses non-list/non-string `workspaces` on direct pipeline calls instead of iterating a bare string into single-character names; `INTRO.md`/`docs/INTEGRATION.md` doc-contract versions and `examples/memory-arbiter.config.example.json` (new config keys) updated for dev5.
- **Workspace-state adversarial review fixes** — schema compaction now uses the distinct `workspace_state_v1` generation so older binaries refuse it; partial/malformed decision tables fail closed without data loss; interrupted legacy compaction resumes safely; rename/migrate reject competing moves after another transaction already forwarded the source; strict unconfirmed retries remain pending without registering canonical/vector state; case/mechanical confirmation reuses one spelling; the default pool cannot be confirmed into a project; old action tombstones return mapped, machine-readable replacement calls.
- **Phase-3 adversarial reviews (rounds 1–2)** — fixed strict `rebuild_evidence` tuple binding and admitted-scope discovery; scoped conflict pagination before LIMIT; preserved empty audit buckets and attributed conflicts to their owning workspace; required semantic-notice members to remain in one authorized workspace; made admitted-neighbor notice delivery, conflict recording, generated conflict/scan follow-up calls, and console status directly executable with the caller workspace; removed the 20-neighbor admission truncation and the 2,048-row evidence-KNN starvation cap; filled notice pages after member filtering and refreshed stale lifecycle in counts; rejected non-finite recall cutoffs; reused one caller scope throughout each product response.
- **Documentation adversarial review (round 3)** — aligned README/INTRO/Integration/help/docstrings with optional admitted-set strict visibility and workspace-review exit semantics; added all new workspace controls to setup/config descriptors/status; corrected the dev5 phase summary; and replaced the duplicated 7.5 KB agent onboarding manual with a compact rule that delegates uncommon procedures to tool help.

## [0.14.0.dev4] — 2026-08-21

Second independent adversarial review round on the dev3 fixes. Closes 2 blockers and several regressions the dev3 changes introduced.

### Fixed

- **Scan reverse-only value mapping (blocker)** — in the new scan Qwen enhancement, a single-direction (B→A) extraction stamped each member with its peer's value because `gate.value_a/value_b` follow the surviving extraction's own input order while members are ordered left/right. Each member is now grounded to its own side's display value, and `value_groups`/`slot_groups` use the corrected per-side values.
- **Detector-version wedge (blocker)** — the dev3 rename of `CONFLICT_DETECTOR_VERSION` to `attribute-value-v1`, combined with the running-vs-persisted gate comparison, could permanently wedge `conflict_scan_required=true` on a database upgraded under dev2 (which persisted the old string). `rearm_conflict_scan_if_drifted` now also treats a persisted-vs-running detector mismatch as drift and re-stamps `conflict_scan_detector_version`, so one full scan clears the flag.
- **record_conflict append IntegrityError (high)** — the open-group append now wraps its CAS UPDATE and returns a structured `duplicate_event` when the event-snapshot index collides with a resolved/not_a_conflict row recorded under a different detector, instead of leaking a raw exception through the tool layer.
- **none-mode explicit filter truncation (high)** — an explicit none-mode workspace filter is now pushed into the recall SQL (`hard_scope`) so the limit applies after scoping; it previously post-filtered the already-limited page and could return empty when the target workspace ranked below the cutoff. weak mode still never hard-filters.
- **needs_authorization livelock (high)** — a `needs_authorization` apply step is now recorded `blocked` (committed) and routed to `replan_conflict` by both guidance surfaces, instead of raising forever with no machine-executable exit.
- **Applying re-entry over-broad suppression** — suppression is now strictly slot-scoped and applied only after the gate resolves the candidate's `slot_key` (with §15.3 revision/action/target preconditions on the trusted context); a different conflict between two co-members of an applying group is examined and surfaced, and an unexamined skip can no longer report `checked_no_notice`.
- **Normalization edge cases** — `_mechanical_normalize` preserves a decimal point between digits (`1.5s` ≠ `15s`) while still compacting size units (`8GB` = `8G`) without corrupting non-numeric tokens (`dotnet` intact); the `__unknown__` sentinel check is case-insensitive; a prose-prefixed top-level array is rejected; the coexistence evolution veto covers the 变为/变更为/调整为/更新为 replacement-verb family.
- **workspace_decision_reason** — admission/inference deadline errors (which say "deadline expired", not "timeout") and other backend errors now map to `qwen_timeout`/`qwen_backend_error` instead of a fabricated `qwen_low_conf`.
- **slot_groups value disagreement** — when one memory extracts different values across two same-slot pairs, the emitted slot group is flagged `value_conflict`/`route=review_candidate` instead of an un-recordable payload.
- **apply response history** — `apply_conflict_action` returns the persisted `apply_summary` (including `history`) instead of a hardcoded `{"plan": ...}`.
- **Docs/tests** — removed a duplicated README sentence; replaced two `or True` tautological assertions and a misleading escalate test with real coverage of the `linked`/`applying_group_exists` outcomes; added regression tests for both blockers and the high/medium fixes.

## [0.14.0.dev3] — 2026-08-21

Post-review hardening of the dev2 baseline: closes the findings of an independent adversarial review round (conflict store CAS paths, applying re-entry, scan wide gate, upgrade scan gate, workspace contracts, and help/doc truthfulness).

### Fixed

- **Applying re-entry suppression** — post-commit semantic jobs for versions produced by an applying conflict's plan (and pairs against members of an applying group, or landing on its slot) no longer create a new user notice for the same conflict; unrelated third-party facts still surface via scan review. The `trusted_applying_context` plumbed from `apply_conflict_action` is now revalidated against the live conflict row and consumed.
- **`checked_no_notice` reachability** — definitive strict-gate negatives (clean model decisions, deterministic coexistence vetoes, `__unknown__` "cannot extract" responses) now complete the bounded check instead of reporting `status=incomplete`; only technical failures, budget exhaustion, grounding-uncertain (`qwen_unverified`), and provenance-insufficient outcomes stay incomplete. `check_degradation` counters no longer record clean model decisions as degradation.
- **Conflict store CAS paths** — `judge` pins the stored member entry matching the memory's CURRENT version (an open group holding several versions of one memory is judgeable instead of permanently `stale_member`); notice `escalate` appends/links into an existing open group at the same slot instead of raising `IntegrityError` (new outcomes `appended`/`linked`; `applying_group_exists` when the group is mid-application); re-recording under a changed detector/candidate identity returns structured `duplicate_event` instead of raising; appends compute the row's `candidate_key` over the combined member snapshot; `record_conflict(status="not_a_conflict")` against an occupied slot returns `open_group_exists` instead of silently appending; a single new member/value group can be appended to an open group (the two-value-group rule now applies to creation only); apply steps preserve `apply_summary.history` written by `replan_conflict`; failed steps with nothing pending now suggest authorized `replan_conflict` (detail/list surfaces and next-call computation) instead of a `resolve_conflict` that would fail `apply_incomplete`; `stale_conflict`/`stale_member` responses carry re-read/replan guidance.
- **`__unknown__` sentinel gating** — the model's protocol-legal "cannot reliably extract" marker is treated as extraction failure for every gate (it can no longer pass as a slot attribute or reach a user notice), and `_normalize_slot` rejects `unknown`/`__unknown__` entity/attribute/scope values. Top-level array-wrapped model output is rejected; `gb/kb/mb/tb` unit spellings normalize equal at the post-gate; the coexistence evolution veto no longer triggers on the bare character `由`.
- **Scan wide gate (spec §7.1)** — `scan_candidates` now runs a bounded per-page Qwen enhancement when `semantic_conflict.scan_enhance=true` (default): rule candidates are enriched with attribute/value member fields and `value_groups`, similarity-only pairs that extract a legal same-attribute-different-value in either direction are unioned into `candidates`, and verified candidates with matching metadata entity/scope are aggregated into `slot_groups`. `scan_max_pairs` (default 8, 0 disables) and `scan_budget_ms` (default 60000) bound the cost; backend absence/failure leaves the deterministic candidate set unchanged.
- **Upgrade scan gate** — the clearing gate compares the persisted requirement against the RUNNING detector identity (one shared `CONFLICT_DETECTOR_VERSION`, also stamped on scan candidates), so an old-detector scan can no longer echo-clear the flag. A memory write between upgrade and scan completion re-arms the epoch against the current live boundary (`conflict_scan_rearmed` in the response) instead of wedging `conflict_scan_required=true` forever. A conflict-only rebuild whose post-transaction validation fails now stamps `phase=failed` like the full-rebuild path.
- **Workspace contracts** — an exact-match write repairs a missing canonical vector (the advertised "retry a write using this workspace" now works); `workspace_decision_reason` distinguishes `qwen_unavailable`/`no_similar_candidates`/`qwen_timeout`/`qwen_rejected_candidate`/`qwen_unrelated` from genuine `qwen_low_conf`; a none-mode read with an EXPLICITLY passed workspace canonicalizes and scopes that query (spec §15.6) while omitted workspace still spans all workspaces; the search loud `attention_required` flag fires again for formally recorded groups (`conflict_source="conflict_group"`).
- **API surface** — the dead legacy wrapper `memory_record_conflict` returns a structured removal error instead of recursing; `action_required_paths` documents `judge_conflict` and `replan_conflict`; the `ask_user` text no longer contradicts `decided_by=[user,agent]`; the judge help marks `resolution_memory_id` as required consistently; the `_memory_govern` docstring drops the removed "correct judgments" wording; the semantic degradation note states that write-time notices (deterministic notify pairs included) fail closed while Qwen is unavailable.
- **Docs/notice text** — README `mema setup` template description, INTEGRATION span-read claim, scan-gate description, onboarding none-mode/record_conflict guidance, blog compatibility-profile sentence, and console conflict-detail captions now match the implementation.

### Added

- `semantic_conflict.max_evidence_units` (default 24, 1–256) bounds evidence units examined per write-time check; the gathering loop also enforces the job deadline, and pair inference receives the remaining job budget as its deadline.

## [0.14.0.dev2] — 2026-08-20

Development state implementing the memory 715 conflict-group and workspace-normalization baseline. This is not a compatibility release.

### Breaking

- **Conflict history is rebuilt, not migrated** — both previous `local_text_evidence_v1` and older derived-index generations are refused at startup and use side-by-side `mema upgrade`. Upgrade in an exclusive window after a WAL-safe source checkpoint/backup, stopping old writers, and draining/stopping the semantic worker. The target omits old `conflicts`, `conflict_judgments`, and `semantic_notices`; memory content/history, workspace/audit/backup state, and evidence remain. The previous evidence generation reuses FTS/evidence/vector tables unchanged and rebuilds only the conflict domain; older generations rebuild evidence vectors. `--yes` skips only the destructive-loss confirmation, not these prerequisites.
- **One conflicts table and lifecycle** — a row now represents one one-to-many event and stores immutable detection snapshots, value groups, the final decision, and application results. Public states are `open`, `applying`, `resolved`, and `not_a_conflict`; old pair judgment/correction wording and public `notify/check/ignore` routing are no longer the contract.

### Added

- **Required post-upgrade scan epoch** — successful rebuild sets persistent `conflict_scan_required=true` and `conflict_scan_epoch`. Only a complete scan covering the upgrade-time active-memory set with the matching detector version may CAS-clear the flag; partial, failed, or old-detector scans cannot.
- **Bidirectional four-field Qwen extraction** — short candidate pairs run A→B and B→A extraction of exactly `attribute_a`, `value_a`, `attribute_b`, and `value_b`. Strict validation checks side mapping, mechanical normalization, quote grounding, complete slot provenance, and deterministic coexistence vetoes. Qwen has no winner-selection, scan-veto, grouping, or mutation authority.
- **Broad scan and strict notices** — scheduled scan unions deterministic KNN/rule candidates with Qwen enhancement and retains uncertain cases as `review_candidate`; write-time notices require two consistent grounded directions and a complete slot. Model failure closes the notice path without shrinking scan recall.
- **CAS application protocol** — `memory(action="judge")` moves an `open` group to `applying` and returns a per-member plan. Authorized `memory_govern(action="apply_conflict_action")` applies one step at a time with the latest revision; authorized `resolve_conflict` closes the group only after all steps succeed.
- **Evidence-scoped reads** — `memory(action="read")` accepts a strict `{start,end}` character span and returns clipped `data.memory.content` plus `data.span`; scan/notice deep-read calls can carry these source windows while full read remains available by omitting span.
- **Conflict replanning** — a failed apply step returns `data.action_required=replan_conflict`; authorized `memory_govern(action="replan_conflict")` CAS-replaces the remaining plan while preserving prior plan history.
- **Delivery and workspace budgets** — `semantic_conflict.notice_sync_wait_ms` defaults to 3000 ms (0–5000) and only controls sync-versus-async notice delivery. An accepted timed-out task continues; queue-full means no accepted task. `semantic_conflict.workspace_qwen_budget_ms` defaults to 750 ms (50–5000) and bounds near-match workspace suggestions independently.

### Changed

- **Workspace normalization is orthogonal to ACL** — `none`, `weak`, and `strict` all write canonical workspace results. `none` performs no workspace ACL and unscoped reads remain whole-library; `weak` adds soft ranking; `strict` forbids silent Qwen merges and keeps new workspaces pending. Automatic vector/Qwen results do not create confirmed aliases.
- **Scan triage persistence is explicit** — `scan_candidates` returns internal `review_candidate`/`notice_ready` states. Agents must call `record_conflict(status="open"|"not_a_conflict")` for each triaged snapshot to gain dedupe semantics. Process-local queue loss/full/restart is recovered by checking coverage, rebuilding evidence to ready, then scanning.
- **Response paths are explicit** — product operation results and `action_required` live under `data`; successful delivery side channels use top-level `notices`, where each notice carries its own action/read call. No generic top-level operation `action_required` is promised.

## [0.13.1] — 2026-08-17

### Changed

- **Write-time semantic candidate quality tightened** — successful writes enqueue Qwen classification asynchronously; strict model-output validation now fails closed, and semantic candidates use specific bounded candidate recall with subject/tag ranking before `pair_limit`, suppressing noisy/common tags while retaining specific tags and subject fallback. The deterministic pair-text gate remains authoritative: `medium` is the default, while `strong` is the conservative option configurable through `semantic_conflict.pair_text_gate` or `MEMORY_ARBITER_SEMANTIC_CONFLICT_GATE`.
- **Semantic notice review instructions completed** — after `notice/read`, Agents execute the returned `left_read_call` and `right_read_call`, read both full memories, and only then tell the user when the advisory candidate appears credible; false positives can be dismissed and already-handled notices resolved.
- **Notice lifecycle and transport boundaries clarified** — attaching a stub performs the `open → open + delivered_at` state transition; `dismiss` and `resolve` are the only public terminal transitions, while stale undelivered snapshots may transition internally to `stale`. Delivery does not create a conflict, submit a judgment, edit a memory, or supersede either side. The database claim is atomic best effort and does not promise transport-level exactly-once delivery.

### Fixed

- **Restored 0.13.0 semantic notice delivery** — the next successful response from any of the four product tools again attaches at most one compact semantic stub when an eligible open notice exists, while update/onboarding/backup notices can coexist.

## [0.13.0] — 2026-08-16

Release hardening for bounded backup replay, strict product validation, subprocess-isolated local semantic inference, governance safety, startup diagnostics, and reproducible publishing.

### Added

- **Bounded, resumable backup replay** — replay reads oversized JSONL records without materializing unbounded lines, rejects unsupported legacy/invalid envelopes and duplicate replay keys, caps mutating pages, reports processed/remaining state, and records post-processing status so interrupted imports can safely resume claim/vector/section work.
- **Semantic subprocess hard timeouts** — local GGUF load and inference run behind a single-flight subprocess boundary with configurable load/inference deadlines, forced termination/restart on timeout, admission gates during disable/shutdown, and observable timeout/restart status.
- **Release artifact smoke coverage** — CI and publish workflows now install the built wheel in a clean environment and verify package import/version, all console entry points, and packaged `AGENT_ONBOARDING.md` data.

### Changed

- **Product input validation is fail-closed for protected fields** — product surfaces strip harmless unknown fields with a warning, reject likely misspellings of protected fields with `did_you_mean`, normalize controlled numeric strings before dispatch, enforce bounded text/revision/offset values, validate section modes, and require finite bounded semantic-control timeouts.
- **Workspace/governance replay behavior is safer** — replay no longer treats registry presence alone as workspace confirmation, keeps pending workspace visibility rules, and exposes post-processing warnings rather than silently presenting a partially indexed import as complete.
- **Startup and shutdown observability improved** — backup replay notice failures degrade to an explicit warning and suggested inspection call; semantic disable/shutdown closes synchronous and worker admissions before draining in-flight work.
- **Release consistency checks expanded** — `sync_version.py --check` is read-only and validates `server.json`, the newest CHANGELOG release, and the editable project version/extras in `uv.lock` against the authoritative package version.
- **CI/publish hardening** — the existing Python 3.11/3.12/3.13 core matrix and sqlite-vec job remain; quality builds and smokes the wheel, release tags must equal the package version, and manual publishing no longer defaults to an obsolete tag.

### Fixed

- Backup replay now detects incomplete post-processing, safely retries already-imported receipts, limits live replay batches, preserves workspace canonicalization, and reports degraded startup inspection instead of failing invisibly.
- The strict mypy baseline check now invokes mypy with the running `sys.executable`, checks its return code, and requires a parseable summary consistent with emitted errors, preventing missing/crashed mypy runs from passing green.
- Semantic runtime disable/unload/shutdown races no longer admit new synchronous workspace suggestions while another inference is draining or a timed-out child is being replaced.

### Compatibility and limitations

- MCP product tool names and the legacy/full compatibility profile remain available; valid numeric strings continue to coerce where documented. Harmless unknown fields are stripped with warnings, while protected-field typos, oversized, non-finite, and out-of-range inputs fail validation instead of reaching storage/runtime code.
- The default install remains free of local-model binaries. `semantic-local` still requires `llama-cpp-python` plus a separately supplied GGUF model; because platform wheel/build availability is variable, installation is an observable manual/release check rather than a normal-CI green gate.
- Subprocess timeouts can terminate a stuck local-model call but cannot make inference deterministic or bundle model files. Backup replay is bounded and resumable, not atomic across an entire JSONL file; warnings require follow-up until post-processing reaches `complete`.
- Production smoke remains an explicit post-release/manual operation against the configured database and is not run by ordinary CI; it creates a uniquely marked record and attempts cleanup on every exit path.

## [0.12.5] — 2026-08-10

Strict workspace isolation, workspace-governance, and semantic-notice release hardening.

### Added

- Added adversarial regression coverage for strict cross-workspace mutating ID paths, pending workspace confirmation, structured-claim isolation, activation claim reindexing, arbitration cleanup, and semantic-notice claim pins.

### Changed

- Strict read ACL now consistently protects by-id mutation helpers when callers provide an explicit `workspace`, while legacy/global activation remains available for the existing low-level repair flow.
- Strict new-workspace guidance now points agents at `memory_govern(action="confirm_pending_workspace")` instead of the legacy `memory_activate` name.
- Semantic notice dedupe/closed checks now include `claim_revision` pins so claim-only metadata/entity changes can reopen semantic review without content-version changes.

### Fixed

- Prevented cross-workspace structured-claim collision scans from creating conflicts/notices under workspace isolation.
- Reindexed and reconciled structured claims immediately when pending memories are activated or confirmed, so active records are not left stale until a later rebuild.
- Blocked explicit-workspace cross-tenant writes for `memory_confirm`, `memory_supersede`, `memory_set_entity`, `memory_store_embedding`, and history cleanup.
- `memory_arbitrate(authorized=True)` now resolves open conflicts touching the loser in the same transaction as superseding it.

## [0.12.4] — 2026-08-09

Architecture baseline refactor for maintainability and release hardening. This patch keeps the MCP tool surface stable while splitting the former `tools.py`, `db.py`, and `doctor.py` God objects into focused pipelines, stores, and doctor check modules.

### Changed

- **MemoryTools facade slimmed** — `tools.py` now composes `ProductSurfaces`, `ReadPipeline`, `WritePipeline`, `OperationsPipeline`, `SectionPipeline`, `ConflictSignalPipeline`, and background workers. Legacy `memory_*` method signatures remain available on `MemoryTools` as thin delegates.
- **MemoryDB facade split into stores** — `memory_arbiter.db` is now a package. `MemoryDB` remains the connection/transaction authority and delegates to `SchemaStore`, `MemoriesStore`, `ConflictStore`, `VectorStore`, `WorkspaceStore`, `SectionStore`, `SemanticNoticeStore`, `AuditStore`, and `MetaStore`.
- **Doctor checks extracted** — health check implementations live in `doctor_checks/all_checks.py`, with shared data model/helpers in `doctor_checks/common.py`; `memory_arbiter.doctor` remains the public facade and preserves legacy private check names for tests/diagnostics.
- **Shared low-level helpers consolidated** — CJK/text/tag helpers moved to `text.py`, time parsing helpers to `timeutil.py`, and cross-module constants/isolation helpers to `constants.py`.

### Fixed

- Preserved legacy monkeypatch/diagnostic seams for `memory_arbiter.tools.search_memories`, `compare_memories`, `extract_claims`, and `_linked_open_items_for_search` after pipeline extraction.
- Fixed a doctor split circular import so `memory_arbiter.doctor_checks.all_checks` can be imported directly from a fresh interpreter.
- Added refactor safety tooling for symbol/signature snapshots, monkeypatch inventory, getattr attribution, and mypy no-new-errors checks.

### Compatibility notes

- MCP tools and `MemoryTools` method signatures remain compatible.
- `MemoryDB` no longer exposes the old one-line structured claim/judgment forwarding methods such as `publish_memory_claims` and `submit_conflict_judgment`; direct internal callers should use `db.claims.*` and `db.judgments.*`. This was an intentional narrowing of the DB God-object surface.

## [0.12.3] — 2026-08-09

Subject-required writes and console/search pagination hardening. This patch tightens the memory data contract, adds a guarded backfill helper for historical empty-subject rows, and fixes strict workspace isolation on expired recall.

### Added

- **Guarded subject backfill helper** — `scripts/backfill_subjects.py` uses the normal `memory_edit` path to backfill historical active rows that predate the subject requirement, preserving version/history/FTS/vector side effects. The built-in plan validates expected workspace and content hash before editing so integer-id collisions in another DB fail closed.
- **Active search offset support** — `memory_search` and the Console memories API now accept `offset` for best-effort query-recall pagination. Empty-query browse and expired recall continue to use exact SQL-backed pagination where available.

### Changed

- **Subject is now required on writes** — `memory_write` and `MemoryDB.insert_memory` reject missing or blank subjects; `memory_edit` also refuses to wipe a subject with an empty `new_subject`.
- Console memories pagination now exposes `total`, `total_precise`, and best-effort labels so the UI does not treat query-recall estimates as exact totals.
- SQLite WAL/SHM sidecar files are ignored by git.

### Fixed

- **Strict isolation on expired recall** — `memory_search_expired` now requires a workspace under `isolation=strict` and hard-filters by canonical workspace, preventing pending/superseded/conflicted records from leaking through expired/history recall or Console expired pages.
- Legacy MCP `memory_search` wrapper now forwards `offset` and its help text no longer claims active search has no offset cursor.

## [0.12.2] — 2026-08-09

Agent onboarding notice + guide. Additive notice/help/docs change only; no memory DB schema change and no breaking changes to existing tool calls.

### Added

- **One-time agent onboarding notice** — successful MCP responses can now include an `agent_onboarding` notice once per `MEMORY_ARBITER_AGENT_ID` and notice version (`agent-onboarding:v1`). The notice asks agents to persist a compact mema rule to their local agent memory file and points to the full guide. Suppression state is stored in the existing sidecar update state JSON, not in the memory database.
- **Agent onboarding help topic** — `memory(action="help", data={"topic": "agent_onboarding"})` returns the same guide content available in the package file.
- **Packaged guide file** — `memory_arbiter/AGENT_ONBOARDING.md` is included in the wheel and serves as the stable local source users can feed to agents that did not retain the onboarding rule.

### Changed

- README now links to the local guide file and GitHub copy instead of embedding the full agent instruction text.

## [0.12.1] — 2026-08-08

Workspace-alias governance + hardening on top of the existing workspace
canonicalization. Additive tool actions and internal tables only; no breaking
changes to existing behavior.

### Added

- **Workspace alias governance** — `memory_govern` gains `accept_workspace_alias`,
  `reject_workspace_alias`, `rename_workspace_canonical`, `migrate_workspace`, and
  `confirm_pending_workspace`. Backed by two new tables: `workspace_aliases`
  (current state) and `workspace_alias_events` (append-only audit); UNIQUE +
  single-transaction, no CAS. A user rejection is never silently reversed by any
  path — reversal requires an explicit `authorized=true`.
- **Rule-first workspace decision layer** (`workspace_rules.py`) — vector distance
  produces candidates; rules (`classify_workspace_quality` / `extract_evidence` /
  `rule_decision` → AUTO/KEEP/ASK/None) decide. Confirmed/rejected aliases and the
  three isolation modes retain veto power.
- **Local-model workspace candidate suggester** — reuses the existing GGUF backend
  to suggest normalization candidates; weak-mode silent merge only for
  identity-grade relations at high confidence; strict never silent-merges; the
  model never overrides confirmed/rejected/strict.
- **Doctor semantic/Qwen enablement chain** and a workspace-alias health check.

### Fixed

- Console memories page (empty query) now browses by recency instead of the
  multi-level safety-net sort that buried recent memories.
- `migrate_workspace`/`rename_workspace_canonical` keep the canonical registry and
  alias table consistent under rejection, case-only rename, and chained ops (no
  phantom canonical, no stranded/self-referential alias, forwarding preserved).
- Loosely-typed MCP JSON (non-string workspace values, string `"false"`
  authorization flags) is rejected with a structured error rather than crashing or
  silently granting an override.
- Model-supplied confidence is clamped to `[0,1]`.

## [0.12.0] — 2026-08-08

### Added

- **Write-time semantic conflict detection (off by default)** — new `semantic_conflict` module runs a local Qwen2.5-0.5B candidate signal after writes, gated by pair-text evidence (`medium` by default for balanced recall, `strong` for lower-noise/higher-confidence). Pipeline: metadata-overlap coarse recall → 0.5B pair classification → pair-text gate → `semantic_notices` row. The small model is only a recall signal; the gate has veto power, so a duplicate/compatible pair never produces a notice even when the model says candidate. The model is not bundled with the default package; install the `semantic-local` extra and point it at a GGUF file.
- **`semantic_notices` table and `memory_repair` notice/semantic_control tasks** — notices are viewable and dismissible via `memory_repair(task="notice", ...)`; runtime control (pause/resume/enable/disable/unload/status) via `memory_repair(task="semantic_control", ...)`. `memory_status` exposes a `semantic_conflict` diagnostics block.
- **`SemanticConflictWorker`** — single-threaded async worker (`max_concurrency` reserved to 1; configured values are clamped) with a per-pair budget floor (`min_pair_budget_ms`, default 1000) so the deadline gates *between* pairs and never starts an inference it cannot afford. `on_write=off` avoids spinning up the thread or preloading the model for a queue that stays empty.

### Removed

- **Legacy vector conflict-candidate scan** — the `memory_scan_conflict_candidates` MCP tool and the underlying `scan_conflict_candidates` / `pairs_closed_for_scan` / `purge_stale_dismissals` / `_bulk_backfill_meta` / `_scan_log_append` db methods are removed. The old KNN candidate scanner is no longer a conflict path; `embedding`/`sqlite-vec` remain for semantic recall, section recall, and workspace aliasing, but no longer feed a conflict scanner. `memory_doctor` no longer warns on a missing/stale `scan_log.jsonl` (the writer is gone); it reports open conflicts from the `conflicts` table and treats the legacy log as INFO context.

### Fixed

- **Semantic signal JSON no longer truncated on nested objects** — `model_signal_from_text` used a non-greedy `{.*?}` regex that stopped at the first `}`, so a model reply with a nested JSON value (e.g. `{"reason_code": "value_diff", "parsed": {...}}`) was rejected as `invalid_json` and the candidate was silently dropped on a high-recall path. JSON is now extracted by brace-balanced scanning that respects string literals, returning the first complete top-level object.
- **`todo_done` pair detection is direction-agnostic** — the gate previously fired only when the left text said "待办/todo" and the right said "已完成". Because the caller may pass new/old memories in either order, a new "done" write against an old "todo" peer was missed, and an unrelated todo/done pair could falsely pair. It now accepts both orientations and requires shared content tokens, so unrelated statements no longer couple.
- **Semantic worker error/progress writes are now lock-guarded** — `SemanticConflictWorker._last_error` and `_processed` were written from the job thread outside the condition lock while `status()` read them inside it, a benign-but-sloppy race. `_run` now updates both under `_cond`, and external timeout errors go through a new `set_error()` method instead of reaching into the private attribute from `_process_semantic_conflict_job`.
- **Per-pair budget guard prevents worker stalls near the deadline** — the job deadline could only be checked *between* pairs, so when the remaining budget was a few milliseconds the loop would still launch another non-interruptible model inference and overrun. A new `semantic_conflict.min_pair_budget_ms` floor (default 1000) makes the job stop early with a clear `last_error` instead, and `semantic_control status` now exposes `min_pair_budget_ms`, `last_pair_duration_ms`, and a `job_deadline_behavior` note explaining that a single stuck in-flight call still requires process-level isolation.
- **Worker `last_error` cleared on successful job** — once any job errored, `SemanticConflictWorker.status()["last_error"]` stayed populated forever even if every subsequent job succeeded, leaving a stale error in `memory_status`. A clean run now clears the prior transient error.
- **Doctor no longer warns on the deprecated scan_log** — `_check_conflicts_open` used to emit a WARNING when `scan_log.jsonl` had no `completed` entry or was stale, but the writer was removed with the legacy vector scan, so vec-enabled installs permanently saw a WARN pointing at a deprecated feature. The check now reports from the `conflicts` table and treats the legacy scan_log as INFO context only.
- **Version comment aligned** — a leftover "removed in v0.12" comment referenced an unreleased version; corrected to avoid the unreleased-version mismatch with the `[Unreleased]` changelog.

## [0.11.0] — 2026-08-07

### Changed

- **Default MCP surface is now task-oriented** — new clients see four product tools by default: `memory`, `memory_review`, `memory_govern`, and `memory_repair`. This replaces the previous default surface of many low-level tools, reducing always-loaded MCP schema/token overhead and making the daily agent path easier to choose.
- **Low-level tools are internal by default** — existing `MemoryTools` implementations remain in the codebase and are reused by the product tools, but low-level MCP tools such as `memory_write`, `memory_search`, `memory_supersede`, `memory_rebuild_claims`, and scan/record helpers are no longer registered by default.
- **Advanced compatibility profile** — set `MEMORY_ARBITER_TOOL_PROFILE=legacy_full` (or `full`) to expose the legacy low-level MCP tool surface alongside the new product tools.

### Added

- **Progressive tool help** — product tools support `help` actions/views/tasks and validation errors return command-specific hints so detailed low-level parameter guidance is loaded only when needed, not in every MCP tool schema.
- **Judgment constraints in help** — `memory(action="help")` and `memory_govern(action="help")` now expose `judge_constraints`: the allowed `verdict`, `recommended_use`, `usage_context`, `confidence_hint`, `resolution_kind`, and `conflict_scope` values plus their cross-field rules, so an agent can fill a `judge` / `correct_judgment` request correctly on the first try instead of iterating against `invalid_*` outcomes.

### Fixed

- **Missing-id forwards no longer raise** — `memory(action="read"|"update")`, `memory_govern(action="retire"|"confirm"|"correct_judgment"|"resolve_conflict")`, and `memory_repair(task="split"|"set_entity"|"activate_pending")` previously forwarded an empty payload to a method with a required positional id and raised `TypeError`. They now return `ok=false` with an error and help, matching the rest of the bad-input contract. `cleanup_history` with no `memory_id` is unchanged (it is a valid full-cleanup request gated by `authorized`).
- **Bad secondary int args no longer raise** — loosely-typed values like `memory(action="find", data={"limit": "abc"})`, `memory_govern(retire, superseded_by="xyz")`, `memory_review(conflicts/expired, limit="abc")`, and `memory_repair(cleanup_history, older_than_days="abc")` raised `ValueError` from an unguarded `int()` inside the low-level method. All product forwards now run through a wrapper that turns stray `TypeError`/`ValueError` into `ok=false` errors. Numeric strings (`"5"`) still coerce and succeed, matching common MCP-client JSON.
- **Judgment field ordering** — `memory_govern(correct_judgment)` previously reported a non-integer `conflict_id` before checking other required fields, forcing an agent to fix fields one at a time. The required-fields check now runs first; `judge` and `correct_judgment` also coerce `conflict_id` explicitly so a non-numeric id gets a clear `conflict_id must be an integer` error instead of an obscure `invalid_input` deep in the submit path.
- **Non-dict `data` rejected** — product tools previously coerced a non-dict `data` argument (string/list/int) to `{}` via `_payload_dict`, so `memory(action="remember", data="x")` silently wrote an empty record. They now return `ok=false` with `data must be a JSON object`.
- **Non-list `tags` no longer split into characters** — `MemoryRecord.from_input` did `list(payload.get("tags") or [])`, so `tags="todo"` became `['t','o','d','o']` and silently corrupted the tag index. Non-list/tuple tags now coerce to `[]`.

## [0.10.3] — 2026-08-07

### Added

- **Console Support Panel** — `mema console` now includes a Support card and top-bar shortcuts for GitHub Star, feature requests, and bug reports. Star opens the project repository; feedback opens a Console form that generates a prefilled public GitHub issue URL.

### Security

- The Support Panel does not use GitHub OAuth, does not store tokens, does not call GitHub APIs, and does not write to the local memory database. It only opens GitHub after an explicit user click. Prefilled issues include only coarse diagnostics by default (version, route, doctor status, open-conflict count), never memory content, conflict content, or local database paths unless the user manually types them into the form.

## [0.10.2] — 2026-08-06

### Added

- **Resolution guidance schema for conflict judgments** — `memory_submit_conflict_judgment` and `memory_correct_conflict_judgment` now accept optional `resolution_kind` (`partial_update`, `merge`, `contextual_keep_both`, `near_duplicate`, `full_replacement`, `not_a_conflict`) and `conflict_scope` (`field`, `section`, `record`, `whole_memory`, `unknown`). The values are stored append-only on `conflict_judgments` and projected onto `conflicts` for current guidance.
- **Machine-readable resolution actions** — judgment responses, `memory_list_conflicts`, and search `conflict_signal` guidance now expose `recommended_resolution_action` and `supersede_candidate`. Partial update/merge maps to `update_or_merge`; contextual guidance maps to `use_contextual_guidance`; near-duplicate/full-replacement maps to `supersede_old_memory` as a suggestion only.
- **Console read-only resolution guidance** — `mema console` now displays resolution chips and guidance callouts on conflict list/detail views. It remains read-only and does not expose resolve/edit/supersede actions.

### Changed

- **Partial conflicts are guarded against whole-memory winner misuse** — Arbiter still does not classify semantics itself; the host LLM submits the classification, and Arbiter validates that partial update/merge cannot carry a single winner, field/section conflicts cannot be full replacements, and supersede candidates require record/whole-memory scope plus an explicit winner.
- **Search can surface resolved guidance** — active search results can now attach reusable resolved `evolution`/`compatible` guidance when the exact memory snapshot still matches the stored CAS pins, without hiding active memories. Resolved pairs carrying a contradiction verdict are deliberately not surfaced as guidance and remain ringable by the vector scan for a second look; the two queries are intentionally asymmetric.
- **Scan terminal-pair filtering is now unified** — vector conflict scan skips `not_a_conflict` dismissals and manually resolved pinned pairs in addition to resolved evolution/compatible judgments, and reopens them when version/claim pins change.
- **Resolution-action classification is a single source of truth** — the `recommended_resolution_action` / `supersede_candidate` mapping now lives on `ConflictJudgmentStore` (`resolution_action` / `is_supersede_candidate`) and the duplicate copies in `tools.py` / `console_api.py` were removed. The superseded `MemoryDB.resolved_guidance_pairs_for` helper (subsumed by `pairs_closed_for_scan` in v0.10.1) was deleted. Internal refactor only; no API, behavior, or stored-data change.

## [0.10.1] — 2026-08-06

### Fixed

- **Conflict scan no longer re-rings already judged evolution/compatible pairs** — vector `memory_scan_conflict_candidates` now skips pairs whose exact memory snapshot already has resolved `evolution` or `compatible` guidance. The close is CAS-pinned by memory `version` and, when present, `claim_revision`, so editing either side naturally reopens the pair for scanning.

### Changed

- **`evolution` no longer implies whole-memory supersede** — tool descriptions now define it as same-topic change over time. Partial evolution should be handled with update/merge/contextual guidance; only near-duplicate or full-replacement cases should become supersede suggestions.

## [0.10.0] — 2026-08-05

### Added

- **Read-only local Console MVP (`mema console`)** — starts a localhost-only governance UI with a top language switch, sidebar navigation, bilingual `mema` / `迷码` branding, Overview metrics, open conflict review, conflict detail left/right memory comparison, read-only memory search, doctor findings, and read-only Settings descriptors. The Console defaults to `127.0.0.1:18876`, uses stdlib HTTP + a single embedded HTML/CSS/JS page, and deliberately exposes no write/resolve/supersede/config-save actions.
- **Config Registry** — a single bilingual descriptor table (`memory_arbiter/config_registry.py`) drives the Settings page labels/descriptions and is the future source of truth for config help text, avoiding doc drift.

### Security

- Console is local-only by default (`127.0.0.1`); non-localhost hosts and forged/empty `Host` headers are rejected. No write, resolve, supersede, confirm, or config-save actions are exposed. Memory content is rendered as escaped text, never `innerHTML`.

## [0.9.10] — 2026-08-05

### Added

- **`mema` short command / 迷码中文名** — the PyPI console scripts now include `mema` as a short alias for `memory-arbiter` / `memory-arbiter-mcp`. README usage examples now prefer `mema setup`, `mema doctor`, and `uvx --from memory-arbiter-mcp mema`, while the long entry points remain compatible.

### Changed

- **README repositioned around fact governance** — the project is now described as a trustworthy local fact layer for AI agents, not merely a shared memory layer. The README leads with trust, freshness, conflicts, history, structured claim gates, section recall, workspace isolation, and doctor diagnostics before installation/API details.
- **MCP SDK dependency capped below 2.0** — `requirements.txt` now matches `pyproject.toml` with `mcp>=1.2.0,<2`, because the current server uses `from mcp.server.fastmcp import FastMCP`, which is not available in `mcp 2.x`.

## [0.9.9] — 2026-08-05

### Fixed

- **Doctor update status no longer shows stale cached latest below the installed version** — after upgrading from an older cached PyPI check, `memory-arbiter doctor` could render a confusing line such as `当前版本: 0.9.8` with `最新已知: 0.9.5` while still saying `up_to_date`. The update monitor now floors stale cached `latest_version` to the observed installed version and records `latest_source=installed_version`, so doctor/status output stays internally consistent until the next PyPI refresh.

## [0.9.8] — 2026-08-05

### Added

- **Async split reindex worker (`SplitReindexWorker`)** — the rules-path
  section split in `memory_write` / content `memory_edit` moved from
  synchronous to asynchronous. Previously a long structured document blocked
  the tool call for ~0.3s × section count (a 14-section doc measured 4.4s of
  GGUF embedding inside the call, risking the 30s MCP timeout). Write/edit
  now completes DB write + memory-level embed + claims rebuild and returns
  immediately (~0.5s); per-section embedding runs on a single background
  daemon thread serialised via an in-process dict queue (GGUF inference is
  not thread-safe — parallel embeds deadlock, so a thread pool would still
  serialise behind a lock with no benefit).
  - **Response field change**: `split.mode` is now `rules_async` (was
    `rules`), `applied` is `false` (was `true`), and the block carries
    `reindex_pending: true` plus `section_count_estimated`. `split_status`
    stays `NULL` until the background publish lands, then flips to `active`.
  - **Search degrade is recall-safe**: while `split_status=NULL`,
    `memory_search` returns the full memory (`content_scope=full_memory`) —
    section-level precision is temporarily absent but no result is dropped;
    memory-level vector recall keeps working. Section precision restores
    automatically once the worker publishes.
  - **`memory_status`** exposes `split_reindex_pending: [memory_id, …]` —
    the ids currently queued or being processed, so the host can poll
    progress.
  - **Concurrency safety reuses the existing 5-point CAS** in
    `_publish_sections` (status / content_hash / version / split_status /
    split_revision checked inside `BEGIN IMMEDIATE`): a stale snapshot from
    an edit that bumped `split_revision` fails CAS and is safely discarded;
    the new edit re-enqueues. Cross-process writers are serialised by WAL +
    `busy_timeout`; a later publisher loses CAS and wastes one embed pass
    but corrupts nothing.
  - **Embedder double-checked locking** (`_embedder_lock`) fixes a TOCTOU in
    `_ensure_embedder` — the worker is the first background thread to call
    it, so the check-then-`build_embedder` race (which would double-mmap the
    GGUF) had to be closed first.
  - **`embed_text` call serialisation** (`ManagedEmbedder._embed_lock`) closes
    a regression the async worker introduced: while the worker embeds sections
    on its background thread, a concurrent `memory_search` on the main thread
    vectorises the query through the *same* `Llama` instance. llama-cpp's
    GGUF inference (both `create_embedding` and `tokenize`) is not thread-safe
    — the two overlapping calls could deadlock. The lock wraps the whole
    `embed_text` body so every llama-cpp call is mutually exclusive. Embed is
    already single-threaded by nature, so this costs nothing; it only makes
    the "one caller at a time" invariant explicit. Affects all call sites
    (write/search/edit/doctor/rebuild) uniformly.
  - **Startup scan** enqueues up to 100 `split_status IS NULL` long
    memories on boot, recovering crash-leftover NULL rows and old-library
    upgrades; the rest self-heal on next edit.
  - **`memory_doctor_overview`** no longer misreports the async window: the
    `split.long_unsplit_backlog` check excludes ids currently inflight on
    the worker, so a memory mid-reindex is not flagged as a backlog needing
    Agent continuation (which would needlessly race the worker's CAS).

## [0.9.7] — 2026-08-04

### Added

- **Workspace isolation (`none` / `weak` / `strict`)** — the long-reserved
  `workspace` field can now partition recall, controlled globally by
  `MEMORY_ARBITER_ISOLATION` (default `none`). The three levels are monotonic —
  workspace goes from *ineffective* → *affects ranking* → *controls visibility*:
  - **`none`** (default): workspace is a pure label. Recall ignores it entirely
    (never filtered, never ranked). Backward-compatible with v0.7.4 behavior.
  - **`weak`**: recall is always whole-library — workspace never enters a WHERE
    clause. Passing a workspace only soft-reranks (same-workspace boosted,
    cross-workspace demoted, nothing dropped). A new workspace emits
    `write_hints.new_workspace_detected`.
  - **`strict`**: workspace is mandatory on write (empty → error) and on recall
    (missing → error). Recall hard-filters to the same canonical workspace. A
    brand-new workspace on write is blocked as `status=pending` (excluded from
    active recall) and returns `action_required=confirm_new_workspace`.
- **`memory_activate` tool** — activates a memory held as `pending` by strict
  workspace isolation. Clears the new-workspace gate without the trust/lock
  promotion `memory_confirm` applies. Requires `authorized=true`; rejects
  non-pending memories.
- **Workspace alias canonicalization (double-store)** — `memories.workspace`
  keeps the raw string; the new `memories.workspace_canonical` column holds the
  resolved name. Names are merged by cosine similarity over a
  `workspace_canonicals` registry + vector table (e.g. `金营项目` and
  `金科营销项目` collapse to one canonical). Cutoff is
  `workspace_match_distance` (env `MEMORY_ARBITER_WORKSPACE_MATCH_DISTANCE`,
  default `0.25`). Resolution runs only under `weak`/`strict`; without an
  embedder it degrades to exact string identity. Old rows get a lazy NULL
  canonical via idempotent migration — no backfill required.

### Notes

- Default `isolation=none` preserves all prior recall behavior. Enabling
  `strict` trades recallability for isolation: a wrong or inconsistently-spelled
  workspace silently isolates memories. When unsure, use `weak` (never drops,
  only demotes). See the Workspace Isolation section in the README.

### Fixed (pre-release review)

Six strict-isolation defects found by adversarial review before release, all
fixed in this version:
- **strict fallback leaked cross-workspace memories** — when a strict search's
  candidate pool emptied after workspace filtering, `_recent_fallback` was
  called without `ws_canonical`, returning whole-DB recent memories. strict now
  returns empty instead of falling back.
- **bm25 direct hits ignored workspace** — `_search_bm25`'s FTS/LIKE SQL had no
  workspace predicate; under `RANKING_MODE=bm25` + strict, cross-workspace
  matches leaked directly.
- **pool saturation hid same-workspace hits** — `_wide_recall` filled the pool
  from the whole DB before strict post-filtered, so other-workspace matches
  could crowd out a real same-workspace hit. Workspace predicates are now pushed
  down into every recall channel (FTS main/OR, subject/tags LIKE, content LIKE,
  memory-vec KNN, section-vec KNN).
- **`linked_open_items` leaked cross-workspace todos** — the side query had no
  workspace constraint under strict, surfencing other workspaces' todo metadata.
- **`total_estimate`/`has_more` counted cross-workspace** — non-empty query +
  filters used a whole-DB count, mismatching the strict-filtered result list.
- **`attention_summary` pointed at the wrong tool** — the strict new-workspace
  block told users to call `memory_confirm` (which rejects pending memories);
  corrected to `memory_activate`.

## [0.9.6] — 2026-08-04

### Added

- **Update discovery & notices (side channel)** — the MCP server now performs
  a one-shot background PyPI check when due (at most every 24h, retry after 6h
  on failure) and may attach a top-level `notices` array to successful tool
  responses. Notices are a side channel: `data` is unchanged, and callers that
  do not understand them can ignore them. Two notice types: `update_available`
  (a newer version exists on PyPI) and `post_upgrade_doctor_recommended` (the
  installed version changed and doctor has not been run on it yet). Each notice
  is suppressed for 7 days per version; there is no ack protocol and no
  automatic upgrade. Disable with `{"update_check":{"enabled":false}}` in
  `~/.config/memory-arbiter/config.json`. State is cached in
  `~/.local/share/memory-arbiter/update_state.json`. The background check is a
  daemon thread that exits after one fetch (success/failure/timeout); it is not
  a persistent worker and is never auto-restarted.

- **Doctor fix metadata (P1)** — each doctor finding now carries lightweight
  machine-readable repair metadata (`fix_kind`, `fix_tool`,
  `requires_authorized`, `risk`) for high-value repair paths, instead of only a
  human-readable `fix_hint`. `fix_kind` values: `mcp_tool`, `agent_assisted`,
  `manual_config`, `dependency_install`, `model_download`, `manual_or_none`,
  `none`.

- **`memory_status` update-check state** — `memory_status` now reports cached
  update-check state (`update_check` object: `enabled`, `status`,
  `current_version`, `latest_version`, `latest_checked_at`, `cache_stale`,
  `last_doctor_run_version`, etc.) and whether the background check is enabled.
  Doctor never performs a network check; it only displays cached state and
  records that doctor has run for the current installed version.

### Fixed

- **Disabled update_check no longer writes state file** — `_write_state_locked`
  early-returns when `enabled=False`, covering all write paths (`__init__`
  version observation, `record_doctor_run`, `consume_notices`, background
  check). Previously, running the CLI doctor with `update_check` disabled would
  still create `~/.local/share/memory-arbiter/update_state.json`.

## [0.9.5] — 2026-08-03

### Fixed

- **MCP SDK compatibility** — pins the Python MCP SDK dependency to
  `mcp>=1.2.0,<2`. `mcp 2.0.0` removed/relocated `mcp.server.fastmcp`, which
  `memory-arbiter-mcp` uses for its server entry point; without the upper bound,
  fresh `uvx memory-arbiter-mcp` installs could resolve `mcp 2.0.0` and fail at
  startup. No memory schema or API behavior changes from v0.9.4.

## [0.9.4] — 2026-08-03

### Added

- **`memory_search_expired` tool** — searches non-active non-deleted memories
  (superseded + conflicted + pending) with vec-hybrid recall (vec channel with
  `parent_status` predicate + FTS). Replaces the old `include_superseded=true`
  parameter on `memory_search`. The query domain is physically isolated from
  active queries so superseded results never crowd out active ones. Default
  `limit=20`, hard cap 50, configurable via `MEMORY_ARBITER_SUPERSEDED_LIMIT`.
  v0.9.4 also adds `offset` cursor pagination: exact on the empty-query +
  filters path (SQL `OFFSET` + precise `count_filtered_memories`), best-effort
  on the query-recall path (relevance pool widened to `offset + limit`).
  `memory_search` deliberately remains unpaginated. Offset clamped to
  `[0, 10000]`.

- **`memory_resync_vec_parent_status` tool** — repairs `vec.parent_status` to
  match `memories.status` when drift is detected (direct DB edits, migration
  bugs, or failed transactions). Uses `dry_run=true` (default) to preview
  mismatches; actual repair uses `dry_run=false` and does not require
  `authorized` because it is a non-destructive metadata UPDATE. The doctor
  `consistency.vec_parent_status_sync` check reports drift and points at this
  tool.

- **Schema migration: `vec0` metadata `parent_status` column** — idempotent
  startup migration adds `parent_status TEXT` to `memories_vec` and
  `memory_sections_vec` (DROP+CREATE+re-insert because vec0 does not support
  ALTER). Parent status is derived from `memories.status` with `COALESCE(...,
  'deleted')` for orphan rows. Requires `sqlite-vec>=0.1.6`.

### Changed

- **`memory_search` now returns ONLY active memories** — removes
  `include_superseded` parameter. Status filter is hard-coded to `status='active'`.
  For superseded history recall use the new `memory_search_expired`.

- **vec KNN filter generalized to `parent_status_filter`** — `vec_knn` and
  `section_vec_knn` replace the boolean `include_superseded` with a string
  enum: `"active"` (`= 'active'`), `"expired"` (`NOT IN ('active','deleted')`,
  covering superseded + conflicted + pending), `"all"` (`!= 'deleted'`). This
  aligns the vec channel's domain with FTS semantics: `status_filter="all"`
  now returns all non-deleted from both channels (previously vec returned
  active-only, missing semantically-relevant superseded vectors).

- **`memory_search_expired` recall domain expanded** — returns all non-active
  non-deleted memories, not just `superseded`. This eliminates "dead vectors":
  conflicted/pending memories had `parent_status` set to those literal statuses
  by `update_memory`, but were invisible to both active and (old)
  superseded-only recall. `memory_search` is unchanged (active only).

- **Supersede/arbitrate vec sync consolidated** — vector `parent_status` is
  synced solely by `update_memory`'s built-in vec synchronization (which writes
  both `memories_vec` and `memory_sections_vec` in the same transaction as the
  status flip); the redundant `mark_vectors_for_memory` calls in the
  `supersede`/`arbitrate` paths are removed. `mark_vectors_for_memory` is
  retained as a primitive for the future revive path (`mark(id, 'active')`).

- **`memory_cleanup_inactive_vectors` semantics changed** — now performs two
  phases: (1) resync `parent_status` mismatches via `_resync_vec_parent_status`,
  (2) purge true orphans (vec rows whose parent memory/section no longer exists).
  It no longer deletes superseded vectors (they are kept for audit recall).

- **Doctor `consistency.inactive_vectors_slowpath` retired** — replaced by
  `consistency.vec_parent_status_sync` which reports mismatches between
  `vec.parent_status` and `memories.status`. The old check assumed inactive
  vectors were always problematic; the retain-vectors design intentionally
  keeps superseded vectors for history recall.

- **KNN query optimization (metadata predicate)** — `vec_knn` and
  `section_vec_knn` now use the `parent_status` metadata predicate on the
  vec0 MATCH query, eliminating the O(n) exact-distance slow path that was
  triggered whenever inactive vectors existed.

### Fixed

- **Expired-search pagination UX** — `memory_search_expired` now honors
  `offset` on empty-query browse and bm25 paths, returns caller-friendly
  pagination metadata (`effective_limit`, `next_offset`, clamp/cap flags,
  `pagination_precision`), and preserves `total_estimate` for out-of-range
  exact-pagination requests.
- **Defensive active/expired domain isolation** — vec MATCH queries now pair
  the vec0 `parent_status` predicate with the current `memories.status` JOIN
  predicate, preventing stale vec metadata from leaking superseded rows into
  active recall or active rows into expired recall before resync runs.
- **Doctor orphan-vector reporting** — `consistency.orphan_vectors` now counts
  both memory-level and section-level orphan vectors and points to
  `memory_cleanup_inactive_vectors` for the authorized orphan purge path.
- **Migration and cleanup wording** — legacy `include_superseded` callers get a
  clearer migration error; cleanup dry-run hints distinguish non-destructive
  `memory_resync_vec_parent_status(dry_run=false)` from authorized orphan purge.

### Migration Notes

1. **Backup required**: The vec0 schema migration is destructive (DROP+CREATE).
   Backup `memory.sqlite3` before first startup on v0.9.4.
2. **sqlite-vec version**: Ensure `sqlite-vec>=0.1.6` (metadata column support).
3. **API signature change**: `vec_knn` / `section_vec_knn`
   `include_superseded: bool` → `parent_status_filter: str`. Direct callers
   must update. Pass `"expired"` where `include_superseded=True` was used;
   `"active"` is the default (was the `False` behavior).
4. **`status_filter="superseded"` is aliased to `"expired"`** in
   `search_memories` for backward compatibility, but new code should use
   `"expired"` (the broader, accurate domain).
5. **Behavior reversal**: Old `include_superseded=true` vector recall no longer
   works (it returned active+superseded mixed). Use `memory_search` (active) or
   `memory_search_expired` (non-active non-deleted) explicitly.

## [0.9.3] — 2026-08-03

### Fixed

- **Doctor `split.index_integrity` offset spot-check now scoped to active
  parents** — the offset continuity sub-query still flagged anomalies on
  superseded/deleted parents, whose sections are retained as audit history by
  design (v0.9.2). It now joins `m.status='active'`, matching the other two
  sub-queries in the same check and the v0.9.2 `section_vec_coverage` scoping.

## [0.9.2] — 2026-08-03

### Added

- **`memory_cleanup_inactive_vectors` tool** — physically deletes
  `memories_vec` / `memory_sections_vec` rows whose parent memory is not
  active (superseded/deleted/orphan). `dry_run=true` (default) reports counts;
  actual deletion requires `dry_run=false` + `authorized=true`. Only vector
  rows are touched — memory content, FTS and audit history are preserved, and
  vectors can be rebuilt from content via `memory_rebuild_embeddings` if ever
  needed. The doctor `consistency.inactive_vectors_slowpath` fix_hint now
  points at this tool instead of manual SQL.

### Changed

- **Supersede / arbitrate now cascade-delete the loser's vectors** —
  `memory_supersede` and the `memory_arbitrate` auto-apply path remove the
  superseded memory's memory-level and section-level vectors, so they stop
  accumulating in the vec tables and permanently forcing KNN onto the
  exact-distance slow path introduced in v0.9.1. Side effect: explicit
  `include_superseded=true` **vector** recall of a superseded memory is no
  longer possible (its vectors are gone); auditing its content/FTS is
  unaffected. This restores the vec0 MATCH fast path for all KNN once existing
  inactive vectors are cleared.

## [0.9.1] — 2026-08-02

### Fixed

- **Superseded/deleted vectors no longer poison semantic top-k recall** —
  memory and section recall now uses the vec0 KNN fast path only when every
  vector is eligible; if an inactive parent exists, it computes exact distance
  over rows pre-filtered by authoritative `memories.status` before applying
  LIMIT. Inactive vectors therefore cannot consume all KNN slots, while
  `include_superseded=true` remains the explicit audit path and deleted rows
  stay excluded. Doctor now distinguishes safely retained inactive sections
  from true physical orphans. No duplicate lifecycle state or vec schema
  change is introduced.

### Added

- **Doctor slow-path visibility** — a new `consistency.inactive_vectors_slowpath`
  check reports how many inactive (superseded/deleted/orphan) memory/section
  vectors exist and warns that, while they remain, every KNN query falls back
  to the exact-distance slow path. Recall correctness is unaffected; this is
  purely a latency signal with a manual-cleanup hint.

## [0.9.0] — 2026-08-02

### Added

- **Real-time structured conflict detection** — writes and edits now extract conservative deterministic claims (explicit key/value pairs, Markdown tables, number+unit facts, and semantic versions) into the separate `memory_claims` derived-index table. Claims are keyed by canonical `entity + attribute + scope`; code fences, URLs/references, ambiguous same-key values, and common CJK bookkeeping fields are excluded. The existing semantic scan remains the broad-coverage backstop.
- **Mandatory host-LLM judgment receipt** — a new structured collision is persisted as `pending_llm` and returned with a top-level `attention_required` / `action_required=judge_conflict_before_use` plus evidence and CAS pins. The host model submits its verdict through `memory_submit_conflict_judgment`; the result is stored in append-only `conflict_judgments`. Low-risk assessed conflicts become non-blocking query guidance, while uncertain, protected-vs-protected, and high-impact code/config/write/external-action uses escalate to `pending_user`.
- **Human correction without rewriting history** — `memory_correct_conflict_judgment` appends an authorized human judgment and makes it active; prior LLM/policy judgments remain auditable. LLM judgment never edits or supersedes either memory.
- **Entity and backfill operations** — `memory_set_entity`, `memory_list_entities`, and bounded `memory_rebuild_claims` support incremental entity cleanup and existing-library claim backfill. Entity/scope-only changes bump `claim_revision`, not content `version` or content history.
- **Operational visibility and kill switch** — `memory_status` and doctor now self-report `arbiter_version` so operators can confirm which build a running server is on. `memory_status` reports `structured_claim_mode`; doctor reports indexed/stale/unreconciled/ambiguous claims, pending judgment backlog, structured-path latency/candidate counts, structured-only / scan-only / both coverage, and real-time lead time. `structured_claim_mode=beta_all` is the default for the current trusted beta cohort; `off` is the emergency switch.
- **Recoverable post-index reconciliation** — `claims_reconciled_revision` advances only after pair resolution/upsert and judgment-request preparation all succeed. A crash or query failure after claim publication therefore remains visible to doctor and is selected by `memory_rebuild_claims` instead of becoming a permanently missed conflict.
- **Structured/scan coexistence** — one conflict row retains independent first-detection timestamps for both channels. A scheduled scan may enrich an existing structured conflict but cannot clear its version/claim-revision pins or bypass the mandatory judgment gate; scan-first rows are upgraded in place when structured detection later supplies a valid snapshot.

### Changed

- Search now distinguishes `structured_claim_candidate` (pending LLM receipt, loud), `open_table` / `ask_user` (human decision required, loud), and `conflict_guidance` (already assessed, non-blocking). A source revision change invalidates the active judgment through version + claim-revision CAS and reopens the pair.
- Schema startup migration adds `memory_claims`, `conflict_judgments`, claim index/reconciliation state, channel provenance, latency telemetry, and conflict judgment state idempotently. Zero extracted claims count as a successful index publication and reconciliation; extraction/storage or post-index reconciliation failure remains doctor-visible and fail-open.

389 tests pass.

## License policy

Memory Arbiter version 0.8.2 and later are offered under the Apache License 2.0 going forward. Prior MIT grants remain valid for copies previously distributed under MIT (including 0.8.0 and 0.8.1). Versions before 0.8.2 were released under MIT.

## [0.8.8] — 2026-08-01

### Added

- **`attention_required` 响铃计数 + doctor 统计** — 0.8.7 把 `attention_required` 提升为顶层强标志(写入路径 + 搜索路径),但响多少次、按来源(open_table / runtime_metadata_hint / 写入重复·演化)各多少,没有可观测的数——而 cry-wolf 疲劳是静默发生的(用户会下意识忽略、却说不清忽略了多少)。现每次 `attention_required` 触发时,往 `attention_log.jsonl`(与 `memory.sqlite3` 同目录,复用 scan_log 的 best-effort 追加纪律、绝不抛错)写一条 `{ts, trigger(write/search), source, ids}`;doctor 新增 `capacity.attention_volume` 检查(第 19 项),报告近 7 天响铃总数 + 按 **trigger(write/search) 和来源** 拆分——write 是事件性(每次写入、低频),search 是查询性(同一冲突被反复检索会重复响),两者混在一个 total 会让高频 search 掩盖 write 的真信号,故拆开看（by_trigger + by_source_trigger 交叉表）。**始终 INFO、绝不 WARN**——是否刷屏由人判断,doctor 不替你下结论(0.8.7 决定让 advisory 的 runtime_metadata_hint 也响铃、先跑跑看;这条计数就是"跑跑看"的硬依据:若 runtime_hint 的量是 open_table 的数倍,就该考虑只让 open_table 响)。

- **搜索路径的 advisory 信号不再响强制铃,改由 LLM 按内容判断** — `runtime_metadata_hint`(advisory、未验证,且同一对重叠记忆每次检索都会重复触发)此前和 `open_table` 一样置顶层 `attention_required` + MUST-surface,误报会在每次搜索反复打断用户(cry-wolf)。现:仅 `open_table`(已验证、可 `memory_resolve_conflict` 关闭)响强制铃;`runtime_metadata_hint` 保留为 per-result signal,docstring 改为要求调用方**按内容判断**(对比命中正文与 `conflict_peer` snippet,确实矛盾才提示,只是话题相近就静默)——保留 advisory 的召回又不 nag 误报。doctor 的 by_source 仍记录 advisory 出现次数(即便不响铃),便于看它到底多频繁。

- **write advisory 落表 + dismiss(not_a_conflict) 永久止闹** — 推翻 id=264「write 不落 conflicts 表」:write 命中 duplicate 现记录为 `source='metadata_write_hint'` 的 open 行,可 `resolve_conflict(status='not_a_conflict')` 永久 dismiss。新增 `is_pair_dismissed`(version CAS:编辑其中一条 memory 即失效 dismiss、重新可响)。`_build_open_table_signal` 按行 source 路由(`metadata_write_hint`→`runtime_metadata_hint` advisory,不冒充 verified `open_table`)。`memory_supersede` 连清涉及该 memory 的 not_a_conflict 行;scan GC 版本过期的 dismissed 行;`memory_list_conflicts` 加 source 过滤参;doctor `conflicts_open` 按 source 拆。口径 **(A)**:search advisory 不响强制铃(LLM 软判),write 响(低频)+ 可 dismiss。insert-first 硬规则:记忆先 active 入库再跑 advisory,best-effort、fail-open(dismiss 检查抛错也照常响)。

### Fixed

- 搜索 attention 块显式带上 `include_conflict_signal` 门;search 摘要现仅引用 open_table(advisory 不再进入强制铃路径,无需动态冠词)。
- **NULL version CAS 修复(ZCode review id=434)** — write 落表时 candidate version 没兜底(`get_memory_version(cand_id)` 无 `or 1`),race/遗留 NULL 版本会让 dismiss 行的 version CAS 静默失效:`NULL = x` 恒为 NULL → `is_pair_dismissed` 误判「未 dismiss」→ 止闹失效、nag 复活;`purge_stale_dismissals` 的 `IS NOT NULL` 又护着死行不清理(同一 NULL 行,一处当失效、一处当有效,自相矛盾)。修:write 端 `or 1` 堵源头;`is_pair_dismissed` 与 `dismissed_pairs_for` 改 `IS NULL OR =`(NULL pin 视为有效 dismiss,不 re-nag);`dismissed_pairs_for` 补回 `IN (result_ids)` 约束(不再全表扫)。补 NULL 边界回归测试。

350 tests pass (342 + 8 new).

## [0.8.7] — 2026-08-01

### Added

- **写入时的重复/演化提示现在会要求 agent 转告用户** — `memory_write` 早已在响应里返回 `write_hints`(subject/tags 高度重叠时标记 `possible_duplicate` / `possible_evolution_of`),但调用方 agent 几乎从不把它告诉用户,两个原因:① 工具描述对 `write_hints` / "要不要提醒用户"只字未提,agent 根本不知道有这回事;② 信号藏在嵌套的 `data.write_hints.possible_supersede_targets`、字段名又是 "possible"(读着像可忽略)。现修两处:`memory_write` 的 docstring 显式要求写入后**必须**检查 `attention_required` / `write_hints`,命中时**必须**转告用户(给出现成话术)再视为写入完成;`_enrich_write_response` 命中时在 `data` 顶层新增布尔 `attention_required` + 现成可读的 `attention_summary`(如 "Possible duplicate/evolution of memory #42 (api-token-policy)"),让 agent 扫一眼响应就能撞见、不必钻进 `write_hints`。仍是 advisory、且只覆盖 metadata 重叠(**重复/演化**),**不查语义矛盾**(那是 0.9 结构化冲突检测的范围)。无法"强制"(MCP 工具不能指令 LLM 必须说话),但描述 + 顶层强标志双管齐下,把"从不提示"推到"大多数情况会提示";写入那一刻漏的,由搜索路径已有的 `conflict_signal` 兜底。
- **搜索路径的 `conflict_signal` 同步提升为顶层 `attention_required` 强标志** — 上一条只改了写入路径,搜索路径的 `conflict_signal`(v0.7.6,每个 result 各自附 `open_table` / `runtime_metadata_hint`)仍是嵌套字段,agent 不一定逐条翻看,会和写入提示存在同样的"有信号但不被注意"的不对称。现补齐对称:`memory_search` 在任一 direct 命中带 `conflict_signal` 时,在 `data` 顶层同样置 `attention_required` + `attention_summary`(如 "Search hit #1 (revenue) carries a open_table signal vs #2 (revenue-v2) and 1 more"),并要求 agent 命中时转告用户(给出核实话术)。细节仍在各 result 的 `conflict_signal`,顶层标志只是"快扫描"用的响铃。两路(write/search)现在用同一套 `attention_required` / `attention_summary` 字段名,agent 的检查逻辑可统一。

341 tests pass (338 + 3 new).

## [0.8.6] — 2026-08-01

### Fixed

- **`memory_edit` content 模式现支持 `add_tags`/`remove_tags`** — content 编辑路径(整体 `new_content` 替换 / `old_text`+`new_text` 局部替换)此前静默丢弃 `add_tags`/`remove_tags`,只转发 `new_tags` 给 `db.edit_memory`。"重写正文同时微调一个 tag"在 tag 侧是空操作,调用方不得不追加一次 `tags_only` 编辑(v0.8.5 回写 todo 时连踩 3 次发现)。现修复:`add`/`remove` 在 content 路径内合并,叠加在 `new_tags`(若传)或现有 tags 之上,复用 `update_tags_low_side_effect` 的保序去重算法(先 remove 再 add)。`new_tags` 单独传仍是全量替换;`new_tags=None` 且无增删时透传 None(`edit_memory` 保持原 tags),既有调用零影响。

338 tests pass (337 + 1 new).

## [0.8.5] — 2026-07-31

### Breaking

- **`memory_arbitrate` 参数 `apply` 改名为 `authorized`** — 此前 `apply=true` 是全库唯一无授权门的破坏性路径（LLM 传 `apply=true` 即按启发式自动退役败方）。现统一为 `authorized` 语义：默认 false 只返回比对结果，`authorized=true`（用户确认后）才自动将非保护败方标记为 superseded。mark_conflict 半边不变、只读比对不变；响应键 `applied` 保持不变（仅参数名变更）。内层保护检查（跳过 LOCKED / USER_CONFIRMED 败方）作为纵深防御保留。传入旧的 `apply` 参数会返回**显式迁移错误**（指向 `authorized`），而非静默忽略——防止已部署的旧 Agent 误以为生效（静默失效）。
- **`memory_confirm` 新增 `authorized` 门** — 此前任意调用方可把任意记忆自封为 USER_CONFIRMED + LOCKED，绕过 source_type 信任模型。现要求 `authorized=true` 才执行（与 `memory_supersede` 对齐）——提升到最高信任/保护档必须是显式、用户确认的动作。

### Added

- **`memory_search` 空 query + filter 召回（G6）** — 此前 `query="" + tags_filter / after_time / before_time / source_type` 是死路径（返回空 + "query required" warning，id=211 risk#9 推迟到 v0.7.4+）。现改为 filter 驱动召回、按 `ingest_time` 倒序返回，解锁 list-by-tag / by-source_type / by-time（如 `memory_search(query="", tags_filter=["todo"])` 列出所有待办）。沿用 `count_filtered_memories` 的 WHERE 子句（抽出公共 `_filter_clauses`，SQL 召回与计数同源），无分页、靠 `has_more`+`total_estimate`；`retrieval_mode="direct"`（与非空 + filter 路径一致，conflict_signal / linked_open_items 正常触发）。无新工具（id=210：list + 过滤 == search）。

### Fixed

- **`__init__.py` 版本号漂移** — `__version__` 此前停在 0.8.2（连续两个版本漏改，复现发版自查清单的老问题），本次同步到 0.8.5，四处（`__init__.py` / `pyproject.toml` / `server.json` ×2）一致。
- **`edit_memory` 并发竞态（#3）** — 改用 `write_transaction`（BEGIN IMMEDIATE），读前取写锁串行化并发编辑；此前用 `connection()`，两个并发编辑都读 v=1 都写 v=2 → 丢更新 + `memory_history` 版本号重复。
- **`load_policy` 启动崩溃（#4）** — 损坏的 policy.json 不再让 server 启动崩溃：try/except 回退默认全放行策略并 warning（对齐 `load_config_file`）。
- **`client_defaults` / `default_enabled` 字符串真值（#5）** — 手写的 `"false"` 不再被当真值；load 时统一 `parse_bool`。
- **`memory-arbiter setup` 配置写入原子化（#6）** — 先写临时文件再原子 rename，写入失败不再让 config_path 缺失/半写。
- **search 通道 3/4 + bm25 容错（#7）** — 此前无 try/except、`conn.close()` 不在 finally，瞬时 SQLite 错误会泄露连接并中断整个搜索；现对齐通道 1/2 的 per-channel try/except。
- **embedder 失败不再永久缓存** — `_embedder_loaded` 仅在构建成功后置 True；此前一次失败（模型缺失/维度不符）后永久返回 None，重装模型也无法自愈。
- **`memory_edit(new_content="")` 拒绝静默清空** — 空/纯空白 new_content 返回错误而非抹空正文。
- **`get_memory_summaries` 过滤 `status='active'`** — 对齐其 docstring 契约（非 active 视为"消失"），不再把 superseded 摘要混进 conflict_signal。
- **`count_filtered_memories` 去掉遗留 `workspace` 形参** — v0.7.4 起不过滤 workspace，去掉误导性死参数，与新 `recall_by_filters` 对称（zcode review 反馈）。

### Docs

- **21 条 MCP 工具描述英文化** — `server.py` 全部 `@app.tool()` docstring 由中文改为英文（MCP 生态惯例），与项目面向全球的定位（双语 README、PyPI、awesome-mcp-servers、「any AI client」）一致。保留全部操作护栏（tags_filter 废 vec、空 query + filter 行为、ASCII+CJK 空格分隔、limit 是单页非上限、has_more 无翻页、split 仅内部续接、supersede/confirm/edit-of-protected 需 authorized 等），只换语言；过长描述顺手收紧。不做全文双语（避免 2x token，与省 token 卖点冲突）。测试不依赖描述文本，无回归。

337 tests pass (328 + 9 new).

## [0.8.4] — 2026-07-27

### Fixed

- **`memory-arbiter setup` 保留用户已有的向量模型** — 修复 v0.8.3 的体验缺陷：当用户已在 `config.json` 里配置了 `embedding.model_path` 且该路径指向真实存在的模型文件时，setup 不再用 embeddinggemma 默认路径覆盖，而是**沿用用户的模型**，只在配置文件顶部打印一条 `ℹ 检测到你已配置的模型: X（沿用，未覆盖）` 提示。这避免了"用户装了 BGE / nomic 等别的 GGUF 模型，跑完 setup 后被覆盖成 embeddinggemma"的问题。新用户（无 config 或 model_path 指向不存在的文件）行为不变，仍走 embeddinggemma 默认。`--force` 显式绕过此保留逻辑，强制重置为默认模型。用户的自定义模型不做大小基准校验（无法预知正确大小），仅报文件大小作参考。7 个新测试覆盖：无 config / 路径失效 / 默认 embeddinggemma / 真实自定义模型 / 损坏 JSON / 端到端保留 / `--force` 绕过。328 tests pass (321 + 7 new).

## [0.8.3] — 2026-07-27

### Added

- **`memory-arbiter setup` 子命令** — 一键生成 `~/.config/memory-arbiter/config.json` + 环境自检 + 精确指引。半自动方案：setup 检测 `sqlite-vec` / `llama-cpp-python` / GGUF 模型文件是否就位，对缺失项打印**可直接复制**的安装命令（含 llama-cpp-python CPU 预构建 wheel 的正确 `--extra-index-url`）和 HuggingFace + ModelScope（国内镜像）下载链接；Python 版本若不在预构建 wheel 覆盖的 3.10–3.12 范围会给出提示。**不调 pip、不下模型、不碰网络**——依赖装失败是用户环境问题，不是 setup 的 bug。参数：`--print-config`（dry-run 预览）、`--no-config`（只跑自检）、`--force`（覆盖已有 config 不备份）、`--config-path`（自定义写入路径）。macOS + Windows 通用，路径全程 `Path.home()` 自动适配。沿用 `doctor` 的 `argv[1]` 分流模式，不新增 console script。321 tests pass (20 new).

## [0.8.2] — 2026-07-24

License switch release. Starting with this version, the project is offered under Apache License 2.0 going forward. See the License policy section above for the full statement.

### Changed

- **License: MIT → Apache-2.0** for version 0.8.2 and later. Prior versions (0.8.0, 0.8.1 and earlier) remain MIT; existing MIT grants for previously distributed copies stay valid.
- **Copyright holder declared as 张志维 (billy12151)** — the real name behind the billy12151 handle. Applied consistently across `NOTICE`, `pyproject.toml` authors, and README license sections. Apache-2.0's reciprocal patent grant (§3) makes copyright-holder determinacy more material than under MIT, which is why the real name is used while the handle is retained in parentheses as a brand reference.
- `LICENSE` replaced with the Apache License 2.0 full text; `NOTICE` added.
- `server.json` MCP registry metadata updated: top-level `license` MIT → Apache-2.0.

No code or behaviour changes. 301 tests pass unchanged.

## [0.8.1] — 2026-07-24

Docstring-only hotfix. Three MCP tool docstrings in `server.py` were stale relative to the v0.8.0 release (the `tools.py` method layer and README were already correct). Since MCP clients and AI agents derive tool descriptions from these docstrings at runtime, the drift could mislead callers. No behaviour change — 301 tests pass unchanged.

### Fixed

- **`memory_get`** — the docstring still described the pre-v0.8 behaviour ("section_ids empty → returns all sections"). Rewritten to the v0.8 contract: default `sections="catalog"`, `section_ids` precedence with `missing_section_ids`, `matched` rejected (no search context), and the returned `split` sub-object (`status` / `legacy_status` / `revision` / `section_count` / `content_hash`).
- **`memory_split`** — documented the missing `decline` branch, the `allowed_decision` return values (`rebuild` for active records, `split` otherwise), the CAS snapshot checks on publish, and that provenance is fixed to `"agent"` (the rules path runs inside `memory_write`/`memory_edit`, not via this tool).
- **`memory_recent`** — dropped the pre-v0.7.4 "specified workspace" wording; workspace is reserved metadata and does not filter results. The parameter remains for interface stability.

## [0.8.0] — 2026-07-23

The section-split experience is internalized into the write path. Users only express "remember this"; structured long docs are split server-side, unstructured long docs are handed to the calling Agent's own LLM via a `split_request`. The Core never configures or calls an LLM provider, never does mechanical fallback, and never writes `pending`/`fallback_active`.

### Breaking

- **Removed MCP tools `get_sections` and `memory_split_status`.** Their read capability is merged into `memory_get` (new `sections` / `section_ids` params), `memory_search` (returns full matched sections), and `doctor`.
- **`memory_search` partial / zero-match return protocol changed.** Partial hits now return the full matched-section text (`content_scope=matched_sections`); zero-match returns the full memory (`content_scope=full_memory`). Removed `content_omitted` / `content_truncated` / the bounded preview.
- **`memory_get` signature changed** — new `sections` (`none`|`catalog`|`all`, default `catalog`) and `section_ids` params; `matched` is rejected (no search context). Returns a `split` sub-object (`status` / `legacy_status` / `revision` / `section_count` / `content_hash`).
- **Removed config `split_enabled` and `section_zero_match_preview_chars`.** Split capability is now bound to vec readiness. A residual `split.enabled` in config produces a deprecated warning and is ignored (never blocks startup).
- **`memory_status`** no longer shows `split_enabled`; it shows `split_capability = {available, reason}`.

### Changed

- **`memory_split` retained but repositioned** as the Agent internal continuation / history-repair / active-rebuild entry — not the "sole split entry". `prepare` no longer sets `requires_user_confirmation`; active records return `allowed_decision=rebuild` instead of erroring. Provenance is now an explicit caller argument (`parser` for rules, `agent` for `memory_split`), no longer inferred from anchor-vs-heading text. `max_section_chars` is now a hard publish gate (`section_too_large`).
- **`memory_write` / content `memory_edit`** now run a post-write split decision: vec ready + long + a fenced-code-safe Markdown heading plan (2..`max_sections` sections, each ≤ `max_section_chars`) → synchronous rules auto-split; otherwise a full `split_request` (content + snapshot + schema) is returned and `split_status` stays `NULL`. `tags_only` edits are unchanged.

### Added

- **Doctor split checks** (replacing the single `split.enabled` toggle): `split.capability`, `split.long_unsplit_backlog` (active + long + `split_status IS NULL`), `split.failed_count`, `split.legacy_declined`, `split.legacy_unknown_status` (pending/fallback_active surfaced read-only), `split.index_integrity` (active-but-no-sections / <2 / missing section vec / non-positive offsets). Backlog/failed/integrity are n/a when vec is not ready.
- **Unified publish helper** `_publish_sections`, shared by the rules path and the Agent path (replaces the inline validate-then-write block). `rebuild` failures never touch `split_status` — old active sections are preserved.

### Migration

No schema change. Historical `declined` records are read compatibly; unknown legacy statuses are surfaced by doctor for repair. Config migration only warns + ignores the removed keys.

## [0.7.6] — 2026-07-23

### Added

- **`memory_search` conflict signals** — on a genuine query hit (`retrieval_mode=direct`), each result may now carry a `conflict_signal` field indicating it is involved in an unresolved conflict. Two sources are strongly distinguished: `open_table` (from scan/record-verified conflicts, carrying `conflict_type`/`conflict_point`/`suggested_winner`/`confidence_hint`/`source`/`conflict_peer`) and `runtime_metadata_hint` (computed from subject/tags overlap + trust disparity; advisory only, not LLM-verified). Pass `include_conflict_signal=false` to suppress. Open conflicts are batch-fetched in a single SQL (no N+1). If the conflict peer was cut by `limit`, a lightweight peer summary is still attached.
- **`memory_write` write_hints** — after a successful write, the response may carry a `write_hints.possible_supersede_targets` array listing up to 3 active memories that share high subject/tags overlap. Two hint types: `possible_duplicate` and `possible_evolution_of` (new content ≥1.3× candidate length). Hints are advisory only — never written to the conflicts table. Computed synchronously; failures degrade silently (write still succeeds).
- **`memory_edit(tags_only=true)`** — a low-side-effect tag-only edit mode: pass `tags_only=true` with `add_tags`/`remove_tags` to update tags without writing `memory_history`, bumping `version`, re-embedding, or re-splitting. FTS tags are re-synced. `locked`/`user_confirmed` still require `authorized=true` (re-checked inside the transaction to close the TOCTOU window). Idempotent: removing a tag that is already absent returns `no_change` (zero writes).
- **`memory_record_conflict(refresh=true)`** — when an open conflict already exists on the same pair, `refresh=true` updates the enrichment fields in place (returns `refreshed`); `refresh=false` (default) preserves the old `deduped` behavior. Use this in the scan task when re-running LLM after a memory version or model change.
- **Schema** — `conflicts` table gains 5 columns for scan-refresh provenance: `left_version`, `right_version`, `scan_prompt_version`, `scan_model`, `refreshed_at` (idempotent migration). Three ordinary indexes added: `idx_conflicts_status_left`, `idx_conflicts_status_right`, `idx_conflicts_status_created`.

### Changed

- **`memory_record_conflict` docstring** — `conflict_type` semantics expanded: `evolution` now explicitly covers `stale_active_memory` (new version should supersede old but both are still active).
- **`memory_compare` / `memory_arbitrate`** — docstring downgraded to "low-frequency diagnostic / compatibility-retained tool". New conflict workflows should use `scan_conflict_candidates` → `record_conflict` → `list_conflicts` → `supersede`/`resolve`. `memory_arbitrate(mark_conflict=true)` still uses the legacy `record_conflict` path (no enrichment fields); this is documented to avoid confusion with enriched conflicts.
- **`memory_scan_conflict_candidates` / `memory_record_conflict` / `memory_resolve_conflict`** — docstring clarifies these are for agent-side scheduled/manual scan loops, not general conversation tools.

### Removed

- **`memory_complete_open_item`** — the MCP tool entry, `tools.memory_complete_open_item()`, `db.complete_open_item()`, and ~10 associated tests have been removed. Completing a todo is now done via `memory_edit(tags_only=true, remove_tags=["todo"])`, which is strictly lower-side-effect (no history write, no version bump, no re-embedding). Breaking change (0.7.5 had no users on this interface).

### Docs

- `README.md` — feature list and agent instructions updated for conflict signals, write_hints, tags-only edit, and complete_open_item removal. MCP tools table reflects new parameters.
- `docs/INTEGRATION.md` — agent guidance updated for conflict_signal consumption, scan prompt templates (with refresh), batch-arbitration workflow, and tags-only todo completion.

## [0.7.5] — 2026-07-23

### Added

- **Conflict scan (path-B, id=243)** — three new MCP tools for a scan→compare→record→resolve loop, with the core package remaining headless (no LLM, no network):
  - **`memory_scan_conflict_candidates`** — vector-recalls candidate conflict pairs via sqlite-vec KNN. Incremental (only new `id > watermark` + recently edited memories), same-workspace filtered, pair-canonicalised (`left<right`), distance-ranked, `max_pairs` truncated. Writes a diagnostic `scan_log.jsonl` entry for doctor freshness tracking. Returns `scanned=False` with a hint when sqlite-vec is unavailable (config state, not an error).
  - **`memory_record_conflict`** — persists a conflict with enrichment fields (`conflict_type` / `conflict_point` / `suggested_winner` / `confidence_hint` / `source`). Idempotent: a duplicate open pair returns `deduped=True` without writing.
  - **`memory_resolve_conflict`** — closes a single open conflict by `conflict_id` (dismiss a false positive without touching either memory). Distinct from `resolve_conflicts_for` (which closes all conflicts touching a memory).
- **`get_embedding`** (db helper) — reads a memory's embedding back as `list[float]` via `struct.unpack` on the vec0 binary blob (sqlite-vec stores JSON input as packed float32 internally; SELECT returns bytes, not the JSON that was written).
- **Schema** — `conflicts` table gains 5 columns: `conflict_type`, `conflict_point`, `suggested_winner`, `confidence_hint`, `source` (idempotent migration via `_migrate_add_column`).

### Changed

- **`_check_conflicts_open` (doctor)** — rewritten as a three-state sentinel: warns "never scanned" when `scan_log.jsonl` has no `completed` entry; warns "stale" if last scan > 15 days; reports open-count normally when fresh. When sqlite-vec is off, falls back to the legacy table-count behaviour. The old bare `SELECT count(*)` systematically false-negatived once scan became the primary conflict source.
- **Workspace is reserved metadata** (carried over from v0.7.4) — scan candidates are same-workspace filtered at the Python layer (`vec_knn` itself does not filter by workspace).

### Removed

- **`docs/scheduled_conflict_check.py`** — the tag-overlap + `memory_compare` cron script, superseded by the vector-recall MCP tools. `docs/INTEGRATION.md` references updated to point at the new tools (EN + ZH sections).

### Fixed (v0.7.4.1 review follow-ups, bundled into this release)

- `_linked_open_items_for_search` docstring no longer over-promises "single read snapshot" — the bare SELECTs don't share a WAL snapshot; corrected to "best-effort read".
- bm25 legacy path's `retrieval_mode` inference no longer relies on an inline warning literal — extracted to module constant `_NO_DIRECT_MATCH_PREFIX`.
- `authorized` flag documented (README EN/ZH + docstrings) as a "caller-side confirmation gate", not strong authentication.

## [0.7.4] — 2026-07-22

### Added

- **`linked_open_items`** — `memory_search` now attaches up to 5 active todos (memories tagged `todo`) that share meaningful tags with the current result set, in a separate `linked_open_items` field alongside `results`. Pure read-only enhancement; never affects ranking. Fires only on genuine query hits (`retrieval_mode=direct`), never on browse/fallback/empty. A generic-tag stoplist (tag == `todo`, single-char, or appearing in ≥20% of active memories with df≥3) keeps noise out. Failures degrade to `[]` + a warning, never crashing the main search. Pass `include_linked_open_items=false` to suppress.
- **`memory_complete_open_item`** — closes the todo loop: atomically removes the `todo` tag from an active memory (preserving all other tags), writes a `memory_history` snapshot, bumps the version, and re-syncs FTS — all in one `BEGIN IMMEDIATE` transaction (re-read + protection check + writes share the write lock, closing the TOCTOU window). Content/subject/sections/embeddings are never touched. Protected (`locked`/`user_confirmed`) memories require `authorized=true`. An active memory already lacking `todo` returns `already_completed=true` with zero writes (idempotent).
- **`retrieval_mode`** — every `memory_search` response now carries a `retrieval_mode` (`direct` / `recent_fallback` / `recent_browse` / `empty` / `unavailable`) describing how the rows were produced. `search_memories` returns a `SearchOutcome` dataclass instead of a bare 4-tuple; callers use attribute access.

### Changed

- **`workspace` is now reserved metadata and no longer filters results.** This is a **behaviour change**: `memory_search` and `memory_recent` return matches across the whole shared library regardless of the `workspace` argument. The parameter remains in all signatures for interface stability and is still written/returned as a field, but it does not enter any SQL or vector post-filter. memory-arbiter is a shared memory layer — filtering by workspace made cross-project knowledge invisible. If you relied on workspace isolation, filter client-side until an explicit scope API lands.

### Internal

- `json_valid(tags)` SQL guard introduced (first use in the codebase) so malformed-tag rows are silently skipped by the linked-items side query without raising or emitting a warning.
- M1: the stoplist rule is uniform — no longer relaxed when few todos exist.
- M5 hardening: `complete_open_item` uses `write_transaction()` (`BEGIN IMMEDIATE`) instead of a deferred `connection()`, so the re-read and protection check run inside the same locked transaction as the writes. The fallback "no direct match" warning no longer says "from this workspace" (results are library-wide since the workspace change).
- Test coverage: added 4 tests for gaps the original v0.7.4 suite omitted — FTS-failure transaction rollback (no partial write), linked-items sort stability (score → ingest_time → id), duplicate-tag no-inflation, and MCP server-wrapper pass-through of `include_linked_open_items`. 212 tests pass (was 208).

## [0.7.3] — 2026-07-19

### Added

- **Tag scoring via token overlap** (`_score_tags_surface`) replaces contiguous-substring matching on the `tags` field. Query is split on whitespace, both sides normalized (v-prefix stripped on version-like tokens); each query token is matched against the tag set. Pure-CJK tokens use prefix/suffix substring (no middle — blocks bigram-artifact tags like `版历`); ASCII/mixed tokens use equality only (blocks `v0.7` matching `v0.7.0` and mixed-token leakage like `0.7.2发版`). Match ratio → strong/medium/weak/none. A memory whose tags contain both query tokens now reaches `strong` instead of being capped at `medium`. Fixes id=206.
- **`memory_search` filters**: `tags_filter` (AND semantics), `after_time` / `before_time` (ISO 8601), `source_type`. Empty query + filters still applies the post-filter to the recalled pool (filters never recall on their own). Responses now carry `has_more` and `total_estimate` (`has_more = total_estimate > len(reranked)`). `limit` is now page size, not a recall ceiling. bm25 mode warns when filter params are passed (it can't honour them).
- `db.count_filtered_memories` — SQL push-down of the same filters (`json_each` for tags AND, ISO 8601 string compare for time) mirroring the Python post-filter so COUNT and reranked stay consistent.

### Changed

- **Subject anchor overlap tightened.** `classify_match_level` `specific_coverage` threshold `0.4 → 0.6`. Hitting half the query's specific anchors (coverage 0.5) now drops to `weak` instead of `medium` — an incidental-subject record (subject mentions one query word) no longer suppresses a tag-precise record. This is the root-cause fix for the id=206/id=105 dogfooding case.
- **Tag weights parity with subject**: `7/4/1.5 → 10/6/2`, `_TAGS_SCORE_CAP` `7.0 → 10.0` so a `strong` tag score isn't capped below its weight.

### Fixed

- `_score_tags_surface` now skips query tokens that normalize to empty (stray punctuation) before incrementing the denominator, so the match ratio reflects only tokens actually attempted.
- README `source_type` values corrected to match the `SourceType` enum (`user_confirmed` / `agent_generated` / `document_extracted` / `unknown` / `pending`) — the previously documented values (`requirement` / `decision` / `doc_summary` / `research` / `progress`) were silently unfilterable.
- Version sync: `memory_arbiter/__init__.py` was stale at `0.6.6` since v0.6.6 — bumped to `0.7.3` alongside `pyproject.toml`.

### Internal

- 9 targeted unit tests added for `classify_match_level`'s coverage threshold (zero prior coverage) guarding both directions: relaxing reintroduces the id=105 regression, raising breaks single-specific-anchor queries. 176 tests pass.

## [0.7.2] — 2026-07-17

### Improved

- **Doctor: actionable `db.unopenable` hints.** When the CLI can't open the DB, the fallback report now distinguishes two failure modes instead of a generic message:
  - **File does not exist at the resolved path** → points the user at `--db` and `~/.config/memory-arbiter/config.json` (the common case when there's no config.json and doctor defaults to `cwd/memory_arbiter.sqlite3`, which isn't where the real DB lives).
  - **File exists but won't open** → points at corruption / lock / recovery (the original hint).
  The `evidence` now includes `file_exists` and `db_path` so scripts can branch on the cause. No change to path resolution itself — doctor still reads `config.json` > env > cwd via `Settings.from_env()`.

## [0.7.1] — 2026-07-17

### Fixed

- **Doctor: `vec_effective` / `mode` self-consistency.** Previously, when the environment was fully configured for semantic recall (model + `vec.enabled` + extension + auto all on) but the database had never built its `memories_vec` table (e.g. an older DB created before vec was enabled, or a fresh config pointed at an old DB), the doctor report could contradict itself: all 5 vector-chain links green yet `mode=sqlite_vec` while no vec table existed. Now:
  - `vec_effective` requires both the 5-link chain to pass **and** the `memories_vec` table to actually exist (capability ready + data ready).
  - `mode` (summary + `config.degradation_mode` finding) is grounded in the actual table existence, not just the MCP process's startup-time probe (`runtime_state.mode`), which goes stale if the vec table is later dropped or the DB swapped. MCP mode is downgraded to `fts5`/`like` when the vec table is absent.
  - `vec.link3.extension_loaded` notes in its detail when the extension is loadable but the vec table hasn't been created yet, so users understand why `vec_effective` is False despite all-green links.

## [0.7.0] — 2026-07-17

### Added

- **Doctor health diagnostics** — a read-only, one-shot health check for memory-arbiter, exposed as both an MCP tool and a standalone CLI.
  - **MCP tool**: `memory_doctor_overview(deep=false)` returns a graded report. Run it in-conversation to ask "is my setup healthy?".
  - **CLI ambulance**: `memory-arbiter doctor [--json] [--deep] [--db PATH]` works even when the MCP process is down or the DB is read-only — it opens its own read-only connection and never touches the write lock. Exit codes: `0` clean / `1` warnings / `2` criticals (script/CI friendly). If the DB can't be opened at all, it degrades to a single critical report instead of crashing.
  - **18 checks across 5 dimensions**: config integrity (parse warnings, write-probe, degradation mode), the **vector-enablement chain** (5-link short-circuit: model configured → `vec.enabled` → extension loaded → model usable → auto flags — catches the classic "configured a model but recall still doesn't work" case), split state, data consistency (orphaned sections/vectors, version-chain breaks, section-vector coverage), and capacity (conflicts, superseded ratio, history bloat, DB size).
  - Each finding carries a `severity` (`info`/`warning`/`critical`) and a config-specific `fix_hint` — not a flat field dump.
  - **Two-layer error defense**: per-check try/except isolation (one check failing never aborts the other 17) + a platform-entry `except Exception` fallback that guarantees doctor always returns a structured report.
  - `deep=true` additionally loads the GGUF model for a dimension probe; the MCP path reuses an already-loaded embedder at zero cost.
- README: bilingual (EN/CN) sections for the doctor feature (Features list, MCP tools table, dedicated CLI section).

### Changed

- `MemoryDB` gained a `diagnostic_connection()` method — a read-only (`mode=ro`) connection context manager that loads sqlite-vec when available, for doctor's check SQL to run against the vec0 virtual tables. Does not affect existing `connection()` / `write_transaction()` behavior.

## [0.6.3] — 2026-07-15

### Added
- **Channel 6 — section-vec KNN recall.** New recall channel that catches
  long-document dilution: a query semantically matches a late chapter the
  memory-level embedding (truncated to ~3600 chars) never saw. Runs KNN over
  section vectors instead of the single memory vector. Pure gap-filler —
  existing channels are untouched.
- **`section_zero_match_preview_chars` config.** Bounds the zero-match preview
  length (default 2000, clamped [100, 10000]) to prevent token explosion.
- **Section provenance attribution.** Each published section is tagged
  `provenance="parser"` (anchor matches a document heading) or `"agent"`
  (anchor supplied by the caller/LLM). `memory_split` prepare auto-detects
  Markdown headings; callers with structured docs can skip the LLM entirely.

### Changed
- **Zero-match returns a bounded preview, not full text.** When zero sections
  match, `memory_search` returns a truncated preview + section catalog instead
  of the full content. `content_omitted` changes `true→false`; new
  `content_truncated` flag indicates whether the preview was shortened.
- **Long-content penalty exempts split-active memories.** A legitimately
  sectioned long document is no longer penalized for length. Non-split long
  memories are penalized as before.
- **Content normalization in `_attach_sections`.** Channel 6 candidates
  (content="") now get their content filled from `current_mem_map` at the top
  of the result loop, fixing empty-content returns in fulltext/invariant/
  gate-closed branches.
- `vec_knn` (Channel 5) now returns `split_status`, a prerequisite for the
  penalty exemption.

## [0.6.2] — 2026-07-15

### Changed
- **Workspace no longer used as a search filter.** `memory_search` and
  `memory_recent` no longer fall back to `settings.workspace` when the caller
  does not explicitly pass a workspace. All memories are searchable regardless
  of their workspace field. The field is kept on records for future use.

## [0.6.0] — 2026-07-06

### Added
- **Long-document section split.** Memories exceeding `split.threshold`
  (default 4000 chars) can be split into semantic sections with per-section
  vectors. `memory_search` returns only matched section metadata instead of
  the full text. Two-phase: `memory_split` prepare returns content for the
  caller; publish validates offsets and atomically writes sections + vectors.
- New tools: `memory_split`, `get_sections`, `memory_split_status`,
  `memory_rebuild_embeddings`.
- Config: `split.enabled`, `split.threshold`, `section_vec_distance_threshold`
  (calibrated 0.42 on embeddinggemma-300m), `section_fulltext_threshold`,
  `max_sections`, `max_section_chars`.

### Fixed
- Guard against empty embeddings from the never-raises embedder contract.
- Section split state handling hardened; version sync, space-id invariant,
  single-batch protocol.

## [0.5.0] — 2026-06-29

### Added
- **Auto embedding via GGUF.** `embedding.provider = "gguf"` with a local model
  path enables automatic query encoding and write-time embedding — no external
  API calls. sqlite-vec stores vectors locally.
- Config: `embedding.provider`, `embedding.model_path`, `embedding.auto_query`,
  `embedding.auto_write`, vec dim.

## [0.4.0] — 2026-06-21

### Added
- **In-place version chain.** `memory_edit` rewrites content in place; old
  versions are preserved in `memory_history`. `memory_history` traces the full
  edit timeline; `memory_cleanup_history` trims old snapshots.
- README configuration guide with config-file / env-var tables.

## [0.3.1] — 2026-06-15

### Added
- **Optional semantic recall (Channel 5).** `vec_knn` over memory-level
  embeddings surfaces memories with zero lexical overlap. Candidates get a vec
  floor score (2.5) so they beat content-only noise but lose to subject/tags
  hits. `query_embedding` parameter added to `memory_search`.
- Config: `recall_pool_cap`, `content_like_cap`.

### Fixed
- Diagnose vec-disabled cause in status warnings.

## [0.3.0] — 2026-06-14

### Added
- **Wide-recall + soft-rerank.** Multi-channel recall (FTS5 AND, FTS5 OR, LIKE,
  content LIKE) feeds a candidate pool; soft-rerank applies subject/tags/content
  scoring with penalties for noise governance.

## [0.2.6] — 2026-06-10

### Added
- **`memory_supersede`.** Explicit retire with audit trail; bypasses
  user-confirmed protection with authorization. Superseded records sink below
  active in search ranking and are excluded by default (`include_superseded`
  flag restores them for audit walkthroughs).

## [0.2.4] — 2026-06-08

### Added
- `memory_supersede` tool (refined in 0.2.6).

### Fixed
- FTS5 query sanitization.
