# Changelog

All notable changes to memory-arbiter-mcp are documented here.
Versions follow semantic versioning.

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
