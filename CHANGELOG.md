# Changelog

All notable changes to memory-arbiter-mcp are documented here.
Versions follow semantic versioning.

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
