# Changelog

All notable changes to memory-arbiter-mcp are documented here.
Versions follow semantic versioning.

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
