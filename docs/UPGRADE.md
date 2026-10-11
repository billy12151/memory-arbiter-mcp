# Upgrading from an older database

> Operational guide, moved out of the README (readme is the product front door). Applies to databases older than schema generation `workspace_state_v1` (pre-0.14.8 claim/vector-era databases are refused at startup without modification).



> ⚠️ **0.14.8 upgrade warning:** current runtime startup accepts only schema generation `workspace_state_v1`. Both `conflict_groups_v2` and `local_text_evidence_v1`, plus older claim/memory-vector/section-vector databases, are refused without modification. Run the public side-by-side `mema upgrade`. Every schema migration declares `vector_effect=preserve|rebuild`; the migrations from the two previous evidence generations preserve vector payloads regardless of current model availability. Compatibility is evaluated separately: a different configured embedding space records `state=mismatch`, disables vector reads, and is repaired later with `memory_repair(rebuild_evidence)`. Both paths compact current workspace redirect/negative-decision state and discard the obsolete workspace decision event ledger.

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
