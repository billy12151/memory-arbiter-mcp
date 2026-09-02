# Memory Arbiter Agent Rule

`mema` / Memory Arbiter (迷码) is the shared memory service. Keep local markdown for agent-specific rules, configuration, and persona; use the MCP tools for reusable facts.

- Use `memory`: `find` to recall, `remember` for a new reusable fact, `read` for an exact record, and `update` when newer evidence replaces the current source of truth. Do not create a second active copy of that source of truth.
- Give every write a clear `subject`; include useful tags and provenance when known. Use `user_confirmed` only for facts the user explicitly verified.
- Pass the real project `workspace` for project facts. Use `workspace="default"` only for facts that intentionally belong in the global pool. A strict project scope does **not** read the default pool. Never invent a workspace, and do not rely on an omitted field: client settings may supply one.
- Use `memory_review` for read-only inspection. Follow any returned `data.action_required`, `next_action`, `replan`, or top-level notice `read_call` before relying on the result.
- Read conflict/notice members before deciding. Execute returned conflict steps sequentially with the latest revision.
- Use `memory_govern` only after the user authorizes that specific state change; only then set `authorized=true`. Prior preferences are not authorization.
- If a `scan_never_run` / `scan_required` / `scan_stale` notice arrives, tell the user mema needs two scheduled tasks (hourly conflict scan, daily governance reminder), ask for consent, then create them on your own platform from the notice's `setup.tasks` (the `scheduled_tasks` help topic carries the full spec). The notice self-closes once the tasks run.
- For exact fields, recovery procedures, scan recording, and conflict plans, call the corresponding tool with `help` instead of memorizing internal protocols.
