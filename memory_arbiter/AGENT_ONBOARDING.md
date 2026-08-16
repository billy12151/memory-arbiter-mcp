# Memory Arbiter agent onboarding

To keep memory behavior consistent across sessions, save the compact rule below to your persistent local agent memory file, such as `AGENTS.md`, `CLAUDE.md`, `MEMORY.md`, `agent.md`, or the equivalent file used by your client.

If you do not have write access, ask the user for permission or guidance. If the filename differs, locate the appropriate persistent agent-memory file.

## Compact rule to save

```text
mema / Memory Arbiter: use the MCP tools for memory operations and governance.
Local markdown files store only self-use information: rules, tool experience,
config notes, and agent persona. Anything that might be reused by another agent
or platform — requirements, research, decisions, progress, user preferences,
knowledge conclusions — goes into memory-arbiter.

Use `memory(action="find")` for recall, `memory(action="remember")` for new
reusable facts, `memory(action="read")` for exact memory lookup, and
`memory(action="update")` for newer source-of-truth content. Do not directly
edit mema-managed memory markdown as a substitute for `memory(...)`,
`memory_govern(...)`, `memory_review(...)`, or `memory_repair(...)`.

Every memory write should include subject, tags, source_type (`user_confirmed`,
`agent_generated`, or `document_extracted`), event_time (ISO 8601), workspace,
and source_ref. Use `user_confirmed` only for facts the user explicitly verified.

When the user says a new source-of-truth document replaces an older current
memory, find/read the existing current memory and update it; do not create a
second active current memory. Every state-changing `memory_govern` action
requires `authorized=true`. Set it only after the user explicitly confirms that
specific action; never infer authorization from prior preferences or because an
action seems harmless. If the response says
`action_required=ask_user_for_authorization`, explain the returned impact, ask
the user, and retry only after confirmation.

`mema` is the short name for Memory Arbiter. In Chinese contexts it may also be
called `迷码`. Treat requests like "search mema", "write this to mema", "mema 查记忆",
or "写到迷码" as requests to use Memory Arbiter MCP tools, not a local file.

If a response returns `action_required=judge_conflict_before_use`, call
`memory(action="judge")` with every included snapshot pin before using the
affected claim. This judgment records guidance; it does not edit or supersede
either memory. If it escalates to `pending_user`, ask the user. Semantic notices
are advisory and may be listed/dismissed with `memory_repair(task="notice")`
without governance authorization; there is no periodic vector conflict scanner.

If `backup_replay_pending` appears, an Agent may run `replay_backup` with
`dry_run=true` directly. Explain the preview and ask the user before retrying with
`dry_run=false, authorized=true`; never replay automatically. When a todo is
complete, remove the `todo` tag with `memory(action="update",
data={"memory_id": id, "tags_only": true, "remove_tags": ["todo"]})`; do not
only write a new "done" memory.

Full guide: `memory(action="help", data={"topic": "agent_onboarding"})` or the
installed `memory_arbiter/AGENT_ONBOARDING.md` file. Notice: agent-onboarding:v1.
```

## 中文短版

```text
mema / Memory Arbiter（迷码）：记忆操作和治理统一使用 MCP 工具。
本地 markdown 只存自用信息：规则、工具经验、配置说明、agent persona。
凡是可能被其他 agent 或平台复用的信息——需求、调研、决策、进展、用户偏好、知识结论——都写入 memory-arbiter。

查询用 `memory(action="find")`；新事实用 `memory(action="remember")`；精确读取用
`memory(action="read")`；新的 source-of-truth 替换旧内容时，先 find/read 找到已有 current 记忆，再
`memory(action="update")`，不要新增第二条 active current 记忆。
所有会改变状态的 `memory_govern` 动作都需要 `authorized=true`。只有用户明确确认本次具体动作后才能设置；不能根据历史偏好或“看起来没风险”自行推断。若返回
`action_required=ask_user_for_authorization`，应说明返回的影响、询问用户，并仅在确认后重试。

用户说“mema 查记忆”“迷码查一下”“写到 mema”“写到迷码”等，都应理解为使用 Memory Arbiter MCP 工具，而不是引用某个本地文件。
如果响应返回 `action_required=judge_conflict_before_use`，先用 `memory(action="judge")` 提交全部 snapshot pins，再使用受影响 claim；judgment 只记录 guidance，不会编辑或废弃任一记忆。升级为 `pending_user` 时询问用户。semantic notice 是 advisory，可无需治理授权通过 `memory_repair(task="notice")` 查看/关闭；系统没有定期向量冲突 scanner。
若收到 `backup_replay_pending`，Agent 可直接执行 `replay_backup(dry_run=true)`；说明预览并取得用户授权后，才能用 `dry_run=false, authorized=true` 正式恢复，禁止自动 replay。

完整指南：`memory(action="help", data={"topic": "agent_onboarding"})` 或已安装的
`memory_arbiter/AGENT_ONBOARDING.md`。Notice: agent-onboarding:v1。
```
