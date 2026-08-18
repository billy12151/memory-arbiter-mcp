# Memory Arbiter Agent Onboarding

Use Memory Arbiter MCP tools for reusable memory. Local markdown should contain only agent-specific rules, tool notes, configuration, and persona.

## Compact Rule

```text
mema / Memory Arbiter (迷码) is the shared memory service.

Use memory(action="find") for recall, remember for new reusable facts, read for
exact lookup, and update when a newer source replaces an existing current
memory. Do not create a second active copy of the same source of truth.

Writes should include subject, tags, source_type, event_time, workspace, and
source_ref. Use user_confirmed only for facts explicitly verified by the user.

Use memory_review for read-only inspection. Use memory_govern only after the
user explicitly authorizes that specific state-changing action. Never infer
authorized=true from earlier preferences.

Memory Arbiter indexes local-text evidence from subjects, headings, and nearby
sentences. Conflict candidates route as notify, check, or ignore. Notify remains
visible without Qwen. Check requires Qwen and becomes ignore when Qwen is not
available. Qwen is only a filter; it never edits memory or confirms a conflict.

A successful product response may include one advisory notice stub. Read the
notice, then execute both returned memory read calls. Only after reading both
complete memories should you tell the user that the candidate appears credible.
Dismiss false positives and resolve handled notices. A notice never edits,
retires, or automatically creates a formal conflict.

Formal conflict judgments use the left and right memory versions as snapshot
pins. A judgment records guidance only. If it requires user action, ask the user.

Preview backup replay with dry_run=true. Apply it only after explicit user
authorization with dry_run=false and authorized=true. Complete a todo by removing
its todo tag instead of writing a separate done memory.
```

## 中文短版

```text
mema / Memory Arbiter（迷码）是共享记忆服务。

查询使用 memory(find)，新增可复用事实使用 remember，精确读取使用 read；新的
source-of-truth 替换旧内容时，先找到现有记忆再 update，不要新增第二条 active 副本。

写入应包含 subject、tags、source_type、event_time、workspace、source_ref。
只有用户明确核实的事实才使用 user_confirmed。

只读检查使用 memory_review。改变状态的 memory_govern 动作必须先取得用户对本次
具体动作的明确授权，不能根据历史偏好推断 authorized=true。

系统从标题、Markdown heading 和局部语句生成统一 evidence。冲突候选分为
notify、check、ignore：Qwen 不可用时 notify 继续提醒，check 降级为忽略。
Qwen 只负责降噪，不编辑记忆，也不确认冲突。

收到 advisory notice 后，先读取两侧完整记忆再判断；误报 dismiss，已处理 resolve。
正式 conflict judgment 只用左右 memory version 做快照固定，judgment 只记录建议。

备份恢复先 dry_run=true 预览，取得用户明确授权后才可正式执行。
```
