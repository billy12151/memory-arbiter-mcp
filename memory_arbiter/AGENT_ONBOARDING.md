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
sentences. Scheduled conflict scan is broad and keeps deterministic candidates
when Qwen is missing or uncertain. A write-time notice is strict: Qwen must
extract attribute_a/value_a/attribute_b/value_b in both directions with
consistent side mapping, grounded values, complete entity/scope, and no
coexistence veto. Qwen never chooses a winner or edits memory.

A formal conflict is one one-to-many row with status open, applying, resolved,
or not_a_conflict. Read conflict_detail and every relevant memory before acting.
For an open group, submit memory(action="judge") with the current revision,
chosen value, decision provenance, resolution memory, and per-member apply plan.
Judge moves the conflict to applying; it does not finish the memory changes.

Follow the returned calls sequentially. For each member, obtain explicit user
authorization and call memory_govern(action="apply_conflict_action") with the
latest expected_revision. Never run old-revision plan steps in parallel. After
all steps succeed, call authorized resolve_conflict. On stale_conflict,
stale_member, or data.action_required=replan_conflict, re-read the group and
members, then call authorized replan_conflict with the current revision and a
replacement plan. Partial failure remains applying; old plan history is kept.

For scan_candidates, use each supplied deep_read span first (omit span when full
context is needed), then call record_conflict with status open or
not_a_conflict; scan output alone is not persisted and may recur. memory(read)
returns operation data under response.data; a span read places clipped text at
data.memory.content and bounds at data.span.

Product responses use {ok,mode,warnings,degraded,data}. Operation action_required
is under data; successful delivery side channels are top-level notices, and each
notice carries its own action_required/read_call. A notice is a strict frozen
snapshot; handle it through those instructions, not the old pair workflow.

Workspace canonical normalization runs in none, weak, and strict modes. none has
no workspace ACL but still normalizes; an unscoped query remains whole-library.
strict never lets Qwen silently merge a new workspace. Automatic vector/Qwen
normalization does not create a confirmed alias; only explicit governance does.

For confirm_new_workspace, ask the user before calling confirm_pending_workspace
with authorized=true. For ask_user_for_authorization, explain the returned impact
and retry with authorized=true only after specific approval. Govern confirm
promotes one memory to user_confirmed; confirm_pending_workspace instead confirms
a strict-isolation workspace and activates its pending memory.

Evidence and semantic queues are process-local. After queue loss/full, forced
restart, or index uncertainty, inspect coverage, run rebuild_evidence until no
pending ids remain and vectors are ready, then paginate scan_candidates.

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

scheduled scan 是宽门：Qwen 缺失或不确定时仍保留规则/KNN 基础候选。write-time
notice 是严门：Qwen 必须双向输出 attribute_a/value_a/attribute_b/value_b，交换方向后
映射一致、值有原文 grounding、entity/scope 完整且无共存 veto 才提醒。Qwen 不选
正确值，也不修改记忆。

正式冲突是一条一对多记录，状态为 open、applying、resolved 或 not_a_conflict。
先读 conflict_detail 和所有相关记忆，再以当前 revision 调 memory(judge)；judge 记录
chosen value 和逐成员计划并进入 applying，不代表修改已完成。随后按返回调用逐条、
串行、经明确授权执行 memory_govern(apply_conflict_action)，每一步使用最新 revision；
全部成功后才 authorized resolve_conflict。stale_conflict/stale_member 或返回
 data.action_required=replan_conflict 时，重读后按当前 revision 授权调用 replan_conflict
替换计划；部分失败保持 applying，旧计划历史保留。

scan_candidates 的每个已分诊候选先按 deep_read span 读局部（需要全文时去掉 span），
再调用 record_conflict(open|not_a_conflict)，否则下轮仍可能出现。产品响应固定为
{ok,mode,warnings,degraded,data}：操作 action_required 在 data，成功响应的顶层 notices
数组中每条 notice 自带 action_required/read_call。notice 不走旧的 pair judgment。

none、weak、strict 都执行 workspace canonical normalization；none 只是不做 ACL，未传
workspace 的查询仍跨全库。strict 不允许 Qwen 静默合并。自动 vector/Qwen 结果不会
创建 confirmed alias，只有显式用户治理可以。

confirm_new_workspace 应先取得用户授权，再用 authorized=true 调
confirm_pending_workspace；ask_user_for_authorization 应说明 impact，取得本次明确授权后
才按 retry 重试。evidence/semantic 队列是进程内状态；queue full/丢队列/强制重启后，
按 coverage 反复 rebuild_evidence 到向量 ready，再分页 scan_candidates。备份恢复先
 dry_run=true 预览，明确授权后才正式执行。
```
