# Memory Arbiter MCP

Memory Arbiter（迷码，命令 `mema`）让多个 AI 客户端共享同一个本地 SQLite 事实库，并显式保存来源、事实时间、版本历史、workspace 边界和治理结果。当前文档对应 `0.14.0.dev2` 开发状态。

## 单一 Evidence 主线

每条记忆只保存一份完整原文。后台从 `subject`、Markdown heading、局部语句和必要的重叠窗口生成统一 evidence 向量；语义召回和冲突候选召回共用这一套可重建索引。

## 冲突识别：scan 宽门，notice 严门

Evidence KNN 只负责召回和排序。可选的本地 Qwen2.5-0.5B 对短 pair 分别执行 A→B 与 B→A 抽槽，每次只能输出四个字段：

```json
{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型","value_b":"SQLite"}
```

Qwen 不输出 conflict/coexistence/winner，也不编辑记忆。代码负责校验双向映射、机械属性/值归一、原文 quote grounding、duplicate/compatible guard、作用域与演进等共存条件，以及 `workspace_canonical + entity + attribute + scope` 槽位身份。

- Scheduled scan 是宽门：规则/KNN 基础候选始终保留，任一方向合法可增强召回；单向失败、grounding 弱、entity/scope 不足都保留为 `review_candidate`。Qwen 缺失、超时或非法不会缩小基础候选集。
- Write-time notice 是严门：两方向都必须是合法四字段且交换后映射一致，值严格 grounding、槽位完整、无共存 veto，才可形成用户 notice；否则转入 scan，不向用户倾倒不确定候选。

`notice_sync_wait_ms` 默认 3000 ms，只决定同步返回或已接受的同一任务继续异步，不参与识别；`0` 表示完全异步，queue full 则根本没有已接受任务。`checked_no_notice` 仅表示本次有界检查完成，不代表全库无冲突。evidence/semantic 队列都是进程内队列，崩溃、强制关停、queue full 或模型子进程重启后要按 coverage 恢复：先反复 `rebuild_evidence` 到 eligible coverage 完整/向量 ready，再分页执行 `scan_candidates`。

## 单一冲突表与治理协议

`conflicts` 是唯一冲突事实表。一行代表一次一对多事件，同时保存不可变 member/evidence 快照、value groups、CAS `revision`、最终选择和逐成员应用结果。公开生命周期为：

- `open`：已确认值得治理，可由 scan 追加同槽成员；
- `applying`：已裁决，正在按计划修改成员；
- `resolved`：计划全部完成并复查；
- `not_a_conflict`：该候选快照可共存或是误报。

治理顺序固定为：

1. `memory(action="judge")` 以 `expected_revision` 裁决 `open` 组并进入 `applying`；
2. 按返回计划逐条、串行调用授权的 `memory_govern(action="apply_conflict_action")`，每步使用最新 revision；
3. 全部计划完成后再授权调用 `memory_govern(action="resolve_conflict")`。

部分失败保持 `applying`，返回 `data.action_required="replan_conflict"` 时必须重读 conflict/member，以当前 revision 和新计划调用授权的 `memory_govern(action="replan_conflict")`；旧计划不会被静默重试。普通 update 不能伪造 conflict context 来压制 notice。不存在独立、append-only 的 judgment 公共记录。

## Workspace 归一与隔离

`none`、`weak`、`strict` 三种模式都执行 canonical normalization；归一与 ACL 正交：

- `none`：按与 weak 相同的 exact/confirmed/vector/rule/Qwen 流程归一，但不启用 workspace ACL；未指定 workspace 的查询仍跨全库。
- `weak`：同样归一，并把 workspace 用作软排序和提示信号。
- `strict`：exact/confirmed 和安全机械规则可复用 canonical；Qwen 不得静默合并，新 workspace 保持 pending，需用户确认。

自动 vector/Qwen 结果只写本条 memory 的 `workspace_canonical`，不会生成 confirmed alias。rejected alias 永不被模型重新推荐。workspace Qwen 使用独立 `workspace_qwen_budget_ms`（默认 750 ms）；超时保留 raw canonical 并返回 review hint，不阻塞 notice 门禁。

## 四个产品工具

- `memory`：写入、查询、读取、更新、冲突裁决、状态
- `memory_review`：只读检查、历史、冲突组、审计
- `memory_govern`：需要用户明确授权的治理和逐步冲突应用
- `memory_repair`：evidence 重建、scan/record_conflict、备份恢复、notice 和运行时维护

所有产品响应固定为 `{ok, mode, warnings, degraded, data}`：操作自己的 `action_required/next_action/replan` 在 `data` 内；成功响应可另带顶层 `notices`，notice 的动作在各 `notices[*]` 内。不要把 `action_required` 当成通用顶层字段。`memory(read)` 默认返回全文，也可传严格整数 `span={start,end}` 只读局部窗口；返回切片在 `data.memory.content`，实际范围在 `data.span`。

## 0.14.0.dev2 破坏性升级

当前 runtime 只接受 `conflict_groups_v2`。紧邻的 `local_text_evidence_v1` 和更老的 claim/memory-vector/section-vector 库都会只读识别为 legacy，并统一走公开的旁路 `mema upgrade`，没有公开的原地 conflict-only 升级。

升级必须停止旧 writer，并 drain/停止 semantic worker 后在排他窗口执行。先用 `PRAGMA wal_checkpoint(TRUNCATE)`（busy 必须为 0）再复制主库做回滚备份，不能在 WAL 尚有已提交帧时只复制 `.sqlite3`。升级保留记忆正文/历史、workspace alias/canonical、audit、backup receipt 和逻辑 evidence units，但重建 FTS 与 evidence vectors；旧 `conflicts`、`conflict_judgments`、`semantic_notices` 历史不迁移。目标通过 coverage/fingerprint 和 WAL checkpoint 后才切配置。`--yes` 只跳过“writer 已停且接受历史丢失”的确认，不会替你停进程、checkpoint 或备份。

重建后设置持久化 `conflict_scan_required=true` 和 scan epoch。只有覆盖升级时 active memory 集合、且 detector version 匹配的完整全库 scan 成功后，才能用 CAS 清除该标志；部分分页、失败或旧 detector scan 都不能清除。仍成立的冲突由 scan 重新发现，已经通过正文更新解决的旧冲突不得复活。

详见 [README.md](README.md) 和 [docs/INTEGRATION.md](docs/INTEGRATION.md)。
