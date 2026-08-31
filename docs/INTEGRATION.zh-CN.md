# 集成指南

**[English](INTEGRATION.md) | 中文**

本指南描述 `0.14.12` 的正式契约。

## MCP 接口面

把命令配置为 `mema`，使用四个产品工具。用 `memory(action="help")` 或对应工具的 help 发现当前字段。

stdio 是默认传输。要让多个本地客户端共享一个社区版进程，设置 `mcp.transport="streamable-http"`，把每个客户端连接到 `http://127.0.0.1:8000/mcp`（地址/端口用 `mcp.http.host`/`mcp.http.port` 配置；接口路径固定为 `/mcp`）。HTTP 按无状态请求处理（固定行为）：记忆和 semantic notice 保存在 SQLite 中，异步任务产生的 pending notice 会由后续任一次成功工具调用领取，不依赖发起任务时的 MCP 连接；服务重启也不会让客户端卡在已失效的内存 session。进程重启仍可能中断尚未把 notice 落库的 worker 任务。

无论哪种传输，服务器都要求显式配置身份：在 config.json 里设置 `client` 和 `agent_id`，或使用环境变量 `MEMORY_ARBITER_CLIENT`/`MEMORY_ARBITER_AGENT_ID`（保留的 6 个启动上下文变量之二，见「配置面」）。没有内建默认值，任一为空都会拒绝启动。stdio 下该配置身份即进程级调用身份——归因和策略判定都用它，工具 `data` 里的 `agent_id`/`client` 字段不会被当作身份来源。streamable-http 则以下述逐请求头为调用身份。

在每个 MCP server 配置中设置固定的 `X-Mema-Client` 和 `X-Mema-Agent-Id` 请求头；客户端随后会在 initialize、工具发现和工具调用时自动携带。**不要**让 agent 往工具 `data` 里动态附加身份。任一请求头缺失、为空、非法、重复或与工具数据冲突时，HTTP 模式都会 fail closed。它只绑定 loopback：请求头提供的是本地来源标记和策略输入，不是认证、租户隔离，也不是把服务暴露到公网的许可。

写入要求非空 `subject`；已知时带上 `source_type`、`event_time`、`source_ref` 和有用的 tags。项目事实传真实的项目 `workspace`。只有刻意要存进全局池的事实才显式传 `workspace="default"`；不要依赖省略，因为客户端配置可能已提供 workspace。strict 隔离要求必须有 workspace。`user_confirmed` 只用于用户显式验证过的事实。当新来源替换已有 current 来源时，先 find/read 找到原记忆再做 `update`，而不是创建第二条 active 副本。

## 配置面

0.15.0 起**配置只认文件**：所有用户可调项都在 `~/.config/memory-arbiter/config.json`（或 `MEMORY_ARBITER_CONFIG` 启动上下文变量指向的文件）。完整配置面共 18 键：

```json
{
  "db_path": "…", "backup_jsonl": "…",
  "client": "…", "agent_id": "…", "workspace": "default",
  "isolation": "none", "policy_path": null,
  "update_check": {"enabled": true},
  "embedding": {"model_path": "…", "auto_query": true, "auto_write": true},
  "semantic_conflict": {"enabled": true, "model_path": "…", "on_write": "async", "max_notice_pairs": 2},
  "mcp": {"transport": "stdio", "http": {"host": "127.0.0.1", "port": 8000}}
}
```

意图语义：

- `embedding.model_path` 指向本地 GGUF 模型就是启用 sqlite-vec 证据召回的唯一意图——不再有 `vec.enabled`/`embedding.provider`/`vec.dim`。向量维度取自模型本身；数据库把活跃维度记录为库内事实源，换成不同输出维度的模型时会在启动时按新维度 DROP 并重建向量表，索引翻为 `state=mismatch`，待全量重建把数据重新发布进新表。
- `semantic_conflict.model_path` 指向本地 Qwen2.5-0.5B GGUF 即启用语义冲突运行时，并在启动时加载、常驻不卸载（`preload`/`resident` 冻结为 true）。`semantic_conflict.enabled=false` 是显式关闭的逃生口；不设 + 有 `model_path` 即视为启用。
- 排序固定为 hybrid（字面 + 证据倒数排名融合），没有排序模式可选。
- HTTP 接口路径固定 `/mcp`，请求体上限固定 4 MB。

环境变量只保留 6 个启动上下文：`MEMORY_ARBITER_CONFIG`、`MEMORY_ARBITER_DB_PATH`、`MEMORY_ARBITER_BACKUP_JSONL`、`MEMORY_ARBITER_MCP_TRANSPORT`、`MEMORY_ARBITER_CLIENT`、`MEMORY_ARBITER_AGENT_ID`。它们只选择进程上下文（用哪个配置文件、哪个库、哪种传输、什么身份），同名时以配置文件为准。**其余 `MEMORY_ARBITER_*` 变量一律不再读取**——残留的导出会在 `mema doctor`、控制台设置页和 `memory(action="status")` 里收到 "no longer read" 警告；读/检索响应不再携带配置期警告。config.json 里出现被删文件键同样会提示 "no longer configurable" 并被忽略。

### 0.14 → 0.15 键迁移

| 旧键（0.14.x） | 处置 |
| --- | --- |
| `vec.enabled`、`embedding.provider`、`embedding.model_path` 的 env 别名（`MEMORY_ARBITER_GGUF`） | 语义合并——`embedding.model_path` 是唯一意图 |
| `vec.dim` | 删除——维度取自模型 / 库内 `active_dim` 事实源 |
| `ranking_mode`（env） | 删除——排序固定 hybrid |
| `tool_profile` | 删除——四个产品工具是唯一接口面 |
| `semantic_conflict.backend`、`semantic_conflict.max_concurrency` | 删除——死旋钮（单一本地后端、串行 worker） |
| `semantic_conflict.preload`、`semantic_conflict.resident` | 常量冻结为 true——配置了模型即启动加载并常驻 |
| `semantic_conflict.n_ctx` / `n_threads` / `n_batch` | 常量冻结（1024 / 4 / 128） |
| `semantic_conflict.job_timeout_ms` / `inference_timeout_ms` / `load_timeout_ms` / `min_pair_budget_ms` | 常量冻结（5000 / 30000 / 120000 / 1000 ms） |
| `semantic_conflict.queue_max_size`、`semantic_conflict.max_evidence_units` | 常量冻结（100 / 24） |
| `semantic_conflict.scan_enhance`、`semantic_conflict.scan_max_pairs`、`semantic_conflict.scan_budget_ms` | 常量冻结（true / 8 / 60000） |
| `semantic_conflict.notice_sync_wait_ms`、`semantic_conflict.workspace_qwen_budget_ms` | 常量冻结（5000 / 750 ms） |
| `embedding.n_ctx`、`embedding.reserved_tokens`、`embedding.max_unit_chars` | 常量冻结（2048 / 64 / 3600） |
| `workspace_match_distance`、`workspace_qwen_candidate_distance`、`workspace_qwen_candidate_top_k` | 常量冻结（0.25 / 0.25 / 3） |
| `workspace_weak_vector_weight`、`workspace_min_name_len`、`workspace_recall_admission`、`workspace_recall_cutoff` | 常量冻结（false / 3 / true / 0.25） |
| `recall_pool_cap`、`content_like_cap`、`superseded_limit` | 常量冻结（50 / 30 / 20） |
| `mcp.http.path`、`mcp.http.stateless`、`mcp.http.json_response`、`mcp.http.max_request_body_size` | 常量冻结（`/mcp` / 无状态 / 4 MB） |
| 其余所有 `MEMORY_ARBITER_*` 环境变量 | 不再读取（doctor/控制台/status 警告） |

冻结常量位于 `memory_arbiter/constants.py`，取值为原默认值；升级前自定义过其中某项的用户，升级后应预期回到原默认行为；更换 embedding 模型导致引擎空间变化时，仍会像以前一样触发一次 mismatch/重建。

## 证据召回

配置 sqlite-vec 和本地 embedding 模型后，写入会异步发布从已存原文派生的句子/段落级证据。字面和证据通道独立召回，按记忆用倒数排名融合合并，再经过信任、时间、过滤和 workspace 调整。证据偏移量定位相关原文。

`memory(action="read", data={"memory_id":42})` 返回完整原文。加 `"span":{"start":120,"end":640}` 则只返回 `data.memory.content[120:640]` 外加 `data.span.{start,end,total_chars}`。边界必须是严格 JSON 整数且满足 `0 <= start < end`；超大的 end 裁剪到正文长度，而 start 超出正文会报错。`scan_candidates.deep_read` 可能提供可直接使用的 span；语义 notice 的 read call 按设计是整记忆读取，需要完整上下文时不要带 span 参数。

用 `memory_repair(task="rebuild_evidence", data={"dry_run":true})` 检查缺失/过期的覆盖率，再排队有界的重建页。重复执行 execute/dry-run 直到没有剩余 id、status 报告合格覆盖完整。embedding 空间变化会报告 `state=mismatch` 并停用证据通道，直到每条合格记忆都重新发布。证据单元和向量都是派生的。升级只在源向量索引为 `ready` 且 active space ID 与当前模型/管线空间完全一致时复用；否则在当前空间重建索引。

## 冲突检测契约

### 双向四字段抽取

证据 KNN 只提供有界的短 pair 召回和排序。可选的本地 Qwen2.5-0.5B 分别以 A→B 和 B→A 各跑一次。每次结果必须是恰好四个有界字符串字段的严格 JSON 对象：

```json
{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型","value_b":"SQLite"}
```

模型不得输出最终的 conflict/coexistence 判定、赢家或任何修改。代码校验：

1. 每个方向都有具体的同属性/不同值抽取；
2. 交换两侧后，两个方向在归一属性和值到来源的映射上互相一致；
3. 值在对应证据引用中有 grounding，机械的大小写/单位/数字/已确认别名推导除外；
4. 归一后的值确实不同；
5. 确定性的重复、兼容、环境/版本/地域/对象、观察时间、历史/当前、演进和测量范围 veto 均不命中；
6. 正式槽位具备充分的 `workspace_canonical + entity + attribute + scope` 来源。

模糊的属性相似不能创建正式槽位。Qwen 失败或缺席无权否决确定性的扫描候选。

### 定时扫描：宽门

`memory_repair(task="scan_candidates")` 枚举有界的 KNN/规则候选，不需要把整个库读进 agent 会话。扫描保留确定性基线，并且当本地 Qwen 运行时可用时（扫描增强恒开，冻结常量），对该页执行有界增强：规则候选被丰富为带抽取的 `attribute/value` 成员字段和 `value_groups`；原本需 `include_check` 显式开启的纯相似 pair，只要在任一方向抽取出合法的同属性/不同值，就并入 `candidates`。每页 Qwen pair 评估数上限 8、单页截止时间 60 s（冻结常量）；元数据 `entity/scope` 一致的已验证候选聚合为 `slot_groups`。单向输出、grounding 弱或 entity/scope 缺失保留为 `review_candidate`，供 agent 深读；Qwen 缺席/非法/超时/预算失败永远不会缩小基线集合，也不会移除任何规则候选。

候选携带成员版本、证据 span、候选身份和深读调用。`scan_candidates` 本身不持久化分诊结果。对每个已复查候选，调用 `memory_repair(task="record_conflict")` 并传 `status="open"` 或 `status="not_a_conflict"`；否则它可能在后续扫描中再次出现。`slot_key` 只在 `status="open"` 时传——`not_a_conflict` 分诊仅通过 `candidate_key` 记录；若该槽位已有 open 组，会返回 `open_group_exists`。仅候选的 `not_a_conflict` 行使用 `candidate_key`，不会虚构 `scope="unknown"`。

### 写入时 notice：严门

一条用户可见 notice 要求：两个方向都合法、方向映射一致、严格引用 grounding、归一值确实不同、槽位来源完整、无共存 veto。任何失败都会关闭 notice 路径，把该案例留给定时扫描复查。Notice 快照冻结成员版本、值分组、槽位来源、detector/prompt 版本、任务 id 和去重键。

写入成功后，服务器最多等待 5000 ms（冻结的投递期常量）以完成有界的 notice 任务。等待超时后写入照常成功返回，同一个已接受任务继续异步执行——不会被取消或重算。队列满/入队被拒是另一回事：那时根本没有可等待任务。`checked_no_notice` 只表示该有界写入时任务内的每个候选都完成了严门检查，**不是**全库无冲突的声明。定时扫描仍是持久的召回兜底。

job 预算（5000 ms，冻结）是队列公平预算，不是推理超时。只有后面已有其他 semantic job 等待时才启用，并且只在候选 pair 之间检查。已经开始的 Qwen 请求只受推理超时（30000 ms，冻结）约束；即使 job 预算期间耗尽，也会先完成当前 pair，再在开始下一 pair 前让出 worker。没有积压时，job 预算不生效。

## 单一冲突表

没有独立的公开裁决存储。一行 `conflicts` 代表一次一对多事件，保留其不可变检测快照、`value_groups`、裁决和应用结果。重要的身份/并发字段包括 `revision`、`workspace_canonical`、`slot_key`、`slot_key_hash`、`candidate_key`、`candidate_key_hash`、`member_versions` 和 `member_fingerprint`。

公开生命周期：

- `open`：已确认值得治理；scan 可以用 `expected_revision` CAS 为同一槽位追加新成员。
- `applying`：已有裁决和应用计划；普通成员不能再追加，重复提醒只对已验证的计划动作抑制。
- `resolved`：所有计划修改完成并复查通过；事件关闭，永不扩张。
- `not_a_conflict`：该候选快照经复查为可共存/误报。

同一 canonical workspace+slot 只能存在一条 `open`/`applying` 行。解决之后出现新值/新版本会创建新事件。

## 裁决工作流

1. 读 `memory_review(view="conflict_detail")` 和所有相关记忆。展示共享槽位、每个值分组及其成员。
2. 用当前 `expected_revision`、选定值、裁决依据/理由、resolution memory 和逐成员计划调用 `memory(action="judge")`。这一步把 `open` 推进到 `applying`。
3. 跟随返回的机器调用。一次执行一条带授权的 `memory_govern(action="apply_conflict_action")`，每步使用最新 revision。永远不要并发执行预计算步骤。
4. 遇到 `stale_conflict` 或 `stale_member` 时重读/重计划。失败的 apply 返回 `ok=false` 及 `data.action_required="replan_conflict"` 和 `data.replan`；部分失败保持 `applying`。重读冲突组和成员后，用当前 revision、替换 `apply_plan` 和可选的替换 `resolution_memory_id` 调用授权的 `memory_govern(action="replan_conflict")`。replan 保留之前的计划历史并递增 revision。
5. 当前计划的每一步都成功后，用最新 revision 调用授权的 `memory_govern(action="resolve_conflict")`。

优先修正不正确的 current 事实；历史/外部/锁定的上下文酌情保留；可能时复用一条已存在的正确 active 记忆作为 `resolution_memory_id`。普通 `memory(update)` 不能把 conflict id 当作 notice 抑制开关使用。

## Workspace 归一与 ACL

Canonical 归一在每种隔离模式下都会运行，与 ACL 相互独立：

- `none`：精确/确认/向量/规则/Qwen 归一行为与 weak 相同，但不应用 workspace ACL。省略 workspace 过滤时返回所有 workspace。
- `weak`：同样的归一，外加软性排序/提示；没有硬可见性过滤。
- `strict`：精确/确认和安全的机械规则可以复用 canonical；Qwen 不能静默合并。新 workspace 保持 pending，直到授权的 `confirm_pending_workspace`。可见性使用 guarded 向量准入（0.15.0 起恒开，冻结常量）：workspace 敏感的 recall/read/repair 操作、冲突/notice 工作流和 console 内容/计数视图共享调用方 canonical 加上所有在守卫余弦距离（0.25，冻结）之内、且通过 default 池、短名称和泛化子串 guard 的 canonical。进程级维护（如语义运行时控制、备份回放、doctor、settings）不按此限定。向量缺失或 sqlite-vec 降级时回退到精确 canonical 作用域；绝缘的 `default` 池永远不会被准入 strict 项目作用域。

在 `none` 和 `weak` 中，首次写入并注册 canonical workspace 时，响应返回非阻断的顶层 notice：`type=workspace_review`、`action_required=review_workspace_registry`，并附带 doctor 复查调用和需要用户另行授权的 `confirm_workspaces` 调用。重复写入已有 canonical 不重复提醒。`strict` 使用原有 pending workspace 阻断流程。

解析顺序是：内部已确认/负向 workspace 决策 → 精确 canonical → 有界向量候选 → 确定性 `AUTO|KEEP|ASK` → 仅对未决近似项调 Qwen。Qwen 必须从提供的候选中选择，可以提示 `alias`/`typo`/`same_project` 关系，但自动归一只写记忆的 `workspace_canonical`，不创建持久 redirect。负决策抑制重复提议。产品治理使用 rename、migrate、pending 确认和全注册表复查；内部决策行不是产品工作流。

Workspace 和冲突推理共享一个串行本地 worker。近似匹配 Qwen 预算（750 ms，冻结）是独立的短预算：超时/忙碌时保留原始 canonical 并返回复查提示，而不是阻塞写入时 notice 门。

## 响应信封

四个产品工具都返回 `{ok, mode, warnings, degraded, data}`。操作结果和所需步骤在 `data` 下，例如 `data.action_required`、`data.next_action` 或 `data.replan`。仅在成功响应上，投递侧通道可以附加顶层 `notices` 数组。语义 notice 存根的 `action_required` 和 `read_call` 位于该 notice 项内部。客户端应同时检查 `response.data.action_required` 和每个 `response.notices[*].action_required`；没有通用的顶层 action-required 字段。

## Notice 生命周期与恢复

Notice 投递是原子、尽力而为的数据库认领，不是传输层 exactly-once。内部的 `pending` 和 `delivered` 对外都显示为 `open`；附加存根把 `pending` 变为 `delivered` 并设置 `delivered_at`。公开的终态动作是 `dismiss`（误报，冲突变为 `not_a_conflict`）和 `resolve`（已处理）。如果投递前任一钉住的记忆快照已变化，服务器可能内部把该 notice 标记为 `stale`。投递本身不修改记忆、不裁决/解决正式冲突，也不会 supersede 任何一侧。注意：被归类为只读的接口仍会推进投递状态——notice list/read 和 `memory(action="status")` 的投递会认领 pending notice（`pending → delivered`，或钉住快照已变时标记 `stale`）；对被策略拒绝的 HTTP 身份也是如此，因为它们按设计是只读的。

证据和语义队列是进程内的。崩溃、强制关停、队列满丢弃或模型子进程重启都可能丢失未处理的索引/分类工作，即使记忆写入已提交。恢复是覆盖率驱动的：查看 `memory(action="status")`，反复运行 `rebuild_evidence` 直到 dry-run 为空、合格覆盖完整、向量状态 ready；然后分页 `scan_candidates` 并记录每个已复查候选。不要假设重启能重建旧队列。`rebuild_evidence` 恢复源派生单元/向量；随后的扫描找回错过的冲突发现。

## 备份与升级

JSONL 回放无需授权即可预览。应用回放需要显式用户授权，并按回放键/载荷哈希幂等。JSONL 只存记忆记录及其选定的 canonical；不恢复内部 redirect 或负决策。需要 workspace 决策状态存活时，保留 SQLite 源库/升级产物。

### 升级矩阵

| 源数据库 | 运行时行为 | 公开升级路径 | 保留 / 重建 |
| --- | --- | --- | --- |
| 新建/空库或 `workspace_state_v1` | 正常启动 | 无 | 现有 current 数据 |
| `conflict_groups_v2` 或 `local_text_evidence_v1` | 原样拒绝启动 | Side-by-side `mema upgrade`；仅空间完全匹配时走 conflict-only | 核心/公开数据保留；冲突历史和旧 workspace 决策事件省略；空间兼容时复用 FTS/证据/向量，否则重建 |
| 更老的 claim + memory/section-vector 代次 | 原样拒绝启动 | 同样的 side-by-side `mema upgrade` | 核心/公开数据保留；证据单元生成/保留，向量重建 |
| 未知、不完整、失败/恢复中的目标库 | 拒绝 | 用 `mema doctor --json` 诊断；只能用低层迁移工作流修复/续跑 | 验证成功前永远不作为 current 打开 |

没有公开的原地升级：两条路径都构建并验证一个 side-by-side 目标。完整重建需要 sqlite-vec、可读的已配置 GGUF embedding 模型、`llama-cpp-python`（`semantic-local` extra 同时提供 embedding 运行时）、可写目标目录和足够磁盘。只有源向量状态为 `ready` 且 active space ID 与当前模型/管线完全一致时，conflict-only 路径才复用现有证据/向量且不加载模型。独立的可选语义冲突 Qwen 模型不是迁移前置条件。

### WAL 安全流程

1. 用 `mema upgrade --dry-run` 预览。
2. 停掉所有旧版本写入方/客户端，排空或终止语义 worker。
3. 只在 checkpoint 已停止的源库之后创建回滚副本：

   ```bash
   sqlite3 /absolute/path/to/memory.sqlite3 "PRAGMA wal_checkpoint(TRUNCATE);"
   cp /absolute/path/to/memory.sqlite3 /absolute/path/to/memory.pre-0.14.sqlite3
   ```

   checkpoint 报告非零 `busy` 时中止；当 `-wal` 中还有已提交帧时只拷贝主文件是不完整的。关闭前用 SQLite 的在线 `.backup` 命令也是安全的。
4. 运行 `mema upgrade`；重启并用 `mema doctor --json` 验证。
5. 完成 status/doctor 显示的带 epoch 固定的完整 `scan_candidates` 全库扫描。

side-by-side 拷贝保留记忆内容/历史、备份回放回执、workspace canonical 和当前 redirect/负决策状态及审计。已废弃的 workspace 决策事件账本被刻意省略。每个结构迁移显式声明 `vector_effect=preserve|rebuild`：preserve 原样复制 FTS/证据/向量 payload，只替换冲突域，再独立判断空间兼容性；不兼容时写入 `mismatch` 并禁用向量读取，等待后续重建，不阻断结构迁移。它刻意以空的冲突/notice 状态开始：旧 `conflicts`、`conflict_judgments` 和 `semantic_notices` 历史不迁移。目标发布要求指纹稳定、破坏性表为空、generation 与完成时间在同一事务提交，并在移除 WAL/SHM、切换配置前完成 `wal_checkpoint(TRUNCATE)`；rebuild 路径额外要求合格证据全覆盖且目标向量状态在预期空间为 `ready`。

成功时记录 `conflict_scan_required=true` 和一个持久 epoch。只有覆盖升级时活跃集且 detector 匹配的完整扫描才能 CAS 清除它。部分页、失败扫描或旧 detector 都不能。`--yes` 只跳过"确认写入方已停止、冲突/裁决/notice 历史永久丢失"的提示；它不会停止进程、checkpoint/备份源库，也不放松验证。`--no-switch` 不动配置。被环境变量覆盖/不匹配的配置路径需要按打印的手动步骤切换。
