# Integration Guide / 集成指南

**[English](#english) | [中文](#中文)**

---

<a id="english"></a>

## English

Memory Arbiter is a local fact-governance and targeted-retrieval layer for MCP clients. The default surface has four product tools: `memory`, `memory_review`, `memory_govern`, and `memory_repair`. Low-level compatibility tools are available under `MEMORY_ARBITER_TOOL_PROFILE=legacy_full`/`full`, except for the removed vector conflict scanner.

### Configuration and dependency boundary

Resolution order is `MEMORY_ARBITER_CONFIG` → `~/.config/memory-arbiter/config.json` → environment/defaults. Keep shared database/vector/model settings in the user-owned config file and per-client identity in each MCP env block.

The default core needs no model. Lexical recall, structured-claim detection, governance, history, and repair work without sqlite-vec or GGUF. Optional dependencies:

```bash
pip install 'memory-arbiter-mcp[vec]'             # sqlite-vec semantic/section/workspace candidate recall
pip install 'memory-arbiter-mcp[semantic-local]'  # local GGUF runtime
```

On Python 3.13, `[semantic-local]` may compile `llama-cpp-python` if no matching wheel is available; install a C/C++ toolchain and CMake when needed.

Key configuration:

| JSON path | Env fallback | Default | Meaning |
|---|---|---|---|
| `db_path` | `MEMORY_ARBITER_DB_PATH` | `./memory_arbiter.sqlite3` | Shared SQLite location. |
| `backup_jsonl` | `MEMORY_ARBITER_BACKUP_JSONL` | `./memory_arbiter.backup.jsonl` | Schema-1 append fallback when SQLite is unavailable or unwritable. |
| `structured_claim_mode` | `MEMORY_ARBITER_STRUCTURED_CLAIM_MODE` | `beta_all` | Deterministic write/edit claim gate; `off` is an emergency kill switch. There is no periodic vector scan fallback. |
| `isolation` | `MEMORY_ARBITER_ISOLATION` | `none` | `none`, `weak`, or `strict`. |
| `workspace_match_distance` | `MEMORY_ARBITER_WORKSPACE_MATCH_DISTANCE` | `0.25` | Cosine-distance cutoff for optional workspace candidates, not final authority. |
| `vec.enabled` | `MEMORY_ARBITER_ENABLE_SQLITE_VEC` | `false` | Enable sqlite-vec. |
| `vec.dim` | `MEMORY_ARBITER_VEC_DIM` | `768` | Must match the embedding model. |
| `embedding.model_path` | `MEMORY_ARBITER_EMBEDDING_MODEL_PATH` | none | Local embedding GGUF; provider is inferred as `gguf`. |
| `embedding.auto_query` / `auto_write` | matching env vars | `true` / `true` | Automatically embed queries/writes when the embedding path is configured and vec is ready. |
| `semantic_conflict.enabled` | `MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED` | `false` | Optional write-time semantic notices. A configured model path auto-enables unless explicitly false. |
| `semantic_conflict.model_path` | `MEMORY_ARBITER_SEMANTIC_CONFLICT_MODEL_PATH` | none | Local classifier GGUF. Current backend: `local_gguf`. |
| `semantic_conflict.pair_text_gate` | `MEMORY_ARBITER_SEMANTIC_CONFLICT_GATE` | `medium` | `medium` or lower-noise `strong`. |
| `semantic_conflict.on_write` | `MEMORY_ARBITER_SEMANTIC_CONFLICT_ON_WRITE` | `async` | `async` or `off`; writes do not wait. |
| `semantic_conflict.queue_max_size` | matching env var | `100` | Bounded queue; same-memory jobs coalesce. |
| `semantic_conflict.candidate_limit` / `pair_limit` | matching env vars | `30` / `10` | Metadata recall bound and pair bound per job. |
| `semantic_conflict.n_ctx` / `n_threads` / `n_batch` | matching env vars | `1024` / `4` / `128` | Local GGUF sizing. |
| `semantic_conflict.resident` / `preload` | matching env vars | `true` / `false` | Retain after use / load at startup. |
| `semantic_conflict.job_timeout_ms` | matching env var | `5000` | Between-pair job budget. It does not interrupt an inference already started. |
| `semantic_conflict.inference_timeout_ms` | matching env var | `30000` | Hard timeout for one started inference. |
| `semantic_conflict.load_timeout_ms` | matching env var | `120000` | Separate model startup/load timeout. |
| `semantic_conflict.min_pair_budget_ms` | matching env var | `1000` | Minimum remaining job budget before starting another pair. |

`semantic_conflict.max_concurrency` is reserved and clamped to `1`. Runtime/backend status, generation, PID/in-flight state, timeout/restart counters, and budgets are exposed through `memory(action="status")` and `memory_repair(task="semantic_control", data={"action":"status"})`.

See [`examples/memory-arbiter.config.example.json`](../examples/memory-arbiter.config.example.json) for the complete sample.

### Pattern A — Targeted retrieval

```text
user asks a question
  → memory(action="find", data={"query": "specific subject/version/decision"})
  → use only the returned active facts
```

Use `tags_filter`, `source_type`, `after_time`, and `before_time` when precision matters. Under `strict`, always send the caller workspace; by-ID/detail paths are also workspace-filtered.

### Pattern B — Write-time conflict handling

There is no periodic or incremental vector conflict scanner.

1. **Deterministic structured claims (default).** After a write/edit, the server extracts conservative explicit claims and compares the same canonical `entity + attribute + scope`. A collision is persisted as `pending_llm` with memory-version and claim-revision snapshot pins.
2. **Optional semantic notices.** After a successful write, an async worker performs specific bounded candidate recall with subject/tag ranking, bounded pair selection, local Qwen GGUF classification, then a deterministic pair-text gate. Noisy/common tags are suppressed, specific tags and subject fallback remain eligible, and ranking occurs before `pair_limit`. Model output must satisfy the strict semantic JSON schema. The `medium` pair-text gate is the default; `strong` is the conservative option, selected with `semantic_conflict.pair_text_gate` or `MEMORY_ARBITER_SEMANTIC_CONFLICT_GATE`. Passing candidates become open `semantic_notices`; they are review hints, not conflict judgments.
3. **Manual/on-demand review.** Agents may inspect `memory_review(view="conflicts")`, compare evidence through compatibility tools, and record/resolve conflicts explicitly.

When a product response returns `action_required=judge_conflict_before_use`, call `memory(action="judge")` with all returned pins before using the affected claim. The judgment receipt records guidance only and never edits or supersedes either memory. If it returns `pending_user`/`ask_user`, the user is the final authority. State-changing governance remains separately authorized.

The semantic worker is strictly serial in one child process. The write path is fail-open and asynchronous. Historical jobs with stale memory/version/claim snapshots are skipped. The 5 s default job budget only decides whether another pair may begin; each started inference has a 30 s hard timeout, and loading has a separate 120 s timeout. A timed-out generation is terminated before another process generation starts.

On the next successful call to any of the four product tools, the server may append at most one compact semantic stub to top-level `notices`; existing update/onboarding/backup notices may appear alongside it. The stub instructs the Agent to call `memory_repair(task="notice", data={"action":"read", "notice_id": ...})`. Read the full notice, then execute its returned `left_read_call` and `right_read_call` to read both full memories. Only after both reads succeed, assess the advisory candidate and tell the user if it appears credible without calling it a confirmed conflict; dismiss false positives and resolve already-handled notices. Delivery is the state transition `open → open + delivered_at`; public `dismiss`/`resolve` calls make terminal transitions, while stale undelivered snapshots may transition internally to `stale`. Dismiss/resolve reasons are persisted, and an opposite repeated terminal action returns `already_terminal` rather than success. Stale cleanup scans a bounded batch per claim; if no fresh row is found in that batch, delivery returns no semantic stub. Under strict isolation, claim/list/read/dismiss/resolve require both notice memories to be visible in the caller's canonical workspace; omitted caller workspace falls back to `settings.workspace`. `none` and `weak` retain their existing unscoped notice behavior. The database claim is atomic best effort, but transport does not guarantee exactly-once delivery.

Semantic notices never automatically create a conflict, submit a judgment, edit a memory, or supersede either side. `list`, `read`, `dismiss`, and `resolve` use `memory_repair(task="notice", ...)`; these operations do not require governance authorization because they manage advisory notices, not memory facts. Runtime `pause`, `resume`, `enable`, `unload`, `disable`, and `status` use `semantic_control` and also do not require governance authorization.

### Pattern C — Workspace-aware sharing

| Isolation | Behavior |
|---|---|
| `none` | Workspace is stored but ignored for recall. |
| `weak` | Same canonical workspace is boosted; cross-workspace results are demoted; unresolved candidates become hints. |
| `strict` | Caller workspace is required for isolated paths; searches/details hard-filter; an unresolved new canonical is stored as `pending`. |

For weak/strict writes, resolution checks exact canonicals and confirmed/rejected aliases first. If embeddings are ready, vector similarity supplies only a shortlist. Rules decide `AUTO`/`KEEP`/`ASK`; an optional GGUF model may suggest among candidates but is never the final authority. Busy, unavailable, or uncertain inference falls back to `ASK`. Weak mode auto-merges only high-confidence identity-grade relations; strict mode keeps unresolved workspaces pending. User-authorized alias acceptance/rejection and pending confirmation are durable and final.

### Pattern D — Long-document section split

The original row commits first. If vec is ready and a long document has safe Markdown headings, a background worker asynchronously applies the rule-based split. If headings are absent/unsafe or limits are exceeded, the response includes `split_request`; the Agent uses its own model to prepare section metadata and calls `memory_repair(task="split")`. `memory(action="read")` supports `sections=none|catalog|all` and specific `section_ids`.

### Backup-only replay

SQLite unavailable **or unwritable** triggers the JSONL fallback. A successful fallback writes one `backup_schema=1` envelope using a single `O_APPEND` write; if the JSONL path is unavailable/unwritable or the write is short, the memory write fails rather than claiming a backup. Backup-only rows have no SQLite `memory_id` and are not searchable.

Once SQLite is usable, an Agent may run dry-run without authorization:

```text
memory_repair(task="replay_backup", data={"dry_run": true, "limit": 1000, "offset": 0})
```

Formal replay requires explicit user authorization and `dry_run=false, authorized=true`. It processes at most 200 physical entries per call and reports pagination. The main memory row and replay receipt commit atomically; repeated runs are idempotent by replay key/hash. Claims, embeddings, split, and semantic enqueue run after commit and may leave a retryable receipt warning without rolling back the recovered row. Invalid entries do not block valid ones. The original JSONL is retained. Only schema 1 is supported; legacy flat rows are reported as unsupported and are not converted automatically. Under `strict`, the backed-up canonical is preserved and an unconfirmed canonical remains `pending`.

### Product input validation

Known fields are validated before authorization/business execution. Main limits: content 2 MiB UTF-8; subject 2,000 chars; query 32,000 chars; 100 tags × 256 chars; metadata 256 KiB serialized JSON; workspace/source references 2,000 chars; batch IDs 1,000. Invalid enums/timestamps and NaN/Inf are rejected. Harmless unknown fields are stripped with warnings; likely misspellings of protected fields are rejected with `did_you_mean`. Controlled coercion remains for IDs and bounded integer/timeout numeric strings; booleans are not accepted as IDs.

### CI and production smoke

The GitHub Actions workflow contains required jobs for the Python 3.11/3.12/3.13 core matrix, Python 3.12 sqlite-vec tests, quality/security checks, and build + twine validation. The post-release `mema-production-smoke --expected-version X.Y.Z` is manual and optional; it is not a CI, publish, or release gate.

---

<a id="中文"></a>

## 中文

Memory Arbiter 是 MCP 客户端下面的一层本地事实治理与精准召回层。默认工具面只有 `memory`、`memory_review`、`memory_govern`、`memory_repair` 四个产品工具。`MEMORY_ARBITER_TOOL_PROFILE=legacy_full`/`full` 可暴露兼容低层工具，但不会恢复已经移除的向量冲突扫描器。

### 配置与依赖边界

读取顺序为 `MEMORY_ARBITER_CONFIG` → `~/.config/memory-arbiter/config.json` → 环境变量/default。共享数据库、向量和模型设置放用户配置文件；每客户端身份放各自 MCP env。

默认核心无需模型：字面召回、结构化 claim 检测、治理、历史和修复都不要求 sqlite-vec/GGUF。可选依赖：

```bash
pip install 'memory-arbiter-mcp[vec]'             # sqlite-vec 语义/section/workspace 候选召回
pip install 'memory-arbiter-mcp[semantic-local]'  # 本地 GGUF runtime
```

Python 3.13 若没有匹配的 `llama-cpp-python` wheel，`[semantic-local]` 可能本机编译，需要 C/C++ toolchain 和 CMake。

关键配置：

| JSON 路径 | env 兜底 | 默认值 | 含义 |
|---|---|---|---|
| `db_path` | `MEMORY_ARBITER_DB_PATH` | `./memory_arbiter.sqlite3` | 共享 SQLite 路径。 |
| `backup_jsonl` | `MEMORY_ARBITER_BACKUP_JSONL` | `./memory_arbiter.backup.jsonl` | SQLite 不可用或不可写时的 schema-1 追加备份。 |
| `structured_claim_mode` | `MEMORY_ARBITER_STRUCTURED_CLAIM_MODE` | `beta_all` | 确定性写入/编辑 claim 门禁；`off` 仅作熔断，没有 periodic vector scan 兜底。 |
| `isolation` | `MEMORY_ARBITER_ISOLATION` | `none` | `none`、`weak`、`strict`。 |
| `workspace_match_distance` | `MEMORY_ARBITER_WORKSPACE_MATCH_DISTANCE` | `0.25` | 可选 workspace 候选余弦距离阈值，不是最终裁决。 |
| `vec.enabled` | `MEMORY_ARBITER_ENABLE_SQLITE_VEC` | `false` | 启用 sqlite-vec。 |
| `vec.dim` | `MEMORY_ARBITER_VEC_DIM` | `768` | 必须和 embedding 模型一致。 |
| `embedding.model_path` | `MEMORY_ARBITER_EMBEDDING_MODEL_PATH` | 无 | 本地 embedding GGUF；provider 推断为 `gguf`。 |
| `embedding.auto_query` / `auto_write` | 对应 env | `true` / `true` | 配置模型且 vec ready 时自动向量化查询/写入。 |
| `semantic_conflict.enabled` | `MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED` | `false` | 可选写入时语义 notice；设置模型路径后自动启用，除非显式 false。 |
| `semantic_conflict.model_path` | `MEMORY_ARBITER_SEMANTIC_CONFLICT_MODEL_PATH` | 无 | 本地分类 GGUF；当前 backend 只有 `local_gguf`。 |
| `semantic_conflict.pair_text_gate` | `MEMORY_ARBITER_SEMANTIC_CONFLICT_GATE` | `medium` | `medium` 或低打扰 `strong`。 |
| `semantic_conflict.on_write` | `MEMORY_ARBITER_SEMANTIC_CONFLICT_ON_WRITE` | `async` | `async` 或 `off`；写入不等待。 |
| `semantic_conflict.queue_max_size` | 对应 env | `100` | 有界队列；同 memory job 合并。 |
| `semantic_conflict.candidate_limit` / `pair_limit` | 对应 env | `30` / `10` | 每 job 的 metadata 候选与 pair 上限。 |
| `semantic_conflict.n_ctx` / `n_threads` / `n_batch` | 对应 env | `1024` / `4` / `128` | 本地 GGUF 参数。 |
| `semantic_conflict.resident` / `preload` | 对应 env | `true` / `false` | 使用后常驻 / 启动时预加载。 |
| `semantic_conflict.job_timeout_ms` | 对应 env | `5000` | pair 之间的 job budget，不中断已启动推理。 |
| `semantic_conflict.inference_timeout_ms` | 对应 env | `30000` | 单次已启动推理硬超时。 |
| `semantic_conflict.load_timeout_ms` | 对应 env | `120000` | 独立模型启动/加载超时。 |
| `semantic_conflict.min_pair_budget_ms` | 对应 env | `1000` | 开始下一 pair 所需最小剩余预算。 |

`semantic_conflict.max_concurrency` 是保留字段，固定钳制为 `1`。`memory(action="status")` 和 `memory_repair(task="semantic_control", data={"action":"status"})` 可查看 runtime/backend、generation、PID/in-flight、超时/重启计数和预算。

完整样例见 [`examples/memory-arbiter.config.example.json`](../examples/memory-arbiter.config.example.json)。

### 模式 A —— 精准召回

```text
用户提问
  → memory(action="find", data={"query": "具体主题/版本/决策"})
  → 只使用返回的 active 事实
```

需要更精确时加 `tags_filter`、`source_type`、`after_time`、`before_time`。`strict` 下始终传 caller workspace；按 ID/detail 路径也会隔离。

### 模式 B —— 写入时冲突处理

系统不再有 periodic/incremental vector conflict scanner。

1. **确定性结构化 claims（默认）**：写入/编辑后提取保守显式 claim，比较相同 canonical `entity + attribute + scope`；碰撞持久化为 `pending_llm`，携 memory version 与 claim revision pins。
2. **可选语义 notice**：写入成功后，异步 worker 走 specific bounded candidate recall（具体且有界的候选召回）与 subject/tag 排序、有界 pair 选择、本地 Qwen GGUF 分类、确定性 pair-text gate。常见/嘈杂 tag 会被压制，具体 tag 与 subject fallback 仍可入选，排序发生在 `pair_limit` 截断前。模型输出必须包含全部必填字段，字段类型与 `reason_code` enum 必须严格匹配；允许附加解释字段。pair-text gate 默认 `medium`，保守档为 `strong`，通过 `semantic_conflict.pair_text_gate` 或 `MEMORY_ARBITER_SEMANTIC_CONFLICT_GATE` 选择。通过者写为 open `semantic_notices`，只是审阅提示，不是 conflict judgment。
3. **人工/按需审查**：Agent 可用 `memory_review(view="conflicts")` 查看，通过兼容工具比较证据，并显式落表/关闭冲突。

产品响应返回 `action_required=judge_conflict_before_use` 时，必须携全部 pins 调 `memory(action="judge")`，再使用受影响 claim。judgment receipt 只记录 guidance，不编辑或 supersede 任何一侧。若返回 `pending_user`/`ask_user`，最终权在用户；改变事实状态的治理仍需单独授权。

语义 worker 在单子进程内严格串行；写路径 fail-open、异步。memory/version/claim snapshot 已 stale 的历史 job 会跳过。默认 5 秒 job budget 只决定是否开始下一 pair；单次推理硬超时 30 秒，加载另有 120 秒。超时代际必须终止后才能启动新代际。

四个产品工具的下一次成功调用都可能在顶层 `notices` 附加至多 1 个紧凑 semantic stub；现有 update/onboarding/backup notice 可与它并存。stub 指示 Agent 调用 `memory_repair(task="notice", data={"action":"read", "notice_id": ...})`。先读取完整 notice，再执行返回的 `left_read_call` 与 `right_read_call` 读取两侧完整记忆；只有两次读取都成功后，才判断 advisory candidate，并在看起来可信时提示用户，但不得称为已确认冲突。误报 dismiss，已处理 notice resolve。投递状态迁移为 `open → open + delivered_at`；公开的 `dismiss`/`resolve` 才进入终态，未投递 stale snapshot 可由内部迁移到 `stale`。本地数据库 claim 是原子 best effort，transport 不保证 exactly-once delivery。

semantic notice 不会自动创建 conflict、提交 judgment、编辑 memory 或 supersede 任一侧。`list`、`read`、`dismiss`、`resolve` 都走 `memory_repair(task="notice", ...)`；它们只管理 advisory notice，不需要 governance 授权。`semantic_control` 的 `pause`、`resume`、`enable`、`unload`、`disable`、`status` 同样无需治理授权。

### 模式 C —— Workspace 共享与隔离

| isolation | 行为 |
|---|---|
| `none` | workspace 存储但召回忽略。 |
| `weak` | 同 canonical 加权，跨 workspace 降权；未决候选作为 hint。 |
| `strict` | 隔离路径要求 caller workspace；搜索/detail 硬过滤；未决新 canonical 写为 `pending`。 |

weak/strict 写入先查 exact canonical 和 confirmed/rejected alias。embedding ready 时，向量只提供 shortlist；规则决定 `AUTO`/`KEEP`/`ASK`，可选 GGUF 只能在候选中建议，忙、不可用或不确定时 fallback 到 `ASK`。weak 只对高置信身份级关系自动合并；strict 保持未决 workspace pending。用户授权的 alias accept/reject 和 pending confirm 会持久化，并拥有最终权。

### 模式 D —— 长文档分段

原始 row 先提交。vec ready 且长文 Markdown 标题安全时，后台 worker 异步执行规则分段；没有安全标题或超出限制时响应返回 `split_request`，Agent 用自己的模型准备 section metadata 后调用 `memory_repair(task="split")`。`memory(action="read")` 支持 `sections=none|catalog|all` 和指定 `section_ids`。

### Backup-only replay

SQLite 不可用**或不可写**都会触发 JSONL fallback。成功 fallback 使用单次 `O_APPEND` 写一个 `backup_schema=1` envelope；路径不可用/不可写或 short write 时，memory write 失败，不会虚报备份。backup-only 记录没有 SQLite `memory_id`，恢复前不可搜索。

SQLite 可用后，Agent 可无需授权执行 dry-run：

```text
memory_repair(task="replay_backup", data={"dry_run": true, "limit": 1000, "offset": 0})
```

正式 replay 需要用户明确授权并传 `dry_run=false, authorized=true`。单次最多处理 200 个物理条目并返回分页。主 memory row 与 replay receipt 同事务提交，按 replay key/hash 幂等。claims、embedding、split、semantic enqueue 在提交后执行；失败可留下 receipt warning 供重试，不回滚主记录。坏行不阻断其他有效条目，原 JSONL 保留。只支持 schema 1；旧 flat row 明确报告 unsupported，不自动转换。`strict` 下保留备份 canonical，未确认 canonical 保持 `pending`。

### 产品输入校验

已知字段在授权与业务执行前校验。主要上限：正文 2 MiB UTF-8、subject 2,000 字符、query 32,000 字符、100 个 tag × 256 字符、metadata 序列化 JSON 256 KiB、workspace/source reference 2,000 字符、批量 ID 1,000 个。非法 enum/时间和 NaN/Inf 会拒绝。普通未知字段剥离并 warning；疑似受保护字段拼写错误拒绝并给 `did_you_mean`。ID 和有界整数/timeout 的数字字符串保留受控 coercion，bool 不能作为 ID。

### CI 与 production smoke

GitHub Actions workflow 包含 Python 3.11/3.12/3.13 core matrix、Python 3.12 sqlite-vec 测试、质量/安全检查和 build + twine 校验。发版后 `mema-production-smoke --expected-version X.Y.Z` 是人工可选检查，不是 CI、publish 或 release 门禁。
