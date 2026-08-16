# Memory Arbiter MCP

### 让多个 AI 工具共享一层可检索、可追溯、可治理的本地事实库

你在 ZCode 里确认的架构决策，切换到 Codex 或 Cursor 后不应重新解释；不同 Agent 写下互相矛盾的结论时，也不应靠 last-write-wins 静默覆盖。Memory Arbiter（迷码，短命令 `mema`）以标准 MCP Server 连接多个客户端，让它们读写同一个本地 SQLite，并把来源、时间、版本、冲突和 workspace 边界显式化。

核心默认无需模型：字面召回、结构化 claim 检测、历史、治理和修复都能独立运行。sqlite-vec、本地 GGUF embedding 和本地 GGUF 语义冲突 notice 都是可选增强。

---

## 它解决什么问题

### 1. 多工具记忆孤岛

不同客户端通常有各自的记忆文件。Memory Arbiter 让它们通过同一协议按需检索同一份事实库，并记录 `agent_id`、`workspace`、`source_type`、`event_time` 和 `ingest_time`。共享数据库路径是跨工具可见的前提；`strict` workspace 隔离则会主动限制可见范围。

### 2. 记忆冲突不是自动裁决

冲突发现有两条当前实现路径：

- 默认开启的确定性结构化 claim 检测，在写入/编辑后比较同一 `entity + attribute + scope` 的显式 key/value、表格、数字单位和 semver claim；
- 可选本地 GGUF 写入时语义 notice，通过 specific bounded candidate recall（具体且有界的候选召回）与 subject/tag 排序、模型候选信号和 pair-text gate 提示可能的语义冲突；pair-text gate 默认 `medium`，保守低打扰档为 `strong`。

旧的 `memory_scan_conflict_candidates` 向量扫描器和 periodic vector scan 已移除。sqlite-vec 仍可用于语义召回、section 召回和 workspace 候选，但不再向冲突扫描器供数。

结构化碰撞会持久化为 `pending_llm`，响应返回 `action_required=judge_conflict_before_use` 和 snapshot pins。宿主 LLM 必须调用 `memory(action="judge")` 提交判断 receipt，之后才能使用受影响 claim。judgment 只记录 guidance，不会自动编辑、覆盖或 supersede 任一记忆；不确定、双保护或高影响用途会升级为 `pending_user`。整条记忆过期、冲突关闭、workspace alias 决策等治理操作仍由用户最终授权。

四个产品工具的成功响应可投递至多一个紧凑 semantic notice stub。Agent 先 `memory_repair(task="notice", data={"action":"read", ...})`，再执行返回的 `left_read_call` 与 `right_read_call` 读取两侧完整记忆；只有两次读取成功后，才在候选看起来可信时提示用户，且不得称为已确认冲突。投递只做 `open → open + delivered_at` 状态迁移；公开 `dismiss`/`resolve` 才进入终态，未投递 stale snapshot 可由内部迁移到 `stale`。数据库 claim 是原子 best effort，transport 不保证 exactly-once delivery。

### 3. 用户确认事实与演进历史

`user_confirmed` 和 `locked` 让权威事实可见并阻止普通自动覆盖。新 source-of-truth 替换旧 current 内容时，应更新已有记忆；只有用户明确要求整条旧记忆退役时才调用治理工具。版本快照、judgment 历史和 supersede 链保留变化依据。

---

## 快速开始

需要 Python 3.11+；CI 覆盖 3.11、3.12、3.13。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# 可选：sqlite-vec 语义召回
pip install -e '.[vec]'

mema
```

也可使用发布包：

```bash
uvx --from memory-arbiter-mcp mema
```

配置助手：

```bash
mema setup
```

它会生成 `~/.config/memory-arbiter/config.json` 并检查环境，但不会自动安装依赖或下载模型。

### 接入 MCP 客户端

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "/absolute/path/to/.venv/bin/memory-arbiter-mcp",
      "env": {
        "MEMORY_ARBITER_CLIENT": "zcode",
        "MEMORY_ARBITER_AGENT_ID": "zcode-default",
        "MEMORY_ARBITER_WORKSPACE": "default"
      }
    }
  }
}
```

共享数据库、向量和模型设置建议放在 `~/.config/memory-arbiter/config.json`；每客户端身份放各自 MCP env。MCP Server 通常在客户端会话启动时加载，改完配置后需要新建会话。

---

## 默认产品工具

v0.11 起默认只暴露四个任务型工具：

| 工具 | 用途 |
|---|---|
| `memory` | `remember`、`find`、`read`、`update`、`judge`、`status`。 |
| `memory_review` | 只读 overview、doctor、conflicts、conflict_detail、judgments、history、expired、audit、entities。 |
| `memory_govern` | 需要用户明确授权的 retire、resolve、confirm、judgment correction 和 workspace 治理。 |
| `memory_repair` | split、claims/embedding 重建、清理、向量同步、entity 修复、backup replay、semantic runtime/notice 维护。 |

低层实现仍被内部复用，但默认不暴露 schema。兼容需要时可设置 `MEMORY_ARBITER_TOOL_PROFILE=legacy_full`（或 `full`）；已移除的向量冲突 scanner 不会因此恢复。

### 长文档分段

分段能力绑定 vec readiness。满足阈值且 Markdown 标题安全时，原文先提交，再由后台 worker 异步执行规则分段；结构不适合规则分段时返回 `split_request`，由 Agent 使用自己的 LLM 生成 section metadata 后调用 `memory_repair(task="split")`。短文不会触发分段。

### 可选本地语义冲突 notice

安装运行时并配置分类 GGUF：

```bash
pip install 'memory-arbiter-mcp[semantic-local]'
```

设置 `semantic_conflict.model_path` 后会自动启用，除非显式 `enabled=false`。当前仅支持 `local_gguf`。写入 job 异步入队，GGUF 推理全局严格串行；默认 5 秒 job budget 只控制是否开始下一 pair，单次推理硬超时为 30 秒，模型加载超时为 120 秒。超时子进程终止后才会启动新代际。状态可通过 `memory(action="status")` 或 `memory_repair(task="semantic_control", data={"action":"status"})` 查看。

Python 3.13 上若没有匹配的 `llama-cpp-python` wheel，安装 `[semantic-local]` 可能本机编译，需要 C/C++ toolchain 和 CMake。

### Workspace 规则

- `none`：workspace 仅存储，不过滤召回；
- `weak`：同 canonical workspace 加权，跨 workspace 降权；未决候选以 hint 提示；
- `strict`：搜索/按 ID 读取硬过滤；新且未确认的 canonical 写成 `pending`，确认前不进入 active recall。

`weak`/`strict` 先使用 exact canonical 和已确认/拒绝 alias。向量相似度只提供候选 shortlist；规则决定 `AUTO`/`KEEP`/`ASK`，可选 GGUF 只能建议候选，忙或不可用时 fallback 到 `ASK`。只有 weak 下高置信身份级关系可自动合并；用户 accept/reject alias 和 confirm pending workspace 拥有最终权并持久化。

---

## SQLite 降级与 JSONL 恢复

SQLite 不可用或不可写时，写入尝试追加一个 `backup_schema=1` envelope。JSONL 采用单次 `O_APPEND` 写；追加失败时本次写入失败，不会声称已备份。backup-only 记录没有 SQLite `memory_id`，恢复前不可搜索。

SQLite 恢复后可直接执行只读预览：

```text
memory_repair(task="replay_backup", data={"dry_run": true})
```

正式 replay 必须先取得用户授权，再传 `dry_run=false, authorized=true`。主 memory row 与 replay receipt 在同一事务提交；重复执行按 `replay_key + payload_hash` 幂等。claims、embedding、split 和 semantic enqueue 是提交后的可重试后处理，不会因其失败回滚主记录。正式单次最多处理 200 条并返回分页信息。原 JSONL 保留；旧 flat JSONL 不自动兼容，会明确报告 unsupported legacy entry。`strict` replay 对未确认 canonical 保持 `pending`。

---

## 输入校验边界

四个产品工具会拒绝非法已知字段类型、enum、ISO 时间、NaN/Inf 和超限资源。主要上限：

- content：2 MiB UTF-8；subject：2,000 字符；query：32,000 字符；
- tags：最多 100 个，单 tag 最多 256 字符；metadata JSON：256 KiB；
- workspace/source_ref 等文本字段：2,000 字符；批量 ID：1,000 个。

未知普通字段会从 payload 剥离并返回 warning；疑似受保护字段的拼写错误会拒绝并给 `did_you_mean`。ID 和有界整数/timeout 保留受控数字字符串 coercion，bool 不能冒充 ID。

---

## CI、发布后 smoke 与 License

CI workflow 包含 Python 3.11/3.12/3.13 core matrix、Python 3.12 sqlite-vec 全套测试、质量/安全检查以及 build + twine 校验。`mema-production-smoke --expected-version X.Y.Z` 是发版后对指定正式环境的人工可选检查，不是 CI 或发布门禁。

当前项目使用 Apache License 2.0。0.8.2 起的版本按 Apache-2.0 提供；此前已经按 MIT 分发的副本继续保有原 MIT 授权，包括 0.8.0、0.8.1 及更早版本。
