mcp-name: io.github.billy12151/memory-arbiter-mcp

# memory-arbiter-mcp

**[中文](#中文) | [English](#english)**

---

<a id="english"></a>

## English

A lightweight, fully local MCP Server that gives your AI coding tools a **shared memory store** with built-in conflict arbitration.

Every tool — ZCode, Codex, Cursor, Claude Code — has its own memory. They don't talk to each other. Memory Arbiter fixes this: one SQLite database, all tools read and write through the same MCP protocol, conflicts are resolved by structured rules (not LLM guesswork).

### Why Memory Arbiter?

Most AI clients load their entire memory file (`MEMORY.md` + `memory/*.md`) into the system prompt **every turn**. As memory grows, so does your token bill — 5K–20K tokens of context burned before the model even reads your question. Memory Arbiter is a **token-optimization middleware**: precise retrieval replaces full-file loading.

| Scenario | Full-file loading | With Memory Arbiter | Saving |
|---|---|---|---|
| Per-turn memory load | `MEMORY.md` + `memory/*.md` all in context (5K–20K tokens) | `memory_search("keyword")` returns 3–5 relevant entries (200–800 tokens) | ~80%+ |
| Conflict detection | LLM compares every pair (N², thousands of tokens) | `memory_compare(id1, id2)` returns a structured verdict, LLM only confirms (~200 tokens) | ~90% |
| Periodic audit | LLM scans the whole library (10K+ tokens) | `memory_list_conflicts` + `memory_recent` give structured candidates; LLM only makes the final call | ~70% |
| Cross-tool sharing | Each tool keeps its own memory, duplicated storage & loading | One SQLite, isolated by `workspace`/`agent_id`, write once — use everywhere | storage 100% dedup |

**Positioning, in four lines:**
- ✅ Structured memory storage — SQLite + dual timeline + source trust levels
- ✅ Conflict arbitration engine — rule-based verdicts with explainable rationale
- ✅ Cross-tool sharing layer — one database, every AI client shares it
- ✅ Token-optimization middleware — precise retrieval replaces full-file loading
- ❌ Not an LLM, does not do semantic reasoning — semantic judgement stays with the AI client

#### Real-world example: cross-tool task delegation

The user runs OpenClaw (planning) + ZCode (coding). OpenClaw drafts a task spec; ZCode implements it.

- **Old way**: OpenClaw writes the spec to a file → ZCode reads it. Requires agreed paths, manual sync, version drift. Or the user copy-pastes — information loss + wasted tokens.
- **Memory Arbiter way**: OpenClaw calls `memory_write` with the spec → the user switches to ZCode → ZCode runs `memory_search("the task")` and gets the full spec with **zero file handoff**.

**Tokens for handing off a 2000-word spec: ~3000 (old) → ~500 (new). Saving ~83%.** This very project ships its own release tasks this way — it dogfoods itself. Full step-by-step in [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

> See [`docs/INTEGRATION.md`](docs/INTEGRATION.md) for three concrete usage patterns (per-turn retrieval, scheduled audit, write-time conflict check) and the full cross-tool delegation walkthrough.

### Features

- **Structured memory write**: `content`, `agent_id`, `workspace`, `tags`, `source_type`, `event_time`, `ingest_time`, `confidence`, `protection_level`, and more.
- **Source trust levels**: `user_confirmed` > `document_extracted` > `agent_generated` > `unknown`.
- **Dual timeline arbitration**: resolves conflicts by user confirmation → event time → source trust → ingest time. Every decision comes with an explainable rationale.
- **Locked protection**: `user_confirmed` memories are automatically locked — no agent can overwrite them.
- **Client policy system**: per-client enable/disable, agent allow/deny lists for multi-agent governance.
- **Graceful degradation**: `sqlite-vec` → FTS5 → `LIKE` → JSONL backup. Never crashes.
- **Zero cloud, zero LLM calls**: pure local SQLite. No Postgres, Redis, or external services.

### Quick Start

**Requirements**: Python 3.11+

```bash
# Clone
git clone https://github.com/billy12151/memory-arbiter-mcp.git
cd memory-arbiter-mcp

# Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Optional: semantic recall via sqlite-vec
pip install '.[vec]'

# Run
memory-arbiter-mcp
```

### Connect Your Tool

Add to your tool's MCP config (see `examples/` for ready-made templates):

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "/path/to/memory-arbiter-mcp/.venv/bin/memory-arbiter-mcp",
      "env": {
        "MEMORY_ARBITER_CLIENT": "zcode",
        "MEMORY_ARBITER_AGENT_ID": "zcode-default",
        "MEMORY_ARBITER_DB_PATH": "~/.local/share/memory-arbiter/memory.sqlite3"
      }
    }
  }
}
```

> Change `MEMORY_ARBITER_CLIENT` for each tool (`zcode`, `codex`, `cursor`, `claude-code`). All tools share the same `DB_PATH` — that's the whole point.

> ⚠️ **New session required**: MCP servers are loaded at session startup. Already-open sessions won't see the new tools. Start a fresh session after configuring.

### Client Config Locations

| Client | Config Location |
|---|---|
| ZCode | `~/.zcode/v2/` MCP config |
| Codex CLI | `~/.codex/` MCP config |
| Claude Code | `.mcp.json` in project root |
| Cursor | `~/.cursor/mcp.json` |

### MCP Tools

| Tool | Description |
|---|---|
| `memory_write` | Write a memory (`source_type=user_confirmed` auto-locks) |
| `memory_search` | Search memories (FTS5 → LIKE fallback) |
| `memory_compare` | Compare two memories, returns explanation only |
| `memory_arbitrate` | Arbitrate conflict, can record result (`apply=true`) |
| `memory_confirm` | Promote a memory to user-confirmed and locked |
| `memory_list_conflicts` | List unresolved conflicts |
| `memory_audit_summary` | Per-workspace stats overview (counts, oldest/newest, open conflicts, source_type distribution) |
| `memory_status` | Show current mode, degradation status, storage paths |

### Data Migration

Moving to a new machine? Just copy the SQLite file:

```bash
# Copy the database
cp ~/.local/share/memory-arbiter/memory.sqlite3 /new/machine/~/.local/share/memory-arbiter/

# Reinstall the project (don't copy .venv — rebuild it)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Testing

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m pytest
```

### License

MIT

---

<a id="中文"></a>

## 中文

一个轻量、完全本地运行的 MCP Server，给你的 AI 编程工具提供**共享记忆库**，内置冲突仲裁机制。

你同时用 ZCode、Codex、Cursor、Claude Code——每个工具都有各自的记忆，互不相通。Memory Arbiter 解决这个问题：一个 SQLite 数据库，所有工具通过同一个 MCP 协议读写，冲突由结构化规则仲裁，不依赖大模型。

### 为什么需要 Memory Arbiter？

多数 AI 客户端每轮对话都把整个记忆文件（`MEMORY.md` + `memory/*.md`）塞进 system prompt。记忆越多，token 消耗越大——模型还没读你的问题，5K–20K token 的上下文已经烧掉了。Memory Arbiter 是一层 **token 优化中间件**：用精准检索替代全文加载。

| 场景 | 全文加载 | 用 memory-arbiter | 节省 |
|------|---------|-------------------|------|
| 每轮对话记忆加载 | `MEMORY.md` + `memory/*.md` 全塞 context（5K–20K tokens） | `memory_search("关键词")` 返回 3–5 条相关记忆（200–800 tokens） | ~80%+ |
| 冲突检测 | LLM 逐条比对全部记忆（N² 复杂度，数千 tokens） | `memory_compare(id1, id2)` 返回结构化裁决，LLM 只确认（~200 tokens） | ~90% |
| 定期审查 | LLM 扫全库生成报告（万级 tokens） | `memory_list_conflicts` + `memory_recent` 拿结构化候选，LLM 只做最终判断 | ~70% |
| 跨工具共享 | 每个工具各自维护记忆，重复存储重复加载 | 统一 SQLite，按 `workspace`/`agent_id` 隔离，一次写入处处可用 | 存储 100% 去重 |

**定位（四句话）：**
- ✅ 结构化记忆存储 — SQLite + 双时间轴 + 来源可信度
- ✅ 冲突仲裁引擎 — 规则化裁决，输出可解释理由
- ✅ 跨工具共享层 — 一个数据库，所有 AI 客户端共享
- ✅ Token 优化中间件 — 精准检索替代全文加载
- ❌ 不是 LLM、不做语义推理 — 语义判断交给 AI 客户端

#### 真实案例：跨工具任务委派

用户同时用 OpenClaw（负责规划）+ ZCode（负责写代码）。OpenClaw 出任务规格，ZCode 执行。

- **传统方式**：OpenClaw 把规格写成文件 → ZCode 读文件，要约定路径、手动同步、版本还容易乱；或者用户口述/复制粘贴，信息损耗又浪费 token。
- **memory-arbiter 方式**：OpenClaw 调 `memory_write` 写入规格 → 用户切到 ZCode → ZCode 跑 `memory_search("那个任务")` 直接拿到完整规格，**零文件传递**。

**一份 2000 字规格的交接成本：~3000 tokens（传统）→ ~500 tokens（现在），省 83%。** 这个项目自己的发版任务就是这么跑的——用自己的产品喂自己的产品。完整步骤见 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)。

> 三种典型用法（每轮按需检索、定时审查、写入时冲突检测）和完整的跨工具委派步骤见 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)。

### 核心能力

- **结构化写入**：`content`、`agent_id`、`workspace`、`tags`、`source_type`、`event_time`、`ingest_time`、`confidence`、`protection_level` 等。
- **来源可信度**：`user_confirmed` > `document_extracted` > `agent_generated` > `unknown`。
- **双时间轴仲裁**：按 用户确认 → 事件发生时间 → 来源可信度 → 录入时间 的优先级判定，输出可解释的裁决理由。
- **锁定保护**：`user_confirmed` 的记忆自动锁定，任何 Agent 都不能自动覆盖。
- **客户端策略**：按客户端启用/禁用，Agent 级别的 allow/deny 白名单控制。
- **逐级降级**：`sqlite-vec` → FTS5 → `LIKE` → JSONL 备份，不会崩。
- **零云依赖、零大模型调用**：纯本地 SQLite，不需要 Postgres、Redis 或外部服务。

### 快速开始

**要求**：Python 3.11+

```bash
# 克隆
git clone https://github.com/billy12151/memory-arbiter-mcp.git
cd memory-arbiter-mcp

# 安装
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 可选：启用语义召回增强（sqlite-vec）
pip install '.[vec]'

# 启动
memory-arbiter-mcp
```

### 接入工具

在你的工具的 MCP 配置中加入（完整示例见 `examples/` 目录）：

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "/path/to/memory-arbiter-mcp/.venv/bin/memory-arbiter-mcp",
      "env": {
        "MEMORY_ARBITER_CLIENT": "zcode",
        "MEMORY_ARBITER_AGENT_ID": "zcode-default",
        "MEMORY_ARBITER_DB_PATH": "~/.local/share/memory-arbiter/memory.sqlite3"
      }
    }
  }
}
```

> 每个工具改一下 `MEMORY_ARBITER_CLIENT` 标识（`zcode`、`codex`、`cursor`、`claude-code`），共享同一个 `DB_PATH`——这就是跨工具记忆共享的关键。

> ⚠️ **需要新建会话**：MCP Server 在客户端启动时加载，已经打开的会话不会识别新添加的 Server。配置好后请新建一个会话。

### 客户端配置位置

| 客户端 | 配置文件位置 |
|---|---|
| ZCode | `~/.zcode/v2/` 下 MCP 配置 |
| Codex CLI | `~/.codex/` 下 MCP 配置 |
| Claude Code | 项目根目录 `.mcp.json` |
| Cursor | `~/.cursor/mcp.json` |

### MCP 工具

| 工具 | 说明 |
|---|---|
| `memory_write` | 写入记忆（`source_type=user_confirmed` 自动锁定） |
| `memory_search` | 搜索记忆（FTS5 → LIKE 自动降级） |
| `memory_compare` | 比较两条记忆，只返回解释 |
| `memory_arbitrate` | 仲裁冲突，自动判定胜者（`apply=true` 时落记录） |
| `memory_confirm` | 用户确认某条记忆，锁定保护 |
| `memory_list_conflicts` | 列出未解决的冲突 |
| `memory_audit_summary` | 各 workspace 记忆统计概览（条目数、最旧/最新、open 冲突数、来源分布） |
| `memory_status` | 查看运行状态、模式、降级原因 |

### 数据迁移

换电脑只需拷贝一个文件：

```bash
# 拷贝数据库
cp ~/.local/share/memory-arbiter/memory.sqlite3 新电脑:~/.local/share/memory-arbiter/

# 重新安装项目（.venv 不要拷贝，新机器上重建）
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 测试

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m pytest
```

### License

MIT
