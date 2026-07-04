# Memory Arbiter MCP

### 你的 AI 工具记不住彼此说过的话。

你用 ZCode 写了一个架构决策，转头打开 Codex，它一脸茫然。你在 Cursor 里踩过的坑，换到 ZCode 又踩一遍。每个工具都有自己的记忆系统，彼此隔离，互不相通。

**Memory Arbiter MCP 就是为了解决这个问题。**

---

## 它解决什么问题

### 1. 多工具记忆孤岛

你同时在用 ZCode、Codex、Cursor、Claude Code……每个工具都号称有"记忆"，但它们的记忆互不相通：

- 你在 ZCode 里定下了"这个项目用 REST 不用 GraphQL"
- 打开 Codex，它完全不知道这个约定，继续给你生成 GraphQL
- 你在 Codex 里排查了一个依赖冲突的根因
- 换到 Cursor，同样的问题从头再来一遍

Memory Arbiter 让所有工具共享同一个记忆库。你在任何一个工具里告诉它的事，其他工具都能查到。

### 2. 记忆冲突，谁说了算

不同工具（或不同 Agent）可能记下互相矛盾的信息：

- ZCode 记录："数据库用 PostgreSQL"
- Codex 记录："数据库用 MySQL"

Memory Arbiter 内置仲裁机制：按用户确认 > 事件发生时间 > 来源可信度 > 录入时间的优先级自动判定，给出可解释的裁决理由，而不是默默留两条矛盾记录。

### 3. 用户确认的事实不该被覆盖

你在对话中明确确认过的事情（"对，这个接口就用 v2 版本"），不应该被某个 Agent 的推测覆盖。Memory Arbiter 支持 `user_confirmed` 标记和 `locked` 保护级别——确认即锁定，谁都不能自动推翻。

---

## 核心创新：多客户端记忆共享

这是 Memory Arbiter 和其他记忆方案最大的不同。

| | 传统方案 | Memory Arbiter |
|---|---|---|
| 记忆存储 | 每个工具各自存储 | 一个 SQLite，所有工具共享 |
| 跨工具查询 | 不支持 | 任意工具可搜全量记忆 |
| 冲突处理 | 无（各记各的） | 双时间轴仲裁 + 自动降级 |
| 来源追踪 | 无 | 每条记忆标记来源工具和 Agent |
| 用户确认保护 | 无 | 锁定后禁止自动覆盖 |
| 云依赖 | 通常需要 | 完全本地，零云依赖 |

**工作原理一句话**：作为一个标准 MCP Server 运行在每个工具的侧边，所有工具通过同一个协议读写同一个本地数据库，冲突由结构化规则仲裁，不依赖大模型判断。

---

## 适合谁

- **同时使用多个 AI 编程工具的开发者**（ZCode + Codex + Cursor + Claude Code 等）
- 希望工具之间共享项目知识、决策、踩坑经验
- 受够了每个工具都要重新解释一遍项目背景
- 对记忆冲突有治理需求（而不是放任不管）

---

## 快速开始

### 安装

需要 Python 3.11+。

```bash
# 克隆或进入项目目录
cd ~/OpenClawProject/memory-arbiter-mcp

# 创建虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate

# 安装依赖和本包
pip install -r requirements.txt
pip install -e .

# 可选：启用语义召回增强（sqlite-vec）
pip install '.[vec]'
```

验证安装：

```bash
memory-arbiter-mcp
# 输出 MCP Server running in stdio mode... 即正常
```

### 接入 ZCode（示例）

ZCode 的 MCP 配置文件在 `~/.zcode/v2/` 下。编辑（或创建）MCP 配置：

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "/Users/zhangzhiwei17/OpenClawProject/memory-arbiter-mcp/.venv/bin/memory-arbiter-mcp",
      "env": {
        "MEMORY_ARBITER_CLIENT": "zcode",
        "MEMORY_ARBITER_AGENT_ID": "zcode-default",
        "MEMORY_ARBITER_DB_PATH": "~/.local/share/memory-arbiter/memory.sqlite3"
      }
    }
  }
}
```

> **注意**：`command` 建议写虚拟环境的绝对路径（`.venv/bin/memory-arbiter-mcp`），避免客户端找不到 PATH 里的可执行文件。

### 接入其他工具

同样的结构，改一下 `MEMORY_ARBITER_CLIENT` 标识即可：

| 客户端 | `MEMORY_ARBITER_CLIENT` | 配置文件位置 |
|---|---|---|
| ZCode | `"zcode"` | `~/.zcode/v2/` 下 MCP 配置 |
| Codex | `"codex"` | `~/.codex/` 下 MCP 配置 |
| Claude Code | `"claude-code"` | 项目根目录 `.mcp.json` |
| Cursor | `"cursor"` | `~/.cursor/mcp.json` |

所有客户端共享同一个 `MEMORY_ARBITER_DB_PATH`，这才是跨工具记忆共享的关键。

完整配置示例见 `examples/` 目录。

### ⚠️ 重要：必须新建会话

MCP Server 在客户端启动时加载。**已经打开的会话不会自动识别新添加的 MCP Server**，这是客户端的限制，不是 Memory Arbiter 的问题。

正确操作：

1. 关闭当前会话（或直接新开一个）
2. 确认客户端已加载 `memory-arbiter` MCP Server
3. 在新会话中正常使用

如果工具列表里看不到 `memory_search`、`memory_write` 等，大概率就是当前会话启动时还没配置好 MCP，**新建一个会话**即可。

---

## 数据迁移：换电脑怎么办

所有记忆数据都在一个 SQLite 文件里，迁移非常简单：

### 1. 拷贝数据库文件

```bash
# 默认路径
scp ~/.local/share/memory-arbiter/memory.sqlite3 新电脑:~/.local/share/memory-arbiter/
```

### 2. 重新安装项目

```bash
cd ~/OpenClawProject/memory-arbiter-mcp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> `.venv` 不要直接拷贝，里面包含绝对路径，换机器会失效。新电脑上重建即可。

### 3. 配置 MCP Server

在新电脑的客户端里配好 `mcpServers`，指向相同的 `MEMORY_ARBITER_DB_PATH`，新建会话即可使用。

### 后续：云同步

本地版够用时不用折腾。如果后续需要多设备实时同步，计划通过 Turso（云端 libSQL）实现，改动很小——加一个 `MEMORY_ARBITER_REMOTE_URL` 环境变量即可开启，不影响纯本地模式。

---

## 支持的 MCP 工具

| 工具 | 用途 |
|---|---|
| `memory_write` | 写入一条记忆（`source_type=user_confirmed` 自动锁定） |
| `memory_search` | 搜索已有记忆（FTS5 → LIKE 自动降级） |
| `memory_compare` | 比较两条记忆是否冲突（只返回解释，不落记录） |
| `memory_arbitrate` | 仲裁冲突，自动判定胜者（`apply=true` 时落记录） |
| `memory_confirm` | 用户确认某条记忆，锁定保护 |
| `memory_list_conflicts` | 列出未解决的冲突 |
| `memory_status` | 查看运行状态、模式、降级原因 |

---

## 仲裁规则

冲突发生时，按以下优先级判定：

1. **用户确认 > 一切**：`source_type=user_confirmed` 的记忆自动锁定，不可被覆盖
2. **事件发生时间（event_time）**：越接近事实发生时间点越可信
3. **来源可信度（source_type）**：`document_extracted` > `agent_generated` > `unknown`
4. **录入时间（ingest_time）**：以上都相同时，先录入的优先

裁决结果可解释：每次仲裁都输出结构化理由，包含胜方、败方、判定依据。

---

## 降级策略

零依赖崩溃，逐级降级：

| 层级 | 条件 | 行为 |
|---|---|---|
| 1. sqlite-vec | 可用 | 语义召回增强（可选） |
| 2. FTS5 | sqlite-vec 缺失 | 全文搜索（默认主模式） |
| 3. LIKE | FTS5 缺失 | 关键词模糊匹配 |
| 4. JSONL | SQLite 不可写 | append-only 文件备份 |

降级状态通过 `memory_status` 查询，所有响应都会带 `warnings` 和 `degraded` 标记。

---

## 设计原则

- **本地优先**：所有数据存在本地 SQLite，不依赖任何云服务
- **轻量**：不跑大模型、不启 Web 服务、不需要 Postgres/Redis
- **可降级**：sqlite-vec → FTS5 → 关键词 → JSONL 备份，不会崩
- **MCP 标准**：任何支持 MCP 的客户端都能接入
- **零侵入**：不修改工具本身，只通过 MCP 协议旁路接入

---

## 兼容性

支持所有兼容 MCP stdio 协议的客户端：

- ✅ ZCode
- ✅ Codex CLI
- ✅ Claude Code
- ✅ Cursor
- ✅ Cline / Roo Code
- ✅ 任何支持 `mcpServers` 配置的工具

---

## License

MIT
