# memory-arbiter-mcp

一个完整、轻量、本地运行的跨工具记忆外挂 MCP Server 原型，用于验证“基于双时间轴、来源标记与插件化适配的多智能体记忆冲突仲裁方法及系统”。

MVP 的事实源是 SQLite；`sqlite-vec` 只做可选语义召回增强，仲裁只信结构化字段。默认不调用大模型、不云同步、不启动 Web 服务，也不依赖 Postgres、Redis 或外部向量服务。

## 能力

- 结构化写入：`content`、`agent_id`、`workspace`、`tags`、`source_type`、`source_ref`、`event_time`、`ingest_time`、`confidence`、`protection_level`、`status`、`subject`、`metadata`。
- 来源标记：`user_confirmed`、`document_extracted`、`agent_generated`、`unknown`、`pending`。
- 双时间轴仲裁：先看用户确认/锁定保护，再按信息发生时间 `event_time`、来源强度、置信度、录入时间 `ingest_time` 输出可解释结果。
- 适配策略：Codex、ZCode 默认启用；OpenClaw 默认不启用，可通过 allow/deny agent 白名单控制。
- MCP tools：`memory_write`、`memory_search`、`memory_compare`、`memory_arbitrate`、`memory_list_conflicts`、`memory_confirm`、`memory_status`。
- 降级：`sqlite-vec` 不可用时降到 FTS5；FTS5 不可用时降到 LIKE/关键词；SQLite 不可写时尽量写 JSONL append-only 备份。

所有工具返回统一 JSON：

```json
{
  "ok": true,
  "mode": "fts5",
  "warnings": ["sqlite-vec unavailable: ..."],
  "degraded": true,
  "data": {}
}
```

## 安装

需要 Python 3.11+。

```bash
cd /Users/zhangzhiwei17/OpenClawProject/memory-arbiter-mcp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

可选安装语义召回增强：

```bash
pip install '.[vec]'
```

启动 stdio MCP Server：

```bash
memory-arbiter-mcp
```

如果没有安装 MCP Python SDK，命令会提示安装 `pip install -r requirements.txt` 或 `pip install mcp`。

## 环境变量

可从 `.env.example` 复制：

```bash
export MEMORY_ARBITER_DB_PATH=~/.local/share/memory-arbiter/memory.sqlite3
export MEMORY_ARBITER_BACKUP_JSONL=~/.local/share/memory-arbiter/backup.jsonl
export MEMORY_ARBITER_CLIENT=codex
export MEMORY_ARBITER_AGENT_ID=codex-default
export MEMORY_ARBITER_WORKSPACE=/path/to/workspace
```

## Codex 接入

参考 `examples/codex.mcp.json`：

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "memory-arbiter-mcp",
      "env": {
        "MEMORY_ARBITER_CLIENT": "codex",
        "MEMORY_ARBITER_AGENT_ID": "codex-default",
        "MEMORY_ARBITER_WORKSPACE": "${workspaceFolder}",
        "MEMORY_ARBITER_DB_PATH": "~/.local/share/memory-arbiter/memory.sqlite3"
      }
    }
  }
}
```

## ZCode 接入

参考 `examples/zcode.mcp.json`，把 MCP 配置加入 ZCode 的 MCP Server 列表：

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "memory-arbiter-mcp",
      "env": {
        "MEMORY_ARBITER_CLIENT": "zcode",
        "MEMORY_ARBITER_AGENT_ID": "zcode-default",
        "MEMORY_ARBITER_WORKSPACE": "${workspaceFolder}",
        "MEMORY_ARBITER_DB_PATH": "~/.local/share/memory-arbiter/memory.sqlite3"
      }
    }
  }
}
```

## OpenClaw 接入

OpenClaw 自带记忆体系，所以建议默认关闭，只给部分 agent 启用。参考 `examples/openclaw.agent-policy.json`：

```json
{
  "client_defaults": {
    "codex": true,
    "zcode": true,
    "openclaw": false
  },
  "default_enabled": true,
  "allow_agents": ["research-agent", "coding-agent"],
  "deny_agents": ["private-notes-agent"]
}
```

MCP 配置示例：

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "memory-arbiter-mcp",
      "env": {
        "MEMORY_ARBITER_CLIENT": "openclaw",
        "MEMORY_ARBITER_AGENT_ID": "coding-agent",
        "MEMORY_ARBITER_POLICY": "/Users/zhangzhiwei17/OpenClawProject/memory-arbiter-mcp/examples/openclaw.agent-policy.json",
        "MEMORY_ARBITER_DB_PATH": "~/.local/share/memory-arbiter/memory.sqlite3"
      }
    }
  }
}
```

策略规则：

- `deny_agents` 优先级最高。
- `allow_agents` 命中则启用。
- 没命中 allow/deny 时使用 `client_defaults`。
- 未知 client 使用 `default_enabled`。

## 工具说明

- `memory_write`：写入结构化记忆。`source_type=user_confirmed` 会自动设为 `locked`，禁止自动覆盖。
- `memory_search`：按 FTS5 或 LIKE 检索。`sqlite-vec` 可用时会在状态里显示，但 MVP 不把向量结果作为事实仲裁依据。
- `memory_compare`：比较两条记忆，只返回解释，不落冲突记录。
- `memory_arbitrate`：比较并可记录冲突；`apply=true` 时只会把非保护输家标为 `superseded`。
- `memory_list_conflicts`：列出冲突记录。
- `memory_confirm`：把记忆提升为用户确认、锁定保护。
- `memory_status`：查看当前模式、降级原因、存储路径和适配策略。

## 测试

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m pytest
```

覆盖写入、搜索、仲裁、用户确认保护、降级状态。

## 降级行为

- `sqlite-vec` 缺失：返回 warning，使用 FTS5 或关键词检索。
- FTS5 缺失：返回 warning，使用 `LIKE`/关键词检索。
- SQLite 不可写：返回 warning，尽量写入 JSONL append-only 备份；搜索和仲裁能力会受限。
- MCP SDK 缺失：server 启动失败并输出安装提示，核心 Python 模块仍可被测试或嵌入调用。
