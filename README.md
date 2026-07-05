mcp-name: io.github.billy12151/memory-arbiter-mcp

# memory-arbiter-mcp

**[中文](#中文) | [English](#english)**

---

<a id="english"></a>

## English

A lightweight, fully local **AI enhancement middleware** — not another memory tool, but a layer that makes every AI client you already use noticeably smarter, by fixing the one thing they all get wrong: **context quality**.

From planning (OpenClaw) to design (OpenDesign) to code (ZCode, Codex, Cursor, Claude Code) — Memory Arbiter connects your entire AI toolchain through one shared SQLite database.

### Why Memory Arbiter?

Your AI client loads `MEMORY.md` + `memory/*.md` into the system prompt **every turn**. As knowledge grows, 5K–20K tokens burn before the model even reads your question — and worse, the model drowns in noise, losing track of what's current, what's confirmed, and what's stale.

Memory Arbiter replaces this with a SQLite-backed search: only the relevant entries come back, everything else stays on disk.

**The result isn't just cheaper — it's sharper.** When the model receives precise, de-duplicated, trust-ranked context, the same model produces noticeably better output. Most AI execution errors aren't the model being dumb — they're the model acting on stale, contradictory, or diluted context. Fix the input, and your existing model jumps a level.

**Works with one tool. Scales to many.**

#### Better output quality (the real win)

Token savings are easy to measure. But the deeper, more impactful change is **output quality** — because Memory Arbiter fixes what the model sees before it even starts thinking.

| What pollutes AI context | How it degrades output | How Memory Arbiter fixes it |
|---|---|---|
| Key info buried in a 20K-token blob | Model attention is spread thin; it grabs the wrong detail or hallucinates. | `memory_search` returns 3–5 laser-relevant entries. High signal-to-noise. |
| Stale and current info mixed together | Model follows an outdated constraint. | Dual timeline + conflict arbitration: outdated entries are flagged or superseded. |
| "The user confirmed this" vs "the AI guessed this" — indistinguishable | Model treats a guess as ground truth. | `source_type` labels + `user_confirmed` lock. The model knows what to trust. |
| Tool switch = context reset | Each tool re-derives understanding from scratch; errors compound. | Shared memory layer: every tool starts from the same verified facts. |

**Same model. Better context. Better output.** This is the core value — everything else (token savings, cross-tool sharing, audit trails) flows from giving your AI clean, structured, de-conflicted context.

#### What does it actually enhance?

Memory Arbiter doesn't add new capabilities to your AI — it **removes the blind spots** that cap the capabilities it already has:

| Bottleneck | Without Memory Arbiter | With Memory Arbiter |
|---|---|---|
| **Attention precision** | Model scans 20K tokens, attention diluted across everything | Gets 3–5 laser-relevant entries. Attention bandwidth is freed for actual reasoning. |
| **Memory consistency** | Old and new info coexist; model may follow stale constraints | Dual timeline + conflict arbitration ensure the model only sees current, validated facts |
| **Trust calibration** | Model can't distinguish user-confirmed facts from AI guesses | `source_type` + `user_confirmed` lock — the model knows exactly what to trust |
| **Cross-tool continuity** | Switching tools = context reset, understanding drifts | Shared memory layer: every tool starts from the same verified baseline |
| **Compounding knowledge** | Memory degrades over time as files grow messy | Structured database gets richer and more precise the more you use it — a positive feedback loop |

**Model-agnostic.** Whether you run GLM, Claude, GPT, or Gemini — the stronger the model, the more sensitive it is to context quality, and the bigger the uplift Memory Arbiter delivers.

#### Solo user: still worth it with one tool

Even if you only use Cursor, Claude Code, Codex, or ZCode — Memory Arbiter upgrades your memory from flat markdown to a queryable database:

| What you get | Without Memory Arbiter | With Memory Arbiter |
|---|---|---|
| **Per-turn token cost** | Entire `MEMORY.md` + `memory/*.md` in system prompt (5K–20K tokens) | `memory_search("keyword")` returns 3–5 hits (200–800 tokens). **~80%+ saving.** |
| **Memory scale ceiling** | Bigger = slower = more expensive. You manually trim. | Thousands of entries, retrieval cost near zero. Stop trimming. |
| **Conflict detection** | AI holds old and new info simultaneously and may not notice contradictions. | `memory_compare(id1, id2)` returns a structured verdict. `memory_arbitrate` resolves it by rules. **~90% cheaper** than LLM-based comparison. |
| **Source trust levels** | All memories are flat text — "the user said this" and "the AI guessed this" look the same. | `user_confirmed` > `document_extracted` > `agent_generated` > `unknown`. High-trust entries are auto-locked. |
| **Audit trail** | Edited in-place, no history. | Every entry carries `agent_id`, `ingest_time`, `event_time`. Full traceability. |
| **Periodic self-check** | Hope the AI remembers correctly, or manually proofread. | `memory_list_conflicts` + `memory_audit_summary` — one call, structured report. |

#### Multi-tool collaboration (bonus)

Using two or more AI clients? All solo benefits apply to each tool, **plus** a shared memory layer:

| What you get | Without Memory Arbiter | With Memory Arbiter |
|---|---|---|
| **Task handoff** | Write spec to file → other tool reads file. Path sync, version drift, copy-paste. | Tool A calls `memory_write`. Tool B calls `memory_search`. **Zero file handoff.** |
| **Cross-tool visibility** | Each tool has its own memory silo. What happened in Cursor is invisible to Claude Code. | One SQLite, isolated by `workspace` / `agent_id`. Write once, search everywhere. |
| **Conflict resolution across tools** | Two tools disagree → user manually reconciles. | `memory_arbitrate` applies the same trust/timeline rules across all tools. |
| **Storage dedup** | Each tool keeps its own copy of shared knowledge. | One database, zero duplication. **100% storage dedup.** |

#### Token savings at a glance

| Scenario | Full-file loading | With Memory Arbiter | Saving |
|---|---|---|---|
| Per-turn memory load | 5K–20K tokens in system prompt | 200–800 tokens via `memory_search` | ~80%+ |
| Conflict detection | LLM compares pairs (N², thousands of tokens) | `memory_compare` returns structured verdict (~200 tokens) | ~90% |
| Periodic audit | LLM scans entire library (10K+ tokens) | `memory_list_conflicts` + `memory_audit_summary` serve structured candidates | ~70% |
| Spec handoff (2000 words) | ~3000 tokens loaded into context | ~500 tokens via targeted `memory_search` | ~83% |

**Positioning, in five lines:**
- ✅ **Execution quality booster** — clean, de-conflicted context → fewer hallucinations, sharper results (**this is the core value**)
- ✅ Structured memory storage — SQLite + dual timeline + source trust levels
- ✅ Token-optimization middleware — precise retrieval replaces full-file loading
- ✅ Conflict arbitration engine — rule-based verdicts with explainable rationale
- ✅ Cross-tool sharing layer — one database, every AI client shares it (optional but powerful)
- ❌ Not an LLM, does not do semantic reasoning — semantic judgement stays with the AI client

#### Real-world example: full creative pipeline (plan → design → code)

The user runs a three-tool pipeline: **OpenClaw** (planning & specs) → **OpenDesign** (page & slide design) → **ZCode** (implementation). Memory Arbiter connects the entire chain.

- **Old way**: OpenClaw writes a spec to a file → OpenDesign reads it, designs something → screenshots/specs are handed to ZCode. Path sync, version drift, copy-paste at every handoff. Or the user repeats the context each time — information loss + wasted tokens.
- **Memory Arbiter way**: OpenClaw calls `memory_write` with the spec → OpenDesign calls `memory_search("the project")` and gets full context, produces designs, writes back design decisions → ZCode calls `memory_search("the project")` and receives both the spec **and** the design decisions in one query — **zero file handoff across three tools.**

**A spec + design handoff that used to cost ~5000 tokens of repeated context loading now costs ~800 tokens via targeted `memory_search`.** This very project ships its own release tasks this way — it dogfoods itself. Full step-by-step in [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

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

> Change `MEMORY_ARBITER_CLIENT` for each tool (`openclaw`, `opendesign`, `zcode`, `codex`, `cursor`, `claude-code`). Point multiple tools at the same `DB_PATH` to enable cross-tool memory sharing.

> ⚠️ **New session required**: MCP servers are loaded at session startup. Already-open sessions won't see the new tools. Start a fresh session after configuring.

### Client Config Locations

| Client | Config Location |
|---|---|
| ZCode | `~/.zcode/v2/` MCP config |
| Codex CLI | `~/.codex/` MCP config |
| Claude Code | `.mcp.json` in project root |
| Cursor | `~/.cursor/mcp.json` |
| OpenDesign | OpenDesign MCP settings |
| OpenClaw | `~/.openclaw/openclaw.json` MCP config |

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

一个轻量、完全本地运行的 **AI 增强中间件**——不是又一个记忆工具，而是一层让你正在用的每个 AI 客户端都明显变聪明的底层设施，修的是所有 AI 模型的共同短板：**上下文质量**。

从规划（OpenClaw）到设计（OpenDesign）到编码（ZCode、Codex、Cursor、Claude Code）——memory-arbiter 通过一个共享的 SQLite 数据库串起你的整条 AI 工具链。

### 为什么需要 Memory Arbiter？

你的 AI 客户端每轮对话都把 `MEMORY.md` + `memory/*.md` 整个塞进 system prompt。记忆越多，token 消耗越大——模型还没读你的问题，5K–20K token 的上下文已经烧掉了。更要命的是，模型淹没在噪声里，分不清什么是最新的、什么是用户确认的、什么是过时的。

Memory Arbiter 用 SQLite 检索替代全文加载：只有相关的条目返回，其余的留在磁盘上。

**结果不只是更省钱——而是更准。** 当模型拿到的上下文是精准的、去重的、按可信度排序的，同一个模型，输出质量会明显跃升。AI 执行出错，大部分时候不是模型笨，而是它拿到的输入是过时的、自相矛盾的、或者关键信息被噪声淹没了。修好输入端，你现有的模型直接提升一个等级。

**一个工具就能用。多个工具更香。**

#### 输出质量提升（核心价值）

省 token 好量化。但更深层、更关键的改动是**输出质量**——因为 Memory Arbiter 修的是模型在开始思考之前看到的东西。

| 污染 AI 上下文的问题 | 导致的执行偏差 | memory-arbiter 怎么修 |
|---|---|---|
| 关键信息埋在 20K token 的大段文本里 | 模型注意力分散，抓错重点或直接编造 | `memory_search` 只返回 3–5 条高度相关的结果，信噪比拉满 |
| 旧信息和新信息混在一起 | 模型照着过时的约束去执行 | 双时间轴 + 冲突仲裁，过时条目被标记或淘汰 |
| "用户确认的"和"AI 猜的"分不清 | 模型把猜测当事实用 | `source_type` 标记来源，`user_confirmed` 自动锁定，模型知道该信什么 |
| 换个工具上下文就丢了 | 每个工具重新理解一遍，理解偏差累积 | 共享记忆层，所有工具从同一套已验证的事实出发 |

**同一个模型，上下文对了，输出就准了。** 这是核心价值——其他一切（省 token、跨工具共享、审计追溯）都是“给 AI 干净、结构化、无冲突的上下文”自然带来的结果。

#### 它到底增强了什么？

memory-arbiter 没有给你的 AI 加新能力——它**移除了卡住现有能力上限的盲区**：

| 瓶颈 | 没有 memory-arbiter | 用了之后 |
|---|---|---|
| **注意力精度** | 模型扫 20K token，注意力被稀释 | 拿到 3–5 条精准结果，注意力带宽释放出来做真正的推理 |
| **记忆一致性** | 新旧信息共存，模型可能照着过时的来 | 双时间轴 + 冲突仲裁，模型只看到当前已验证的事实 |
| **可信度判断** | 模型分不清用户确认的事实和 AI 的猜测 | `source_type` + `user_confirmed` 锁定，模型明确知道该信什么 |
| **跨工具连续性** | 换工具 = 上下文重置，理解偏差累积 | 共享记忆层，每个工具从同一套已验证的事实出发 |
| **知识复利** | 记忆随时间退化，文件越来越乱 | 结构化数据库越用越丰富、越精准——正反馈循环 |

**模型无关。** 不管你用 GLM、Claude、GPT 还是 Gemini——模型越强，对上下文质量越敏感，memory-arbiter 带来的提升越大。

#### 单客户端用户：只用一个工具也值得

即使你只用 Cursor、Claude Code、Codex 或 ZCode，Memory Arbiter 也能把你的记忆从扁平 markdown 升级成可查询的数据库：

| 收益 | 没有 memory-arbiter | 用了之后 |
|---|---|---|
| **每轮 token 消耗** | 整个 `MEMORY.md` + `memory/*.md` 塞 system prompt（5K–20K tokens） | `memory_search("关键词")` 返回 3–5 条（200–800 tokens）。**省 80%+。** |
| **记忆规模天花板** | 越大越慢越贵，被迫手动精简 | SQLite 存几千条，检索成本接近零。不用再精简。 |
| **冲突发现** | AI 同时记住旧信息和新信息，自己未必发现得了矛盾 | `memory_compare(id1, id2)` 返回结构化裁决，`memory_arbitrate` 按规则判定。比 LLM 逐条比对**省 90%**。 |
| **来源可信度** | 全是平铺文本，"用户说的"和"AI 猜的"看着一样 | `user_confirmed` > `document_extracted` > `agent_generated` > `unknown`，高可信条目自动锁定。 |
| **审计追溯** | 改了就改了，谁改的、什么时候改的不知道 | 每条记忆有 `agent_id` / `ingest_time` / `event_time`，完整溯源。 |
| **定时自检** | 靠 AI 自觉，或手动翻 | `memory_list_conflicts` + `memory_audit_summary` 一个调用出报告。 |

#### 多客户端协作（进阶收益）

同时用两个或更多 AI 工具？单客户端的所有收益 **每个工具都享受**，再加一层共享记忆：

| 收益 | 没有 memory-arbiter | 用了之后 |
|---|---|---|
| **任务交接** | 规格写成文件 → 另一个工具读文件，路径同步、版本混乱、复制粘贴 | 工具 A 调 `memory_write`，工具 B 调 `memory_search`。**零文件传递。** |
| **跨工具可见性** | 每个工具有自己的记忆孤岛，Cursor 里发生的事 Claude Code 看不到 | 一个 SQLite，按 `workspace` / `agent_id` 隔离，写一次处处可搜。 |
| **跨工具冲突仲裁** | 两个工具对同一件事写的不一样 → 用户手动调和 | `memory_arbitrate` 用同一套可信度/时间线规则跨工具裁决。 |
| **存储去重** | 每个工具各存一份共享知识 | 一个数据库，零重复。**存储 100% 去重。** |

#### Token 节省一览

| 场景 | 全文加载 | 用 memory-arbiter | 节省 |
|---|---|---|---|
| 每轮记忆加载 | system prompt 塞 5K–20K tokens | `memory_search` 返回 200–800 tokens | ~80%+ |
| 冲突检测 | LLM 逐条比对（N²，数千 tokens） | `memory_compare` 返回结构化裁决（~200 tokens） | ~90% |
| 定期审查 | LLM 扫全库（万级 tokens） | `memory_list_conflicts` + `memory_audit_summary` 出结构化候选 | ~70% |
| 规格交接（2000 字） | 加载进 context ~3000 tokens | 精准 `memory_search` ~500 tokens | ~83% |

**定位（五句话）：**
- ✅ **执行质量放大器** — 干净、无冲突的上下文 → 更少幻觉、更准的结果（**核心价值**）
- ✅ 结构化记忆存储 — SQLite + 双时间轴 + 来源可信度
- ✅ Token 优化中间件 — 精准检索替代全文加载
- ✅ 冲突仲裁引擎 — 规则化裁决，输出可解释理由
- ✅ 跨工具共享层 — 一个数据库，所有 AI 客户端共享（可选，但用了就回不去）
- ❌ 不是 LLM、不做语义推理 — 语义判断交给 AI 客户端

#### 真实案例：完整创意管线（规划 → 设计 → 编码）

用户跑的是三工具管线：**OpenClaw**（规划 & 规格）→ **OpenDesign**（页面 & PPT 设计）→ **ZCode**（代码实现）。memory-arbiter 串起整条链路。

- **传统方式**：OpenClaw 把规格写成文件 → OpenDesign 读文件做设计 → 设计稿和规格再交给 ZCode，每次交接都要约定路径、手动同步、版本混乱；或者用户每次重复描述背景，信息损耗大、token 浪费多。
- **memory-arbiter 方式**：OpenClaw 调 `memory_write` 写入规格 → OpenDesign 调 `memory_search("项目")` 拿到完整上下文，做设计，把设计决策写回 → ZCode 调 `memory_search("项目")` 一次拿到规格**和**设计决策——**三个工具之间零文件传递。**

**一份规格 + 设计交接：传统重复加载上下文约 ~5000 tokens，现在精准检索约 ~800 tokens。** 这个项目自己的发版任务就是这么跑的——用自己的产品喂自己的产品。完整步骤见 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)。

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

> 每个工具改一下 `MEMORY_ARBITER_CLIENT` 标识（`openclaw`、`opendesign`、`zcode`、`codex`、`cursor`、`claude-code`）。多个工具指向同一个 `DB_PATH` 即可启用跨工具记忆共享。

> ⚠️ **需要新建会话**：MCP Server 在客户端启动时加载，已经打开的会话不会识别新添加的 Server。配置好后请新建一个会话。

### 客户端配置位置

| 客户端 | 配置文件位置 |
|---|---|
| ZCode | `~/.zcode/v2/` 下 MCP 配置 |
| Codex CLI | `~/.codex/` 下 MCP 配置 |
| Claude Code | 项目根目录 `.mcp.json` |
| Cursor | `~/.cursor/mcp.json` |
| OpenDesign | OpenDesign MCP 设置 |
| OpenClaw | `~/.openclaw/openclaw.json` MCP 配置 |

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
