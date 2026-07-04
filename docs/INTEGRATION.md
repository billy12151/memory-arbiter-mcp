# Integration Guide / 集成指南

**[English](#english) | [中文](#中文)**

---

<a id="english"></a>

## English

Memory Arbiter is a token-optimization middleware: it replaces full-file memory loading with precise retrieval. This document describes three typical usage patterns and one real-world cross-tool delegation example.

### Pattern A — Per-turn retrieval (replace full-file loading)

The default in most AI clients is to load the entire `MEMORY.md` (+ `memory/*.md`) into the system prompt every turn. With Memory Arbiter, the client searches **only when needed** and loads **only what matches**.

```
user asks a question
  → memory_search(question keywords)      # 3–5 relevant entries, 200–800 tokens
  → answer with just those memories
```

Compare: the traditional way loads the whole `MEMORY.md` into the system prompt every turn, even when most of it is irrelevant to the current question. Memory Arbiter keeps the index in the prompt (small) and pulls details on demand.

**When to use**: every conversational turn. This is the default pattern and where the ~80%+ token saving comes from.

### Pattern B — Scheduled audit (cron / scheduled task)

Run periodically (e.g. daily) to catch conflicts and drift without burning model tokens on a full scan.

```
scheduled trigger
  → memory_audit_summary()                 # cheap per-workspace overview, decide if a deep dive is needed
  → memory_list_conflicts(status="open")   # unresolved conflicts
  → memory_recent(workspace="xxx")         # browse latest memories for anomalies
  → spot a suspicious pair → memory_compare(id1, id2)
  → confirmed conflict     → memory_arbitrate(mark_conflict=true)
  → generate a report and notify the user
```

`memory_audit_summary` is the cheapest entry point — pure SQL aggregation, no semantic work. Use it to decide whether a deeper, model-assisted review is worth it.

**When to use**: scheduled maintenance, knowledge-base hygiene, before handing off to a new agent.

### Pattern C — Write-time conflict check

Detect conflicts at the moment a new memory is written, before it silently diverges from existing knowledge.

```
about to write new knowledge
  → memory_search(relevant keywords)            # find existing memories on the same topic
  → memory_compare(new, existing)               # structural verdict, no LLM guess
  → conflict? → memory_arbitrate(mark_conflict=true)   # record + (optionally) supersede the loser
  → no conflict? → memory_write(...)
```

**When to use**: when a tool learns something that might contradict prior knowledge (config changes, policy updates, corrected facts).

### Real-world example — Cross-tool task delegation

**Scenario**: the user runs both OpenClaw (personal assistant, planning) and ZCode (coding tool, execution). OpenClaw drafts the task spec; ZCode implements it.

**The old way hurts**:
- OpenClaw writes the spec to a file → ZCode reads the file: requires agreed paths, manual sync, version drift.
- Or the user copy-pastes the spec into ZCode: information loss + wasted tokens.

**The Memory Arbiter way**:

```
Step 1  OpenClaw calls memory_write with the full task spec
        → memory_search("v0.2.1 release task") finds it later

Step 2  User switches to ZCode and says "memory_search the xxx task"
        → ZCode reads the complete spec, zero file handoff

Step 3  ZCode logs problems/progress with memory_write during execution
        → OpenClaw can memory_search to see ZCode's progress

Step 4  ZCode writes the result with memory_write when done
        → OpenClaw verifies, then memory_confirm locks it
```

**Token comparison**:
- **Old way**: OpenClaw writes a 2000-word spec doc → ZCode loads the whole doc into context ≈ 3000 tokens.
- **Memory Arbiter way**: ZCode `memory_search` gets the structured data ≈ 500 tokens (only the relevant content).
- **Saving**: ~83%.

**Why it works**:
1. **Structured storage** — task specs carry `subject`/`tags`/`workspace`, far easier to retrieve precisely than a bare file.
2. **Bidirectional visibility** — OpenClaw and ZCode read/write the same SQLite; no intermediary protocol.
3. **Built-in audit trail** — every write records `agent_id` + `ingest_time`; who wrote what and when is unambiguous.
4. **Conflict-safe** — if two tools record different understandings of the same task, `memory_compare` surfaces it and `memory_arbitrate` resolves it.

**Applicable to**:
- AI assistant + coding tool collaboration (OpenClaw ↔ ZCode / Cursor / Claude Code).
- Multi-agent division of labor (planner agent + executor agents).
- Team knowledge sharing (each member uses their own AI tool, memories interop).

---

<a id="中文"></a>

## 中文

Memory Arbiter 是一层 token 优化中间件：用精准检索替代全文加载。本文档介绍三种典型使用模式和一个真实的跨工具委派案例。

### 模式 A — 每轮对话按需检索（替代全文加载）

多数 AI 客户端的默认做法是每轮把整个 `MEMORY.md`（+ `memory/*.md`）塞进 system prompt。用 Memory Arbiter，客户端**只在需要时检索**，**只加载命中的内容**。

```
用户提问
  → memory_search(提问关键词)        # 命中 3–5 条相关记忆，200–800 tokens
  → 只带这些记忆回答
```

对比：传统方式每轮都把整个 `MEMORY.md` 塞进 system prompt，哪怕大部分内容和当前问题无关。Memory Arbiter 把索引留在 prompt 里（很小），细节按需拉取。

**适用时机**：每一轮对话。这是默认模式，80%+ 的 token 节省主要来自这里。

### 模式 B — 定时审查（cron / scheduled task）

定期跑（比如每天一次），用很低的成本发现冲突和漂移，不必让模型扫全库。

```
定时触发
  → memory_audit_summary()                 # 廉价的按 workspace 概览，先决定要不要深入
  → memory_list_conflicts(status="open")   # 未解决的冲突
  → memory_recent(workspace="xxx")         # 浏览最新记忆找异常
  → 发现可疑对 → memory_compare(id1, id2)
  → 确认冲突 → memory_arbitrate(mark_conflict=true)
  → 生成报告通知用户
```

`memory_audit_summary` 是最廉价的入口——纯 SQL 聚合，不做任何语义判断。用它决定是否值得做一次需要模型介入的深入审查。

**适用时机**：定时维护、知识库保洁、向新 agent 交接前。

### 模式 C — 写入时冲突检测

在写入新记忆的那一刻就检测冲突，避免它悄悄偏离已有知识。

```
准备写入新知识
  → memory_search(相关关键词)                 # 找同主题的已有记忆
  → memory_compare(new, existing)             # 结构化裁决，不用 LLM 猜
  → 有冲突？ → memory_arbitrate(mark_conflict=true)   # 记录，可选地把败方标记为 superseded
  → 无冲突？ → memory_write(...)
```

**适用时机**：工具学到可能与既有知识矛盾的内容时（配置变更、策略更新、纠正事实）。

### 最佳实践案例 — 跨工具任务委派

**场景**：用户同时使用 OpenClaw（个人助手，负责规划）和 ZCode（编程工具，负责执行）。OpenClaw 出任务规格，ZCode 实现。

**传统方式的问题**：
- OpenClaw 把任务规格写成文件 → ZCode 读文件：需要约定路径、手动同步、版本混乱。
- 或者用户把规格口述/复制粘贴给 ZCode：信息损耗 + 浪费 token。

**memory-arbiter 方式**：

```
Step 1  OpenClaw 用 memory_write 写入完整任务规格
        → 之后 memory_search("v0.2.1 发版任务") 即可找到

Step 2  用户切到 ZCode，说 "memory_search 查一下 xxx 任务"
        → ZCode 直接读到完整规格，零文件传递

Step 3  ZCode 执行过程中用 memory_write 记录问题/进展
        → OpenClaw 可以 memory_search 查到 ZCode 的进展

Step 4  ZCode 完成后用 memory_write 写入结果
        → OpenClaw 验证后用 memory_confirm 锁定
```

**Token 消耗对比**：
- **传统方式**：OpenClaw 写 2000 字规格文档 → ZCode 加载整个文档到 context ≈ 3000 tokens。
- **memory-arbiter 方式**：ZCode `memory_search` 拿到结构化数据 ≈ 500 tokens（只有相关内容）。
- **节省**：~83%。

**为什么有效**：
1. **结构化存储**：任务规格带 `subject`/`tags`/`workspace`，比裸文件更容易精确检索。
2. **双向可见**：OpenClaw 和 ZCode 读写同一个 SQLite，不需要中间协议。
3. **天然审计**：每次写入都记录 `agent_id` + `ingest_time`，谁写的、什么时候写的，一清二楚。
4. **冲突安全**：两个工具对同一任务写了不同理解时，`memory_compare` 能发现，`memory_arbitrate` 能裁决。

**适用场景**：
- AI 助手 + 编程工具协同（OpenClaw ↔ ZCode / Cursor / Claude Code）。
- 多 agent 分工（主 agent 规划 + 子 agent 执行）。
- 团队多成员共享知识（每人用自己的 AI 工具，记忆互通）。
