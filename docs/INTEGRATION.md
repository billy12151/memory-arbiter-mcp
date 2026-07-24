# Integration Guide / 集成指南

**[English](#english) | [中文](#中文)**

---

<a id="english"></a>

## English

Memory Arbiter is a token-optimization middleware: it replaces full-file memory loading with precise retrieval. This document describes three typical usage patterns and one real-world cross-tool delegation example.

### Configuration

Configuration is read in this order:

1. `MEMORY_ARBITER_CONFIG` if set.
2. `~/.config/memory-arbiter/config.json`.
3. Environment variables and defaults.

No config file is required for lexical recall. For semantic recall, prefer the config file for shared paths and vec/model settings because it is user-owned XDG config and will not be overwritten by pip installs or MCP client reinstall/migration. Each row below shows the JSON path and its env fallback (config file wins when both are set). Keep per-client identity (`MEMORY_ARBITER_CLIENT`, `MEMORY_ARBITER_AGENT_ID`) in each MCP client's env block so tools do not all collapse to one global identity.

**Config file fields** (`~/.config/memory-arbiter/config.json`) — paths, vector, and embedding settings:

| JSON path | Env fallback | Default | Purpose |
|---|---|---|---|
| `db_path` | `MEMORY_ARBITER_DB_PATH` | `./memory_arbiter.sqlite3` | SQLite database location. Set to a shared path if multiple tools must see the same store. |
| `backup_jsonl` | `MEMORY_ARBITER_BACKUP_JSONL` | `./memory_arbiter.backup.jsonl` | Append-only JSONL backup, used only when SQLite is read-only. |
| `policy_path` | `MEMORY_ARBITER_POLICY` | _(none)_ | Path to a JSON policy file for per-client enable/disable and agent allow/deny lists. |
| `vec.enabled` | `MEMORY_ARBITER_ENABLE_SQLITE_VEC` | `false` | Enable sqlite-vec for semantic recall. Off by default (lexical recall works without it). Requires `pip install memory-arbiter-mcp[vec]`. Falls back gracefully if the package is missing. |
| `vec.dim` | `MEMORY_ARBITER_VEC_DIM` | `768` | Embedding dimension. **Must match** the model you backfill with (e.g. bge-small-zh=512, bge-base=768). Changing it requires dropping and recreating `memories_vec`. |
| `recall_pool_cap` | `MEMORY_ARBITER_RECALL_POOL_CAP` | `50` | Max candidates pooled across all recall channels before soft-rerank. Raise to 100–200 when your store exceeds ~100 entries to avoid losing matches at the pool edge. |
| `content_like_cap` | `MEMORY_ARBITER_CONTENT_LIKE_CAP` | `30` | Max candidates the content-LIKE补漏 channel contributes. Raise if many same-topic memories exist. |
| `embedding.provider` | `MEMORY_ARBITER_EMBEDDING_PROVIDER` | inferred as `gguf` only when `embedding.model_path` is set | Only `gguf` is supported in v0.5.0. Without a model path, auto-embedding stays off. |
| `embedding.model_path` | `MEMORY_ARBITER_EMBEDDING_MODEL_PATH` (or legacy `MEMORY_ARBITER_GGUF`) | _(none)_ | Path to the GGUF embedding model for v0.5.0 auto-embedding. |
| `embedding.auto_query` | `MEMORY_ARBITER_EMBEDDING_AUTO_QUERY` | `true` | Auto-encode plain-text queries so semantic recall works without explicit `query_embedding`. |
| `embedding.auto_write` | `MEMORY_ARBITER_EMBEDDING_AUTO_WRITE` | `true` | Auto-embed new writes/edits so they enter semantic recall immediately. |

**Environment variables** — keep per-client identity in each MCP client's env block. Some fields also have config-file equivalents, but config wins; use env here when the value must differ by client/session.

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_ARBITER_CLIENT` | `codex` | Client identifier (e.g. `codex`, `claude-code`, `cursor`, `zcode`). Used for policy checks. |
| `MEMORY_ARBITER_AGENT_ID` | `default` | Agent identity within a client. |
| `MEMORY_ARBITER_WORKSPACE` | `default` | Record field on each memory. Not used for search filtering (v0.6.2). |
| `MEMORY_ARBITER_CONFIG` | _(none)_ | Optional path to an alternate JSON config file. If set, memory-arbiter reads that file instead of the default `~/.config/memory-arbiter/config.json`; file values still override other env fallbacks. |
| `MEMORY_ARBITER_RANKING_MODE` | `hybrid` | `hybrid` (wide recall + soft rerank, default) or `bm25` (legacy v0.2.6 single-FTS). No config-file equivalent. |
| `MEMORY_ARBITER_GGUF` | _(none)_ | Legacy GGUF model path fallback. Prefer `embedding.model_path` in the config file for v0.5.0 auto-embedding. |

**When to tune**: if you notice relevant memories missing from results as the store grows, the first knob to turn is `MEMORY_ARBITER_RECALL_POOL_CAP`. The default `50` is conservative; `100` is a safe bump for stores up to a few hundred entries.

For a full config-file template, see [`examples/memory-arbiter.config.example.json`](../examples/memory-arbiter.config.example.json).

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

**v0.7.5–v0.7.6 conflict scan**: `memory_scan_conflict_candidates` vector-recalls candidate conflict pairs (incremental: only new + recently edited memories). It returns pairs with distance/excerpt/tags — no LLM, no writes. The calling agent runs LLM comparison on each pair, then persists the verdict with `memory_record_conflict` (idempotent, carries `conflict_type`/`suggested_winner`/`source`). If an open conflict already exists but the memory version or scan model changed since it was recorded, re-run LLM and persist with `memory_record_conflict(refresh=true)` to update the enrichment fields in place. Dismiss false positives with `memory_resolve_conflict`. `memory_doctor_overview` reports scan freshness via `scan_log.jsonl` (warns if never scanned or stale > 15 days).

**v0.7.6 consuming conflict signals**: when `memory_search` returns a result with a `conflict_signal` field, read `conflict_source` to decide:
- `open_table`: the conflict is scan/record-verified. Mention to the user "this memory has an unresolved conflict" and optionally guide them to `memory_list_conflicts` for details. Use `suggested_winner`/`confidence_hint` to decide who to trust.
- `runtime_metadata_hint`: advisory only, not LLM-verified. Surface as a low-confidence hint ("there may be a duplicate"), don't auto-delete or auto-supersede based on it alone.

**Batch arbitration workflow** (v0.7.6): when the user says "handle these conflicts" or "arbitrate by suggestion":
1. `memory_list_conflicts(status="open")` → filter to `confidence_hint == "high"` with a `suggested_winner`.
2. For each conflict, the loser is the side that is NOT `suggested_winner`. Check the loser's `protection_level`/`source_type` — skip `locked`/`user_confirmed` losers unless the user explicitly confirms.
3. `memory_supersede(memory_id=loser_id, superseded_by=suggested_winner, authorized=true, reason="batch arbitrate: conflict #N")` for safe losers.
4. Low-confidence (`confidence_hint == "low"`) conflicts are always skipped — leave for manual review.

**When to use**: scheduled maintenance, knowledge-base hygiene, before handing off to a new agent.

### Pattern C — Write-time conflict check

Detect conflicts at the moment a new memory is written, before it silently diverges from existing knowledge.

```
write new knowledge via memory_write(...)
  → check response for write_hints.possible_supersede_targets
  → hint present? → memory_search to verify, then memory_supersede the stale one
  → no hint? → done (low-confidence conflicts will be caught by scheduled scan)
```

**v0.7.6**: `memory_write` now synchronously returns `write_hints` when an active memory shares high subject/tags overlap with the just-written record. Two hint types: `possible_duplicate` (likely the same thing) and `possible_evolution_of` (new content is significantly longer — the new one may supersede the candidate). Hints are advisory; they never write to the conflicts table. If a hint fires, the agent can prompt the user or immediately supersede the stale one. Semantic conflicts (where subject/tags don't overlap) are still the domain of the scheduled vector scan — not write-time hints.

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

### 配置

配置读取顺序：

1. 如果设置了 `MEMORY_ARBITER_CONFIG`，先读它。
2. 再读 `~/.config/memory-arbiter/config.json`。
3. 最后用环境变量和默认值兜底。

纯字面检索不需要配置文件。开启语义检索时，建议把共享路径和 vec/model 配置放在配置文件里，因为这是用户自己的 XDG 配置目录，不会被 pip 安装或 MCP 客户端重装/迁移覆盖。下面每行同时给出 JSON 路径和对应的 env 兜底（两者都设时配置文件优先）。每客户端身份（`MEMORY_ARBITER_CLIENT`、`MEMORY_ARBITER_AGENT_ID`）仍放各 MCP 客户端 env，避免所有工具被全局 config 覆盖成同一个身份。

**配置文件字段**（`~/.config/memory-arbiter/config.json`）——路径、向量、embedding 设置：

| JSON 路径 | env 兜底 | 默认值 | 用途 |
|---|---|---|---|
| `db_path` | `MEMORY_ARBITER_DB_PATH` | `./memory_arbiter.sqlite3` | SQLite 数据库路径。多个工具要共享记忆时设成同一路径。 |
| `backup_jsonl` | `MEMORY_ARBITER_BACKUP_JSONL` | `./memory_arbiter.backup.jsonl` | 追加式 JSONL 备份，仅在 SQLite 只读时启用。 |
| `policy_path` | `MEMORY_ARBITER_POLICY` | _(无)_ | 策略 JSON 文件路径，可按客户端开关、按 agent 允许/拒绝。 |
| `vec.enabled` | `MEMORY_ARBITER_ENABLE_SQLITE_VEC` | `false` | 启用 sqlite-vec 走语义检索。默认关闭（纯字面检索不需要它）。需要 `pip install memory-arbiter-mcp[vec]`，包缺失时优雅降级。 |
| `vec.dim` | `MEMORY_ARBITER_VEC_DIM` | `768` | embedding 维度。**必须和你灌向量用的模型一致**（bge-small-zh=512、bge-base=768）。改维度要重建 `memories_vec` 表。 |
| `recall_pool_cap` | `MEMORY_ARBITER_RECALL_POOL_CAP` | `50` | 多路召回合并进候选池的上限，之后才走软重排。记忆超过约 100 条时建议调到 100–200，避免边界漏召回。 |
| `content_like_cap` | `MEMORY_ARBITER_CONTENT_LIKE_CAP` | `30` | content LIKE 补漏路最多贡献的候选数。同主题记忆多时调大。 |
| `embedding.provider` | `MEMORY_ARBITER_EMBEDDING_PROVIDER` | 仅在设置 `embedding.model_path` 时推断为 `gguf` | v0.5.0 只支持 `gguf`。没有模型路径时，自动向量化保持关闭。 |
| `embedding.model_path` | `MEMORY_ARBITER_EMBEDDING_MODEL_PATH`（或 legacy `MEMORY_ARBITER_GGUF`） | _(无)_ | GGUF embedding 模型路径，用于 v0.5.0 自动向量化。 |
| `embedding.auto_query` | `MEMORY_ARBITER_EMBEDDING_AUTO_QUERY` | `true` | 自动 encode 纯文本查询，不传 `query_embedding` 也能触发语义检索。 |
| `embedding.auto_write` | `MEMORY_ARBITER_EMBEDDING_AUTO_WRITE` | `true` | 新写入/编辑自动灌向量，立即进语义召回。 |

**环境变量**——每客户端身份建议放在各自 MCP env 段。部分字段也有配置文件对应项，但 config 优先；当某个值必须按客户端/会话变化时再放 env。

| 变量 | 默认值 | 用途 |
|---|---|---|
| `MEMORY_ARBITER_CLIENT` | `codex` | 客户端标识（如 `codex`、`claude-code`、`cursor`、`zcode`），用于策略判断。 |
| `MEMORY_ARBITER_AGENT_ID` | `default` | 客户端内的 agent 身份。 |
| `MEMORY_ARBITER_WORKSPACE` | `default` | 记忆记录上的字段。v0.6.2 起不再用于搜索过滤。 |
| `MEMORY_ARBITER_CONFIG` | _(无)_ | 可选：指定另一个 JSON 配置文件路径。设置后读取该文件，而不是默认的 `~/.config/memory-arbiter/config.json`；配置文件里的字段仍然优先于其他 env 兜底值。 |
| `MEMORY_ARBITER_RANKING_MODE` | `hybrid` | `hybrid`（宽召回 + 软重排，默认）或 `bm25`（legacy v0.2.6 单 FTS）。无配置文件对应。 |
| `MEMORY_ARBITER_GGUF` | _(无)_ | 旧版 GGUF 模型路径兜底。v0.5.0 自动向量化建议改用配置文件里的 `embedding.model_path`。 |

**什么时候调**：记忆库变大后，如果发现相关记忆从结果里消失了，第一个该调的就是 `MEMORY_ARBITER_RECALL_POOL_CAP`。默认 `50` 偏保守，记忆到几百条时调到 `100` 比较稳。

完整配置文件模板见 [`examples/memory-arbiter.config.example.json`](../examples/memory-arbiter.config.example.json)。

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

**v0.7.5–v0.7.6 冲突扫描**：`memory_scan_conflict_candidates` 向量召回候选冲突对（增量：只扫新增 + 最近编辑的记忆），返回带 distance/excerpt/tags 的候选对——无 LLM、不写库。调用方 agent 对每对跑 LLM 比对后，用 `memory_record_conflict` 落表（幂等，带 `conflict_type`/`suggested_winner`/`source`）。如果同一对已有 open 冲突、但之后记忆版本或扫描模型变了，重跑 LLM 后用 `memory_record_conflict(refresh=true)` 原地更新富化字段。误报用 `memory_resolve_conflict` 关闭。`memory_doctor_overview` 通过 `scan_log.jsonl` 报告扫描新鲜度（从未扫描或超过 15 天会 WARN）。

**v0.7.6 消费冲突信号**：当 `memory_search` 返回结果带 `conflict_signal` 字段时，按 `conflict_source` 决定怎么处理：
- `open_table`：经 scan/record 验证过的冲突。可以提示用户"这条记忆有未解决冲突"，并引导到 `memory_list_conflicts` 看详情；用 `suggested_winner`/`confidence_hint` 判断该信哪一边。
- `runtime_metadata_hint`：仅运行时启发式提示，未经 LLM 验证。当作低置信线索（"可能有重复"）处理，不要据此单独 auto-delete 或 auto-supersede。

**批量仲裁工作流**（v0.7.6）：当用户说"处理一下这些冲突"或"按建议仲裁"时：
1. `memory_list_conflicts(status="open")` → 筛出 `confidence_hint == "high"` 且有 `suggested_winner` 的。
2. 每条冲突的败方 = 不是 `suggested_winner` 的那一侧。检查败方的 `protection_level`/`source_type`——`locked`/`user_confirmed` 的败方跳过，除非用户明确确认。
3. 对安全的败方：`memory_supersede(memory_id=败方id, superseded_by=suggested_winner, authorized=true, reason="批量仲裁: 冲突 #N")`。
4. 低置信（`confidence_hint == "low"`）的冲突一律跳过，留作人工审查。

**适用时机**：定时维护、知识库保洁、向新 agent 交接前。

### 模式 C — 写入时冲突检测

在写入新记忆的那一刻就检测冲突，避免它悄悄偏离已有知识。

```
memory_write(...) 写入新知识
  → 检查响应里的 write_hints.possible_supersede_targets
  → 有 hint？ → memory_search 核实，然后 memory_supersede 废弃旧的那条
  → 没有？   → 完成（低置信冲突留给定时扫描）
```

**v0.7.6**：`memory_write` 现在会在写入后同步返回 `write_hints`——当某条 active 记忆与新写入的 subject/tags 高度重叠时触发。两种 hint 类型：`possible_duplicate`（疑似同一条）和 `possible_evolution_of`（新内容明显更长——新的可能要取代旧候选）。hint 仅供参考，绝不写 conflicts 表。命中时 agent 可以提示用户或直接 supersede 旧的那条。语义层面的冲突（subject/tags 不重叠）仍归定时向量扫描管，不是写入时 hint 的职责。

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
