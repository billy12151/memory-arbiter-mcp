# memory-arbiter 长文分段检索详细设计（实施基线）

> 目标版本：v0.6.0
> 依赖现状：v0.5.4，代码库 `/Users/zhangzhiwei17/OpenClawProject/memory-arbiter-mcp`
> 前身：`doc/历史/section-split-detailed-design-v0.5.0-已废弃.md`（三层索引方案，因复杂度过高废弃）
> 真实场景驱动：金营项目-完整知识库 18264 字符（12 个 `##` 章节）、鹊桥小金库超级赞-接口文档 52238 字符，均为 split_threshold（4000）的 4.5~13 倍。

## 0. 为什么重做这个设计

### 0.1 废弃的旧方案做了什么

旧方案（v0.5.0 设计，已废弃）为每条长记忆新建 **三张表**：
- `memory_sections`（段落元数据 + 偏移量）
- `memory_sections_fts`（段落级 FTS5 精确召回，external content 模式）
- `memory_sections_vec`（段落级 vec0 语义召回）

三张表各有独立的写入、校验、降级、编辑清理路径。其中 FTS external content 模式要求 `memory_sections` 表冗余存一份 `body` 列；embedding 任一生成失败就要整批不写（强一致发布）；编辑场景要事务内删三张表。复杂度集中在"如何维护三套独立索引的一致性"上。

此外，旧方案的 `memory_write` 走"待确认"两阶段协议——长文本到达后**先不落库**，等用户确认分段后才写。这引入一个尴尬的"已知取舍"：用户既未同意也未拒绝、对话就中断，原文直接丢失。

### 0.2 新方案的核心思想

**一句话**：原文照常入库（继续走现有 memory 级 FTS/Vec/LIKE 召回，不新增召回通道；memory Vec 只增加向量空间一致性门禁），LLM 只负责给每个段落打标题/摘要 + 锚点（无标题纯文本路径），arbiter 用锚点算出偏移量 + 生成段落向量存进两张表；查询时命中 memory 后，再用段落级语义匹配把相关段落挑出来返回。

**核心定位：sections 只是增强，不影响原有能力。** 向量是分段发布的**硬前提**——只有 `sqlite_vec 可用 + embedding 配置可用` 时才允许新建/重建 sections。能力从一开始就不可用时，长记忆照常普通入库（`split_status=NULL`），走现有 memory 级检索，不触发分段流程；已经存在 active sections、但 query-time Vec 门禁临时关闭时，search 必须回退返回全文。

**关键改进：原文立即入库，分段是事后补齐的增强。** 长文本写入时照常落库，返回里带一句"已保存，建议分段以提升检索精度"；用户同意后走 `memory_split` 补齐 `memory_sections` + `memory_sections_vec` 两张表。这样：
- 对话中断 → 原文已安全（零丢失，彻底消除旧方案的"已知取舍"）
- 分段入口统一归 `memory_split`，`memory_write` 回归"只管写入"
- 向量不可用 → 不分段，原文照常入库检索（sections 增强不激活，但不影响原有能力）

对比旧方案：

| 维度 | 旧方案（废弃） | 新方案 |
|------|---------------|--------|
| 新增表 | 3 张业务索引表（sections + FTS + Vec） | **2 张业务索引表**（sections + sections_vec）+ 1 张全局 KV 元数据表（`_vec_index_meta`） |
| 新增召回通道 | 2 条（section FTS + section Vec，改 `_wide_recall` 主路径） | **0 条**（不改召回主路径；section Vec 仅用于命中后挑段） |
| 向量依赖 | 强（分段依赖向量，但 embedding 失败会降级） | **硬前提**（向量不可用就不分段，不存在"分段了但没向量"的状态） |
| 编辑清理 | 事务内删三张表 | **删两张表** |
| 原文落库时机 | 用户确认后 | **立即**（消除对话中断丢原文的风险） |
| 降级路径 | 三层索引全挂 → 返回全文 | 向量不可用 → 不分段（走现有检索，返回全文） |

### 0.3 设计原则

1. **原文为唯一主数据** — `memories.content` 存完整原文，不改一个字
2. **段落元数据是派生索引** — section 从原文偏移生成，坏了可重建，删了不影响原文
3. **存储/检索/返回三者解耦** — 检索走现有 memory 级通道；分段只是"命中后挑哪段返回"的事
4. **全文能力始终保底** — `memory_get` 永远返回完整原文；即使分段全失败，记忆照常写入、照常搜到
5. **原文先入库，分段是事后增强** — 分段不阻塞写入，分段失败不丢数据，分段入口统一走 `memory_split`
6. **向量是分段发布的硬前提** — 新 sections 只有在全部向量生成成功时才允许发布；已发布 sections 在查询时若 Vec 门禁临时关闭，必须返回全文并关闭 section 增强，原有检索能力不得退化
7. **一次响应只使用一个一致快照** — `memory_search` 返回前重新物化当前 memory + sections；禁止把 rerank 阶段缓存的 `split_status/content` 与稍后读取的 sections 混用
8. **昂贵计算使用乐观并发控制** — LLM/embedding 在事务外执行，split 发布时 CAS `content/version + split_status + split_revision + embedding space`；迁移提交还要 CAS `migration epoch + lease owner + expected cursor`。冲突可见且不得覆盖别人的成功结果
9. **LLM 决定批次内部的语义边界** — 对于没有 Markdown 标题的纯文本，LLM 通过语义理解决定每个 batch 内部的段落边界（给出 anchor）；arbiter 只做确定性验证和 offset 计算。为满足上下文窗口，非重叠 batch 的首尾是强制技术边界，LLM 不得跨 batch 合并 section；该约束必须在提示和诊断中对调用方可见
10. **事务必须独占连接** — SQLite transaction 是 connection-scoped；每个工具调用/事务使用独立连接，禁止在并发调用间共享一个长期 `self.conn`。耗时 LLM/embedding 一律在事务外，事务内只做快照读取、CAS 和短写入

### 0.4 准入条件（高阶功能，默认关闭）

分段检索是一个**高阶增强功能**，不是默认行为。开启它需要满足以下**全部**条件：

1. `split_enabled = True`（配置显式开启）
2. `sqlite_vec_available = True`（sqlite-vec 扩展已安装并加载成功）
3. 托管 GGUF embedding 配置可用（provider/model_path 已配好，能正常生成向量）
4. `_vec_index_meta.state == ready`（不在模型迁移/失败/unmanaged 状态）
5. `len(content) > split_threshold`（内容超过字符阈值）

**条件 1 不满足**：`memory_write` 不返回 `split_hint`，`memory_split` 拒绝调用。所有记忆按普通路径处理。

**条件 2 或 3 不满足**：向量不可用 → 分段增强无法激活。`memory_write` 不返回 `split_hint`，`memory_split` 首次校验直接返回“分段增强不可用，请先配置托管 GGUF embedding”。所有长记忆按普通路径入库，走现有 memory 级检索。**不会出现“分段了但没向量”的状态。**

**条件 4 不满足**：向量索引正在迁移、迁移失败或处于 unmanaged。`memory_write` 不返回 `split_hint`，`memory_split` 返回当前 `vec_index_state` 和明确提示；用户先完成 `memory_rebuild_embeddings`。维护窗口内不发布新的 sections，避免出现“split_status=active 但 section Vec 全局不可用”的惊讶状态。

**条件 5 不满足**：内容太短，分段无意义。`memory_write` 不返回 `split_hint`，`memory_split` 返回“无需分段”。

**为什么默认关闭**：分段需要 1..N 个外部 LLM 批次（生成 sections，次数由正文长度和调用方上下文预算决定），需要 embedding 服务运行正常（生成段落向量），需要 sqlite-vec 扩展已安装。对大多数用户来说，这些依赖不是必需的——他们的记忆长度可能永远不超过阈值，或者他们不需要段落级检索精度。只在确认所有依赖可用、且确实有长文检索需求时，才应开启。

**`split_threshold` 为什么推荐 4000**：Python `len()` 统计的是字符数，中文一字一字符，4000 字符 = 4000 个中文字。大约对应 8-20 个自然段落（中文偏多，英文偏少）。低于这个长度的文档，用户扫一眼就能找到关键信息，分段带来的检索精度提升不大，却仍要消耗至少 1 个 LLM 批次。4000 的阈值能把 token 花在刀刃上——只对真正需要分段的长文触发。详见 5 节配置项。

**不开启分段的用户体验**：分段相关行为与 v0.5.4 一致，`split_status` 永远为 NULL，`memory_sections` + `memory_sections_vec` 不会写入数据。唯一的全局变化是 memory Vec 增加向量空间一致性门禁：旧库首次升级且已有无法确认来源的向量时，会暂时禁用 Vec、保留 FTS/LIKE，并明确提示执行一次向量重建；详见 1.1b 和 6.2。

## 1. Schema 变更

### 1.1 新增表 `memory_sections`

```sql
CREATE TABLE IF NOT EXISTS memory_sections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id INTEGER NOT NULL,
  section_index INTEGER NOT NULL,            -- 从 0 开始的有序序号
  title TEXT,                                -- LLM 生成（纯文本）或解析器提取（结构文档标题）
  title_path TEXT,                           -- 层级路径，展示用，可空
  summary TEXT,                              -- LLM 生成，目录展示用
  anchor_text TEXT,                          -- 段落边界锚点（arbiter 算 offset 用）
  occurrence_index INTEGER NOT NULL DEFAULT 0,-- 锚点在原文第几次出现（0-based）
  start_offset INTEGER NOT NULL,             -- 字符偏移，arbiter 算，写入时填好
  end_offset INTEGER NOT NULL,               -- 字符偏移，arbiter 算，写入时填好
  provenance TEXT NOT NULL,                  -- 'parser' | 'llm'：title 来源
  embedding_truncated INTEGER NOT NULL DEFAULT 0, -- 1=该 section 的 embedding body 因字符预限或 token 预算被截断
  embedding_original_tokens INTEGER NOT NULL DEFAULT 0, -- prefix+原始 body 的 tokenizer 计数
  embedding_used_tokens INTEGER NOT NULL DEFAULT 0,     -- 实际送入模型的 token 数；用于诊断截断比例
  created_at TEXT NOT NULL,
  FOREIGN KEY(memory_id) REFERENCES memories(id),
  UNIQUE(memory_id, section_index)           -- DB 级防重：同一 memory 下 section_index 唯一
);
-- 不再单独建 UNIQUE INDEX：表级 UNIQUE 约束已自动创建隐式索引
```

**不存 `body` 列**：原文片段永远通过 `content[start_offset:end_offset]` 现算现取，零冗余。没有 section FTS 就不需要 body 列。

**关键约束（应用层校验 + DB 约束）：**
- 同一 `memory_id` 下 `section_index` 从 0 连续、不重叠（DB UNIQUE 约束兜底防重）
- `start_offset` 严格递增
- 相邻 section：`sections[i].end_offset == sections[i+1].start_offset`
- 最后一个 section：`end_offset == len(content)`
- 第一个 section：`start_offset == 0`（含标题前 preamble）
- **section 数量约束**：单条 memory 的最终 section 数量必须 ≥ 2 且 ≤ `max_sections`（默认 50）。少于 2 段 → 分段无意义（等于没分），拒绝；超出上限 → 分段失败，走保底。防止 LLM 对超长文档生成了过多碎片段落，导致返回膨胀
- **单段 embedding 上限**：helper 先用 tokenizer 统计 `prefix + 完整原始 body` 得到诊断用 `original_tokens`，但只取 `max_section_chars`（默认 3600）的 body 构造 embedding 候选，再按 `n_ctx-reserved_tokens`（包含 title_path）二次截断；任一截断发生都标记 `embedding_truncated=1`。完整原文只用于计数和按 offset 返回，绝不整段送入模型
- **截断可诊断**：同时记录 `embedding_original_tokens` 和 `embedding_used_tokens`；不能只存布尔值，否则无法区分“仅少几个 token”和“只覆盖了前 10%”
- 拼接还原：`"".join(content[s.start_offset:s.end_offset] for s in sections) == content`

这些约束**只在写入时校验一次**，运行期不再维护——因为 section 表是只写的派生索引，除了"编辑时整表清空"之外不会被局部修改。

### 1.1b 段落级语义匹配（命中后定位，不参与召回）

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memory_sections_vec USING vec0(
  id INTEGER PRIMARY KEY,      -- = memory_sections.id（不是 memory_id）
  embedding float[{dim}]       -- dim 复用 settings.vec_dim
);
```

向量语义输入为 `title_path + content[start_offset:end_offset]`，其中 `title_path` 来源取决于 `provenance`（parser 提取 vs LLM 生成）；实际送入模型前必须经过下述统一 token-safe helper，必要时只截断 body、不改变原文 offset。

**向量空间身份（`embedding_space_id`）**：切换模型后，旧向量和新 query 向量在不同向量空间比较 → 结果静默错误。需要一种可靠的方式判断"向量空间是否一致"。

**身份计算（v0.6.0 的托管 GGUF 路径）**：
- **模型身份**（`content_digest`）：模型文件的 SHA-256，server 启动时计算一次并内存缓存（300MB 文件约 1-2 秒）。`size + mtime` 仅作诊断信息，不参与兼容性判断
- **向量空间身份**（`embedding_space_id`）：模型身份 + 向量维度 + 所有会影响向量输出的有效配置，使用稳定、无歧义的 canonical JSON 计算

```python
payload = {
    "provider": "gguf",
    "model_sha256": model_content_digest,
    "dim": dim,
    "pipeline_version": EMBEDDING_PIPELINE_VERSION,
    # 只放确实影响输出的有效配置；示例：pooling / normalize / query、document prefix
    "effective_config": effective_embedding_config,
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
embedding_space_id = sha256(canonical.encode("utf-8")).hexdigest()
# ⚠️ 不得使用 Python 内置 hash()——它跨进程随机化，重启后值不同
```

⚠️ `model_path`、`size`、`mtime` 只用于诊断，不参与 `embedding_space_id` 计算——换目录不应误判 mismatch，touch 文件不应触发重建。任何影响 pooling、normalize、query/document prefix 或输入构造的代码变化，都必须修改有效配置或递增 `EMBEDDING_PIPELINE_VERSION`。

**统一的 token-safe embedding 输入规则**：所有 memory/query/section embedding 必须走同一个 helper，不能把任意长度字符串直接交给 `create_embedding`。真实样例最长 52238 字符，远超 GGUF 的 `n_ctx=2048`；如果没有统一截断，`memory_write` 和 `memory_rebuild_embeddings` 会直接失败。

```python
embed_text(prefix, body) -> (embedding, truncated, original_tokens, used_tokens):
    # 使用当前模型 tokenizer 精确计数；字符数只能做快速预筛，不能作为最终边界
    original_tokens = tokenizer.count(prefix + body)  # 可超过 n_ctx，只计数，不送入模型
    body_candidate = body[:max_body_chars] if max_body_chars else body  # section 传 max_section_chars；memory/query 传 None
    token_budget = n_ctx - reserved_tokens  # v0.6 建议 reserved_tokens=64，容纳 BOS/EOS/模板开销
    优先保留 prefix（memory=subject，section=title_path），剩余预算给 body_candidate
    超预算时按 token 截断 body_candidate；used_tokens=实际模型输入计数，生成 embedding
```

- section：`prefix=title_path`、`body=content[start:end]`，写入 truncated + original/used token counts
- memory：`prefix=subject`、`body=content`，在 `memories.metadata._embedding` 记录 `memory_embedding_truncated/original_tokens/used_tokens` 并返回 warning；`memory_write`、`memory_edit`、`memory_rebuild_embeddings` 必须复用同一 helper
- query：通常很短；若仍超预算，同样 token 截断并返回 warning
- `max_section_chars` 只作为进入 tokenizer 前的防御性字符上限，且预算必须包含 `title_path`，不能先取满 3600 字符再额外拼标题
- `n_ctx`、`reserved_tokens`、截断策略版本、`max_section_chars` 都属于 `effective_embedding_config`；变化后必须产生新的 `embedding_space_id` 并重建

**维护窗口语义（非渐进迁移）**：采用全局维护窗口，不做 per-memory 渐进恢复。理由：本地单用户 MCP，用户主动切模型、主动跑重建，一个维护窗口足够。不实现"一批可用、一批不可用"的混合状态。

新建元数据表（**不用 PRAGMA user_version**——user_version 应留给 schema 版本）：
```sql
CREATE TABLE IF NOT EXISTS _vec_index_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- 存储以下键值对：
-- active_space_id   = 当前向量空间身份（重建成功后原子更新；尚未建立可信空间时该 key 不存在）
-- target_space_id   = 目标空间身份（重建期间 != active；ready 时该 key 不存在）
-- state             = unmanaged | ready | mismatch | failed
-- migration_cursor  = 已连续完成迁移的最大 memory_id；尚未完成任何一条时该 key 不存在
-- migration_epoch   = 当前目标迁移世代 UUID；每次建立/更换 target_space_id 时生成，ready 时不存在
-- migration_lease_owner      = 当前执行迁移的调用 UUID；无持有者时不存在
-- migration_lease_expires_at = lease 的 UTC 过期时间；无持有者时不存在
-- last_error        = 失败原因（state=failed 时；无错误时该 key 不存在）
```

**状态流转**：
- `unmanaged`：未配置可由 server 识别的 GGUF embedder，仅使用 v0.5.4 的显式 `memory_store_embedding/query_embedding` 路径。server 无法证明向量空间兼容，保留旧行为并在 status 中暴露 `vec_space_unmanaged=true`；分段增强不允许激活
- `ready`：正常工作，memory Vec + section Vec 全部可用
- `mismatch`：server 启动时检测到当前 embedder 的 `embedding_space_id` != `active_space_id` → 设置 `target_space_id=当前 embedding_space_id`、生成新的 `migration_epoch`、删除 `migration_cursor` 和旧 lease keys，并**全局禁用所有 Vec 通道**（memory Vec Channel 5 + section Vec 匹配）。FTS/LIKE 正常使用。`memory_search` 的 warnings 中明确 `vec_disabled=embedding_space_mismatch`。提示用户执行 `memory_rebuild_embeddings`
- `failed`：迁移中某条可重建 memory 的 embedding/CAS、未知 status 或数据库操作真正失败。Vec 继续禁用，`last_error` 记录原因；`migration_cursor` 保持在失败项之前的连续成功/清理位置。deleted/missing/orphan 属于 cleanup，不进入 failed。当前 embedder 仍等于 `target_space_id` 时，下一次调用从该位置自动续跑
- 重建全部成功 → 原子更新 `active_space_id` 为当前 `embedding_space_id`，`state` 切回 `ready` → 所有 Vec 通道恢复
- 未来异步化时可追加 `rebuilding` 状态，当前版本不存（同步执行时该状态不会被读到）

**启动时状态归并**：
- 以下归并必须与 schema 初始化一样，在 `BEGIN IMMEDIATE` 写事务内重新读取当前 meta 后执行；不得用获取写锁前的缓存状态覆盖另一个进程刚建立的 target/epoch/lease
- 没有可用的托管 GGUF embedder → `unmanaged`；不对既有显式向量做错误猜测，分段增强保持关闭
- 当前 `embedding_space_id == active_space_id` → `ready`
- 当前空间 != active，且 `target_space_id == 当前空间`、原状态为 `mismatch/failed` → 保留原状态和 `migration_cursor`，允许分批续跑
- 当前空间 != active，且 `target_space_id != 当前空间`（迁移中再次换模型）→ 新目标覆盖旧目标，设 `target_space_id=当前空间`、生成新 `migration_epoch`、删除 `migration_cursor` 和旧 lease keys、`state=mismatch`，从头把表内所有现存向量重建到新空间；旧 runner 即使稍后返回，也会因 epoch/owner CAS 失败而无法提交

**Legacy 初始化**（首次升级到 v0.6.0）：
- `_vec_index_meta` 表不存在 → 先创建，再检查两张 vec 表
- 托管 GGUF embedder 可用且 vec 表均为空（全新安装）→ `active_space_id = 当前 embedding_space_id`，`state = ready`
- 托管 GGUF embedder 可用、任一 vec 表有数据但无 meta 记录 → 不写 `active_space_id`/`migration_cursor` key，写 `target_space_id=当前 embedding_space_id`、新 `migration_epoch`、`state=mismatch`，提示用户执行 `memory_rebuild_embeddings`；不猜测旧向量是否兼容
- 托管 embedder 不可用 → `state=unmanaged`，保留现有显式 memory Vec 行为，但 `memory_split` 不可用

**`state=ready` 的不变量**：两张 vec 表（`memories_vec` + `memory_sections_vec`）中所有现存向量都属于 `active_space_id` 对应的向量空间；并且每条 `split_status='active'`、memory status 为 active/superseded 的 section 都有且只有一条对应 section Vec。`mismatch`/`failed` 期间 Vec 全局禁用，期间产生的新向量（如 `memory_write` 自动 embedding）属于当前 embedder 的空间，但不会被查询到（Vec 禁用），且必须在最终重建完成判定中纳入目标集合。

**不再需要 per-memory fingerprint**：删除 `memories.metadata._split.embedding_fingerprint`、查询时逐 memory 比较 fingerprint、section Vec 渐进恢复逻辑。全局一个 `active_space_id` 管所有。

**托管模式下，所有向量入口都必须经过同一空间门禁**，否则全局不变量无法成立：
- server 自动生成的 memory/query/section embedding，内部自动绑定当前 `embedding_space_id`
- 显式 `memory_store_embedding` 增加可选参数 `embedding_space_id`；进入 v0.6 托管空间后，该参数缺失或不等于当前允许写入空间时拒绝写入，且不得先删除旧向量。`ready` 时允许写入空间=`active_space_id`；`mismatch/failed` 时允许写入空间=`target_space_id`
- 显式 `memory_search(query_embedding=...)` 增加 `query_embedding_space_id`；缺失或不等于 `active_space_id` 时跳过 memory Vec 和 section Vec，并返回 `vec_disabled=explicit_query_space_unknown_or_mismatch`，FTS/LIKE 仍正常
- v0.6.0 自动计算只支持现有 GGUF provider。远程 API/自定义 embedding 继续属于 `unmanaged` 的 memory 级高级用法，不启用 sections；后续若支持托管远程 provider，必须由 provider 配置提供不可变模型 revision 和稳定空间 ID，不能使用会漂移的模型别名

**与现有 `memories_vec` 的关系**：`memories_vec` 存 memory 级单向量（语义输入为 `subject+content`，实际按统一 token 预算安全截断），用于 `_wide_recall` Channel 5。`memory_sections_vec` 存段落级向量，**不参与召回**——仅用于 `memory_search` 返回时，对已命中的 memory 做 section 级语义匹配（"这条命中了，它里面哪段最相关？"）。两者独立，互不影响。

**为什么没有 section FTS，也没有段落级关键词匹配**：段落级匹配只走 section Vec（语义匹配，命中后定位，不参与召回），不增加 section FTS，也不引入段落级关键词匹配。向量是分段发布的硬前提；查询时 Vec 门禁若因迁移、空间不匹配或 query embedding 失败而关闭，则该条直接返回全文，不伪装成“零段落命中”。Vec 可以覆盖”二期规划/第二阶段”这类同义表达，memory 级 FTS 继续负责完整正文的字面召回。是否需要 section FTS 留给真实漏召回数据决定，本版本不预先增加第三套派生索引。

### 1.1c SQLite 连接与事务所有权

当前 `MemoryDB` 的长期共享 `self.conn` 必须重构为**连接工厂 + 每操作/每事务独立连接**。SQLite 的事务、提交和回滚都属于连接；若多个 MCP 调用共享同一连接，一个调用可能提交/回滚另一个调用，或触发 `cannot start a transaction within a transaction`。实现不得依赖 FastMCP 当前是否恰好串行调度，因为多客户端、多线程或未来异步化都会打破该隐含前提。

连接契约：

1. 普通无事务读写可使用一次性 operation connection；显式事务从 `BEGIN` 到 `COMMIT/ROLLBACK` 必须始终使用同一个、且只属于当前调用的 connection
2. 每个新 connection 都必须统一设置 `row_factory`、`PRAGMA foreign_keys=ON`、有限 `busy_timeout`，并加载 sqlite-vec；`journal_mode=WAL` 在初始化连接上确保一次，其他连接验证/复用该模式
3. schema migration 在服务接受工具调用前完成：初始化连接先确保 WAL，再用 `BEGIN IMMEDIATE` 获取跨进程写锁，并在锁内重新执行 `PRAGMA table_info` 后再决定是否 ALTER/CREATE；因此两个 MCP 进程同时首次启动也不会同时添加同一列。运行期连接不得重复跑 schema migration
4. wide recall、LLM、tokenizer 和 embedding 不得放进数据库事务；`memory_search.attach_sections` 只包最终物化短读快照，split/edit/rebuild 只包 CAS 与原子发布短写事务
5. connection 必须在 `finally` 中关闭；遇 `SQLITE_BUSY` 只做有上限、带抖动的重试，耗尽后返回稳定错误码和当前阶段

若实现方选择全局串行化而非独立连接，必须将“单进程、全部 DB 工具严格串行”列为产品限制，并放弃并发读写目标；v0.6 的默认实现不采用该方案。

### 1.2 `memories` 表新增列

```sql
ALTER TABLE memories ADD COLUMN split_status TEXT;  -- 迁移时默认 NULL
ALTER TABLE memories ADD COLUMN split_revision INTEGER NOT NULL DEFAULT 0;
```

`split_revision` 是**分段派生状态版本**，与记录正文的 `memories.version` 分工不同：正文、分段发布、分段清理或 decline 中任一会使在途 split/rebuild 结果失效的状态变化，都必须在同一事务内令 `split_revision = split_revision + 1`。prepare 返回该值，publish/decline 必须原样带回并做 CAS。不能只检查 `split_status`：两个并发 rebuild 的状态始终都是 `active`，仍会互相覆盖。

**`split_status` 取值：**

| 值 | 含义 |
|---|---|
| `NULL` | 从未尝试过分段（未超阈值 / 普通短记忆 / 向量不可用 / 环境未开分段） |
| `active` | 分段生效，`memory_search` 返回 `matched_sections` |
| `failed` | 尝试过分段但失败，按普通长记忆处理。失败阶段记录在 `metadata._split.last_split_error.stage`（`validation` / `embedding` / `publish`） |
| `declined` | 用户拒绝分段（记录到 metadata 防重复询问，不影响原文检索） |

**可归因于当前 prepare 的业务失败标 `failed`**：offset/batch 校验失败、LLM 输出非法、section embedding 生成失败，只有在 content/version/split revision 仍匹配该 prepare 时，才以 CAS 事务标 `split_status='failed'` 并记录 `last_split_error`。并发 edit/split/rebuild、Vec space 变化、`SQLITE_BUSY` 等发布冲突不是分段内容失败，保持当前状态和旧 sections 不变，返回可重试错误，不能用 `failed` 覆盖并发成功结果。

新增列的幂等迁移仿照现有 `_migrate_add_version_column`：`PRAGMA table_info(memories)` 探测，缺失才 `ALTER`。旧记录的 `split_revision` 初始化为 0。

### 1.3 拒绝标记

不持久化 `declined_content_hash`：`split_status='declined'` 已表达当前内容版本被拒绝；content edit 会在同一事务把状态重置为 NULL 并递增 `split_revision`。decline 请求携带的 `decision_content_hash` 只作为一次性 stale-decision CAS，hash 不匹配就拒绝，不把针对旧正文的决定落到新正文。

⚠️ `metadata._split` 和 `metadata._embedding` 的更新必须在事务内 merge，不能整体覆盖用户 metadata（read-modify-write 时先读完整 metadata → 更新内部子键 → 写回）。

## 2. API（MCP 工具）

### 2.1 `memory_write`（不扩展，回归纯写入）

`memory_write` **完全不改 API**，保持 v0.5.4 的签名和行为。长文本照常入库，`split_status=NULL`。

唯一变化：写入成功后，若 `split_enabled=True`、`len(content) > split_threshold`、托管向量可用且 `_vec_index_meta.state=ready`，在返回的 `data` 里**追加一个提示字段**（不影响现有字段）：

```python
# 正常写入返回（v0.5.4 现有结构，不变）
data = {"id": memory_id, "backup_only": False, "record": {...}}
# 超阈值 + 向量可用时，data 追加 split_hint
data["split_hint"] = {
    "char_count": len(content),
    "split_threshold": settings.split_threshold,
    "prompt": "已保存。该记忆 {char_count} 字符，分段可提升检索精度（LLM 调用次数取决于安全批次数）。如需分段，调用 memory_split(memory_id={id})。",
    "memory_id": memory_id
}
```

`split_hint` 只是建议，Agent 可忽略。原文已经入库了，丢不了。向量不可用时不返回 `split_hint`（因为分段增强无法激活）。

**分段的所有逻辑统一归 `memory_split`（见 2.3）**，`memory_write` 不掺和。

### 2.2 `memory_split_status(memory_id)`

只读工具。返回某条记忆的 `split_status`、`split_revision`、section 目录（`section_id` + `title` + `title_path` + `summary`）、content hash，以及全局 `vec_index_state`、`active_space_id`、`target_space_id`、`migration_cursor`、`migration_epoch`、`migration_in_progress`、`section_vec_available`。lease owner 仅用于内部 CAS，不向普通状态响应暴露；供 Agent 区分“没有拆”“已拆但 Vec 正在维护”“显式向量处于 unmanaged”三类情况。

现有 `memory_status` 也追加同一组全局向量状态和最近一次迁移错误，避免用户必须挑一条 memory 才能诊断整个索引。

### 2.3 `memory_split(...)`

prepare（可多批读取）+ atomic publish 工具，分段的**唯一入口**。对**已存在**的长记忆补齐分段信息。完整流程见 3.3。

**参数：**
```python
memory_split(
  memory_id: int,                                  # 必填，目标已存在记忆
  split_decision: Optional[str] = None,            # None=首次 / "split"=二次提交 / "decline"=拒绝 / "rebuild"=重建
  decision_content_hash: Optional[str] = None,     # 除第一次 prepare 外必填
  decision_memory_version: Optional[int] = None,   # 除第一次 prepare 外必填；检测 prepare 后的并发 edit
  decision_split_status: Optional[str] = None,     # 除第一次 prepare 外必填；必须等于 prepare 返回状态
  decision_split_revision: Optional[int] = None,   # 除第一次 prepare 外必填；防并发 split/rebuild 覆盖
  sections: Optional[list[dict]] = None,           # "split"/"rebuild"时提供，结构见下
  prepare_batch_index: int = 0,                    # prepare/rebuild prepare 要取的批次，0-based
  llm_batch_chars: int = 12000,                    # 外部 LLM 单批字符预算；调用方可按模型上下文调小
)
```

**五种调用模式：**
- **首次 prepare**（`split_decision=None, prepare_batch_index=0`）：确定性生成 batch manifest，返回第 0 批正文、`content_hash`、`split_revision` 和 schema，不落库。前置条件：`split_status ∈ {NULL, failed, declined}`。
- **继续取 prepare 批次**（`split_decision=None, prepare_batch_index=N>0`）：调用方带回第一次返回的 `decision_content_hash + decision_memory_version + decision_split_status + decision_split_revision + llm_batch_chars`；server 重新计算同一 manifest，只返回第 N 批。若任一快照已变化，拒绝并要求重新 prepare。
- **二次提交**（`split_decision="split"` + `sections` + 四个 decision 快照字段）：仅在 `vec_index_state=ready` 且完整 CAS 通过时发布，切 `split_status='active'` 并递增 `split_revision`。
- **拒绝分段**（`split_decision="decline"` + 四个 decision 快照字段）：hash/version/status/revision 任一不匹配即拒绝；成功后标记 `split_status='declined'` 并递增 `split_revision`。
- **重建 prepare/publish**（`split_decision="rebuild"`）：`sections=None` 时按上述批次协议读取 active 记忆；`sections` 非空时做原子替换。prepare 和 publish 都要求预期状态为 `active`；成功 publish 后保持 active 并递增 `split_revision`，失败时旧 sections 保持不动。

**关于 rebuild 的长期维护能力**：量化模型变更是必然的运维场景。`memory_split(rebuild)` 用于**单条记忆的分段信息重建**（需要 LLM 重新生成 sections）。模型切换后批量重建向量层（不需要 LLM，只重算 embedding）使用 `memory_rebuild_embeddings` 工具（见 2.4b）。

#### 外部 LLM 安全分批协议

不能假设外部 LLM 能一次接收完整正文。真实样例 52238 字符，再加 system prompt、工具 schema、对话历史和 JSON 输出预算，可能超过 16K/32K 上下文；截断后只生成前半篇 sections，最终只能得到难诊断的“覆盖校验失败”。因此 prepare **不得无条件返回完整 content**，而是按 `llm_batch_chars` 返回可重放的确定性批次。

prepare 响应示例：

```python
{
  "content_hash": str,
  "memory_version": int,
  "split_status": str | None,
  "split_revision": int,
  "parser_detected": bool,
  "llm_batch_chars": int,
  "batch_count": int,
  "batches": [
    {"batch_index": 0, "batch_id": str, "char_count": int,
     "structure_hint": "structured" | "normal" | "low_structure"},
    ...
  ],
  "current_batch": {
    "batch_index": int,
    "batch_id": str,
    "structure_hint": "structured" | "normal" | "low_structure", # 按当前 batch 判定，不是文档级字段
    "content": str,              # 只含当前批次正文，不暴露全局 offset
    "source_sections": [...]     # parser 路径有；纯文本路径为空
  },
  "estimated_llm_calls": batch_count,
  "split_schema": {...},
}
```

批次规则：

1. 批次覆盖完整正文、严格有序、互不重叠；`batch_id = sha256(canonical_json(content_hash, memory_version, split_status, split_revision, parser_mode, llm_batch_chars, batch_index, batch_start, batch_end))`。server 不新增临时 job 表，后续请求根据相同参数确定性重算。
2. **Parser 路径**：先按确定性 Markdown 边界得到最终 source sections，再按完整 section 分组为批次；单个 source section 超过预算时，在最近的空行/段落/句末标点边界确定性细分为最终 sections，仍记 `provenance='parser'`。LLM 只为 `source_section_index` 对应内容生成 title/summary。
3. **纯文本路径**：先在空行/段落/句末标点边界附近确定性切成不超过预算的非重叠批次；LLM 只在批次内部生成 sections。每批第一段强制从局部 0 开始，后续段落用**批次内** `anchor_text + occurrence_index`；arbiter 用私有的 batch 起点换算全局 offset。非重叠 batch 的首尾因此是最终 section 的强制技术边界，不能宣称 batch 对边界完全无影响。
4. publish 时 server 重新生成 manifest，校验所有预期 `batch_id` 恰好出现一次、没有未知或缺失批次；缺失时返回 `missing_batch_ids`，不得用模糊的 offset 错误代替。
5. 批次只解决外部 LLM 上下文，不改变最终原子发布：所有批次结果收齐、全局 offset/覆盖校验和全部 section embedding 成功后，才一次性替换旧 sections。
6. `llm_batch_chars` 是保守字符预算，不宣称等于 token 数；调用方知道模型上下文时可以调小。普通文档可能只有 1 批，超长文档为 N 批，产品提示必须展示 `estimated_llm_calls`，不得再承诺固定“1 次 LLM 调用”。
7. prepare 必须在调用 LLM 前验证 `llm_batch_chars > 0`，并计算理论最少最终 section 数：parser 路径按确定性最终 source sections 数；纯文本路径中 `normal` batch 至少 1 段、`low_structure` batch 因 prompt 要求至少 2 段，最后再取全局最小值 2，即 `max(2, normal_batch_count + 2 * low_structure_batch_count)`。若该值超过 `max_sections`，立即返回 `too_many_batches_for_max_sections`，提示调用方在模型允许范围内提高批次预算，不能等所有批次完成后才失败。

`structure_hint` 必须确定性计算：parser batch 固定为 `structured`；纯文本 batch 将 CRLF/CR 仅为检测目的视为 LF，若 batch 内不存在任何换行字符则为 `low_structure`，否则为 `normal`。检测不得改写返回给 LLM 的原始 content，也不得使用模型或启发式“看起来像段落”的判断。

**`sections` 元素结构（Agent/LLM 输出，不含全局 offset）：**
```python
{
  "batch_id": str,                 # prepare 返回，publish 原样带回
  "batch_section_index": int,      # 当前批次内从 0 连续
  "source_section_index": Optional[int], # parser 路径必填；纯文本为空
  "title": str,                    # LLM 生成（纯文本）或解析器提取（结构文档标题）
  "summary": str,                  # LLM 生成，目录展示用
  "anchor_text": Optional[str],    # 纯文本批次中除第一段外必填；parser 路径不需要
  "occurrence_index": int,         # anchor 在当前批次第几次出现（0-based），不是全文序号
  "parent_title": Optional[str],   # 层级，展示用
  "title_path": Optional[str],     # 层级路径，展示用
}
```

**arbiter 只接受 batch 身份、parser source index 或局部 anchor，绝不接受 LLM 给的 `start_offset/end_offset`。** 单批文档的 `current_batch.content` 等于全文；多批文档任何一次响应都只返回当前批次。

### 2.4 `get_sections(memory_id, section_ids)`

返回指定 section 的完整原文片段（`content[start_offset:end_offset]`）+ 元数据。

- `section_ids` 为空列表 → 返回空列表
- `section_ids` 中的某个 ID 不存在 → 该 ID 放入 `missing_section_ids` 返回给调用方（不静默忽略）
- 返回结构（顶层）：
  ```python
  {
    "memory_id": int,
    "sections": [
      {"section_id": int, "title": str, "title_path": str, "summary": str,
       "start_offset": int, "end_offset": int,
       "provenance": str, "embedding_truncated": bool,
       "embedding_original_tokens": int, "embedding_used_tokens": int, "created_at": str,
       "content": str},  # content[start_offset:end_offset] 原文片段
      ...
    ],
    "found_count": int,              # 实际找到的 section 数（顶层）
    "missing_section_ids": [int],    # 未找到的 section_id 列表
  }
  ```

### 2.4b `memory_rebuild_embeddings(memory_ids?, dry_run?, batch_size?)`

批量重建向量工具。用于 embedding 模型切换后，用当前 embedder 全量重算 memory 级向量（`memories_vec`）+ section 级向量（`memory_sections_vec`）。

**不需要 LLM 调用**——只重算向量，不重新分段。section 的 offset/title 不变，只是用新模型重新 embedding。

**参数：**
```python
memory_rebuild_embeddings(
  memory_ids: Optional[list[int]] = None,  # 迁移模式忽略；ready 状态下必须指定，表示局部修复
  dry_run: bool = True,                    # True=只返回清单（默认）/ False=执行重建
  batch_size: Optional[int] = 50,          # 默认最多 50 条，防止同步 MCP 超时；显式 None 才表示不限
)
```

**执行逻辑：**

1. 确定向量可用（`sqlite_vec_available=True` 且 embedding 配置可用），否则返回错误
2. **确定运行模式与目标集合**：
   - `_vec_index_meta.state in {mismatch, failed}` 时进入**迁移模式**：要求当前 `embedding_space_id == target_space_id`，忽略 `memory_ids`；目标为两张 vec 表中存在任一向量的 memory ID 并集，按 memory ID 升序，只处理 `memory_id > migration_cursor` 的连续后缀（cursor key 不存在时视为负无穷，从第一条开始）
   - `_vec_index_meta.state == ready` 时进入**局部修复模式**：目标=`memory_ids` 指定的子集，`memory_ids=None` 返回错误"当前已 ready，请指定 memory_ids 做修复"；局部修复不使用 `migration_cursor`
   - 目标并集必须覆盖：① `memories_vec` 中有 memory 级向量的 ID；② 通过 `memory_sections JOIN memory_sections_vec` 能映射出的 memory ID；③ status 为 active/superseded、`split_status='active'` 且存在 section 缺对应 Vec 的 memory ID。第三类是完整性修复目标，避免“Vec 行不存在”因不在任一 Vec 表而永远不被扫描
   - 生成目标前先识别 `memory_sections_vec.id` 无对应 `memory_sections.id` 的**无归属 section vec 孤儿**；该类记录没有 memory_id，不能进入 cursor 序列，必须作为独立 preflight cleanup 清理并计数。`dry_run` 只报告，不删除；实际删除使用短写事务并在事务内重新验证仍无对应 section，避免误删并发刚完成的合法写入
3. **按主数据分类，而不是把非 active 一律过滤掉**：
   - memory 存在且 `status in {'active', 'superseded'}` → `rebuild`。`superseded` 仍需保留 Vec，保证 `include_superseded=True` 在模型切换前后能力一致
   - memory 不存在，或 `status='deleted'` → `cleanup`。清理其 memory vec、可映射的 section vec 和 section rows；这是可预期的索引垃圾回收，不是迁移失败
   - 未知 status → `error`，停止迁移并暴露原值，禁止擅自删除
4. `dry_run=True` → 返回分类清单（`rebuild_targets` / `cleanup_deleted` / `cleanup_orphans` + 当前 `embedding_space_id` vs `active_space_id`），不申请 lease、不执行；`batch_size` 非 NULL 时必须 `> 0`
5. `dry_run=False` 的迁移模式必须先用一个短 `BEGIN IMMEDIATE` **申请数据库 lease**：
   - 调用生成随机 `run_id`，在事务内读取并固定 `migration_epoch`、`target_space_id`、`state` 和 `migration_cursor`
   - 无 lease、lease 已过期，或 owner 就是本调用时，写入/续期 `migration_lease_owner=run_id`、`migration_lease_expires_at=now+lease_ttl`；存在其他未过期 owner 时立即返回 `migration_in_progress`，不得并行计算
   - 申请 lease 的同一事务中检查“active/superseded active-split section 缺 Vec”的最小 memory ID。若该 ID `<= migration_cursor`，说明已完成前缀后来出现完整性破坏：把 cursor 安全回退到 `missing_memory_id - 1`（若不存在更小合法位置则删除 cursor key），允许后续重算该后缀；不得保持 cursor 不变而让 final-ready 永久卡住
   - `lease_ttl` 必须覆盖单条最大预期 embedding 时间（建议 10 分钟），并在每条计算开始前和提交成功后续期；调用正常结束时用 owner CAS 释放，进程崩溃则等待过期回收
   - lease 只避免重复工作；真正正确性仍由下述 `epoch + owner + expected_cursor` CAS 保证。任何旧 runner 在 lease 过期后返回，都不得提交
   - ready 状态下的局部修复不使用全局 migration lease，也不读写 cursor；同一 memory 的重复修复由主数据/section snapshot CAS 保底
6. 获得 lease 后，**先清理无归属 section vec 孤儿，再按 memory ID 升序处理；只有真正失败才停止**：
   - **`rebuild` 项事务外**：读 memory 的 status/subject/content/version/split_status/split_revision，通过统一 token-safe helper 计算 memory embedding + truncated/original/used token metadata；对 `split_status='active'` 的，读取 sections 并计算 `section_snapshot_hash = sha256(canonical_json([{id,title_path,start_offset,end_offset,embedding_truncated,embedding_original_tokens,embedding_used_tokens}, ...]))`，再按快照和同一 helper 计算 section embedding + 每段新的 token metadata
   - **`rebuild` 项单事务（BEGIN IMMEDIATE）**：
     - 事务外计算前记录 `expected_cursor`（包括“cursor key 不存在”这一状态）以及本调用固定的 `expected_epoch/expected_target_space_id/run_id`
     - ⚠️ **双层 CAS 校验**：事务内先校验 migration `state in {mismatch, failed}`、epoch、target space、lease owner 均仍等于期望值，且当前 cursor 与 `expected_cursor` 完全相同；再校验 memory 的 status/version/content_hash/split_status/split_revision 和 `section_snapshot_hash`。任一不匹配 → 丢弃计算结果、不得 DELETE/INSERT/推进 cursor，返回 `migration_lease_lost`、`migration_epoch_changed` 或 `migration_cursor_conflict`
     - DELETE 该 memory 的旧 memory_vec + 旧 section_vec
     - INSERT 新 memory_vec + 新 section_vec，更新各 section 的 truncated/original/used token metadata，并 merge 更新 memory 级同名 metadata
     - 迁移模式下，在**同一个事务**把 `_vec_index_meta.migration_cursor` 单调推进到该 memory ID 并续期 lease；事务回滚时 cursor 和 lease 续期也必须回滚。逐条事务不得修改 `active_space_id` 或切 `ready`
   - **`cleanup` 项单事务（BEGIN IMMEDIATE）**：先执行同样的 epoch/target/owner/expected-cursor CAS，再确认 memory 仍不存在或仍为 `deleted`；随后 DELETE 该 memory 的 section vec → section rows → memory vec，并在**同一事务**推进 `migration_cursor`、续期 lease。若分类已变化（例如并发恢复为 active），回滚后按新分类重试，不能删除已恢复数据
   - cleanup 成功视为连续前缀成功，计入 `cleaned` 而非 `failed`；一条孤儿/已删除记录不得永久卡住后续迁移
   - 迁移真正失败 → 只有仍持有同一 epoch/owner 时才可 CAS `state=failed`、记录 `last_error`；保留最后一个连续成功/清理的 `migration_cursor`。lease/epoch/cursor 冲突属于陈旧 runner 退出，不得把新迁移标 failed
   - 局部修复模式中，active/superseded 按 rebuild 处理，deleted/missing 按 cleanup 处理；单项错误只返回给调用方，**全局 state 保持 ready**
7. 迁移模式下，处理完本批后开启一个最终 `BEGIN IMMEDIATE` 短事务，先 CAS 当前 epoch/target/lease owner/cursor 仍等于本调用最后成功值，再在写锁内确认：当前 `embedding_space_id == target_space_id`，不存在无归属 section vec 孤儿，不存在 active/superseded active-split section 缺 Vec，并且不存在 `memory_id > migration_cursor` 的剩余 rebuild/cleanup/完整性修复目标。确认完成后才原子执行：`active_space_id=target_space_id`、删除 `target_space_id`/`migration_cursor`/`migration_epoch`/lease/`last_error` keys、`state=ready`。如果仍有目标则保持 `mismatch/failed`，释放本调用 lease，等待下一批。⚠️ 逐条事务绝不能切 ready——第一条成功就切 ready 会导致新旧向量空间混合查询
8. 返回：
```python
{
  "processed": N,           # 本次处理的 memory 数
  "succeeded": M,           # rebuild 成功数
  "cleaned": C,             # deleted/missing memory 的索引清理成功数
  "orphan_section_vecs_cleaned": O, # 无法映射 memory_id 的 section vec 清理数
  "failed": F,              # 失败数
  "errors": [...],          # 失败详情（memory_id + 原因）
  "migration_cursor": int | None, # 服务端已持久化的连续成功位置；调用方不回传
  "migration_epoch": str | None,  # 诊断字段；调用方不得回传或指定
  "lease_acquired": bool,
  "has_more": bool,         # 是否仍有待处理目标
  "migration_complete": bool,
  "total_target": T,        # 目标 memory 数（rebuild + cleanup，不含无 memory_id 的 section vec 孤儿）
  "global_state": "unmanaged" | "ready" | "mismatch" | "failed",  # 最终全局状态
}
```

**续跑语义**：cursor 是服务端持久状态，不接受调用方任意指定，避免错误地传入过大的 cursor 跳过旧空间向量。rebuild 成功和 deleted/missing cleanup 成功都会推进连续前缀；只有 embedding、主数据 CAS、未知 status 或数据库错误才停止。用户修复问题后再次调用 `memory_rebuild_embeddings(dry_run=False, batch_size=...)` 即可自动续跑并申请新 lease。若响应丢失或 server 重启，`migration_cursor` 和 `migration_epoch` 仍在 `_vec_index_meta`，不会丢进度；旧 lease 过期后可安全接管。只有所有现存 rebuild/cleanup/完整性修复目标完成、无无归属 section vec 孤儿且没有 active section 缺 Vec，最终事务才切 `ready`。

**与 `memory_split(rebuild)` 的关系**：
- `memory_split(rebuild)`：重建**单条**记忆的**分段信息**（需要 LLM 重新生成 sections + 重新算 offset + 重新算向量）。用于分段信息本身需要更新的场景。
- `memory_rebuild_embeddings`：批量重建**多条**记忆的**向量**（不需要 LLM，不重新分段，只重算 embedding）。用于模型切换后的向量层迁移。

**典型使用流程**（模型切换）：
```
1. 用户切换 embedding 模型（改 config.json 的 model_path）
2. 重启 server → _vec_index_meta.state = mismatch，所有 Vec 通道全局禁用
3. memory_rebuild_embeddings()              → 默认 dry_run=True，看影响范围
4. memory_rebuild_embeddings(dry_run=False, batch_size=50) → 分批执行；重复调用会自动续跑
5. 重建全部成功 → active_space_id 原子更新，state 切 ready → 所有 Vec 通道恢复
6. memory_search 正常工作，所有向量已迁移到新模型空间
```

**性能考量**：全量重建可能涉及上千条记忆，每条算 1 个 memory 向量 + N 个 section 向量。按实测 ~0.3s/embedding，1000 条记忆（平均每条 10 段）≈ 1000×11×0.3 ≈ 55 分钟。建议实现时支持**后台异步执行 + 进度查询**，不阻塞 MCP 调用。当前版本可以先同步实现，后续优化为异步。

**迁移期间新写入的已知冗余计算**：mismatch/failed 期间新 memory 由写入门禁保证使用 `target_space_id`，但若其自增 ID 落在 cursor 尚未处理的后缀，当前动态目标扫描仍可能再算一次相同空间向量。该行为浪费少量计算但不影响正确性，v0.6 明确接受，不为此引入 per-memory fingerprint；若实测迁移写入量很大，后续可在迁移开始时记录 `migration_upper_bound`，只迁移当时已有 ID。

### 2.4c `memory_store_embedding`（显式向量写入守门）

现有工具签名追加可选参数：

```python
memory_store_embedding(
  memory_id: int,
  embedding: list[float],
  embedding_space_id: Optional[str] = None,
)
```

进入 v0.6 托管空间后，显式写入必须提供空间 ID。`ready` 时只接受 `active_space_id`；`mismatch/failed` 时只接受 `target_space_id`。缺失或不匹配直接拒绝，旧向量保持不动。这样可以防止手工 backfill 把未知空间的向量混入已经声明为 `ready` 的全局索引。`unmanaged` 状态保留 v0.5.4 行为，但该状态不能启用分段增强，兼容性由显式向量调用方负责。

### 2.5 `memory_search`（返回结构扩展，召回路径不变）

**召回路径**：`_wide_recall` 的现有 5 个通道（FTS main / FTS OR / subject-tags LIKE / content LIKE / memory Vec KNN）保留，不加任何新通道。前 4 个字面通道行为不变；托管模式下 memory Vec Channel 5 仅在 `_vec_index_meta.state == ready` 且 query 向量空间等于 `active_space_id` 时执行。`unmanaged` 仅保留旧版显式 query 路径并返回 warning；其他状态跳过 Vec。返回值仍是 memory 级去重的候选 `pool`。

**显式 query embedding 的空间校验**：`memory_search` 签名追加 `query_embedding_space_id: Optional[str] = None`。托管模式下，server 自动生成 query embedding 时内部自动填当前空间 ID；调用方显式传 `query_embedding` 时必须同时传空间 ID。缺失或 mismatch 不报整个搜索失败，只禁用 memory/section 两个 Vec 路径，FTS/LIKE 继续工作。`unmanaged` 状态保留 v0.5.4 的显式 query 行为并返回 `vec_space_unmanaged` warning，提醒兼容性由调用方负责。

**返回前增强 + token 节省**（在 `_soft_rerank` 打完分、切到 limit 后，对每条结果追加分段信息）：

⚠️ **核心机制：active 分段命中时省略 content 字段**。当前 rerank result 自带完整 `content`（可能几万字）。分段的目的就是省 token——命中 active memory 时，`content` 字段设为 `null`，用 `content_omitted: true` 标记，Agent 需要全文时调 `memory_get(memory_id)`。`matched_sections` 只含结构化元数据（title/summary），不含原文——需要段落原文调 `get_sections`。

⚠️ **最终物化必须使用同一个 SQLite read snapshot**。rerank result 中的 `split_status/content/version` 只是候选阶段缓存，不能直接与稍后单独查询的 sections 拼接。切到 limit 后，`attach_sections` 对本批结果开启一个短 `BEGIN` 读事务，在同一快照内重新读取：当前 memory 行、可见性状态、`_vec_index_meta`、所有 sections 及 section Vec 命中；返回字段以该快照为准。这样即使两次读之间有 `memory_edit` 提交，也不会出现“旧 `split_status=active` + 新 sections=[]”或“旧正文 + 新状态”。WAL 下该短读事务不阻塞写者；不得把耗时的 wide recall/embedding 包进这个事务。

```
result_ids = rerank_and_limit(...)

BEGIN;  # 最终物化短读事务；以下读取共享一个 snapshot
current_memories = SELECT current rows WHERE id IN result_ids
current_vec_state = SELECT _vec_index_meta ...
current_sections = SELECT sections WHERE memory_id IN result_ids
current_section_vec_ids = SELECT ids FROM memory_sections_vec JOIN current_sections ...

对每条 current memory：
  先按当前 status/include_superseded 重新应用可见性规则；不再可见则从响应移除
  用当前行覆盖 rerank 缓存中的 content/version/split_status

  if current_memory.split_status == "active":
      sections = current_sections[current_memory.id]
      total_sections = len(sections)

      # 最后一道不变量防护：active 理论上应至少有 1 段，但旧库/人工改库/bug 都可能破坏它
      if total_sections == 0:
          → 返回当前快照中的完整原文
          → content_omitted = false, section_enhancement_applied = false
          → warnings += "split_invariant_broken_empty_sections"
          → continue  # 严禁进入 matched_count / total_sections

      if total_sections == 1:
          → 返回当前快照中的完整原文
          → content_omitted = false, section_enhancement_applied = false
          → warnings += "split_invariant_broken_too_few_sections"
          → continue

      # Vec 门禁关闭不是“真实零命中”；分段增强不可用时必须恢复原搜索返回能力
      if not snapshot_vec_gate_open:
          → 返回当前快照中的完整原文
          → content_omitted = false, section_enhancement_applied = false
          → warnings += snapshot_vec_disabled_reason
          → continue

      # active 发布要求每个 section 都有 Vec；缺失不能伪装成“不相关”
      if ids(sections) - current_section_vec_ids[current_memory.id] is not empty:
          → 返回当前快照中的完整原文
          → content_omitted = false, section_enhancement_applied = false
          → warnings += "split_invariant_broken_missing_section_vec"
          → continue

      # 只有门禁打开后才执行 section Vec 语义匹配
      # 查询也在本 read snapshot 内执行；任何孤儿 vec hit 还要与本快照 section_id 集合求交
      vec_hits = section_vec_match(query_embedding, memory_id)
      vec_hits = vec_hits ∩ ids(sections)

      matched = vec_hits
      matched_count = len(matched)

      if matched_count == 0:
          → content = null, content_omitted = true
          → section_enhancement_applied = true
          → 返回 section_catalog（全部段落目录）+ hint
      elif matched_count / total_sections >= section_fulltext_threshold:
          → content = 完整原文（大部分段落都命中，直接返回全文）
          → content_omitted = false, section_enhancement_applied = true
          → 附带 matched_sections（供参考）+ hint "{pct}% 段落命中，建议直接看全文"
      else:
          → content = null, content_omitted = true, section_enhancement_applied = true
          → 返回所有 matched_sections（title/summary，不含原文）
          → 返回 section_catalog（未命中的段落目录）
          → Agent 需要段落原文 → 调 get_sections；需要全文 → 调 memory_get
  else:
      返回当前快照中的全文（结构不变，content_omitted 不出现）

COMMIT;
```

该短事务只保证**最终响应自洽**；若 memory 在 rerank 后被编辑，排序分数可能仍来自编辑前版本，但响应正文、状态和 sections 不会互相矛盾。下一次搜索会用新内容重新排名。若未来要求“排序和响应也必须是同一时点”，才考虑把完整 search 放进读事务；v0.6 不用长快照换取这一极端并发语义。

**返回字段变化说明**：

```python
# 未分段记忆（split_status 为 NULL/failed/declined）— 与 v0.5.4 完全一致
{
  "id": 50, "content": "完整原文...", "subject": "...", ...
}

# active 分段命中，部分段落匹配（省 token 的核心场景）
{
  "id": 50, "content": null, "content_omitted": true,
  "subject": "...", ...
  "split_status": "active",
  "matched_sections": [
    {"section_id": 101, "title": "二期建设"},
    ...
  ],
  "section_catalog": [
    {"section_id": 102, "title": "接口设计", "title_path": "技术方案 / 接口设计",
     "summary": "接口边界、请求字段与错误码",
     "embedding_truncated": false, "embedding_original_tokens": 812,
     "embedding_used_tokens": 812}, ...
  ],
  "hint": "已返回命中段落元数据，用 get_sections 获取段落原文，memory_get 获取全文"
}

# active 分段命中，≥80% 段落匹配（返回全文）
{
  "id": 50, "content": "完整原文...", "content_omitted": false,
  "split_status": "active",
  "matched_sections": [...],  # 附带，供参考
  "hint": "83% 段落命中，建议直接看全文"
}

# active 分段命中，零段落匹配
{
  "id": 50, "content": null, "content_omitted": true,
  "split_status": "active",
  "section_catalog": [...],  # 全部段落目录
  "hint": "已拆分为 12 段，可用 get_sections 获取"
}

# active 分段存在，但本次 query 的 Vec 门禁关闭（迁移/空间不匹配/query embedding 失败或为空）
{
  "id": 50, "content": "完整原文...", "content_omitted": false,
  "split_status": "active", "section_enhancement_applied": false,
  "warnings": ["vec_disabled=embedding_space_mismatch"]
}
```

**返回逻辑的设计原则**：分段是增强，不能以牺牲准确度为代价。只有 query-time Vec 门禁已经打开并真正执行了 section 匹配，才存在“零命中”语义；门禁关闭时直接返回全文。门禁打开后，命中的 section 数量不被人为截断——匹配了多少就返回多少。如果大部分段落都命中（≥ `section_fulltext_threshold`），说明 query 和整条记忆都相关，直接返回全文（片段定位此时没有价值）。如果只有少数段落命中，省略 content 返回命中段落元数据，Agent 按需取原文。如果真实零命中，返回目录供 Agent 自行决定。

`section_catalog` 无论是零命中全目录还是部分命中的未命中目录，都统一返回 `section_id/title/title_path/summary/embedding_truncated/embedding_original_tokens/embedding_used_tokens`。这些字段已存库，增加它们不产生额外 LLM 成本，可让 Agent 在不追加 round trip 的情况下先判断是否值得取原文，并诊断向量只覆盖了多少原文。

**section Vec 语义匹配**（在返回时执行，不参与召回）：

⚠️ **不能用 vec0 的 `MATCH ... k=N` 全局 KNN 语法**。vec0 引擎先做全库 top-K 再用 `WHERE memory_id=?` 过滤——如果目标 memory 的 section 向量在全局排不进 top-K，就会被漏掉（sqlite-vec 0.1.9 已确认此行为）。

⚠️ **不能从 vec0 表 SELECT embedding 列后 `json.loads`**。sqlite-vec 0.1.9 的 embedding 列返回 raw BLOB bytes，`json.loads` 会 `UnicodeDecodeError`。

**正确做法是 SQL 内 `vec_distance_cosine` 函数**：一个 memory 的 section 数量很少（10-20 个），直接在 SQL 内算距离，不需要取出向量：

```sql
SELECT
  s.id AS section_id,
  s.title,
  s.title_path,
  s.start_offset,
  s.end_offset,
  vec_distance_cosine(v.embedding, :query_embedding) AS distance
FROM memory_sections s
JOIN memory_sections_vec v ON v.id = s.id
WHERE s.memory_id = :memory_id
  AND vec_distance_cosine(v.embedding, :query_embedding) <= :threshold
ORDER BY distance;
-- :query_embedding 使用 server 当前 embedder 生成的向量，或携带匹配 query_embedding_space_id 的显式向量
-- :threshold = section_vec_distance_threshold（relevance gate）
```

`vec_distance_cosine` 是 sqlite-vec 提供的 SQL 标量函数，直接在数据库层计算，避免了 BLOB → Python 的序列化开销。应用层通过参数绑定传入已通过空间校验的 query embedding（sqlite-vec 接受 JSON 数组格式的字符串）；未通过门禁时不得执行这段 SQL。

**relevance gate（`section_vec_distance_threshold`）**：`vec_distance_cosine` 只有排序没有"命中/未命中"语义——distance 很大的 section 也会被返回。如果不设阈值，所有 section 都算"命中" → matched_count 趋近 total → 80% 阈值频繁回退全文 → 分段失效。必须设 distance 上限：distance 超过此值的 section 不算命中。该阈值需用真实数据校准（建议：取"明显相关"的 section 的 distance P90 作为初始值）。

`matched_sections` 每条结构：

```python
{
  "section_id": int, "title": str, "title_path": str,
  "summary": str,          # LLM 在分段时生成的简短摘要（几十字），让 Agent 不用二次调用就能大致了解段落内容
  "embedding_truncated": bool,  # 该 section 的 embedding 是否被截断（true=语义覆盖不完整）
  "embedding_original_tokens": int,
  "embedding_used_tokens": int, # 二者可推导实际覆盖比例，便于解释“为什么这段没命中”
}
# 不截断，命中几段返回几段
# Agent 根据 title + summary 判断相关性
# summary 是 LLM 在分段时已生成并存入 memory_sections.summary 列的，返回时零成本带上
# 需要段落原文 → 调 get_sections(memory_id, [section_id])
# 需要全文 → 调 memory_get(memory_id)
```

**未分段记忆**（`split_status` 为 NULL/`failed`/`declined`）：返回结构不变，仍是全文。

### 2.6 `memory_get(memory_id)`

**不变**。永远返回完整原文（原则 4）。

## 3. 写入流程

### 3.1 `memory_write`（原样写入 + 建议提示）

```
content 到达
  ↓
普通写入（走现有 insert_memory，split_status=NULL）
  ↓
if split_enabled and len(content) > split_threshold and 托管向量可用 and vec_index_state == "ready":
    返回里追加 split_hint（建议调用 memory_split）
else:
    返回不变
```

`memory_write` 只管写原文，不触发任何分段逻辑。现有 auto memory embedding 改为调用 1.1b 的统一 token-safe helper：长文被截断时正文仍完整入库，在 `metadata._embedding` 记录 truncated/original/used token metadata，并在 warnings 暴露向量覆盖不完整。超阈值且向量索引 ready 时仅返回一个**建议字段**，Agent/用户可自行决定是否调 `memory_split`。向量不可用或索引维护中不返回 `split_hint`。

### 3.2 偏移量计算（arbiter 负责，offset 绝不交给 LLM）

以下计算在 `memory_split` 二次提交时执行（见 3.3），是分段的核心逻辑。

**两条路径，LLM 介入程度不同：**

| | `provenance="parser"`（有 Markdown 标题） | `provenance="llm"`（无标题纯文本） |
|---|---|---|
| 边界判定（切哪） | **按 Markdown 标题/段落预算确定性切分，零 LLM** | arbiter 先确定性切 batch，LLM 只给 batch 内 anchor |
| title/summary（打什么标签） | LLM 生成（标题名不一定有语义） | LLM 生成 |
| offset | parser 已知的全局字符位置 | arbiter 用私有 batch 起点 + 局部 anchor 算 |
| LLM 调用次数 | 1..N 批（只生成标签） | 1..N 批（批内边界 + 标签） |

**为什么不区分文件格式**：memory-arbiter 的 `content` 是纯文本字符串，不解析文件。Excel 等结构化文档由调用方（Agent）在写入前转成 Markdown（每个 sheet 用 `## {sheet名}` 标题 + 表格内容），arbiter 只看 Markdown 文本结构。不管来源是 Markdown 文档、Excel 转的 Markdown、还是会议纪要，处理方式完全一致。

#### 路径 A：`provenance="parser"`（有 Markdown 标题）

1. **确定性检测 Markdown 标题**：v0.6 只认 fenced code block（三反引号或三波浪号围栏）之外、行首的 ATX H1-H6（`^#{1,6}\s+.+`），允许尾随 closing hashes；不识别 Setext、缩进或 blockquote 内标题。相同输入必须产生相同 source boundaries
2. **标题 ≥2 个 → 按标题切分**：每个标题的字符位置是 source section 起点。第一个标题之前的 preamble 归入第一段（`start_offset=0`）；重复标题按出现顺序处理，不依赖标题文本唯一性
3. **上下文预算细分**：单个 source section 超过 `llm_batch_chars` 时，在不超过预算的最近空行/段落/句末标点边界确定性细分为最终 sections；再把完整最终 sections 分组为 batch。任何 batch 都不得超过预算，也不得跨越既定 section 边界
4. **LLM 生成标签**：每批只为 `source_section_index` 对应内容生成 `title/summary`。arbiter 以 parser manifest 为准，不接受 LLM 修改边界

#### 路径 B：`provenance="llm"`（无标题纯文本）

1. **arbiter 先切 batch**：优先在空行/段落边界附近切分，其次在句末标点（中文 `。！？`，英文 `. ! ?` 后跟空白或换行）附近切分，最后才在硬字符上限处切；批次严格有序、无重叠、无空洞，且每批 `len(content) <= llm_batch_chars`。因此 batch 首尾是最终 section 的强制技术边界；这是上下文窗口约束，不是 LLM 的全局语义判断
2. **LLM 生成批内 sections**：title/summary，以及除批内第一段外的 `anchor_text/occurrence_index`。第一段的局部起点由 arbiter 强制为 0，不要求 anchor。`structure_hint` 是 batch 级字段：parser batch 为 `structured`，普通纯文本 batch 为 `normal`；无标题且当前批段落分隔符极少时为 `low_structure`。仅对 `low_structure` batch，`split_prompt` 明确指示 LLM “该批文本无自然段落结构，请根据语义主题变化自行识别批内边界，至少分成 2 段；不得跨 batch 合并”
3. **arbiter 计算全局 offset**：
   - 对批内第 2..N 段，用 `find_nth_occurrence(anchor_text, occurrence_index)` 只在**当前 batch content** 中定位局部起点
   - `occurrence_index` 是 anchor 在当前 batch 内的 0-based 序号，不是全文序号
   - `global_start = private_batch_start + local_start`；`private_batch_start` 不暴露给 LLM
   - 批内起点不严格递增、anchor 未找到或 occurrence 越界 → 校验失败；错误必须包含 `batch_id + batch_section_index`

#### 公共步骤（两条路径都执行）

1. **先校验批次与数量，不做 embedding**：重新生成 manifest；每个预期 `batch_id` 必须恰好出现一次，未知/重复/缺失批次直接拒绝；随后校验 `2 <= len(sections) <= max_sections`（少于 2 段 → 分段无意义，拒绝）。数量超限或不足必须在 offset 遍历和 embedding 之前失败，避免白算最多几十次向量
2. 推导 end_offset：批内 section 的 end 为下一段 start，批内最后一段 end 为该 batch 私有 end；合并批次后得到全局有序 offsets
3. **全局校验**：
   - start_offset 严格递增
   - 相邻不重叠且连续
   - 第一个 `start == 0`，最后 `end == len(content)`
   - 拼接还原 == content：`"".join(content[s:e] for s,e in offsets) == content`
4. **单段长度预限**：section body 超过 `max_section_chars`（默认 3600）时先做字符级预截断并标记 `embedding_truncated=1`。这是防御性上限，不替代下一步的 tokenizer 精确预算
5. **生成 section embedding**（向量是硬前提，这步必须成功）：调用 1.1b 的统一 helper，输入 `prefix=title_path`、完整 `body=content[start_offset:end_offset]` 和 `max_body_chars=max_section_chars`；helper 先记录完整 token 数，再做字符预限和精确 token 截断。向量存入 `memory_sections_vec`，并写 truncated/original/used token metadata；space ID 由全局 `_vec_index_meta` 管理。任一失败 → 全部分段不发布

**offset 绝不交给 LLM**（LLM 的 tokenizer 算的是 token 数不是字符数，offset 给错 = 段落原文硬损坏）。结构化文档路径的 offset 由 parser 给；纯文本路径由 arbiter 在确定性 batch 中用 `content.find()` 定位局部 anchor 再换算全局位置。两条路径的全局 offset 都不经过 LLM。

### 3.3 `memory_split`（可分批 prepare + 原子 publish，分段唯一入口）

**场景**：长记忆已入库（`split_status=NULL`），用户想分段提升检索精度。此刻库里只有完整原文，没有任何 sections 数据。`memory_split` 给 Agent 一个入口：先按安全预算逐批取得正文并让外部 LLM 生成 sections，再一次性提交发布。

**首次调用（`split_decision` 缺省）—— 待确认：**

前置校验：
- memory 存在且 status 为 active（superseded/deleted 拒绝）
- `split_status` ∈ {NULL, `failed`, `declined`}（已是 `active` → 提示用 `split_decision="rebuild"` 重建）
- `split_enabled == True`
- **托管向量可用且 `_vec_index_meta.state == ready`**—— 向量是硬前提；mismatch/failed/unmanaged 时返回可诊断状态，要求先完成迁移或配置 GGUF
- `len(content)` 当前超 `split_threshold`（若内容变短到阈值内 → 返回"无需分段"）

通过后按 2.3 生成 manifest 并返回当前批次（**不落任何库改动**）：
```
requires_user_confirmation = true
content_hash = sha256(content)
memory_version = memories.version
split_revision = memories.split_revision
char_count = len(content)
split_status = <当前状态>
batch_count = <确定性批次数>
current_batch = {batch_index, batch_id, structure_hint, content, source_sections}
split_prompt = "该记忆 {char_count} 字符，预计需要 {batch_count} 个 LLM 批次。原文已完整保存，分段仅影响检索精度。每批独立处理，batch 首尾是强制技术边界，不得跨批合并。是否分段？"
split_schema = <2.3 中定义的 sections 结构>
ok = true
```

Agent 逐批处理：
- **无标题路径**：LLM 为当前批生成局部 sections；第一段从局部 0 开始，后续段带局部 anchor/occurrence。当前批 `structure_hint=low_structure` 时，split_prompt 额外指示 LLM 根据语义主题变化自行识别批内边界并至少分 2 段
- **有标题路径**：LLM 按当前批 `source_section_index` 只生成 title/summary
- 处理下一批时带回 `content_hash + memory_version + split_status + split_revision + llm_batch_chars`；任一快照不匹配立即停止，避免继续为过期正文付费

**最终 publish（`split_decision="split"` 或 `"rebuild"` + 全部批次 `sections` + 四个 decision 快照字段）—— 执行：**

> `split` 和 `rebuild` 的唯一区别：`split` 要求 `split_status ∈ {NULL, failed, declined}`，`rebuild` 要求 `split_status == 'active'`（用于分段边界、标题或标签本身需要重新生成的场景；模型切换只需 `memory_rebuild_embeddings`）。事务体完全相同（都先 DELETE 旧 section 两表再 INSERT），因为 DELETE 在空表上是空操作。

1. 重新读取当前 memory 快照和 `_vec_index_meta.state`。校验 content hash、memory version、split revision、决策对应的预期 split status；任一不匹配在昂贵 embedding 前拒绝。Vec 不是 `ready` 时同样拒绝，旧状态/旧 sections 不变
2. 按相同 `llm_batch_chars` 重新生成 manifest，重新计算理论最小 section 数，校验 batch 完整性和 `2 <= len(sections) <= max_sections`；再执行 3.2 的局部 anchor → 全局 offset 和全文覆盖校验
3. 生成全部 section embedding。**向量是硬前提，前两步和 embedding 必须全成功才进入发布事务**。任一失败 → 走失败保底
4. 事务体（短事务原子发布，`BEGIN IMMEDIATE` 获取写锁防止并发）：
   ```sql
   BEGIN IMMEDIATE;
     -- ⚠️ 完整 CAS：LLM/embedding 期间 memory_edit、split 或 rebuild 都可能提交
     -- 用 BEGIN IMMEDIATE 获取写锁，如果 memory_edit 已持有锁 → SQLITE_BUSY → 重试或返回 busy
     current = SELECT status, content, version, split_status, split_revision FROM memories WHERE id = ?;
     IF vec_state != 'ready' OR active_space_id != generated_embedding_space_id:
         ROLLBACK;
         返回 code = "vec_space_changed"
     IF current.status != 'active'
        OR sha256(current.content) != decision_content_hash
        OR current.version != decision_memory_version:
         ROLLBACK;
         返回 code = "memory_changed"，附当前 hash/version
     IF current.split_status IS NOT decision_split_status
        OR current.split_revision != decision_split_revision
        OR (decision == 'split' AND decision_split_status NOT IN (NULL, 'failed', 'declined'))
        OR (decision == 'rebuild' AND decision_split_status != 'active'):
         ROLLBACK;
         返回 code = "split_revision_conflict"，附当前 status/revision

     -- 所有 CAS 通过后才允许删除旧派生索引
     DELETE FROM memory_sections_vec WHERE id IN (SELECT id FROM memory_sections WHERE memory_id = ?);
     DELETE FROM memory_sections WHERE memory_id = ?;
     -- 写 sections 元数据（不存 body，offset 在 3.2 已算好）
     INSERT INTO memory_sections(memory_id, section_index, title, title_path, summary,
                                 anchor_text, occurrence_index,
                                 start_offset, end_offset, provenance, embedding_truncated,
                                 embedding_original_tokens, embedding_used_tokens, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
     -- 写 section vec（向量是硬前提，3.2 已生成好）
     INSERT INTO memory_sections_vec(id, embedding) VALUES (?, ?);
     -- 切换状态并推进派生状态版本；原文不 UPDATE
     UPDATE memories
        SET split_status = 'active', split_revision = split_revision + 1
      WHERE id = ? AND split_revision = decision_split_revision;
     ASSERT changes() == 1;
   COMMIT;
   ```

**decline 发布**不经过 LLM/embedding，但仍是一次状态发布，必须使用同样的乐观锁：`BEGIN IMMEDIATE` 后校验当前 memory status=active、content hash、memory version、当前 `split_status IS decision_split_status`、`decision_split_status in {NULL, failed, declined}` 与 `split_revision`；全部匹配才执行 `split_status='declined', split_revision=split_revision+1`。任一快照不匹配时返回当前值并要求用户重新确认，不能把针对旧正文的拒绝应用到新正文。

**失败保底**：`memory_split` 针对**已存在**的 memory，原文已在库中，失败时**不重写原文、不删原文**。
- batch/offset/sections 校验失败或 section embedding 失败 → 不执行发布事务。首次 split 只有在 `content/version + split_status + split_revision` 仍匹配 prepare 快照时，才用短 CAS 事务标 `failed`、写 `last_split_error` 并递增 revision；CAS 已冲突则不写失败状态，避免覆盖并发成功结果
- CAS 校验失败（memory changed 或 split revision conflict）→ 事务 ROLLBACK，`split_status`、旧 sections 和旧向量完全不变；并发冲突不是业务失败，禁止标 `failed`
- Vec space CAS 失败（事务内不再 ready 或 active_space_id 已变化）→ 事务 ROLLBACK，`split_status` 和旧 sections 不变，提示先完成向量迁移后重试。
- **rebuild 失败** → 保持 `split_status='active'`，旧 sections + 旧向量不动。`last_rebuild_error` 也只能在 revision 仍匹配时 CAS merge；若另一个 rebuild 已成功，当前失败不得把错误写到新 revision 上

> **状态机说明**：可归因于当前 prepare 的 validation/embedding 失败用 CAS 标 `failed`，并发/空间/busy 冲突保持数据库当前状态。不存在“分段了但没向量”的中间态——要么 offset + embedding 全成功后原子切 `active`，要么新 sections 整体不发布。

### 3.4 编辑场景（memory_edit 触发 section 清理）

content 变化 → section 偏移失效：

1. `edit_memory` 内部检测到 content 变化
2. 无论当前 `split_status` 是什么，都在 `edit_memory` 现有事务内（`INSERT memory_history → UPDATE memories → FTS delete+insert` 链路）执行：
   ```sql
   DELETE FROM memory_sections_vec WHERE id IN (SELECT id FROM memory_sections WHERE memory_id = ?);
   DELETE FROM memory_sections WHERE memory_id = ?;
   UPDATE memories
      SET split_status = NULL,
          split_revision = split_revision + 1
    WHERE id = ?;
   ```
   即使当前没有 sections，也必须递增 `split_revision`，使 edit 前已经发出的 prepare/batch/publish 全部失效
3. 若用户希望重新分段新内容 → 走 `memory_split(memory_id)` 主动触发（见 3.3）

**清理两张表**（对比旧方案三张表，少了 FTS 表的 external content 删除路径）。

只改 subject、不改 content 时不清 sections、也不递增 `split_revision`；但 `memories.version` 会变化，memory 级向量必须按新的 `subject + content` 通过统一 helper 重建。在途 split/rebuild 因 `decision_memory_version` 不匹配而重新 prepare，避免与 edit 的 metadata/向量更新交错发布。

## 4. 查询流程

### 4.1 召回（现有 5 通道不变）

`_wide_recall` 的现有 5 个通道（FTS main / FTS OR / subject-tags LIKE / content LIKE / memory vec KNN）保留，不加任何新通道。前 4 个通道原样运行；托管模式的 memory Vec KNN 只有 `_vec_index_meta.state == ready` 且 query 空间匹配时运行。`unmanaged` 仅允许旧版显式 query 并返回兼容性 warning；门禁关闭时查询明确降级到 FTS/LIKE，而不是把不同空间的向量混算。

**产品定位（明确声明）**：本方案是"命中后定位"——section Vec **不参与召回**，不能让仅局部语义相关的长记忆进入 top-N。一条长记忆能否被搜到，取决于 memory 级 FTS（字面匹配完整正文）和 memory 级单向量（token-safe 前缀语义）。section Vec 只在 memory 已命中后挑出相关段落。若 query 只与超长文档后部章节语义相关、原文又没有对应字面词，FTS 与 memory Vec 都可能漏召回。这是本版有意识接受的取舍——不新增召回通道，换取实现简单和低回归风险。

### 4.2 不改 `_soft_rerank` 的打分

打分逻辑、排序 key 全部保持现状。分段**不影响打分**——检索复杂度不随分段数量增长。一条长记忆即使分了 20 段，它在 pool 里仍只算 1 条 memory，`_final_score` 基于全文算。

### 4.3 返回前的 section 匹配（新增一步，hybrid 和 bm25 模式共享）

**两种排名模式都执行此步骤**：当前代码有 hybrid（`_soft_rerank`）和 bm25（`_search_bm25`）两种模式，都会在切到 limit 后返回结果列表。section 匹配是在**结果列表返回前**的后处理，不绑定特定排名模式。实现建议：抽成独立的 `attach_sections(results, query, query_embedding)` 函数，两种模式在 return 前统一调用。

`attach_sections` 的第一职责不是匹配，而是**最终一致性物化**：提取 limit 后的 result IDs，开启一个短 `BEGIN` 读事务，在同一 snapshot 内批量重读当前 memories、全局 Vec 状态、sections 和 section Vec 匹配。候选结果缓存的 `split_status/content` 不参与返回决策。若记录的当前 status 已不满足本次 `include_superseded` 可见性规则，则移除该结果；允许并发极端情况下返回少于 limit，不为补位重跑召回。

对同一快照中每条 `split_status == "active"` 的结果：

1. 读取已批量捞出的 sections。若 `total_sections == 0`，视为派生索引不变量损坏：返回该快照中的完整正文，设置 `content_omitted=false`、`section_enhancement_applied=false` 并追加 `split_invariant_broken_empty_sections` warning；**立即结束该条处理，禁止计算除法**。若只有 1 段，同样违反发布时 `sections>=2` 的不变量，全文降级并返回 `split_invariant_broken_too_few_sections`。search 是只读路径，不在这里自动修库
2. 判断本次 query 的 section Vec 门禁。sqlite-vec 不可用、`state!=ready`、query embedding 不存在/生成失败或 query 空间不匹配时，**不得把它解释为零命中**：返回当前快照全文，设置 `content_omitted=false`、`section_enhancement_applied=false`，附具体 `vec_disabled` warning 后结束该条处理
3. 门禁打开后，先比较当前 snapshot 的 section IDs 与 section Vec IDs。任一 active section 缺 Vec 时返回全文、`section_enhancement_applied=false` 和 `split_invariant_broken_missing_section_vec`；不得把缺失向量的 section 当作“不相关”。修复由 `memory_rebuild_embeddings(memory_ids=[...])` 或迁移模式完成，search 不写库
4. 完整性通过后执行 section Vec 语义匹配（**不截断，命中几段返回几段**）：
   - 用 `query_embedding` 对该 memory 的 sections 做 section Vec 语义匹配
   - Vec 命中必须与当前 snapshot 的 section ID 集合求交，丢弃孤儿/已删除 section ID
5. 根据 `matched_count / total_sections` 比例决定返回策略：
   - **= 0**：返回 `section_catalog`（全部段落目录）+ hint
   - **≥ `section_fulltext_threshold`（默认 0.8）**：返回全文 + `matched_sections`（附带供参考）+ hint "{pct}% 段落命中，建议看全文"
   - **其他**：返回所有 `matched_sections`（不截断）+ `section_catalog`（未命中的）

**不截断的设计理由**：分段是增强，不能以牺牲准确度为代价。人为截断到 3 段，但实际有 8 段都相关 → 信息丢失 → 比不分段还差。匹配了多少就返回多少。如果大部分都命中了（≥80%），说明整条都相关，直接返回全文（片段定位此时无价值）。

事务在本批结果物化完成后立即 COMMIT。该快照使用 1.1c 定义的调用独占 connection；WAL 下读者持有旧 snapshot 时 writer 可继续提交。同时配置有限 `busy_timeout`，遇 `SQLITE_BUSY` 时做有上限重试并返回可诊断 warning，禁止无限等待。

## 5. 配置项

> **警告：这是高阶配置。** 分段功能依赖 sqlite-vec 扩展 + embedding 服务 + LLM 调用，三个条件全部就绪才能正常工作。默认全部关闭。请勿在生产环境随意开启，先在测试环境验证依赖可用性。

`Settings` 新增（config.py）：

```python
# ========== 分段增强功能开关 ==========

# 是否启用分段增强。默认关。开启前请确认：
#   1. sqlite-vec 已安装并加载成功
#   2. 托管 GGUF embedding 已配置并可用
#   3. _vec_index_meta.state == ready（不在迁移/失败/unmanaged）
#   4. 确实有长文检索需求（记忆经常超过阈值）
# 不满足任一条件 → 不要开，开了也不会生效（见 0.4 准入条件）
split_enabled: bool = False              # MEMORY_ARBITER_SPLIT_ENABLED

# ========== 分段触发阈值 ==========

# 内容超过此字符数时才触发分段提示。默认 4000。
# 推荐值：4000（中文约 4000 字，英文约 800 词）。这是”一屏扫不完、分段有价值”的临界点。
# 核心目的：节省 LLM token。低于此值的短文档用户一眼能扫完，分段提供的“定位到章节”价值不大，
# 却要消耗至少 1 个 LLM 批次（生成 title/summary/anchor_text），ROI 低。
# 不要设太低：< 4000 会导致几乎每条记忆都触发分段，LLM 调用频繁，token 浪费严重。4000 字符 = 4000 个中文字
#   已经是“扫不完”的临界点，再低分段就失去意义了。
# 不要设太高：无硬上限。只要用户有长文检索需求，20000-30000 字符（如完整 API 文档、项目计划书）都值得分段。
#   分段只影响检索精度，不影响内容完整性——即使不分段，原文也照常入库、照常可搜。
# 适用场景：知识库文档（3000-8000 字/篇）、API 文档（5000+ 字/篇）、项目计划、长文报告。
# 不适用场景：短笔记（< 2000 字）、对话摘要、代码片段、周报——这些通常无需分段，也省不了 token。
split_threshold: int = 4000              # MEMORY_ARBITER_SPLIT_THRESHOLD

# ========== 返回增强参数 ==========

# section Vec 语义匹配的 distance 上限。超过此值的 section 不算”命中”。
# KNN 只有排序没有”命中/未命中”语义——如果不设阈值，所有 section 的 distance 都会被当成命中，
# matched_count 趋近 total，80% 全文回退阈值频繁触发，分段失效。
# 此阈值需用真实数据校准：取”明显相关”的 section 的 distance P90 作为初始值。
# 推荐值：需校准。开发期临时值设 0.7（cosine distance），上线后根据实际命中情况调整。
# 不要设太大：> 1.5 几乎不过滤，等于没有 relevance gate。
# 不要设太小：< 0.3 会过度过滤，只有极相似的 section 才命中。
section_vec_distance_threshold: float = 0.7  # MEMORY_ARBITER_SECTION_VEC_DISTANCE_THRESHOLD

# 发布门：0.7 只是开发期临时值，不是跨模型常数。v0.6.0 默认开启前必须用真实语料标注“相关/不相关”section，
# 至少记录相关 distance P90、不相关 distance P10 和全文回退触发率；若两类无法有效分离，保持 split_enabled=False。

# 当命中段落占该 memory 总段数的比例达到此阈值时，放弃返回片段、改为返回全文。
# 默认 0.8（80%）。这是”分段增强不能牺牲准确度”原则的体现：
#   - 如果 12 段命中了 3 段（25%）→ 返回 3 段片段（片段定位有价值）
#   - 如果 12 段命中了 10 段（83%）→ 返回全文（整条都相关，片段已无定位价值）
#   - 如果 12 段命中了 0 段（0%）→ 返回目录（memory 级命中但段落级未命中，让 Agent 自行选）
# 推荐值：0.8。低于 0.6 会导致频繁返回全文（失去分段意义），高于 0.95 会导致几乎不触发全文回退
#   （即使整条都相关也强行返回片段，信息碎片化）。
# matched_sections 不截断——命中几段返回几段，不人为丢弃匹配结果。
section_fulltext_threshold: float = 0.8  # MEMORY_ARBITER_SECTION_FULLTEXT_THRESHOLD

# 单条 memory 的 section 数量上限。超出 → 分段失败，走保底。
# 少于 2 段 → 分段无意义（等于没分），同样拒绝。
# 防止 LLM 对超长文档生成过多碎片段落，导致返回膨胀和 embedding 耗时过长。
# 推荐值：50。一条 5 万字文档拆 20-30 段已足够细，超过 50 段通常是 LLM 切碎了。
max_sections: int = 50                    # MEMORY_ARBITER_MAX_SECTIONS

# 单个 section body 构造 embedding 候选时的字符级防御上限。完整 body 可先被 tokenizer 计数用于诊断，
# 但超过该值的后缀不送入模型并标记 embedding_truncated。这不是最终上下文边界：helper 仍须做精确 token 预算，
# token_budget = n_ctx - reserved_tokens；任一阶段截断都必须显式标记。
# embedder.py 的 Llama() 初始化设 n_ctx=2048；推荐 max_section_chars=3600 只用于限制最坏输入规模。
max_section_chars: int = 3600             # MEMORY_ARBITER_MAX_SECTION_CHARS
```

外部 LLM 的 `llm_batch_chars` 不做全局 Settings：它取决于调用方本轮选用的模型上下文，由 `memory_split` prepare 参数显式传入（默认 12000）。manifest 和 `batch_id` 绑定该值，同一次 prepare/publish 不得中途改变；需要调整时重新 prepare。这样避免把外部模型能力散落成一个容易过期的服务端配置。

**配置优先级**：config.json > 环境变量 > 代码默认值。所有配置项均为可选，有默认值。config.json 中已配置的值会覆盖环境变量（如 config.json 中 `vec.dim=512` 会覆盖 `MEMORY_ARBITER_VEC_DIM=768`），环境变量仅在 config.json 中未配置时生效。这是有意设计的——配置文件一目了然，好维护；环境变量是隐式的，容易忘。

**`split_threshold` 推荐值 4000 的推导**：
- 中文：Python `len()` 统计字符数，中文一字一字符。4000 字符 = 4000 个中文字。一个段落大约 200-300 字 → 4000 字 ≈ 15-20 个自然段落。
- 英文：4000 字符 ≈ 800 词（平均 1 词 ≈ 5 字符）。一个段落大约 100-150 词 → 800 词 ≈ 5-8 个自然段落。
- 结论：4000 字符对应的内容大约有 8-20 个自然段落（中文偏多，英文偏少），这个长度下用户已经无法”一眼扫完”，分段带来的检索精度提升开始有实际价值。低于 4000 的中等长度文档，用户扫一眼就能找到关键信息，分段的意义不大，却要消耗至少 1 个 LLM 批次，ROI 低。
- **核心目的：节省 LLM token**。分段是付费增强——每条超阈值的记忆需要 1..N 个 LLM 批次生成 sections。如果阈值太低，LLM 调用频繁，token 浪费严重。4000 字符的阈值能过滤掉大部分中等长度文档，只对真正需要分段的长文触发，把 token 花在刀刃上；超长文档则用更多批次换取不截断和可诊断性。
- 真实场景验证：金营项目知识库 18264 字符（12 章节）、鹊桥接口文档 52238 字符（约 20 章节），均为 4000 的 4.5~13 倍，充分受益于分段。

**为什么说这是高阶配置**：分段不是“开了就能用”的轻量功能。它需要：
1. **sqlite-vec 扩展已安装** — 不是所有 SQLite 环境都支持。
2. **embedding 服务运行正常** — v0.6.0 的分段路径只支持现有本地 GGUF provider。远程 API/自定义 embedding 在本版仅保留 memory 级显式向量用法，不满足 sections 准入条件。
3. **LLM 调用成本** — 每条超阈值记忆的首次分段需要 1..N 个 LLM 批次（生成 sections）；prepare 会先返回 `estimated_llm_calls`，用户确认后再继续。
4. **维护成本** — `memory_edit` 改内容会清空 sections（需重新分段），分段失败需要 Agent 诊断。

对大多数用户来说，现有的 memory 级 FTS + Vec 召回已经足够。分段增强只适用于“有大量长文知识库、需要段落级检索精度”的场景。**如果你的记忆大部分是短笔记、对话摘要、代码片段，不要开启分段。**

**功能作用（供 README 使用）**：本功能旨在解决长文（>4000 字符）检索时”定位到具体章节”困难的问题。通过将长文分段并生成段落级向量，让 `memory_search` 在命中一条长文后，能返回最相关的段落（`matched_sections`），而不是返回全文。命中段落数不截断——匹配多少返回多少；如果大部分段落都命中（≥80%），说明整条都相关，自动回退为返回全文。这能显著提升 Agent 处理长文的效率，同时通过 `split_threshold=4000` 的设计，把 LLM 调用（生成 sections）的 token 消耗控制在必要范围内，实现”用最少的 token，解决最痛的长文检索问题”.

## 6. 迁移与兼容

### 6.1 Schema 迁移（幂等，启动时自动跑）
- 新建 `memory_sections`（`CREATE IF NOT EXISTS`）
- 新建 `memory_sections_vec`（`CREATE IF NOT EXISTS`，仅在 `sqlite_vec_available=True` 时）
- 新建 `_vec_index_meta`（`CREATE IF NOT EXISTS`）
- `memories` 加 `split_status` 和 `split_revision NOT NULL DEFAULT 0` 两列：必须先用 `PRAGMA table_info(memories)` 分别探测，**仅当该列不存在时**执行对应 `ALTER TABLE`
- 首次增加列时，SQLite 通过列默认值使旧记忆表现为 `split_status=NULL, split_revision=0`；这是一条一次性迁移语义，**禁止在启动时执行 blanket `UPDATE memories SET split_status=NULL, split_revision=0`**。列已存在时必须完整保留 active/failed/declined 状态和 revision，否则会让已有 sections 与主状态失配
- 按 1.1b 的 Legacy 初始化规则建立全局向量空间状态；已有向量但无来源元数据时安全降级为 `mismatch`
- schema migration 必须在服务接受工具调用前由初始化连接完成：先确保 WAL，再在 `BEGIN IMMEDIATE` 内重新探测并执行 schema 变更，兼容多个 MCP 进程并发首次启动；完成后运行期按 1.1c 使用每操作/每事务独立连接
- SQLite 启动契约显式要求 `PRAGMA journal_mode=WAL`（现有实现已设置）并为每个运行期连接配置有限 `busy_timeout`。`memory_search` 的最终物化使用短 read snapshot；split/edit/rebuild 使用短 `BEGIN IMMEDIATE`。遇 `SQLITE_BUSY` 做有上限重试后返回可诊断错误，禁止无限阻塞 MCP 请求

### 6.2 行为兼容
- `split_enabled=False`（默认）→ 所有写入走普通分段路径，`split_status` 永远 NULL，`memory_search` 不追加 section 返回字段
- `memory_write` / `memory_edit` / `memory_get` 的主数据行为与 v0.5.4 一致；分段不开启时不会写 `memory_sections` 两表
- **明确的一次性兼容例外**：旧库已有 `memories_vec`、且配置了托管 GGUF embedder，但没有 `_vec_index_meta` 时，无法证明旧向量由当前模型生成。首次升级进入 `mismatch`，memory Vec 临时禁用，FTS/LIKE 正常；执行 `memory_rebuild_embeddings` 后恢复。不能再宣称此场景“行为完全一致”
- 未配置托管 embedder、仅使用显式向量的旧库进入 `unmanaged`，保留 v0.5.4 memory Vec 行为并给出 warning；sections 不可用，不强迫用户进入一个无法自动完成的重建流程
- 显式 `memory_store_embedding` / `memory_search(query_embedding=...)` 的高级调用新增可选空间 ID；在托管空间中缺失或 mismatch 时安全拒绝写入/跳过 Vec，避免静默污染
- `memory_sections` + `memory_sections_vec` 是可重建派生索引；`_vec_index_meta` 是全局正确性元数据，不能随意删除。删除后若 vec 表有数据，启动时会再次进入 `mismatch`

### 6.3 现有 bug 顺带修（server.py dead code）
`server.py` 第 60-68 行：`memory_get` handler 之后紧跟的 `memory_store_embedding` 注册。当前代码已把 `memory_store_embedding` 抽成独立 `@app.tool()`（见 server.py 第 65-68 行），无 dead code。本版本确认此 bug 已修复，无需再动。

## 7. 实现拆解（建议 PR 粒度）

| PR | 内容 | 风险 | 改动文件 |
|---|---|---|---|
| 1 | Schema 幂等迁移 + SQLite connection factory：`memory_sections` / Vec / meta 表、split 列；每操作/事务独立连接、逐连接 PRAGMA/扩展初始化、WAL/busy timeout | 中（改变 DB 生命周期） | db.py |
| 2 | 统一 token-safe embedder helper + `embedding_space_id` + `_vec_index_meta` 状态读写；write/edit/query 复用同一输入策略 | 中（影响全部自动 embedding 路径） | embedder.py, db.py, tools.py |
| 3 | `memory_sections` / vec 的 db 层 CRUD + 确定性 batch manifest + 局部 anchor→全局 offset 校验 + section embedding | 中 | db.py, tools.py |
| 4 | `memory_write` 追加 `split_hint`（超阈值 + 向量可用时建议调用 memory_split） | 低（只追加返回字段） | tools.py |
| 5 | `memory_split` 多批 prepare + `split_revision` 完整 CAS + 原子 publish；并发冲突/失败保底 | 高（新工具，核心并发协议） | tools.py, server.py |
| 6 | `memory_search`：全局 Vec 空间门禁 + 显式 query space 校验；hybrid/bm25 共享短 read-snapshot `attach_sections` | 中（不新增召回通道，但改变最终物化） | search.py, tools.py, server.py |
| 7 | `memory_store_embedding` 守门 / `get_sections` / 状态工具 / rebuild 的 active+superseded 重建、deleted+orphan cleanup、migration epoch/lease/expected-cursor CAS | 高（涉及迁移状态机、租约与并发 CAS） | db.py, tools.py, server.py |
| 8 | 单元测试 + 并发多连接压测 + 1.8 万/5.2 万字符真实文档校准 | — | tests/, docs/ |

**关键变化**：相比旧方案，`memory_write` 从"高风险的两阶段协议改主入口"降级为"低风险的返回字段追加"。分段发布复杂度集中在 `memory_split`；全局 embedding 正确性单独放在 PR 2/6/7，避免与 section CRUD 混在一个提交里。search 不新增召回通道，但会在空间不可信时明确关闭现有 Vec 通道。

## 8. 测试要点

1. **offset 校验**：anchor 不在 content / occurrence 越界 / 起点无序 → 分段失败，事务不执行，`split_status` 标为 `failed`（`metadata._split.last_split_error.stage=validation`），原文可搜
2. **连续覆盖还原**：`"".join(content[s.start:s.end]) == content` 必须成立
3. **原文立即入库**：长文本 `memory_write` 后，返回 `data.id` 非空、`data.backup_only=false`，且记忆可立即搜到（不等用户确认分段）
4. **分段不影响召回**：`_vec_index_meta.state=ready` 时，一条 18264 字符记忆分 12 段后，`memory_search` 的候选 pool 数量、打分、排序与未分段时一致（验证没有新增 section recall 通道）
5. **section Vec 语义匹配与门禁降级**：配置 auto query embedding + sqlite_vec 且空间门禁为 ready 后，query 与某段向量语义相近（distance ≤ 阈值）→ `matched_sections` 含该段；没有有效 query embedding、空间不匹配或 state 非 ready 时不执行 section 匹配，返回当前全文、`section_enhancement_applied=false` 和具体 warning
6. **relevance gate**：section Vec 的 distance 超过 `section_vec_distance_threshold` 的 section 不算命中；不设阈值时所有 section 都"命中"导致全文回退失效
7. **最小段数约束**：LLM 对超阈值文档只返回 1 段 → 分段拒绝（`split_status` 不变），不执行 embedding；要求 ≥ 2 段才接受
8. **不截断**：一条 12 段的记忆，query 命中了其中 8 段 → `matched_sections` 返回全部 8 段（不截断）
9. **全文回退**：一条 12 段的记忆，query 命中了 10 段（≥80%）→ 返回全文（`content_omitted=false`）+ `matched_sections`（附带）+ hint "建议看全文"
10. **真实零命中返回目录**：section Vec 门禁已打开且实际查询执行成功，但没有任何段落通过 relevance gate → `content_omitted=true` + 返回带 title/title_path/summary 和 embedding 截断 token metadata 的 `section_catalog` + hint；不得用 Vec 不可用模拟零命中
11. **content_omitted 省省 token**：active 分段部分命中时，`content=null, content_omitted=true`，`matched_sections` 只含 title/summary（无原文）；Agent 需要原文调 `get_sections`
12. **向量是发布硬前提、查询可降级**：关闭 sqlite-vec/GGUF embedding，或把 vec state 置为 mismatch/failed/unmanaged，`memory_write` 均不返回 `split_hint`；`memory_split` 返回具体状态和恢复建议；维护窗口内不发布新 sections。已有 active sections 的 search 必须返回全文，不返回 catalog-only
13. **业务失败 CAS 标记**：batch/offset 校验或 embedding 失败时，仅在 content/version/split revision 仍匹配 prepare 的情况下标 `failed` 并递增 revision；并发冲突、Vec space 变化、busy 不得标 failed
14. **完整 publish CAS**：embedding 完成后事务内再次校验 memory status、content hash、memory version、预期 split status、split revision 和 vec state/space；任一变化 → ROLLBACK，旧 sections 不变并返回稳定错误码
15. **编辑触发清理与失效**：`memory_edit` 改 content 后，两张 section 表清空、`split_status` 归 NULL、`split_revision+1`；即使原本无 sections 也递增，使所有在途 batch/publish 失效
16. **memory 级 limit**：一条记忆分 12 段，search 仍算 1 条 memory
17. **memory_get 兜底**：分段后 `memory_get` 仍返回完整原文
18. **`memory_split` 五模式**：首次/后续 batch prepare 都不落库；split/rebuild publish 带 content hash、memory version、split status、split revision 四字段快照并原子替换；decline 做同样 CAS；单批响应只含 `current_batch.content`，多批时不得泄露完整正文
19. **section 数量约束前置**：LLM 生成的 sections 少于 2 段或超过 `max_sections`（默认 50）→ 在 offset/embedding 前失败，embedding 调用次数必须为 0；若多个 `low_structure` batch 的理论下限 `2 * batch_count` 已超限，prepare 在第一次 LLM 调用前返回 `too_many_batches_for_max_sections`
20. **UNIQUE 约束**：尝试写入重复 `(memory_id, section_index)` → DB 报错（验证约束生效）
21. **向量空间不一致**：切换 embedding 模型后重启 server，`_vec_index_meta.state=mismatch`，所有 Vec 通道禁用（memory Vec + section Vec），`memory_search` warnings 含 `vec_disabled=embedding_space_mismatch`，FTS/LIKE 正常
22. **超长 section 截断诊断**：分别构造“字符数超限”和“字符数未超限但 title_path+body 的 token 超预算”两种 section → body 被安全截断，`embedding_truncated=1` 且 original_tokens > used_tokens；未截断时二者相等，offset/原文保持不变
23. **有标题 vs 无标题路径**：代码围栏外的行首 ATX H1-H6 标题 ≥2 个 → parser 路径；围栏内 `#`、Setext、缩进/blockquote 标题不误切；无标题路径按段落确定性分批并使用局部 anchor
24. **bm25 模式 section postprocess**：`RANKING_MODE=bm25` 时，`_search_bm25` 返回的结果也执行 section 匹配（验证 `attach_sections` 被两种排名模式调用）
25. **金营知识库实测**：18264 字符、12 章节，验证"问二期规划"时 `memory_search` 的 `matched_sections` 只包含 `## 四` 对应的 section，`split_status` 为 `active`
26. **批量向量重建分类**：active/superseded 正常重建，deleted/missing 清理并推进 cursor，无归属 section vec 在 preflight 清理；所有目标完成且无孤儿前不得切 ready
27. **rebuild 单条分段**：对 active 记忆 rebuild → 原子替换、保持 active、`split_revision+1`；并发两个 rebuild 只能一个发布成功，失败者返回 `split_revision_conflict` 且不得删除胜者结果
28. **迁移首条失败与自动续跑**：某批第一条 embedding 或 memory/section 主数据快照 CAS 失败 → `migration_cursor` 保持原值（第一批首条失败时该 key 仍不存在）、`state=failed`、`migration_complete=false`；修复后再次调用无需传 cursor，从失败项继续，不跳项。epoch/lease/expected-cursor CAS 冲突属于陈旧 runner 退出，不得改 state
29. **迁移中再次换模型**：target=B、已完成部分迁移时切到模型 C 并重启 → `target_space_id=C`、生成新 `migration_epoch`、删除 `migration_cursor` 和旧 lease keys、`state=mismatch`，从头把 A/B 混合表全部重建为 C；B runner 的迟到提交全部被 epoch CAS 拒绝
30. **局部 repair 失败不全局降级**：`state=ready` 时指定 `memory_ids` 修复，某条失败后旧向量保留，错误对调用方可见，但全局 state 仍为 ready
31. **显式向量空间守门**：托管模式下 `memory_store_embedding` 缺失/传错 space ID 时拒绝且旧向量不删；显式 `query_embedding` 缺失/传错 `query_embedding_space_id` 时跳过两个 Vec 路径并返回 warning，字面检索正常
32. **Legacy 首次升级**：有托管 GGUF 时，空 vec 表初始化为 current/ready，已有 vec 但无 meta 时初始化为 mismatch，完成重建后恢复 ready；无托管 embedder 的显式向量旧库进入 unmanaged，memory Vec 保持可用、sections 不激活并返回 warning
33. **状态可诊断**：`memory_status` 和 `memory_split_status` 都返回 vec state、active/target space、migration cursor/epoch、migration_in_progress、last_error；单条状态额外返回 split revision；不泄露内部 lease owner；mismatch/failed/unmanaged 不得静默
34. **memory token-safe embedding**：用 1.8 万和 5.2 万字符正文执行 write/edit/rebuild，不得因超过 n_ctx 失败；memory embedding 的 truncated/original/used token metadata 和 warning 可见，三条路径生成输入完全一致，改变截断策略会改变 `embedding_space_id`
35. **distance 发布校准**：用真实 query-section 标注集输出相关 P90、不相关 P10、零命中率和 ≥80% 全文回退率；把最终阈值及样本记录进测试基线，未完成校准时 `split_enabled` 保持默认关闭
36. **search TOCTOU 压测**：在召回读到 active 后并发 edit 清空 sections；最终物化不得除零，必须返回同一 read snapshot 的当前正文/状态。人工构造 active+0 sections 时返回 `split_invariant_broken_empty_sections`
37. **并发首次 split**：两个 Agent 对同一 NULL revision prepare；A 发布后 revision+1，B 必须冲突，A 的 sections/vec 不得被 B 删除或覆盖
38. **超长文档上下文预算**：18264/52238 字符的 Markdown 与无标题纯文本，在 `llm_batch_chars=8000/12000` 下每批均不超预算，batch 严格覆盖全文；漏交、重复、未知 batch 返回明确 ID 列表
39. **batch 期间 edit**：取完部分 batch 后修改正文；后续取批和最终 publish 都因 hash/version/revision 不匹配拒绝，且不继续生成 embedding
40. **迁移垃圾不阻塞**：低 ID orphan/deleted 后还有 active 目标时，cleanup 与 cursor 同事务提交，迁移能够越过垃圾记录并最终切 ready；cleanup 分类在事务前后变化时不得误删恢复后的 active 数据
41. **连接与事务隔离**：同一进程并发执行 search 短读快照、edit 和 split/rebuild 短写事务；每个事务使用不同 connection，不出现嵌套事务、跨调用 commit/rollback 或线程错误。每个新 connection 均启用 foreign_keys/busy_timeout/sqlite-vec，连接最终关闭
42. **并发 migration runner**：A/B 同时从 cursor=100 开始，只有一个获得 lease；模拟 A lease 过期、B 接管并推进到 102 后，A 的旧 101 结果必须因 epoch/owner/expected-cursor CAS 失败，cursor 不得回退，state 不得被 A 改成 failed
43. **迁移换模型隔离旧 runner**：模型 B 的 runner 计算期间切换 target=C 并生成新 epoch；B 的任何 item/final-ready 提交都必须被拒绝，不得删除 C 空间向量或覆盖 meta
44. **Schema migration 真幂等**：从无 split 列的旧库升级后旧行得到 NULL/0；构造 active sections 后连续重启/重复跑 migration，split_status/revision/sections/vec 完全不变，且不得执行 blanket reset UPDATE
45. **active section 缺 Vec 修复**：人工删除一条 active section Vec；dry-run 必须把其 memory 列为完整性修复目标，迁移完成判定不得切 ready。若缺失 memory ID 已不大于 cursor，申请 lease 时必须先回退 cursor 再重算后缀；补齐后才允许 ready
46. **默认批量上限**：不传 `batch_size` 的实际执行最多处理 50 条并正确返回 `has_more`；显式 `None` 才允许同步不限量，0/负数在任何 embedding 前拒绝
47. **并发首次启动迁移**：两个 MCP 进程同时从无 split 列的旧库启动；初始化连接通过 `BEGIN IMMEDIATE` 串行化并在锁内复查 schema，两者最终都成功，列只添加一次且无 duplicate-column/半迁移状态
48. **search 检测缺失 section Vec**：在 `state=ready` 的 active memory 中人工删除一条 section Vec；最终物化必须在同一 snapshot 发现 section IDs 与 Vec IDs 不一致，返回全文、`section_enhancement_applied=false` 和 `split_invariant_broken_missing_section_vec`，不得返回误导性的部分命中/零命中目录
49. **并发启动状态归并**：两个进程同时发现 active=A/current=B；在 `BEGIN IMMEDIATE` 内重读 meta 后只能建立一个 target=B 的 migration epoch，后进入者保留已有 epoch/cursor/lease，不得用锁外旧快照覆盖

## 9. 与废弃方案的对比

| 维度 | 废弃方案（v0.5.0） | 新方案（v0.6.0） |
|------|-------------------|-----------------|
| 新增表 | 3 张业务索引表（sections + FTS + Vec） | **2 张业务索引表**（sections + sections_vec）+ 1 张全局 KV 元数据表 |
| `memory_write` 改动 | 两阶段协议（高风险改主入口） | **仅追加返回字段**（低风险） |
| 召回主路径改动 | Channel 6/7（section FTS + section Vec） | **无**（`_wide_recall` 5 通道不变；section Vec 仅用于返回增强） |
| 打分逻辑改动 | 4 处（高风险） | 不改（section Vec candidate 走现有 floor 逻辑） |
| 向量依赖 | 强（失败则 `embedding_failed` 半成品态） | **硬前提**（不可用就不分段，不存在半成品态） |
| `split_status` 取值 | 5 种（含 `embedding_failed`） | **4 种**（去掉 `embedding_failed`） |
| 编辑清理 | 删 3 张表 | 删 2 张表 |
| 原文落库时机 | 用户确认后 | **立即** |
| 对话中断丢原文 | 是（"已知取舍"） | **否** |
| offset 校验 | 相同 | 相同 |
| 原文保底 | 失败时重写原文 | **原文已在库，无需保底逻辑** |
| 实现复杂度 | 高 | 中 |

**核心差异一句话**：废弃方案让 `memory_write` 承担分段的两阶段协议（高风险改主入口），且 embedding 失败会产生 `embedding_failed` 半成品态；新方案让 `memory_write` 回归纯写入，分段统一走 `memory_split`（事后增强），向量是硬前提（不可用就不分段，不存在半成品态）。原文立即入库，分段失败零损失。

## 10. 已知取舍

以下三点是有意识的架构取舍，在设计上自洽：

1. **无 section FTS，无段落级关键词匹配，无 section recall channel**：段落级匹配只走 section Vec（语义，返回时执行），不参与召回，也不引入段落级关键词匹配。向量是分段发布的硬前提；查询时 Vec 门禁关闭则回退全文，不提供段落定位。这意味着一条长记忆能否被搜到完全依赖 memory 级 FTS/Vec。memory FTS 覆盖完整正文，但单个 memory Vec 受模型 token 上限约束，只覆盖 `subject + content` 的 token-safe 前缀；query 只与后部章节语义相关且没有字面词重合时，可能无法进入候选 pool。这是用“召回路径不新增通道”换实现简单和低回归风险的取舍。

   **风险分级**：
   - **中等长度（4000-8000 字）**：风险**中低**。memory Vec 可能已经发生 token 截断，但 subject 和文档前部通常仍提供足够主题信号；FTS 覆盖完整正文。
   - **超长文档（50000+ 字）**：风险**较高**。memory Vec 明确只覆盖 token-safe 前缀；若 query 只和后部章节语义相关且原文没有同义字面词，memory Vec/FTS 都可能漏召回。

   **后续版本缓解方案**：如果实测确认超长文档召回不足，可以追加 section Vec 作为召回通道（不改现有 5 通道，只在 `_wide_recall` 末尾追加 Channel 6）。这个改动不影响现有架构——section Vec 表和 embedding 已经存在，只是从"返回时匹配"升级为"也参与召回"。当前版本不做，先用真实数据验证风险是否真实存在。

2. **memory 级打分仍使用完整正文的字面特征**：一条长记忆即使分了段，`_soft_rerank` 的 FTS/LIKE/anchor/long-content 相关特征仍基于完整正文；memory Vec 则使用 token-safe 输入。分段只影响返回内容，不改变候选 memory 的打分规则。如果未来发现 `long_content_penalty` 对分段记忆造成实际问题，再基于真实数据决定是否豁免，本版本不预先修改。

3. **`get_sections` 暂不做跨 memory 批量接口**：v0.6 先保持 `get_sections(memory_id, section_ids)`，因为零命中/部分命中的 catalog 已返回 summary 和截断覆盖信息，Agent 可先判断是否需要取正文。若遥测显示一次 search 后经常对多条 memory 各发一次 `get_sections`，再追加 `get_sections_batch(items=[...])`；这是延迟优化，不阻塞正确性首版。

## 11. 文档与配置交付物（实现时必须同步更新）

分段功能引入了新的配置项、MCP 工具和用户行为。以下三个文件**必须在实现 PR 中同步更新**，不能只改代码不改文档。实现者容易只关注 db.py/tools.py 的逻辑变更而遗漏这些交付物，此处显式列出以防遗漏。

### 11.1 `memory_arbiter/config.py`

`Settings` dataclass 必须新增以下字段，并在 `from_env()` 中按 config.json > 环境变量 > 默认值解析：

| 字段 | 类型 | 默认值 | 环境变量 | config.json key |
|------|------|--------|----------|-----------------|
| `split_enabled` | bool | `False` | `MEMORY_ARBITER_SPLIT_ENABLED` | `split.enabled` |
| `split_threshold` | int | `4000` | `MEMORY_ARBITER_SPLIT_THRESHOLD` | `split.threshold` |
| `section_vec_distance_threshold` | float | `0.7` | `MEMORY_ARBITER_SECTION_VEC_DISTANCE_THRESHOLD` | `split.section_vec_distance_threshold` |
| `section_fulltext_threshold` | float | `0.8` | `MEMORY_ARBITER_SECTION_FULLTEXT_THRESHOLD` | `split.section_fulltext_threshold` |
| `max_sections` | int | `50` | `MEMORY_ARBITER_MAX_SECTIONS` | `split.max_sections` |
| `max_section_chars` | int | `3600` | `MEMORY_ARBITER_MAX_SECTION_CHARS` | `split.max_section_chars` |

`from_env()` 必须新增 `split` 配置段解析（`cfg.get("split")`），与现有 `vec_cfg`/`emb_cfg` 模式一致。float 类型需新增 `parse_float` 函数和 `pick_float_field` 内部 helper（当前只有 `parse_int`）。`llm_batch_chars` **不做全局 Settings**，它是 `memory_split` 的调用参数，不属于服务端配置。

### 11.2 `examples/memory-arbiter.config.example.json`

必须新增 `split` 配置段，含 `_readme` 注释说明各项含义、推荐值和前置条件。分段默认关闭，示例中 `"enabled": false`。用户按需开启时修改为 `true` 并确认 sqlite-vec + GGUF embedding 已就绪。

### 11.3 `README.md`

必须新增分段功能章节（中英双语，与现有文档格式一致），至少包含：

1. **功能概述**：解决长文（>4000 字符）检索时"定位到具体段落"困难的问题；适用场景（知识库文档、API 文档、项目计划）
2. **前置条件**：sqlite-vec 扩展已安装、托管 GGUF embedding 已配置且 `_vec_index_meta.state=ready`、外部 LLM 可用
3. **配置方法**：config.json `split` 段完整示例 + 环境变量对照表；明确标注默认关闭、distance 阈值需校准
4. **新增 MCP 工具**：`memory_split`、`get_sections`、`memory_split_status`、`memory_rebuild_embeddings` 的签名、参数和用途
5. **`memory_search` 返回结构变化**：命中分段记忆时的 `content_omitted`、`matched_sections`、`section_catalog`、`section_enhancement_applied` 字段说明；Vec 门禁关闭时返回全文的行为
6. **典型使用流程**：`memory_write` → `split_hint` → `memory_split`（prepare → LLM 分批 → publish）→ `memory_search` 返回 `matched_sections`
7. **注意事项**：默认关闭的原因、distance 阈值校准是发布门、`memory_edit` 改 content 后需重新分段、模型切换后需 `memory_rebuild_embeddings`

同时更新现有的 MCP 工具表和配置项参考表，把新增工具和配置项加入。

### 11.4 PR 粒度

文档和配置更新可以与对应功能 PR 合并提交（如 config.py 在 PR 2、README 工具表在 PR 5/6/7），也可以在 PR 8（测试与校准）中统一补充。**但不允许在功能 PR 合并后仍遗漏这些文件更新。** 建议在每个功能 PR 的 checklist 中加入"已更新 README 对应章节"和"已更新 config.py / example config"的检查项。
