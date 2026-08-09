# mema 架构整理与重构可行性方案

> 版本：v0.12.3 之后 / 预 v0.12.4
> 日期：2026-08-09
> 目标：交付一个维护性与可读性都处于顶级的代码库
> 硬约束：**两轨执行**。架构拆分轨仍要求零行为漂移（纯移动 + 委托 + 兼容 facade）；已确认的原始缺陷进入显式 hardening 轨，每个行为变更必须有独立测试、独立提交与 CHANGELOG 说明，禁止混在“纯移动”提交里。已验证必须保留的怪癖列在 §6.3；已确认原始缺陷与修复边界列在 §9.6。

---

## 0. TL;DR

本方案不再只是“把大文件拆小”的维护性重构，也包含一组已确认原始缺陷的安全/一致性加固。执行时必须把“纯重构不漂移”和“hardening 有意变更”分开提交、分开测试、分开审查。

mema 的**模块边界设计是合理的**——models / config / db / search / tools / server 的分层方向正确，store 组合模式（`StructuredClaimStore`、`ConflictJudgmentStore`）也是正确先例。问题不在架构方向，而在**架构纪律没有跟上功能增长**：

1. 三个核心文件严重超限：`tools.py` 4871 行（120 个方法）、`db.py` 3813 行（98 个方法）、`doctor.py` 1880 行（31 个检查）。`MemoryDB` 和 `MemoryTools` 都已退化为 God Object。
2. 一批真正的领域逻辑（分章管线、冲突信号、产品面调度、语义冲突 worker、sections CRUD、workspace 别名治理）被埋在巨型类内部，无法独立理解、独立测试、独立复用。
3. 底层小方法存在系统性重复：CJK/token 文本工具 3 处、ISO 时间解析 4 处、settings 防御性 `getattr` 50+ 处、隔离字符串字面量散落 6 个文件、conflict source/status 字面量贯穿 tools+search+db。
4. 硬编码契约常量在注释里反复声明"single source of truth"（`_NO_DIRECT_MATCH_PREFIX`），恰恰说明它们需要一个真正的家。（注：方案初稿另列 `_RECENT_FALLBACK_WARNING_PREFIX`，经核查该常量**在代码中不存在**，系误记，已删除。）

**方案：两条轨道交付 v0.12.4。第一条是 6 个阶段、约 17 天的架构拆分轨，使用纯代码移动 + 委托模式，要求行为不漂移；第二条是 hardening 轨，把本轮对抗性 review 发现的授权、隔离、事务、配置容错、输入规范化、并发生命周期缺陷作为显式行为变更逐项修复。阶段仅为提交节奏，全程只发一个补丁版本。**

---

## 1. 现状盘点

### 1.1 文件规模（包内 18210 行 + 测试 12612 行）

| 文件 | 行数 | 顶层结构 | 状态 |
|---|---|---|---|
| `tools.py` | **4871** | 3 个类（SplitReindexWorker / SemanticConflictWorker / MemoryTools），MemoryTools 含 38 个 `memory_*` 端点 + ~60 个私有方法 | 🔴 必须拆 |
| `db.py` | **3813** | `MemoryDB` 98 个方法，尾部挂着模块级文本工具函数 | 🔴 必须拆 |
| `doctor.py` | **1880** | 31 个 `_check_*` 健康检查 + 报告组装 | 🔴 必须拆 |
| `search.py` | **1498** | 单入口 `search_memories` + 召回/重排/过滤，模块内部已清晰 | 🟡 仅配合 text.py 抽离微调 |
| `conflict_judgments.py` | 718 | 单 store，内聚 | 🟢 保留 |
| `claims_db.py` / `claims.py` / `arbitration.py` / `semantic_conflict.py` | 286–530 | 单职责 store/纯函数 | 🟢 保留，是本方案分层正确性的证据 |
| `tests/test_memory_arbiter.py` | **4091**（183 个测试） | 单文件 | 🔴 按域拆分 |

### 1.2 分层依赖现状

```
server.py  → tools.py ──┬─→ db.py ──┬─→ claims_db.py / conflict_judgments.py
                        │           ├─→ config.py / models.py / degrade.py
                        ├─→ search.py ─→ db.py / anchors.py
                        ├─→ semantic_conflict.py（纯函数 + GGUF backend）
                        ├─→ embedder.py / workspace_rules.py / arbitration.py
                        └─→ update_monitor.py
console_api.py → tools.py；doctor.py → config/degrade（独立旁路）
```

**依赖方向总体正确**：store 不反向依赖 tools，search 不依赖 tools。三个已识别的坏味道：

- **跨层导入私有函数**：`tools.py` 从 `db.py` 导入 `_canon_entity`/`_canon_scope`，从 `search.py` 导入 `_linked_open_items_for_search`；`search.py` 从 `db.py` 导入 `row_to_dict`；`tools.py` 内部以 `MemoryDB._fetch_memory(conn, mid)` 这种"类名.私有方法"方式跨实例调用 db 私有静态方法（`_attach_sections` 内，tools.py:3900 前后）。
- **测试直 import 私有名**：`_soft_rerank`、`_sanitize_fts_query`、`_coerce_tags`、`_normalize_alias_key`、`_coerce_tags_db`、`_passes_filters` 等 12+ 个私有符号被测试直接锁定。
- **反向渗透**：`claims_db.py`/`conflict_judgments.py` 持有 `MemoryDB` 并触达其私有成员（`db._db_available`、`db._fetch_memory`，合计 28 处回指）——这是组合式 store 的既定模式，重构时**保持不动**，只把更多"暂时挂在 MemoryDB 上的领域方法"按同样模式收编为 store。

### 1.3 配置体系双轨（本次最确定的"硬编码未抽离"证据）

- `config.py` 的 `Settings.from_env()` 已经实现了"配置文件 + 环境变量 + 默认值"三级合并，并附带 `pick_str/pick_int/pick_bool/clamp_*` 校验器。
- 但 `search.py:124-137` 的 `_get_ranking_mode()` 绕开 Settings，直接 `os.environ.get("MEMORY_ARBITER_RANKING_MODE")`，且用裸字符串 `"bm25"/"hybrid"`。
- `tools.py` 中 `getattr(self.settings, "...", 默认值)` 出现 **40 处**（doctor.py 8 处、search.py 2 处、console_api.py 2 处）——对 `@dataclass` 强类型字段用 `getattr` 兜底，等于声明"我不信任 Settings 是完整的"。默认值因此同时存在于 `config.py` 字段定义和所有调用点，是两份真相。

### 1.4 领域语义靠裸字符串传递

- **isolation 等级**：`"none"/"weak"/"strict"` 字面量贯穿 `tools.py`（写、搜、语义 worker 三处独立重读并重实现判定）、`search.py`（9 处 `isolation == "strict" and ws_canonical` 重复表达式）、`doctor.py`、`console_api.py`。没有任何 Enum/常量定义点，全靠每个调用点拼对字符串。
- **conflict source/status**：`"open_table"`、`"structured_claim_candidate"`、`"metadata_write_hint"`、`"runtime_metadata_hint"`、`"metadata_overlap"` 在 `tools.py` 的信号组装、`db.py` 的记录、`search.py` 的透传中反复出现，靠人脑保持拼写一致。
- **响应包络**：`self.db.state.response(...)` 在 `tools.py` 有 **139 处**返回点，错误字典的形状（`{"error": ..., "help": ...}`）靠约定而非类型。

### 1.5 重复实现的底层方法

| 功能 | 现状位置 | 问题 |
|---|---|---|
| ISO 时间解析 | `search._parse_time`、`search._parse_ingest_time`、`arbitration.parse_time`、`update_monitor._parse_time` | 4 份，语义略有差异但无人说得清差异是否故意 |
| CJK 判定/切分 | `search._is_cjk_token/_split_cjk_token/_is_pure_cjk_token/_cjk_substring_match/_normalize_token_for_tag_match`、`anchors._is_cjk_char/_cjk_bigrams`、`db._subject_tokens/_CJK_CHAR_RE`（且 db.py 在第 3789 行出现第二次 `import re as _re`） | 3 处实现，正则范围各自为政 |
| subject token 重叠度 | `db.find_metadata_overlap_candidates`（用 `_subject_tokens`）与 `tools._write_duplicate_hints:1877-1893`（手写 `.lower().split()` 集合重叠） | 同一启发式两份实现，阈值可能漂移 |
| tags 归一化 | `search._coerce_tags` 与 `db._coerce_tags_db` | 注释自认是镜像（"Mirrors search._coerce_tags"） |
| 文本/证据工具 | `claims.py` 的 `canon_token/canon_scope/_evidence`、`db.py` 尾部的 `_canon_entity/_canon_scope` | canon_scope 在两处各定义一次 |

### 1.6 注释债务

`tools.py` 含 27 处 `design §/r4 §/636 §` 引用，`search.py` 27 处，`doctor.py` 24 处。这些外部设计文档（部分在 `BillyProject/docs`，不入库）是唯一的上下文来源——链接失效风险高。同时 `# v0.3.0:`、`# v0.7.4.1:` 这类版本墓碑注释遍布，CHANGEOG 已有记录，代码内的是噪音。

---

## 2. 架构判断：现有结构是否合理

**结论：骨架合理，无需推翻；需要做的是"把已证明正确的模式贯彻到底"。**

合理的部分：
- **分层方向**：entry（server/CLI/console）→ orchestration（tools）→ domain（search/claims/arbitration/semantic）→ persistence（db+stores）→ infrastructure（config/models/degrade/embedder）。没有环，没有下层依赖上层。
- **store 组合模式**：`StructuredClaimStore`/`ConflictJudgmentStore` 证明了"MemoryDB 只做连接/事务权威，领域 store 持有 `MemoryDB` 句柄"是可行的——这正是 db.py 其余部分的拆分模板。
- **不可变枚举骨架**：`models.py` 的 `MemoryStatus/SourceType/ProtectionLevel` 已就位，只是没用透。

不合理的部分（都需要在本方案内解决）：
1. **God Object × 2**：`MemoryTools` 把 MCP 端点、产品面路由、写管线编排、读管线编排、分章管线、冲突信号、语义 worker 管理、help 文案共 8 种职责塞进一个类；`MemoryDB` 把连接工厂、schema 迁移、FTS/vec 探测、向量 KNN、memory CRUD、冲突表读写、workspace 别名治理、audit、history、sections 存储共 10 种职责塞进一个类。
2. **隐式契约大于显式类型**：检索/判定/信号语义靠字符串字面量和注释传递（§1.4），修改一处需要全局搜索拼写。
3. **领域逻辑没有独立家**：分章（split）管线 ~500 行、冲突信号 ~330 行、产品面调度 ~470 行——全是高内聚逻辑，却都长在 `MemoryTools` 类体里，没法单独阅读或测试。
4. **底层方法无共享库**：文本/时间工具四处重复（§1.5），且因为没有公共位置，新功能倾向于在最近的文件里再写一份。

---

## 3. 目标架构

```
memory_arbiter/
├── __init__.py
├── text.py              # 新增：CJK/token/subject 工具（唯一实现）
├── timeutil.py          # 新增：parse_iso8601 / utc_now_iso（唯一实现）
├── constants.py         # 新增：Isolation 等共享常量（若不用 StrEnum 则常量模块）
├── models.py            # 增补：IsolationLevel / ConflictSource / ConflictStatus / RetrievalMode 收敛于此
├── config.py            # 增补：Settings 补齐 ranking_mode 等遗漏字段
├── db/
│   ├── __init__.py      # MemoryDB  facade，re-export 全量旧符号（含私有别名）
│   ├── core.py          # 连接工厂、事务、诊断连接、degrade 接线
│   ├── schema.py        # DDL、迁移、FTS/vec 探测与建表
│   ├── vectors.py       # vec/section-vec KNN、embedding 存取、orphan/resync
│   ├── workspaces.py    # 别名治理：resolve/upsert/rename/migrate/events
│   ├── memories.py      # memory CRUD、filter 查询、history、edit
│   ├── sections_store.py# sections CRUD（现 db.py 尾部 _insert/_get/_delete 系列）
│   ├── audit.py         # audit_summary、attention/scan 日志
│   └── conflicts.py     # conflict 行读写、dismiss 判定（claims_db/judgments 仍为独立模块）
├── search/
│   ├── __init__.py      # search_memories + SearchOutcome，re-export 旧私有符号
│   ├── recall.py        # 六通道 _wide_recall、bm25 路径、recent_fallback
│   ├── rerank.py        # _soft_rerank 及全部打分常量
│   ├── filters.py       # _passes_filters、tags/time 过滤、linked_open_items
│   └── fts.py           # FTS 查询构造（_sanitize_fts_query 系列）
├── pipeline/
│   ├── __init__.py
│   ├── write.py         # WritePipeline：remember/update 的后写编排
│   ├── signals.py       # 冲突信号组装（_attach_conflict_signals/_build_open_table_signal/_compute_runtime_hint）
│   └── sections.py      # SectionPipeline：_attach_sections/_rule_plan_sections/_publish_sections/_after_write_split/_compute_offsets
├── workers.py           # SplitReindexWorker + SemanticConflictWorker（纯移动）
├── surfaces.py          # ProductSurfaces：memory/memory_review/memory_govern/memory_repair 路由、help、参数守卫
├── claims.py / claims_db.py / conflict_judgments.py / semantic_conflict.py
├── arbitration.py / anchors.py / workspace_rules.py / embedder.py / degrade.py
├── update_monitor.py / config_registry.py
├── doctor.py            # 瘦壳：Finding/Severity/编排/CLI 绑定
├── doctor_checks/       # __init__ 含 CHECKS 注册表 + 每个维度一个模块
├── tools.py             # 瘦壳：MemoryTools 组合上述服务，仅保留编排 + 兼容委托
├── server.py / setup_cli.py / doctor_cli.py
└── console_api.py / console_server.py / console_cli.py / console_static.py
```

**关键不变量**：`memory_arbiter.db`、`memory_arbiter.tools`、`memory_arbiter.search`、`memory_arbiter.doctor` 四个模块路径下的所有现存公开与私有符号，在目标架构中**全部仍可 import**（经 facade `__init__.py` re-export）。外部使用者和存量测试零迁移成本。

---

## 4. 拆分设计

### 4.1 `db.py`（3813 行 → core ~350 + 7 个领域 store）

**模式**：沿用已验证的 store 组合。`MemoryDB` 保留为唯一连接/事务权威 + facade；领域方法按内聚性搬入持有 `MemoryDB` 句柄的 store 类（`__init__(self, db)`），facade 上保留同名委托方法。

分组（按现方法清单，行数为近似值）：

| store | 收录方法 | 规模 |
|---|---|---|
| `core.py` (MemoryDB 本体) | 连接三件套、`db_available`、`_init_database`、`_init_vec_state` 的 db 侧 | ~350 |
| `schema.py` | `_init_schema`、`_migrate_v090_claims`、`_migrate_add_column`、`_probe_*`、`_rebuild_fts`、`_ensure_fts/vec/section_vec/workspace_vec`、`_migrate_vec_parent_status` | ~600 |
| `vectors.py` | `store/get/delete_embedding`、`vec_knn`、`section_vec_knn`、`mark/delete/purge/resync` 向量、`get/init_vec_index_state` | ~700 |
| `workspaces.py` | `resolve_workspace_canonical`、`upsert/get/list/rename/migrate` 别名系列、`set_memory_workspace_canonical` | ~600 |
| `memories.py` | `insert/get/update/list/edit_memory`、`_filter_clauses`、`recall_by_filters`、`count_filtered_memories`、`history` 三件套 | ~600 |
| `conflicts.py` | `record_conflict(_enriched)`、`resolve_*`、`list_conflicts`、`is_pair_dismissed`、`is_structured_pair_closed_for_snapshot` | ~400 |
| `audit.py` | `audit_summary`、`log_attention`、scan log、`get_memory_summaries` | ~200 |
| `sections_store.py` | `_insert_section`、`_store_section_vec`、`_delete/_get_sections*`、`get_sections_by_*`、`section_vec_distance_match` | ~250 |

**去委托化**：`publish_memory_claims` 起的 26 个一行转发方法（db.py:2660-2830）删除；`MemoryDB.__init__` 中已有 `self._claim_store`/`self._judgment_store`，调用点改为显式 `db.claims.publish_memory_claims(...)` / `db.judgments.submit_conflict_judgment(...)`（`self._claim_store` 提升为公开只读属性 `claims`/`judgments`，保留旧私有别名兼容）。跨层 import 的 `_canon_entity/_canon_scope` 移入 `text.py`，`db/__init__.py` re-export 旧路径。

### 4.2 `tools.py`（4871 行 → facade ~1200 + 4 个新模块）

| 新模块 | 内容（现位置） | 规模 |
|---|---|---|
| `workers.py` | `SplitReindexWorker`（tools.py:45-139）、`SemanticConflictWorker`（140-292）——两个类自包含，只回调 tools 公开方法 | ~300 |
| `surfaces.py` | 4 个产品面路由 `memory/memory_review/memory_govern/memory_repair`（600-863）、`_product_help` 文案（420-486）、`_judge_constraints`、`_forward/_alias_id/_int_product_arg/_require_id/_coerce_product_id/_is_truthy/_require_ws_strings` 守卫 | ~470 |
| `pipeline/write.py` | `memory_write` 编排体（1158-1348）、`_enrich_write_response`（1350-1445）、`_apply_structured_gate`、`_index_and_reconcile_claims(_impl)`（1479-1858）、`_write_duplicate_hints`（1859-1914）、`memory_edit` 的写后部分 | ~900 |
| `pipeline/signals.py` | `_trust_score/_confidence_rank`、`_attach_conflict_signals`（3583-3648）、`_build_open_table_signal`（3649-3748）、`_compute_runtime_hint`（3749-3824） | ~330 |
| `pipeline/sections.py` | `_catalog_entry`、`_attach_sections`（3837-3969）、`_find_nth_occurrence/_compute_offsets`（4022-4114）、`_rule_plan_sections`（4115-4218）、`_split_snapshot_error`、`_publish_sections`（4270-4422）、`_after_write_split`（4423-4510）、`memory_split` 编排（4511-4644）、`_mark_split_failed`、`memory_rebuild_embeddings` | ~900 |
| `tools.py` 剩余 | `MemoryTools`：`__init__` 组合、embedder/semantic 生命周期（`_ensure_embedder/_ensure_semantic_backend/_semantic_*`）、各 `memory_*` 端点的薄编排 + 对 pipeline/surfaces 的委托 | ~1200 |

**拆分类的构造协议**：pipeline/surfaces 服务类持有 `tools` 句柄（与 worker 现行模式一致：`SplitReindexWorker(self)`），facade 保留全部原方法名做薄委托——`MemoryTools().memory_write(...)`、`_attach_sections`、`_after_write_split` 等签名与返回逐字节不变。

### 4.3 `doctor.py`（1880 行 → 壳 ~400 + checks 包）

31 个 `_check_*` 按维度分 6 组：

| doctor_checks 模块 | 检查项 |
|---|---|
| `config_env.py` | `_check_config_warnings`、`_check_db_writable`、`_check_degradation_mode`、`_check_attention_volume` |
| `vector.py` | `_check_vector_chain`、`_check_vec_index_state`、`_check_orphan_vectors`、`_check_vec_parent_status_sync`、`_check_section_vec_coverage` |
| `semantic.py` | `_check_semantic_chain`、`_shallow_gguf_probe`、`_embedder_shallow_probe` |
| `split.py` | `_check_split_capability/_backlog/_failed/_legacy_declined/_legacy_unknown/_index_integrity` |
| `integrity.py` | `_check_structured_claims`、`_check_history_version_chain`、`_check_orphan_sections`、`_check_history_bloat`、`_check_db_size`、`_check_superseded_ratio` |
| `conflicts.py` | `_check_conflicts_open`、`_open_conflicts_by_source`、`_check_workspace_alias_health` |

壳保留 `Finding/Severity/OverviewReport`、`_scalar/_table_exists/_max_severity/_na` 等共享辅助、`run_all_checks` 编排、CLI/MCP 绑定与 `report_to_dict`。检查函数签名统一为 `(conn, settings, runtime_state) -> Finding|list[Finding]|None` 并收进 `CHECKS` 注册表，`run_all_checks` 改为数据驱动遍历——新增检查从"改两处"变成"加一个文件 + 一行注册"。

### 4.4 `search.py`（1498 行 → search/ 包 4 模块）

模块内部已经是"fts 构造 → 召回 → 重排 → 过滤 → 入口"的清晰流水线，按此切分即可。`_get_ranking_mode` 读取的 ranking mode 并入 Settings 链路（Settings 集中读环境变量），`_RANKING_MODES` 收敛为常量。**注意：ranking mode 存在运行时中途切换的测试/调参路径**（见审查修订 R1），`Settings.ranking_mode` 若设为普通字段会在 from_env 时被缓存，行为漂移。方案：`search` 侧保留对 `MEMORY_ARBITER_RANKING_MODE` 的**运行时直读**，只把"环境变量名 + 合法值 bm25/hybrid + 默认 hybrid"收敛为常量，不改为 Settings 缓存字段。`search/__init__.py` re-export 全部现有符号（含测试依赖的 `_soft_rerank` 等私有名）。

### 4.5 测试拆分（4091 行 → tests/ 目录按域分组）

`test_memory_arbiter.py` 的 183 个测试按域归入：`test_write_pipeline.py`、`test_search_recall.py`、`test_arbitration.py`、`test_supersede_confirm.py`、`test_edit_history.py`、`test_audit.py`、`test_vec_embedding.py`。版本化测试文件（`test_v074_*` 等）保留文件名不动。提取 `tests/conftest.py` 共享 fixture（临时 db、MemoryTools 工厂）——目前各文件各自 new 实例。

---

## 5. 抽象与去硬编码方案

### 5.1 `text.py`（唯一文本工具库）

收编：`anchors._is_cjk_char/_cjk_bigrams`、`search._is_cjk_token/_split_cjk_token/_is_pure_cjk_token/_cjk_substring_match/_normalize_token_for_tag_match`、`db._subject_tokens/_CJK_CHAR_RE`、`claims.canon_token/canon_scope`、`db._canon_entity/_canon_scope`、`search._coerce_tags`+`db._coerce_tags_db`（合并为一个 `_coerce_tags`）。原位置全部留 re-export 别名。CJK 正则统一为一个常量（现三处范围不同，以覆盖面最广的 `search._CJK_RE` 为准，其余两处的行为差异见 §6.3 先验证再合并——若测试证明差异是有意的，则保留两个具名常量并注释原因）。

### 5.2 `timeutil.py`

收编 4 处 ISO 解析为 `parse_iso8601(value) -> Optional[datetime]`（naive→UTC、容错返回 None）+ `utc_now_iso()`（从 models 迁入并 re-export）。`arbitration.parse_time` 是**非 Optional 签名**（空值/解析失败返回 `datetime.min(UTC)`，用于仲裁排序兜底，永不返回 None），不得合并进 Optional 版本——保留为薄包装并注明差异。另三处（`search._parse_time`/`search._parse_ingest_time`/`update_monitor._parse_time`）语义接近但容错粒度不同，收编前逐调用点核对返回 None 与抛/默认值的处理分支。

### 5.3 枚举与常量收敛（`models.py` / `constants.py`）

| 收敛项 | 现状散落 | 目标 |
|---|---|---|
| `IsolationLevel`（none/weak/strict）+ `isolation_active(level)` | tools ×3、search ×9、doctor、console_api | `models.py` StrEnum；search 的 9 处 `isolation == "strict" and ws_canonical` 折叠为 `strict_ws(isolation, ws_canonical)` 一处实现 |
| `ConflictSource`（open_table/structured_claim_candidate/metadata_write_hint/runtime_metadata_hint/metadata_overlap）、`ConflictStatus` | tools 信号组装、db 记录、search 透传 | `models.py` StrEnum |
| `RetrievalMode` | search.py Literal 别名 | 迁入 models（StrEnum），保持 JSON 序列化兼容 |
| `_NO_DIRECT_MATCH_PREFIX` | search.py（注释自述 single source of truth） | 移至 `constants.py`，bm25 嗅探逻辑不变（`_RECENT_FALLBACK_WARNING_PREFIX` 经核查不存在，已删） |
| 响应包络 `{"ok","mode","warnings","degraded","data"}` | 139 处散落调用 | 保持 `DegradeState.response` 唯一出口（不动），新增类型化 `ToolResponse` 仅做文档/校验，不改运行行为 |

### 5.4 Settings 单一真相

- `Settings` 读取链路保持集中（from_env 统一读环境变量是本文件做得好的部分，不动）；ranking mode 按 §4.4 修订**例外处理**（保留运行时直读）。
- **消除 50+ 处 `getattr(settings, ..., default)`**：所有字段在 dataclass 上已有默认值，直接属性访问。这是纯机械替换，逐文件进行。**但须先完成审查修订 R5 的归因**——确认这些 getattr 兜底的默认值与 dataclass 字段默认值逐一相等（不相等的不是冗余而是行为差异，照抄现状不得"纠正"），并确认不存在依赖属性缺失的 try/except AttributeError 防御路径。
- 保留 `config.py` 现有的 `pick_*/clamp_*` 校验链（这部分写得是好的）。

### 5.5 底座可选增强（建议但非必须）

`db/core.py` 增加两个 contextmanager helper，消除"availability 检查 + try/sqlite3.Error/返回降级值"的重复样板（db.py 内有 ~30 处同构方法）：

```python
def _read_one(self, sql, params, default=None): ...      # 自带可用性检查与异常降级
def _read_all(self, sql, params) -> list[dict]: ...
```

只在新 store 内部使用，不改既有方法的可见行为。此项放 P2，若与零行为变更承诺冲突则放弃。

---

## 6. 兼容性与已识别陷阱

### 6.1 兼容矩阵

| 消费方 | 依赖 | 保障 |
|---|---|---|
| MCP 客户端/存量 agent | 4 个产品面 + 全部 legacy `memory_*` 工具的 payload 契约 | server.py 注册签名不动；MemoryTools 方法名与返回字典形状不动 |
| tests/（含 12+ 私有符号直 import） | `from memory_arbiter.X import _y` | 各 facade `__init__.py` re-export；tools.py 顶部保留 `from .db import _canon_entity, _canon_scope` 等兼容 import |
| console_api / doctor_cli / setup_cli | tools/doctor 的公开类 | 构造签名不变 |
| 已安装旧版升级的存量数据库 | schema 与迁移 | `_init_schema`/迁移函数纯移动，DDL 一字不改 |

### 6.2 技术原则

1. **纯移动 + 委托**：任何函数在搬运时不改函数体（除 import 与接收方 `self.` → `self._db.` 的机械改写）。逻辑修复留给后续版本。
2. **小步快跑**：每个 phase 内部按"一个 store / 一个 pipeline 模块 / 一组检查"为单位提交，每步跑全量测试。
3. **类型护栏先行**：阶段 0 先跑通 `mypy --strict`（或 pyright）基线，重构期间用类型错误兜底跨文件搬运的遗漏。
4. **测试不变原则**：除 import 路径外不改测试；若某测试在纯移动后失败，视为重构引入的回归，修重构代码而非测试。

### 6.3 已验证必须保留的怪癖（不得"顺手修复"）

- `_sanitize_fts_query` 的 CJK 三元组 OR 逻辑（2 字 CJK 必须走 LIKE 兜底）。
- bm25 模式通过嗅探 `_NO_DIRECT_MATCH_PREFIX` 警告前缀推断 retrieval_mode 的耦合。
- `MemoryRecord.from_input` 对空 workspace 填 "default"（写路径严格判空必须看 raw payload）。
- `settings.policy` 的 client/agent 双键及 `_allowed` 告警形状。
- `_is_truthy` 的 allow-list 语义（字符串 "false" 不得为真——安全相关）。
- Channel 6 候选 `content=""` 依赖 `_attach_sections` 回填全文。
- strict 模式"新 workspace 写为 pending"的副作用顺序（subject 校验必须先于 workspace 注册，tools.py:1163-1173 的注释明确要求）。
- `resolve_workspace_canonical` 在 isolation=none 时完全跳过（不动写路径 embedder 不变量）。
- CJK 正则三处范围差异（§5.1）：先加特性测试锁定现状，再决定是否合并。

---

## 7. 实施计划（两轨：重构 + hardening）

### 7.1 架构拆分轨

| # | 阶段 | 内容 | 风险 | 工作量 | 验收 |
|---|---|---|---|---|---|
| -1 | 行为矩阵与缺陷 characterization | 建立授权矩阵、isolation 矩阵、事务边界矩阵；为 §6.3 怪癖与 §9.6 原始缺陷补 characterization tests；生成旧符号/签名/`__module__`/monkeypatch 清单 | 中 | 1.5 天 | 基线绿；矩阵入库；缺陷测试能证明“现状是什么” |
| 0 | 安全网 | mypy/pyright 基线；全量测试绿基线；`git tag pre-refactor-baseline`；为 §6.3 怪癖补特性测试（若缺失） | 低 | 1 天 | 基线绿 + 类型基线文件入库 |
| 1 | 纯函数抽离 | `text.py`/`timeutil.py`/`constants.py`；models 增补枚举；全部调用点改 import；原位置 re-export | 低 | 2 天 | 全绿；`grep -c "import re as _re" db.py` == 0；CJK 差异按 R11 保留或有测试证明可合并 |
| 2 | Settings 收敛 | 消除 50+ 处 `getattr` 兜底；isolation 判定收敛为 helper；ranking mode 继续运行时直读（R1） | 低-中 | 1.5 天 | 全绿；`getattr` 兜底默认值归因清单为空或逐项说明 |
| 3 | db.py 拆分 | db/ 包 8 个文件逐 store 迁移 + facade 委托；调用点改 `db.claims.*`；不改变事务边界 | 中 | 4 天 | 全绿；db/__init__ re-export 清单与旧符号 diff 为空；AST 搬运证明通过 |
| 4 | tools.py 拆分 | workers/surfaces/pipeline 三包迁移；MemoryTools 瘦身为组合 + 委托；共享状态 owner/锁规则落文档 | 中-高 | 4 天 | 全绿；tools.py ≤1300 行；monkeypatch 清单真实生效 |
| 5 | doctor/search/tests + 文档 | doctor_checks 注册表化；search/ 包；test_memory_arbiter.py 按域拆分 + conftest；清理版本墓碑注释；README/CLAUDE.md 结构图更新 | 中 | 4 天 | 全绿；无文件 >1500 行（测试除外但单文件 ≤1200）；wheel/sdist import smoke 通过 |

### 7.2 Hardening 轨（显式行为变更）

Hardening 轨不得与“纯移动”提交混在一起。每个修复项需要：失败用例先行、行为变更说明、CHANGELOG 条目、回归测试。

| # | 修复项 | 内容 | 风险 | 验收 |
|---|---|---|---|---|
| H1 | 授权布尔规范化 | 收紧 `_is_truthy()` 为严格 allow-list：仅 `True`、`"true"/"1"/"yes"/"on"`、可选整数 `1` 为真；`2`/`-1`/`NaN`/`Infinity`/list/dict/null 均不得授权；所有 product/legacy 授权参数统一使用该语义 | 中 | negative tests 覆盖 confirm/activate/supersede/correct/edit 等路径；`authorized="false"`、`2`、`NaN` 均拒绝 |
| H2 | Policy 基础语义 | `default_enabled=false` 对未知 client/agent 生效；`client_defaults` key 归一化；policy 状态页与实际行为一致 | 中 | policy 单测覆盖 allow/deny/client_defaults/default_enabled 大小写 |
| H3 | Config 容错 | config 顶层合法 JSON 但非 object 时 warning + fallback，不 AttributeError 崩启动 | 低 | config `[]`/`"x"`/`1` 测试不崩 |
| H4 | Tags 输入规范化 | 写入入口把 tags 规范为 `list[str]`：缺失/null→`[]`；非 list 拒绝；list 内非 string 拒绝；string strip；空 tag 删除；去重保留顺序 | 中 | `tags="todo"`、`tags=[1,"x"]`、空/重复 tag 等 loose payload 测试覆盖写入、FTS、search tags |
| H5 | `memory_edit` TOCTOU | protection/status 检查下沉到 `db.edit_memory()` 事务内重检 | 中 | 并发/monkeypatch 测试证明事务内遇到 locked/user_confirmed/superseded 会拒绝 |
| H6 | workspace confirm 原子化 | `memory_confirm_pending_workspace()` 合并 alias/canonical/status 为单个 DB 事务 | 中-高 | failure injection 任一步失败全回滚，无半更新 |
| H7 | supersede 原子化 | `memory_supersede()` 合并 status/conflict/audit 为单个 DB 事务 | 中-高 | failure injection 任一步失败全回滚，conflict/audit/status 一致 |
| H8-a | Policy 扩展到 mutating tools | 建立 mutating/read-only/ambiguous 工具清单；所有会改 DB 或 runtime state 的 tool 走 policy gate（dry_run 仍验证不写） | 高 | policy-disabled client 无法调用 edit/supersede/govern/repair/store_embedding/semantic_control 等 mutating 路径；read-only 路径不误伤 |
| H8-b | Strict read ACL 决策/实现 | 若本次修 strict 读隔离，不改 payload，使用 `settings.workspace` 解析 canonical 后过滤 `memory_get`/review/history/conflict_detail/console detail；无 workspace context 的 console detail strict 下拒绝敏感内容 | 高 | strict 下跨 workspace read-by-id/review/console 被拒绝或隐藏；same-workspace 正常；无需新增工具参数 |
| H8-c | Policy parse fail 口径 | 显式配置了 policy 文件但解析失败时 fail-closed；未配置 policy 仍默认 allow；重复归一化 client key 视为 invalid policy | 中-高 | 损坏 policy 文件 + 显式 policy 路径时 mutating gate 关闭；未配置 policy 不影响默认可用性；冲突 key 有测试 |
| H8-d | Semantic lifecycle 口径 | 定义 unload/disable 与 in-flight inference 的语义；本次最低要求统一 backend lifecycle lock，保留 in-flight 推理 best-effort 完成，不实现取消 | 中 | semantic_control unload/disable 与 worker job 并发时状态一致，无未加锁读写；status 明确 best-effort |
| H9 | Metadata 输入规范化 | `metadata` 必须是 JSON object；缺失/null→`{}`；dict 浅拷贝；禁止 `dict(value)` 魔法转换 list/string | 中 | `metadata="x"`、`metadata=[]` 拒绝；`metadata={"a":1}` 正常；product/legacy 两路径覆盖 |
| H10 | Numeric 输入规范化 | `confidence` 等 numeric 字段必须是 finite number，范围按字段定义校验（confidence 建议 `[0.0,1.0]`），无效值结构化拒绝，不 silent clamp | 中 | `confidence="abc"`/`"nan"`/`"inf"`/`-1`/`2` 拒绝，合法边界 0/1 通过 |

### 7.3 v0.12.x 版本节奏（1–2 个 patch 收口）

用户确认：后续不要动不动跳大版本号，也不要把计划拆成 v0.12.4 到 v0.12.10 这种过碎节奏。v0.12.x 清理线按 **1–2 个 patch** 集中解决；除非公开 API 或 strict read ACL 被最终确认是产品语义大改，否则不预设跳 v0.13。

| 版本 | 主题 | 范围 |
|---|---|---|
| v0.12.4 | 结构收口 + 兼容层治理 + 入口 hardening | 当前已拆结构对齐文档；`db/__init__.py`、`search/__init__.py` 区分 public API 与 compatibility-only private exports；legacy import smoke；内部代码不再新增/依赖 compatibility re-export；行为测试改新路径、兼容测试专测旧路径；scripts/docs/examples 只修 import；H1/H2/H3/H4/H8-a/H8-c/H9/H10：authorized、config、tags、metadata、numeric、policy 基础语义与 mutating gate。**v0.12.4 保留兼容层作为迁移安全网，不删除旧 import/private re-export** |
| v0.12.5（仅必要时） | 一致性 + 隔离 + lifecycle hardening + 兼容层择干净 | **兼容层清理**：删除旧 import 依赖与 compatibility-only private re-export/private exports，把 v0.12.4 保留的兼容层择干净；**事务一致性**：H5 `memory_edit()` 事务内重检 protection/status，H6 `memory_confirm_pending_workspace()` 单事务原子化，H7 `memory_supersede()` status/conflict/audit 单事务原子化；**strict read ACL**：H8-b 覆盖 `memory_get`、review/history/conflict_detail、console detail 等 read-by-id/detail 路径，不新增 payload，使用 `settings.workspace` 作为 caller workspace context；**semantic lifecycle**：H8-d 统一 backend lifecycle lock，in-flight inference best-effort 完成，不实现取消。若 v0.12.4 review 体量可控，也可并入 v0.12.4；若 strict read ACL 影响过大，再作为例外单独评估 |

每个 patch 只承载清晰主题，并配独立 regression tests、migration note 与 release note。CHANGELOG 必须列出对应 patch 的行为变化；未完成的 H 项不得写成已修复。

---

## 8. 风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| 循环 import（db ↔ stores ↔ text） | 中 | 分层铁律：text/timeutil/models 不得 import 任何上层；stores 用 `TYPE_CHECKING` 引用 MemoryDB（现行模式已如此）；facade 最后组装 |
| facade 委托遗漏某个方法导致 AttributeError | 中 | 阶段 3/4 完成后跑"旧符号清单 diff"脚本：对重构前 `dir(MemoryDB)`/`dir(MemoryTools)` 快照逐一断言存在 |
| 私有符号 re-export 遗漏（测试 import） | 中 | 同上的符号快照含私有名；CI 加 import 冒烟 |
| 拆分时"顺手优化"引入行为漂移 | 高 | 纪律：函数体 diff 必须为空（除 import/self 改写）；PR 自审清单第一条 |
| doctor 检查注册表化改变执行顺序 | 低-中 | CHECKS 表按现行 `run_all_checks` 调用顺序排列，报告 dict 的 key 顺序由测试锁定 |
| 字符串字面量收敛为枚举后序列化差异 | 低 | 沿用仓库现有惯例 `class X(str, Enum)`（models.py 已用此型；不用 StrEnum，保持 3.11 一致与现状统一），JSON 输出与裸字符串逐字节一致 |
| 并行开发冲突（功能分支 vs 重构） | 中 | 冻结窗口：阶段 3–4 期间不接受功能 PR；或功能先行合入再启动 |
| hardening 与纯移动混在一个 diff | 高 | 提交/PR 分轨；AST diff 非空必须标为 hardening；纯移动提交不得改断言语义 |
| 安全修复破坏 loose 客户端 | 中-高 | release note + migration note；negative tests 明确拒绝 `authorized="false"`、非 string tags、非 object metadata、NaN confidence |
| 事务原子化引入 nested transaction | 高 | 禁止 outer transaction 调 public DB method 伪原子；必须抽出接收 `conn` 的 internal helper 或新增单个 DB method 在同一 `write_transaction()` 内完成 |
| strict read ACL 误伤 console/review | 高 | 先定义 workspace context；不改 payload 时使用 `settings.workspace`；无 workspace context 的 console strict 下拒绝敏感 detail 而非跨库泄漏 |
| policy fail-closed 造成可用性下降 | 中 | 仅显式配置 policy 文件但解析失败时 fail-closed；未配置 policy 仍默认 allow；重复归一化 key 视为 invalid policy 并测试 |

---

## 9. 验收标准（"顶级可维护性"的操作化定义）

1. **规模**：包内无文件 >1500 行；核心域文件（pipeline/stores）300–900 行；无类方法数 >40。
2. **唯一真相**：任一领域语义（隔离等级、冲突源、检索模式、文本工具、时间解析、配置默认值）全库仅一处定义，其余位置为 import 或委托；`grep` 隔离字符串字面量在 models.py 外出现次数 == 0（委托 helper 内部除外）。
3. **零私有跨层**：`from memory_arbiter.X import _y`（X≠facade）在包内为零；跨层调用私有方法（`MemoryDB._fetch_memory` 式）为零。
4. **类型**：mypy strict 无 error（允许现状已存在的显式 ignore 注释）。
5. **测试**：全部既有测试不改动断言逻辑即通过；总测试数不减少；覆盖率不下降；monkeypatch 目标可随实现位置迁移但必须证明旧 facade 路径仍可影响真实执行或测试目标已显式更新。
6. **行为**：MCP 工具面的集成测试（test_v080_protocol / test_v090_structured_claims）逐字节通过；`mema doctor` 输出 schema 不变；架构拆分轨不得改变 §6.3 怪癖与 §9.6 hardening 前的 characterization 结果。
7. **授权/隔离矩阵**：所有 mutating tools 的 policy gate / authorized gate 状态有测试覆盖；strict isolation 下 search/read/review/recent/console 的行为有矩阵测试，未进入 H8 决策前不得暗改。
8. **事务原子性**：H5/H6/H7 必须有 failure-injection 测试，证明失败时无 status/history/conflict/audit/FTS/alias 半更新。
9. **包兼容**：源码树、wheel、sdist 解包后三处 import smoke 均通过，覆盖 `memory_arbiter.db` / `memory_arbiter.search` 旧公开与私有符号。
10. **搬运证明**：阶段 3/4/5 的移动函数需提供 AST 归一化 diff（允许 import 与 `self` 接收方机械改写），非空 diff 必须标注为 hardening 或独立行为变更。
11. **Hardening negative tests**：`authorized="false"`/`2`/`NaN` 不得通过；`default_enabled=false` 对未知 client 生效；config `[]` 不崩；非 string tags、非 object metadata、NaN/越界 confidence 被结构化拒绝；edit 事务内遇到 protected/superseded 拒绝；supersede/confirm pending 中途失败全回滚。
12. **测试分层**：Phase -1 characterization tests 用于证明现状，可短期 xfail 或放在专门文件；hardening 合入时必须改写为长期 regression tests。main 不得长期保留无理由 xfail 或“旧漏洞存在”的正向断言。
13. **事务实现约束**：H5/H6/H7 原子化不得通过外层事务包 public DB method 实现；验收需检查核心 SQL 在同一 `write_transaction()`/同一 `conn` helper 内执行。

---

## 9.5 对抗性审查（2026-08-09）与修订

> 本节是对上文方案的独立对抗性审查结论。审查方法：把方案当作"即将执行的 diff"，逐条到代码中找反证。结果：**方案的方向、目标架构、兼容性策略经审查成立**；发现 3 处事实性错误（已直接修订进 §4.4/§5.2/§8）和 8 个方案未覆盖的盲区（如下，需在执行前吸收进对应阶段）。

### 9.5.1 已确认并直接修订的事实错误

- **R1（§4.4 ranking_mode）**：方案提议把 `_get_ranking_mode()` 并入 `Settings.ranking_mode` 缓存字段。但 `scripts/tune_tag_weights.py`、`benchmark_section_recall.py` 等调参工具以及测试存在**运行时切换 ranking mode 做 A/B** 的用法；Settings 在 `from_env()` 时一次性固化，改成缓存字段会让运行中切换失效 → 行为漂移。已修订为：保留运行时直读，仅收敛常量。
- **R2（§5.2 arbitration.parse_time）**：方案原描述"抛异常语义"**错误**。实际 `arbitration.parse_time` 空值/失败返回 `datetime.min(UTC)`（非 Optional），与另外三处 Optional 语义不同。已修订为不得合并、保留薄包装。
- **R3（§8 StrEnum 表述）**：仓库现状用 `class X(str, Enum)`（models.py 三处），无 StrEnum。方案原写"StrEnum"与现状惯例不一致且对 3.11 无必要。已修订为沿用 `str, Enum`。

### 9.5.2 审查确认的盲区（执行前必须吸收）

| # | 盲区 | 证据 | 影响 | 吸收到 |
|---|---|---|---|---|
| R4 | **测试 monkeypatch 锁定了私有方法的归属模块与对象**，与 facade 委托方案冲突 | `test_v080_protocol.py:296,321` patch `tools._publish_sections`；`test_memory_arbiter.py:1828,1857` patch `memory_arbiter.tools.search_memories`；`test_v090:766` patch `memory_arbiter.tools.extract_claims`；`test_v090:750,881` patch `tools.db.publish_memory_claims`；`test_doctor_unit.py:141,146` patch `doc._check_db_size`；`test_v076:598` patch `tools.db.is_pair_dismissed`；`test_update_monitor.py:248` patch `UpdateMonitor._load_state` | 阶段 3/4/5 把方法搬进 store/pipeline/doctor_checks 后，这些 patch 打到旧位置将**打不中真实现**，测试假绿或运行期 AttributeError。方案"测试零改动"承诺与"纯移动"在 monkeypatch 语义下不可兼得 | **阶段 0 新增**：先生成"monkeypatch 清单"（本表即起点），对每一处决定 (a) 保留 facade 委托使旧 patch 点仍生效，或 (b) 同步更新 patch 目标并在 PR 中显式标注。验收标准 §9.5"测试不改断言"需放宽为"不改断言逻辑，patch 目标可随实现位置迁移" |
| R5 | **`getattr(settings,…,default)` 未归因即承诺清零**，可能误删有意防御 | tools.py 40 处、doctor 8、console_api 2、search 2。已确认**不存在**"读了但未定义为字段"的属性（§对抗核查 2 的 comm 结果为空），但是否每处的兜底值都等于字段默认值、是否有 try/except AttributeError 依赖属性缺失，方案未验证 | 若某处 getattr 默认值 ≠ 字段默认值，机械替换为属性访问会改变行为；若为延迟初始化遗留防御，清零可能暴露 AttributeError | **阶段 2 前置**：逐处比对 getattr 兜底值与字段默认值，列出"值不等"清单单独评审（照抄不纠正）；全库 grep 确认无 `except AttributeError` 依赖 settings 属性缺失 |
| R6 | **"旧符号清单 diff"验收无法证明行为不变**，且委托会改变 `__module__`/`__qualname__` | 方案验收依赖 `dir()` 快照比对 + 测试全绿。`dir()` 只能证明名字存在，不能证明指向同一实现；method 搬到 store 后经 facade 委托，`__module__` 改变 | 若有任何基于 `__module__`/pickle/inspect 的隐式依赖（当前未发现，但未穷尽核查），委托会静默破坏 | **阶段 0 新增**：基线不仅快照 `dir()`，还快照 `inspect.signature` + 关键方法的 `__module__`；验收改为"签名 diff 为空 + `__module__` 变化白名单逐条确认" |
| R7 | **拆分类持有 `tools` 句柄的循环引用 + 共享可变状态（`_embedder`/`_split_worker`/`_semantic_worker`/`_embedder_warnings`）的线程归属**，方案未设计 | `tools.py:36` 处触及这些共享句柄；`SplitReindexWorker._process_one` 回调 `tools._publish_sections`，`SemanticConflictWorker` 回调 `tools._ensure_semantic_backend/_process_semantic_conflict_job`；embedder 有独立 `_embedder_lock`；`_embedder_warnings` 是 list 被多处 extend | pipeline/workers 拆出后与 MemoryTools 形成双向引用（现状 worker 已如此，但 pipeline 是新引入）；共享状态若由 facade 与 pipeline 双写，锁边界需在文档中明确，否则拆分本身引入竞态 | **阶段 4 前置**：明确"共享可变状态（embedder/worker 句柄）只归 MemoryTools 持有并加锁，pipeline 经 tools 公开方法访问，不得自持副本"的所有权规则；`_embedder_warnings` 的 extend 路径逐一确认在锁内或启动期串行 |
| R8 | **`db/` 与 `search/` 改为包后，`from memory_arbiter.db import X` 与 `from memory_arbiter import db` 两种用法、`import memory_arbiter.db as db` 及 `db.something` 属性访问必须全部经 `__init__.py` 重建**，方案只说 re-export 未覆盖子模块属性与 `pkgutil`/pickle 路径 | scripts/backfill_subjects.py 用 `from memory_arbiter.db import MemoryDB`；console_api 用 `self.tools.db._scan_log_last_completed()`（私有方法，实例级，非模块级） | 模块→包时，模块级私有函数（`db._normalize_alias_key`、尾部 `_subject_tokens` 等）若在 `__init__.py` 漏 re-export，`from memory_arbiter.db import _x` 立即 ImportError | **阶段 3/5 验收新增**：`__init__.py` 的 re-export 清单由"旧模块 `dir()` 全集 - 显式排除项"自动生成，而非手工列举；CI 加 `python -c "from memory_arbiter.db import *"` 与逐私有名 import 冒烟 |
| R9 | **doctor 检查注册表化后，`run_all_checks` 的报告 key 顺序 / 重复 check_id / 维度归组的运行时差异**，方案只提"按现行顺序"未提"顺序由谁保证" | `run_all_checks` 现行用嵌套 `_run(check_id, fn, dimension)` 逐个登记，报告 dict 按插入序；`test_doctor_unit.py` 对顺序/键有断言 | 注册表若用 dict（3.7+ 保序）可保序，但若分组模块的 import 顺序改变，CHECKS 拼接顺序随之改变 → 报告 key 顺序漂移 | **阶段 5**：CHECKS 采用显式有序 list 而非各模块自动收集；模块只定义检查函数，顺序在 `doctor_checks/__init__.py` 一处显式排列（单点控制） |
| R10 | **方案对"拆分时顺手优化"的纪律约束无法机械化执行**，全靠 reviewer 自觉 | §8 风险表第 4 行只写"PR 自审清单第一条" | tools.py 4871 行、db.py 3813 行的体量下，人工保证"函数体 diff 为空"不可靠；一处顺手改名即漂移 | **阶段 3/4/5 工具化**：拆分 PR 附"搬运证明"——对被移动函数做 AST 归一化（剥 import/self 改写）后与源函数 diff，diff 为空才允许合入；可用 `ast.dump` 比对脚本固化到 CI |
| R11 | **`text.py` CJK 正则"以最广为准"的预设与 §6.3"差异可能是有意的"自相矛盾，且已确认 db 侧与 search/anchors 侧正则确实不同**（实测） | `db.py:3791` `_CJK_CHAR_RE = r"[㐀-鿿豈-﫿぀-ヿ가-힯]"`（字面 CJK 字符）；`anchors.py:32`/`search.py:55` 用 `㐀-䶿一-鿿豈-﫿...`（转义区间，含 Ext-A 㐀-䶿）。两组**不等价**：db 版缺 Ext-A 区间且边界字符需逐一核对 | 若按方案"以最广（search._CJK_RE）为准"直接统一，`db._subject_tokens`/`find_metadata_overlap_candidates` 对含 Ext-A 的 subject 判定改变 → 行为漂移 | **阶段 1 前置**：先写 Ext-A 边界字符（如 㐀）的特性测试锁定 db 与 search 两侧现状差异，再决定保留两个具名常量（推荐：`CJK_RE_CORE` 与 `CJK_RE_WITH_EXT_A`，注释各自语义）而非强行合一。**默认取"保留两个常量"**，除非测试证明现状差异是 bug |
| R11-补 | **执行后实测修正（2026-08-09，阶段 1 落地时）**：R11 原表述有两处错误，已在实施中以特性测试 `tests/test_cjk_characterization.py` 锁定真实现状并据此实现 | (a) 真实差集**不是**"db 缺 Ext-A"，而是：search 版 `3400-4DBF + 4E00-9FFF` 挖空了 U+4DC0-4DFF，db 版 `3400-9FFF` 连续区间覆盖了它们——db 是 search 的**严格超集，仅多 64 个易经卦象符 U+4DC0-4DFF**。(b) db 的 `_CJK_CHAR_RE` **只被 `_subject_tokens` 使用**（subject 切分），不影响 FTS/锚点 | 若误合一到 search 版，含易经符的 subject 不再按 CJK 切分（行为漂移）；误合一到 db 版则 FTS/anchor 路径对易经符判定改变。两侧都不能动 | 已实现为 `text.CJK_RE_SEARCH` 与 `text.CJK_RE_SUBJECT` **两个具名常量**（实测与原始编译正则逐码点等价），原 `db._CJK_CHAR_RE`/`search._CJK_RE` 均 re-export 委托。`db.py` 的 `import re as _re`（3789 行重复 import）已移除——这是 R11 唯一的目标"unsafe"符号变化，属预期内部别名移除，无外部/测试依赖 |

### 9.5.3 审查确认"成立、无需改"的关键判断

- 模块骨架合理、store 组合模式可复用——**确认成立**，claims_db/conflict_judgments 是正确模板。
- 兼容性策略（facade re-export + 委托）总体可行——**确认成立**，但须叠加 R4/R6/R8 的机械化验收才能兑现"零迁移"。
- "不合并 claims_db/conflict_judgments""不动 schema""不引新依赖"等排除项——**确认合理**。
- §6.3 怪癖清单——抽查 `_is_truthy`、Channel 6 content="" 回填、strict pending 顺序、`MemoryRecord.from_input` 填 "default"，**均与代码一致**，清单可信。

### 9.5.4 对抗性审查净结论

方案**可以执行，但必须带着 R1–R11 进入对应阶段**。最大的三处实质风险按严重度：

1. **R4 monkeypatch 冲突**（高）：它直接戳破"测试零改动"承诺，是阶段 3/4/5 的共同前置。
2. **R5 getattr 归因**（高）：52 处 `getattr` 在未逐处比对默认值前承诺清零，是行为漂移的最大单点来源。
3. **R11 CJK 正则合并**（中）：方案自己内部矛盾（"以最广为准" vs "差异可能有意"），必须按"默认保留两个常量"收敛，否则召回/标签匹配对扩展 A 区汉字行为改变。

其余 R6–R10 是把"靠自觉的纪律"换成"可机械验证的验收"，成本低、应全部采纳。

---

## 9.6 更高层对抗性审查：原始缺陷与 hardening 范围

> §9.5 主要审查“重构机械风险”（monkeypatch、re-export、CJK、Settings、AST 搬运等）。本节站在更高层审查“原代码本来就存在的安全/隔离/事务/并发缺陷”。结论：这些缺陷不能再被“零行为变更”口号掩盖；但修复也不能混在纯移动提交里。v0.12.4 采用两轨执行：重构轨保持行为不漂移，hardening 轨逐项有意改变行为并配测试。

### 9.6.1 已确认原始缺陷

| # | 缺陷 | 证据 | 影响 | v0.12.4 处理 |
|---|---|---|---|---|
| H-01 | `AgentPolicy.default_enabled` 被解析/展示但未参与判定，实际 default-allow | `config.py:17` 定义字段；`config.py:21-30` `enabled_for()` 最后 `return True`；`config.py:410-414` 解析；`tools.py:2853-2858` status 展示 | 用户设置 `default_enabled=false` 时，未知 client/agent 仍允许 | H2 修复：让字段生效，并补 allow/deny/client_defaults/default_enabled 测试 |
| H-02 | policy 只保护 `memory_write`，其他 mutating tools 未统一 gate | `tools.py:383-388` `_allowed()`；`tools.py:1158-1161` 仅 write 调用；resolve/confirm/alias/supersede/edit/store_embedding 等写路径无 policy gate | 若 policy 被理解为“禁用某 client 的写能力”，当前存在绕过 | H8-a 修复：先建立授权矩阵，再把 policy gate 扩展到 mutating/runtime-state tools |
| H-03 | `authorized` 参数多处直接 truthiness 判断，字符串 `"false"` 可绕过 | `_is_truthy()` 在 `tools.py:563-579` 已声明安全语义；但 `memory_confirm`、`memory_activate`、`memory_supersede`、`memory_correct_conflict_judgment`、`memory_edit` 等仍用 `if not authorized` | loose JSON/MCP payload 传 `authorized="false"` 时非空字符串为真，可能通过授权 | H1 修复：统一 `_is_truthy()` 或严格 bool，补 negative tests |
| H-04 | strict workspace isolation 只约束 search，不约束 read-by-id/review/recent/console detail | `tools.py:1950-1971` search strict 强制 workspace；`tools.py:2286-2315` get 按 id 读；`tools.py:2376-2382` recent 跨全库；`tools.py:682-709` review/conflict/history 按 id 读；`console_api.py:113-136` conflict detail 读左右 memory | 若 strict 被理解为 workspace 级读隔离，则是信息泄露；若只是 recall 隔离，则需文档明确 | H8-b 修复/决策：先用 isolation 矩阵锁定现状；若不升 v0.13.0，则按 settings.workspace 实现 strict read ACL |
| H-05 | `memory_confirm_pending_workspace()` 多事务，可能半更新 | `tools.py:2651-2672` 依次调 `upsert_workspace_alias`→`set_memory_workspace_canonical`→`update_memory`，三步各自独立事务，无整体回滚 | 崩溃/并发时 alias/canonical/status 可能不一致 | H6 修复：DB 单事务 confirm pending workspace，failure injection 测全回滚 |
| H-06 | `memory_supersede()` status/conflict/audit 多事务不原子 | `tools.py:2768-2795` 先 update status，再 resolve conflicts，再 record audit | 可能 status 已 superseded 但 conflict 仍 open 或 audit 缺失 | H7 修复：DB 单事务 supersede，failure injection 测全回滚 |
| H-07 | `memory_edit()` protection/status 检查 TOCTOU | `tools.py:3219-3235` service 层检查；`db.py:3312-3367` 事务内未重检；tags-only 路径 `db.py:3134-3163` 反而有事务内校验 | 并发下检查后被 lock/supersede 的 memory 仍可能被 content edit | H5 修复：检查下沉到 `db.edit_memory()` 事务内 |
| H-08 | config 顶层合法 JSON 但非 object 会崩启动 | `config.py:503-509` 返回任意 JSON；`config.py:97-116` 假设 `cfg.get` | `[]`/`"x"`/`1` 等 config 会 AttributeError，而非 warning fallback | H3 修复：顶层非 dict 视为无效配置，warning + fallback |
| H-09 | `client_defaults` key 大小写不归一 | `config.py:26-28` lookup 使用 lower-case client；`config.py:411` load 时 key 未 lower/casefold | policy 写 `"ZCode"` 可能匹配不上运行时 `zcode` | H2 修复：load 时 normalize key，补大小写测试 |
| H-10 | policy 文件解析失败 fallback allow-all，安全上偏宽 | `config.py:398-408` 解析失败返回默认 `AgentPolicy()`；`enabled_for()` 默认 allow | 损坏 policy 文件时系统允许所有 client/agent，只靠 warnings 提醒 | H8-c 修复：显式 policy 配置解析失败 fail-closed；未配置 policy 仍默认 allow |
| H-11 | tags 元素级未规范到 `str`（入口已把非 list/tuple 归一为 `[]`，见 `models.py:75-77`；但 list 内非字符串元素不过滤） | `models.py:76` `list(raw_tags)` 原样保留元素；`db.py:1950-1955` FTS `" ".join(record.tags)` 遇非 str 元素 TypeError；`search.py:1145-1170` 搜索侧又只保留字符串 | `[1,"x"]` 等 loose payload 可写失败或写/读 tags 口径不一致 | H4 修复：入口对 list 内元素做 str 校验/拒绝，先固定口径再实现 |
| H-12 | semantic backend unload/disable 与 worker inference 生命周期竞态 | `_ensure_semantic_backend()` 在 `tools.py:913-926` 加锁创建；`_semantic_control()` 在 `tools.py:985-1017` 未持同锁 unload/disable；worker 在 `tools.py:1109-1113` 推理后直接 unload；`semantic_conflict.py:407-427` 推理/unload 锁粒度不同 | runtime control 与后台 job 同时运行时 backend 状态可能非预期 | H8-d：定义 in-flight inference 语义并统一 lifecycle lock |
| H-13 | `metadata` 输入未严格要求 JSON object | `models.py:91` 使用 `dict(payload.get("metadata") or {})`，字符串/list-of-pairs 等会抛异常或被魔法转换 | loose payload 可导致非结构化异常，或把非 object 静默转成 dict | H9 修复：metadata 缺失/null→`{}`，dict 浅拷贝，其他类型结构化拒绝 |
| H-14 | numeric 输入未做 finite/range 校验 | `models.py:87` 直接 `float(payload.get("confidence", 0.5))` | `"nan"`/`"inf"`/越界值可能进入模型与 JSON 响应，破坏标准 JSON 或排序/判断语义 | H10 修复：confidence 等 numeric 字段 finite + range 校验，无效结构化拒绝 |

### 9.6.2 本次修复 vs 待决项

**本次纳入 hardening 轨修复（H1–H10）**：

1. `authorized` coercion：统一 `_is_truthy()`/严格 bool，堵住字符串 truthiness。
2. Policy 基础语义：`default_enabled` 生效，`client_defaults` key normalize。
3. Config 容错：顶层非 object warning + fallback。
4. Tags 输入规范化：写入口与 FTS/search 口径一致。
5. `memory_edit()` 事务内重检 protection/status。
6. `memory_confirm_pending_workspace()` 单事务原子化。
7. `memory_supersede()` 单事务原子化。
8. Policy 扩展到 mutating tools、strict read ACL、policy parse fail 口径、semantic lifecycle lock 按 H8-a/b/c/d 的细化口径实施；若 strict read ACL 被判定为产品语义大改，则升级版本号。
9. Metadata 输入规范化。
10. Numeric 输入 finite/range 校验。

**仍需执行前确认细则，不得暗改的子决策**：

1. mutating 工具清单必须列完整：write/edit/update/confirm/activate/supersede/resolve/correct、workspace alias govern、store_embedding、rebuild/cleanup/activate/resync/set_entity、semantic_control、govern retire/resolve/confirm/correct；doctor/review/search/read/recent 默认 read-only，dry_run repair 必须测试不写。
2. strict read ACL 若实施，不改 payload：用 `settings.workspace` 作为 caller workspace context；跨 workspace 的 `memory_get`/review/history/conflict_detail/console detail 返回 not_found/forbidden 或隐藏敏感内容（最终口径固定），same-workspace 正常。
3. policy parse fail：仅显式配置 policy 文件但解析失败时 fail-closed；未配置 policy 仍默认 allow；归一化后重复 client key 视为 invalid policy。
4. semantic backend unload/disable：本次最低语义为“加 lifecycle lock，in-flight inference best-effort 完成，不实现取消”；若要等待/取消，另立更大设计。

### 9.6.3 执行纪律

- Hardening 轨每项必须“失败测试先行”，但测试分层要清楚：characterization tests 证明当前坏行为，red tests 表达修复后期望，regression tests 是最终长期保留用例。
- Hardening 提交不得混入纯移动。若一个 diff 同时移动函数并改逻辑，必须拆成两个提交。
- CHANGELOG 必须列出所有用户可见行为变化；不能再写 “no behavior change”。
- 对安全修复的兼容影响要明说：例如 `authorized="false"` 从曾经可能通过变为拒绝，是有意破坏 loose 输入兼容。
- H8 子项按 H8-a/b/c/d 执行；若某子项未完成产品决策，只能保留矩阵测试与文档说明，不得在 v0.12.4 中顺手修。
- H5/H6/H7 原子化不能通过“外层 transaction + 调 public DB method”实现；必须使用同一 `conn` 的 internal helper，避免 nested transaction 或提前 commit。
- H4/H9/H10 的输入规范化必须先固定 reject/sanitize 口径；本方案默认采用结构化拒绝，只有 tags 的空字符串做 strip 后删除。

---

## 10. 不做的事（明确排除）

- 不改任何 MCP 工具签名、payload 契约、help 文案内容（文案可搬家不可改写）；H1–H10 只改变校验/授权/事务一致性语义，不新增/删除工具参数。
- 不动 sqlite schema、迁移逻辑、FTS/vec 建表 SQL；H5–H7 只调整事务边界与校验下沉，不做 schema migration。
- 不引入新依赖（不引 pydantic 进热路径、不引 ORM）。
- 不重命名任何现有公开方法/模块路径。
- 不合并 claims_db/conflict_judgments（已是对的）。
- 不把 H8 子项（全 mutating policy gate、strict read ACL、policy fail-closed、semantic in-flight 语义）伪装成“顺手修复”；未按 H8-a/b/c/d 固定语义前不得落代码。
- 不处理 `scripts/` 的业务逻辑（独立调参工具，不属于产品代码面）；但 wheel/sdist import smoke 必须覆盖其旧 import 路径。
- 注释只清理"版本墓碑"类，设计 §引用注释随代码搬家保留，后续再统一建立 docs/design/ 索引（另立任务）。
