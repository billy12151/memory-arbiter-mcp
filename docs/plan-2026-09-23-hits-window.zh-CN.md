# 实施方案：hits 模式邻句窗口（hit_window）

- 日期：2026-09-23（v3；v2 经一轮对抗性 review，v3 按 owner 拍板修正——处置记录见 §7）
- 状态：**owner 已拍板 F1/F2**（§2）；D1-D4、D7 按推荐结论定稿，可直接开发
- 版本归属：建议随 0.17.0 未发版收敛包一并入库（当前 `memory_arbiter/__init__.py:3` 为 0.17.0，CHANGELOG 0.17.0 标注"未发版一次性收敛"）；若发版切分另有安排则落 0.17.1 槽
- 范围：find / batch_find / read / batch_read 四个调用的 `content_mode="hits"` 档；不改召回层排序逻辑、不改存储层、不改默认档

---

## §0 背景与目标

0.17.0 行级化（C2-C5）后，向量召回与 conflict 管线共用 `memory_row` / `memory_row_vec`，每行是完整句子/表格行，带精确源字符偏移（DDL：`memory_arbiter/db/additive.py:55-76`；偏移契约 `_clean(content[start:end])` 含 unit 文本的出处是 `memory_arbiter/evidence.py:141-150`，强制校验在 `memory_arbiter/db/evidence_store.py:709`）。find 的 `content_mode="hits"` 已能把命中句以 `hit_spans[]`（text + start/end offset）返回，但有一个缺口：

**语义截断**：只返回命中句本身。中文记忆大量指代（"这个方案""它"）和跨句条件（"X 支持 Y。但 Z 场景除外"），单句命中丢失上下文。

目标（一件，P0）：

- **P0-1**：`hit_window` 参数——hits 档按 `row_index ±N` 把命中句的相邻句一并返回（完整句子，非字符截断）。

~~P0-2 词法通道补命中~~：**owner 拍板不做（2026-09-23）**——向量通道未命中本身就是"没有足够相似句"的答案，服务端再挑一句自相矛盾；向量不可用时更无从找相似句。FTS/LIKE 命中条目维持纯 preview 形状（`read.py:257-259` 现状不变）。

附带的行为修正（review F1 发现、owner 拍板修法）：hits 档此前对"旧版本行坐标切新版本原文"无防御，会静默切错区域；本次顺带修复，并按 owner 要求**显式提示 agent 版本已变、请重新查询**。

非目标（明确不做）：

- 不把 sentence+邻句设为默认档（`read.py:28-30` owner 红线："服务端绝不替 agent 挑选重要命中"）。
- 不动 preview 默认档、full 兜底、≥50% 覆盖升级规则本身。
- 不改 offset 坐标系（保持 Python 0 基 code-point 偏移）。
- `memory_search_expired` 无 content_mode，不在本次范围。
- 不做词法通道的句子级补命中（见上）。

---

## §1 既定事实锚点（开发前必读；经 review R1 对码核对）

存储与召回：

- `memory_row` DDL：`memory_arbiter/db/additive.py:55-76`；字段含 `memory_id, memory_version, content_hash, row_index, kind, text, start_offset, end_offset`；`UNIQUE(memory_id, row_index)`；`kind='subject'` 行的 span 是哨兵 `(0,0)`（`:57` 注释），无定位意义。
- `_evidence_hits` 构造：`search.py:770-789`（hit dict 字面量 `:781-789`），逐 hit 字段 `{evidence_id, kind, text, start_offset, end_offset, distance, score}`——**当前无 `row_index`、无行版本**（Step 2a 要加）；attach 到 pool 行 `search.py:805-821`；`_` 字段剥离 `search.py:1282-1286`（`debug_ranking=True` 时整块跳过——**debug 响应会原样外发 `_evidence_hits`，新增字段属 debug 面 additive 可见变化**，现状契约如此，非回归）；`keep_evidence_hits` 参数声明 `search.py:1016`。
- `row_knn`：`db/evidence_store.py:373-383`，SQL `:423-434`（SELECT 含 `m.version AS memory_row_version`，`:427`；memory_row 列经 `r.*` 入结果）；**SQL 无行版本谓词**——召回可能拿到旧 version 行，这是 F1 的根源（Step 2a/2b 在消费侧修，不改 SQL）。
- 两条提前 return 路径（recent_browse `search.py:1110-1116`、filter-driven `:1123-1140`）不经过页面组装段——**空 query 浏览页和纯 filter 页的 hits 档永远不会有 hit_spans**，文案要写明（review F6）。

返回层：

- `_hit_spans`：`read.py:169-212`（丢弃 subject 命中 `:187`、int 严格校验拒绝 bool `:192-195`、`0<=s<e<=len(content)` `:198`、排序后合并重叠/相接区间 `:202-208`、`text=content[s:e]` `:210`、无命中返回 None `:200-201`）。**注意：只做边界校验，不校验版本/内容一致性**——旧 version 的合法区间会切出新 content 的错误区域（F1）。
- `_preview_item`：`read.py:215-263`；hits 档消费 `_evidence_hits` 后无条件 pop（`:260`——`row_index` 等字段不会经 find/batch_find 非 debug 路径泄漏）；覆盖率升级判断 `read.py:250-256`；`_HIT_SPANS_FULL_COVERAGE = 0.5`（`read.py:30`）。生产调用点仅 `read.py:506`（find）与 `read.py:873`（batch_find `_preview`）两处（review A5 核实）。
- outline 批量预取现例：`read.py:499-504` 调 `outline_rows_for_ids`（`db/evidence_store.py:191-193`，SQL 范式 `:203-224`：`(memory_id, memory_version) IN ((?,?),...)` 每 200 条 chunk + `ORDER BY memory_id, row_index`）。
- `_unit_aligned_hits`：`read.py:44-98`（read/batch_read 的 hits 档）；`text_unit_rows`（`db/evidence_store.py:79-86`）返回 keys `unit_index(=row_index 的 SQL 别名), kind, text, start_offset, end_offset`，恒过滤 `kind != 'subject'`；span=None 时返回全量非 subject 行（现状即全量——read 侧 window>0 取全量行**不是新增开销**）；batch_read 走预取未过滤行 + Python 侧 span 过滤（`read.py:74-82`，预取点 `:1447-1449`）。**键名注意**：evidence_store 批量接口统一用 `unit_index` 别名，与 Step 2a 给 `_evidence_hits` 加的 `row_index` 是同一列的两个名字，join 时心里要有数（review R1 ③）。
- batch_find 合并页 `_preview`：`read.py:863-873`（debug 字段只放行 `_evidence_hits`）；batch_find 以 `debug_ranking=True` 调 search（`read.py:831-839`），find 以 `keep_evidence_hits=(content_mode=="hits" and not debug_ranking)` 调（`read.py:393`）。
- `_evidence_hits` 全部消费方（review R1 grep 核实）：`read.py:249/260/393/865/871`、`search.py:814/851/1285`、`scripts/eval_relevance_floor.py:94`（新增字段对它 additive 安全，无需改动）。

校验边界：

- MCP 单工具 `memory(action, data)`（`server.py:349-381`），无 per-field JSON schema；`data` 为自由 dict。
- 未知字段被 `_v_unknown_fields` **pop 丢弃** + warning `"unknown field ignored: {key}"`（`validation.py:227-280`）——**`hit_window` 必须进 `PRODUCT_FIELD_REGISTRY` 白名单，否则静默失效**。白名单位置：find `validation.py:63-68`、batch_find `:69-72`、read `:73`、batch_read `:74`。
- **golden 联动（review A2）**：`scripts/gen_golden_validation.py:296-309` 遍历 registry 每个 key 的每个 allowed field 生成用例，`FIELD_KIND`（`:69-104`）与 `LEGAL_SAMPLE`（`:118+`）必须同步补 `hit_window` 条目，否则下次 regen 直接 KeyError；`tests/test_golden_validation.py:10-11` 规定 regen 必须是独立 commit、diff 逐行 review。
- batch_find 的 `queries` 子项硬限制 `{id, query}` 两字段（`validation.py:307-309`）——`hit_window` 是 batch 级参数，不动此处。
- content_mode 现有校验即 read.py 四处 `_CONTENT_MODES` 检查（`read.py:344-352` / `:787-796` / `:1213-1217` / `:1368-1373`），validation.py 不管。
- **零改动声明（review A5 核实）**：`surfaces.py` 的 action 分发（`:776-813`）走 `self._forward(..., **payload)` 透传，不需要动签名；`tools.py` 四处薄封装中 `memory_batch_find :1767-1768`、`memory_batch_read :1770-1771`、`memory_get :1797-1806` 为 `**payload` 透传零改动；`memory_search :1758-1765` 是显式参数+`**_`，`hit_window` 落进 `_` 也能原样转发，但建议提为显式参数防漏参漂移（review F7，对齐 `include_size` 事故教训）。

文案触点（review C1 核定共七处；v3 口径已随 P0-2 取消而调整——原"items without vector hits keep the plain preview shape"承诺**继续成立**，只需补 hit_window 与 stale 提示）：

1. `server.py:364-369` docstring content_mode 段；
2. `read.py:554-562` find hits 档 display_hint；
3. `surfaces.py:125-126` find_size_metering 块；
4. `surfaces.py:166-168` batch_find_semantics 块（review E1：原方案误写为"batch_read :166"，surfaces.py 不存在 batch_read 散文式 help 块）；
5. `AGENT_ONBOARDING.md:6` content_mode 描述；
6. `README.md:77`；
7. `docs/INTEGRATION.md:74`。

统一改写口径：hits 档补"`hit_window=N`（默认 0）带出命中句 ±N 相邻完整句；hit_spans 只出现在 query 召回页，浏览/纯 filter 页没有（review F6）；命中句版本落后于记忆当前版本时 span 被丢弃并显式提示重新查询（F1 拍板）"。server.py docstring 的 offset 契约句用 review A7 措辞："offsets are 0-based Unicode code-point offsets into the content as indexed for the item's version; the evidence index may lag right after an edit (see vector_lag)"。

测试落点：

- 现有 hits 测试：`tests/test_find_enhancements.py:415-578`（find/batch_find）；`tests/test_batch_read.py:132-249`（read/batch_read，含 `_insert_units` 直接 SQL INSERT 造行范式 `:34-51`）；`test_row_store.py:29-124`（publish_rows 范式）；`test_find_enhancements.py:545-578`（FakeEmbedder 全管线范式）。validation 测试在 `tests/test_product_validation.py`。

---

## §2 设计决策（D1/D2/D3/D4/D7 按推荐定稿；F1/F2 为 owner 拍板）

- **D1（窗口单位）**：`hit_window=N` 按 **`row_index ±N` 句**扩展（同 memory、同 version、排除 subject 行），不按字符数。整句扩展不碰"永不截断"红线；表格行按 row_index 相邻，表头折叠行随之带出（散文/表格混排的"邻句"语义即 row 序语义）。
- **D2（默认值、上限、非法值）**：默认 `0`（行为与现状逐字节一致，F1 修正除外，见 §4 第 1 条），上限 `_HIT_WINDOW_MAX = 5`。规整收敛到统一 helper `_coerce_hit_window(value) -> int`：`int()` 失败→0；负数→clamp 0；超上限→clamp 5 **并追加 warning**（review F7：静默 clamp 会掩盖 caller 笔误）；bool 按 int 处理（无害展示参数，不套用 span 的 bool 拒绝）。**`hit_window` 与非 hits 档同传时静默忽略**（对齐 `limit_per_query` 惯例，review A6）。
- **D3（覆盖率口径）**：≥50% 升级全文按**扩展后**的合并区间计算——窗口大、命中多的文章本就接近"整篇相关"，直接给全文省一次 read。配套保护见 Step 3 的 batch_read 字节预算推广（review F3）。
- **D4（`matched` 标记；v3 简化——P0-2 取消后只剩窗口邻句需要标记）**：
  - 窗口带入的邻句 span：恒带 `matched: false`；
  - 检索器实际命中的 span：默认形态（无新键）；仅当同一条目的 span 列表中存在邻句 span 时，命中 span 补 `matched: true` 消歧；
  - **推论：`hit_window=0` 时输出形状与 v0.15.10 逐字节一致**。
- **F1（owner 拍板 2026-09-23）：旧版本命中丢弃 + 显式提示。** 命中行的 `memory_version` 与条目当前 version 不一致时丢弃该命中（全部丢弃则该条目无 hit_spans），**同时**：
  - 条目上加 `stale_hit_spans: {"evidence_version": <行版本>, "memory_version": <当前版本>}`；
  - 响应级 extra_warnings 加一条人话提示："item #\<id\>: hit spans dropped — the memory was likely edited after your previous read (evidence index vX vs memory vY); re-query or re-read to get fresh spans"；
  - read/batch_read 侧的对应面：`content_mode="hits"` 因当前 version 无行而回落、以及 full+span 走 legacy 字符切片回落（`read.py:1273-1277`、`:1309-1316`）时，各加一条同语义 warning（这两个回落今天完全静默）。
  - owner 原话依据："不显示没问题，不过你要明确提示 Agent 版本号不正确，可能有人在 Agent 读取后更新过数据，请重新查询。"
- **F2（owner 拍板 2026-09-23）：词法通道补命中不做。** 理由：向量通道未命中即"没有足够相似句"，服务端再挑一句自相矛盾；向量不可用时无从找相似句。Step 6/6.5、row_knn 正向 memory_id 过滤、`_LEXICAL_HIT_FLOOR`、校准探针全部撤销；`origin="lexical_fallback"` 标记体系随之取消（D4 已相应简化）。
- **D7（read/batch_read 同步）**：`_unit_aligned_hits` 同步支持 hit_window，四调用语义一致。

---

## §3 实施步骤

### Step 1：evidence_store 批量行 span 预取

新增**范围受限**的批量行 span 预取（review F4：行数无上限，只取命中行 ±w 范围，不全量取）：

```python
def row_spans_for_ids(
    self, entries: list[tuple[int, int, int, int]],
) -> dict[int, list[dict[str, Any]]]:
    """(memory_id, version, lo_row_index, hi_row_index) 批量取窗口行的定位信息。

    返回 {memory_id: [{"unit_index", "kind", "start_offset", "end_offset"}]}；
    WHERE 按条目 OR 拼接 (memory_id=? AND memory_version=? AND row_index BETWEEN ? AND ?)，
    每 200 条 chunk（同 :203-224 范式）；过滤 kind != 'subject'；不带 text。
    """
```

lo/hi 由调用方从该条目 `_evidence_hits` 的 `row_index` min/max ∓ window 算出（多簇命中取覆盖区间即可，w≤5 下失真可忽略）。**不带 `text`**（邻句只需 offset）。

（v2 的 Step 1b——row_knn 正向 `memory_id` 过滤与 `db/core.py` 代理改动——随 P0-2 取消，不再需要。）

### Step 2：find/batch_find 的窗口扩展 + F1 版本对齐

**2a. `_evidence_hits` 增加 `row_index` 与行版本字段**（`search.py:781-789` 构造处）：从 row_knn 返回行补拷 `row_index` 与 `memory_version`（`r.*` 已含 memory_row 全列；若与 `m.version AS memory_row_version` 存在列名歧义，在 `db/evidence_store.py:423-434` 的 SELECT 显式加 `r.memory_version AS row_version` 并取该名）。additive；`_evidence_hits` 在非 debug 路径被 `_hit_spans` 消费后 pop（`read.py:260`），debug 路径原样外发属既有契约（§1 已声明）。
**这一步连带修复潜伏 bug（review F1）**：此前旧 version 行的合法 offset 会切出新 content 的错误区域；Step 2b 的版本丢弃对 `hit_window=0` 的既有 hits 也生效——漂移窗口内的错位命中变为"丢弃+显式提示"，CHANGELOG 中注明此行为修正。

**2b. `_hit_spans` 扩展签名与逻辑**（`read.py:169-212`）：

```python
def _hit_spans(
    raw_hits, content, *, window: int = 0,
    window_rows: list[dict] | None = None,
    memory_version: int | None = None,
) -> tuple[list[dict] | None, bool]:
```

返回值从 `list | None` 改为 `(spans, dropped_stale)` 二元组（`dropped_stale=True` 表示存在因版本不匹配被丢弃的命中；调用方 `_preview_item` 据此设置条目级 `stale_hit_spans` 字段与响应级 warning，见 F1）：

- **版本对齐（F1 拍板）**：先丢弃 `memory_version` 与条目 version 不一致的命中（字段缺失视为不一致，防御），记录是否有丢弃。
- 现存逻辑不变产出基础命中区间（subject 丢弃、int/bool 校验、`0<=s<e<=len(content)`）。
- `window>0` 且 `window_rows` 非空时：对每个幸存命中，取 `window_rows` 中 `unit_index ∈ [row_index−window, row_index+window]` 的行的 `(start_offset, end_offset)` 并入区间集（同样过边界校验）。`window_rows` 为空（version 漂移/行未建）→ 不扩展。
- 合并逻辑（`:202-208`）不变，对扩展后区间集统一跑。
- D4 标记：窗口带入的 span 带 `matched: false`；存在邻句 span 时命中 span 补 `matched: true`。
- 无幸存命中 → `(None, dropped_stale)`。

**2c. `_preview_item` 接线**（`read.py:215-263`）：签名加 `hit_window: int = 0`、`window_rows: list[dict] | None = None`，hits 分支透传（`memory_version` 取 `item["version"]`）；`dropped_stale=True` 时条目加 `stale_hit_spans = {"evidence_version": <被丢弃命中的行版本>, "memory_version": <item version>}`（多个被丢弃版本取 max，一个条目一个字段）。覆盖率判断（`:250-256`）自然作用于扩展后的 spans（D3）。

**2d. `memory_search`**（`read.py:322` 起）：签名加 `hit_window: int = 0`；在 `_CONTENT_MODES` 校验块（`:344-352`）旁调 `_coerce_hit_window`（Step 5）规整。`content_mode=="hits" and hit_window>0` 时，在 outline 预取（`:499-504`）同点调 `row_spans_for_ids`（entries 由页面条目 `_evidence_hits` 的 row_index 范围算出），按 `r["id"]` 分发进 `_preview_item`。页面组装后若有任何条目带 `stale_hit_spans`，向 `extra_warnings` 追加 F1 的人话提示（每条目一条，含 memory id 与两个版本号）。

**2e. `memory_batch_find`**（`read.py:743` 起；review A3 明确落点）：签名加 `hit_window` 并同样规整；**预取点在 query 循环结束后、merge 循环（`:876` 起）开始前**：以 `[(int(r["id"]), int(r.get("version") or 1)), ... for _, _, _, r in collected]` 算范围并调一次 `row_spans_for_ids` 得 `_window_map`（函数内部 `sorted({...})` 去重，照 `evidence_store.py:201` 范式）；`_preview`（`:863-873`）闭包内按 `int(row["id"])` 查 `_window_map` 传入 `_preview_item`。deduplicate=false 分支共用同一闭包，天然覆盖。**严禁**把预取放进 `_preview` 内部（每 item 一次查询，违反 §4 性能门）。`stale_hit_spans` 提示同样经 `_preview_item` 落在条目上，并汇总进 batch 响应的 extra_warnings。

### Step 3：read/batch_read 同步（`_unit_aligned_hits`，`read.py:44-98`）

- 签名加 `window: int = 0`。
- `window>0` 时行集取全量非 subject 行：`rows=None` 分支（memory_get）改为不带 span 调 `text_unit_rows`（span=None 全量取是现状已有路径，**不是新增开销**），span 过滤在 Python 侧照 `:74-82` 现逻辑做；`rows` 参数分支（batch_read 预取）本来就是未过滤全量，直接用。
- span 选中的行集合照旧产出 hit_spans 条目；再按 `unit_index ±window` 从全量行集带入邻行（带 `matched: false`）；重叠/相接合并；覆盖率升级判断（`:94-98`）作用于扩展后。
- `memory_get`（`read.py:1186` 起）签名加 `hit_window`，在 `:1213-1217` 校验块旁规整后透传；`memory_batch_read`（`:1346` 起）同（`:1368-1373` 旁）。
- **F1 拍板的 read 侧提示**：`content_mode="hits"` 且 `_unit_aligned_hits` 返回 None（当前 version 无行）走全文回落（`:1273-1277`）时、以及 full+span 走 legacy 字符切片回落（`:1309-1316`）时，各向 extra_warnings 加一条"evidence index lags the memory version（可能刚被编辑）；若是依据旧 offset 读取，请重新查询确认"。batch_read 同语义逐条进响应 warnings。
- **batch_read hits 档字节预算（review F3）**：`BATCH_READ_MAX_HITS=50`（`constants.py:291`）× 升级全文在窗口下可轻易造出无预算的大响应（docstring `:34-37` 的"hits 结构性有界"前提被窗口打破）。把 full 档的字节预算降级逻辑（`read.py:1518-1556`）推广：**hits 档页面中凡携带 content（被升级）的条目计入同一 `BATCH_READ_FULL_BUDGET_BYTES` 预算**，超预算走同一结构化 over-long 响应（永不静默截断）。

### Step 4：校验边界与文案

- `validation.py` `PRODUCT_FIELD_REGISTRY` 四处白名单加 `hit_window`（find `:63-68`、batch_find `:69-72`、read `:73`、batch_read `:74`）。
- **golden 生成器联动（review A2）**：`scripts/gen_golden_validation.py` 的 `FIELD_KIND`（`:69-104`，`hit_window` 归 `"int_range"` 或 `"none"`）与 `LEGAL_SAMPLE`（`:118+`，样例值如 `2`）同步补条目；按 `tests/test_golden_validation.py:10-11` 的规矩以**独立 commit** 重新生成 `tests/golden/validation.json`，diff 逐行 review。
- 七处文案按 §1 口径统一改写（`server.py:364-369`、`read.py:554-562`、`surfaces.py:125-126`、`surfaces.py:166-168`、`AGENT_ONBOARDING.md:6`、`README.md:77`、`docs/INTEGRATION.md:74`）。

### Step 5：常量与规整 helper

- `read.py` 模块级：`_HIT_WINDOW_MAX = 5`（照 `_HIT_SPANS_FULL_COVERAGE` 先例，`read.py:30`）；`_coerce_hit_window(value) -> int`（int() 失败→0；clamp `[0, _HIT_WINDOW_MAX]`，clamp 生效时追加 warning；bool 按 int 处理）。

（v2 的 `_LEXICAL_HIT_FLOOR` 随 P0-2 取消。）

### Step 6：测试

数据构造范式照现例（`tests/test_batch_read.py:34-51 _insert_units` 直接 SQL INSERT；`tests/test_find_enhancements.py:545-578` FakeEmbedder；`test_row_store.py:29-124` publish_rows）。

`tests/test_find_enhancements.py` 新增：

- `test_hits_window_default_zero_byte_identical`：逐键断言（review B3：非录制比对）——不显式传 hit_window 时 span 条目**无 `matched` 键**、无 `row_index` 外发、形状与 v0.15.10 完全一致。
- `test_hits_window_one_includes_neighbors`：5 句 memory，命中第 3 句，window=1 → spans 覆盖第 2-4 句，且每 span `text == content[s:e]`；邻句 span `matched=false`、命中 span `matched=true`。
- `test_hits_window_never_pulls_subject`：命中 row_index=1，window=1 → 无 (0,0) 区间、subject 文本不出现。
- `test_hits_window_coverage_upgrade`：window 使扩展后覆盖 ≥50% → 升级全文且 hit_spans 保留。
- `test_hits_window_clamped_and_invalid`：`hit_window=99` 按 5 生效且有 warning；`hit_window="x"` 回退 0；`hit_window=-1` → 0（review A4）。
- `test_hits_window_adjacent_spans_stay_separate`：命中句与邻句间有分隔空白 → 两个 span 条目分别精确（不强并）。
- `test_hits_stale_version_dropped_with_signal`（F1 拍板，find 侧）：`_insert_units` 造行后 bump memory version（edit）不重嵌 → 旧 version 命中被丢弃、条目**不切错区域**、带 `stale_hit_spans={"evidence_version","memory_version"]}`、响应 warnings 含"重新查询"提示。
- `test_hits_partial_stale_window_survives`：部分命中过期部分新鲜 → 新鲜命中正常出 span（含窗口扩展），`stale_hit_spans` 如实上报。

`tests/test_batch_find.py` 新增（review B2，原方案完全遗漏）：

- `test_batch_find_hits_window_merged_page`：两 query 命中同 memory、deduplicate=true → hit_spans 与 `matched_query_ids`/`best_query_id` 共存，window 逐 id 生效，debug 字段除 `_evidence_hits` 外无残留。

`tests/test_batch_read.py` 新增：

- `test_read_hits_window_expands_neighbors`：span 选中 1 行，window=1 → hit_spans 含 3 行且 unit 完整（无半句），带入行 `matched=false`。
- `test_read_hits_window_version_mismatch_noop`：memory_row 的 `memory_version` 与 memory 当前 version 不一致 → 不扩展、不报错、带 stale 提示 warning。
- `test_read_full_span_legacy_fallback_warns`（F1 拍板）：full+span 且当前 version 无行 → legacy 切片照常返回但带"索引可能滞后"warning。
- `test_batch_read_hits_window`：spans + window 组合逐 id 生效。
- `test_batch_read_hits_window_budget`（review F3）：多 id 升级全文超 `BATCH_READ_FULL_BUDGET_BYTES` → 结构化 over-long 响应，无静默截断。

边界用例（review B4）：空 content（`_hit_spans` 恒 None，钉死防回归）；单行 memory（window 退化为自身 + 升级交互）；`hit_window` 与非 hits 档同传静默忽略（D2/A6）。

`tests/test_product_validation.py` 新增：`hit_window` 进白名单（不再出现 `unknown field ignored` warning）；golden regen 随 Step 4 独立 commit。

### Step 7：CHANGELOG

按 0.17.0 未发版收敛包惯例追加条目（参照 0.17.1 工作区豁免条行文），注明：owner 拍板日期与结论（F1 修+显式提示；P0-2 不做）、以及 2a/2b 连带的旧版本命中错位修正（行为修正，单独一句说明）。

---

## §4 验收标准

1. **行为门**：`hit_window` 缺省/为 0 时，四个调用的输出与实施前逐字节一致——**唯一例外**是 F1 行为修正：漂移窗口内（行版本 ≠ 记忆版本）原先会切错区域的 hit_spans 现在被丢弃并带 `stale_hit_spans` 提示（owner 拍板的预期变更）。锁法：现有 hits 测试套件（`test_find_enhancements.py:415-578`、`test_batch_read.py:132-249`）不改动即通过 + 新 stale 用例钉住例外路径。
2. 新增测试全绿；`mypy` 与 `ruff` 通过。
3. 性能：hits+window 页面新增查询固定为 1 次 `row_spans_for_ids`（批量、范围受限）；preview/full 档零新增查询；batch_read hits+window 页受 `BATCH_READ_FULL_BUDGET_BYTES` 预算约束（F3）。
4. 红线复核：任何路径不返回被截断的句子；默认档不含 content；subject 文本永不进入 hit_spans；过期命中永不被静默切错（F1）。

---

## §5 风险与缓解

| 风险 | 缓解 |
|---|---|
| **旧 version 行命中切错区域（review F1，含无窗口时的潜伏 bug）**：row_knn 无 version 谓词，异步重建窗口期内 offset 与新 content 错位 | Step 2a hit dict 带行版本；`_hit_spans` 入口丢弃版本不匹配命中；条目 `stale_hit_spans` + 响应 warning 显式提示重新查询（owner 拍板）；find/read 两侧各有测试锁定 |
| 窗口扩展后 agent 分不清哪句是真命中 | D4：邻句 span 恒带 `matched=false`，混合列表中命中 span 补 `matched=true`；window=0 形状不变 |
| batch_read hits+window 打穿"结构性有界"前提（review F3） | hits 档升级出的 content 计入 `BATCH_READ_FULL_BUDGET_BYTES`，超预算走结构化 over-long 响应 |
| `row_spans_for_ids` 全量取行在病理长文上无界（review F4） | SQL 按命中 row_index ±w 范围收缩（Step 1）；read 侧 span=None 全量取是现状路径，无回归 |
| `hit_window` 未进白名单被静默 pop / golden 生成器 KeyError（review A2） | Step 4 硬步骤 + validation 测试 + golden regen 独立 commit |
| batch_find 预取放错位置变成逐 item 查询（review A3） | Step 2e 明确落点：query 循环后、merge 循环前，一次批量调用 |
| 窗口与 ≥50% 升级相互作用使 hits 档频繁升全文 | D3 是期望行为；`_HIT_WINDOW_MAX=5` 封顶；升级后 hit_spans 保留作标注 |
| agent 依据旧 offset 做 read span，读到编辑后的错位内容 | F1 拍板：当前 version 无行时 read hits 回落与 legacy 切片回落均带"重新查询"warning；docstring 写明 offset 随 version 失效（§1 文案口径） |

---

## §6 工作量估计（v3）

Step 1（约 60 行 + SQL）、Step 2-3（约 150 行，含 F1 信号链路）、Step 4（白名单 + golden regen + 七处文案）、Step 5（常量 + helper）、Step 6（约 200 行测试）、Step 7。召回层排序逻辑与存储层零改动；无校准阻塞项（随 P0-2 取消）。预计 **1.5-2 天**含测试。

---

## §7 决策与 review 处置记录

### owner 拍板（2026-09-23）

| 项 | 拍板 |
|---|---|
| F1（旧版本命中错位） | **修**：版本不匹配的命中丢弃、不显示；**但必须显式提示** agent"版本号不正确，可能读取后数据被更新过，请重新查询"（条目级 `stale_hit_spans` + 响应级 warning + read 侧两个静默回落补 warning） |
| F2 / P0-2（词法通道补命中） | **不做**：向量通道未命中即"没有足够相似句"，服务端再挑一句自相矛盾；向量不可用时更无从找相似句。Step 6/6.5、row_knn 正向过滤、阈值校准全部撤销 |

### 对抗性 review 处置（2026-09-23，一轮：R1 对码 / R2 设计 / R3 落地与测试）

| 发现 | 来源 | 处置 |
|---|---|---|
| E1 surfaces.py "batch_read :166" 张冠李戴（实为 batch_find_semantics 块） | R1 | 采纳，§1 修正 |
| E2 `_clean` 契约出处应为 evidence.py:141-150 + evidence_store.py:709 | R1 | 采纳，§0 修正 |
| E3 §1 `_evidence_hits` 概览行号偏宽 | R1 | 采纳，修正为 :770-789 |
| R1③ unit_index/row_index 别名、debug 面可见性、eval_relevance_floor.py 消费面 | R1 | 采纳，§1 补注 |
| F1 版本漂移：命中侧旧坐标切错区域（HIGH） | R2 | 采纳并按 owner 拍板强化（显式提示），Step 2a/2b/Step 3 |
| F2 D5 同形透出撞红线（HIGH） | R2 | 发现成立；owner 拍板 P0-2 整体不做，问题随功能撤销而消解 |
| F3 batch_read hits+window 无字节预算（MEDIUM-HIGH） | R2 | 采纳，Step 3 推广 full 档预算降级 |
| F4 `row_spans_for_ids` 全量取行无界（MEDIUM） | R2 | 采纳，Step 1 改范围受限 |
| F5 0.45 尺度风险 + 校准顺序（MEDIUM） | R2 | 随 P0-2 取消而消解 |
| F6 "hit_spans 只在 query 召回页"文案（LOW-MEDIUM） | R2 | 采纳，§1 文案口径 |
| F7 tools.py 显式参数、clamp warning（LOW） | R2 | 采纳，§1/D2 |
| A1 `db/core.py:425` row_knn 代理漏改（P1） | R3 | 随 P0-2 取消而消解（不再需要正向 memory_id 过滤） |
| A2 golden 生成器随注册表联动（P1） | R3 | 采纳，Step 4 + Step 6 |
| A3 batch_find 预取落点与键源（P2） | R3 | 采纳，Step 2e 明确 |
| A4 `_coerce_hit_window` helper 与 read 侧落点、负数 clamp（P2） | R3 | 采纳，Step 5/D2 |
| A5 调用点清单与零改动声明（P2） | R3 | 采纳，§1 补结论 |
| A6/A7 非 hits 档行为、docstring 措辞（P3） | R3 | 采纳，D2/§1 |
| B1 fallback 测试夹具配方（P1） | R3 | 随 P0-2 取消而消解 |
| B2 batch_find 窗口零直接测试（P1） | R3 | 采纳（窗口部分保留），Step 6 |
| B3 "逐字节一致"测试方法论 + 标记键按需出现（P2） | R3 | 采纳，D4/Step 6 |
| B4 空 content/单行/负数的边界用例（P2） | R3 | 采纳，Step 6 |
| B5 过时测试 docstring（P3） | R3 | 随 P0-2 取消而消解（原 docstring 恢复准确） |
| C1 文案触点清单（P2） | R3 | 采纳（七处保留，口径随 P0-2 取消调整） |
| C3 AGENT_ONBOARDING 具体改法（P3） | R3 | 采纳，Step 4 |
| C4 性能验收口径（P3） | R3 | 采纳并随 P0-2 取消简化，§4 第 3 条 |
| C5 fallback 的 stale-version 风险（P3） | R3 | 并入 F1 处置（find 侧版本对齐保留） |
| 验收第 1 条判定：成立但需写明生效边界 | R3 | 采纳并随 P0-2 取消简化，§4 第 1 条 |

---

## §8 实施记录（2026-09-23，随 0.17.0 未发版收敛包入库）

实施完成：Step 1-7 全落地。验证基线：全量 2691 passed + 1 skipped、mypy 65 文件零错、ruff 全绿、golden 门 1040 passed（PYTHONHASHSEED=0）。改动面：pipeline/read.py（核心）、db/evidence_store.py（row_spans_for_ids）、search.py（hit dict 加 row_index/row_version）、validation.py（白名单四处）、tools.py（memory_search 显式参数）、golden 生成器+corpus、七处文案、CHANGELOG、四个测试文件追加 15+3 用例。

### 与方案的有意偏离（已 review 确认成立）

1. **row_version 缺失 fail-open（R1 抓出方案自相矛盾）**：§2b「字段缺失视为不一致→丢弃」与 §4.1 行为门（现有测试不改即过）不可兼得——存量合成 hit 不带 row_version，严格丢弃会全灭。实现改为「row_version 存在且 ≠ 条目版本才丢弃；缺失无法判定→保留」。生产路径无缺口：row_knn 的 r.* 恒带 memory_version，_evidence_hits 恒填 row_version。
2. **_hit_spans 返回三元组**（spans, dropped_stale, stale_evidence_version）：方案的两元组承载不了 §2c「多丢弃版本取 max」——调用方需要版本号值才能落 stale_hit_spans。
3. **golden 生成器存量缺口一并修复**：registry 与生成器三表在 2026-09-14（0b3637a）后漂移——claims（P2-5）与 claims_backfill 五字段（after_id/mode/model_path/results/slow_lane）从未进 FIELD_KIND/LEGAL_SAMPLE，gen 脚本对现行 registry 直接 KeyError。本次补齐（claims=manual/[]，claims_backfill 按语义定型），regen 属等价更新（891,809 字节级复现）+ hit_window 新用例。

### 第一轮对码 review（PASS-with-notes，0 P0/P1）

- [P2 工作区] .git/sequencer 有进行中的 revert 序列（workspace 豁免线 C1/C2 回滚，与本实施零文件交集）——保持原样不代决，commit 按路径拆分，遗留 owner 处置。
- [P3 已修] evidence_lag 措辞中性化（None 也可能来自 span 不相交/查询异常，不全是 lag）；coercion 挪过 include_superseded 早退防 clamp warning 丢失；memory_get not-found 合并 read_warnings；surfaces batch_find 块与 AGENT_ONBOARDING 补口径（后者受 3000 字节帽约束，压缩同行冗余后 2999 字节）；batch_read 两个 F1 warning 直接用例补上。

### 第二轮对抗性 review（SHIP-with-notes，4 P3 实锤全修，无 P0/P1/P2）

26 个实测探针击穿 F1/窗口/预算主防线全部失败（debug_ranking 页 stale 照常生效、evidence-only preserved 链保留 _evidence_hits、row_version 异型值按设计、跨 chunk/参数上限实测通过、性能门每页恰 1 次预取）。已修 P3：①batch_read 回落措辞改「metadata-only record」（其回落本就无 content，与单条 read 的 full-record 回落区分——存量行为差异，顺带钉死）；②memory_get span 校验/batch_read ids 与 cap 早退带 read_warnings；③batch_find deduplicate=false 同一 stale memory 不再重复 N 条 warning（按 id 去重）；④预算测试注释归因修正（单行 100% 覆盖 window=0 同样升级，预算门与窗口无关照常施加）。
理论风险记录（不修）：find hits 页无字节预算为 window=0 时代既有形态（方案 §5 只对 batch_read 立预算，有意）；row_spans_for_ids sqlite3.Error fail-open 空图（窗口静默缺失退 window=0 形状）；float hit_window 截断（D2 未约定，无害）。

### 其他记录

- 全量套件偶发一例 test_first_run_demo 竞态失败（复跑即过，与本改动无交集，疑似异步 worker 竞态存量 flake）。
- AGENT_ONBOARDING.md 受 help topic 3000 字节帽约束，七处文案中该处的 F6 半句以同行冗余压缩换入（2999/3000 字节）。

### harness 回归（重启服务后全量三套件，2026-09-23）

- 带 hits-window 全量三套件 `all-hits-window-0170`：Recall@10 0.9333（42/45）、自召回 97/98、相似提示 12/64（true_near_dup 12/16）、冲突 Recall 0.2093（9/43）、Precision 0.6923、共存误报 0/18。
- 对 baseline-0.16.12 回归门 FAILED 6 项（conflict recall/write_opposition/noisy/irrelevant_false_pulls）。**AB 归因实验：stash 本次改动后在 HEAD 原样重跑 conflict+recall，六项差异全部逐位复现（conflict Recall 0.2093、write_opposition 0.5、noisy 0.0385、无关误召回 2/7 完全一致；唯一抖动=scan_evolution 一对 sync/async 边界）——hits-window 零贡献，回归门失败全部是 0.17.0 行级化检测线（行级化/subject 行排除/internal 预算帽，8c69047/1377378）相对行级化前基线 baseline-0.16.12 的存量差距，基线重写属 0.17.0 发版前动作。**
