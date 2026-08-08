# Workspace 语义归一 —— 实施方案 (636/637)

## 现状核对(动手前的关键前提)

对着当前代码逐条比对 636,**核心已实现**,这次只做增量:

| 636/637 要点 | 现状 | 位置 |
|---|---|---|
| workspace_canonicals + vec 表、workspace_canonical 列 | ✅ 已有 | db.py:187/281/294 |
| resolver: exact → 向量 topK(cosine 0.25)→ new | ✅ 已有 | db.py:778 `resolve_workspace_canonical` |
| 写路径接入 resolver | ✅ 已有(两处) | tools.py:1061 / 1748 |
| isolation 三档真正 gate 行为 | ✅ none 不碰 / weak 软加权 / strict 硬过滤 | tools.py:1058-1099, search.py:260 `_workspace_bonus` |
| strict 新 workspace → pending + action_required | ✅ 已有 | tools.py:1070-1092 |
| 本地 GGUF 模型脚手架 | ✅ 已有(用于 semantic_conflict) | semantic_conflict.py:304 `LocalGGUFSemanticBackend` |

**真正的增量(本次全做,按依赖排序):**

---

## 阶段 A —— alias 治理层(637,地基,最独立)

637 明确:**方案 1.5 = 当前态表 + append-only 事件日志 + UNIQUE + 单事务,不引入 conflict_judgments 的 CAS**。

### A1. 两张新表(db.py `_initialize_schema` 内,紧跟 workspace_canonicals 之后 ~db.py:285)
```sql
CREATE TABLE IF NOT EXISTS workspace_aliases (
  alias_workspace TEXT NOT NULL,          -- normalized key
  canonical TEXT NOT NULL,
  relation TEXT NOT NULL,                 -- alias|typo|same_project|same_family|related|unrelated|uncertain
  status TEXT NOT NULL,                   -- confirmed|rejected
  source TEXT NOT NULL,                   -- user|agent|rule|qwen
  updated_at TEXT NOT NULL,
  UNIQUE(alias_workspace)                 -- 天然防分裂,不靠 CAS
);
CREATE TABLE IF NOT EXISTS workspace_alias_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  alias_workspace TEXT NOT NULL,
  old_canonical TEXT, new_canonical TEXT,
  old_status TEXT,    new_status TEXT,
  action TEXT NOT NULL,                   -- accept|reject|rename|migrate|correct
  judge_type TEXT NOT NULL,               -- user|agent|rule|qwen
  reason TEXT,
  created_at TEXT NOT NULL                -- append-only, 永不 UPDATE/DELETE
);
CREATE INDEX IF NOT EXISTS idx_ws_alias_events ON workspace_alias_events(alias_workspace, created_at);
```
migration 走既有 idempotent `CREATE TABLE IF NOT EXISTS` 约定(与 workspace_canonicals 同款);无需 CAS 列。`workspace_raw` 637 说等于现有 `memories.workspace`(raw 已保留),**不新增列**。

### A2. db.py 治理方法(单事务 upsert 当前态 + append 事件)
- `upsert_workspace_alias(alias, canonical, relation, status, source, action, judge_type, reason)` → 一个事务里 `INSERT OR REPLACE workspace_aliases` + `INSERT workspace_alias_events`;并发撞 UNIQUE 一方失败重试,状态机自然收敛(637 语义)。
- `get_workspace_alias(alias)` → resolver 用的 O(1) 单行查询。
- `list_workspace_alias_events(alias=None)` → 审计读。

### A3. resolver 接入 confirmed/rejected(db.py:778 `resolve_workspace_canonical` 顶部)
在 exact 命中(step 1)**之前**插入:先查 `workspace_aliases`:
- `status=confirmed` → 直接返回其 canonical,`matched_by="confirmed_alias"`,**不调向量/不调 Qwen**(636 第 1/8 步)。
- `status=rejected` 的 pair → 记入 result,后续候选/向量阶段过滤掉该 pair(636 第 4 步 "过滤 rejected alias pair",第 8 步抑制重复提示)。

### A4. 治理 API(tools.py `memory_govern`,复用现有 action dispatch,~tools.py:641)
新增 actions(与 retire/resolve_conflict 同款 `_forward` 模式):
- `accept_workspace_alias` {alias, canonical, reason, authorized}
- `reject_workspace_alias` {alias, canonical, reason}
- `rename_workspace_canonical` {old, new}
- `migrate_workspace` {from, to}(bulk 改 memories.workspace_canonical)
- `confirm_pending_workspace`(把 strict 下 pending 的记忆 + 其 workspace 确认为 confirmed alias/canonical,联动现有 memory_activate)
每个都在 help doc(tools.py:423 区块)登记 example。

### A5. 测试(tests/test_workspace_isolation.py 扩展 + 新建 test_workspace_alias_governance.py)
accept→confirmed 归一、reject→抑制、并发 UNIQUE 收敛、事件 append-only、rename/migrate、confirm_pending 闭环。

---

## 阶段 B —— 规则优先决策层(636 第 2/5 步)

现状 resolver 只有"向量 < 0.25 就归一",缺规则闸门。改成 **规则先于向量**。

### B1. 新模块 memory_arbiter/workspace_rules.py(纯函数,无 IO,好测)
- `classify_workspace_quality(ws_raw) -> empty|default|generic|specific|suspicious`(636 第 2 步)。generic 词表:实施计划/项目二期/经营方案/月报/复盘… 内置常量。
- `rule_decision(ws_raw, title, evidence, candidates) -> KEEP|ASK|AUTO|None`(636 第 5 步):
  - KEEP:参考/借鉴/模板/经验/月报/指标/复盘/通用,或技术主题与业务候选明显无关,或命中 rejected。
  - ASK:empty/default、generic、多候选接近、同客户不同子域、证据弱/冲突。
  - AUTO:confirmed/exact/机械标准化/固定别名/明显 typo/中英名互指/标题关键句强命中完整 canonical 且无 KEEP/ASK veto。
  - None:规则定不下来 → 交给阶段 C(Qwen)。

### B2. 接入 resolver / 写路径
resolver 或写路径调用顺序变为:**confirmed/rejected(A3)→ 机械标准化 → rule_decision(B1)→ 向量候选 → Qwen(C)**。向量只产候选,规则/confirmed 优先裁决(636 第 4/5 步:"向量只负责候选,不直接裁决")。

### B3. evidence 抽取(636 第 3 步)
`extract_evidence(record) -> {title, headings, subject, first_para, key_sentences, ws_terms}`;长文不整篇喂模型。放 workspace_rules.py 或复用现有 section/anchor 逻辑(anchors.py)。

### B4. 测试 test_workspace_rules.py
quality 分类、各 KEEP/ASK/AUTO 分支、generic 词表、evidence 抽取。回归跑 test_workspace_isolation.py 确认无行为漂移。

---

## 阶段 C —— Qwen 候选发现(636 第 6 步)

定位:**candidate suggester,非裁决器**。仅在 B1 返回 None(规则无法高置信 AUTO/KEEP)或需解释归一时调用。

### C1. 复用现有本地模型宿主
`LocalGGUFSemanticBackend`(semantic_conflict.py:304)已有 llama-cpp 装载/锁/降级。新增一个 workspace 用途方法(或轻量姊妹类),**不重建装载逻辑**。

### C2. 新方法 `suggest_workspace_candidate(evidence, candidates, examples)`
- 输入:短 evidence(B3)+ topK 候选(向量)+ candidate examples。
- 输出:建议候选、relation(alias/typo/same_project/same_family/related/unrelated/uncertain)、confidence band、evidence、notice 文案草稿。
- 约束:高置信仅在 **weak** 下作静默归一"加分",**不能覆盖 confirmed/rejected/strict**(636 第 6/7 步)。模型缺失/超时 → 降级为 ASK/pending,绝不硬失败(沿用现有 degrade 约定)。

### C3. 按 isolation 落决策(636 第 7 步,补齐 weak 分支)
- **none**:不参与召回/排序/可见性,不打扰;可后台记 suggestion(现状已如此,基本无改动)。
- **weak**:规则强证据/Qwen 高置信 → 静默归一;Qwen 中低置信 → **先 active 入库,再轻提示 write_hints**(现状 weak 只有 new_workspace hint,需扩成"候选归一提示")。tools.py:1093 分支扩展。
- **strict**:仅 exact/confirmed/极机械标准化可 active;语义候选(即使 Qwen 概率大)默认 pending/action_required(现状已有 strict_block,扩成"候选也进 pending 并带候选列表")。

### C4. strict 切换前治理提示(636 第 9 步)
用户启用 strict 时(config 切换点 / setup_cli.py),提示先做 workspace 治理:用主 LLM 批量审查现有 workspace、确认 alias/reject/pending,再启用强隔离。

### C5. 测试 test_workspace_qwen_candidate.py(模型可选,缺失则 skip + 降级路径断言)
Qwen 不覆盖 confirmed/rejected/strict;weak 中低置信走 active+提示;降级不硬失败。

---

## 执行顺序与验证

1. A(治理层)→ 2. B(规则层)→ 3. C(Qwen 层)。每阶段独立可测、可单独 commit。
2. 全程测试入口固定用 **Python 3.12 venv 的 pytest**(项目 memory: 避免 Homebrew pytest/3.14;MCP SDK `mcp>=1.2.0,<2`)。每阶段跑该阶段测试 + `test_workspace_isolation.py` 回归。
3. 不新增细粒度用户配置(weak_auto_policy/qwen_threshold 等),内置默认策略(636 第 10 步)。
4. 保留 raw workspace + alias/reject/change history 审计(637)。
5. 版本对齐:637 旁记的 pyproject/server.json 不一致**已解决**(现 dynamic version + 0.12.0),无需处理。

## 不做 / 明确排除
- 不引入 CAS/版本锁/自引用 supersede 链到 alias 表(637 明确过度设计)。
- 不新增 workspace_raw 列(raw 已在 memories.workspace)。
- 不重建本地模型装载逻辑(复用 LocalGGUFSemanticBackend)。
- 不暴露多策略开关配置。
