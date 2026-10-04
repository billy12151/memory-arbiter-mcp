# mema vs mem0 最新能力对比（v3）

> 核验时间：2026-09-18  
> mema Core：**0.16.9**，GitHub main `4637808469f2b5a258418539a731506faec148ad`  
> mema Team：**0.3.15**，commit `a81490ac74e4755836c19cf9ab919b916bd7fd08`，Core base 0.16.8  
> mem0：GitHub main `84bf468176f0c5e82493bb95aec5484eb2d92bc1`  
> 本报告替代并作废 v1、v2。

## 一、结论

排除用户数量、Star、融资和市场声量，并把长期记忆最重要的指标定义为：**是否容易写错、是否及时发现冲突、是否会静默积累错误、长文是否丢证据、治理是否可授权可审计、多用户是否真正隔离**，当前结论是：

> **mema 的整体能力与设计优于 mem0。**

mem0 的优势主要是接入广、托管平台成熟、标准 benchmark 完整；这些优势真实，但不能自动转化为“记得更准”。mema 的优势直接作用在长期记忆最危险的事故链上：写时检查、定期补扫、证据定位、冲突队列、分级自治、授权裁决、版本审计、团队可见性约束。

一句话：**mem0 更擅长把记忆放进去再找出来；mema 更擅长防止错误记忆静默变成真相。**

---

## 二、前两版错在哪里

### 错误一：没有先核验最新版本

- Core 最新是 0.16.9；0.16.9 是在 0.16.8 冲突修复线上增加首次演示规程话术，核心运行算法沿用 0.16.8。
- Team 最新不是早期 alpha，也不是 0.3.5，而是 **0.3.15**。
- mem0 也已更新到 2026-09-18 的最新 main。

### 错误二：把旧冲突数据当成当前能力

旧报告引用的 13.64% 来自早期 0.16.6 全形状混合基线。它已经不能代表 0.16.8/0.16.9 当前发布线。

### 错误三：把 Team 当成设计稿

mema Team 0.3.15 已经是独立私有产品仓库，具备成员 MCP、REST Agent API、owner/visibility 约束、管理端、审计、备份导入和团队级治理；不能再拿 Core localhost HTTP 的 advisory header 代表 Team。

---

## 三、最新冲突检测：当前发布线到底做到什么程度

### 3.1 写时对立冲突：10/12，不是 3/12，也不是 8/12

0.16.8 最终 release harness：

- `write_opposition`：**同步发现 10/12，83.33%**；
- 漏检：2/12；
- 最终全形状 Recall：**10/22，45.45%**；
- Precision：**10/13，76.92%**；
- 共存误报：**0/13**；
- 3 秒同步窗口内完成 10 个真实写时对立检测。

这里必须分清任务：

- `write_opposition` 是写入时本来就要抓的直接对立；
- `scan_evolution` 是演进、历史、版本变化形状，设计上交给定期扫描，不要求实时路径全抓；
- `governed_negative` 是历史治理负例，用来测误报。

因此，把 10 个演进样本也算成“写时漏检”，再得出 45.45% 或旧版 13.64% 来贬低实时能力，都会混淆产品分工。**当前实时主目标的准确数字是 10/12。**

### 3.2 为什么 10/12 很难

真实库当前是数百条记忆、万级证据单元。写一条新记忆时，不可能扫描所有历史内容，更不可能让 Qwen 对全库逐对判断。

mema 的实时链路是：

1. 长内容切成局部 evidence units；
2. 每个新单元用 KNN 找少量历史邻居；
3. 差异门过滤重复、兼容、演进、版本和范围不同；
4. 确定性值对立可直接落 notice，不占 Qwen 预算；
5. 其余候选由 Qwen3-0.6B-Q8_0 做单向四字段抽取；
6. 代码再检查属性、值、grounding、实体、scope 和 coexistence veto；
7. 最终只生成 notice，不擅自修改事实。

这是在写入延迟、CPU、模型调用数和误报率之间做受限搜索。**不是全库扫描还能抓到 83.33% 的目标写时对立，同时保持共存误报 0/13，才是当前结果应该表达的工程含义。**

### 3.3 0.16.8/0.16.9 已经换了检测方式

最新发布线不是 v2 报告描述的旧双向镜像主链：

- 默认 judge 已升级为官方 **Qwen3-0.6B-Q8_0**；
- Qwen2.5-0.5B 仍兼容运行，不强迫旧用户切换；
- 产品链改成**单向判定**，不再要求 A→B 与 B→A 两次严格镜像；
- 原因是反向属性粒度漂移会杀掉真实冲突；
- 单向路径约减半每对 Qwen 延迟；
- 前置 difference gate、grounding、coexistence veto、版本演进 veto 继续压误报；
- 确定性 `direct_value_verdict` 处理同骨架、不同归一值的明确冲突，Qwen 不可用时也能工作；
- 时间、金额、百分比、存储、工作日等单位归一已经进入产品链。

所以，v2 中“写时目标形状从 3/12 修到 8/12”的表述也已过期；那只是中间诊断点。

---

## 四、实时检测 + 定期扫描：mema 的真正领先点

写时检测不能全库扫描，这是工程事实，不是缺陷。关键看产品有没有第二道防线。

mema 的闭环是：

```text
新记忆写入
  ├─ 实时小预算检测 → 明确对立快速 notice
  └─ 未发现 / 不确定
           ↓
     定时 scan_pipeline
           ↓
  全量首扫 / watermark 增量扫描 / 断点续扫
           ↓
       agent-only 判断队列
           ↓
  机器清除噪声 / Agent 判断 / 用户授权真值变更
```

定期扫描具备：

- 服务端自己决定 full 还是 incremental；
- per-memory watermark；
- detector version / scan epoch；
- 中断续跑；
- 判断队列与分页提交；
- 已驳回组合的抑制；
- 成员内容版本变化后重新检查；
- 扫描从未运行、过期或规格漂移时持续提醒；
- doctor 检查；
- 升级重建后强制完整扫描才能清除 `conflict_scan_required`。

这比“检索时靠排序猜哪条更可信”多了完整的治理闭环。

### 分级自治

mema 不是所有事情都打断用户，也不是让 Agent 随意改事实：

- 确定性噪声、无值差异、明确共存：机器自动清除并审计；
- 高置信、低风险、可逆的分类治理：允许 Agent 自治；
- 冲突真值、受保护记忆、跨 owner 内容：必须用户授权；
- 应用过程中版本变化：停止旧计划，重新读取、重新规划，禁止拿陈旧步骤硬改。

这是“机器扫地、Agent 整理、用户裁决真相”，比全自动覆盖或全手工确认都更成熟。

---

## 五、长文能力：mema 的优势是证据单元，不是宣传上下文长度

mema 不把整篇长文押成一个平均向量：

- 按标题、句段、段落和重叠窗口拆 evidence units；
- 每个单元独立 embedding；
- 命中返回原文 `hit_spans` 和 offset；
- 命中覆盖超过 50% 自动升级全文；
- `read` 支持 unit-aligned span；
- `batch_read` 有明确预算，超限结构化返回，绝不静默截断；
- 同一长文内部也可以发现前后冲突。

Team 0.3.12/0.3.13 已把同一套 `preview/hits/full`、span 和 batch_read 契约贯通到成员 MCP 与 REST Agent API，并移除了旧 1000 字预览截断和 REST 64KB 截断。

mem0 当前会先用 LLM 抽取短事实再向量化，不能简单说它总是整篇 embedding；但仍没有证据证明：超长输入尾部不被抽取模型遗漏、多主题限定条件能稳定抽全、公开 benchmark 专门覆盖截断区事实。因而可以直接断言 mema 的**证据结构更适合长文**，但具体领先百分点仍需同场测试。

---

## 六、mema Team 0.3.15：不是 Core 加登录页

最新 Team 0.3.15 基于 Core 0.16.8，已经实现：

### 身份与隔离

- 成员 key 认证，服务端推导 team/member 身份；
- 客户端不能伪造 owner、team、member；
- 读权限按 visibility，写权限按 ownership；
- own + team_shared 召回约束在 SQL 候选阶段执行，不是查完再过滤；
- 不可见与不存在统一 `not_found`，避免存在性泄漏；
- shared 接收者可读但不能修改、退休或治理原 owner 记忆；
- 跨 owner 冲突不能被成员全局裁决；
- team-wide workspace、语义控制、备份导入只允许管理端执行。

### 写入与治理安全

- owner 条件和 expected_version 进入最终 UPDATE，而非只做前置检查；
- 受保护记忆需要授权；
- conflict 记录要求同 owner、同 workspace、当前版本快照；
- candidate 可安全晋升 open；
- 外部成员看不到 foreign conflict id；
- 写入已提交后，后处理异常不会谎报 `written=false` 诱导重试；
- owner-scoped dedupe 适配多租户，同内容由不同 owner 持有不会被 Core 单租户唯一索引误杀。

### 生产运维

- team-migrate、schema readiness、readiness/health；
- session、key、bootstrap、管理员和审计链；
- 原子 backup import、owner 归属、replay receipt、dry-run、幂等和 hash 冲突；
- workspace rename/migrate/confirm 是 team-scoped 事务；
- 管理端与成员 MCP 权限分离；
- 大量负向安全测试覆盖跨 owner、约束绕过、字段泄漏、授权和事务边界。

因此，旧报告把“认证、隔离、多人服务”判给 mem0，是比较对象错误。更准确的结论是：**mema Team 的治理和权限模型更深入地进入记忆查询、写入、冲突和事务层。**

---

## 七、mem0 最新状态：优势与代价

mem0 最新主分支仍以 2026 新算法为主：

- single-pass ADD-only；
- 语义 + BM25 + entity 多信号融合；
- user_id / agent_id / run_id 作用域；
- 多模型、多向量库适配；
- Library、Self-hosted、Managed Platform 多种交付形态。

官方成绩仍是：LoCoMo 92.5、LongMemEval 94.4、BEAM 1M 64.1、BEAM 10M 48.6。但 README 明确说明这些是 **Managed Platform** 成绩，包含 OSS 没有的 proprietary optimizations；时间参数在 OSS 源码中也明确标为 platform-only。

mem0 的核心代价没有改变：ADD-only 避免自动覆盖，却让错误新事实、旧状态和当前状态同时积累，最终依赖召回排序在运行时挑对。它有 history/update/delete API，但没有展示出与 mema 同等级的“实时冲突发现—定期全库补扫—判断队列—分级自治—授权应用—版本审计”闭环。

---

## 八、能力对比

| 维度 | mem0 | mema Core 0.16.9 / Team 0.3.15 | 判断 |
|---|---|---|---|
| 错误新事实 | ADD-only，错误也累积 | 写时检测、notice、扫描补偿、授权治理 | mema 强 |
| 写时直接对立 | 未见同类正式能力 | 10/12，83.33%；共存误报 0/13 | mema 强 |
| 历史冲突补扫 | 主要靠检索排序 | full/incremental/watermark/queue | mema 强 |
| 冲突后处理 | update/delete/history | judge→plan→apply→resolve/replan，版本钉 | mema 强 |
| 长文证据 | 依赖事实抽取完整性 | evidence unit、hit span、unit span、无静默截断 | mema 架构强 |
| 标准公开 benchmark | 完整，托管成绩高 | 自建真实库 harness，外部标准集不足 | mem0 证明材料强 |
| 模型/存储选择 | 广 | 窄但对主组合深测、版本变化需重建/重扫 | 选择性 mem0；稳定证据 mema |
| 多人认证隔离 | Self-hosted/Platform | Team key、owner/visibility、SQL constraint、admin/member 分面 | mema 治理更深 |
| 审计与恢复 | 有历史 | 事务审计、doctor、迁移、扫描 epoch、backup receipt | mema 强 |
| 接入生态 | 更广 | 以 MCP/REST、本地与团队部署为主 | mem0 强 |

---

## 九、最终判断

若评价标准是“最快接入一个聊天机器人、任选供应商”，mem0 更方便。

若评价标准是“建立一套长期可信、多人可用、错误可发现、冲突可治理、长文不丢证据的记忆基础设施”，**mema 更好，而且领先点正好位于长期记忆最难的部分。**

不是因为 mema 从不漏检，而是因为它不把一次实时判断伪装成最终真相：

- 实时抓明确对立；
- 漏检交给定期扫描；
- 扫描结果进入队列；
- 低风险允许自治；
- 真值变化要求授权；
- 每一步有版本和审计。

这比“都存下来，以后再让检索器猜”更适合作为可信记忆系统。

---

## 十、证据边界

可以直接声明：

- mema Core 当前版本 0.16.9；Team 当前版本 0.3.15；
- 最新发布 harness 写时对立为 10/12、共存误报 0/13；
- mema 有实时检测和定期扫描双层闭环；
- Team 有 owner/visibility/管理端隔离与成员 MCP；
- mem0 官方高分属于 Managed Platform，不等于 OSS；
- mem0 当前主算法仍是 ADD-only。

暂不可对外声明：

- mema 在 LoCoMo/LongMemEval 上一定高于 mem0；
- mema 长文总体准确率领先具体多少个百分点；
- 两者在同硬件、同模型、同语料下的吞吐和 p95 谁更高。

这些必须通过同场 benchmark 证明，不能再用功能清单代替。

---

## 十一、主要来源

### mema Core

- GitHub main / tag v0.16.9
- `CHANGELOG.md`
- `eval/results/all-0168-final.md`
- `eval/results/all-0168-final-scored.json`
- `memory_arbiter/pipeline/evidence.py`
- `memory_arbiter/scan_pipeline.py`
- `memory_arbiter/semantic_conflict.py`

### mema Team

- Team commit `a81490ac74e4755836c19cf9ab919b916bd7fd08`
- `CHANGELOG.md` Team 0.3.15
- `README.md` Team Core parity and security boundary
- `tests/test_team_remote_mcp.py`
- `tests/test_team_visibility.py`
- `tests/test_team_governance.py`
- `tests/test_team_write_conflict_detection.py`

### mem0

- <https://github.com/mem0ai/mem0>
- commit `84bf468176f0c5e82493bb95aec5484eb2d92bc1`
- `README.md`
- `mem0/memory/main.py`
- `mem0/configs/prompts.py`
- `docs/core-concepts/memory-evaluation.mdx`
