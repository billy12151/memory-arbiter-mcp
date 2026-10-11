# Memory Arbiter（迷码）— 中文说明

**[English](README.md) | 中文**

<!-- mcp-name: io.github.billy12151/memory-arbiter-mcp -->

## 一句话说明白

**迷码是 Agent 能力放大器——用上它，你的全部 AI 工具都升级成有记忆、有灵魂的超级助手。**

今天 Claude 聊透的决策，明天 Cursor 写代码直接能用，后天换别的 Agent 接着干——大家共享同一份记忆：现行、可信、有出处、改不坏；谁跟谁打过架、哪些是你拍过板的，全都记得清清楚楚。它跑在你自己电脑上（SQLite 一个文件，不联网、不上传），通过 MCP 标准接口接入所有工具。当前正式版本 `0.17.2`；`0.18.0`（已在 main）带浏览器知识库工作台。

## 一分钟看懂迷码给你带来什么

- **召回是真的准。** 词法和"按句找证据"两条路一起搜、按排名融合，发版考试自召回 **65/65**。不够格的候选根本不进查询结果页——返回的都是你能直接引用的；每条命中带原文偏移，`read span=[start,end]` 按字符偏移精确切出原文。
- **冲突藏得住，也报得准。** 两条事实打架时，自训练的本地 mDeBERTa 小模型来判定——它只会给三种回答：冲突 / 没冲突 / 拿不准，不听任何暗示词，也绝不替你决定留哪条删哪条。高置信的冲突才来打扰你，拿不准的只作灰区参考，其余默默落进定时扫描——**门再严也漏不掉，门开着也不乱吵**：发版考试里报出来的冲突，九成以上是真打架（精确 90.91%）。
- **事实是你的，而且改不坏。** 完整原文存在你机器上的一个 SQLite 文件里，每条带出处、每次修改有版本，信任级别锁住你确认过的东西。向量、索引都是"目录"，原文才是"正文"，丢了随时重建。平时不联网——唯一可选的主动外联是一次 PyPI 版本检查（控制台学习工作台走你自己填的 LLM 端点）。
- **知识库由你策展。** 本地浏览器控制台里圈定哪些记忆算"知识"（`kb_member` 独立治理位，Agent 重写标签也冲不掉），按 工作区 → 标签 浏览、markdown 渲染、一跳关系图；还带自带 LLM 的学习工作台——总结、梳理、出题、费曼讲解、时间线，产物存回记忆库。

## 评测

### 公开评测：LoCoMo-Refined

在 [LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined) 上全量评测（1,382 题，官方判分 Qwen3-14B）：

| 配置 | 总体 | 文本类 |
|---|---:|---:|
| **mema + 写入代理**（deepseek-v4.1-flash + 公开规则） | **61.00%** | **65.62%** |
| **mema 零抽取**（写入端无 LLM，原话照存 + `event_time`） | 43.13% | **49.48%** |

对照：Mem0 完整管线 48.91%（文本类口径，组委会重判）——**写入端不带任何 LLM 的裸 mema 与之打平**；带写入代理后，文本类口径成绩仅次于 MemoraX。完整方法论与复现：[`evals/locomo/RESULTS.md`](./evals/locomo/RESULTS.md)。

### 内部发版考试（出厂配置实测）

每次发版必须过内部考试台——固定考卷：42 道召回题覆盖五种题型（改述/查细节/法条/远因/关键词）、108 对冲突考题（四个标签族、其中 36 条真冲突）、四组相似度考题（口径：`0.17.2` 出厂判定模型与阈值；`eval/runner.py --suite all` 可复现）：

| 考试门 | 成绩 |
|---|---|
| 自召回（65 个检索目标，top-10） | **65/65** |
| 冲突仲裁（D4 综合） | **召回 55.56%（20/36）· 精确 90.91%（20/22）** |
| batch_find 跨查询一致性 | **0 处不一致**（42 题 × 6 批） |
| 冲突窗完成率 | **51/51** |

考卷刻意偏向精确：判定门拦下的不会丢——定时扫描还会用宽门把全库重走一遍，可疑的进只有 Agent 能看到的裁决队列，**漏报有兜底、不等于丢失**。这就是每次发版都重考一遍的质量底线。

## 交给 AI Agent 安装

把下面这段话原样粘给 Codex / Claude Code / Cursor 等有终端权限的编码 Agent：

```text
Read the latest README at https://github.com/billy12151/memory-arbiter-mcp.
Install and configure the latest mema release for my operating system and current AI client.
Preserve any existing config and database; do not overwrite or delete existing data.
Ask me before choosing between materially different install modes, changing existing config,
or performing any destructive or privileged action. When finished, run mema doctor and report
the install method, config path, database path, client integration, and verification result.
```

Agent 会以本 README 为准、按本地环境在 `uvx`/core/`vec`/`semantic-local` 里选安装方式，拿不准就停下来问你。装完必须跑 `mema doctor` 并如实汇报每条警告——没过体检的安装不算装完。

**装完先查一次演示标记**：`memory(action="find", data={"query": "", "tags_filter": ["mema-first-run-demo-status"]})`——命中说明演示已做过或已拒绝（全局只此一次），别再提；没命中，征得同意后可按随包的 `FIRST_RUN_DEMO.zh-CN.md` 规程做一次约一分钟、可回滚的产品演示。

## 手动安装（备用）

```bash
# 一条命令全搞定：装包 + 装依赖 + 下 embedding 模型（约 330MB，断点续传/国内镜像）+ 写配置
curl -fsSL https://memarbiter.cn/install.sh | bash
```

或者分步（需要 Python 3.11+）：

```bash
pip install "memory-arbiter-mcp[vec,semantic-local]"  # core + sqlite-vec + 本地 GGUF 运行时
mema setup --install   # 下载 embedding 模型并写好配置（判定模型需另行手动安装）
mema doctor            # 体检验证
```

然后从 [`examples/`](examples/) 挑 `.mcp.json` 接入客户端，用 `mema` 启动。能力缺失会以持续降级横幅 + 补救指引出现——**装一半不会冒充装好了**。

**上手前只知道两件事：** ① `client`/`agent_id` 必须在 config.json 里显式配置（无内置默认，留空拒绝启动）；② `remember` 必须带 `workspace`——先 `memory_review(view="workspaces")` 列出桶，再传一个已存在的桶名（`default` 是全局池）。

## 四个工具

| 工具 | 能做什么 |
|---|---|
| `memory` | `remember` 存、`find`/`batch_find` 找、`read`/`batch_read` 读、`update` 更新、`judge` 裁决、`status` 体检、`help` 指南 |
| `memory_review` | 只读体检：健康、冲突组/详情、历史、过期审计、工作区视图 |
| `memory_govern` | 需显式授权的治理：废弃、冲突方案执行/解决、各类确认 |
| `memory_repair` | 维修：证据重建、广域冲突扫描、历史清理、备份回放 |

每次调用返回 `{ok, mode, warnings, degraded, data}`；要你处理的事在 `notices` 或 `data.action_required` 里。让 Agent 记住打包好的规矩：`memory(action="help", data={"topic":"agent_onboarding"})`。

日常就四个动作：`remember` 存一条可复用的事实、`find` 召回、`read` 精确读、`update` 用新来源替换旧结论（**同一事实永远只有一份现行版本**）。`find` 是索引页（元数据 + 字符数 + 带原文偏移的目录）；`batch_find` 一次合并 8 个查询；`read span=[start,end]` 按字符位置精确引用；搜不到就明说，不硬塞近期记忆。

## 里面有什么

**冲突仲裁，装好就有。** 候选对先过几道不花模型的粗筛（句级预筛 → 记忆级筛查 → 相似度带 → 规则证据），然后交给本地 mDeBERTa——两句裸文本进、校准概率出。高置信的才报警，拿不准的只作参考，其余一律落回定时扫描：服务端自己走全库，可疑项进仅 Agent 可见的裁决队列，绝不进用户可见列表。判定模型永不选边、永不改记忆——发现的事进一张可审计的 `conflicts` 表（`open → applying → resolved`），由你的 Agent 在你逐次授权下走 `judge → apply_conflict_action → resolve_conflict` 解决。

**知识库工作台，装好就有。** `memory-arbiter console` 打开本地控制台：一键策展知识库（`kb_member` 独立数据列）、按 工作区 → 标签 浏览（markdown 渲染 + 一跳关系图）、对任意记忆跑六个一键学习动作（快速总结/知识点梳理/生成考题/费曼讲解/反过来考我/时间线梳理）——用你自己的 OpenAI 兼容端点，产物走标准写管道存回，归因给控制台（Key 只存本地配置文件，后端强制 `0600` 权限并代理调用，页面拿不到；策展由你决定——写入侧不会来问"这条算不算知识"）。

**信任是 enforced 的。** 每条写带出处、每次改有版本、`normal`/`protected`/`locked` 信任级别 + `user_confirmed` 标记，所有改状态的治理动作都要逐次 `authorized=true`。工作区像抽屉：`none` 不设防、`weak` 轻轻把你往自己的抽屉引、`strict` 下新抽屉要先经你确认才能用，而且在项目抽屉里干活也看不到公共抽屉（default）的内容。写进 `recall_blacklist.jsonl`（数据库文件旁边）的工作区会从无范围召回里消失，`mema doctor` 报告生效清单。

## 配置

只认文件，在 `~/.config/memory-arbiter/config.json`。全部可调的就 28 个键——完整模板在 [`examples/memory-arbiter.config.example.json`](examples/memory-arbiter.config.example.json)，逐键说明与迁移表见 [`docs/INTEGRATION.zh-CN.md`](docs/INTEGRATION.zh-CN.md)。第一天要关心的：

```json
{
  "client": "your-client",
  "agent_id": "your-agent-id",
  "workspace": "default",
  "isolation": "none",
  "embedding": { "model_path": ".../embedding.gguf", "auto_query": true, "auto_write": true },
  "semantic_conflict": {
    "mdeberta_ckpt": ".../mdeberta-v58_ep3_fp16.pt"
  },
  "mcp": { "transport": "stdio" }
}
```

- `embedding.model_path` —— 指向本地 GGUF 模型 = 启用证据召回的唯一意图。
- `semantic_conflict.mdeberta_ckpt` —— 指向判定模型 = 启用写入时仲裁（见下）。
- `mcp.transport` —— `stdio`（默认）或多客户端共享时的 `streamable-http`。
- 启动环境变量只剩 6 个（`MEMORY_ARBITER_CONFIG`/`DB_PATH`/`BACKUP_JSONL`/`MCP_TRANSPORT`/`CLIENT`/`AGENT_ID`）；配置文件优先，其余 `MEMORY_ARBITER_*` 导出会警告"不再读取"。

## HTTP 模式（多个客户端共用一个迷码）

stdio（默认）不需要后台进程。要多客户端共用：配置里切 `mcp.transport` 为 `"streamable-http"` → 让进程常驻（迷码不带守护进程，macOS 用 [`examples/`](examples/com.memory-arbiter.mema.plist) 里的 launchd 模板）→ 客户端指向 `http://127.0.0.1:8000/mcp`，带 [`examples/streamable-http.mcp.json`](examples/streamable-http.mcp.json) 里那两个固定身份头（`X-Mema-Client`/`X-Mema-Agent-Id`）。只绑环回；头是来源归因，不是鉴权。Claude Desktop / Claude Code 走一行 `mcp-remote` 桥接条目——照抄同一个示例文件。

## 判定模型（mDeBERTa，可选）

一次性手动安装（约 531MB 的 fp16 checkpoint 不随 PyPI 分发）：

1. `pip install memory-arbiter-mcp[mdeberta]`；
2. 从 mini-clash release 制品下载 `mdeberta-v58_ep3_fp16.pt`（+ `mdeberta-base` tokenizer 目录）；离线安装包（`memarbiter.cn` 的 bundle）里已包含，装离线包可跳过这步；
3. config.json 里把 `semantic_conflict.mdeberta_ckpt` 指过去——自动启用、启动预载；`mema doctor` 核验摘要与标签契约。

不配置 = 写入时仲裁停用；定时扫描和 Agent 兜底不受影响。

## 降级说明

- 没有 sqlite-vec / embedding 模型：词法召回和治理照常，证据索引不可用。
- 没有判定模型：写入时仲裁暂停；定时扫描继续返回确定性候选。
- SQLite 不可用：写入落追加式 JSONL 兜底；工作区决策状态依赖数据库。

## 从旧版本升级

schema 世代早于 `workspace_state_v1` 的数据库启动时直接拒绝——走并排迁移 `mema upgrade`（先 `--dry-run` 预览；源库永不删除）。完整操作指南（WAL 安全备份、哪些会带走哪些刻意丢弃、升级后的扫描纪元）在 [`docs/UPGRADE.zh-CN.md`](docs/UPGRADE.zh-CN.md)（英文 [`docs/UPGRADE.md`](docs/UPGRADE.md)）。

## 开发

```bash
uv run pytest -q
python scripts/sync_version.py --check
```

开发期 package/docs 可能描述未发布 dev 版本，而 `server.json`（MCP Registry 清单）保持最后发布版；`scripts/sync_version.py` 发版时推进，`--check` 防漂移。中文文档的白话基调是硬约定：重写不得丢失中文叙述。更多背景见 [INTRO.md](INTRO.md) 与 [docs/INTEGRATION.zh-CN.md](docs/INTEGRATION.zh-CN.md)。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
