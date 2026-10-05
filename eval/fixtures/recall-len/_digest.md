## L01 [t-9cc9b5d181f7] lang=en len=500-1k ws=AgentLane
subject: AgentLane run monitor UI + SQLite default state store pushed; MCP now plaintext-only (f9dd7ec)
tags: agent-lane, implementation-done, run-monitor-ui, sqlite, state-store, step-description, adversarial-review, pushed, v0.1.0a4, commit-a7e3c5f
content_head: AgentLane run monitor UI + SQLite default state store implemented, reviewed, pushed. Latest commit f9dd7ec (`fix: return plaintext-only MCP tool results`) on origin/feat/read-only-ui. Earlier a7e3c5f implements run monitor UI + SQLite default state store. Final gates before f9dd7ec: ruff/mypy/pytest 757 passed, 3 warnings. Change: `agentlane-mcp` no longer emits outputSchema/structuredContent; the strict action envelope is serialized in `content[0].text` only, avoiding duplicate token consumption for agent consumers. Tests updated to assert plaintext-only wire format.

## L02 [t-0a9bf971eb00] lang=en len=500-1k ws=memory-arbiter-mcp
subject: Memory Arbiter MCP 0.14.1 release completed
tags: mema, release, v0.14.1, pypi, github-release
content_head: Memory Arbiter MCP 0.14.1 was released from commit 13236010261ab3a5d71565e4683ac6e11c4b291d. Main and codex/local-text-evidence point to the release commit; annotated tag v0.14.1 and the GitHub Release are published; GitHub CI and the PyPI trusted-publishing workflow passed; PyPI contains the 0.14.1 wheel and sdist. The release includes the reviewed semantic timing fix: notice_sync_wait_ms defaults to 5000 ms, job_timeout_ms is backlog fairness measured from the oldest pending enqueue and only gates between pairs, in-flight Qwen uses inference_timeout_ms, and semantic drain waits for pending p

## L03 [t-109137f8f912] lang=en len=1k-2k ws=default
subject: plan-mode policy: low-risk continuation and execution-review loop
tags: plan-mode, skill, approval, review-loop, low-risk-continuation
content_head: User confirmed the plan-mode workflow policy should connect two behaviors: (1) under the established low-risk continuation口径, clear low-risk tasks—especially read-only review/analysis/research—may proceed after showing a plan/key steps instead of forcing a separate approval turn every time; high-risk actions still require explicit separate confirmation; (2) after execution, an execution review is required, and if review finds non-trivial issues (failed tests, goal drift, missed requirements, new risk, weak evidence, or user-requested adjustment), the agent should re-enter the plan/execute/revi

## L04 [t-402ed7557374] lang=en len=1k-2k ws=jd-proxy
subject: JD socrates local proxy switched to v1 compatibility layer
tags: jd-socrates, proxy, zcode, gpt-5.5, claude-opus-4.8, configuration, 2026-08-03
content_head: 2026-08-03 19:11 Asia/Shanghai: Updated local proxy `/Users/zhangzhiwei17/.openclaw/workspace/tools/jd-opus-proxy.mjs` after backing it up to `/Users/zhangzhiwei17/.openclaw/workspace/tools/jd-opus-proxy.mjs.bak-20260803-191109`. Proxy now defaults to socrates `v1` (`UPSTREAM_VERSION=v1`) as an API-key compatibility layer and no longer requires/injects Cookie by default (`needCookie=false`); v2 Cookie mode remains available via `JD_OPUS_UPSTREAM_VERSION=v2` or `JD_OPUS_INJECT_COOKIE=1`. Kept Anthropic/Opus sanitization for unsupported thinking/cache_control/system-array fields. Added OpenAI/GP

## L05 [t-40edc74fc882] lang=en len=1k-2k ws=jd-proxy
subject: JD socrates proxy route-split configuration
tags: jd-socrates, proxy, route-split, zcode, gpt-5.5, claude-opus-4.8, configuration, 2026-08-03
content_head: 2026-08-03 19:40 Asia/Shanghai: Updated `/Users/zhangzhiwei17/.openclaw/workspace/tools/jd-opus-proxy.mjs` to route-split mode after backing up the previous v1-compat script to `/Users/zhangzhiwei17/.openclaw/workspace/tools/jd-opus-proxy.mjs.bak-routesplit-20260803-193907`. Current proxy listens on `127.0.0.1:47821` and healthz shows `messages` route -> upstream `v2` with Cookie (`needCookie=true`), `chatCompletions` route -> upstream `v1` without Cookie, default -> `v1` without Cookie. Existing field-sanitization remains: Anthropic messages strip thinking/output_config/cache_control and hist

## L06 [t-2a8dc5563bd2] lang=en len=1k-2k ws=jd-proxy
subject: JD socrates proxy context-length cap for Claude Code (DeepSeek/GPT) 2026-08-03
tags: jd-socrates, proxy, context-length, 40001, claude-code, deepseek-v4-pro, gpt-5.5, max-tokens-cap, 2026-08-03
content_head: 2026-08-03 Asia/Shanghai: Fixed Claude Code "API Error: 400 上下文长度超出限制" when calling DeepSeek-V4-Pro/GPT-5.5 through local proxy `jd-opus-proxy.mjs`. Root cause: Claude Code sends `max_tokens=32000` with large system/tools/history payloads; JD socrates v1 counts input context + reserved output budget together and rejects some combinations with 40001. Direct tests showed DeepSeek-V4-Pro fails at ~1MB input + 32000, while GPT-5.5 fails at ~3MB + 32000 but has non-monotonic behavior at 2.6MB (8192 fails, 4096 works, 2048 fails, 1024 works). Implemented model-aware max_tokens capping in proxy: Deep

## L07 [t-cf994b35a85a] lang=en len=2k-4k ws=jd-proxy
subject: JD socrates / ZCode proxy troubleshooting summary 2026-08-03
tags: jd-socrates, zcode, claude-code, proxy, claude-opus-4.8, gpt-5.5, 40002, extended-thinking, route-split, troubleshooting, 2026-08-03
content_head: 2026-08-03 Asia/Shanghai full troubleshooting summary for ZCode/Claude Code using JD socrates models through local proxy. Initial issue: when using local proxy for `Claude-Opus-4.8`, both ZCode and Claude Code reported `provider_code=40002`, `invalid_request`, status 400. Located local proxy at `/Users/zhangzhiwei17/.openclaw/workspace/tools/jd-opus-proxy.mjs`, listening on `127.0.0.1:47821`, managed by `/Users/zhangzhiwei17/.openclaw/workspace/tools/proxy.sh`. Added temporary 4xx/5xx request/error logging and found failures were due to Anthropic extended-thinking/new protocol fields: top-leve

## L08 [t-fb27870780f6] lang=en len=2k-4k ws=memory-arbiter-mcp
subject: memory-arbiter update notice UX decision: 7-day suppression, no ack
tags: memory-arbiter, update-check, doctor, ux, decision
content_head: 2026-08-04 decision for memory-arbiter-mcp update discovery UX: implement background/runtime update discovery with a simple reminder policy, not an ack protocol. When an update notice is shown/delivered, suppress the same version for 7 days before showing it again. Do not add a complex memory_ack_notice style acknowledgement flow. If the user upgrades memory-arbiter, the version cache/runtime state must update so the old update notice stops appearing. For post-upgrade doctor reminders, persist doctor-run state per installed version. If the user has already run doctor on the current version, do

## L09 [t-80939d044dc9] lang=en len=>4k ws=memory-arbiter-mcp
subject: mema Team v1 工程实施方案（两轮对抗审查后）
tags: mema, memory-arbiter, team-v1, implementation-plan, admin-portal, member-readonly-portal, M1, Agent-Key, Grant, security-review, Tencent-Cloud
content_head: 2026-08-10 完成 mema / memory-arbiter Team v1 工程实施方案，并进行了两轮独立对抗性审查（安全/架构审查 + 交付/API/前端审查）。本方案基于 M1 方案 id=647 与 Team v1 产品规格 id=659，作为后续编码实施依据。

结论：可以指导开发，但必须按阶段执行，不能跳过前置安全架构。Team v1 是腾讯云单实例 SQLite 部署形态，包含 Team Admin Portal、Member Read-only Portal、Agent/MCP read-only wrapper。v1 不注册 MCP 写入、MCP share/revoke、Member Portal 写入记忆、Agent Key 写 scope。

## 核心假设

1. 当前本地 Console（memory_arbiter/console_api.py、console_static.py）是本地只读调试入口，不作为 Team v1 云端权限系统复用；生产 Team server 默认不注册这些路由。
2. core memories.id 是 INTEGER PRIMARY KEY AUTOINCREMENT；Team v1 中所有与 core memories.id 关联的 memory_id 字段统一使用 INTEGER NOT NULL。
3.

## L10 [t-7a7f77833361] lang=mixed len=<500 ws=memory-arbiter-mcp
subject: memory-arbiter v0.3.0 公司群推广文案定稿
tags: memory-arbiter, 推广, v0.3.0, 公司群
content_head: ## memory-arbiter v0.3.0 公司群推广文案

用户要求写一段推广词发到公司群，让同事试用 memory-arbiter。

### 定稿要点
- 突出是用户自己的开源项目
- 整体介绍产品，不只介绍 v0.3.0
- 加实操贴士：让用户跟 Agent 说"以后查记忆用 memory-arbiter 的 memory_search"
- 不提 MEMORY.md（agent 强制加载的，说了是废话）
- pip install 不加版本号（默认下载最新）
- 不举噪音问题的具体例子
- v0.3.0 只说解决了经典噪音问题即可
- MIT 协议，欢迎 star/issue/PR

### 文案存于对话中，未单独存文件。

## L11 [t-afd1b6ddb6e1] lang=mixed len=<500 ws=memory-arbiter-mcp
subject: README "The Bigger Picture" 概念段（v0.4.0 发版）
tags: memory-arbiter, README, 发版, 概念传播, The Bigger Picture, 事实掌控力
content_head: ## ✅ 已完成：README "The Bigger Picture" 概念段（v0.4.0 发版）

### 状态
已在 v0.4.0 commit 并发版（见 id=106 / id=108）。原"未 commit/未发版"为旧状态，现更新。

### 改动内容
英文和中文各加一个 "The Bigger Picture" / "更大的图景" 段，插在 "Why Memory Arbiter?" 和 Features/核心能力 之间。核心论点：
> 智能趋于平权，数据定义高下。Agent 的终极竞争力是事实掌控力——memory-arbiter 是这条原则的工程实现。

### 传播价值
- GitHub README + PyPI 项目描述自动同步（PyPI 读 README）
- 概念先行再引出功能，比直接列 feature 更有传播力

### 关联
- 概念短文：`~/OpenClawProject/doc/智能趋同时代_事实即竞争力.md`
- 配置文档：id=104（README 补充向量模型配置指南，已同版交付）

## L12 [t-968fccbf3d26] lang=mixed len=<500 ws=memory-arbiter-mcp
subject: 飞天小虾 memory-arbiter 私有桥接自检
tags: 飞天小虾, memory-arbiter, bridge, self-test
content_head: 飞天小虾私有 memory-arbiter 桥接自检：此记录用于验证 companion 私有桥接可写、可查，且不修改 memory-arbiter 仓库代码。

## L13 [t-efc2500e449f] lang=mixed len=<500 ws=default
subject: plan-mode-mcp v0.1.0 代码审查结论
tags: 审查, 代码质量, plan-mode-mcp, v0.1.0, 决策
content_head: WorkBuddy 对 plan-mode-mcp v0.1.0 进行了全量代码审查。审查范围：5 源文件（server/models/db/config/tools）+ 5 测试文件，~2000 行，73 测试用例。总体评价：架构分层清晰（server→tools→db，无循环依赖）、测试覆盖全面、错误处理有三重 elicitation 降级、WeakKeyDictionary 会话隔离设计精巧。可直接用于生产。发现 8 个问题全部为 P2-P4 级别（文档不一致、代码异味），无 P0/P1 安全或正确性问题。

## L14 [t-d85e2b84b8a6] lang=mixed len=<500 ws=default
subject: Agnes image/video skills 已应用本地下载输出规则
tags: Agnes, skill, 已应用, 本地下载, 输出路径, agnes-image-2.1-flash, agnes-video-v2.0
content_head: 2026-07-29 用户确认应用 Agnes skill 后续更新。已成功应用 proposal `agnes-video-v2-20260729-43d7961b8f` 和 `agnes-image-21-flash-20260729-b703c1daf9`。live skill 已读回验证：video skill 现在要求解析国内站完成响应顶层 `url`（fallback `metadata.url`）、支持顶层 `size_mapping`，并要求完成视频输出下载到用户指定 working document/path 后才算完成；image skill 现在要求生成/编辑后的图片下载或解码到用户指定 working document/path，远程 Agnes URL 不能单独视为完成，除非用户明确只要 URL。两者仍坚持 provider-level 配置从 `models.providers.agnes` 继承，不硬编码 baseUrl/apiKey/auth。

## L15 [t-1c286f849601] lang=mixed len=<500 ws=memory-arbiter-mcp
subject: memory-arbiter 0.9 本地规则文档更新
tags: memory-arbiter, 0.9, rules, AGENTS, structured-claims, documentation
content_head: 2026-08-02 已确认本机 memory-arbiter 运行版本为 0.9.0，structured_claim_mode=beta_all。已更新本地 Markdown 规则：AGENTS.md 及 pm/patent-writer/software-architect/knowledge-keeper 的 memory/工具与环境.md 从 0.8.5 调整为 0.9；补充 structured claims 规则：memory_write/memory_edit 后必须检查 action_required、verification_status、conflict_judgment_requests；若返回 judge_conflict_before_use，先读取双方证据并调用 memory_submit_conflict_judgment，判断前不要使用冲突结论；若 user_action_required=true 再询问用户。knowledge-keeper 的 memory-arbiter审查规范.md 也加入该冲突处理口径。

## L16 [t-8aea8aba76ff] lang=mixed len=<500 ws=default
subject: mema 是 memory-arbiter 的短称
tags: mema, memory-arbiter, 迷码, alias, instruction, memory
content_head: 用户明确确认：`mema` 是 `memory-arbiter` 的短称，中文语境也可叫“迷码”。用户说“mema 查记忆”、“查一下 mema”、“迷码查一下”、“写到 mema”、“写到迷码”、“mema 记一下”、“remember this in mema”等，都应理解为使用 memory-arbiter 工具，而不是引用某个本地文件。查询/读取类请求映射到 memory_search 或 memory_get；保存/记一下类请求映射到 memory_write，并补齐 subject、tags、source_type、event_time、workspace、source_ref 等 metadata。

## L17 [t-fa0b3159cd0a] lang=mixed len=<500 ws=default
subject: mema workspace 归一不只是相似度：需要语义归一/项目族识别
tags: mema, memory-arbiter, workspace, normalization, semantic-normalization, canonical, project-family, vector, design-decision
content_head: 用户纠正并确认：workspace/claim 归一不能只理解为向量相似度；还需要语义归一与项目族/事实域识别。例如“金营项目/经营项目”“项目二期”“经营方案”“实施计划”等文本可能不是同一个 workspace 字符串，但需要识别出它们属于同一个项目或同一事实域的相关 workspace。纯向量只能提供候选相似度，不能单独完成 canonical 命名、父子层级、同域但非同槽位的语义归一；方案需要引入 canonical registry/alias/family 层，以及必要时由主大模型或用户确认归一关系。

## L18 [t-eff5747e2b0c] lang=mixed len=500-1k ws=default
subject: patent-writer工具与环境配置
tags: 工具, 环境, patent-writer
content_head: # 工具与环境

## 系统
- macOS 15.6.1 (arm64), MacBook
- OpenClaw 运行在本地，默认模型 zai/glm-5.2
- z.ai / 智谱官方 (open.bigmodel.cn) 的 API key 通用，baseUrl 可互换（2026-06-10 验证）
- Fallback 链：GLM-5.2 → GLM-4.7 → Agnes 2.0 Flash

## 网络
- Clash Verge Ninja 代理: `127.0.0.1:6789`（HTTP），访问 Poe/Google 等需要代理
- Poe API 通过代理可访问，需配合 `--http1.1` 避免 TLS 问题
- Chrome 调试模式 (port 9222) 用于 Playwright 自动化访问内网
- Chrome 调试模式必须用 `--user-data-dir`，默认 profile 不支持

## 搜索
- web-search-prime（z.ai MCP）：国内直连，速度快，优先使用
- web_search（Tavily）：免费 1000 次/月，备选
- 百度 MCP 搜索已配但需检查开通状态

## 模型 Provider
- zai（智谱）：主力，Max 会员
- poe：Claude/GPT/Gemini 等，走代理
- agnes

## L19 [t-96a6a2026dd6] lang=mixed len=500-1k ws=memory-arbiter-mcp
subject: memory-arbiter-mcp 安装 Runbook：其他系统安装时优先引用
tags: memory-arbiter, mcp, install-runbook, jingleai, reuse-doc
content_head: memory-arbiter-mcp 安装 Runbook 已整理完成，供其他 JingleAI / Agent 系统复用。文档路径：/Users/zhangzhiwei17/OpenClawProject/memory-arbiter-mcp/docs/JINGLEAI_MEMORY_ARBITER_INSTALL.md。使用规则：当用户要求给其他系统安装 memory-arbiter、输出安装文档、配置 memory-arbiter MCP 时，优先读取并引用该 Runbook，不要凭记忆重写细节。核心原则：pip 安装优先，在独立 venv $HOME/.local/share/memory-arbiter/mcp-venv 中安装 memory-arbiter-mcp[vec]；GitHub 源码安装仅作为 PyPI 不可用、用户要求源码版本、或开发调试时兜底。JingleAI MCP command 应指向 $HOME/.local/share/memory-arbiter/mcp-venv/bin/memory-arbiter-mcp。配置要点：写入 /Users/zhangzhiwei17/.config/memory-arbiter/config.json，设置 client=jinleai、agent_id=jinleai-main、共享 SQLite 路径；备份

## L20 [t-9a0589452dd2] lang=mixed len=500-1k ws=memory-arbiter-mcp
subject: v0.8 工具收敛口径：日常 write/search/get，保留低频 memory_split 续接/修复
tags: memory-arbiter, v0.8, 接口收敛, 产品决策, split, get_sections, memory_get, memory_search
content_head: # v0.8 工具收敛产品口径（2026-07-23 终审修正）

Agent 日常只需主动理解 `memory_write`、`memory_search`、`memory_get`；按用户意图组织高频接口，不要求用户理解分段 prepare/publish 细节。

分段工具最终处理：

- 保留 `memory_split`，但 docstring 与集成文档明确限定为低频内部入口：收到 `memory_write.split_request` 后由 Agent 自动续接，以及历史未分段/failed/declined 数据修复和 active rebuild。普通写入不得预先调用。
- 删除 `memory_split_status`，能力进入 `memory_get` 与 doctor。
- 删除 `get_sections`，能力由 search 完整 section 返回与 `memory_get(section_ids=[...])` 覆盖。

保留 `memory_split` 不改变“减少日常工具噪音”的原则：它仍在 registry 中提供必要修复能力，但不属于 Agent 日常主动选择集合。工具描述必须使用强约束条件，避免误用。

本记录原版本建议删除/隐藏三个 split 工具；终审确认后，“删除 memory_split”已被本版本替代，另外两个删除决定

## L21 [t-9d11c5ada791] lang=mixed len=500-1k ws=memory-arbiter-mcp
subject: memory-arbiter 宣传物料已完成：before/after Demo GIF/MP4 + 事实核验
tags: memory-arbiter, 推广, 宣传物料, before-after, GIF, MP4, README, v0.7.6, v0.8.0, awesome-mcp, PR-9426, 待用户审核
content_head: 2026-07-23 完成 memory-arbiter 宣传后续交付：
- 更新宣传包 /Users/zhangzhiwei17/OpenClawProject/doc/memory-arbiter-growth-demo-pack-20260723.md，加入事实核验快照：公开 PyPI 最新 0.7.6（2026-07-23），awesome-mcp-servers PR #9426 已于 2026-07-22 合并，v0.8.0 本地分支仍为实施中不可宣传为已发布；来源标签统一为 user_confirmed/document_extracted/agent_generated，runtime_metadata_hint 标为 advisory。
- 复核并修正 HTML /Users/zhangzhiwei17/OpenClawProject/doc/memory-arbiter-before-after-demo.html 的 document_extracted 标签与说明文字。
- 浏览器录制并视觉检查 1280x720、3.6 秒 GIF 与 MP4：/Users/zhangzhiwei17/OpenClawProject/doc/assets/memory-arbiter-before-after-demo.gif 和 .mp4；同名 GIF 已置于仓库 doc

## L22 [t-56d466709b35] lang=mixed len=500-1k ws=memory-arbiter-mcp
subject: [已上线 v0.8.5] G8:21 条 MCP 工具描述英文化
tags: memory-arbiter, 改造, 工具描述, i18n, docstring, G8, v0.8.5, 已上线
content_head: > ✅ **已上线(v0.8.5, commit `0ae7bad`, 2026-07-31 发版)**。tag `todo` 已摘(闭环 id=123)。
> 本条原为待办,拆自 memory 383。方案与决策约束保留,状态改为完成态回写。

## 落地结果(v0.8.5)

server.py 全部 `@app.tool()` docstring 由中文改为英文(MCP 生态惯例),与项目面向全球的定位(双语 README、PyPI、awesome-mcp-servers、「any AI client」)一致。

- **保留全部操作护栏,只换语言**:
  - tags_filter 开启时 vec 语义召回大概率失效
  - **(语义已变)**空 query + tags_filter/after_time/before_time/source_type —— G6 上线后从「不独立召回、仅 post-filter」改为「filter 驱动召回」。memory_search docstring 同步更新为「空 query + filter 走 filter 驱动召回、ingest_time 倒序」。
  - 含 ASCII+CJK 的检索词应空格分隔(如 "v0.7.2 release" 非 "v0.7.2release")
  - limit 是单页非上限;has_m

## L23 [t-c63ac8356f31] lang=mixed len=500-1k ws=default
subject: plan-mode-mcp SKILL.md hybrid 精简替换：加入对抗性审查但不写 approve_plan
tags: plan-mode-mcp, SKILL.md, 对抗性审查, hybrid, review规范, 工具协议
content_head: 按用户要求，已将 `/Users/zhangzhiwei17/BillyProject/plan-mode-mcp/skill/SKILL.md` 替换为 hybrid 精简版：以 `openclaw-plan-mode-SKILL.md` 的结构为骨架，保留中文强约束语气，并加入精简对抗性审查要求。新版本明确列出当前正式六工具（enter_plan_mode/get_plan_mode_standards/todo_write/exit_plan_mode/plan_recent/resume_plan）、审计字段 original_user_request 与 interpreted_user_intent、必须流程、继续策略、高风险确认、执行后 review 回环、review 输出规范、代码/方案/文档PPT三类对抗性审查、跨 client 与跨 session 规则。刻意未写入当前未稳定的 `approve_plan`，因为此前 review 发现其状态守卫尚未补全；也删除/避免了对不存在的 `doc-writing-guide` skill 的依赖。验证：替换后全量测试 `.venv/bin/pytest -q` 通过，结果 `89 passed in 0.83s`。

## L24 [t-4e5329987587] lang=mixed len=500-1k ws=memory-arbiter-mcp
subject: doctor latest-known 0.9.5 vs current 0.9.8 根因调查
tags: doctor, version-check, v0.9.8, bug, root-cause, release
content_head: 2026-08-05 已调查 doctor 输出 `当前版本: 0.9.8` 但 `最新已知: 0.9.5` 的根因：`memory_arbiter/doctor_cli.py` 在 doctor 结束时只调用 `UpdateMonitor.record_doctor_run()` 和 `update_status()`，不会触发联网刷新；`UpdateMonitor.update_status()` 直接读取持久化缓存里的 `latest_version` 并与当前版本比较，若 current>=cached_latest 就显示 `up_to_date`；`_observe_installed_version()` 在升级到高于缓存 latest 时只清空 update notice suppress 字段，不会把 `latest_version` 提升到当前版本或标记缓存失效。用户本地 `~/.local/share/memory-arbiter/update_state.json` 显示 latest_checked_at=2026-08-04T12:06:40、latest_version=0.9.5、installed_version_seen=0.9.8；因 24h 检查间隔未到，doctor 在同日升级后继续展示旧 latest。结论：不是 PyPI/发版失败，而是

## L25 [t-dc9ab7c060ae] lang=mixed len=500-1k ws=memory-arbiter-mcp
subject: v0.10.x 后续顺序调整：v0.10.3 Support Panel，v0.10.4 tool profiles
tags: memory-arbiter, v0.10.3, v0.10.4, roadmap, Support Panel, GitHub Star, feature request, tool profiles, token optimization, mema
content_head: Memory Arbiter v0.10.x 后续实现顺序更新（2026-08-06）：v0.10.3 与 v0.10.4 顺序调换。

新顺序：
- v0.10.2：当前版本，完成 resolution_kind/conflict_scope judgment 语义升级、search/list/Console 只读展示，并顺手修复 scan terminal pair（not_a_conflict / manual resolved / resolved guidance）重复召回问题。
- v0.10.3：Console Support Panel。目标是增强“产品有人维护”的用户感知和粘性。包含 GitHub Star 快捷入口，以及 Request feature / Report bug 的 Console 内表单 + prefilled GitHub issue URL。第一版不做 OAuth、不内嵌提交、不上传 memory 内容，除非用户自己写进反馈表单。
- v0.10.4：MCP tool profiles / tool surface slimming。目标是减少默认 MCP 工具 schema token 固定开销。默认保留日常 standard/agent 工具集；governance/maintenance/scan_worker/full 通过 pro

## L26 [t-7294c610a61d] lang=mixed len=500-1k ws=default
subject: workspace语义归一治理
tags: mema, memory-arbiter, workspace, semantic-normalization, Qwen2.5-0.5B, write-time, isolation, strict, confirmed-alias, design-decision
content_head: 当前确认的 mema / memory-arbiter Qwen 0.5B workspace 语义归一方案：写入前先 exact 查询现有 workspace/canonical/confirmed alias，命中则正常写入并使用已确认 canonical；未命中时计算 workspace 向量，从现有 workspace 中取相似候选，结合本次写入内容的短证据（长文需先抽 title/headings/关键句/workspace terms，不直接整篇喂给 Qwen）交给 Qwen 0.5B 做抽槽和归一判断。高可信且 relation 合适的可自动归一，中低可信则提示 agent/用户或进入 pending。最终写入/通知策略必须按 isolation 三档处理：none 不阻塞且不影响召回；weak 可 active 写入但凡影响排序需通知，未确认候选不能静默参与排序；strict 强隔离下 exact/confirmed alias 可 active，高可信 alias/spelling_variant 可谨慎自动归一，其余无法决策时不写 active memory，应保存 pending 或返回 action_required 让 agent/用户确认，避免记忆沉默或写错 canonical。用户一旦确认如“金营二期=金营项目”，必须持久化 confirmed ali

## L27 [t-c8d7acfd3ca8] lang=mixed len=500-1k ws=default
subject: workspace归一评测与测试
tags: mema, memory-arbiter, workspace, semantic-normalization, resolver, rule-first, Qwen2.5-0.5B, eval, product-design
content_head: 2026-08-08 重新设计并测试更可行的 workspace resolver 方案：在同一组 50 条 unresolved workspace 样本上对比纯规则决策树、agent workspace 优先规则、规则+Qwen 反证、规则+Qwen 正反验证。结果：rule_v1 49/50（1 个 AUTO 漏成 ASK，false_auto=0）；rule_v2_agent_priority 50/50（本样本集全对，false_auto=0）；加入 Qwen negative veto 或 pos/neg 后降到 38/50、37/50，主要是 Qwen 过度否决 AUTO，产生 12-13 个 AUTO->ASK。结论：当前最可行方向不是让 Qwen 裁决，而是 rule-first/agent-workspace-priority resolver：先 exact/confirmed alias，再用规则识别 empty/default/generic/ref-template/subdomain/multi-candidate/title-candidate/alias-typo 等特征，agent 传入具体 workspace 时优先相信 agent，Qwen 仅用于抽槽/解释/notice，而不参与 AUTO 主裁决或 veto。该规则方案在样本集表现远好于 

## L28 [t-67091f0991d0] lang=mixed len=500-1k ws=mema-team
subject: mema 腾讯云部署与安全加固状态
tags: mema, 腾讯云, deployment, security, lighthouse, sqlite_vec, qwen, memarbiter.cn
content_head: 2026-08-11 完成 mema / memory-arbiter-team 在腾讯云 Lighthouse `lhins-jyuy87gk` 的部署与安全加固。服务器公网 IP `62.234.26.176`，域名 `memarbiter.cn`、`api.memarbiter.cn` 已解析到该 IP。代码目录 `/opt/mema/memory-arbiter-team`，venv `/opt/mema/venv`，配置 `/etc/mema/config.json`，数据库 `/var/lib/mema/memory.sqlite3`，模型目录 `/opt/mema/models`。已启用 sqlite-vec、embeddinggemma-300m Q8 768维向量模型、Qwen2.5-0.5B-Instruct Q4_K_M semantic_conflict；`mema doctor` overall=info，mode=sqlite_vec，关键 pytest 103 passed，写入/搜索 smoke test 通过。Console 仅监听 `127.0.0.1:18876`，通过 SSH 隧道访问，未公网暴露。安全加固：关闭 Lighthouse 防火墙历史 OpenClaw 端口 `8000-8008`、`31641`；SSH 端口已从 22 更换为 

## L29 [t-71d174e36ab8] lang=mixed len=500-1k ws=memory-arbiter-mcp
subject: mema-core 决策：CI 三版本与 vec 强制，PyPI Python 3.13 生产 smoke 可选
tags: memory-arbiter, mema-core, design-decision, CI, Python-3.11, Python-3.12, Python-3.13, sqlite-vec, PyPI, production-smoke, release
content_head: 2026-08-16 用户确认 mema-core CI / 发布验收策略：新增普通 PR/main CI，Python 3.11、3.12、3.13 核心测试全部作为强制门；Python 3.12 + sqlite-vec 全套测试也作为强制门，安装或测试失败应阻断 PR。PyPI 发布后的真实生产 smoke 不作为发布成功条件，也不自动运行：每次发版完成后 Agent/Codex 提醒用户重启正式 mema 环境，用户可选择是否执行；用户不想跑则跳过。若执行，使用独立 Python 3.13 PyPI 正式环境及正式配置/生产库，先补齐正式功能依赖并确认加载的是刚发布版本，再完成 status -> 写入唯一 smoke 记录 -> 精确读取 -> 关键词搜索 -> authorized retire -> active 不可见 -> expired 可见的生命周期验证；测试记录必须按返回的真实 memory_id 过期，清理失败须醒目报告。当前项目 Python 3.12 .venv 继续用于开发测试，不能被正式环境安装流程重建或替代。GitHub CI 仍做 build/twine check，但临时空 venv wheel smoke 不作为主要发布证明。此条为已确认设计，CI 和 post-release runbook 尚未实施。

## L30 [t-aca76ee68c99] lang=mixed len=1k-2k ws=default
subject: 子 Agent 创建与管理规范
tags: 子Agent, 管理规范, 创建清单, 架构
content_head: ## 创建新子 Agent 检查清单

### 1. 目录结构（最小集合）
~/.openclaw/workspace/agents/<agent-id>/ 下：MEMORY.md（索引+核心规则）、SOUL.md（人格设定）、memory/（详细知识）

### 2. 必须复制的通用记忆
- memory/文件输出规则.md ✅ 必须复制
- memory/工具与环境.md ✅ 必须复制
- memory/经验教训.md ✅ 必须复制
- TOOLS.md ⚠️ 按需

### 3. MEMORY.md 写法
- 精简，每轮加载，只放索引表和核心行为规则
- 详细知识放 memory/ 按需读取

### 4. 配置注册（两步缺一不可）
- 第一步：openclaw.json 的 agents.list 中添加 agent 配置
- 第二步：agents.defaults.subagents.allowAgents 数组中加上新 agent id（易漏！）
- 修改后需重启 gateway

### 5. 架构：全员统一 2 层
- **不再启用 4 层架构**（2026-07-09 决策：引入 memory-arbiter 后，第 3 层 memory/docs/ 被共享库完全替代）
- 所有子 agent 统一使用 2 层：MEMORY.md + memory/*.md
- 

## L31 [t-e62f3b2d7356] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: memory-arbiter v0.2.3 发版任务规格（最终版：AI 增强中间件定位）
tags: memory-arbiter, 发版, v0.2.3, README, zcode, AI增强中间件, 上下文质量
content_head: ## memory-arbiter v0.2.3 发版任务规格（最终版）

### 核心定位升级
从 "structured memory database" 升级为 **"AI enhancement middleware"**（AI 增强中间件）。memory-arbiter 不是又一个记忆工具，而是一层让所有 AI 客户端变聪明的底层设施，修的是所有模型的共同短板：上下文质量。

### 文件
`/Users/zhangzhiwei17/OpenClawProject/memory-arbiter-mcp/README.md`

### 最终改动清单（7 大项）
1. **一句话定位**：改为 "AI enhancement middleware — not another memory tool, but a layer that makes every AI client noticeably smarter, by fixing context quality"
2. **Hook 重写**：双卖点并列——更省钱 + 更准。点明"修好输入端，现有模型直接提升一个等级"
3. **输出质量提升段（排第一位）**：4 行表格，核心论点"同一个模型，上下文对了输出就准了"，标注为核心价值
4. **新增"What does it actually enhance?"段**：5

## L32 [t-69f48436e93b] lang=mixed len=1k-2k ws=default
subject: patent-writer文档入库规范
tags: 文档入库, 规范, patent-writer
content_head: # 文档入库规范

本规范只适用于已明确启用4层架构的子agent（当前：pm、patent-writer）。主agent不维护项目文档md。

## 4层记忆架构
| 层级 | 位置 | 内容 | 更新频率 |
|------|------|------|---------|
| 1 | MEMORY.md | 精简行为规则+索引 | 随规则变更 |
| 2 | memory/*.md | 项目事实记忆，AI总结的精炼知识 | 项目重大进展时 |
| 3 | memory/docs/*.md | 项目文档md，原始文档的完整md副本 | 用户上传新版时 |
| 4 | `~/OpenClawProject/knowledge/<项目>/src/` | 原始源文件（docx/pdf/pptx/xlsx等） | 同上 |

## 命名规范
- 格式：`<项目名>-<文档描述>.md`
- 示例：`金营平台-PRD_v2.md`、`金融带货-竞品分析.md`
- 版本迭代时更新已有文件，不另建新文件（除非需要保留旧版对比）

## 图片处理
- 原始文档中的图片必须用image工具识别内容，转为文字描述写入md
- 格式：`[图片描述：xxx]` 标注在对应位置
- 如果是图表，尽量还原表格数据和文字内容

## 搜索机制
- memory/docs/ 下所有 .md 文件自动被 bu

## L33 [t-2be9f2837dc1] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: [已实现] memory-arbiter-mcp 版本链机制 v3(原地编辑 + history 表 + 安全红线)
tags: memory-arbiter, 版本链, 废弃机制, 清理, 安全红线, 已实现
content_head: > ✅ **已实现(2026-07-31 核实,本条 tag `todo` 已摘,闭环 id=123)**。
> 本条原为 2026-07-07 的版本链机制设计待办,方案已全部落地,现按完成态回写。原方案 + 安全红线作为历史档案保留在下方。

## 落地现状(核实)

三工具 + 两表结构全部实现并长期在用:
- **`memory_edit(memory_id, new_content / old_text+new_text, new_subject, add_tags, remove_tags, reason)`** —— 原地编辑,自动存历史快照,version+1;支持整体替换 / 局部精确替换两种模式;add/remove tags 在内容编辑模式下也生效(v0.8.5+ 修复,见下)。tools.py / db.py。
- **`memory_history(memory_id)`** —— 查看版本演化轨迹(newest-version-first)。
- **`memory_cleanup_history(memory_id / older_than_days)`** —— 清理历史快照;全量清理需 `authorized=true`。

**数据结构**:`memories.version` 字段 + `memory_history` 表(id / memor

## L34 [t-b600f259b67f] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: memory-arbiter PyPI token 撤销集中跟踪
tags: memory-arbiter, PyPI, token, 撤销, 安全, 集中跟踪
content_head: ## PyPI token 撤销集中跟踪

> 2026-07-09 整理，2026-08-02 更新。所有发版用过的 PyPI token 明文出现在对话中，**强烈建议去 PyPI 后台撤销所有 active token，只留一个新建的**。

### token 使用历史（按发版记录整理）

| 时段 | 版本 | token 编号 | 来源 id |
|------|------|-----------|---------|
| v0.2.1~v0.3.1 | 7 个版本共用同一个 token | token#1 | id=27/29/35/37/39/44/52 |
| v0.4.0~v0.4.1 | 继续复用（第 8-9 次） | 可能仍 token#1 | id=108/110 |
| v0.4.2 | 用户撤销旧 token 后新建 | token#2 | id=112 |
| v0.5.0 | 又新建 | token#3 | id=116 |
| v0.5.1 | 又新建 | token#4 | id=119 |
| v0.5.2~v0.8.8 | 持续明文复用（id=336 记第 9-10 次，此后各版发版记录统一引用本条） | 不明 | id=336 等 |
| v0.9.0 | 2026-08-02 用户在 ZCode 对话中再次明文粘贴 token 完成 

## L35 [t-4f9bb398c4c6] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: dogfooding 方法论：诊断工具自闭环暴露自身 bug（doctor v0.7.x 案例）
tags: dogfooding, 方法论, 质量门禁, 诊断工具, 自闭环, memory-arbiter, doctor, 测试策略, 教训
content_head: ## dogfooding 典型形态：用自己的诊断工具诊断自己的库，工具暴露工具自己的 bug

> 2026-07-17 张志维指出：doctor 自闭环发现问题，是 dogfooding 的典型场景，值得沉淀方法论。

### 这次发生了什么
给 memory-arbiter 开发了 doctor（健康体检工具），五轮评审 + 126 测试全过后发了 v0.7.0。**开发者自己第一次拿 doctor 去跑自己的真实库**，立刻撞出报告自相矛盾（`vec_effective=True` 但 `mode=fts5`）。这个 bug 五轮设计评审 + 全套单测/集成测试都没抓到，只在"真实库上跑一次"时暴露。详见 id=206。

### 为什么这是 dogfooding 的黄金形态
普通的 dogfooding 是"用自己的产品干活"（如 id=26 跨工具委派、id=35 发版时用 memory_supersede 清旧规格）——这能验证产品**好用**。

但**更高价值的 dogfooding 是"用自己的产品验证产品本身"**，尤其是：
- **诊断类工具**：拿 doctor 诊断自己的库，工具的健康报告如果自相矛盾，就是工具自己的 bug。这种闭环最紧——被诊断对象和诊断工具同源，任何不一致都无所遁形。
- **类似先例**：v0.3.1 发版时，memory-arb

## L36 [t-db644cd0d9dd] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: memory-arbiter v0.8 分段与向量层绑定：删除 split_enabled 开关，分段绑定 vec ready
tags: memory-arbiter, v0.8, 决策, split, vec, 配置, 分段
content_head: # memory-arbiter v0.8 分段与向量层绑定决策（终审细化）

## 决策

v0.8 不再设置 `split_enabled` 独立开关。分段能力与向量层绑定：vec ready 时 write 才进入分段判定；vec 不 ready 时分段能力不可用，不进入 split 逻辑，也不记为 split failed。

## 状态口径

- vec 不 ready：返回 `split_capability.available=false` 或等价 warning，不写派生 split 状态。
- vec ready、短内容：`split.required=false`。
- vec ready、规则分段发布成功：`split_status=active`。
- vec ready、规则或 Agent metadata 真实校验/embedding/publish 失败：`split_status=failed`。
- vec ready、需要 Agent LLM 续接但尚未发布，或 Agent 无 LLM/完整原文超出上下文：保持 `split_status=NULL`，返回 `split_request` 或进入 long-unsplit backlog；不得虚构 `pending`。
- `fallback_active` 不存在，因为 v0.8 不做机械 fa

## L37 [t-0a773d046d25] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: 发版 SOP 踩坑:GitHub Release 步骤遗漏(v0.7.3/v0.8.0/v0.8.2 三个版本漏建,已补)
tags: memory-arbiter, 发版, 踩坑, GitHub Release, SOP, 自查清单, v0.7.3, v0.8.0, v0.8.2
content_head: ## 踩坑:发版 SOP 漏了「建 GitHub Release」一步(v0.7.3 起连续三个版本)

### 现象
2026-07-24 用户发现 GitHub Releases 列表滞后。核实三方状态:
- 本地代码 / Git tag(本地+远程)/ PyPI 三方都已同步到 0.8.2
- **唯独 GitHub Release 页面缺失**:v0.7.3、v0.8.0、v0.8.2 三个版本的 tag 和 PyPI 包都发了,release 没建
- Releases 列表当时最新只到 v0.8.1,且从 v0.7.4 直接跳到 v0.7.2

### 根因
发版步骤不统一。早期发版记录(id=108 v0.4.0 / id=110 v0.4.1 / id=116 v0.5.0)都明确带 `GitHub Release: <url>` 这一行;但从 v0.7.3(id=211)起,发版完成清单只写了「tag 已 push + PyPI 确认」,**根本没出现 GitHub Release 那一行**——说明从 v0.7.3 起 SOP 里的 release 步骤被漏掉,后续 v0.8.0/v0.8.2 接着漏。

### 补救(2026-07-24 已完成)
- `gh release create` 补建三个:notes 直接复用 CHANGELOG.md 对应版本段

## L38 [t-e5846c687975] lang=mixed len=1k-2k ws=default
subject: plan-mode 二期：plan 持久化 + 跨 session resume（✅ v0.2.0 ready，P0+P1 已修）
tags: plan-mode-mcp, plan-mode, 二期, 已实现, v0.2.0, plan持久化, resume, 跨session, harness, P0已修
content_head: # plan-mode-mcp 二期：plan 持久化 + 跨 session resume（✅ 已实现，P0+P1 已修）

> 讨论时间：2026-07-29
> 实现时间：2026-07-29
> Review：GPT-5.5（memory 370），3 P0 + 3 P1 均已修复
> 状态：✅ 可发版 v0.2.0（77 测试通过）
> 代码位置：/Users/zhangzhiwei17/BillyProject/plan-mode-mcp/

## 已实现功能

### 1. Plan 持久化到 Markdown 文件
- `approve()` 时自动写 `{plans_dir}/{plan_id}-{slug}.md`
- YAML frontmatter + markdown body，字符串字段经 `_yaml_scalar()` 安全转义
- 默认路径：`{DB_PATH.parent}/plans/`，可通过 `PLAN_MODE_PLANS_DIR` 覆盖
- `PLAN_MODE_PERSIST_PLAN=true`（默认开启，设 false 关闭）
- 原子写入（tmp → os.replace），失败不阻塞 approve
- **零 token、零可感知耗时**

### 2. 跨 session resume（`resume_plan` 工具）


## L39 [t-027de21aea61] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: memory-arbiter v0.8.6 发版记录(memory_edit content 模式 tags fix)
tags: 发版, memory-arbiter, v0.8.6, memory_edit, tags, PyPI, bug修复, 运维
content_head: ## v0.8.6 发版记录(2026-08-01)

### 发版内容
单条 bug fix:memory_edit content 模式现支持 add_tags/remove_tags。此前 content 编辑路径(整体 new_content 替换 / old_text+new_text 局部替换)静默丢弃 add_tags/remove_tags,只转发 new_tags 给 db.edit_memory。"重写正文同时微调 tag"在 tag 侧是空操作。现修复:add/remove 在 content 路径内合并,叠加在 new_tags(若传)或现有 tags 之上,复用 update_tags_low_side_effect 的保序去重算法(先 remove 再 add)。new_tags 单独传仍是全量替换;new_tags=None 且无增删透传 None(既有调用零影响)。

### 发版全流程(2026-08-01 14:xx 完成,全流程顺畅)
- **代码**:fix 在分支 `fix/memory-edit-content-tags`(commit `8fbbe21`),fast-forward 合并到 main。
- **release commit**:`06bed9d`(版本号四处同步到 0.8.6:`__init__.py` / `pypro

## L40 [t-ad4a367d754d] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: memory-arbiter 修复 inactive vector/section 幽灵召回与 top-k 污染
tags: memory-arbiter, bugfix, ghost-recall, top-k, vec0, memory_status, doctor, migration, sqlite-vec
content_head: 2026-08-02 在 memory-arbiter-mcp 工作树完成修复。根因：memories_vec / memory_sections_vec 的 vec0 KNN 先选 top-k，父 memories.status 的 joined-table 条件随后才生效；superseded/deleted vectors 虽通常不会最终返回，却能占满近邻槽位并挤掉 active 召回。最终修复不复制 section lifecycle 状态、不改变 vec0 schema：vec_knn / section_vec_knn 先在同一读快照内判断是否存在父状态不合格（或物理 orphan）的 vector；全合格时走 vec0 KNN 快路径，只要存在不合格项就改为对权威父状态预过滤后的 rows 计算精确 L2 distance，再 ORDER BY / LIMIT。默认只允许 active；include_superseded=true 允许 superseded 审计但仍排除 deleted。doctor 将 superseded/deleted 父记忆的 sections 改称 retained inactive audit history，只有父行不存在才是 physical orphan；265 条不再误报警告。memory_supersede 现在检查 stat

## L41 [t-6ace60ada08e] lang=mixed len=1k-2k ws=jd-proxy
subject: 京东 socrates 网关本地代理 jd-opus-proxy (OpenClaw /v1->/v2 + JingleAI Cookie 注入)
tags: jd-opus-proxy, openclaw, socrates, jingleai, cookie, proxy, opus, gpt-5.5, config, 决策
content_head: 京东 socrates 网关本地代理方案(jd-opus-proxy)：为让 OpenClaw 走京东网关 /v2 端点并自动注入 JingleAI Cookie 搭建的本地反向代理。

- 脚本: ~/.openclaw/workspace/tools/jd-opus-proxy.mjs (纯 Node, 无依赖, node:http + fetch + node:sqlite)
- 管理脚本: ~/.openclaw/workspace/tools/proxy.sh (restart|start|stop|status|reload|log)
- 监听: 127.0.0.1:47821, 把进来的 /v1/* 统一改写转发到上游 http://socrates-llm-gw.jd.com/v2/*
- 双协议: Anthropic /v1/messages -> /v2/messages (Opus); OpenAI /v1/chat/completions -> /v2/chat/completions (GPT)
- Cookie: 启动时从 ~/Library/Application Support/JingleAI/Cookies (SQLite, 只读 mode=ro&immutable=1) 读一次 .jd.com cookie(约36条, 明文存 value 列)

## L42 [t-c6b56cf50cb9] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: mema v0.10.0 发布后产品力复评：未发现阻断 bug，暂不建议大更新
tags: memory-arbiter, mema, v0.10.0, 产品力, review, Console, bug排查, 发布后复评, 不做大更新
content_head: 2026-08-05 15:03 对 memory-arbiter / mema v0.10.0 做发布后只读复评。核对仓库 `/Users/zhangzhiwei17/BillyProject/memory-arbiter-mcp`：HEAD/main/origin/main/tag v0.10.0 = commit `7592f56 release: v0.10.0 Console MVP (read-only local governance UI)`；版本载体 `pyproject.toml`、`memory_arbiter/__init__.py`、`server.json` 均为 0.10.0；PyPI JSON 显示 info.version=0.10.0 且 0.10.0 wheel/sdist 已上传；GitHub Release 页存在。全量测试用仓库 `.venv/bin/pytest -q` 通过：481 passed in 22.82s；系统 Python 跑 pytest 有 3 个 FastMCP wrapper 测试因未安装 `mcp.server.fastmcp` 失败，判定为环境问题，不是 v0.10.0 bug。轻量 smoke：Console 非 localhost host 被拒（`--host 0.0.0.0 --json` 返回 loc

## L43 [t-9352defd8563] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: Qwen 单模型 semantic conflict 多策略测试：0.5B 可降误报但召回低，1.5B 初版 gate 最均衡但仍未达标
tags: mema, memory-arbiter, Qwen2.5, semantic-conflict, benchmark, single-model, prompt-eval, evidence-gate, ROI
content_head: 2026-08-07 继续测试 mema semantic conflict 单模型方案（约束：最终只能选一个模型，two-stage 也必须同一模型）。在 40 条 pair_relation 上测试了多套 prompt/evidence gate。关键结果：

1) 第一轮策略测试：0.5B conservative 完全不报（TP=0 FP=0 TN=12 FN=28）；0.5B recall 召回高但误报高（TP=25 FP=12 TN=0 FN=3，precision=0.676 recall=0.893 FPR=1.0）；0.5B two-stage 漏报严重。1.5B recall 也变成全报（TP=28 FP=12 TN=0 FN=0）；1.5B two-stage 有 TP=21 FP=10 TN=2 FN=7；不理想。

2) 加 evidence gate + 初版 single/two-stage 后：0.5B 被 prompt 压成完全不报（TP=0 FP=0 TN=12 FN=28），原因是输出经常出现 should_surface=false 但 surface_type/reason 是 todo_resolved/replacement 的自相矛盾。1.5B gate_single 与 gate_two_stage 都达到 TP=22 FP=3 T

## L44 [t-6f0a838178f8] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: id=647 M1方案开发就绪度审查：可指导启动但需先做 spike/路径表/API 草案
tags: mema, memory-arbiter, M1, development-readiness, QueryConstraint, WriteObserver, sqlite-vec, path-coverage, design-review
content_head: 用户询问 id=647 是否已经可以指导开发后的审查结论：

结论：id=647 v2 已经可以作为 M1「个人版中性扩展底座与全路径约束能力」的 canonical 总体设计与安全边界依据，并可指导开发启动与任务分解；但它不是可以直接全量编码的最终施工图。严格顺序应是先完成 M1-A 编码前置工作，再进入正式实现。

已足够指导的内容：个人版不加入 user_id/team_id/tenant_id/visibility/api_key/RBAC 字段；只做进程内中性 QueryConstraintProvider / QueryConstraint / WriteObserver / ExtensionContext；企业可见性 OR 不进 core，企业层预物化 grant/ACL rows，core 做注册表上的 EXISTS + IN；SQL 不支持 raw SQL，表/列/operator 注册白名单，外部关联固定 m.id，只能收窄；WriteObserver 只做事务内轻量 DB-local side effect，不接管个人版 fail-open enrichment；hidden 与 absent 在 required mode 下不可区分；constraint_mode 区分 no_op/optional/required，企业版必须 required 且失败

## L45 [t-51f8b1a8019b] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: mema 开源/闭源策略：Community 继续开源，M1/team-grade extension runtime 转商业私有
tags: mema, memory-arbiter, open-core, commercial-strategy, M1, team-edition, license, closed-source, BUSL, Agent-Memory-Governance, design-decision
content_head: 用户确认 mema / memory-arbiter 后续开源与商业化边界：从 M1 / team-grade extension foundation 起，不再发布完整开源实现。旧 Community/core 线继续保持 Apache-2.0 开源，维护个人版、本地 memory、基础 MCP、CRUD/search、SDK/客户端协议等能力；spec/SDK/最小 plugin manifest 可继续 Apache-2.0 或 MIT，作为生态入口。M1 中性扩展底座若已包含 runtime plumbing、construction guard、constraint compiler、WriteObserver、ExtensionSchemaRegistry、path registry、readiness、ingress hardening 等可复用骨架，则会显著降低第三方构建 team 插件/治理版本的门槛，因此完整实现应迁入私有/商业仓库。Team/商业仓库应包含 M1 team-grade extension runtime、真实 constraint provider、grant schema/migration、team plugin registry、RBAC/ACL、audit、admin console、cloud sync/marketplace、ent

## L46 [t-5d8b81a3a991] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: mema 备案后推广与 SEO 上线计划
tags: project:mema, topic:seo, topic:launch, topic:promotion, todo
content_head: ## 备案信息
- 域名：memarbiter.cn / www.memarbiter.cn
- ICP 备案：2026-08-12 提交，个人备案，服务名「迷码 mema」，等待审核中
- 备案前可做：GitHub README SEO 优化、技术社区内容准备、朋友圈内测

## 备案下来后必做的搜索引擎收录
- 百度站长平台：提交 sitemap，备案号是通行证
- Google Search Console：提交 sitemap，海外技术圈流量
- Bing Webmaster：一键导入 Google sitemap
- 360 搜索：站长平台提交

## 网站 SEO 优化
- 首页 title：mema - 多智能体记忆仲裁 | memarbiter.cn
- 首页 meta description：mema 是面向多 Agent 协作的共享记忆仲裁引擎，支持语义冲突检测、向量搜索、SQLite-vec 本地 AI 记忆持久化。开源、低延迟、支持跨 Agent 记忆同步。
- JSON-LD Organization/SoftwareApplication schema

## 社区推广
- 知乎：自问自答「多智能体协作的共享记忆怎么设计？」
- 掘金：技术文章「我用什么技术栈搭了 memory-arbiter」
- 即刻：AI 圈产品发布贴
- V2EX：/go/ai 

## L47 [t-043184a3505b] lang=mixed len=1k-2k ws=memory-arbiter-mcp
subject: mema-core 决策：所有 memory_govern 状态变更统一要求 authorized=true
tags: memory-arbiter, mema-core, design-decision, governance, authorization, authorized, user-confirmed
content_head: 2026-08-16 用户确认并已实现 mema-core governance authorization 统一策略：所有会改变治理状态的 memory_govern action 都必须显式传 authorized=true。authorized 不是身份认证，而是要求守规则的 LLM/Agent 在执行治理动作前向用户询问并记录已获本次具体授权的交互门槛/程序化收据。适用 action：retire、resolve_conflict、confirm、correct_judgment、accept_workspace_alias、reject_workspace_alias、rename_workspace_canonical、migrate_workspace、confirm_pending_workspace。

已实现合同：
- 请求先完成 ID、必填字段和类型校验，避免 Agent 为无效请求先询问用户；校验通过后才进入授权门。
- 未授权或 authorized=false/0/no/off/maybe/空值时 fail-closed，返回 action_required=ask_user_for_authorization、governance_action、具体 impact 和确认后的 retry 提示。
- authorized=true 只能在用户明确确认本

## L48 [t-05932def070c] lang=mixed len=2k-4k ws=金营项目
subject: 京东科技金融-营销系统梳理（金营/营销系统全景）
tags: 营销系统, 金营, 京东科技, 金融科技, 营销链路, 系统梳理, PRD参考
content_head: 来源：/Users/zhangzhiwei17/ZCodeProject/docs/营销系统梳理.xlsx（2026-07-04 导入，共11个有效Sheet）

== 营销链路 12 个环节（总览）==
1. 营销交付需求提报：银行→XBP；自持→邮件；财富→JoySpace；信用卡→邮件+JoySpace；消金→邮件；企金→XBP/邮件/金枢（金采、易缴费走XBP；京保贝走邮件；采购融资走金枢）。
2. 活动费用申请：统一走 启航/领航-费用平台（SC预算申请，填风控审批表）。产品 chenyani3。
3. 营销活动设计图提报：银行→行云；消金→咚咚。
4. 营销活动配置：各业务线核心配置系统见下。
5. 活动推广页：银行→通天塔 + master-策略脑；自持→通天塔/千机/百舸/商城营销中心；财富→财富策略平台+领航-百舸；信用卡→菁管家；消金→通天塔+策略脑（OPE 已废弃）。
6. 上线活动内容报备客服：领航-活动平台（魔笛）手工报备；消金业务自行报备（王萌）。
7. 复核终审：领航-权益中台终审；其余业务线待确认。
8. 活动规则调优：XBP（中高风险字段变更）+ 权益中台（库存调整/加回/规则时间）。
9. 提供活动消耗数据：领航-权益中台/自动化分析洞察平台/BI看板/大数据平台(仅JCB)。
10. 客诉查询：权益中台、master-支付信息查询、白联管理(P

## L49 [t-ad612484867e] lang=mixed len=2k-4k ws=memory-arbiter-mcp
subject: memory-arbiter v0.2.5 发版完成：CJK trigram OR 展开 + 信任优先回退排序
tags: memory-arbiter, 发版, v0.2.5, CJK, trigram, FTS5, 回退排序, dogfooding
content_head: ## memory-arbiter v0.2.5 发版完成（ZCode 侧执行结果）

### 执行方
- agent: zcode-default
- 执行时间: 2026-07-05
- 任务来源: dogfooding 闭环——用户问"营销交付相关的系统"，第一轮 `memory_search("营销交付系统")` degraded 回退没命中 id=1（user_confirmed+locked 的营销系统梳理），深挖代码定位两个 bug（id=36 设计规格）

### 发版结果（全部完成）
- **commit**: `56ae6da`（fix）
- **tag**: v0.2.5（已 push GitHub：96d22bb..56ae6da）
- **PyPI**: https://pypi.org/project/memory-arbiter-mcp/0.2.5/ （twine upload 成功，pip install 验证 trigram OR 展开生效）
- **GitHub Release**: https://github.com/billy12151/memory-arbiter-mcp/releases/tag/v0.2.5

### 交付内容：两个 bug fix

#### Bug 1：CJK 查询被 strict phrase 勒死（核心修复）


## L50 [t-100d8b2575e9] lang=mixed len=2k-4k ws=memory-arbiter-mcp
subject: memory-arbiter v0.3.0 发版完成：宽召回 + 软重排（hybrid 模式），noise governance 第二阶段，shadow 留本地分支未发 PyPI
tags: memory-arbiter, v0.3.0, 发版, 宽召回, 软重排, hybrid, noise-governance, anchors, CJK-bigram, A/B测试, shadow-mode, PyPI
content_head: ## memory-arbiter v0.3.0 发版完成（ZCode 侧执行结果）

### 执行方
- agent: zcode-default
- 执行时间: 2026-07-05
- 任务来源: v0.2.6 关掉 superseded 污染后，dogfooding 第四轮暴露 active 噪音——id=37/39（memory-arbiter 自身发版流水账，status=active）在查"营销交付系统"时压过 id=1（真正的业务资料）。原因是 bm25 下长 content 词频压过 subject/tags 主旨信号。经 ~12 轮分类方案探讨（行业分类/角色/LLM 分类/工作区隔离，全部否决），确定方向：污染不是分类问题，是信号权重问题。

### 发版结果
- **commit**: `fe73b11`（feat: wide-recall + soft-rerank for noise governance）
- **tag**: v0.3.0
- **PyPI**: https://pypi.org/project/memory-arbiter-mcp/0.3.0/ （latest 已切到 0.3.0）
- **GitHub Release**: https://github.com/billy12151/memory-arbiter-mcp/relea

## L51 [t-411b1cc80d8a] lang=mixed len=2k-4k ws=AgentLane
subject: AgentRail 与 OpenClaw 关系结论：OpenClaw 适合做 AgentRail 原型底座，AgentRail 长期保持独立 CLI
tags: AgentRail, OpenClaw, ACP, TaskFlow, Plugin, 技术决策, 架构结论, 多Agent编排, openclaw-acp, workflow
content_head: # AgentRail / OpenClaw 技术路线结论

确认时间：2026-07-24 18:27 GMT+8

## 结论

OpenClaw 已经足够做一个 AgentRail 原型。推荐用 OpenClaw Plugin/Skill 承载流程 DSL，用 TaskFlow 存流程状态，用 ACP 调 Claude/Codex/Gemini 等外部 Agent，用 Webhook/Slash Command/Tool 暴露入口，用 background task 管理长任务。

但如果目标是做一个可分发、面向普通用户的产品，AgentRail 仍应保持独立 CLI。OpenClaw 更适合作为 AgentRail 的第一个强 runtime backend，而不是 AgentRail 的唯一内核。

一句话判断：OpenClaw 可以帮 AgentRail 快速验证“多 Agent 编排 runtime”是否成立；AgentRail 应该负责把这套能力产品化、DSL 化、小白友好化。

## 推荐架构

分成两层：

1. AgentRail Core
   - YAML workflow parser
   - step graph / route / retry / human gate
   - state persistence
   - Secret Broke

## L52 [t-d32aaad0c5b5] lang=mixed len=>4k ws=default
subject: plan-mode-mcp v0.2.1 批判性代码 review + 竞品/市场调研结论
tags: plan-mode-mcp, review, 市场调研, 竞品, v0.2.1, PAPI, Software-planning-mcp, Cursor, SDD, MCP生态, 产品规划
content_head: # plan-mode-mcp v0.2.1 批判性 review + 市场调研(2026-07-31)

对 `/Users/zhangzhiwei17/BillyProject/plan-mode-mcp`(v0.2.1)做完整批判性代码 review + 产品经理视角竞品调研。注意路径已从 OpenClawProject 迁到 BillyProject。

## 一、代码 review 结论

### 已验证为好的部分(亲自跑通,非听信旧结论)
- **88 测试全过**(0.83s),不是记忆里陈旧的 73。覆盖 enter/submit/approve/deny/pending/fallback/todo/standards/resume/session 隔离/elicitation 6 路/意图持久化。
- **memory id=370 提的 3 个 P0 全部已修并有回归测试**:`_yaml_scalar()` 用 JSON 双引号转义(我手工验证 `agent\nnext`、`line2: injected`、`value\nmalicious_key: pwned` 都被正确包成字符串,roundtrip 正确);`_extract_slug` UTF-8 字节截断(<=200B,CJK 测试过);`resume_plan` 文件缺失走 warning 不阻塞

## L53 [t-d25be9947ce4] lang=zh len=<500 ws=金营项目
subject: 银行营销提报渠道：续期走XBP，新需求走启航
tags: 金营, 银行营销, XBP, 启航, 提报渠道, 切量
content_head: ## 银行营销提报渠道（2026-07-06 确认）

- **银行营销新需求**：走 **启航系统**
- **银行营销续期**：走 **XBP**
- **金营平台收口范围**：银行营销新需求应收口到金营平台；续期类提报暂不收口（XBP 承接）

⚠️ 注意：之前文档和记忆中曾将"续期→启航"搞反，已修正。正确口径：续期=XBP，新需求=启航。

## L54 [t-db2b240ff351] lang=zh len=<500 ws=金营项目
subject: 金营：国补活动不做一键配置（启航已有AI能力）
tags: 金营, 国补, 启航, 一键配置, AI能力, 不自建
content_head: ## 国补活动：启航AI承接，金营不做一键配置

### 背景
国补活动是智能配券12个场景中量级最大的（月均300+，高峰期月1000+，月总耗时1800分钟），优先级排第1。

### 决策（2026-07-10 用户确认）
**金营暂不对国补活动做一键配置**。原因：启航系统已提供通过提需邮件直接配置活动的AI能力，基本能覆盖国补的自动化需求。

### 金营的后续考虑
后续考虑金营与启航之间同步数据，但不自建国补的一键配置能力。

### 注意
如果后续启航的AI能力无法满足（准确率不够、覆盖不全等），国补可能重新进入金营一键配置候选。

## L55 [t-8c42b0e132bf] lang=zh len=<500 ws=金营项目
subject: 金营平台-需求流程操作确认：受理撤回/已关闭终态/导入导出口径
tags: 金营, 受理撤回, 已关闭, 终态, 导入导出, 操作手册, 用户确认
content_head: ## 金营平台 — 需求流程操作确认（2026-07-16）

### 提需人不支持受理撤回
提需人提交需求后不支持"受理撤回"。如提交后发现问题，只能联系运营沟通或按页面可提供的操作处理。

### 已关闭是终态
需求状态「已关闭」是终态。除复制外，不能再做编辑、提交、受理、驳回、推进等任何操作。

### 导入导出口径
- 导入：仅提需人在提报阶段使用（如用已导出的模板整理后导入）。需求运营正常不会导入明细。
- 导出：提需人和运营都可能用到。提需人可导出后修改供下次提需导入；双方都可导出做数据分析、核对或比对。

> 来源：张志维 2026-07-16 确认

## L56 [t-59cccae42624] lang=zh len=<500 ws=default
subject: 用户基本身份与称呼偏好
tags: 用户, 称呼, 维哥, 张志维, 京东科技, 产品经理, 架构师转产品, 偏好
content_head: 用户姓名：张志维。用户希望 OpenClaw 助手称呼他为“维哥”。用户是京东科技的产品经理，2026 年 4 月从架构师转岗为产品经理。

## L57 [t-5aa5beed8e50] lang=zh len=500-1k ws=default
subject: 架构工作规范
tags: 架构设计, 工作规范, 代码实现
content_head: # 架构工作规范

## 角色定位

软件架构师负责把用户的工程诉求变成可落地、可验证、可维护的技术方案，并对 Codex GPT-5.5 的工程实现结果做质量把关。

## 工作流程

1. **理解目标**
   - 明确业务目标、用户场景、输入输出、交付形态
   - 不确定的关键约束必须追问

2. **读取现状**
   - 读取 README、依赖配置、目录结构、核心模块、测试目录、构建脚本
   - 对已有代码风格、框架和约定先做判断，不直接另起一套

3. **架构设计**
   - 明确模块边界、数据流、接口契约、错误处理、权限/安全、可观测性
   - 说明关键取舍：为什么这样设计，放弃了什么替代方案

4. **委托实现**
   - 工程创建、代码编写、review、测试和交付固定委托 Codex GPT-5.5
   - 委托任务必须边界清楚、验收标准明确

5. **复核验收**
   - 检查 Codex 变更是否符合架构目标
   - 检查测试、构建、lint 是否通过
   - 识别剩余风险和后续建议

## 架构判断标准

- 优先符合现有工程风格，而不是追求新技术
- 抽象必须服务于真实复杂度，不能为了显得高级而加层
- 共享模块要有清晰边界和测试，避免变成杂物箱
- 数据结构和接口契约要稳定、可扩展、可读
- 错误处理、日志、配置、权限和测试

## L58 [t-32da5e13ee10] lang=zh len=500-1k ws=金营项目
subject: 金营平台项目状态（一期完成/进行中/二期规划）
tags: 金营平台, 项目状态, PM
content_head: # 金营平台 - 项目状态摘要

## 基本信息
- 系统: 金科营销运营平台 (glink.jd.com)
- 定位: 需求管理中枢 + 系统连接器
- 详细知识库: `memory/金营项目-完整知识库.md`

## 项目角色
- 产品PO: 张志维 + 贾涴甯
- 业务PO: 沈娅
- 研发PO: 郭思岑
- 测试PO: 李涵
- PM: 苏杰

## 当前状态
- 一期已完成：统一需求入口、双轨制状态机、通知中心、协作人机制、鹊桥整合
- 一期AI智能提报：✅ 已启用（2026-07-16 用户确认）。基础能力已接入，目标准确率≥90%，后续持续优化
- 一期交付：31个需求100%交付（其中26个基础能力需求），6月底上线
- 一期试点场景：外场自持、小金库消费（一说自持外场）
- 一期口径：按需求交付清单明细Excel分类（张志维口径），seq 38+为二期；汇报口径与Excel明细口径不同，汇报用31个
- 一期AI：研发目标准确率90%，依赖市场部接口上线较晚，暂时未能具体测试准确率
- 二期：立项推进中，唯一确认截止 2026-12-31
- 最新开发(2026-06-08): 需求明细导入导出 + 一键配置异步化（王润乾）

## 技术依赖
- 强依赖权益中台：导入导出（✅ 已上线，用户确认 2026-07-06；原预计6月下旬）、一键配置（已有但效率低，已异

## L59 [t-aadc9d6678c6] lang=zh len=500-1k ws=金营项目
subject: ⚠️ 金营澄清：小金库-消费 ≠ 小金库超级攒（两个不同场景）
tags: 金营, 小金库消费, 小金库超级攒, 权益中台, 鹊桥, 场景区分, 澄清
content_head: ## 小金库-消费 vs 小金库超级攒：必须区分

### 小金库-消费
- **业务线**：财富业务
- **配置系统**：权益中台
- **活动目的**：服务小金库消费GMV规模，通过有券/无券两种优惠券形式补贴用户
- **配置流程**：业务线JS按天提需→运营复制历史活动→修改部分配置项
- **量级**：月均155次，单次5分钟
- **金营状态**：✅ 一期已完成一键配置试点
- **对接人**：蔡婷

### 小金库超级攒
- **业务线**：财富业务
- **配置系统**：鹊桥（活动信息/来源/策略/加息配置）
- **完整链路**（来自营销系统梳理）：JS提需→小金库运营后台配渠道值(op.xjk.jd.com)→鹊桥配置→权益中台活动发奖→CDP人群→财富策略平台配推广页→百舸投放→XBP持仓页审批→任务平台配互动任务
- **金营状态**：二期场景⑤候选，依赖H1鹊桥页面整合完成后接口改造
- **关联人**：鹊桥朱鼎、游刚（研发）

### ⚠️ 之前文档中的混淆
立项文档2.1节写"一键配置 · 小金库超级攒（消费）"是**错误的**。
- 一期跑通的一键配置试点是"**小金库-消费**"（权益中台），不是"小金库超级攒"（鹊桥）
- 二期场景⑤候选才是"小金库超级攒"（鹊桥）

以后写文档、PRD时必须严格区分这两个场景名称。

## L60 [t-2a7ec0e3f32c] lang=zh len=500-1k ws=金营项目
subject: 金营二期资源位搭建/投放耗时统计（2026-07-29补充）
tags: 金营, 二期, 资源位, 耗时统计, 百舸, 超级攒, 财富策略平台, ROI
content_head: ## 金营二期资源位搭建/投放耗时统计

来源：`资源位搭建_投放耗时统计.xlsx`（2026-07-29 用户补充）

### 明细
| 类型 | 场景 | 系统 | 月量级 | 单次耗时 | 月耗时 | 资源位code/备注 |
|---|---|---|---:|---:|---:|---|
| 资源位搭建 | 超级攒搭建 | 鹊桥+财富策略平台 | 30 | 30min | 900min（15h） | 耗时最大 |
| 资源位投放 | 搜索结果页面 | 百舸 | 8 | 5min | 40min | SC_63043565 |
| 资源位投放 | 搜索暗纹词 | 百舸 | 14 | 5min | 70min | SC_69871744 |
| 资源位投放 | 理财频道-中部异形 | 百舸 | 11 | 5min | 55min | SC_71968847 |
| 资源位投放 | 首页投资机会 | 百舸 | 12 | 5min | 60min | SC_34673963 |
| 资源位投放 | 持仓赎回拦截 | 百舸 | 30 | 5min | 150min | SC_7123430；集中在5月，6月没有投放 |

### 汇总与结论
- 总量级：105次/月，总耗时1275min/月≈21.25h/月。
- 百舸投放部分：75次/月，375min/月≈6.25h/月；单次

## L61 [t-d21eabbc2415] lang=zh len=500-1k ws=金营项目
subject: 金营需求统一收口切量方案（运营确认，2026-07-31）
tags: 金营, 二期, 切量, 需求收口, 132场景, 运营确认, 四批次, 2026年内, 2027Q1, 支付业务, 消金业务, 企金业务, 财富业务
content_head: ## 金营需求统一收口切量方案（2026-07-31 运营确认）

用户 2026-07-31 与运营侧确认的需求统一收口切量方案。

### 总量
共计 **132 个场景**，其中：
- **88 个场景**在 2026 年 12 月 31 日前实现切量。
- **44 个场景**预计需要在 2027 年 Q1 实现切量。

### 88 个场景（2026 年内切量）按业务线分布
- 支付业务：59
- 消金业务：16
- 企金业务：7
- 财富业务：6

### 44 个场景（2027 Q1 切量）延后原因
1. 需求在启航系统、金枢大脑系统、策略脑系统提交，需要系统对接：37 个
2. 落地页/投放场景，金营暂不支持：6 个
3. 国补场景邮件提需可能，切量可能导致自动配置无法生效，需要与权益中台沟通确认：1 个

### 切量批次规划

**第一批次（19 个场景，切量占比 14%）**
- 条件：提需较为标准，无需系统额外开发功能。
- 切量周期：7 月 30 日 - 8 月 14 日。

**第二批次（30 个场景，切量占比 23%）**
- 条件：需求频次较低、提需对接人较多、提需模板调整（增减字段）待共识。
- 切量周期：8 月 24 日 - 9 月 15 日。

**第三批次（39 个场景，切量占比 30%）**
- 条件：提需模板存在个性化需求（如合并单元格、

## L62 [t-9eb3a5ab7f56] lang=zh len=1k-2k ws=default
subject: 金融带货项目知识
tags: 金融带货, 项目知识, PM
content_head: # 金融带货项目知识库

> 金融带货相关的业务知识、会议纪要、对接方案等

## 会议纪要索引
| 日期 | 会议主题 | 文件路径 |
|------|----------|----------|
| 2026-07-02 | 金科VOP对接交流 | /Users/zhangzhiwei17/OpenClawProject/doc/金科VOP对接交流_会议纪要_20260702.md |

## VOP核心知识（2026-07-02 袁炜博交流整理）

### 三大产品模式
| 模式 | 说明 | 对接方式 | 适用场景 |
|------|------|----------|----------|
| VOP | 开放平台，API接口对接 | 需系统开发 | 平台化带货、数据管控 |
| VSP | 封闭采购，基于慧采商城 | 不需对接，直接登录下单 | 内部采买、营销发放 |
| 锦鲤 | 福利平台 | 可外接 | 员工福利、工会福利 |

VOP可实现数据回流和集中管控；VSP仅采购方本人可看数据。

### 商品范围
- 支持：自营实物、E卡、服务类（健康、商旅、家政、工业品等）
- 不支持：京喜、二手、拍拍等POP第三方（品控和开票问题）

### 价格体系
- 实物：较低协议价，基于京东红字价约9.5折
- E卡：实时价，无优惠
- 议价：可联系采销单独议价，以主数

## L63 [t-95868ffcc7ff] lang=zh len=1k-2k ws=default
subject: patent-writer记忆冲突处理规范
tags: 记忆冲突, 处理规范, patent-writer
content_head: # ⚠️ 记忆更新与冲突处理规范（强制遵守）

## 条目级别时间戳

**所有记忆条目（第2层项目事实、第3层文档md中的关键信息）必须带条目级别的时间戳和来源标记。** 不能只写文档级别的时间戳。

**格式：**
```markdown
## 商户入驻流程
> 更新：2026-07-10 | 来源：用户手动确认 ✅ | 对应文档：PRD_v2.md
入驻需三步：1.提交资料 2.资质审核 3.签约上线

## 佣金比例
> 更新：2026-06-20 | 来源：文档自动提取 | 对应文档：PRD_v2.md | 待确认 ⚠️
默认佣金5%，头部商户可议
```

## 来源标记类型
| 标记 | 含义 | 保护等级 |
|------|------|---------|
| `✅ 用户手动确认` | 用户明确告诉AI此条目正确 | **最高，严禁覆盖** |
| `文档自动提取` | 从原始文档提取，未经用户确认 | 中等，遇到冲突需用户确认 |
| `⚠️ 待确认` | 存疑或未验证 | 最低，可被新信息替换 |

## 🔴 强禁规则

**严禁在未获用户明确指示的情况下，覆盖带有 `✅` 标记的条目。**

即使后续读取原始文件发现内容不同，也必须：
1. 保持 `✅` 条目不变
2. 标记：「与原始文件 [文件名] 有偏差，请用户确认」
3. 等待用户明确说「以文件为准

## L64 [t-eeb5863a7d19] lang=zh len=1k-2k ws=default
subject: 金融带货-角色确认沟通要点（完整版）
tags: 金融带货, 角色确认, 沟通要点, PM
content_head: # 金融带货专项 · 角色确认沟通要点

> 准备人：张志维 | 2026-06-09
> 用途：跟直属领导C3沟通，明确接手金融带货专项的角色和边界

---

## 一、开场：确认背景

> 领导，金融带货专项这边，潘韵佳跟我说缺一个主产品，想让我来接。但我记得您之前说的是让我"协助"，想跟您对齐一下我的定位。

---

## 二、需要确认的4个核心问题

### 1. 我的角色是什么？

| 选项 | 含义 | 影响 |
|------|------|------|
| A. 协助 | 辅助潘韵佳，她决策我执行 | 不背结果，但没有话语权 |
| B. 主产品 | 我来负责产品规划、需求、SOP | 要背结果，需要对应的决策权 |
| C. 产品负责人（正式任命） | 明确写在项目组里，有签字权 | 最清晰，权责对等 |

**我要表达的态度**：不管哪个角色我都可以干，但需要明确，否则推不动。

---

### 2. 如果是主产品，我需要什么支持？

**权限层面：**
- 产线接入优先级，我有权排吗？
- SOP流程方案，我有权定稿吗？
- 与姜涛（研发）的需求排期，我有权协调吗？

**组织层面：**
- 需要一个正式的项目启动会，C-3/C-2管理者在场，明确我的角色
- 各产线接口人的配合，需要上层发过声（综合运营推不动BGBU）

**资源层面：**
- 我手

## L65 [t-a729f24df543] lang=zh len=1k-2k ws=金营项目
subject: 金营项目-一期建设成果
tags: 金营平台, 一期, 交付成果, 双轨制状态机, PM
content_head: # 金营项目 - ② 一期建设成果

> 来源：金营项目-完整知识库.md（三章），文档更新 2026-06-08 V1.1

## 一期核心转变

- ✅ 已实现：从「多入口分散」→「单入口统一」；从「无标准」→「有标准」（需求模板、状态流转、通知机制）
- ⚠️ 进行中：从「依赖人防」→「人防+基础技防」（必填校验、字段锁定已实现，智能风控待三期）
- 🔮 规划中：从「经验运营」→「数据驱动运营」（数据已沉淀，分析能力待建设）

## 一期完成情况

| 功能模块 | 完成状态 | 说明 |
|---------|---------|------|
| 统一需求入口 | ✅ 已完成 | 5+入口统一为1个 |
| 需求状态管理 | ✅ 已完成 | 双轨制状态机完整实现 |
| 通知中心 | ✅ 已完成 | 京ME推送+平台内消息 |
| 协作人机制 | ✅ 已完成 | 最多3人协作 |
| 系统设置 | ✅ 已完成 | 模板管理、受理人推荐等 |
| 鹊桥页面整合 | ✅ 已完成 | 小金库超级攒配置页面整合 |
| AI智能提报 | ✅ 已启用 | 2026-07-16 用户确认已启用；目标准确率≥90%，后续持续优化 |
| 金营↔鹊桥数据打通 | ❌ 未完成 | 仅完成页面整合，字段映射/数据回写未打通 |
| 金营↔财富策略数据打通 | ❌ 未完成 | 仅页面整合，数据

## L66 [t-b786985ee5ed] lang=zh len=1k-2k ws=金营项目
subject: 金营项目：智能配券12个横向场景全景梳理（2026-07-10）
tags: 金营, 智能配券, 横向场景, 一键配置, 机会点梳理, 二期
content_head: ## 金营项目智能配券横向场景梳理

> 2026-08-03 纠偏：本条保留 2026-07-10《智能配券机会点梳理》的历史全景数据，但其中“银行一键绑卡：二期横向场景③”“SMB金采&采购专属金：二期横向场景④”等二期横向场景优先级结论已被用户最新确认口径取代。当前《金营项目二期立项》横向扩展场景以 memory id=484 为准：SMB 金采 & 采购专属金营销、白分期/消金标准活动、小金库超级攒（鹊桥）三个场景；不能再把银行营销-一键绑卡写成本次二期横向扩展场景。

来源：`智能配券机会点梳理.xlsx` Sheet1（2026-07-10 用户提供）

### 12个场景全景（按优先级排序）

| # | 场景 | 业务线 | 配置系统 | 月均量 | 单次min | 月总min | 对接人 |
|---|------|--------|---------|--------|---------|---------|--------|
| 1 | 国补 | 支付-自持营销 | 权益平台 | 300+(高峰1000+) | 6 | 1800 | 张羽 |
| 2 | 小金库-消费 | 财富 | 权益中台 | 155 | 5 | 775 | 蔡婷 |
| 3 | 银行一键绑卡 | 支付-银行营销 | 权益平台 | 150+(季度300+) | 10 | 2500 | 梁婉蓉

## L67 [t-cb5de116e687] lang=zh len=1k-2k ws=金营项目
subject: 金营二期切量统计与推进口径 v4（2026-07-29 最新底表）
tags: 金营, 二期, 切量, 月均量级, 提需部门, 底表, H2, 全量切金营, v4
content_head: ## 金营二期切量统计与推进口径（v4，2026-07-29 最新底表）

来源：`提需部门_月均量级---a664125c-0785-4d58-9362-ba4a988233b8.xlsx` 的《底表》sheet，作为后续切量口径基准。

### 老板要求的切量原则
- 部门人多的可以少切、灰度切：先选代表性活动、高频标准需求切入，沉淀模板和数据资产后再逐步扩大。
- 部门人少的要在 H2 完成全量切金营提需：少人/低阻力/标准化基础好的场景必须在 H2 收口到金营。
- 系统提需类暂缓强制手工迁移：启航、金枢等系统提需场景后续探索系统对接方式。

### 少人部门：H2 全量切金营（约 545+ 笔/月，另含按需新场景）
| 业务线 | 场景/部门 | 人数 | 提需方式 | 月均量级 | H2口径 |
|---|---|---:|---|---:|---|
| 支付-自持营销 | 银行营销组（戚雪静） | 不确定 | XBP | 20+ | 纳入 H2 全量切金营，先跑通 XBP 类提需收口 |
| 支付-自持营销 | 白条产品部（王琳） | 1 | XBP | 新场景按需 | H2 全量走金营提需 |
| 支付-自持营销 | 支付生态合作部 | 4 | 邮件 | 10+（暂未计算国补2.0） | 先与权益中台确认格式兼容，H2 纳入全量切金营 |
| 消金-白分期 | 白条

## L68 [t-cccf0e93927d] lang=zh len=2k-4k ws=金营项目
subject: 京东VOP解决方案知识库（完整版）
tags: VOP, 京东, 解决方案, 金营, PM
content_head: # 京东VOP解决方案 - 完整知识库

> 来源: 京东VOP-解决方案（标准）.pdf | 京东集团-京东零售-政企事业部
> 读取时间: 2026-06-09

---

## 一、VOP定位

**京东VOP（Vendor Open Platform）** = 电商采购供应链 + 数智化运营平台

- 面向 ToG、ToB 大中型企业客户
- 以**标准API接口 + 现成组件**形式，与客户采购平台实现商品、交易、物流、财务等环节对接
- 核心价值：提供以"商品+技术"为核心的**采购基础设施解决方案**
- 实现商流、物流、信息流、资金流的高效链接协同
- 开放平台地址: https://vop.jd.com

---

## 二、政策背景

多部门出台政策推进采购数字化改革（2015-2022），关键文件包括：

| 时间 | 机关 | 核心要求 |
|------|------|----------|
| 2015 | 国务院 | 电子化政府采购交易平台建设 |
| 2015 | 国资委 | 央国企集中采购与电子化采购 |
| 2019 | 工信部 | 《企业数字化采购实施指南》 |
| 2022 | 全国人大 | 十四五规划——数字化转型、数据赋能全产业链 |

---

## 三、企业采购数字化核心诉求

1. **提升成本控制**：采购物资标品化、流程线上化，

## L69 [t-de4f21263f11] lang=zh len=2k-4k ws=default
subject: 金融带货专项知识库（完整版）
tags: 金融带货, 知识库, PM
content_head: # 金融带货专项进展汇报

> 来源文档: 20260325 初始汇报 + 20260522 进展更新

---

## 一、项目背景

基于骨肉相连2.0推进金融带货，综合运营部牵头建立金科群组的**商品库**（实物商品+虚拟服务）及互通机制。

## 二、项目目标

### 1. 内部管控收口
- **建立流程规范**: 金科各业务场景实物商品和权益营销对接的标准SOP，把控关键节点风险，明确干系人权责
- **业务数据沉淀**: 商品库线上化，系统整合规范数据管理口径，压缩隐形成本、提高资源投入投入效率
- **营销健康度管理**: 商品和权益营销活动ROI监控，保障盈利性

### 2. 业务拓展
- **跨BGBU资源拉通**: 建立群组商品库和商品货架上新协同机制，统一结算价格管控、营销成本压降
- **金融业务反向赋能**: 通过金融营销拉动零售等BGBU用户、订单、GMV增长，实现带货联动

## 三、前期梳理的业务痛点
1. 对接流程无标准，协议规范尚未建立
2. 商品结算价格不统一、结算方式复杂多样
3. 货品来源多样，系统依赖跨多体系
4. 新需求品类合作重复对接效率低
5. 缺少商品状态监控

---

## 四、整体进展 (截至2026-05-22)

### 项目任务规划框架

| 阶段 | 时间 | 内容 |
|------|------|------

## L70 [t-eb356062ebd1] lang=zh len=2k-4k ws=default
subject: 金融带货个人工作路线图（完整版）
tags: 金融带货, 工作路线图, PM
content_head: # 金融带货 - 张志维个人工作路线图（2026年6月-12月）

> 来源：`00_我的个人工作路线图.drawio` (v1.1, 2026-06-10更新)

## 角色定位
- **业务产品经理 + 运营规划**
- **核心交付**：数据看板（展示层）、PRD、姜涛团队协作

---

## 👤 角色分工（我 vs 潘韵佳）

### 🟢 我牵头（对内 + 产品）
- 📊 数据看板（技术+展示层）
- 📝 PRD 撰写
- 🤝 姜涛团队协作

### 🟡 潘韵佳牵头（对外 + 流程 + 指标）
- 💼 对接 BGBU
- 📋 SOP / 业务流程
- 📐 数据指标体系（数据同事协助）

### 🤝 共同协同
- 指标到PRD转换
- BGBU 需求翻译
- 周例会同步
- 月度运营报告
- 对上汇报

---

## 📅 三阶段工作规划

### 阶段一：摸清家底 + 协作机制建立（2026年6月-7月）

**📊 数据看板：**
- 配合潘&数据同事，理解指标体系初稿
- 配合调研数据源/取数链路（数据同事主导）
- 看板信息架构设计（哪些页面/模块/钻取路径）
- 数据看板 PRD v0.5 草案（展示层+交互）

**📝 PRD 撰写：**
- 制定 PRD 模板（业务背景/目标/用户故事/流程/指标）
- 搞清楚 VOP 9大接口域能做什么
- 搞清楚金科权益平台

## L71 [t-a3486f0b18a1] lang=zh len=>4k ws=default
subject: 专利撰写指南（完整版）
tags: 专利, 撰写指南, patent-writer
content_head: # 专利撰写指南 - 京东专利技术交底书

> 基于2026-01-09更新的京东专利模板和范例整理，协助完成发明/实用新型/外观设计专利技术交底书撰写。

---

## 一、专利类型与对应文档格式

### 1. 发明及实用新型专利技术交底书
- **模板来源**：发明及实用新型专利技术交底书模板_20260109更新.docx
- **范例参考**：一种新型无人机（发明及实用新型专利技术交底书-范例2）

### 2. 外观设计专利交底书
- **范例参考1**：产品外观（转运车）
- **范例参考2**：GUI界面（荐服饰功能界面）

---

## 二、发明及实用新型专利技术交底书 - 标准结构

### 2.1 表头信息
| 字段 | 说明 |
|------|------|
| 交底书名称 | 专利名称 |
| 技术联系人姓名 | 默认填写：张志维（用于与外部代理沟通，发明人信息在ERP系统中填写） |
| 技术联系人电话 | 默认填写：13810138598 |
| 技术联系人Email | 默认填写：zhangzhiwei3@jd.com（工作邮箱，禁止私人邮箱与代理人沟通） |

### 2.2 正文结构（六大章节）

#### 第1章：现有技术
- **内容要求**：记载某个应用场景或解决某个技术问题当前所采用的技术
- **写法**：可以概述描述，也可以仅给

## L72 [t-8936aaafd844] lang=zh len=>4k ws=金营项目
subject: 京东VOP实操手册（完整版）
tags: VOP, 京东, 实操手册, 金营, PM
content_head: # 京东VOP实操手册 - 完整知识库

> 来源: 京东VOP【实物+虚拟服务】标准接口对接帮助文档（外部用户申请权限请输入访问密码）.pdf
> 文档版本: 2.5.3
> 总页数: 394页
> 读取时间: 2026-06-09
> 外部客户访问密码: NOHz0h
> 最终解释权: 京东政企VOP

---

## 一、文档概述

本文档是VOP的**完整API技术手册**，涵盖从授权认证到商品、地址、价格、库存、订单、售后、发票、消息等全部接口的详细说明，包括请求参数、响应参数、示例代码、错误码等。

### API对接三大优势
1. **接入速度极快**：一个工程师30分钟搞定
2. **发展速度极快**：可同时与多客户对接，互不影响
3. **极大降低研发成本**：一次开发，不断复用

---

## 二、文档完整目录（14大模块）

### 模块1：概述
### 模块2：名词解释及注意事项
- API限流说明：触发限流返回错误码2010
- 接口响应统一格式：result/resultCode/success/resultMessage
- 出入参会扩展增加，需兼容处理
- 异常码不承诺稳定不变

### 模块3：API系统对接流程及说明
- **3.1** API调用流程图
- **3.2** 授权API（HTTPS调用）
  - **3.2.1** HTTPS方
