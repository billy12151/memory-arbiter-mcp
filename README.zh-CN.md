# Memory Arbiter（迷码）— 中文说明

**[English](README.md) | 中文**

> 这份文档写给所有人，不需要你是专业研发。精确的字段级契约见[集成指南](docs/INTEGRATION.zh-CN.md)。
> 当前正式版本 `0.15.7`（写时重复提醒升级为「主题+标签」向量召回；新增 `scan_duplicates` 一次有界清扫全库近重复）。

## 一句话说明白

**迷码是你所有 AI 工具共用的"记事本"。**

你今天用 Claude 聊了一个决策，明天换 Cursor 写代码，后天用别的 Agent 干活——每个工具都各自记事，互相不知道对方记了什么，时间一长：记重的、记错的、互相打架的、过时没说清的，越攒越多。

迷码把这些事记在**你自己电脑上的一个数据库文件**里（SQLite，不联网、不上传），并且帮你管起来：哪些是新的、哪些是旧结论、哪些互相矛盾、哪些是用户亲口确认过的，全都标得清清楚楚。

## 一分钟看懂它怎么工作

```text
  Claude Code   Cursor    ZCode     其他 Agent
      │           │         │           │
      └───────────┴────┬────┴───────────┘
                       │  都通过同一个标准接口（MCP）读写
                       ▼
              ┌─────────────────┐
              │   迷码 (mema)    │   跑在你本机的一个小程序
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  memory.sqlite3 │   就这一个文件，在你自己电脑上
              │  （事实的原文）   │   原文只有一份，永不悄悄覆盖
              └─────────────────┘
```

几个关键设计，用白话说：

- **原文只有一份，索引随时可以重建。** 就像一本书：正文是宝贝，目录和索引丢了重印一份就行。迷码里搜索用的向量、全文索引全是"目录"，原文才是"正文"。
- **每条记忆都带来源。** 谁写的、什么时候写的、说的是什么时候的事，都有记录。"用户亲口确认过"的事实会被锁住，AI 不能悄悄改。
- **AI 只负责发现矛盾，没权力改。** 发现两条记忆打架时，AI（本地小模型 Qwen）只能"举手报告"，选哪个、改哪个，必须你点头。
- **改任何东西都留痕。** 每次修改都有历史版本，可以查到谁、什么时候、把什么改成了什么。

## 交给 AI Agent 安装

把下面这段话直接发给 Codex、Claude Code、Cursor，或其他能操作终端的编程 Agent：

```text
请阅读 https://github.com/billy12151/memory-arbiter-mcp 的最新 README，
根据我的操作系统和当前 AI 客户端安装并配置最新版 mema。
请保留已有配置和数据库，不要覆盖或删除现有数据；
需要我选择不同安装模式、修改已有配置，或执行任何破坏性、提权操作时先询问我。
完成后运行 mema doctor，并向我报告安装方式、配置路径、数据库路径、客户端接入方式和验收结果。
```

Agent 应以这份 README 为事实来源，先检查本机环境，再判断使用 `uvx`、核心版、`vec` 或 `semantic-local`。无法安全判断时必须询问用户；没有运行 `mema doctor` 并报告所有警告，就不算安装完成。

## 手动安装（备用）

前提：电脑上装了 Python 3.11+。

```bash
# 1. 安装
pip install memory-arbiter-mcp

# 2. 生成配置（会自检环境，不会乱下载东西）
mema setup

# 3. 在你的 AI 客户端里接入（照抄示例改一下就行）
#    示例文件：examples/ 目录下的 *.mcp.json
```

日常就 4 个动作，AI 工具会自动调用，你只需要知道概念：

| 动作 | 白话解释 |
| --- | --- |
| `remember` | 记一条新事实（"项目数据库选了 PostgreSQL"） |
| `find` | 用的时候搜一下，返回索引页（元数据 + 各条长度 + 目录），看中哪条再 read 原文，不用整本翻 |
| `read` | 按编号看某条的完整原文 |
| `update` | 同一个事实有新说法时，更新原记录，**不是**再记一条新的 |

> 为什么要强调"更新而不是再记一条"？因为同一件事记两条，迟早打架。迷码会帮你发现打架（见下文），但最好别制造打架。

## 多条记忆打架了怎么办？

这是迷码最核心的本事。先看图：

```text
  写入"数据库用 PostgreSQL"
        │
        ▼
  系统发现已有一条"数据库用 MySQL"  ← 本地小模型双向核对，要求很严格：
        │                            两边都得说清"哪个属性、哪个值"，
        ▼                            说不清就不烦你
  给你发一条"提醒"（notice）
        │
        ├── 你看了觉得是误报 → dismiss（这条提醒关掉，记为"不是冲突"）
        ├── 你已经自己处理好了 → resolve（提醒关闭，不变成正式冲突）
        └── 确实是矛盾 → 升级成"正式冲突单"（conflict）
                          │
                          ▼
                  ┌──────────────────────────┐
                  │ 冲突单生命周期：            │
                  │ open（待裁决）              │
                  │   → applying（正在按方案改） │
                  │   → resolved（改完复查通过） │
                  └──────────────────────────┘
                          │
        处理顺序固定三步：judge（选定哪个对）
          → apply_conflict_action（一条一条改，每步都要你授权）
          → resolve_conflict（全部改完才关闭）
```

几个让人安心的点：

- **宁可漏报，不乱报。** 写入时的提醒门槛很高：模型必须两个方向都给出一致的"属性/值"抽取，还得在原文里找得到出处。不满足就静默转入后台扫描，不拿不确定的东西打扰你。
- **后台还有一道兜底扫描。** 定时扫描（`scan_candidates`）门槛宽、保召回，模型不在线也能跑出确定性候选。两道门配合：宽门保证不漏，严门保证不吵。
- **想清一遍全库的重复记忆？** 用 `memory_repair(task="scan_duplicates")` 一次拿全：服务端把所有页的近重复对聚合成一份有界结果（最多 200 对，默认只给 id/主题/workspace/原因/距离这类轻量字段，`include_quotes=true` 再附上证据引文），不会把 agent 会话撑爆。分诊后该合并的走 `merge_memories`，误报的照旧用 `record_conflict` 压掉。
- **装完记得设两个定时任务。** 迷码自己不带定时器（扫描的价值闭环在 agent 分诊，得靠外部调度器到点唤醒 agent）：①每小时跑一轮 `scan_candidates` 分页循环（把每页返回的 `next_anchor_memory_id` 当下一页的入参，直到返回空）；②每天跑一次 `doctor` 做治理提醒。每完成一轮完整扫描边界（某页返回 `next_anchor_memory_id=null` 且确有 anchor 被扫）才向 `scan_log.jsonl` 记一行轻量审计记录；在看到运行证据之前，agent 会收到设置引导提示，doctor 也会对「重建欠账」（`conflicts.scan_required`）和「扫描停滞超过 14 天」（`conflicts.scan_stale`）亮黄灯——任务跑起来后它们会自动安静。完整规格：`memory(action="help", data={"topic": "scheduled_tasks"})`。
- **改坏了能查。** 冲突单里钉死了当时的记忆版本和证据原文位置，每一步修改都有记录，改错了能追回来。

## 多个项目怎么隔开？

迷码用 workspace（工作区）区分项目，就像给每个项目一个抽屉：

```text
  isolation=none              isolation=weak               isolation=strict
  ┌───────────────────┐       ┌───────────────────┐        ┌───────────────────┐
  │ 所有抽屉全打通，      │       │ 都能翻到，但同抽屉的   │        │ 只给你看当前抽屉，     │
  │ 谁都能翻            │       │ 排在前面            │        │ 新抽屉要先经你确认     │
  └───────────────────┘       └───────────────────┘        └───────────────────┘
   适合自己一个人随便用          适合多项目日常                适合项目间不能串的场景
```

细节（名字相近的项目怎么合并、`default` 全局池为什么特殊等）见[集成指南](docs/INTEGRATION.zh-CN.md#workspace-归一与-acl)。

### 召回黑名单（0.15.5）

有些抽屉的内容不该混进日常搜索——比如 mema-twin（迷码分身）存的工作偏好：它们已经通过"编译成 persona prompt"这条正式通道注入，再被普通搜索翻出来就是重复打扰，还挤占搜索结果名额。

在数据库文件旁边放一个 `recall_blacklist.jsonl`（每行一个 workspace 名，空行和 `#` 注释随意），列进去的抽屉就从**不带 workspace 参数的普通 find 召回**里消失：

```text
  find（不指定 workspace）      find（显式指定 workspace）      治理视图 / 精确过滤 / 按 id 读
  ┌───────────────────┐       ┌───────────────────┐        ┌───────────────────┐
  │ 黑名单抽屉不出现      │       │ 指到哪个抽屉就看哪个，  │        │ 一切照旧，不受影响     │
  │                    │       │ 黑名单抽屉也一样     │        │                    │
  └───────────────────┘       └───────────────────┘        └───────────────────┘
```

不用重启，改完下次 find 就生效。没有这个文件时用内置默认（`mema-twin`）；文件建出来就完全以文件为准，清空文件=全部放行。`doctor` 会报告当前生效的黑名单。

## 出问题了怎么办？

```bash
mema doctor          # 体检：配置、索引、冲突积压……有毛病会告诉你怎么修
mema doctor --deep   # 连模型也一起检查
mema console         # 打开本地网页控制台（只读，看看库里都有啥）
```

常见情况：

- **没装向量组件**：照样能用，只是"按意思搜"降级为"按字面搜"。
- **本地小模型没配**：冲突发现不会停，只是写入时的即时提醒会关掉，由后台扫描兜底。
- **数据库文件坏了或没法写**：写入会改走一个只追加的备份文件（JSONL），不会假装存上了。
- **崩溃/强制关机后**：先跑 `memory(action="status")` 看索引覆盖率，再按提示重建索引、跑一遍扫描。索引是"目录"，重建它是安全无害的。

## 从旧版本升级（重要，先看再动）

> ⚠️ **0.14.8 升级警告：** 当前 schema generation 仍为 `workspace_state_v1`；更早的数据库**直接启动会被拒绝**，必须走迁移命令。迁移会**丢弃旧的冲突/裁决/提醒历史**（记忆原文和历史都保留），升级完成后需要跑一遍全库扫描重新发现冲突。结构迁移与向量兼容现在独立判断：空间不一致会禁用向量并等待显式重建，不再阻断结构升级。

```bash
# 1. 先预览，看看会动什么（不动真格）
mema upgrade --dry-run

# 2. 停掉所有连着迷码的 AI 客户端，然后做备份：
sqlite3 /绝对路径/memory.sqlite3 "PRAGMA wal_checkpoint(TRUNCATE);"
cp /绝对路径/memory.sqlite3 /绝对路径/memory.pre-0.14.sqlite3

# 3. 正式迁移（会自动切换配置；旧数据库文件永远不会被删）
mema upgrade

# 4. 重启客户端，验证
mema doctor --json
```

为什么要先跑那条 `sqlite3` 命令？因为数据库的"最新流水"可能还在 WAL 暂存文件里，只拷主文件会拷丢。这条命令先把流水并入主文件，再拷贝才是完整的。

## 配置（进阶）

配置文件在 `~/.config/memory-arbiter/config.json`（`mema setup` 会帮你生成）。0.15.0 起**配置只认文件**：所有可调项就是下面这 19 个键，引擎参数、超时、阈值、上限全部冻结为内置常量，不用再操心。

| 配置项 | 白话说明 | 默认值 |
| --- | --- | --- |
| `client` / `agent_id` | 调用身份（"是谁在写"），必填：没有内建默认值，缺失或为空时服务拒绝启动。stdio 下这组配置身份即进程级调用身份（归因、策略判定都用它）；HTTP 模式以请求头为准 | 无（必填） |
| `db_path` | 数据库文件放哪 | 安装时生成 |
| `backup_jsonl` | 数据库写不进去时的兜底备份文件（只追加） | 安装时生成 |
| `workspace` | 写入时不带 workspace 用的默认工作区 | `default` |
| `isolation` | workspace 隔离档：`none` / `weak` / `strict` | `none` |
| `policy_path` | 可选：按客户端/agent 控制工具开关的策略文件 | 无 |
| `mcp.transport` | `stdio`（默认，一对一）或 `streamable-http`（多个客户端共享一个本地服务） | `stdio` |
| `mcp.http.host` / `mcp.http.port` | HTTP 模式的监听地址（只允许本机）和端口；地址路径固定是 `/mcp`，不用配 | `127.0.0.1:8000` |
| `update_check.enabled` | 唯一会联网的功能：偶尔查一下 PyPI 有没有新版本。关掉就完全不联网 | `true` |
| `embedding.model_path` | 本地 embedding 模型（GGUF 文件）路径——**填了就是"我要用按意思搜"**，不用再开别的开关；向量维度自动跟着模型走，换不同维度的模型会在启动时自动按新维度重建向量表 | 无 |
| `embedding.auto_query` / `embedding.auto_write` | 查询/写入时自动算向量 | `true` |
| `semantic_conflict.model_path` | 本地 Qwen 小模型路径，用于写入时的冲突核对——填了就自动启用，并且启动时加载、常驻内存 | 无 |
| `semantic_conflict.enabled` | 显式关掉语义冲突的逃生口；不填时指向模型即启用，显式 `false` 优先 | 自动 |
| `semantic_conflict.on_write` | 写入时的冲突检测：`async`（异步提醒）或 `off`（关闭） | `async` |
| `semantic_conflict.max_notice_pairs` | 一次写入最多提醒几对（1–3） | `2` |
| `include_size` | 召回复量总开关（0.15.6）：开着，`find` / `read` / 过期审计 / 历史版本四个召回面都带 `size` 块（返回字符数 + 条数 + **token 预估**），agent 汇报成本用同一把尺子；关了就全都不带 | `true` |

关于"指向模型就是意图"再说两句：以前要 `vec.enabled`、`embedding.provider`、`vec.dim` 三个开关凑齐才算开了向量，现在**只看 `embedding.model_path` 填没填**；Qwen 那边同理，以前 `preload`/`resident` 默认不加载，现在配了模型就启动即加载、常驻不卸载。

环境变量只留 6 个"启动上下文"：`MEMORY_ARBITER_CONFIG` / `DB_PATH` / `BACKUP_JSONL` / `MCP_TRANSPORT` / `CLIENT` / `AGENT_ID`，用来告诉进程用哪个配置文件、哪个库、哪种传输、什么身份（launchd 等场景需要），同名时以配置文件为准。**其余旧环境变量全部失效**，设置了会在 `mema doctor`、控制台设置页和 `memory(action="status")` 里收到 "no longer read" 警告；被删的旧文件键会被忽略并提示 "no longer configurable"。旧键怎么迁移见[集成指南](docs/INTEGRATION.zh-CN.md)。

注意：`mema setup` 生成的模板里 `client`/`agent_id` **留空待填**——在 config.json 里填上，或在客户端接入配置里用环境变量 `MEMORY_ARBITER_CLIENT`/`MEMORY_ARBITER_AGENT_ID` 提供（`examples/` 下的 stdio 示例就是这么做的）。写入时不要在 `remember` 的 `data` 里传 `agent_id`/`client`，这些字段会被当未知字段剔除，归因只来自上述可信身份。

完整的配置示例：[`examples/memory-arbiter.config.example.json`](examples/memory-arbiter.config.example.json)。

多客户端共享一个本地服务（HTTP 模式）时，每个客户端要在配置里写死两个请求头 `X-Mema-Client` 和 `X-Mema-Agent-Id`，客户端会自动带上。注意：这只用于区分"是谁在写"，**不是密码**——服务只监听本机，请勿暴露到网络。示例：[`examples/streamable-http.mcp.json`](examples/streamable-http.mcp.json)。

## HTTP 模式：让多个客户端共用一个迷码

默认的 stdio 模式不用你操心进程：每个 AI 客户端自己把 `mema` 当子进程拉起来，用完关掉。但如果你想让**多个客户端共用同一个本地迷码服务**，就开 HTTP 模式。先看两者区别：

| | stdio（默认） | streamable-http |
| --- | --- | --- |
| 谁启动迷码 | 每个客户端各自拉子进程 | 你自己起一个常驻进程，多个客户端连过来 |
| 要不要常驻后台 | **不要** | **要**，否则关终端就没了 |
| 客户端怎么配 | 写命令和参数 | 写 url + 两个固定请求头 |
| 适合 | 一个人、一个客户端 | 一台机器上多个客户端共享同一份记忆 |

开 HTTP 分三步：

1. **配置**：`~/.config/memory-arbiter/config.json` 里 `mcp.transport` 改成 `"streamable-http"`（或设环境变量 `MEMORY_ARBITER_MCP_TRANSPORT=streamable-http`）。
2. **常驻**：迷码**不自带守护进程功能**，得用进程管理器让它一直在后台跑。macOS 用 launchd，模板见 [`examples/com.memory-arbiter.mema.plist`](examples/com.memory-arbiter.mema.plist)（把里面的 `__MEMA_BIN__` 换成 `which mema` 的绝对路径，丢进 `~/Library/LaunchAgents/` 后 `launchctl load`）。临时试水也可以 `tmux new -d -s mema 'mema'`。
3. **客户端**：照抄 [`examples/streamable-http.mcp.json`](examples/streamable-http.mcp.json)，填上 `X-Mema-Client` 和 `X-Mema-Agent-Id` 两个请求头。

几个关键提醒：

- **HTTP 固定无状态**：迷码的记忆和 semantic notice 状态在 SQLite，不在 MCP session，所以请求按无状态处理（0.15.0 起固定如此，不再可配）。服务重启后，客户端不会因为旧 session ID 卡住；异步生成并已落库的 notice 仍会附在后续成功的工具响应里。只有尚未落库、仍停留在进程内 worker 队列中的任务可能被进程重启中断。
- **请求头客户端写死一次**，每次请求自动带，别让 agent 在单次工具调用里动态加身份，那样会被拒。任一头缺失/为空/重复/冲突 → 直接 400 拒绝，不回退默认身份。
- **只监听 127.0.0.1**，迷码不会绑公网。这两个头只是"是谁在写"的来源标记，**不是密码**。要给远程机器用，自己在前面加带认证的反代，别让迷码直接暴露。
- **launchd 跑的时候 PATH 和工作目录跟你终端不一样**：`ProgramArguments` 里直接写 `mema` 的**绝对路径**；GGUF 模型路径在 config.json 里也写绝对路径，别用 `~`。

在 `none` / `weak` 模式下，首次成功注册一个 canonical workspace 时，写入响应会返回非阻断的顶层 `workspace_review` notice，并保留 `data.write_hints.new_workspace_detected`。先检查是否与已有 workspace 重复，再由用户授权执行 `confirm_workspaces`；重复写入已有 workspace 不会重复提醒。`strict` 模式继续使用阻断式 `action_required=confirm_new_workspace`，不会再叠加这条非阻断 notice。

### Claude Desktop / Claude Code 连接本机 HTTP

Claude 的本地 MCP 配置入口启动的是 stdio 命令。要复用已经常驻的 mema HTTP 服务、避免再启动第二份 mema，可在 `~/.claude.json` 的 `mcpServers` 下保留这一条配置（当前 Claude Desktop/Cowork 与 Claude Code 可能共用这个用户级文件）：

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "/opt/homebrew/bin/npx",
      "args": [
        "-y",
        "mcp-remote@0.1.43",
        "http://127.0.0.1:8000/mcp",
        "--allow-http",
        "--transport", "http-only",
        "--header", "X-Mema-Client:claude",
        "--header", "X-Mema-Agent-Id:claude",
        "--silent"
      ],
      "env": {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "NO_PROXY": "127.0.0.1,localhost"
      }
    }
  }
}
```

`command` 要换成你机器上 `which npx` 返回的绝对路径。删除原来直接启动 `mema` / `memory-arbiter-mcp` 的同名配置，否则 Claude 可能再拉起第二个服务进程。修改后完全退出并重开 Claude Desktop，Claude Code 会话也要重启。`mcp-remote` 是第三方桥接器，上面固定的是已与 mema 实测通过的版本。若你的 Claude 安装明确使用独立的 Desktop MCP 配置文件，就把同一条配置放在那里，但不要两边重复注册。

## 想深入了解

- [INTRO.md](INTRO.md)——架构和设计动机（中文）
- [docs/INTEGRATION.zh-CN.md](docs/INTEGRATION.zh-CN.md)——精确的工具契约、冲突协议、升级矩阵
- [CHANGELOG.md](CHANGELOG.md)——版本历史
- 给 Agent 看的内置规则：`memory(action="help", data={"topic":"agent_onboarding"})`
