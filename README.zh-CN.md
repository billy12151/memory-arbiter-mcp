# Memory Arbiter（迷码）— 中文说明

**[English](README.md) | 中文**

> 这份文档写给所有人，不需要你是专业研发。精确的字段级契约见[集成指南](docs/INTEGRATION.zh-CN.md)。
> 当前正式版本 `0.14.6`。

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

## 3 分钟上手

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
| `find` | 用的时候搜一下，只返回相关的几条，不用整本翻 |
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

> ⚠️ **0.14.6 升级警告：** 当前 schema generation 仍为 `workspace_state_v1`；更早的数据库**直接启动会被拒绝**，必须走迁移命令。迁移会**丢弃旧的冲突/裁决/提醒历史**（记忆原文和历史都保留），升级完成后需要跑一遍全库扫描重新发现冲突。结构迁移与向量兼容现在独立判断：空间不一致会禁用向量并等待显式重建，不再阻断结构升级。

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

配置文件在 `~/.config/memory-arbiter/config.json`（`mema setup` 会帮你生成）。常用的几项：

| 配置项 | 白话说明 | 默认值 |
| --- | --- | --- |
| `db_path` | 数据库文件放哪 | 安装时生成 |
| `mcp.transport` | `stdio`（默认，一对一）或 `streamable-http`（多个客户端共享一个本地服务） | `stdio` |
| `mcp.http.stateless` | HTTP 请求无状态处理；服务重启后客户端不会持有失效 session | `true` |
| `isolation` | workspace 隔离档：`none` / `weak` / `strict` | `none` |
| `update_check.enabled` | 唯一会联网的功能：偶尔查一下 PyPI 有没有新版本。关掉就完全不联网 | `true` |
| `vec.enabled` | 开启后支持"按意思搜" | 关闭 |
| `embedding.model_path` | 本地 embedding 模型（GGUF 文件）路径 | 无 |
| `semantic_conflict.model_path` | 本地 Qwen 小模型路径，用于写入时的冲突核对 | 无 |

完整的配置说明和示例：[`examples/memory-arbiter.config.example.json`](examples/memory-arbiter.config.example.json)、[`.env.example`](.env.example)。

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

- **HTTP 默认无状态**：迷码的记忆和 semantic notice 状态在 SQLite，不在 MCP session。服务重启后，客户端不会因为旧 session ID 卡住；异步生成并已落库的 notice 仍会附在后续成功的工具响应里。只有尚未落库、仍停留在进程内 worker 队列中的任务可能被进程重启中断。只有客户端明确依赖服务端 MCP session 或主动 SSE 消息时，才设置 `mcp.http.stateless=false`。
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
