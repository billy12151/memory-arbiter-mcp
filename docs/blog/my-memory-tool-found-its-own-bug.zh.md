# 我给 AI 做的记忆工具，被它自己的搜索扒出了一个祖传 bug

*我做了个叫 [`memory-arbiter`](https://github.com/billy12151/memory-arbiter-mcp) 的 MCP server，让 ZCode、Codex、Cursor、Claude Code 这些 AI 工具共享同一份记忆。发版用它自己的时候，它自己的搜索暴露了它自己的 FTS5 bug。这个闭环让我对"做开发者工具"这件事想了很多。*

---

## 先说它解决什么

我每天用两个 AI 工具：**OpenClaw**（我的规划助手）和 **ZCode**（我的写码工具）。和所有同时用好几个 AI 客户端的人一样，我有个一直隐隐烦我的问题：

> Cursor 学会了我的编码规范，Claude Code 完全不知道；ZCode 忘了 OpenClaw 刚定的事。我每天都在把同样的上下文重新解释给五个不同的工具听。

所以我做了 `memory-arbiter` —— 一个本地跑的 MCP server。**一个 SQLite 文件，所有工具通过同一套 MCP 协议读写**，冲突用结构化规则仲裁（用户确认 → 事件时间 → 来源可信度 → 录入时间），不靠大模型猜。没云、没 API key、没按次计费。

顺带还有一个被低估的好处：**精准检索替代全文加载**。多数 AI 客户端每轮都把整个 `MEMORY.md` 塞进 system prompt，记忆越多 token 烧得越快；改成 `memory_search("关键词")` 只取相关切片，能省 80%+ 的 token。

卖点自己就写好了：**别再给你的工具栈重复上课了。写一次，所有工具都知道，还更省 token。**

## 发 v0.2.1 —— dogfooding 的那一刻

发 v0.2.1 的时候，我决定真用一下自己做的东西。我没有把发版规格写成文件再让 ZCode 去读，而是让 OpenClaw 直接把规格写进 `memory-arbiter`：

```
OpenClaw  →  memory_write(写入 "v0.2.1 发版任务" 规格)
ZCode     →  memory_search("v0.2.1 发版任务")  ← 直接捞出来执行，零文件传递
```

跑通了。ZCode 检索到规格，执行发版，把结果写回去。一份 ~2000 字规格的交接成本从 ~3000 tokens（整篇加载）降到 ~500 tokens（只取相关切片），**省了 83%**。我当时觉得自己挺聪明的。

然后我认真看了一眼返回值。

## 差点错过的东西

每一条 `memory_search` 返回的 `warnings` 字段里，都埋着这一行：

```
"warnings": [
  "FTS5 query failed: fts5: syntax error near \".\". Falling back to LIKE search."
]
```

我一直在搜 `v0.2.1 发版任务`。这个串里有个 `.`。而我的全文索引——"精准检索替代全文加载"这整个卖点的根基——**对所有含 `.`、`:`、`*`、`(`、`)`、`-` 的查询，都在悄悄降级成慢速的 `LIKE '%query%'` 全表扫描**。

换句话说：**版本号、文件路径、URL、配置键**——一个开发者记忆库里真正会被搜的最常见的东西——全部跑在慢路径上。而我完全没察觉，因为结果还是返回的。这个 bug 不崩，它只是小声哼了一下。

这是最糟糕的那种 bug。崩溃逼着你修。静默的性能/质量退化就那么躺着，慢慢啃食你的招牌功能，你还在给自己庆祝发版。

## 根因

代码简单到有点侮辱人：

```python
# search.py —— 原来那一行
rows = db.conn.execute(
    "SELECT ... WHERE memories_fts MATCH ? ...",
    [query],  # ← 用户原始 query，直接塞进 MATCH
)
```

我把用户 query 直接传进了 FTS5 的 `MATCH`。但 FTS5 有自己的查询语法——`. : * " ( ) - + AND OR NOT` 全是特殊字符。一个裸的 `v0.2.1` 在它眼里就是乱码。

我去查了 `git blame`。那一行从 `v0.1.0` 第一个 commit 起就在那。**天生就坏。** 我甚至还跟朋友说过"我记得 OpenAI 模型在某次重构里修过"。没有。我查了历史：根本没动过。这段记忆是错的。（对一个记忆工具来说，讽刺到家了。）

## 修法

FTS5 没有 SQL 那种参数化转义。标准做法是把每个 token 包成**双引号 phrase**——phrase 内部的特殊字符失去特殊含义。

```python
def _sanitize_fts_query(query: str) -> str:
    tokens = [tok for tok in query.split() if tok]
    if not tokens:
        return ""
    quoted = ['"' + tok.replace('"', '""') + '"' for tok in tokens]
    return " AND ".join(quoted)

_sanitize_fts_query("v0.2.1")                # '"v0.2.1"'
_sanitize_fts_query("v0.2.1 release task")   # '"v0.2.1" AND "release" AND "task"'
_sanitize_fts_query('a"b')                    # '"a""b"'
```

十二行代码。招牌功能终于在最该用上的查询上跑 BM25 排序了。

我把它作为 [`v0.2.2`](https://github.com/billy12151/memory-arbiter-mcp/releases/tag/v0.2.2) 发了，附带三个回归测试，**显式断言**对含 `.`、`:`、`*`、`(`、`)` 的查询不再出现 `FTS5 query failed` 警告。

## 我真正想聊的部分

bug 本身很平庸——十二行代码，标准写法，你大概率见过。我一直在想的是**产出这个 bug 的那个循环**：

1. 我做了一个让 AI 工具之间共享记忆的工具。
2. 我用这个工具去发这个工具的版。
3. **工具自己的记忆搜索，暴露了工具自己记忆搜索里的 bug。**
4. 我修了。工具的记忆搜索更可靠了。

这是**小尺度上的自举（self-hosting）**，也是我能给这个工具的、最有说服力的背书——比任何功能列表都有用。

大多数"开发者工具"博客是从外往里写的：*我做了这个，酷吧。* 诚实的是从里往外写的：*它在哪咬了我一口，我学到了什么。* dogfooding 重要，不是因为它是个营销姿态，是因为**作者自己真用的工具，才有机会成为你也想用的工具**。作者从没真刀真枪跑过的工具，只是 demo。

`memory-arbiter` 很小。目前只有一个用户（我）加几个 agent。从市场任何维度看，它都"不算成功"。但它挺过了我唯一信任的那种测试：**它跑了自己的发版，它产出的失败是真实的、具体的、可修的**——不是 roadmap 上的假设。

## 我学到的几条

2026 年做小工具，几条我会传下去的：

- **payload 里一个静默 warning，比崩溃更糟。** 崩溃会被修，JSON 字段里的 warning 会被划过去。如果你的兜底会削弱核心功能，要么把它喊响，要么别发这个兜底。
- **`git blame` 优先于"信自己的记忆"。** 我真心相信这个 bug 在某次重构里修过。历史不同意。（是的，我懂这个讽刺。）
- **dogfooding 不是营销话术，是过滤器。** 如果你没法连续一周拿自己的工具干自己的活儿，再怎么定位都救不了。如果你能，bug 报告会自己写出来——而且那是你能发布的最有可信度的内容。
- **MCP 生态还很早。** 官方 registry 2025 年 9 月才上线。你在做 MCP server 的话，门槛低、受众饿。先把小东西发出来。

## 试一下

如果你用好几个 AI 编程工具，受够了重复解释自己：

```bash
pip install memory-arbiter-mcp
```

仓库：[github.com/billy12151/memory-arbiter-mcp](https://github.com/billy12151/memory-arbiter-mcp)
PyPI：[pypi.org/project/memory-arbiter-mcp](https://pypi.org/project/memory-arbiter-mcp/)

MIT 协议，完全本地，不会把你的记忆发到任何你没同意的地方。

---

*我是一个人在公开做东西。你试了之后踩到坑，提个 issue——每一条我都看。*
