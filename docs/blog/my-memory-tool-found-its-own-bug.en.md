# My memory tool found its own bug while I was shipping it

*How dogfooding `memory-arbiter` exposed a silent FTS5 bug in its own search — and what that cycle taught me about building developer tools.*

---

## The setup

I use two AI tools every day: **OpenClaw** (my planning assistant) and **ZCode** (my coding tool). Like most people using multiple AI clients, I had a problem that quietly bugged me:

> Cursor learned my coding conventions. Claude Code had no idea. ZCode forgot what OpenClaw decided. I was re-explaining the same context to five different tools, every single day.

So I built **[`memory-arbiter`](https://github.com/billy12151/memory-arbiter-mcp)** — a tiny local MCP server. One SQLite file. Every tool reads and writes through the same MCP protocol. Conflicts get arbitrated by structured rules (user-confirmation → event time → source trust → ingest time), not by an LLM guessing. No cloud, no API keys, no per-call cost.

The pitch writes itself: **stop re-teaching your stack the same facts. Write once, every tool knows.**

## Shipping v0.2.1 — the dogfooding moment

For the v0.2.1 release, I decided to actually use the thing I built. Instead of writing the release spec into a file and pointing ZCode at it, I had OpenClaw write the spec straight into `memory-arbiter`:

```
OpenClaw  →  memory_write(spec for "v0.2.1 release task")
ZCode     →  memory_search("v0.2.1 release task")  ← picks it up, zero file handoff
```

It worked. ZCode retrieved the spec, executed the release, wrote the result back. The handoff for a ~2000-word spec went from ~3000 tokens (loading the whole doc) to ~500 tokens (just the relevant slice). **83% saved.** I felt pretty clever.

Then I actually looked at the response payload.

## The thing I almost missed

Buried in the `warnings` field of every `memory_search` response:

```
"warnings": [
  "FTS5 query failed: fts5: syntax error near \".\". Falling back to LIKE search."
]
```

I had been searching for `v0.2.1 release task`. That string contains a `.`. And my full-text search index — the entire reason "precise retrieval replaces full-file loading" — was **silently falling back to slow `LIKE '%query%'` table scans** for any query containing a `.`, `:`, `*`, `(`, `)`, or `-`.

In other words: **version numbers, file paths, URLs, config keys** — the most common things you'd actually search for in a developer's memory store — were all running on the slow path. And I hadn't noticed, because results still came back. The bug didn't crash. It whispered.

This is the worst kind of bug. A crash forces you to fix it. A silent performance/clarity degradation just sits there, eroding your headline feature while you congratulate yourself on shipping.

## Root cause

The code was almost insulting in its simplicity:

```python
# search.py — the original line
rows = db.conn.execute(
    "SELECT ... WHERE memories_fts MATCH ? ...",
    [query],  # ← raw user query, straight into MATCH
)
```

I was passing the user's query directly into FTS5's `MATCH`. But FTS5 has its own query grammar — `. : * " ( ) - + AND OR NOT` are all special characters. A bare `v0.2.1` parses as garbage.

I checked `git blame`. That line had been there since `v0.1.0` — the very first commit. **It had been broken from birth.** I had even told a friend "I think OpenAI's models fixed this in a refactor once." Nope. I checked the history: never touched. The memory was false. (Ironic, for a memory tool.)

## The fix

FTS5 doesn't have SQL-style parameterized escaping. The standard idiom is to wrap each token as a **double-quoted phrase** — special characters inside a phrase lose their meaning.

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

Twelve lines. The headline feature now actually uses BM25 ranking for the queries that matter most.

I shipped this as [`v0.2.2`](https://github.com/billy12151/memory-arbiter-mcp/releases/tag/v0.2.2) with three regression tests that explicitly assert **no `FTS5 query failed` warning** appears for queries containing `.`, `:`, `*`, `(`, `)`.

## The part I actually want to talk about

The bug fix itself is mundane — twelve lines, standard idiom, you've seen it before. What I keep thinking about is the loop that produced it:

1. I built a tool to share memory across AI tools.
2. I used that tool to ship a release of that tool.
3. **The tool's own memory search exposed a bug in the tool's own memory search.**
4. I fixed it. The tool's memory search is now more reliable.

This is **self-hosting** at a tiny scale, and it's the strongest argument I have that the tool is worth your time — stronger than any feature list.

Most "developer tool" blog posts are written from the outside looking in: *here's what I built, isn't it neat.* The honest ones are written from the inside: *here's where it bit me, here's what I learned.* The reason dogfooding matters isn't that it's a marketing flex. It's that **a tool its author actually uses has a chance of being a tool you'll actually want to use.** A tool its author has never run in anger is a demo.

`memory-arbiter` is small. It has one user right now (me) and a handful of agents. It is not, by any market measure, "a success." But it has now survived the only test that I trust: **it ran its own release, and the failure it produced was real, specific, and fixable** — not a hypothetical from a roadmap.

## What I took away

A few things I'd pass on to anyone building small tools in 2026:

- **A silent warning in a payload field is worse than a crash.** Crashes get fixed. Warnings in JSON fields get scrolled past. If your fallback degrades the core feature, make it loud or don't ship the fallback.
- **`git blame` before `git trust your memory`.** I genuinely believed the bug had been fixed in a refactor. The history disagreed. (Yes, I see the irony.)
- **Dogfooding is not a marketing line. It's a filter.** If you can't stand using your own tool for your own real work across a full week, no amount of positioning will save it. If you can, the bug reports write themselves — and they're the most credible thing you can publish.
- **The MCP ecosystem is early.** The official registry only launched in September 2025. If you're building MCP servers, the bar is low and the audience is hungry. Ship the small thing now.

## Try it

If you use more than one AI coding tool and you're tired of re-explaining yourself:

```bash
pip install memory-arbiter-mcp
```

Repo: [github.com/billy12151/memory-arbiter-mcp](https://github.com/billy12151/memory-arbiter-mcp)
PyPI: [pypi.org/project/memory-arbiter-mcp](https://pypi.org/project/memory-arbiter-mcp/)

It's Apache-2.0 licensed for the 0.8.2 line going forward, fully local, and won't send your memory anywhere you don't want it to.

---

*I'm one person building in public. If you try it and hit something, file an issue — I read every one.*
