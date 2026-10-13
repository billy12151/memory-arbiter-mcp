<!-- mcp-name: io.github.billy12151/memory-arbiter-mcp -->
# Memory Arbiter MCP

[![PyPI](https://img.shields.io/pypi/v/memory-arbiter-mcp)](https://pypi.org/project/memory-arbiter-mcp/)
[![Website](https://img.shields.io/badge/website-memarbiter.cn-brightgreen)](https://memarbiter.cn/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

**English | [中文](README.zh-CN.md)**

**Official site & docs: [memarbiter.cn](https://memarbiter.cn/)**

Memory Arbiter (迷码) is a capability amplifier for your AI agents: plug it in over MCP and every agent you use — Claude Code, Cursor, Codex, ZCode — becomes one assistant with a running memory. Decisions made in Claude today are usable in Cursor tomorrow; every fact carries its provenance, collisions get surfaced before they bite, and what you've locked cannot be silently overwritten. One SQLite file on your machine holds every fact in full; every model — embeddings, conflict judge — runs locally.

> **Current release: `0.18.0`** · local-first · MCP-native · 4 tools · knowledge workbench in `0.18.0` (on main now)

## In one minute: what mema does for you

Every coding agent you use — Claude Code, Cursor, Codex, ZCode — keeps its own notes and knows nothing about the others. Facts get duplicated, contradicted, and silently go stale. mema replaces that with **one trusted fact store all your agents share**:

- **Recall that actually works.** Lexical and sentence-level evidence channels search in parallel and fuse by rank: **65/65 self-recall** on the release exam. Sub-threshold candidates never reach a result page — what it returns is what you can cite — and every hit carries source offsets, so `read span=[start,end]` slices the exact original text.
- **Conflicts get surfaced, not buried — and not spammed.** When two facts disagree, a self-trained local mDeBERTa judge (three classes, no prompt engineering, no winner-picking) calls it: only high-confidence conflicts interrupt you, the grey zone shows up as low-key reference, and everything else quietly goes to the scheduled scan — so a strict threshold never means a lost finding, and an open gate never means noise. On the release exam, **90.9% of the conflicts it interrupts you with are real**.
- **Facts stay yours — and stay safe.** Full original text in one SQLite file on your machine, provenance on every write, version history on every edit, trust levels that lock what you confirmed. mema dials out nowhere on its own — the only optional call is a PyPI update check (the console workbench talks to the LLM endpoint you configure).
- **A knowledge base, curated by you.** The local browser console lets you pick which memories count as knowledge, browse them by workspace → tag with rendered markdown and relation graphs, and study them with a bring-your-own-LLM workbench (summarize / outline / quiz / Feynman / quiz-me / timeline) whose outputs write back through the standard pipeline.

## Benchmarks

### Public: LoCoMo-Refined

Evaluated on [LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined) (all 1,382 questions, official judge `Qwen3-14B`):

| Configuration | Overall | Text-only |
|---|---:|---:|
| **mema + writer-agent** (deepseek-v4.1-flash writer, public rules) | **61.00%** | **65.62%** |
| **mema verbatim** (zero-LLM write path, utterances + `event_time` as-is) | 43.13% | **49.48%** |

Context: Mem0's full pipeline scores 48.91% on the same text-only basis (re-scored by the benchmark authors) — **bare mema with no LLM in the write path matches it**; with a writer-agent, mema's text-only score trails only MemoraX on the leaderboard. Full methodology and reproduction: [`evals/locomo/RESULTS.md`](./evals/locomo/RESULTS.md).

### Internal release exam (shipped configuration)

Every release must pass the internal exam suite — fixed fixtures: 42 recall queries across five query kinds (paraphrase / lookup / legal / far / keyword), 108 conflict pairs across four label classes (36 true conflicts), and four similarity-label groups (as of `0.17.2`, shipped judge and thresholds; reproduce with `eval/runner.py --suite all`):

| Gate | Result |
|---|---|
| Self-recall (65 targets, top-10) | **65/65** |
| Conflict arbitration (D4 headline, any-channel) | **Recall 55.56% (20/36) · Precision 90.91% (20/22)** |
| batch_find cross-query consistency | **0 mismatches** (42 queries × 6 batches) |
| Conflict-window completion | **51/51** |

The exam is deliberately conservative and precision-favored: whatever the strict gate withholds is not lost — the scheduled scan re-walks the library with a wider, model-free funnel. This is the quality bar that ships, re-measured on every release.

## Install & quickstart

### Install with your AI Agent

Paste this into Codex, Claude Code, Cursor, or another coding agent with terminal access:

```text
Read the latest README at https://github.com/billy12151/memory-arbiter-mcp.
Install and configure the latest mema release for my operating system and current AI client.
Preserve any existing config and database; do not overwrite or delete existing data.
Ask me before choosing between materially different install modes, changing existing config,
or performing any destructive or privileged action. When finished, run mema doctor and report
the install method, config path, database path, client integration, and verification result.
```

The agent treats this README as the source of truth, picks `uvx`/core/`vec`/`semantic-local` from your environment, and stops for user input when a safe choice cannot be inferred. A successful install is not complete until `mema doctor` has run and every warning has been reported. On a fresh install the agent may also offer a one-minute product demo — it first checks the `mema-first-run-demo-status` marker memory (offered at most once, globally; skipped silently if it exists).

### Install manually

```bash
# One command: package + deps + embedding model + config.json (resumable downloads, ModelScope fallback)
curl -fsSL https://memarbiter.cn/install.sh | bash
```

Or step by step:

```bash
pip install "memory-arbiter-mcp[vec,semantic-local]"  # core + sqlite-vec + local GGUF runtime
mema setup --install   # downloads the embedding GGUF (~330MB, resumable); the mDeBERTa judge is a separate manual install
mema doctor            # verify
```

Then wire your MCP client from [`examples/*.mcp.json`](examples/) and start the server with `mema`. Missing capabilities (models not downloaded) surface as a persistent degraded banner with remediation — an incomplete install cannot pass for a complete one.

Full documentation, the Team edition and the personal edition live at **[memarbiter.cn](https://memarbiter.cn/)** (docs: [memarbiter.cn/docs](https://memarbiter.cn/docs/)).

**Two things to know before first use:**

1. **Identity is explicit.** Set `client` and `agent_id` in config.json — no built-in defaults, the server refuses to start when either is blank.
2. **`remember` requires `workspace`.** Call `memory_review(view="workspaces")` first to list existing buckets, then pass one; `'default'` is the global pool.

## The four tools

| Tool | Surface |
|---|---|
| `memory` | `remember`, `find`, `batch_find`, `read`, `batch_read`, `update`, `judge`, `status`, `help` |
| `memory_review` | read-only health, conflict groups/details, history, expired/audit, entities, workspaces |
| `memory_govern` | explicitly authorized retirement, conflict-plan application/resolution, confirmations |
| `memory_repair` | evidence rebuild, broad conflict scanning, history cleanup, backup replay |

Every call returns `{ok, mode, warnings, degraded, data}`; things needing your attention arrive as `notices` or `data.action_required`. Point any agent at the packaged rule: `memory(action="help", data={"topic":"agent_onboarding"})`.

The daily loop is four calls — `remember` a reusable fact, `find` to recall, `read` for exact lookup, `update` when a newer source replaces an old fact (never a second active copy of one source of truth). `find` is an index page (metadata + `content_chars` + outline with source offsets); `batch_find` merges 8 queries into one round-trip; `read span=[start,end]` quotes exactly; an empty recall says so honestly instead of stuffing recent memories.

## What's inside

**Conflict arbitration, included.** Candidate pairs surviving a deterministic funnel (sentence prefilter → memory screen → cosine band → rule evidence) are judged by the local mDeBERTa encoder — two bare row texts in, calibrated probabilities out. Only high-confidence conflicts notify; the grey zone surfaces as low-key reference; everything else fails open into the scheduled scan, which walks the library server-side and lands suspects in an agent-only judgment queue — never in user-facing lists. The judge never picks a winner and never edits memory: findings live in one auditable `conflicts` lifecycle (`open → applying → resolved`) your agent walks under explicit authorization (`judge → apply_conflict_action → resolve_conflict`, CAS-pinned revisions).

**A knowledge workbench, included.** `memory-arbiter console` opens a local browser console: curate your knowledge base (one-click `kb_member` add/remove — a dedicated column agents cannot drop by rewriting tags), browse by workspace → tag with markdown and one-hop relation graphs, and run six one-click study actions on any memory (summarize / outline / quiz / Feynman / quiz-me / timeline) with your own OpenAI-compatible API — outputs write back as memories, attributed to the console.

**Trust, enforced.** Provenance on every write, append-only version history, `normal`/`protected`/`locked` trust levels plus the `user_confirmed` label, and per-action `authorized=true` on every state-changing governance call. Workspace isolation comes in `none`/`weak`/`strict` — under strict, a new workspace stays `pending` until you confirm it, and the reserved `default` pool is invisible from project scopes. Workspaces listed in a `recall_blacklist.jsonl` file next to the database are excluded from unscoped recall; `mema doctor` reports the effective list.

## Configuration

File-only, in `~/.config/memory-arbiter/config.json`. The complete surface is 28 keys — full template at [`examples/memory-arbiter.config.example.json`](examples/memory-arbiter.config.example.json), per-key docs and migration tables in [`docs/INTEGRATION.md`](docs/INTEGRATION.md). The keys that matter on day one:

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

- `embedding.model_path` — pointing at a local GGUF model is the sole intent to enable evidence recall.
- `semantic_conflict.mdeberta_ckpt` — pointing at the judge checkpoint (see below) enables write-time arbitration.
- `mcp.transport` — `stdio` (default) or `streamable-http` for one shared local server.
- Six launch-context environment variables exist (`MEMORY_ARBITER_CONFIG`/`DB_PATH`/`BACKUP_JSONL`/`MCP_TRANSPORT`/`CLIENT`/`AGENT_ID`); config file wins, and any other `MEMORY_ARBITER_*` export warns "no longer read".

## HTTP mode: sharing one local server

stdio (default) needs no background process. To share one server across several clients: set `mcp.transport` to `"streamable-http"`, keep the process alive (launchd template in [`examples/`](examples/com.memory-arbiter.mema.plist) — mema has no built-in daemon), and point clients at `http://127.0.0.1:8000/mcp` with the fixed `X-Mema-Client`/`X-Mema-Agent-Id` headers from [`examples/streamable-http.mcp.json`](examples/streamable-http.mcp.json). Loopback only; headers are provenance, not authentication. Claude Desktop / Claude Code connect through a one-entry `mcp-remote` bridge — copy it from the same example file.

## Conflict judge model (mDeBERTa, optional)

One-time manual install (the ~531MB fp16 checkpoint does not ship on PyPI):

1. `pip install memory-arbiter-mcp[mdeberta]`;
2. download `mdeberta-v58_ep3_fp16.pt` (+ the `mdeberta-base` tokenizer directory) from the mini-clash release artifacts (already included in the offline install bundle from `memarbiter.cn` — skip this step there);
3. point `semantic_conflict.mdeberta_ckpt` at it — auto-enables and preloads at startup; `mema doctor` verifies the digest and label contract.

Unconfigured = write-time arbitration disabled; the scheduled scan and agent fallback are unaffected.

## Degradation

- No sqlite-vec / embedding model: lexical recall and governance continue, evidence indexing is unavailable.
- No judge: write-time arbitration pauses; the scheduled scan keeps returning deterministic candidates.
- SQLite unavailable: writes fall back to the append-only JSONL envelope; workspace decision state requires the database.

## Upgrading from an older database

Databases older than schema generation `workspace_state_v1` are refused at startup — run the side-by-side `mema upgrade` (preview with `--dry-run`, the source database is never deleted). The full operational guide — WAL-safe backup, what gets carried over and what is intentionally dropped, the post-upgrade scan epoch — lives in [`docs/UPGRADE.md`](docs/UPGRADE.md) (中文版 [`docs/UPGRADE.zh-CN.md`](docs/UPGRADE.zh-CN.md)).

## Development

```bash
uv run pytest -q
python scripts/sync_version.py --check
```

During development, package/docs may describe an unreleased dev version while `server.json` (the MCP Registry manifest) stays at the last released version; `scripts/sync_version.py` advances it at release time and `--check` prevents drift.

## 中文摘要

Memory Arbiter（迷码）是面向 AI Agent 的本地可信事实层：每条事实只存一份完整原文，向量与索引皆可重建；写入带出处、编辑带版本、信任级别防静默改写。冲突仲裁与知识库工作台开箱即得——mDeBERTa 三分类判定走宽严双门（宽门保召回、严门保安静），`conflicts` 表走 `open → applying → resolved` 可审计生命周期；浏览器控制台提供知识库策展（`kb_member` 独立治理位）与自带 LLM 的学习工作台。公开评测 LoCoMo-Refined 文本类口径写入代理 65.62%、零抽取打平 Mem0；内部发版考试自召回 65/65。**完整中文文档见 [README.zh-CN.md](README.zh-CN.md)**；另见 [INTRO.md](INTRO.md)、[docs/INTEGRATION.zh-CN.md](docs/INTEGRATION.zh-CN.md) 与 [docs/UPGRADE.zh-CN.md](docs/UPGRADE.zh-CN.md)。

## License

Apache-2.0 — see [LICENSE](LICENSE).
