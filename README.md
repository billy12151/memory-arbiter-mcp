mcp-name: io.github.billy12151/memory-arbiter-mcp

# memory-arbiter-mcp

**[中文](#中文) | [English](#english)**

---

<a id="english"></a>

## English

A lightweight, fully local **AI enhancement middleware** — not another memory tool, but a layer that makes every AI client you already use noticeably smarter, by fixing the one thing they all get wrong: **context quality**.

From planning (OpenClaw) to design (OpenDesign) to code (ZCode, Codex, Cursor, Claude Code) — Memory Arbiter connects your entire AI toolchain through one shared SQLite database.

### Why Memory Arbiter?

Your AI client loads `MEMORY.md` + `memory/*.md` into the system prompt **every turn**. As knowledge grows, 5K–20K tokens burn before the model even reads your question — and worse, the model drowns in noise, losing track of what's current, what's confirmed, and what's stale.

Memory Arbiter replaces this with a SQLite-backed search: only the relevant entries come back, everything else stays on disk.

**The result isn't just cheaper — it's sharper.** When the model receives precise, de-duplicated, trust-ranked context, the same model produces noticeably better output. Most AI execution errors aren't the model being dumb — they're the model acting on stale, contradictory, or diluted context. Fix the input, and your existing model jumps a level.

**Works with one tool. Scales to many.**

#### Better output quality (the real win)

Token savings are easy to measure. But the deeper, more impactful change is **output quality** — because Memory Arbiter fixes what the model sees before it even starts thinking.

| What pollutes AI context | How it degrades output | How Memory Arbiter fixes it |
|---|---|---|
| Key info buried in a 20K-token blob | Model attention is spread thin; it grabs the wrong detail or hallucinates. | `memory_search` returns 3–5 laser-relevant entries. High signal-to-noise. |
| Stale and current info mixed together | Model follows an outdated constraint. | Dual timeline + conflict arbitration: outdated entries are flagged or superseded. |
| "The user confirmed this" vs "the AI guessed this" — indistinguishable | Model treats a guess as ground truth. | `source_type` labels + `user_confirmed` lock. The model knows what to trust. |
| Tool switch = context reset | Each tool re-derives understanding from scratch; errors compound. | Shared memory layer: every tool starts from the same verified facts. |

**Same model. Better context. Better output.** This is the core value — everything else (token savings, cross-tool sharing, audit trails) flows from giving your AI clean, structured, de-conflicted context.

#### What does it actually enhance?

Memory Arbiter doesn't add new capabilities to your AI — it **removes the blind spots** that cap the capabilities it already has:

| Bottleneck | Without Memory Arbiter | With Memory Arbiter |
|---|---|---|
| **Attention precision** | Model scans 20K tokens, attention diluted across everything | Gets 3–5 laser-relevant entries. Attention bandwidth is freed for actual reasoning. |
| **Memory consistency** | Old and new info coexist; model may follow stale constraints | Dual timeline + conflict arbitration ensure the model only sees current, validated facts |
| **Trust calibration** | Model can't distinguish user-confirmed facts from AI guesses | `source_type` + `user_confirmed` lock — the model knows exactly what to trust |
| **Cross-tool continuity** | Switching tools = context reset, understanding drifts | Shared memory layer: every tool starts from the same verified baseline |
| **Compounding knowledge** | Memory degrades over time as files grow messy | Structured database gets richer and more precise the more you use it — a positive feedback loop |

**Model-agnostic.** Whether you run GLM, Claude, GPT, or Gemini — the stronger the model, the more sensitive it is to context quality, and the bigger the uplift Memory Arbiter delivers.

#### Solo user: still worth it with one tool

Even if you only use Cursor, Claude Code, Codex, or ZCode — Memory Arbiter upgrades your memory from flat markdown to a queryable database:

| What you get | Without Memory Arbiter | With Memory Arbiter |
|---|---|---|
| **Per-turn token cost** | Entire `MEMORY.md` + `memory/*.md` in system prompt (5K–20K tokens) | `memory_search("keyword")` returns 3–5 hits (200–800 tokens). **~80%+ saving.** |
| **Memory scale ceiling** | Bigger = slower = more expensive. You manually trim. | Thousands of entries, retrieval cost near zero. Stop trimming. |
| **Conflict detection** | AI holds old and new info simultaneously and may not notice contradictions. | `memory_compare(id1, id2)` returns a structured verdict. `memory_arbitrate` resolves it by rules. **~90% cheaper** than LLM-based comparison. |
| **Source trust levels** | All memories are flat text — "the user said this" and "the AI guessed this" look the same. | `user_confirmed` > `document_extracted` > `agent_generated` > `unknown`. High-trust entries are auto-locked. |
| **Audit trail** | Edited in-place, no history. | Every entry carries `agent_id`, `ingest_time`, `event_time`. Full traceability. |
| **Periodic self-check** | Hope the AI remembers correctly, or manually proofread. | `memory_list_conflicts` + `memory_audit_summary` — one call, structured report. |

#### Multi-tool collaboration (bonus)

Using two or more AI clients? All solo benefits apply to each tool, **plus** a shared memory layer:

| What you get | Without Memory Arbiter | With Memory Arbiter |
|---|---|---|
| **Task handoff** | Write spec to file → other tool reads file. Path sync, version drift, copy-paste. | Tool A calls `memory_write`. Tool B calls `memory_search`. **Zero file handoff.** |
| **Cross-tool visibility** | Each tool has its own memory silo. What happened in Cursor is invisible to Claude Code. | One SQLite, isolated by `workspace` / `agent_id`. Write once, search everywhere. |
| **Conflict resolution across tools** | Two tools disagree → user manually reconciles. | `memory_arbitrate` applies the same trust/timeline rules across all tools. |
| **Storage dedup** | Each tool keeps its own copy of shared knowledge. | One database, zero duplication. **100% storage dedup.** |

#### Token savings at a glance

| Scenario | Full-file loading | With Memory Arbiter | Saving |
|---|---|---|---|
| Per-turn memory load | 5K–20K tokens in system prompt | 200–800 tokens via `memory_search` | ~80%+ |
| Conflict detection | LLM compares pairs (N², thousands of tokens) | `memory_compare` returns structured verdict (~200 tokens) | ~90% |
| Periodic audit | LLM scans entire library (10K+ tokens) | `memory_list_conflicts` + `memory_audit_summary` serve structured candidates | ~70% |
| Spec handoff (2000 words) | ~3000 tokens loaded into context | ~500 tokens via targeted `memory_search` | ~83% |

**Positioning, in five lines:**
- ✅ **Execution quality booster** — clean, de-conflicted context → fewer hallucinations, sharper results (**this is the core value**)
- ✅ Structured memory storage — SQLite + dual timeline + source trust levels
- ✅ Token-optimization middleware — precise retrieval replaces full-file loading
- ✅ Conflict arbitration engine — rule-based verdicts with explainable rationale
- ✅ Cross-tool sharing layer — one database, every AI client shares it (optional but powerful)
- ❌ Not an LLM, does not do semantic reasoning — semantic judgement stays with the AI client

#### Real-world example: full creative pipeline (plan → design → code)

The user runs a three-tool pipeline: **OpenClaw** (planning & specs) → **OpenDesign** (page & slide design) → **ZCode** (implementation). Memory Arbiter connects the entire chain.

- **Old way**: OpenClaw writes a spec to a file → OpenDesign reads it, designs something → screenshots/specs are handed to ZCode. Path sync, version drift, copy-paste at every handoff. Or the user repeats the context each time — information loss + wasted tokens.
- **Memory Arbiter way**: OpenClaw calls `memory_write` with the spec → OpenDesign calls `memory_search("the project")` and gets full context, produces designs, writes back design decisions → ZCode calls `memory_search("the project")` and receives both the spec **and** the design decisions in one query — **zero file handoff across three tools.**

**A spec + design handoff that used to cost ~5000 tokens of repeated context loading now costs ~800 tokens via targeted `memory_search`.** This very project ships its own release tasks this way — it dogfoods itself. Full step-by-step in [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

> See [`docs/INTEGRATION.md`](docs/INTEGRATION.md) for three concrete usage patterns (per-turn retrieval, scheduled audit, write-time conflict check) and the full cross-tool delegation walkthrough.

### The Bigger Picture

> **Intelligence is converging. Facts are the differentiator.**

AI models are getting smarter in lockstep. GPT-4o, Claude 3.5, Gemini 1.5, Llama 3, GLM-5, DeepSeek — run a blind test and most people can't tell them apart. Reasoning, generation quality, multimodal understanding: the gaps that once separated generations are closing fast. **Base intelligence is becoming a public good.**

Two narratives dominate the industry: the "Data Moat" (companies compete on proprietary data) and the "Context" school (agents need good context to perform). Both are right — but both miss the competitive axis that matters most for agents:

> **When every agent is smart enough, the winner is the one that commands more verified facts.**

This isn't about *having* data (a static asset). It's about **factual command** — the dynamic ability to retrieve, verify, connect, and act on the right information at the right time. An agent with precise, de-duplicated, trust-ranked context outperforms a "smarter" model drowning in noise — every time.

**Memory Arbiter is the engineering answer to this principle.** It doesn't make your model smarter. It makes your model *better informed*. Structured storage, conflict arbitration, trust levels, cross-tool sharing, semantic recall — every feature is designed around one thesis:

> **In the age of convergent intelligence, factual mastery is the ultimate competitive edge.**

### Features

- **Structured memory write**: `content`, `agent_id`, `workspace`, `tags`, `source_type`, `event_time`, `ingest_time`, `confidence`, `protection_level`, and more.
- **Source trust levels**: `user_confirmed` > `document_extracted` > `agent_generated` > `unknown`.
- **Dual timeline arbitration**: resolves conflicts by user confirmation → event time → source trust → ingest time. Every decision comes with an explainable rationale.
- **Locked protection**: `user_confirmed` memories are automatically locked — no agent can overwrite them.
- **Client policy system**: per-client enable/disable, agent allow/deny lists for multi-agent governance.
- **Graceful degradation**: `sqlite-vec` → FTS5 → `LIKE` → JSONL backup. Never crashes.
- **Zero cloud, zero LLM calls**: pure local SQLite. No Postgres, Redis, or external services.

### Quick Start

**Requirements**: Python 3.11+ (3.11, 3.12, or 3.13 — any of them works).

```bash
# Clone
git clone https://github.com/billy12151/memory-arbiter-mcp.git
cd memory-arbiter-mcp

# Setup — use whichever python3.1x you have (>=3.11). No python3.11? Use python3.12 / python3.13.
python3.11 -m venv .venv        # or: python3.12 / python3.13
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Optional: semantic recall via sqlite-vec
pip install '.[vec]'

# Run
memory-arbiter-mcp
```

#### Zero-install via `uvx` (recommended for non-Python users)

If you just want to run the server without managing a Python env — install [`uv`](https://docs.astral.sh/uv/) once, then:

```bash
uvx --from memory-arbiter-mcp memory-arbiter
```

This pulls the published package and launches the `memory-arbiter` entry point. No venv, no `pip install`. The entry points `memory-arbiter` and `memory-arbiter-mcp` are equivalent — use the shorter one. Note: `uvx` only shortens the **install** step; embedding model and `sqlite-vec` still need separate setup (see [Semantic Recall](#semantic-recall-optional)).

### Connect Your Tool

Add to your tool's MCP config (see `examples/` for ready-made templates). With a local venv:

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "/path/to/memory-arbiter-mcp/.venv/bin/memory-arbiter-mcp",
      "env": {
        "MEMORY_ARBITER_CLIENT": "zcode",
        "MEMORY_ARBITER_AGENT_ID": "zcode-default",
        "MEMORY_ARBITER_WORKSPACE": "${workspaceFolder}"
      }
    }
  }
}
```

Or, zero-install via `uvx` (no local clone needed):

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "uvx",
      "args": ["--from", "memory-arbiter-mcp", "memory-arbiter"],
      "env": {
        "MEMORY_ARBITER_CLIENT": "zcode",
        "MEMORY_ARBITER_AGENT_ID": "zcode-default",
        "MEMORY_ARBITER_WORKSPACE": "${workspaceFolder}"
      }
    }
  }
}
```

> Change `MEMORY_ARBITER_CLIENT` for each tool (`openclaw`, `zcode`, `codex`, `cursor`, `claude-code`). Put shared paths/vector/model settings in `~/.config/memory-arbiter/config.json`; keep per-client identity in the MCP env block. If you are not using a config file, put `MEMORY_ARBITER_DB_PATH` in each client's env and point them at the same SQLite file. (GUI tools like OpenDesign inherit the host CLI's config — no separate client name.)

> ⚠️ **New session required**: MCP servers are loaded at session startup. Already-open sessions won't see the new tools. Start a fresh session after configuring.

### Client Config Locations

| Client | Config Location |
|---|---|
| ZCode | `~/.zcode/v2/` MCP config |
| Codex CLI | `~/.codex/` MCP config |
| Claude Code | `.mcp.json` in project root |
| Cursor | `~/.cursor/mcp.json` |
| OpenClaw | `~/.openclaw/openclaw.json` MCP config |

> **OpenDesign / OpenClaw GUI tools**: these run on top of a host CLI (Codex CLI, Claude Code, etc.) and do **not** have their own MCP config entry. Whatever MCP server the host client has loaded is automatically available — e.g. once Codex CLI configures Memory Arbiter, OpenDesign running on top of Codex can call `memory_search` / `memory_write` natively with no extra setup.

### MCP Tools

| Tool | Description |
|---|---|---|
| `memory_write` | Write a memory (`source_type=user_confirmed` auto-locks) |
| `memory_search` | Search memories (FTS5 → LIKE fallback) |
| `memory_get` | Get a single memory by ID. Use when you already know the `memory_id` (e.g. from conflict lists, audit results, or previous search results) to quickly fetch full details without re-running a search. Read-only. |
| `memory_compare` | Compare two memories, returns explanation only |
| `memory_arbitrate` | Arbitrate conflict, can record result (`apply=true`) |
| `memory_confirm` | Promote a memory to user-confirmed and locked |
| `memory_supersede` | Explicitly retire a memory; bypasses user-confirmed/locked protection (`authorized=true` required) |
| `memory_list_conflicts` | List unresolved conflicts |
| `memory_audit_summary` | Per-workspace stats overview (counts, oldest/newest, open conflicts, source_type distribution) |
| `memory_store_embedding` | (optional) Store or replace an embedding manually. v0.5.0 can also auto-embed writes/searches when configured. |
| `memory_edit` | (v0.4.0) In-place edit a memory's content (full or partial `old_text`→`new_text`), archiving the prior version to a history table and re-syncing FTS. `locked`/`user_confirmed` records need `authorized=true`. The right tool for *partial* corrections — `supersede` retires the whole record, which also sinks the parts you didn't mean to negate. |
| `memory_history` | (v0.4.0) View the version chain (historical snapshots) of a memory, newest version first. Read-only. |
| `memory_cleanup_history` | (v0.4.0) Delete historical snapshots from `memory_history` (never touches active records). Per-memory / by-age / full; full cleanup requires `authorized=true`. |
| `memory_status` | Show current mode, degradation status, storage paths |
| `memory_split` | (v0.6.0) Split a long memory into sections for paragraph-level retrieval. Two-phase: prepare returns content batches for an external LLM; publish validates and atomically publishes sections + vectors. Requires sqlite-vec + GGUF embedding + `split.enabled`. |
| `get_sections` | (v0.6.0) Get full text + metadata of specific sections by ID. Use after `memory_search` returns `matched_sections` to fetch only the relevant paragraphs. |
| `memory_split_status` | (v0.6.0) Check a memory's section-split status, section catalog, and global vector index state. |
| `memory_rebuild_embeddings` | (v0.6.0) Batch-rebuild all embeddings after switching embedding models. Processes memory-level + section-level vectors. No LLM needed. | (feed this to your agent)

If your client also keeps local markdown (ZCode's `MEMORY.md`, Codex's `AGENTS.md`, etc.), paste this rule into your agent's instructions so it knows what to write where:

> Local md files store only self-use info (rules, tool experience, config notes, agent persona). Anything that might be reused by another agent or platform — not just project info: requirements, research, decisions, progress, user preferences, knowledge conclusions — goes into `memory-arbiter`. Every write must fill `subject`, `tags`, `source_type` (one of `requirement` / `decision` / `doc_summary` / `research` / `progress`), `event_time` (ISO 8601), `workspace` (project name), `source_ref`. Search via `memory_search` first; read source files only for detail. When you find a contradiction, don't overwrite — use `memory_supersede` or `memory_arbitrate`. When a to-do entry is done, write the status back to the original record (update status/subject to done); don't just mention "done" in a new memory, or the old entry stays in to-do state and misleads future searches.

### Optional: Semantic Recall (v0.5.0)

By default, memory-arbiter uses **lexical recall** (FTS5 trigram + BM25 + soft-rerank) — no embedding model, no heavy dependencies, fully local. This is enough for most cases.

For queries where wording differs but meaning is the same ("happy" vs "joyful", "金营平台" vs "金融带货"), you can opt into **semantic recall**. memory-arbiter does **not** bundle an embedding model — you bring your own, so the default install stays lightweight and you keep full control over the model, language, and cost. When configured, normal `memory_write` and `memory_search(query="...")` calls auto-embed; callers do not need to pass `query_embedding` manually.

**Setup (4 steps):**

1. Install sqlite-vec and the GGUF runtime:
   ```bash
   pip install memory-arbiter-mcp[vec]
   pip install llama-cpp-python
   ```
2. Choose an embedding model. For automatic embedding in v0.5.0, the built-in runtime supports local GGUF models:
   - **GGUF (local, recommended)** — works with any GGUF embedding model via `llama-cpp-python`. Reuses models you may already have (e.g. from OpenClaw/llama.cpp). Point the script at the file:
     `embeddinggemma-300m-qat-Q8_0.gguf` is a good 768-dim default.
   - **sentence-transformers (local)** — HuggingFace PyTorch models (`bge-small-zh`, `bge-base-en`, etc.):
     use your own backfill/query script and pass vectors to `memory_store_embedding` / `memory_search(query_embedding=...)`.
   - **Remote API (OpenAI / Zhipu / Tongyi)** — call the API in your own backfill script, pass vectors to `memory_store_embedding`. memory-arbiter only needs `pip install memory-arbiter-mcp[vec]`; no model runtime on this side.

   **Embedding Model Quick Reference** (pick one, then match `MEMORY_ARBITER_VEC_DIM` to its dimension):

   | Path | Recommended model | Dim | Source | Best for |
   |---|---|---|---|---|
   | GGUF (local) | `embeddinggemma-300m-qat-Q8_0.gguf` | 768 | [HuggingFace](https://huggingface.co/google/embeddinggemma-300m-qat) | Reusing a model you already have; no Python ML stack |
   | sentence-transformers | `BAAI/bge-small-zh-v1.5` (CN) / `bge-base-en-v1.5` (EN) | 512 / 768 | [HuggingFace](https://huggingface.co/BAAI) | Best quality-to-size ratio; needs PyTorch |
   | Remote API | `text-embedding-3-small` (OpenAI) / `embedding-3` (Zhipu) | 1536 / 1024 | Provider dashboard | No local compute; per-call cost |

   **End-to-end flow for auto-embedding:** pick GGUF model → put vec/model settings in `~/.config/memory-arbiter/config.json` → restart the MCP server → run `docs/semantic_example.py` once to backfill old memories. New writes and plain-text searches auto-embed after that.
3. Copy the config template and edit paths:
   ```bash
   mkdir -p ~/.config/memory-arbiter
   # Source checkout:
   cp examples/memory-arbiter.config.example.json ~/.config/memory-arbiter/config.json
   # Or pip-installed users:
   curl -L https://raw.githubusercontent.com/billy12151/memory-arbiter-mcp/main/examples/memory-arbiter.config.example.json \
     -o ~/.config/memory-arbiter/config.json
   ```
   Keep shared paths/vector/model settings here instead of each MCP client's env block. Keep `MEMORY_ARBITER_CLIENT`, `MEMORY_ARBITER_AGENT_ID`, and `MEMORY_ARBITER_WORKSPACE` in each client's env so tools keep separate identities. `~/.config/memory-arbiter/` is user-owned XDG config, so pip installs and client reinstallers do not overwrite it. The first auto-embedding call lazily loads the model and may be noticeably slower; later calls reuse it.
4. Backfill embeddings into existing memories, then search normally:
   ```bash
   # From a source checkout:
   python docs/semantic_example.py                 # backfill all active memories
   python docs/semantic_example.py --query "金营平台营销"   # try a semantic search
   ```
   The backfill helper currently ships as a source-tree script. If you installed only from pip, clone the repo or use `memory_store_embedding` from your own script for existing memories. New writes/edits are auto-embedded once the server is configured.

After configuration, normal `memory_search(query="...")` can generate the query vector automatically. Explicit `query_embedding` still works and takes precedence.

**How it ranks:** semantic candidates get a *floor score* just below content matches — they beat content-only noise but never outrank a real subject/tags hit. The arbitration and trust layer is untouched.

**Measured impact (small sample, not a formal benchmark):** on the same 15 golden queries + 18 pairwise constraints used to validate v0.3.0, enabling semantic recall improved Top-3 hit rate and pairwise ordering. The biggest win was pairwise pass rate reaching 100% — every "should-rank-above" constraint held. SQLite-side latency overhead was ~8ms per query; local embedding generation depends on your model/runtime.

| Metric | bm25 (v0.2.6) | hybrid (v0.3.0) | hybrid + semantic (v0.3.1) |
|---|---|---|---|
| Top-1 hit rate | 46.7% | 53.3% | 53.3% |
| Top-3 hit rate | 60.0% | 66.7% | **73.3%** |
| Pairwise pass rate | 77.8% | 88.9% | **100.0%** |

### Optional: Long-Document Section Split (v0.6.0)

When a memory exceeds `split.threshold` (default 4000 chars), `memory_search` returns the entire content — potentially tens of thousands of tokens. Section split solves this by dividing long documents into semantic sections with per-section vectors, so `memory_search` returns only the relevant section metadata (`matched_sections`) instead of the full text. Full design spec: [`docs/section-split-design.md`](docs/section-split-design.md).

**Prerequisites (all required):**
1. Semantic recall configured — sqlite-vec + GGUF embedding (see above)
2. `split.enabled = true` in `config.json`
3. `_vec_index_meta.state == ready` (verify with `memory_status`)
4. An external LLM to generate section titles, summaries, and boundary anchors

**How it works:**

```
memory_write(long_doc)
  → saves content normally; response includes split_hint if > threshold
  → split_hint is just a suggestion — content is already saved, no data loss

memory_split(memory_id)                    ← prepare, no DB writes
  → returns content in safe batches (llm_batch_chars) + section schema
  → Agent sends each batch to external LLM → gets back title/summary/anchor

memory_split(memory_id, split_decision="split", sections=[...])
  → arbiter validates offsets deterministically (never trusts LLM offsets)
  → generates per-section embeddings (hard prerequisite — all must succeed)
  → atomically publishes sections + vectors, sets split_status=active

memory_search("query")
  → recall works the same (5 channels, no new recall path)
  → for active-split memories: section Vec matching finds relevant paragraphs
  → returns matched_sections (title+summary only) + section_catalog
  → content=null, content_omitted=true → saves tokens

get_sections(memory_id, [section_id, ...])  ← fetch specific paragraphs
memory_get(memory_id)                       ← fetch full text if needed
```

**Vec gate closed** (model migration, space mismatch, query embedding failure): `memory_search` returns the full text with `section_enhancement_applied=false` — search capability never degrades, section enhancement is purely additive.

**Configuration** (`config.json` `split` section):

| JSON path | Env fallback | Default | Description |
|---|---|---|---|
| `split.enabled` | `MEMORY_ARBITER_SPLIT_ENABLED` | `false` | Master switch. All prerequisites must also be met. |
| `split.threshold` | `MEMORY_ARBITER_SPLIT_THRESHOLD` | `4000` | Min char count to trigger split hint. |
| `split.section_vec_distance_threshold` | `MEMORY_ARBITER_SECTION_VEC_DISTANCE_THRESHOLD` | `0.7` | Cosine distance cutoff for section matching. ⚠️ **Calibrate before production.** |
| `split.section_fulltext_threshold` | `MEMORY_ARBITER_SECTION_FULLTEXT_THRESHOLD` | `0.8` | When ≥80% of sections match, return full text. |
| `split.max_sections` | `MEMORY_ARBITER_MAX_SECTIONS` | `50` | Max sections per memory. Min is 2 (fewer = pointless). |
| `split.max_section_chars` | `MEMORY_ARBITER_MAX_SECTION_CHARS` | `3600` | Char limit for section embedding body (truncation tracked). |

**Example `config.json` with split enabled:**

```json
{
  "db_path": "~/.local/share/memory-arbiter/memory.sqlite3",
  "backup_jsonl": "~/.local/share/memory-arbiter/memory.backup.jsonl",
  "vec": { "enabled": true, "dim": 768 },
  "embedding": {
    "provider": "gguf",
    "model_path": "~/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf",
    "auto_query": true,
    "auto_write": true
  },
  "split": {
    "enabled": true,
    "threshold": 4000,
    "section_vec_distance_threshold": 0.7,
    "section_fulltext_threshold": 0.8,
    "max_sections": 50,
    "max_section_chars": 3600
  }
}
```

**Important notes:**
- `section_vec_distance_threshold` (0.7) is a development placeholder. Before production use, calibrate with real query-section pairs (see design doc §5). If the threshold is wrong, section split provides no value.
- `memory_edit` on content clears all sections and resets `split_status` to NULL. Re-split with `memory_split` if needed.
- After switching embedding models, run `memory_rebuild_embeddings(dry_run=True)` to preview impact, then `memory_rebuild_embeddings(dry_run=False, batch_size=50)` to rebuild all vectors.
- Default is **off**. If your memories are mostly short notes, code snippets, or conversation summaries, don't enable this — the overhead (LLM calls + embedding generation) isn't worth it.

### Configuration

Configuration can come from `MEMORY_ARBITER_CONFIG`, then `~/.config/memory-arbiter/config.json`, then environment variables/defaults. **Durable vector/model settings belong in the config file** so they survive MCP client reinstall/migration; each row below shows the JSON path and its env fallback (config file wins when both are set). Environment variables are still useful for simple client identity and CI overrides. Full explanations in [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

**Config file fields** (`~/.config/memory-arbiter/config.json`) — paths, vector, and embedding settings:

| JSON path | Env fallback | Default | What to tune |
|---|---|---|---|
| `db_path` | `MEMORY_ARBITER_DB_PATH` | `./memory_arbiter.sqlite3` | Shared path for cross-tool memory. |
| `backup_jsonl` | `MEMORY_ARBITER_BACKUP_JSONL` | `./memory_arbiter.backup.jsonl` | Append-only JSONL backup, used only when SQLite is read-only. |
| `policy_path` | `MEMORY_ARBITER_POLICY` | _(none)_ | Path to a JSON policy file (per-client enable/disable, agent allow/deny). |
| `vec.enabled` | `MEMORY_ARBITER_ENABLE_SQLITE_VEC` | `false` | Set `true` to enable semantic recall (requires `pip install memory-arbiter-mcp[vec]`). |
| `vec.dim` | `MEMORY_ARBITER_VEC_DIM` | `768` | Must match your embedding model. Changing it requires dropping and recreating `memories_vec`. |
| `recall_pool_cap` | `MEMORY_ARBITER_RECALL_POOL_CAP` | `50` | **Raise to 100–200 when your store exceeds ~100 entries** — first knob to turn if matches go missing. |
| `content_like_cap` | `MEMORY_ARBITER_CONTENT_LIKE_CAP` | `30` | Raise if many same-topic memories exist. |
| `embedding.provider` | `MEMORY_ARBITER_EMBEDDING_PROVIDER` | inferred as `gguf` only when `embedding.model_path` is set | Only `gguf` is supported in v0.5.0. Without a model path, auto-embedding stays off. |
| `embedding.model_path` | `MEMORY_ARBITER_EMBEDDING_MODEL_PATH` (or legacy `MEMORY_ARBITER_GGUF`) | _(none)_ | Path to the GGUF embedding model. |
| `embedding.auto_query` | `MEMORY_ARBITER_EMBEDDING_AUTO_QUERY` | `true` | Auto-encode plain-text queries for semantic recall. |
| `embedding.auto_write` | `MEMORY_ARBITER_EMBEDDING_AUTO_WRITE` | `true` | Auto-embed new writes/edits so they enter semantic recall immediately. |
| `split.enabled` | `MEMORY_ARBITER_SPLIT_ENABLED` | `false` | Enable long-document section split (v0.6.0). Requires vec + embedding configured. See [Section Split](#optional-long-document-section-split-v060). |
| `split.threshold` | `MEMORY_ARBITER_SPLIT_THRESHOLD` | `4000` | Min char count to trigger section split. |
| `split.section_vec_distance_threshold` | `MEMORY_ARBITER_SECTION_VEC_DISTANCE_THRESHOLD` | `0.7` | Section Vec cosine distance cutoff. ⚠️ Calibrate before production. |
| `split.section_fulltext_threshold` | `MEMORY_ARBITER_SECTION_FULLTEXT_THRESHOLD` | `0.8` | Return full text when ≥X% of sections match. |
| `split.max_sections` | `MEMORY_ARBITER_MAX_SECTIONS` | `50` | Max sections per memory (min 2). |
| `split.max_section_chars` | `MEMORY_ARBITER_MAX_SECTION_CHARS` | `3600` | Char limit for section embedding input. |

**Environment variables** — keep per-client identity in each MCP client's env block. Some fields also have config-file equivalents, but config wins; use env here when the value must differ by client/session.

| Variable | Default | What to tune |
|---|---|---|
| `MEMORY_ARBITER_CLIENT` | `codex` | Per-tool identity (`codex`, `claude-code`, `cursor`, `zcode`, ...). |
| `MEMORY_ARBITER_AGENT_ID` | `default` | Agent identity within a client. |
| `MEMORY_ARBITER_WORKSPACE` | `default` | Isolation key. |
| `MEMORY_ARBITER_CONFIG` | _(none)_ | Optional path to an alternate JSON config file. If set, memory-arbiter reads that file instead of the default `~/.config/memory-arbiter/config.json`; file values still override other env fallbacks. |
| `MEMORY_ARBITER_RANKING_MODE` | `hybrid` | `hybrid` (default) or `bm25` (legacy). No config-file equivalent. |
| `MEMORY_ARBITER_GGUF` | _(none)_ | Legacy GGUF path fallback; prefer `embedding.model_path` in the config file. |

### Data Migration

Moving to a new machine? Just copy the SQLite file:

```bash
# Copy the database
cp ~/.local/share/memory-arbiter/memory.sqlite3 /new/machine/~/.local/share/memory-arbiter/

# Reinstall the project (don't copy .venv — rebuild it)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Testing

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m pytest
```

### License

MIT

---

<a id="中文"></a>

## 中文

一个轻量、完全本地运行的 **AI 增强中间件**——不是又一个记忆工具，而是一层让你正在用的每个 AI 客户端都明显变聪明的底层设施，修的是所有 AI 模型的共同短板：**上下文质量**。

从规划（OpenClaw）到设计（OpenDesign）到编码（ZCode、Codex、Cursor、Claude Code）——memory-arbiter 通过一个共享的 SQLite 数据库串起你的整条 AI 工具链。

### 为什么需要 Memory Arbiter？

你的 AI 客户端每轮对话都把 `MEMORY.md` + `memory/*.md` 整个塞进 system prompt。记忆越多，token 消耗越大——模型还没读你的问题，5K–20K token 的上下文已经烧掉了。更要命的是，模型淹没在噪声里，分不清什么是最新的、什么是用户确认的、什么是过时的。

Memory Arbiter 用 SQLite 检索替代全文加载：只有相关的条目返回，其余的留在磁盘上。

**结果不只是更省钱——而是更准。** 当模型拿到的上下文是精准的、去重的、按可信度排序的，同一个模型，输出质量会明显跃升。AI 执行出错，大部分时候不是模型笨，而是它拿到的输入是过时的、自相矛盾的、或者关键信息被噪声淹没了。修好输入端，你现有的模型直接提升一个等级。

**一个工具就能用。多个工具更香。**

#### 输出质量提升（核心价值）

省 token 好量化。但更深层、更关键的改动是**输出质量**——因为 Memory Arbiter 修的是模型在开始思考之前看到的东西。

| 污染 AI 上下文的问题 | 导致的执行偏差 | memory-arbiter 怎么修 |
|---|---|---|
| 关键信息埋在 20K token 的大段文本里 | 模型注意力分散，抓错重点或直接编造 | `memory_search` 只返回 3–5 条高度相关的结果，信噪比拉满 |
| 旧信息和新信息混在一起 | 模型照着过时的约束去执行 | 双时间轴 + 冲突仲裁，过时条目被标记或淘汰 |
| "用户确认的"和"AI 猜的"分不清 | 模型把猜测当事实用 | `source_type` 标记来源，`user_confirmed` 自动锁定，模型知道该信什么 |
| 换个工具上下文就丢了 | 每个工具重新理解一遍，理解偏差累积 | 共享记忆层，所有工具从同一套已验证的事实出发 |

**同一个模型，上下文对了，输出就准了。** 这是核心价值——其他一切（省 token、跨工具共享、审计追溯）都是“给 AI 干净、结构化、无冲突的上下文”自然带来的结果。

#### 它到底增强了什么？

memory-arbiter 没有给你的 AI 加新能力——它**移除了卡住现有能力上限的盲区**：

| 瓶颈 | 没有 memory-arbiter | 用了之后 |
|---|---|---|
| **注意力精度** | 模型扫 20K token，注意力被稀释 | 拿到 3–5 条精准结果，注意力带宽释放出来做真正的推理 |
| **记忆一致性** | 新旧信息共存，模型可能照着过时的来 | 双时间轴 + 冲突仲裁，模型只看到当前已验证的事实 |
| **可信度判断** | 模型分不清用户确认的事实和 AI 的猜测 | `source_type` + `user_confirmed` 锁定，模型明确知道该信什么 |
| **跨工具连续性** | 换工具 = 上下文重置，理解偏差累积 | 共享记忆层，每个工具从同一套已验证的事实出发 |
| **知识复利** | 记忆随时间退化，文件越来越乱 | 结构化数据库越用越丰富、越精准——正反馈循环 |

**模型无关。** 不管你用 GLM、Claude、GPT 还是 Gemini——模型越强，对上下文质量越敏感，memory-arbiter 带来的提升越大。

#### 单客户端用户：只用一个工具也值得

即使你只用 Cursor、Claude Code、Codex 或 ZCode，Memory Arbiter 也能把你的记忆从扁平 markdown 升级成可查询的数据库：

| 收益 | 没有 memory-arbiter | 用了之后 |
|---|---|---|
| **每轮 token 消耗** | 整个 `MEMORY.md` + `memory/*.md` 塞 system prompt（5K–20K tokens） | `memory_search("关键词")` 返回 3–5 条（200–800 tokens）。**省 80%+。** |
| **记忆规模天花板** | 越大越慢越贵，被迫手动精简 | SQLite 存几千条，检索成本接近零。不用再精简。 |
| **冲突发现** | AI 同时记住旧信息和新信息，自己未必发现得了矛盾 | `memory_compare(id1, id2)` 返回结构化裁决，`memory_arbitrate` 按规则判定。比 LLM 逐条比对**省 90%**。 |
| **来源可信度** | 全是平铺文本，"用户说的"和"AI 猜的"看着一样 | `user_confirmed` > `document_extracted` > `agent_generated` > `unknown`，高可信条目自动锁定。 |
| **审计追溯** | 改了就改了，谁改的、什么时候改的不知道 | 每条记忆有 `agent_id` / `ingest_time` / `event_time`，完整溯源。 |
| **定时自检** | 靠 AI 自觉，或手动翻 | `memory_list_conflicts` + `memory_audit_summary` 一个调用出报告。 |

#### 多客户端协作（进阶收益）

同时用两个或更多 AI 工具？单客户端的所有收益 **每个工具都享受**，再加一层共享记忆：

| 收益 | 没有 memory-arbiter | 用了之后 |
|---|---|---|
| **任务交接** | 规格写成文件 → 另一个工具读文件，路径同步、版本混乱、复制粘贴 | 工具 A 调 `memory_write`，工具 B 调 `memory_search`。**零文件传递。** |
| **跨工具可见性** | 每个工具有自己的记忆孤岛，Cursor 里发生的事 Claude Code 看不到 | 一个 SQLite，按 `workspace` / `agent_id` 隔离，写一次处处可搜。 |
| **跨工具冲突仲裁** | 两个工具对同一件事写的不一样 → 用户手动调和 | `memory_arbitrate` 用同一套可信度/时间线规则跨工具裁决。 |
| **存储去重** | 每个工具各存一份共享知识 | 一个数据库，零重复。**存储 100% 去重。** |

#### Token 节省一览

| 场景 | 全文加载 | 用 memory-arbiter | 节省 |
|---|---|---|---|
| 每轮记忆加载 | system prompt 塞 5K–20K tokens | `memory_search` 返回 200–800 tokens | ~80%+ |
| 冲突检测 | LLM 逐条比对（N²，数千 tokens） | `memory_compare` 返回结构化裁决（~200 tokens） | ~90% |
| 定期审查 | LLM 扫全库（万级 tokens） | `memory_list_conflicts` + `memory_audit_summary` 出结构化候选 | ~70% |
| 规格交接（2000 字） | 加载进 context ~3000 tokens | 精准 `memory_search` ~500 tokens | ~83% |

**定位（五句话）：**
- ✅ **执行质量放大器** — 干净、无冲突的上下文 → 更少幻觉、更准的结果（**核心价值**）
- ✅ 结构化记忆存储 — SQLite + 双时间轴 + 来源可信度
- ✅ Token 优化中间件 — 精准检索替代全文加载
- ✅ 冲突仲裁引擎 — 规则化裁决，输出可解释理由
- ✅ 跨工具共享层 — 一个数据库，所有 AI 客户端共享（可选，但用了就回不去）
- ❌ 不是 LLM、不做语义推理 — 语义判断交给 AI 客户端

#### 真实案例：完整创意管线（规划 → 设计 → 编码）

用户跑的是三工具管线：**OpenClaw**（规划 & 规格）→ **OpenDesign**（页面 & PPT 设计）→ **ZCode**（代码实现）。memory-arbiter 串起整条链路。

- **传统方式**：OpenClaw 把规格写成文件 → OpenDesign 读文件做设计 → 设计稿和规格再交给 ZCode，每次交接都要约定路径、手动同步、版本混乱；或者用户每次重复描述背景，信息损耗大、token 浪费多。
- **memory-arbiter 方式**：OpenClaw 调 `memory_write` 写入规格 → OpenDesign 调 `memory_search("项目")` 拿到完整上下文，做设计，把设计决策写回 → ZCode 调 `memory_search("项目")` 一次拿到规格**和**设计决策——**三个工具之间零文件传递。**

**一份规格 + 设计交接：传统重复加载上下文约 ~5000 tokens，现在精准检索约 ~800 tokens。** 这个项目自己的发版任务就是这么跑的——用自己的产品喂自己的产品。完整步骤见 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)。

> 三种典型用法（每轮按需检索、定时审查、写入时冲突检测）和完整的跨工具委派步骤见 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)。

### 更大的图景

> **智能趋于平权，数据定义高下。**

AI 模型的能力正在同步收敛。GPT-4o、Claude 3.5、Gemini 1.5、Llama 3、GLM-5、DeepSeek——做盲测，普通人已经分不清谁是谁了。推理、生成、多模态理解，曾经拉开代差的维度正以肉眼可见的速度缩小。**基础智能正在变成公共品。**

行业里两派主流叙事："Data Moat"（企业靠独占数据竞争）和 "Context"（Agent 需要好的上下文）。都对，但都没触及 Agent 竞争的最关键维度：

> **当所有 Agent 都足够聪明时，胜者属于掌握更多已验证事实的那个。**

这不是"拥有数据"（静态资产），而是**事实掌控力**——在正确的时间检索、验证、关联、调用正确信息的动态能力。一个拿着精准、去重、可信度排序的上下文的 Agent，每次都能击败淹没在噪声里的"更聪明"的模型。

**Memory Arbiter 就是这条原则的工程实现。** 它不会让你的模型更聪明，但它会让你的模型**更知情**。结构化存储、冲突仲裁、可信度分层、跨工具共享、语义检索——每一个能力都围绕一个论点设计：

> **在智能趋同的时代，事实掌控力是终极竞争力。**

### 核心能力

- **结构化写入**：`content`、`agent_id`、`workspace`、`tags`、`source_type`、`event_time`、`ingest_time`、`confidence`、`protection_level` 等。
- **来源可信度**：`user_confirmed` > `document_extracted` > `agent_generated` > `unknown`。
- **双时间轴仲裁**：按 用户确认 → 事件发生时间 → 来源可信度 → 录入时间 的优先级判定，输出可解释的裁决理由。
- **锁定保护**：`user_confirmed` 的记忆自动锁定，任何 Agent 都不能自动覆盖。
- **客户端策略**：按客户端启用/禁用，Agent 级别的 allow/deny 白名单控制。
- **逐级降级**：`sqlite-vec` → FTS5 → `LIKE` → JSONL 备份，不会崩。
- **零云依赖、零大模型调用**：纯本地 SQLite，不需要 Postgres、Redis 或外部服务。

### 快速开始

**要求**：Python 3.11+（3.11、3.12、3.13 均可）。

```bash
# 克隆
git clone https://github.com/billy12151/memory-arbiter-mcp.git
cd memory-arbiter-mcp

# 安装 —— 用你机器上任意一个 3.11 及以上的 python 即可；没有 3.11 就用 3.12 / 3.13
python3.11 -m venv .venv        # 也可：python3.12 / python3.13
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 可选：启用语义召回增强（sqlite-vec）
pip install '.[vec]'

# 启动
memory-arbiter-mcp
```

#### 用 `uvx` 零安装启动（推荐非 Python 用户）

只想跑起来、不想管 Python 环境——装一次 [`uv`](https://docs.astral.sh/uv/)，然后：

```bash
uvx --from memory-arbiter-mcp memory-arbiter
```

这会拉取已发布的包并启动 `memory-arbiter` 入口。无需 venv、无需 `pip install`。`memory-arbiter` 和 `memory-arbiter-mcp` 两个入口等价，用短的那个即可。注意：`uvx` 只省掉**安装**这一步，embedding 模型和 `sqlite-vec` 仍需单独配置（见 [语义召回](#语义召回可选)）。

### 接入工具

在你的工具的 MCP 配置中加入（完整示例见 `examples/` 目录）。用本地 venv：

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "/path/to/memory-arbiter-mcp/.venv/bin/memory-arbiter-mcp",
      "env": {
        "MEMORY_ARBITER_CLIENT": "zcode",
        "MEMORY_ARBITER_AGENT_ID": "zcode-default",
        "MEMORY_ARBITER_WORKSPACE": "${workspaceFolder}"
      }
    }
  }
}
```

或用 `uvx` 零安装（无需本地 clone）：

```json
{
  "mcpServers": {
    "memory-arbiter": {
      "command": "uvx",
      "args": ["--from", "memory-arbiter-mcp", "memory-arbiter"],
      "env": {
        "MEMORY_ARBITER_CLIENT": "zcode",
        "MEMORY_ARBITER_AGENT_ID": "zcode-default",
        "MEMORY_ARBITER_WORKSPACE": "${workspaceFolder}"
      }
    }
  }
}
```

> 每个工具改一下 `MEMORY_ARBITER_CLIENT` 标识（`openclaw`、`zcode`、`codex`、`cursor`、`claude-code`）。共享路径、向量、模型配置放 `~/.config/memory-arbiter/config.json`；每客户端身份放 MCP env 段。如果不用配置文件，再把 `MEMORY_ARBITER_DB_PATH` 放到每个客户端 env，并指向同一个 SQLite 文件。（OpenDesign 这类 GUI 工具继承宿主 CLI 的配置，不需要单独的 client 名称。）

> ⚠️ **需要新建会话**：MCP Server 在客户端启动时加载，已经打开的会话不会识别新添加的 Server。配置好后请新建一个会话。

### 客户端配置位置

| 客户端 | 配置文件位置 |
|---|---|
| ZCode | `~/.zcode/v2/` 下 MCP 配置 |
| Codex CLI | `~/.codex/` 下 MCP 配置 |
| Claude Code | 项目根目录 `.mcp.json` |
| Cursor | `~/.cursor/mcp.json` |
| OpenClaw | `~/.openclaw/openclaw.json` MCP 配置 |

> **OpenDesign / OpenClaw GUI 类工具**：这类工具寄宿在底层 CLI（Codex CLI、Claude Code 等）之上，**没有自己的 MCP 配置入口**。宿主客户端加载了哪个 MCP Server，GUI 工具就天然能用——例如 Codex CLI 配好了 memory-arbiter，跑在 Codex 上的 OpenDesign 就能直接调用 `memory_search` / `memory_write`，无需额外设置。

### MCP 工具

| 工具 | 说明 |
|---|---|---|
| `memory_write` | 写入记忆（`source_type=user_confirmed` 自动锁定） |
| `memory_search` | 搜索记忆（FTS5 → LIKE 自动降级） |
| `memory_get` | 通过 ID 直接获取单条记忆的完整信息。当已知 `memory_id`（如从冲突列表、审计结果、搜索结果中获取）时，直接用此工具获取记忆详情，无需重新搜索。只读。 |
| `memory_compare` | 比较两条记忆，只返回解释 |
| `memory_arbitrate` | 仲裁冲突，自动判定胜者（`apply=true` 时落记录） |
| `memory_confirm` | 用户确认某条记忆，锁定保护 |
| `memory_supersede` | 显式废弃某条记忆；可突破 user_confirmed/locked 保护（需 `authorized=true`） |
| `memory_list_conflicts` | 列出未解决的冲突 |
| `memory_audit_summary` | 各 workspace 记忆统计概览（条目数、最旧/最新、open 冲突数、来源分布） |
| `memory_store_embedding` | （可选）手动存入或替换某条记忆的语义向量。v0.5.0 配置后也可自动为写入/查询生成向量。 |
| `memory_edit` | （v0.4.0）原地编辑记忆正文（整体替换 `new_content` 或局部替换 `old_text`→`new_text`），旧版本自动存入历史表并同步 FTS。`locked`/`user_confirmed` 记忆需 `authorized=true`。**部分否定的正确做法**——`supersede` 会整条沉掉（连同你没否定那部分），`edit` 只改你要改的。 |
| `memory_history` | （v0.4.0）查看一条记忆的版本演化轨迹（历史快照，按版本号倒序）。只读。 |
| `memory_cleanup_history` | （v0.4.0）清理历史表快照（**绝不碰活跃记录**）。支持单条 / 按时间 / 全量；全量清理需 `authorized=true`。 |
| `memory_status` | 查看运行状态、模式、降级原因 |
| `memory_split` | （v0.6.0）将长记忆分段，实现段落级检索。两阶段：prepare 返回内容批次供外部 LLM 生成段落信息；publish 验证偏移量并原子发布段落 + 向量。需 sqlite-vec + GGUF embedding + `split.enabled`。 |
| `get_sections` | （v0.6.0）按 section ID 获取段落完整原文 + 元数据。`memory_search` 返回 `matched_sections` 后，用此工具取相关段落原文，不必拉取整篇文档。 |
| `memory_split_status` | （v0.6.0）查看某条记忆的分段状态、段落目录、全局向量索引状态。 |
| `memory_rebuild_embeddings` | （v0.6.0）切换 embedding 模型后批量重建所有向量（memory 级 + section 级）。不需要 LLM，只重算向量。 |

### 信息存储规则（喂给你的 Agent）

如果你的客户端同时维护本地 markdown（ZCode 的 `MEMORY.md`、Codex 的 `AGENTS.md` 等），把下面这条规则贴进你 agent 的系统指令，让它知道什么该写到哪里：

> 本地 md 只存自用信息（规则/经验/配置/角色）。凡是有可能被其他 agent 或平台复用的信息（不只是项目信息——需求、调研、决策、进展、用户偏好、知识结论等），一律写入 `memory-arbiter`，必填 `subject`、`tags`、`source_type`（限 requirement/decision/doc_summary/research/progress）、`event_time`（ISO 8601）、`workspace`（项目名）、`source_ref`。查找先 `memory_search`，细节读源文件。发现矛盾不覆盖，用 `memory_supersede` 或 `memory_arbitrate` 处理。待办处理完成后回写原条目（更新 status/subject 标注已完成），不要只在新记忆里提及，否则旧条目仍呈待办状态会误导检索。

### 可选：语义检索（v0.5.0）

默认情况下，memory-arbiter 用的是**字面检索**（FTS5 trigram + BM25 + 软重排）——不依赖 embedding 模型、不引入重依赖、完全本地。绝大多数场景这就够了。

对于"措辞不同但语义相同"的查询（比如搜"快乐"想命中"开心"、搜"金融带货"想命中"金营平台"），你可以**可选开启语义检索**。memory-arbiter **不内置** embedding 模型——你自己带模型，这样默认安装保持轻量，模型选择、语言、成本完全由你掌控。配置完成后，普通 `memory_write` 和 `memory_search(query="...")` 会自动向量化，调用方不用手动传 `query_embedding`。

**四步开启：**

1. 安装 sqlite-vec 和 GGUF 运行时：
   ```bash
   pip install memory-arbiter-mcp[vec]
   pip install llama-cpp-python
   ```
2. 选一个 embedding 模型。v0.5.0 的自动向量化内置支持本地 GGUF：
   - **GGUF（本地，推荐）**——通过 `llama-cpp-python` 跑任意 GGUF embedding 模型。能复用你已有的模型（比如 OpenClaw/llama.cpp 用的）。把脚本指到模型文件：
     `embeddinggemma-300m-qat-Q8_0.gguf` 是一个 768 维的默认选择。
   - **sentence-transformers（本地）**——HuggingFace PyTorch 模型（`bge-small-zh`、`bge-base-en` 等）：
     用你自己的 backfill/query 脚本，把向量传给 `memory_store_embedding` / `memory_search(query_embedding=...)`。
   - **远程 API（OpenAI / 智谱 / 通义）**——在你自己的灌向量脚本里调 API，把向量传给 `memory_store_embedding`。memory-arbiter 这侧只需 `pip install memory-arbiter-mcp[vec]`，不跑任何模型。

   **向量模型速查表**（任选一个，然后把 `MEMORY_ARBITER_VEC_DIM` 设成它的维度）：

   | 方式 | 推荐模型 | 维度 | 来源 | 适用场景 |
   |---|---|---|---|---|
   | GGUF（本地） | `embeddinggemma-300m-qat-Q8_0.gguf` | 768 | [HuggingFace](https://huggingface.co/google/embeddinggemma-300m-qat) | 复用已有模型；不想搭 Python ML 环境 |
   | sentence-transformers | `BAAI/bge-small-zh-v1.5`（中）/ `bge-base-en-v1.5`（英） | 512 / 768 | [HuggingFace](https://huggingface.co/BAAI) | 性价比最高；需要 PyTorch |
   | 远程 API | `text-embedding-3-small`（OpenAI）/ `embedding-3`（智谱） | 1536 / 1024 | 各平台控制台 | 不想本地算力；按调用计费 |

   **自动向量化完整流程一句话**：选 GGUF 模型 → 把 vec/model 配置写到 `~/.config/memory-arbiter/config.json` → 重启 MCP server → 跑一次 `docs/semantic_example.py` 给旧记忆补向量。之后新写入和普通文本查询会自动向量化。
3. 复制配置模板并修改路径：
   ```bash
   mkdir -p ~/.config/memory-arbiter
   # 源码 checkout：
   cp examples/memory-arbiter.config.example.json ~/.config/memory-arbiter/config.json
   # 或 pip 安装用户：
   curl -L https://raw.githubusercontent.com/billy12151/memory-arbiter-mcp/main/examples/memory-arbiter.config.example.json \
     -o ~/.config/memory-arbiter/config.json
   ```
   共享路径、向量、模型配置建议放这里，不放每个 MCP 客户端的 env 段。`MEMORY_ARBITER_CLIENT`、`MEMORY_ARBITER_AGENT_ID`、`MEMORY_ARBITER_WORKSPACE` 仍放各客户端 env，避免所有工具被全局 config 覆盖成同一个身份。`~/.config/memory-arbiter/` 是用户自己的 XDG 配置目录，pip 安装和客户端重装不会覆盖。第一次自动向量化会懒加载模型，可能明显慢一次；后续复用已加载模型。
4. 给现有记忆补向量，然后正常搜索：
   ```bash
   # 从源码 checkout 运行：
   python docs/semantic_example.py                 # 给所有活跃记忆补向量
   python docs/semantic_example.py --query "金营平台营销"   # 试一次语义检索
   ```
   backfill 辅助脚本目前是源码树脚本。只通过 pip 安装的用户，可以 clone 仓库后运行脚本，或用自己的脚本调用 `memory_store_embedding` 给旧记忆补向量。服务配置好后，新写入/编辑会自动写向量。

配置完成后，普通 `memory_search(query="...")` 可以自动生成查询向量。显式 `query_embedding` 仍然支持，并且优先级更高。

**排序规则**：语义召回的候选会给一个*保底分*（略低于正文命中分）——它能压过"正文顺带提及"的噪音，但永远不会盖过真正的标题/标签命中。仲裁和可信度分层逻辑完全不动。

**实测效果（小样本，非正式 benchmark）**：在 v0.3.0 验证用的同一套 15 条黄金查询 + 18 条 pairwise 约束上，开启语义检索后 Top-3 命中率和排序质量进一步提升。最大的亮点是 pairwise 通过率冲到 100%——所有"该排前面的都排在前面了"。SQLite 侧查询延迟约多 8ms；本地 embedding 生成耗时取决于模型和运行时。

| 指标 | bm25 (v0.2.6) | hybrid (v0.3.0) | hybrid + 语义 (v0.3.1) |
|---|---|---|---|
| Top-1 命中率 | 46.7% | 53.3% | 53.3% |
| Top-3 命中率 | 60.0% | 66.7% | **73.3%** |
| Pairwise 通过率 | 77.8% | 88.9% | **100.0%** |

### 可选：长文分段检索（v0.6.0）

当一条记忆超过 `split.threshold`（默认 4000 字符）时，`memory_search` 会返回完整原文——可能几万个 token。分段功能把长文档按语义切成多个 section，每个 section 独立向量化，`memory_search` 只返回最相关段落的元数据（`matched_sections`），不再返回全文。完整设计见 [`docs/section-split-design.md`](docs/section-split-design.md)。

**前置条件（全部满足才能生效）：**
1. 已配置语义检索——sqlite-vec + GGUF embedding（见上方）
2. `config.json` 中 `split.enabled = true`
3. `_vec_index_meta.state == ready`（用 `memory_status` 确认）
4. 有外部 LLM 可用于生成段落标题、摘要和边界锚点

**工作流程：**

```
memory_write(长文档)
  → 正常入库；超过阈值时返回里带 split_hint 建议分段
  → split_hint 只是建议——原文已保存，不丢数据

memory_split(memory_id)                    ← prepare，不写库
  → 按安全批次（llm_batch_chars）返回正文 + section schema
  → Agent 把每批发给外部 LLM → 拿回 title/summary/anchor

memory_split(memory_id, split_decision="split", sections=[...])
  → arbiter 确定性验证偏移量（绝不信任 LLM 的 offset）
  → 生成每段向量（硬前提——必须全部成功）
  → 原子发布段落 + 向量，split_status 切 active

memory_search("查询")
  → 召回路径不变（5 通道，不新增召回通道）
  → 对 active 分段记忆：section Vec 匹配找出相关段落
  → 返回 matched_sections（只有 title+summary）+ section_catalog
  → content=null, content_omitted=true → 省 token

get_sections(memory_id, [section_id, ...])  ← 取特定段落原文
memory_get(memory_id)                       ← 取全文（需要时）
```

**Vec 门禁关闭时**（模型迁移、空间不匹配、query embedding 失败）：`memory_search` 直接返回全文，`section_enhancement_applied=false`——检索能力不会退化，分段增强纯粹是加法。

**配置项**（`config.json` 的 `split` 段）：

| JSON 路径 | env 兜底 | 默认值 | 说明 |
|---|---|---|---|
| `split.enabled` | `MEMORY_ARBITER_SPLIT_ENABLED` | `false` | 总开关。所有前置条件也必须满足。 |
| `split.threshold` | `MEMORY_ARBITER_SPLIT_THRESHOLD` | `4000` | 触发分段提示的最小字符数。 |
| `split.section_vec_distance_threshold` | `MEMORY_ARBITER_SECTION_VEC_DISTANCE_THRESHOLD` | `0.7` | section Vec 余弦距离上限。⚠️ **上线前必须用真实数据校准。** |
| `split.section_fulltext_threshold` | `MEMORY_ARBITER_SECTION_FULLTEXT_THRESHOLD` | `0.8` | 命中段落占比 ≥80% 时返回全文。 |
| `split.max_sections` | `MEMORY_ARBITER_MAX_SECTIONS` | `50` | 每条记忆最大段数（最小 2）。 |
| `split.max_section_chars` | `MEMORY_ARBITER_MAX_SECTION_CHARS` | `3600` | 段落 embedding 输入的字符上限（超出部分截断，有诊断标记）。 |

**完整配置示例（含分段）：**

```json
{
  "db_path": "~/.local/share/memory-arbiter/memory.sqlite3",
  "backup_jsonl": "~/.local/share/memory-arbiter/memory.backup.jsonl",
  "vec": { "enabled": true, "dim": 768 },
  "embedding": {
    "provider": "gguf",
    "model_path": "~/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf",
    "auto_query": true,
    "auto_write": true
  },
  "split": {
    "enabled": true,
    "threshold": 4000,
    "section_vec_distance_threshold": 0.7,
    "section_fulltext_threshold": 0.8,
    "max_sections": 50,
    "max_section_chars": 3600
  }
}
```

**注意事项：**
- `section_vec_distance_threshold`（0.7）是开发期临时值。上线前必须用真实 query-section 对校准（见设计文档 §5）。阈值不对，分段等于白做。
- `memory_edit` 改 content 后会清空所有 section 并重置 `split_status` 为 NULL。需要时用 `memory_split` 重新分段。
- 切换 embedding 模型后，先跑 `memory_rebuild_embeddings(dry_run=True)` 看影响范围，再 `memory_rebuild_embeddings(dry_run=False, batch_size=50)` 批量重建向量。
- 默认**关闭**。如果你的记忆大部分是短笔记、代码片段、对话摘要，不要开启——LLM 调用 + 向量生成的开销不值得。

### 配置

配置读取顺序：`MEMORY_ARBITER_CONFIG` 指定文件 → `~/.config/memory-arbiter/config.json` → 环境变量/default。**耐久的向量和模型配置建议放配置文件**，避免 MCP 客户端重装/迁移时丢失；下面每行同时给出 JSON 路径和对应的 env 兜底（两者都设时配置文件优先）。环境变量仍适合简单 client 标识和 CI 覆盖。完整说明见 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)。

**配置文件字段**（`~/.config/memory-arbiter/config.json`）——路径、向量、embedding 设置：

| JSON 路径 | env 兜底 | 默认值 | 什么时候调 |
|---|---|---|---|
| `db_path` | `MEMORY_ARBITER_DB_PATH` | `./memory_arbiter.sqlite3` | 跨工具共享记忆时设成同一路径。 |
| `backup_jsonl` | `MEMORY_ARBITER_BACKUP_JSONL` | `./memory_arbiter.backup.jsonl` | 追加式 JSONL 备份，仅在 SQLite 只读时启用。 |
| `policy_path` | `MEMORY_ARBITER_POLICY` | _(无)_ | 策略 JSON 文件路径，可按客户端开关、按 agent 允许/拒绝。 |
| `vec.enabled` | `MEMORY_ARBITER_ENABLE_SQLITE_VEC` | `false` | 设 `true` 开启语义检索（需 `pip install memory-arbiter-mcp[vec]`）。 |
| `vec.dim` | `MEMORY_ARBITER_VEC_DIM` | `768` | 必须和你的 embedding 模型一致。改维度要重建 `memories_vec` 表。 |
| `recall_pool_cap` | `MEMORY_ARBITER_RECALL_POOL_CAP` | `50` | **记忆超过约 100 条时调到 100–200**——发现结果里漏了相关记忆，第一个就调它。 |
| `content_like_cap` | `MEMORY_ARBITER_CONTENT_LIKE_CAP` | `30` | 同主题记忆多时调大。 |
| `embedding.provider` | `MEMORY_ARBITER_EMBEDDING_PROVIDER` | 仅在设置 `embedding.model_path` 时推断为 `gguf` | v0.5.0 只支持 `gguf`。没有模型路径时，自动向量化保持关闭。 |
| `embedding.model_path` | `MEMORY_ARBITER_EMBEDDING_MODEL_PATH`（或 legacy `MEMORY_ARBITER_GGUF`） | _(无)_ | GGUF embedding 模型路径。 |
| `embedding.auto_query` | `MEMORY_ARBITER_EMBEDDING_AUTO_QUERY` | `true` | 自动 encode 纯文本查询触发语义检索。 |
| `embedding.auto_write` | `MEMORY_ARBITER_EMBEDDING_AUTO_WRITE` | `true` | 新写入/编辑自动灌向量，立即进语义召回。 |
| `split.enabled` | `MEMORY_ARBITER_SPLIT_ENABLED` | `false` | 开启长文分段检索（v0.6.0）。需 vec + embedding 已配置。详见 [长文分段](#可选长文分段检索v060)。 |
| `split.threshold` | `MEMORY_ARBITER_SPLIT_THRESHOLD` | `4000` | 触发分段的最小字符数。 |
| `split.section_vec_distance_threshold` | `MEMORY_ARBITER_SECTION_VEC_DISTANCE_THRESHOLD` | `0.7` | section Vec 余弦距离上限。⚠️ 上线前校准。 |
| `split.section_fulltext_threshold` | `MEMORY_ARBITER_SECTION_FULLTEXT_THRESHOLD` | `0.8` | 命中段落占比达到此值时返回全文。 |
| `split.max_sections` | `MEMORY_ARBITER_MAX_SECTIONS` | `50` | 每条记忆最大段数（最小 2）。 |
| `split.max_section_chars` | `MEMORY_ARBITER_MAX_SECTION_CHARS` | `3600` | 段落 embedding 输入字符上限。 |

**环境变量**——每客户端身份建议放在各自 MCP env 段。部分字段也有配置文件对应项，但 config 优先；当某个值必须按客户端/会话变化时再放 env。

| 变量 | 默认值 | 什么时候调 |
|---|---|---|
| `MEMORY_ARBITER_CLIENT` | `codex` | 每个工具一个标识（`codex`、`claude-code`、`cursor`、`zcode`…）。 |
| `MEMORY_ARBITER_AGENT_ID` | `default` | 客户端内的 agent 身份。 |
| `MEMORY_ARBITER_WORKSPACE` | `default` | 工作区隔离键。 |
| `MEMORY_ARBITER_CONFIG` | _(无)_ | 可选：指定另一个 JSON 配置文件路径。设置后读取该文件，而不是默认的 `~/.config/memory-arbiter/config.json`；配置文件里的字段仍然优先于其他 env 兜底值。 |
| `MEMORY_ARBITER_RANKING_MODE` | `hybrid` | `hybrid`（默认）或 `bm25`（legacy）。无配置文件对应。 |
| `MEMORY_ARBITER_GGUF` | _(无)_ | 旧版 GGUF 路径兜底；建议改用配置文件里的 `embedding.model_path`。 |

### 数据迁移

换电脑只需拷贝一个文件：

```bash
# 拷贝数据库
cp ~/.local/share/memory-arbiter/memory.sqlite3 新电脑:~/.local/share/memory-arbiter/

# 重新安装项目（.venv 不要拷贝，新机器上重建）
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 测试

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m pytest
```

### License

MIT
