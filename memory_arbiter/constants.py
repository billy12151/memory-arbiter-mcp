"""Shared contract constants — single source of truth for cross-module literals.

Phase 1 (v0.12.4) consolidation. Constants that couple two or more modules (or a
module and its tests) live here so the literal has exactly one home. Original
locations keep re-export aliases so existing imports keep working.
"""
from __future__ import annotations

import unicodedata


# ---------------------------------------------------------------------------
# Reserved default workspace pool (single source).
# ---------------------------------------------------------------------------
# Raw workspace strings that mean "no project" resolve to ONE global default
# pool. Every consumer (resolver, publish guards, internal workspace decisions,
# doctor) must test membership through these definitions — never the "default" literal,
# which silently misses 默认/none/null/unknown/未知 and lets a synonym act as a
# second, phantom default pool.
DEFAULT_WORKSPACE_NAME = "default"
DEFAULT_TERMS = frozenset({"", DEFAULT_WORKSPACE_NAME, "默认", "none", "null", "unknown", "未知"})


def is_default_workspace_term(name: str | None) -> bool:
    """True when a workspace string is a reserved default-pool synonym."""
    if name is None:
        return False
    # NFKC first: a full-width IME spelling (ｄｅｆａｕｌｔ, ＮＵＬＬ) must fold to
    # its ASCII twin before the synonym comparison. Exact-codepoint matching
    # would treat the visually identical word as a brand-new workspace and
    # register a phantom second default pool instead of the global one.
    return unicodedata.normalize("NFKC", name.strip()).casefold() in DEFAULT_TERMS


# ---------------------------------------------------------------------------
# Workspace isolation levels (single source for the three literals).
# ---------------------------------------------------------------------------
# These values are part of the persisted/API contract (settings, env, payload
# fields), so they remain plain strings — NOT enums — to keep JSON/payload
# byte-identical (§9.6 / R3). This class only groups the literals + the two
# predicate helpers that were previously re-implemented at 15+ call sites.


class Isolation:
    """Workspace isolation level literals + predicates.

    none  — omitted workspace spans the library; an explicit workspace scopes that read.
    weak  — no hard ACL; workspace provides a soft ranking/hint signal.
    strict— hard scope to the caller canonical; optional guarded vector admission may add neighbors.
    """

    NONE = "none"
    WEAK = "weak"
    STRICT = "strict"

    #: Values accepted from config/env; anything else falls back to NONE.
    ALL = (NONE, WEAK, STRICT)




def strict_ws(level: str, ws_canonical: str | None) -> str | None:
    """Return ``ws_canonical`` only under strict isolation, else None.

    Folds the repeated ``ws_canonical if (isolation == "strict" and ws_canonical)
    else None`` idiom that appeared 9× across search.py / tools.py. Under weak or
    none the caller must NOT hard-filter by workspace, so this returns None.
    """
    if level == Isolation.STRICT and ws_canonical:
        return ws_canonical
    return None


# ---------------------------------------------------------------------------
# Frozen configuration constants (v0.15.0 config slimming).
# ---------------------------------------------------------------------------
# These were user-facing config knobs through 0.14.x and are frozen at their
# former from_env defaults (plan doc: docs/mema-config-slim-plan-2026-08-31,
# mema 804). Changing a value here is a semantic decision (embedding space
# identity, recall behavior), not a tuning move. Consumers import from here
# instead of Settings fields.

# embedding engine (part of embedding_space_id via effective_config)
EMBEDDING_N_CTX = 2048
EMBEDDING_RESERVED_TOKENS = 64
EMBEDDING_MAX_SECTION_CHARS = 3600
# Offline fallback only: disk-size estimation before any model/DB dim is
# known (vnext estimates). Never used to accept or reject vectors.
EMBEDDING_DEFAULT_DIM = 768

# semantic-conflict (Qwen) engine
# n_ctx 2048 (0.15.8): 1024 left no headroom — system(178) + frame/metadata
# (~101) + two 400-char quotes (~460) + the 384-token output budget exceeded
# the window, so long-prompt pairs had their JSON generation truncated at the
# context wall (the top qwen_invalid_output source).
SEMANTIC_N_CTX = 2048
SEMANTIC_N_THREADS = 4
SEMANTIC_N_BATCH = 128
SEMANTIC_JOB_TIMEOUT_MS = 5000
SEMANTIC_INFERENCE_TIMEOUT_MS = 30000
SEMANTIC_LOAD_TIMEOUT_MS = 120000
SEMANTIC_MIN_PAIR_BUDGET_MS = 1000
# Pair-extraction feedback retry (pair-v6): a single protocol invalid output
# (over-limit field / truncated JSON / wrong schema) earns one retry with a
# feedback turn. Truncation retries shrink the evidence quotes and widen the
# output budget — worst-case n_ctx: system(~250) + metadata(~101) +
# 2x400-char quotes(~460) + previous raw(<=384) + feedback(~30) + 384 output
# ~= 1610 < 2048; the truncation retry (~240-char quotes, 512 output) lands
# lower still. Unknown-field/backend failures never retry.
SEMANTIC_PAIR_MAX_ATTEMPTS = 2
SEMANTIC_PAIR_RETRY_QUOTE_CHARS = 240
SEMANTIC_PAIR_RETRY_MAX_TOKENS = 512
# A1 ring (0.15.14): recent examined-pair samples kept for status/doctor
# aggregation (mean/p95 pair_ms, retried ratio, long-decode ratio). A sample
# whose generated tokens reach this share of the 384-token output budget
# counts as a long decode — the slow-but-valid rambling/copy mode that
# neither queue competition nor retries explain.
SEMANTIC_PAIR_RING_SIZE = 20
SEMANTIC_PAIR_LONG_DECODE_TOKENS = 256
# C2 (0.17.0 unit retirement): the worker merge moved indexing into the
# semantic job, so this queue now carries the combined index+detect load —
# it inherits the old evidence-queue watermark (200) instead of 100.
SEMANTIC_QUEUE_MAX_SIZE = 200
EVIDENCE_QUEUE_MAX_SIZE = 200
# C2: wall-clock cap for the job's embed+publish phase. Normal batched embeds
# are ~300ms/memory; a GPU rebuild-class stall must fail the job fast
# (incomplete → retry) instead of parking the single worker thread and
# starving every queued job behind it.
SEMANTIC_EMBED_PHASE_TIMEOUT_MS = 30000
# C7: streaming collection batch size. The producer embeds one batch ahead on
# a single worker thread (queue depth 1) so the main thread's KNN+gates
# overlap the GPU work; value-anchor ranking happens BEFORE batching, so a
# deadline/cap hit stops later batches and always cuts the lowest-value tail.
SEMANTIC_STREAM_BATCH_ROWS = 16
# Adversarial-review follow-up: internal (same-memory) pair construction is
# O(n²) in segment count — row granularity multiplied n (a 300-row table is
# ~45k pairs). Independent from the cross-loop cap so E10① (internal keepers
# land despite cross truncation) keeps its own headroom.
SEMANTIC_INTERNAL_MAX_ROWS = 256
# Harness-found regression: row granularity multiplied internal keepers
# (~dozens per long memory) and they consumed the WHOLE shared Qwen pair
# budget (cross pairs never reached Qwen — recall 0.372→0.163). Internal
# keepers now take at most this many Qwen slots (value-anchored ranking
# already ordered them); E10①'s land-first guarantee is unchanged.
SEMANTIC_INTERNAL_QWEN_MAX_PAIRS = 3
# Owner insight 2026-09-23: the detection window must surface CONFLICT
# candidates, not the globally most-similar rows — a wide window deduped per
# peer keeps every plausible opponent represented; the per-peer dict then
# keeps each opponent's closest row as its representative.
SEMANTIC_CROSS_KNN_WINDOW = 16
# Gate-v2 G2 (owner 拍板 6): an evidence-channel best whose TRUE cosine
# clears this is the text the user asked for (#91: raw KNN first yet
# find-rank 15 after RRF rank-flattening) — the fusion boost bypasses rank
# arithmetic entirely. The detection side reuses the same ceiling: pairs at
# or above it are duplicates, not conflicts (G4 repeatability skip).
COS_EXACT_BOOST = 0.98
# Gate-v2 G4 candidate cosine band on TRUE row-to-row cosine (calibration
# table §1: true conflicts 0.80-0.97, same-topic non-conflicts 0.64-0.75,
# random same-bucket p5=0.554 — below-floor pairs are noise; at/above-ceil
# pairs are duplicates that belong to the similarity/duplicates channel,
# hence the ceil == COS_EXACT_BOOST). Non-unit row vectors (|v|≈16.5) make
# L2-to-cos conversion unreliable, so the gate runs on fetched vectors, one
# batched IN query per collection loop.
SEMANTIC_CANDIDATE_COS_FLOOR = 0.60
SEMANTIC_CANDIDATE_COS_CEIL = COS_EXACT_BOOST
# Gate-v2 G5 title coarse-screen width (方案: 30-50 条记忆, 初值 30-50,
# 宽不罚——后续层会筛; 窄才漏). One subject-row KNN per write.
SEMANTIC_NEIGHBOR_SCREEN = 50
# 0.15.14 (A5): unit cap covers the real-library maximum (62 observed in #956);
# collection cost per unit is one k=5 KNN + rule gate (milliseconds) — the
# expensive resource is Qwen pairs, bounded separately below.
SEMANTIC_MAX_EVIDENCE_UNITS = 64
# 0.15.14 (A5): deterministic second gate on Qwen work per write-check — the
# fair job deadline remains the first. Formula (plan mema-01514 §A5):
# clamp(6, 16, round(20s target check budget ÷ p95 pair wall)); with the
# grammar-free decode (A2) the measured p95 pair ≈ 1.6s ×1.5 load margin
# ⇒ 10. Pairs beyond the cap report incomplete reason=pairs_examined_capped.
SEMANTIC_MAX_EXAMINED_PAIRS = 10
# 0.17.0 P2-3.1: row cap for the row-level conflict channel (sentences +
# header-folded table rows; ~35 rows per typical memory, 256 covers the
# real-library tail). Rows sort value-anchored-first before the cap bites
# (P2-3.1, plan §5); the truncation reason is rows_capped. Initial value —
# P2-3.2 recalibrates on the noisy corpus (constants keep the evidence chain).
SEMANTIC_MAX_ROWS = 256
# 0.17.0 P2-3.4: candidate pair_score weights (order-only, never a verdict).
# Base is the C4 subject/tags overlap; value features outrank topic
# similarity because 98% of same-topic pairs are continuations, not
# conflicts (12th/13th-round evidence). Weights sum to 1.0; recalibrated
# with the P2-0 corpus during P2-3.2.
# 0.17.0 P2-3.3/P2-5.3: claims channel attr-vector gate τ (8th-round spike:
# A/B recall 4/4, C2+D1 accepted as advisory FPs; asymmetric-benefit doctrine).
CLAIM_ATTR_TAU = 0.70
# 0.17.0 review A3: the claims channel is zero-Qwen with no natural pairs
# cap — notices per write are bounded here instead (overflow visible).
CLAIMS_MAX_NOTICES_PER_WRITE = 5
PAIR_SCORE_W_OVERLAP = 0.40
PAIR_SCORE_W_NUMERIC_ROUTE = 0.30
PAIR_SCORE_W_BOTH_VALUES = 0.30
SEMANTIC_PRELOAD = True
SEMANTIC_RESIDENT = True

# scheduled-task guidance notice (scan_log.jsonl freshness): a library whose
# newest completed scan is older than this, or that has never completed one,
# prompts the agent to offer setting up the two scheduled tasks.
SCAN_TASK_STALE_DAYS = 14
# C5 broken-chain alarm (0.15.13): a routine scan's page-progress kv that is
# incomplete and older than this many hours, with no completion line after
# it, means the round was interrupted mid-walk (a whole chain is 15-20 min).
SCAN_CHAIN_STALE_HOURS = 1
# Negative-cache TTL for the notice check: without it every product response
# would re-read scan_log.jsonl end to end.
SCAN_TASK_RECHECK_SECONDS = 3600

# Write-time duplicate hint: subject gate over candidate recall, then a
# content-confirmation gate (owner 2026-09-16 redesign, three-measurement
# evidence from the eval harness). The old tag-Jaccard second gate is GONE:
# on the real library it blocked true cross-habit rewrites (duplicate writes
# whose tags drifted, Jaccard 0.29–0.70) while waving through same-subject
# serials (98% of subject-similar pairs are same-topic continuations, not
# duplicates — content-similarity median 0.15). The subject bar 0.95→0.80
# same day (gate only fired at ratio ≥0.968, missing natural-suffix
# rewrites; 0.80 recovers 8/12 on the natural-gradient corpus, 0/36 false
# positives — see eval/sweep_similar_threshold.py).
# Content gate: full-body char-trigram cosine (semantic_conflict._char_ngrams
# + _cosine, the deterministic pre-filter's own implementation) ≥ 0.40.
# Calibration on 14 positives / 154 negatives (real library + corpus):
# negatives top out at 0.374, positives ≥0.427 → 0.40 sits mid-band with
# recall 11/14 and 0 false positives. Bodies under the char floor skip the
# confirmation and hint anyway flagged low_confidence (set variance is too
# high on short texts; prefer recall). Vector confirmation was measured and
# REJECTED: paraphrase-type near-dups (0.87–0.92) and serials (0.81–0.93)
# are inseparable at 0.5B-embedding resolution — that split belongs to the
# Qwen scan line, not this millisecond sync channel.
WRITE_SIMILAR_SUBJECT_RATIO = 0.8
WRITE_SIMILAR_CONTENT_COSINE = 0.4
# 0.17.0 P2-7 校准轮（cand1 数据）：双轴 OR 规则——(subject≥0.45 且
# content≥0.22) 或 (subject≥0.80 且 content≥0.15)。语料实测分布：真近重复
# (含 noisy 改写) s∈[0.32,1.0]/c∈[0.17,0.74]，组内样板互撞带 s≈0.10/c∈
# [0.60,0.74]（旧豁免线 0.60 正落在带内致 sim07 假阳性），负例带 c≤0.16。
# 地板 0.22 取负例带之上、真值地板之下；0.45/0.80 分层兜 sim12 型低内容对。
WRITE_SIMILAR_SUBJECT_FLOOR = 0.45
WRITE_SIMILAR_CONTENT_FLOOR = 0.22
WRITE_SIMILAR_SUBJECT_STRONG = 0.80
WRITE_SIMILAR_CONTENT_MIN = 0.15
WRITE_SIMILAR_MIN_CONTENT_CHARS = 40
WRITE_SIMILAR_MAX_HINTS = 2
# Recall channel (0.15.3): with a loaded embedder the hint recalls candidates
# by subject+tags KNN over subject_tags_vec instead of scanning every
# same-workspace active row; without one the legacy scan keeps a hard row cap.
# 20→10 (2026-09-16): one index query either way — the wider window bought
# nothing (true duplicates rank at the top) and only fattened the candidate
# pool the content gate then has to chew.
WRITE_DUPLICATE_VEC_TOP_K = 10
WRITE_SIMILAR_FALLBACK_SCAN_LIMIT = 500

# memory_repair(task="scan_duplicates") — full-library near-duplicate sweep.
# One-shot response bounded by a global pair cap (default lightweight fields;
# include_quotes adds evidence quotes), so an agent session never has to
# ingest the unbounded duplicates enumeration that per-page scan_candidates
# would require. The sweep loop itself is bounded by a hard page ceiling
# (pages × batch anchors): a clean library walks every page, but no call can
# run unbounded work; hitting the ceiling reports truncated=true.
SCAN_DUPLICATES_MAX_RESULTS = 200
SCAN_DUPLICATES_BATCH = 100
SCAN_DUPLICATES_MAX_PAGES = 200

# workspace normalization Qwen guard (A/B: top-3 beats top-5; over-distance
# candidates must never reach the model — see tools._suggest_workspace_candidate)
QWEN_CANDIDATE_DISTANCE = 0.25
QWEN_CANDIDATE_TOP_K = 3
QWEN_BUDGET_MS = 750

# workspace recall / normalization thresholds (global; NOT per-isolation)
WORKSPACE_MATCH_DISTANCE = 0.25
WORKSPACE_RECALL_ADMISSION = True
WORKSPACE_RECALL_CUTOFF = 0.25
WORKSPACE_WEAK_VECTOR_WEIGHT = False
WORKSPACE_MIN_NAME_LEN = 3

# retrieval / paging caps
RECALL_POOL_CAP = 50
CONTENT_LIKE_CAP = 30
# v0.15.9 relevance floor (docs/eval-relevance-floor-2026-09-08.md): reranked
# candidates below this final_score never enter a find query-recall result
# page. Originally calibrated on the live library (340 labeled candidates):
# the 7.6-8.1 band measured 94% irrelevant; F=8.1 kept 41/45 relevant, cut 73%
# of irrelevant candidates and cleared 91% of legal-form noise.
# 0.16.11 recalibration (2026-09-20, owner): fine floor sweep on corpus
# recall-v1 with _final_score instrumented runs (eval/results/
# recall-floorcurve81.json; offline curve double-validated against measured
# 8.1/8.6 runs). The weak false-pull engine (vec-only fringe candidates,
# 2.5 vec floor + ~4.9 fusion + bonuses) clusters at 8.212-8.218 while the
# nearest paraphrase true-positive sits at 8.278 — the gap (8.218, 8.278]
# kills 5/7 legal-form false pulls at ZERO recall cost (7/21 -> 2/8,
# Recall@10 unchanged at 43/45). The window is corpus-narrow (~0.06): any
# embedder swap or scoring-dimension change reopens the question (mema
# id=1029); the structural fix (a separate bar for vec-only candidates) is
# the recorded follow-up. Scope unchanged: active query-recall direct path
# only (browse / filter-driven recall / expired audit are exempt).
QUERY_RECALL_SCORE_FLOOR = 8.25
# v0.15.9: bounded reserved pool seats for channel-3 surface hits (exact
# subject/tags token matches). Without them the fusion-order trim starves
# surface rows — they enter the pool last (worst lexical ranks) and get cut.
SURFACE_ADMISSION_QUOTA = 10

# v0.15.9 batch_find bounds (mema 923 §6)
MAX_BATCH_FIND_QUERIES = 8
BATCH_FIND_DEFAULT_LIMIT_PER_QUERY = 3
BATCH_FIND_MAX_LIMIT_PER_QUERY = 20
BATCH_FIND_TOTAL_BYTES = 64 * 1024
SUPERSEDED_LIMIT = 20
NOTICE_SYNC_WAIT_MS = 3000

# 0.16.0 batch read caps (plan §1.5/§6⑭/§6⑰; owner-pinned numbers).
# preview/hits items carry structurally bounded payloads so a per-call count
# cap suffices; full content is bounded per memory at 2MB, so a byte budget is
# the real gate: over-budget batches return a structured over-long prompt
# (never a silent truncation) and the agent re-reads items individually.
BATCH_READ_MAX_PREVIEW = 50
BATCH_READ_MAX_HITS = 50
BATCH_READ_MAX_FULL = 10
BATCH_READ_FULL_BUDGET_BYTES = 80 * 1024
BATCH_READ_FULL_BUDGET_MAX_BYTES = 100 * 1024

# 0.16.0 tag discipline (plan §6⑮): a single memory's persisted tag total is
# capped — tags are a retrieval dimension, not an event log. One-call inputs
# keep the MAX_TAGS=100 bound; the total cap applies to the merged on-row set.
MAX_MEMORY_TOTAL_TAGS = 32

# Pipeline kick defaults: one kick is a bounded synchronous batch (the task
# re-kicks until complete; no resident walker per §6⑦).
SCAN_PIPELINE_KICK_TIME_BUDGET_S = 45.0
SCAN_PIPELINE_KICK_MAX_MEMORIES = 400
SCAN_PIPELINE_NEIGHBOR_K = 10
# 0.16.2 §1.5: machine-decidable check routes only generate within the top-3
# neighbour ranks; notify routes keep the full top-10 (real-conflict recall
# has no threshold). A rank tightening, not an absolute distance band.
SCAN_MACHINE_ROUTE_TOP_K = 3
# 0.17.0 P2-6.2: slow-lane anchors per kick (owner default 20, adjustable).
SCAN_SLOW_LANE_PER_KICK = 20

# 0.16.0 workspace-normalization gate (plan §6⑫); 0.16.2 recalibrates the
# vote threshold from the absolute >=8/10 (based on the 930/950 cases owner
# later re-adjudicated as true moves) to a proportional gate judged ONLY
# through normalize_gate.normalize_gate — five consumers, one function.
# Protected buckets (E6 + owner confirm): NO autonomous move in either
# direction — persona isolation depends on it. Manual authorized moves are
# unaffected; protected-involved suspects surface as user hints only.
PROTECTED_WORKSPACES = frozenset({"mema-twin", "mema-twin-dev"})
NORMALIZE_VOTE_NEIGHBORS = 10
NORMALIZE_VOTE_MIN_FOREIGN = 4  # top foreign bucket absolute floor
NORMALIZE_FOREIGN_SHARE_MIN = 0.60  # top foreign votes / all foreign votes
NORMALIZE_MIN_CONF = 0.8

# HTTP transport fixed surface
MCP_HTTP_PATH = "/mcp"
MCP_HTTP_BODY_LIMIT = 4 * 1024 * 1024

# Environment variables dropped in 0.15.0 (config is file-only; these are
# scanned at startup so a stale export produces a visible warning instead of
# silently not taking effect). Launch-context vars (CONFIG/DB_PATH/
# BACKUP_JSONL/MCP_TRANSPORT/CLIENT/AGENT_ID) are NOT listed — they remain.
REMOVED_ENV_NAMES = (
    "MEMORY_ARBITER_CONTENT_LIKE_CAP",
    "MEMORY_ARBITER_EMBEDDING_AUTO_QUERY",
    "MEMORY_ARBITER_EMBEDDING_AUTO_WRITE",
    "MEMORY_ARBITER_EMBEDDING_MAX_UNIT_CHARS",
    "MEMORY_ARBITER_EMBEDDING_MODEL_PATH",
    "MEMORY_ARBITER_EMBEDDING_N_CTX",
    "MEMORY_ARBITER_EMBEDDING_PROVIDER",
    "MEMORY_ARBITER_EMBEDDING_RESERVED_TOKENS",
    "MEMORY_ARBITER_ENABLE_SQLITE_VEC",
    "MEMORY_ARBITER_GGUF",
    "MEMORY_ARBITER_ISOLATION",
    "MEMORY_ARBITER_MCP_HTTP_HOST",
    "MEMORY_ARBITER_MCP_HTTP_JSON_RESPONSE",
    "MEMORY_ARBITER_MCP_HTTP_MAX_REQUEST_BODY_SIZE",
    "MEMORY_ARBITER_MCP_HTTP_PATH",
    "MEMORY_ARBITER_MCP_HTTP_PORT",
    "MEMORY_ARBITER_MCP_HTTP_STATELESS",
    "MEMORY_ARBITER_NOTICE_SYNC_WAIT_MS",
    "MEMORY_ARBITER_POLICY",
    "MEMORY_ARBITER_RANKING_MODE",
    "MEMORY_ARBITER_RECALL_POOL_CAP",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_BACKEND",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_INFERENCE_TIMEOUT_MS",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_JOB_TIMEOUT_MS",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_LOAD_TIMEOUT_MS",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_MAX_CONCURRENCY",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_MAX_EVIDENCE_UNITS",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_MAX_NOTICE_PAIRS",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_MIN_PAIR_BUDGET_MS",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_MODEL_PATH",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_N_BATCH",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_N_CTX",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_N_THREADS",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_ON_WRITE",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_PRELOAD",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_QUEUE_MAX_SIZE",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_RESIDENT",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_SCAN_BUDGET_MS",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_SCAN_ENHANCE",
    "MEMORY_ARBITER_SEMANTIC_CONFLICT_SCAN_MAX_PAIRS",
    "MEMORY_ARBITER_SUPERSEDED_LIMIT",
    "MEMORY_ARBITER_TOOL_PROFILE",
    "MEMORY_ARBITER_UPDATE_CHECK_ENABLED",
    "MEMORY_ARBITER_VEC_DIM",
    "MEMORY_ARBITER_WORKSPACE",
    "MEMORY_ARBITER_WORKSPACE_MATCH_DISTANCE",
    "MEMORY_ARBITER_WORKSPACE_MIN_NAME_LEN",
    "MEMORY_ARBITER_WORKSPACE_QWEN_BUDGET_MS",
    "MEMORY_ARBITER_WORKSPACE_QWEN_CANDIDATE_DISTANCE",
    "MEMORY_ARBITER_WORKSPACE_QWEN_CANDIDATE_TOP_K",
    "MEMORY_ARBITER_WORKSPACE_RECALL_ADMISSION",
    "MEMORY_ARBITER_WORKSPACE_RECALL_CUTOFF",
    "MEMORY_ARBITER_WORKSPACE_WEAK_VECTOR_WEIGHT",
)

