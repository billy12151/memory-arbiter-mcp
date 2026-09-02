"""Shared contract constants — single source of truth for cross-module literals.

Phase 1 (v0.12.4) consolidation. Constants that couple two or more modules (or a
module and its tests) live here so the literal has exactly one home. Original
locations keep re-export aliases so existing imports keep working.
"""
from __future__ import annotations

import unicodedata

# Recent-fallback warning prefix. Emitted by search._recent_fallback when a
# non-empty query produced no direct match and recent memories are returned
# instead. Tests match on the prefix substring — keep the prefix stable, never
# inline it at a call site. Single source; ``search._NO_DIRECT_MATCH_PREFIX``
# re-exports this. (Until 0.14.x the removed bm25 path also sniffed this
# prefix to infer retrieval_mode; since 0.15.0 ranking is hybrid-only and the
# warning is purely user-facing.)
NO_DIRECT_MATCH_PREFIX = "No direct memory match"


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


def isolation_active(level: str) -> bool:
    """True when isolation policy applies by default (weak or strict).

    In ``none`` mode an explicitly supplied workspace can still scope one read;
    this predicate only answers whether the configured mode itself is active.
    """
    return level != Isolation.NONE


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
SEMANTIC_N_CTX = 1024
SEMANTIC_N_THREADS = 4
SEMANTIC_N_BATCH = 128
SEMANTIC_JOB_TIMEOUT_MS = 5000
SEMANTIC_INFERENCE_TIMEOUT_MS = 30000
SEMANTIC_LOAD_TIMEOUT_MS = 120000
SEMANTIC_MIN_PAIR_BUDGET_MS = 1000
SEMANTIC_SCAN_ENHANCE = True
SEMANTIC_SCAN_MAX_PAIRS = 8
SEMANTIC_SCAN_BUDGET_MS = 60000
SEMANTIC_QUEUE_MAX_SIZE = 100
SEMANTIC_MAX_EVIDENCE_UNITS = 24
SEMANTIC_PRELOAD = True
SEMANTIC_RESIDENT = True

# scheduled-task guidance notice (scan_log.jsonl freshness): a library whose
# newest completed scan is older than this, or that has never completed one,
# prompts the agent to offer setting up the two scheduled tasks.
SCAN_TASK_STALE_DAYS = 14
# Negative-cache TTL for the notice check: without it every product response
# would re-read scan_log.jsonl end to end.
SCAN_TASK_RECHECK_SECONDS = 3600

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
SUPERSEDED_LIMIT = 20
NOTICE_SYNC_WAIT_MS = 5000

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

