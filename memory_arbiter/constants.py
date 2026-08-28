"""Shared contract constants — single source of truth for cross-module literals.

Phase 1 (v0.12.4) consolidation. Constants that couple two or more modules (or a
module and its tests) live here so the literal has exactly one home. Original
locations keep re-export aliases so existing imports keep working.
"""
from __future__ import annotations

import unicodedata
from typing import Optional

# Recent-fallback warning prefix. The legacy bm25 search path infers
# retrieval_mode by SNIFFING this warning (it has no structured mode signal), so
# the literal must never be inlined at a call site. Tests match on the prefix
# substring — keep the prefix stable. Single source; ``search._NO_DIRECT_MATCH_PREFIX``
# re-exports this.
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


def is_default_workspace_term(name: Optional[str]) -> bool:
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


def strict_ws(level: str, ws_canonical: Optional[str]) -> Optional[str]:
    """Return ``ws_canonical`` only under strict isolation, else None.

    Folds the repeated ``ws_canonical if (isolation == "strict" and ws_canonical)
    else None`` idiom that appeared 9× across search.py / tools.py. Under weak or
    none the caller must NOT hard-filter by workspace, so this returns None.
    """
    if level == Isolation.STRICT and ws_canonical:
        return ws_canonical
    return None
