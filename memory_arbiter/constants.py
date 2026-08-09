"""Shared contract constants — single source of truth for cross-module literals.

Phase 1 (v0.12.4) consolidation. Constants that couple two or more modules (or a
module and its tests) live here so the literal has exactly one home. Original
locations keep re-export aliases so existing imports keep working.
"""
from __future__ import annotations

from typing import Optional

# Recent-fallback warning prefix. The legacy bm25 search path infers
# retrieval_mode by SNIFFING this warning (it has no structured mode signal), so
# the literal must never be inlined at a call site. Tests match on the prefix
# substring — keep the prefix stable. Single source; ``search._NO_DIRECT_MATCH_PREFIX``
# re-exports this.
NO_DIRECT_MATCH_PREFIX = "No direct memory match"


# ---------------------------------------------------------------------------
# Workspace isolation levels (single source for the three literals).
# ---------------------------------------------------------------------------
# These values are part of the persisted/API contract (settings, env, payload
# fields), so they remain plain strings — NOT enums — to keep JSON/payload
# byte-identical (§9.6 / R3). This class only groups the literals + the two
# predicate helpers that were previously re-implemented at 15+ call sites.


class Isolation:
    """Workspace isolation level literals + predicates.

    none  — workspace fully ignored on the read/write path.
    weak  — soft rerank nudge toward the query workspace; empty workspace allowed.
    strict— hard filter to the query workspace; workspace required on every op.
    """

    NONE = "none"
    WEAK = "weak"
    STRICT = "strict"

    #: Values accepted from config/env; anything else falls back to NONE.
    ALL = (NONE, WEAK, STRICT)


def isolation_active(level: str) -> bool:
    """True when the workspace is consulted at all (weak or strict)."""
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
