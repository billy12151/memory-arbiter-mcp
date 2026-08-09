"""Shared contract constants — single source of truth for cross-module literals.

Phase 1 (v0.12.4) consolidation. Constants that couple two or more modules (or a
module and its tests) live here so the literal has exactly one home. Original
locations keep re-export aliases so existing imports keep working.
"""
from __future__ import annotations

# Recent-fallback warning prefix. The legacy bm25 search path infers
# retrieval_mode by SNIFFING this warning (it has no structured mode signal), so
# the literal must never be inlined at a call site. Tests match on the prefix
# substring — keep the prefix stable. Single source; ``search._NO_DIRECT_MATCH_PREFIX``
# re-exports this.
NO_DIRECT_MATCH_PREFIX = "No direct memory match"
