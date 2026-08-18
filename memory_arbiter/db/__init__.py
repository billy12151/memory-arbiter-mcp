"""memory_arbiter.db — persistence facade.

Phase 3 (v0.12.4): the former single-file ``db.py`` is now a package. The
``MemoryDB`` class and module-level helpers live in ``db/core.py`` (and, as later
sub-stores are extracted, in sibling modules). This ``__init__`` re-exports the
full original symbol surface — public AND private — so every existing
``from memory_arbiter.db import X`` keeps working unchanged (R8).

It also re-exports the third-party/stdlib names the original db.py surfaced as a
side effect of its top-level imports (``json``, ``re``, ``sqlite3``, …), so that
``memory_arbiter.db.<name>`` attribute access and the symbol snapshot stay
byte-identical to the pre-split module.
"""
from __future__ import annotations

from .core import (
    MemoryDB,
    _BUSY_TIMEOUT_MS,
    _CJK_CHAR_RE,
    _canon_entity,
    _canon_scope,
    _coerce_tags_db,
    _coerce_ws,
    _normalize_alias_key,
    _subject_tokens,
    row_to_dict,
)

from .core import (  # noqa: F401
    ConflictJudgmentStore,
    DegradeState,
    MemoryRecord,
    Path,
    Settings,
    Tuple,
    contextmanager,
    datetime,
    json,
    re,
    sqlite3,
    time,
    timezone,
    utc_now_iso,
    uuid,
)
from .core import Optional, Iterator, Any  # noqa: F401

__all__ = [
    "MemoryDB",
    "row_to_dict",
    "_BUSY_TIMEOUT_MS",
    "_CJK_CHAR_RE",
    "_canon_entity",
    "_canon_scope",
    "_coerce_tags_db",
    "_coerce_ws",
    "_normalize_alias_key",
    "_subject_tokens",
    "Any",
    "ConflictJudgmentStore",
    "DegradeState",
    "Iterator",
    "MemoryRecord",
    "Optional",
    "Path",
    "Settings",
    "Tuple",
    "contextmanager",
    "datetime",
    "json",
    "re",
    "sqlite3",
    "time",
    "timezone",
    "utc_now_iso",
    "uuid",
]
