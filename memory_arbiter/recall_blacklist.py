"""Recall blacklist: workspaces excluded from *unscoped* semantic recall.

File ``recall_blacklist.jsonl`` lives next to the active database file. One
bare workspace name per line; blank lines and ``#``-comment lines are ignored.
Edits take effect on the next find (mtime-cached per path, no restart needed).
An empty file means "exclude nothing" — an explicit way to opt back in.

Semantics (design 861, v0.15.5):
- Only the *default* recall pool is filtered: a find with no explicit
  workspace. An explicit workspace filter — including a blacklisted one — is
  always honored (that is how twin/governance reach the bucket on purpose).
- Filter-driven recall (empty query + tags_filter/…) and the expired-audit
  path never consult the blacklist: those are explicit, cursor-paginated
  queries, not ambient recall.
- Governance views (review/doctor/audit/conflicts) and id-based reads are
  unaffected.

If the file does not exist the built-in default applies (``mema-twin``, the
twin preference bucket: persona material consumed via compiled prompt
injection, not semantic recall — keeping it out stops double-channel
injection and stops it crowding the recall pool). Creating the file replaces
the default entirely; delete the line to re-admit a workspace.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_BLACKLIST: tuple[str, ...] = ("mema-twin",)

_BLACKLIST_FILENAME = "recall_blacklist.jsonl"
_MAX_NAME_CHARS = 256
_MAX_ENTRIES = 490  # each SQL binds 2N exclusion params — safe even on legacy 999-var sqlite
# Same shape rule as workspace names themselves — keeps NOT IN SQL boring.
_FORBIDDEN_CHARS = ("/", "\\", "\x00")

# path -> (mtime_ns, size, ino, names, warnings); re-read only when the stat
# triple changes so a find storm costs one stat() per call.
_CACHE: dict[Path, tuple[int, int, int, frozenset[str], list[str]]] = {}


def blacklist_path(db_path: str | Path) -> Path:
    """Blacklist file sits in the data directory (next to the DB file)."""
    return Path(db_path).parent / _BLACKLIST_FILENAME


def _parse_lines(text: str) -> tuple[frozenset[str], list[str]]:
    names: set[str] = set()
    warnings: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) > _MAX_NAME_CHARS or any(c in line for c in _FORBIDDEN_CHARS):
            warnings.append(f"recall_blacklist.jsonl:{lineno} invalid entry skipped: {raw[:48]!r}")
            continue
        names.add(line)
    if len(names) > _MAX_ENTRIES:
        warnings.append(
            f"recall_blacklist.jsonl truncated at {_MAX_ENTRIES} entries; extra lines ignored")
        names = set(sorted(names)[:_MAX_ENTRIES])
    return frozenset(names), warnings


def load_blacklist(path: str | Path) -> tuple[frozenset[str], list[str]]:
    """Load the exclusion set for ``path``. Never raises — unreadable file
    degrades to the default set plus a warning so recall keeps working."""
    p = Path(path)
    try:
        stat = p.stat()
        key = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except OSError:
        # Absent/unreadable → built-in default (the shipped prefill).
        _CACHE.pop(p, None)
        return frozenset(DEFAULT_BLACKLIST), []
    cached = _CACHE.get(p)
    if cached is not None and cached[:3] == key:
        return cached[3], cached[4]
    try:
        # utf-8-sig: a BOM must not silently corrupt the first entry into a
        # phantom workspace name that matches nothing.
        names, warnings = _parse_lines(p.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as e:
        warnings = [f"recall_blacklist.jsonl unreadable ({e}); default blacklist applied"]
        _CACHE[p] = (*key, frozenset(DEFAULT_BLACKLIST), warnings)
        return frozenset(DEFAULT_BLACKLIST), warnings
    _CACHE[p] = (*key, names, warnings)
    return names, warnings


def reset_cache() -> None:
    """Test hook — drop the mtime cache."""
    _CACHE.clear()
