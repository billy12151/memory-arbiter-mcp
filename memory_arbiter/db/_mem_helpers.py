"""memories 共享 helper 叶子（拆分批 ②b：memories 与 _mem_edit 共用，避免环）。

从 memories.py 原样搬出；memories.py re-export 保活（tests/scripts 直取不变）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


_RETIRED_METADATA_KEYS = ("entity", "scope")


def _strip_retired_metadata_keys(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``metadata`` without the retired keys."""
    return {key: value for key, value in metadata.items() if key not in _RETIRED_METADATA_KEYS}


def content_sha(text: str) -> str:
    """sha256 over the raw UTF-8 content bytes — the dedup identity.

    Never normalised: any normalisation folds distinct memories onto one
    hash. Mirrors additive._add_content_sha_dedupe's backfill."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("tags", "metadata", "structured_details"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data


class DuplicateActiveContentError(ValueError):
    """Raising this BEFORE a pending->active flip keeps the caller's rollback
    contract while naming the dedup-gate cause (0.16.6)."""


