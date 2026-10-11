"""memories 共享 helper（拆分批 ②b：memories 与 _mem_edit 共用）。

从 memories.py 原样搬出；memories.py re-export 保活（tests/scripts 直取不变）。
C4 批（2026-10-11）起 content_sha 委托 evidence.evidence_content_hash——
不再是纯叶子，方向与 db 包其余模块（meta/_mem_vectors）一致。
"""
from __future__ import annotations

import json
from typing import Any

from ..evidence import evidence_content_hash


_RETIRED_METADATA_KEYS = ("entity", "scope")


def _strip_retired_metadata_keys(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``metadata`` without the retired keys."""
    return {key: value for key, value in metadata.items() if key not in _RETIRED_METADATA_KEYS}


def content_sha(text: str) -> str:
    """sha256 over the raw UTF-8 content bytes — the dedup identity.

    Never normalised: any normalisation folds distinct memories onto one
    hash. Delegates to evidence.evidence_content_hash (reuse batch C4:
    single source for the content-hash convention)."""
    return evidence_content_hash(text)


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


