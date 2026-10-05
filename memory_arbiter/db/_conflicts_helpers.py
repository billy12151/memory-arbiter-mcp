"""conflicts 共享模块级 helper（拆分批 ②c 断环下沉，纯移动）。"""
from __future__ import annotations

import json
from typing import Any

_MAX_FIELD_CHARS = 16_384
_MAX_SLOT_JSON = 4_096


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _member_ref(member: dict[str, Any]) -> str:
    return f"{int(member['memory_id'])}@{int(member['version'])}"


def _decode_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("slot_key", "candidate_key", "member_versions", "value_groups", "apply_summary"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    return data
