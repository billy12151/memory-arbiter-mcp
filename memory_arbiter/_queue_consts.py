"""queue 共享常量与判定 helper（从 queue_protocol.py 搬出，拆分批 ⑦ 纯移动，断环叶子）。
ASSEMBLY_WINDOW 留守 queue_protocol（模块 patch 缝）；本模块名值经 queue_protocol re-export 保活测试 import 面。"""
from __future__ import annotations

from typing import Any


DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 30
GROUP_MEMBER_CAP = 10
INTERNAL_PAIRS_CAP = 8
GROUP_HASHES_CAP = 5


def _decision_truthy(value: Any) -> bool:
    """Decision-item booleans arrive loosely typed from MCP clients
    ("true"/"false" strings, 1/0, bools): only unambiguous true shapes
    count, so a "false" string can never silently waive a guard."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes"}
