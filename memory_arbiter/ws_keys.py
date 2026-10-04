"""Workspace 折叠键的唯一定义（拆分批 ①，抽象收敛）。

四个键函数此前有三份实现：db/workspaces.py 的原版、twin_redirect.py 的
有意复制（A9，0.17.1 修复批——当时的理由是「import 路径经 db/__init__
有副作用」）。本模块收拢为**顶层叶子**（不属于 db/ 包，import 它不触发
db/__init__.py 拉入 core 全链），两处消费方都从这里 import，原路径各自
re-export 保活。

键语义（原 docstring 全文随迁）：
- ``_normalize_alias_key``：治理表别名键（casefold + 空白折叠）；
- ``_mechanical_ws_key``：机械折叠键（再去 ``-_\\s``，拼写变体碰撞）；
- ``_normalize_ws_group_key``：批量归并分组键（str.lower 严于 casefold，
  错并不可恢复故从窄）；
- ``_coerce_ws``：松散 JSON 到 str 的矫正。
"""
from __future__ import annotations

import re
from typing import Any

from .constants import DEFAULT_TERMS

# resolve_workspace_canonical 与 admitted_canonicals 共用的默认词 SQL 片段。
_DEFAULT_TERM_SQL_PARAMS = tuple(sorted({t.casefold() for t in DEFAULT_TERMS if t}))
_DEFAULT_TERM_SQL_NOT_IN = (
    " AND lower(c.name) NOT IN (" + ",".join("?" for _ in _DEFAULT_TERM_SQL_PARAMS) + ")"
)


def _normalize_alias_key(ws: str | None) -> str:
    """Normalize a workspace string into a stable alias-governance key.

    Case-folded + whitespace-collapsed so "金营项目 " and "金营项目" map to the
    same alias row. This is only the governance lookup key; the display
    canonical is stored verbatim in workspace_aliases.canonical. Non-string
    inputs (loosely-typed MCP JSON) are coerced to str rather than crashing.
    """
    if ws is None:
        return ""
    s = ws.strip() if isinstance(ws, str) else str(ws).strip()
    if not s:
        return ""
    return " ".join(s.split()).casefold()


def _mechanical_ws_key(ws: str | None) -> str:
    """Fold a workspace string to a deterministic case/separator-insensitive key.

    Unlike _normalize_alias_key (which only case-folds + collapses whitespace
    for the governance table), this also strips hyphens and underscores so pure
    spelling variants of one canonical collide: AgentLane / agent-lane /
    agent_lane -> "agentlane". Used only to reuse an EXISTING canonical, never to
    invent a new spelling. Returns "" for empty/whitespace input so blank
    workspaces never collapse together here.
    """
    if not isinstance(ws, str):
        ws = "" if ws is None else str(ws)
    return re.sub(r"[\s_\-]+", "", ws).casefold()


def _normalize_ws_group_key(ws: str | None) -> str:
    """Grouping key for ``normalize_workspace_canonicals`` — deliberately
    STRICER than ``_mechanical_ws_key``.

    Same separator stripping, but ``str.lower`` instead of ``casefold``:
    'Straße' vs 'strasse' and the 'ﬁ' ligature vs 'file' stay distinct
    instead of collapsing. Normalize is a bulk destructive merge, so its
    grouping key errs toward NOT merging (a missed variant is recoverable, a
    wrong merge is not); the non-destructive orthography reuse in the
    decision primitive / resolver / migrate keeps the full casefold of
    ``_mechanical_ws_key``. The whole planner (grouping, respected-rejection
    match, shadowed-redirect match, rejected-only twin map) uses this one key
    space so every comparison stays consistent with the groups built from it.
    """
    if not isinstance(ws, str):
        ws = "" if ws is None else str(ws)
    return re.sub(r"[\s_\-]+", "", ws).lower()


def _coerce_ws(ws: Any) -> str:
    """Coerce a possibly-non-string workspace value to a trimmed str.

    MCP clients send loosely-typed JSON; alias/canonical may arrive as int/
    list/dict. Coerce rather than raise AttributeError on ``.strip()``.
    """
    if ws is None:
        return ""
    return ws.strip() if isinstance(ws, str) else str(ws).strip()
