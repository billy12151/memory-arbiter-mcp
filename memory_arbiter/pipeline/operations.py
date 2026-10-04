"""Governance, edit, status, and maintenance operations for MemoryTools (Phase 4 extraction)."""
from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from ..acl import CallerWorkspace, WorkspaceScope
from ..embedder import ManagedEmbedder
from ._ops_conflicts import _OpsConflicts
from ._ops_ws_lifecycle import _OpsWsLifecycle
from ._ops_ws_move import _OpsWsMove
from ._ops_content import _OpsContent
from ._ops_status import _OpsStatus


def _embed_input_profile(record: dict[str, Any] | None) -> tuple[str, str, str]:
    """Derived-embedding input profile (0.16.12 P2-T1): the normalized
    (subject, tags, content) triple the two recall-vector refreshes re-embed
    from. Row-level equality of this triple means the refreshed vectors would
    be byte-identical, so the re-embed is skippable. Accepts tags in either
    shape — the raw JSON string (SQL rows) or an already-parsed list
    (missing_*_rows snapshots) — so profiles compare equal across sources."""
    if not record:
        return ("", "", "")
    raw_tags = record.get("tags")
    if isinstance(raw_tags, str):
        try:
            tags = json.loads(raw_tags or "[]")
        except (TypeError, ValueError):
            tags = []
    elif isinstance(raw_tags, (list, tuple)):
        tags = list(raw_tags)
    else:
        tags = []
    # Mirror _subject_tags_embed_text/_summary_embed_text exactly:
    # strip each tag, drop empties, THEN sort — an unstripped join would let
    # distinct embed inputs share a profile (second-round adversarial finding).
    cleaned = sorted(str(t).strip() for t in tags if str(t).strip())
    return (
        str(record.get("subject") or ""),
        " ".join(cleaned),
        str(record.get("content") or ""),
    )

if TYPE_CHECKING:
    from ..update_monitor import UpdateMonitor
    from ..tools import MemoryTools


class _SubjectTagView:
    """Minimal read-only view for write-time duplicate-hint checks.

    ``_similar_active_notice`` reads ``.subject``/``.tags``/``.content``; this
    lightweight view lets status-change paths (confirm/activate) reuse the
    same check without building a full ``MemoryRecord``. Content is coerced
    to str so the content-confirmation gate sees a real body (a missing
    content would silently degrade the hint to low_confidence-only).
    """

    __slots__ = ("subject", "tags", "content")

    def __init__(
        self, subject: str | None, tags: list[str] | None, content: str | None = None,
    ) -> None:
        self.subject = subject
        self.tags = tags or []
        self.content = str(content or "")


class OperationsPipeline(_OpsConflicts, _OpsWsLifecycle, _OpsWsMove, _OpsContent, _OpsStatus):
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools
        self.db = tools.db
        self.settings = tools.settings

    @property
    def _update_monitor(self) -> "UpdateMonitor | None":
        # Assigned on MemoryTools after pipeline construction: resolve lazily.
        return self._tools._update_monitor

    def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace":
        return self._tools._caller_workspace(*args, **kwargs)

    def _conflict_detail_for_workspace(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._conflict_detail_for_workspace(*args, **kwargs)

    def _embedding_configured(self) -> bool:
        return self._tools._embedding_configured()

    def _post_commit(
        self, *args: Any, **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._tools._post_commit(*args, **kwargs)

    def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_active_embedder()

    def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_embedder()

    def _get_memory_visible(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._get_memory_visible(*args, **kwargs)

    def _is_truthy(self, *args: Any, **kwargs: Any) -> bool:
        return self._tools._is_truthy(*args, **kwargs)

    def _semantic_notice_workspace_scope(self, *args: Any, **kwargs: Any) -> "WorkspaceScope":
        return self._tools._semantic_notice_workspace_scope(*args, **kwargs)

    def _semantic_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._tools._semantic_status(*args, **kwargs)

    def _strict_acl_unavailable(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._strict_acl_unavailable(*args, **kwargs)

    def current_agent_id(self) -> "str | None":
        return self._tools.current_agent_id()

    def current_client(self) -> "str | None":
        return self._tools.current_client()


    def wait_semantic_worker_drained(self, *args: Any, **kwargs: Any) -> bool:
        # C2: replay's B-D2 guarantee (complete receipt ⇒ index persisted)
        # drains the semantic queue — indexing lives there now.
        return self._tools.wait_semantic_worker_drained(*args, **kwargs)

    @staticmethod
    def _compare_memories(*args: Any, **kwargs: Any) -> Any:
        # Preserve legacy patch seam for memory_arbiter.tools.compare_memories.
        from .. import tools as tools_mod
        return getattr(tools_mod, "compare_memories")(*args, **kwargs)
