from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class SourceType(str, Enum):
    USER_CONFIRMED = "user_confirmed"
    DOCUMENT_EXTRACTED = "document_extracted"
    AGENT_GENERATED = "agent_generated"
    UNKNOWN = "unknown"
    PENDING = "pending"


class ProtectionLevel(str, Enum):
    NORMAL = "normal"
    PROTECTED = "protected"
    LOCKED = "locked"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"
    PENDING = "pending"
    DELETED = "deleted"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_iso(value: Optional[str]) -> str:
    if not value:
        return utc_now_iso()
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    except ValueError:
        return value


@dataclass
class MemoryRecord:
    content: str
    agent_id: str
    workspace: str
    tags: list[str] = field(default_factory=list)
    source_type: str = SourceType.UNKNOWN.value
    source_ref: Optional[str] = None
    event_time: str = field(default_factory=utc_now_iso)
    ingest_time: str = field(default_factory=utc_now_iso)
    confidence: float = 0.5
    protection_level: str = ProtectionLevel.NORMAL.value
    status: str = MemoryStatus.ACTIVE.value
    subject: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    @classmethod
    def from_input(cls, payload: dict[str, Any], defaults: dict[str, str]) -> "MemoryRecord":
        source_type = payload.get("source_type") or SourceType.UNKNOWN.value
        protection = payload.get("protection_level") or ProtectionLevel.NORMAL.value
        status = payload.get("status") or MemoryStatus.ACTIVE.value
        if source_type == SourceType.USER_CONFIRMED.value:
            protection = ProtectionLevel.LOCKED.value
            status = MemoryStatus.ACTIVE.value
        return cls(
            content=str(payload["content"]).strip(),
            agent_id=str(payload.get("agent_id") or defaults.get("agent_id") or "default"),
            workspace=str(payload.get("workspace") or defaults.get("workspace") or "default"),
            tags=list(payload.get("tags") or []),
            source_type=str(source_type),
            source_ref=payload.get("source_ref"),
            event_time=normalize_iso(payload.get("event_time")),
            ingest_time=normalize_iso(payload.get("ingest_time")),
            confidence=float(payload.get("confidence", 0.5)),
            protection_level=str(protection),
            status=str(status),
            subject=payload.get("subject"),
            metadata=dict(payload.get("metadata") or {}),
        )
