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


@dataclass(frozen=True)
class ConflictMember:
    memory_id: int
    version: int
    attribute_raw: str
    value_raw: str
    normalized_attribute: str
    normalized_value: str
    evidence_quote: str
    evidence_span: tuple[int, int]
    content_hash: str
    direction: str
    prompt_version: Optional[str]
    detector_version: str
    evidence_unit: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id, "version": self.version,
            "attribute_raw": self.attribute_raw, "value_raw": self.value_raw,
            "normalized_attribute": self.normalized_attribute,
            "normalized_value": self.normalized_value,
            "evidence_quote": self.evidence_quote,
            "evidence_span": list(self.evidence_span), "content_hash": self.content_hash,
            "direction": self.direction, "prompt_version": self.prompt_version,
            "detector_version": self.detector_version, "evidence_unit": self.evidence_unit,
        }


@dataclass(frozen=True)
class ConflictValueGroup:
    normalized_value: str
    display_value: str
    members: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"normalized_value": self.normalized_value, "display_value": self.display_value,
                "members": list(self.members)}


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 (µs stripped). Implementation: timeutil.utc_now_iso (Phase 1)."""
    from .timeutil import utc_now_iso
    return utc_now_iso()


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
        # tags must be a list; a string like "todo" would otherwise be split
        # into ['t','o','d','o'] by list(). Coerce non-list values to [] so a
        # malformed payload cannot silently corrupt the tag index.
        raw_tags = payload.get("tags")
        tags = list(raw_tags) if isinstance(raw_tags, (list, tuple)) else []
        return cls(
            content=str(payload["content"]).strip(),
            agent_id=str(payload.get("agent_id") or defaults.get("agent_id") or "default"),
            workspace=str(payload.get("workspace") or defaults.get("workspace") or "default"),
            tags=tags,
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
