from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .models import ProtectionLevel, SourceType


SOURCE_RANK = {
    SourceType.USER_CONFIRMED.value: 100,
    SourceType.DOCUMENT_EXTRACTED.value: 70,
    SourceType.AGENT_GENERATED.value: 45,
    SourceType.PENDING.value: 20,
    SourceType.UNKNOWN.value: 10,
}


def compare_memories(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    left_protected = is_user_protected(left)
    right_protected = is_user_protected(right)
    if left_protected and not right_protected:
        return decision(left, right, left, "left", ["left is user_confirmed or locked; automatic overwrite is forbidden"])
    if right_protected and not left_protected:
        return decision(left, right, right, "right", ["right is user_confirmed or locked; automatic overwrite is forbidden"])
    if left_protected and right_protected:
        return decision(left, right, None, "manual_review", ["both records are user protected; manual review required"])

    left_event = parse_time(left.get("event_time"))
    right_event = parse_time(right.get("event_time"))
    if left_event != right_event:
        winner = left if left_event > right_event else right
        side = "left" if winner is left else "right"
        reasons.append(f"{side} has newer event_time; fact occurrence time has priority")
        return decision(left, right, winner, side, reasons)

    left_source = SOURCE_RANK.get(left.get("source_type"), 0)
    right_source = SOURCE_RANK.get(right.get("source_type"), 0)
    if left_source != right_source:
        winner = left if left_source > right_source else right
        side = "left" if winner is left else "right"
        reasons.append(f"{side} has stronger source_type")
        return decision(left, right, winner, side, reasons)

    left_conf = float(left.get("confidence") or 0)
    right_conf = float(right.get("confidence") or 0)
    if left_conf != right_conf:
        winner = left if left_conf > right_conf else right
        side = "left" if winner is left else "right"
        reasons.append(f"{side} has higher confidence")
        return decision(left, right, winner, side, reasons)

    left_ingest = parse_time(left.get("ingest_time"))
    right_ingest = parse_time(right.get("ingest_time"))
    if left_ingest != right_ingest:
        winner = left if left_ingest > right_ingest else right
        side = "left" if winner is left else "right"
        reasons.append(f"{side} has newer ingest_time after equal event_time/source/confidence")
        return decision(left, right, winner, side, reasons)

    return decision(left, right, None, "tie", ["records are equivalent under configured arbitration rules"])


def is_user_protected(record: dict[str, Any]) -> bool:
    return record.get("source_type") == SourceType.USER_CONFIRMED.value or record.get("protection_level") in {
        ProtectionLevel.LOCKED.value,
        ProtectionLevel.PROTECTED.value,
    }


def parse_time(value: Any) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def decision(left: dict[str, Any], right: dict[str, Any], winner: Optional[dict[str, Any]], side: str, reasons: list[str]) -> dict[str, Any]:
    loser = None
    if winner:
        loser = right if winner.get("id") == left.get("id") else left
    return {
        "winner_side": side,
        "winner_id": winner.get("id") if winner else None,
        "loser_id": loser.get("id") if loser else None,
        "manual_review": side in {"manual_review", "tie"},
        "reasons": reasons,
        "rule_order": [
            "user_confirmed/locked protection",
            "event_time",
            "source_type",
            "confidence",
            "ingest_time",
        ],
    }
