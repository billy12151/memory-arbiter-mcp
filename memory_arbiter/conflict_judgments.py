"""Version-pinned, append-only conflict judgments."""
from __future__ import annotations

import sqlite3
from typing import Any, Optional, TYPE_CHECKING

from .models import utc_now_iso

if TYPE_CHECKING:
    from .db import MemoryDB


class ConflictJudgmentStore:
    RESOLUTION_KINDS = {
        "partial_update", "merge", "contextual_keep_both",
        "near_duplicate", "full_replacement", "not_a_conflict",
    }
    CONFLICT_SCOPES = {"field", "section", "record", "whole_memory", "unknown"}
    PARTIAL_KINDS = {"partial_update", "merge"}
    SUPERSEDE_KINDS = {"near_duplicate", "full_replacement"}
    VERDICTS = {"contradiction", "evolution", "compatible", "uncertain"}
    RECOMMENDATIONS = {"left", "right", "contextual", "merge", "ask_user", "none"}
    # usage_context/confidence_hint drive the pending_user escalation gates;
    # free-text values silently fall out of every set membership and weaken
    # the gate, so they are validated like the other enums.
    USAGE_CONTEXTS = {
        "answer", "code", "config", "memory_write", "external_action",
        "unrelated", "unknown",
    }
    CONFIDENCE_HINTS = {"low", "medium", "high"}

    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    @classmethod
    def resolution_action(cls, kind: Optional[str]) -> str:
        if kind in cls.PARTIAL_KINDS:
            return "update_or_merge"
        if kind == "contextual_keep_both":
            return "use_contextual_guidance"
        if kind in cls.SUPERSEDE_KINDS:
            return "supersede_old_memory"
        if kind == "not_a_conflict":
            return "none"
        return "unknown"

    @classmethod
    def is_supersede_candidate(cls, kind: Optional[str]) -> bool:
        return kind in cls.SUPERSEDE_KINDS

    @classmethod
    def _validate_input(
        cls, conflict_id: int, verdict: str, recommended_use: str,
        suggested_winner: Optional[int], reason: str,
        resolution_kind: Optional[str], conflict_scope: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if verdict not in cls.VERDICTS or recommended_use not in cls.RECOMMENDATIONS or not reason.strip():
            return {"outcome": "invalid_input", "conflict_id": conflict_id}
        if resolution_kind is not None and resolution_kind not in cls.RESOLUTION_KINDS:
            return {"outcome": "invalid_resolution_kind", "conflict_id": conflict_id}
        if conflict_scope is not None and conflict_scope not in cls.CONFLICT_SCOPES:
            return {"outcome": "invalid_conflict_scope", "conflict_id": conflict_id}
        if conflict_scope in {"field", "section"} and resolution_kind in cls.SUPERSEDE_KINDS:
            return {"outcome": "invalid_resolution_scope", "conflict_id": conflict_id}
        if resolution_kind in cls.PARTIAL_KINDS and recommended_use not in {"merge", "contextual", "ask_user"}:
            return {"outcome": "invalid_recommendation", "conflict_id": conflict_id}
        if resolution_kind == "contextual_keep_both" and (
            recommended_use != "contextual" or suggested_winner is not None
        ):
            return {"outcome": "invalid_recommendation", "conflict_id": conflict_id}
        if resolution_kind == "not_a_conflict" and (
            recommended_use != "none" or suggested_winner is not None
        ):
            return {"outcome": "invalid_recommendation", "conflict_id": conflict_id}
        if resolution_kind in cls.SUPERSEDE_KINDS and (
            recommended_use not in {"left", "right"}
            or suggested_winner is None
            or conflict_scope not in {"record", "whole_memory"}
        ):
            return {"outcome": "invalid_resolution_scope", "conflict_id": conflict_id}
        return None

    @staticmethod
    def _snapshot_matches(
        conflict: dict[str, Any], left: dict[str, Any], right: dict[str, Any],
        left_version: int, right_version: int,
    ) -> bool:
        return (
            int(left.get("version") or 1) == left_version
            and int(right.get("version") or 1) == right_version
            and conflict.get("left_version") == left_version
            and conflict.get("right_version") == right_version
        )

    @staticmethod
    def _expected_winner(conflict: dict[str, Any], recommended_use: str) -> Optional[int]:
        if recommended_use == "left":
            return int(conflict["left_id"])
        if recommended_use == "right":
            return int(conflict["right_id"])
        return None

    @staticmethod
    def _memory_evidence(memory: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "id", "version", "subject", "content", "source_type",
            "protection_level", "confidence", "event_time", "source_ref",
        )
        return {key: memory.get(key) for key in keys}

    def build_conflict_judgment_request(self, conflict_id: int) -> Optional[dict[str, Any]]:
        return self.build_conflict_judgment_requests([conflict_id]).get(int(conflict_id))

    def build_conflict_judgment_requests(
        self, conflict_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        requests: dict[int, dict[str, Any]] = {}
        with self._db.connection() as conn:
            for conflict_id in dict.fromkeys(map(int, conflict_ids)):
                row = conn.execute(
                    "SELECT * FROM conflicts WHERE id=? AND status='open'",
                    (conflict_id,),
                ).fetchone()
                if row is None:
                    continue
                conflict = dict(row)
                left = self._db._fetch_memory(conn, int(conflict["left_id"]))
                right = self._db._fetch_memory(conn, int(conflict["right_id"]))
                if not left or not right:
                    continue
                if conflict.get("left_version") is None or conflict.get("right_version") is None:
                    continue
                left_version = int(conflict["left_version"])
                right_version = int(conflict["right_version"])
                if not self._snapshot_matches(
                    conflict, left, right, left_version, right_version,
                ):
                    continue
                judge_data = {
                    "conflict_id": conflict_id,
                    "expected_left_version": left_version,
                    "expected_right_version": right_version,
                    "verdict": None,
                    "recommended_use": None,
                    "suggested_winner": None,
                    "confidence_hint": None,
                    "affects_current_output": None,
                    "usage_context": None,
                    "reason": None,
                }
                requests[conflict_id] = {
                    "conflict_id": conflict_id,
                    "verification_status": conflict.get("judgment_status") or "pending_agent",
                    "left": self._memory_evidence(left),
                    "right": self._memory_evidence(right),
                    "required_tool": "memory(action='judge')",
                    "judge_call": {"action": "judge", "data": judge_data},
                }
        return requests

    @staticmethod
    def _insert_judgment(
        conn: sqlite3.Connection,
        *,
        conflict_id: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        confidence_hint: Optional[str],
        reason: str,
        judge_type: str,
        judge_ref: Optional[str],
        left_version: int,
        right_version: int,
        supersedes_judgment_id: Optional[int],
        resolution_kind: Optional[str],
        conflict_scope: Optional[str],
    ) -> int:
        cur = conn.execute(
            """INSERT INTO conflict_judgments(
                 conflict_id,verdict,recommended_use,suggested_winner,
                 confidence_hint,reason,judge_type,judge_ref,left_version,
                 right_version,supersedes_judgment_id,resolution_kind,
                 conflict_scope,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                conflict_id, verdict, recommended_use, suggested_winner,
                confidence_hint, reason, judge_type, judge_ref, left_version,
                right_version, supersedes_judgment_id, resolution_kind,
                conflict_scope, utc_now_iso(),
            ),
        )
        if cur.lastrowid is None:
            raise sqlite3.Error("judgment insert did not return an id")
        return int(cur.lastrowid)

    def _read_current_snapshot(
        self, conn: sqlite3.Connection, conflict_id: int, *, require_open: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
        sql = "SELECT * FROM conflicts WHERE id=?"
        if require_open:
            sql += " AND status='open'"
        row = conn.execute(sql, (conflict_id,)).fetchone()
        if row is None:
            return None
        conflict = dict(row)
        left = self._db._fetch_memory(conn, int(conflict["left_id"]))
        right = self._db._fetch_memory(conn, int(conflict["right_id"]))
        if not left or not right:
            return None
        return conflict, left, right

    def submit_conflict_judgment(
        self,
        conflict_id: int,
        expected_left_version: int,
        expected_right_version: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        confidence_hint: Optional[str],
        reason: str,
        affects_current_output: bool,
        usage_context: str,
        judge_ref: Optional[str] = None,
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
    ) -> dict[str, Any]:
        error = self._validate_input(
            conflict_id, verdict, recommended_use, suggested_winner, reason,
            resolution_kind, conflict_scope,
        )
        if error:
            return error
        if usage_context not in self.USAGE_CONTEXTS:
            return {"outcome": "invalid_input", "conflict_id": conflict_id, "error": "usage_context must be one of the declared values (see judge_constraints)"}
        if confidence_hint is not None and confidence_hint not in self.CONFIDENCE_HINTS:
            return {"outcome": "invalid_input", "conflict_id": conflict_id, "error": "confidence_hint must be low, medium, high, or null"}
        with self._db.write_transaction() as conn:
            snapshot = self._read_current_snapshot(conn, conflict_id, require_open=True)
            if snapshot is None:
                return {"outcome": "not_open", "conflict_id": conflict_id}
            conflict, left, right = snapshot
            if not self._snapshot_matches(
                conflict, left, right,
                expected_left_version, expected_right_version,
            ):
                return {"outcome": "stale_snapshot", "conflict_id": conflict_id}
            if suggested_winner != self._expected_winner(conflict, recommended_use):
                return {"outcome": "invalid_recommendation", "conflict_id": conflict_id}

            judgment_id = self._insert_judgment(
                conn,
                conflict_id=conflict_id,
                verdict=verdict,
                recommended_use=recommended_use,
                suggested_winner=suggested_winner,
                confidence_hint=confidence_hint,
                reason=reason,
                judge_type="llm",
                judge_ref=judge_ref,
                left_version=expected_left_version,
                right_version=expected_right_version,
                supersedes_judgment_id=conflict.get("active_judgment_id"),
                resolution_kind=resolution_kind,
                conflict_scope=conflict_scope,
            )
            high_impact = affects_current_output and usage_context in {
                "code", "config", "memory_write", "external_action", "unknown",
            }
            pending_user = (
                verdict == "uncertain"
                or recommended_use in {"merge", "ask_user"}
                or confidence_hint in {None, "low"}
                or high_impact
            )
            status = "open" if pending_user or verdict == "contradiction" else "resolved"
            judgment_status = "pending_user" if pending_user else "llm_assessed"
            conn.execute(
                """UPDATE conflicts SET
                     status=?,resolved_at=?,judgment_status=?,active_judgment_id=?,
                     suggested_winner=?,winner_id=?,confidence_hint=?,reason=?,
                     source='llm_informed',refreshed_at=?,resolution_kind=?,conflict_scope=?
                   WHERE id=?""",
                (
                    status, None if status == "open" else utc_now_iso(),
                    judgment_status, judgment_id, suggested_winner,
                    suggested_winner, confidence_hint, reason, utc_now_iso(),
                    resolution_kind, conflict_scope, conflict_id,
                ),
            )
            return {
                "outcome": "judged",
                "conflict_id": conflict_id,
                "judgment_id": judgment_id,
                "conflict_status": status,
                "user_action_required": pending_user,
                "recommended_resolution_action": self.resolution_action(resolution_kind),
            }

    def correct_conflict_judgment(
        self,
        conflict_id: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        reason: str,
        expected_judgment_id: int,
        expected_left_version: int,
        expected_right_version: int,
        judge_ref: Optional[str] = None,
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
    ) -> dict[str, Any]:
        error = self._validate_input(
            conflict_id, verdict, recommended_use, suggested_winner, reason,
            resolution_kind, conflict_scope,
        )
        if error:
            return error
        with self._db.write_transaction() as conn:
            snapshot = self._read_current_snapshot(conn, conflict_id, require_open=False)
            if snapshot is None:
                return {"outcome": "not_found", "conflict_id": conflict_id}
            conflict, left, right = snapshot
            if int(conflict.get("active_judgment_id") or 0) != expected_judgment_id:
                return {"outcome": "stale_judgment", "conflict_id": conflict_id}
            if not self._snapshot_matches(
                conflict, left, right,
                expected_left_version, expected_right_version,
            ):
                return {"outcome": "stale_snapshot", "conflict_id": conflict_id}
            if suggested_winner != self._expected_winner(conflict, recommended_use):
                return {"outcome": "invalid_recommendation", "conflict_id": conflict_id}

            judgment_id = self._insert_judgment(
                conn,
                conflict_id=conflict_id,
                verdict=verdict,
                recommended_use=recommended_use,
                suggested_winner=suggested_winner,
                confidence_hint=None,
                reason=reason,
                judge_type="human",
                judge_ref=judge_ref or "authorized-human",
                left_version=expected_left_version,
                right_version=expected_right_version,
                supersedes_judgment_id=expected_judgment_id,
                resolution_kind=resolution_kind,
                conflict_scope=conflict_scope,
            )
            resolved = verdict in {"evolution", "compatible"}
            conn.execute(
                """UPDATE conflicts SET
                     status=?,resolved_at=?,judgment_status='human_confirmed',
                     active_judgment_id=?,suggested_winner=?,winner_id=?,reason=?,
                     source='human_confirmed',refreshed_at=?,resolution_kind=?,conflict_scope=?
                   WHERE id=?""",
                (
                    "resolved" if resolved else "open",
                    utc_now_iso() if resolved else None,
                    judgment_id, suggested_winner, suggested_winner, reason,
                    utc_now_iso(), resolution_kind, conflict_scope, conflict_id,
                ),
            )
            return {
                "outcome": "corrected",
                "conflict_id": conflict_id,
                "judgment_id": judgment_id,
            }

    def list_conflict_judgments(self, conflict_id: int) -> list[dict[str, Any]]:
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM conflict_judgments WHERE conflict_id=? ORDER BY id",
                (conflict_id,),
            ).fetchall()
        return [dict(row) for row in rows]
