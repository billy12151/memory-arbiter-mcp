"""Persistence and CAS state transitions for structured conflict judgments.

This store records LLM, policy, and human judgments.  It delegates connection
and transaction ownership to MemoryDB so every multi-statement transition keeps
the same SQLite safety guarantees as the original implementation.
"""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, Optional


def is_protected_memory(memory: dict[str, Any]) -> bool:
    return memory.get("protection_level") == "locked" or memory.get("source_type") == "user_confirmed"
from .models import utc_now_iso

if TYPE_CHECKING:
    from .db import MemoryDB


class ConflictJudgmentStore:
    """Judgment persistence backed by MemoryDB's transaction factory."""

    RESOLUTION_KINDS = {
        "partial_update", "merge", "contextual_keep_both",
        "near_duplicate", "full_replacement", "not_a_conflict",
    }
    CONFLICT_SCOPES = {"field", "section", "record", "whole_memory", "unknown"}
    PARTIAL_KINDS = {"partial_update", "merge"}
    SUPERSEDE_KINDS = {"near_duplicate", "full_replacement"}

    def __init__(self, db: "MemoryDB"):
        self._db = db

    @classmethod
    def resolution_action(cls, resolution_kind: Optional[str]) -> str:
        """Map a resolution_kind to the machine-readable resolution action.

        Single source of truth — tools/console call this instead of keeping
        their own copy, so a new resolution_kind only needs updating here.
        """
        if resolution_kind in cls.PARTIAL_KINDS:
            return "update_or_merge"
        if resolution_kind == "contextual_keep_both":
            return "use_contextual_guidance"
        if resolution_kind in cls.SUPERSEDE_KINDS:
            return "supersede_old_memory"
        if resolution_kind == "not_a_conflict":
            return "none"
        return "unknown"

    @classmethod
    def is_supersede_candidate(cls, resolution_kind: Optional[str]) -> bool:
        """True only for near_duplicate/full_replacement (suggestion-only)."""
        return resolution_kind in cls.SUPERSEDE_KINDS

    @classmethod
    def _validate_resolution(
        cls,
        *,
        resolution_kind: Optional[str],
        conflict_scope: Optional[str],
        recommended_use: str,
        suggested_winner: Optional[int],
        conflict_id: int,
    ) -> Optional[dict[str, Any]]:
        if resolution_kind is not None and resolution_kind not in cls.RESOLUTION_KINDS:
            return {
                "outcome": "invalid_resolution_kind",
                "conflict_id": int(conflict_id),
                "error": "invalid resolution_kind",
            }
        if conflict_scope is not None and conflict_scope not in cls.CONFLICT_SCOPES:
            return {
                "outcome": "invalid_conflict_scope",
                "conflict_id": int(conflict_id),
                "error": "invalid conflict_scope",
            }
        if conflict_scope in {"field", "section"} and resolution_kind in cls.SUPERSEDE_KINDS:
            return {
                "outcome": "invalid_resolution_scope",
                "conflict_id": int(conflict_id),
                "error": "field/section conflicts cannot be near_duplicate/full_replacement",
            }
        if resolution_kind in cls.PARTIAL_KINDS and recommended_use not in {"merge", "contextual", "ask_user"}:
            return {
                "outcome": "invalid_recommendation",
                "conflict_id": int(conflict_id),
                "error": "partial_update/merge requires recommended_use=merge|contextual|ask_user",
            }
        if resolution_kind == "contextual_keep_both" and (
            recommended_use != "contextual" or suggested_winner is not None
        ):
            return {
                "outcome": "invalid_recommendation",
                "conflict_id": int(conflict_id),
                "error": "contextual_keep_both requires recommended_use=contextual and no winner",
            }
        if resolution_kind == "not_a_conflict" and (
            recommended_use != "none" or suggested_winner is not None
        ):
            return {
                "outcome": "invalid_recommendation",
                "conflict_id": int(conflict_id),
                "error": "not_a_conflict requires recommended_use=none and no winner",
            }
        if resolution_kind in cls.SUPERSEDE_KINDS:
            if recommended_use not in {"left", "right"} or suggested_winner is None:
                return {
                    "outcome": "invalid_recommendation",
                    "conflict_id": int(conflict_id),
                    "error": "near_duplicate/full_replacement requires left/right winner",
                }
            if conflict_scope not in {"record", "whole_memory"}:
                return {
                    "outcome": "invalid_resolution_scope",
                    "conflict_id": int(conflict_id),
                    "error": "near_duplicate/full_replacement requires record or whole_memory scope",
                }
        return None

    def build_conflict_judgment_request(
        self, conflict_id: int,
    ) -> Optional[dict[str, Any]]:
        requests = self.build_conflict_judgment_requests([int(conflict_id)])
        return requests.get(int(conflict_id))

    def build_conflict_judgment_requests(
        self, conflict_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        """Build multiple host-LLM requests on one consistent read connection."""
        db = self._db
        ids = list(dict.fromkeys(int(conflict_id) for conflict_id in conflict_ids))
        if not db._db_available or not ids:
            return {}
        try:
            with db.connection() as conn:
                requests: dict[int, dict[str, Any]] = {}
                for conflict_id in ids:
                    request = self._build_request_on_connection(conn, conflict_id)
                    if request is not None:
                        requests[conflict_id] = request
                return requests
        except sqlite3.Error:
            return {}

    def _build_request_on_connection(
        self, conn: sqlite3.Connection, conflict_id: int,
    ) -> Optional[dict[str, Any]]:
        db = self._db
        row = conn.execute(
            "SELECT * FROM conflicts WHERE id=? AND status='open'",
            (int(conflict_id),),
        ).fetchone()
        if not row:
            return None
        conflict = dict(row)
        left = db._fetch_memory(conn, int(conflict["left_id"]))
        right = db._fetch_memory(conn, int(conflict["right_id"]))
        if not left or not right:
            return None
        pins = (
            conflict.get("left_version"), conflict.get("right_version"),
            conflict.get("left_claim_revision"),
            conflict.get("right_claim_revision"),
        )
        if any(value is None for value in pins):
            return None
        if not self._snapshot_matches(
            conflict, left, right,
            int(conflict["left_version"]), int(conflict["right_version"]),
            int(conflict["left_claim_revision"]),
            int(conflict["right_claim_revision"]),
        ):
            return None
        if (
            left.get("claims_indexed_revision") != left.get("claim_revision")
            or right.get("claims_indexed_revision") != right.get("claim_revision")
        ):
            return None
        claim_rows = conn.execute(
            """
            SELECT l.entity, l.attribute, l.scope,
                   l.value AS left_value, l.raw_value AS left_raw_value,
                   l.evidence AS left_evidence, l.start_offset AS left_start_offset,
                   l.end_offset AS left_end_offset,
                   r.value AS right_value, r.raw_value AS right_raw_value,
                   r.evidence AS right_evidence, r.start_offset AS right_start_offset,
                   r.end_offset AS right_end_offset, l.extractor_rule
            FROM memory_claims l JOIN memory_claims r
              ON r.entity=l.entity AND r.attribute=l.attribute
            WHERE l.memory_id=? AND r.memory_id=?
              AND l.claim_revision=? AND r.claim_revision=?
              AND l.value<>r.value
              AND NOT (l.scope<>'' AND r.scope<>'' AND l.scope<>r.scope)
            ORDER BY l.attribute
            """,
            (
                int(conflict["left_id"]), int(conflict["right_id"]),
                int(left.get("claim_revision") or 1),
                int(right.get("claim_revision") or 1),
            ),
        ).fetchall()
        claims = [dict(claim_row) for claim_row in claim_rows]
        if not claims:
            return None

        def memory_evidence(memory: dict[str, Any]) -> dict[str, Any]:
            content = str(memory.get("content") or "")
            return {
                "id": int(memory["id"]),
                "version": int(memory.get("version") or 1),
                "claim_revision": int(memory.get("claim_revision") or 1),
                "subject": memory.get("subject"),
                "source_type": memory.get("source_type"),
                "protection_level": memory.get("protection_level"),
                "confidence": memory.get("confidence"),
                "event_time": memory.get("event_time"),
                "ingest_time": memory.get("ingest_time"),
                "source_ref": memory.get("source_ref"),
                "content": content if len(content) <= 2000 else None,
                "content_truncated": len(content) > 2000,
            }

        judge_data = {
            "conflict_id": int(conflict["id"]),
            "expected_left_version": int(conflict["left_version"]),
            "expected_right_version": int(conflict["right_version"]),
            "expected_left_claim_revision": int(conflict["left_claim_revision"]),
            "expected_right_claim_revision": int(conflict["right_claim_revision"]),
            "verdict": None,
            "recommended_use": None,
            "suggested_winner": None,
            "confidence_hint": None,
            "affects_current_output": None,
            "usage_context": None,
            "reason": None,
        }
        return {
            "conflict_id": int(conflict["id"]),
            "verification_status": conflict.get("judgment_status") or "pending_llm",
            "left": memory_evidence(left),
            "right": memory_evidence(right),
            "claims": claims,
            "allowed_verdicts": [
                "contradiction", "evolution", "compatible", "uncertain",
            ],
            "allowed_recommendations": [
                "left", "right", "contextual", "merge", "ask_user", "none",
            ],
            "allowed_resolution_kinds": sorted(self.RESOLUTION_KINDS),
            "allowed_conflict_scopes": sorted(self.CONFLICT_SCOPES),
            "resolution_guidance": (
                "LLM/host agent must classify resolution_kind/conflict_scope. "
                "Arbiter validates consistency but never auto-edits or supersedes."
            ),
            "required_tool": "memory(action='judge')",
            "judge_call": {"action": "judge", "data": judge_data},
        }

    @staticmethod
    def _snapshot_matches(
        conflict: dict[str, Any],
        left: dict[str, Any],
        right: dict[str, Any],
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
    ) -> bool:
        return (
            int(left.get("version") or 1) == int(expected_left_version)
            and int(right.get("version") or 1) == int(expected_right_version)
            and int(left.get("claim_revision") or 1) == int(expected_left_claim_revision)
            and int(right.get("claim_revision") or 1) == int(expected_right_claim_revision)
            and conflict.get("left_version") == int(expected_left_version)
            and conflict.get("right_version") == int(expected_right_version)
            and conflict.get("left_claim_revision") == int(expected_left_claim_revision)
            and conflict.get("right_claim_revision") == int(expected_right_claim_revision)
        )

    @staticmethod
    def _insert_judgment(
        conn: sqlite3.Connection,
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
        left_claim_revision: int,
        right_claim_revision: int,
        supersedes_judgment_id: Optional[int],
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
    ) -> int:
        cur = conn.execute(
            """
            INSERT INTO conflict_judgments(
              conflict_id, verdict, recommended_use, suggested_winner,
              confidence_hint, reason, judge_type, judge_ref,
              left_version, right_version, left_claim_revision,
              right_claim_revision, supersedes_judgment_id,
              resolution_kind, conflict_scope, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(conflict_id), verdict, recommended_use, suggested_winner,
                confidence_hint, reason, judge_type, judge_ref,
                int(left_version), int(right_version), int(left_claim_revision),
                int(right_claim_revision), supersedes_judgment_id,
                resolution_kind, conflict_scope, utc_now_iso(),
            ),
        )
        return int(cur.lastrowid)

    def submit_conflict_judgment(
        self,
        conflict_id: int,
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
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
        verdicts = {"contradiction", "evolution", "compatible", "uncertain"}
        recommendations = {"left", "right", "contextual", "merge", "ask_user", "none"}
        contexts = {
            "answer", "code", "config", "memory_write",
            "external_action", "unrelated", "unknown",
        }
        if (
            verdict not in verdicts
            or recommended_use not in recommendations
            or usage_context not in contexts
        ):
            return {"outcome": "invalid_input", "conflict_id": int(conflict_id)}
        if confidence_hint not in {None, "low", "medium", "high"}:
            return {
                "outcome": "invalid_input", "conflict_id": int(conflict_id),
                "error": "invalid confidence_hint",
            }
        if not str(reason or "").strip():
            return {
                "outcome": "invalid_input", "conflict_id": int(conflict_id),
                "error": "reason is required",
            }
        resolution_error = self._validate_resolution(
            resolution_kind=resolution_kind,
            conflict_scope=conflict_scope,
            recommended_use=recommended_use,
            suggested_winner=suggested_winner,
            conflict_id=int(conflict_id),
        )
        if resolution_error is not None:
            return resolution_error
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable", "conflict_id": int(conflict_id)}
        try:
            with db.write_transaction() as conn:
                row = conn.execute(
                    "SELECT * FROM conflicts WHERE id=? AND status='open'",
                    (int(conflict_id),),
                ).fetchone()
                if not row:
                    return {"outcome": "not_open", "conflict_id": int(conflict_id)}
                conflict = dict(row)
                if any(
                    conflict.get(key) is None for key in (
                        "left_version", "right_version",
                        "left_claim_revision", "right_claim_revision",
                    )
                ):
                    return {
                        "outcome": "invalid_structured_snapshot",
                        "conflict_id": int(conflict_id),
                    }
                left = db._fetch_memory(conn, int(conflict["left_id"]))
                right = db._fetch_memory(conn, int(conflict["right_id"]))
                if not left or not right:
                    return {"outcome": "stale_snapshot", "conflict_id": int(conflict_id)}
                if not self._snapshot_matches(
                    conflict, left, right,
                    expected_left_version, expected_right_version,
                    expected_left_claim_revision, expected_right_claim_revision,
                ):
                    conn.execute(
                        "UPDATE conflicts SET judgment_status='pending_llm', "
                        "active_judgment_id=NULL, suggested_winner=NULL, "
                        "winner_id=NULL, confidence_hint=NULL, source='structured_claim', "
                        "reason='Structured claim snapshot changed; pending new judgment', "
                        "resolution_kind=NULL, conflict_scope=NULL WHERE id=?",
                        (int(conflict_id),),
                    )
                    return {"outcome": "stale_snapshot", "conflict_id": int(conflict_id)}
                if suggested_winner is not None and int(suggested_winner) not in {
                    int(conflict["left_id"]), int(conflict["right_id"])
                }:
                    return {"outcome": "invalid_winner", "conflict_id": int(conflict_id)}
                expected_winner = (
                    int(conflict["left_id"])
                    if recommended_use == "left"
                    else int(conflict["right_id"])
                    if recommended_use == "right"
                    else None
                )
                if expected_winner is not None and suggested_winner != expected_winner:
                    return {
                        "outcome": "invalid_recommendation",
                        "conflict_id": int(conflict_id),
                        "error": (
                            f"recommended_use={recommended_use} "
                            f"requires suggested_winner={expected_winner}"
                        ),
                    }
                if expected_winner is None and suggested_winner is not None:
                    return {
                        "outcome": "invalid_recommendation",
                        "conflict_id": int(conflict_id),
                        "error": (
                            "contextual/merge/ask_user/none must not carry "
                            "a single winner"
                        ),
                    }

                if conflict.get("active_judgment_id") is not None:
                    active = conn.execute(
                        "SELECT judge_type FROM conflict_judgments WHERE id=?",
                        (int(conflict["active_judgment_id"]),),
                    ).fetchone()
                    if active and active["judge_type"] in {"human", "policy"}:
                        return {
                            "outcome": "higher_priority_judgment_active",
                            "conflict_id": int(conflict_id),
                            "judge_type": active["judge_type"],
                        }

                prior_active = conflict.get("active_judgment_id")
                llm_id = self._insert_judgment(
                    conn, int(conflict_id), verdict, recommended_use,
                    int(suggested_winner) if suggested_winner is not None else None,
                    confidence_hint, str(reason).strip(), "llm", judge_ref,
                    expected_left_version, expected_right_version,
                    expected_left_claim_revision, expected_right_claim_revision,
                    int(prior_active) if prior_active is not None else None,
                    resolution_kind=resolution_kind,
                    conflict_scope=conflict_scope,
                )
                active_id = llm_id
                effective_use = recommended_use
                effective_winner = (
                    int(suggested_winner) if suggested_winner is not None else None
                )
                effective_reason = str(reason).strip()
                protected = [
                    is_protected_memory(left), is_protected_memory(right),
                ]
                protected_count = sum(1 for value in protected if value)
                high_impact = bool(affects_current_output) and usage_context in {
                    "code", "config", "memory_write", "external_action", "unknown",
                }
                pending_user = (
                    verdict == "uncertain"
                    or recommended_use in {"merge", "ask_user"}
                    or resolution_kind in self.PARTIAL_KINDS
                    or resolution_kind in self.SUPERSEDE_KINDS
                    or confidence_hint in {None, "low"}
                    or (verdict == "contradiction" and recommended_use == "none")
                    or high_impact
                    or (protected_count >= 2 and verdict != "compatible")
                )
                source = "llm_informed"
                if (
                    verdict == "contradiction"
                    and protected_count == 1
                    and not high_impact
                    and resolution_kind not in self.PARTIAL_KINDS
                    and recommended_use not in {"merge", "ask_user"}
                ):
                    effective_winner = int(
                        left["id"] if protected[0] else right["id"]
                    )
                    effective_use = (
                        "left"
                        if effective_winner == int(left["id"])
                        else "right"
                    )
                    effective_reason = (
                        "Policy prefers the single locked/user_confirmed side "
                        "for low-risk use."
                    )
                    active_id = self._insert_judgment(
                        conn, int(conflict_id), verdict, effective_use,
                        effective_winner, confidence_hint, effective_reason,
                        "policy", "protected-memory-policy-v1",
                        expected_left_version, expected_right_version,
                        expected_left_claim_revision,
                        expected_right_claim_revision, llm_id,
                        resolution_kind=resolution_kind,
                        conflict_scope=conflict_scope,
                    )
                    source = "policy_informed"

                if verdict in {"evolution", "compatible"} and not pending_user:
                    status = "resolved"
                    judgment_status = "llm_assessed"
                    resolved_at = utc_now_iso()
                else:
                    status = "open"
                    judgment_status = (
                        "pending_user" if pending_user else "llm_assessed"
                    )
                    resolved_at = None
                conn.execute(
                    """
                    UPDATE conflicts SET status=?, resolved_at=?, judgment_status=?,
                      active_judgment_id=?, suggested_winner=?, winner_id=?,
                      confidence_hint=?, reason=?, source=?, refreshed_at=?,
                      resolution_kind=?, conflict_scope=?
                    WHERE id=?
                    """,
                    (
                        status, resolved_at, judgment_status, active_id,
                        effective_winner, effective_winner, confidence_hint,
                        effective_reason, source, utc_now_iso(),
                        resolution_kind, conflict_scope, int(conflict_id),
                    ),
                )
                disclosure_required = (
                    bool(affects_current_output)
                    and usage_context == "answer"
                    and judgment_status == "llm_assessed"
                )
                return {
                    "outcome": "judged",
                    "conflict_id": int(conflict_id),
                    "judgment_id": active_id,
                    "judgment_status": judgment_status,
                    "conflict_status": status,
                    "recommended_use": effective_use,
                    "suggested_winner": effective_winner,
                    "resolution_kind": resolution_kind,
                    "conflict_scope": conflict_scope,
                    "recommended_resolution_action": self.resolution_action(resolution_kind),
                    "supersede_candidate": self.is_supersede_candidate(resolution_kind),
                    "disclosure_required": disclosure_required,
                    "user_action_required": pending_user,
                    "disclosure": (
                        f"Used memory #{effective_winner} for this answer: "
                        f"{effective_reason}"
                        if disclosure_required and effective_winner is not None
                        else (
                            f"Used {effective_use} conflict guidance for this answer: "
                            f"{effective_reason}"
                            if disclosure_required else None
                        )
                    ),
                }
        except sqlite3.Error as exc:
            return {
                "outcome": "error", "conflict_id": int(conflict_id),
                "error": str(exc),
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
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
        judge_ref: Optional[str] = None,
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
    ) -> dict[str, Any]:
        verdicts = {"contradiction", "evolution", "compatible", "uncertain"}
        recommendations = {"left", "right", "contextual", "merge", "ask_user", "none"}
        if (
            verdict not in verdicts
            or recommended_use not in recommendations
            or not str(reason or "").strip()
        ):
            return {"outcome": "invalid_input", "conflict_id": int(conflict_id)}
        resolution_error = self._validate_resolution(
            resolution_kind=resolution_kind,
            conflict_scope=conflict_scope,
            recommended_use=recommended_use,
            suggested_winner=suggested_winner,
            conflict_id=int(conflict_id),
        )
        if resolution_error is not None:
            return resolution_error
        db = self._db
        if not db._db_available or not db.state.sqlite_writable:
            return {"outcome": "unavailable", "conflict_id": int(conflict_id)}
        try:
            with db.write_transaction() as conn:
                row = conn.execute(
                    "SELECT * FROM conflicts WHERE id=?", (int(conflict_id),)
                ).fetchone()
                if not row:
                    return {"outcome": "not_found", "conflict_id": int(conflict_id)}
                conflict = dict(row)
                if any(
                    conflict.get(key) is None for key in (
                        "left_version", "right_version",
                        "left_claim_revision", "right_claim_revision",
                    )
                ):
                    return {
                        "outcome": "invalid_structured_snapshot",
                        "conflict_id": int(conflict_id),
                    }
                if int(conflict.get("active_judgment_id") or 0) != int(
                    expected_judgment_id
                ):
                    return {
                        "outcome": "stale_judgment",
                        "conflict_id": int(conflict_id),
                    }
                left = db._fetch_memory(conn, int(conflict["left_id"]))
                right = db._fetch_memory(conn, int(conflict["right_id"]))
                if not left or not right or not self._snapshot_matches(
                    conflict, left, right,
                    expected_left_version, expected_right_version,
                    expected_left_claim_revision, expected_right_claim_revision,
                ):
                    return {
                        "outcome": "stale_snapshot",
                        "conflict_id": int(conflict_id),
                    }
                if suggested_winner is not None and int(suggested_winner) not in {
                    int(conflict["left_id"]), int(conflict["right_id"])
                }:
                    return {
                        "outcome": "invalid_winner",
                        "conflict_id": int(conflict_id),
                    }
                expected_winner = (
                    int(conflict["left_id"])
                    if recommended_use == "left"
                    else int(conflict["right_id"])
                    if recommended_use == "right"
                    else None
                )
                if expected_winner is not None and suggested_winner != expected_winner:
                    return {
                        "outcome": "invalid_recommendation",
                        "conflict_id": int(conflict_id),
                    }
                if expected_winner is None and suggested_winner is not None:
                    return {
                        "outcome": "invalid_recommendation",
                        "conflict_id": int(conflict_id),
                    }
                judgment_id = self._insert_judgment(
                    conn, int(conflict_id), verdict, recommended_use,
                    int(suggested_winner) if suggested_winner is not None else None,
                    None, str(reason).strip(), "human",
                    judge_ref or "authorized-human",
                    expected_left_version, expected_right_version,
                    expected_left_claim_revision, expected_right_claim_revision,
                    int(expected_judgment_id),
                    resolution_kind=resolution_kind,
                    conflict_scope=conflict_scope,
                )
                resolved = verdict in {"evolution", "compatible"}
                conn.execute(
                    """
                    UPDATE conflicts SET status=?, resolved_at=?,
                      judgment_status='human_confirmed',
                      active_judgment_id=?, suggested_winner=?, winner_id=?,
                      reason=?, source='human_confirmed', refreshed_at=?,
                      resolution_kind=?, conflict_scope=? WHERE id=?
                    """,
                    (
                        "resolved" if resolved else "open",
                        utc_now_iso() if resolved else None,
                        judgment_id, suggested_winner, suggested_winner,
                        str(reason).strip(), utc_now_iso(),
                        resolution_kind, conflict_scope, int(conflict_id),
                    ),
                )
                return {
                    "outcome": "corrected",
                    "conflict_id": int(conflict_id),
                    "judgment_id": judgment_id,
                    "judgment_status": "human_confirmed",
                    "conflict_status": "resolved" if resolved else "open",
                    "resolution_kind": resolution_kind,
                    "conflict_scope": conflict_scope,
                    "recommended_resolution_action": self.resolution_action(resolution_kind),
                    "supersede_candidate": self.is_supersede_candidate(resolution_kind),
                }
        except sqlite3.Error as exc:
            return {
                "outcome": "error", "conflict_id": int(conflict_id),
                "error": str(exc),
            }

    def list_conflict_judgments(
        self, conflict_id: int,
    ) -> list[dict[str, Any]]:
        db = self._db
        if not db._db_available:
            return []
        try:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM conflict_judgments "
                    "WHERE conflict_id=? ORDER BY id",
                    (int(conflict_id),),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            return []
