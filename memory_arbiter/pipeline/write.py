"""Memory write path for the local-text evidence architecture."""
from __future__ import annotations

import difflib
import re
from typing import Any, Protocol, TYPE_CHECKING

from ..embedder import ManagedEmbedder

from .. import workspace_rules
from ..constants import (
    WRITE_DUPLICATE_VEC_TOP_K,
    WRITE_SIMILAR_FALLBACK_SCAN_LIMIT,
    WRITE_SIMILAR_MAX_HINTS,
    WRITE_SIMILAR_SUBJECT_RATIO,
    WRITE_SIMILAR_TAG_JACCARD,
    is_default_workspace_term,
)
from ..models import MemoryRecord, MemoryStatus
from ..validation import validate_product_payload

if TYPE_CHECKING:
    from ..tools import MemoryTools


class _SubjectTagRecord(Protocol):
    """Structural requirement for ``_similar_active_notice``: subject + tags."""

    subject: str | None
    tags: list[str]


class WritePipeline:
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools: "MemoryTools" = tools
        self.db = tools.db
        self.settings = tools.settings

    def _allowed(self, *args: Any, **kwargs: Any) -> "tuple[bool, list[str]]":
        return self._tools._allowed(*args, **kwargs)

    def _post_commit(
        self, *args: Any, **kwargs: Any,
    ) -> "tuple[dict[str, Any], dict[str, Any]]":
        return self._tools._post_commit(*args, **kwargs)

    def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_active_embedder()

    def _suggest_workspace_candidate(self, *args: Any, **kwargs: Any) -> Any:
        return self._tools._suggest_workspace_candidate(*args, **kwargs)

    def current_agent_id(self) -> str | None:
        return self._tools.current_agent_id()

    _SIMILARITY_SPACE = re.compile(r"\s+")
    _DIGIT_RUN = re.compile(r"\d+")

    @classmethod
    def _normalized_subject(cls, subject: str) -> str:
        return cls._SIMILARITY_SPACE.sub(" ", str(subject or "").casefold().strip())

    @classmethod
    def _tag_jaccard(cls, left: list[str], right: list[str]) -> float:
        # Tags normalize with the same whitespace folding as subjects, so
        # "金营 项目" and "金营  项目" are the same tag.
        left_set = {
            cls._SIMILARITY_SPACE.sub(" ", str(tag).casefold().strip())
            for tag in left if str(tag).strip()
        }
        right_set = {
            cls._SIMILARITY_SPACE.sub(" ", str(tag).casefold().strip())
            for tag in right if str(tag).strip()
        }
        if not left_set and not right_set:
            # Neither side carries tags: the subject bar alone decides (a
            # shared empty vocabulary is trivially full overlap).
            return 1.0
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)

    @classmethod
    def _subject_tags_embed_text(cls, subject: Any, tags: Any) -> str:
        """Canonical text embedded into subject_tags_vec (owner spec: "subject
        + 排序后 tags"). Tags are sorted so tag ORDER never changes the
        vector; case/whitespace are left to the model like every other
        embedded body."""
        cleaned = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        body = f"{str(subject or '').strip()}\n{' '.join(sorted(cleaned))}".strip()
        return body

    def _duplicate_hint_candidate_rows(
        self, memory_id: int, record: _SubjectTagRecord, workspace_canonical: str,
    ) -> list[dict[str, Any]]:
        """Candidate recall for the write-time duplicate hint (0.15.3).

        Primary path: embed the new memory's "subject + sorted tags", publish
        it into subject_tags_vec, and KNN-recall the top-k same-workspace
        active rows. Fallback (no embedder / vec index unavailable / KNN
        error): the legacy full scan capped by WRITE_SIMILAR_FALLBACK_SCAN_
        LIMIT rows. Fail-open like the notice itself — any unexpected error
        degrades to the fallback scan.
        """
        subject = str(getattr(record, "subject", None) or "").strip()
        if not subject:
            return []
        embedder, _ = self._ensure_active_embedder()
        if embedder is not None and self.db.state.sqlite_vec_available:
            try:
                er = embedder.embed_text(
                    prefix="",
                    body=self._subject_tags_embed_text(
                        subject, getattr(record, "tags", None),
                    ),
                )
                if er is not None and er.embedding:
                    vector = [float(x) for x in er.embedding]
                    self.db.upsert_subject_tags_vector(memory_id, vector)
                    rows = self.db.subject_tags_knn(
                        vector,
                        k=WRITE_DUPLICATE_VEC_TOP_K,
                        exclude_memory_id=memory_id,
                        workspace_canonical=workspace_canonical,
                    )
                    if rows:
                        return rows
            except Exception:
                pass
        return self.db.active_subject_tag_rows(
            memory_id, workspace_canonical, limit=WRITE_SIMILAR_FALLBACK_SCAN_LIMIT,
        )

    def refresh_subject_tags_vector(self, memory_id: int) -> bool:
        """Re-embed one memory's subject+tags after an in-place edit.

        Keeps subject_tags_vec aligned with the row's CURRENT subject/tags
        (tags-only and content edits both reach here). Deletes the vector
        when the memory is no longer active. Best-effort: a failed refresh
        self-heals on the next process start via the startup backfill.
        """
        try:
            record = self.db.get_memory(int(memory_id))
            if record is None:
                return self.db.delete_subject_tags_vector(int(memory_id))
            if str(record.get("status") or "") != "active":
                return self.db.delete_subject_tags_vector(int(memory_id))
            embedder, _ = self._ensure_active_embedder()
            if embedder is None or not self.db.state.sqlite_vec_available:
                return False
            er = embedder.embed_text(
                prefix="",
                body=self._subject_tags_embed_text(
                    record.get("subject"), record.get("tags"),
                ),
            )
            if er is None or not er.embedding:
                return False
            return self.db.upsert_subject_tags_vector(
                int(memory_id), [float(x) for x in er.embedding],
            )
        except Exception:
            return False

    def _similar_active_notice(
        self, memory_id: int, record: _SubjectTagRecord, workspace_canonical: str | None,
    ) -> dict[str, Any] | None:
        """Write-time duplicate hint over subject/tags similarity (owner spec).

        Recall is vector-based since 0.15.3: the hint KNN-recalls the top-k
        same-workspace active rows over subject_tags_vec (fallback: capped
        legacy scan when no embedder/index is available) and then applies the
        deterministic model-free fine-ranking. Long documents rarely compare
        equal on content, but a forgotten near-duplicate keeps a
        near-identical subject and tag set. Fires only when BOTH bars clear.
        Subjects identical modulo digit runs (release/checklist series such
        as "0.15.1 发版清单" vs "0.15.2 发版清单") are suppressed outright;
        other deliberate series entries (tier-1 vs tier-2 plans sharing most
        tags) stay quiet unless the subjects are near-identical.
        Same-workspace active rows only (no cross-workspace leakage);
        fail-open on any error.
        """
        if not workspace_canonical:
            return None
        try:
            subject = self._normalized_subject(str(record.subject or ""))
            if not subject:
                return None
            rows = self._duplicate_hint_candidate_rows(
                int(memory_id), record, workspace_canonical,
            )
            if not rows:
                return None
            record_tags = list(record.tags or [])
            subject_series = self._DIGIT_RUN.sub("#", subject)
            scored: list[tuple[float, float, dict[str, Any]]] = []
            for row in rows:
                row_subject = self._normalized_subject(str(row["subject"] or ""))
                if not row_subject:
                    continue
                # Length gate: ratio() = 2*M/(len_a+len_b) with M bounded by
                # the shorter subject, so when even a full subsequence match
                # cannot clear the bar the SequenceMatcher run is skipped.
                shorter = min(len(subject), len(row_subject))
                if 2.0 * shorter < WRITE_SIMILAR_SUBJECT_RATIO * (len(subject) + len(row_subject)):
                    continue
                if row_subject != subject and self._DIGIT_RUN.sub("#", row_subject) == subject_series:
                    # Series entries: identical modulo digit runs but not
                    # exact duplicates — stay quiet.
                    continue
                ratio = difflib.SequenceMatcher(None, subject, row_subject).ratio()
                if ratio < WRITE_SIMILAR_SUBJECT_RATIO:
                    continue
                jaccard = self._tag_jaccard(record_tags, row["tags"])
                if jaccard < WRITE_SIMILAR_TAG_JACCARD:
                    continue
                scored.append((ratio, jaccard, row))
            if not scored:
                return None
            scored.sort(key=lambda item: (-item[0], -item[1]))
            matches = [
                {
                    "memory_id": row["id"],
                    "subject": row["subject"],
                    "event_time": row.get("event_time"),
                    "subject_similarity": round(ratio, 4),
                    "tag_jaccard": round(jaccard, 4),
                }
                for ratio, jaccard, row in scored[:WRITE_SIMILAR_MAX_HINTS]
            ]
            return {
                "type": "similar_active_memory",
                "severity": "info",
                "matches": matches,
                "agent_instruction": (
                    "This write closely matches existing active memories. Triage silently: "
                    "ignore it if this is a deliberate series entry or cross-reference. If it "
                    "duplicates an existing memory, prefer updating that memory instead of "
                    "keeping two active copies. If the fix needs retiring or merging (governance) "
                    "or you are unsure, ask the user."
                ),
            }
        except Exception:
            return None

    def memory_write(self, **payload: Any) -> dict[str, Any]:
        payload = dict(payload)
        validation = validate_product_payload("memory", "remember", payload)
        if validation.error is not None:
            error = dict(validation.error)
            if error.get("field") in {"content", "subject"} and str(error.get("reason") or "").startswith("is required"):
                error["error"] = f"{error['field']} is required"
            return self._tools.db.state.response(
                {"written": False, **error}, ok=False,
            )
        # Policy is evaluated against the trusted request identity, never
        # caller-supplied payload fields (agent_id/client are not write inputs).
        allowed, policy_warnings = self._allowed()
        if not allowed:
            return self._tools.db.state.response(
                {"written": False}, ok=False, extra_warnings=policy_warnings,
            )
        if not str(payload.get("subject") or "").strip():
            return self._tools.db.state.response(
                {"written": False, "error": "subject is required"},
                ok=False, extra_warnings=policy_warnings,
            )
        if self.settings.isolation == "strict" and not str(payload.get("workspace") or "").strip():
            return self._tools.db.state.response(
                {"written": False, "error": "isolation=strict requires a workspace on every write"},
                ok=False, extra_warnings=policy_warnings,
            )
        # Authoritative guard for callers that bypass product-surface validation
        # (console API, direct MemoryTools use): superseded/conflicted/deleted
        # are lifecycle outcomes, never caller-supplied write inputs.
        status_value = payload.get("status")
        if status_value is not None and status_value not in (
            MemoryStatus.ACTIVE.value, MemoryStatus.PENDING.value,
        ):
            return self._tools.db.state.response(
                {
                    "written": False,
                    "error": "invalid_input",
                    "field": "status",
                    "reason": "must be 'active' (default) or 'pending'; superseded/conflicted/deleted are lifecycle outcomes, not write inputs",
                },
                ok=False, extra_warnings=policy_warnings,
            )

        try:
            record = MemoryRecord.from_input(payload, self.settings.defaults())
            # Attribution channel: the trusted request identity (HTTP headers or
            # the stdio process identity) owns agent_id. Payload/agent input can
            # no longer smuggle provenance; when no trusted identity exists
            # (direct MemoryTools use), from_input's env/config fallback stands.
            record.agent_id = self.current_agent_id() or record.agent_id
            workspace = self._resolve_write_workspace(record)
            if workspace["strict_block"]:
                record.status = MemoryStatus.PENDING.value

            memory_id, write_warnings = self.db.insert_memory(
                record, workspace["canonical"], workspace.get("canonical_embedding"),
                register_workspace_canonical=not workspace["strict_block"],
            )
            if any("workspace canonical vector publish failed" in warning for warning in write_warnings):
                workspace["vector_publish_pending"] = True
            data: dict[str, Any] = {
                "id": memory_id,
                "backup_only": memory_id is None,
                "record": {**record.__dict__, "id": memory_id},
                "workspace_canonical": workspace["canonical"],
                "workspace_matched_by": workspace["matched_by"],
            }
            self._apply_workspace_response(data, workspace)
            if memory_id is not None:
                data["evidence_index"], data["semantic_conflict_check"] = (
                    self._post_commit(memory_id, data["record"], recheck_conflicts=True)
                )
            response = self._tools.db.state.response(
                data,
                extra_warnings=(
                    policy_warnings + validation.warnings + write_warnings + workspace["warnings"]
                ),
            )
            if data.get("evidence_index", {}).get("status") == "busy":
                depth = int(data["evidence_index"].get("queue_depth") or 0)
                response.setdefault("warnings", []).append(
                    f"evidence indexer is busy (queue depth {depth}); memory written but "
                    "evidence indexing is delayed — slow down writes or retry later."
                )
            if workspace["is_new"] and not workspace["strict_block"]:
                response.setdefault("notices", []).append({
                    "type": "workspace_review",
                    "severity": "info",
                    "workspace": workspace["canonical"],
                    "message": (
                        f"New workspace {workspace['canonical']!r} was registered. "
                        "Review the workspace registry for duplicates before confirming it."
                    ),
                    "action_required": "review_workspace_registry",
                    "review_call": {
                        "tool": "memory_review",
                        "view": "doctor",
                        "data": {},
                    },
                    "confirm_call": {
                        "tool": "memory_govern",
                        "action": "confirm_workspaces",
                        "data": {},
                    },
                    "authorization_required": True,
                })
            if memory_id is not None and record.status == MemoryStatus.ACTIVE.value:
                similar_notice = self._similar_active_notice(
                    int(memory_id), record, workspace["canonical"],
                )
                if similar_notice is not None:
                    response.setdefault("notices", []).append(similar_notice)
            return response
        except Exception as exc:
            return self._tools.db.state.response(
                {"written": False, "error": str(exc)},
                ok=False, extra_warnings=policy_warnings + validation.warnings,
            )

    def _resolve_write_workspace(self, record: MemoryRecord) -> dict[str, Any]:
        raw = record.workspace
        result: dict[str, Any] = {
            "canonical": raw,
            "is_new": False,
            "matched_by": "fallback",
            "similar": [],
            "warnings": [],
            "decision": None,
            "decision_reason": None,
            "candidate": None,
            "vector_publish_pending": False,
            "strict_block": False,
            "canonical_embedding": None,
        }
        isolation = self.settings.isolation

        embedder, warnings = self._ensure_active_embedder()
        result["warnings"].extend(warnings)
        # Resolution is read-only. Registration of only the final policy result
        # happens atomically in insert_memory.
        resolved = self.db.resolve_workspace_canonical(raw, embedder, register_new=False)
        result.update({
            "canonical": resolved["canonical"],
            "is_new": bool(resolved["is_new"]),
            "matched_by": resolved["matched_by"],
            "similar": resolved.get("similar") or [],
            "vector_publish_pending": bool(resolved.get("vector_publish_pending")),
        })
        candidate_embedding = resolved.get("candidate_embedding")
        if result["canonical"] == raw and candidate_embedding:
            result["canonical_embedding"] = candidate_embedding
        result["warnings"].extend(resolved.get("warnings") or [])
        # Confirmed aliases short-circuit before embedding their canonical text.
        # Backfill a missing canonical vector once, outside the memory write
        # transaction; insert_memory publishes this prepared vector post-commit.
        # Exact matches take the same repair path so the "retry a write using
        # this workspace" guidance actually republishes a missing vector.
        if result["matched_by"] in {"confirmed_alias", "exact"}:
            result["canonical_embedding"] = (
                self.db.workspaces.prepare_missing_workspace_canonical_embedding(
                    result["canonical"], embedder,
                )
            )
        evidence = workspace_rules.extract_evidence(record)
        rule = workspace_rules.rule_decision(raw, resolved, evidence)
        result["decision"] = rule["decision"]
        result["decision_reason"] = rule["reason"]
        if isolation == "strict" and result["matched_by"] == "vector":
            # Vector similarity is a candidate, not a safe mechanical identity.
            # Strict mode must never silently merge it.
            result["canonical"] = raw.strip() or result["canonical"]
            result["is_new"] = True
            result["matched_by"] = "strict_candidate"
            result["decision"] = "ASK"
            result["decision_reason"] = "strict_requires_confirmation"
            result["canonical_embedding"] = candidate_embedding
        elif rule["decision"] == "KEEP" and result["matched_by"] == "vector":
            result["canonical"] = raw.strip() or result["canonical"]
            result["is_new"] = True
            result["matched_by"] = "rule_keep"
            result["canonical_embedding"] = candidate_embedding
        elif rule["decision"] is None:
            suggestion = self._suggest_workspace_candidate(raw, evidence, result["similar"])
            result["candidate"] = suggestion
            rejected = bool(
                suggestion is not None and suggestion.candidate
                and suggestion.candidate in (resolved.get("rejected_canonicals") or [])
            )
            if (
                suggestion is not None and suggestion.candidate
                and suggestion.relation in {"alias", "typo", "same_project"}
                and isolation in {"none", "weak"} and (suggestion.confidence or 0.0) >= 0.85
                and not rejected
            ):
                result["canonical"] = suggestion.candidate
                result["is_new"] = False
                result["matched_by"] = "qwen"
                result["decision"] = "AUTO"
                result["decision_reason"] = "qwen_high_conf"
                # The selected candidate already exists; never publish the raw
                # query embedding under the chosen canonical.
                result["canonical_embedding"] = None
            else:
                # Behavior is unchanged (keep raw canonical + workspace_review);
                # the reason distinguishes model-absent/timeout/uncertain from a
                # genuine low-confidence model output (spec §8 diagnostics).
                result["decision"] = "ASK"
                err = str(suggestion.error).lower() if (suggestion and suggestion.error) else ""
                if suggestion is None:
                    result["decision_reason"] = "qwen_unavailable" if result["similar"] else "no_similar_candidates"
                elif rejected:
                    result["decision_reason"] = "qwen_rejected_candidate"
                elif "timeout" in err or "deadline" in err:
                    # Admission/inference deadlines surface as "... deadline
                    # expired ..." not the literal "timeout"; both are technical.
                    result["decision_reason"] = "qwen_timeout"
                elif err:
                    # Any other backend error (disabled/crashed/invalid child)
                    # is a technical failure, not a low-confidence model output.
                    result["decision_reason"] = "qwen_backend_error"
                elif not suggestion.candidate and suggestion.relation == "unrelated":
                    result["decision_reason"] = "qwen_unrelated"
                else:
                    result["decision_reason"] = "qwen_low_conf"
        # Empty/default workspace: offer a NON-binding placement suggestion from
        # the memory's own subject (thought A: nearest existing memory's
        # workspace). A real-library A/B showed this is accurate but its
        # distance scale is not comparable to the name-vector threshold, so it
        # must never auto-assign — global memories (user prefs/identity) belong
        # in default. Read-only: suggest, let the agent/user decide.
        # all reserved default synonyms resolve to the same canonical
        # ("default"), so the hint fires for 默认/none/未知 writes too.
        if is_default_workspace_term(str(result["canonical"] or "")):
            result["placement_suggestion"] = self._suggest_placement_for_default(record)
        result["strict_block"] = isolation == "strict" and result["is_new"]
        return result

    def _suggest_placement_for_default(self, record: MemoryRecord) -> dict[str, Any] | None:
        """Read-only subject-based placement hint for a default/empty workspace.

        Embeds the subject and finds the nearest existing memory that lives in a
        real (non-default) workspace. Returns a suggestion dict or None. Never
        writes, never auto-assigns; global memories intentionally stay in
        default when no confident neighbor exists.
        """
        # Under strict isolation the unscoped KNN would leak the existence and
        # canonical name of a foreign workspace's memory to a caller whose read
        # ACL hides it; the hint is a none/weak convenience only.
        if getattr(self.settings, "isolation", "none") == "strict":
            return None
        subject = str(getattr(record, "subject", None) or "").strip()
        if not subject:
            return None
        embedder, _ = self._ensure_active_embedder()
        if embedder is None or not self.db.state.sqlite_vec_available:
            return None
        try:
            er = embedder.embed_text(prefix="", body=subject)
        except Exception:
            return None
        if not er or not er.embedding:
            return None
        try:
            hits = self.db.evidence_knn(list(er.embedding), k=8)
        except Exception:
            return None
        for hit in hits:
            peer = self.db.get_memory(int(hit.get("memory_id") or 0))
            if not peer or peer.get("status") != "active":
                continue
            ws = str(peer.get("workspace_canonical") or peer.get("workspace") or "").strip()
            if ws and not is_default_workspace_term(ws):
                return {
                    "suggested_workspace": ws,
                    "from_memory_id": int(peer.get("id") or 0),
                    "basis": "subject_nearest_memory",
                    "note": (
                        "Not auto-assigned: this memory is in 'default'. Its subject is "
                        f"closest to memory #{peer.get('id')} in workspace {ws!r}. If it belongs "
                        "there, rewrite with that workspace or use governance to move it."
                    ),
                }
        return None

    def _apply_workspace_response(self, data: dict[str, Any], workspace: dict[str, Any]) -> None:
        if workspace["vector_publish_pending"]:
            data["workspace_vector_publish"] = {
                "status": "pending_retry",
                "canonical": workspace["canonical"],
                "repair_task_available": False,
                "retry": "Write another memory using this workspace after sqlite-vec recovers.",
            }
        if workspace["strict_block"]:
            data.update({
                "attention_required": True,
                "action_required": "confirm_new_workspace",
                "verification_status": "pending_user",
                "workspace_is_new": True,
                "pending_workspace": {
                    "canonical": workspace["canonical"],
                    "similar_workspaces": workspace["similar"],
                },
                "attention_summary": (
                    f"strict isolation: workspace {workspace['canonical']!r} is new; "
                    "confirm it with memory_govern(action='confirm_pending_workspace')"
                ),
            })
        elif workspace["is_new"]:
            data.setdefault("write_hints", {})["new_workspace_detected"] = {
                "canonical": workspace["canonical"],
                "similar_workspaces": workspace["similar"],
            }
        if workspace["decision"] is not None:
            data["workspace_decision"] = workspace["decision"]
            data["workspace_decision_reason"] = workspace["decision_reason"]
        suggestion = workspace.get("candidate")
        if suggestion is not None and suggestion.candidate:
            data["workspace_candidate"] = {
                "candidate": suggestion.candidate,
                "relation": suggestion.relation,
                "confidence": suggestion.confidence,
                "evidence": suggestion.evidence,
            }
        if workspace["decision"] == "ASK" and not workspace["strict_block"]:
            similar = workspace["similar"]
            options: list[dict[str, Any]] = [
                {"decision": "keep_separate", "action": None},
            ]
            if similar:
                options.insert(0, {
                    "decision": "merge",
                    "call": {
                        "tool": "memory_govern",
                        "action": "migrate_workspace",
                        "data": {
                            "from": data["record"]["workspace"],
                            "to": similar[0].get("name"),
                        },
                    },
                    "authorization_required": True,
                })
            data.setdefault("write_hints", {})["workspace_review"] = {
                "raw": data["record"]["workspace"],
                "reason": workspace["decision_reason"],
                "similar_workspaces": similar,
                "options": options,
                "note": "Merge only after user confirmation; otherwise keep the workspace separate.",
            }
        placement = workspace.get("placement_suggestion")
        if placement:
            # Non-binding: the memory was still written to default. Surface a
            # subject-based placement hint so the agent/user can move it if it
            # truly belongs elsewhere.
            data.setdefault("write_hints", {})["placement_suggestion"] = placement
