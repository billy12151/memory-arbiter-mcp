"""Judgment-queue protocol (0.16.0 plan §6㉑/§2 commit 5).

The agent processes the scan_queue page by page — "handle page 1, submit its
dispositions with the next page fetch" — while the server does all the
transport work:

- Page assembly builds TRANSITIVE-CLOSURE GROUPS from pair rows (A↔B + A↔C →
  {A,B,C}) purely as judgment units (§6⑥): groups are never persisted; the
  landed state is per-pair. Oversized components (>N members) are split back
  into their constituent edges.
- Submission is server-side land-from-reference: the agent submits
  ``{candidate_key_hash, status, reason}`` plus — only for confirms — the
  slot key and per-group display values; the server resolves the frozen
  envelope from the queue row and drives the existing ``record_conflict``
  (confirm → open, dismiss → not_a_conflict suppression source). Queue rows
  NEVER enter ``conflicts`` themselves.
- Page boundaries are natural breakpoints: a decision that arrives after a
  crash resumes from wherever the queue stands; nothing is lost because
  nothing was held in memory.
"""
from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from .constants import is_default_workspace_term
from .models import utc_now_iso

if TYPE_CHECKING:
    from .tools import MemoryTools

# §1.5: 10-30 groups per page (owner-pinned band; default at the band floor).
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 30
# §6⑬: a closure component with more than N members is split back into its
# edges — one judgment item per pair — instead of one mega-group.
GROUP_MEMBER_CAP = 10
# Fetch window for group assembly per page call (rows, not groups).
ASSEMBLY_WINDOW = 400


class QueueProtocol:
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools
        self.db = tools.db

    # ── page fetch ──────────────────────────────────────────────────────────

    def page(self, *, page_size: int = DEFAULT_PAGE_SIZE, page_token: int = 0) -> dict[str, Any]:
        page_size = max(1, min(int(page_size or DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE))
        page_token = max(0, int(page_token or 0))
        items: list[dict[str, Any]] = []
        meta_cache: dict[int, dict[str, Any]] = {}
        last_id = page_token
        # Priority: workspace suspects first (rare, cheap to judge, gate
        # autonomous moves), then internal contradictions, then conflict
        # groups — a big conflict backlog must not starve the other kinds.
        for row in self._fetch_workspace_rows():
            if len(items) >= page_size:
                break
            items.append(self._workspace_item(row, meta_cache))
        if len(items) < page_size:
            for row in self.db.internal_conflicts.list_pending(limit=page_size):
                if len(items) >= page_size:
                    break
                items.append(self._internal_item(row, meta_cache))
        if len(items) < page_size:
            rows = self._fetch_conflict_rows(page_token)
            for group in self._assemble_groups(rows):
                last_id = max(last_id, group["last_queue_id"])
                if len(items) >= page_size:
                    break
                if len(group["member_ids"]) > GROUP_MEMBER_CAP:
                    # §6⑬: oversize component → split back into edges.
                    for edge in group["pairs"]:
                        if len(items) >= page_size:
                            break
                        items.append(self._pair_item(edge, meta_cache))
                else:
                    items.append(self._group_item(group, meta_cache))
        remaining = self.db.scan_queue_backlog()
        response: dict[str, Any] = {
            "ok": True,
            "items": items[:page_size],
            "count": len(items[:page_size]),
            "queue_backlog": remaining,
            "detector_version": self._detector_version(),
            "instruction": (
                "Judge each item from the evidence quotes; upgrade an uncertain item with "
                "memory(action='batch_read', content_mode='hits', spans={...evidence_span...}) "
                "before deciding, and read full texts before any confirm-driven edit. Submit "
                "dispositions with memory_repair(task='scan_queue', action='submit'): per-pair "
                "{candidate_key_hash, status: confirmed|dismissed, reason} + (confirms) "
                "slot_key + value_groups with display_value per member group. A whole group "
                "that is noise can be dismissed in one entry with the group's group_token."
            ),
        }
        # has_more: any backlog beyond what this page displayed — the queue
        # backlog itself is authoritative (workspace/internal rows live in
        # separate tables and may not all fit this page).
        has_more = remaining > len(items[:page_size])
        if items and (last_id > page_token or remaining):
            response["next_page_token"] = last_id
        response["has_more"] = bool(has_more)
        if self.db.settings.include_size and items:
            from .tokens import meter_payloads

            response["size"] = meter_payloads(items[:page_size])
        return response

    @staticmethod
    def _component_token(pairs: list[dict[str, Any]]) -> str:
        import hashlib

        joined = ":".join(sorted(str(pair["candidate_key_hash"]) for pair in pairs))
        return hashlib.sha256(joined.encode()).hexdigest()[:16]

    def _fetch_conflict_rows(self, after_id: int) -> list[dict[str, Any]]:
        if not self.db.db_available:
            return []
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """SELECT id,kind,workspace_canonical,status,candidate_key_hash,
                              member_versions,evidence,reason,severity,source,detail
                       FROM scan_queue WHERE status='pending' AND kind='conflict' AND id>?
                       ORDER BY id LIMIT ?""",
                    (int(after_id), ASSEMBLY_WINDOW),
                ).fetchall()
        except Exception:
            return []
        decoded: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("member_versions", "evidence", "detail"):
                if isinstance(item.get(key), str):
                    try:
                        item[key] = json.loads(item[key])
                    except (TypeError, json.JSONDecodeError):
                        item[key] = None
            decoded.append(item)
        return decoded

    def _assemble_groups(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Union-find pair rows into closure groups by shared member refs."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        row_refs: dict[int, list[str]] = {}
        for row in rows:
            refs = [
                f"{int(member['memory_id'])}@{int(member.get('version') or 1)}"
                for member in (row.get("member_versions") or [])
                if member.get("memory_id") is not None
            ]
            row_refs[int(row["id"])] = refs
            for ref in refs:
                union(ref, refs[0])

        components: dict[str, dict[str, Any]] = {}
        for row in rows:
            refs = row_refs[int(row["id"])]
            root = find(refs[0]) if refs else f"row:{row['id']}"
            component = components.setdefault(root, {"pairs": [], "member_ids": set(), "last_queue_id": 0})
            component["pairs"].append(row)
            component["last_queue_id"] = max(component["last_queue_id"], int(row["id"]))
            for member in row.get("member_versions") or []:
                if member.get("memory_id") is not None:
                    component["member_ids"].add(int(member["memory_id"]))
        return sorted(components.values(), key=lambda c: c["last_queue_id"])

    def _member_meta(self, memory_id: int, cache: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
        if memory_id not in cache:
            record = self.db.get_memory(memory_id)
            if not record:
                cache[memory_id] = None
            else:
                tags = record.get("tags")
                cache[memory_id] = {
                    "subject": str(record.get("subject") or "")[:200],
                    "tags": (tags if isinstance(tags, list) else [])[:20],
                    "event_time": record.get("event_time"),
                    "workspace": record.get("workspace_canonical") or record.get("workspace"),
                    "version": int(record.get("version") or 1),
                }
        return cache[memory_id]

    def _pair_item(self, row: dict[str, Any], cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        members = row.get("member_versions") or []
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        return {
            "kind": "conflict",
            "group_token": f"pair:{row['candidate_key_hash'][:16]}",
            "pair_count": 1,
            "member_ids": sorted({int(m["memory_id"]) for m in members if m.get("memory_id") is not None}),
            "member_meta": {
                str(m["memory_id"]): self._member_meta(int(m["memory_id"]), cache)
                for m in members if m.get("memory_id") is not None
            },
            "pairs": [{
                "queue_id": int(row["id"]),
                "candidate_key_hash": row["candidate_key_hash"],
                "severity": row.get("severity"),
                "reason": row.get("reason"),
                "workspace": row.get("workspace_canonical"),
                "evidence": row.get("evidence") or [],
                "candidate_key": detail.get("candidate_key"),
            }],
        }

    def _group_item(self, group: dict[str, Any], cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        members = group["pairs"][0].get("member_versions") or []
        return {
            "kind": "conflict",
            "group_token": f"group:{self._component_token(group['pairs'])}",
            "pair_count": len(group["pairs"]),
            "member_ids": sorted(group["member_ids"]),
            "member_meta": {
                str(mid): self._member_meta(mid, cache) for mid in sorted(group["member_ids"])
            },
            "pairs": [{
                "queue_id": int(row["id"]),
                "candidate_key_hash": row["candidate_key_hash"],
                "severity": row.get("severity"),
                "reason": row.get("reason"),
                "workspace": row.get("workspace_canonical"),
                "evidence": row.get("evidence") or [],
                "candidate_key": (row.get("detail") or {}).get("candidate_key")
                if isinstance(row.get("detail"), dict) else None,
            } for row in group["pairs"]],
        }

    def _internal_item(self, row: dict[str, Any], cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        return {
            "kind": "internal",
            "internal_id": int(row["id"]),
            "memory_id": int(row["memory_id"]),
            "memory_version": int(row["memory_version"]),
            "member_meta": {str(row["memory_id"]): self._member_meta(int(row["memory_id"]), cache)},
            "quote_a": row.get("quote_a"),
            "quote_b": row.get("quote_b"),
            "span_a": row.get("span_a"),
            "span_b": row.get("span_b"),
            "reason": row.get("reason"),
            "instruction": (
                "One memory contradicting itself: dismiss if compatible, or fix the memory "
                "(memory action='update') — the row auto-stales once the version lifts."
            ),
        }

    def _fetch_workspace_rows(self) -> list[dict[str, Any]]:
        if not self.db.db_available:
            return []
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """SELECT id,kind,workspace_canonical,status,candidate_key_hash,
                              member_versions,evidence,reason,severity,source,detail
                       FROM scan_queue WHERE status='pending' AND kind='workspace'
                       ORDER BY id LIMIT ?""",
                    (ASSEMBLY_WINDOW,),
                ).fetchall()
        except Exception:
            return []
        decoded: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("member_versions", "evidence", "detail"):
                if isinstance(item.get(key), str):
                    try:
                        item[key] = json.loads(item[key])
                    except (TypeError, json.JSONDecodeError):
                        item[key] = None
            decoded.append(item)
        return decoded

    def _workspace_item(self, row: dict[str, Any], cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        member = (row.get("member_versions") or [{}])[0]
        memory_id = int(member.get("memory_id") or 0)
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        outline: list[dict[str, Any]] = []
        content_chars = 0
        record = self.db.get_memory(memory_id)
        if record:
            from .pipeline.read import _content_outline

            content = str(record.get("content") or "")
            content_chars = len(content)
            outline = _content_outline(str(record.get("subject") or ""), content)
        return {
            "kind": "workspace",
            "queue_id": int(row["id"]),
            "candidate_key_hash": row["candidate_key_hash"],
            "memory_id": memory_id,
            "memory_version": member.get("version"),
            "current_workspace": detail.get("current_workspace") or row.get("workspace_canonical"),
            "suspected_workspace": detail.get("suspected_workspace"),
            "votes": detail.get("votes") or {},
            "protected_involved": bool(detail.get("protected_involved")),
            "member_meta": {str(memory_id): self._member_meta(memory_id, cache)},
            "content_chars": content_chars,
            "outline": outline,
            "reason": row.get("reason"),
            "instruction": (
                "E5 summary input: judge placement from subject/tags/outline. Confirm with "
                "{kind:'workspace', memory_id, status:'confirmed', target_workspace, conf} — "
                "the server re-runs the vector vote and only moves when the two signals agree "
                "(protected buckets are never moved autonomously)."
            ),
        }

    def _detector_version(self) -> str:
        from .db_generation import CONFLICT_DETECTOR_VERSION

        return CONFLICT_DETECTOR_VERSION

    # ── submission (server-side land-from-reference) ───────────────────────

    def submit(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(decisions, list) or not decisions:
            return {"ok": False, "error": "decisions must be a non-empty list"}
        if len(decisions) > 200:
            return {"ok": False, "error": "at most 200 decisions per submission"}
        results: list[dict[str, Any]] = []
        for index, raw in enumerate(decisions):
            results.append(self._submit_one(index, raw))
        handled = {"confirmed", "dismissed", "resolved", "skipped", "moved",
                   "protected_bucket_hint", "multi_family_hint"}
        ok = all(item.get("outcome") in handled for item in results)
        return {
            "ok": ok,
            "results": results,
            "queue_backlog": self.db.scan_queue_backlog(),
        }

    def _submit_one(self, index: int, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {"index": index, "outcome": "invalid_input", "error": "decision must be an object"}
        status = str(raw.get("status") or "").strip().lower()
        reason = str(raw.get("reason") or "")
        kind = str(raw.get("kind") or "conflict").strip().lower()
        if kind == "workspace":
            return self._submit_workspace(index, status, reason, raw)
        if kind == "internal":
            internal_id = raw.get("internal_id")
            if status not in {"dismissed", "resolved"}:
                return {"index": index, "outcome": "invalid_input",
                        "error": "internal decisions accept status dismissed|resolved"}
            outcome = self.db.internal_conflicts.decide(
                int(internal_id), status, reason=reason,
            )
            if outcome.get("outcome") == "updated":
                outcome = {"outcome": status, **{
                    key: value for key, value in outcome.items() if key != "outcome"
                }}
            return {"index": index, "kind": "internal", **outcome}
        if status not in {"confirmed", "dismissed"}:
            return {"index": index, "outcome": "invalid_input",
                    "error": "status must be confirmed|dismissed"}
        group_token = raw.get("group_token")
        if group_token and not raw.get("candidate_key_hash"):
            return self._submit_group(index, str(group_token), status, reason, raw)
        candidate_hash = str(raw.get("candidate_key_hash") or "")
        if len(candidate_hash) != 64:
            return {"index": index, "outcome": "invalid_input",
                    "error": "candidate_key_hash must be the 64-char hash from the queue page"}
        return self._decide_row(index, candidate_hash, status, reason, raw)

    def _submit_workspace(
        self, index: int, status: str, reason: str, raw: dict[str, Any],
    ) -> dict[str, Any]:
        from .constants import (
            NORMALIZE_MIN_CONF, NORMALIZE_VOTE_NEIGHBORS, NORMALIZE_VOTE_SHARE_MIN,
            PROTECTED_WORKSPACES,
        )
        from .scan_pipeline import _workspace_identity

        if status not in {"confirmed", "dismissed"}:
            return {"index": index, "outcome": "invalid_input",
                    "error": "workspace decisions accept status confirmed|dismissed"}
        memory_id = raw.get("memory_id")
        if not isinstance(memory_id, int) or memory_id <= 0:
            return {"index": index, "outcome": "invalid_input", "error": "memory_id required"}
        record = self.db.get_memory(memory_id)
        if not record or record.get("status") != "active":
            return {"index": index, "outcome": "not_found", "memory_id": memory_id}
        version = int(record.get("version") or 1)
        current = str(record.get("workspace_canonical") or record.get("workspace") or "")
        if status == "dismissed":
            self._expire_workspace_rows(memory_id, version, "dismissed", reason)
            return {"index": index, "outcome": "dismissed", "memory_id": memory_id}
        # confirmed: the four-part gate (E7) re-runs at decision time.
        target = str(raw.get("target_workspace") or "").strip()
        try:
            conf = float(raw.get("conf") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if not target:
            return {"index": index, "outcome": "invalid_input",
                    "error": "confirmed workspace moves need target_workspace"}
        if current in PROTECTED_WORKSPACES or target in PROTECTED_WORKSPACES:
            # E6: protected buckets are never moved autonomously — user hint.
            self._expire_workspace_rows(memory_id, version, "dismissed",
                                        f"protected bucket involved: {current!r}->{target!r}")
            return {
                "index": index, "outcome": "protected_bucket_hint",
                "memory_id": memory_id, "current": current, "target": target,
                "hint": "受保护桶不自动搬——请向用户提示疑似写错桶，由用户自行处置",
            }
        if conf < NORMALIZE_MIN_CONF:
            return {"index": index, "outcome": "gate_failed", "gate": "conf",
                    "memory_id": memory_id, "conf": conf}
        multi = self._multi_family_mentions(record, target)
        if multi:
            # E7-4: multi-family mentions downgrade to a user hint, no move.
            self._expire_workspace_rows(memory_id, version, "dismissed",
                                        f"multi-family mention: {multi}")
            return {"index": index, "outcome": "multi_family_hint",
                    "memory_id": memory_id, "families": multi}
        vote = self._workspace_vote(memory_id)
        if vote is None:
            return {"index": index, "outcome": "gate_failed", "gate": "vote_unavailable",
                    "memory_id": memory_id}
        top_bucket, share, neighbours = vote
        if top_bucket != target or share < NORMALIZE_VOTE_SHARE_MIN:
            return {
                "index": index, "outcome": "gate_failed", "gate": "vote",
                "memory_id": memory_id, "top_bucket": top_bucket,
                "share": share, "neighbours": neighbours,
            }
        moved, warnings = self._execute_auto_move(memory_id, current, target, vote, conf)
        if not moved:
            return {"index": index, "outcome": "move_failed", "warnings": warnings}
        self._expire_workspace_rows(memory_id, version, "confirmed", reason)
        return {
            "index": index, "outcome": "moved", "memory_id": memory_id,
            "from": current, "to": target, "vote": {"top": top_bucket, "share": share},
            "warnings": warnings,
        }

    def _expire_workspace_rows(self, memory_id: int, version: int, status: str, why: str) -> None:
        try:
            with self.db.write_transaction() as conn:
                conn.execute(
                    """UPDATE scan_queue SET status=?, decided_reason=?, decided_at=CURRENT_TIMESTAMP,
                       updated_at=CURRENT_TIMESTAMP
                       WHERE kind='workspace' AND status='pending'
                         AND EXISTS(SELECT 1 FROM json_each(scan_queue.member_versions) AS m
                                    WHERE CAST(json_extract(m.value,'$.memory_id') AS INTEGER)=?)""",
                    (status, why, int(memory_id)),
                )
        except Exception:
            pass

    def _multi_family_mentions(self, record: dict[str, Any], target: str) -> list[str]:
        """E7-4: subject/tags mentioning >=2 registered project families
        downgrades the case to a user hint (cross-project meta content)."""
        tags = record.get("tags") if isinstance(record.get("tags"), list) else []
        text = (str(record.get("subject") or "") + " " + " ".join(str(t) for t in tags)).casefold()
        if not text.strip():
            return []
        try:
            with self.db.connection() as conn:
                names = [str(r["name"]) for r in conn.execute(
                    "SELECT name FROM workspace_canonicals").fetchall()]
        except Exception:
            return []
        mentioned = sorted({
            name for name in names
            if len(name) >= 4 and not is_default_workspace_term(name)
            and name.casefold() in text
        })
        return mentioned if len(mentioned) >= 2 else []

    def _workspace_vote(self, memory_id: int) -> "tuple[str, int, int] | None":
        """Decision-time vector vote (E7①: 现算票). Returns (top_foreign_bucket,
        its_share, neighbours_checked); None without vectors/numpy."""
        try:
            import numpy as np
        except ImportError:
            return None
        from .constants import NORMALIZE_VOTE_NEIGHBORS

        vectors = self.db.memories.all_summary_vectors()
        if memory_id not in vectors:
            return None
        all_ids = sorted(vectors)
        workspaces = {mid: str(vectors[mid][0] or "") for mid in all_ids}
        own = workspaces[memory_id]
        matrix = np.array([vectors[mid][1] for mid in all_ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        unit = matrix / norms[:, None]
        row = all_ids.index(memory_id)
        sims = unit @ unit[row]
        sims[row] = -1.0
        k = min(NORMALIZE_VOTE_NEIGHBORS, len(all_ids) - 1)
        if k <= 0:
            return None
        order = np.argsort(-sims, kind="stable")[:k]
        votes: dict[str, int] = {}
        for col in order:
            bucket = workspaces[all_ids[int(col)]]
            votes[bucket] = votes.get(bucket, 0) + 1
        top_bucket, top_votes = max(votes.items(), key=lambda item: item[1], default=("", 0))
        return top_bucket, top_votes, k

    def _execute_auto_move(
        self, memory_id: int, current: str, target: str,
        vote: "tuple[str, int, int]", conf: float,
    ) -> "tuple[bool, list[str]]":
        from .constants import NORMALIZE_VOTE_SHARE_MIN

        try:
            with self.db.write_transaction() as conn:
                moved, move_warnings = self.db.workspaces.move_memory_workspace_on_conn(
                    conn, memory_id, target,
                )
                if not moved:
                    return False, move_warnings
                cur = conn.execute(
                    """INSERT INTO normalize_audit(
                         memory_id, from_workspace, to_workspace, gate, status, created_at)
                       VALUES(?,?,?,?, 'applied', ?)""",
                    (int(memory_id), current, target,
                     json.dumps({"vote_top": vote[0], "vote_share": f"{vote[1]}/{vote[2]}",
                                 "share_min": NORMALIZE_VOTE_SHARE_MIN, "conf": conf},
                                ensure_ascii=False),
                     utc_now_iso()),
                )
            return True, move_warnings
        except Exception as exc:
            return False, [f"auto move failed: {exc}"]

    def _submit_group(
        self, index: int, group_token: str, status: str, reason: str,
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        """Group-level dismissal (§6⑥): one entry suppresses every pair of
        the group. Confirms stay per-pair (each pair's slot/values differ)."""
        if status != "dismissed":
            return {"index": index, "outcome": "invalid_input",
                    "error": "group decisions accept status=dismissed only; confirm per pair"}
        rows = self._fetch_conflict_rows(0)
        wanted = group_token.split(":", 1)[-1]
        targets = None
        for component in self._assemble_groups(rows):
            if self._component_token(component["pairs"]) == wanted:
                targets = component["pairs"]
                break
        if not targets:
            return {"index": index, "outcome": "not_found", "group_token": group_token}
        results = [
            self._decide_row(f"{index}.{i}", row["candidate_key_hash"], "dismissed", reason, {})
            for i, row in enumerate(targets)
        ]
        return {
            "index": index, "outcome": "dismissed", "group_token": group_token,
            "pairs": results,
        }

    def _decide_row(
        self, index: Any, candidate_hash: str, status: str, reason: str,
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        row = self._queue_row(candidate_hash)
        if row is None:
            return {"index": index, "outcome": "not_found",
                    "candidate_key_hash": candidate_hash}
        if row["status"] not in {"pending", "in_review"}:
            return {"index": index, "outcome": "already_terminal", "status": row["status"],
                    "candidate_key_hash": candidate_hash}
        members = row["member_versions"] or []
        detail = row["detail"] if isinstance(row["detail"], dict) else {}
        candidate_key = detail.get("candidate_key")
        # Migrated legacy rows freeze their ORIGINAL detector identity — the
        # intake gate rejects a member/detector mismatch (D1 family), so the
        # disposition runs under the row's own stamp, never the running one.
        row_detector = str(
            (members[0] or {}).get("detector_version") or ""
        ).strip() or self._detector_version()
        if status == "dismissed":
            result = self.db.record_conflict_group(
                workspace_canonical=row["workspace_canonical"],
                slot_key=None,
                members=members,
                value_groups=[],
                candidate_key=candidate_key,
                status="not_a_conflict",
                detector_version=row_detector,
                source="scan_queue",
                detection_reason=reason or "dismissed from scan queue",
            )
            outcome = result.get("outcome")
            if outcome in {"inserted", "deduped"}:
                self._mark_decided(candidate_hash, "dismissed", decided_ref=result.get("conflict_id"), reason=reason)
                return {"index": index, "outcome": "dismissed",
                        "conflict_id": result.get("conflict_id")}
            if outcome == "stale_snapshot":
                # §6㉑④: version drift — expire and let the pipeline re-enqueue
                # the current identity; never a silent drop.
                self._expire_row(candidate_hash, "member versions drifted")
                return {"index": index, "outcome": "stale_snapshot",
                        "requeued": True, "detail": result}
            return {"index": index, "outcome": "dismiss_failed", "detail": result}
        # confirmed → open promotion: the agent supplies slot + per-member
        # display values; the server enriches the frozen envelope (D1: the
        # stored normalized_value is derived mechanically from value_raw).
        slot_key = raw.get("slot_key")
        value_groups = raw.get("value_groups")
        if not isinstance(slot_key, dict) or not isinstance(value_groups, list):
            return {"index": index, "outcome": "invalid_input",
                    "error": "confirm requires slot_key and value_groups"}
        enriched, enrich_error = self._enrich_members(members, value_groups, slot_key)
        if enrich_error:
            return {"index": index, "outcome": "invalid_input", "error": enrich_error}
        normalized_groups, groups_error = self._normalize_groups(value_groups)
        if groups_error:
            return {"index": index, "outcome": "invalid_input", "error": groups_error}
        result = self.db.record_conflict_group(
            workspace_canonical=row["workspace_canonical"],
            slot_key=slot_key,
            members=enriched,
            value_groups=normalized_groups,
            candidate_key=candidate_key,
            status="open",
            detector_version=row_detector,
            source="scan_queue",
            detection_reason=reason or "confirmed from scan queue",
        )
        outcome = result.get("outcome")
        if outcome in {"inserted", "deduped"}:
            self._mark_decided(candidate_hash, "confirmed", decided_ref=result.get("conflict_id"), reason=reason)
            return {"index": index, "outcome": "confirmed",
                    "conflict_id": result.get("conflict_id"), "revision": result.get("revision")}
        if outcome == "stale_snapshot":
            self._expire_row(candidate_hash, "member versions drifted")
            return {"index": index, "outcome": "stale_snapshot", "requeued": True, "detail": result}
        if outcome == "workspace_mismatch":
            # A member moved since enqueue: expire; the move voided nothing
            # here because the queue row is not a conflicts ticket, but the
            # pair can no longer land in one bucket.
            self._expire_row(candidate_hash, "workspace mismatch after move")
            return {"index": index, "outcome": "workspace_mismatch", "expired": True}
        return {"index": index, "outcome": "confirm_failed", "detail": result}

    @staticmethod
    def _normalize_groups(
        value_groups: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Complete the agent's display-value groups into the storage contract:
        normalized_value is mechanically derived (D1) — the agent never
        supplies it."""
        from .semantic_conflict import normalize_value

        normalized: list[dict[str, Any]] = []
        for group in value_groups:
            display = str(group.get("display_value") or "")
            refs = [str(ref) for ref in (group.get("members") or [])]
            if not display or not refs:
                return None, "each value_group needs display_value and members"
            normalized.append({
                "normalized_value": normalize_value(display),
                "display_value": display,
                "members": sorted(set(refs)),
            })
        return normalized, None

    def _queue_row(self, candidate_hash: str) -> dict[str, Any] | None:
        if not self.db.db_available:
            return None
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    "SELECT * FROM scan_queue WHERE candidate_key_hash=?",
                    (candidate_hash,),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        item = dict(row)
        for key in ("member_versions", "evidence", "detail"):
            if isinstance(item.get(key), str):
                try:
                    item[key] = json.loads(item[key])
                except (TypeError, json.JSONDecodeError):
                    item[key] = None
        return item

    @staticmethod
    def _enrich_members(
        members: list[dict[str, Any]], value_groups: list[dict[str, Any]],
        slot_key: dict[str, Any],
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Fill value/attribute fields from the agent's judgment (D1-safe).

        The queue envelope carries value-less deterministic members; an open
        promotion needs every member's normalized_value to match its group.
        The agent's value_groups give display values per member ref — the
        server derives value_raw/normalized_value mechanically so the D1
        intake gate holds by construction.
        """
        from .semantic_conflict import normalize_value

        by_ref = {str(g.get("members")): None for g in ()}  # noqa: F841 (shape doc)
        ref_to_value: dict[str, str] = {}
        for group in value_groups:
            display = str(group.get("display_value") or "")
            refs = group.get("members") or []
            if not display or not refs:
                return None, "each value_group needs display_value and members"
            for ref in refs:
                ref_to_value[str(ref)] = display
        enriched: list[dict[str, Any]] = []
        for member in members:
            item = dict(member)
            ref = f"{int(item['memory_id'])}@{int(item.get('version') or 1)}"
            display = ref_to_value.get(ref)
            if display is None:
                return None, f"member {ref} is not covered by any value_group"
            item["attribute_raw"] = str(slot_key.get("attribute") or "")
            item["normalized_attribute"] = str(slot_key.get("attribute") or "")
            item["value_raw"] = display
            item["normalized_value"] = normalize_value(display)
            enriched.append(item)
        if len(enriched) != len(ref_to_value):
            return None, "value_groups must cover exactly the pair members"
        return enriched, None

    def _mark_decided(
        self, candidate_hash: str, status: str, *, decided_ref: Any, reason: str,
    ) -> None:
        now = utc_now_iso()
        try:
            with self.db.write_transaction() as conn:
                conn.execute(
                    "UPDATE scan_queue SET status=?, decided_ref=?, decided_reason=?, "
                    "decided_at=?, updated_at=? WHERE candidate_key_hash=?",
                    (status, str(decided_ref) if decided_ref is not None else None,
                     reason, now, now, candidate_hash),
                )
        except Exception:
            pass

    def _expire_row(self, candidate_hash: str, why: str) -> None:
        now = utc_now_iso()
        try:
            with self.db.write_transaction() as conn:
                conn.execute(
                    "UPDATE scan_queue SET status='expired', decided_reason=?, decided_at=?, "
                    "updated_at=? WHERE candidate_key_hash=?",
                    (why, now, now, candidate_hash),
                )
        except Exception:
            pass
