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
# 0.16.4 §3: per-memory internal aggregation — pairs preview cap per page
# item. The FULL count travels as pair_count; a memory-level dismissal of a
# pair_count beyond this cap requires expanded=true (the agent read every
# pair via batch_read hits or the per-row channel first).
INTERNAL_PAIRS_CAP = 8
# 0.16.4 live-judgment review: group-level hash preview cap — full hash
# lists dominated real page bytes; group_token dispositions re-assemble the
# complete group server-side (see _submit_group), so the cap is display-only.
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


class QueueProtocol:
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools
        self.db = tools.db

    @staticmethod
    def _scope_sql(workspace_canonical_column: str, scope) -> "tuple[str, list[Any]]":
        from .acl import workspace_scope_sql

        return workspace_scope_sql(workspace_canonical_column, scope)

    # ── page fetch ──────────────────────────────────────────────────────────

    def page(self, *, page_size: int = DEFAULT_PAGE_SIZE, page_token: int = 0, caller=None) -> dict[str, Any]:
        self.db.internal_conflicts.expire_stale()
        self._caller = caller
        scope = caller.scope_canonicals() if caller is not None and caller.isolation == "strict" else None
        page_size = max(1, min(int(page_size or DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE))
        page_token = max(0, int(page_token or 0))
        items: list[dict[str, Any]] = []
        meta_cache: dict[int, dict[str, Any]] = {}
        last_id = page_token
        # Priority: workspace suspects first (rare, cheap to judge, gate
        # autonomous moves), then internal contradictions, then conflict
        # groups — a big conflict backlog must not starve the other kinds.
        for row in self._fetch_workspace_rows(page_token, scope):
            if len(items) >= page_size:
                break
            if not self._row_visible(row):
                continue
            items.append(self._workspace_item(row, meta_cache))
            # The cursor advances through workspace rows too: a page whose
            # items are all workspace/internal used to echo the caller's
            # token back unchanged (0 on the first page), which a
            # defensive agent reads as a pagination loop and stops early
            # (live repro 2026-09-13: the weekly task judged 67 of 125
            # suspects, never reached a single conflict pair, and exited
            # believing it was done). Judgement-free rows no longer block
            # the page either — the next page moves past them instead of
            # re-serving the same head forever.
            last_id = max(last_id, int(row["id"]))
        if len(items) < page_size:
            # Internal items are capped per page (0.16.4 §3: one aggregated
            # row per MEMORY — information-dense items, and the cap keeps a
            # fragment-noise flood from starving conflict judgment).
            internal_cap = min(3, page_size)
            for group in self.db.internal_conflicts.list_pending_grouped(
                limit_memories=internal_cap, pairs_cap=INTERNAL_PAIRS_CAP,
            ):
                if len(items) >= page_size:
                    break
                if not self._member_visible(int(group["memory_id"])):
                    continue
                items.append(self._internal_memory_item(group, meta_cache))
        if len(items) < page_size:
            rows = self._fetch_conflict_rows(page_token, scope)
            for group in self._assemble_groups(rows):
                if len(items) >= page_size:
                    # Page full: the cursor must stay on the last DISPLAYED
                    # group — advancing past the boundary here would strand
                    # the undisplayed group between pages forever (adversarial
                    # review repro #3).
                    break
                if not all(self._row_visible(edge) for edge in group["pairs"]):
                    # Fail closed: hide the whole component from a caller who
                    # cannot read every member (matches conflict_detail).
                    continue
                if len(group["member_ids"]) > GROUP_MEMBER_CAP:
                    # §6⑬: oversize component → split back into edges.
                    for edge in group["pairs"]:
                        if len(items) >= page_size:
                            break
                        items.append(self._pair_item(edge, meta_cache))
                        last_id = max(last_id, int(edge["id"]))
                else:
                    items.append(self._group_item(group, meta_cache))
                    last_id = max(last_id, group["last_queue_id"])
        remaining = self.db.scan_queue_backlog() + len(
            self.db.internal_conflicts.list_pending(limit=10**6)
        )
        if not items and page_token > 0 and remaining > 0:
            # Cursor wrap-around: the pass advanced past every row but some
            # backlog remains (rows the caller skipped or failed to judge).
            # Returning an empty page with the echoed token would signal a
            # pagination loop; restart from the head instead so the agent
            # gets another pass over the survivors. Terminates: the wrapped
            # call runs with page_token=0 and cannot re-enter this branch.
            return self.page(page_size=page_size, page_token=0, caller=caller)
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
                "that is noise can be dismissed in one entry with just the group's group_token "
                "(pair_hashes are a preview; the server re-assembles the whole group). "
                "internal_memory items (aggregated per memory): one "
                "{kind:'internal_memory', memory_id, status: dismissed|resolved, reason} "
                "clears the whole memory — pair_count beyond the pairs preview requires "
                "reading every pair first and resubmitting with expanded=true. Workspace "
                "suspects: confirmed with target_workspace + "
                "conf (server re-runs the vote gate); if NO suitable bucket exists, "
                "confirmed with target_workspace='default' AND fallback=true + reason "
                "(vote gate waived, audited, reported to the user — use sparingly)."
            ),
        }
        # has_more: any backlog beyond what this page displayed — the queue
        # backlog itself is authoritative (workspace/internal rows live in
        # separate tables and may not all fit this page).
        has_more = remaining > len(items[:page_size])
        if has_more:
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

    def _fetch_conflict_rows(self, after_id: int, scope=None) -> list[dict[str, Any]]:
        if not self.db.db_available:
            return []
        scope_sql, scope_params = self._scope_sql("workspace_canonical", scope)
        if scope_sql:
            scope_sql = " AND " + scope_sql
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """SELECT id,kind,workspace_canonical,status,candidate_key_hash,
                              member_versions,evidence,reason,severity,source,detail
                       FROM scan_queue WHERE status='pending' AND kind='conflict' AND id>?
                       """ + scope_sql + " ORDER BY id LIMIT ?",
                    (int(after_id), *scope_params, ASSEMBLY_WINDOW),
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

    def _member_visible(self, memory_id: int) -> bool:
        """Strict-isolation fail-closed check (adversarial review #2): a
        member the caller cannot read takes its whole item off the page and
        blocks its dispositions."""
        if self._caller is None or self._caller.isolation != "strict":
            return True
        return self._tools._get_memory_visible(int(memory_id), self._caller) is not None

    def _row_visible(self, row: dict[str, Any]) -> bool:
        return all(
            self._member_visible(int(member["memory_id"]))
            for member in (row.get("member_versions") or [])
            if member.get("memory_id") is not None
        )

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
            "pair_hashes": [str(row["candidate_key_hash"]) for row in group["pairs"][:GROUP_HASHES_CAP]],
            # 0.16.4 live-judgment review: the hash list is a PREVIEW (byte
            # budget — full lists dominated real pages); pair_count stays the
            # full total and a group_token disposition re-assembles the whole
            # group server-side, so the cap cannot strand pairs.
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

    def _internal_memory_item(self, group: dict[str, Any], cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        """0.16.4 §3: one judgment item per MEMORY — pairs cap 8 preview,
        full pair_count, reason distribution. The judgment unit is the
        memory, so no dedicated cursor: a disposition rolls the page."""
        memory_id = int(group["memory_id"])
        pairs = [
            {
                "internal_id": int(row["id"]),
                "quote_a": row.get("quote_a"),
                "quote_b": row.get("quote_b"),
                "span_a": row.get("span_a"),
                "span_b": row.get("span_b"),
                "reason": row.get("reason"),
            }
            for row in group["pairs"]
        ]
        return {
            "kind": "internal_memory",
            "memory_id": memory_id,
            "pair_count": int(group["pair_count"]),
            "pairs": pairs,
            "reasons_summary": group.get("reasons_summary") or {},
            "member_meta": {str(memory_id): self._member_meta(memory_id, cache)},
            "instruction": (
                "One memory contradicting itself (aggregated: the pairs list is a "
                f"preview of up to {INTERNAL_PAIRS_CAP}, pair_count is the full total). "
                "Dismiss the whole memory with {kind:'internal_memory', memory_id, "
                "status:'dismissed', reason} — if pair_count exceeds the preview, first "
                "read EVERY pair (memory action='batch_read', content_mode='hits', spans "
                "from the pairs' span_a/span_b) and resubmit with expanded=true. Resolve "
                "only for confirmed real contradictions. Fixing the memory "
                "(memory action='update') also works — rows auto-stale on version lift."
            ),
        }

    def _fetch_workspace_rows(self, after_id: int = 0, scope=None) -> list[dict[str, Any]]:
        if not self.db.db_available:
            return []
        scope_sql, scope_params = self._scope_sql("workspace_canonical", scope)
        if scope_sql:
            scope_sql = " AND " + scope_sql
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """SELECT id,kind,workspace_canonical,status,candidate_key_hash,
                              member_versions,evidence,reason,severity,source,detail
                       FROM scan_queue WHERE status='pending' AND kind='workspace'
                           AND id>? """ + scope_sql + " ORDER BY id LIMIT ?",
                    (int(after_id), *scope_params, ASSEMBLY_WINDOW),
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

    def submit(self, decisions: list[dict[str, Any]], caller=None) -> dict[str, Any]:
        if not isinstance(decisions, list) or not decisions:
            return {"ok": False, "error": "decisions must be a non-empty list"}
        if len(decisions) > 200:
            return {"ok": False, "error": "at most 200 decisions per submission"}
        self._caller = caller
        results: list[dict[str, Any]] = []
        for index, raw in enumerate(decisions):
            results.append(self._submit_one(index, raw))
        handled = {"confirmed", "dismissed", "resolved", "skipped", "moved",
                   "protected_bucket_hint", "multi_family_hint",
                   # Terminal-for-this-decision states: the server did the
                   # right thing (expired + requeue, or the row was already
                   # decided) — the submission envelope stays green so an
                   # idempotent retry is not punished.
                   "stale_snapshot", "already_terminal", "workspace_mismatch",
                   "not_found"}
        ok = all(item.get("outcome") in handled for item in results)
        return {
            "ok": ok,
            "results": results,
            # 0.16.4 live-judgment review: same backlog semantics as page()
            # (scan_queue rows + internal pending) — a mixed submission that
            # just cleared internal rows must not report an unchanged number.
            "queue_backlog": self.db.scan_queue_backlog() + len(
                self.db.internal_conflicts.list_pending(limit=10**6)
            ),
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
            if not isinstance(internal_id, int) or isinstance(internal_id, bool) or internal_id <= 0:
                return {"index": index, "outcome": "invalid_input",
                        "error": "internal decisions need a positive integer internal_id"}
            if status not in {"dismissed", "resolved"}:
                return {"index": index, "outcome": "invalid_input",
                        "error": "internal decisions accept status dismissed|resolved"}
            # Same fail-closed visibility rule as the memory-level channel:
            # no dispositions on rows whose memory the caller cannot read.
            row_memory = self.db.internal_conflicts.memory_id_of(int(internal_id))
            if row_memory is None or not self._member_visible(row_memory):
                return {"index": index, "outcome": "not_found",
                        "error": "internal row not visible to this caller"}
            outcome = self.db.internal_conflicts.decide(
                int(internal_id), status, reason=reason,
            )
            if outcome.get("outcome") == "updated":
                outcome = {"outcome": status, **{
                    key: value for key, value in outcome.items() if key != "outcome"
                }}
            return {"index": index, "kind": "internal", **outcome}
        if kind == "internal_memory":
            # 0.16.4 §3: one disposition clears a memory's whole pending set.
            memory_id = raw.get("memory_id")
            if not isinstance(memory_id, int) or isinstance(memory_id, bool) or memory_id <= 0:
                return {"index": index, "outcome": "invalid_input",
                        "error": "internal_memory decisions need a positive integer memory_id"}
            if status not in {"dismissed", "resolved"}:
                return {"index": index, "outcome": "invalid_input",
                        "error": "internal_memory decisions accept status dismissed|resolved"}
            if not self._member_visible(memory_id):
                # Same fail-closed visibility rule as the per-row channel.
                return {"index": index, "outcome": "not_found", "memory_id": memory_id}
            pair_count = self.db.internal_conflicts.pending_pair_count(memory_id)
            if pair_count == 0:
                # No current-version pending rows: already judged (idempotent
                # green not_found) or version-drifted (stale_snapshot) —
                # decide_memory tells them apart.
                outcome = self.db.internal_conflicts.decide_memory(memory_id, status, reason=reason)
                return {"index": index, "kind": "internal_memory", "memory_id": memory_id,
                        "pair_count": 0, **outcome}
            if (
                status == "dismissed"
                and pair_count > INTERNAL_PAIRS_CAP
                and not _decision_truthy(raw.get("expanded"))
            ):
                # 0.16.4 review P1: the preview showed only INTERNAL_PAIRS_CAP
                # pairs — dismissing the whole memory sight-unseen is exactly
                # the blind-judge hole 72% of the live stock sits behind.
                # expanded=true is the explicit declaration that every pair
                # was read (batch_read hits / per-row channel).
                return {
                    "index": index, "outcome": "invalid_input", "memory_id": memory_id,
                    "error": (
                        f"pair_count={pair_count} exceeds the preview cap "
                        f"({INTERNAL_PAIRS_CAP}): read every pair first "
                        "(batch_read content_mode='hits' on the pairs' spans, or the "
                        "per-internal_id channel), then resubmit with expanded=true"
                    ),
                }
            outcome = self.db.internal_conflicts.decide_memory(
                memory_id, status, reason=reason,
            )
            return {"index": index, "kind": "internal_memory", "memory_id": memory_id,
                    "pair_count": pair_count, **outcome}
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
            NORMALIZE_MIN_CONF, NORMALIZE_VOTE_NEIGHBORS,
            PROTECTED_WORKSPACES,
        )
        from .normalize_gate import normalize_gate
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
            self._expire_workspace_rows(memory_id, "dismissed", reason)
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
            self._expire_workspace_rows(memory_id, "dismissed",
                                        f"protected bucket involved: {current!r}->{target!r}")
            return {
                "index": index, "outcome": "protected_bucket_hint",
                "memory_id": memory_id, "current": current, "target": target,
                "hint": "受保护桶不自动搬——请向用户提示疑似写错桶，由用户自行处置",
            }
        if target == current:
            return {"index": index, "outcome": "invalid_input",
                    "error": "target_workspace equals the current bucket; nothing to move"}
        if conf < NORMALIZE_MIN_CONF:
            return {"index": index, "outcome": "gate_failed", "gate": "conf",
                    "memory_id": memory_id, "conf": conf}
        # 0.16.3 default fallback (owner rule): an agent that genuinely
        # cannot find a suitable bucket may confirm the suspect BACK into
        # the global default pool — under strict isolation default is the
        # only bucket outside the caller's own that still participates in
        # recall. Explicit declaration required, the vector-vote gate is
        # waived by definition (an unreliable vote IS the "no suitable
        # bucket" finding), conf >= 0.8 still applies, and the audit trail
        # plus response hint keep it user-visible.
        fallback = _decision_truthy(raw.get("fallback"))
        if fallback and not is_default_workspace_term(target):
            return {"index": index, "outcome": "invalid_input",
                    "error": "fallback=true is only valid with target_workspace=default"}
        if fallback:
            # Fold accepted synonyms (默认/none/…) onto the canonical name —
            # a raw "默认" would otherwise land the memory in a phantom
            # bucket, split off from the real default pool in recall
            # scoping and pairing (same defence as the memory_govern path).
            from .constants import DEFAULT_WORKSPACE_NAME

            target = DEFAULT_WORKSPACE_NAME
        if fallback and not str(reason or "").strip():
            return {"index": index, "outcome": "invalid_input",
                    "error": "fallback=true requires a reason (why no suitable bucket exists)"}
        multi = self._multi_family_mentions(record, target)
        if multi:
            # E7-4: multi-family mentions downgrade to a user hint, no move.
            self._expire_workspace_rows(memory_id, "dismissed",
                                        f"multi-family mention: {multi}")
            return {"index": index, "outcome": "multi_family_hint",
                    "memory_id": memory_id, "families": multi}
        if fallback:
            gate_evidence = {"default_fallback": True, "reason": reason}
        else:
            vote = self._workspace_vote(memory_id)
            if vote is None:
                return {"index": index, "outcome": "gate_failed", "gate": "vote_unavailable",
                        "memory_id": memory_id}
            votes, own_bucket, neighbours = vote
            passed, gate_evidence = normalize_gate(votes, own_bucket)
            if gate_evidence["top_bucket"] != target or not passed:
                return {
                    "index": index, "outcome": "gate_failed", "gate": "vote",
                    "memory_id": memory_id, "top_bucket": gate_evidence["top_bucket"],
                    "share": gate_evidence["top_votes"], "neighbours": neighbours,
                }
        moved, warnings = self._execute_auto_move(memory_id, current, target, gate_evidence, conf)
        if not moved:
            return {"index": index, "outcome": "move_failed", "warnings": warnings}
        self._expire_workspace_rows(memory_id, "confirmed", reason)
        result = {
            "index": index, "outcome": "moved", "memory_id": memory_id,
            "from": current, "to": target,
            "vote": {"top": gate_evidence.get("top_bucket", "default"),
                     "share": gate_evidence.get("top_votes", "fallback")},
            "warnings": warnings,
        }
        if fallback:
            result["default_fallback"] = {
                "note": (
                    "Parked in the default pool (no suitable bucket found) — "
                    "tell the user: re-home via memory_govern(action="
                    "'move_memories_workspace') when a bucket is decided."
                ),
                "reason": reason,
            }
        return result

    def _expire_workspace_rows(self, memory_id: int, status: str, why: str) -> None:
        now = utc_now_iso()
        try:
            with self.db.write_transaction() as conn:
                conn.execute(
                    """UPDATE scan_queue SET status=?, decided_reason=?, decided_at=?, updated_at=?
                       WHERE kind='workspace' AND status='pending'
                         AND EXISTS(SELECT 1 FROM json_each(scan_queue.member_versions) AS m
                                    WHERE CAST(json_extract(m.value,'$.memory_id') AS INTEGER)=?)""",
                    (status, why, now, now, int(memory_id)),
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

    def _workspace_vote(self, memory_id: int) -> "tuple[dict[str, int], str, int] | None":
        """Decision-time vector vote (E7①: 现算票). Returns (votes, own_bucket,
        neighbours_checked); None without vectors/numpy. Judged by the shared
        normalize_gate at the call site — this method only counts votes."""
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
        # The own bucket stays in the dict: normalize_gate excludes it before
        # judging (a top that is the current bucket means the content peers
        # agree with the placement and the gate must fail).
        return votes, own, k

    def _execute_auto_move(
        self, memory_id: int, current: str, target: str,
        gate_evidence: dict[str, Any], conf: float,
    ) -> "tuple[bool, list[str]]":
        try:
            with self.db.write_transaction() as conn:
                moved, move_warnings = self.db.workspaces.move_memory_workspace_on_conn(
                    conn, memory_id, target,
                    allow_default=bool(gate_evidence.get("default_fallback")),
                )
                if not moved:
                    return False, move_warnings
                cur = conn.execute(
                    """INSERT INTO normalize_audit(
                         memory_id, from_workspace, to_workspace, gate, status, created_at)
                       VALUES(?,?,?,?, 'applied', ?)""",
                    (int(memory_id), current, target,
                     json.dumps({**gate_evidence, "conf": conf}, ensure_ascii=False),
                     utc_now_iso()),
                )
            return True, move_warnings
        except Exception as exc:
            return False, [f"auto move failed: {exc}"]

    def _fetch_all_conflict_rows(self, scope=None) -> list[dict[str, Any]]:
        """Every pending conflict row, paged past the ASSEMBLY_WINDOW.

        Group-level dispositions must re-assemble the COMPLETE closure even
        on libraries deeper than one fetch window — a token-only submit on a
        63-pair group in a large library must not strand the tail. Defensive
        ceiling keeps a pathological queue bounded.
        """
        rows: list[dict[str, Any]] = []
        cursor = 0
        ceiling = 50  # 50 × ASSEMBLY_WINDOW(400) = 20k rows defensive cap
        for _ in range(ceiling):
            batch = self._fetch_conflict_rows(cursor, scope)
            rows.extend(batch)
            if len(batch) < ASSEMBLY_WINDOW:
                return rows
            cursor = int(batch[-1]["id"])
        return rows

    def _submit_group(
        self, index: int, group_token: str, status: str, reason: str,
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        """Group-level dismissal (§6⑥): one entry suppresses every pair of
        the group. Confirms stay per-pair (each pair's slot/values differ)."""
        if status != "dismissed":
            return {"index": index, "outcome": "invalid_input",
                    "error": "group decisions accept status=dismissed only; confirm per pair"}
        explicit = raw.get("pair_hashes")
        targets: list[dict[str, Any]] = []
        if isinstance(explicit, list) and explicit:
            # Primary path: resolve by the caller-supplied pair hashes DIRECTLY
            # (no id window, no re-assembly) — per-pair decisions earlier in
            # the same batch or depth beyond the assembly window cannot
            # strand the remaining pairs (adversarial review #5).
            wanted_hashes = [str(h) for h in explicit]
            placeholders = ",".join("?" for _ in wanted_hashes)
            rows_by_hash: dict[str, dict[str, Any]] = {}
            try:
                with self.db.connection() as conn:
                    db_rows = conn.execute(
                        f"""SELECT id,kind,workspace_canonical,status,candidate_key_hash,
                                   member_versions,evidence,reason,severity,source,detail
                            FROM scan_queue WHERE candidate_key_hash IN ({placeholders})""",
                        tuple(wanted_hashes),
                    ).fetchall()
                for db_row in db_rows:
                    item = dict(db_row)
                    for key in ("member_versions", "evidence", "detail"):
                        if isinstance(item.get(key), str):
                            try:
                                item[key] = json.loads(item[key])
                            except (TypeError, json.JSONDecodeError):
                                item[key] = None
                    rows_by_hash[str(item["candidate_key_hash"])] = item
            except Exception:
                rows_by_hash = {}
            targets = [rows_by_hash[h] for h in wanted_hashes if h in rows_by_hash]
            missing = len(wanted_hashes) - len(targets)
            if missing:
                # Already terminal (decided earlier in this batch or a prior
                # run): dismissing what remains lands the same outcome.
                reason = f"{reason} (+{missing} already terminal)"
        # 0.16.4 live-judgment review: the page now CAPS the displayed
        # pair_hashes (byte budget — full hash lists dominated real pages),
        # so a caller may legitimately hold only a subset of the group. The
        # token re-assembly below runs in ADDITION to the explicit hashes
        # and merges whatever same-group pending rows it finds — a partial
        # hash list can never strand the rest of the group. If assembly
        # cannot find the token (closure drift from same-batch per-pair
        # decisions, or depth), the explicit hashes remain the fallback.
        wanted = group_token.split(":", 1)[-1]
        seen_hashes = {str(row["candidate_key_hash"]) for row in targets}
        for component in self._assemble_groups(self._fetch_all_conflict_rows()):
            if self._component_token(component["pairs"]) == wanted:
                for row in component["pairs"]:
                    if str(row["candidate_key_hash"]) not in seen_hashes:
                        targets.append(row)
                        seen_hashes.add(str(row["candidate_key_hash"]))
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
        probe = dict(row)
        probe["member_versions"] = row["member_versions"] or []
        if not self._row_visible(probe):
            # Fail closed under strict isolation: no dispositions on rows the
            # caller cannot fully read.
            return {"index": index, "outcome": "not_found",
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
