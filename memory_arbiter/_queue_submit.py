"""queue 提交侧 mixin：submit/判定/执行/过期全族（从 queue_protocol.py 搬出，拆分批 ⑦ 纯移动）。caller 参数链不改（P2#15 契约）。"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, TYPE_CHECKING

from .constants import is_default_workspace_term
from ._queue_consts import INTERNAL_PAIRS_CAP, _decision_truthy
from .models import utc_now_iso

if TYPE_CHECKING:
    from .config import Settings
    from .db import MemoryDB
    from .tools import MemoryTools


class _QueueSubmit:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"
        settings: "Settings"
        def _member_visible(self, *args: Any, **kwargs: Any) -> bool: ...
        def _row_visible(self, *args: Any, **kwargs: Any) -> bool: ...
        def _member_meta(self, *args: Any, **kwargs: Any) -> Any: ...
        def _fetch_conflict_rows(self, *args: Any, **kwargs: Any) -> "list[dict[str, Any]]": ...
        def _fetch_all_conflict_rows(self, *args: Any, **kwargs: Any) -> "list[dict[str, Any]]": ...
        def _fetch_workspace_rows(self, *args: Any, **kwargs: Any) -> "list[dict[str, Any]]": ...
        def _scope_sql(self, *args: Any, **kwargs: Any) -> Any: ...
        def _assemble_groups(self, *args: Any, **kwargs: Any) -> "list[dict[str, Any]]": ...
        def _component_token(self, *args: Any, **kwargs: Any) -> str: ...
        def _detector_version(self) -> str: ...
        def _pair_item(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...
        def _group_item(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...
        def _internal_memory_item(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...
        def _workspace_item(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...

    def submit(self, decisions: list[dict[str, Any]], caller: Any = None) -> dict[str, Any]:
        if not isinstance(decisions, list) or not decisions:
            return {"ok": False, "error": "decisions must be a non-empty list"}
        if len(decisions) > 200:
            return {"ok": False, "error": "at most 200 decisions per submission"}
        results: list[dict[str, Any]] = []
        for index, raw in enumerate(decisions):
            results.append(self._submit_one(index, raw, caller))
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
            # 0.16.4 live-judgment review: backlog semantics — a mixed
            # submission that just cleared internal rows must not report an
            # unchanged number. Global caliber (P2 #17 scoped the PAGE's
            # numbers; submit's caller mix makes a per-caller count here
            # ambiguous, so it stays library-wide).
            "queue_backlog": self.db.scan_queue_backlog() + len(
                self.db.internal_conflicts.list_pending(limit=10**6)
            ),
        }

    def _submit_one(self, index: int, raw: Any, caller: Any = None) -> dict[str, Any]:
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
            if row_memory is None or not self._member_visible(row_memory, caller):
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
            if not self._member_visible(memory_id, caller):
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
            return self._submit_group(index, str(group_token), status, reason, raw, caller=caller)
        candidate_hash = str(raw.get("candidate_key_hash") or "")
        if len(candidate_hash) != 64:
            return {"index": index, "outcome": "invalid_input",
                    "error": "candidate_key_hash must be the 64-char hash from the queue page"}
        return self._decide_row(index, candidate_hash, status, reason, raw, caller=caller)

    def _submit_workspace(
        self, index: int, status: str, reason: str, raw: dict[str, Any],
    ) -> dict[str, Any]:
        from .constants import NORMALIZE_MIN_CONF, PROTECTED_WORKSPACES
        from .normalize_gate import normalize_gate

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
            self._expire_workspace_rows(memory_id, "dismissed", reason, durable_record=True)
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

    def _expire_workspace_rows(
        self, memory_id: int, status: str, why: str, *, durable_record: bool = False,
    ) -> None:
        """Settle ALL pending kind='workspace' rows of one memory (decision APIs
        judge per memory, not per bucket). With ``durable_record`` (agent
        explicit dismiss only) each row's (version, suspected) identity lands
        in ``workspace_dismissals`` in the SAME transaction — the queue row
        and its durable suppression flip together or not at all. Hint
        branches (protected/multi_family) and confirmed pass durable_record=
        False: they only flip queue status, never mint a durable exemption.

        Two hardening degrades (R2 review):
        - workspace_dismissals missing (additive skipped on a read-only/
          damaged DB, boot degrades to a warning): the combined transaction
          would roll the queue flip back — the agent decision MUST land, so
          the except path retries the legacy pure-UPDATE (pre-C2 semantics);
          the durable exemption is lost, one more dismiss re-mints it.
        - The durable identity pins the CURRENT memory version, not the row's
          pinned one: an edit-before-dismiss sequence must not re-open the
          exact noise the agent is settling (reopen is the edit's job when it
          comes after the dismiss)."""
        now = utc_now_iso()
        current_version: int | None = None
        if status == "dismissed" and durable_record:
            record = self.db.get_memory(int(memory_id))
            if record is not None:
                current_version = int(record.get("version") or 1)
        try:
            with self.db.write_transaction() as conn:
                rows = conn.execute(
                    """SELECT id, member_versions, detail FROM scan_queue
                       WHERE kind='workspace' AND status='pending'
                         AND EXISTS(SELECT 1 FROM json_each(scan_queue.member_versions) AS m
                                    WHERE CAST(json_extract(m.value,'$.memory_id') AS INTEGER)=?)""",
                    (int(memory_id),),
                ).fetchall()
                if not rows:
                    return
                durable: list[tuple[int, int, str, str]] = []
                row_ids: list[int] = []
                for row in rows:
                    row_ids.append(int(row["id"]))
                    if status != "dismissed" or not durable_record:
                        continue  # hint/confirmed 分支只翻状态，不留持久 record
                    try:
                        # 钉当前版本：行钉可能是 enqueue 之后的旧版本（edit 先于
                        # dismiss），按行钉落表会让刚处置的噪音立即复发。
                        version = current_version
                        if version is None:
                            version = int(json.loads(str(row["member_versions"] or "[]"))[0]["version"])
                        suspected = str(json.loads(str(row["detail"] or "{}"))["suspected_workspace"])
                    except (IndexError, KeyError, TypeError, ValueError, OverflowError,
                            json.JSONDecodeError):
                        continue  # envelope 损坏：行照常了结，只是无法留持久豁免
                    durable.append((int(memory_id), version, suspected, why))
                if durable:
                    self.db.scan_queue.record_workspace_dismissals_on_conn(conn, durable)
                conn.executemany(
                    """UPDATE scan_queue SET status=?, decided_reason=?, decided_at=?, updated_at=?
                       WHERE id=? AND status='pending'""",  # CAS（仓库 B10 纪律）
                    [(status, why, now, now, rid) for rid in row_ids],
                )
        except sqlite3.OperationalError as exc:
            # F1 (R2)：durable 表缺失时 INSERT 连坐队列翻转（假绿 dismissed +
            # 行永 pending + §九永久卡死）——队列翻转必须落地，回退旧路径。
            # 0.17.0 review R2：降级只针对「workspace_dismissals 表缺失」这
            # 一设计内形态；写锁等瞬时错误曾被同一裸 except 吞进降级（豁免
            # 静默丢失 + 假绿 dismissed）——其余照抛，counted never silent。
            if "no such table" in str(exc) and "workspace_dismissals" in str(exc):
                self._expire_workspace_rows_legacy_fallback(memory_id, status, why, now)
            else:
                raise

    def _expire_workspace_rows_legacy_fallback(
        self, memory_id: int, status: str, why: str, now: str,
    ) -> None:
        """Pre-C2 pure-UPDATE path — guarantees the queue flip lands whenever the
        old code would have landed it (workspace_dismissals absent/degraded)."""
        try:
            with self.db.write_transaction() as conn:
                conn.execute(
                    """UPDATE scan_queue SET status=?, decided_reason=?, decided_at=?, updated_at=?
                       WHERE kind='workspace' AND status='pending'
                         AND EXISTS(SELECT 1 FROM json_each(scan_queue.member_versions) AS m
                                    WHERE CAST(json_extract(m.value,'$.memory_id') AS INTEGER)=?)""",
                    (status, why, now, now, int(memory_id)),
                )
        except sqlite3.Error:
            pass  # 库不可写（只读/损坏）：行留 pending，agent 可再处置

    def _multi_family_mentions(self, record: dict[str, Any], target: str) -> list[str]:
        """E7-4: subject/tags mentioning >=2 registered project families
        downgrades the case to a user hint (cross-project meta content)."""
        raw_tags = record.get("tags")
        tags: list[Any] = raw_tags if isinstance(raw_tags, list) else []
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
        normalize_gate at the call site — this method only counts votes, and
        the counting itself is the shared compute_summary_votes so the
        decision-time tally can never drift from the generation-time one."""
        from .normalize_gate import compute_summary_votes

        vectors = self.db.memories.all_summary_vectors()
        # path="single": decision-time votes were counted by gemv before the
        # extraction; keep the exact old formula (see scan_pipeline's caller
        # note on gemv/gemm tie stability).
        vote = compute_summary_votes(vectors, [memory_id], path="single").get(memory_id)
        if vote is None:
            return None
        return vote["votes"], vote["own"], vote["k"]

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

    def _submit_group(
        self, index: int, group_token: str, status: str, reason: str,
        raw: dict[str, Any], *, caller: Any = None,
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
            self._decide_row(f"{index}.{i}", row["candidate_key_hash"], "dismissed", reason, {}, caller=caller)
            for i, row in enumerate(targets)
        ]
        return {
            "index": index, "outcome": "dismissed", "group_token": group_token,
            "pairs": results,
        }

    def _decide_row(
        self, index: Any, candidate_hash: str, status: str, reason: str,
        raw: dict[str, Any], *, caller: Any = None,
    ) -> dict[str, Any]:
        row = self._queue_row(candidate_hash)
        if row is None:
            return {"index": index, "outcome": "not_found",
                    "candidate_key_hash": candidate_hash}
        if row["status"] != "pending":
            return {"index": index, "outcome": "already_terminal", "status": row["status"],
                    "candidate_key_hash": candidate_hash}
        probe = dict(row)
        probe["member_versions"] = row["member_versions"] or []
        if not self._row_visible(probe, caller):
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
            member_display = ref_to_value.get(ref)
            if member_display is None:
                return None, f"member {ref} is not covered by any value_group"
            item["attribute_raw"] = str(slot_key.get("attribute") or "")
            item["normalized_attribute"] = str(slot_key.get("attribute") or "")
            item["value_raw"] = member_display
            item["normalized_value"] = normalize_value(member_display)
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
                    "decided_at=?, updated_at=? WHERE candidate_key_hash=? AND status='pending'",
                    (status, str(decided_ref) if decided_ref is not None else None,
                     reason, now, now, candidate_hash),
                )
        except Exception:
            pass
        # The status='pending' term is a CAS guard (0.16.6 audit B10): the
        # pre-check at the submit entry ran outside this transaction, so a
        # concurrent expire/void can land in between — last-writer-wins would
        # silently overwrite that terminal state. rowcount 0 = lost race,
        # the conflicts-side record remains the source of truth either way.
        # It does NOT deduplicate record_conflict_group calls; that is the
        # conflicts table's own uniqueness bidding.

    def _expire_row(self, candidate_hash: str, why: str) -> None:
        now = utc_now_iso()
        try:
            with self.db.write_transaction() as conn:
                conn.execute(
                    "UPDATE scan_queue SET status='expired', decided_reason=?, decided_at=?, "
                    "updated_at=? WHERE candidate_key_hash=? AND status='pending'",
                    (why, now, now, candidate_hash),
                )
        except Exception:
            pass
        # Same CAS term as _mark_decided: expiring an already-decided row
        # would only swap one terminal state for another — skip instead.
