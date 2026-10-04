"""scan 枚举域 mixin（从 scan_pipeline.py 搬出，拆分批 ⑤ 纯移动）。

_lightweight_scan_candidate(s)/scan_rule_candidates/memory_scan_workspace_anomalies；
surfaces 直调 scan_rule_candidates 经 mixin 方法名不变保活。
"""
from __future__ import annotations

import itertools
import json
from typing import Any, TYPE_CHECKING

from .acl import scope_names, workspace_scope_sql
from .scan_admission import _candidate_pair_member
from .semantic_conflict import decide_evidence, is_cross_evolution
from .normalize_gate import compute_summary_votes, normalize_gate

if TYPE_CHECKING:
    from .acl import WorkspaceScope
    from .db import MemoryDB
    from .tools import MemoryTools


class _ScanEnumerate:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"
        QUOTE_LIGHT_CHARS: int

    def _lightweight_scan_candidate(self, item: dict[str, Any]) -> dict[str, Any]:
        """C1 lightweight projection of one scan candidate for the default page.

        The full candidate payload (full quotes/spans/members/slot payloads)
        was calibrated for batch=2 reads and explodes the response at the
        spec's batch sizes (12MB pages). The default page keeps only the
        triage identity — pair ids, workspace, reasons, route/state and a
        short quote per side — while include_quotes=true restores the full
        envelope (whose members/slot_key/value_groups record_conflict needs).
        The full payload is computed first and projected last so enhancement
        order and suppression counting are unaffected.
        """
        members = item.get("members")
        if not isinstance(members, list):
            members = []

        def member_quote(index: int) -> str:
            if 0 <= index < len(members):
                quote = str((members[index] or {}).get("evidence_quote") or "")
                if quote:
                    return quote[:self.QUOTE_LIGHT_CHARS]
            return str(item.get("left_snippet") or item.get("right_snippet") or "")[:self.QUOTE_LIGHT_CHARS]

        workspace = item.get("workspace")
        if not workspace and members:
            left_mem = self.db.get_memory(int((members[0] or {}).get("memory_id") or 0))
            if left_mem:
                workspace = (
                    left_mem.get("workspace_canonical")
                    or left_mem.get("workspace")
                )
        light: dict[str, Any] = {
            "left_id": item.get("left_id"),
            "right_id": item.get("right_id"),
            "workspace": workspace,
            "state": item.get("state"),
            "route": item.get("route"),
            "reasons": list(item.get("reasons") or []),
            "distance": item.get("distance"),
            "left_quote": member_quote(0),
            "right_quote": member_quote(1),
        }
        # 0.17.1: judged notices carry model_signal; legacy qwen_signal rows
        # keep rendering through the old shape (one-release compat read).
        model_signal = item.get("model_signal") if isinstance(item.get("model_signal"), dict) else None
        if model_signal:
            light["model_signal"] = {
                key: model_signal.get(key) for key in ("label", "probs", "mechanism", "model_version")
            }
        qwen_signal = item.get("qwen_signal") if isinstance(item.get("qwen_signal"), dict) else None
        if qwen_signal:
            light["qwen_signal"] = {
                key: qwen_signal.get(key) for key in ("state", "reason", "prompt_version")
            }
        return light

    def _lightweight_scan_candidates(self, result: dict[str, Any]) -> dict[str, Any]:
        """Apply the C1 lightweight projection to a finished scan page.

        Candidates carry pair ids/workspace/state/reasons and a short quote
        per side; the full quotes/spans/members/value_groups envelope comes
        back only with include_quotes=true (record_conflict needs it).
        similarity_pool/duplicates_pool pairs get the same treatment via the
        shared per-item projection. slot_groups stay untouched: they are the
        grouping evidence for triage, not per-pair payload bloat.
        """
        for key in ("candidates", "similarity_pool", "duplicates_pool"):
            items = result.get(key)
            if isinstance(items, list):
                result[key] = [
                    (self._lightweight_scan_candidate(item) if isinstance(item, dict) else item)
                    for item in items
                ]
        return result

    def scan_rule_candidates(
        self,
        *,
        after_memory_id: int = 0,
        anchor_batch: int = 50,
        neighbor_k: int = 10,
        include_check: bool = False,
        max_distance: float | None = None,
        workspace: "WorkspaceScope" = None,
        similarity_pool_limit: int = 0,
        include_duplicates: bool = False,
        suspected_anomalies: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Enumerate conflict-candidate pairs for an external scan loop.

        Scheduled LLM review cannot load the whole library into a session,
        so the server enumerates the clues: for every active memory's
        current evidence units, KNN neighbours (rank-based, like the
        write-time notice path but with a wider window) pass through the
        deterministic decide_evidence rule. By default only rule-level
        notify routes (numeric/polarity/todo change) are returned —
        similarity-only check pairs are legion in topic-clustered
        libraries and are opt-in via include_check. Each pair carries the
        triggering unit snippets so the agent can triage without reading
        full memories. Pairs with an open conflict, or a version-pinned
        not_a_conflict dismissal, are filtered out.

        include_duplicates additionally exposes same-value near-duplicate
        pairs (ignore/equivalent_value|compatible_evidence) as a bounded
        duplicates_pool for governance merge; recorded pairs are suppressed
        with the same candidate-hash contract.

        C3b (0.15.13): pairing is workspace-grouped. Each anchor's KNN is
        scoped to the anchor's OWN bucket, so cross-bucket pairs are never
        generated (they cannot satisfy record_conflict's single-bucket
        group identity and used to loop weekly without landing).
        suspected_anomalies ({memory_id: suspected_bucket} from the active
        workspace_review notices) additionally sweeps each suspected
        misplaced memory against its SUSPECTED bucket: those hits are
        cross-bucket by construction and surface in a separate
        cross_bucket_references list for the running agent — authoritative
        disposition is the workspace_review notice (move), never
        record_conflict.

        Calibrated on a real 474-memory production copy: absolute vector
        distance has no discrimination there (random same-workspace pairs
        overlap notice pairs), so ranking + rules do the work and
        max_distance stays an optional extra gate.

        0.17.0 R2: this pairing orchestration moved here from the db layer
        (EvidenceStore) — rule/candidate policy belongs to the pipeline; the
        store keeps only the KNN/vector primitives. SQL and semantics are
        carried over verbatim.
        """
        db = self.db
        if not db.state.sqlite_vec_available:
            return {"error": "sqlite_vec_unavailable"}
        workspace_anchor_sql = ""
        anchor_params: list[Any] = []
        workspace_names = scope_names(workspace)
        echo_workspace = workspace_names[0] if workspace_names else None
        if workspace is not None:
            # Strict callers must not anchor on — or leak snippets from —
            # memories outside their admitted workspace set.
            anchor_scope_sql, anchor_scope_params = workspace_scope_sql(
                "COALESCE(NULLIF(workspace_canonical,''),workspace)", workspace,
            )
            if anchor_scope_sql:
                workspace_anchor_sql = f"AND {anchor_scope_sql} "
                anchor_params.extend(anchor_scope_params)
        with db.connection() as conn:
            anchors = [
                int(row["id"]) for row in conn.execute(
                    "SELECT id FROM memories WHERE status='active' AND id > ? "
                    + workspace_anchor_sql
                    + "ORDER BY id LIMIT ?",
                    (int(after_memory_id), *anchor_params, max(1, int(anchor_batch))),
                )
            ]
            if not anchors:
                return {
                    "anchors_scanned": 0, "next_anchor_memory_id": None,
                    "candidates": [], "counts": {"knn_pairs": 0, "rule_pass": 0,
                                                 "filtered_open": 0, "filtered_dismissed": 0,
                                                 "duplicates": 0},
                    "duplicates_pool": [], "duplicates_truncated": False,
                    "cross_bucket_references": [], "anchor_buckets": [],
                }
            # The group schema has no left/right columns. Suppression is tied
            # to the exact candidate snapshot successfully persisted by
            # record_conflict, not merely to a memory pair. That keeps an
            # unrecorded external review repeatable and allows changed member
            # versions/evidence to be reconsidered.
            recorded_candidate_statuses: dict[str, str] = {}
            active_group_members: list[frozenset[str]] = []
            dismissed_group_members: list[frozenset[str]] = []
            for row in conn.execute(
                "SELECT status,candidate_key_hash,member_versions FROM conflicts "
                "WHERE status IN ('open','applying','not_a_conflict')"
            ):
                candidate_hash = str(row["candidate_key_hash"] or "")
                status = str(row["status"])
                if candidate_hash:
                    recorded_candidate_statuses[candidate_hash] = status
                try:
                    members = json.loads(str(row["member_versions"] or "[]"))
                    refs = frozenset(
                        f"{int(member['memory_id'])}@{int(member['version'])}"
                        for member in members
                    )
                except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                    continue
                # C2 (0.15.13): a dismissed not_a_conflict pair now suppresses
                # by MEMORY-PAIR @version, not only by its exact evidence
                # snapshot. The same pair re-enumerated through different unit
                # slices (new hashes) used to resurface forever. Version
                # pinning stays: an edited memory lifts the suppression and
                # the pair is reconsidered. open/applying stay a separate
                # set so a pair with BOTH an open group and a dismissal
                # keeps the open group's precedence.
                if refs and status in {"open", "applying"}:
                    active_group_members.append(refs)
                elif refs and status == "not_a_conflict":
                    dismissed_group_members.append(refs)
            candidates: dict[tuple[int, int], dict[str, Any]] = {}
            # Spec §7.1 wide gate: similarity-only pairs dropped from the
            # default candidate set stay available as a bounded pool for the
            # caller's Qwen union instead of vanishing outright.
            similarity_pool: dict[tuple[int, int], dict[str, Any]] = {}
            # Near-duplicate (ignore/equivalent_value|compatible_evidence)
            # pairs, exposed for governance merge only when include_duplicates
            # is set. Same suppression contract as real candidates: pairs
            # already recorded (not_a_conflict/open/applying) are not
            # re-enumerated — the candidate_hash lookup runs inside the ignore
            # branch, ahead of the historical silent drop.
            duplicates_pool: dict[tuple[int, int], dict[str, Any]] = {}
            duplicates_truncated = False
            duplicates_cap = 2 * max(1, int(anchor_batch))
            pool_limit = max(0, int(similarity_pool_limit))
            knn_pair_count = 0
            stale_anchors = 0
            filtered_open = 0
            filtered_dismissed = 0
            cross_bucket_refs: dict[tuple[int, int], dict[str, Any]] = {}
            anchor_buckets: dict[str, dict[str, int]] = {}
            # C3b: suspected misplaced memories (from ACTIVE workspace_review
            # notices) are ALSO paired against their suspected bucket. Those
            # pairs are cross-bucket, cannot land in record_conflict, and go
            # to a reference list only.
            suspected = {
                int(mid): str(bucket)
                for mid, bucket in (suspected_anomalies or {}).items()
            }
            for anchor_id in anchors:
                anchor_row = conn.execute(
                    "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=?",
                    (anchor_id,),
                ).fetchone()
                # C3b grouping: pair only within the anchor's own bucket. The
                # strict caller scope (workspace) already bounds the whole
                # page; the anchor bucket narrows pairing further.
                anchor_bucket = (
                    str(anchor_row["workspace"] or "").strip() if anchor_row else ""
                )
                # C5 per-group page accounting: doctor's broken-chain alarm
                # needs to know which bucket the walk was last inside.
                if anchor_bucket:
                    entry = anchor_buckets.setdefault(
                        anchor_bucket, {"count": 0, "last_anchor": 0},
                    )
                    entry["count"] += 1
                    entry["last_anchor"] = max(entry["last_anchor"], anchor_id)
                # C2/C5: rows are the scan source — the job no longer
                # publishes unit vectors. rowseg emits no heading rows and
                # (pre-C3) no subject rows, so the old kind='text' filter's
                # intent (subjects/headings excluded) holds by construction.
                # C3 A+ guard: subject rows never ORIGINATE a scan pair
                # (subject version progression is timeline evolution, not a
                # numeric clue) — the row-channel counterpart of the unit
                # channel's kind='text' anchor filter.
                units = conn.execute(
                    """SELECT r.id AS eid, r.text AS text, v.embedding AS embedding,
                              r.memory_version AS memory_version, r.content_hash AS content_hash,
                              r.start_offset AS start_offset, r.end_offset AS end_offset
                       FROM memory_row r
                       JOIN memory_row_vec v ON v.id=r.id
                       WHERE r.memory_id=? AND r.memory_version=(
                           SELECT version FROM memories WHERE id=?)
                         AND r.kind != 'subject'
                       ORDER BY r.id""",
                    (anchor_id, anchor_id),
                )
                first_unit = units.fetchone()
                if first_unit is None:
                    # Async republish window or a permanently failed publish:
                    # surface it instead of silently skipping forever.
                    stale_anchors += 1
                anchor_content_row = conn.execute(
                    "SELECT content FROM memories WHERE id=?", (anchor_id,),
                ).fetchone()
                anchor_content = str(anchor_content_row["content"]) if anchor_content_row else ""
                peer_content_cache: dict[int, str] = {}

                def peer_content(peer_mid: int) -> str:
                    if peer_mid not in peer_content_cache:
                        row = conn.execute(
                            "SELECT content FROM memories WHERE id=?", (peer_mid,),
                        ).fetchone()
                        peer_content_cache[peer_mid] = str(row["content"]) if row else ""
                    return peer_content_cache[peer_mid]

                def locate_span(content: str, unit_text: str, hint_start: int, hint_end: int) -> dict[str, int] | None:
                    """Validate an exact evidence span and pad it for review.

                    Evidence pipeline v2 guarantees that cleaning the source
                    slice equals the unit text. Do not search for the text:
                    repeated phrases make search ambiguous and can silently
                    choose the wrong occurrence. A failed invariant drops the
                    span and falls back to a full read.
                    """
                    from .evidence import _clean

                    start, end = int(hint_start), int(hint_end)
                    if not (content and unit_text and 0 <= start < end <= len(content)):
                        return None
                    if _clean(content[start:end]) != unit_text:
                        return None
                    return {
                        "start": max(0, start - 128),
                        "end": min(len(content), end + 128),
                    }

                def _pool_near_duplicate(
                    peer: int, hit: "dict[str, Any]", anchor: int,
                    unit_row: "dict[str, Any]", text_a: str, reason: str,
                ) -> None:
                    """Gate-v2 G4: shared duplicates_pool admission for BOTH
                    near-duplicate sources — the deterministic ignore routes
                    and the at-ceil cosine pairs. Same suppression contract,
                    same free-dict-replace cap accounting."""
                    nonlocal duplicates_truncated
                    pair_key = (min(anchor, peer), max(anchor, peer))
                    member_refs, _key, candidate_hash = self.db.evidence._unit_pair_identity(
                        anchor, unit_row, peer, hit,
                    )
                    recorded = recorded_candidate_statuses.get(candidate_hash)
                    if recorded is not None or any(
                        member_refs <= group_members for group_members in active_group_members
                    ) or any(
                        member_refs <= group_members for group_members in dismissed_group_members
                    ):
                        return
                    hit_text = str(hit.get("text") or "")
                    # Re-hitting an already-pooled pair is a free dict
                    # replace, not pool growth — it must not count against
                    # the cap or flag truncation that never happened.
                    if pair_key in duplicates_pool or len(duplicates_pool) < duplicates_cap:
                        duplicates_pool[pair_key] = {
                            "left_id": pair_key[0], "right_id": pair_key[1],
                            "reason": reason,
                            "distance": float(hit.get("distance") or 0),
                            "candidate_key_hash": candidate_hash,
                            "left_snippet": text_a[:200] if pair_key[0] == anchor else hit_text[:200],
                            "right_snippet": hit_text[:200] if pair_key[1] == peer else text_a[:200],
                            "members": [
                                _candidate_pair_member(
                                    pair_key[0], is_anchor=(pair_key[0] == anchor),
                                    unit=unit_row, hit=hit,
                                    anchor_text=text_a, peer_text=hit_text,
                                ),
                                _candidate_pair_member(
                                    pair_key[1], is_anchor=(pair_key[1] == anchor),
                                    unit=unit_row, hit=hit,
                                    anchor_text=text_a, peer_text=hit_text,
                                ),
                            ],
                        }
                    else:
                        duplicates_truncated = True


                # C3b: same-bucket pairing. Under a strict caller scope the
                # anchor bucket must stay inside the admitted set.
                admitted_names = set(workspace_names) if workspace_names else set()
                pairing_scope = anchor_bucket if (
                    anchor_bucket and (workspace is None or anchor_bucket in admitted_names)
                ) else workspace
                suspect_bucket = suspected.get(anchor_id)
                from .pipeline.gates import candidate_cos_gate
                for unit in (() if first_unit is None else itertools.chain((first_unit,), units)):
                    text = str(unit["text"] or "")
                    if not text:
                        continue
                    unit_vector = self.db.evidence._blob_to_vector(bytes(unit["embedding"]))
                    hits = self.db.row_knn(
                        unit_vector,
                        k=max(1, int(neighbor_k)) + 1,
                        workspace=pairing_scope,
                        exclude_memory_id=anchor_id,
                        include_subject_rows=False,
                    )
                    # Gate-v2 G4 余弦门 (diagnostic-channel leg): below-floor
                    # pairs are noise; AT/ABOVE-ceil pairs are near-duplicates
                    # — they route into duplicates_pool below instead of being
                    # dropped (that pool IS their governance consumer). The
                    # suspect-bucket sweep stays OUTSIDE the gate: those hits
                    # are cross-bucket references, not conflict candidates.
                    gate_vectors = self.db.evidence.row_vectors_for_ids(
                        [int(hit["id"]) for hit in hits], conn=conn,
                    )
                    _passed, _below, at_ceil_pairs = candidate_cos_gate(unit_vector, hits, gate_vectors)
                    hits = [hit for hit, _cos in _passed]
                    if include_duplicates:
                        for dup_hit, _dup_cos in at_ceil_pairs:
                            _pool_near_duplicate(
                                int(dup_hit["memory_id"]), dup_hit, anchor_id, unit,
                                text, "near_duplicate_cosine",
                            )
                    # C3b: the suspected-bucket sweep for misplaced memories.
                    suspect_hits: list[dict[str, Any]] = []
                    if suspect_bucket and suspect_bucket != anchor_bucket:
                        suspect_hits = self.db.row_knn(
                            unit_vector,
                            k=max(1, int(neighbor_k)) + 1, include_subject_rows=False,
                            workspace=suspect_bucket,
                            exclude_memory_id=anchor_id,
                        )
                    for hit in itertools.chain(hits, suspect_hits):
                        # C2/C5: rows carry no 'text' kind — the old unit-
                        # channel filter's intent (exclude subject/heading
                        # units) holds by construction (rowseg emits neither).
                        peer_id = int(hit["memory_id"])
                        if peer_id == anchor_id:
                            continue
                        peer_bucket = str(hit.get("workspace_canonical") or hit.get("workspace") or "").strip()
                        if peer_bucket and anchor_bucket and peer_bucket != anchor_bucket:
                            # C3b: cross-bucket hits exist only in the
                            # suspected-bucket sweep (regular pairing is
                            # bucket-scoped). Reference-only: they can never
                            # satisfy record_conflict's single-bucket group
                            # identity. Authority is the workspace_review
                            # notice (move), never a conflict group.
                            pair_key = (min(anchor_id, peer_id), max(anchor_id, peer_id))
                            decision = decide_evidence(text, str(hit.get("text") or ""))
                            ref = cross_bucket_refs.setdefault(pair_key, {
                                "left_id": pair_key[0], "right_id": pair_key[1],
                                "workspace": anchor_bucket,
                                "suspected_workspace": suspect_bucket,
                                "reasons": set(), "distance": float(hit.get("distance") or 0),
                                "left_snippet": text[:200], "right_snippet": str(hit.get("text") or "")[:200],
                                "note": "cross-bucket reference only; disposition via the workspace_review notice (move), not record_conflict",
                            })
                            ref["reasons"].add(decision.reason)
                            ref["distance"] = min(ref["distance"], float(hit.get("distance") or 0))
                            continue
                        knn_pair_count += 1
                        # Every unit pair is judged: an earlier equivalent
                        # match (e.g. identical subjects) must not blacklist
                        # the peer, or a later numeric-change unit on the
                        # same pair would be lost.
                        decision = decide_evidence(text, str(hit.get("text") or ""))
                        # 0.16.4 §1/§0.5: the diagnostic channel routes
                        # through the SAME shared predicate as the scan
                        # pipeline and the write-time KNN loop — evolution-
                        # domain pairs must not surface here either, or the
                        # retroactive void's "re-enqueue and surface the gap"
                        # design would leak them back as notice_ready.
                        if is_cross_evolution(decision):
                            continue
                        if decision.action == "ignore":
                            if include_duplicates and decision.reason in {"equivalent_value", "compatible_evidence"}:
                                _pool_near_duplicate(
                                    peer_id, hit, anchor_id, unit, text, decision.reason,
                                )
                            continue
                        # Numeric deltas remain a deterministic scan baseline
                        # candidate even though they can no longer directly
                        # produce a write-time notice.
                        similarity_only = (
                            decision.action == "check"
                            and decision.reason != "numeric_value_candidate"
                            and not include_check
                        )
                        if similarity_only and pool_limit <= 0:
                            continue
                        if max_distance is not None and float(hit.get("distance") or 0) > float(max_distance):
                            continue
                        pair = (min(anchor_id, peer_id), max(anchor_id, peer_id))
                        distance = float(hit.get("distance") or 0)
                        store = similarity_pool if similarity_only else candidates
                        existing = store.get(pair)
                        hit_text = str(hit.get("text") or "")
                        anchor_span = locate_span(
                            anchor_content, text,
                            int(unit["start_offset"] or 0), int(unit["end_offset"] or 0),
                        )
                        peer_span = locate_span(
                            peer_content(peer_id), hit_text,
                            int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0),
                        )
                        member_refs, candidate_key, candidate_hash = self.db.evidence._unit_pair_identity(
                            anchor_id, unit, peer_id, hit,
                        )
                        recorded_status = recorded_candidate_statuses.get(candidate_hash)
                        if recorded_status is None and any(
                            member_refs <= group_members for group_members in active_group_members
                        ):
                            recorded_status = "open"
                        if recorded_status is None and any(
                            member_refs <= group_members for group_members in dismissed_group_members
                        ):
                            # C2: pair@version dismissal. Counted as dismissed,
                            # never silently folded into filtered_open — the
                            # counter is the convergence observability face.
                            recorded_status = "not_a_conflict"
                        if recorded_status is not None:
                            if recorded_status == "not_a_conflict":
                                filtered_dismissed += 1
                            else:
                                filtered_open += 1
                            continue
                        if existing is None:
                            state = "notice_ready" if decision.action == "notify" else "review_candidate"
                            store[pair] = {
                                "left_id": pair[0], "right_id": pair[1],
                                "state": state, "route": state,
                                "reasons": {decision.reason}, "distance": distance,
                                "candidate_key": candidate_key,
                                "candidate_key_hash": candidate_hash,
                                "members": [
                                    _candidate_pair_member(
                                        pair[0], is_anchor=(pair[0] == anchor_id),
                                        unit=unit, hit=hit,
                                        anchor_text=text, peer_text=hit_text,
                                    ),
                                    _candidate_pair_member(
                                        pair[1], is_anchor=(pair[1] == anchor_id),
                                        unit=unit, hit=hit,
                                        anchor_text=text, peer_text=hit_text,
                                    ),
                                ],
                                "value_groups": [], "slot_key": None, "slot_provenance": None,
                                "left_snippet": text[:200] if pair[0] == anchor_id else hit_text[:200],
                                "right_snippet": hit_text[:200] if pair[0] == anchor_id else text[:200],
                                # Pre-built deep-read calls: reading just the
                                # triggering region (plus context) instead of
                                # the full text keeps triage token cost low.
                                "deep_read": {
                                    "left": {
                                        "memory_id": pair[0],
                                        "span": anchor_span if pair[0] == anchor_id else peer_span,
                                        **({"workspace": echo_workspace} if echo_workspace else {}),
                                    },
                                    "right": {
                                        "memory_id": pair[1],
                                        "span": peer_span if pair[0] == anchor_id else anchor_span,
                                        **({"workspace": echo_workspace} if echo_workspace else {}),
                                    },
                                },
                            }
                        else:
                            # notice_ready outranks review_candidate when
                            # different unit pairs on the same memory pair
                            # disagree. Snippets and spans track the strongest
                            # signal, not the first discovery.
                            numeric_upgrade = (
                                decision.reason == "numeric_value_candidate"
                                and "numeric_value_candidate" not in existing["reasons"]
                            )
                            if not similarity_only and (
                                (existing["state"] == "review_candidate" and decision.action == "notify")
                                or numeric_upgrade
                            ):
                                # 0.17.1 owner ③（窗口 32）：同 memory pair 的
                                # numeric discovery 到来时必须接管 deep_read/
                                # snippets——先到的 similarity-only discovery
                                #（如 filler 区同文对）不能永久占住 span，否则
                                # 判定页把 Agent 带到无冲突证据的文本区。
                                if numeric_upgrade and existing["state"] == "review_candidate":
                                    existing["left_snippet"] = text[:200] if pair[0] == anchor_id else hit_text[:200]
                                    existing["right_snippet"] = hit_text[:200] if pair[0] == anchor_id else text[:200]
                                    existing["deep_read"] = {
                                        "left": {
                                            "memory_id": pair[0],
                                            "span": anchor_span if pair[0] == anchor_id else peer_span,
                                            **({"workspace": echo_workspace} if echo_workspace else {}),
                                        },
                                        "right": {
                                            "memory_id": pair[1],
                                            "span": peer_span if pair[0] == anchor_id else anchor_span,
                                            **({"workspace": echo_workspace} if echo_workspace else {}),
                                        },
                                    }
                                if existing["state"] == "review_candidate" and decision.action == "notify":
                                    existing["state"] = "notice_ready"
                                    existing["route"] = "notice_ready"
                                    existing["left_snippet"] = text[:200] if pair[0] == anchor_id else hit_text[:200]
                                    existing["right_snippet"] = hit_text[:200] if pair[0] == anchor_id else text[:200]
                                    existing["deep_read"] = {
                                        "left": {
                                            "memory_id": pair[0],
                                            "span": anchor_span if pair[0] == anchor_id else peer_span,
                                            **({"workspace": echo_workspace} if echo_workspace else {}),
                                        },
                                        "right": {
                                            "memory_id": pair[1],
                                            "span": peer_span if pair[0] == anchor_id else anchor_span,
                                            **({"workspace": echo_workspace} if echo_workspace else {}),
                                        },
                                    }
                            existing["reasons"].add(decision.reason)
                            existing["distance"] = min(existing["distance"], distance)
            ordered = [candidates[pair] for pair in sorted(candidates)]
            for item in ordered:
                item["reasons"] = sorted(item["reasons"])
            similarity_ordered = sorted(
                similarity_pool.values(), key=lambda item: float(item.get("distance") or 9),
            )[:pool_limit]
            for item in similarity_ordered:
                item["reasons"] = sorted(item["reasons"])
            next_anchor = anchors[-1]
            with db.connection() as conn:
                more = conn.execute(
                    "SELECT 1 FROM memories WHERE status='active' AND id > ? "
                    + workspace_anchor_sql
                    + "LIMIT 1",
                    (next_anchor, *anchor_params),
                ).fetchone()
            duplicates_ordered = [
                duplicates_pool[pair] for pair in sorted(duplicates_pool)
            ]
            cross_refs_ordered = [
                {**cross_bucket_refs[pair], "reasons": sorted(cross_bucket_refs[pair]["reasons"])}
                for pair in sorted(cross_bucket_refs)
            ]
            return {
                "anchors_scanned": len(anchors),
                "next_anchor_memory_id": int(next_anchor) if more else None,
                "candidates": ordered,
                "similarity_pool": similarity_ordered,
                "duplicates_pool": duplicates_ordered,
                "duplicates_truncated": duplicates_truncated,
                "cross_bucket_references": cross_refs_ordered,
                "anchor_buckets": [
                    {"workspace": ws, "anchors_scanned": entry["count"],
                     "last_anchor": entry["last_anchor"]}
                    for ws, entry in sorted(anchor_buckets.items())
                ],
                "counts": {
                    "knn_pairs": knn_pair_count,
                    "rule_pass": len(ordered),
                    "similarity_pool": len(similarity_ordered),
                    "duplicates": len(duplicates_ordered),
                    "filtered_open": filtered_open,
                    "filtered_dismissed": filtered_dismissed,
                    "stale_anchors": stale_anchors,
                },
            }

    def memory_scan_workspace_anomalies(self, **_: Any) -> dict[str, Any]:
        """C3a workspace anomaly check: single-pass matmul over all summary vectors.

        One SELECT reads every active memory's summary vector; numpy computes
        the N×N cosine in row-blocks (bounded memory); each row votes over its
        top-10 neighbours through the shared proportional normalize_gate. A
        memory whose neighbourhood passes the gate is a suspected misplacement:
        one kind='workspace' scan_queue row per memory (same queue, same gate
        as the pipeline's incremental suspects), capped at 10 new rows per run.
        Zero Qwen, milliseconds. numpy absence degrades with a structured
        outcome (it is not a declared dependency — llama-cpp-python normally
        brings it).
        """
        try:
            import numpy  # noqa: F401  # presence probe only: the structured error below must distinguish numpy-missing from an empty library
        except ImportError:
            return self.db.state.response({
                "error": "numpy_unavailable", "detail": (
                    "workspace anomaly check needs numpy (bundled with the "
                    "semantic-local extra); install numpy to run it"
                ),
            }, ok=False)
        # Fresh-boot coverage: the first run after an upgrade (before any
        # write) has no summary vectors yet — the write-path publish and the
        # startup backfill both ride the first embedder load. Ensure that load
        # happens here so the weekly task never no-ops its first round.
        if self.db.missing_summary_vec_rows():
            embedder, _warnings = self._tools._ensure_embedder()
            if embedder is not None:
                try:
                    self._tools._backfill_memory_summary_vectors(embedder)
                except Exception:
                    pass
        vectors = self.db.all_summary_vectors()
        if not vectors:
            return self.db.state.response({
                "status": "ok", "checked": 0, "suspected": 0, "queued": 0,
                "note": "no summary vectors yet (backfill pending or empty library)",
            })
        ids = sorted(vectors)
        n = len(ids)
        # The STABLE-sorting discipline lives in compute_summary_votes: equal
        # similarities (FakeEmbedder's binary vectors, duplicated content)
        # must pick the same neighbours on every machine/numpy version —
        # np.argpartition left ties arbitrary and CI once selected one beta
        # neighbour where the local run selected nine.
        votes_by_id = compute_summary_votes(vectors, ids)
        from .doctor import load_confirmed_workspaces

        confirmed = load_confirmed_workspaces(self.db.settings)
        dismissed = self.db.scan_queue.load_workspace_dismissals()
        suspected: list[dict[str, Any]] = []
        for row_mid in ids:
            vote = votes_by_id.get(row_mid)
            if vote is None:
                continue
            votes = vote["votes"]
            own = vote["own"]
            own_best = vote["own_best"]
            foreign_best = vote["foreign_best"]
            # Shared proportional gate (0.16.2 §1.1): the weekly
            # backstop judges by the SAME function as the pipeline's
            # suspect generation and the decision-time re-vote.
            passed, gate_evidence = normalize_gate(votes, own)
            if passed:
                top = str(gate_evidence["top_bucket"])
                if own in confirmed and top in confirmed:
                    continue  # owner-confirmed pair: never suspected
                entries = dismissed.get(row_mid)
                if entries:
                    record = self.db.get_memory(row_mid)
                    if record is not None:
                        top_ws = str(gate_evidence["top_bucket"])
                        if (int(record.get("version") or 1), top_ws) in entries:
                            continue  # durable dismissal: never suspected
                suspected.append({
                    "memory_id": row_mid,
                    "workspace": own,
                    "suspected_workspace": gate_evidence["top_bucket"],
                    "foreign_votes": gate_evidence["top_votes"],
                    "neighbours_checked": vote["k"],
                    "foreign_neighbour_id": foreign_best[1],
                    "own_neighbour_id": own_best[1],
                    "votes": dict(votes),
                })
        suspected.sort(key=lambda item: (-item["foreign_votes"], item["memory_id"]))
        # Lazy staleness (the notice channel's heir), BEFORE selecting the
        # cap: a pending suspect row whose subject already left its pinned
        # bucket is resolved — retire it even when the memory no longer shows
        # up in this sweep's findings (that is precisely why it is stale).
        self._tools._expire_relocated_workspace_rows()
        capped = suspected[:10]
        queued = 0
        # 0.16.2 §1.3: findings land in the judgment queue (same queue, same
        # gate as the pipeline's incremental suspects) — the workspace_review
        # notice channel no longer produces new findings. Identity reuses
        # _workspace_identity so a pipeline suspect and the weekly suspect
        # for the same memory@version+suspicion share one row (INSERT OR
        # IGNORE dedupes re-runs; dismissal keeps it dismissed).
        from .constants import PROTECTED_WORKSPACES
        from .scan_pipeline import _workspace_identity

        for item in capped:
            memory_id = int(item["memory_id"])
            record = self.db.get_memory(memory_id)
            if record is None or str(record.get("status") or "") != "active":
                continue
            version = int(record.get("version") or 1)
            own = str(item["workspace"])
            best_bucket = str(item["suspected_workspace"])
            detail = {
                "suspected_workspace": best_bucket,
                "current_workspace": own,
                "votes": item.get("votes") or {},
                "neighbours_checked": item["neighbours_checked"],
                "protected_involved": bool(
                    own in PROTECTED_WORKSPACES or best_bucket in PROTECTED_WORKSPACES
                ),
                "channel": "weekly_backstop",
            }
            outcome = self.db.scan_queue.enqueue(
                kind="workspace",
                workspace_canonical=own,
                candidate_key_hash=_workspace_identity(memory_id, version, best_bucket),
                member_versions=[{"memory_id": memory_id, "version": version}],
                evidence=[],
                reason=(
                    f"weekly vote {item['foreign_votes']}/{item['neighbours_checked']}"
                    f" -> {best_bucket!r}"
                ),
                severity="normal",
                source="workspace_anomaly_scan",
                detail=detail,
            )
            if str(outcome.get("outcome") or "") == "queued":
                queued += 1
        return self.db.state.response({
            "status": "ok",
            "checked": n,
            "suspected": len(suspected),
            "returned": len(capped),
            "queued": queued,
            "cap": 10,
            **({"capped": True} if len(suspected) > len(capped) else {}),
            "findings": [
                {
                    "memory_id": item["memory_id"],
                    "workspace": item["workspace"],
                    "suspected_workspace": item["suspected_workspace"],
                    "votes": f"{item['foreign_votes']}/{item['neighbours_checked']}",
                }
                for item in capped
            ],
        })
