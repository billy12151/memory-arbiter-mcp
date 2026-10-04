"""ws 别名决策与守卫原语（从 workspaces.py 搬出，拆分批 ② 纯移动）。"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, TYPE_CHECKING

from ..config import Settings
from ..constants import (
    is_default_workspace_term,
)
from ..degrade import DegradeState
from ..models import utc_now_iso
from ..ws_keys import (
    _DEFAULT_TERM_SQL_NOT_IN as _DEFAULT_TERM_SQL_NOT_IN,
    _DEFAULT_TERM_SQL_PARAMS as _DEFAULT_TERM_SQL_PARAMS,
    _coerce_ws as _coerce_ws,
    _mechanical_ws_key as _mechanical_ws_key,
    _normalize_alias_key as _normalize_alias_key,
    _normalize_ws_group_key as _normalize_ws_group_key,
)

if TYPE_CHECKING:
    from .core import MemoryDB


class _WsAliasMixin:
    """ws 别名决策与守卫原语（从 workspaces.py 搬出，拆分批 ② 纯移动）。"""

    # 拆分批 ②：声明式注解（mypy strict；形态对齐主类 @property/@contextmanager）
    if TYPE_CHECKING:
        _db: "MemoryDB"
        from typing import Any as _Any
        @property
        def _db_available(self) -> bool: ...
        @property
        def settings(self) -> "Settings": ...
        @property
        def state(self) -> "DegradeState": ...
        def connection(self) -> "_Any": ...
        def write_transaction(self) -> "_Any": ...
    @staticmethod
    def _apply_alias_decision_on_conn(
        conn: sqlite3.Connection,
        workspace_name: str,
        canonical: str,
        *,
        status: str = "confirmed",
        force: bool = False,
    ) -> tuple[bool, list[str]]:
        """Record one alias→canonical decision (confirmed redirect or rejection).

        Single primitive behind record_workspace_decision_on_conn and
        _install_workspace_redirect_on_conn. Guards are the UNION of both
        legacy call sites (never the intersection): non-empty inputs,
        default-term refused in both directions, status enum, and self-pair
        no-op. ``force`` lets an operator override a prior rejection.
        """
        key = _normalize_alias_key(workspace_name)
        if not key:
            return False, ["workspace name must be non-empty."]
        canonical = _coerce_ws(canonical)
        if not canonical:
            return False, ["canonical must be a non-empty workspace string."]
        default_refusal = [
            "default is a reserved global pool and cannot be merged in either "
            "direction; workspace decisions require two non-default names."
        ]
        if is_default_workspace_term(workspace_name) or is_default_workspace_term(canonical):
            return False, default_refusal
        if status not in {"confirmed", "rejected"}:
            return False, [f"status={status!r} invalid; expected confirmed|rejected."]
        if key == _normalize_alias_key(canonical):
            # Self-pair: a name needs no redirect to itself (exact matching
            # already resolves it), and rejecting it is meaningless. No-op.
            return True, []
        # First-seen orthography: reuse the registered mechanical twin spelling
        # (AgentLane vs agent-lane) instead of storing a variant. A rejected
        # target is never registered, so a ghost spelling stays verbatim here.
        mechanical = _WsAliasMixin._mechanical_canonical_on_conn(conn, canonical)
        if mechanical is not None:
            canonical = mechanical
            # The substitution can land on a registered default spelling
            # ('de-fault' -> 'default' once the pool row exists): re-apply the
            # reserved-pool refusal to the substituted canonical so the
            # mechanical fold cannot bypass the front guard, then re-check the
            # self-pair against the substituted spelling.
            if is_default_workspace_term(canonical):
                return False, default_refusal
            if key == _normalize_alias_key(canonical):
                return True, []
        # Rejected-pair check by mechanical key on BOTH sides, not exact
        # string: a ghost spelling variant of the rejected target must not
        # bypass the refusal, and neither may a spelling-variant ALIAS key
        # (rejected 'agent_lane'→'X' must also block confirm 'agent-lane'→'X'
        # even though the two alias keys differ verbatim). The table is a
        # small governance table, so scan the rejected rows and match
        # mechanically; the exact-key disjunct keeps degenerate
        # separator-only aliases behaving exactly as before.
        key_mech = _mechanical_ws_key(key)
        canonical_mech = _mechanical_ws_key(canonical)
        # Mechanical twins are ONE workspace identity (the write path folds
        # them unconditionally), so a keep-separate decision between them is
        # unenforceable and would split one workspace's memories across two
        # buckets — degrading conflict detection and recall. Refuse it; a
        # genuine two-project pair with coincidentally similar names must be
        # disambiguated by renaming one side instead (owner 2026-10-01).
        if status == "rejected" and canonical_mech and canonical_mech == key_mech:
            return False, [
                f"workspace {key!r} and {canonical!r} are spelling variants of "
                "the same mechanical identity (case/separator-insensitive); "
                "keeping them separate would split one workspace's memories "
                "across two buckets and degrade conflict detection and recall. "
                "If they really are two different projects, rename one of them "
                "first."
            ]
        rejected_match = [
            (str(row["alias_workspace"]), str(row["canonical"]))
            for row in conn.execute(
                "SELECT alias_workspace, canonical FROM workspace_aliases "
                "WHERE status='rejected'",
            )
            if (
                str(row["alias_workspace"]) == key
                or (key_mech and _mechanical_ws_key(str(row["alias_workspace"])) == key_mech)
            )
            and _mechanical_ws_key(str(row["canonical"])) == canonical_mech
        ]
        if rejected_match and status == "confirmed":
            if not force:
                return False, [
                    f"workspace {key!r} was explicitly kept separate from "
                    f"{rejected_match[0][1]!r}; refusing to reverse that decision silently."
                ]
            # force override: clear every matched rejected row under its own
            # stored alias spelling before confirming.
            for rejected_alias, rejected_canonical in rejected_match:
                conn.execute(
                    "DELETE FROM workspace_aliases "
                    "WHERE alias_workspace=? AND canonical=? AND status='rejected'",
                    (rejected_alias, rejected_canonical),
                )
        now = utc_now_iso()
        if status == "confirmed":
            # A confirmation always wins over stale conflicting confirmed rows
            # for the same alias (unconditional, as at the committed baseline).
            # An intermediate state of the uncommitted working tree had narrowed
            # this to force-only dead code (B-A1); this primitive restores the
            # unconditional semantics.
            conn.execute(
                "DELETE FROM workspace_aliases "
                "WHERE alias_workspace=? AND status='confirmed' AND canonical<>?",
                (key, canonical),
            )
        conn.execute(
            """INSERT INTO workspace_aliases(alias_workspace,canonical,status,updated_at)
               VALUES(?,?,?,?)
               ON CONFLICT(alias_workspace,canonical) DO UPDATE SET
                 status=excluded.status,updated_at=excluded.updated_at""",
            (key, canonical, status, now),
        )
        if status == "confirmed":
            # A9（0.17.1 修复批）：别名确认同样不得注册保护桶变体——毒行
            # 落地后精确命中保护键、劫持 twin 本体写入（P1-3 同族）。
            from ..twin_redirect import protected_bucket_variant

            if protected_bucket_variant(canonical):
                return False, [
                    f"workspace {canonical!r} is a protected-bucket spelling variant "
                    "and cannot be confirmed; use the canonical name"
                ]
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES(?,?)",
                (canonical, now),
            )
        return True, []

    def record_workspace_decision_on_conn(
        self,
        conn: sqlite3.Connection,
        workspace_name: str,
        canonical: str,
        *,
        status: str = "confirmed",
        force: bool = False,
    ) -> tuple[bool, list[str]]:
        return self._apply_alias_decision_on_conn(
            conn, workspace_name, canonical, status=status, force=force,
        )

    def record_workspace_decision(
        self,
        workspace_name: str,
        canonical: str,
        *,
        status: str = "confirmed",
        force: bool = False,
        conn: sqlite3.Connection | None = None,
    ) -> tuple[bool, list[str]]:
        if conn is not None:
            return self.record_workspace_decision_on_conn(
                conn, workspace_name, canonical, status=status, force=force,
            )
        if not self._db_available or not self.state.sqlite_writable:
            return False, ["SQLite write unavailable; workspace decision not written."]
        try:
            with self.write_transaction() as txn_conn:
                return self.record_workspace_decision_on_conn(
                    txn_conn, workspace_name, canonical, status=status, force=force,
                )
        except sqlite3.Error as exc:
            return False, [f"record_workspace_decision failed: {exc}"]

    def get_workspace_decision(self, workspace_name: str) -> dict[str, Any] | None:
        if not self._db_available:
            return None
        workspace_key = _normalize_alias_key(workspace_name)
        if not workspace_key:
            return None
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT alias_workspace,canonical,status,updated_at "
                    "FROM workspace_aliases WHERE alias_workspace=? "
                    "ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END,"
                    "updated_at DESC,canonical ASC LIMIT 1",
                    (workspace_key,),
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.Error:
            return None

    @staticmethod
    def _repoint_workspace_targets_on_conn(
        conn: sqlite3.Connection,
        old: str,
        new: str,
        *,
        exclude_aliases: tuple[str, ...] = (),
    ) -> list[str]:
        """Repoint alias decisions from ``old`` to ``new``; returns warnings.

        A rejected decision whose alias is a mechanical twin of ``new`` would
        become an unenforceable, memory-splitting row: the write path folds
        twins unconditionally and the resolver now honors rejections *against*
        that fold. Drop it with a warning instead of repointing — mirroring
        the governance guard that refuses creating such rows (0.17.1).
        """
        now = utc_now_iso()
        warnings: list[str] = []
        exclusions = ""
        params: list[Any] = [new, now, old]
        if exclude_aliases:
            placeholders = ",".join("?" for _ in exclude_aliases)
            exclusions = f" AND alias_workspace NOT IN ({placeholders})"
            params.extend(exclude_aliases)
        new_mech = _mechanical_ws_key(new)
        if new_mech:
            doomed: list[tuple[str, str]] = []
            for row in conn.execute(
                "SELECT alias_workspace, canonical FROM workspace_aliases "
                "WHERE canonical=? AND status='rejected'" + exclusions,
                [old, *exclude_aliases],
            ):
                alias_ws = str(row["alias_workspace"])
                if _mechanical_ws_key(alias_ws) == new_mech:
                    doomed.append((alias_ws, str(row["canonical"])))
            for alias_ws, rejected_canonical in doomed:
                conn.execute(
                    "DELETE FROM workspace_aliases "
                    "WHERE alias_workspace=? AND canonical=? AND status='rejected'",
                    (alias_ws, rejected_canonical),
                )
                warnings.append(
                    f"rejected workspace decision ({alias_ws!r} kept separate from "
                    f"{rejected_canonical!r}) dropped while repointing {old!r} to {new!r}: "
                    "the alias is a spelling variant of the destination, and a twin-pair "
                    "rejection would split one workspace's memories across two buckets."
                )
        conn.execute(
            "INSERT OR IGNORE INTO workspace_aliases("
            "alias_workspace,canonical,status,updated_at) "
            "SELECT alias_workspace,?,status,? FROM workspace_aliases "
            "WHERE canonical=?" + exclusions,
            params,
        )
        conn.execute(
            "DELETE FROM workspace_aliases WHERE canonical=?" + exclusions,
            (old, *exclude_aliases),
        )
        return warnings

    @staticmethod
    def _mechanical_canonical_on_conn(
        conn: sqlite3.Connection, workspace_name: str,
    ) -> str | None:
        key = _mechanical_ws_key(workspace_name)
        if not key:
            return None
        for row in conn.execute("SELECT name FROM workspace_canonicals"):
            name = str(row["name"])
            if _mechanical_ws_key(name) == key:
                return name
        return None

    def registered_mechanical_canonical(self, workspace_name: str) -> str | None:
        """Return the registered canonical with the same mechanical key, if any.

        Read-only twin of the destination-orthography fold migrate_workspace
        applies: callers that point memories at a destination spelling must
        land on the already-registered orthography instead of re-splitting
        the canonical registry into a mechanical twin.
        """
        name = _coerce_ws(workspace_name)
        if not name:
            return None
        try:
            with self.connection() as conn:
                return self._mechanical_canonical_on_conn(conn, name)
        except sqlite3.Error:
            return None

    def confirmed_alias_canonical(self, workspace_name: str) -> str | None:
        """Return the confirmed-redirect canonical for an alias spelling, if any.

        Mirrors the write path's confirmed-alias short-circuit: a move
        destination that is a confirmed alias must land rows on the decision
        canonical, not re-register the alias spelling as a shadow canonical
        that ordinary writes would never create.
        """
        name = _coerce_ws(workspace_name)
        key = _normalize_alias_key(name)
        if not name or not key:
            return None
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT canonical FROM workspace_aliases "
                    "WHERE alias_workspace=? AND status='confirmed' "
                    "ORDER BY updated_at DESC, canonical ASC LIMIT 1",
                    (key,),
                ).fetchone()
            return str(row["canonical"]) if row is not None else None
        except sqlite3.Error:
            return None

    @staticmethod
    def _sha_collision_warning_text(to_ws: str, ids: "list[int]") -> str:
        listed = ", ".join(f"#{int(value)}" for value in ids)
        return (
            f"content duplicate collision: moving {listed} into {to_ws!r} would create "
            "byte-identical ACTIVE memories in one workspace (dedup gate); "
            "govern the duplicates first (merge/retire) and retry"
        )

    @staticmethod
    def _content_sha_collision_warning_on_conn(
        conn: sqlite3.Connection, to_ws: str, *,
        from_ws: str | None = None, only_id: int | None = None,
    ) -> str | None:
        """0.16.6 dedup gate on workspace moves: moving an ACTIVE row into a
        workspace that already holds the same content_sha ACTIVE row would
        violate idx_memories_content_sha mid-bulk-UPDATE (aborting the whole
        transaction). Refuse up front with the colliding ids, mirroring the
        slot-collision pre-check pattern."""
        canonical = "COALESCE(NULLIF(workspace_canonical, ''), workspace)"
        target = "COALESCE(NULLIF(t.workspace_canonical, ''), t.workspace)"
        if only_id is not None:
            source = "m.id = ?"
            params: "tuple[int | str | None, str]" = (int(only_id), to_ws)
        else:
            source = f"{canonical} = ?"
            params = (from_ws, to_ws)
        rows = conn.execute(
            f"SELECT m.id, m.subject FROM memories m "
            f"WHERE {source} AND m.status='active' AND m.content_sha IS NOT NULL "
            f"AND EXISTS (SELECT 1 FROM memories t WHERE {target} = ? "
            "AND t.status='active' AND t.content_sha = m.content_sha AND t.id != m.id) "
            "ORDER BY m.id LIMIT 5",
            params,
        ).fetchall()
        if not rows:
            return None
        return _WsAliasMixin._sha_collision_warning_text(
            to_ws, [int(r["id"]) for r in rows],
        )

    @staticmethod
    def _content_sha_collision_ids_on_conn(
        conn: sqlite3.Connection, to_ws: str, memory_ids: "list[int]",
    ) -> "set[int]":
        """P2 #10 batched dedup gate: which of ``memory_ids`` hold ACTIVE
        content whose sha already exists as ACTIVE in ``to_ws`` (pre-existing
        rows only). Same-request siblings are gated by the bulk caller as
        they land, preserving the sequential per-id semantics bit-for-bit."""
        ids = sorted({int(value) for value in memory_ids})
        if not ids:
            return set()
        placeholders = ",".join("?" * len(ids))
        target = "COALESCE(NULLIF(t.workspace_canonical, ''), t.workspace)"
        rows = conn.execute(
            f"SELECT m.id FROM memories m "
            f"WHERE m.id IN ({placeholders}) "
            f"AND m.status='active' AND m.content_sha IS NOT NULL "
            f"AND EXISTS (SELECT 1 FROM memories t WHERE {target} = ? "
            "AND t.status='active' AND t.content_sha = m.content_sha AND t.id != m.id)",
            (*ids, to_ws),
        ).fetchall()
        return {int(row["id"]) for row in rows}

    @staticmethod
    def _pending_queue_rows_by_memory_id_on_conn(
        conn: sqlite3.Connection, memory_ids: "list[int]",
    ) -> "dict[int, list[dict[str, Any]]]":
        """P2 #10 batched pending scan_queue lookup: every pending
        conflict/workspace queue row keyed by its member memory ids — the
        json_each-join single-SQL form of the per-id EXISTS probe (the same
        shape as operations._merge_conflict_membership_on_conn)."""
        wanted = sorted({int(value) for value in memory_ids})
        if not wanted:
            return {}
        try:
            rows = conn.execute(
                "SELECT q.id, q.candidate_key_hash, "
                "CAST(json_extract(m.value,'$.memory_id') AS INTEGER) AS memory_id "
                "FROM scan_queue AS q "
                "JOIN json_each(q.member_versions) AS m "
                "JOIN json_each(?) AS wanted "
                "ON CAST(json_extract(m.value,'$.memory_id') AS INTEGER)=CAST(wanted.value AS INTEGER) "
                "WHERE q.kind IN ('conflict','workspace') AND q.status = 'pending'",
                (json.dumps(wanted, ensure_ascii=False, sort_keys=True, separators=(",", ":")),),
            ).fetchall()
        except sqlite3.Error:
            return {}
        grouped: "dict[int, list[dict[str, Any]]]" = {}
        for row in rows:
            bucket = grouped.setdefault(int(row["memory_id"]), [])
            if any(int(existing["id"]) == int(row["id"]) for existing in bucket):
                continue  # a queue row may pin the same memory at two versions
            bucket.append({
                "id": int(row["id"]),
                "candidate_key_hash": str(row["candidate_key_hash"] or ""),
            })
        return grouped

    @staticmethod
    def _conflict_slot_collision_warning_on_conn(
        conn: sqlite3.Connection, source: str, destination: str,
    ) -> list[str] | None:
        """Refuse a workspace move that would collide active conflict slots.

        ``idx_conflicts_active_slot`` is a partial unique index over
        (workspace_canonical, slot_key_hash) WHERE status IN ('open','applying'),
        and a slot_key carries no workspace — so re-pointing the source
        workspace's active conflicts to the destination can collide with the
        destination's own active row for the same slot. The bulk UPDATE would
        abort the whole migration transaction. Silently resolving either
        side (closing/dropping an open conflict) would fabricate a triage
        decision, so the move is refused up front with the colliding ids so
        the operator can triage via the normal governance flow and retry.
        """
        rows = conn.execute(
            "SELECT s.id AS src_id, d.id AS dst_id FROM conflicts s "
            "JOIN conflicts d ON d.slot_key_hash = s.slot_key_hash "
            "WHERE s.workspace_canonical = ? AND d.workspace_canonical = ? "
            "AND s.status IN ('open','applying') AND d.status IN ('open','applying') "
            "AND s.slot_key_hash IS NOT NULL",
            (source, destination),
        ).fetchall()
        if not rows:
            return None
        pairs = ", ".join(f"#{int(row['src_id'])}->#{int(row['dst_id'])}" for row in rows)
        return [
            f"workspace move {source!r} -> {destination!r} refused: active "
            f"conflicts would collide on the same slot in the destination "
            f"workspace (source conflict id -> destination conflict id: {pairs}). "
            f"Resolve or triage the listed conflicts first "
            f"(memory_review conflict_detail + memory_govern resolve_conflict), "
            f"then retry the move."
        ]

    def _competing_move_warning_on_conn(
        self, conn: sqlite3.Connection, source: str, destination: str,
    ) -> list[str] | None:
        source_exists = conn.execute(
            "SELECT 1 FROM workspace_canonicals WHERE name=? UNION ALL "
            "SELECT 1 FROM memories WHERE "
            "COALESCE(NULLIF(workspace_canonical,''),workspace)=? LIMIT 1",
            (source, source),
        ).fetchone() is not None
        if source_exists:
            return None
        mechanical = self._mechanical_canonical_on_conn(conn, source)
        if mechanical and mechanical != source:
            if mechanical == destination:
                return []
            return [
                f"workspace {source!r} already exists as {mechanical!r}; "
                f"refusing a competing move to {destination!r}."
            ]
        decision = conn.execute(
            "SELECT canonical,status FROM workspace_aliases WHERE alias_workspace=? "
            "ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END,"
            "updated_at DESC,canonical ASC LIMIT 1",
            (_normalize_alias_key(source),),
        ).fetchone()
        if decision is None:
            return None
        existing = str(decision["canonical"])
        if existing == destination:
            return []
        return [
            f"workspace {source!r} already has a decision for {existing!r}; "
            f"refusing a competing move to {destination!r}."
        ]

    @staticmethod
    def _install_workspace_redirect_on_conn(
        conn: sqlite3.Connection, workspace_name: str, canonical: str,
        *, force: bool = False,
    ) -> bool:
        ok, _errors = _WsAliasMixin._apply_alias_decision_on_conn(
            conn, workspace_name, canonical, status="confirmed", force=force,
        )
        return ok
