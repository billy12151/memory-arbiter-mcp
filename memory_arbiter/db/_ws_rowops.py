"""ws 单行搬运（从 workspaces.py 搬出，拆分批 ② 纯移动）。"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, TYPE_CHECKING

from ..config import Settings
from ..constants import (
    EMBED_PREFIX_STS,
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


class _WsRowopsMixin:
    """ws 单行搬运（从 workspaces.py 搬出，拆分批 ② 纯移动）。"""

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
        _sha_collision_warning_text: "Callable[..., Any]"
        _content_sha_collision_warning_on_conn: "Callable[..., Any]"
        _conflict_slot_collision_warning_on_conn: "Callable[..., Any]"
        _competing_move_warning_on_conn: "Callable[..., Any]"
    def prepare_workspace_canonical_embedding(self, canonical: str, embedder: Any = None) -> list[float] | None:
        """Compute canonical embedding before caller takes a SQLite write lock."""
        canonical = _coerce_ws(canonical)
        if (
            embedder is not None
            and self.state.sqlite_vec_available
            and canonical
            and not is_default_workspace_term(canonical)
        ):
            try:
                er = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=canonical)
                return list(er.embedding) if er and er.embedding else None
            except Exception:
                return None
        return None

    def set_memory_workspace_canonical_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        canonical: str,
        *,
        precomputed_embedding: list[float] | None = None,
    ) -> tuple[bool, list[str]]:
        canonical = _coerce_ws(canonical)
        if not canonical:            return False, ["canonical must be a non-empty workspace string."]
        current = conn.execute(
            "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS bucket "
            "FROM memories WHERE id = ?",
            (int(memory_id),),
        ).fetchone()
        # 0.16.6 dedup gate BEFORE the void block (same ordering rationale as
        # move_memory_workspace_on_conn — review round-1 P2-1).
        sha_collision = self._content_sha_collision_warning_on_conn(
            conn, canonical, only_id=int(memory_id),
        )
        if sha_collision is not None:
            return False, [sha_collision]
        bucket_changed = bool(current and str(current["bucket"] or "") != canonical)
        if bucket_changed:
            # 0.16.0 §6⑤/§6⑯: a bucket reassignment is a move — void the
            # memory's old-bucket tickets and invalidate its scan watermark
            # so the pipeline re-pairs it inside the new bucket.
            self._db.conflicts.void_conflicts_on_conn(
                conn, [int(memory_id)],
                reason=f"canonical reassignment -> {canonical!r}",
            )
        # A9 补漏（R3 对码实证）：搬运路径（set canonical / move by id）此前
        # 无保护桶守卫——以 twin 身份走 move 即可把变体注册进 canonical 表，
        # 绕过写路径的 insert_memory 守卫。
        from ..twin_redirect import protected_bucket_variant

        if protected_bucket_variant(canonical):
            return False, [
                f"workspace {canonical!r} is a protected-bucket spelling variant "
                "and cannot be assigned; use the canonical name"
            ]
        cur = conn.execute(
            "UPDATE memories SET workspace_canonical = ?, "
            "scan_watermark = CASE WHEN "
            "COALESCE(NULLIF(workspace_canonical,''),workspace) != ? "
            "THEN NULL ELSE scan_watermark END "
            "WHERE id = ?",
            (canonical, canonical, int(memory_id)),
        )
        if (cur.rowcount or 0) == 0:
            return False, ["memory id not found."]
        # Scope membership changed without COUNT/version movement — the
        # linked-df fingerprint cannot see this; drop the cache explicitly.
        self._db.invalidate_linked_df_cache()
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
            (canonical, utc_now_iso()),
        )
        if precomputed_embedding is not None and not is_default_workspace_term(canonical):
            row = conn.execute(
                "SELECT id FROM workspace_canonicals WHERE name = ?", (canonical,)
            ).fetchone()
            if row is not None:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                        (int(row["id"]), json.dumps(precomputed_embedding)),
                    )
                except sqlite3.Error as exc:
                    return True, [
                        f"workspace canonical vector publish failed for {canonical!r}; retry a write using this workspace after sqlite-vec and embedding configuration recover: {exc}"
                    ]
        return True, []

    def move_memory_workspace_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        workspace: str,
        *,
        precomputed_embedding: list[float] | None = None,
        allow_default: bool = False,
        current_bucket: str | None = None,
        sha_collision: bool | None = None,
        queue_rows: "list[Any] | None" = None,
    ) -> tuple[bool, list[str]]:
        """Reassign one memory's workspace bucket and canonical together.

        Companion to set_memory_workspace_canonical_on_conn for governance
        moves by id: both the raw bucket column (``workspace``) and
        ``workspace_canonical`` are written so the moved memory stops
        resolving through its old bucket name. Default stays reserved —
        callers refuse a default destination before opening the transaction;
        the guard here is defensive for direct callers. ``allow_default``
        opens the 0.16.3 explicitly-declared fallback (agent cannot find a
        suitable bucket → park in the global pool with audit + notice).

        P2 #10 bulk-call prefetch hooks (all optional, keyword-only):
        ``current_bucket`` (the row's COALESCE bucket), ``sha_collision``
        (the dedup-gate verdict against the destination), and ``queue_rows``
        (this memory's pending scan_queue rows) let a multi-id caller run
        each probe ONCE per request on the transaction snapshot. Every
        default None keeps the per-id self-read, so single-id callers
        (queue_protocol._execute_auto_move) are untouched.
        """
        workspace = _coerce_ws(workspace)
        if not workspace or (is_default_workspace_term(workspace) and not allow_default):
            return False, ["move destination must be a non-default workspace string."]
        # A9 补漏（R3 对码实证）：搬运路径此前无保护桶守卫——以 twin 身份
        # 走 move 即可把变体注册进 canonical 表，绕过写路径的 insert_memory
        # 守卫（R3 实测 ok=True + canon rows 含 mema_twin）。
        from ..twin_redirect import protected_bucket_variant

        if protected_bucket_variant(workspace):
            return False, [
                f"workspace {workspace!r} is a protected-bucket spelling variant "
                "and cannot be a move destination; use the canonical name"
            ]
        if current_bucket is None:
            row = conn.execute(
                "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS bucket "
                "FROM memories WHERE id = ?",
                (int(memory_id),),
            ).fetchone()
            current_bucket = str(row["bucket"] or "") if row is not None else ""
        bucket_changed = bool(current_bucket and current_bucket != workspace)
        move_warnings: list[str] = []
        # 0.16.6 dedup gate runs BEFORE any ticket voiding: a bulk-move caller
        # commits sibling successes in the same transaction, so a collision
        # refusal after the void block would destroy pending judgment tickets
        # for a row that never moved (review round-1 P2-1).
        if sha_collision is None:
            sha_collision = self._content_sha_collision_warning_on_conn(
                conn, workspace, only_id=int(memory_id),
            ) is not None
        if sha_collision:
            return False, [self._sha_collision_warning_text(workspace, [int(memory_id)])]
        if bucket_changed:
            # 0.16.0 §6⑤/§6⑯: move 视同编辑 — void old-bucket tickets
            # (releasing slot/candidate identities, suppressing nothing) and
            # clear the scan watermark so the pipeline re-pairs the memory in
            # its new bucket. Same-bucket no-ops stay side-effect free.
            voided = self._db.conflicts.void_conflicts_on_conn(
                conn, [int(memory_id)],
                reason=f"workspace move -> {workspace!r}",
            )
            # §6⑤ companion: pending JUDGMENT QUEUE rows that pair the moved
            # memory with its old-bucket peers describe a bucket identity that
            # just died — void them (the pipeline re-enqueues valid pairs in
            # the new bucket); stale workspace suspects likewise. The
            # candidate identity is REWRITTEN, not kept: the UNIQUE hash spans
            # all statuses, and a kept hash would burn the pair@version
            # identity forever (a move-back could never re-enqueue it —
            # adversarial review #4). The UPDATE guards on status='pending'
            # so a row an earlier same-request move already voided (possible
            # only with prefetched queue_rows) is left exactly as the
            # sequential per-id flow would have skipped it.
            try:
                if queue_rows is None:
                    queue_rows = conn.execute(
                        """SELECT id, candidate_key_hash FROM scan_queue
                           WHERE kind IN ('conflict','workspace') AND status = 'pending'
                         AND EXISTS(SELECT 1 FROM json_each(scan_queue.member_versions) AS m
                                    WHERE CAST(json_extract(m.value,'$.memory_id') AS INTEGER)=?)""",
                        (int(memory_id),),
                    ).fetchall()
                from .additive import voided_identity_hash

                now_ts = utc_now_iso()
                for queue_row in queue_rows:
                    conn.execute(
                        """UPDATE scan_queue SET status='voided',
                           candidate_key_hash=?, decided_reason='member moved to '||?,
                           decided_at=?, updated_at=?
                           WHERE id=? AND status='pending'""",
                        (voided_identity_hash(str(queue_row["candidate_key_hash"]), int(queue_row["id"])),
                         workspace, now_ts, now_ts, int(queue_row["id"])),
                    )
            except sqlite3.Error:
                pass
            if voided:
                # Structured sentinel consumed by the move surface (0.16.0):
                # "voided_conflict_tickets:<n>" is reported in the response,
                # never shown as a raw warning.
                move_warnings.append(f"voided_conflict_tickets:{voided}")
        cur = conn.execute(
            "UPDATE memories SET workspace = ?, workspace_canonical = ?, "
            "scan_watermark = CASE WHEN "
            "COALESCE(NULLIF(workspace_canonical,''),workspace) != ? "
            "THEN NULL ELSE scan_watermark END "
            "WHERE id = ?",
            (workspace, workspace, workspace, int(memory_id)),
        )
        if (cur.rowcount or 0) == 0:
            return False, ["memory id not found."]
        # Same fingerprint blind spot as the canonical setter above.
        self._db.invalidate_linked_df_cache()
        if not is_default_workspace_term(workspace):
            # The global pool is a reserved term, never a registry row.
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
                (workspace, utc_now_iso()),
            )
        if precomputed_embedding is not None:
            row = conn.execute(
                "SELECT id FROM workspace_canonicals WHERE name = ?", (workspace,),
            ).fetchone()
            if row is not None:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                        (int(row["id"]), json.dumps(precomputed_embedding)),
                    )
                except sqlite3.Error as exc:
                    return True, move_warnings + [
                        f"workspace canonical vector publish failed for {workspace!r}; retry a write using this workspace after sqlite-vec and embedding configuration recover: {exc}"
                    ]
        return True, move_warnings

    def set_memory_workspace_canonical(
        self,
        memory_id: int,
        canonical: str,
        embedder: Any = None,
        *,
        conn: sqlite3.Connection | None = None,
        precomputed_embedding: list[float] | None = None,
    ) -> tuple[bool, list[str]]:
        """Directly set a memory's workspace_canonical column.

        update_memory() intentionally whitelists only trust/status/metadata
        fields, so it silently drops a workspace_canonical write. This helper writes the column directly and
        registers the canonical in workspace_canonicals — AND its vector row when
        an embedder + sqlite-vec are available — so the resolver's KNN can later
        fuzzy-match this canonical instead of re-splitting it into a sibling.
        """
        if conn is not None:
            return self.set_memory_workspace_canonical_on_conn(
                conn,
                memory_id,
                canonical,
                precomputed_embedding=precomputed_embedding,
            )
        if not self._db_available or not self.state.sqlite_writable:
            return False, ["SQLite write unavailable; workspace_canonical not set."]
        canonical = _coerce_ws(canonical)
        if not canonical:
            return False, ["canonical must be a non-empty workspace string."]
        # Compute the canonical embedding OUTSIDE the write txn (embedder calls
        # can be slow / must not hold the write lock).
        embedding = precomputed_embedding
        if embedding is None:
            embedding = self.prepare_workspace_canonical_embedding(canonical, embedder)
        try:
            with self.write_transaction() as txn_conn:
                return self.set_memory_workspace_canonical_on_conn(
                    txn_conn,
                    memory_id,
                    canonical,
                    precomputed_embedding=embedding,
                )
        except sqlite3.Error as exc:
            return False, [f"set_memory_workspace_canonical failed: {exc}"]
