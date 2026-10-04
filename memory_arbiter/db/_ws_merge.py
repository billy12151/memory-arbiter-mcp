"""ws rename/migrate/normalize 合并套件（从 workspaces.py 搬出，拆分批 ② 纯移动）。"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, TYPE_CHECKING

from ..config import Settings
from ..constants import (
    EMBED_PREFIX_STS,
    is_default_workspace_term,
)
from ..db_generation import database_startup_lock
from ..degrade import DegradeState
from ..models import utc_now_iso
from ._ws_alias import _WsAliasMixin
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


class _WsMergeMixin:
    """ws rename/migrate/normalize 合并套件（从 workspaces.py 搬出，拆分批 ② 纯移动）。"""

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
        _repoint_workspace_targets_on_conn: "Callable[..., Any]"
        _install_workspace_redirect_on_conn: "Callable[..., Any]"
        _mechanical_canonical_on_conn: "Callable[..., Any]"
    def rename_workspace_canonical(
        self, old: str, new: str,
    ) -> tuple[int, list[str], bool]:
        """Rename or merge a canonical and keep old-name forwarding stable.

        A5（0.17.1 修复批）：返回第三元素 ``committed`` —— 事务是否已越过
        全部前置守卫并提交（含 no-op 与空源桶：它们同样"已处理"）。调用方
        （operations）据此落治理审计与判定响应 ok：此前以 ``not warnings``
        推断，repoint 警告形态（UPDATE 已提交）会同时报失败且无审计
        （产品面实测：库内已改名、响应 renamed=False、审计 0 行）。
        """
        if not self._db_available or not self.state.sqlite_writable:
            return 0, ["SQLite write unavailable; rename skipped."], False
        old = _coerce_ws(old)
        new = _coerce_ws(new)
        if not old or not new:
            return 0, ["rename requires non-empty old and new canonical."], False
        # renaming into or out of default would merge the global pool
        # with one project — refuse in both directions.
        if is_default_workspace_term(old) or is_default_workspace_term(new):
            return 0, [
                "default is a reserved global pool and cannot be merged in either "
                "direction; rename requires two non-default workspace names."
            ], False
        if old == new:
            # no-op：无写入但"已处理"（保持既有审计行为：no-op 也落一行）。
            return 0, [], True
        # A9（0.17.1 修复批）：目标名不得是保护桶的机械等价变体——否则
        # rename 会把保护桶拼写注册进攻击者桶（实测 rename→mematwin 成功，
        # 随后 twin 本体写入被折进攻击者桶）。
        from ..twin_redirect import protected_bucket_variant

        if protected_bucket_variant(new):
            return 0, [
                f"workspace {new!r} is a protected-bucket spelling variant and "
                "cannot be a rename target; use the canonical name"
            ], False
        # Destination orthography: when `new` is a mechanical variant of an
        # already-registered canonical other than `old`, rename into the
        # registered spelling. The verbatim branch would otherwise
        # double-register the variant while the redirect normalizes to the
        # registered twin — splitting one canonical across two spellings. A
        # twin that IS `old` is a genuine spelling change of the same row
        # (e.g. a case-only rename) and proceeds untouched.
        try:
            with self.connection() as conn:
                registered = self._mechanical_canonical_on_conn(conn, new)
        except sqlite3.Error:
            registered = None
        if registered is not None and registered != old:
            new = registered
            # The fold can land on a registered default spelling; keep the
            # reserved pool refused in both directions (mirrors the guard).
            if is_default_workspace_term(new):
                return 0, [
                    "default is a reserved global pool and cannot be merged in either "
                    "direction; rename requires two non-default workspace names."
                ], False
        now = utc_now_iso()
        try:
            # Serialize against migrate/normalize with the same advisory flock
            # (always taken before the write transaction, in this order).
            with database_startup_lock(self.settings.db_path), self.write_transaction() as conn:
                competing = self._competing_move_warning_on_conn(conn, old, new)
                if competing is not None:
                    return 0, competing, False
                collision = self._conflict_slot_collision_warning_on_conn(conn, old, new)
                if collision is not None:
                    return 0, collision, False
                sha_collision = self._content_sha_collision_warning_on_conn(conn, new, from_ws=old)
                if sha_collision is not None:
                    return 0, [sha_collision], False
                cur = conn.execute(
                    # 疑似#9（owner 2026-10-04 拍板：换桶后要扫新桶冲突）：
                    # rename 与 migrate 统一清 scan_watermark——被重指派到
                    # 新桶的行须对新桶的既有内容做一轮冲突扫描。
                    "UPDATE memories SET workspace_canonical = ?, scan_watermark = NULL "
                    "WHERE COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?",
                    (new, old),
                )
                updated = cur.rowcount or 0
                if updated:
                    # Scope membership changed without COUNT/version movement
                    # — the linked-df fingerprint cannot see a rename.
                    self._db.invalidate_linked_df_cache()
                conn.execute(
                    "UPDATE conflicts SET workspace_canonical=? WHERE workspace_canonical=?",
                    (new, old),
                )
                # If `new` already exists, a plain rename would hit UNIQUE(name)
                # and (with OR IGNORE) silently orphan `old`. Instead merge: drop
                # the old row so the surviving `new` canonical is authoritative.
                new_exists = conn.execute(
                    "SELECT 1 FROM workspace_canonicals WHERE name = ?", (new,)
                ).fetchone()
                if new_exists:
                    old_row = conn.execute(
                        "SELECT id FROM workspace_canonicals WHERE name = ?", (old,)
                    ).fetchone()
                    if old_row is not None:
                        if self.state.sqlite_vec_available:
                            try:
                                conn.execute(
                                    "DELETE FROM workspace_canonicals_vec WHERE id = ?",
                                    (int(old_row["id"]),),
                                )
                            except sqlite3.Error:
                                pass  # vec table may be absent; canonical delete still proceeds
                        conn.execute(
                            "DELETE FROM workspace_canonicals WHERE name = ?", (old,)
                        )
                else:
                    conn.execute(
                        "UPDATE workspace_canonicals SET name = ? WHERE name = ?",
                        (new, old),
                    )
                fwd_key = _normalize_alias_key(old)
                new_key = _normalize_alias_key(new)
                if fwd_key == new_key:
                    repoint_warnings = self._repoint_workspace_targets_on_conn(conn, old, new)
                    conn.execute(
                        "DELETE FROM workspace_aliases WHERE alias_workspace=? AND canonical=?",
                        (new_key, new),
                    )
                    return updated, repoint_warnings, True
                repoint_warnings = self._repoint_workspace_targets_on_conn(
                    conn, old, new, exclude_aliases=(fwd_key, new_key),
                )
                conn.execute(
                    "DELETE FROM workspace_aliases "
                    "WHERE alias_workspace=? AND canonical IN (?,?)",
                    (new_key, old, new),
                )
                conn.execute(
                    "DELETE FROM workspace_aliases "
                    "WHERE alias_workspace=? AND canonical=? AND status='confirmed'",
                    (fwd_key, old),
                )
                self._install_workspace_redirect_on_conn(conn, old, new)
            return updated, repoint_warnings, True
        except OSError as exc:
            # The advisory flock itself is unavailable (e.g. <db>.startup.lock
            # is a directory, or the database directory is read-only) — report
            # a structured warning like normalize does instead of letting the
            # OSError escape.
            return 0, [f"workspace migration lock unavailable: {exc}"], False
        except sqlite3.Error as exc:
            return 0, [f"rename_workspace_canonical failed: {exc}"], False

    @staticmethod
    def _merge_workspace_core_on_conn(
        conn: sqlite3.Connection,
        from_ws: str,
        to_ws: str,
        *,
        to_embedding: list[float] | None = None,
        db: "MemoryDB | None" = None,
    ) -> tuple[int, list[str], bool]:
        """Merge canonical ``from_ws`` into ``to_ws`` on an open write connection.

        Full merge suite shared by ``migrate_workspace`` and
        ``normalize_workspace_canonicals``: re-point memories (COALESCE raw
        fallback) and conflicts, register the winner, optionally publish the
        winner vector (a publish failure lands in warnings and never aborts the
        merge), drop the loser canonical row and its vec row, re-point alias
        targets, clear the self-referencing alias rows, and install the
        loser→winner redirect. The caller holds the write transaction (and the
        startup flock); guards such as default-insulation or the competing-move
        check stay with the caller.

        A5（0.17.1 修复批）：第三元素 ``committed``——本函数自己的两道前置
        守卫（slot/sha 冲突）也在 UPDATE 之前 return，故它们必须报
        committed=False（此前调用方以 not warnings 推断，这两条恰好一致；
        A5 改为显式返回，语义不再依赖警告文案）。
        """
        warnings: list[str] = []
        # A9（0.17.1 修复批）：注册目标不得是保护桶变体——本函数是
        # migrate 与 normalize 两条路径的公共注册点（后者绕过 migrate 的
        # 入口检查），故在此兜底（拒绝而非静默跳过：调用方以 warnings 归因）。
        from ..twin_redirect import protected_bucket_variant

        if protected_bucket_variant(to_ws):
            return 0, [
                f"workspace {to_ws!r} is a protected-bucket spelling variant and "
                "cannot be a merge target; use the canonical name"
            ], False
        # Guard shared with rename: a slot collision would abort the bulk
        # conflict re-point mid-transaction, and auto-resolving either side
        # would fabricate a triage decision — refuse instead.
        collision = _WsAliasMixin._conflict_slot_collision_warning_on_conn(conn, from_ws, to_ws)
        if collision is not None:
            return 0, collision, False
        sha_collision = _WsAliasMixin._content_sha_collision_warning_on_conn(
            conn, to_ws, from_ws=from_ws,
        )
        if sha_collision is not None:
            return 0, [sha_collision], False
        alias_key = _normalize_alias_key(from_ws)
        to_key = _normalize_alias_key(to_ws)
        cur = conn.execute(
            "UPDATE memories SET workspace_canonical = ?, scan_watermark = NULL "
            "WHERE COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?",
            (to_ws, from_ws),
        )
        updated = cur.rowcount or 0
        if updated and db is not None:
            # Same linked-df fingerprint blind spot as rename/move (staticmethod:
            # the caller passes the store's db handle).
            db.invalidate_linked_df_cache()
        conn.execute(
            "UPDATE conflicts SET workspace_canonical=? WHERE workspace_canonical=?",
            (to_ws, from_ws),
        )
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
            (to_ws, utc_now_iso()),
        )
        if to_embedding is not None:
            # vec0 ignores OR IGNORE (mema #794): merging into a target that
            # already owns a vector raised UNIQUE and produced a misleading
            # "retry after sqlite-vec recovers" warning. Probe first — the
            # merge never renames the target, so an existing vector stays
            # authoritative and is kept as-is (post-commit probe pattern).
            try:
                row = conn.execute(
                    "SELECT c.id, v.id AS vector_id "
                    "FROM workspace_canonicals c "
                    "LEFT JOIN workspace_canonicals_vec v ON v.id = c.id "
                    "WHERE c.name = ?",
                    (to_ws,),
                ).fetchone()
            except sqlite3.Error as exc:
                row = None
                warnings.append(
                    f"workspace canonical vector probe failed for {to_ws!r}: {exc}"
                )
            if row is not None and row["vector_id"] is None:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                        (int(row["id"]), json.dumps(to_embedding)),
                    )
                except sqlite3.Error as exc:
                    warnings.append(
                        f"workspace canonical vector publish failed for {to_ws!r}; retry a write using this workspace after sqlite-vec and embedding configuration recover: {exc}"
                    )
        from_row = conn.execute(
            "SELECT id FROM workspace_canonicals WHERE name = ?", (from_ws,)
        ).fetchone()
        if from_row is not None:
            try:
                conn.execute(
                    "DELETE FROM workspace_canonicals_vec WHERE id = ?",
                    (int(from_row["id"]),),
                )
            except sqlite3.Error:
                pass  # vec table may be absent; canonical delete still proceeds
            conn.execute("DELETE FROM workspace_canonicals WHERE name = ?", (from_ws,))
        warnings.extend(_WsAliasMixin._repoint_workspace_targets_on_conn(
            conn, from_ws, to_ws, exclude_aliases=(alias_key, to_key),
        ))
        conn.execute(
            "DELETE FROM workspace_aliases "
            "WHERE alias_workspace=? AND canonical IN (?,?)",
            (to_key, from_ws, to_ws),
        )
        conn.execute(
            "DELETE FROM workspace_aliases "
            "WHERE alias_workspace=? AND canonical=? AND status='confirmed'",
            (alias_key, from_ws),
        )
        _WsAliasMixin._install_workspace_redirect_on_conn(conn, from_ws, to_ws)
        return updated, warnings, True

    def migrate_workspace(
        self, from_ws: str, to_ws: str, *, embedder: Any = None,
    ) -> tuple[int, list[str], bool]:
        """Merge one canonical into another and keep old-name forwarding stable.

        A5（0.17.1 修复批）：第三元素 ``committed`` 同 rename——事务已越过
        前置守卫并提交（含 no-op/自折叠；空源桶同样合法 committed=True，
        它仍会删 loser canonical 行并装 redirect）。
        """
        if not self._db_available or not self.state.sqlite_writable:
            return 0, ["SQLite write unavailable; migrate skipped."], False
        from_ws = _coerce_ws(from_ws)
        to_ws = _coerce_ws(to_ws)
        if not from_ws or not to_ws:
            return 0, ["migrate requires non-empty from and to workspace."], False
        # migrate is a merge-into path like rename; default stays
        # reserved in both directions.
        if is_default_workspace_term(from_ws) or is_default_workspace_term(to_ws):
            return 0, [
                "default is a reserved global pool and cannot be merged in either "
                "direction; migrate requires two non-default workspace names."
            ], False
        if from_ws == to_ws:
            return 0, [], True
        # A9（0.17.1 修复批）：同 rename——目标名不得是保护桶变体。
        from ..twin_redirect import protected_bucket_variant

        if protected_bucket_variant(to_ws):
            return 0, [
                f"workspace {to_ws!r} is a protected-bucket spelling variant and "
                "cannot be a migrate target; use the canonical name"
            ], False
        # Destination orthography: when the destination is a mechanical variant
        # of an already-registered canonical, merge into the registered
        # spelling. Re-pointing memories to the verbatim variant while the
        # redirect normalizes to the registered twin would split one canonical
        # across two spellings and double-register it.
        try:
            with self.connection() as conn:
                registered = self._mechanical_canonical_on_conn(conn, to_ws)
        except sqlite3.Error:
            registered = None
        if registered is not None:
            to_ws = registered
            # The fold can land on a registered default spelling; keep the
            # reserved pool refused in both directions (mirrors the guard).
            if is_default_workspace_term(to_ws):
                return 0, [
                    "default is a reserved global pool and cannot be merged in either "
                    "direction; migrate requires two non-default workspace names."
                ], False
            if from_ws == to_ws:
                # migrate('AgentLane', 'agent-lane') folds the destination back
                # onto the source: a self-merge no-op (executing it would
                # delete the winner canonical row).
                return 0, [], True
        # Embedding for the destination canonical, computed outside the txn.
        to_embedding = None
        if embedder is not None and self.state.sqlite_vec_available:
            try:
                er = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=to_ws)
                to_embedding = list(er.embedding) if er and er.embedding else None
            except Exception:
                to_embedding = None
        try:
            # Same advisory flock as rename/normalize, always before the write
            # transaction so the three paths serialize in one order.
            with database_startup_lock(self.settings.db_path), self.write_transaction() as conn:
                competing = self._competing_move_warning_on_conn(conn, from_ws, to_ws)
                if competing is not None:
                    return 0, competing, False
                updated, publish_warnings, merged = self._merge_workspace_core_on_conn(
                    conn, from_ws, to_ws, to_embedding=to_embedding, db=self._db,
                )
            return updated, publish_warnings, merged
        except OSError as exc:
            # Same advisory-flock failure mode as rename/normalize: report a
            # structured warning instead of letting the OSError escape.
            return 0, [f"workspace migration lock unavailable: {exc}"], False
        except sqlite3.Error as exc:
            return 0, [f"migrate_workspace failed: {exc}"], False

    def normalize_workspace_canonicals(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Fold registered spelling variants of one canonical into its first-seen row.

        Stock migration for legacy double-registration (AgentLane / agent-lane /
        agent_lane). Grouping uses ``_normalize_ws_group_key``, which is
        deliberately STRICTER than the ``_mechanical_ws_key`` used by the
        resolver/decision primitive/migrate: it lowercases instead of
        casefolding, so 'Straße' vs 'strasse' (or the 'ﬁ' ligature vs 'file')
        are NOT treated as spelling variants. Normalize is a bulk destructive
        operation — its grouping key errs toward not merging (a missed variant
        is recoverable by re-running, a wrong merge is not). The whole
        scan+merge runs under the same advisory flock as rename/migrate (taken
        before any write transaction, in that order). ``dry_run=True`` (the
        default) only reads inside the flock — no write transaction is opened —
        and returns the plan; ``dry_run=False`` executes every merge plus the
        rejected-canonical normalization in ONE write transaction. A second run
        is a no-op.

        Concurrency honesty: the write-path resolver registers canonicals
        OUTSIDE any process-level lock, so a fresh variant double-registration
        window can still exist after this migration. It is converged by
        first-seen wins on later resolves plus re-running this cleanup; the
        window is NOT claimed closed.

        Returns {ok, dry_run, groups, merged, rejected_normalized, skipped,
        warnings} where groups are the multi-member mechanical groups
        ({key, winner, losers, skipped}), merged the executed/planned
        loser→winner merges ({from, to, memories_updated}),
        rejected_normalized the rejected alias rows whose canonical spelling
        was aligned to the registered twin, and skipped the pairs/groups left
        untouched with a reason (an explicit user rejection in ANY spelling of
        the pair, the default pool) plus third-party confirmed redirects the
        merge would silently drop behind a same-alias rejection
        (confirmed_redirect_shadowed_by_rejection — reported AND merged, with
        the rejection winning).
        """
        result: dict[str, Any] = {
            "ok": True,
            "dry_run": bool(dry_run),
            "groups": [],
            "merged": [],
            "rejected_normalized": [],
            "skipped": [],
            "warnings": [],
        }
        if not self._db_available:
            result["ok"] = False
            result["warnings"].append("SQLite unavailable; normalize skipped.")
            return result
        if not dry_run and not self.state.sqlite_writable:
            result["ok"] = False
            result["warnings"].append("SQLite write unavailable; normalize skipped.")
            return result
        try:
            with database_startup_lock(self.settings.db_path):
                if dry_run:
                    # Plan-only: read inside the flock, never open a write txn.
                    with self.connection() as conn:
                        self._plan_workspace_normalization_on_conn(conn, result, execute=False)
                else:
                    with self.write_transaction() as conn:
                        self._plan_workspace_normalization_on_conn(conn, result, execute=True)
        except OSError as exc:
            # The advisory flock itself is unavailable (e.g. a read-only
            # database directory makes os.open fail) — report a structured
            # result on the dry-run and the execute path alike instead of
            # letting the OSError escape.
            result["ok"] = False
            result["error"] = f"workspace normalization lock unavailable: {exc}"
        except sqlite3.Error as exc:
            result["ok"] = False
            result["warnings"].append(f"normalize_workspace_canonicals failed: {exc}")
        return result

    def _plan_workspace_normalization_on_conn(
        self, conn: sqlite3.Connection, result: dict[str, Any], *, execute: bool,
    ) -> None:
        """Single-source plan (and optional execution) for normalize.

        ``execute=False`` computes the identical plan read-only so a dry run
        reports exactly what the real run would do: the respected-rejection
        and shadowed-redirect checks reason over the same alias-row snapshot,
        and the rejected-only phase below tracks the targets it has already
        planned, so two drifted rows folding to one registered spelling report
        rewrite + dropped_duplicate in both modes.
        """
        rows = conn.execute(
            "SELECT id, name FROM workspace_canonicals ORDER BY id ASC"
        ).fetchall()
        groups: dict[str, list[tuple[int, str]]] = {}
        for row in rows:
            name = str(row["name"])
            key = _normalize_ws_group_key(name)
            if not key:
                continue
            groups.setdefault(key, []).append((int(row["id"]), name))

        # Snapshot the small governance table once for the respected-rejection
        # and shadowed-redirect checks. One group's merge can never rewrite a
        # row relevant to another group (their group keys are disjoint), so
        # the pre-merge snapshot stays valid for every group.
        alias_rows = conn.execute(
            "SELECT alias_workspace, canonical, status FROM workspace_aliases"
        ).fetchall()
        rejected_rows = [row for row in alias_rows if str(row["status"]) == "rejected"]

        now = utc_now_iso()
        for key, members in groups.items():
            if len(members) < 2:
                continue
            # id ASC == first-seen: the earliest registered spelling wins.
            winner = members[0][1]
            losers = [name for _id, name in members[1:]]
            if any(is_default_workspace_term(name) for _id, name in members):
                # The default global pool is bidirectionally insulated; never
                # merge a reserved term in either direction.
                result["groups"].append({
                    "key": key, "winner": None,
                    "losers": [name for _id, name in members],
                    "skipped": True,
                    "reason": "group contains a reserved default term; default stays unmerged.",
                })
                result["skipped"].append({
                    "key": key,
                    "members": [name for _id, name in members],
                    "reason": "default_reserved",
                })
                continue
            result["groups"].append({
                "key": key, "winner": winner, "losers": losers, "skipped": False,
            })
            # Respect an explicit user rejection touching this group in ANY
            # spelling: a rejected row whose alias AND canonical both fold to
            # the group's key keeps the WHOLE group separate. This covers the
            # loser→winner and winner→loser rows and — unlike exact-key
            # lookups, which would miss it — a row recorded under a third
            # spelling of the pair (the resolve-refusal flow's typical
            # product, e.g. 'agent_lane'→'AgentLane'); merging on top of it
            # would leave a confirmed redirect and the rejection side by side.
            respected: dict[str, str] | None = None
            for row in rejected_rows:
                alias_spelling = str(row["alias_workspace"])
                rejected_canonical = str(row["canonical"])
                if (
                    _normalize_ws_group_key(alias_spelling) != key
                    or _normalize_ws_group_key(rejected_canonical) != key
                ):
                    continue
                if alias_spelling == _normalize_alias_key(winner):
                    direction = "winner_to_loser"
                elif alias_spelling in {_normalize_alias_key(loser) for loser in losers}:
                    direction = "loser_to_winner"
                else:
                    direction = "cross_spelling"
                respected = {
                    "direction": direction,
                    "alias_workspace": alias_spelling,
                    "canonical": rejected_canonical,
                }
                break
            if respected is not None:
                for loser in losers:
                    result["skipped"].append({
                        "from": loser,
                        "to": winner,
                        "direction": respected["direction"],
                        "rejected_alias_workspace": respected["alias_workspace"],
                        "rejected_canonical": respected["canonical"],
                        "reason": "rejected_pair_respected: user explicitly kept this pair separate.",
                    })
                continue
            # Surface any third-party confirmed redirect the merge would
            # silently drop: re-pointing (alias→loser, confirmed) copies it to
            # (alias→winner) with INSERT OR IGNORE, so a rejected row under
            # the SAME alias whose canonical folds into this group wins the
            # PRIMARY KEY and the user's confirmed decision evaporates (the
            # alias would later re-register as a new canonical — the double
            # registration this migration exists to cure). The merge still
            # runs (the rejection is the conservative outcome); the collision
            # must be visible.
            shadowed_reported: set[str] = set()
            for row in alias_rows:
                if str(row["status"]) != "confirmed":
                    continue
                alias_spelling = str(row["alias_workspace"])
                confirmed_canonical = str(row["canonical"])
                if (
                    _normalize_ws_group_key(alias_spelling) == key
                    or _normalize_ws_group_key(confirmed_canonical) != key
                    or alias_spelling in shadowed_reported
                ):
                    continue
                blocking = next(
                    (
                        str(rejected["canonical"])
                        for rejected in rejected_rows
                        if str(rejected["alias_workspace"]) == alias_spelling
                        and _normalize_ws_group_key(str(rejected["canonical"])) == key
                    ),
                    None,
                )
                if blocking is None:
                    continue
                shadowed_reported.add(alias_spelling)
                result["skipped"].append({
                    "type": "confirmed_redirect_shadowed_by_rejection",
                    "alias_workspace": alias_spelling,
                    "confirmed_canonical": confirmed_canonical,
                    "rejected_canonical": blocking,
                    "key": key,
                    "reason": (
                        "confirmed redirect and rejected decision collide under one "
                        "alias; the merge proceeds and the rejection wins the "
                        "(alias, winner) row."
                    ),
                })
                result["warnings"].append(
                    f"workspace alias {alias_spelling!r}: confirmed redirect to "
                    f"{confirmed_canonical!r} is shadowed by the rejected row for "
                    f"{blocking!r}; merging group {key!r} keeps the rejection and "
                    "drops the confirmed redirect."
                )
            for loser in losers:
                if execute:
                    updated, merge_warnings, merged = self._merge_workspace_core_on_conn(
                        conn, loser, winner, db=self._db,
                    )
                    result["warnings"].extend(merge_warnings)
                    if merge_warnings and not merged:
                        # The merge was refused (e.g. active-conflict slot
                        # collision) — report it as skipped-with-reason, not
                        # as a zero-row merge that silently did not happen.
                        result["skipped"].append({
                            "type": "merge_refused",
                            "from": loser,
                            "to": winner,
                            "reason": " ".join(merge_warnings),
                        })
                        continue
                    # merged=True with warnings (A5 committed semantics, e.g.
                    # repoint dropped a mechanical-twin rejected row): the
                    # merge HAPPENED and must be reported as such — falling
                    # into the refused branch here would report the merged
                    # group as skipped, contradicting both the dry-run plan
                    # and the committed library state.
                else:
                    # Plan honesty: the same read-only collision guard runs
                    # in dry-run so the plan reports a would-be refusal as
                    # skipped, not as a merge the execute pass must refuse.
                    collision = self._conflict_slot_collision_warning_on_conn(conn, loser, winner)
                    if collision is not None:
                        result["skipped"].append({
                            "type": "merge_refused",
                            "from": loser,
                            "to": winner,
                            "reason": " ".join(collision),
                        })
                        continue
                    updated = int(conn.execute(
                        "SELECT COUNT(*) AS c FROM memories "
                        "WHERE COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?",
                        (loser,),
                    ).fetchone()["c"])
                result["merged"].append({
                    "from": loser, "to": winner, "memories_updated": updated,
                })

        # Rejected-only normalization: a rejected target was never registered,
        # so its stored spelling may drift from the registered mechanical twin
        # (rejected 'project-x' vs registered 'ProjectX'). Align the spelling so
        # later suppression/expansion matches consistently.
        registered: dict[str, str] = {}
        for row in conn.execute("SELECT name FROM workspace_canonicals ORDER BY id ASC"):
            name = str(row["name"])
            registered.setdefault(_normalize_ws_group_key(name), name)
        drifted_rows = conn.execute(
            "SELECT alias_workspace, canonical FROM workspace_aliases "
            "WHERE status='rejected' ORDER BY alias_workspace, canonical"
        ).fetchall()
        # (alias, canonical) targets this phase has already rewritten (or, in
        # dry-run, plans to rewrite). The duplicate check consults it alongside
        # the table so a dry run reports the same rewrite + dropped_duplicate
        # sequence as the real run instead of claiming two physically
        # impossible rewrites to one PRIMARY KEY.
        planned_targets: set[tuple[str, str]] = set()
        for row in drifted_rows:
            alias = str(row["alias_workspace"])
            canonical = str(row["canonical"])
            twin = registered.get(_normalize_ws_group_key(canonical))
            if twin is None or twin == canonical:
                continue
            duplicate = (alias, twin) in planned_targets or conn.execute(
                "SELECT 1 FROM workspace_aliases WHERE alias_workspace=? AND canonical=?",
                (alias, twin),
            ).fetchone() is not None
            if execute:
                if duplicate:
                    # (alias, registered spelling) already exists — drop the
                    # drifted row instead of colliding with the PRIMARY KEY.
                    conn.execute(
                        "DELETE FROM workspace_aliases "
                        "WHERE alias_workspace=? AND canonical=? AND status='rejected'",
                        (alias, canonical),
                    )
                else:
                    conn.execute(
                        "UPDATE workspace_aliases SET canonical=?, updated_at=? "
                        "WHERE alias_workspace=? AND canonical=? AND status='rejected'",
                        (twin, now, alias, canonical),
                    )
            if not duplicate:
                planned_targets.add((alias, twin))
            result["rejected_normalized"].append({
                "alias_workspace": alias,
                "from": canonical,
                "to": twin,
                "action": "dropped_duplicate" if duplicate else "rewritten",
            })
