"""Workspace canonicalization and alias governance for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Optional, Tuple, TYPE_CHECKING

from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


def _normalize_alias_key(ws: Optional[str]) -> str:
    """Normalize a workspace string into a stable alias-governance key.

    Case-folded + whitespace-collapsed so "金营项目 " and "金营项目" map to the
    same alias row. This is only the governance lookup key; the display
    canonical is stored verbatim in workspace_aliases.canonical. Non-string
    inputs (loosely-typed MCP JSON) are coerced to str rather than crashing.
    """
    if ws is None:
        return ""
    s = ws.strip() if isinstance(ws, str) else str(ws).strip()
    if not s:
        return ""
    return " ".join(s.split()).casefold()


def _coerce_ws(ws: Any) -> str:
    """Coerce a possibly-non-string workspace value to a trimmed str.

    MCP clients send loosely-typed JSON; alias/canonical may arrive as int/
    list/dict. Coerce rather than raise AttributeError on ``.strip()``.
    """
    if ws is None:
        return ""
    return ws.strip() if isinstance(ws, str) else str(ws).strip()

class WorkspaceStore:
    def __init__(self, db: "MemoryDB"):
        self._db = db

    def __getattr__(self, name: str) -> Any:
        # Proxy to MemoryDB for connection/write_transaction/state/settings and
        # cross-store helpers used by the copied methods. This keeps the Phase 3
        # extraction close to a pure move; later hardening can tighten the seam.
        return getattr(self._db, name)

    def _publish_missing_workspace_canonical_vector(
        self,
        canonical: str,
        embedder: Any,
        result: dict[str, Any],
    ) -> None:
        """Idempotently backfill a missing canonical vector on write paths.

        The existence probe and embedding happen before the short write
        transaction. The transaction rechecks the vector row so concurrent
        retries cannot replace an already-published vector.
        """
        if not (
            canonical
            and embedder is not None
            and self.state.sqlite_writable
            and self.state.sqlite_vec_available
        ):
            return
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT c.id, v.id AS vector_id "
                    "FROM workspace_canonicals c "
                    "LEFT JOIN workspace_canonicals_vec v ON v.id = c.id "
                    "WHERE c.name = ?",
                    (canonical,),
                ).fetchone()
        except sqlite3.Error:
            return
        if row is not None and row["vector_id"] is not None:
            return

        try:
            er = embedder.embed_text(prefix="", body=canonical)
            embedding = list(er.embedding) if er and er.embedding else None
        except Exception:
            embedding = None
        if not embedding:
            return

        try:
            with self.write_transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
                    (canonical, utc_now_iso()),
                )
                canonical_row = conn.execute(
                    "SELECT c.id, v.id AS vector_id "
                    "FROM workspace_canonicals c "
                    "LEFT JOIN workspace_canonicals_vec v ON v.id = c.id "
                    "WHERE c.name = ?",
                    (canonical,),
                ).fetchone()
                if canonical_row is not None and canonical_row["vector_id"] is None:
                    conn.execute(
                        "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                        (int(canonical_row["id"]), json.dumps(embedding)),
                    )
        except sqlite3.Error as exc:
            result["vector_publish_pending"] = True
            result["warnings"].append(
                f"workspace canonical vector publish failed for {canonical!r}; retry a write using this workspace after sqlite-vec and embedding configuration recover: {exc}"
            )

    def prepare_missing_workspace_canonical_embedding(
        self,
        canonical: str,
        embedder: Any = None,
    ) -> Optional[list[float]]:
        """Embed canonical text only when its derived vector is missing.

        Both the existence probe and model call happen before the authoritative
        memory write transaction. A concurrent publisher is harmless because the
        post-commit publication uses INSERT OR IGNORE.
        """
        canonical = _coerce_ws(canonical)
        if not (
            canonical
            and embedder is not None
            and self.state.sqlite_writable
            and self.state.sqlite_vec_available
        ):
            return None
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT v.id AS vector_id FROM workspace_canonicals c "
                    "LEFT JOIN workspace_canonicals_vec v ON v.id = c.id "
                    "WHERE c.name = ?",
                    (canonical,),
                ).fetchone()
            if row is not None and row["vector_id"] is not None:
                return None
        except sqlite3.Error:
            return None
        try:
            er = embedder.embed_text(prefix="", body=canonical)
            return list(er.embedding) if er and er.embedding else None
        except Exception:
            return None

    def publish_workspace_canonical_vector(
        self,
        canonical: str,
        embedding: Optional[list[float]],
    ) -> list[str]:
        """Publish a prepared canonical vector after the canonical write commits.

        Canonical registration and the memory row are the authoritative atomic
        transaction. The vector is a derived index: a failure here must leave
        that committed write successful and return an actionable warning.
        """
        if not embedding or not self.state.sqlite_vec_available:
            return []
        try:
            with self.write_transaction() as conn:
                row = conn.execute(
                    "SELECT id FROM workspace_canonicals WHERE name = ?", (canonical,)
                ).fetchone()
                if row is None:
                    return []
                conn.execute(
                    "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                    (int(row["id"]), json.dumps(embedding)),
                )
            return []
        except sqlite3.Error as exc:
            return [
                f"workspace canonical vector publish failed for {canonical!r}; "
                "retry a write using this workspace after sqlite-vec and embedding "
                f"configuration recover: {exc}"
            ]

    def resolve_workspace_canonical(
        self,
        ws_raw: Optional[str],
        embedder: Any = None,
        *,
        match_distance: Optional[float] = None,
        register_new: bool = True,
    ) -> dict[str, Any]:
        """Resolve a raw workspace string to its canonical name (alias merge).

        Strategy (double-store: raw stays in memories.workspace, resolved name
        goes to memories.workspace_canonical):
          1. Exact match against workspace_canonicals.name → reuse it.
          2. If an embedder + sqlite-vec are available, embed the raw string and
             KNN against workspace_canonicals_vec; if the nearest canonical is
             within ``match_distance`` reuse it (handles 金营项目 / 金科营销项目).
          3. Otherwise it is a NEW canonical. When ``register_new`` is True, the
             raw string is registered as a new canonical (+ its vector); when
             False (read/query path) nothing is written.

        Returns a dict:
          {canonical, is_new, matched_by: exact|vector|new|fallback,
           distance, similar: [{name, distance}, ...]}

        Never raises — degrades to exact string identity so callers can rely on
        a canonical always coming back (falls back to the raw string itself).
        """
        raw = (ws_raw or "").strip()
        result: dict[str, Any] = {
            "canonical": raw or "default",
            "is_new": False,
            "matched_by": "fallback",
            "distance": None,
            "similar": [],
            "rejected_canonicals": [],
            "warnings": [],
            "vector_publish_pending": False,
            # Prepared outside the eventual memory write transaction. Callers
            # may publish it only for the final canonical selected by policy.
            "candidate_embedding": None,
        }
        if not raw:
            result["canonical"] = "default"
            return result
        if not self._db_available:
            return result
        if match_distance is None:
            # Explicit None check (not truthy fallback): 0.0 is a legitimate
            # "exact-vector-only" setting and must not be swallowed to 0.25.
            configured = self.settings.workspace_match_distance
            match_distance = float(0.25 if configured is None else configured)

        try:
            with self.connection() as conn:
                # 0. Confirmed/rejected alias governance (design 637 / 636 §1,8).
                #    A confirmed alias short-circuits everything (no vector, no
                #    Qwen). Rejected pairs are recorded so downstream candidate
                #    ranking can filter them out and suppress repeat prompts.
                alias_key = _normalize_alias_key(raw)
                try:
                    arow = conn.execute(
                        "SELECT canonical, status FROM workspace_aliases "
                        "WHERE alias_workspace = ? "
                        "ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1",
                        (alias_key,),
                    ).fetchone()
                except sqlite3.Error:
                    arow = None
                if arow is not None:
                    if str(arow["status"]) == "confirmed":
                        result.update({
                            "canonical": arow["canonical"],
                            "is_new": False,
                            "matched_by": "confirmed_alias",
                            "distance": 0.0,
                        })
                        if register_new:
                            self._publish_missing_workspace_canonical_vector(
                                str(arow["canonical"]), embedder, result,
                            )
                        return result
                    if str(arow["status"]) == "rejected":
                        # Rejections accumulate per (raw, canonical).  Confirmed
                        # aliases have singular precedence above; absent one, skip
                        # every explicitly rejected target during candidate ranking.
                        rejected_rows = conn.execute(
                            "SELECT canonical FROM workspace_aliases "
                            "WHERE alias_workspace=? AND status='rejected'",
                            (alias_key,),
                        ).fetchall()
                        result["rejected_canonicals"] = [row["canonical"] for row in rejected_rows]

                # 1. Exact canonical hit.
                exact = conn.execute(
                    "SELECT id, name FROM workspace_canonicals WHERE name = ?",
                    (raw,),
                ).fetchone()
                if exact:
                    result.update({"canonical": exact["name"], "is_new": False, "matched_by": "exact", "distance": 0.0})
                    # A prior new-canonical write may have registered the canonical
                    # while vector publication was unavailable. A later write to the
                    # exact same workspace is the executable retry path once vec and
                    # the embedder are healthy again.
                    if register_new and self.state.sqlite_writable and self.state.sqlite_vec_available and embedder is not None:
                        try:
                            er = embedder.embed_text(prefix="", body=raw)
                            exact_embedding = list(er.embedding) if er and er.embedding else None
                        except Exception:
                            exact_embedding = None
                        if exact_embedding:
                            try:
                                conn.execute(
                                    "INSERT OR REPLACE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                                    (int(exact["id"]), json.dumps(exact_embedding)),
                                )
                                conn.commit()
                            except sqlite3.Error as exc:
                                result["vector_publish_pending"] = True
                                result["warnings"].append(
                                    f"workspace canonical vector publish failed for {raw!r}; retry a write using this workspace after sqlite-vec and embedding configuration recover: {exc}"
                                )
                    return result

                # 2. Vector nearest-canonical (only when embedding is available).
                vec_ok = self.state.sqlite_vec_available and embedder is not None
                embedding = None
                if vec_ok:
                    try:
                        er = embedder.embed_text(prefix="", body=raw)
                        embedding = list(er.embedding) if er and er.embedding else None
                    except Exception:
                        embedding = None
                if embedding:
                    result["candidate_embedding"] = embedding
                    try:
                        query_json = json.dumps(embedding)
                        # Full-scan cosine (not MATCH/L2): the canonical table is
                        # tiny (one row per project) and embeddinggemma vectors are
                        # unnormalized, so cosine is the scale-invariant choice —
                        # sqlite-vec returns cosine distance for this index.
                        rows = conn.execute(
                            """SELECT c.name AS name,
                                      vec_distance_cosine(v.embedding, ?) AS distance
                               FROM workspace_canonicals_vec v
                               JOIN workspace_canonicals c ON c.id = v.id
                               ORDER BY distance
                               LIMIT 5""",
                            (query_json,),
                        ).fetchall()
                        # Skip any canonical the user has explicitly rejected for
                        # this alias (636 §4): the nearest non-rejected candidate
                        # within threshold wins. Also drop rejected names from the
                        # returned `similar` list so downstream write-hints never
                        # re-surface a pair the user already rejected.
                        rejected = set(result.get("rejected_canonicals") or [])
                        result["similar"] = [
                            {"name": r["name"], "distance": float(r["distance"])}
                            for r in rows if r["name"] not in rejected
                        ]
                        best = next(
                            (r for r in rows if r["name"] not in rejected),
                            None,
                        )
                        if best is not None and float(best["distance"]) <= match_distance:
                            result.update({
                                "canonical": best["name"],
                                "is_new": False,
                                "matched_by": "vector",
                                "distance": float(best["distance"]),
                            })
                            return result
                    except sqlite3.Error:
                        pass  # vec query failed — fall through to new-canonical path

                # 3. New canonical.
                result.update({"canonical": raw, "is_new": True, "matched_by": "new"})
                if register_new and self.state.sqlite_writable:
                    try:
                        now = utc_now_iso()
                        cur = conn.execute(
                            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
                            (raw, now),
                        )
                        row = conn.execute(
                            "SELECT id FROM workspace_canonicals WHERE name = ?", (raw,)
                        ).fetchone()
                        if row and embedding and self.state.sqlite_vec_available:
                            try:
                                conn.execute(
                                    "INSERT OR REPLACE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                                    (int(row["id"]), json.dumps(embedding)),
                                )
                            except sqlite3.Error as exc:
                                result["vector_publish_pending"] = True
                                result["warnings"].append(
                                    f"workspace canonical vector publish failed for {raw!r}; retry a write using this workspace after sqlite-vec and embedding configuration recover: {exc}"
                                )
                        conn.commit()
                    except sqlite3.Error as exc:
                        result["warnings"].append(f"workspace canonical registration failed for {raw!r}: {exc}")
                return result
        except sqlite3.Error:
            return result

    # ------------------------------------------------------------------
    #  Workspace alias governance (design 637). Current-state table +
    #  append-only event log + UNIQUE + single transaction. NO CAS.
    # ------------------------------------------------------------------
    def upsert_workspace_alias_on_conn(
        self,
        conn: sqlite3.Connection,
        alias: str,
        canonical: str,
        *,
        relation: str = "alias",
        status: str = "confirmed",
        source: str = "user",
        action: str = "accept",
        judge_type: str = "user",
        reason: Optional[str] = None,
        force: bool = False,
    ) -> Tuple[bool, list[str]]:
        alias_key = _normalize_alias_key(alias)
        if not alias_key:
            return False, ["alias must be a non-empty workspace string."]
        canonical = _coerce_ws(canonical)
        if not canonical:
            return False, ["canonical must be a non-empty workspace string."]
        if status not in {"confirmed", "rejected"}:
            return False, [f"status={status!r} invalid; expected confirmed|rejected."]
        now = utc_now_iso()
        pair = conn.execute(
            "SELECT canonical, status FROM workspace_aliases "
            "WHERE alias_workspace=? AND canonical=?",
            (alias_key, canonical),
        ).fetchone()
        confirmed = conn.execute(
            "SELECT canonical, status FROM workspace_aliases "
            "WHERE alias_workspace=? AND status='confirmed' LIMIT 1",
            (alias_key,),
        ).fetchone()
        prev = confirmed or pair
        old_canonical = prev["canonical"] if prev else None
        old_status = prev["status"] if prev else None
        # Guard only the exact rejected pair. Other rejected targets are retained
        # as independent negative decisions and do not block a singular confirm.
        if pair is not None and pair["status"] == "rejected" and status == "confirmed" and not force:
            return False, [
                f"workspace {alias_key!r} was explicitly rejected as an alias of "
                f"{canonical!r}; refusing to confirm it silently. Pass an "
                "authorized override to change this decision."
            ]
        if status == "confirmed":
            # At most one positive mapping per raw alias. Rejected pairs remain so
            # governance history continues to suppress those alternatives.
            conn.execute(
                "DELETE FROM workspace_aliases WHERE alias_workspace=? AND status='confirmed' AND canonical<>?",
                (alias_key, canonical),
            )
        conn.execute(
            """INSERT INTO workspace_aliases
                 (alias_workspace, canonical, relation, status, source, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(alias_workspace,canonical) DO UPDATE SET
                 relation=excluded.relation,
                 status=excluded.status,
                 source=excluded.source,
                 updated_at=excluded.updated_at""",
            (alias_key, canonical, relation, status, source, now),
        )
        conn.execute(
            """INSERT INTO workspace_alias_events
                 (alias_workspace, old_canonical, new_canonical,
                  old_status, new_status, action, judge_type, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (alias_key, old_canonical, canonical, old_status, status,
             action, judge_type, reason, now),
        )
        # Register the target canonical for confirmed aliases so the
        # resolver's confirmed-alias short-circuit never returns a name
        # that has no row in workspace_canonicals (which would silently
        # disable KNN fuzzy-merge and let a later near-miss re-split).
        # Rejected aliases don't register — a rejection doesn't create
        # a workspace, it only records that this key is NOT that name.
        if status == "confirmed":
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
                (canonical, now),
            )
        return True, []

    def upsert_workspace_alias(
        self,
        alias: str,
        canonical: str,
        *,
        relation: str = "alias",
        status: str = "confirmed",
        source: str = "user",
        action: str = "accept",
        judge_type: str = "user",
        reason: Optional[str] = None,
        force: bool = False,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Tuple[bool, list[str]]:
        """Upsert the current alias state AND append an audit event in one txn.

        Consistency comes from the UNIQUE(alias_workspace) constraint + the
        transaction, not from a version/CAS column (637). A concurrent writer
        that loses the race hits UNIQUE and is expected to retry; the state
        machine converges naturally.

        A prior *rejection* is never silently flipped to confirmed: writing a
        ``confirmed`` alias over an existing ``rejected`` row for the same key is
        refused (returns ok=False + warning) unless ``force=True``. This mirrors
        rename_workspace_canonical's rejection-preserving guard so no governance
        path can reverse a user's explicit "keep separate" decision by accident.
        Re-asserting a rejection, or overriding with force, is always allowed.
        """
        if conn is not None:
            return self.upsert_workspace_alias_on_conn(
                conn,
                alias,
                canonical,
                relation=relation,
                status=status,
                source=source,
                action=action,
                judge_type=judge_type,
                reason=reason,
                force=force,
            )
        if not self._db_available or not self.state.sqlite_writable:
            return False, ["SQLite write unavailable; workspace alias not written."]
        try:
            # write_transaction (BEGIN IMMEDIATE) so the read-then-write of the
            # predecessor state is serialized: concurrent writers block here
            # instead of both reading the same old_canonical and writing an
            # event log whose predecessor snapshot can't reconstruct the chain.
            with self.write_transaction() as txn_conn:
                return self.upsert_workspace_alias_on_conn(
                    txn_conn,
                    alias,
                    canonical,
                    relation=relation,
                    status=status,
                    source=source,
                    action=action,
                    judge_type=judge_type,
                    reason=reason,
                    force=force,
                )
        except sqlite3.Error as exc:
            return False, [f"upsert_workspace_alias failed: {exc}"]

    def get_workspace_alias(self, alias: str) -> Optional[dict[str, Any]]:
        """O(1) current-state lookup used by the resolver."""
        if not self._db_available:
            return None
        alias_key = _normalize_alias_key(alias)
        if not alias_key:
            return None
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT alias_workspace, canonical, relation, status, source, updated_at "
                    "FROM workspace_aliases WHERE alias_workspace = ? "
                    "ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1",
                    (alias_key,),
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.Error:
            return None

    def list_workspace_alias_events(
        self, alias: Optional[str] = None, *, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Read the append-only audit trail (optionally filtered by alias)."""
        if not self._db_available:
            return []
        try:
            with self.connection() as conn:
                if alias:
                    rows = conn.execute(
                        "SELECT * FROM workspace_alias_events WHERE alias_workspace = ? "
                        "ORDER BY created_at DESC, id DESC LIMIT ?",
                        (_normalize_alias_key(alias), int(limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM workspace_alias_events "
                        "ORDER BY created_at DESC, id DESC LIMIT ?",
                        (int(limit),),
                    ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    def rename_workspace_canonical(
        self, old: str, new: str, *, judge_type: str = "user", reason: Optional[str] = None
    ) -> Tuple[int, list[str]]:
        """Rename a canonical everywhere: memories + canonicals table + audit.

        Returns (rows_updated, warnings). Best-effort on the vec table (kept in
        sync via workspace_canonicals.id, untouched here).
        """
        if not self._db_available or not self.state.sqlite_writable:
            return 0, ["SQLite write unavailable; rename skipped."]
        old = _coerce_ws(old)
        new = _coerce_ws(new)
        if not old or not new:
            return 0, ["rename requires non-empty old and new canonical."]
        if old == new:
            return 0, []
        now = utc_now_iso()
        try:
            with self.write_transaction() as conn:
                cur = conn.execute(
                    "UPDATE memories SET workspace_canonical = ? "
                    "WHERE COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?",
                    (new, old),
                )
                updated = cur.rowcount or 0
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
                # Snapshot the forwarding-alias decision BEFORE the blanket
                # repoint, so the guard below reasons about real prior state and
                # not a row the repoint has already moved to `new`.
                fwd_key = _normalize_alias_key(old)
                new_key = _normalize_alias_key(new)
                prior = conn.execute(
                    "SELECT canonical, status FROM workspace_aliases WHERE alias_workspace = ?",
                    (fwd_key,),
                ).fetchone()
                if fwd_key == new_key:
                    # Case/whitespace-only rename (e.g. 'Foo'->'foo'): the alias
                    # KEY is unchanged, so there is nothing to forward and the
                    # normal DELETE-self-alias/forwarding logic would wrongly
                    # destroy the existing row. Only refresh the display-name
                    # (canonical column) on rows that pointed at the old name.
                    conn.execute(
                        "UPDATE workspace_aliases SET canonical = ?, updated_at = ? "
                        "WHERE canonical = ?",
                        (new, now, old),
                    )
                    return updated, []
                # Repoint OTHER aliases that targeted `old` to `new` (rename = the
                # canonical formerly-called-`old` IS now `new`, so both confirmed
                # and rejected references follow). Exclude the forwarding key
                # (handled below) and any row that would become a self-alias
                # ("new is not new"); delete the self-aliasing rows.
                conn.execute(
                    "UPDATE workspace_aliases SET canonical = ?, updated_at = ? "
                    "WHERE canonical = ? AND alias_workspace != ? AND alias_workspace != ?",
                    (new, now, old, fwd_key, new_key),
                )
                conn.execute(
                    "DELETE FROM workspace_aliases WHERE canonical = ? AND alias_workspace = ?",
                    (old, new_key),
                )
                # Insert a forwarding alias normalize(old) -> new so a later write
                # with the OLD raw workspace string resolves to `new` instead of
                # re-registering `old` as a fresh canonical (re-splitting the
                # workspace). NEVER clobber an existing user decision on this key:
                # a rejected row stays rejected; a confirmed row's prior snapshot
                # is recorded truthfully.
                if prior is None:
                    conn.execute(
                        """INSERT INTO workspace_aliases
                             (alias_workspace, canonical, relation, status, source, updated_at)
                           VALUES (?, ?, 'alias', 'confirmed', 'user', ?)""",
                        (fwd_key, new, now),
                    )
                    conn.execute(
                        """INSERT INTO workspace_alias_events
                             (alias_workspace, old_canonical, new_canonical,
                              old_status, new_status, action, judge_type, reason, created_at)
                           VALUES (?, ?, ?, NULL, 'confirmed', 'rename', ?, ?, ?)""",
                        (fwd_key, old, new, judge_type, reason, now),
                    )
                elif prior["status"] != "rejected":
                    # Existing confirmed alias on this key: repoint it to `new`
                    # (the repoint UPDATE above excluded fwd_key, so do it here)
                    # and record the true prior snapshot for audit.
                    conn.execute(
                        "UPDATE workspace_aliases SET canonical = ?, updated_at = ? "
                        "WHERE alias_workspace = ?",
                        (new, now, fwd_key),
                    )
                    conn.execute(
                        """INSERT INTO workspace_alias_events
                             (alias_workspace, old_canonical, new_canonical,
                              old_status, new_status, action, judge_type, reason, created_at)
                           VALUES (?, ?, ?, ?, ?, 'rename', ?, ?, ?)""",
                        (fwd_key, prior["canonical"], new, prior["status"], prior["status"],
                         judge_type, reason, now),
                    )
                else:
                    # prior is a rejection — preserve it, record that rename left
                    # it untouched so the audit trail explains the no-op.
                    conn.execute(
                        """INSERT INTO workspace_alias_events
                             (alias_workspace, old_canonical, new_canonical,
                              old_status, new_status, action, judge_type, reason, created_at)
                           VALUES (?, ?, ?, 'rejected', 'rejected', 'rename', ?, ?, ?)""",
                        (fwd_key, prior["canonical"], prior["canonical"], judge_type,
                         (reason or "") + " [forwarding-alias skipped: key is user-rejected]", now),
                    )
            return updated, []
        except sqlite3.Error as exc:
            return 0, [f"rename_workspace_canonical failed: {exc}"]

    def migrate_workspace(
        self, from_ws: str, to_ws: str, *, judge_type: str = "user",
        reason: Optional[str] = None, embedder: Any = None,
    ) -> Tuple[int, list[str]]:
        """Bulk-move memories from one canonical to another (no canonical rename).

        Registers `to_ws` in workspace_canonicals (+ vec row when an embedder is
        available), records a confirmed alias from_ws -> to_ws, and repoints any
        existing aliases that targeted `from_ws` — mirroring
        rename_workspace_canonical so a migrate can't leave a phantom canonical
        or stale alias behind. Returns (rows_updated, warnings).
        """
        if not self._db_available or not self.state.sqlite_writable:
            return 0, ["SQLite write unavailable; migrate skipped."]
        from_ws = _coerce_ws(from_ws)
        to_ws = _coerce_ws(to_ws)
        if not from_ws or not to_ws:
            return 0, ["migrate requires non-empty from and to workspace."]
        if from_ws == to_ws:
            return 0, []
        alias_key = _normalize_alias_key(from_ws)
        now = utc_now_iso()
        # Embedding for the destination canonical, computed outside the txn.
        to_embedding = None
        if embedder is not None and self.state.sqlite_vec_available:
            try:
                er = embedder.embed_text(prefix="", body=to_ws)
                to_embedding = list(er.embedding) if er and er.embedding else None
            except Exception:
                to_embedding = None
        publish_warnings: list[str] = []
        try:
            # Single transaction: move the memories AND record the confirming
            # alias together, so a crash can't leave memories relocated with no
            # alias (which would let the next write re-split the workspace).
            with self.write_transaction() as conn:
                # (1) Snapshot the forwarding-alias decision BEFORE any UPDATE,
                #     so the rejection guard reasons about real prior state, not
                #     a row a later blanket repoint has already mutated.
                fwd_prev = conn.execute(
                    "SELECT canonical, status FROM workspace_aliases WHERE alias_workspace = ? "
                    "ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1",
                    (alias_key,),
                ).fetchone()
                fwd_prev_canonical = fwd_prev["canonical"] if fwd_prev else None
                fwd_prev_status = fwd_prev["status"] if fwd_prev else None

                # (2) Move the memories.
                cur = conn.execute(
                    "UPDATE memories SET workspace_canonical = ? "
                    "WHERE COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?",
                    (to_ws, from_ws),
                )
                updated = cur.rowcount or 0

                # (3) Register the destination canonical (+ vec).
                conn.execute(
                    "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
                    (to_ws, now),
                )
                if to_embedding is not None:
                    row = conn.execute(
                        "SELECT id FROM workspace_canonicals WHERE name = ?", (to_ws,)
                    ).fetchone()
                    if row is not None:
                        try:
                            conn.execute(
                                "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                                (int(row["id"]), json.dumps(to_embedding)),
                            )
                        except sqlite3.Error as exc:
                            publish_warnings.append(
                                f"workspace canonical vector publish failed for {to_ws!r}; retry a write using this workspace after sqlite-vec and embedding configuration recover: {exc}"
                            )

                # (4) Delete the phantom `from_ws` canonical + its vec row.
                #     migrate subsumes from_ws into to_ws; leaving from_ws in the
                #     registry lets a later raw-from_ws write exact-match it and
                #     re-split. The forwarding alias (step 6) preserves resolution.
                from_row = conn.execute(
                    "SELECT id FROM workspace_canonicals WHERE name = ?", (from_ws,)
                ).fetchone()
                if from_row is not None:
                    if self.state.sqlite_vec_available:
                        try:
                            conn.execute(
                                "DELETE FROM workspace_canonicals_vec WHERE id = ?",
                                (int(from_row["id"]),),
                            )
                        except sqlite3.Error:
                            pass
                    conn.execute("DELETE FROM workspace_canonicals WHERE name = ?", (from_ws,))

                # (5) Repoint OTHER aliases that targeted `from_ws` to `to_ws`.
                #     Exclude the forwarding key (handled in step 6) and any row
                #     that would become a self-alias (alias_workspace == the
                #     normalized target), which is meaningless ("to is not to").
                #     Delete the self-aliasing rows outright.
                to_key = _normalize_alias_key(to_ws)
                conn.execute(
                    "UPDATE workspace_aliases SET canonical = ?, updated_at = ? "
                    "WHERE canonical = ? AND alias_workspace != ? AND alias_workspace != ?",
                    (to_ws, now, from_ws, alias_key, to_key),
                )
                conn.execute(
                    "DELETE FROM workspace_aliases WHERE canonical = ? AND alias_workspace = ?",
                    (from_ws, to_key),
                )

                # (6) Forwarding decision for normalize(from_ws), using the
                #     pre-UPDATE snapshot. A prior rejection is preserved (never
                #     silently confirmed); otherwise forward from_ws -> to_ws.
                if fwd_prev is not None and fwd_prev_status == "rejected":
                    conn.execute(
                        """INSERT INTO workspace_alias_events
                             (alias_workspace, old_canonical, new_canonical,
                              old_status, new_status, action, judge_type, reason, created_at)
                           VALUES (?, ?, ?, 'rejected', 'rejected', 'migrate', ?, ?, ?)""",
                        (alias_key, fwd_prev_canonical, fwd_prev_canonical, judge_type,
                         (reason or "") + " [forwarding-alias skipped: key is user-rejected]", now),
                    )
                else:
                    conn.execute(
                        """INSERT INTO workspace_aliases
                             (alias_workspace, canonical, relation, status, source, updated_at)
                           VALUES (?, ?, 'alias', 'confirmed', 'user', ?)
                           ON CONFLICT(alias_workspace,canonical) DO UPDATE SET
                             relation=excluded.relation, status=excluded.status,
                             source=excluded.source, updated_at=excluded.updated_at""",
                        (alias_key, to_ws, now),
                    )
                    conn.execute(
                        """INSERT INTO workspace_alias_events
                             (alias_workspace, old_canonical, new_canonical,
                              old_status, new_status, action, judge_type, reason, created_at)
                           VALUES (?, ?, ?, ?, 'confirmed', 'migrate', ?, ?, ?)""",
                        (alias_key, fwd_prev_canonical, to_ws, fwd_prev_status, judge_type, reason, now),
                    )
            return updated, publish_warnings
        except sqlite3.Error as exc:
            return 0, [f"migrate_workspace failed: {exc}"]

    def prepare_workspace_canonical_embedding(self, canonical: str, embedder: Any = None) -> Optional[list[float]]:
        """Compute canonical embedding before caller takes a SQLite write lock."""
        canonical = _coerce_ws(canonical)
        if embedder is not None and self.state.sqlite_vec_available and canonical:
            try:
                er = embedder.embed_text(prefix="", body=canonical)
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
        precomputed_embedding: Optional[list[float]] = None,
    ) -> Tuple[bool, list[str]]:
        canonical = _coerce_ws(canonical)
        if not canonical:
            return False, ["canonical must be a non-empty workspace string."]
        cur = conn.execute(
            "UPDATE memories SET workspace_canonical = ? WHERE id = ?",
            (canonical, int(memory_id)),
        )
        if (cur.rowcount or 0) == 0:
            return False, ["memory id not found."]
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
            (canonical, utc_now_iso()),
        )
        if precomputed_embedding is not None:
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

    def set_memory_workspace_canonical(
        self,
        memory_id: int,
        canonical: str,
        embedder: Any = None,
        *,
        conn: Optional[sqlite3.Connection] = None,
        precomputed_embedding: Optional[list[float]] = None,
    ) -> Tuple[bool, list[str]]:
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
