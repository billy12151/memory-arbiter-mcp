"""Workspace canonicalization and internal redirect/decision state."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from typing import Any, Optional, Tuple, TYPE_CHECKING

from ..constants import DEFAULT_WORKSPACE_NAME, DEFAULT_TERMS, is_default_workspace_term
from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB

# Case-folded non-empty default terms, for SQL NOT IN guards. lower() in
# SQLite is ASCII-only, which covers the terms that have case at all.
_DEFAULT_TERM_SQL_PARAMS = tuple(sorted({t.casefold() for t in DEFAULT_TERMS if t}))
_DEFAULT_TERM_SQL_NOT_IN = (
    " AND lower(c.name) NOT IN (" + ",".join("?" for _ in _DEFAULT_TERM_SQL_PARAMS) + ")"
)


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


def _mechanical_ws_key(ws: Optional[str]) -> str:
    """Fold a workspace string to a deterministic case/separator-insensitive key.

    Unlike _normalize_alias_key (which only case-folds + collapses whitespace
    for the governance table), this also strips hyphens and underscores so pure
    spelling variants of one canonical collide: AgentLane / agent-lane /
    agent_lane -> "agentlane". Used only to reuse an EXISTING canonical, never to
    invent a new spelling. Returns "" for empty/whitespace input so blank
    workspaces never collapse together here.
    """
    if not isinstance(ws, str):
        ws = "" if ws is None else str(ws)
    return re.sub(r"[\s_\-]+", "", ws).casefold()


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
            and not is_default_workspace_term(canonical)
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
            and not is_default_workspace_term(canonical)
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
        # default never gets a vector — a vectorless default row can't
        # appear in KNN candidates, keeping the global pool un-mergeable.
        if not embedding or not canonical or is_default_workspace_term(canonical) or not self.state.sqlite_vec_available:
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
            "canonical": raw or DEFAULT_WORKSPACE_NAME,
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
        # every reserved default synonym (default/默认/none/null/
        # unknown/未知, case-insensitive) is the ONE global pool. Resolve to the
        # canonical name without alias lookup, KNN, or registration — default is
        # bidirectionally insulated from the whole vector/alias system.
        if not raw or is_default_workspace_term(raw):
            result["canonical"] = DEFAULT_WORKSPACE_NAME
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
                # 0. Durable redirect/negative-decision state. A confirmed
                #    redirect short-circuits vector/model matching; negative
                #    pairs suppress those candidates on later writes.
                alias_key = _normalize_alias_key(raw)
                try:
                    arow = conn.execute(
                        "SELECT canonical, status FROM workspace_aliases "
                        "WHERE alias_workspace = ? "
                        "ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END, "
                        "updated_at DESC, canonical ASC LIMIT 1",
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

                # 1. Exact canonical hit. A default term can never reach here
                #    (early return above), so the publish-repair below is
                #    additionally guarded for legacy/defensive safety.
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
                    if (
                        register_new
                        and not is_default_workspace_term(raw)
                        and self.state.sqlite_writable
                        and self.state.sqlite_vec_available
                        and embedder is not None
                    ):
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

                # 1b. Mechanical variant of an existing canonical: same string
                #     once case, whitespace, hyphens and underscores are folded
                #     (AgentLane / agent-lane / agent_lane). This is a purely
                #     deterministic identity with no semantic risk, so spec §11
                #     allows it to reuse the canonical without vector or Qwen.
                #     The already-registered spelling wins; a new variant never
                #     renames it.
                variant_key = _mechanical_ws_key(raw)
                if variant_key:
                    variant = next(
                        (
                            row for row in conn.execute(
                                "SELECT id, name FROM workspace_canonicals"
                            ).fetchall()
                            if _mechanical_ws_key(str(row["name"])) == variant_key
                        ),
                        None,
                    )
                    if variant is not None:
                        result.update({
                            "canonical": variant["name"], "is_new": False,
                            "matched_by": "mechanical_variant", "distance": 0.0,
                        })
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
                        # reserved default terms are excluded at the
                        # source — no vector-published default can ever swallow a
                        # real project into the global pool via AUTO merge.
                        rows = conn.execute(
                            f"""SELECT c.name AS name,
                                      vec_distance_cosine(v.embedding, ?) AS distance
                               FROM workspace_canonicals_vec v
                               JOIN workspace_canonicals c ON c.id = v.id
                               WHERE 1=1{_DEFAULT_TERM_SQL_NOT_IN}
                               ORDER BY distance
                               LIMIT 5""",
                            (query_json, *_DEFAULT_TERM_SQL_PARAMS),
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
                        # never publish a vector for a default term — a
                        # vectorless default row can't enter the KNN candidates
                        # even if a legacy DB registered one.
                        if (
                            row
                            and embedding
                            and self.state.sqlite_vec_available
                            and not is_default_workspace_term(raw)
                        ):
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

    def canonical_distance_map(
        self,
        query_canonical: str,
        canonicals: Any,
    ) -> dict[str, float]:
        """Precompute cosine distances from one query canonical to a bounded
        set of canonicals in a single query.

        Powers the recall-side vector admission/weighting without giving the
        pure scoring leaves DB access. Read-only: the query canonical's
        already-published vector is looked up (never embedded or backfilled on
        a read path); each record canonical maps to its cosine distance.
        Returns {} on any degradation (no sqlite-vec, missing query vector,
        DB error) — callers fall back to exact-equality semantics.
        """
        query = _coerce_ws(query_canonical)
        names = sorted({(str(name) or "").strip() for name in canonicals} - {""})
        # A default-term query never participates in the vector system.
        if not query or not names or is_default_workspace_term(query):
            return {}
        if not self._db_available or not self.state.sqlite_vec_available:
            return {}
        try:
            with self.connection() as conn:
                qrow = conn.execute(
                    "SELECT v.embedding AS embedding FROM workspace_canonicals c "
                    "JOIN workspace_canonicals_vec v ON v.id = c.id WHERE c.name = ?",
                    (query,),
                ).fetchone()
                if qrow is None or qrow["embedding"] is None:
                    return {}
                placeholders = ",".join("?" for _ in names)
                rows = conn.execute(
                    "SELECT c.name AS name, vec_distance_cosine(v.embedding, ?) AS distance "
                    "FROM workspace_canonicals c "
                    "JOIN workspace_canonicals_vec v ON v.id = c.id "
                    f"WHERE c.name IN ({placeholders})",
                    (qrow["embedding"], *names),
                ).fetchall()
                distance_map: dict[str, float] = {}
                for row in rows:
                    distance = row["distance"]
                    if distance is None:
                        # Degenerate vectors (all-zero / NaN) make sqlite-vec
                        # return SQL NULL — not a sqlite3.Error. Treat that
                        # canonical as vectorless so its records fall back to
                        # the binary step instead of crashing the search.
                        continue
                    try:
                        distance_map[str(row["name"])] = float(distance)
                    except (TypeError, ValueError):
                        continue
                return distance_map
        except sqlite3.Error:
            return {}

    def admitted_canonicals(
        self,
        query_canonical: str,
        *,
        cutoff: float,
        min_name_len: int = 3,
    ) -> tuple[str, ...]:
        """Canonicals a strict caller may read: its own plus in-radius neighbours.

        Strict vector admission. The caller's own canonical is ALWAYS first and
        always present, so a degraded lookup (no sqlite-vec, no published
        vector, DB error) returns exactly ``(query_canonical,)`` — the previous
        exact-equality scope. Every candidate passes the same shared guards as
        the weak weighting path (default-pool insulation, short-name guard,
        substring/generic-only proximity), so `w` never admits a neighbour and
        `main` never admits `openclaw-main`. The registry is one row per project;
        every guarded in-radius canonical is returned (no silent top-N truncation).
        """
        from ..workspace_rules import workspace_admit

        own = _coerce_ws(query_canonical)
        if not own:
            return ()
        # default is bidirectionally insulated: the global pool neither admits
        # nor is admitted by any project workspace.
        if is_default_workspace_term(own) or not self._db_available or not self.state.sqlite_vec_available:
            return (own,)
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    "SELECT c.name AS name FROM workspace_canonicals c "
                    "JOIN workspace_canonicals_vec v ON v.id = c.id "
                    f"WHERE c.name <> ?{_DEFAULT_TERM_SQL_NOT_IN}",
                    (own, *_DEFAULT_TERM_SQL_PARAMS),
                ).fetchall()
            names = [str(row["name"]) for row in rows]
        except sqlite3.Error:
            return (own,)
        if not names:
            return (own,)
        distance_map = self.canonical_distance_map(own, names)
        if not distance_map:
            return (own,)
        admitted = sorted(
            (
                (distance, name) for name, distance in distance_map.items()
                if workspace_admit(own, name, distance_map, cutoff, min_name_len=min_name_len)
            ),
            key=lambda item: (item[0], item[1]),
        )
        return (own, *[name for _distance, name in admitted])

    # ------------------------------------------------------------------
    #  Internal workspace redirect / negative-decision state.
    # ------------------------------------------------------------------
    def record_workspace_decision_on_conn(
        self,
        conn: sqlite3.Connection,
        workspace_name: str,
        canonical: str,
        *,
        status: str = "confirmed",
        force: bool = False,
    ) -> Tuple[bool, list[str]]:
        workspace_key = _normalize_alias_key(workspace_name)
        if not workspace_key:
            return False, ["workspace name must be non-empty."]
        canonical = _coerce_ws(canonical)
        if not canonical:
            return False, ["canonical must be a non-empty workspace string."]
        if is_default_workspace_term(workspace_name) or is_default_workspace_term(canonical):
            return False, [
                "default is a reserved global pool and cannot be merged in either "
                "direction; workspace decisions require two non-default names."
            ]
        if status not in {"confirmed", "rejected"}:
            return False, [f"status={status!r} invalid; expected confirmed|rejected."]
        pair = conn.execute(
            "SELECT status FROM workspace_aliases "
            "WHERE alias_workspace=? AND canonical=?",
            (workspace_key, canonical),
        ).fetchone()
        if pair is not None and pair["status"] == "rejected" and status == "confirmed" and not force:
            return False, [
                f"workspace {workspace_key!r} was explicitly kept separate from "
                f"{canonical!r}; refusing to reverse that decision silently."
            ]
        now = utc_now_iso()
        if status == "confirmed":
            conn.execute(
                "DELETE FROM workspace_aliases "
                "WHERE alias_workspace=? AND status='confirmed' AND canonical<>?",
                (workspace_key, canonical),
            )
        conn.execute(
            """INSERT INTO workspace_aliases(alias_workspace,canonical,status,updated_at)
               VALUES(?,?,?,?)
               ON CONFLICT(alias_workspace,canonical) DO UPDATE SET
                 status=excluded.status,updated_at=excluded.updated_at""",
            (workspace_key, canonical, status, now),
        )
        if status == "confirmed":
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES(?,?)",
                (canonical, now),
            )
        return True, []

    def record_workspace_decision(
        self,
        workspace_name: str,
        canonical: str,
        *,
        status: str = "confirmed",
        force: bool = False,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Tuple[bool, list[str]]:
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

    def get_workspace_decision(self, workspace_name: str) -> Optional[dict[str, Any]]:
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
    ) -> None:
        now = utc_now_iso()
        exclusions = ""
        params: list[Any] = [new, now, old]
        if exclude_aliases:
            placeholders = ",".join("?" for _ in exclude_aliases)
            exclusions = f" AND alias_workspace NOT IN ({placeholders})"
            params.extend(exclude_aliases)
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

    @staticmethod
    def _mechanical_canonical_on_conn(
        conn: sqlite3.Connection, workspace_name: str,
    ) -> Optional[str]:
        key = _mechanical_ws_key(workspace_name)
        if not key:
            return None
        for row in conn.execute("SELECT name FROM workspace_canonicals"):
            name = str(row["name"])
            if _mechanical_ws_key(name) == key:
                return name
        return None

    def _competing_move_warning_on_conn(
        self, conn: sqlite3.Connection, source: str, destination: str,
    ) -> Optional[list[str]]:
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

    def _install_workspace_redirect_on_conn(
        self, conn: sqlite3.Connection, workspace_name: str, canonical: str,
        *, force: bool = False,
    ) -> bool:
        key = _normalize_alias_key(workspace_name)
        if not key or key == _normalize_alias_key(canonical):
            return False
        rejected = conn.execute(
            "SELECT 1 FROM workspace_aliases "
            "WHERE alias_workspace=? AND canonical=? AND status='rejected'",
            (key, canonical),
        ).fetchone()
        if rejected is not None and not force:
            return False
        if rejected is not None:
            conn.execute(
                "DELETE FROM workspace_aliases "
                "WHERE alias_workspace=? AND canonical=? AND status='rejected'",
                (key, canonical),
            )
        conn.execute(
            "DELETE FROM workspace_aliases "
            "WHERE alias_workspace=? AND status='confirmed' AND canonical<>?",
            (key, canonical),
        )
        conn.execute(
            "INSERT INTO workspace_aliases(alias_workspace,canonical,status,updated_at) "
            "VALUES(?,?,'confirmed',?) "
            "ON CONFLICT(alias_workspace,canonical) DO UPDATE SET "
            "status='confirmed',updated_at=excluded.updated_at",
            (key, canonical, utc_now_iso()),
        )
        return True

    def rename_workspace_canonical(
        self, old: str, new: str,
    ) -> Tuple[int, list[str]]:
        """Rename or merge a canonical and keep old-name forwarding stable."""
        if not self._db_available or not self.state.sqlite_writable:
            return 0, ["SQLite write unavailable; rename skipped."]
        old = _coerce_ws(old)
        new = _coerce_ws(new)
        if not old or not new:
            return 0, ["rename requires non-empty old and new canonical."]
        # renaming into or out of default would merge the global pool
        # with one project — refuse in both directions.
        if is_default_workspace_term(old) or is_default_workspace_term(new):
            return 0, [
                "default is a reserved global pool and cannot be merged in either "
                "direction; rename requires two non-default workspace names."
            ]
        if old == new:
            return 0, []
        now = utc_now_iso()
        try:
            with self.write_transaction() as conn:
                competing = self._competing_move_warning_on_conn(conn, old, new)
                if competing is not None:
                    return 0, competing
                cur = conn.execute(
                    "UPDATE memories SET workspace_canonical = ? "
                    "WHERE COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?",
                    (new, old),
                )
                updated = cur.rowcount or 0
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
                    self._repoint_workspace_targets_on_conn(conn, old, new)
                    conn.execute(
                        "DELETE FROM workspace_aliases WHERE alias_workspace=? AND canonical=?",
                        (new_key, new),
                    )
                    return updated, []
                self._repoint_workspace_targets_on_conn(
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
            return updated, []
        except sqlite3.Error as exc:
            return 0, [f"rename_workspace_canonical failed: {exc}"]

    def migrate_workspace(
        self, from_ws: str, to_ws: str, *, embedder: Any = None,
    ) -> Tuple[int, list[str]]:
        """Merge one canonical into another and keep old-name forwarding stable."""
        if not self._db_available or not self.state.sqlite_writable:
            return 0, ["SQLite write unavailable; migrate skipped."]
        from_ws = _coerce_ws(from_ws)
        to_ws = _coerce_ws(to_ws)
        if not from_ws or not to_ws:
            return 0, ["migrate requires non-empty from and to workspace."]
        # migrate is a merge-into path like rename; default stays
        # reserved in both directions.
        if is_default_workspace_term(from_ws) or is_default_workspace_term(to_ws):
            return 0, [
                "default is a reserved global pool and cannot be merged in either "
                "direction; migrate requires two non-default workspace names."
            ]
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
            with self.write_transaction() as conn:
                competing = self._competing_move_warning_on_conn(conn, from_ws, to_ws)
                if competing is not None:
                    return 0, competing
                cur = conn.execute(
                    "UPDATE memories SET workspace_canonical = ? "
                    "WHERE COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?",
                    (to_ws, from_ws),
                )
                updated = cur.rowcount or 0
                conn.execute(
                    "UPDATE conflicts SET workspace_canonical=? WHERE workspace_canonical=?",
                    (to_ws, from_ws),
                )
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
                to_key = _normalize_alias_key(to_ws)
                self._repoint_workspace_targets_on_conn(
                    conn, from_ws, to_ws, exclude_aliases=(alias_key, to_key),
                )
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
                self._install_workspace_redirect_on_conn(conn, from_ws, to_ws)
            return updated, publish_warnings
        except sqlite3.Error as exc:
            return 0, [f"migrate_workspace failed: {exc}"]

    def prepare_workspace_canonical_embedding(self, canonical: str, embedder: Any = None) -> Optional[list[float]]:
        """Compute canonical embedding before caller takes a SQLite write lock."""
        canonical = _coerce_ws(canonical)
        if (
            embedder is not None
            and self.state.sqlite_vec_available
            and canonical
            and not is_default_workspace_term(canonical)
        ):
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
