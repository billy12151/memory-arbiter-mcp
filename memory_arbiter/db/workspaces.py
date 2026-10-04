"""Workspace canonicalization and internal redirect/decision state."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, TYPE_CHECKING
from ..config import Settings
from ..degrade import DegradeState

from ..constants import (
    EMBED_PREFIX_STS,
    DEFAULT_WORKSPACE_NAME,
    DEFAULT_TERMS,
    WORKSPACE_MATCH_DISTANCE,
    is_default_workspace_term,
)
from ..db_generation import database_startup_lock
from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB

# Case-folded non-empty default terms, for SQL NOT IN guards. lower() in
# SQLite is ASCII-only, which covers the terms that have case at all.
_DEFAULT_TERM_SQL_PARAMS = tuple(sorted({t.casefold() for t in DEFAULT_TERMS if t}))
_DEFAULT_TERM_SQL_NOT_IN = (
    " AND lower(c.name) NOT IN (" + ",".join("?" for _ in _DEFAULT_TERM_SQL_PARAMS) + ")"
)


def _normalize_alias_key(ws: str | None) -> str:
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


def _mechanical_ws_key(ws: str | None) -> str:
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


def _normalize_ws_group_key(ws: str | None) -> str:
    """Grouping key for ``normalize_workspace_canonicals`` — deliberately
    STRICTER than ``_mechanical_ws_key``.

    Same separator stripping, but ``str.lower`` instead of ``casefold``:
    'Straße' vs 'strasse' and the 'ﬁ' ligature vs 'file' stay distinct
    instead of collapsing. Normalize is a bulk destructive merge, so its
    grouping key errs toward NOT merging (a missed variant is recoverable, a
    wrong merge is not); the non-destructive orthography reuse in the
    decision primitive / resolver / migrate keeps the full casefold of
    ``_mechanical_ws_key``. The whole planner (grouping, respected-rejection
    match, shadowed-redirect match, rejected-only twin map) uses this one key
    space so every comparison stays consistent with the groups built from it.
    """
    if not isinstance(ws, str):
        ws = "" if ws is None else str(ws)
    return re.sub(r"[\s_\-]+", "", ws).lower()


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

    @property
    def _db_available(self) -> bool:
        return self._db._db_available

    @property
    def settings(self) -> "Settings":
        return self._db.settings

    @property
    def state(self) -> "DegradeState":
        return self._db.state

    @contextmanager
    def connection(self) -> "Iterator[sqlite3.Connection]":
        with self._db.connection() as conn:
            yield conn

    @contextmanager
    def write_transaction(self) -> "Iterator[sqlite3.Connection]":
        with self._db.write_transaction() as conn:
            yield conn

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
            er = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=canonical)
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
    ) -> list[float] | None:
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
            er = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=canonical)
            return list(er.embedding) if er and er.embedding else None
        except Exception:
            return None

    def publish_workspace_canonical_vector(
        self,
        canonical: str,
        embedding: list[float] | None,
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

    def rebuild_workspace_canonical_vectors(
        self, embedder: Any, embedding_space_id: str,
    ) -> dict[str, Any]:
        """Atomically replace every non-default canonical vector."""
        if embedder is None or not self.state.sqlite_vec_available:
            return {"ok": False, "error": "workspace_vector_runtime_unavailable"}
        try:
            with self.connection() as conn:
                marker = conn.execute(
                    "SELECT value FROM _vec_index_meta "
                    "WHERE key='workspace_rebuild_space_id'"
                ).fetchone()
                names = [
                    str(row["name"])
                    for row in conn.execute(
                        "SELECT name FROM workspace_canonicals ORDER BY id"
                    )
                    if not is_default_workspace_term(str(row["name"] or ""))
                ]
                expected_ids = {
                    int(row["id"])
                    for row in conn.execute("SELECT id,name FROM workspace_canonicals")
                    if not is_default_workspace_term(str(row["name"] or ""))
                }
                vector_ids = {
                    int(row["id"])
                    for row in conn.execute("SELECT id FROM workspace_canonicals_vec")
                }
                if (
                    marker is not None
                    and str(marker["value"]) == embedding_space_id
                    and vector_ids == expected_ids
                ):
                    return {"ok": True, "rebuilt": 0, "already_current": True}
            vectors: dict[str, list[float]] = {}
            for name in names:
                result = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=name)
                embedding = list(result.embedding) if result and result.embedding else []
                if not embedding:
                    return {
                        "ok": False,
                        "error": "workspace_vector_embedding_failed",
                        "canonical": name,
                    }
                vectors[name] = embedding
            with self.write_transaction() as conn:
                current = [
                    str(row["name"])
                    for row in conn.execute(
                        "SELECT name FROM workspace_canonicals ORDER BY id"
                    )
                    if not is_default_workspace_term(str(row["name"] or ""))
                ]
                if current != names:
                    return {"ok": False, "error": "workspace_registry_changed"}
                conn.execute("DELETE FROM workspace_canonicals_vec")
                for name in names:
                    row = conn.execute(
                        "SELECT id FROM workspace_canonicals WHERE name=?", (name,),
                    ).fetchone()
                    if row is None:
                        raise sqlite3.IntegrityError("workspace canonical disappeared")
                    conn.execute(
                        "INSERT INTO workspace_canonicals_vec(id,embedding) VALUES(?,?)",
                        (int(row["id"]), json.dumps(vectors[name])),
                    )
                conn.execute(
                    "INSERT INTO _vec_index_meta(key,value) VALUES("
                    "'workspace_rebuild_space_id',?) ON CONFLICT(key) DO UPDATE "
                    "SET value=excluded.value",
                    (embedding_space_id,),
                )
            return {"ok": True, "rebuilt": len(names), "already_current": False}
        except sqlite3.Error as exc:
            return {"ok": False, "error": f"workspace_vector_rebuild_failed: {exc}"}

    def resolve_workspace_canonical(
        self,
        ws_raw: str | None,
        embedder: Any = None,
        *,
        match_distance: float | None = None,
    ) -> dict[str, Any]:
        """Resolve a raw workspace string to its canonical name (alias merge).

        Strategy (double-store: raw stays in memories.workspace, resolved name
        goes to memories.workspace_canonical):
          1. Exact match against workspace_canonicals.name → reuse it.
          2. If an embedder + sqlite-vec are available, embed the raw string and
             KNN against workspace_canonicals_vec; if the nearest canonical is
             within ``match_distance`` reuse it (handles 金营项目 / 金科营销项目).
          3. Otherwise it is a NEW canonical — the resolution itself writes
             nothing (P2 #9: the register_new parameter retired with its dead
             branches; the write path registers the final canonical atomically
             in insert_memory).

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
            match_distance = WORKSPACE_MATCH_DISTANCE

        try:
            with self.connection() as conn:
                # 0. Durable redirect/negative-decision state. A confirmed
                #    redirect short-circuits vector/model matching; negative
                #    pairs suppress those candidates on later writes.
                alias_key = _normalize_alias_key(raw)
                ghost_alias_keys: list[str] = []
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
                if arow is not None and str(arow["status"]) == "confirmed":
                    # Confirmed identity pair: the hot path returns before any
                    # ghost-spelling scan (P2 #5 keeps the confirmed short
                    # circuit free of extra reads; confirmed rows can never
                    # join a suppression list).
                    result.update({
                        "canonical": arow["canonical"],
                        "is_new": False,
                        "matched_by": "confirmed_alias",
                        "distance": 0.0,
                    })
                    return result
                # Ghost-spelling siblings: a rejection recorded under one
                # spelling ("agent-lane") must also cover separator/case
                # variants ("agent_lane") — the mechanical fold would
                # otherwise silently bypass it (the alias lookup key only
                # case-folds + collapses whitespace). P2 #5: the sibling set
                # is computed for the direct REJECTED hit too, so every
                # same-mechanical-key spelling's rejections aggregate on
                # both paths. Only rejected rows participate (SQL pushes
                # status='rejected' down; a confirmed alias is an exact
                # identity pair and must not silently redirect an unrelated
                # registered bucket that happens to share the mechanical key).
                mkey = _mechanical_ws_key(raw)
                if mkey:
                    try:
                        ghost_rows = conn.execute(
                            "SELECT alias_workspace FROM workspace_aliases "
                            "WHERE status='rejected'"
                        ).fetchall()
                    except sqlite3.Error:
                        ghost_rows = []
                    ghost_rejected = [
                        row for row in ghost_rows
                        if _mechanical_ws_key(str(row["alias_workspace"])) == mkey
                    ]
                    if ghost_rejected:
                        ghost_alias_keys = [
                            str(row["alias_workspace"])
                            for row in ghost_rejected
                        ]
                        if arow is None:
                            arow = ghost_rejected[0]
                if arow is not None:
                    # arow is rejected here (confirmed returned above).
                    # Rejections accumulate per (raw, canonical). Exact
                    # key ∪ ghost mechanical keys — unconditionally — so a
                    # direct hit under one spelling also eats its siblings'
                    # rejections (agent-chancellor carries two rejected rows
                    # under two spellings).
                    list_keys = list(dict.fromkeys([alias_key, *ghost_alias_keys]))
                    placeholders = ",".join("?" * len(list_keys))
                    rejected_rows = conn.execute(
                        f"SELECT canonical FROM workspace_aliases "
                        f"WHERE status='rejected' "
                        f"AND alias_workspace IN ({placeholders})",
                        list_keys,
                    ).fetchall()
                    suppressed = [str(row["canonical"]) for row in rejected_rows]
                    # Exact-match suppression alone is bypassable through a
                    # ghost spelling variant (rejected targets are never
                    # registered, so "projectb" would not suppress a
                    # registered "ProjectB"). Expand mechanically so every
                    # registered spelling of a rejected target is suppressed
                    # for all downstream consumers (vector path, rule
                    # decision, qwen candidate check).
                    rejected_keys = {
                        _mechanical_ws_key(name) for name in suppressed
                    }
                    rejected_keys.discard("")
                    if rejected_keys:
                        for reg in conn.execute("SELECT name FROM workspace_canonicals"):
                            name = str(reg["name"])
                            if (
                                _mechanical_ws_key(name) in rejected_keys
                                and name not in suppressed
                            ):
                                suppressed.append(name)
                    result["rejected_canonicals"] = suppressed

                # 1. Exact canonical hit. A default term can never reach here
                #    (early return above), so the publish-repair below is
                #    additionally guarded for legacy/defensive safety.
                exact = conn.execute(
                    "SELECT id, name FROM workspace_canonicals WHERE name = ?",
                    (raw,),
                ).fetchone()
                if exact:
                    result.update({"canonical": exact["name"], "is_new": False, "matched_by": "exact", "distance": 0.0})
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
                    # The mechanical fold must respect rejected targets: a
                    # keep-separate decision on this identity pair suppresses
                    # the fold (suppressed spellings were expanded mechanically
                    # in the rejected branch above, so name exclusion here is
                    # sufficient).
                    suppressed_names = set(result.get("rejected_canonicals") or [])
                    variant = next(
                        (
                            row for row in conn.execute(
                                "SELECT id, name FROM workspace_canonicals"
                            ).fetchall()
                            if _mechanical_ws_key(str(row["name"])) == variant_key
                            and str(row["name"]) not in suppressed_names
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
                        er = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=raw)
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

                # 3. New canonical. Resolution is read-only (P2 #9: the dead
                #    register_new branches are gone); registration of the final
                #    policy result happens atomically in insert_memory, and the
                #    write path publishes the prepared vector post-commit.
                result.update({"canonical": raw, "is_new": True, "matched_by": "new"})
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
        mechanical = WorkspaceStore._mechanical_canonical_on_conn(conn, canonical)
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
        return WorkspaceStore._sha_collision_warning_text(
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
        ok, _errors = WorkspaceStore._apply_alias_decision_on_conn(
            conn, workspace_name, canonical, status="confirmed", force=force,
        )
        return ok

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
        collision = WorkspaceStore._conflict_slot_collision_warning_on_conn(conn, from_ws, to_ws)
        if collision is not None:
            return 0, collision, False
        sha_collision = WorkspaceStore._content_sha_collision_warning_on_conn(
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
        warnings.extend(WorkspaceStore._repoint_workspace_targets_on_conn(
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
        WorkspaceStore._install_workspace_redirect_on_conn(conn, from_ws, to_ws)
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
                    updated, merge_warnings, _merged = self._merge_workspace_core_on_conn(
                        conn, loser, winner, db=self._db,
                    )
                    result["warnings"].extend(merge_warnings)
                    if merge_warnings:
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
                else:
                    # Plan honesty: the same read-only collision guard runs
                    # in dry-run so the plan reports a would-be refusal as
                    # skipped, not as a merge the execute pass must refuse.
                    collision = WorkspaceStore._conflict_slot_collision_warning_on_conn(conn, loser, winner)
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
        if not canonical:
            return False, ["canonical must be a non-empty workspace string."]
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
