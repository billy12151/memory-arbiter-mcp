"""Workspace canonicalization and internal redirect/decision state."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, TYPE_CHECKING
from ..config import Settings
from ..degrade import DegradeState

from ..constants import (
    EMBED_PREFIX_STS,
    DEFAULT_WORKSPACE_NAME,
    WORKSPACE_MATCH_DISTANCE,
    is_default_workspace_term,
)
from ._ws_alias import _WsAliasMixin
from ._ws_merge import _WsMergeMixin
from ._ws_rowops import _WsRowopsMixin
from ._ws_vectors import _WsVectorsMixin
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

# Case-folded non-empty default terms, for SQL NOT IN guards. lower() in
# SQLite is ASCII-only, which covers the terms that have case at all.


class WorkspaceStore(_WsVectorsMixin, _WsAliasMixin, _WsMergeMixin, _WsRowopsMixin):
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

























