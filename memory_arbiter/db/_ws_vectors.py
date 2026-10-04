"""ws 向量发布与召回准入（从 workspaces.py 搬出，拆分批 ② 纯移动）。"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, TYPE_CHECKING

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


class _WsVectorsMixin:
    """ws 向量发布与召回准入（从 workspaces.py 搬出，拆分批 ② 纯移动）。"""

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

        # A9 补漏：保护桶变体不得在此注册（该函数是 boot/backfill 路径上的
        # 注册点，R3 对码实证遗漏——它会在 canonical 未注册时 INSERT 一行）。
        from ..twin_redirect import protected_bucket_variant

        if protected_bucket_variant(canonical):
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
