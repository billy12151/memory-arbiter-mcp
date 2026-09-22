"""conflict_backlog store coverage — 0.17.0 P2-4.1（方案 §6 + review A7）。

幂等入队（candidate_key UNIQUE）、500 帽按分淘汰、take_next 按分取最高、
版本漂移过期（refresh_stale 平移 scan_queue 语义）。
store 直接实例化（core.py 属性接线随 P2-4.2 落地）。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.db.conflict_backlog import (  # noqa: E402
    CONFLICT_BACKLOG_MAX,
    ConflictBacklogStore,
    conflict_backlog_ddl,
)


def _make_db(tmp: Path) -> ConflictBacklogStore:
    settings = Settings(
        db_path=tmp / "t.sqlite3",
        backup_jsonl=tmp / "t.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        isolation="weak",
    )
    db = MemoryDB(settings)
    store = ConflictBacklogStore(db)
    with db.connection() as conn:  # executescript 自带隐式提交，不进 write_transaction
        conn.executescript(conflict_backlog_ddl())
        conn.commit()
    return store


def _seed_memory(
    store: ConflictBacklogStore, memory_id: int, version: int, status: str = "active"
) -> None:
    with store._db.write_transaction() as conn:
        conn.execute(
            """INSERT INTO memories(id, workspace, content, subject, status, version,
                                    content_sha, created_at, ingest_time, event_time,
                                    agent_id, source_type, tags)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                memory_id,
                "default",
                f"content-{memory_id}",
                f"subject-{memory_id}",
                status,
                version,
                f"sha-{memory_id}",
                "2026-09-22T00:00:00+00:00",
                "2026-09-22T00:00:00+00:00",
                "2026-09-22T00:00:00+00:00",
                "agent-a",
                "user_confirmed",
                "[]",
            ),
        )


def _enqueue(
    store: ConflictBacklogStore, key: str, score: float, left: int = 1, right: int = 2
) -> dict:
    return store.enqueue(
        candidate_key_hash=key,
        left_memory_id=left,
        left_version=1,
        right_memory_id=right,
        right_version=1,
        left_text=f"left-{key}",
        right_text=f"right-{key}",
        pair_score=score,
    )


def test_enqueue_idempotent_and_take_next_by_score(tmp_path: Path) -> None:
    store = _make_db(tmp_path)
    _seed_memory(store, 1, 1)
    _seed_memory(store, 2, 1)
    assert _enqueue(store, "k1", 0.3)["outcome"] == "queued"
    assert _enqueue(store, "k1", 0.9)["outcome"] == "duplicate"  # 幂等：同 key 重放
    assert _enqueue(store, "k2", 0.8)["outcome"] == "queued"
    row = store.take_next()
    assert row is not None and row["candidate_key_hash"] == "k2"  # 分高优先
    assert store.complete(row["id"]) is True
    assert store.take_next()["candidate_key_hash"] == "k1"
    assert store.counts() == {"pending": 1, "done": 1}  # 零桶不出现在 GROUP BY


def test_cap_evicts_lowest_score_and_counts(tmp_path: Path) -> None:
    store = _make_db(tmp_path)
    _seed_memory(store, 1, 1)
    _seed_memory(store, 2, 1)
    for i in range(CONFLICT_BACKLOG_MAX):
        _enqueue(store, f"k{i:04d}", float(i) / CONFLICT_BACKLOG_MAX)
    assert store.counts()["pending"] == CONFLICT_BACKLOG_MAX
    # 超帽入队：淘汰分最低的 pending（k0000），新条目在位
    result = _enqueue(store, "k-new", 0.5)
    assert result["outcome"] == "queued" and result["evicted"] == 1
    counts = store.counts()
    assert counts["pending"] == CONFLICT_BACKLOG_MAX
    remaining = {row["candidate_key_hash"] for row in _all_rows(store)}
    assert "k0000" not in remaining and "k-new" in remaining


def _all_rows(store: ConflictBacklogStore) -> list[dict]:
    with store._db.connection() as conn:
        return [
            dict(r) for r in conn.execute("SELECT * FROM conflict_backlog").fetchall()
        ]


def test_refresh_stale_expires_on_version_drift(tmp_path: Path) -> None:
    store = _make_db(tmp_path)
    _seed_memory(store, 1, 1)
    _seed_memory(store, 2, 1)
    _seed_memory(store, 3, 1)
    _enqueue(store, "k1", 0.5)  # 钉 id=1@v1（左）
    _enqueue(store, "k2", 0.6, left=2, right=3)  # 不含 id=1
    with store._db.write_transaction() as conn:
        conn.execute("UPDATE memories SET version=2 WHERE id=1")  # 左成员被编辑
    expired = store.refresh_stale()
    assert expired == 1  # 仅钉住 id=1@v1 的条目过期
    counts = store.counts()
    assert counts["stale"] == 1 and counts["pending"] == 1


def test_extraction_roundtrip(tmp_path: Path) -> None:
    import json

    store = _make_db(tmp_path)
    _seed_memory(store, 1, 1)
    _seed_memory(store, 2, 1)
    store.enqueue(
        candidate_key_hash="kx",
        left_memory_id=1,
        left_version=1,
        right_memory_id=2,
        right_version=1,
        left_text="l",
        right_text="r",
        pair_score=0.7,
        extraction={"attribute": "超时", "values": ["500ms", "3s"]},
    )
    row = store.take_next()
    assert row is not None
    assert row["extraction"] == {"attribute": "超时", "values": ["500ms", "3s"]}
    assert json.dumps(row, ensure_ascii=False)  # 可序列化


def test_ddl_idempotent_on_real_db(tmp_path: Path) -> None:
    """真实 MemoryDB init 后 DDL 二次执行幂等。"""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_db(Path(tmp))
        with store._db.connection() as conn:  # 二次执行幂等（executescript 隐式提交）
            conn.executescript(conflict_backlog_ddl())
            conn.commit()
        assert store.counts() == {}
