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


def test_drain_consumes_stored_extraction_and_lands_notice(tmp_path: Path) -> None:
    """P2-4.2：stored extraction 直接过确定性门落 notice、条目 done；
    version 漂移条目被 refresh_stale 收编不消化。"""
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    from test_vnext_evidence import make_tools

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "prod"}
    a = tools.memory_write(
        content="连接池上限为 10。", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(
        content="连接池上限为 99。", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    result = tools.db.conflict_backlog.enqueue(
        candidate_key_hash="kb-1",
        left_memory_id=a["id"], left_version=1,
        right_memory_id=b["id"], right_version=1,
        left_text="连接池上限为 10。", right_text="连接池上限为 99。",
        pair_score=0.6,
        extraction={"attribute": "连接池上限", "value_a": "10", "value_b": "99"},
    )
    assert result["outcome"] == "queued"
    processed = tools._evidence.drain_conflict_backlog(limit=2)
    assert processed == 1
    assert tools.db.conflict_backlog.counts().get("done") == 1
    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == a["id"]]
    assert len(notices) == 1 and notices[0]["payload"].get("backlog") is True

    # version 漂移：入队后编辑左侧 → 消化时判 stale，不落 notice 不计 processed
    tools.db.conflict_backlog.enqueue(
        candidate_key_hash="kb-2",
        left_memory_id=a["id"], left_version=1,
        right_memory_id=b["id"], right_version=1,
        left_text="x", right_text="y", pair_score=0.1,
    )
    tools.db.edit_memory_intent(a["id"], new_content="连接池上限为 20。", reason="edit")
    refreshed = tools._evidence.drain_conflict_backlog(limit=2)
    assert refreshed == 0
    assert tools.db.conflict_backlog.counts().get("stale", 0) >= 1


def test_drain_no_backend_skip_rotation_is_bounded(tmp_path: Path, monkeypatch) -> None:
    """R2-W1 回归：无后端 + 无 extraction 的积压在 drain 里零进展——skip-
    rotate 不设上限时 while processed<limit 只能靠 take_next 扫完全部
    pending 才停（take_next 每次新开 sqlite 连接 ⇒ 空闲 tick 每 5s 一轮
    O(N) 连接空转）。超界即停；条目未 complete 仍 pending，后端出现后
    照常重试。"""
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    from test_vnext_evidence import make_tools

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "prod"}
    a = tools.memory_write(
        content="连接池上限为 10。", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(
        content="连接池上限为 99。", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    pending_n = 40
    for i in range(pending_n):
        result = tools.db.conflict_backlog.enqueue(
            candidate_key_hash=f"knb-{i:03d}",
            left_memory_id=a["id"], left_version=1,
            right_memory_id=b["id"], right_version=1,
            left_text="连接池上限为 10。", right_text="连接池上限为 99。",
            pair_score=0.5 + i / 1000,  # 全部无 extraction
        )
        assert result["outcome"] == "queued"

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: None)
    store = tools.db.conflict_backlog
    real_take_next = store.take_next
    take_calls: list[int] = []

    def _counting_take_next(*args, **kwargs):
        take_calls.append(1)
        return real_take_next(*args, **kwargs)

    monkeypatch.setattr(store, "take_next", _counting_take_next)

    processed = tools._evidence.drain_conflict_backlog(limit=2)
    assert processed == 0, "无后端不可能有进展"
    # 上界 2*limit ⇒ take_next 最多 2*limit+1 次（末次取空判定）；未设上限
    # 时这里会是 41 次（扫完全部 pending 才停）。
    assert len(take_calls) <= 2 * 2 + 1, (
        f"skip-rotate 未设上限：take_next 调用 {len(take_calls)} 次"
    )
    assert store.counts().get("pending", 0) == pending_n, "skip 条目必须保持 pending"


def test_worker_idle_drain_skipped_when_off(tmp_path: Path) -> None:
    """R2 P2：on_write="off" 语义=无检测活动——worker idle 轮不得自动消化
    backlog（会加载 Qwen 并产 notice，与 start() 不预加载及 runtime_state
    的 off 报告自相矛盾）。积压留待恢复 on_write 后消化；idle tick 周期
    5s，此处等待两轮确认零调用。"""
    import sys
    import time
    sys.path.insert(0, str(REPO / "tests"))
    from test_vnext_evidence import make_tools

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "prod"}
    a = tools.memory_write(
        content="连接池上限为 10。", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(
        content="连接池上限为 99。", subject="b", tags=[], metadata=meta)["data"]
    tools.db.conflict_backlog.enqueue(
        candidate_key_hash="kw-off-1",
        left_memory_id=a["id"], left_version=1,
        right_memory_id=b["id"], right_version=1,
        left_text="连接池上限为 10。", right_text="连接池上限为 99。",
        pair_score=0.6,
        extraction={"attribute": "连接池上限", "value_a": "10", "value_b": "99"},
    )
    assert tools.wait_semantic_worker_drained(timeout=5)
    drain_calls: list[int] = []
    real_drain = tools._evidence.drain_conflict_backlog

    def _counting_drain(limit: int = 2):
        drain_calls.append(limit)
        return real_drain(limit=limit)

    tools._evidence.drain_conflict_backlog = _counting_drain
    time.sleep(10.5)  # ≥2 个 idle tick（5s 间隔）
    assert drain_calls == [], "off 语义下 idle 轮不得自动 drain"
    assert tools.db.conflict_backlog.counts().get("pending", 0) == 1, "积压必须保留"
    # 恢复 on_write 后手动 drain 仍可用（消化权在显式恢复侧）
    tools._evidence.drain_conflict_backlog = real_drain
    tools.settings.semantic_conflict_on_write = "sync"
    processed = tools._evidence.drain_conflict_backlog(limit=2)
    assert processed == 1


def test_drain_judge_pool_terminates(tmp_path: Path, monkeypatch) -> None:
    """0.17.1 review P0 回归：判定池分支的条目 pass2 前不 complete 且
    processed 不增，必须同时进 take_next 排除表——否则 while 每轮重取同一
    头部条目无限循环（judge_pool 无限膨胀）。"""
    import signal

    sys.path.insert(0, str(REPO / "tests"))
    from test_vnext_evidence import make_tools

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "sync"
    meta = {"entity": "svc", "scope": "prod"}
    a = tools.memory_write(
        content="连接池上限为 10。", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(
        content="连接池上限为 99。", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    store = tools.db.conflict_backlog
    result = store.enqueue(
        candidate_key_hash="kjp-1",
        left_memory_id=a["id"], left_version=1,
        right_memory_id=b["id"], right_version=1,
        left_text="连接池上限为 10。", right_text="连接池上限为 99。",
        pair_score=0.7,  # 无 extraction → 走判定池分支
    )
    assert result["outcome"] == "queued"

    from memory_arbiter.semantic_judge import PairVerdict

    class _StubBackend:
        def judge_pairs(self, pairs):
            return [PairVerdict("no_conflict", {}, None, "stub") for _ in pairs]

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _StubBackend())

    def _handler(signum, frame):
        raise TimeoutError("drain did not terminate — judge-pool livelock")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(15)
    try:
        processed = tools._evidence.drain_conflict_backlog(limit=2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert processed == 1
    assert store.counts().get("done") == 1, "判定 clear 条目必须出队"
