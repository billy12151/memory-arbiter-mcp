"""ClaimsStore coverage — 0.17.0 P2-5.1（方案 §7 + review A4/R1-7）。

UNIQUE 去重、version 钉死（编辑后旧 claims 不再匹配）、attr 精确候选、
自共存查询（A4）、覆盖计数（doctor 挂点）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.db.claims import ClaimsStore, claims_ddl  # noqa: E402


def _make_store(tmp_path: Path) -> ClaimsStore:
    settings = Settings(
        db_path=tmp_path / "t.sqlite3",
        backup_jsonl=tmp_path / "t.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        isolation="weak",
    )
    db = MemoryDB(settings)
    store = ClaimsStore(db)
    with db.connection() as conn:  # executescript 自带隐式提交
        conn.executescript(claims_ddl())
        conn.commit()
    return store


def _seed(
    store: ClaimsStore, memory_id: int, version: int, status: str = "active"
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
                f"sha-{memory_id}-{version}",
                "2026-09-22T00:00:00+00:00",
                "2026-09-22T00:00:00+00:00",
                "2026-09-22T00:00:00+00:00",
                "agent-a",
                "user_confirmed",
                "[]",
            ),
        )


def _claim(
    attr: str, value: str, vnorm: str | None = None, source: str = "agent"
) -> dict:
    return {
        "attr": attr,
        "attr_norm": attr.lower(),
        "value": value,
        "value_norm": vnorm or value,
        "source": source,
    }


def test_insert_dedup_and_current_version_pinning(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed(store, 1, 1)
    result = store.insert(
        memory_id=1,
        memory_version=1,
        claims=[
            _claim("超时", "500ms", "500|ms"),
            _claim("超时", "500ms", "500|ms"),  # 完全重复 → UNIQUE 吞掉
        ],
    )
    assert result == {"outcome": "ok", "written": 1}
    assert len(store.current_claims(1)) == 1
    # 编辑：version=2 落新 claims，旧 version 行保留但不再匹配
    with store._db.write_transaction() as conn:
        conn.execute("UPDATE memories SET version=2 WHERE id=1")
    assert store.current_claims(1) == []  # v1 claims 不再匹配当前版本
    store.insert(
        memory_id=1, memory_version=2, claims=[_claim("超时", "3s", "3000|ms")]
    )
    current = store.current_claims(1)
    assert len(current) == 1 and current[0]["value_norm"] == "3000|ms"


def test_attr_conflict_candidates_excludes_self_and_inactive(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed(store, 1, 1)
    _seed(store, 2, 1)
    _seed(store, 3, 1, status="retired")
    store.insert(
        memory_id=1, memory_version=1, claims=[_claim("超时", "500ms", "500|ms")]
    )
    store.insert(
        memory_id=2, memory_version=1, claims=[_claim("超时", "3s", "3000|ms")]
    )
    store.insert(
        memory_id=3, memory_version=1, claims=[_claim("超时", "1s", "1000|ms")]
    )
    candidates = store.attr_conflict_candidates(attr_norm="超时", exclude_memory_id=1)
    assert [c["memory_id"] for c in candidates] == [2]  # 自身与 retired 排除


def test_attr_conflict_candidates_limit_truncation_visible(tmp_path: Path) -> None:
    """R2 复评：候选上限必须显式传参且调用方可经 len(rows)==limit 判定截断
    （默认 CLAIMS_EXACT_CANDIDATE_LIMIT=100；截断在通道侧以
    channel_b_exact_capped 回执，不静默）。"""
    store = _make_store(tmp_path)
    for mid in (1, 2, 3, 4):
        _seed(store, mid, 1)
        store.insert(
            memory_id=mid, memory_version=1,
            claims=[_claim("超时", f"{mid}s", f"{mid}|s")],
        )
    rows = store.attr_conflict_candidates(
        attr_norm="超时", exclude_memory_id=1, limit=2,
    )
    assert len(rows) == 2  # 满 limit → 调用方视为「可能截断」信号


def test_coexisting_values_a4(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed(store, 1, 1)
    store.insert(
        memory_id=1,
        memory_version=1,
        claims=[
            _claim("限流", "1000|qps", "1000|qps"),
            _claim("限流", "300|qps", "300|qps"),  # 白天/夜间双值共存
            _claim("超时", "500|ms", "500|ms"),
        ],
    )
    assert sorted(store.coexisting_values(1, "限流")) == ["1000|qps", "300|qps"]
    assert store.coexisting_values(1, "超时") == ["500|ms"]


def test_coverage_counts(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed(store, 1, 1)
    _seed(store, 2, 1)
    assert store.coverage() == {"with_claims": 0, "active_memories": 2}
    store.insert(memory_id=1, memory_version=1, claims=[_claim("a", "1", "1")])
    assert store.coverage() == {"with_claims": 1, "active_memories": 2}
