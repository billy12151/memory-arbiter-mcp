"""C3a (0.15.13): workspace anomaly check — summary vectors + matmul voting.

Two-layer scan, layer one: every active memory carries a summary vector
(subject + sorted tags + each body segment's first 40 chars, ~800 cap);
the check reads the whole matrix in one SELECT, computes the N×N cosine
with numpy in row blocks, votes each row over its top-10 neighbours, and
raises one workspace_review notice per memory whose neighbourhood sits
≥8/10 in one foreign bucket (cap 10 per run). Uses the FakeEmbedder
keyword buckets (postgres / 超时 / default) as workspace proxies.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter.tools import MemoryTools


@pytest.fixture()
def vec_tools(tmp_path: Path):
    pytest.importorskip("numpy")  # rides the semantic-local/llama-cpp extra
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    yield tools


def _beta_clan(tools: MemoryTools, n: int, start: int = 0) -> None:
    for i in range(start, start + n):
        tools.memory_write(
            content=f"生产环境数据库使用 PostgreSQL，兄弟条目 {i}。",
            subject=f"db-note-{i}", tags=["db"], workspace="dbpgsql",
        )


def _alpha_clan(tools: MemoryTools, n: int) -> None:
    for i in range(n):
        tools.memory_write(
            content=f"接口超时配置记录 {i}。",
            subject=f"timeout-note-{i}", tags=["timeout"], workspace="apisvc",
        )


def test_summary_text_shape() -> None:
    from memory_arbiter.tools import MemoryTools as MT

    text = MT._summary_embed_text(
        "subj", ["b", "a"], "第一段落内容。\n\n第二段落内容。".replace("段落", "段落" * 30),
    )
    assert text.startswith("subj\na b")
    # Each segment contributes at most 40 chars; total capped at 800.
    assert len(text) <= MT.SUMMARY_TOTAL_CHARS
    segments = text.split("\n")
    assert all(len(seg) <= MT.SUMMARY_SEGMENT_CHARS for seg in segments[1:])


def test_summary_vectors_cover_writes_and_backfill(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _beta_clan(tools, 2)
    _alpha_clan(tools, 1)
    assert tools.wait_evidence_worker_drained(timeout=5)

    with tools.db.connection() as conn:
        ids = {int(r["id"]) for r in conn.execute("SELECT id FROM memory_summary_vec")}
    assert len(ids) == 3, "write path must publish summary vectors"

    # Simulate a missed publish: delete one, restart-style backfill repairs.
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM memory_summary_vec WHERE id = (SELECT MIN(id) FROM memory_summary_vec)")
    written = tools._backfill_memory_summary_vectors(tools._embedder)  # type: ignore[arg-type]
    assert written >= 1
    with tools.db.connection() as conn:
        ids = {int(r["id"]) for r in conn.execute("SELECT id FROM memory_summary_vec")}
    assert len(ids) == 3


def test_anomaly_scan_flags_misplaced_memory(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _beta_clan(tools, 9)   # 9 PostgreSQL brothers in dbpgsql
    _alpha_clan(tools, 2)  # 2 timeout notes in apisvc
    # Misplaced: PostgreSQL content sitting in apisvc.
    tools.memory_write(
        content="生产环境数据库使用 PostgreSQL，属于 dbpgsql 的错位记忆。",
        subject="misplaced", tags=["db"], workspace="apisvc",
    )
    assert tools.wait_evidence_worker_drained(timeout=10)

    result = tools.memory_repair("scan_workspace_anomalies", {})
    assert result["ok"] is True, result
    data = result["data"]
    assert data["status"] == "ok"
    assert data["checked"] == 12
    findings = data["findings"]
    assert findings, "misplaced memory must be flagged"
    hit = next(f for f in findings if f["memory_id"] == 12)
    assert hit["workspace"] == "apisvc"
    assert hit["suspected_workspace"] == "dbpgsql"

    notices = tools.memory_repair("notice", {"action": "list", "status": "open", "limit": 20})
    rows = (notices.get("data") or {}).get("notices") or (notices.get("data") or {}).get("items") or []
    assert any(
        (n.get("type") or n.get("notice_type")) == "workspace_review" for n in rows
    ), f"workspace_review notice must land: {json.dumps(notices.get('data'), ensure_ascii=False)[:400]}"


def test_anomaly_scan_clean_library_stays_quiet(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _beta_clan(tools, 6)
    _alpha_clan(tools, 6)
    assert tools.wait_evidence_worker_drained(timeout=10)

    result = tools.memory_repair("scan_workspace_anomalies", {})
    data = result["data"]
    assert data["status"] == "ok"
    assert data["suspected"] == 0
    assert data["findings"] == []


def test_anomaly_scan_cap_ten(vec_tools: MemoryTools, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = vec_tools
    _beta_clan(tools, 9)   # the dbpgsql voting base must exist
    _alpha_clan(tools, 2)
    # 12 misplaced memories, all PostgreSQL content in apisvc.
    for i in range(12):
        tools.memory_write(
            content=f"生产环境数据库使用 PostgreSQL，错位条目 {i}。",
            subject=f"misplaced-{i}", tags=["db"], workspace="apisvc",
        )
    assert tools.wait_evidence_worker_drained(timeout=10)

    result = tools.memory_repair("scan_workspace_anomalies", {})
    data = result["data"]
    assert data["suspected"] >= 12
    assert data["returned"] == 10, "cap 10 per run"
    assert data.get("capped") is True


def test_move_stales_the_notice(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _beta_clan(tools, 9)
    _alpha_clan(tools, 2)
    tools.memory_write(
        content="生产环境数据库使用 PostgreSQL，属于 dbpgsql 的错位记忆。",
        subject="misplaced", tags=["db"], workspace="apisvc",
    )
    assert tools.wait_evidence_worker_drained(timeout=10)
    result = tools.memory_repair("scan_workspace_anomalies", {})
    assert result["data"]["findings"]

    moved = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [12], "new_workspace": "dbpgsql",
        "reason": "confirmed placement", "authorized": True,
    })
    assert moved["ok"] is True, moved

    with tools.db.connection() as conn:
        notice_row = conn.execute(
            "SELECT id FROM conflicts WHERE notice_type='workspace_review' "
            "AND notice_dedupe_key='workspace-anomaly:12' LIMIT 1"
        ).fetchone()
    assert notice_row is not None, "the notice must exist"
    # Staleness is lazy: reading the notice after the move flips the frozen
    # member's workspace check and marks it stale (the anomaly is resolved).
    read = tools.db.read_semantic_notice(int(notice_row["id"]))
    assert read is not None
    assert read["freshness"]["fresh"] is False
    assert read["notice_delivery_status"] == "stale", (
        "moving the memory must stale its workspace_review notice (member freshness)"
    )


def test_anomaly_scan_self_heals_missing_vectors(vec_tools: MemoryTools) -> None:
    """Fresh-boot coverage: the task itself backfills missing summary vectors
    (upgrade-before-first-write must not no-op the first weekly round)."""
    tools = vec_tools
    _beta_clan(tools, 2)
    _alpha_clan(tools, 2)
    assert tools.wait_evidence_worker_drained(timeout=5)
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM memory_summary_vec")

    result = tools.memory_repair("scan_workspace_anomalies", {})
    assert result["ok"] is True, result
    assert result["data"]["checked"] == 4, "task must backfill and vote over the full set"
    with tools.db.connection() as conn:
        covered = int(conn.execute("SELECT COUNT(*) FROM memory_summary_vec").fetchone()[0])
    assert covered == 4


def test_anomaly_scan_requires_numpy(vec_tools: MemoryTools, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    tools = vec_tools
    _beta_clan(tools, 2)
    assert tools.wait_evidence_worker_drained(timeout=5)

    real_import = builtins.__import__

    def _no_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_numpy)
    result = tools.memory_repair("scan_workspace_anomalies", {})
    monkeypatch.undo()
    # Structured capability error, same contract as sqlite_vec_unavailable.
    assert result["ok"] is False
    assert result["data"].get("error") == "numpy_unavailable"
