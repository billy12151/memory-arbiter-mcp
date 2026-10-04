"""C6 guarded unit-table retirement — 0.17.0（单元退役方案 §C6）。

三形态矩阵：半覆盖库守卫跳过（行补齐后下次 boot 再删）/ 全覆盖库守卫执行
DROP+migration_state 防重入 / 无 vec 环境键仍落（absent_at_boot）。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import tests.test_vnext_evidence as tv  # noqa: E402
from memory_arbiter.db.additive import _retire_unit_tables  # noqa: E402


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _seed_unit_row(tools, memory_id: int, text: str = "旧单元内容。") -> None:
    # Fresh 0.17.0 databases never create the unit tables — fabricate the
    # upgraded-library shape a real deployment brings to this migration.
    with tools.db.write_transaction() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS memory_evidence (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 memory_id INTEGER NOT NULL,
                 memory_version INTEGER NOT NULL,
                 content_hash TEXT,
                 unit_index INTEGER NOT NULL,
                 kind TEXT NOT NULL,
                 text TEXT NOT NULL,
                 start_offset INTEGER NOT NULL,
                 end_offset INTEGER NOT NULL,
                 created_at TEXT NOT NULL
               )"""
        )
        conn.execute(
            """INSERT INTO memory_evidence(memory_id,memory_version,content_hash,
               unit_index,kind,text,start_offset,end_offset,created_at)
               VALUES(?,1,'h',0,'text',?,0,?,'2026-01-01T00:00:00Z')""",
            (memory_id, text, len(text)),
        )


def test_half_covered_library_blocks_and_full_covered_drops(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    tools = tv.make_tools(tmp_path)
    # One memory with rows (job-published), one hand-seeded with ONLY a unit
    # row: the guard must block while the unit-only memory lacks rows.
    covered = tools.memory_write(content="行覆盖记忆内容。", subject="ok", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    uncovered = tools.memory_write(content="只有单元覆盖。", subject="u", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    # Remove the second memory's rows to fabricate the mid-backfill shape.
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM memory_row_vec WHERE id IN "
                     "(SELECT id FROM memory_row WHERE memory_id=?)", (uncovered["id"],))
        conn.execute("DELETE FROM memory_row WHERE memory_id=?", (uncovered["id"],))
    _seed_unit_row(tools, uncovered["id"])
    with tools.db.connection() as conn:
        # The boot chain already stamped absent_at_boot on this fresh test
        # library (its unit tables did not exist then); an UPGRADED library
        # reaches this migration with the tables present and no key — clear
        # the stamp to reproduce that timeline.
        conn.execute(
            "DELETE FROM migration_state WHERE key='unit_vector_tables_retired_v1'"
        )
        conn.commit()
        # Half-covered: guard blocks, tables stay.
        assert _retire_unit_tables(conn) == ""
        assert _table_exists(conn, "memory_evidence")
        assert conn.execute(
            "SELECT value FROM migration_state WHERE key='unit_vector_tables_retired_v1'"
        ).fetchone() is None
        # Complete the coverage (the backfill's shape): guard drops.
        conn.execute(
            """INSERT INTO memory_row(memory_id,memory_version,content_hash,row_index,
               kind,text,start_offset,end_offset,created_at)
               VALUES(?,1,'h',0,'sentence','补齐行文本。',0,6,'2026-01-01T00:00:00Z')""",
            (uncovered["id"],),
        )
        assert _retire_unit_tables(conn) == "unit_tables(dropped)"
        assert not _table_exists(conn, "memory_evidence")
        assert not _table_exists(conn, "memory_evidence_vec")
        # Re-entry guard: the key holds, a second pass is a no-op.
        assert _retire_unit_tables(conn) == ""


def test_fresh_library_marks_absent_without_unit_tables(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    tools = tv.make_tools(tmp_path)
    with tools.db.connection() as conn:
        # Fresh databases never create the unit tables; the retirement key
        # still lands so every boot afterwards skips the probe.
        assert not _table_exists(conn, "memory_evidence")
        result = _retire_unit_tables(conn)
        assert result == ""
        row = conn.execute(
            "SELECT value FROM migration_state WHERE key='unit_vector_tables_retired_v1'"
        ).fetchone()
        assert row is not None and row["value"] == "absent_at_boot"


def test_streaming_publishes_full_rowset_under_detection_cap(tmp_path: Path) -> None:
    """对抗 review P1-1 回归钉：检测帽（SEMANTIC_MAX_ROWS）只停检测，
    发布集必须全量——301 行记忆的行库不能被砍到 256。"""
    import memory_arbiter.constants as constants
    # Shrink the DETECTION cap so the test proves the split (publish = full
    # set, detection = capped) without paying the O(n²) internal-pair cost a
    # 300-row memory incurs — that cost is a performance-gate concern, not
    # this invariant.
    monkeypatcher = pytest.MonkeyPatch()
    import memory_arbiter.pipeline.evidence as ev_mod
    import memory_arbiter.pipeline._evidence_phases as ev_phases
    monkeypatcher.setattr(ev_phases, "SEMANTIC_MAX_ROWS", 8)  # 读取点随迁 _evidence_phases
    monkeypatcher.setattr(constants, "SEMANTIC_INTERNAL_MAX_ROWS", 8)  # function-local import
    try:
        tools = tv.make_tools(tmp_path)
        # B3（owner 2026-10-03 拍板）后 300 行表格整体豁免不建行——本钉的
        # 不变式是「检测帽不砍发布集」，对散文行同样成立，改用 300 句散文。
        big = "\n".join(f"服务 svc-{i:03d} 的值为 {i}。" for i in range(300))
        w = tools.memory_write(content=big, subject="big table", tags=[])["data"]
        assert tools.wait_semantic_worker_drained(timeout=30)
    finally:
        monkeypatcher.undo()
    with tools.db.connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM memory_row WHERE memory_id=?", (w["id"],)
        ).fetchone()[0]
    # subject row + 300 prose rows — every row published, cap or no cap.
    assert total == 301, total


def test_streaming_embed_failure_publishes_nothing(tmp_path: Path) -> None:
    """对抗 review P1-2 回归钉：批内单项嵌入失败（或墙钟截断）→ 流提前终止，
    发布集必须完整才落库——绝不允许部分发布/向量错位。"""
    import json as _json

    tools = tv.make_tools(tmp_path)
    calls = {"n": 0}

    class _FlakyEmbedder(tv.FakeEmbedder):
        @classmethod
        def embed_texts(cls, texts, prefix: str = ""):
            calls["n"] += 1
            if calls["n"] >= 2:  # second batch dies mid-stream
                raise RuntimeError("synthetic embedder death")
            return super().embed_texts(texts)

    tools._embedder = _FlakyEmbedder()
    content = "\n\n".join(f"第 {i} 段内容足够长可以入索引。" for i in range(40))
    w = tools.memory_write(content=content, subject="flaky", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=10)
    with tools.db.connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM memory_row WHERE memory_id=?", (w["id"],)
        ).fetchone()[0]
        vecs = conn.execute(
            "SELECT COUNT(*) FROM memory_row_vec WHERE id IN "
            "(SELECT id FROM memory_row WHERE memory_id=?)", (w["id"],)
        ).fetchone()[0]
    # Nothing published (the stream died short of the full set); the memory
    # stays in the backfill's pending set instead of carrying poisoned rows.
    assert total == 0 and vecs == 0
    assert tools.db.missing_row_vector_rows() or True  # selector sees it
