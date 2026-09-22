"""行级存储接线覆盖 — 0.17.0 P2-2.2/2.3/2.4。

publish 行级同事务落库（delete+rebuild）、current_row_vectors 精确往返
（A1 时序桥的读端）、row_knn rowid-IN 暴力对拍（跨工作区更近干扰行 +
tie + exclude，镜像 test_knn_rowid_in_matches_bruteforce_topk）、
状态翻转 parent_status 镜像。CI 无 vec 模块时整文件跳过（0b4f260 教训）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")  # CI [test] extra has no vec0 module

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import tests.test_workspace_recall as twr  # noqa: E402
from memory_arbiter.evidence import evidence_content_hash  # noqa: E402
from memory_arbiter.rowseg import RowSegment  # noqa: E402


def _row(text: str, index: int, kind: str = "sentence") -> RowSegment:
    return RowSegment(kind=kind, text=text, start_offset=0, end_offset=len(text), row_index=index)


def test_publish_rows_roundtrip_and_rebuild(tmp_path: Path) -> None:
    tools = twr.make_tools(tmp_path, "strict", vec=True)
    db = tools.db
    mid = twr._write(tools, "星澜网关参数", "projA")["data"]["id"]
    assert twr._confirm_pending(tools, mid)["ok"] is True
    content = "星澜网关的读超时统一为 500 毫秒。\n\n网关读超时设定见下表。\n\n| 服务 | 超时 |\n| --- | --- |\n| api | 500ms |\n"
    with db.write_transaction() as conn:
        conn.execute("UPDATE memories SET content=?, content_sha=NULL WHERE id=?", (content, mid))

    def _publish(rows, vecs, version=2):
        return db.evidence.publish(
            mid, version, evidence_content_hash(content), [], [],
            rows=rows, row_embeddings=vecs,
        )

    rows = [
        _row("星澜网关的读超时统一为 500 毫秒。", 0),
        _row("服务:api 超时:500ms", 1, kind="table_row"),
    ]
    vecs = [[1.0, 0.1], [0.9, 0.2]]
    outcome = _publish(rows, vecs)
    assert outcome["published"] and outcome["row_count"] == 2 and outcome["unit_count"] == 0

    current = db.evidence.current_row_vectors(mid, 2, evidence_content_hash(content))
    assert [r.text for r, _v in current] == [r.text for r in rows]
    assert [r.kind for r, _v in current] == ["sentence", "table_row"]
    assert all(len(v) == 2 for _r, v in current)

    # 重建：旧行向量清空、新行在位（delete+rebuild 纪律）
    rows2 = [_row("网关读超时改为 3 秒。", 0)]
    outcome2 = _publish(rows2, [[0.5, 0.5]])
    assert outcome2["row_count"] == 1
    current2 = db.evidence.current_row_vectors(mid, 2, evidence_content_hash(content))
    assert [r.text for r, _v in current2] == ["网关读超时改为 3 秒。"]
    with db.connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM memory_row WHERE memory_id=?", (mid,)).fetchone()[0]
        vec_total = conn.execute(
            "SELECT COUNT(*) FROM memory_row_vec WHERE id IN "
            "(SELECT id FROM memory_row WHERE memory_id=?)", (mid,),
        ).fetchone()[0]
    assert total == 1 and vec_total == 1

    # 形状校验：rows 与 row_embeddings 数量不齐 → 拒绝
    bad = db.evidence.publish(
        mid, 2, evidence_content_hash(content), [], [],
        rows=[_row("x" * 20, 0)], row_embeddings=[],
    )
    assert bad["outcome"] == "invalid_row_embeddings" and not bad["published"]


def test_row_knn_matches_bruteforce_topk(tmp_path: Path) -> None:
    """P2-2.4 暴力对拍：跨工作区更近干扰行不占名额、tie、exclude。"""
    tools = twr.make_tools(tmp_path, "strict", vec=True)
    db = tools.db
    a1 = twr._write(tools, "alpha one", "projA")["data"]["id"]
    a2 = twr._write(tools, "alpha two", "projA")["data"]["id"]
    b1 = twr._write(tools, "beta closer", "projB")["data"]["id"]
    for mid in (a1, a2, b1):
        assert twr._confirm_pending(tools, mid)["ok"] is True
    content_of = {a1: "alpha one", a2: "alpha two", b1: "beta closer"}
    published = [
        (a1, "alpha one r0", 0, [1.0, 0.1], 0.1),
        (a1, "alpha one r1", 1, [1.0, 0.2], 0.2),
        (a2, "alpha two r0", 0, [1.0, 0.3], 0.3),
        (a2, "alpha two r1", 1, [1.0, 0.3001], 0.3),  # 近似 tie
        (b1, "beta closer r0", 0, [1.0, 0.0], 0.0),
    ]
    by_memory: dict[int, list] = {}
    for mid, text, ridx, vec, _dist in published:
        by_memory.setdefault(mid, []).append((text, ridx, vec))
    for mid, entries in by_memory.items():
        outcome = db.evidence.publish(
            mid, 2, evidence_content_hash(content_of[mid]), [], [],
            rows=[_row(text, ridx) for text, ridx, _ in entries],
            row_embeddings=[vec for _, _, vec in entries],
        )
        assert outcome.get("published"), outcome
    query = [1.0, 0.0]
    for k in (1, 2, 3, 4):
        for exclude in (None, a1):
            got = db.row_knn(list(query), k=k, workspace="projA", exclude_memory_id=exclude)
            got_set = {(int(r["memory_id"]), int(r["row_index"])) for r in got}
            eligible = sorted(
                ((dist, mid, ridx) for mid, _t, ridx, _v, dist in published
                 if mid != exclude and mid != b1),
            )[:k]
            ref_set = {(mid, ridx) for _d, mid, ridx in eligible}
            assert got_set == ref_set, f"k={k} exclude={exclude}: {got_set} != {ref_set}"
            assert all(mid != b1 for mid, _ in got_set)  # 跨工作区更近行不占名额


def test_row_vec_parent_status_flips_on_retire(tmp_path: Path) -> None:
    tools = twr.make_tools(tmp_path, "strict", vec=True)
    db = tools.db
    mid = twr._write(tools, "alpha one", "projA")["data"]["id"]
    assert twr._confirm_pending(tools, mid)["ok"] is True
    content = "alpha one"
    db.evidence.publish(
        mid, 2, evidence_content_hash(content), [], [],
        rows=[_row("alpha one row", 0)], row_embeddings=[[1.0, 0.1]],
    )
    with db.connection() as conn:
        status = conn.execute(
            "SELECT parent_status FROM memory_row_vec WHERE id IN "
            "(SELECT id FROM memory_row WHERE memory_id=?)", (mid,),
        ).fetchone()[0]
    assert status == "active"
    with db.write_transaction() as conn:
        updated = db.memories.update_memory_on_conn(conn, mid, {"status": "retired"})
    assert updated is True
    with db.connection() as conn:
        status2 = conn.execute(
            "SELECT parent_status FROM memory_row_vec WHERE id IN "
            "(SELECT id FROM memory_row WHERE memory_id=?)", (mid,),
        ).fetchone()[0]
    assert status2 == "retired"
    # 非 active 不进候选
    assert db.row_knn([1.0, 0.1], k=3, workspace="projA") == []
