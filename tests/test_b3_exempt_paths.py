"""0.17.1 修复批 A3：B3 超长表格段豁免三处接线 + 全豁免防死循环.

背景（对抗性复核实证）：_filter_exempted_segments 只接检测相与 index-only，
三处后门：index_memory（升级路径，vnext_migration 每记忆调用）、boot
backfill、扫描腿（直读 scan_rows 无过滤）。实测 120 行表：写入仅 1 行 →
index_memory 后 119 行复活 → 一次 kick 落地 internal_conflicts=7021。

R1 硬伤（实测）：空 subject + 全表 >100 行时过滤后 segments 为空 →
publish_rows([],[]) 返回 published=True/row_count=0 → 行被清空 →
missing_row_vector_rows 每次启动重选该记忆 → 永久回填死循环。

钉死契约：
- 正常 subject 巨表：index_memory / backfill / scan 后表格行=0；
- 空 subject 全豁免：index_memory 返回 all_rows_exempted 且**不清空既有行**、
  missing_row_vector_rows 不再重烧（防死循环核心钉）；
- 扫描腿：存量巨表行不再产生 internal_conflicts（对照：此前 7021）；
- 混合记忆：散文行照常；≤100 行表格段照常嵌入。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools

pytest.importorskip("sqlite_vec")

import tests.test_vnext_evidence as tv  # noqa: E402
from tests.test_vnext_evidence import FakeEmbedder  # noqa: E402

GIANT = 120


def make_tools(tmp_path: Path) -> MemoryTools:
    return tv.make_tools(tmp_path)


def _giant_table(rows: int = GIANT) -> str:
    return "\n".join(f"| 项目{i} | 金额 {i * 100} |" for i in range(rows))


def _write(tools: MemoryTools, subject: str, content: str) -> int:
    res = tools.memory_write(content=content, subject=subject, workspace="w", tags=[])
    assert res.get("ok"), res
    return int(res["data"]["id"])


def _rows(tools: MemoryTools, mid: int) -> dict[str, int]:
    with tools.db.connection() as conn:
        return {
            r["kind"]: int(r["c"])
            for r in conn.execute(
                "SELECT kind, COUNT(*) c FROM memory_row WHERE memory_id=? GROUP BY kind", (mid,)
            )
        }


def test_index_memory_respects_exemption(tmp_path: Path) -> None:
    """index_memory（升级路径）不得让豁免表行复活。"""
    tools = make_tools(tmp_path)
    mid = _write(tools, "tbl", _giant_table())
    assert tools.wait_semantic_worker_drained(timeout=120)
    assert _rows(tools, mid).get("table_row", 0) == 0, "写入路径应已豁免"
    out = tools._evidence.index_memory(mid)
    assert out.get("status") == "indexed", out
    assert _rows(tools, mid).get("table_row", 0) == 0, "index_memory 不得复活表格行"
    assert out.get("table_rows_exempted")


def test_boot_backfill_respects_exemption(tmp_path: Path) -> None:
    """boot backfill 不得复活豁免表行（存量库形态）。"""
    tools = make_tools(tmp_path)
    mid = _write(tools, "tbl", _giant_table())
    assert tools.wait_semantic_worker_drained(timeout=120)
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM memory_row_vec WHERE id IN (SELECT id FROM memory_row WHERE memory_id=?)", (mid,))
        conn.execute("DELETE FROM memory_row WHERE memory_id=?", (mid,))
    tools._backfill_memory_row_vectors(FakeEmbedder)
    assert _rows(tools, mid).get("table_row", 0) == 0, "backfill 不得复活表格行"


def test_all_exempt_does_not_clear_rows_or_loop(tmp_path: Path) -> None:
    """核心钉（R1 硬伤）：空 subject + 全表 → 不清空既有行、不重烧。"""
    tools = make_tools(tmp_path)
    mid = _write(tools, "tbl", _giant_table())
    assert tools.wait_semantic_worker_drained(timeout=120)
    # 造 legacy 空 subject 形态：清空 subject（行保留 1 行 subject）
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET subject='' WHERE id=?", (mid,))
    before = _rows(tools, mid)
    out = tools._evidence.index_memory(mid)
    after = _rows(tools, mid)
    if out.get("reason") == "all_rows_exempted":
        assert after == before, "全豁免不得清空既有行（否则回填死循环）"
        assert out.get("table_rows_exempted") == GIANT - 1  # 表格段行数（rowseg 语义）
    # 无论哪条分支：不得把该记忆留在 missing 选集里反复重烧
    miss = tools.db.missing_row_vector_rows()
    assert not any(int(x["id"]) == mid for x in miss), "全豁免记忆不得被回填选集重选"


def test_scan_side_exemption_stops_internal_flood(tmp_path: Path) -> None:
    """扫描腿：存量巨表行不再产生 O(n²) internal_conflicts（此前 7021）。"""
    tools = make_tools(tmp_path)
    mid = _write(tools, "tbl", _giant_table())
    assert tools.wait_semantic_worker_drained(timeout=120)
    # 造存量形态：手工插入 119 行 table_row（绕过写入侧豁免）
    with tools.db.write_transaction() as conn:
        for i in range(1, GIANT):
            conn.execute(
                "INSERT INTO memory_row(memory_id,memory_version,row_index,kind,text,"
                "start_offset,end_offset,content_hash,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (mid, 1, i, "table_row", f"| 项目{i} | 金额 {i*100} |", 0, 10, "h" * 64,
                 "2026-01-01T00:00:00+00:00"),
            )
    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 50, "time_budget_s": 30})
    assert kick.get("ok"), kick
    # 2026-10-05 审查接线：扫描腿豁免计数必须在 kick 回执可见（B3 宣称
    # 「回执可见」；此前写进 outcome 后全链无消费）
    assert (kick.get("data") or {}).get("table_rows_exempted_total", 0) >= GIANT - 1
    with tools.db.connection() as conn:
        ic = conn.execute("SELECT COUNT(*) FROM internal_conflicts WHERE memory_id=?", (mid,)).fetchone()[0]
    assert ic == 0, f"扫描腿必须豁免存量巨表（此前落地 7021 条），实际 {ic}"


def test_mixed_memory_keeps_prose_rows(tmp_path: Path) -> None:
    """混合记忆：120 行表 + 散文 → 散文行照常（各数各的）。"""
    tools = make_tools(tmp_path)
    content = _giant_table() + "\n\n这是一段普通散文，讨论债务转移的注意事项。\n\n第二句散文在此。"
    mid = _write(tools, "mixed", content)
    assert tools.wait_semantic_worker_drained(timeout=120)
    rows = _rows(tools, mid)
    assert rows.get("table_row", 0) == 0, "巨表段豁免"
    assert rows.get("sentence", 0) >= 1, "散文行照常嵌入"


def test_index_only_job_all_exempt_skips_publish(tmp_path: Path) -> None:
    """2026-10-05 审查修复：index_rows_in_job 补 A3 同款全豁免守卫。

    此前 index-only 路径（on_write=off / replay postprocess / conflict-apply
    edits）过滤后段集为空时仍走 publish_rows([],[])：回执谎报 indexed=0 行、
    记忆永久留在 missing_row_vector_rows 选集（already_current 永不命中）、
    每个后续 job 重复一次空 publish 事务。
    """
    tools = make_tools(tmp_path)
    mid = _write(tools, "tbl", _giant_table())
    assert tools.wait_semantic_worker_drained(timeout=120)
    # legacy 空 subject 形态 + 行缺失（模拟 missing 选集成员）
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET subject='' WHERE id=?", (mid,))
        conn.execute(
            "DELETE FROM memory_row_vec WHERE id IN (SELECT id FROM memory_row WHERE memory_id=?)", (mid,),
        )
        conn.execute("DELETE FROM memory_row WHERE memory_id=?", (mid,))
    out = tools._evidence.index_rows_in_job(mid, {"version": 1})
    assert out.get("status") == "skipped", out
    assert out.get("reason") == "all_rows_exempted"
    assert out.get("table_rows_exempted") == GIANT - 1
    assert out.get("index_only") is True
    # 修复前形态对照：此构造下返回 indexed/published（publish_rows([],[])
    # 谎报 0 行成功）；修复后不再进 publish 事务，也不落任何行。
    assert _rows(tools, mid) == {}
    with tools.db.connection() as conn:
        published = conn.execute(
            "SELECT COUNT(*) FROM memory_row_vec WHERE id IN "
            "(SELECT id FROM memory_row WHERE memory_id=?)", (mid,),
        ).fetchone()[0]
    assert published == 0


def test_small_table_still_indexed(tmp_path: Path) -> None:
    """≤100 行表格段照常嵌入（豁免不越界）。"""
    tools = make_tools(tmp_path)
    mid = _write(tools, "small", _giant_table(rows=5))
    assert tools.wait_semantic_worker_drained(timeout=120)
    # 5 行 md 表 → rowseg 产出 4 条 table_row（表头折叠进首行）
    assert _rows(tools, mid).get("table_row", 0) == 4
