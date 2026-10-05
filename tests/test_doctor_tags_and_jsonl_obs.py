"""0.17.1 优化批 B4 + B5：doctor tags 预筛 + jsonl 降级观测字段.

B4（纯性能）：_c_tags_over_limit 此前全表取 tags != '[]' 行 + 逐行
json.loads（30k 行 ~0.26s）。改 SQL 侧 json_valid + json_array_length
预筛。语义等价（坏 JSON 现实现本就 continue 跳过；json_array_length 对
坏 JSON 直接抛错，故必须 AND 形式）。

B5（观测）：jsonl_backup_active 是单向闩（无复位点），不改语义，只补
jsonl_backup_last_used_at（status 输出可见）。

钉死契约：
- B4：超限行仍被报出；坏 JSON 行不报（与现语义等价，且不抛错）；
- B4：预筛不改变 over_limit 结果集（对照 Python 侧全量扫描）；
- B5：降级写入后 status 含 jsonl_backup_last_used_at；
- B5：正常（未降级）库该字段为 None。
"""
from __future__ import annotations

import json
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    return MemoryTools(Settings(db_path=tmp_path / "m.db", backup_jsonl=tmp_path / "b.jsonl"))


def _find(doc: dict, check_id: str) -> dict | None:
    data = doc.get("data") if isinstance(doc.get("data"), dict) else doc
    for f in data.get("findings") or []:
        if f.get("check_id") == check_id or f.get("id") == check_id:
            return f
    return None


def test_tags_over_limit_still_reported(tmp_path: Path) -> None:
    """B4：超限行仍被报出（预筛不改变结果集）。"""
    tools = make_tools(tmp_path)
    res = tools.memory_write(content="x", subject="s", workspace="w", source_type="agent_generated")
    mid = int(res["data"]["id"])
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET tags=? WHERE id=?",
            (json.dumps([f"t{i}" for i in range(40)]), mid),
        )
    doc = tools.memory_doctor_overview(deep=False)
    f = _find(doc, "tags.over_limit")
    assert f is not None, doc
    assert f.get("status") == "warn" or f.get("ok") is False, f
    ids = [item["id"] for item in (f.get("evidence") or {}).get("over_limit") or []]
    assert mid in ids


def test_tags_prefilter_equivalent_to_python_scan(tmp_path: Path) -> None:
    """B4：预筛结果 == Python 全量扫描结果（等价钉，含坏 JSON 行）。"""
    tools = make_tools(tmp_path)
    ids = []
    for i in range(5):
        r = tools.memory_write(content=f"c{i}", subject=f"s{i}", workspace="w",
                               source_type="agent_generated")
        ids.append(int(r["data"]["id"]))
    with tools.db.write_transaction() as conn:
        # 两条超限、一条坏 JSON、其余正常
        conn.execute("UPDATE memories SET tags=? WHERE id=?",
                     (json.dumps([f"t{i}" for i in range(35)]), ids[0]))
        conn.execute("UPDATE memories SET tags=? WHERE id=?",
                     (json.dumps([f"u{i}" for i in range(33)]), ids[1]))
        conn.execute("UPDATE memories SET tags='{not json' WHERE id=?", (ids[2],))
    from memory_arbiter.constants import MAX_MEMORY_TOTAL_TAGS as CAP
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT id, tags FROM memories WHERE status!='deleted' AND tags != '[]'"
        ).fetchall()
    expected = set()
    for row in rows:
        try:
            parsed = json.loads(str(row["tags"]))
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, list) and len(parsed) > CAP:
            expected.add(int(row["id"]))
    doc = tools.memory_doctor_overview(deep=False)
    f = _find(doc, "tags.over_limit")
    got = {item["id"] for item in (f.get("evidence") or {}).get("over_limit") or []}
    assert got == expected == {ids[0], ids[1]}


def test_broken_json_row_does_not_raise(tmp_path: Path) -> None:
    """B4：坏 JSON 行不得让查询抛错（AND 形式；OR 形式实测会抛 malformed JSON）。"""
    tools = make_tools(tmp_path)
    r = tools.memory_write(content="x", subject="s", workspace="w", source_type="agent_generated")
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET tags='{{bad' WHERE id=?", (int(r["data"]["id"]),))
    doc = tools.memory_doctor_overview(deep=False)
    assert _find(doc, "tags.over_limit") is not None


def test_jsonl_backup_last_used_at_present_after_degrade(tmp_path: Path) -> None:
    """B5：降级写入后 status 含 jsonl_backup_last_used_at。"""
    tools = make_tools(tmp_path)
    # 触发 JSONL 降级路径：直接把 sqlite_writable 置 False 后写
    tools.db.state.sqlite_writable = False
    res = tools.memory_write(content="degraded", subject="s", workspace="w",
                             source_type="agent_generated")
    assert res.get("ok"), res
    assert tools.db.state.jsonl_backup_active is True
    assert tools.db.state.jsonl_backup_last_used_at, "降级写入必须记时间戳"
    status = tools.memory_status()
    payload = json.dumps(status.get("data") or {}, ensure_ascii=False)
    assert "jsonl_backup_last_used_at" in payload


def test_jsonl_backup_last_used_at_none_when_healthy(tmp_path: Path) -> None:
    """B5：未降级库该字段为 None（不新增噪声）。"""
    tools = make_tools(tmp_path)
    tools.memory_write(content="x", subject="s", workspace="w", source_type="agent_generated")
    assert tools.db.state.jsonl_backup_last_used_at is None
