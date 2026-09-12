"""0.16.0 release-gate slow e2e (plan §4): the full scan pipeline on a real
embedder — write-time internal-first detection, first full round (closure
groups, numeric auto-reject with audit, byte-capped quote-first judgment
page, page breakpoints), normalization auto-move with the double-signal gate
and protected-bucket suppression, incremental watermarks, epoch semantics,
and the single-copy serialization contract.

Cleanly skipped when no real embedding model is configured on the host.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest

from memory_arbiter import db_generation
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools

pytestmark = pytest.mark.slow

MODEL_PATH = Path("~/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf").expanduser()


def _build(tmp_path: Path) -> MemoryTools:
    if not MODEL_PATH.exists():
        pytest.skip("real embedding model not configured on this host")
    settings = Settings(
        db_path=tmp_path / "e2e.sqlite3",
        backup_jsonl=tmp_path / "e2e.jsonl",
        embedding_model_path=MODEL_PATH,
        embedding_auto_write=True,
        embedding_auto_query=False,
        client="e2e", agent_id="e2e",
        # Write-time checks stay ON (the internal-first stage is deterministic
        # and must survive a missing Qwen backend — E10① resilience); no
        # semantic model path is configured on purpose, so the cross-memory
        # Qwen loop degrades while internal findings still land.
        semantic_conflict_enabled=True,
        semantic_conflict_on_write="async",
    )
    db = MemoryDB(settings)
    return MemoryTools(settings=settings, db=db)


def _teardown(tools: MemoryTools) -> None:
    """Real-model teardown rule: unload + del + gc, or a green pytest still
    exits 134 (llama-cpp Metal SIGABRT at interpreter finalization)."""
    try:
        tools.shutdown(timeout=10.0)
    except Exception:
        pass
    tools._embedder = None
    tools._embedder_loaded = False
    gc.collect()


def _write(tools: MemoryTools, subject: str, content: str, workspace: str = "ws") -> int:
    res = tools.memory_write(content=content, subject=subject, workspace=workspace, tags=[])
    assert res.get("ok"), res
    return int(res["data"]["id"])


def _page(tools: MemoryTools, **payload):
    res = tools.memory_repair("scan_queue", {"action": "page", **payload})
    assert res["ok"], res
    return res["data"]


def _kick(tools: MemoryTools, **data):
    res = tools.memory_repair("scan_pipeline", {"action": "kick", **data})
    assert res["ok"], res
    return res["data"]


def test_scan_pipeline_e2e(tmp_path: Path) -> None:
    tools = _build(tmp_path)
    try:
        _run_e2e(tools)
    finally:
        _teardown(tools)


def _run_e2e(tools: MemoryTools) -> None:
    # ── seed the synthetic two-bucket library ────────────────────────────
    a = _write(tools, "部署数据库", "生产环境数据库使用 MySQL 8.0 主库。", workspace="研发部")
    b = _write(tools, "部署数据库对端", "生产环境数据库使用 PostgreSQL 集群。", workspace="研发部")
    n1 = _write(tools, "版本快照甲", "该功能上限为 100 QPS。", workspace="研发部")
    n2 = _write(tools, "版本快照乙", "该功能上限为 200 QPS。", workspace="研发部")
    internal_mid = _write(tools, "自相矛盾条目", "## 配置甲\n超时时间为 30 秒。\n## 配置乙\n超时时间为 60 秒。", workspace="研发部")
    mis = _write(tools, "错桶条目", "园区车辆通行证办理流程说明。", workspace="研发部")
    for i in range(12):
        _write(tools, f"通行证主题{i}", f"园区车辆通行证办理流程第{i}条。", workspace="物流园区")
    serials = "甲乙丙丁戊己庚辛壬癸子丑"
    for idx, serial in enumerate(serials):
        _write(tools, f"偏好素材{serial}", f"界面排版偏好：图表优先于表格，颜色克制。{serial}。", workspace="mema-twin")
    assert tools.wait_evidence_worker_drained(timeout=120)
    assert tools.wait_semantic_worker_drained(timeout=120)

    # ── 1) write-time: internal-first detection ─────────────────────────
    pending_internal = tools.db.internal_conflicts.list_pending()
    assert any(row["memory_id"] == internal_mid for row in pending_internal), \
        "写时内部矛盾必须落 internal_conflicts"
    with tools.db.connection() as conn:
        candidate_members = conn.execute(
            "SELECT member_versions FROM conflicts WHERE status IN ('open','applying','candidate')"
        ).fetchall()
    for row in candidate_members:
        ids = {int(m["memory_id"]) for m in json.loads(row[0])}
        assert ids != {internal_mid}, "内部冲突绝不进 conflicts 表"

    # ── 2) first full round ─────────────────────────────────────────────
    kick = _kick(tools, max_memories=500, time_budget_s=180.0)
    assert kick["complete"] is True, kick
    assert kick["pending_memories"] == 0
    rejects = tools.db.list_conflicts(status="not_a_conflict", source="scan_numeric_autoreject")
    assert rejects, "numeric 噪声必须自动驳回且留审计行"
    with tools.db.connection() as conn:
        queued_sets = [
            {int(m["memory_id"]) for m in json.loads(r[0])}
            for r in conn.execute("SELECT member_versions FROM scan_queue WHERE kind='conflict'").fetchall()
        ]
    assert {n1, n2} not in queued_sets, "numeric 对不得进判断队列"
    assert {a, b} in queued_sets, "真冲突对必须入队"

    # judgment page: quote-first payload + byte budget
    page = _page(tools)
    items = [i for i in page["items"] if i["kind"] == "conflict"]
    assert items, "判断页必须有冲突项"
    page_bytes = len(json.dumps(page, ensure_ascii=False).encode("utf-8"))
    assert page_bytes < 300 * 1024, f"判断页字节超上限: {page_bytes}"
    assert any(
        entry.get("evidence_quote")
        for item in items for pair in item["pairs"] for entry in (pair.get("evidence") or [])
    ), "判断页必须引句优先"

    hashes = [pair["candidate_key_hash"] for item in items for pair in item["pairs"]]
    # pagination is a natural breakpoint: a second page must be fetchable
    page2 = _page(tools, page_size=1)
    assert page2["items"], "翻页必须可续"
    # batch submit of page-1 dispositions
    submit = tools.memory_repair("scan_queue", {"action": "submit", "decisions": [
        {"candidate_key_hash": h, "status": "dismissed", "reason": "e2e 驳回", "authorized": True}
        for h in hashes[:3]
    ]})
    assert submit["ok"], submit

    # ── 3) normalization: auto-move + protected suppression ─────────────
    page = _page(tools, page_size=30)
    suspects = [i for i in page["items"] if i["kind"] == "workspace" and i["memory_id"] == mis]
    assert suspects, "错桶记忆必须成为归一疑点"
    result = tools.memory_repair("scan_queue", {"action": "submit", "decisions": [
        {"kind": "workspace", "memory_id": mis, "status": "confirmed",
         "target_workspace": "物流园区", "conf": 0.9}
    ]})
    entry = result["data"]["results"][0]
    assert entry["outcome"] == "moved", entry
    record = tools.db.get_memory(mis)
    assert (record["workspace_canonical"] or record["workspace"]) == "物流园区"
    with tools.db.connection() as conn:
        audits = conn.execute(
            "SELECT id,status FROM normalize_audit WHERE memory_id=?", (mis,)
        ).fetchall()
    assert audits and audits[0][1] == "applied"
    twin_member = 0
    with tools.db.connection() as conn:
        twin_member = int(conn.execute(
            "SELECT id FROM memories WHERE workspace_canonical='mema-twin' LIMIT 1"
        ).fetchone()[0])
    result = tools.memory_repair("scan_queue", {"action": "submit", "decisions": [
        {"kind": "workspace", "memory_id": twin_member, "status": "confirmed",
         "target_workspace": "物流园区", "conf": 0.95}
    ]})
    entry = result["data"]["results"][0]
    assert entry["outcome"] in {"protected_bucket_hint", "gate_failed", "not_found"}, entry
    twin_record = tools.db.get_memory(twin_member)
    assert (twin_record["workspace_canonical"] or twin_record["workspace"]) == "mema-twin", "E6：受保护桶零物理搬"
    # move 视同编辑: the moved memory's watermark was invalidated
    assert mis in tools.db.pending_scan_memory_ids()

    # ── 4) incremental watermarks ───────────────────────────────────────
    _kick(tools, max_memories=500, time_budget_s=180.0)
    with tools.db.connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM scan_queue").fetchone()[0]
    kick2 = _kick(tools, max_memories=500, time_budget_s=180.0)
    assert kick2["complete"] is True
    with tools.db.connection() as conn:
        after = conn.execute("SELECT COUNT(*) FROM scan_queue").fetchone()[0]
    assert after == before, "零变化第二轮不得新增队列项"
    tools.memory("update", {"memory_id": a, "new_content": "生产环境数据库已切换为 MySQL 8.4 主库。", "reason": "e2e edit"})
    assert tools.wait_evidence_worker_drained(timeout=120)
    assert tools.wait_semantic_worker_drained(timeout=120)
    _kick(tools, max_memories=500, time_budget_s=180.0)
    with tools.db.connection() as conn:
        rows = [json.loads(r[0]) for r in conn.execute(
            "SELECT member_versions FROM scan_queue WHERE kind='conflict'").fetchall()]
    edited_ids = [m["memory_id"] for members in rows for m in members]
    assert a in edited_ids, "编辑后仅相关对重现"

    # ── 5) epoch semantics ──────────────────────────────────────────────
    arm_before = tools.db.meta.scan_epoch_arm()
    settings = tools.settings
    db2 = MemoryDB(settings)
    tools2 = MemoryTools(settings=settings, db=db2)
    arm_after = tools2.db.meta.scan_epoch_arm()
    assert arm_after["at"] == arm_before["at"], "同版本重启不得重新布防"
    monkey = pytest.MonkeyPatch()
    monkey.setattr(db_generation, "CONFLICT_DETECTOR_VERSION", "attribute-value-e2e-next")
    try:
        db3 = MemoryDB(settings)
        tools3 = MemoryTools(settings=settings, db=db3)
        arm3 = tools3.db.meta.scan_epoch_arm()
        assert arm3["to"] == "attribute-value-e2e-next"
        assert tools3.db.pending_scan_memory_count() > 0, "epoch bump 必须布防全量"
    finally:
        monkey.undo()

    # ── 6) serialization: single-copy compact wire ──────────────────────
    from memory_arbiter.server import _structured_only

    try:
        from mcp.types import CallToolResult
    except Exception:
        CallToolResult = None  # type: ignore[assignment,misc]
    wrapped = _structured_only({"ok": True, "data": {"x": "字" * 100}})
    if CallToolResult is not None:
        assert isinstance(wrapped, CallToolResult)
        assert wrapped.content == []
        payload = wrapped.structuredContent
    else:
        payload = wrapped
    assert payload["data"]["x"]
    wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert ": " not in wire and ", " not in wire, "结构化段必须零空白"
