"""C6 (0.15.13): scan capability end-to-end — real models, release gate.

Seven-step acceptance over a synthetic two-bucket library with the REAL
embedder (embeddinggemma-300m) and REAL Qwen (0.5B pair backend), budget
15-25s (hard bound 60s). Machines without either GGUF skip cleanly, so CI
stays green while the local release gate always exercises the real stack.

Steps (plan mema #963):
  0  summary vectors in place (write path + backfill)
  1  first page with the SPEC sample call: same-bucket conflict candidate,
     page < 200KB even with include_quotes at batch=50 on this corpus,
     anomaly check names the misplaced memory pointing at bucket B,
     workspace_review notice lands
  2  agent-shaped triage following SCHEDULED_TASKS_SPEC's record_conflict
     sample verbatim (spec self-proof)
  3  same-version rescan: the dismissed pair never resurfaces (C2), notice
     not duplicated
  4  suspected-bucket sweep surfaces the planted cross-bucket conflict
  5  after the governance move: notice stales, the memory participates in
     bucket B, absent from the anomaly findings
  6  paged to null: scan_log completion line + page-progress kv complete,
     doctor raises no broken-chain alarm
  7  injected 1h+ stale progress -> doctor reports the broken chain
"""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools

_EMBED_MODEL = Path(
    "~/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf"
).expanduser()
_QWEN_MODEL = Path(
    "~/.local/share/memory-arbiter/models/semantic-conflict/"
    "Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf"
).expanduser()

_PORT_META = {"entity": "dbport", "scope": "production"}


def _release_real_models(tools: MemoryTools) -> None:
    """Teardown contract (owner rule, 2026-09-11): any test that loads a real
    GGUF model must release it EXPLICITLY — unload the semantic backend, close
    the embedder, then drop the references and gc. Leaving finalization to
    interpreter exit crashes inside ggml_metal_device_free (pytest reports all
    green while the process exits 134, polluting the release gate's and CI's
    exit-code checks)."""
    try:
        backend = getattr(tools, "_semantic_backend", None)
        if backend is not None:
            backend.unload(timeout=10.0)
    except Exception:
        pass
    tools._semantic_backend = None
    try:
        embedder = getattr(tools, "_embedder", None)
        if embedder is not None:
            embedder.close()
    except Exception:
        pass
    tools._embedder = None
    tools._embedder_loaded = False
    gc.collect()


def _make_real_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    if not _EMBED_MODEL.exists():
        pytest.skip(f"real embedding model not installed at {_EMBED_MODEL}")
    if not _QWEN_MODEL.exists():
        pytest.skip(f"real semantic model not installed at {_QWEN_MODEL}")
    settings = Settings(
        db_path=tmp_path / "c6.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=_EMBED_MODEL,
        embedding_auto_write=True,
        embedding_auto_query=True,
        semantic_conflict_enabled=True,
        semantic_conflict_model_path=_QWEN_MODEL,
        semantic_conflict_on_write="off",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    return tools


def _bucket_of(tools: MemoryTools, memory_id: int) -> str:
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS w "
            "FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
    return str(row["w"])


def _build_library(tools: MemoryTools) -> dict[str, Any]:
    """Bucket A (apisvc): 3 timeout notes + 1 misplaced PostgreSQL memory.
    Bucket B (dbpgsql): 8 PostgreSQL brothers + the planted same-bucket
    numeric pair (5432 vs 5433)."""
    for i in range(3):
        tools.memory_write(
            content=f"apisvc 网关的接口超时配置为 30s（条目 {i}），熔断阈值独立维护。",
            subject=f"timeout-{i}", tags=["apisvc", "timeout"], workspace="apisvc",
        )
    for i in range(8):
        tools.memory_write(
            content=f"dbpgsql 生产环境使用 PostgreSQL 16，部署方案第 {i} 步已复核。",
            subject=f"pg-plan-{i}", tags=["dbpgsql", "postgres"], workspace="dbpgsql",
        )
    port_a = tools.memory_write(
        content="dbpgsql 生产环境使用 PostgreSQL 16，连接端口配置为 5432。",
        subject="pg-port-a", tags=["dbpgsql", "postgres"], metadata=dict(_PORT_META),
        workspace="dbpgsql",
    )["data"]
    port_b = tools.memory_write(
        content="dbpgsql 生产环境使用 PostgreSQL 16，连接端口配置为 5433。",
        subject="pg-port-b", tags=["dbpgsql", "postgres"], metadata=dict(_PORT_META),
        workspace="dbpgsql",
    )["data"]
    misplaced = tools.memory_write(
        content="dbpgsql 生产环境使用 PostgreSQL 16，连接端口配置为 5432，与部署方案同主题。",
        subject="pg-misplaced", tags=["dbpgsql", "postgres"], metadata=dict(_PORT_META),
        workspace="apisvc",
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=30)
    return {"port_a": port_a, "port_b": port_b, "misplaced": misplaced}


def _scan_page(tools: MemoryTools, anchor: int = 0, **extra: Any) -> dict[str, Any]:
    result = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": anchor, "batch": 50, "k": 10, "include_quotes": True,
        **extra,
    })
    assert result["ok"] is True, result.get("data") or result
    return result["data"]


def _record_conflict_spec_shaped(
    tools: MemoryTools, clue: dict[str, Any], status: str, reason: str,
    *, values: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Execute the SCHEDULED_TASKS_SPEC record_conflict sample SHAPE with the
    page's values (spec self-proof). For a rule-level numeric pair the agent
    fills the slot/values from the page quotes and memory metadata — `values`
    plays that role when the page's Qwen envelope is absent."""
    members = [dict(m) for m in clue["members"]]
    for member in members:
        member.setdefault("attribute_raw", None)
        member.setdefault("value_raw", None)
        member.setdefault("normalized_attribute", None)
        member.setdefault("normalized_value", None)
        member.setdefault("prompt_version", None)
    refs = [f"{m['memory_id']}@{m['version']}" for m in members]
    if clue.get("slot_key") and clue.get("value_groups"):
        slot_key: dict[str, Any] | None = clue["slot_key"]
        value_groups: list[dict[str, Any]] = clue["value_groups"]
    elif values is not None and status == "open":
        slot_key = {"entity": "dbport", "attribute": "连接端口", "scope": "production"}
        for index, member in enumerate(members):
            member["attribute_raw"] = "连接端口"
            member["value_raw"] = values[index]
            member["normalized_attribute"] = "连接端口"
            member["normalized_value"] = values[index]
        value_groups = [
            {"normalized_value": values[0], "display_value": values[0], "members": [refs[0]]},
            {"normalized_value": values[1], "display_value": values[1], "members": [refs[1]]},
        ]
    else:
        slot_key = None
        value_groups = [{
            "normalized_value": "None", "display_value": "no conflict",
            "members": refs,
        }]
    payload = {
        "slot_key": slot_key,
        "members": members,
        "value_groups": value_groups,
        "status": status,
        "detector_version": members[0]["detector_version"],
        "prompt_version": members[0].get("prompt_version"),
        "source": "scheduled_scan",
        "reason": reason,
        "workspace": _bucket_of(tools, int(members[0]["memory_id"])),
    }
    if status == "not_a_conflict":
        payload["authorized"] = True
    return tools.memory_repair("record_conflict", payload)


@pytest.mark.slow
def test_scan_capability_e2e_real_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
) -> None:
    started = time.monotonic()
    tools = _make_real_tools(tmp_path)
    request.addfinalizer(lambda: _release_real_models(tools))
    lib = _build_library(tools)
    port_pair = tuple(sorted((lib["port_a"]["id"], lib["port_b"]["id"])))
    misplaced_id = int(lib["misplaced"]["id"])

    # Step 0: summary vectors cover the whole active set (write path), and
    # the backfill repairs a wiped row.
    with tools.db.connection() as conn:
        covered = int(conn.execute("SELECT COUNT(*) FROM memory_summary_vec").fetchone()[0])
        active = int(conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0])
    assert covered == active, f"write-path summary vectors: {covered}/{active}"
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM memory_summary_vec WHERE id=?", (misplaced_id,))
    assert tools.db.missing_summary_vec_rows()
    written = tools._backfill_memory_summary_vectors(tools._embedder)  # type: ignore[arg-type]
    assert written >= 1
    assert tools.db.missing_summary_vec_rows() == []

    # Step 1: anomaly check names the misplaced memory pointing at dbpgsql;
    # the workspace_review notice lands. Then the FIRST scan page (spec
    # sample call shape, real Qwen enhancement bounded for the test budget).
    anomaly = tools.memory_repair("scan_workspace_anomalies", {})
    assert anomaly["ok"] is True, anomaly["data"]
    findings = anomaly["data"]["findings"]
    hit = next((f for f in findings if f["memory_id"] == misplaced_id), None)
    assert hit is not None, f"misplaced memory must be flagged: {findings}"
    assert hit["suspected_workspace"] == "dbpgsql"
    with tools.db.connection() as conn:
        notice_row = conn.execute(
            "SELECT id FROM conflicts WHERE notice_type='workspace_review' "
            "AND notice_dedupe_key=?", (f"workspace-anomaly:{misplaced_id}",),
        ).fetchone()
    assert notice_row is not None, "workspace_review notice must land"

    monkeypatch.setattr("memory_arbiter.tools.SEMANTIC_SCAN_BUDGET_MS", 12_000)
    page1 = _scan_page(tools)
    clue = next(
        (c for c in page1["candidates"]
         if tuple(sorted((c["left_id"], c["right_id"]))) == port_pair),
        None,
    )
    assert clue is not None, "same-bucket planted conflict must be a candidate"
    page_size = len(json.dumps(page1, ensure_ascii=False, default=str).encode("utf-8"))
    assert page_size < 200_000, f"page too large even with quotes on this corpus: {page_size}"
    # Later rescans skip Qwen (logic under test is suppression, not extraction).
    tools._semantic_runtime_disabled = True

    # Step 2: triage exactly as the spec sample shapes it.
    recorded = _record_conflict_spec_shaped(
        tools, clue, "open", "Reviewed conflicting port values from the scan page.",
        values=("5432", "5433"),
    )
    assert recorded["ok"] is True, recorded["data"]
    assert recorded["data"]["outcome"] in {"inserted", "deduped"}, recorded["data"]

    # Step 3: same-version rescan — the recorded pair never resurfaces.
    page3 = _scan_page(tools)
    assert not any(
        tuple(sorted((c["left_id"], c["right_id"]))) == port_pair
        for c in page3["candidates"]
    )
    assert page3["counts"]["filtered_open"] >= 1
    with tools.db.connection() as conn:
        duplicates = int(conn.execute(
            "SELECT COUNT(*) FROM conflicts WHERE notice_type='workspace_review' "
            "AND notice_dedupe_key=?", (f"workspace-anomaly:{misplaced_id}",),
        ).fetchone()[0])
    assert duplicates == 1, "anomaly notice must not duplicate on re-runs"

    # Step 4: the suspected-bucket sweep surfaces the planted cross-bucket
    # conflict (misplaced x port_b, 5432 vs 5433) the same week.
    refs = {
        tuple(sorted((r["left_id"], r["right_id"]))): r
        for r in page3.get("cross_bucket_references", [])
    }
    cross_key = tuple(sorted((misplaced_id, int(lib["port_b"]["id"]))))
    assert cross_key in refs, f"planted cross-bucket conflict must surface: {sorted(refs)}"
    assert "numeric_value_candidate" in refs[cross_key]["reasons"]

    # Step 5: governance move, then rescan — notice stales, the memory
    # participates in bucket B, and the anomaly check stays quiet about it.
    moved = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [misplaced_id], "new_workspace": "dbpgsql",
        "reason": "confirmed placement", "authorized": True,
    })
    assert moved["ok"] is True, moved["data"]
    read = tools.db.read_semantic_notice(int(notice_row["id"]))
    assert read is not None and read["notice_delivery_status"] == "stale"
    page5 = _scan_page(tools)
    same_bucket_pair = tuple(sorted((misplaced_id, int(lib["port_b"]["id"]))))
    assert any(
        tuple(sorted((c["left_id"], c["right_id"]))) == same_bucket_pair
        for c in page5["candidates"]
    ), "post-move, the conflict pair must be a regular same-bucket candidate"
    rescan_anomaly = tools.memory_repair("scan_workspace_anomalies", {})
    assert not any(
        f["memory_id"] == misplaced_id for f in rescan_anomaly["data"]["findings"]
    ), "moved memory must leave the anomaly findings"

    # Step 6: page to null — scan_log completion line + page-progress kv.
    anchor = 0
    for _ in range(20):
        data = _scan_page(tools, anchor=anchor)
        if data["next_anchor_memory_id"] is None:
            break
        anchor = int(data["next_anchor_memory_id"])
    else:
        raise AssertionError("scan did not reach a null boundary")
    scan_log = tools.settings.db_path.parent / "scan_log.jsonl"
    assert scan_log.exists(), "completion line must be appended"
    last_line = json.loads(scan_log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last_line.get("next_anchor_memory_id") is None
    progress = tools.db.scan_page_progress_state()
    assert progress is not None and progress["complete"] is True
    from memory_arbiter.doctor import run_all_checks
    with tools.db.connection() as conn:
        report = run_all_checks(conn, tools.settings)
    assert not any(f.check_id == "conflicts.scan_chain" for f in report.findings)

    # Step 7: inject a 1h+ stale incomplete progress -> doctor alarms.
    # Real sequence emulated: last completion BEFORE the stalled round — age
    # the scan_log line past the fake stall so no "later completion" resolves
    # the chain.
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    older = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    stale_state = dict(progress or {})
    stale_state["complete"] = False
    stale_state["at"] = old
    with tools.db.connection() as conn:
        conn.execute(
            "INSERT INTO migration_state(key,value,updated_at) "
            "VALUES('scan_page_progress',?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(stale_state, ensure_ascii=False),),
        )
        conn.commit()
    last_line["scan_time"] = older
    lines = scan_log.read_text(encoding="utf-8").strip().splitlines()
    lines[-1] = json.dumps(last_line, ensure_ascii=False)
    scan_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with tools.db.connection() as conn:
        report = run_all_checks(conn, tools.settings)
    chain = next(f for f in report.findings if f.check_id == "conflicts.scan_chain")
    assert chain.severity.value == "warning"
    assert "interrupted at anchor" in chain.detail

    elapsed = time.monotonic() - started
    assert elapsed < 60.0, f"e2e budget exceeded: {elapsed:.1f}s"
    print(f"scan capability e2e: {elapsed:.1f}s")
