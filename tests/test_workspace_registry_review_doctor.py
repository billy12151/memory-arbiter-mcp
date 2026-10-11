

# ── from test_workspace_review_doctor.py ──
# helper make_tools renamed: review_doctor_make_tools (collision)

"""doctor workspace.review 全量确认 + confirm_workspaces 治理动作 (mema 721 期2).

Contract highlights under test:
  - workspace.review diffs workspace_canonicals (default terms excluded)
    against the workspace_review.json sidecar in ONE direction; missing or
    corrupt sidecar = first full review, never raises;
  - the finding is WARNING-only (a pending confirmation must never be
    critical / break CI exit semantics for an otherwise healthy registry);
  - doctor NEVER writes the snapshot — only the authorized
    memory_govern(confirm_workspaces) action does;
  - all eight product-surface wiring points exist for confirm_workspaces.
"""
import json
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import DEFAULT_WORKSPACE_NAME
from memory_arbiter.db import MemoryDB
from memory_arbiter.doctor import Severity, open_ro_connection, run_all_checks
from memory_arbiter.tools import MemoryTools
from contextlib import contextmanager
from types import SimpleNamespace
import sqlite3
import threading
from memory_arbiter.db import MemoryDB, _normalize_alias_key
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryStatus
from tests._helpers import record_conflict
from tests._helpers import register, write


def review_doctor_make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "rev.sqlite3",
        backup_jsonl=tmp_path / "rev.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write(tools: MemoryTools, content: str, workspace: str, subject: str = "test") -> int:
    return tools.memory_write(
        content=content, workspace=workspace, subject=subject,
        source_type="agent_generated",
    )["data"]["id"]


def _run_doctor(tools: MemoryTools):
    with open_ro_connection(Path(tools.settings.db_path)) as conn:
        return run_all_checks(conn, tools.settings)


def test_deep_doctor_owns_integrity_generation_and_vector_health(tmp_path):
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "deep.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "deep.sqlite3",
        backup_jsonl=tmp_path / "deep.jsonl",
        embedding_model_path=model,
    )
    db = MemoryDB(settings)
    # Lazy vec0 tables (0.15.0): create them at dim 2 the way the first
    # successful embedder build does, so the deep vector checks have tables.
    assert db.ensure_vec_tables(2) == []
    with db.write_transaction() as conn:
        conn.execute(
            """INSERT INTO memories(content,agent_id,workspace,tags,source_type,
               event_time,ingest_time,status,subject,metadata,version,created_at)
               VALUES('body','agent','default','[]','agent_generated',
               '2026-01-01T00:00:00Z','2026-01-01T00:00:00Z','active','subject','{}',1,
               '2026-01-01T00:00:00Z')"""
        )
        memory_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute(
            """INSERT INTO memory_row(memory_id,memory_version,content_hash,
               row_index,kind,text,start_offset,end_offset,created_at)
               VALUES(?,1,'hash',0,'sentence','body',0,4,'2026-01-01T00:00:00Z')""",
            (memory_id,),
        )

    with db.diagnostic_connection() as conn:
        shallow = run_all_checks(conn, settings, deep=False)
    assert not any(f.check_id == "database.quick_check" for f in shallow.findings)

    with db.diagnostic_connection() as conn:
        deep = run_all_checks(conn, settings, deep=True)
    findings = {f.check_id: f for f in deep.findings}
    assert findings["database.quick_check"].status == "pass"
    assert findings["database.schema_generation"].status == "pass"
    assert findings["vector.table_dimension"].status == "pass"
    assert findings["vector.evidence_rows"].status == "warn"
    assert findings["vector.evidence_rows"].evidence["missing_vectors"] == 1


def test_deep_doctor_does_not_treat_unqueryable_vec_tables_as_empty(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    # Point at a model so the vector checks are active (model path IS the
    # intent since 0.15.0). Tables that vanish AFTER a dim was recorded must
    # read as unqueryable (warn, vectors None) — never as empty. A
    # never-embedded library (no active dim, lazy tables not yet created)
    # is the healthy pre-first-embed window and must not warn.
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    tools.settings.embedding_model_path = model
    tools.db.set_active_dim(4)
    with tools.db.diagnostic_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS memory_evidence_vec")
        conn.execute("DROP TABLE IF EXISTS workspace_canonicals_vec")
        report = run_all_checks(conn, tools.settings, deep=True)
    findings = {f.check_id: f for f in report.findings}
    assert findings["vector.evidence_rows"].status == "warn"
    assert findings["vector.evidence_rows"].evidence["vectors"] is None
    assert findings["vector.workspace_rows"].status == "warn"
    assert findings["vector.workspace_rows"].evidence["vectors"] is None

    fresh_dir = tmp_path / "fresh"
    fresh_dir.mkdir()
    fresh = review_doctor_make_tools(fresh_dir)
    fresh.settings.embedding_model_path = model
    with fresh.db.diagnostic_connection() as conn:
        report = run_all_checks(conn, fresh.settings, deep=True)
    findings = {f.check_id: f for f in report.findings}
    assert findings["vector.evidence_rows"].status == "pass"
    assert findings["vector.workspace_rows"].status == "pass"


def _review(report):
    return next(f for f in report.findings if f.check_id == "workspace.review")


def _sidecar(tools: MemoryTools) -> Path:
    return Path(tools.settings.db_path).parent / "workspace_review.json"


# ── workspace.review check ───────────────────────────────────────────────────

def test_first_run_lists_all_workspaces_as_unconfirmed(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    _write(tools, "b", "projB")
    # legacy-style default row in the registry must never be listed
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) "
            "VALUES ('default', '2026-01-01T00:00:00Z')",
        )

    report = _run_doctor(tools)
    finding = _review(report)
    assert finding.status == "warn"
    assert finding.severity is Severity.WARNING  # 请确认是例行提示，绝不 critical
    assert finding.severity is not Severity.CRITICAL
    assert sorted(finding.evidence["new"]) == ["projA", "projB"]
    assert "default" not in finding.evidence["current"]
    assert "projA" in finding.detail
    # read-only: a doctor run must not create or refresh the snapshot
    assert not _sidecar(tools).exists()


def test_confirmed_snapshot_clears_check(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    r = tools.memory_govern("confirm_workspaces", {"authorized": True, "reason": "reviewed"})
    assert r["ok"] is True
    assert r["data"]["confirmed"] is True

    snapshot = json.loads(_sidecar(tools).read_text(encoding="utf-8"))
    assert snapshot["version"] == 1
    assert snapshot["confirmed_workspaces"] == ["projA"]
    assert snapshot["confirmed_at"]

    finding = _review(_run_doctor(tools))
    assert finding.status == "pass"
    assert finding.severity is Severity.INFO
    assert finding.evidence["new"] == []


def test_new_workspace_surfaces_as_only_new_item(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    tools.memory_govern("confirm_workspaces", {"authorized": True})

    _write(tools, "b", "projB")
    finding = _review(_run_doctor(tools))
    assert finding.status == "warn"
    assert finding.evidence["new"] == ["projB"]
    assert "projB" in finding.detail


def test_corrupt_sidecar_treated_as_first_full_review(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    tools.memory_govern("confirm_workspaces", {"authorized": True})
    _sidecar(tools).write_text("{not json", encoding="utf-8")

    finding = _review(_run_doctor(tools))  # must not raise
    assert finding.status == "warn"
    assert finding.evidence["new"] == ["projA"]


def test_disappeared_names_are_silently_ignored(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    # manual snapshot mentioning a name that later got merged away
    _sidecar(tools).write_text(
        json.dumps({"confirmed_workspaces": ["projA", "merged-away"], "confirmed_at": "2026-01-01T00:00:00Z", "version": 1}),
        encoding="utf-8",
    )
    finding = _review(_run_doctor(tools))
    assert finding.status == "pass"
    assert finding.evidence["new"] == []


def test_snapshot_after_rename_records_final_registry(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    _write(tools, "b", "projA2")
    renamed = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "projA2", "new": "projA", "reason": "duplicate spelling", "authorized": True,
    })
    assert renamed["ok"] is True

    confirmed = tools.memory_govern("confirm_workspaces", {"authorized": True})
    assert confirmed["data"]["confirmed_workspaces"] == ["projA"]

    finding = _review(_run_doctor(tools))
    assert finding.status == "pass"
    assert finding.evidence["new"] == []


def test_default_terms_never_listed_for_confirmation(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    with tools.db.write_transaction() as conn:
        for name in (DEFAULT_WORKSPACE_NAME, "默认", "none", "未知"):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
                (name,),
            )
    finding = _review(_run_doctor(tools))
    assert finding.status == "pass"
    assert finding.evidence["new"] == []
    assert finding.evidence["current"] == []


# ── confirm_workspaces governance action ─────────────────────────────────────

def test_confirm_workspaces_requires_authorization(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    r = tools.memory_govern("confirm_workspaces", {"reason": "reviewed"})
    assert r["ok"] is False
    assert r["data"]["action_required"] == "ask_user_for_authorization"
    assert r["data"]["governance_action"] == "confirm_workspaces"
    # _GOVERNANCE_IMPACTS has the entry — the authorization error path must
    # not raise KeyError (721 4b 漏一处即崩 item 1).
    assert r["data"]["impact"]
    assert not _sidecar(tools).exists()


def test_confirm_workspaces_default_snapshots_current_registry(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    _write(tools, "b", "projB")
    r = tools.memory_govern("confirm_workspaces", {"authorized": True})
    assert r["ok"] is True
    assert r["data"]["confirmed_workspaces"] == ["projA", "projB"]
    assert r["data"]["count"] == 2
    snapshot = json.loads(_sidecar(tools).read_text(encoding="utf-8"))
    assert snapshot["confirmed_workspaces"] == ["projA", "projB"]
    assert set(snapshot) == {"confirmed_workspaces", "confirmed_at", "version"}


def test_confirm_workspaces_explicit_list_excludes_default_terms(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    r = tools.memory_govern("confirm_workspaces", {
        "authorized": True,
        "workspaces": ["projA", DEFAULT_WORKSPACE_NAME, "默认", " projB "],
    })
    assert r["ok"] is True
    assert r["data"]["confirmed_workspaces"] == ["projA", "projB"]


@pytest.mark.parametrize("bad", [
    "projA",            # not a list
    [],                 # empty list
    [123],              # non-string entry
    [""],               # blank entry
    ["x" * 2001],       # over-long item (validation bound)
    [f"ws-{i}" for i in range(101)],  # over-long list (validation bound)
])
def test_confirm_workspaces_rejects_malformed_lists(tmp_path, bad):
    tools = review_doctor_make_tools(tmp_path)
    r = tools.memory_govern("confirm_workspaces", {"authorized": True, "workspaces": bad})
    assert r["ok"] is False
    assert not _sidecar(tools).exists()


def test_confirm_workspaces_pipeline_bounds_direct_calls(tmp_path):
    """The pipeline re-checks the bound even when called directly (bypassing
    the product-surface validation) — one call cannot write an unbounded
    sidecar."""
    tools = review_doctor_make_tools(tmp_path)
    r = tools.memory_confirm_workspaces(workspaces=["x" * 5000], authorized=True)
    assert r["ok"] is False
    assert "workspaces" in r["data"]["error"]
    r = tools.memory_confirm_workspaces(workspaces=[f"ws-{i}" for i in range(101)], authorized=True)
    assert r["ok"] is False
    assert not _sidecar(tools).exists()


def test_sidecar_write_is_atomic_no_tmp_left_behind(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    tools.memory_govern("confirm_workspaces", {"authorized": True})
    sidecar = _sidecar(tools)
    assert sidecar.exists()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["confirmed_workspaces"] == ["projA"]
    leftovers = [p.name for p in sidecar.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_confirm_pending_workspace_with_default_synonym_raw_no_longer_dead_ends(tmp_path):
    """Round-1 review fix: a pending memory whose raw workspace is a reserved
    default synonym must confirm without recording an (impossible) alias.
    Round-2 review fix: a default-term canonical argument folds to the one
    true spelling instead of re-persisting a phantom synonym canonical."""
    tools = review_doctor_make_tools(tmp_path)
    # Simulate a legacy pending memory written with raw workspace "unknown"
    # (pre-change strict writes produced exactly this shape).
    record = tools.memory_write(
        content="legacy pending memory", workspace="unknown", subject="legacy pending",
        source_type="agent_generated", status="pending",
    )
    mid = record["data"]["id"]
    assert tools.db.get_memory(mid)["status"] == "pending"

    # An agent echoing the memory's own workspace as canonical must NOT
    # create a phantom "unknown" canonical — it folds to "default".
    r = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "default",
        "memory_id": mid, "canonical": "unknown", "authorized": True,
    })
    assert r["ok"] is True, r
    assert r["data"]["confirmed"] is True
    memory = tools.db.get_memory(mid)
    assert memory["status"] == "active"
    assert memory["workspace_canonical"] == DEFAULT_WORKSPACE_NAME
    with tools.db.connection() as conn:
        names = [row["name"] for row in conn.execute("SELECT name FROM workspace_canonicals")]
        aliases = conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases WHERE alias_workspace = 'unknown'"
        ).fetchone()[0]
    assert "unknown" not in names
    assert aliases == 0
    # The confirmed memory is reachable from the default pool.
    found = tools.memory_search(query="legacy pending", workspace="默认")
    assert any(x["id"] == mid for x in found["data"]["results"])


def test_confirm_workspaces_persists_reason_in_snapshot(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    tools.memory_govern("confirm_workspaces", {
        "authorized": True, "reason": "reviewed after merging duplicates",
    })
    snapshot = json.loads(_sidecar(tools).read_text(encoding="utf-8"))
    assert snapshot["reason"] == "reviewed after merging duplicates"
    # doctor still reads the three contract keys regardless of the extra key
    finding = _review(_run_doctor(tools))
    assert finding.status == "pass"


def test_pipeline_rejects_non_list_workspaces_direct_calls(tmp_path):
    """Round-2 review fix: a direct pipeline call bypassing the product
    surface must refuse a bare string instead of confirming its characters."""
    tools = review_doctor_make_tools(tmp_path)
    r = tools.memory_confirm_workspaces(workspaces="projA", authorized=True)
    assert r["ok"] is False
    assert "workspaces" in r["data"]["error"]
    r = tools.memory_confirm_workspaces(workspaces=[None, 123], authorized=True)
    assert r["ok"] is False
    assert not _sidecar(tools).exists()


def test_unknown_fields_warn_but_action_still_works(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    _write(tools, "a", "projA")
    r = tools.memory_govern("confirm_workspaces", {"authorized": True, "bogus": 1})
    assert r["ok"] is True
    assert any("unknown field ignored: bogus" in w for w in r["warnings"])


# ── product-surface wiring (721 4b 八处落点) ─────────────────────────────────

def test_help_and_registry_wire_confirm_workspaces(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    help_doc = tools.memory_govern("help")["data"]
    assert "confirm_workspaces" in help_doc["actions"]
    assert "confirm_workspaces" in help_doc["examples"]
    assert "confirm_workspaces" in help_doc["confirm_actions"]

    from memory_arbiter.surfaces import ProductSurfaces
    assert "confirm_workspaces" in ProductSurfaces._GOVERNANCE_IMPACTS

    from memory_arbiter.validation import PRODUCT_FIELD_REGISTRY
    fields = PRODUCT_FIELD_REGISTRY.get(("memory_govern", "confirm_workspaces"))
    assert fields is not None
    assert {"workspaces", "reason", "authorized"} <= fields


def test_tools_forwarder_exists(tmp_path):
    tools = review_doctor_make_tools(tmp_path)
    r = tools.memory_confirm_workspaces(authorized=True)
    assert r["ok"] is True
    assert _sidecar(tools).exists()


# ── 0.17.1 confirm 清场（prompt suppression 存量治理） ─────────────────────────

def _enqueue_pending_workspace_row(
    tools: MemoryTools, mid: int, own: str, suspected: str, tag: str,
) -> str:
    """Pending workspace row with the FULL detail envelope — 清场解析
    (current_workspace, suspected_workspace)；detail={} 的行自愈/清场不认。"""
    import hashlib

    candidate_key_hash = hashlib.sha256(f"confirm-sweep:{tag}".encode("utf-8")).hexdigest()
    outcome = tools.db.scan_queue.enqueue(
        kind="workspace",
        workspace_canonical=own,
        candidate_key_hash=candidate_key_hash,
        member_versions=[{"memory_id": mid, "version": 1}],
        evidence=[],
        reason="t",
        severity="normal",
        source="test",
        detail={"current_workspace": own, "suspected_workspace": suspected},
    )
    assert outcome.get("outcome") == "queued", outcome
    return candidate_key_hash


def _workspace_row_status(tools: MemoryTools, candidate_key_hash: str) -> str:
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT status FROM scan_queue WHERE candidate_key_hash=?",
            (candidate_key_hash,),
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_confirm_expires_pending_confirmed_pairs(tmp_path):
    """confirm 快照落地后：双确认对的 pending workspace 行即时 expired；
    未确认对（projB→projC，projC 不在快照里）不动。"""
    tools = review_doctor_make_tools(tmp_path)
    mid_a = _write(tools, "a", "projA")
    mid_b = _write(tools, "b", "projB")
    stale_hash = _enqueue_pending_workspace_row(tools, mid_a, "projA", "projB", "stale")
    live_hash = _enqueue_pending_workspace_row(tools, mid_b, "projB", "projC", "live")

    r = tools.memory_govern("confirm_workspaces", {"authorized": True})
    assert r["ok"] is True
    assert r["data"]["suppressed_pending"] == 1
    assert _workspace_row_status(tools, stale_hash) == "expired"
    assert _workspace_row_status(tools, live_hash) == "pending"


def test_confirm_sweep_falls_back_to_canonical_column(tmp_path):
    """R2 F4 补测：detail 缺 current_workspace（旧版/半完整行）时清场回退
    workspace_canonical 列——半 detail 行不比空 detail 行糟，仍须被治理。"""
    tools = review_doctor_make_tools(tmp_path)
    mid_a = _write(tools, "a", "projA")
    _write(tools, "b", "projB")  # 注册 projB，确认快照须含两端才会清场
    import hashlib

    candidate_key_hash = hashlib.sha256(b"confirm-sweep:col-fallback").hexdigest()
    outcome = tools.db.scan_queue.enqueue(
        kind="workspace",
        workspace_canonical="projA",
        candidate_key_hash=candidate_key_hash,
        member_versions=[{"memory_id": mid_a, "version": 1}],
        evidence=[], reason="t", severity="normal", source="test",
        detail={"suspected_workspace": "projB"},  # 无 current_workspace
    )
    assert outcome.get("outcome") == "queued", outcome

    r = tools.memory_govern("confirm_workspaces", {"authorized": True})
    assert r["ok"] is True
    assert r["data"]["suppressed_pending"] == 1
    assert _workspace_row_status(tools, candidate_key_hash) == "expired"


def test_confirm_expiry_failure_warns_not_fails(tmp_path, monkeypatch):
    """清场抛错绝不回滚快照：confirmed 仍 true、suppressed_pending=-1、
    降级 warning（下次 kick 自愈兜底）。"""
    tools = review_doctor_make_tools(tmp_path)
    mid = _write(tools, "a", "projA")
    row_hash = _enqueue_pending_workspace_row(tools, mid, "projA", "projB", "boom")

    def _raising_write_transaction(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tools.db, "write_transaction", _raising_write_transaction)
    r = tools.memory_govern("confirm_workspaces", {"authorized": True})
    assert r["ok"] is True
    assert r["data"]["confirmed"] is True
    assert r["data"]["suppressed_pending"] == -1
    assert any("清场失败" in w for w in r["warnings"]), r["warnings"]
    assert _sidecar(tools).exists(), "快照不得被清场失败回滚"
    assert _workspace_row_status(tools, row_hash) == "pending"