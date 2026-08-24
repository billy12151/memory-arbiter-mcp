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


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "rev.sqlite3",
        backup_jsonl=tmp_path / "rev.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        enable_sqlite_vec=False,
        vec_dim=2,
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
    settings = Settings(
        db_path=tmp_path / "deep.sqlite3",
        backup_jsonl=tmp_path / "deep.jsonl",
        enable_sqlite_vec=True,
        vec_dim=2,
    )
    db = MemoryDB(settings)
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
            """INSERT INTO memory_evidence(memory_id,memory_version,content_hash,
               unit_index,kind,text,start_offset,end_offset,created_at)
               VALUES(?,1,'hash',0,'text','body',0,4,'2026-01-01T00:00:00Z')""",
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
    tools = make_tools(tmp_path)
    tools.settings.enable_sqlite_vec = True
    with tools.db.diagnostic_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS memory_evidence_vec")
        conn.execute("DROP TABLE IF EXISTS workspace_canonicals_vec")
        report = run_all_checks(conn, tools.settings, deep=True)
    findings = {f.check_id: f for f in report.findings}
    assert findings["vector.evidence_rows"].status == "warn"
    assert findings["vector.evidence_rows"].evidence["vectors"] is None
    assert findings["vector.workspace_rows"].status == "warn"
    assert findings["vector.workspace_rows"].evidence["vectors"] is None


def _review(report):
    return next(f for f in report.findings if f.check_id == "workspace.review")


def _sidecar(tools: MemoryTools) -> Path:
    return Path(tools.settings.db_path).parent / "workspace_review.json"


# ── workspace.review check ───────────────────────────────────────────────────

def test_first_run_lists_all_workspaces_as_unconfirmed(tmp_path):
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
    _write(tools, "a", "projA")
    tools.memory_govern("confirm_workspaces", {"authorized": True})

    _write(tools, "b", "projB")
    finding = _review(_run_doctor(tools))
    assert finding.status == "warn"
    assert finding.evidence["new"] == ["projB"]
    assert "projB" in finding.detail


def test_corrupt_sidecar_treated_as_first_full_review(tmp_path):
    tools = make_tools(tmp_path)
    _write(tools, "a", "projA")
    tools.memory_govern("confirm_workspaces", {"authorized": True})
    _sidecar(tools).write_text("{not json", encoding="utf-8")

    finding = _review(_run_doctor(tools))  # must not raise
    assert finding.status == "warn"
    assert finding.evidence["new"] == ["projA"]


def test_disappeared_names_are_silently_ignored(tmp_path):
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
    _write(tools, "a", "projA")
    _write(tools, "b", "projA2")
    renamed = tools.memory_govern("rename_workspace_canonical", {
        "old": "projA2", "new": "projA", "reason": "duplicate spelling", "authorized": True,
    })
    assert renamed["ok"] is True

    confirmed = tools.memory_govern("confirm_workspaces", {"authorized": True})
    assert confirmed["data"]["confirmed_workspaces"] == ["projA"]

    finding = _review(_run_doctor(tools))
    assert finding.status == "pass"
    assert finding.evidence["new"] == []


def test_default_terms_never_listed_for_confirmation(tmp_path):
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
    r = tools.memory_govern("confirm_workspaces", {"reason": "reviewed"})
    assert r["ok"] is False
    assert r["data"]["action_required"] == "ask_user_for_authorization"
    assert r["data"]["governance_action"] == "confirm_workspaces"
    # _GOVERNANCE_IMPACTS has the entry — the authorization error path must
    # not raise KeyError (721 4b 漏一处即崩 item 1).
    assert r["data"]["impact"]
    assert not _sidecar(tools).exists()


def test_confirm_workspaces_default_snapshots_current_registry(tmp_path):
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
    r = tools.memory_govern("confirm_workspaces", {"authorized": True, "workspaces": bad})
    assert r["ok"] is False
    assert not _sidecar(tools).exists()


def test_confirm_workspaces_pipeline_bounds_direct_calls(tmp_path):
    """The pipeline re-checks the bound even when called directly (bypassing
    the product-surface validation) — one call cannot write an unbounded
    sidecar."""
    tools = make_tools(tmp_path)
    r = tools.memory_confirm_workspaces(workspaces=["x" * 5000], authorized=True)
    assert r["ok"] is False
    assert "workspaces" in r["data"]["error"]
    r = tools.memory_confirm_workspaces(workspaces=[f"ws-{i}" for i in range(101)], authorized=True)
    assert r["ok"] is False
    assert not _sidecar(tools).exists()


def test_sidecar_write_is_atomic_no_tmp_left_behind(tmp_path):
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
    r = tools.memory_confirm_workspaces(workspaces="projA", authorized=True)
    assert r["ok"] is False
    assert "workspaces" in r["data"]["error"]
    r = tools.memory_confirm_workspaces(workspaces=[None, 123], authorized=True)
    assert r["ok"] is False
    assert not _sidecar(tools).exists()


def test_unknown_fields_warn_but_action_still_works(tmp_path):
    tools = make_tools(tmp_path)
    _write(tools, "a", "projA")
    r = tools.memory_govern("confirm_workspaces", {"authorized": True, "bogus": 1})
    assert r["ok"] is True
    assert any("unknown field ignored: bogus" in w for w in r["warnings"])


# ── product-surface wiring (721 4b 八处落点) ─────────────────────────────────

def test_help_and_registry_wire_confirm_workspaces(tmp_path):
    tools = make_tools(tmp_path)
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
    tools = make_tools(tmp_path)
    r = tools.memory_confirm_workspaces(authorized=True)
    assert r["ok"] is True
    assert _sidecar(tools).exists()
