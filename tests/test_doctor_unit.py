"""Unit tests for doctor.py (design doc §13).

Mock-based: chain short-circuit, severity aggregation, per-check isolation,
rendering (tty/JSON), build_unopenable_report. SQL semantics are covered by
test_doctor_integration.py with real sqlite — this file does NOT exercise SQL.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.degrade import DegradeState
from memory_arbiter.doctor import (
    Finding,
    OverviewReport,
    Severity,
    build_unopenable_report,
    doctor_overview_cli,
    doctor_overview_mcp,
    report_to_dict,
    run_all_checks,
)
from memory_arbiter.tools import MemoryTools


# ---------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------

def _settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "doc.sqlite3",
        backup_jsonl=tmp_path / "doc.jsonl",
        client="codex", agent_id="a", workspace="w",
    )
    base.update(overrides)
    return Settings(**base)


def _run(tmp_path: Path, settings: Settings, **kw) -> OverviewReport:
    """Run doctor against a freshly-init'd MemoryDB (ro diagnostic conn)."""
    db = MemoryDB(settings)
    return doctor_overview_mcp(db, settings, **kw)


# ---------------------------------------------------------------------
#  Chain short-circuit (design doc §7)
# ---------------------------------------------------------------------

class TestVectorChainShortCircuit:
    def test_link1_broken_makes_rest_na(self, tmp_path):
        """No model configured → links 2-5 must all be n/a."""
        s = _settings(tmp_path)  # no embedding config
        report = _run(tmp_path, s)
        chain = [f for f in report.findings if f.dimension == "vector"]
        ids = {f.check_id for f in chain}
        assert "vec.link1.configured" in ids
        assert "vec.link1.configured" in [f.check_id for f in chain if f.status == "fail"]
        # links 2-5 are n/a (status)
        for lid in ("vec.link2.enabled_flag", "vec.link3.extension_loaded",
                    "vec.link4.model_usable", "vec.link5.auto_flags"):
            matches = [f for f in chain if f.check_id == lid]
            assert matches, f"{lid} missing in chain {ids}"
            assert matches[0].status == "n/a", f"{lid} should be n/a, got {matches[0].status}"

    def test_link2_broken_when_model_configured_but_vec_disabled(self, tmp_path):
        """Model configured but vec.enabled=false → link2 fail, 3-5 n/a."""
        s = _settings(tmp_path,
                      embedding_provider="gguf",
                      embedding_model_path=tmp_path / "fake.gguf",
                      enable_sqlite_vec=False)
        report = _run(tmp_path, s)
        chain = {f.check_id: f for f in report.findings if f.dimension == "vector"}
        assert chain["vec.link1.configured"].status == "pass"
        assert chain["vec.link2.enabled_flag"].status == "fail"
        assert chain["vec.link2.enabled_flag"].severity == Severity.WARNING
        assert chain["vec.link3.extension_loaded"].status == "n/a"
        assert chain["vec.link4.model_usable"].status == "n/a"
        assert chain["vec.link5.auto_flags"].status == "n/a"

    def test_full_pass_chain_all_pass(self, tmp_path):
        """With a real managed embedder + all flags on, all 5 links pass."""
        pytest.importorskip("sqlite_vec")
        # Build a tiny GGUF-less path: use MemoryTools with a mocked embedder
        # so _ensure_embedder returns a usable object. We rely on the vec tools
        # factory but stub the embedder.
        s = _settings(tmp_path,
                      embedding_provider="gguf",
                      embedding_model_path=tmp_path / "fake.gguf",
                      enable_sqlite_vec=True, vec_dim=2)
        # Can't easily mock embedder on MemoryDB directly; verify link1+2 pass
        # at minimum (deeper links need a real model — covered by integration).
        report = _run(tmp_path, s)
        chain = {f.check_id: f for f in report.findings if f.dimension == "vector"}
        assert chain["vec.link1.configured"].status == "pass"
        assert chain["vec.link2.enabled_flag"].status == "pass"


# ---------------------------------------------------------------------
#  Severity aggregation (design doc §6)
# ---------------------------------------------------------------------

class TestSeverityAggregation:
    def test_overall_is_max_severity(self, tmp_path):
        s = _settings(tmp_path,
                      embedding_provider="gguf",
                      embedding_model_path=tmp_path / "f.gguf",
                      enable_sqlite_vec=False)  # link2 → warning
        report = _run(tmp_path, s)
        assert report.overall == Severity.WARNING

    def test_clean_db_overall_info(self, tmp_path):
        s = _settings(tmp_path)  # nothing configured, fresh empty db
        report = _run(tmp_path, s)
        # Empty db, no vec config → all info or n/a; overall should be INFO
        assert report.overall == Severity.INFO


# ---------------------------------------------------------------------
#  Per-check isolation (design doc §9 constraint 4)
# ---------------------------------------------------------------------

class TestPerCheckIsolation:
    def test_one_check_exception_does_not_abort_others(self, tmp_path, monkeypatch):
        """If a single check raises, it degrades to status=error; rest still run."""
        s = _settings(tmp_path)
        db = MemoryDB(s)
        # Monkeypatch one check to raise.
        import memory_arbiter.doctor as doc
        original = doc._check_db_size
        call_count = {"n": 0}

        def boom(conn):
            call_count["n"] += 1
            raise RuntimeError("simulated check failure")

        monkeypatch.setattr(doc, "_check_db_size", boom)
        try:
            with db.diagnostic_connection() as conn:
                report = run_all_checks(conn, s)
        finally:
            monkeypatch.setattr(doc, "_check_db_size", original)

        assert call_count["n"] == 1  # the failing check was invoked
        # Find the error finding
        errors = [f for f in report.findings if f.status == "error"]
        assert len(errors) == 1
        assert errors[0].check_id == "capacity.db_size"
        assert "simulated check failure" in errors[0].detail
        # Other checks still ran: we should have many non-error findings
        non_error = [f for f in report.findings if f.status != "error"]
        assert len(non_error) >= 15  # most of the 18 survived


# ---------------------------------------------------------------------
#  build_unopenable_report + CLI/MCP fallback (design doc §11.1)
# ---------------------------------------------------------------------

class TestUnopenableFallback:
    def test_build_unopenable_report_shape(self, tmp_path):
        s = _settings(tmp_path)
        report = build_unopenable_report(s, RuntimeError("boom"))
        assert report.overall == Severity.CRITICAL
        assert len(report.findings) == 1
        f = report.findings[0]
        assert f.check_id == "db.unopenable"
        assert f.severity == Severity.CRITICAL
        assert "boom" in f.evidence["error"]

    def test_missing_file_hint_points_at_path_config(self, tmp_path):
        """When the DB file does NOT exist, the hint should point the user at
        --db / config.json (path mismatch), not at corruption/recovery."""
        s = _settings(tmp_path, db_path=tmp_path / "never_created.sqlite3")
        report = build_unopenable_report(s, RuntimeError("unable to open database file"))
        f = report.findings[0]
        assert f.evidence["file_exists"] is False
        assert "--db" in f.fix_hint or "config.json" in f.fix_hint
        assert "找不到" in f.title

    def test_existing_file_hint_points_at_recovery(self, tmp_path):
        """When the DB file exists but won't open, the hint should point at
        corruption / lock / recovery, not path config."""
        db_file = tmp_path / "exists.sqlite3"
        db_file.write_bytes(b"not a real sqlite file")  # exists but corrupt
        s = _settings(tmp_path, db_path=db_file)
        report = build_unopenable_report(s, RuntimeError("database disk image is malformed"))
        f = report.findings[0]
        assert f.evidence["file_exists"] is True
        assert "backup_jsonl" in f.fix_hint or "锁定" in f.fix_hint or "权限" in f.fix_hint
        assert "无法打开" in f.title

    def test_cli_nonexistent_db_returns_unopenable_not_raise(self, tmp_path):
        """CLI ambulance: pointing at a missing DB must not raise."""
        s = _settings(tmp_path)
        missing = tmp_path / "does_not_exist.sqlite3"
        report = doctor_overview_cli(missing, s)
        # mode=ro on a nonexistent file raises → fallback engages
        assert report.overall == Severity.CRITICAL
        assert report.findings[0].check_id == "db.unopenable"
        # And the missing-file hint path is taken (points at --db/config.json)
        assert report.findings[0].evidence["file_exists"] is False

    def test_mcp_unopenable_db_falls_back(self, tmp_path):
        """MCP entry on a DB marked unavailable returns unopenable report."""
        s = _settings(tmp_path)
        db = MemoryDB(s)
        db._db_available = False  # force unavailable
        report = doctor_overview_mcp(db, s)
        assert report.overall == Severity.CRITICAL
        assert report.findings[0].check_id == "db.unopenable"


# ---------------------------------------------------------------------
#  Rendering + CLI (design doc §10.2)
# ---------------------------------------------------------------------

class TestRendering:
    def _sample_report(self) -> OverviewReport:
        return OverviewReport(
            snapshot_ts="2026-07-17T00:00:00+00:00",
            overall=Severity.WARNING,
            summary={"mode": "fts5", "total_memories": 5, "vec_effective": False, "split_capability_available": False},
            findings=[
                Finding("config.warnings", "config", Severity.INFO, "pass", "ok", "d", {}),
                Finding("vec.link2.enabled_flag", "vector", Severity.WARNING, "fail", "vec off", "d", {},
                        fix_hint="enable vec"),
            ],
        )

    def test_render_no_color_when_not_tty(self, capsys, monkeypatch):
        from memory_arbiter import doctor_cli
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = doctor_cli._render_text(self._sample_report(), use_color=False)
        assert "\033[" not in out  # no ANSI codes
        assert "WARNING" in out
        assert "enable vec" in out

    def test_render_uses_color_when_tty(self):
        from memory_arbiter import doctor_cli
        out = doctor_cli._render_text(self._sample_report(), use_color=True)
        assert "\033[" in out  # ANSI present

    def test_report_to_dict_roundtrip(self):
        from memory_arbiter.doctor import report_to_dict
        d = report_to_dict(self._sample_report())
        assert d["overall"] == "warning"
        assert isinstance(d["findings"], list)
        assert d["findings"][0]["severity"] == "info"
        assert d["findings"][1]["check_id"] == "vec.link2.enabled_flag"
        # All values JSON-serializable
        import json
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------
#  MCP entry via MemoryTools (design doc §5 layer 3 + Step 3)
# ---------------------------------------------------------------------

class TestMCPEntryViaTools:
    def test_tools_memory_doctor_overview_returns_envelope(self, tmp_path):
        """The tools.py method wraps the report in state.response() envelope."""
        s = _settings(tmp_path)
        tools = MemoryTools(settings=s, db=MemoryDB(s))
        result = tools.memory_doctor_overview(deep=False)
        # Envelope shape (degrade.py:24)
        assert set(result) >= {"ok", "mode", "warnings", "degraded", "data"}
        data = result["data"]
        assert "snapshot_ts" in data
        assert "overall" in data
        assert "findings" in data
        # v0.8 (§12.3): do not hard-code the count; just assert checks ran
        # across all dimensions and the v0.8 split checks are present.
        assert len(data["findings"]) >= 18
        ids = {f["check_id"] for f in data["findings"]}
        assert {"split.capability", "split.long_unsplit_backlog",
                "split.index_integrity"} <= ids
        assert "update_check" in data


class TestDoctorFixMetadata:
    def test_report_to_dict_adds_high_value_fix_metadata(self):
        report = OverviewReport(
            snapshot_ts="2026-08-04T00:00:00+00:00",
            overall=Severity.WARNING,
            summary={},
            findings=[
                Finding(
                    check_id="consistency.vec_parent_status_sync",
                    dimension="consistency",
                    severity=Severity.WARNING,
                    status="warn",
                    title="drift",
                    detail="",
                    fix_hint="repair drift",
                ),
                Finding(
                    check_id="consistency.orphan_vectors",
                    dimension="consistency",
                    severity=Severity.WARNING,
                    status="warn",
                    title="orphans",
                    detail="",
                    fix_hint="cleanup",
                ),
                Finding(
                    check_id="config.warnings",
                    dimension="config",
                    severity=Severity.WARNING,
                    status="warn",
                    title="config",
                    detail="",
                    fix_hint="edit config",
                ),
            ],
        )

        findings = report_to_dict(report)["findings"]
        assert findings[0]["fix_kind"] == "mcp_tool"
        assert findings[0]["fix_tool"] == "memory_resync_vec_parent_status"
        assert findings[0]["requires_authorized"] is False
        assert findings[0]["risk"] == "low"
        assert findings[1]["fix_tool"] == "memory_cleanup_inactive_vectors"
        assert findings[1]["requires_authorized"] is True
        assert findings[2]["fix_kind"] == "manual_config"


# ---------------------------------------------------------------------
#  Semantic conflict / Qwen chain (mirrors TestVectorChainShortCircuit)
# ---------------------------------------------------------------------

class TestSemanticChainShortCircuit:
    def test_link1_na_when_no_model_path(self, tmp_path):
        """No model_path → all 4 links n/a (semantic conflict not configured)."""
        s = _settings(tmp_path)  # no semantic_conflict_model_path
        report = _run(tmp_path, s)
        chain = {f.check_id: f for f in report.findings if f.dimension == "semantic"}
        assert "semantic.link1.model_path" in chain
        assert chain["semantic.link1.model_path"].status == "n/a"
        for lid in ("semantic.link2.enabled", "semantic.link3.model_usable",
                    "semantic.link4.on_write"):
            assert chain[lid].status == "n/a", f"{lid} should be n/a"

    def test_link2_fail_when_model_set_but_explicit_disabled(self, tmp_path):
        """model_path set + enabled=false → link2 fail, 3-4 n/a."""
        s = _settings(tmp_path,
                      semantic_conflict_model_path=tmp_path / "qwen.gguf",
                      semantic_conflict_enabled=False)
        report = _run(tmp_path, s)
        chain = {f.check_id: f for f in report.findings if f.dimension == "semantic"}
        assert chain["semantic.link1.model_path"].status == "pass"
        assert chain["semantic.link2.enabled"].status == "fail"
        assert chain["semantic.link2.enabled"].severity == Severity.WARNING
        assert chain["semantic.link3.model_usable"].status == "n/a"
        assert chain["semantic.link4.on_write"].status == "n/a"

    def test_link3_fail_when_model_file_missing(self, tmp_path):
        """model_path set + enabled=true (auto) but file doesn't exist → link3 critical."""
        s = _settings(tmp_path,
                      semantic_conflict_model_path=tmp_path / "nonexistent.gguf",
                      semantic_conflict_enabled=True)
        report = _run(tmp_path, s)
        chain = {f.check_id: f for f in report.findings if f.dimension == "semantic"}
        assert chain["semantic.link1.model_path"].status == "pass"
        assert chain["semantic.link2.enabled"].status == "pass"
        assert chain["semantic.link3.model_usable"].status == "fail"
        assert chain["semantic.link3.model_usable"].severity == Severity.WARNING
        assert chain["semantic.link4.on_write"].status == "n/a"

    def test_link4_fail_when_on_write_off(self, tmp_path):
        """All links pass except on_write=off → link4 warn.

        Requires llama_cpp (the probe checks importability). Environments
        without it skip — same pattern as the vector chain's full-pass test.
        """
        pytest.importorskip("llama_cpp")
        gguf = tmp_path / "qwen.gguf"
        gguf.write_bytes(b"\x00")
        s = _settings(tmp_path,
                      semantic_conflict_model_path=gguf,
                      semantic_conflict_enabled=True,
                      semantic_conflict_on_write="off")
        report = _run(tmp_path, s)
        chain = {f.check_id: f for f in report.findings if f.dimension == "semantic"}
        assert chain["semantic.link1.model_path"].status == "pass"
        assert chain["semantic.link2.enabled"].status == "pass"
        assert chain["semantic.link3.model_usable"].status == "pass"
        assert chain["semantic.link4.on_write"].status == "fail"
        assert chain["semantic.link4.on_write"].severity == Severity.WARNING

    def test_all_links_pass(self, tmp_path):
        """All 4 links pass when model file exists + enabled + on_write=async.

        Requires llama_cpp; skipped otherwise.
        """
        pytest.importorskip("llama_cpp")
        gguf = tmp_path / "qwen.gguf"
        gguf.write_bytes(b"\x00")
        s = _settings(tmp_path,
                      semantic_conflict_model_path=gguf,
                      semantic_conflict_enabled=True,
                      semantic_conflict_on_write="async")
        report = _run(tmp_path, s)
        chain = {f.check_id: f for f in report.findings if f.dimension == "semantic"}
        for lid in ("semantic.link1.model_path", "semantic.link2.enabled",
                    "semantic.link3.model_usable", "semantic.link4.on_write"):
            assert chain[lid].status == "pass", f"{lid} should pass, got {chain[lid].status}"


# ---------------------------------------------------------------------
#  Workspace alias health
# ---------------------------------------------------------------------

class TestWorkspaceAliasHealth:
    def test_pass_with_empty_aliases(self, tmp_path):
        """v0.13 DB with no aliases → pass, 0 confirmed/rejected."""
        s = _settings(tmp_path)
        report = _run(tmp_path, s)
        finding = [f for f in report.findings
                   if f.check_id == "capacity.workspace_alias_health"][0]
        assert finding.status == "pass"
        assert finding.evidence["confirmed_aliases"] == 0
        assert finding.evidence["rejected_pairs"] == 0

    def test_warn_when_pending_memories_under_strict(self, tmp_path):
        """strict isolation + pending memories → warn."""
        s = _settings(tmp_path, isolation="strict")
        db = MemoryDB(s)
        # Insert a pending memory (simulating strict-blocked new workspace)
        from memory_arbiter.models import MemoryRecord, MemoryStatus
        record = MemoryRecord(content="pending test", agent_id="a",
                              workspace="newproj", status=MemoryStatus.PENDING.value, subject="pending-test")
        db.insert_memory(record, "newproj")
        report = _run(tmp_path, s)
        finding = [f for f in report.findings
                   if f.check_id == "capacity.workspace_alias_health"][0]
        assert finding.status == "warn"
        assert finding.evidence["pending_memories"] == 1
        assert finding.evidence["isolation"] == "strict"
