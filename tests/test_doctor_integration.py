"""Integration tests for doctor.py SQL semantics (design doc §13).

Real sqlite (in-memory or tmp file), NO mocking of SQL. Each check's SQL is
exercised with both an "expected-to-report" scenario and a "should-not-report"
scenario — this is the hard gate that fts_coverage's broken SQL failed for
four review rounds (design doc v1.4 lesson). Never trust SQL that hasn't run
against a real DB with crafted data.

vec0-table checks (orphan_vectors / section_vec_coverage / vec.link3) need the
sqlite-vec extension; they use pytest.importorskip.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.doctor import (
    Finding,
    Severity,
    _check_history_version_chain,
    _check_orphan_sections,
    _check_orphan_vectors,
    _check_section_vec_coverage,
    _check_split_backlog,
    _check_split_failed,
    _check_split_index_integrity,
    _check_split_legacy_declined,
    _check_split_legacy_unknown_status,
    doctor_overview_cli,
    doctor_overview_mcp,
    run_all_checks,
)
from memory_arbiter.tools import MemoryTools


# ---------------------------------------------------------------------
#  Schema bootstrap: mirror the subset of db.py schema the checks touch.
#  Using MemoryDB itself would also work, but a hand-built minimal schema
#  lets us craft exact pathological states (orphans, broken chains) that
#  the normal write path would never produce.
# ---------------------------------------------------------------------

def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT, status TEXT DEFAULT 'active',
            version INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE memory_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER, version INTEGER NOT NULL, changed_at TEXT);
        CREATE TABLE memory_sections(
            id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id INTEGER NOT NULL);
        CREATE TABLE conflicts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT DEFAULT 'open', created_at TEXT);
        CREATE TABLE _vec_index_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    return conn


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "it.sqlite3",
        backup_jsonl=tmp_path / "it.jsonl",
        client="codex", agent_id="a", workspace="w",
    )
    base.update(overrides)
    return Settings(**base)


# =====================================================================
#  version_chain SQL (design doc §9.C) — the v1.0恒真 bug lived here
# =====================================================================

class TestVersionChain:
    def test_normal_edits_not_reported(self):
        """A memory edited twice (version 1→2→3, history stores 1,2) must NOT report."""
        conn = _make_conn()
        conn.execute("INSERT INTO memories(id, version) VALUES (1, 3)")
        conn.execute("INSERT INTO memory_history(memory_id, version, changed_at) VALUES (1,1,'t1'),(1,2,'t2')")
        conn.commit()
        f = _check_history_version_chain(conn)
        assert f.status == "pass"
        assert f.evidence["broken_count"] == 0

    def test_broken_chain_reported(self):
        """live=5 but hist_max=2 (missing 3,4) → must report."""
        conn = _make_conn()
        conn.execute("INSERT INTO memories(id, version) VALUES (1, 5)")
        conn.execute("INSERT INTO memory_history(memory_id, version, changed_at) VALUES (1,1,'t1'),(1,2,'t2')")
        conn.commit()
        f = _check_history_version_chain(conn)
        assert f.status == "warn"
        assert f.severity == Severity.WARNING
        assert f.evidence["broken_count"] == 1
        assert f.evidence["items"][0]["memory_id"] == 1

    def test_version_1_with_no_history_not_reported(self):
        """A never-edited memory (version=1, no history rows) must not report."""
        conn = _make_conn()
        conn.execute("INSERT INTO memories(id, version) VALUES (1, 1)")
        conn.commit()
        f = _check_history_version_chain(conn)
        assert f.status == "pass"


# =====================================================================
#  orphan_sections SQL (design doc §9.C)
# =====================================================================

class TestOrphanSections:
    def test_clean_not_reported(self):
        conn = _make_conn()
        conn.execute("INSERT INTO memories(id) VALUES (1)")
        conn.execute("INSERT INTO memory_sections(id, memory_id) VALUES (10, 1)")
        conn.commit()
        f = _check_orphan_sections(conn)
        assert f.status == "pass"

    def test_section_pointing_to_superseded_reported(self):
        conn = _make_conn()
        conn.execute("INSERT INTO memories(id, status) VALUES (1, 'superseded')")
        conn.execute("INSERT INTO memory_sections(id, memory_id) VALUES (10, 1)")
        conn.commit()
        f = _check_orphan_sections(conn)
        assert f.status == "warn"
        assert f.evidence["stale_status"] == 1

    def test_physical_orphan_reported(self):
        """Section whose memory_id no longer exists (memory row physically gone)."""
        conn = _make_conn()
        conn.execute("INSERT INTO memory_sections(id, memory_id) VALUES (10, 999)")
        conn.commit()
        f = _check_orphan_sections(conn)
        assert f.status == "warn"
        assert f.evidence["physical_orphans"] == 1


# =====================================================================
#  orphan_vectors SQL (design doc §9.C) — needs vec0 table
# =====================================================================

class TestOrphanVectors:
    def test_table_missing_returns_na(self):
        """When memories_vec table doesn't exist → n/a, not exception."""
        conn = _make_conn()
        # No memories_vec table created
        f = _check_orphan_vectors(conn)
        assert f.status == "n/a"

    def test_orphan_vector_reported(self):
        pytest.importorskip("sqlite_vec")
        import sqlite_vec
        conn = _make_conn()
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("CREATE VIRTUAL TABLE memories_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[2])")
        conn.execute("INSERT INTO memories(id) VALUES (1)")  # memory 1 exists
        conn.execute("INSERT INTO memories_vec(id, embedding) VALUES (1, '[0.1,0.2]')")
        conn.execute("INSERT INTO memories_vec(id, embedding) VALUES (99, '[0.3,0.4]')")  # orphan
        conn.commit()
        f = _check_orphan_vectors(conn)
        assert f.status == "warn"
        assert f.evidence["orphan_vectors"] == 1
        assert 99 in f.evidence["vector_ids"]


# =====================================================================
#  section_vec_coverage SQL (design doc §9.C) — needs vec0 table
# =====================================================================

class TestSectionVecCoverage:
    def test_tables_missing_returns_na(self):
        conn = _make_conn()
        f = _check_section_vec_coverage(conn)
        assert f.status == "n/a"

    def test_missing_section_vec_reported(self):
        pytest.importorskip("sqlite_vec")
        import sqlite_vec
        conn = _make_conn()
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("CREATE VIRTUAL TABLE memory_sections_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[2])")
        conn.execute("INSERT INTO memory_sections(id, memory_id) VALUES (10,1),(11,1),(12,1)")
        conn.execute("INSERT INTO memory_sections_vec(id, embedding) VALUES (10,'[0.1,0.2]'),(12,'[0.3,0.4]')")  # 11 missing
        conn.commit()
        f = _check_section_vec_coverage(conn)
        assert f.status == "warn"
        assert f.evidence["missing_section_vec"] == 1

    def test_full_coverage_not_reported(self):
        pytest.importorskip("sqlite_vec")
        import sqlite_vec
        conn = _make_conn()
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("CREATE VIRTUAL TABLE memory_sections_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[2])")
        conn.execute("INSERT INTO memory_sections(id, memory_id) VALUES (10,1)")
        conn.execute("INSERT INTO memory_sections_vec(id, embedding) VALUES (10,'[0.1,0.2]')")
        conn.commit()
        f = _check_section_vec_coverage(conn)
        assert f.status == "pass"


# =====================================================================
#  Empty DB: no spurious warnings (design doc risk: 误诊)
# =====================================================================

class TestEmptyDB:
    def test_fresh_empty_db_no_warnings(self, tmp_path):
        """A freshly-initialized empty DB must report overall INFO, no warnings."""
        s = _settings(tmp_path)
        db = MemoryDB(s)
        report = doctor_overview_mcp(db, s)
        # Empty db, nothing configured → everything info/pass/na
        warnings = [f for f in report.findings if f.severity in (Severity.WARNING, Severity.CRITICAL)]
        assert report.overall == Severity.INFO, f"unexpected warnings: {[(f.check_id, f.title) for f in warnings]}"
        # v0.8 (§12.3): do NOT hard-code the finding count — checks are added/
        # removed per version. Only assert the dimensions are all represented.
        dims = {f.dimension for f in report.findings}
        assert {"config", "vector", "split", "consistency", "capacity"} <= dims


# =====================================================================
#  MCP/CLI consistency (design doc §5: both share run_all_checks)
# =====================================================================

class TestMCPCLIConsistency:
    def test_both_entries_same_overall_on_same_db(self, tmp_path):
        """Same DB → both entries should agree on overall severity."""
        s = _settings(tmp_path)
        db = MemoryDB(s)  # initializes the on-disk db
        mcp_report = doctor_overview_mcp(db, s)
        cli_report = doctor_overview_cli(s.db_path, s)
        assert mcp_report.overall == cli_report.overall
        # And the finding check_id sets match (order may differ only if isolation reorders)
        mcp_ids = {f.check_id for f in mcp_report.findings}
        cli_ids = {f.check_id for f in cli_report.findings}
        assert mcp_ids == cli_ids


# =====================================================================
#  config.db_writable CLI vs MCP path (design doc §5 runtime_state)
# =====================================================================

class TestRuntimeStatePaths:
    def test_mcp_uses_runtime_state_for_writability(self, tmp_path):
        """MCP entry: db_writable reflects the MCP process's write-probe result."""
        s = _settings(tmp_path)
        db = MemoryDB(s)
        report = doctor_overview_mcp(db, s)
        finding = next(f for f in report.findings if f.check_id == "config.db_writable")
        assert finding.evidence["source"] == "MCP runtime state"
        assert finding.status == "pass"  # tmp_path is writable

    def test_cli_uses_os_access_inference(self, tmp_path):
        """CLI entry: db_writable uses os.access (no MCP runtime state)."""
        s = _settings(tmp_path)
        MemoryDB(s)  # init the file
        report = doctor_overview_cli(s.db_path, s)
        finding = next(f for f in report.findings if f.check_id == "config.db_writable")
        assert "CLI" in finding.evidence["source"]


# =====================================================================
#  vec_effective semantics + vec_version probe (regression for reviewer
#  edits: vec_effective now requires all 5 links pass, not just link3;
#  link3 CLI path now probes vec_version() instead of re-importing)
# =====================================================================

class TestVecEffectiveSemantics:
    def test_vec_effective_false_when_link4_fails(self, tmp_path):
        """vec_effective must be False when any of the 5 links fails.

        Regression: an earlier impl set vec_effective from link3 alone, which
        wrongly reported True when link3 passed but link4 (model) failed.
        """
        pytest.importorskip("sqlite_vec")
        s = _settings(tmp_path,
                      embedding_provider="gguf",
                      embedding_model_path=tmp_path / "nonexistent.gguf",
                      enable_sqlite_vec=True, vec_dim=2)
        db = MemoryDB(s)
        # embedder_probe returns None → link4 fails, even though link3 passes.
        report = doctor_overview_mcp(
            db, s, embedder_probe=lambda: (None, ["model not found"]),
            runtime_state=db.state,
        )
        chain = {f.check_id: f.status for f in report.findings if f.dimension == "vector"}
        assert chain["vec.link3.extension_loaded"] == "pass"
        assert chain["vec.link4.model_usable"] == "fail"
        assert report.summary["vec_effective"] is False

    def test_vec_version_probe_path_on_cli(self, tmp_path):
        """CLI link3 detection uses vec_version() scalar, not re-import.

        When sqlite_vec is loadable, the CLI entry's link3 should pass via
        the vec_version probe; evidence source mentions the probe.
        """
        pytest.importorskip("sqlite_vec")
        s = _settings(tmp_path,
                      embedding_provider="gguf",
                      embedding_model_path=tmp_path / "fake.gguf",
                      enable_sqlite_vec=True, vec_dim=2)
        db = MemoryDB(s)  # initialises schema + loads vec on its init conn
        report = doctor_overview_cli(s.db_path, s)
        link3 = next(f for f in report.findings if f.check_id == "vec.link3.extension_loaded")
        assert link3.status == "pass"
        assert "vec_version probe" in link3.evidence["source"]


# =====================================================================
#  Config-ready-but-data-not-ready: env has model+vec.enabled+extension
#  all configured (5 links pass), but the DB never built a memories_vec
#  table (e.g. an old DB created before vec was enabled). Semantic recall
#  is NOT actually working in this state — vec_effective must be False and
#  mode must not be sqlite_vec, otherwise the report contradicts itself.
#  (Regression: found by testing doctor against the wrong DB path, which
#  happened to be exactly this state — config ready, no vec table.)
# =====================================================================

class TestConfigReadyDataNotReady:
    def test_vec_effective_false_without_vec_table(self, tmp_path):
        """5 links pass but no memories_vec table → vec_effective=False."""
        pytest.importorskip("sqlite_vec")
        import sqlite_vec
        # Build a DB with schema but WITHOUT the vec tables (simulate an old
        # DB that predates vec enablement — drop them after MemoryDB init).
        s = _settings(tmp_path,
                      embedding_provider="gguf",
                      embedding_model_path=tmp_path / "fake.gguf",
                      enable_sqlite_vec=True, vec_dim=2)
        db = MemoryDB(s)  # init creates the vec tables...
        # ...so drop them to simulate the "data not ready" state.
        with db.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS memories_vec")
            conn.execute("DROP TABLE IF EXISTS memory_sections_vec")
            conn.commit()
        # Stub embedder so link4 passes (model "usable"); all 5 links green.
        class _StubEmbedder:
            embedding_space_id = "stub"
            def embed_text(self, prefix="", body="", max_body_chars=None):
                from memory_arbiter.embedder import EmbedResult
                return EmbedResult(embedding=[0.1, 0.2], truncated=False,
                                   original_tokens=0, used_tokens=0)
        report = doctor_overview_mcp(
            db, s, embedder_probe=lambda: (_StubEmbedder(), []),
            runtime_state=db.state,
        )
        chain = {f.check_id: f.status for f in report.findings if f.dimension == "vector"}
        # All 5 links pass (capability ready)...
        assert all(chain[k] == "pass" for k in (
            "vec.link1.configured", "vec.link2.enabled_flag",
            "vec.link3.extension_loaded", "vec.link4.model_usable",
            "vec.link5.auto_flags")), chain
        # ...but vec_effective is False because no vec table (data not ready).
        assert report.summary["vec_effective"] is False
        # And mode is NOT sqlite_vec (no vec table → falls back to fts5/like).
        assert report.summary["mode"] != "sqlite_vec"
        # link3 should note the missing table so the user understands why
        # vec_effective is False despite all-green links.
        link3 = next(f for f in report.findings if f.check_id == "vec.link3.extension_loaded")
        assert link3.evidence["vec_table_exists"] is False
        assert "尚未创建" in link3.detail


# =====================================================================
#  v0.8.0 split findings (design doc §6.5) — crafted pathological data.
#  These checks existed but had ZERO data-driven coverage; this block is the
#  "never trust SQL that hasn't run against crafted data" gate for them.
# =====================================================================

def _make_split_conn() -> sqlite3.Connection:
    """Schema with the columns the v0.8 split checks touch (split_status,
    split_revision, metadata, section offsets, a vec-id table). Hand-built so
    we can craft states the normal write path would never produce."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT, status TEXT DEFAULT 'active',
            version INTEGER NOT NULL DEFAULT 1,
            split_status TEXT, split_revision INTEGER NOT NULL DEFAULT 0,
            metadata TEXT);
        CREATE TABLE memory_sections(
            id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id INTEGER NOT NULL,
            section_index INTEGER NOT NULL, start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL);
        CREATE TABLE memory_sections_vec(id INTEGER PRIMARY KEY);
        CREATE TABLE _vec_index_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    return conn


class TestSplitIndexIntegrity:
    def test_active_but_no_sections_reported(self):
        """active memory, split_status='active', but zero section rows → warn."""
        conn = _make_split_conn()
        conn.execute("INSERT INTO memories(id, status, split_status) VALUES (1,'active','active')")
        conn.commit()
        f = _check_split_index_integrity(conn, _settings(Path(".")))
        assert f.status == "warn"
        assert f.severity == Severity.WARNING
        probs = [i["problem"] for i in f.evidence["issues"]]
        assert "active_but_no_sections" in probs

    def test_missing_section_vec_reported(self):
        """active+active memory with a section row but no matching vec row → warn."""
        conn = _make_split_conn()
        conn.execute("INSERT INTO memories(id, status, split_status) VALUES (1,'active','active')")
        # section id=10 has no vec row; section id=11 has a matching vec row (clean).
        conn.execute("INSERT INTO memory_sections(id, memory_id, section_index, start_offset, end_offset) VALUES (10,1,0,0,5),(11,1,1,5,10)")
        conn.execute("INSERT INTO memory_sections_vec(id) VALUES (11)")
        conn.commit()
        f = _check_split_index_integrity(conn, _settings(Path(".")))
        assert f.status == "warn"
        missing = [i for i in f.evidence["issues"] if i["problem"] == "missing_section_vec"]
        assert len(missing) == 1
        assert missing[0]["memory_id"] == 1
        assert missing[0]["missing"] == 1   # only section 10 is missing its vec

    def test_non_positive_section_length_reported(self):
        """a section with end_offset <= start_offset → offset anomaly warn."""
        conn = _make_split_conn()
        conn.execute("INSERT INTO memories(id, status, split_status) VALUES (1,'active','active')")
        conn.execute("INSERT INTO memory_sections(id, memory_id, section_index, start_offset, end_offset) VALUES (10,1,0,5,5)")
        conn.commit()
        f = _check_split_index_integrity(conn, _settings(Path(".")))
        assert f.status == "warn"
        probs = [i["problem"] for i in f.evidence["issues"]]
        assert "non_positive_section_length" in probs

    def test_clean_active_memory_not_reported(self):
        """active+active memory with 2 well-formed sections + vec rows → pass."""
        conn = _make_split_conn()
        conn.execute("INSERT INTO memories(id, status, split_status) VALUES (1,'active','active')")
        conn.execute("INSERT INTO memory_sections(id, memory_id, section_index, start_offset, end_offset) VALUES (10,1,0,0,5),(11,1,1,5,10)")
        conn.execute("INSERT INTO memory_sections_vec(id) VALUES (10),(11)")
        conn.commit()
        f = _check_split_index_integrity(conn, _settings(Path(".")))
        assert f.status == "pass"
        assert f.evidence["issue_count"] == 0


class TestSplitLegacyStatuses:
    def test_unknown_legacy_status_reported(self):
        """a memory with split_status='pending' (non-v0.8) → legacy_unknown_status warn."""
        conn = _make_split_conn()
        conn.execute("INSERT INTO memories(id, status, split_status) VALUES (1,'active','pending')")
        conn.commit()
        f = _check_split_legacy_unknown_status(conn, _settings(Path(".")))
        assert f.status == "warn"
        assert f.severity == Severity.WARNING
        assert f.evidence["unknown_status_count"] == 1
        assert f.evidence["by_status"]["pending"] == 1

    def test_declined_reported_as_legacy(self):
        """historical 'declined' → legacy_declined (info/warn, NOT a failure)."""
        conn = _make_split_conn()
        conn.execute("INSERT INTO memories(id, status, split_status) VALUES (1,'active','declined')")
        conn.commit()
        f = _check_split_legacy_declined(conn, _settings(Path(".")))
        assert f.evidence["legacy_declined_count"] == 1
        assert 1 in f.evidence["sample_memory_ids"]
        # declined is historical/compat, not a v0.8 failure status
        assert f.severity == Severity.INFO

    def test_clean_statuses_not_reported(self):
        """only v0.8 statuses (NULL/active/failed/declined) → unknown-status pass."""
        conn = _make_split_conn()
        conn.execute("INSERT INTO memories(id, status, split_status) VALUES (1,'active','active'),(2,'active','failed'),(3,'active','declined'),(4,'active',NULL)")
        conn.commit()
        fu = _check_split_legacy_unknown_status(conn, _settings(Path(".")))
        assert fu.status == "pass"
        assert fu.evidence["unknown_status_count"] == 0


class TestSplitFailed:
    def test_failed_with_error_metadata_reported(self):
        """split_status='failed' → failed_count warn, with last_split_error surfaced."""
        import json as _json
        conn = _make_split_conn()
        meta = _json.dumps({"_split": {"last_split_error": {"stage": "embedding", "message": "boom"}}})
        conn.execute("INSERT INTO memories(id, status, split_status, metadata) VALUES (1,'active','failed',?)", (meta,))
        conn.commit()
        f = _check_split_failed(conn, _settings(Path(".")))
        assert f.status == "warn"
        assert f.severity == Severity.WARNING
        assert f.evidence["failed_count"] == 1
        assert f.evidence["recent"][0]["last_split_error"]["stage"] == "embedding"


class TestSplitBacklog:
    def test_long_null_content_reported_when_vec_ready(self):
        """vec ready + active + content≥threshold + split_status NULL → backlog."""
        conn = _make_split_conn()
        conn.execute("INSERT INTO _vec_index_meta(key, value) VALUES ('state','ready')")
        conn.execute("INSERT INTO memories(id, status, split_status, content) VALUES (1,'active',NULL,?)", ("x" * 100,))
        conn.commit()
        s = _settings(Path("."), split_threshold=10)
        f = _check_split_backlog(conn, s)
        assert f.evidence["backlog_count"] == 1
        assert 1 in f.evidence["sample_memory_ids"]

    def test_backlog_not_applicable_when_vec_not_ready(self):
        """vec not ready → backlog check returns n/a (capability down, not a split failure)."""
        conn = _make_split_conn()
        # state not set → not ready
        conn.execute("INSERT INTO memories(id, status, split_status, content) VALUES (1,'active',NULL,?)", ("x" * 100,))
        conn.commit()
        f = _check_split_backlog(conn, _settings(Path("."), split_threshold=10))
        assert f.status == "n/a"
