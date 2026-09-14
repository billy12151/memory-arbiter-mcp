"""Equivalence gate for doctor.run_all_checks.

The corpus in tests/golden/doctor.json was generated from the pre-refactor
implementation by scripts/gen_golden_doctor.py. It pins the **ordered** findings
list, because nothing else in the suite can: console_static's doctorSummaryCard
renders ``findings.slice(0, 6)``, while every existing doctor test looks
findings up by check_id and so is blind to a reordering.

Two things a snapshot structurally cannot see are asserted separately below:
how many times the embedder probe is invoked, and when -- both are load-bearing
because the probe on the MCP path is MemoryTools._ensure_embedder, which builds
vec tables and writes _vec_index_meta on first success.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.db import MemoryDB
from memory_arbiter.doctor import report_to_dict, run_all_checks

from doctor_fixtures import (
    FIXTURES,
    PROBES,
    _StubEmbedder,
    _evidence,
    _mask,
    _memory,
    build_settings,
    fx_indexed,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "doctor.json"
GOLDEN: list[dict[str, Any]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
BY_CASE = {snapshot["case_id"]: snapshot["report"] for snapshot in GOLDEN}

FIXTURE_CASES = [
    (name, factory, deep)
    for name, factory, deep_modes in FIXTURES
    for deep in deep_modes
]


def test_corpus_covers_every_check() -> None:
    """A truncated corpus would disarm the gate without failing anything."""
    seen = {f["check_id"] for snap in GOLDEN for f in snap["report"]["findings"]}
    # Conditional checks that only a purpose-built fixture reaches.
    for conditional in (
        "conflicts.scan_epoch", "conflicts.scan_stale", "conflicts.scan_chain",
        "conflicts.spec_drift", "config.warnings",
    ):
        assert conditional in seen, f"{conditional} lost its fixture"
    # Only emitted when the probe hands back a usable embedder.
    for embedder_only in ("vector.device", "evidence.unit_budget"):
        assert embedder_only in seen, f"{embedder_only} lost its probe stub"
    assert len(GOLDEN) == 45


@pytest.mark.parametrize(
    ("name", "factory", "deep"), FIXTURE_CASES,
    ids=[f"{name}/deep={int(deep)}" for name, _factory, deep in FIXTURE_CASES],
)
def test_fixture_report_matches_golden(
    tmp_path: Path, name: str, factory: Any, deep: bool,
) -> None:
    settings, db = factory(tmp_path)
    with db.diagnostic_connection() as conn:
        report = run_all_checks(conn, settings, deep=deep)
    assert _mask(report_to_dict(report)) == BY_CASE[f"{name}/deep={int(deep)}"]


@pytest.mark.parametrize(
    ("probe_name", "probe", "with_model", "prep"), PROBES,
    ids=[name for name, _probe, _with_model, _prep in PROBES],
)
def test_probe_outcome_matches_golden(
    tmp_path: Path, probe_name: str, probe: Any, with_model: bool, prep: Any,
) -> None:
    """All six probe outcomes, including the one that is easy to lose.

    When the probe returns a handle but embed_text raises, vector.device and
    evidence.unit_budget are still produced -- the try only wraps the embed
    call. Folding the embed result into the probe outcome would silently drop
    the GPU-degradation warning.
    """
    settings = build_settings(tmp_path, with_model=with_model)
    db = MemoryDB(settings)
    for mid in (1, 2):
        _memory(db, mid, f"probe subject {mid}")
        _evidence(db, mid, "x" * 200)
    if prep is not None:
        prep(settings, db)
    with db.diagnostic_connection() as conn:
        report = run_all_checks(conn, settings, deep=True, embedder_probe=probe)
    assert _mask(report_to_dict(report)) == BY_CASE[f"probe/{probe_name}"]


def test_probe_is_called_once_on_deep_and_never_otherwise(tmp_path: Path) -> None:
    """Resolving the embedder is expensive and, on the MCP path, mutating."""
    calls: list[int] = []

    def probe() -> tuple[Any, list[str]]:
        calls.append(1)
        return _StubEmbedder(), []

    settings, db = fx_indexed(tmp_path)
    with db.diagnostic_connection() as conn:
        run_all_checks(conn, settings, deep=False, embedder_probe=probe)
        assert calls == [], "deep=False must never resolve the embedder"
        run_all_checks(conn, settings, deep=True, embedder_probe=probe)
        assert len(calls) == 1, f"deep=True must resolve it exactly once, got {len(calls)}"


def test_probe_runs_after_the_vector_state_checks(tmp_path: Path) -> None:
    """The probe must stay lazy, not be resolved while building the context.

    On the MCP path the probe is MemoryTools._ensure_embedder, whose first
    success creates the vec0 tables and writes _vec_index_meta. Resolving it
    up front would therefore change what vector.space, vector.table_dimension,
    vector.evidence_rows and vector.workspace_rows observe. This stub makes the
    mutation visible: if the findings report the post-probe state, the probe
    ran too early.
    """
    settings, db = fx_indexed(tmp_path)

    def mutating_probe() -> tuple[Any, list[str]]:
        raw = sqlite3.connect(str(settings.db_path))
        try:
            raw.execute(
                "INSERT OR REPLACE INTO _vec_index_meta(key,value) VALUES('state','ready')"
            )
            raw.execute(
                "INSERT OR REPLACE INTO _vec_index_meta(key,value)"
                " VALUES('active_space_id','probe-side-effect')"
            )
            # The other half of _ensure_embedder's first-success footprint:
            # vec0-shaped tables spring into existence with a row each. Both
            # halves are needed -- the meta write alone only catches a probe
            # hoisted ahead of the snapshot; the tables catch one hoisted
            # anywhere ahead of the four vector checks.
            raw.execute(
                "CREATE TABLE memory_evidence_vec"
                " (id INTEGER PRIMARY KEY, embedding float[8])"
            )
            raw.execute("INSERT INTO memory_evidence_vec(id, embedding) VALUES (1, 0.0)")
            raw.execute(
                "CREATE TABLE workspace_canonicals_vec"
                " (id INTEGER PRIMARY KEY, embedding float[8])"
            )
            raw.commit()
        finally:
            raw.close()
        return _StubEmbedder(), []

    with db.diagnostic_connection() as conn:
        report = run_all_checks(conn, settings, deep=True, embedder_probe=mutating_probe)
    space = next(f for f in report.findings if f.check_id == "vector.space")
    assert space.evidence is not None
    assert space.evidence["state"] == "unmanaged", (
        "vector.space saw the probe's write: the embedder was resolved before the "
        "vector state checks ran"
    )
    assert space.evidence["active_space_id"] is None
    table_dim = next(f for f in report.findings if f.check_id == "vector.table_dimension")
    evidence_rows = next(f for f in report.findings if f.check_id == "vector.evidence_rows")
    workspace_rows = next(f for f in report.findings if f.check_id == "vector.workspace_rows")
    assert table_dim.evidence is not None and table_dim.evidence["evidence"] is None, (
        "vector.table_dimension saw the probe's tables: the embedder was resolved "
        "too early"
    )
    assert evidence_rows.evidence is not None and evidence_rows.evidence["vectors"] is None, (
        "vector.evidence_rows counted the probe's rows: the embedder was resolved "
        "too early"
    )
    assert workspace_rows.evidence is not None and workspace_rows.evidence["vectors"] is None, (
        "vector.workspace_rows counted the probe's rows: the embedder was resolved "
        "too early"
    )


def test_deep_run_does_not_mutate_vector_state(tmp_path: Path) -> None:
    """doctor is read-only: neither mode may touch the vec bookkeeping."""
    settings, db = fx_indexed(tmp_path)

    def snapshot() -> tuple[list[tuple[str, str]], list[str]]:
        raw = sqlite3.connect(str(settings.db_path))
        try:
            meta = [(str(k), str(v)) for k, v in raw.execute(
                "SELECT key,value FROM _vec_index_meta ORDER BY key"
            )]
            tables = [str(r[0]) for r in raw.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE '%\\_vec%' ESCAPE '\\'"
                " ORDER BY name"
            )]
        finally:
            raw.close()
        return meta, tables

    before = snapshot()
    with db.diagnostic_connection() as conn:
        run_all_checks(conn, settings, deep=False, embedder_probe=lambda: (_StubEmbedder(), []))
        assert snapshot() == before
        run_all_checks(conn, settings, deep=True, embedder_probe=lambda: (_StubEmbedder(), []))
        assert snapshot() == before
