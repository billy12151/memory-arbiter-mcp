"""0.17.1 P2 #20: doctor's semantic.judge_model — model_dir is_dir check and
the enabled-without-ckpt info finding (the silent-misconfig breaker)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.doctor import report_to_dict, run_all_checks


def _report(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "b.jsonl",
        client="doctor", agent_id="doctor",
        **overrides,
    )
    db = MemoryDB(settings)
    with db.diagnostic_connection() as conn:
        return report_to_dict(run_all_checks(conn, settings, deep=False))


def _semantic_finding(report: dict[str, Any]) -> dict[str, Any]:
    return next(f for f in report["findings"] if f["check_id"] == "semantic.judge_model")


def test_model_dir_missing_warns_judge_load_will_fail(tmp_path: Path) -> None:
    ckpt = tmp_path / "mdeberta-v4m_dual_v1.pt"
    ckpt.write_bytes(b"fake-checkpoint")
    report = _report(
        tmp_path,
        semantic_conflict_mdeberta_ckpt=ckpt,
        semantic_conflict_mdeberta_model_dir=tmp_path / "no-such-model-dir",
    )
    finding = _semantic_finding(report)
    assert finding["status"] == "warn"
    assert "judge load will fail" in finding["detail"]
    assert "no-such-model-dir" in finding["detail"]


def test_enabled_without_checkpoint_emits_info_finding(tmp_path: Path) -> None:
    report = _report(tmp_path, semantic_conflict_enabled=True)
    finding = _semantic_finding(report)
    assert finding["status"] == "pass"
    assert "enabled=true but mdeberta_ckpt is not configured" in finding["detail"]


def test_unset_and_disabled_stays_silent(tmp_path: Path) -> None:
    report = _report(tmp_path)
    assert not any(f["check_id"] == "semantic.judge_model" for f in report["findings"])
