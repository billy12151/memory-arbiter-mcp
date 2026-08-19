import json
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.config_registry import CONFIG_DESCRIPTORS, grouped_descriptors
from memory_arbiter.models import utc_now_iso
from memory_arbiter.semantic_conflict import notice_dedupe_key
from memory_arbiter.tools import MemoryTools


def tools(tmp_path: Path) -> MemoryTools:
    return MemoryTools(Settings(db_path=tmp_path / "memory.db", backup_jsonl=tmp_path / "backup.jsonl"))


def write(tool: MemoryTools, content: str) -> int:
    return tool.memory_write(content=content, subject="subject", workspace="w")["data"]["id"]


def test_config_registry_only_describes_current_architecture() -> None:
    paths = {item["path"] for item in CONFIG_DESCRIPTORS}
    assert "embedding.max_unit_chars" in paths
    assert not any("claim" in path or "split" in path or "pair_text_gate" in path for path in paths)
    assert sum(len(group["items"]) for group in grouped_descriptors()) == len(CONFIG_DESCRIPTORS)
    assert all(item["label_en"] and item["label_zh"] and item["editable"] is False for item in CONFIG_DESCRIPTORS)


def test_formal_conflict_and_judgment_use_memory_versions(tmp_path: Path) -> None:
    tool = tools(tmp_path)
    left, right = write(tool, "old"), write(tool, "new")
    conflict = tool.memory_record_conflict(
        left, right, "changed", conflict_type="evolution",
        left_version=1, right_version=1,
    )["data"]
    request = tool.db.judgments.build_conflict_judgment_request(conflict["conflict_id"])
    assert request["judge_call"]["data"]["expected_left_version"] == 1
    assert "expected_left_claim_revision" not in request["judge_call"]["data"]
    result = tool.memory_submit_conflict_judgment(
        conflict["conflict_id"], 1, 1, "evolution", "contextual", None,
        "high", "new context", False, "answer",
        resolution_kind="contextual_keep_both", conflict_scope="record",
    )
    assert result["ok"] is True


def test_stale_formal_judgment_is_rejected(tmp_path: Path) -> None:
    tool = tools(tmp_path)
    left, right = write(tool, "old"), write(tool, "new")
    cid = tool.memory_record_conflict(left, right, "changed", left_version=1, right_version=1)["data"]["conflict_id"]
    tool.memory_edit(left, new_content="newer", reason="update")
    result = tool.memory_submit_conflict_judgment(cid, 1, 1, "compatible", "none", None, "high", "same", False, "answer")
    assert result["data"]["outcome"] == "stale_snapshot"


def test_product_notice_delivery_is_version_pinned(tmp_path: Path) -> None:
    tool = tools(tmp_path)
    left, right = write(tool, "left"), write(tool, "right")
    created = tool.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal", notice_type="semantic_evidence",
        title="candidate", message="check", payload={}, left_version=1, right_version=1,
        dedupe_key=notice_dedupe_key(left, right, 1, 1, "semantic_evidence"),
    )
    response = tool.memory(action="help", data={})
    assert response["notices"][0]["notice_id"] == created["notice_id"]
    assert "message" not in response["notices"][0]


def test_stale_notice_is_not_delivered(tmp_path: Path) -> None:
    tool = tools(tmp_path)
    left, right = write(tool, "left"), write(tool, "right")
    tool.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal", notice_type="semantic_evidence",
        title="candidate", message="check", payload={}, left_version=1, right_version=1,
    )
    tool.memory_edit(left, new_content="changed", reason="update")
    response = tool.memory(action="help", data={})
    assert not response.get("notices")
    assert tool.db.semantic_notice_counts().get("stale") == 1


def test_backup_replay_is_authorized_and_idempotent(tmp_path: Path) -> None:
    tool = tools(tmp_path)
    envelope = {
        "backup_schema": 1, "replay_key": "one", "backup_written_at": utc_now_iso(),
        "workspace_canonical": "w",
        "record": {"content": "restored", "subject": "restore", "workspace": "w", "source_type": "agent_generated", "event_time": utc_now_iso()},
    }
    tool.settings.backup_jsonl.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    denied = tool.memory_repair(task="replay_backup", data={"dry_run": False})
    assert denied["ok"] is False
    first = tool.memory_repair(task="replay_backup", data={"dry_run": False, "authorized": True})
    assert first["data"]["imported_count"] == 1
    second = tool.memory_repair(task="replay_backup", data={"dry_run": False, "authorized": True})
    assert second["data"]["already_replayed_count"] == 1


def test_doctor_text_renderer_shows_every_dimension_and_current_summary() -> None:
    from memory_arbiter.doctor import Finding, OverviewReport, Severity
    from memory_arbiter.doctor_cli import _render_text

    findings = [
        Finding("config.writable", "config", Severity.INFO, "pass", "config.writable", "ok"),
        Finding("evidence.coverage", "evidence", Severity.WARNING, "warn", "evidence.coverage", "3/5 memories indexed"),
        Finding("conflicts.backlog", "conflicts", Severity.INFO, "pass", "conflicts.backlog", "0 open conflicts"),
        Finding("notices.backlog", "notices", Severity.INFO, "pass", "notices.backlog", "0 open notices"),
        Finding("future.thing", "future", Severity.WARNING, "warn", "future.thing", "unknown dimension"),
    ]
    report = OverviewReport(
        "2026-08-19T00:00:00+00:00", Severity.WARNING, findings,
        {"mode": "sqlite", "total_memories": 5, "evidence_indexed": 3, "evidence_units": 40},
    )
    text = _render_text(report, use_color=False)
    for dim in ("config", "evidence", "conflicts", "notices", "future"):
        assert f"[{dim}]" in text
    assert "3/5 memories indexed" in text
    assert "evidence 单元: 40" in text
    # Retired summary fields must not render as literal None.
    assert "None" not in text
    assert "分段能力" not in text and "向量生效" not in text
    # A warning-level report must not claim all checks passed.
    assert "所有检查通过" not in text


def test_doctor_deep_probe_reports_dimension_mismatch(tmp_path: Path) -> None:
    import sqlite3 as _sqlite3
    from memory_arbiter.db.schema import SchemaStore
    from memory_arbiter.doctor import run_all_checks
    from memory_arbiter.config import Settings as Cfg

    settings = Cfg(db_path=tmp_path / "d.sqlite3", backup_jsonl=tmp_path / "b.jsonl", vec_dim=2)
    conn = _sqlite3.connect(settings.db_path)
    SchemaStore(
        type("PseudoDB", (), {"settings": settings, "state": None, "_sqlite_vec_loadable": False})()
    )._init_schema(conn)
    conn.close()

    class ProbeEmbedder:
        embedding_space_id = "probe-space"

        def embed_text(self, prefix="", body=""):
            from memory_arbiter.embedder import EmbedResult
            return EmbedResult([0.1, 0.2, 0.3], False, 1, 1)

    report = run_all_checks(
        _sqlite3.connect(settings.db_path), settings, deep=True,
        embedder_probe=lambda: (ProbeEmbedder(), []),
    )
    probe = [f for f in report.findings if f.check_id == "vector.dimension_probe"]
    assert probe and probe[0].status == "warn"
    assert "3" in probe[0].detail and "2" in probe[0].detail


def test_doctor_cli_deep_probe_uses_settings_embedder(monkeypatch, tmp_path) -> None:
    import memory_arbiter.doctor as doctor_mod
    from memory_arbiter.doctor import doctor_overview_cli, report_to_dict
    from memory_arbiter.doctor_cli import _render_text

    class ProbeEmbedder:
        embedding_space_id = "cli-probe"

        def embed_text(self, prefix="", body=""):
            from memory_arbiter.embedder import EmbedResult
            return EmbedResult([0.1, 0.2], False, 1, 1)

    def fake_build_embedder(*_args, **_kwargs):
        return ProbeEmbedder(), []

    # Configured path: the CLI builds the embedder itself and reports a
    # dimension finding instead of the "no resolver" false warning.
    settings = Settings(
        db_path=tmp_path / "d.sqlite3", backup_jsonl=tmp_path / "b.jsonl",
        vec_dim=2, enable_sqlite_vec=True,
        embedding_provider="gguf", embedding_model_path=tmp_path / "m.gguf",
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    import sqlite3 as _sql
    from memory_arbiter.db.schema import SchemaStore
    conn = _sql.connect(settings.db_path)
    SchemaStore(type("P", (), {"settings": settings, "state": None, "_sqlite_vec_loadable": False})())._init_schema(conn)
    conn.close()
    monkeypatch.setattr(doctor_mod, "report_to_dict", doctor_mod.report_to_dict, raising=False)
    import memory_arbiter.embedder as emb
    monkeypatch.setattr(emb, "build_embedder", fake_build_embedder)
    report = doctor_overview_cli(settings, deep=True)
    probes = [f for f in report.findings if f.check_id == "vector.dimension_probe"]
    assert probes and probes[0].status == "pass"
    assert "no embedder resolver" not in _render_text(report, use_color=False)

    # Unconfigured path: skip note, not a warning — healthy DB stays INFO.
    plain = Settings(db_path=tmp_path / "d.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    report = doctor_overview_cli(plain, deep=True)
    probes = [f for f in report.findings if f.check_id == "vector.dimension_probe"]
    assert probes and probes[0].status == "pass"
    assert "skipped" in probes[0].detail
    assert report.overall.value == "info"
