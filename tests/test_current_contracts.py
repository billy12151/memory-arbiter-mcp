import json
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.config_registry import CONFIG_DESCRIPTORS, grouped_descriptors
from memory_arbiter.models import ConflictMember, ConflictValueGroup, utc_now_iso
from memory_arbiter.semantic_conflict import notice_dedupe_key
from memory_arbiter.tools import MemoryTools


def tools(tmp_path: Path) -> MemoryTools:
    return MemoryTools(Settings(db_path=tmp_path / "memory.db", backup_jsonl=tmp_path / "backup.jsonl"))


def write(tool: MemoryTools, content: str) -> int:
    return tool.memory_write(content=content, subject="subject", workspace="w")["data"]["id"]


def test_config_registry_only_describes_current_architecture() -> None:
    paths = {item["path"] for item in CONFIG_DESCRIPTORS}
    assert "embedding.max_unit_chars" in paths
    assert {
        "workspace_weak_vector_weight", "workspace_min_name_len",
        "workspace_recall_admission", "workspace_recall_cutoff",
    } <= paths
    assert not any("claim" in path or "split" in path or "pair_text_gate" in path for path in paths)
    assert sum(len(group["items"]) for group in grouped_descriptors()) == len(CONFIG_DESCRIPTORS)
    assert all(item["label_en"] and item["label_zh"] and item["editable"] is False for item in CONFIG_DESCRIPTORS)


def _record_current_group(tool: MemoryTools, left: int, right: int) -> dict:
    members = [
        ConflictMember(left, 1, "state", "old", "state", "old", "old", (0, 3), "a" * 64, "a_to_b", "p1", "d1"),
        ConflictMember(right, 1, "state", "new", "state", "new", "new", (0, 3), "b" * 64, "b_to_a", "p1", "d1"),
    ]
    return tool.db.record_conflict_group(
        workspace_canonical="w", slot_key={"entity": "subject", "attribute": "state", "scope": "global"},
        members=members, value_groups=[
            ConflictValueGroup("old", "old", (f"{left}@1",)),
            ConflictValueGroup("new", "new", (f"{right}@1",)),
        ], detection_reason="changed", source="scan", detector_version="d1",
    )


def test_formal_conflict_and_decision_use_revisioned_member_versions(tmp_path: Path) -> None:
    tool = tools(tmp_path)
    left, right = write(tool, "old"), write(tool, "new")
    conflict = _record_current_group(tool, left, right)
    result = tool.db.judge_conflict(
        conflict["conflict_id"], expected_revision=1, chosen_value="new",
        decided_by="user", decided_ref="answer", decision_reason="new context",
        resolution_memory_id=right, apply_plan=[
            {"memory_id": left, "action": "update_current_claim"},
            {"memory_id": right, "action": "use_as_resolution"},
        ],
    )
    assert result["outcome"] == "applying"
    row = tool.db.get_conflict(conflict["conflict_id"])
    assert [member["version"] for member in row["member_versions"]] == [1, 1]
    assert row["revision"] == 2


def test_stale_conflict_revision_is_rejected(tmp_path: Path) -> None:
    tool = tools(tmp_path)
    left, right = write(tool, "old"), write(tool, "new")
    conflict = _record_current_group(tool, left, right)
    first = tool.db.judge_conflict(
        conflict["conflict_id"], expected_revision=1, chosen_value="new",
        decided_by="user", decided_ref=None, decision_reason="same",
        apply_plan=[{"memory_id": left, "action": "update_current_claim"}],
    )
    assert first["outcome"] == "applying"
    stale = tool.memory_govern("apply_conflict_action", {
        "conflict_id": conflict["conflict_id"], "expected_revision": 1,
        "memory_id": left, "action": "update_current_claim", "content": "new",
        "authorized": True,
    })
    assert stale["data"]["outcome"] == "stale_conflict"


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


def test_notice_read_calls_execute_all_frozen_members_and_strict_workspace(tmp_path: Path) -> None:
    setting = Settings(
        db_path=tmp_path / "strict.db", backup_jsonl=tmp_path / "backup.jsonl",
        isolation="strict", workspace="w",
    )
    tool = MemoryTools(setting)
    ids = [write(tool, value) for value in ("left", "middle", "right")]
    payload = {
        "member_versions": [
            {"memory_id": memory_id, "version": 1, "value": value, "content_hash": char * 64}
            for memory_id, value, char in zip(ids, ("left", "middle", "right"), "abc")
        ],
        "value_groups": [
            {"normalized_value": value, "display_value": value, "members": [f"{memory_id}@1"]}
            for memory_id, value in zip(ids, ("left", "middle", "right"))
        ],
    }
    created = tool.db.record_semantic_notice(
        memory_id=ids[0], peer_id=ids[1], severity="normal", notice_type="semantic_evidence",
        title="candidate", message="check", payload=payload, left_version=1, right_version=1,
    )
    result = tool.memory_repair("notice", {"action": "read", "notice_id": created["notice_id"], "workspace": "w"})
    notice = result["data"]["notice"]
    assert len(notice["read_calls"]) == 3
    assert "left_read_call" not in notice and "right_read_call" not in notice
    assert "freshness.fresh" in notice["agent_instruction"] and "complete memory" in notice["agent_instruction"]
    reads = [tool.memory(**{key: value for key, value in call.items() if key != "tool"}) for call in notice["read_calls"]]
    assert [item["data"]["memory"]["id"] for item in reads] == ids
    assert all(call["data"]["workspace"] == "w" for call in notice["read_calls"])


def test_delivered_notice_becomes_stale_on_read_and_list(tmp_path: Path) -> None:
    tool = tools(tmp_path)
    left, right = write(tool, "left"), write(tool, "right")
    created = tool.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal", notice_type="semantic_evidence",
        title="candidate", message="check", payload={}, left_version=1, right_version=1,
    )
    assert tool.db.claim_next_semantic_notice()["notice_id"] == created["notice_id"]
    tool.memory_edit(left, new_content="changed", reason="update")
    read = tool.db.read_semantic_notice(created["notice_id"])
    assert read["status"] == "stale" and read["freshness"]["fresh"] is False
    assert tool.db.list_semantic_notices("open") == []
    assert [item["notice_id"] for item in tool.db.list_semantic_notices("stale")] == [created["notice_id"]]


def test_notice_claim_exception_is_nonfatal_warning_and_observable(tmp_path: Path, monkeypatch) -> None:
    tool = tools(tmp_path)
    monkeypatch.setattr(tool.db, "claim_next_semantic_notice", lambda *args: (_ for _ in ()).throw(RuntimeError("claim broke")))
    result = tool.memory(action="help")
    assert result["ok"] is True and result["degraded"] is True
    assert any("semantic_notice_claim_failed: claim broke" in warning for warning in result["warnings"])
    delivery = tool._semantic_status()["notice_delivery"]
    assert delivery["claim_error_count"] == 1
    assert delivery["last_claim_error"] == "claim broke"
    assert delivery["last_claim_error_at"]


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


def test_strict_backup_replay_keeps_unconfirmed_workspace_pending(tmp_path: Path) -> None:
    setting = Settings(
        db_path=tmp_path / "strict-replay.db",
        backup_jsonl=tmp_path / "strict-replay.jsonl",
        isolation="strict", workspace="default",
    )
    tool = MemoryTools(setting)
    envelope = {
        "backup_schema": 1, "replay_key": "strict-one",
        "backup_written_at": utc_now_iso(), "workspace_canonical": "ReplayNew",
        "record": {
            "content": "restored", "subject": "restore", "workspace": "ReplayNew",
            "source_type": "agent_generated", "event_time": utc_now_iso(),
        },
    }
    tool.settings.backup_jsonl.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    replayed = tool.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert replayed["data"]["imported_count"] == 1
    with tool.db.connection() as conn:
        memory = conn.execute(
            "SELECT status FROM memories WHERE workspace_canonical='ReplayNew'"
        ).fetchone()
        canonical = conn.execute(
            "SELECT 1 FROM workspace_canonicals WHERE name='ReplayNew'"
        ).fetchone()
    assert memory["status"] == "pending"
    assert canonical is None
    retried = tool.memory_write(
        content="second", subject="second", workspace="ReplayNew",
        source_type="agent_generated",
    )
    assert retried["data"]["record"]["status"] == "pending"
    assert retried["data"]["action_required"] == "confirm_new_workspace"


def test_backup_replay_follows_current_workspace_redirect(tmp_path: Path) -> None:
    tool = tools(tmp_path)
    tool.memory_write(
        content="old", subject="old", workspace="Old", source_type="agent_generated",
    )
    moved = tool.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "New", "authorized": True,
    })
    assert moved["ok"] is True
    envelope = {
        "backup_schema": 1, "replay_key": "moved-one",
        "backup_written_at": utc_now_iso(), "workspace_canonical": "Old",
        "record": {
            "content": "restored old", "subject": "restore", "workspace": "Old",
            "source_type": "agent_generated", "event_time": utc_now_iso(),
        },
    }
    tool.settings.backup_jsonl.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    replayed = tool.memory_repair("replay_backup", {"dry_run": False, "authorized": True})
    assert replayed["data"]["imported_count"] == 1
    with tool.db.connection() as conn:
        restored = conn.execute(
            "SELECT workspace_canonical FROM memories WHERE content='restored old'"
        ).fetchone()
        old_registry = conn.execute(
            "SELECT 1 FROM workspace_canonicals WHERE name='Old'"
        ).fetchone()
    assert restored["workspace_canonical"] == "New"
    assert old_registry is None


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


def test_doctor_cli_reports_legacy_database_as_upgrade_required(tmp_path: Path) -> None:
    from memory_arbiter.db import MemoryDB
    from memory_arbiter.doctor import doctor_overview_cli, report_to_dict

    settings = Settings(
        db_path=tmp_path / "legacy.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
    )
    db = MemoryDB(settings)
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    report = doctor_overview_cli(settings)
    payload = report_to_dict(report)
    assert payload["overall"] == "critical"
    assert payload["summary"]["mode"] == "upgrade_required"
    assert payload["findings"][0]["check_id"] == "database.upgrade_required"
    assert payload["findings"][0]["evidence"]["generation"] == "legacy"
    assert "mema upgrade --dry-run" in payload["findings"][0]["fix_hint"]
    assert "notice_type" not in payload["findings"][0]["detail"]
