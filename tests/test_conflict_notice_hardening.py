"""Regression tests for the v3 conflict-notice hardening slice.

Covers: model output with missing parsed keys no longer raising KeyError on
the notice path, per-task degradation counting deduplicated by reason,
pending-workspace memories reporting a skipped (not incomplete) conflict job,
and slot_key entity/scope canonicalisation with raw+canon double-form
suppression matching against applying conflict groups (B-C4).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.test_vnext_evidence as tv
from memory_arbiter.evidence import evidence_content_hash
from memory_arbiter.models import ConflictMember, ConflictValueGroup
from memory_arbiter.semantic_conflict import ModelSignal


def _write_pair(tools, meta: dict, left: str = "database is mysql",
                right: str = "database is sqlite") -> tuple[dict, dict]:
    peer = tools.memory_write(content=left, subject="a", tags=[], metadata=meta)["data"]
    new = tools.memory_write(content=right, subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    return peer, new


def _hit(peer_id: int, text: str, *, row_id: int = 1, distance: float = 0.1) -> dict:
    return {
        "memory_id": peer_id, "id": row_id, "kind": "text", "text": text,
        "start_offset": 0, "end_offset": len(text), "distance": distance,
    }


def test_notice_value_groups_tolerate_missing_parsed_keys(tmp_path: Path, monkeypatch) -> None:
    """A notice_ready gate with value keys missing from parsed must not KeyError."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "MyProject", "scope": "Production"}
    peer, new = _write_pair(tools, meta)
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: [_hit(peer["id"], "database is mysql")])

    class Backend:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            # A well-formed signal whose parsed dict omits the value keys.
            parsed = {"attribute_a": "数据库选型", "attribute_b": "数据库选型"}
            return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: Backend())
    gate = SimpleNamespace(
        state="notice_ready", reason="bidirectional_conflict",
        attribute="数据库选型", value_a="sqlite", value_b="mysql",
    )
    monkeypatch.setattr(
        "memory_arbiter.pipeline.evidence.evaluate_pair_extractions", lambda *a, **k: gate,
    )

    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert result["outcome"] == "notices_created"
    notice = tools.db.list_semantic_notices(status="open", limit=10)[0]
    groups = notice["payload"]["value_groups"]
    # Display values fall back to the gate's normalised values.
    assert [group["display_value"] for group in groups] == ["sqlite", "mysql"]
    # B-C4: the notice slot_key is stored in canonical entity/scope form.
    assert notice["payload"]["slot_key"] == {
        "entity": "myproject", "attribute": "数据库选型", "scope": "production",
    }


def test_same_reason_degradation_counted_once_per_task(tmp_path: Path, monkeypatch) -> None:
    """Two pairs failing with the same technical reason bump the counter once."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    peers = [
        tools.memory_write(content=f"database is {value}", subject=value, tags=[], metadata=meta)["data"]
        for value in ("mysql", "postgres")
    ]
    new = tools.memory_write(content="database is sqlite", subject="sqlite", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    hits = [
        _hit(peers[0]["id"], "database is mysql", row_id=1, distance=0.1),
        _hit(peers[1]["id"], "database is postgres", row_id=2, distance=0.2),
    ]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))

    class BadOutput:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            return ModelSignal(False, "invalid_schema", None, "", None, "bad output")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: BadOutput())

    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert result["status"] == "incomplete"
    assert result["reason"] == "qwen_invalid_output"
    assert result["reasons_seen"] == ["qwen_invalid_output"]
    assert tools._check_degradation_count == 1


def test_pending_memory_conflict_job_reports_skipped(tmp_path: Path) -> None:
    """A pending (workspace-activation) memory skips the job instead of incomplete."""
    tools = tv.make_tools(tmp_path)
    written = tools.memory_write(content="database is mysql", subject="a", tags=[], status="pending")
    assert written["ok"] is True
    memory_id = written["data"]["id"]

    result = tools._process_semantic_conflict_job(
        memory_id, {"memory_id": memory_id, "version": 1, "content_hash": "unused"},
    )

    assert result == {
        "status": "skipped", "reason": "pending_workspace_activation", "notices_created": 0,
    }


def _slot_member(memory_id: int, value: str) -> dict:
    quote = f"连接池上限为 {value}。"
    return ConflictMember(
        memory_id=memory_id, version=1, attribute_raw="连接池上限", value_raw=value,
        normalized_attribute="连接池上限", normalized_value=value, evidence_quote=quote,
        evidence_span=(0, len(quote)), content_hash=(str(memory_id) * 64)[:64],
        direction="a_to_b", prompt_version="p1", detector_version="d1",
    ).to_dict()


@pytest.mark.parametrize("stored_entity,stored_scope", [
    ("MyProject", "Production"),   # legacy raw (pre-canon) storage form
    ("myproject", "production"),   # canonical storage form
])
def test_applying_suppression_matches_raw_and_canon_slot_forms(
    tmp_path: Path, monkeypatch, stored_entity: str, stored_scope: str,
) -> None:
    """Suppression hits when either the canon or the raw slot form matches the
    stored applying group, regardless of metadata casing."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "MyProject", "scope": "Production"}
    a, b = _write_pair(tools, meta, left="连接池上限为 10。", right="连接池上限为 20。")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", tv._strict_pair_backend)

    recorded = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": stored_entity, "attribute": "连接池上限", "scope": stored_scope},
        "members": [_slot_member(a["id"], "10"), _slot_member(b["id"], "20")],
        "value_groups": [
            ConflictValueGroup("10", "10", (f"{a['id']}@1",)).to_dict(),
            ConflictValueGroup("20", "20", (f"{b['id']}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scan",
        "reason": "pool size conflict", "status": "open",
    })
    conflict_id = recorded["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "20",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"},
                       {"memory_id": b["id"], "action": "use_as_resolution"}],
        "resolution_memory_id": b["id"], "authorized": True,
    })
    # Storage canonises slot_key at record time; a group stored BEFORE that
    # canonicalisation existed may still carry the raw form — inject it
    # directly to simulate such a legacy row.
    import json as _json
    legacy_slot = {"entity": stored_entity, "attribute": "连接池上限", "scope": stored_scope}
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE conflicts SET slot_key=? WHERE id=?",
            (_json.dumps(legacy_slot, ensure_ascii=False, sort_keys=True), conflict_id),
        )

    tools.db.edit_memory_intent(a["id"], new_content="连接池上限为 30。", reason="apply")
    updated = tools.db.get_memory(a["id"])
    monkeypatch.setattr(
        tools.db, "evidence_knn", lambda *a_, **k: [_hit(b["id"], "连接池上限为 20。")],
    )
    snapshot = {
        "memory_id": a["id"], "version": updated["version"],
        "content_hash": evidence_content_hash(updated["content"]),
        "trusted_applying_context": {
            "conflict_id": conflict_id, "revision": 2, "memory_id": a["id"],
            "action": "update_current_claim", "chosen_value": "20",
        },
    }

    result = tools._process_semantic_conflict_job(a["id"], snapshot)

    # The pair was examined and slot-suppressed (not an error, no new notice).
    assert result["outcome"] == "checked_no_notice"
    fresh = [n for n in tools.db.list_semantic_notices(status="open")
             if {n.get("memory_id"), n.get("peer_id")} == {a["id"], b["id"]}]
    assert fresh == []


def test_scan_slot_key_uses_canonical_entity_scope(tmp_path: Path, monkeypatch) -> None:
    """Scan-path slot keys canonicalise entity/scope (B-C4 comparison side)."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "SVC", "scope": "Production"}
    tools.memory_write(content="生产数据库使用 mysql 方案", subject="a", tags=[], metadata=meta)
    tools.memory_write(content="生产数据库使用 sqlite 方案", subject="b", tags=[], metadata=meta)
    assert tools.wait_evidence_worker_drained(timeout=2)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", tv._grounded_db_backend)

    result = tools.memory_repair("scan_candidates", {"batch": 50, "k": 10})

    assert result["ok"] is True
    groups = result["data"].get("slot_groups") or []
    assert groups
    # The attribute's normalised form is owned by the extraction gate; the
    # canonicalisation under test here covers entity/scope.
    assert groups[0]["slot_key"]["entity"] == "svc"
    assert groups[0]["slot_key"]["scope"] == "production"
