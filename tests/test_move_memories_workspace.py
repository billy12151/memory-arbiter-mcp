"""memory_govern action move_memories_workspace: by-id workspace bucket moves.

Covers the PR-E contract: authorization gate, default-destination refusal,
destination orthography fold (mechanical twin + confirmed alias), per-id
pre-validation with failed_ids, the divergent-row refuse/force semantics,
strict ACL on both sides, the new-bucket follow-up (canonical registration +
workspace_review notice), and the round-1 review hardening (registry flock,
in-transaction recheck, id coercion, response reconciliation).
"""
from pathlib import Path
import sqlite3

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryStatus
from memory_arbiter.tools import MemoryTools
from memory_arbiter.validation import validate_product_payload


def make_tools(
    tmp_path: Path, isolation: str = "none", *, workspace: str = "default",
) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "move.sqlite3",
        backup_jsonl=tmp_path / "move.jsonl",
        client="codex", agent_id="agent-a", workspace=workspace,
        isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def write(tools: MemoryTools, workspace: str, content: str = "bucket fact") -> int:
    return int(tools.memory_write(
        content=content, subject="bucket", workspace=workspace,
        source_type="agent_generated",
    )["data"]["id"])


def move(tools: MemoryTools, data: dict) -> dict:
    return tools.memory_govern("move_memories_workspace", data)


def row(tools: MemoryTools, memory_id: int) -> dict:
    return tools.db.get_memory(memory_id)


def confirm_pending(tools: MemoryTools, memory_id: int) -> None:
    record = tools.db.get_memory(memory_id)
    assert record["status"] == MemoryStatus.PENDING.value
    outcome = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id,
        "canonical": record["workspace_canonical"] or record["workspace"],
        "authorized": True,
    })
    assert outcome["ok"], outcome


def test_single_and_batch_double_column_write(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    ids = [write(tools, "default", f"fact {i}") for i in range(3)]

    single = move(tools, {"memory_ids": [ids[0]], "new_workspace": "proj-x", "authorized": True})
    assert single["ok"], single
    assert single["data"]["moved_ids"] == [ids[0]]
    assert single["data"]["failed_ids"] == []
    assert row(tools, ids[0])["workspace"] == "proj-x"
    assert row(tools, ids[0])["workspace_canonical"] == "proj-x"

    batch = move(tools, {"memory_ids": [ids[1], ids[2]], "new_workspace": "proj-x", "authorized": True})
    assert batch["ok"], batch
    assert batch["data"]["moved_ids"] == [ids[1], ids[2]]
    for memory_id in ids[1:]:
        assert row(tools, memory_id)["workspace"] == "proj-x"
        assert row(tools, memory_id)["workspace_canonical"] == "proj-x"
    # source=default is allowed; those rows are gone from the default bucket
    assert row(tools, ids[0])["workspace"] != "default"


def test_new_bucket_registered_with_review_notice(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "default")

    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "fresh-bucket", "authorized": True})
    assert outcome["ok"], outcome
    with tools.db.connection() as conn:
        registered = conn.execute(
            "SELECT 1 FROM workspace_canonicals WHERE name = 'fresh-bucket'"
        ).fetchone()
    assert registered is not None
    notices = outcome.get("notices") or []
    assert any(
        notice.get("type") == "workspace_review"
        and notice.get("workspace") == "fresh-bucket"
        and notice.get("action_required") == "review_workspace_registry"
        for notice in notices
    ), notices


def test_existing_bucket_no_review_notice(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    write(tools, "proj-x")  # registers the canonical through the write path
    memory_id = write(tools, "default")

    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "proj-x", "authorized": True})
    assert outcome["ok"], outcome
    assert not any(
        notice.get("type") == "workspace_review" for notice in outcome.get("notices") or []
    )


def test_authorization_gate(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "default")
    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "proj-x"})
    assert not outcome["ok"]
    assert outcome["data"]["action_required"] == "ask_user_for_authorization"
    assert outcome["data"]["governance_action"] == "move_memories_workspace"
    assert row(tools, memory_id)["workspace"] == "default"


def test_default_destination_refused(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "proj-a")
    for destination in ("default", "默认"):
        outcome = move(tools, {
            "memory_ids": [memory_id], "new_workspace": destination, "authorized": True,
        })
        assert not outcome["ok"]
        assert "reserved global pool" in outcome["data"]["error"]
    assert row(tools, memory_id)["workspace"] == "proj-a"


def test_failed_ids_and_partial_success(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    real_id = write(tools, "default")
    outcome = move(tools, {
        "memory_ids": [real_id, 999999], "new_workspace": "proj-x", "authorized": True,
    })
    assert not outcome["ok"]
    assert outcome["data"]["moved_ids"] == [real_id]
    assert outcome["data"]["failed_ids"] == [999999]
    assert outcome["data"]["errors"][0]["reason"] == "not_found_or_forbidden"
    assert row(tools, real_id)["workspace"] == "proj-x"


def test_destination_folds_to_registered_canonical(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    write(tools, "proj-x")  # registered spelling
    memory_id = write(tools, "default")

    outcome = move(tools, {
        "memory_ids": [memory_id], "new_workspace": "Proj X", "authorized": True,
    })
    assert outcome["ok"], outcome
    assert outcome["data"]["new_workspace"] == "proj-x"
    assert outcome["data"]["requested_new_workspace"] == "Proj X"
    assert row(tools, memory_id)["workspace"] == "proj-x"


def test_registry_strips_unrelated_fields(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "default")
    outcome = move(tools, {
        "memory_ids": [memory_id], "new_workspace": "proj-x", "authorized": True,
        "new_content": "ignored", "new_tags": ["ignored"], "content": "ignored",
    })
    assert outcome["ok"], outcome
    warnings = outcome.get("warnings") or []
    assert "unknown field ignored: new_content" in warnings
    assert "unknown field ignored: new_tags" in warnings
    assert row(tools, memory_id)["content"].startswith("bucket fact")


def test_memory_ids_input_validation(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "default")
    for bad_ids in ([], "not-a-list", [0], ["abc"], [memory_id, -1]):
        outcome = move(tools, {"memory_ids": bad_ids, "new_workspace": "proj-x", "authorized": True})
        assert not outcome["ok"], bad_ids
        assert row(tools, memory_id)["workspace"] == "default"
    missing = move(tools, {"new_workspace": "proj-x", "authorized": True})
    assert not missing["ok"]
    missing_ws = move(tools, {"memory_ids": [memory_id], "authorized": True})
    assert not missing_ws["ok"]
    wrong_type = move(tools, {"memory_ids": [memory_id], "new_workspace": ["proj"], "authorized": True})
    assert not wrong_type["ok"]


def test_strict_target_bucket_forbidden(tmp_path: Path) -> None:
    tools = make_tools(tmp_path, isolation="strict", workspace="proj-a")
    memory_id = write(tools, "proj-a")
    confirm_pending(tools, memory_id)

    outcome = move(tools, {
        "memory_ids": [memory_id], "new_workspace": "proj-b", "authorized": True,
        "workspace": "proj-a",
    })
    assert not outcome["ok"]
    assert outcome["data"]["error"].startswith("forbidden_strict_workspace")
    assert row(tools, memory_id)["workspace"] == "proj-a"


def test_strict_source_visibility_per_id(tmp_path: Path) -> None:
    tools = make_tools(tmp_path, isolation="strict", workspace="proj-a")
    inside_id = write(tools, "proj-a")
    confirm_pending(tools, inside_id)
    # Seed an out-of-scope memory through a non-strict handle on the same DB.
    loose = make_tools(tmp_path)
    outside_id = write(loose, "proj-b")

    outcome = move(tools, {
        "memory_ids": [inside_id, outside_id], "new_workspace": "proj-a",
        "authorized": True, "workspace": "proj-a",
    })
    assert not outcome["ok"]
    assert outcome["data"]["moved_ids"] == [inside_id]
    assert outcome["data"]["failed_ids"] == [outside_id]
    assert outcome["data"]["errors"][0]["reason"] == "not_found_or_forbidden"
    # the out-of-scope row stayed in its own bucket
    assert row(loose, outside_id)["workspace"] == "proj-b"


def test_divergent_row_refused_without_authorized(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "ws-a")
    ok_set, warnings = tools.db.set_memory_workspace_canonical(memory_id, "ws-b")
    assert ok_set, warnings

    # Direct operations-level call bypassing the surface authorization gate.
    outcome = tools.memory_move_memories_workspace(
        memory_ids=[memory_id], new_workspace="proj-x", authorized=False,
    )
    assert not outcome["ok"]
    assert outcome["data"]["failed_ids"] == [memory_id]
    assert outcome["data"]["errors"][0]["reason"] == "canonical_diverged"
    assert row(tools, memory_id)["workspace"] == "ws-a"
    assert row(tools, memory_id)["workspace_canonical"] == "ws-b"


def test_divergent_row_forced_with_authorized_and_hint(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "ws-a")
    ok_set, warnings = tools.db.set_memory_workspace_canonical(memory_id, "ws-b")
    assert ok_set, warnings

    outcome = move(tools, {
        "memory_ids": [memory_id], "new_workspace": "proj-x", "authorized": True,
    })
    assert outcome["ok"], outcome
    assert outcome["data"]["moved_ids"] == [memory_id]
    forced = outcome["data"]["forced_reanchored"]
    assert forced["memory_ids"] == [memory_id]
    assert "normalization rules are unchanged" in forced["note"].lower()
    assert "migrate_workspace" in forced["note"]
    assert forced["suggested_call"]["action"] == "migrate_workspace"
    assert forced["suggested_call"]["data"]["from"] == "ws-b"
    assert forced["suggested_call"]["data"]["to"] == "proj-x"
    moved_row = row(tools, memory_id)
    assert moved_row["workspace"] == "proj-x"
    assert moved_row["workspace_canonical"] == "proj-x"


def test_move_is_idempotent(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "default")
    first = move(tools, {"memory_ids": [memory_id], "new_workspace": "proj-x", "authorized": True})
    second = move(tools, {"memory_ids": [memory_id], "new_workspace": "proj-x", "authorized": True})
    assert first["ok"] and second["ok"]
    assert second["data"]["moved_ids"] == [memory_id]
    assert row(tools, memory_id)["workspace"] == "proj-x"
    assert row(tools, memory_id)["workspace_canonical"] == "proj-x"


def record_conflict(tools: MemoryTools, workspace: str, left: int, right: int) -> int:
    members = [
        ConflictMember(
            memory_id=memory_id, version=1, attribute_raw="database", value_raw=value,
            normalized_attribute="database", normalized_value=value,
            evidence_quote=value, evidence_span=(0, len(value)), content_hash=char * 64,
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        )
        for memory_id, value, char in ((left, "mysql", "a"), (right, "sqlite", "b"))
    ]
    outcome = tools.db.record_conflict_group(
        workspace_canonical=workspace,
        slot_key={"entity": "svc", "attribute": "database", "scope": "production"},
        members=members,
        value_groups=[
            ConflictValueGroup("mysql", "MySQL", (f"{left}@1",)),
            ConflictValueGroup("sqlite", "SQLite", (f"{right}@1",)),
        ],
        detection_reason="different", source="scan", detector_version="d1",
    )
    assert outcome["outcome"] == "inserted"
    return int(outcome["conflict_id"])


def test_strict_settings_fallback_caller(tmp_path: Path) -> None:
    tools = make_tools(tmp_path, isolation="strict", workspace="proj-a")
    memory_id = write(tools, "proj-a")
    confirm_pending(tools, memory_id)
    # No payload workspace: the caller resolves from settings.workspace.
    forbidden = move(tools, {"memory_ids": [memory_id], "new_workspace": "proj-b", "authorized": True})
    assert not forbidden["ok"]
    assert forbidden["data"]["error"].startswith("forbidden_strict_workspace")
    same_bucket = move(tools, {"memory_ids": [memory_id], "new_workspace": "proj-a", "authorized": True})
    assert same_bucket["ok"], same_bucket


def test_destination_follows_confirmed_alias(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    ok_decide, decide_warnings = tools.db.record_workspace_decision(
        "旧名", "新名", status="confirmed", force=False,
    )
    assert ok_decide, decide_warnings
    memory_id = write(tools, "default")

    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "旧名", "authorized": True})
    assert outcome["ok"], outcome
    # the destination lands on the decision canonical, never a shadow one
    assert outcome["data"]["new_workspace"] == "新名"
    assert outcome["data"]["requested_new_workspace"] == "旧名"
    assert row(tools, memory_id)["workspace"] == "新名"
    with tools.db.connection() as conn:
        shadow = conn.execute(
            "SELECT 1 FROM workspace_canonicals WHERE name='旧名'"
        ).fetchone()
    assert shadow is None


@pytest.mark.parametrize("variant", ["PROJ-X", "proj_x", "Proj X"])
def test_mechanical_twin_fold_variants(tmp_path: Path, variant: str) -> None:
    tools = make_tools(tmp_path)
    write(tools, "proj-x")  # registered spelling
    memory_id = write(tools, "default")
    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": variant, "authorized": True})
    assert outcome["ok"], outcome
    assert outcome["data"]["new_workspace"] == "proj-x"
    assert outcome["data"]["requested_new_workspace"] == variant


def test_new_workspace_length_cap(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "default")
    outcome = move(tools, {
        "memory_ids": [memory_id], "new_workspace": "x" * 5000, "authorized": True,
    })
    assert not outcome["ok"]
    assert outcome["data"]["error"] == "invalid_input"
    assert outcome["data"]["field"] == "new_workspace"
    assert row(tools, memory_id)["workspace"] == "default"


def test_direct_call_id_coercion_and_batch_cap(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "default")
    for bad_ids in ([True], [1.5], [0], [-1]):
        outcome = tools.memory_move_memories_workspace(
            memory_ids=bad_ids, new_workspace="proj-x", authorized=True,
        )
        assert not outcome["ok"], bad_ids
        assert row(tools, memory_id)["workspace"] == "default"
    too_many = tools.memory_move_memories_workspace(
        memory_ids=list(range(1, 1002)), new_workspace="proj-x", authorized=True,
    )
    assert not too_many["ok"]
    assert "at most 1000" in too_many["data"]["error"]


def test_memory_ids_batch_limit_validation() -> None:
    within = validate_product_payload(
        "memory_govern", "move_memories_workspace",
        {"memory_ids": list(range(1, 1001)), "new_workspace": "x", "authorized": True},
    )
    assert within.error is None
    beyond = validate_product_payload(
        "memory_govern", "move_memories_workspace",
        {"memory_ids": list(range(1, 1002)), "new_workspace": "x", "authorized": True},
    )
    assert beyond.error is not None
    assert beyond.error["field"] == "memory_ids"


def test_divergence_window_recheck_refuses(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "ws-a")
    store = tools.db.workspaces
    original = store.prepare_missing_workspace_canonical_embedding

    def diverge_then_prepare(canonical: str, embedder):
        # Simulate a concurrent change landing between pre-validation and the
        # write transaction: the row becomes divergent in the window.
        ok_set, set_warnings = store.set_memory_workspace_canonical(memory_id, "ws-b")
        assert ok_set, set_warnings
        return original(canonical, embedder)

    monkeypatch.setattr(store, "prepare_missing_workspace_canonical_embedding", diverge_then_prepare)
    outcome = tools.memory_move_memories_workspace(
        memory_ids=[memory_id], new_workspace="proj-x", authorized=False,
    )
    assert not outcome["ok"]
    assert outcome["data"]["failed_ids"] == [memory_id]
    assert outcome["data"]["errors"][0]["reason"] == "canonical_diverged"
    assert row(tools, memory_id)["workspace_canonical"] == "ws-b"


def test_divergence_window_recheck_forces_when_authorized(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "ws-a")
    store = tools.db.workspaces
    original = store.prepare_missing_workspace_canonical_embedding

    def diverge_then_prepare(canonical: str, embedder):
        ok_set, set_warnings = store.set_memory_workspace_canonical(memory_id, "ws-b")
        assert ok_set, set_warnings
        return original(canonical, embedder)

    monkeypatch.setattr(store, "prepare_missing_workspace_canonical_embedding", diverge_then_prepare)
    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "proj-x", "authorized": True})
    assert outcome["ok"], outcome
    forced = outcome["data"]["forced_reanchored"]
    assert forced["memory_ids"] == [memory_id]
    assert forced["details"][0]["workspace_canonical"] == "ws-b"
    assert forced["authorization_required"] is True
    assert row(tools, memory_id)["workspace_canonical"] == "proj-x"


def test_vanished_row_window_fails_per_id(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    doomed_id = write(tools, "default")
    survivor_id = write(tools, "default")
    store = tools.db.workspaces
    original = store.prepare_missing_workspace_canonical_embedding

    def delete_then_prepare(canonical: str, embedder):
        with tools.db.write_transaction() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (doomed_id,))
        return original(canonical, embedder)

    monkeypatch.setattr(store, "prepare_missing_workspace_canonical_embedding", delete_then_prepare)
    outcome = move(tools, {
        "memory_ids": [doomed_id, survivor_id], "new_workspace": "proj-x", "authorized": True,
    })
    assert not outcome["ok"]
    assert outcome["data"]["moved_ids"] == [survivor_id]
    assert outcome["data"]["failed_ids"] == [doomed_id]
    assert outcome["data"]["errors"][0]["reason"] == "not_found_or_forbidden"
    assert row(tools, survivor_id)["workspace"] == "proj-x"


def test_moved_non_active_annotation(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    keep_id = write(tools, "default")
    old_id = write(tools, "default", content="retired fact")
    retired = tools.memory_govern("retire", {
        "memory_id": old_id, "reason": "superseded duplicate", "authorized": True,
    })
    assert retired["ok"], retired

    outcome = move(tools, {
        "memory_ids": [keep_id, old_id], "new_workspace": "proj-x", "authorized": True,
    })
    assert outcome["ok"], outcome
    assert outcome["data"]["moved_non_active"] == [{"memory_id": old_id, "status": "superseded"}]
    assert row(tools, old_id)["status"] == "superseded"


def test_conflict_scope_note_lists_foreign_scoped_conflicts(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = write(tools, "ws-old")
    right = write(tools, "ws-old", content="other value")
    conflict_id = record_conflict(tools, "ws-old", left, right)

    outcome = move(tools, {"memory_ids": [left], "new_workspace": "proj-x", "authorized": True})
    assert outcome["ok"], outcome
    note = outcome["data"]["conflict_scope_note"]
    assert note["conflict_ids"] == [conflict_id]
    assert "scan_candidates" in note["note"]


def test_governance_impact_and_help_disclosures(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "default")
    gate = move(tools, {"memory_ids": [memory_id], "new_workspace": "proj-x"})
    assert not gate["ok"]
    assert "re-anchor" in gate["data"]["impact"].lower()
    help_doc = tools.memory_govern("help")["data"]
    assert "pending" in help_doc["workspace_move_vs_migrate"]
    assert "moved_non_active" in help_doc["workspace_move_vs_migrate"]


def test_alias_fold_is_single_hop(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    ok1, w1 = tools.db.record_workspace_decision("a-side", "Bee", status="confirmed")
    ok2, w2 = tools.db.record_workspace_decision("Bee", "Cee", status="confirmed")
    assert ok1 and ok2, (w1, w2)
    memory_id = write(tools, "default")

    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "a-side", "authorized": True})
    assert outcome["ok"], outcome
    # exactly one alias hop — the same canonical a write using 'a-side' lands on
    assert outcome["data"]["new_workspace"] == "Bee"
    assert row(tools, memory_id)["workspace"] == "Bee"


def test_alias_fold_beats_mechanical_twin(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    write(tools, "AgentLane")  # registers the mechanical twin of 'agent-lane'
    ok_decide, decide_warnings = tools.db.record_workspace_decision(
        "agent-lane", "Elsewhere", status="confirmed",
    )
    assert ok_decide, decide_warnings
    # the write path follows the alias first; the move fold must agree
    resolved = tools.db.resolve_workspace_canonical("agent-lane", None, register_new=False)
    assert resolved["canonical"] == "Elsewhere"

    memory_id = write(tools, "default")
    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "agent-lane", "authorized": True})
    assert outcome["ok"], outcome
    assert outcome["data"]["new_workspace"] == "Elsewhere"
    assert row(tools, memory_id)["workspace"] == "Elsewhere"


def test_vanish_then_abort_reconciles_without_duplicates(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    doomed_id = write(tools, "default", content="doomed")
    other_id = write(tools, "default", content="other")
    store = tools.db.workspaces
    original_prepare = store.prepare_missing_workspace_canonical_embedding
    original_move = store.move_memory_workspace_on_conn

    def delete_then_prepare(canonical: str, embedder):
        with tools.db.write_transaction() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (doomed_id,))
        return original_prepare(canonical, embedder)

    def flaky_move(conn, memory_id, workspace, *, precomputed_embedding=None):
        if int(memory_id) == other_id:
            raise sqlite3.OperationalError("disk I/O error")
        return original_move(
            conn, memory_id, workspace, precomputed_embedding=precomputed_embedding,
        )

    monkeypatch.setattr(store, "prepare_missing_workspace_canonical_embedding", delete_then_prepare)
    monkeypatch.setattr(store, "move_memory_workspace_on_conn", flaky_move)

    outcome = move(tools, {
        "memory_ids": [doomed_id, other_id], "new_workspace": "proj-x", "authorized": True,
    })
    assert not outcome["ok"]
    failed = outcome["data"]["failed_ids"]
    assert sorted(failed) == [doomed_id, other_id]
    assert len(failed) == len(set(failed))  # the vanished id left the pending queue
    reasons = {entry["memory_id"]: entry["reason"] for entry in outcome["data"]["errors"]}
    assert reasons[doomed_id] == "not_found_or_forbidden"
    assert reasons[other_id].startswith("aborted:")
    assert outcome["data"]["moved_ids"] == []


def test_inlock_refold_drops_stale_embedding(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    write(tools, "Fresh")  # registered spelling
    memory_id = write(tools, "default")
    store = tools.db.workspaces

    # Blind the OUTSIDE fold so it misses the registered twin, while the
    # in-lock re-fold (on the transaction connection) still sees it — the
    # concurrency window round 2 proved must drop the prepared embedding.
    monkeypatch.setattr(store, "registered_mechanical_canonical", lambda name: None)
    monkeypatch.setattr(
        store, "prepare_missing_workspace_canonical_embedding",
        lambda canonical, embedder: [1.0, 2.0],
    )

    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "fresh", "authorized": True})
    assert outcome["ok"], outcome
    assert outcome["data"]["new_workspace"] == "Fresh"
    assert outcome["data"]["requested_new_workspace"] == "fresh"
    assert row(tools, memory_id)["workspace"] == "Fresh"
    # the stale 'fresh' embedding was NOT published under 'Fresh': with vec
    # unavailable a surviving embedding would surface as a publish-failed
    # warning — assert it does not
    assert not any(
        "workspace canonical vector publish failed" in warning
        for warning in outcome.get("warnings") or []
    )


def test_inlock_blocked_destination_reconciles_all_ids(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, isolation="strict", workspace="proj-a")
    memory_id = write(tools, "proj-a")
    confirm_pending(tools, memory_id)
    store = tools.db.workspaces

    # Blind the outside fold; a concurrent alias decision lands in the window
    # between the fold and the lock, so only the in-lock re-fold sees it.
    monkeypatch.setattr(store, "registered_mechanical_canonical", lambda name: None)
    monkeypatch.setattr(store, "confirmed_alias_canonical", lambda name: None)
    original_prepare = store.prepare_missing_workspace_canonical_embedding

    def alias_lands_in_window(canonical: str, embedder):
        ok_decide, decide_warnings = tools.db.record_workspace_decision(
            "proj-a", "elsewhere", status="confirmed",
        )
        assert ok_decide, decide_warnings
        return original_prepare(canonical, embedder)

    monkeypatch.setattr(store, "prepare_missing_workspace_canonical_embedding", alias_lands_in_window)

    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "proj-a", "authorized": True})
    assert not outcome["ok"]
    assert outcome["data"]["error"].startswith("forbidden_strict_workspace")
    # reconciliation: every requested id is accounted for
    assert outcome["data"]["moved_ids"] == []
    assert outcome["data"]["failed_ids"] == [memory_id]
    assert outcome["data"]["errors"][0]["reason"] == "destination_changed_after_recheck"
    untouched = row(tools, memory_id)
    assert untouched["workspace"] == "proj-a"
    assert untouched["workspace_canonical"] == "proj-a"


def test_abort_path_no_double_counting(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    id_a = write(tools, "default", content="fact a")
    id_b = write(tools, "ws-b", content="fact b")
    id_c = write(tools, "default", content="fact c")
    store = tools.db.workspaces
    original_prepare = store.prepare_missing_workspace_canonical_embedding
    original_move = store.move_memory_workspace_on_conn

    def diverge_b_then_prepare(canonical: str, embedder):
        ok_set, set_warnings = store.set_memory_workspace_canonical(id_b, "ws-diverted")
        assert ok_set, set_warnings
        return original_prepare(canonical, embedder)

    def flaky_move(conn, memory_id, workspace, *, precomputed_embedding=None):
        if int(memory_id) == id_c:
            raise sqlite3.OperationalError("disk I/O error")
        return original_move(
            conn, memory_id, workspace, precomputed_embedding=precomputed_embedding,
        )

    monkeypatch.setattr(store, "prepare_missing_workspace_canonical_embedding", diverge_b_then_prepare)
    monkeypatch.setattr(store, "move_memory_workspace_on_conn", flaky_move)

    # Direct operations-level call with authorized=False: the in-transaction
    # divergence recheck must fail id_b while id_a/id_c hit the abort path.
    outcome = tools.memory_move_memories_workspace(
        memory_ids=[id_a, id_b, id_c], new_workspace="proj-x", authorized=False,
    )
    assert not outcome["ok"]
    assert outcome["data"]["moved_ids"] == []
    failed = outcome["data"]["failed_ids"]
    assert sorted(failed) == [id_a, id_b, id_c]
    assert len(failed) == len(set(failed))  # no double counting
    reasons = {entry["memory_id"]: entry["reason"] for entry in outcome["data"]["errors"]}
    assert reasons[id_b] == "canonical_diverged"
    assert reasons[id_a].startswith("aborted:")
    assert reasons[id_c].startswith("aborted:")
    # rollback: nothing landed, bucket not registered
    assert row(tools, id_a)["workspace"] == "default"
    with tools.db.connection() as conn:
        registered = conn.execute(
            "SELECT 1 FROM workspace_canonicals WHERE name='proj-x'"
        ).fetchone()
    assert registered is None


def test_abort_path_drops_vec_publish_warning(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    id_a = write(tools, "default", content="fact a")
    id_b = write(tools, "default", content="fact b")
    store = tools.db.workspaces
    calls: list[int] = []

    def fake_move(conn, memory_id, workspace, *, precomputed_embedding=None):
        calls.append(int(memory_id))
        if int(memory_id) == id_a:
            return True, [
                "workspace canonical vector publish failed for 'proj-x'; "
                "retry a write using this workspace after sqlite-vec and embedding "
                "configuration recover: simulated"
            ]
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "move_memory_workspace_on_conn", fake_move)
    monkeypatch.setattr(
        store, "prepare_missing_workspace_canonical_embedding", lambda canonical, embedder: None,
    )

    outcome = move(tools, {"memory_ids": [id_a, id_b], "new_workspace": "proj-x", "authorized": True})
    assert not outcome["ok"]
    assert len(calls) == 2
    # the rollback undid the canonical registration, so the publish-failed
    # retry guidance must not leak into the aborted response
    assert not any(
        "workspace canonical vector publish failed" in warning
        for warning in outcome.get("warnings") or []
    )
    assert "workspace_vector_publish" not in outcome["data"]


def test_destination_rejection_accounts_request_ids(tmp_path: Path) -> None:
    tools = make_tools(tmp_path, isolation="strict", workspace="proj-a")
    memory_id = write(tools, "proj-a")
    confirm_pending(tools, memory_id)
    # A confirmed alias on the caller's own name redirects the caller's
    # resolution; the rejected destination must still account for the ids.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO workspace_aliases(alias_workspace, canonical, status, updated_at) "
            "VALUES (?, ?, 'confirmed', '2026-08-28T00:00:00Z')",
            ("proj-a", "elsewhere"),
        )

    outcome = move(tools, {
        "memory_ids": [memory_id], "new_workspace": "proj-a", "authorized": True,
        "workspace": "proj-a",
    })
    assert not outcome["ok"]
    assert outcome["data"]["error"].startswith("forbidden_strict_workspace")
    assert outcome["data"]["moved_ids"] == []
    assert outcome["data"]["failed_ids"] == [memory_id]
    assert outcome["data"]["errors"][0]["reason"] == "destination_rejected"
    untouched = row(tools, memory_id)
    assert untouched["workspace"] == "proj-a"


def test_alias_fold_prefers_newest_confirmed_row(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # Drift state (only reachable via manual SQL): two confirmed rows for one
    # alias key. The fold must pick the same winner the write-path resolver
    # picks (updated_at DESC, canonical ASC).
    from memory_arbiter.db import _normalize_alias_key
    alias_key = _normalize_alias_key("drift-alias")
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO workspace_aliases(alias_workspace, canonical, status, updated_at) "
            "VALUES (?, 'aaa-canonical', 'confirmed', '2026-01-01T00:00:00Z'), "
            "(?, 'zzz-canonical', 'confirmed', '2027-01-01T00:00:00Z')",
            (alias_key, alias_key),
        )
    resolved = tools.db.resolve_workspace_canonical("drift-alias", None, register_new=False)
    assert resolved["canonical"] == "zzz-canonical"
    assert tools.db.workspaces.confirmed_alias_canonical("drift-alias") == "zzz-canonical"

    memory_id = write(tools, "default")
    outcome = move(tools, {"memory_ids": [memory_id], "new_workspace": "drift-alias", "authorized": True})
    assert outcome["ok"], outcome
    assert outcome["data"]["new_workspace"] == "zzz-canonical"
    assert row(tools, memory_id)["workspace"] == "zzz-canonical"


def test_sqlite_integer_id_upper_bound(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = write(tools, "default")
    outcome = move(tools, {
        "memory_ids": [2**63], "new_workspace": "proj-x", "authorized": True,
    })
    assert not outcome["ok"]
    assert outcome["data"]["error"] == "invalid_input"
    assert row(tools, memory_id)["workspace"] == "default"
    direct = tools.memory("read", {"memory_id": 2**63})
    assert direct["ok"] is False
