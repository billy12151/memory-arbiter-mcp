# ── from test_workspace_normalize.py ──

"""Stock migration: workspace spelling-variant normalization (PR-C3).

``WorkspaceStore.normalize_workspace_canonicals`` folds legacy
double-registered canonicals that collapse to one ``_mechanical_ws_key``
(AgentLane / agent-lane / agent_lane) into the first-seen row, using the same
``_merge_workspace_core_on_conn`` suite as ``migrate_workspace``. Covered
contract points:
  * mechanical grouping with first-seen (min id) winner, independent of
    insertion order;
  * the full merge suite re-points memories/conflicts/alias targets, drops the
    loser canonical + vec row, and installs a redirect the resolver honors;
  * an explicit user rejection of the (loser, winner) pair is respected in
    ANY spelling — loser→winner, winner→loser, or a rejected row recorded
    under a THIRD spelling of the pair ('cross_spelling') — the whole group
    is skipped, never merged, and the skipped entry records the direction and
    the rejected row's verbatim spellings;
  * a third-party confirmed redirect the merge would silently drop behind a
    same-alias rejection is reported in skipped/warnings
    (confirmed_redirect_shadowed_by_rejection) while the merge still runs;
  * rejected-only normalization aligns a drifted rejected canonical spelling
    with the registered twin (rewritten, or dropped on PRIMARY KEY collision),
    and a dry run reports the IDENTICAL rewrite + dropped_duplicate sequence
    as the real run;
  * grouping deliberately folds less than casefold ('Straße' vs 'strasse'
    stay distinct projects) while ASCII spelling variants still merge;
  * dry_run (the default) plans without writing; a real run is idempotent;
  * the reserved default pool is never merged, even as spelling variants;
  * migrate_workspace/rename_workspace_canonical fold a mechanical-variant
    destination onto the already-registered spelling (no double registration,
    no memory/redirect split), and a destination that folds back onto the
    source is a self-merge no-op for migrate / a genuine spelling rename for
    rename; an unavailable advisory flock (<db>.startup.lock as a directory)
    is a structured warning, not an escaping OSError;
  * the memory_repair surface dispatches task=normalize_workspaces with
    dry_run defaulting to True, executing (dry_run=False) requires
    authorized=True (same gate shape as replay_backup), and strict isolation
    without a caller workspace is denied (same ACL gate as scan_candidates);
"""
import json
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup
from memory_arbiter.tools import MemoryTools

NOW = "2026-01-01T00:00:00+00:00"


def make_tools(tmp_path: Path, *, vec: bool = False) -> MemoryTools:
    # vec=True points at a (fake) GGUF model — the model path IS the intent
    # since 0.15.0 — and mirrors the first successful embedder build by
    # creating the lazy vec0 tables at dim 2.
    model = tmp_path / "fake.gguf"
    settings = Settings(
        db_path=tmp_path / "norm.sqlite3",
        backup_jsonl=tmp_path / "norm.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        embedding_model_path=model if vec else None, isolation="weak",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    if vec:
        model.write_bytes(b"fake")
        assert db.ensure_vec_tables(2) == []
    return tools


def register(tools: MemoryTools, *names: str) -> None:
    """Register canonicals directly, bypassing the resolver's mechanical-twin
    fold, to simulate legacy double-registration."""
    with tools.db.write_transaction() as conn:
        for name in names:
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES(?,?)",
                (name, NOW),
            )


def insert_alias(tools: MemoryTools, alias: str, canonical: str, status: str) -> None:
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_aliases("
            "alias_workspace,canonical,status,updated_at) VALUES(?,?,?,?)",
            (alias, canonical, status, NOW),
        )


def canonical_names(tools: MemoryTools) -> list[str]:
    with tools.db.connection() as conn:
        return [
            str(row["name"])
            for row in conn.execute("SELECT name FROM workspace_canonicals ORDER BY id")
        ]


def alias_rows(tools: MemoryTools) -> list[tuple[str, str, str]]:
    with tools.db.connection() as conn:
        return [
            (str(row["alias_workspace"]), str(row["canonical"]), str(row["status"]))
            for row in conn.execute(
                "SELECT alias_workspace,canonical,status FROM workspace_aliases "
                "ORDER BY alias_workspace,canonical"
            )
        ]


def write(tools: MemoryTools, workspace: str, content: str = "workspace fact") -> int:
    return int(tools.memory_write(
        content=content, subject="workspace", workspace=workspace,
        source_type="agent_generated",
    )["data"]["id"])


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


def memory_canonicals(tools: MemoryTools) -> dict[int, str]:
    with tools.db.connection() as conn:
        return {
            int(row["id"]): str(row["workspace_canonical"])
            for row in conn.execute("SELECT id,workspace_canonical FROM memories")
        }


def conflict_workspaces(tools: MemoryTools) -> list[str]:
    with tools.db.connection() as conn:
        return [
            str(row["workspace_canonical"])
            for row in conn.execute("SELECT workspace_canonical FROM conflicts")
        ]


def test_grouping_first_seen_winner_regardless_of_order(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # 'zebra-lane' registered before its twin; 'BetaProject' before its twin.
    # Min id (first-seen) wins in both, not lexical order.
    register(tools, "zebra-lane", "ZebraLane", "BetaProject", "beta-project")
    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=True)
    assert result["ok"] and result["dry_run"] is True
    by_key = {group["key"]: group for group in result["groups"]}
    assert by_key["zebralane"]["winner"] == "zebra-lane"
    assert by_key["zebralane"]["losers"] == ["ZebraLane"]
    assert by_key["betaproject"]["winner"] == "BetaProject"
    assert by_key["betaproject"]["losers"] == ["beta-project"]
    assert {merge["to"] for merge in result["merged"]} == {"zebra-lane", "BetaProject"}
    # dry_run wrote nothing.
    assert canonical_names(tools) == ["zebra-lane", "ZebraLane", "BetaProject", "beta-project"]


def test_merge_repoints_everything_and_installs_redirect(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    tools = make_tools(tmp_path, vec=True)
    register(tools, "AgentLane", "agent-lane")
    loser_memory = write(tools, "agent-lane", "loser fact")
    other_loser_memory = write(tools, "agent-lane", "loser fact two")
    winner_memory = write(tools, "AgentLane", "winner fact")
    # Conflict recorded under the loser before the merge (record_conflict
    # asserts outcome == "inserted"); the re-point check happens below.
    record_conflict(tools, "agent-lane", loser_memory, other_loser_memory)
    insert_alias(tools, "legacy-name", "agent-lane", "confirmed")

    loser_vec_id = winner_vec_id = None
    if tools.db.state.sqlite_vec_available:
        with tools.db.connection() as conn:
            loser_vec_id = int(conn.execute(
                "SELECT id FROM workspace_canonicals WHERE name='agent-lane'"
            ).fetchone()["id"])
            winner_vec_id = int(conn.execute(
                "SELECT id FROM workspace_canonicals WHERE name='AgentLane'"
            ).fetchone()["id"])
        with tools.db.write_transaction() as conn:
            for vec_id in (loser_vec_id, winner_vec_id):
                conn.execute(
                    "INSERT OR IGNORE INTO workspace_canonicals_vec(id,embedding) VALUES(?,?)",
                    (vec_id, json.dumps([1.0, 0.0])),
                )

    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert result["ok"] and result["dry_run"] is False
    assert result["warnings"] == []
    assert result["merged"] == [
        {"from": "agent-lane", "to": "AgentLane", "memories_updated": 2}
    ]
    # memories + conflicts re-pointed; winner's own memory untouched.
    assert memory_canonicals(tools)[loser_memory] == "AgentLane"
    assert memory_canonicals(tools)[other_loser_memory] == "AgentLane"
    assert memory_canonicals(tools)[winner_memory] == "AgentLane"
    assert conflict_workspaces(tools) == ["AgentLane"]
    # loser canonical row gone; alias target re-pointed; redirect installed.
    assert canonical_names(tools) == ["AgentLane"]
    assert ("legacy-name", "AgentLane", "confirmed") in alias_rows(tools)
    assert ("agent-lane", "AgentLane", "confirmed") in alias_rows(tools)
    resolved = tools.db.resolve_workspace_canonical("agent-lane", None, register_new=False)
    assert resolved["canonical"] == "AgentLane"
    assert resolved["matched_by"] == "confirmed_alias"
    if loser_vec_id is not None:
        with tools.db.connection() as conn:
            vec_ids = {
                int(row["id"])
                for row in conn.execute("SELECT id FROM workspace_canonicals_vec")
            }
        assert loser_vec_id not in vec_ids
        assert winner_vec_id in vec_ids


def test_rejected_pair_is_respected_not_merged(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane", "agent-lane")
    ok, warnings = tools.db.record_workspace_decision(
        "agent-lane", "AgentLane", status="rejected",
    )
    assert ok, warnings
    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert result["ok"]
    assert result["merged"] == []
    assert any(
        entry.get("from") == "agent-lane" and entry.get("to") == "AgentLane"
        and entry.get("direction") == "loser_to_winner"
        for entry in result["skipped"]
    )
    # both canonicals survive and the rejection row is untouched.
    assert canonical_names(tools) == ["AgentLane", "agent-lane"]
    assert ("agent-lane", "AgentLane", "rejected") in alias_rows(tools)


def test_reverse_rejected_pair_is_respected_not_merged(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # Rejection recorded BEFORE either spelling was registered: the ghost
    # spelling stays verbatim under the WINNER's alias key...
    ok, warnings = tools.db.record_workspace_decision(
        "AgentLane", "agent-lane", status="rejected",
    )
    assert ok, warnings
    # ...then legacy double-registration happens later.
    register(tools, "AgentLane", "agent-lane")
    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert result["ok"]
    assert result["merged"] == []
    assert any(
        entry.get("from") == "agent-lane" and entry.get("to") == "AgentLane"
        and entry.get("direction") == "winner_to_loser"
        for entry in result["skipped"]
    )
    # Both canonicals survive, and the reverse rejection row is kept (its
    # spelling aligned to the registered twin by rejected-only normalization).
    assert canonical_names(tools) == ["AgentLane", "agent-lane"]
    assert ("agentlane", "AgentLane", "rejected") in alias_rows(tools)


def test_rejected_only_spelling_normalization(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "ProjectX")
    # Drifted ghost spelling of the registered twin on two aliases; one alias
    # already carries the registered spelling (PRIMARY KEY collision case).
    insert_alias(tools, "some-proj", "project-x", "rejected")
    insert_alias(tools, "other-proj", "project-x", "rejected")
    insert_alias(tools, "other-proj", "ProjectX", "rejected")
    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert result["ok"]
    assert result["merged"] == []
    normalized = {
        (entry["alias_workspace"], entry["from"]): entry
        for entry in result["rejected_normalized"]
    }
    assert normalized[("some-proj", "project-x")]["action"] == "rewritten"
    assert normalized[("some-proj", "project-x")]["to"] == "ProjectX"
    assert normalized[("other-proj", "project-x")]["action"] == "dropped_duplicate"
    rows = alias_rows(tools)
    assert ("some-proj", "ProjectX", "rejected") in rows
    assert ("other-proj", "ProjectX", "rejected") in rows
    assert not any(row[1] == "project-x" for row in rows)


def test_dry_run_plans_without_writing(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane", "agent-lane")
    loser_memory = write(tools, "agent-lane")
    insert_alias(tools, "legacy-name", "agent-lane", "confirmed")
    before = (canonical_names(tools), alias_rows(tools), memory_canonicals(tools))
    result = tools.db.workspaces.normalize_workspace_canonicals()  # default dry_run=True
    assert result["ok"] and result["dry_run"] is True
    assert result["merged"] == [
        {"from": "agent-lane", "to": "AgentLane", "memories_updated": 1}
    ]
    after = (canonical_names(tools), alias_rows(tools), memory_canonicals(tools))
    assert before == after
    assert memory_canonicals(tools)[loser_memory] == "agent-lane"


def test_normalize_is_idempotent(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane", "agent-lane", "agent_lane")
    write(tools, "agent-lane")
    write(tools, "agent_lane")
    first = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert first["ok"]
    assert len(first["merged"]) == 2
    assert canonical_names(tools) == ["AgentLane"]
    second = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert second["ok"]
    assert second["groups"] == []
    assert second["merged"] == []
    assert second["rejected_normalized"] == []
    assert second["skipped"] == []


def test_default_pool_variants_never_merged(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "Default", "default")
    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert result["ok"]
    assert result["merged"] == []
    assert any(entry.get("reason") == "default_reserved" for entry in result["skipped"])
    default_groups = [g for g in result["groups"] if g["key"] == "default"]
    assert len(default_groups) == 1
    assert default_groups[0]["skipped"] is True
    assert canonical_names(tools) == ["Default", "default"]


def test_surface_dispatches_normalize_workspaces(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane", "agent-lane")
    write(tools, "agent-lane")
    planned = tools.memory_repair("normalize_workspaces", {})
    assert planned["ok"] and planned["dry_run"] is True
    assert planned["merged"][0]["to"] == "AgentLane"
    assert canonical_names(tools) == ["AgentLane", "agent-lane"]  # still untouched
    applied = tools.memory_repair(
        "normalize_workspaces", {"dry_run": False, "authorized": True},
    )
    assert applied["ok"] and applied["dry_run"] is False
    assert applied["merged"][0]["memories_updated"] == 1
    assert canonical_names(tools) == ["AgentLane"]


def test_surface_execute_requires_authorization(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane", "agent-lane")
    write(tools, "agent-lane")
    before = (canonical_names(tools), alias_rows(tools), memory_canonicals(tools))
    # dry_run=False without authorized is refused with the same gate shape as
    # replay_backup — and writes nothing.
    refused = tools.memory_repair("normalize_workspaces", {"dry_run": False})
    assert refused["ok"] is False
    assert refused["dry_run"] is False
    assert refused["error"]
    assert refused["action_required"] == "ask_user_for_authorization"
    assert refused["merged"] == []
    assert (canonical_names(tools), alias_rows(tools), memory_canonicals(tools)) == before
    applied = tools.memory_repair(
        "normalize_workspaces", {"dry_run": False, "authorized": True},
    )
    assert applied["ok"] and applied["dry_run"] is False
    assert canonical_names(tools) == ["AgentLane"]


def test_surface_help_lists_normalize_workspaces(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    help_payload = tools.memory_repair("help", {})
    assert "normalize_workspaces" in help_payload["data"]["tasks"]
    assert "normalize_workspaces" in help_payload["data"]["examples"]


# ---------------------------------------------------------------------------
#  migrate/rename destination orthography (registered mechanical twin)
# ---------------------------------------------------------------------------

def test_migrate_folds_destination_onto_registered_mechanical_twin(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane")
    winner_memory = write(tools, "AgentLane", "winner fact")
    first = write(tools, "old-ws", "old fact one")
    second = write(tools, "old-ws", "old fact two")
    updated, warnings = tools.db.workspaces.migrate_workspace("old-ws", "agent-lane")
    assert warnings == []
    assert updated == 2
    # Every memory lands on the registered spelling; the verbatim variant is
    # never registered, and the redirect points at the registered spelling.
    canonicals = memory_canonicals(tools)
    assert canonicals[first] == "AgentLane"
    assert canonicals[second] == "AgentLane"
    assert canonicals[winner_memory] == "AgentLane"
    assert canonical_names(tools) == ["AgentLane"]
    assert ("old-ws", "AgentLane", "confirmed") in alias_rows(tools)
    resolved = tools.db.resolve_workspace_canonical("old-ws", None, register_new=False)
    assert resolved["canonical"] == "AgentLane"


def test_migrate_onto_own_mechanical_twin_is_noop(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane")
    memory_id = write(tools, "AgentLane", "winner fact")
    updated, warnings = tools.db.workspaces.migrate_workspace("AgentLane", "agent-lane")
    assert (updated, warnings) == (0, [])
    # The winner row and its data are intact; no variant registered, no rows.
    assert canonical_names(tools) == ["AgentLane"]
    assert memory_canonicals(tools)[memory_id] == "AgentLane"
    assert alias_rows(tools) == []


def test_migrate_into_target_with_existing_vector_keeps_it(tmp_path: Path) -> None:
    """mema #794: vec0 ignores OR IGNORE — merging into a target that already
    owns a vector used to raise UNIQUE and warn misleadingly about sqlite-vec
    recovery. The probe keeps the existing vector and stays silent."""
    pytest.importorskip("sqlite_vec")
    tools = make_tools(tmp_path, vec=True)

    class Embedder:
        def embed_text(self, *, prefix: str = "", body: str = "", max_body_chars: int = 0):
            return type("ER", (), {"embedding": [0.25, 0.75], "last_encode_error": None})()

    register(tools, "target-ws")
    write(tools, "target-ws", "winner fact")
    tools.db.workspaces.publish_workspace_canonical_vector("target-ws", [0.25, 0.75])
    with tools.db.connection() as conn:
        target_row = conn.execute(
            "SELECT c.id, v.id AS vector_id FROM workspace_canonicals c "
            "LEFT JOIN workspace_canonicals_vec v ON v.id=c.id WHERE c.name='target-ws'"
        ).fetchone()
    assert target_row is not None and target_row["vector_id"] is not None

    write(tools, "source-ws", "old fact")
    updated, warnings = tools.db.workspaces.migrate_workspace(
        "source-ws", "target-ws", embedder=Embedder(),
    )
    assert updated == 1
    assert warnings == [], "existing target vector must be kept without a UNIQUE warning"
    with tools.db.connection() as conn:
        kept = conn.execute(
            "SELECT v.id FROM workspace_canonicals c "
            "JOIN workspace_canonicals_vec v ON v.id=c.id WHERE c.name='target-ws'"
        ).fetchone()
    assert kept is not None and int(kept["id"]) == int(target_row["id"])


def test_rename_folds_destination_onto_registered_mechanical_twin(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane")
    winner_memory = write(tools, "AgentLane", "winner fact")
    first = write(tools, "old-ws", "old fact one")
    second = write(tools, "old-ws", "old fact two")
    updated, warnings = tools.db.workspaces.rename_workspace_canonical("old-ws", "agent-lane")
    assert warnings == []
    assert updated == 2
    canonicals = memory_canonicals(tools)
    assert canonicals[first] == "AgentLane"
    assert canonicals[second] == "AgentLane"
    assert canonicals[winner_memory] == "AgentLane"
    assert canonical_names(tools) == ["AgentLane"]
    assert ("old-ws", "AgentLane", "confirmed") in alias_rows(tools)
    resolved = tools.db.resolve_workspace_canonical("old-ws", None, register_new=False)
    assert resolved["canonical"] == "AgentLane"


def test_rename_onto_own_mechanical_twin_stays_a_spelling_rename(tmp_path: Path) -> None:
    # A destination whose registered mechanical twin IS the source is a
    # genuine spelling change of the same row, not a self-merge (the baseline
    # case-only rename contract).
    tools = make_tools(tmp_path)
    memory_id = write(tools, "ProjectX")
    updated, warnings = tools.db.workspaces.rename_workspace_canonical("ProjectX", "projectx")
    assert warnings == []
    assert updated == 1
    assert canonical_names(tools) == ["projectx"]
    assert memory_canonicals(tools)[memory_id] == "projectx"


# ---------------------------------------------------------------------------
#  Respected rejections in any spelling / shadowed redirects / plan parity
# ---------------------------------------------------------------------------

def test_cross_spelling_rejected_row_skips_whole_group(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # The rejection lives under a THIRD spelling of the pair (the resolve
    # refusal flow's typical product): neither the loser's nor the winner's
    # alias key carries the row, so exact-key lookups would miss it and merge
    # on top of an explicit user rejection.
    ok, warnings = tools.db.record_workspace_decision(
        "agent_lane", "AgentLane", status="rejected",
    )
    assert ok, warnings
    register(tools, "AgentLane", "agent-lane")
    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert result["ok"]
    assert result["merged"] == []
    entry = next(
        skipped for skipped in result["skipped"]
        if skipped.get("direction") == "cross_spelling"
    )
    assert entry["from"] == "agent-lane"
    assert entry["to"] == "AgentLane"
    # The skipped entry carries the rejected row's verbatim spellings as evidence.
    assert entry["rejected_alias_workspace"] == "agent_lane"
    assert entry["rejected_canonical"] == "AgentLane"
    # Both canonicals survive and no confirmed redirect sits next to the rejection.
    assert canonical_names(tools) == ["AgentLane", "agent-lane"]
    assert ("agent_lane", "AgentLane", "rejected") in alias_rows(tools)
    assert not any(row[2] == "confirmed" for row in alias_rows(tools))


def test_confirmed_redirect_shadowed_by_rejection_is_visible(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane", "agent-lane")
    # A third-party alias carries a confirmed redirect INTO the group and a
    # rejected row for the winner spelling: the merge's INSERT OR IGNORE loses
    # the confirmed row to the PRIMARY KEY and the rejection wins. That used
    # to evaporate silently; it must now be reported, in dry-run and execute.
    insert_alias(tools, "legacy", "agent-lane", "confirmed")
    insert_alias(tools, "legacy", "AgentLane", "rejected")
    planned = tools.db.workspaces.normalize_workspace_canonicals(dry_run=True)
    assert any(
        entry.get("type") == "confirmed_redirect_shadowed_by_rejection"
        and entry.get("alias_workspace") == "legacy"
        for entry in planned["skipped"]
    )
    assert any("legacy" in warning for warning in planned["warnings"])
    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert result["ok"]
    assert result["merged"] == [
        {"from": "agent-lane", "to": "AgentLane", "memories_updated": 0}
    ]
    shadowed = [
        entry for entry in result["skipped"]
        if entry.get("type") == "confirmed_redirect_shadowed_by_rejection"
    ]
    assert len(shadowed) == 1
    assert shadowed[0]["alias_workspace"] == "legacy"
    assert shadowed[0]["confirmed_canonical"] == "agent-lane"
    assert shadowed[0]["rejected_canonical"] == "AgentLane"
    assert any("legacy" in warning for warning in result["warnings"])
    # Conservative end state: the rejection wins, the confirmed redirect is
    # gone, and the loser-side self redirect is installed by the merge.
    rows = alias_rows(tools)
    assert ("legacy", "AgentLane", "rejected") in rows
    assert ("agent-lane", "AgentLane", "confirmed") in rows
    assert not any(row[0] == "legacy" and row[2] == "confirmed" for row in rows)


def test_dry_run_matches_real_run_for_duplicate_rejected_rewrites(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "ProjectX")
    # Two drifted rejected spellings of the same registered twin under ONE
    # alias: the real run rewrites the first and drops the second as a
    # PRIMARY KEY duplicate. The dry run must report the identical sequence
    # (it used to claim two physically impossible rewrites).
    insert_alias(tools, "some-proj", "project-x", "rejected")
    insert_alias(tools, "some-proj", "project_x", "rejected")
    planned = tools.db.workspaces.normalize_workspace_canonicals(dry_run=True)
    applied = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert planned["rejected_normalized"] == applied["rejected_normalized"]
    assert [entry["action"] for entry in applied["rejected_normalized"]] == [
        "rewritten", "dropped_duplicate",
    ]
    assert alias_rows(tools) == [("some-proj", "ProjectX", "rejected")]


def test_casefold_only_pairs_are_never_merged(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # casefold() would fold 'Straße' -> 'strasse' and 'ﬁle' -> 'file'; the
    # normalize grouping key deliberately does not, so two legitimately
    # distinct projects are never destroyed as "spelling variants".
    register(tools, "Straße", "strasse", "ﬁle", "file")
    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert result["ok"]
    assert result["groups"] == []
    assert result["merged"] == []
    assert canonical_names(tools) == ["Straße", "strasse", "ﬁle", "file"]
    # ASCII spelling variants still merge normally.
    register(tools, "AgentLane", "agent-lane")
    result = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert result["merged"] == [
        {"from": "agent-lane", "to": "AgentLane", "memories_updated": 0}
    ]
    assert canonical_names(tools) == ["Straße", "strasse", "ﬁle", "file", "AgentLane"]


def test_surface_normalize_workspaces_strict_acl_gate(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane", "agent-lane")
    tools.settings.isolation = "strict"
    # normalize is a global operation (the payload carries no workspace
    # filter), but under strict isolation it still requires a resolvable
    # caller workspace — the same ACL gate as scan_candidates/record_conflict.
    tools.settings.workspace = ""
    denied = tools.memory_repair("normalize_workspaces", {})
    assert denied["ok"] is False
    assert denied["data"]["error"] == "forbidden_strict_workspace"
    assert denied["data"]["reason"] == "missing_caller_workspace"
    assert canonical_names(tools) == ["AgentLane", "agent-lane"]
    # With a caller workspace the global dry-run proceeds normally.
    tools.settings.workspace = "default"
    planned = tools.memory_repair("normalize_workspaces", {})
    assert planned["ok"] is True and planned["dry_run"] is True
    assert planned["merged"][0]["to"] == "AgentLane"


def test_rename_and_migrate_report_unavailable_startup_lock(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane")
    memory_id = write(tools, "old-ws", "old fact")
    # os.open() on a directory raises IsADirectoryError (an OSError): the
    # advisory flock is unavailable and must surface as a structured warning,
    # not an escaping exception (normalize already had this branch).
    lock_path = Path(str(tools.settings.db_path) + ".startup.lock")
    lock_path.unlink()  # MemoryDB startup created the regular lock file
    lock_path.mkdir()
    renamed, rename_warnings = tools.db.workspaces.rename_workspace_canonical(
        "old-ws", "agent-lane",
    )
    assert renamed == 0
    assert len(rename_warnings) == 1
    assert "workspace migration lock unavailable" in rename_warnings[0]
    migrated, migrate_warnings = tools.db.workspaces.migrate_workspace(
        "old-ws", "agent-lane",
    )
    assert migrated == 0
    assert len(migrate_warnings) == 1
    assert "workspace migration lock unavailable" in migrate_warnings[0]
    # Nothing was written by either attempt.
    assert canonical_names(tools) == ["AgentLane", "old-ws"]
    assert memory_canonicals(tools)[memory_id] == "old-ws"
    assert alias_rows(tools) == []


# ── from test_workspace_qwen_candidate.py ──
# helper make_tools renamed: qwen_candidate_make_tools (collision)

"""Qwen/local-model workspace candidate suggester (design 636 §6, §7, §9).

The real GGUF model is optional; these tests exercise the parser and the
per-isolation policy with a stub backend so they run without a model file.
"""
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools
from memory_arbiter.semantic_conflict import (
    WorkspaceCandidateSignal,
    workspace_candidate_from_text,
)


# ── parser ───────────────────────────────────────────────────────────────────

def test_parse_valid_candidate():
    raw = '{"candidate": "金营项目", "relation": "alias", "confidence": 0.92, "evidence": "同一项目不同写法"}'
    sig = workspace_candidate_from_text(raw, ["金营项目", "其他项目"])
    assert sig.candidate == "金营项目"
    assert sig.relation == "alias"
    assert sig.confidence == 0.92


def test_parse_drops_hallucinated_candidate():
    # model returns a candidate not in the offered list → dropped
    raw = '{"candidate": "不存在项目", "relation": "alias", "confidence": 0.9}'
    sig = workspace_candidate_from_text(raw, ["金营项目"])
    assert sig.candidate is None
    assert sig.relation == "uncertain"  # downgraded since no candidate


def test_parse_missing_json_is_uncertain():
    sig = workspace_candidate_from_text("no json here", ["a"])
    assert sig.candidate is None and sig.relation == "uncertain"
    assert sig.error == "missing_json"


def test_parse_unknown_relation_normalized():
    raw = '{"candidate": "a", "relation": "bogus", "confidence": 0.5}'
    sig = workspace_candidate_from_text(raw, ["a"])
    assert sig.relation == "uncertain"


def test_parse_clamps_out_of_range_confidence():
    hi = workspace_candidate_from_text('{"candidate":"a","relation":"alias","confidence":5.0}', ["a"])
    assert hi.confidence == 1.0
    lo = workspace_candidate_from_text('{"candidate":"a","relation":"alias","confidence":-3}', ["a"])
    assert lo.confidence == 0.0


# ── per-isolation policy (stub backend) ──────────────────────────────────────

class _StubBackend:
    """Minimal stand-in for LocalGGUFSemanticBackend."""
    def __init__(self, signal: WorkspaceCandidateSignal):
        self._signal = signal

    def suggest_workspace_candidate(self, ws_raw, evidence, candidates):
        return self._signal


def qwen_candidate_make_tools(tmp_path: Path, isolation: str) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "qwen.sqlite3",
        backup_jsonl=tmp_path / "qwen.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _force_undecided_with_candidate(t: MemoryTools, backend, similar_name="金营项目", distance=0.2):
    """Patch resolver → undecided + a candidate, and inject a stub backend.

    The candidate distance defaults to 0.2 (inside workspace_match_distance): a
    real near-miss the vector brought within range, which is the only situation
    where Qwen is allowed to arbitrate an AUTO merge. Over-distance candidates
    (e.g. 0.4) are filtered out before Qwen sees them by design."""
    def fake_resolve(ws_raw, embedder=None, *, match_distance=None, register_new=True):
        return {
            "canonical": ws_raw, "is_new": True, "matched_by": "new",
            "distance": None, "similar": [{"name": similar_name, "distance": distance}],
            "rejected_canonicals": [],
        }
    t.db.resolve_workspace_canonical = fake_resolve  # type: ignore
    t._ensure_semantic_backend = lambda: backend  # type: ignore


def test_weak_high_confidence_silent_merge(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "weak")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.95, "同项目"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    assert d["workspace_decision"] == "AUTO"
    assert d["workspace_canonical"] == "金营项目"
    assert d["workspace_matched_by"] == "qwen"


def test_none_high_confidence_normalizes_without_acl_or_confirmed_alias(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "none")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.95, "同项目"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    assert r["data"]["workspace_canonical"] == "金营项目"
    assert t.db.get_workspace_decision("金营") is None
    # none stays ACL-free: an unscoped search spans all workspaces, while an
    # explicit filter canonicalizes then scopes that one query (spec §15.6).
    # (The stub backend normalizes both writes to the same canonical.)
    t.memory_write(content="y", workspace="其他", source_type="agent_generated", subject="other")
    assert len(t.memory_search(query="")["data"]["results"]) == 2
    scoped = t.memory_search(query="", workspace="金营项目")["data"]["results"]
    assert {item.get("workspace_canonical") or item["workspace"] for item in scoped} == {"金营项目"}
    assert t.memory_search(query="", workspace="别的项目")["data"]["results"] == []


def test_weak_low_confidence_asks_not_merge(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "weak")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "related", 0.5, "可能相关"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    assert d["workspace_decision"] == "ASK"
    # NOT silently merged
    assert d["workspace_canonical"] != "金营项目"
    review = d.get("write_hints", {}).get("workspace_review")
    assert review
    merge = next(option for option in review["options"] if option["decision"] == "merge")
    assert merge["authorization_required"] is True
    assert "authorized" not in merge["call"]["data"]


def test_near_miss_registers_only_final_canonical(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "weak")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.95, "同项目"))
    _force_undecided_with_candidate(t, backend)
    t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    with t.db.connection() as conn:
        names = {row["name"] for row in conn.execute("SELECT name FROM workspace_canonicals")}
    assert "金营项目" in names
    assert "金营" not in names


def test_strict_never_silent_merges_even_high_conf(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "strict")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.99, "同项目"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    # strict: high-conf candidate does NOT auto-merge; memory stays pending
    assert d["workspace_canonical"] != "金营项目"
    assert d.get("action_required") == "confirm_new_workspace"


def test_no_backend_falls_back_to_ask(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "weak")
    _force_undecided_with_candidate(t, None)  # no backend
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    assert r["data"]["workspace_decision"] == "ASK"


# ── strict-switch governance advisory (636 §9) ───────────────────────────────

def test_strict_emits_governance_advisory(tmp_path, monkeypatch):
    # isolation is file-only since 0.15.0; the strict advisory still fires.
    import json

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"isolation": "strict"}), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg))
    s = Settings.from_env()
    assert any("confirm_pending_workspace" in w and "migrate" in w for w in s.config_warnings)


# ── decision-reason distinguishes model-absent from low-confidence (spec §8) ──

def test_no_backend_reason_is_qwen_unavailable_not_low_conf(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "weak")
    _force_undecided_with_candidate(t, None)  # backend absent, candidates exist
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    assert d["workspace_decision"] == "ASK"
    assert d["workspace_decision_reason"] == "qwen_unavailable"


def test_genuine_low_confidence_reason_is_qwen_low_conf(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "weak")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.5, "可能相关"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    assert d["workspace_decision"] == "ASK"
    assert d["workspace_decision_reason"] == "qwen_low_conf"


def test_rejected_candidate_reason_is_qwen_rejected(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "none")
    # Qwen suggests a candidate the user already rejected. The rule stays
    # undecided (a second, non-rejected near-miss keeps it a near_miss), so the
    # write path reaches the Qwen branch and must refuse the rejected merge.
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.95, "同项目"))

    def fake_resolve(ws_raw, embedder=None, *, match_distance=None, register_new=True):
        return {
            "canonical": ws_raw, "is_new": True, "matched_by": "new", "distance": None,
            "similar": [
                {"name": "金营项目", "distance": 0.18},
                {"name": "别的项目", "distance": 0.22},
            ],
            "rejected_canonicals": ["金营项目"],
        }
    t.db.resolve_workspace_canonical = fake_resolve  # type: ignore
    t._ensure_semantic_backend = lambda: backend  # type: ignore
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    # A user-rejected candidate immediately overrides the high-confidence auto.
    assert d["workspace_decision"] == "ASK"
    assert d["workspace_canonical"] == "金营"
    assert d["workspace_decision_reason"] == "qwen_rejected_candidate"


# ── constrained decoding at the inference call site (2026-08-21 dry-run) ─────
#
# A real-library dry-run showed every workspace suggestion failing with
# missing_json: Qwen2.5-0.5B answered in prose ("candidate: AgentLane\nrelation:
# same_family") because suggest_workspace_candidate never passed a
# response_format, unlike classify_pair. The parser tests above could not catch
# it — only the call site can — so assert the schema is wired in and bounded.

def test_workspace_suggester_uses_constrained_decoding() -> None:
    from memory_arbiter.semantic_conflict import (
        LocalGGUFSemanticBackend, _WORKSPACE_PROMPT, _WORKSPACE_RESPONSE_FORMAT,
    )

    schema = _WORKSPACE_RESPONSE_FORMAT["schema"]
    assert _WORKSPACE_RESPONSE_FORMAT["type"] == "json_object"
    assert set(schema["required"]) == {"candidate", "relation", "confidence", "evidence"}
    assert schema["additionalProperties"] is False
    # relation is constrained to the spec's enum, so the model cannot invent one.
    assert set(schema["properties"]["relation"]["enum"]) == {
        "alias", "typo", "same_project", "same_family", "related", "unrelated", "uncertain",
    }
    # evidence is bounded: an unbounded field let the model paste whole memory
    # bodies in and blow past max_tokens, truncating the JSON.
    assert schema["properties"]["evidence"]["maxLength"] == 200
    assert "只输出 JSON" in _WORKSPACE_PROMPT

    captured: dict = {}

    class _Llm:
        @staticmethod
        def create_chat_completion(**kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": '{"candidate":"金营项目","relation":"alias","confidence":0.9,"evidence":"同项目"}'}}]}

    backend = LocalGGUFSemanticBackend.__new__(LocalGGUFSemanticBackend)
    import threading
    backend._infer_lock = threading.Lock()
    backend._cond = threading.Condition(threading.Lock())
    backend._acquire_llm_for_call = lambda: _Llm()          # type: ignore[method-assign]
    backend._release_llm_for_call = lambda: None            # type: ignore[method-assign]

    signal = backend.suggest_workspace_candidate(
        "金营", {"title": "金营项目排期", "key_sentences": ["交付计划"]}, ["金营项目"],
    )
    assert captured.get("response_format") is _WORKSPACE_RESPONSE_FORMAT
    assert signal.candidate == "金营项目"
    assert signal.relation == "alias"
    assert signal.error is None


# ── Qwen only arbitrates in-threshold candidates (2026-08-21 real-lib A/B) ───
#
# A dry-run over the real 547-memory library had Qwen answer
# "same_project@0.95" merging openclaw into proto-test at cosine 0.357 — far
# past the 0.25 threshold — because the AUTO gate looked at Qwen confidence but
# not vector distance. Fix: _suggest_workspace_candidate drops candidates beyond
# workspace_qwen_candidate_distance before Qwen sees them, so an over-distance
# name can never be resurrected into an AUTO merge.

def test_over_distance_candidate_is_filtered_before_qwen(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "none")
    calls = []

    class _SpyBackend:
        def suggest_workspace_candidate(self, ws_raw, evidence, candidates, **kw):
            calls.append(list(candidates))
            return WorkspaceCandidateSignal(candidates[0] if candidates else None,
                                            "same_project", 0.95, "hallucinated")

    # Only over-distance neighbors (0.357 > 0.25): Qwen must not even be asked.
    _force_undecided_with_candidate(t, _SpyBackend(), similar_name="proto-test", distance=0.357)
    r = t.memory_write(content="x", workspace="openclaw", source_type="agent_generated", subject="s")
    assert calls == []  # no candidate survived the distance bound
    assert r["data"]["workspace_canonical"] == "openclaw"  # stays NEW, not merged
    assert r["data"]["workspace_decision"] == "ASK"


def test_in_threshold_candidate_still_reaches_qwen_and_auto_merges(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "none")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "same_project", 0.95, "同项目"))
    _force_undecided_with_candidate(t, backend, distance=0.2)  # inside threshold
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="s")
    assert r["data"]["workspace_canonical"] == "金营项目"
    assert r["data"]["workspace_decision"] == "AUTO"


def test_qwen_candidate_pool_capped_at_top_k(tmp_path):
    t = qwen_candidate_make_tools(tmp_path, "none")
    seen = []

    class _SpyBackend:
        def suggest_workspace_candidate(self, ws_raw, evidence, candidates, **kw):
            seen.append(list(candidates))
            return WorkspaceCandidateSignal(None, "uncertain", None, "")

    def fake_resolve(ws_raw, embedder=None, *, match_distance=None, register_new=True):
        # Five in-threshold neighbors; only top-3 should reach Qwen.
        return {"canonical": ws_raw, "is_new": True, "matched_by": "new", "distance": None,
                "similar": [{"name": f"c{i}", "distance": 0.10 + i * 0.02} for i in range(5)],
                "rejected_canonicals": []}
    t.db.resolve_workspace_canonical = fake_resolve  # type: ignore
    t._ensure_semantic_backend = lambda: _SpyBackend()  # type: ignore
    t.memory_write(content="x", workspace="w", source_type="agent_generated", subject="s")
    assert seen and len(seen[0]) == 3
    assert seen[0] == ["c0", "c1", "c2"]


# ── from test_workspace_alias_governance.py ──
# make_tools renamed: alias_governance_make_tools; write/record_conflict identical to test_workspace_normalize.py versions -> single shared copy kept above

"""Internal workspace redirect and negative-decision state.

Pairwise alias actions are intentionally absent from the product surface. Tests
cover the behaviors that remain load-bearing: deterministic resolver redirects,
negative candidate suppression, rename/migrate forwarding, strict pending
confirmation, collision safety, compact persistence, and rollback.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import threading

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB, _normalize_alias_key
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryStatus
from memory_arbiter.tools import MemoryTools


def alias_governance_make_tools(
    tmp_path: Path, isolation: str = "weak", *, vec: bool = False,
) -> MemoryTools:
    # vec=True points at a (fake) GGUF model — the model path IS the intent
    # since 0.15.0 — and mirrors the first successful embedder build by
    # creating the lazy vec0 tables at dim 2.
    model = tmp_path / "fake.gguf"
    settings = Settings(
        db_path=tmp_path / "gov.sqlite3",
        backup_jsonl=tmp_path / "gov.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        embedding_model_path=model if vec else None, isolation=isolation,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    if vec:
        model.write_bytes(b"fake")
        assert db.ensure_vec_tables(2) == []
    return tools


def decide(
    tools: MemoryTools, workspace: str, canonical: str,
    *, status: str = "confirmed", force: bool = False,
) -> None:
    ok, warnings = tools.db.record_workspace_decision(
        workspace, canonical, status=status, force=force,
    )
    assert ok, warnings




def test_normalize_workspace_decision_key() -> None:
    assert _normalize_alias_key("  金营项目 ") == _normalize_alias_key("金营项目")
    assert _normalize_alias_key("Project  X") == _normalize_alias_key("project x")
    assert _normalize_alias_key("") == ""
    assert _normalize_alias_key(None) == ""


def test_compact_schema_has_no_event_ledger(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    with tools.db.connection() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
        events = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workspace_alias_events'"
        ).fetchone()
    assert columns == ["alias_workspace", "canonical", "status", "updated_at"]
    assert events is None


def test_confirmed_redirect_short_circuits_resolver(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "金营二期", "金营项目")
    resolved = tools.db.resolve_workspace_canonical(" 金营二期 ", None, register_new=False)
    assert resolved["canonical"] == "金营项目"
    assert resolved["matched_by"] == "confirmed_alias"
    state = tools.db.get_workspace_decision("金营二期")
    assert state["status"] == "confirmed"


def test_negative_decisions_accumulate_and_suppress_candidates(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "raw", "candidate-a", status="rejected")
    decide(tools, "raw", "candidate-b", status="rejected")
    resolved = tools.db.resolve_workspace_canonical("raw", None, register_new=False)
    assert set(resolved["rejected_canonicals"]) == {"candidate-a", "candidate-b"}
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT canonical,status FROM workspace_aliases "
            "WHERE alias_workspace='raw' ORDER BY canonical"
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("candidate-a", "rejected"), ("candidate-b", "rejected"),
    ]


def test_exact_negative_requires_force_to_reverse(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "raw", "target", status="rejected")
    ok, warnings = tools.db.record_workspace_decision("raw", "target")
    assert ok is False and "kept separate" in warnings[0]
    decide(tools, "raw", "target", force=True)
    assert tools.db.get_workspace_decision("raw")["status"] == "confirmed"


def test_one_confirmed_redirect_preserves_unrelated_negatives(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "raw", "candidate-a", status="rejected")
    decide(tools, "raw", "canonical")
    decide(tools, "raw", "new-canonical")
    with tools.db.connection() as conn:
        confirmed = conn.execute(
            "SELECT canonical FROM workspace_aliases "
            "WHERE alias_workspace='raw' AND status='confirmed'"
        ).fetchall()
        rejected = conn.execute(
            "SELECT canonical FROM workspace_aliases "
            "WHERE alias_workspace='raw' AND status='rejected'"
        ).fetchall()
    assert [row["canonical"] for row in confirmed] == ["new-canonical"]
    assert [row["canonical"] for row in rejected] == ["candidate-a"]


def test_removed_pairwise_actions_are_non_mutating_tombstones(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    for action in ("accept_workspace_alias", "reject_workspace_alias"):
        result = tools.memory_govern(action, {
            "alias": "raw", "canonical": "target", "authorized": True,
        })
        assert result["ok"] is False
        assert result["data"]["outcome"] == "removed"
        assert result["data"]["error_code"] == "workspace_alias_action_removed"
        assert result["data"]["removed_action"] == action
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspace_aliases").fetchone()[0] == 0


def test_removed_accept_guides_to_supported_flows(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    result = tools.memory_govern("accept_workspace_alias", {
        "alias": "raw", "canonical": "target",
    })
    actions = {
        item.get("suggested_call", {}).get("action")
        for item in result["data"]["replacements"]
        if item.get("suggested_call")
    }
    assert {"migrate_workspace", "rename_workspace_canonical", "confirm_pending_workspace"} <= actions
    migrate = next(
        item["suggested_call"] for item in result["data"]["replacements"]
        if (item.get("suggested_call") or {}).get("action") == "migrate_workspace"
    )
    assert migrate == {
        "tool": "memory_govern", "action": "migrate_workspace",
        "data": {"from": "raw", "to": "target"},
    }
    reject = tools.memory_govern("reject_workspace_alias", {})
    assert reject["data"]["replacements"][0]["suggested_call"] is None


def test_rename_moves_memories_and_prevents_old_name_resplit(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    memory_id = write(tools, "OldName")
    result = tools.memory_govern("rename_workspace_canonical", {
        "old": "OldName", "new": "NewName", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "NewName"
    resolved = tools.db.resolve_workspace_canonical("OldName", None, register_new=False)
    assert resolved["canonical"] == "NewName"
    assert resolved["matched_by"] == "confirmed_alias"


def test_rename_moves_conflicts_with_their_members(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    left = write(tools, "OldName", "database is mysql")
    right = write(tools, "OldName", "database is sqlite")
    conflict_id = record_conflict(tools, "OldName", left, right)

    result = tools.memory_govern("rename_workspace_canonical", {
        "old": "OldName", "new": "NewName", "authorized": True,
    })

    assert result["ok"] is True
    conflict = tools.db.get_conflict(conflict_id)
    assert conflict["workspace_canonical"] == "NewName"
    assert tools.memory_review(
        "conflict_detail", {"conflict_id": conflict_id, "workspace": "NewName"},
    )["ok"] is True


def test_migrate_moves_memories_and_prevents_source_resplit(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    memory_id = write(tools, "Sub2")
    result = tools.memory_govern("migrate_workspace", {
        "from": "Sub2", "to": "Main", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "Main"
    resolved = tools.db.resolve_workspace_canonical("Sub2", None, register_new=False)
    assert resolved["canonical"] == "Main"
    assert resolved["matched_by"] == "confirmed_alias"


def test_migrate_moves_conflicts_with_their_members(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    left = write(tools, "Sub2", "database is mysql")
    right = write(tools, "Sub2", "database is sqlite")
    conflict_id = record_conflict(tools, "Sub2", left, right)

    result = tools.memory_govern("migrate_workspace", {
        "from": "Sub2", "to": "Main", "authorized": True,
    })

    assert result["ok"] is True
    assert tools.db.get_conflict(conflict_id)["workspace_canonical"] == "Main"


def test_migrate_conflict_slot_collision_rolls_back_everything(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    old_left = write(tools, "Old", "old mysql")
    old_right = write(tools, "Old", "old sqlite")
    new_left = write(tools, "New", "new mysql")
    new_right = write(tools, "New", "new sqlite")
    old_conflict = record_conflict(tools, "Old", old_left, old_right)
    new_conflict = record_conflict(tools, "New", new_left, new_right)

    result = tools.memory_govern("migrate_workspace", {
        "from": "Old", "to": "New", "authorized": True,
    })

    assert result["ok"] is False
    assert any("migrate_workspace failed" in warning for warning in result["warnings"])
    assert tools.db.get_memory(old_left)["workspace_canonical"] == "Old"
    assert tools.db.get_memory(old_right)["workspace_canonical"] == "Old"
    assert tools.db.get_conflict(old_conflict)["workspace_canonical"] == "Old"
    assert tools.db.get_conflict(new_conflict)["workspace_canonical"] == "New"


def test_migrate_without_existing_rows_installs_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    result = tools.memory_govern("migrate_workspace", {
        "from": "mema", "to": "memory-arbiter-mcp", "authorized": True,
    })
    assert result["ok"] is True
    assert result["data"]["memories_updated"] == 0
    resolved = tools.db.resolve_workspace_canonical("mema", None, register_new=False)
    assert resolved["canonical"] == "memory-arbiter-mcp"


def test_exact_negative_blocks_rename_forwarding(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "Old", "New", status="rejected")
    result = tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "New", "authorized": True,
    })
    assert result["ok"] is True
    resolved = tools.db.resolve_workspace_canonical("Old", None, register_new=False)
    assert resolved["matched_by"] != "confirmed_alias"
    assert "New" in resolved["rejected_canonicals"]


def test_unrelated_negative_does_not_block_rename_forwarding(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "Old", "Other", status="rejected")
    tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "New", "authorized": True,
    })
    resolved = tools.db.resolve_workspace_canonical("Old", None, register_new=False)
    assert resolved["canonical"] == "New"
    with tools.db.connection() as conn:
        rejected = conn.execute(
            "SELECT canonical FROM workspace_aliases "
            "WHERE alias_workspace='old' AND status='rejected'"
        ).fetchall()
    assert [row["canonical"] for row in rejected] == ["Other"]


def test_repoint_is_collision_safe_and_preserves_existing_decision(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "foo", "Old", status="rejected")
    decide(tools, "foo", "New", status="rejected")
    result = tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "New", "authorized": True,
    })
    assert result["ok"] is True
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT canonical,status FROM workspace_aliases "
            "WHERE alias_workspace='foo'"
        ).fetchall()
    assert [tuple(row) for row in rows] == [("New", "rejected")]


def test_case_only_rename_leaves_no_self_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    memory_id = write(tools, "ProjectX")
    result = tools.memory_govern("rename_workspace_canonical", {
        "old": "ProjectX", "new": "projectx", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "projectx"
    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases "
            "WHERE alias_workspace='projectx' AND canonical='projectx'"
        ).fetchone()[0] == 0


def test_strict_retries_remain_pending_until_confirmation(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    first = write(tools, "Unconfirmed")
    second = write(tools, "Unconfirmed", "second fact")
    assert tools.db.get_memory(first)["status"] == MemoryStatus.PENDING.value
    assert tools.db.get_memory(second)["status"] == MemoryStatus.PENDING.value
    with tools.db.connection() as conn:
        canonical = conn.execute(
            "SELECT 1 FROM workspace_canonicals WHERE name='Unconfirmed'"
        ).fetchone()
    assert canonical is None


def test_confirm_pending_case_variant_reuses_raw_spelling(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "BrandNew")
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "brandnew", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "BrandNew"
    with tools.db.connection() as conn:
        names = [row["name"] for row in conn.execute(
            "SELECT name FROM workspace_canonicals WHERE lower(name)='brandnew'"
        )]
    assert names == ["BrandNew"]


def test_default_pending_cannot_be_confirmed_into_project(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    written = tools.memory_write(
        content="global pending", subject="global", workspace="default",
        source_type="agent_generated", status="pending",
    )
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": written["data"]["id"], "canonical": "ProjectX", "authorized": True,
    })
    assert result["ok"] is False
    assert "reserved default" in result["data"]["error"]
    assert tools.db.get_memory(written["data"]["id"])["status"] == MemoryStatus.PENDING.value


def test_competing_move_does_not_split_memory_and_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    memory_id = write(tools, "Old")
    first = tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "A", "authorized": True,
    })
    second = tools.memory_govern("rename_workspace_canonical", {
        "old": "Old", "new": "B", "authorized": True,
    })
    assert first["ok"] is True
    assert second["ok"] is False
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "A"
    assert tools.db.resolve_workspace_canonical("Old", None, register_new=False)["canonical"] == "A"


def test_confirm_pending_exact_name_activates_without_self_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "BrandNew")
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.PENDING.value
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "BrandNew", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.ACTIVE.value
    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases WHERE alias_workspace='brandnew'"
        ).fetchone()[0] == 0


def test_confirm_pending_different_name_records_redirect_atomically(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "abbrev")
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "CanonicalProject", "authorized": True,
    })
    assert result["ok"] is True
    record = tools.db.get_memory(memory_id)
    assert record["status"] == MemoryStatus.ACTIVE.value
    assert record["workspace_canonical"] == "CanonicalProject"
    assert tools.db.resolve_workspace_canonical("abbrev", None, register_new=False)["canonical"] == "CanonicalProject"


def test_confirm_pending_rolls_back_decision_assignment_and_activation(tmp_path: Path, monkeypatch) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "abbrev")
    original = tools.db.set_memory_workspace_canonical_on_conn

    def fail(*args, **kwargs):
        return False, ["injected failure"]

    monkeypatch.setattr(tools.db, "set_memory_workspace_canonical_on_conn", fail)
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "CanonicalProject", "authorized": True,
    })
    assert result["ok"] is False
    monkeypatch.setattr(tools.db, "set_memory_workspace_canonical_on_conn", original)
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.PENDING.value
    assert tools.db.get_workspace_decision("abbrev") is None


def test_confirm_pending_error_does_not_leak_foreign_record(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "Other", "TOP SECRET")
    result = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id, "canonical": "Caller",
        "workspace": "Caller", "authorized": True,
    })
    assert result["ok"] is False
    assert result["data"]["record"] is None
    assert "TOP SECRET" not in str(result)


def test_default_pool_cannot_enter_internal_decision_state(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    for left, right in (("default", "project"), ("project", "默认")):
        ok, warnings = tools.db.record_workspace_decision(left, right)
        assert ok is False
        assert "reserved global pool" in warnings[0]


def test_concurrent_confirmed_decisions_leave_one_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def worker(target: str) -> None:
        barrier.wait()
        outcomes.append(tools.db.record_workspace_decision("raw", target)[0])

    threads = [threading.Thread(target=worker, args=(target,)) for target in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(outcomes)
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT canonical FROM workspace_aliases "
            "WHERE alias_workspace='raw' AND status='confirmed'"
        ).fetchall()
    assert len(rows) == 1


def test_negative_decision_filters_real_vector_candidate(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    tools = alias_governance_make_tools(tmp_path, vec=True)
    if not tools.db.state.sqlite_vec_available:
        pytest.skip("sqlite-vec unavailable")

    class Embedder:
        def embed_text(self, prefix="", body=""):
            return SimpleNamespace(embedding=[1.0, 0.0])

    embedder = Embedder()
    tools.db.resolve_workspace_canonical("Target", embedder, register_new=True)
    decide(tools, "raw", "Target", status="rejected")
    resolved = tools.db.resolve_workspace_canonical("raw", embedder, register_new=False)
    assert resolved["canonical"] != "Target"
    assert "Target" not in [item["name"] for item in resolved["similar"]]


def test_workspace_decision_schema_normalization_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "raw", "candidate-a", status="rejected")
    del tools
    reopened = MemoryDB(Settings(
        db_path=path if path.exists() else tmp_path / "gov.sqlite3",
        backup_jsonl=tmp_path / "other.jsonl",
    ))
    with reopened.connection() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
    assert columns == ["alias_workspace", "canonical", "status", "updated_at"]


# ── from test_workspace_rules.py ──
# helper make_tools renamed: rules_make_tools (collision)

"""Rule-first workspace decision layer (design 636 §2, §3, §5)."""
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools
from memory_arbiter import workspace_rules as wr


# ── quality classification ───────────────────────────────────────────────────

def test_quality_empty_default():
    assert wr.classify_workspace_quality("") == "empty"
    assert wr.classify_workspace_quality("   ") == "empty"
    assert wr.classify_workspace_quality("default") == "default"
    assert wr.classify_workspace_quality("默认") == "default"


def test_quality_generic():
    assert wr.classify_workspace_quality("实施计划") == "generic"
    assert wr.classify_workspace_quality("月报") == "generic"
    assert wr.classify_workspace_quality("notes") == "generic"


def test_quality_specific():
    assert wr.classify_workspace_quality("金营项目") == "specific"
    assert wr.classify_workspace_quality("project-x") == "specific"


def test_quality_suspicious():
    assert wr.classify_workspace_quality("/etc/passwd") == "suspicious"
    assert wr.classify_workspace_quality("http://x.com") == "suspicious"
    assert wr.classify_workspace_quality("a" * 90) == "suspicious"


# ── evidence extraction ──────────────────────────────────────────────────────

def test_extract_evidence_from_dict():
    ev = wr.extract_evidence({
        "subject": "金营项目周报",
        "content": "# 概述\n\n本周完成了 X。还做了 Y。\n\n## 细节\n更多内容。",
    })
    assert ev["subject"] == "金营项目周报"
    assert ev["title"] == "金营项目周报"
    assert "概述" in ev["headings"]
    assert ev["key_sentences"]


# ── rule decision ────────────────────────────────────────────────────────────

def test_decision_auto_on_confirmed_alias():
    resolved = {"matched_by": "confirmed_alias", "canonical": "金营项目", "similar": []}
    d = wr.rule_decision("金营二期", resolved)
    assert d["decision"] == "AUTO" and d["canonical"] == "金营项目"


def test_decision_keep_reference_material():
    resolved = {"matched_by": "vector", "canonical": "金营项目",
                "similar": [{"name": "金营项目", "distance": 0.1}]}
    ev = {"title": "参考金营项目的月报模板", "first_para": "借鉴其结构"}
    d = wr.rule_decision("模板库", resolved, ev)
    assert d["decision"] == "KEEP" and d["reason"] == "reference_material"


def test_decision_keep_rejected_pair():
    resolved = {"matched_by": "vector", "canonical": "金营项目",
                "rejected_canonicals": ["金营项目"],
                "similar": [{"name": "金营项目", "distance": 0.1}]}
    d = wr.rule_decision("金营培训", resolved)
    assert d["decision"] == "KEEP" and d["reason"] == "rejected_pair"


def test_decision_ask_on_generic():
    resolved = {"matched_by": "new", "canonical": "月报", "similar": []}
    d = wr.rule_decision("月报", resolved)
    assert d["decision"] == "ASK"


def test_decision_ask_on_near_tie():
    resolved = {"matched_by": "vector", "canonical": "A",
                "similar": [{"name": "A", "distance": 0.20}, {"name": "B", "distance": 0.22}]}
    d = wr.rule_decision("金营", resolved)
    assert d["decision"] == "ASK" and d["reason"] == "candidate_near_tie"


def test_decision_auto_new_specific():
    resolved = {"matched_by": "new", "canonical": "赛博项目", "similar": []}
    d = wr.rule_decision("赛博项目", resolved)
    assert d["decision"] == "AUTO" and d["reason"] == "new_specific_canonical"


# ── review regression: rejected candidate at similar[1] must NOT block a valid
#    merge into the resolver's chosen non-rejected canonical (workspace_rules:143)

def test_rejected_at_similar0_does_not_block_valid_chosen_canonical():
    # resolver skipped rejected ProjectC (similar[0]) and chose ProjectD.
    resolved = {
        "matched_by": "vector", "canonical": "ProjectD",
        "rejected_canonicals": ["ProjectC"],
        "similar": [{"name": "ProjectC", "distance": 0.10},
                    {"name": "ProjectD", "distance": 0.20}],
    }
    d = wr.rule_decision("aliasX", resolved)
    # must merge into the valid ProjectD, NOT keep-separate on the rejected pair
    assert d["decision"] == "AUTO"
    assert d["canonical"] == "ProjectD"


def test_rejected_at_similar1_does_not_trigger_spurious_near_tie():
    # ProjectD is the clean winner; rejected ProjectC sits at similar[1].
    resolved = {
        "matched_by": "vector", "canonical": "ProjectD",
        "rejected_canonicals": ["ProjectC"],
        "similar": [{"name": "ProjectD", "distance": 0.20},
                    {"name": "ProjectC", "distance": 0.22}],
    }
    d = wr.rule_decision("ProjectDvariant", resolved)
    # only one non-rejected candidate → no tie → AUTO, not a re-prompt
    assert d["decision"] == "AUTO"
    assert d["reason"] == "vector_strong"


def test_chosen_canonical_equal_to_rejected_keeps_separate():
    # defensive: if the chosen canonical itself is a rejected name (via a
    # non-confirmed/non-exact path), keep apart rather than merge.
    resolved = {
        "matched_by": "fallback", "canonical": "ProjectC",
        "rejected_canonicals": ["ProjectC"], "similar": [],
    }
    d = wr.rule_decision("ProjectC", resolved)
    assert d["decision"] == "KEEP" and d["reason"] == "rejected_pair"


# ── integration: write path surfaces decision ───────────────────────────────

def rules_make_tools(tmp_path: Path, isolation: str = "weak") -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "rules.sqlite3",
        backup_jsonl=tmp_path / "rules.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_write_surfaces_ask_for_generic_workspace(tmp_path):
    t = rules_make_tools(tmp_path)
    r = t.memory_write(content="some plan", workspace="月报", source_type="agent_generated", subject="test")
    data = r["data"]
    assert data["workspace_decision"] == "ASK"
    assert data.get("write_hints", {}).get("workspace_review")


def test_write_auto_for_specific_workspace(tmp_path):
    t = rules_make_tools(tmp_path)
    r = t.memory_write(content="alpha", workspace="金营项目", source_type="agent_generated", subject="test")
    assert r["data"]["workspace_decision"] == "AUTO"


# ── mechanical variant of an existing canonical (2026-08-21) ─────────────────
#
# A real-library dry-run showed agent-lane failing to reach its existing
# AgentLane canonical: exact match is case/separator sensitive and the vector
# tier could miss a low-frequency canonical. A deterministic case/hyphen/
# underscore/whitespace fold reuses the registered spelling without vector/Qwen.

def test_mechanical_variant_reuses_existing_canonical(tmp_path):
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute("INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES('AgentLane',datetime('now'))")

    for raw in ["agent-lane", "AGENTLANE", "agent_lane", "Agent Lane"]:
        r = db.resolve_workspace_canonical(raw, None, register_new=False)
        assert r["canonical"] == "AgentLane", raw
        assert r["matched_by"] == "mechanical_variant", raw

    # Exact spelling still takes the exact tier.
    exact = db.resolve_workspace_canonical("AgentLane", None, register_new=False)
    assert exact["matched_by"] == "exact"

    # A genuinely different name is NOT folded into the canonical.
    other = db.resolve_workspace_canonical("agentlanes-cli", None, register_new=False)
    assert other["matched_by"] == "new"
    assert other["canonical"] == "agentlanes-cli"


def test_mechanical_variant_does_not_collapse_blanks(tmp_path):
    # Empty/whitespace keys must never collide via the mechanical fold.
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute("INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES('AgentLane',datetime('now'))")
    r = db.resolve_workspace_canonical("   ", None, register_new=False)
    assert r["canonical"] == "default"
    assert r["matched_by"] == "fallback"


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
