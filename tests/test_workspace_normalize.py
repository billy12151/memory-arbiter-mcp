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
    settings = Settings(
        db_path=tmp_path / "norm.sqlite3",
        backup_jsonl=tmp_path / "norm.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        enable_sqlite_vec=vec, vec_dim=2, isolation="weak",
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


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
