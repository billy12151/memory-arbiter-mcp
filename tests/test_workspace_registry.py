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


_fact_serial = 0


def write(tools: MemoryTools, workspace: str, content: str | None = None) -> int:
    # 0.16.6 write gate: byte-identical ACTIVE content can no longer coexist
    # in one workspace, so the default content is uniquified per call. Callers
    # that need an exact value still pass one.
    global _fact_serial
    if content is None:
        content = f"workspace fact #{_fact_serial}"
        _fact_serial += 1
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
    resolved = tools.db.resolve_workspace_canonical("agent-lane", None)
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
    # G 守卫（0.17.1）后 twin rejected 行只能以存量形态存在：直插等价行。
    insert_alias(tools, "agent-lane", "AgentLane", "rejected")
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
    # Rejection recorded BEFORE either spelling was registered (legacy row —
    # G 守卫后此类行不再能经治理产生，直插等价形态): the ghost
    # spelling stays verbatim under the WINNER's alias key...
    insert_alias(tools, "agentlane", "agent-lane", "rejected")
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
    updated, warnings, _committed = tools.db.workspaces.migrate_workspace("old-ws", "agent-lane")
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
    resolved = tools.db.resolve_workspace_canonical("old-ws", None)
    assert resolved["canonical"] == "AgentLane"


def test_migrate_onto_own_mechanical_twin_is_noop(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    register(tools, "AgentLane")
    memory_id = write(tools, "AgentLane", "winner fact")
    updated, warnings, _committed = tools.db.workspaces.migrate_workspace("AgentLane", "agent-lane")
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

        def embed_texts(self, texts, prefix: str = ""):
            return [self.embed_text(prefix="", body=t) for t in texts]

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
    updated, warnings, _committed = tools.db.workspaces.migrate_workspace(
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
    updated, warnings, _committed = tools.db.workspaces.rename_workspace_canonical("old-ws", "agent-lane")
    assert warnings == []
    assert updated == 2
    canonicals = memory_canonicals(tools)
    assert canonicals[first] == "AgentLane"
    assert canonicals[second] == "AgentLane"
    assert canonicals[winner_memory] == "AgentLane"
    assert canonical_names(tools) == ["AgentLane"]
    assert ("old-ws", "AgentLane", "confirmed") in alias_rows(tools)
    resolved = tools.db.resolve_workspace_canonical("old-ws", None)
    assert resolved["canonical"] == "AgentLane"


def test_rename_onto_own_mechanical_twin_stays_a_spelling_rename(tmp_path: Path) -> None:
    # A destination whose registered mechanical twin IS the source is a
    # genuine spelling change of the same row, not a self-merge (the baseline
    # case-only rename contract).
    tools = make_tools(tmp_path)
    memory_id = write(tools, "ProjectX")
    updated, warnings, _committed = tools.db.workspaces.rename_workspace_canonical("ProjectX", "projectx")
    assert warnings == []
    assert updated == 1
    assert canonical_names(tools) == ["projectx"]
    assert memory_canonicals(tools)[memory_id] == "projectx"


# ---------------------------------------------------------------------------
#  Respected rejections in any spelling / shadowed redirects / plan parity
# ---------------------------------------------------------------------------

def test_cross_spelling_rejected_row_skips_whole_group(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # The rejection lives under a THIRD spelling of the pair (legacy row —
    # G 守卫后经治理不再产生，直插等价形态): neither the loser's nor the
    # winner's alias key carries the row, so exact-key lookups would miss it
    # and merge on top of an explicit user rejection.
    insert_alias(tools, "agent_lane", "AgentLane", "rejected")
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
    renamed, rename_warnings, _rc = tools.db.workspaces.rename_workspace_canonical(
        "old-ws", "agent-lane",
    )
    assert renamed == 0
    assert len(rename_warnings) == 1
    assert "workspace migration lock unavailable" in rename_warnings[0]
    migrated, migrate_warnings, _mc = tools.db.workspaces.migrate_workspace(
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


# ── parser ───────────────────────────────────────────────────────────────────






# ── per-isolation policy (stub backend) ──────────────────────────────────────

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
    def fake_resolve(ws_raw, embedder=None, *, match_distance=None):
        return {
            "canonical": ws_raw, "is_new": True, "matched_by": "new",
            "distance": None, "similar": [{"name": similar_name, "distance": distance}],
            "rejected_canonicals": [],
        }
    t.db.resolve_workspace_canonical = fake_resolve  # type: ignore
    t._ensure_semantic_backend = lambda: backend  # type: ignore


def test_model_suggester_retired_undecided_asks(tmp_path):
    """0.17.1 owner 拍板：模型建议器随 Qwen 判定引擎退役——undecided 规则
    直接 ASK（suggester_retired_ask / no_similar_candidates）；旧 AUTO 合并
    与全部 qwen_* decision_reason 不复存在。"""
    t = qwen_candidate_make_tools(tmp_path, "weak")
    tools = t["tools"] if isinstance(t, dict) else t
    assert not hasattr(tools, "_suggest_workspace_candidate"), (
        "retired suggester must be gone from MemoryTools"
    )
    result = tools.memory_write(
        content="占位内容", subject="zcode-data-migrat",
        workspace="zcode-data-migrat", metadata={"entity": "x", "scope": "y"},
    )
    data = result["data"]
    reason = data.get("workspace_decision_reason")
    assert reason not in {"qwen_high_conf", "qwen_low_conf", "qwen_timeout",
                          "qwen_backend_error", "qwen_unavailable", "qwen_rejected"}, reason



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
    resolved = tools.db.resolve_workspace_canonical(" 金营二期 ", None)
    assert resolved["canonical"] == "金营项目"
    assert resolved["matched_by"] == "confirmed_alias"
    state = tools.db.get_workspace_decision("金营二期")
    assert state["status"] == "confirmed"


def test_negative_decisions_accumulate_and_suppress_candidates(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    decide(tools, "raw", "candidate-a", status="rejected")
    decide(tools, "raw", "candidate-b", status="rejected")
    resolved = tools.db.resolve_workspace_canonical("raw", None)
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
        "workspace": "default",
        "old": "OldName", "new": "NewName", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "NewName"
    resolved = tools.db.resolve_workspace_canonical("OldName", None)
    assert resolved["canonical"] == "NewName"
    assert resolved["matched_by"] == "confirmed_alias"


def test_rename_moves_conflicts_with_their_members(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    left = write(tools, "OldName", "database is mysql")
    right = write(tools, "OldName", "database is sqlite")
    conflict_id = record_conflict(tools, "OldName", left, right)

    result = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
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
        "workspace": "default",
        "from": "Sub2", "to": "Main", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "Main"
    resolved = tools.db.resolve_workspace_canonical("Sub2", None)
    assert resolved["canonical"] == "Main"
    assert resolved["matched_by"] == "confirmed_alias"


def test_migrate_moves_conflicts_with_their_members(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    left = write(tools, "Sub2", "database is mysql")
    right = write(tools, "Sub2", "database is sqlite")
    conflict_id = record_conflict(tools, "Sub2", left, right)

    result = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "Sub2", "to": "Main", "authorized": True,
    })

    assert result["ok"] is True
    assert tools.db.get_conflict(conflict_id)["workspace_canonical"] == "Main"


def test_migrate_conflict_slot_collision_refused_up_front(tmp_path: Path) -> None:
    # v0.15.12+: the move is refused BEFORE any write with the colliding
    # conflict ids spelled out, instead of aborting mid-transaction on the
    # partial unique index with a bare sqlite error. Auto-resolving either
    # open conflict would fabricate a triage decision, so refusal is the only
    # honest outcome; triage then happens via the normal governance flow.
    tools = alias_governance_make_tools(tmp_path)
    old_left = write(tools, "Old", "old mysql")
    old_right = write(tools, "Old", "old sqlite")
    new_left = write(tools, "New", "new mysql")
    new_right = write(tools, "New", "new sqlite")
    old_conflict = record_conflict(tools, "Old", old_left, old_right)
    new_conflict = record_conflict(tools, "New", new_left, new_right)

    result = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "Old", "to": "New", "authorized": True,
    })

    assert result["ok"] is False
    warning = next(w for w in result["warnings"] if "would collide" in w)
    assert f"#{old_conflict}->#{new_conflict}" in warning
    assert tools.db.get_memory(old_left)["workspace_canonical"] == "Old"
    assert tools.db.get_memory(old_right)["workspace_canonical"] == "Old"
    assert tools.db.get_conflict(old_conflict)["workspace_canonical"] == "Old"
    assert tools.db.get_conflict(new_conflict)["workspace_canonical"] == "New"

    # Resolving the colliding source conflict unblocks the move. The full
    # open -> applying -> resolved chain is exercised in the lifecycle
    # tests; here we only need the row out of the active-slot index.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE conflicts SET status='resolved' WHERE id=?", (old_conflict,),
        )
    retry = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "Old", "to": "New", "authorized": True,
    })
    assert retry["ok"] is True
    assert tools.db.get_memory(old_left)["workspace_canonical"] == "New"
    assert tools.db.get_conflict(new_conflict)["workspace_canonical"] == "New"


def test_rename_conflict_slot_collision_refused_up_front(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    old_left = write(tools, "Old", "old mysql")
    old_right = write(tools, "Old", "old sqlite")
    new_left = write(tools, "New", "new mysql")
    new_right = write(tools, "New", "new sqlite")
    old_conflict = record_conflict(tools, "Old", old_left, old_right)
    new_conflict = record_conflict(tools, "New", new_left, new_right)

    result = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "New", "authorized": True,
    })

    assert result["ok"] is False
    warning = next(w for w in result["warnings"] if "would collide" in w)
    assert f"#{old_conflict}->#{new_conflict}" in warning
    assert tools.db.get_memory(old_left)["workspace_canonical"] == "Old"
    assert tools.db.get_conflict(new_conflict)["workspace_canonical"] == "New"


def test_migrate_different_slot_conflicts_do_not_collide(tmp_path: Path) -> None:
    # Same-slot pairs collide; different slots must keep moving normally.
    tools = alias_governance_make_tools(tmp_path)
    left = write(tools, "Old", "old mysql")
    right = write(tools, "Old", "old sqlite")
    conflict_id = record_conflict(tools, "Old", left, right)

    result = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "Old", "to": "New", "authorized": True,
    })

    assert result["ok"] is True
    assert tools.db.get_memory(left)["workspace_canonical"] == "New"
    assert tools.db.get_conflict(conflict_id)["workspace_canonical"] == "New"


def test_slot_collision_guard_ignores_terminal_status_rows(tmp_path: Path) -> None:
    # Only 'open'/'applying' rows occupy the active-slot index; resolved /
    # not_a_conflict rows sharing a slot must NOT trigger the refusal.
    tools = alias_governance_make_tools(tmp_path)
    a_left, a_right = write(tools, "A", "a mysql"), write(tools, "A", "a sqlite")
    b_left, b_right = write(tools, "B", "b mysql"), write(tools, "B", "b sqlite")
    c_left, c_right = write(tools, "C", "c mysql"), write(tools, "C", "c sqlite")
    a_conflict = record_conflict(tools, "A", a_left, a_right)
    b_conflict = record_conflict(tools, "B", b_left, b_right)
    c_conflict = record_conflict(tools, "C", c_left, c_right)
    with tools.db.write_transaction() as conn:
        # All three share one slot (record_conflict's fixed slot key) but only
        # B stays active.
        conn.execute("UPDATE conflicts SET status='resolved' WHERE id=?", (a_conflict,))
        conn.execute("UPDATE conflicts SET status='not_a_conflict' WHERE id=?", (c_conflict,))

    from memory_arbiter.db.workspaces import WorkspaceStore
    with tools.db.connection() as conn:
        assert WorkspaceStore._conflict_slot_collision_warning_on_conn(conn, "B", "A") is None
        assert WorkspaceStore._conflict_slot_collision_warning_on_conn(conn, "B", "C") is None
        assert WorkspaceStore._conflict_slot_collision_warning_on_conn(conn, "A", "C") is None


def test_normalize_reports_refused_merge_as_skipped_in_plan_and_execute(tmp_path: Path) -> None:
    # A would-be collision must show up as skipped-with-reason in the DRY-RUN
    # plan (plan honesty: not as a merge the execute pass must refuse), and
    # the execute pass must refuse the same merge and keep both canonicals.
    tools = alias_governance_make_tools(tmp_path)
    register(tools, "AgentLane", "agent-lane")
    winner_left = write(tools, "AgentLane", "winner mysql")
    winner_right = write(tools, "AgentLane", "winner sqlite")
    loser_left = write(tools, "agent-lane", "loser mysql")
    loser_right = write(tools, "agent-lane", "loser sqlite")
    record_conflict(tools, "AgentLane", winner_left, winner_right)
    record_conflict(tools, "agent-lane", loser_left, loser_right)

    plan = tools.db.workspaces.normalize_workspace_canonicals(dry_run=True)
    assert plan["ok"] is True
    refused = [item for item in plan["skipped"] if item.get("type") == "merge_refused"]
    assert len(refused) == 1
    assert refused[0]["from"] == "agent-lane" and refused[0]["to"] == "AgentLane"
    assert any(merge["from"] != "agent-lane" for merge in plan["merged"]) is False or all(
        merge["from"] != "agent-lane" for merge in plan["merged"]
    )

    executed = tools.db.workspaces.normalize_workspace_canonicals(dry_run=False)
    assert executed["ok"] is True
    assert [item for item in executed["skipped"] if item.get("type") == "merge_refused"]
    assert tools.db.get_memory(loser_left)["workspace_canonical"] == "agent-lane"
    assert tools.db.get_memory(winner_left)["workspace_canonical"] == "AgentLane"
    assert any("would collide" in warning for warning in executed["warnings"])
    result = tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "mema", "to": "memory-arbiter-mcp", "authorized": True,
    })
    assert result["ok"] is True
    assert result["data"]["memories_updated"] == 0
    resolved = tools.db.resolve_workspace_canonical("mema", None)
    assert resolved["canonical"] == "memory-arbiter-mcp"


def test_exact_negative_blocks_rename_forwarding(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "Old", "New", status="rejected")
    result = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "New", "authorized": True,
    })
    assert result["ok"] is True
    resolved = tools.db.resolve_workspace_canonical("Old", None)
    assert resolved["matched_by"] != "confirmed_alias"
    assert "New" in resolved["rejected_canonicals"]


def test_unrelated_negative_does_not_block_rename_forwarding(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    write(tools, "Old")
    decide(tools, "Old", "Other", status="rejected")
    tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "New", "authorized": True,
    })
    resolved = tools.db.resolve_workspace_canonical("Old", None)
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
        "workspace": "default",
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
        "workspace": "default",
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
        "workspace": "BrandNew",
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
        "workspace": "default",
        "memory_id": written["data"]["id"], "canonical": "ProjectX", "authorized": True,
    })
    assert result["ok"] is False
    assert "reserved default" in result["data"]["error"]
    assert tools.db.get_memory(written["data"]["id"])["status"] == MemoryStatus.PENDING.value


def test_competing_move_does_not_split_memory_and_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path)
    memory_id = write(tools, "Old")
    first = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "A", "authorized": True,
    })
    second = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "Old", "new": "B", "authorized": True,
    })
    assert first["ok"] is True
    assert second["ok"] is False
    assert tools.db.get_memory(memory_id)["workspace_canonical"] == "A"
    assert tools.db.resolve_workspace_canonical("Old", None)["canonical"] == "A"


def test_confirm_pending_exact_name_activates_without_self_redirect(tmp_path: Path) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "BrandNew")
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.PENDING.value
    result = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "BrandNew",
        "memory_id": memory_id, "canonical": "BrandNew", "authorized": True,
    })
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["status"] == MemoryStatus.ACTIVE.value
    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases WHERE alias_workspace='brandnew'"
        ).fetchone()[0] == 0


def test_confirm_pending_different_name_records_redirect_atomically(tmp_path: Path) -> None:
    """redirect 力学（raw≠canonical → confirmed alias 原子落库）在 none 隔离 +
    显式 pending 写入下钉住——strict 下该形态被第二道校验正确禁止（见
    test_confirm_pending_strict_forbids_cross_canonical）。"""
    tools = alias_governance_make_tools(tmp_path, isolation="none")
    written = tools.memory_write(
        content="abbrev scope", subject="redirect", workspace="abbrev",
        source_type="agent_generated", status="pending",
    )
    memory_id = written["data"]["id"]
    result = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "abbrev",
        "memory_id": memory_id, "canonical": "CanonicalProject", "authorized": True,
    })
    assert result["ok"] is True
    record = tools.db.get_memory(memory_id)
    assert record["status"] == MemoryStatus.ACTIVE.value
    assert record["workspace_canonical"] == "CanonicalProject"
    assert tools.db.resolve_workspace_canonical("abbrev", None)["canonical"] == "CanonicalProject"


def test_confirm_pending_rolls_back_decision_assignment_and_activation(tmp_path: Path, monkeypatch) -> None:
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    memory_id = write(tools, "abbrev")
    original = tools.db.set_memory_workspace_canonical_on_conn

    def fail(*args, **kwargs):
        return False, ["injected failure"]

    monkeypatch.setattr(tools.db, "set_memory_workspace_canonical_on_conn", fail)
    result = tools.memory_govern("confirm_pending_workspace", {
        "workspace": "default",
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
    # P2 #9: resolve is read-only now — register the fixture canonical
    # (+vector) through the store's idempotent backfill helper.
    tools.db.workspaces._publish_missing_workspace_canonical_vector(
        "Target", embedder, {},
    )
    decide(tools, "raw", "Target", status="rejected")
    resolved = tools.db.resolve_workspace_canonical("raw", embedder)
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
        r = db.resolve_workspace_canonical(raw, None)
        assert r["canonical"] == "AgentLane", raw
        assert r["matched_by"] == "mechanical_variant", raw

    # Exact spelling still takes the exact tier.
    exact = db.resolve_workspace_canonical("AgentLane", None)
    assert exact["matched_by"] == "exact"

    # A genuinely different name is NOT folded into the canonical.
    other = db.resolve_workspace_canonical("agentlanes-cli", None)
    assert other["matched_by"] == "new"
    assert other["canonical"] == "agentlanes-cli"


def test_mechanical_variant_does_not_collapse_blanks(tmp_path):
    # Empty/whitespace keys must never collide via the mechanical fold.
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute("INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES('AgentLane',datetime('now'))")
    r = db.resolve_workspace_canonical("   ", None)
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
            """INSERT INTO memory_row(memory_id,memory_version,content_hash,
               row_index,kind,text,start_offset,end_offset,created_at)
               VALUES(?,1,'hash',0,'sentence','body',0,4,'2026-01-01T00:00:00Z')""",
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
        "workspace": "default",
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
        "workspace": "default",
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


# ── 0.17.1 confirm 清场（prompt suppression 存量治理） ─────────────────────────

def _enqueue_pending_workspace_row(
    tools: MemoryTools, mid: int, own: str, suspected: str, tag: str,
) -> str:
    """Pending workspace row with the FULL detail envelope — 清场解析
    (current_workspace, suspected_workspace)；detail={} 的行自愈/清场不认。"""
    import hashlib

    candidate_key_hash = hashlib.sha256(f"confirm-sweep:{tag}".encode("utf-8")).hexdigest()
    outcome = tools.db.scan_queue.enqueue(
        kind="workspace",
        workspace_canonical=own,
        candidate_key_hash=candidate_key_hash,
        member_versions=[{"memory_id": mid, "version": 1}],
        evidence=[],
        reason="t",
        severity="normal",
        source="test",
        detail={"current_workspace": own, "suspected_workspace": suspected},
    )
    assert outcome.get("outcome") == "queued", outcome
    return candidate_key_hash


def _workspace_row_status(tools: MemoryTools, candidate_key_hash: str) -> str:
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT status FROM scan_queue WHERE candidate_key_hash=?",
            (candidate_key_hash,),
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_confirm_expires_pending_confirmed_pairs(tmp_path):
    """confirm 快照落地后：双确认对的 pending workspace 行即时 expired；
    未确认对（projB→projC，projC 不在快照里）不动。"""
    tools = review_doctor_make_tools(tmp_path)
    mid_a = _write(tools, "a", "projA")
    mid_b = _write(tools, "b", "projB")
    stale_hash = _enqueue_pending_workspace_row(tools, mid_a, "projA", "projB", "stale")
    live_hash = _enqueue_pending_workspace_row(tools, mid_b, "projB", "projC", "live")

    r = tools.memory_govern("confirm_workspaces", {"authorized": True})
    assert r["ok"] is True
    assert r["data"]["suppressed_pending"] == 1
    assert _workspace_row_status(tools, stale_hash) == "expired"
    assert _workspace_row_status(tools, live_hash) == "pending"


def test_confirm_sweep_falls_back_to_canonical_column(tmp_path):
    """R2 F4 补测：detail 缺 current_workspace（旧版/半完整行）时清场回退
    workspace_canonical 列——半 detail 行不比空 detail 行糟，仍须被治理。"""
    tools = review_doctor_make_tools(tmp_path)
    mid_a = _write(tools, "a", "projA")
    _write(tools, "b", "projB")  # 注册 projB，确认快照须含两端才会清场
    import hashlib

    candidate_key_hash = hashlib.sha256(b"confirm-sweep:col-fallback").hexdigest()
    outcome = tools.db.scan_queue.enqueue(
        kind="workspace",
        workspace_canonical="projA",
        candidate_key_hash=candidate_key_hash,
        member_versions=[{"memory_id": mid_a, "version": 1}],
        evidence=[], reason="t", severity="normal", source="test",
        detail={"suspected_workspace": "projB"},  # 无 current_workspace
    )
    assert outcome.get("outcome") == "queued", outcome

    r = tools.memory_govern("confirm_workspaces", {"authorized": True})
    assert r["ok"] is True
    assert r["data"]["suppressed_pending"] == 1
    assert _workspace_row_status(tools, candidate_key_hash) == "expired"


def test_confirm_expiry_failure_warns_not_fails(tmp_path, monkeypatch):
    """清场抛错绝不回滚快照：confirmed 仍 true、suppressed_pending=-1、
    降级 warning（下次 kick 自愈兜底）。"""
    tools = review_doctor_make_tools(tmp_path)
    mid = _write(tools, "a", "projA")
    row_hash = _enqueue_pending_workspace_row(tools, mid, "projA", "projB", "boom")

    def _raising_write_transaction(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tools.db, "write_transaction", _raising_write_transaction)
    r = tools.memory_govern("confirm_workspaces", {"authorized": True})
    assert r["ok"] is True
    assert r["data"]["confirmed"] is True
    assert r["data"]["suppressed_pending"] == -1
    assert any("清场失败" in w for w in r["warnings"]), r["warnings"]
    assert _sidecar(tools).exists(), "快照不得被清场失败回滚"
    assert _workspace_row_status(tools, row_hash) == "pending"


# ── 0.17.1 追加（2026-10-01）：mechanical_variant AUTO 补判 + rejected 机械通道免疫 ──
#
# 方案：ZCodeProject/docs/mema-ws-normalization-cleanup-plan-2026-10-01.md（v4）。
# G 守卫（变体对拒分）之后，新的 twin rejected 行无法再经治理产生；本节的
# rejected 行一律直插表内，模拟的是存量行（legacy）——免疫逻辑消费的正是它们。


def _insert_legacy_rejected_row(tools, alias: str, canonical: str) -> None:
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO workspace_aliases(alias_workspace,canonical,status,updated_at) "
            "VALUES(?,?,'rejected',datetime('now'))",
            (_normalize_alias_key(alias), canonical),
        )


def test_rule_decision_mechanical_variant_is_auto():
    resolved = {"matched_by": "mechanical_variant", "canonical": "AgentLane",
                "similar": [], "rejected_canonicals": []}
    d = wr.rule_decision("agent-lane", resolved, {"title": "t", "first_para": "内容"})
    assert d["decision"] == "AUTO"
    assert d["reason"] == "mechanical_variant"
    assert d["canonical"] == "AgentLane"


def test_rejected_legacy_row_blocks_mechanical_fold(tmp_path):
    """被拒原名（legacy rejected 行）不再被 1b 机械折叠进被拒桶（C 层1）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    _insert_legacy_rejected_row(t, "agent-lane", "AgentLane")
    r = db.resolve_workspace_canonical("agent-lane", None)
    assert r["matched_by"] == "new"
    assert r["canonical"] == "agent-lane"
    assert "AgentLane" in r["rejected_canonicals"]


def test_rejected_ghost_variant_second_hop_blocks_fold(tmp_path):
    """幽灵变体（agent-lane 被拒后写 agent_lane）经机械键第二跳吃到免疫（C 层2）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    _insert_legacy_rejected_row(t, "agent-lane", "AgentLane")
    r = db.resolve_workspace_canonical("agent_lane", None)
    assert r["matched_by"] == "new"
    assert r["canonical"] == "agent_lane"
    assert "AgentLane" in r["rejected_canonicals"]


def test_rejected_twin_does_not_block_exact_canonical(tmp_path):
    """桶本名 exact 命中不受 rejected 影响，规则层 AUTO（合法写入）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    _insert_legacy_rejected_row(t, "agent-lane", "AgentLane")
    r = db.resolve_workspace_canonical("AgentLane", None)
    assert r["matched_by"] == "exact"
    d = wr.rule_decision("AgentLane", r, {"title": "t", "first_para": "x"})
    assert d["decision"] == "AUTO"


def test_confirmed_ghost_variant_still_folds(tmp_path):
    """confirmed 幽灵变体不参与第二跳，仍走 1b 折叠（第二跳只消费 rejected）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    ok, errors = db.workspaces.record_workspace_decision(
        "agent-lane", "AgentLane", status="confirmed",
    )
    assert ok, errors
    r = db.resolve_workspace_canonical("agent_lane", None)
    assert r["matched_by"] == "mechanical_variant"
    assert r["canonical"] == "AgentLane"


def test_multiple_rejected_twin_rows_all_aggregated(tmp_path):
    """同机械键多行 rejected 全量聚合（真实库 agent-chancellor 双行形状，R2-P0c）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        for name in ("AgentLane", "MemoryBank"):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
                "VALUES(?,datetime('now'))",
                (name,),
            )
    _insert_legacy_rejected_row(t, "agent-chancellor", "AgentLane")
    _insert_legacy_rejected_row(t, "agent-chancellor", "MemoryBank")
    r = db.resolve_workspace_canonical("agent_chancellor", None)
    assert "AgentLane" in r["rejected_canonicals"]
    assert "MemoryBank" in r["rejected_canonicals"]
    assert r["matched_by"] == "new"
    assert r["canonical"] == "agent_chancellor"


def test_direct_rejected_hit_aggregates_sibling_spellings(tmp_path):
    """P2 #5：直中（alias key 精确命中拒绝行）也聚合机械兄弟拼写的拒绝——
    separate "agent-lane"→X 与 "agent_lane"→Y 后再解析 "agent-lane"，
    X/Y 同时进抑制名单，向量并桶被拒。"""
    pytest.importorskip("sqlite_vec")
    t = alias_governance_make_tools(tmp_path, vec=True)
    if not t.db.state.sqlite_vec_available:
        pytest.skip("sqlite-vec unavailable")

    class Embedder:
        embedding_space_id = "test"
        dim = 2

        def embed_text(self, prefix="", body=""):
            return SimpleNamespace(embedding=[1.0, 0.0])

    embedder = Embedder()
    for target in ("AlphaBucket", "BetaBucket"):
        t.db.workspaces._publish_missing_workspace_canonical_vector(target, embedder, {})
    ok1, e1 = t.db.workspaces.record_workspace_decision(
        "agent-lane", "AlphaBucket", status="rejected",
    )
    ok2, e2 = t.db.workspaces.record_workspace_decision(
        "agent_lane", "BetaBucket", status="rejected",
    )
    assert ok1 and ok2, (e1, e2)

    r = t.db.resolve_workspace_canonical("agent-lane", embedder)
    # both spellings' rejections aggregated on the direct hit
    assert "AlphaBucket" in r["rejected_canonicals"]
    assert "BetaBucket" in r["rejected_canonicals"]
    # the vector merge into the (distance-0) rejected targets is refused
    assert r["matched_by"] == "new"
    assert r["canonical"] == "agent-lane"
    assert {s["name"] for s in r["similar"]} == set()


def test_ghost_hop_does_not_redirect_unrelated_registered_twin(tmp_path):
    """跨身份 rejected（foo-bar→ProjectX）不误伤已注册的孪生拼写 foo_bar（exact 优先）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('foo_bar',datetime('now'))"
        )
    ok, errors = db.workspaces.record_workspace_decision(
        "foo-bar", "ProjectX", status="rejected",
    )
    assert ok, errors
    r = db.resolve_workspace_canonical("foo_bar", None)
    assert r["matched_by"] == "exact"
    assert r["canonical"] == "foo_bar"


def test_separate_refuses_mechanical_twin_pair(tmp_path):
    """G 守卫：同一机械身份的变体对 separate 直接拒绝（owner 2026-10-01 拍板）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    result = t.memory_govern("separate_workspace_alias", {
        "workspace": "default",
        "alias": "agent-lane", "canonical": "AgentLane",
        "reason": "try to split", "authorized": True,
    })
    assert result["ok"] is False, result["data"]
    assert "spelling variants" in result["data"]["error"]
    # 幽灵拼写同样被拒
    ok2, errors2 = db.workspaces.record_workspace_decision(
        "agent_lane", "AgentLane", status="rejected",
    )
    assert not ok2 and errors2
    with db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspace_aliases").fetchone()[0] == 0
    # confirmed 方向不受守卫影响（ twin 确认=折叠语义，合法）
    ok3, errors3 = db.workspaces.record_workspace_decision(
        "agent-lane", "AgentLane", status="confirmed",
    )
    assert ok3, errors3


def test_strict_rejected_name_goes_pending(tmp_path):
    """strict 下被拒原名落 new → strict_block → PENDING + confirm 流程（新桶须确认）。"""
    t = rules_make_tools(tmp_path, "strict")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    _insert_legacy_rejected_row(t, "agent-lane", "AgentLane")
    r = t.memory_write(
        content="x", subject="s", workspace="agent-lane",
        source_type="agent_generated",
    )
    assert r["ok"], r
    data = r["data"]
    assert data["workspace_canonical"] == "agent-lane"
    assert data.get("action_required") == "confirm_new_workspace"
    assert db.get_memory(data["id"])["status"] == MemoryStatus.PENDING.value


def test_strict_mechanical_variant_reuses_active(tmp_path):
    """strict 下机械变体（确定性身份）直接 ACTIVE 复用，不 ASK 不 PENDING。"""
    t = rules_make_tools(tmp_path, "strict")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    r = t.memory_write(
        content="x", subject="s", workspace="agent-lane",
        source_type="agent_generated",
    )
    assert r["ok"], r
    data = r["data"]
    assert data["workspace_decision"] == "AUTO"
    assert data["workspace_canonical"] == "AgentLane"
    assert db.get_memory(data["id"])["status"] == MemoryStatus.ACTIVE.value


def test_migrate_drops_twin_contradicting_rejection_with_warning(tmp_path):
    """R2-P1：migrate repoint 不得制造孪生 rejected 行——会咬合劈桶的行丢弃+警告。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    register(t, "OldProj", "agent-lane")
    # 跨身份合法 rejected（agent_lane ↛ OldProj，G 允许创建）
    _insert_legacy_rejected_row(t, "agent_lane", "OldProj")
    updated, warnings, _committed = db.workspaces.migrate_workspace("OldProj", "agent-lane")
    assert updated >= 0
    assert any("dropped while repointing" in w and "spelling variant" in w for w in warnings), warnings
    # 行已丢弃（不再存在会咬合的孪生 rejected），折叠语义恢复
    r = db.resolve_workspace_canonical("agent_lane", None)
    assert r["matched_by"] == "mechanical_variant"
    assert r["canonical"] == "agent-lane"
    # 非孪生 rejected 行照常跟随迁移
    _insert_legacy_rejected_row(t, "unrelated", "agent-lane")
    _u2, w2, _c2 = db.workspaces.migrate_workspace("agent-lane", "MemoryBank")
    assert not any("dropped while repointing" in w for w in w2), w2
    with db.connection() as conn:
        rows = [
            (str(row["alias_workspace"]), str(row["canonical"]), str(row["status"]))
            for row in conn.execute(
                "SELECT alias_workspace,canonical,status FROM workspace_aliases"
            )
        ]
    assert ("unrelated", "MemoryBank", "rejected") in rows


def test_multiple_rejected_rows_across_spellings_all_aggregated(tmp_path):
    """同机械键**不同拼写** alias 的多行 rejected 全量聚合（R2-P2 变异实测钉性缺口）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        for name in ("AgentLane", "MemoryBank"):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
                "VALUES(?,datetime('now'))",
                (name,),
            )
    _insert_legacy_rejected_row(t, "agent-chancellor", "AgentLane")
    _insert_legacy_rejected_row(t, "agent_chancellor", "MemoryBank")
    # 原名带空格形态：精确键双 miss，只可能经第二跳聚合
    r = db.resolve_workspace_canonical("agent chancellor", None)
    assert "AgentLane" in r["rejected_canonicals"]
    assert "MemoryBank" in r["rejected_canonicals"]
    assert r["matched_by"] == "new"


def test_workspace_required_on_govern_actions_surface(tmp_path: Path) -> None:
    """C1 必传（owner 2026-10-03）：五个改桶动作缺/空 workspace 在 validation
    层打回，报错带 workspaces 列表指路；move/separate 补钉（remember/confirm/
    rename 已入 golden 语料）。"""
    tools = alias_governance_make_tools(tmp_path, isolation="none")
    for action, payload in (
        ("separate_workspace_alias", {"alias": "agent-lane", "canonical": "AgentLane", "authorized": True}),
        ("move_memories_workspace", {"memory_ids": [1], "new_workspace": "ws", "authorized": True}),
    ):
        missing = tools.memory_govern(action, dict(payload))
        assert missing["ok"] is False, action
        assert missing["data"]["field"] == "workspace"
        assert "memory_review(view='workspaces')" in missing["data"]["reason"]
        empty = tools.memory_govern(action, {**payload, "workspace": "  "})
        assert empty["ok"] is False, action
        assert empty["data"]["field"] == "workspace"


def test_memory_review_workspaces_view(tmp_path: Path) -> None:
    """C2：workspaces 列表——计数/别名/空桶标注/排序/limit 单 SQL 口径。"""
    tools = alias_governance_make_tools(tmp_path, isolation="none")
    write(tools, "Alpha")  # pending（none 下 generic 新名走 ASK→pending 或 active，断言口径放宽）
    write(tools, "Beta")
    result = tools.memory_review("workspaces", {"limit": 50})
    assert result["ok"] is True
    ws = {b["canonical"]: b for b in result["data"]["workspaces"]}
    assert "Alpha" in ws and "Beta" in ws
    alpha = ws["Alpha"]
    for key in ("active_count", "pending_count", "alias_count", "last_write_at", "empty"):
        assert key in alpha
    limited = tools.memory_review("workspaces", {"limit": 1})
    assert limited["ok"] is True
    assert limited["data"]["count"] == 1


def test_memory_review_workspaces_strict_admitted_only(tmp_path: Path) -> None:
    """C2 strict ACL：workspaces 视图只见 admitted 集；无 canonical=denied 优先。"""
    tools = alias_governance_make_tools(tmp_path, isolation="strict")
    tools.settings.workspace = "Alpha"
    write(tools, "Alpha")
    write(tools, "Beta")
    with tools.db.connection() as conn:
        pending = conn.execute(
            "SELECT id FROM memories WHERE workspace='Alpha' AND status='pending'"
        ).fetchone()
    if pending:
        tools.memory_govern("confirm_pending_workspace", {
            "workspace": "Alpha", "memory_id": pending[0],
            "canonical": "Alpha", "authorized": True,
        })
    result = tools.memory_review("workspaces", {"workspace": "Alpha"})
    assert result["ok"] is True
    names = {b["canonical"] for b in result["data"]["workspaces"]}
    assert "Alpha" in names
    assert "Beta" not in names, "strict 调用者不得看见 scope 外桶"
