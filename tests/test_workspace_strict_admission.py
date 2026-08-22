"""strict 向量准入 (mema 721 期3).

The plan's core trap: strict recall is locked at the SQL layer, so changing
only a Python post-filter is a no-op that still passes naive tests. These
tests therefore assert BOTH halves of the contract:

  - recall widens (a query in workspace A returns memories of the in-radius
    workspace B), AND
  - the same memory is readable by id (memory_read/recent/conflict paths) —
    proving search and ACL share one admitted set, no "搜得到读不到".

Plus the guardrails: default flood insulation, short-name and generic-word
guards, the mema/abbreviation backstop, COUNT/pagination consistency, and the
`workspace_recall_admission=off` full rollback to exact-equality behavior.
"""
import json
from pathlib import Path

import pytest

from memory_arbiter.acl import CallerWorkspace, scope_names, workspace_scope_sql
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import MemoryStatus
from memory_arbiter.tools import MemoryTools

# Two-dimensional unit vectors whose cosine distances are known exactly, so the
# tests pin admission behavior without depending on a real embedding model.
VEC_SELF = [1.0, 0.0]
VEC_NEAR = [0.99, 0.141]      # cosine distance ≈ 0.01 → inside the 0.25 cutoff
VEC_FAR = [0.0, 1.0]          # cosine distance 1.0 → outside any sane cutoff


def make_tools(
    tmp_path: Path,
    *,
    admission: bool = True,
    cutoff: float = 0.25,
    min_name_len: int = 3,
    isolation: str = "strict",
) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "adm.sqlite3",
        backup_jsonl=tmp_path / "adm.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        enable_sqlite_vec=True,
        vec_dim=2,
        isolation=isolation,
        workspace_recall_admission=admission,
        workspace_recall_cutoff=cutoff,
        workspace_min_name_len=min_name_len,
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    if not tools.db.state.sqlite_vec_available:  # pragma: no cover - env guard
        pytest.skip("sqlite-vec unavailable")
    return tools


def publish(tools: MemoryTools, name: str, vector: list[float]) -> None:
    """Register a canonical with a published vector (no embedder needed)."""
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
            (name,),
        )
        row = conn.execute("SELECT id FROM workspace_canonicals WHERE name = ?", (name,)).fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
            (int(row["id"]), json.dumps(vector)),
        )


def active_write(tools: MemoryTools, content: str, workspace: str, subject: str = "test") -> int:
    mid = tools.memory_write(
        content=content, workspace=workspace, subject=subject,
        source_type="agent_generated",
    )["data"]["id"]
    record = tools.db.get_memory(mid)
    if record["status"] == MemoryStatus.PENDING.value:
        confirmed = tools.memory_govern("confirm_pending_workspace", {
            "memory_id": mid,
            "canonical": record["workspace_canonical"] or record["workspace"],
            "authorized": True,
        })
        assert confirmed["ok"] is True
    return mid


def results(search: dict) -> list[int]:
    return [r["id"] for r in (search.get("data") or {}).get("results") or []]


# ── shared SQL/scope helpers ────────────────────────────────────────────────

def test_workspace_scope_sql_collapses_to_equality_for_one_name():
    sql, params = workspace_scope_sql("WS", ["only"])
    assert sql == "WS = ?"
    assert params == ["only"]


def test_workspace_scope_sql_in_clause_and_dedup():
    sql, params = workspace_scope_sql("WS", ["a", "b", "a", "  ", "c"])
    assert sql == "WS IN (?,?,?)"
    assert params == ["a", "b", "c"]
    assert workspace_scope_sql("WS", []) == ("", [])
    assert workspace_scope_sql("WS", None) == ("", [])
    # a bare string is a one-element scope
    assert workspace_scope_sql("WS", "solo") == ("WS = ?", ["solo"])


def test_scope_names_normalizes():
    assert scope_names(None) == []
    assert scope_names("x") == ["x"]
    assert scope_names(["x", " x ", "", "y"]) == ["x", "y"]


def test_caller_workspace_scope_defaults_to_own_canonical():
    caller = CallerWorkspace(isolation="strict", workspace="w", canonical="proj", source="explicit")
    assert caller.scope_canonicals() == ("proj",)
    widened = CallerWorkspace(
        isolation="strict", workspace="w", canonical="proj", source="explicit",
        admitted=("proj", "proj-sibling"),
    )
    assert widened.scope_canonicals() == ("proj", "proj-sibling")


# ── admitted-set computation ────────────────────────────────────────────────

def test_admitted_canonicals_includes_near_excludes_far(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)

    admitted = tools.db.workspaces.admitted_canonicals(
        "agent-lane", cutoff=0.25, min_name_len=3,
    )
    assert admitted[0] == "agent-lane"          # own canonical always first
    assert "agent-rail" in admitted
    assert "unrelated-ws" not in admitted


def test_admitted_canonicals_do_not_truncate_valid_neighbors(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "tenant-root", VEC_SELF)
    for index in range(25):
        publish(tools, f"neighbor-{index:03d}", VEC_SELF)
    admitted = tools.db.workspaces.admitted_canonicals("tenant-root", cutoff=0.25)
    assert len(admitted) == 26
    assert "neighbor-024" in admitted


def test_evidence_knn_does_not_starve_scoped_hit_after_global_2048(tmp_path):
    """A scoped evidence hit after >2048 closer out-of-scope units remains
    reachable; admission is not a fixed global-window post-filter."""
    tools = make_tools(tmp_path)
    far_memory = active_write(tools, "far evidence owner", "unrelated-ws", "far evidence")
    target_memory = active_write(tools, "target admitted evidence", "agent-rail", "target evidence")
    with tools.db.write_transaction() as conn:
        for index in range(2050):
            cur = conn.execute(
                "INSERT INTO memory_evidence(memory_id,memory_version,content_hash,unit_index,kind,text,start_offset,end_offset,created_at) "
                "VALUES(?,1,?,?,?,?,0,1,'2026-01-01T00:00:00Z')",
                (far_memory, "f" * 64, index, "sentence", f"far-{index}"),
            )
            conn.execute(
                "INSERT INTO memory_evidence_vec(id,parent_status,embedding) VALUES(?,'active',?)",
                (int(cur.lastrowid), json.dumps(VEC_SELF)),
            )
        cur = conn.execute(
            "INSERT INTO memory_evidence(memory_id,memory_version,content_hash,unit_index,kind,text,start_offset,end_offset,created_at) "
            "VALUES(?,1,?,0,'sentence','target',0,6,'2026-01-01T00:00:00Z')",
            (target_memory, "t" * 64),
        )
        conn.execute(
            "INSERT INTO memory_evidence_vec(id,parent_status,embedding) VALUES(?,'active',?)",
            (int(cur.lastrowid), json.dumps(VEC_FAR)),
        )
    hits = tools.db.evidence_knn(
        VEC_SELF, k=1, workspace=("agent-lane", "agent-rail"),
    )
    assert [int(hit["memory_id"]) for hit in hits] == [target_memory]


def test_admitted_canonicals_degrades_to_own_canonical(tmp_path):
    tools = make_tools(tmp_path)
    # No published vector for the query canonical → degraded to exact scope.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES ('vecless', '2026-01-01T00:00:00Z')",
        )
    publish(tools, "agent-rail", VEC_NEAR)
    assert tools.db.workspaces.admitted_canonicals("vecless", cutoff=0.25) == ("vecless",)
    # default is insulated in both directions
    publish(tools, "default", VEC_SELF)
    assert tools.db.workspaces.admitted_canonicals("default", cutoff=0.25) == ("default",)


def test_admitted_canonicals_applies_short_name_guard(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "w", VEC_SELF)               # 1 char — below min_name_len
    publish(tools, "claw", VEC_NEAR)            # near in vector space
    assert tools.db.workspaces.admitted_canonicals("w", cutoff=0.25) == ("w",)
    # Control: the same near vectors DO admit between two long, unrelated names.
    publish(tools, "alpha-proj", VEC_SELF)
    assert "claw" in tools.db.workspaces.admitted_canonicals("alpha-proj", cutoff=0.25)


def test_admitted_canonicals_applies_generic_substring_guard(tmp_path):
    tools = make_tools(tmp_path)
    # Identical vectors (distance 0) but the names share only the substring
    # "main" — the 721 §3d hazard the guard exists for.
    publish(tools, "main", VEC_SELF)
    publish(tools, "openclaw-main", VEC_SELF)
    assert tools.db.workspaces.admitted_canonicals("main", cutoff=0.25) == ("main",)


def test_admitted_canonicals_excludes_default_terms_as_targets(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "默认", VEC_NEAR)             # a legacy synonym canonical
    admitted = tools.db.workspaces.admitted_canonicals("agent-lane", cutoff=0.25)
    assert admitted == ("agent-lane",)


# ── the core adversarial pair: recall AND read widen together ───────────────

def test_strict_admission_widens_recall_and_read(tmp_path):
    """Plan §3 核心验收: strict 下查 agent-lane 能召回 agent-rail 的记忆,
    且该记忆 memory_read 也放行（证明 ACL 同步、非「搜到读不到」）。"""
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "release checklist lives here", "agent-rail", "release checklist")
    lane_id = active_write(tools, "lane local note", "agent-lane", "lane note")

    found = results(tools.memory_search(query="release checklist", workspace="agent-lane"))
    assert rail_id in found, "recall did not widen — SQL admission is a no-op"

    read = tools.memory_get(memory_id=rail_id, workspace="agent-lane")
    assert read["ok"] is True, "search widened but read did not — ACL out of sync"
    assert read["data"]["memory"]["id"] == rail_id

    recent = tools.memory_recent(workspace="agent-lane", limit=50)
    recent_ids = [r["id"] for r in recent["data"]["results"]]
    assert rail_id in recent_ids and lane_id in recent_ids


def test_admission_off_restores_exact_isolation(tmp_path):
    """开关 off 时完全回退精确等值旧行为（recall + read 都不放宽）。"""
    tools = make_tools(tmp_path, admission=False)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "release checklist lives here", "agent-rail", "release checklist")

    assert rail_id not in results(tools.memory_search(query="release checklist", workspace="agent-lane"))
    assert tools.memory_get(memory_id=rail_id, workspace="agent-lane")["ok"] is False
    recent = tools.memory_recent(workspace="agent-lane", limit=50)
    assert rail_id not in [r["id"] for r in recent["data"]["results"]]


def test_far_workspace_never_admitted(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "unrelated-ws", VEC_FAR)
    far_id = active_write(tools, "unrelated secret content", "unrelated-ws", "unrelated subject")

    assert far_id not in results(tools.memory_search(query="unrelated secret", workspace="agent-lane"))
    assert tools.memory_get(memory_id=far_id, workspace="agent-lane")["ok"] is False


def test_default_pool_never_flooded_into_strict_recall(tmp_path):
    """default 洪水测试: strict 下任意查询不召回 default 池记忆。"""
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "default", VEC_SELF)  # even at distance 0
    # default writes are not new workspaces, so they are active immediately.
    default_id = tools.memory_write(
        content="global preference note", workspace="default", subject="global preference",
        source_type="agent_generated",
    )["data"]["id"]

    assert default_id not in results(tools.memory_search(query="global preference", workspace="agent-lane"))
    assert tools.memory_get(memory_id=default_id, workspace="agent-lane")["ok"] is False


def test_short_name_workspace_stays_isolated_end_to_end(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "w", VEC_SELF)
    publish(tools, "claw", VEC_NEAR)
    claw_id = active_write(tools, "claw project content", "claw", "claw subject")

    assert claw_id not in results(tools.memory_search(query="claw project", workspace="w"))
    assert tools.memory_get(memory_id=claw_id, workspace="w")["ok"] is False


def test_generic_substring_workspace_stays_isolated_end_to_end(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "main", VEC_SELF)
    publish(tools, "openclaw-main", VEC_NEAR)
    other_id = active_write(tools, "openclaw main content", "openclaw-main", "openclaw subject")

    assert other_id not in results(tools.memory_search(query="openclaw main", workspace="main"))
    assert tools.memory_get(memory_id=other_id, workspace="main")["ok"] is False


def test_over_cutoff_abbreviation_uses_workspace_migration(tmp_path):
    """Over-cutoff names stay separate until the user merges their workspaces."""
    tools = make_tools(tmp_path, cutoff=0.25)
    publish(tools, "memory-arbiter-mcp", VEC_SELF)
    publish(tools, "mema", VEC_FAR)  # too far to admit
    mema_id = active_write(tools, "mema abbreviation content", "mema", "mema subject")

    assert mema_id not in results(
        tools.memory_search(query="mema abbreviation", workspace="memory-arbiter-mcp")
    )
    merged = tools.memory_govern("migrate_workspace", {
        "from": "mema", "to": "memory-arbiter-mcp", "authorized": True,
    })
    assert merged["ok"] is True
    assert tools.memory_get(memory_id=mema_id, workspace="mema")["ok"] is True


# ── consistency: COUNT / pagination / expired / filters ────────────────────

def test_counts_and_pagination_match_admitted_scope(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    for index in range(3):
        active_write(tools, f"shared tagged item {index}", "agent-lane", f"lane item {index}")
    for index in range(2):
        active_write(tools, f"shared tagged item rail {index}", "agent-rail", f"rail item {index}")
    active_write(tools, "shared tagged item far", "unrelated-ws", "far item")

    # filter-driven path (empty query + tags/source filter) is SQL-exact
    page = tools.memory_search(
        query="", workspace="agent-lane", source_type="agent_generated", limit=100,
    )
    ids = results(page)
    assert len(ids) == 5, ids                      # 3 lane + 2 rail, never the far one
    assert page["data"]["total_estimate"] == 5     # COUNT agrees with the page

    first = tools.memory_search(
        query="", workspace="agent-lane", source_type="agent_generated", limit=2, offset=0,
    )
    second = tools.memory_search(
        query="", workspace="agent-lane", source_type="agent_generated", limit=2, offset=2,
    )
    assert first["data"]["total_estimate"] == second["data"]["total_estimate"] == 5
    assert not set(results(first)) & set(results(second))


def test_expired_recall_uses_same_admitted_scope(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "retired rail decision", "agent-rail", "retired rail")
    # Retire it from a caller that can see it, then read the expired domain
    # from the admitted neighbour — both halves must use the same scope.
    retired = tools.memory_supersede(
        memory_id=rail_id, reason="superseded for test", authorized=True, workspace="agent-rail",
    )
    assert retired["ok"] is True, retired

    expired = tools.memory_search_expired(query="retired rail", workspace="agent-lane")
    assert rail_id in [r["id"] for r in expired["data"]["results"]]


def test_supersede_authorization_widens_with_admission(tmp_path):
    """An admitted neighbour may govern the memory it can now read (unified
    admission: read and governance share one set)."""
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "rail governance target", "agent-rail", "rail governance")

    retired = tools.memory_supersede(
        memory_id=rail_id, reason="admitted neighbour retires it", authorized=True,
        workspace="agent-lane",
    )
    assert retired["ok"] is True, retired
    assert tools.db.get_memory(rail_id)["status"] == "superseded"


def test_recent_fallback_scopes_to_admitted_set(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    rail_id = active_write(tools, "rail body", "agent-rail", "rail subject")
    far_id = active_write(tools, "far body", "unrelated-ws", "far subject")

    # empty query + no filters → recent browse, scoped in SQL
    browse = tools.memory_search(query="", workspace="agent-lane", limit=100)
    ids = results(browse)
    assert rail_id in ids
    assert far_id not in ids


# ── governance/aggregate consistency ───────────────────────────────────────

def test_audit_summary_reports_each_admitted_workspace(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    active_write(tools, "lane content", "agent-lane", "lane subject")
    active_write(tools, "rail content", "agent-rail", "rail subject")
    active_write(tools, "far content", "unrelated-ws", "far subject")

    summary = tools.memory_audit_summary(workspace="agent-lane")["data"]
    assert set(summary["workspaces"]) == {"agent-lane", "agent-rail"}
    assert summary["total_memories"] == 2


def test_audit_summary_preserves_empty_caller_bucket_with_admission_off(tmp_path):
    tools = make_tools(tmp_path, admission=False)
    summary = tools.memory_audit_summary(workspace="empty-project")["data"]
    assert summary["workspaces"] == {
        "empty-project": {
            "count": 0, "oldest": None, "newest": None,
            "open_conflicts": 0, "by_source_type": {},
        }
    }
    assert summary["total_memories"] == 0


def test_strict_rebuild_evidence_scopes_discovery_flag_off_and_on(tmp_path):
    """Round-1 high-severity regression: a strict scope tuple must expand to
    SQL parameters, never bind as one scalar; admission widens discovery only
    when enabled."""
    tools = make_tools(tmp_path, admission=True)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    lane = active_write(tools, "index lane body", "agent-lane", "index lane")
    rail = active_write(tools, "index rail body", "agent-rail", "index rail")
    far = active_write(tools, "index far body", "unrelated-ws", "index far")

    widened = tools.memory_repair(
        "rebuild_evidence", {"dry_run": True, "workspace": "agent-lane", "batch_size": 50},
    )
    assert widened["ok"] is True, widened
    assert set(widened["data"]["memory_ids"]) == {lane, rail}
    assert far not in widened["data"]["memory_ids"]

    tools.settings.workspace_recall_admission = False
    exact = tools.memory_repair(
        "rebuild_evidence", {"dry_run": True, "workspace": "agent-lane", "batch_size": 50},
    )
    assert exact["ok"] is True, exact
    assert exact["data"]["memory_ids"] == [lane]


def test_entities_listing_scopes_to_admitted_set(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    lane_id = active_write(tools, "lane content", "agent-lane", "lane subject")
    rail_id = active_write(tools, "rail content", "agent-rail", "rail subject")
    far_id = active_write(tools, "far content", "unrelated-ws", "far subject")
    tools.memory_set_entity(memory_id=lane_id, entity="lane-entity", workspace="agent-lane")
    tools.memory_set_entity(memory_id=rail_id, entity="rail-entity", workspace="agent-rail")
    tools.memory_set_entity(memory_id=far_id, entity="far-entity", workspace="unrelated-ws")

    data = tools.memory_list_entities(workspace="agent-lane")["data"]
    listed = json.dumps(data, ensure_ascii=False)
    assert "lane-entity" in listed and "rail-entity" in listed
    assert "far-entity" not in listed


def test_console_browse_and_status_use_admitted_scope(tmp_path):
    from memory_arbiter.console_api import ConsoleAPI

    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    rail_id = active_write(tools, "rail console content", "agent-rail", "rail subject")
    rail_peer = active_write(tools, "rail console peer", "agent-rail", "rail peer")
    far_id = active_write(tools, "far console content", "unrelated-ws", "far subject")
    notice = tools.db.record_semantic_notice(
        memory_id=rail_id, peer_id=rail_peer, severity="normal",
        notice_type="semantic_evidence", title="console notice", message="console",
        payload={}, left_version=1, right_version=1,
    )
    assert notice["outcome"] == "created"

    api = ConsoleAPI(tools)
    browse = api.memories(query=None, workspace="agent-lane", limit=100, offset=0)
    ids = [item["id"] for item in (browse.get("items") or [])]
    assert rail_id in ids
    assert far_id not in ids

    counts = api._status_counts(workspace="agent-lane")
    assert counts["total"] == 2  # the two admitted rail rows; never the far one
    assert counts["active"] == 2
    overview = api.overview(workspace="agent-lane")
    assert overview["status"]["semantic_conflict"]["notices"].get("open") == 1


def test_conflict_detail_visible_across_admitted_workspaces(tmp_path):
    """A conflict recorded in an admitted neighbour is inspectable, and its
    members read back — the authorization path shares the admitted set."""
    import hashlib
    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    first = active_write(tools, "rail says sqlite", "agent-rail", "rail database")
    second = active_write(tools, "rail says mysql", "agent-rail", "rail database alt")

    members = []
    groups = []
    for index, memory_id in enumerate((first, second)):
        record = tools.db.get_memory(memory_id)
        value = f"value-{index}"
        quote = record["content"]
        members.append(ConflictMember(
            memory_id=memory_id, version=record["version"], attribute_raw="database",
            value_raw=value, normalized_attribute="database", normalized_value=value,
            evidence_quote=quote, evidence_span=(0, len(quote)),
            content_hash=hashlib.sha256(quote.encode()).hexdigest(),
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ))
        groups.append(ConflictValueGroup(value, value, (f"{memory_id}@{record['version']}",)))
    conflict_id = tools.db.record_conflict_group(
        workspace_canonical="agent-rail",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=members, value_groups=groups, detection_reason="different values",
        source="scan", detector_version="d1", conflict_point="database",
    )["conflict_id"]

    detail = tools.memory_review("conflict_detail", {"conflict_id": conflict_id, "workspace": "agent-lane"})
    assert detail["ok"] is True, detail
    assert detail["data"]["next_executable_call"]["data"]["workspace"] == "agent-lane"
    listing = tools.memory_list_conflicts(status="open", limit=50, workspace="agent-lane")
    assert conflict_id in [c["id"] for c in listing["data"]["conflicts"]]

    judged = tools.memory("judge", {
        "workspace": "agent-lane", "conflict_id": conflict_id,
        "expected_revision": 1, "chosen_value": "value-1",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [
            {"memory_id": first, "action": "preserve_historical_record"},
            {"memory_id": second, "action": "use_as_resolution"},
        ],
        "resolution_memory_id": second,
    })
    assert judged["ok"] is True, judged
    assert judged["data"]["next_action"]["data"]["workspace"] == "agent-lane"


def test_conflict_outside_admitted_set_stays_hidden(tmp_path):
    import hashlib
    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "unrelated-ws", VEC_FAR)
    first = active_write(tools, "far says sqlite", "unrelated-ws", "far database")
    second = active_write(tools, "far says mysql", "unrelated-ws", "far database alt")

    members = []
    groups = []
    for index, memory_id in enumerate((first, second)):
        record = tools.db.get_memory(memory_id)
        value = f"value-{index}"
        quote = record["content"]
        members.append(ConflictMember(
            memory_id=memory_id, version=record["version"], attribute_raw="database",
            value_raw=value, normalized_attribute="database", normalized_value=value,
            evidence_quote=quote, evidence_span=(0, len(quote)),
            content_hash=hashlib.sha256(quote.encode()).hexdigest(),
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ))
        groups.append(ConflictValueGroup(value, value, (f"{memory_id}@{record['version']}",)))
    conflict_id = tools.db.record_conflict_group(
        workspace_canonical="unrelated-ws",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=members, value_groups=groups, detection_reason="different values",
        source="scan", detector_version="d1", conflict_point="database",
    )["conflict_id"]

    detail = tools.memory_review("conflict_detail", {"conflict_id": conflict_id, "workspace": "agent-lane"})
    assert detail["ok"] is False
    listing = tools.memory_list_conflicts(status="open", limit=50, workspace="agent-lane")
    assert conflict_id not in [c["id"] for c in listing["data"]["conflicts"]]


def test_admitted_neighbor_notice_auto_delivery_and_retry_calls(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    left = active_write(tools, "rail notice left", "agent-rail", "rail notice")
    right = active_write(tools, "rail notice right", "agent-rail", "rail notice peer")
    created = tools.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal",
        notice_type="semantic_evidence", title="rail candidate", message="review rail",
        payload={}, left_version=1, right_version=1,
    )
    assert created["outcome"] == "created", created

    delivered = tools.memory(action="help", data={"workspace": "agent-lane"})
    notice_stub = next(n for n in delivered.get("notices", []) if n.get("notice_id") == created["notice_id"])
    assert notice_stub["read_call"]["data"]["workspace"] == "agent-lane"
    read = tools.memory_repair(
        notice_stub["read_call"]["task"], notice_stub["read_call"]["data"],
    )
    assert read["ok"] is True, read
    calls = read["data"]["notice"]["read_calls"]
    assert len(calls) == 2
    assert all(call["data"]["workspace"] == "agent-lane" for call in calls)
    assert all(isinstance(call["data"]["workspace"], str) for call in calls)


def test_notice_delivery_reuses_operation_scope(tmp_path, monkeypatch):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    calls = 0
    original = tools.db.workspaces.admitted_canonicals

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tools.db.workspaces, "admitted_canonicals", counted)
    response = tools.memory("find", {"query": "nothing", "workspace": "agent-lane"})
    assert response["ok"] is True
    assert calls == 1


def test_notice_claim_skips_full_hidden_page_in_one_call(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    left = active_write(tools, "hidden notice left", "agent-rail", "hidden left")
    right = active_write(tools, "hidden notice right", "agent-rail", "hidden right")
    for index in range(25):
        created = tools.db.record_semantic_notice(
            memory_id=left, peer_id=right, severity="normal",
            notice_type="semantic_evidence", title=f"hidden-{index}", message="hidden",
            payload={}, dedupe_key=f"hidden-{index}", left_version=1, right_version=1,
        )
        assert created["outcome"] == "created"
    assert tools.db.set_memory_workspace_canonical(right, "unrelated-ws")[0] is True
    valid_left = active_write(tools, "valid notice left", "agent-rail", "valid left")
    valid_right = active_write(tools, "valid notice right", "agent-rail", "valid right")
    valid = tools.db.record_semantic_notice(
        memory_id=valid_left, peer_id=valid_right, severity="normal",
        notice_type="semantic_evidence", title="valid", message="valid",
        payload={}, dedupe_key="valid-after-hidden", left_version=1, right_version=1,
    )
    delivered = tools.memory("help", {"workspace": "agent-lane"})
    assert any(n.get("notice_id") == valid["notice_id"] for n in delivered.get("notices", []))


def test_notice_counts_refresh_version_stale_before_read(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    left = active_write(tools, "stale count left", "agent-rail", "count left")
    right = active_write(tools, "stale count right", "agent-rail", "count right")
    created = tools.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal",
        notice_type="semantic_evidence", title="stale-count", message="count",
        payload={}, left_version=1, right_version=1,
    )
    assert created["outcome"] == "created"
    edited = tools.memory_edit(
        memory_id=left, new_content="stale count left edited", reason="version bump",
        workspace="agent-rail",
    )
    assert edited["ok"] is True
    status = tools.memory_status(workspace="agent-lane")
    assert status["data"]["semantic_conflict"]["notices"].get("stale") == 1
    assert status["data"]["semantic_conflict"]["notices"].get("open", 0) == 0


def test_stale_notice_list_fills_past_hidden_rows(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    visible_left = active_write(tools, "visible stale left", "agent-rail", "visible left")
    visible_right = active_write(tools, "visible stale right", "agent-rail", "visible right")
    visible = tools.db.record_semantic_notice(
        memory_id=visible_left, peer_id=visible_right, severity="normal",
        notice_type="semantic_evidence", title="visible-stale", message="visible",
        payload={}, dedupe_key="visible-stale", left_version=1, right_version=1,
    )
    tools.memory_edit(
        memory_id=visible_left, new_content="visible stale edited", reason="version bump",
        workspace="agent-rail",
    )
    hidden_left = active_write(tools, "hidden stale left", "agent-rail", "hidden stale left")
    hidden_right = active_write(tools, "hidden stale right", "agent-rail", "hidden stale right")
    for index in range(12):
        tools.db.record_semantic_notice(
            memory_id=hidden_left, peer_id=hidden_right, severity="normal",
            notice_type="semantic_evidence", title=f"hidden-stale-{index}", message="hidden",
            payload={}, dedupe_key=f"hidden-stale-{index}", left_version=1, right_version=1,
        )
    tools.db.set_memory_workspace_canonical(hidden_right, "unrelated-ws")
    listed = tools.memory_repair("notice", {
        "action": "list", "status": "stale", "limit": 1, "workspace": "agent-lane",
    })
    assert [n["id"] for n in listed["data"]["notices"]] == [visible["notice_id"]]


def test_stale_notice_hides_member_moved_outside_scope(tmp_path):
    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    left = active_write(tools, "rail public member", "agent-rail", "rail public")
    right = active_write(tools, "SECRET MOVED MEMBER", "agent-rail", "rail secret")
    created = tools.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal",
        notice_type="semantic_evidence", title="secret candidate", message="SECRET MESSAGE",
        payload={}, left_version=1, right_version=1,
    )
    assert created["outcome"] == "created", created
    assert tools.db.set_memory_workspace_canonical(right, "unrelated-ws")[0] is True

    read = tools.memory_repair("notice", {
        "action": "read", "notice_id": created["notice_id"], "workspace": "agent-lane",
    })
    assert read["ok"] is False
    listed = tools.memory_repair("notice", {
        "action": "list", "status": "stale", "limit": 10, "workspace": "agent-lane",
    })
    encoded = json.dumps(listed, ensure_ascii=False)
    assert str(created["notice_id"]) not in json.dumps(
        [n.get("id") for n in listed["data"]["notices"]]
    )
    assert "SECRET MOVED MEMBER" not in encoded
    assert "SECRET MESSAGE" not in encoded


def test_record_conflict_in_admitted_neighbor_uses_member_workspace(tmp_path):
    import hashlib

    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    ids = [
        active_write(tools, "rail value sqlite", "agent-rail", "rail sqlite"),
        active_write(tools, "rail value mysql", "agent-rail", "rail mysql"),
    ]
    members = []
    groups = []
    for index, memory_id in enumerate(ids):
        record = tools.db.get_memory(memory_id)
        value = ("sqlite", "mysql")[index]
        quote = record["content"]
        members.append({
            "memory_id": memory_id, "version": record["version"],
            "attribute_raw": "database", "value_raw": value,
            "normalized_attribute": "database", "normalized_value": value,
            "evidence_quote": quote, "evidence_span": [0, len(quote)],
            "content_hash": hashlib.sha256(quote.encode()).hexdigest(),
            "direction": "a_to_b", "prompt_version": "p1", "detector_version": "d1",
        })
        groups.append({
            "normalized_value": value, "display_value": value,
            "members": [f"{memory_id}@{record['version']}"],
        })
    result = tools.memory_repair("record_conflict", {
        "workspace": "agent-lane",
        "slot_key": {"entity": "project", "attribute": "database", "scope": "global"},
        "members": members, "value_groups": groups,
        "detector_version": "d1", "source": "scan", "reason": "different values",
        "conflict_point": "database",
    })
    assert result["ok"] is True, result
    conflict = tools.db.get_conflict(result["data"]["conflict_id"])
    assert conflict["workspace_canonical"] == "agent-rail"

    audit = tools.memory_audit_summary(workspace="agent-lane")["data"]
    assert audit["workspaces"]["agent-rail"]["open_conflicts"] == 1
    assert audit["workspaces"]["agent-lane"]["open_conflicts"] == 0
    assert audit["total_open_conflicts"] == 1


# ── anti-no-op: SQL layer must be the one doing the scoping ────────────────

def test_wide_recall_sql_scopes_to_admitted_set_directly(tmp_path):
    """Directly exercise the recall SQL with an admitted set: if a future change
    reverted the SQL to single-canonical equality (leaving only a Python
    post-filter), this fails — the plan's 空操作 detector."""
    from memory_arbiter.search import _wide_recall

    tools = make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "sql level admission probe", "agent-rail", "probe subject")

    pool = _wide_recall(
        tools.db, "sql level admission probe", None, None,
        "m.status = 'active'", "status = 'active'",
        ws_canonical=("agent-lane", "agent-rail"),
    )
    assert rail_id in [row["id"] for row in pool]

    narrow = _wide_recall(
        tools.db, "sql level admission probe", None, None,
        "m.status = 'active'", "status = 'active'",
        ws_canonical=("agent-lane",),
    )
    assert rail_id not in [row["id"] for row in narrow]
