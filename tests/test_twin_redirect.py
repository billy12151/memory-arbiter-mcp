"""0.16.2 twin write routing tests (plan §1.2): a non-twin caller's write,
move, or pending activation resolving to ``mema-twin`` lands in
``mema-twin-dev`` with a redirect notice; the twin itself is unaffected;
the canonical embedding is re-prepared for the destination (no name-vector
poisoning); the boot migration relocates existing violations."""
from __future__ import annotations

import json
import struct

import pytest

from pathlib import Path

from memory_arbiter.db import additive
from memory_arbiter.request_identity import RequestIdentity, request_identity_scope
from memory_arbiter.tools import MemoryTools
from memory_arbiter.twin_redirect import twin_redirect_target

from test_scan_pipeline import make_tools


def _bucket(record: dict) -> str:
    return str(record.get("workspace_canonical") or record.get("workspace") or "")


def _response_bucket(result: dict) -> str:
    # memory_write's `record` echoes the RAW input (no canonical field);
    # the resolved landing spot lives on the response envelope.
    return str(result.get("workspace_canonical") or "")


# ── unit: the routing predicate ─────────────────────────────────────────────

def test_twin_redirect_predicate() -> None:
    assert twin_redirect_target("mema-twin", client=None, agent_id=None) == "mema-twin-dev"
    assert twin_redirect_target("mema-twin", client="zcode", agent_id="agent-x") == "mema-twin-dev"
    # The twin itself passes on either identity channel.
    assert twin_redirect_target("mema-twin", client="mema-twin", agent_id=None) is None
    assert twin_redirect_target("mema-twin", client=None, agent_id="mema-twin") is None
    # Only the exact persona bucket; mema-twin-dev stays writable by anyone.
    assert twin_redirect_target("mema-twin-dev", client="zcode", agent_id="a") is None
    assert twin_redirect_target("proja", client=None, agent_id=None) is None


# ── write path ───────────────────────────────────────────────────────────────

def test_non_twin_write_redirects_to_dev(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    with request_identity_scope(RequestIdentity(client="zcode", agent_id="agent-x")):
        response = tools.memory_write(
            content="偏好素材内容", subject="pref", tags=[], workspace="mema-twin",
        )
    result = response["data"]
    assert _response_bucket(result) == "mema-twin-dev", result.get("workspace_canonical")
    notices = [
        n for n in response.get("notices") or []
        if n.get("type") == "protected_bucket_redirect"
    ]
    assert notices, "the redirect must be explained in the response notices"
    with tools.db.connection() as conn:
        bucket = conn.execute(
            "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) FROM memories WHERE id=?",
            (result["id"],),
        ).fetchone()[0]
    assert bucket == "mema-twin-dev"


def test_twin_write_passes(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    with request_identity_scope(RequestIdentity(client="mema-twin", agent_id="mema-twin")):
        result = tools.memory_write(
            content="本体偏好素材", subject="pref", tags=[], workspace="mema-twin",
        )["data"]
    assert _response_bucket(result) == "mema-twin"
    assert not any(
        n.get("type") == "protected_bucket_redirect" for n in result.get("notices") or []
    )


def test_redirect_does_not_poison_canonical_vector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact/alias resolution prepared the canonical embedding under the
    ORIGINAL name; after the redirect the published vector must be the
    DESTINATION's, not mema-twin's. The stock FakeEmbedder folds both names
    into one direction, so a distinguishing stub embedder proves the
    re-preparation actually happened."""
    tools = make_tools(tmp_path)

    from memory_arbiter.embedder import EmbedResult

    class NameAwareEmbedder:
        embedding_space_id = "name-aware-space"
        dim = 2
        last_encode_error = None

        @staticmethod
        def embed_text(prefix: str, body: str, max_body_chars=None) -> EmbedResult:
            text = f"{prefix}\n{body}".casefold()
            # DISTINCT directions per name — byte-equality would prove the
            # twin's name vector was published under mema-twin-dev.
            if "dev" in text:
                return EmbedResult([0.0, 1.0], False, len(text), len(text))
            return EmbedResult([1.0, 0.0], False, len(text), len(text))

    class StubManaged:
        def __getattr__(self, name):
            return getattr(NameAwareEmbedder, name)

    monkeypatch.setattr(
        type(tools), "_ensure_active_embedder",
        lambda self: (StubManaged(), []),
    )
    with request_identity_scope(RequestIdentity(client="zcode", agent_id="agent-x")):
        result = tools.memory_write(
            content="偏好素材内容", subject="pref", tags=[], workspace="mema-twin",
        )["data"]
    assert _response_bucket(result) == "mema-twin-dev"
    with tools.db.connection() as conn:
        vec = conn.execute(
            """SELECT v.embedding FROM workspace_canonicals_vec v
               JOIN workspace_canonicals c ON c.id=v.id WHERE c.name='mema-twin-dev'"""
        ).fetchone()
    assert vec is not None and vec["embedding"] is not None, (
        "the destination canonical must carry its own vector"
    )
    stored = struct.unpack(f"{len(bytes(vec['embedding'])) // 4}f", bytes(vec["embedding"]))
    assert list(stored) == [0.0, 1.0], (
        "mema-twin's name vector leaked into mema-twin-dev — anti-poison re-prepare missing"
    )


# ── move path (governance included) ─────────────────────────────────────────

def test_move_into_twin_redirects_even_authorized(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="普通项目内容", subject="note", tags=[], workspace="proja",
    )["data"]
    with request_identity_scope(RequestIdentity(client="claude-code", agent_id="agent-a")):
        moved = tools.memory_govern("move_memories_workspace", {
            "memory_ids": [written["id"]], "new_workspace": "mema-twin",
            "reason": "owner said so", "authorized": True,
        })
    assert moved["ok"] is True, moved
    record = tools.db.get_memory(written["id"])
    assert _bucket(record) == "mema-twin-dev"
    assert any("#976" in w or "mema-twin-dev" in w for w in moved.get("warnings") or [])


def test_move_into_twin_by_twin_passes(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="普通项目内容", subject="note", tags=[], workspace="proja",
    )["data"]
    with request_identity_scope(RequestIdentity(client="mema-twin", agent_id="mema-twin")):
        moved = tools.memory_govern("move_memories_workspace", {
            "memory_ids": [written["id"]], "new_workspace": "mema-twin",
            "reason": "twin reorganizing", "authorized": True,
        })
    assert moved["ok"] is True, moved
    record = tools.db.get_memory(written["id"])
    assert _bucket(record) == "mema-twin"


# ── pending activation path ─────────────────────────────────────────────────

def test_pending_activation_into_twin_redirects(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.settings.isolation = "strict"
    with request_identity_scope(RequestIdentity(client="zcode", agent_id="agent-x")):
        written = tools.memory_write(
            content="待确认内容", subject="pending", tags=[], workspace="mema-twin",
        )["data"]
        record = tools.db.get_memory(written["id"])
        status = record.get("status")
        if status == "pending":
            result = tools.memory_govern("confirm_pending_workspace", {
                "memory_id": written["id"], "canonical": "mema-twin",
                "reason": "confirm", "authorized": True,
            })
            assert result["ok"] is True, result
    record = tools.db.get_memory(written["id"])
    assert _bucket(record) == "mema-twin-dev", "activation cannot land in mema-twin for a non-twin caller"
    # Adversarial regression (R1 review): the confirm path must NOT write a
    # confirmed alias mema-twin→mema-twin-dev — that would reroute the twin's
    # OWN future writes (alias resolution bypasses the identity-keyed
    # redirect for everyone).
    with tools.db.connection() as conn:
        poison = conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases "
            "WHERE alias_workspace='mema-twin' AND canonical='mema-twin-dev' "
            "AND status='confirmed'"
        ).fetchone()[0]
    assert poison == 0, "confirm redirect poisoned the alias registry"
    # The twin itself still writes its bucket directly after the redirect.
    with request_identity_scope(RequestIdentity(client="mema-twin", agent_id="mema-twin")):
        again = tools.memory_write(
            content="本体后续写入", subject="twin-after", tags=[], workspace="mema-twin",
        )["data"]
    assert _response_bucket(again) == "mema-twin", (
        "twin's own write must still land in mema-twin (no alias reroute)"
    )


# ── boot migration ──────────────────────────────────────────────────────────

def test_boot_migration_moves_non_twin_residents(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # The twin's own row must be written FIRST and AS the twin: the stock
    # FakeEmbedder folds every non-keyword name into one direction, so once
    # another bucket exists "mema-twin" would vector-resolve into it. Writing
    # first registers the mema-twin canonical cleanly.
    with request_identity_scope(RequestIdentity(client="mema-twin", agent_id="mema-twin")):
        twin_row = tools.memory_write(
            content="本体内容", subject="twin-note", tags=[], workspace="mema-twin",
        )
    assert _response_bucket(twin_row["data"]) == "mema-twin"
    # Stock: a jingleAI-style row lands in mema-twin BEFORE this release
    # (simulated by direct UPDATE — the live redirect now prevents new ones),
    # plus a pending workspace suspect pinning the old bucket.
    victim = tools.memory_write(
        content="误写内容", subject="stray", tags=[], workspace="proja",
    )["data"]
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET workspace='mema-twin', workspace_canonical='mema-twin' WHERE id=?", (victim["id"],))
        conn.execute(
            """UPDATE memories SET agent_id='jingleAI-default' WHERE id=?""", (victim["id"],),
        )
        conn.execute(
            """INSERT INTO scan_queue(kind,workspace_canonical,status,candidate_key_hash,
                 member_versions,evidence,reason,severity,source,detail,created_at,updated_at)
               VALUES('workspace','mema-twin','pending',?,
                 ?,'[]','vector vote 10/10','normal','scan_pipeline',?, ?, ?)""",
            (
                "a" * 64,
                json.dumps([{"memory_id": victim["id"], "version": 1}]),
                json.dumps({"suspected_workspace": "mema-twin-dev", "current_workspace": "mema-twin"}),
                "2026-09-13T00:00:00+00:00", "2026-09-13T00:00:00+00:00",
            ),
        )
    # The tools' first boot already burned the one-shot guard (moved=0, the
    # stock row did not exist yet); simulate the UPGRADE boot that meets the
    # stock row by resetting the guard, as a fresh deploy would see it.
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM migration_state WHERE key='twin_write_redirect_migration_v1'")
    with tools.db.connection() as conn:
        applied = additive.ensure_additive_structures(conn)
    assert any("twin_redirect_migration" in item for item in applied), applied
    record = tools.db.get_memory(victim["id"])
    assert _bucket(record) == "mema-twin-dev", "non-twin stock must relocate"
    twin_record = tools.db.get_memory(twin_row["data"]["id"])
    assert _bucket(twin_record) == "mema-twin", "the twin's own rows are untouched"
    with tools.db.connection() as conn:
        suspect = conn.execute(
            "SELECT status FROM scan_queue WHERE kind='workspace' AND status='pending'"
        ).fetchall()
        assert suspect == [], "the stale suspect pinning the old bucket must expire"
        # Idempotent: a second boot is a no-op.
        applied2 = additive.ensure_additive_structures(conn)
    assert not any("twin_redirect_migration" in item for item in applied2)
