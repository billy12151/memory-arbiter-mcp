"""0.16.3 default-fallback tests (owner rule): when an agent genuinely
cannot find a suitable bucket, moving memories BACK to the global default
pool is allowed as an explicitly declared escape hatch — under strict
isolation default is the only bucket outside the caller's own that still
participates in recall. Guard rails pinned here: explicit flag, non-empty
reason, conf gate on the queue path, audit trail, user-facing notice."""
from __future__ import annotations

import json
from pathlib import Path

from test_scan_pipeline import make_tools, _write


def _bucket(record: dict) -> str:
    return str(record.get("workspace_canonical") or record.get("workspace") or "")


# ── move entry: explicit flag + reason required ────────────────────────────

def test_move_to_default_without_flag_still_refused(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "主题甲", "正文内容甲", workspace="proja")
    result = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [a], "new_workspace": "default",
        "reason": "no bucket", "authorized": True,
    })
    assert not result["ok"], "the default ban stands without the explicit flag"
    assert "default_fallback" in result["data"]["error"] or "reserved" in result["data"]["error"]


def test_move_fallback_requires_reason(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "主题甲", "正文内容甲", workspace="proja")
    result = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [a], "new_workspace": "default",
        "default_fallback": True, "authorized": True,
    })
    assert not result["ok"] and "reason" in result["data"]["error"]


def test_move_fallback_requires_default_target(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "主题甲", "正文内容甲", workspace="proja")
    result = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [a], "new_workspace": "projb",
        "default_fallback": True, "reason": "x", "authorized": True,
    })
    assert not result["ok"] and "default" in result["data"]["error"]


def test_move_fallback_lands_audits_and_notifies(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "主题甲", "正文内容甲", workspace="proja")
    result = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [a], "new_workspace": "默认",  # any reserved synonym folds
        "default_fallback": True, "reason": "投票无显著去向，跨项目内容",
        "authorized": True,
    })
    assert result["ok"], result
    record = tools.db.get_memory(a)
    assert _bucket(record) == "default", "reserved synonyms fold to the one true spelling"
    # audit row
    with tools.db.connection() as conn:
        audit = conn.execute(
            "SELECT gate, status FROM normalize_audit WHERE memory_id=? ORDER BY id DESC LIMIT 1",
            (a,),
        ).fetchone()
    assert audit and audit["status"] == "manual_move"
    assert json.loads(audit["gate"])["default_fallback"] is True
    # user-facing notice
    notices = [n for n in result.get("notices") or [] if n.get("type") == "default_fallback"]
    assert notices, "the fallback landing must be user-visible"
    assert result["data"]["default_fallback"]["count"] == 1
    # default never becomes a registry canonical
    with tools.db.connection() as conn:
        reg = conn.execute(
            "SELECT COUNT(*) FROM workspace_canonicals WHERE name='default'"
        ).fetchone()[0]
    assert reg == 0


# ── queue suspect submit: fallback=true waives the vote gate ───────────────

def _seed_suspect(tools, memory_id: int, current: str = "pgsqlproj") -> None:
    now = "2026-09-13T00:00:00+00:00"
    with tools.db.write_transaction() as conn:
        conn.execute(
            """INSERT INTO scan_queue(kind,workspace_canonical,status,candidate_key_hash,
                 member_versions,evidence,reason,severity,source,detail,created_at,updated_at)
               VALUES('workspace',?, 'pending',?,?,'[]','vector vote 4/10 -> ''default''',
                      'normal','scan_pipeline',?, ?, ?)""",
            (
                current, "f" * 64,
                json.dumps([{"memory_id": memory_id, "version": 1}]),
                json.dumps({"suspected_workspace": "default", "current_workspace": current}),
                now, now,
            ),
        )


def _submit(tools, decisions):
    return tools.memory_repair("scan_queue", {"action": "submit", "decisions": decisions})["data"]


def test_submit_fallback_moves_to_default_without_vote_gate(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "跨项目备忘", "正文内容", workspace="pgsqlproj")
    _seed_suspect(tools, a)
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": a, "status": "confirmed",
        "target_workspace": "default", "conf": 0.9, "fallback": True,
        "reason": "跨项目内容，投票无单一显著去处",
    }])
    entry = result["results"][0]
    assert entry["outcome"] == "moved", entry
    assert entry["default_fallback"]["reason"]
    record = tools.db.get_memory(a)
    assert _bucket(record) == "default"
    with tools.db.connection() as conn:
        audit = conn.execute(
            "SELECT gate FROM normalize_audit WHERE memory_id=? ORDER BY id DESC LIMIT 1", (a,),
        ).fetchone()
    assert json.loads(audit["gate"])["default_fallback"] is True


def test_submit_fallback_still_requires_conf(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "跨项目备忘", "正文内容", workspace="pgsqlproj")
    _seed_suspect(tools, a)
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": a, "status": "confirmed",
        "target_workspace": "default", "conf": 0.5, "fallback": True,
        "reason": "低置信",
    }])
    assert result["results"][0]["outcome"] == "gate_failed"
    assert tools.db.get_memory(a) and _bucket(tools.db.get_memory(a)) == "pgsqlproj"


def test_submit_fallback_rejected_for_project_target(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "跨项目备忘", "正文内容", workspace="pgsqlproj")
    _seed_suspect(tools, a)
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": a, "status": "confirmed",
        "target_workspace": "proja", "conf": 0.9, "fallback": True,
        "reason": "x",
    }])
    assert result["results"][0]["outcome"] == "invalid_input"


def test_submit_without_fallback_flag_keeps_vote_gate(tmp_path: Path) -> None:
    """A confirmed default target WITHOUT the flag is not a fallback: the vote
    gate must judge it like any target (and the seeded 4/10 vote fails)."""
    tools = make_tools(tmp_path)
    a = _write(tools, "跨项目备忘", "正文内容", workspace="pgsqlproj")
    _seed_suspect(tools, a)
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": a, "status": "confirmed",
        "target_workspace": "default", "conf": 0.9,
        "reason": "silent default attempt",
    }])
    entry = result["results"][0]
    assert entry["outcome"] == "gate_failed", entry
    assert _bucket(tools.db.get_memory(a)) == "pgsqlproj"


# ── doctor visibility ───────────────────────────────────────────────────────

def test_doctor_counts_default_fallback_landings(tmp_path: Path) -> None:
    from memory_arbiter.doctor import run_all_checks

    tools = make_tools(tmp_path)
    a = _write(tools, "主题甲", "正文内容甲", workspace="proja")
    tools.memory_govern("move_memories_workspace", {
        "memory_ids": [a], "new_workspace": "default",
        "default_fallback": True, "reason": "no bucket", "authorized": True,
    })
    with tools.db.connection() as conn:
        report = run_all_checks(conn, tools.settings)
    finding = next(
        (f for f in report.findings if f.check_id == "normalize.autonomy"), None,
    )
    assert finding is not None
    assert finding.evidence["default_fallback"] == 1
    assert "parked in default" in finding.detail


def test_submit_fallback_synonym_folds_to_canonical_default(tmp_path: Path) -> None:
    """0.16.4 review P2: an accepted synonym (默认) must FOLD onto the
    canonical 'default' — a raw landing would create a phantom bucket split
    off from the real default pool in recall scoping."""
    tools = make_tools(tmp_path)
    a = _write(tools, "同义词折叠", "正文内容", workspace="pgsqlproj")
    _seed_suspect(tools, a)
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": a, "status": "confirmed",
        "target_workspace": "默认", "conf": 0.9, "fallback": True,
        "reason": "无合适桶（同义词入口）",
    }])
    entry = result["results"][0]
    assert entry["outcome"] == "moved", entry
    record = tools.db.get_memory(a)
    assert _bucket(record) == "default", _bucket(record)


def test_doctor_counts_queue_channel_fallback(tmp_path: Path) -> None:
    """0.16.4 review P2: the audit counter covers BOTH entrances — the
    memory_govern manual_move path and the judgment-queue applied path."""
    tools = make_tools(tmp_path)
    a = _write(tools, "队列通道计数", "正文内容", workspace="pgsqlproj")
    _seed_suspect(tools, a)
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": a, "status": "confirmed",
        "target_workspace": "default", "conf": 0.9, "fallback": True,
        "reason": "队列通道",
    }])
    assert result["results"][0]["outcome"] == "moved"
    report = tools.memory_doctor_overview(deep=False)
    payload = report.get("data") or report
    finding = next(
        f for f in payload["findings"] if f["check_id"] == "normalize.autonomy"
    )
    assert finding["evidence"]["default_fallback"] >= 1, finding
