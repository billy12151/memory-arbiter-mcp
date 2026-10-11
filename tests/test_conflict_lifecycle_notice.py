def _pass_cos_gate(monkeypatch, cos: float = 0.85):
    """Gate-v2 G4: pass every hand-built hit through the cosine band
    (embedder-agnostic — patches the gates module, which the detection loop
    imports at call time)."""
    import memory_arbiter.pipeline.gates as _gates
    monkeypatch.setattr(
        _gates, "candidate_cos_gate",
        lambda own, hits, vecs: ([(h, cos) for h in hits], [], []),
    )

"""Regression tests for the v3 conflict-notice hardening slice.

Covers: model output with missing parsed keys no longer raising KeyError on
the notice path, per-task degradation counting deduplicated by reason,
pending-workspace memories reporting a skipped (not incomplete) conflict job,
and slot_key entity/scope canonicalisation with raw+canon double-form
suppression matching against applying conflict groups (B-C4).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.test_vnext_evidence as tv
from memory_arbiter.config import Settings
from memory_arbiter.semantic_conflict import decide_evidence, notice_dedupe_key
from memory_arbiter.tools import MemoryTools
from memory_arbiter.evidence import evidence_content_hash
from memory_arbiter.models import ConflictMember, ConflictValueGroup


def _write_pair(tools, meta: dict, left: str = "database is mysql",
                right: str = "database is sqlite") -> tuple[dict, dict]:
    peer = tools.memory_write(content=left, subject="a", tags=[], metadata=meta)["data"]
    new = tools.memory_write(content=right, subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    return peer, new


def _hit(peer_id: int, text: str, *, row_id: int = 1, distance: float = 0.1,
         metadata: dict | None = None) -> dict:
    # 0.16.2 write-time provenance gate reads hit['metadata'] exactly like
    # the real knn row does — hand-built hits carry it explicitly.
    return {
        "memory_id": peer_id, "id": row_id, "kind": "text", "text": text,
        "start_offset": 0, "end_offset": len(text), "distance": distance,
        "metadata": metadata,
    }


def test_notice_value_groups_carry_quote_sides(tmp_path: Path, monkeypatch) -> None:
    """0.17.1: judged notices carry the two ROW TEXTS as the value sides —
    no extraction, so value_groups hold the quotes themselves (empty
    normalised value, quote as display). The old missing-parsed-keys
    tolerance died with the extraction paradigm (nothing parses)."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "MyProject", "scope": "Production"}
    peer, new = _write_pair(tools, meta, left="database mysql 8 and 16", right="database sqlite 3 and 4")
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: [_hit(peer["id"], "database mysql 8 and 16", metadata=dict(meta))])
    _pass_cos_gate(monkeypatch)

    class Backend:
        @staticmethod
        def judge_pair(text_a, text_b):
            from memory_arbiter.semantic_judge import PairVerdict
            return PairVerdict(
                "conflict", {"conflict": 0.95, "no_conflict": 0.03, "possible_conflict": 0.02},
                "numeric_value", "mdeberta-v4m:test",
            )
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: Backend())

    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert result["outcome"] == "notices_created"
    notice = tools.db.list_semantic_notices(status="open", limit=10)[0]
    groups = notice["payload"]["value_groups"]
    assert [g["display_value"] for g in groups] == [
        "database sqlite 3 and 4", "database mysql 8 and 16",
    ]
    payload = notice["payload"]
    assert payload["model_signal"]["label"] == "conflict"
    assert payload["model_signal"]["model_version"].startswith("mdeberta-v4m:")


def test_same_reason_degradation_counted_once_per_task(tmp_path: Path, monkeypatch) -> None:
    """Two pairs failing with the same technical reason bump the counter once."""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    peers = [
        tools.memory_write(
            content=f"database {value} 8 and 16", subject=value, tags=[], metadata=meta,
        )["data"]
        for value in ("mysql", "postgres")
    ]
    new = tools.memory_write(
        content="database sqlite 3 and 4", subject="sqlite", tags=[], metadata=meta,
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    hits = [
        _hit(peers[0]["id"], "database mysql 8 and 16", row_id=1, distance=0.1, metadata=dict(meta)),
        _hit(peers[1]["id"], "database postgres 8 and 16", row_id=2, distance=0.2, metadata=dict(meta)),
    ]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: list(hits))
    _pass_cos_gate(monkeypatch)

    class BadOutput:
        @staticmethod
        def judge_pair(text_a, text_b):
            from memory_arbiter.semantic_judge import PairVerdict
            # both pairs fail with the SAME technical reason
            return PairVerdict("no_conflict", {}, None, "mdeberta:unavailable", error="bad output")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: BadOutput())

    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert result["status"] == "incomplete"
    assert result["reason"] == "judge_backend_error"
    assert result["reasons_seen"] == ["judge_backend_error"]
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




@pytest.mark.parametrize("stored_entity,stored_scope", [
    # Gate-v2 G3: slot identity is workspace + own subject now — both sides
    # build the canon form directly, so the old raw-vs-canon mismatch
    # (pre-canon legacy rows) has no metadata source to diverge from.
    ("default", "a"),
])

def _evidence_tools(tmp_path: Path, **overrides) -> MemoryTools:
    values = {
        "db_path": tmp_path / "memory.db",
        "backup_jsonl": tmp_path / "backup.jsonl",
    }
    values.update(overrides)
    return MemoryTools(Settings(**values))


def test_deterministic_routes_are_explainable() -> None:
    numeric = decide_evidence("PostgreSQL port is 5432", "PostgreSQL port is 3306")
    assert numeric.action == "check"
    assert numeric.reason == "numeric_value_candidate"
    assert decide_evidence("pgsql", "PostgreSQL").action == "ignore"
    assert decide_evidence("database connection pool", "database connection policy").action == "check"



def test_notice_dedupe_is_symmetric_and_version_pinned() -> None:
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_evidence") == notice_dedupe_key(
        2, 1, 4, 3, "semantic_evidence"
    )
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_evidence") != notice_dedupe_key(
        1, 2, 4, 4, "semantic_evidence"
    )


def test_notice_freshness_uses_only_memory_versions(tmp_path: Path) -> None:
    tools = _evidence_tools(tmp_path)
    left = tools.memory_write(content="left", subject="s", workspace="w")["data"]["id"]
    right = tools.memory_write(content="right", subject="s", workspace="w")["data"]["id"]
    created = tools.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal",
        notice_type="semantic_evidence", title="candidate", message="check",
        payload={}, left_version=1, right_version=1,
        dedupe_key=notice_dedupe_key(left, right, 1, 1, "semantic_evidence"),
    )
    notice = tools.db.read_semantic_notice(created["notice_id"])
    assert notice["freshness"]["fresh"] is True
    tools.memory_edit(left, new_content="changed", reason="new")
    notice = tools.db.read_semantic_notice(created["notice_id"])
    assert notice["freshness"]["fresh"] is False


def test_decimal_point_preserved_in_normalization() -> None:
    from memory_arbiter.semantic_conflict import normalize_value
    assert normalize_value("1.5s") != normalize_value("15s")
    # Integer/fractional pair stays distinct.
    assert normalize_value("1.5") != normalize_value("15")


# ── 2026-09-16 (eval cf-oppo-10/12): direction-dependent fillers ──────────

def test_normalize_value_strips_approximator_prefix_and_measure_ge() -> None:
    from memory_arbiter.semantic_conflict import normalize_value
    assert normalize_value("近 90 天") == normalize_value("90天")
    assert normalize_value("约 500 元") == normalize_value("500元")
    assert normalize_value("大约 3 小时") == normalize_value("3小时")
    assert normalize_value("5个工作日") == normalize_value("5工作日")
    # Real value differences survive the filler stripping.
    assert normalize_value("近 90 天") != normalize_value("180天")
    assert normalize_value("5个工作日") != normalize_value("72小时")


def test_workday_calendar_day_units_convert() -> None:
    from memory_arbiter.semantic_conflict import canonical_unit_value as cuv
    assert cuv("5个工作日") == cuv("5days") == "432000000ms"
    assert cuv("3自然日") == cuv("3日") == "259200000ms"
    assert cuv("5businessdays") == cuv("5calendardays") == "432000000ms"
    assert cuv("5workdays") == cuv("5workingdays") == "432000000ms"
    # 同值方向的行为变化是拍板口径：工作日按自然天 24h，3 工作日 = 72 小时。
    assert cuv("3个工作日") == cuv("72小时") == "259200000ms"


def test_cf_oppo_12_direct_path_lands_after_workday_unit() -> None:
    from memory_arbiter.semantic_conflict import (
        decide_evidence, direct_value_verdict, _normalized_values, _numeric_stripped_skeleton,
    )
    left, right = "对美专线清关承诺 72 小时内完成。", "对美专线清关承诺 5 个工作日内完成。"
    decision = decide_evidence(left, right)
    assert decision.reason == "numeric_value_candidate"
    assert decision.left_value == "259200000ms" and decision.right_value == "432000000ms"
    assert _numeric_stripped_skeleton(left) == _numeric_stripped_skeleton(right)
    hit = direct_value_verdict(left, right, decision)
    assert hit is not None and hit[1] == "259200000ms" and hit[2] == "432000000ms"


def test_unit_canonical_conversion() -> None:
    from memory_arbiter.semantic_conflict import canonical_unit_value, normalize_value
    # Exact physical relations convert to the canonical base (ms / kb / 元).
    assert normalize_value("0.5s") == normalize_value("500ms")
    assert normalize_value("72小时") == normalize_value("3天")
    assert normalize_value("1GB") == normalize_value("1024mb")
    assert normalize_value("90秒") == normalize_value("1.5分钟")
    assert normalize_value("1.5 hours") == normalize_value("90 minutes")
    assert normalize_value("5万") == "50000"
    assert normalize_value("500块") == normalize_value("500元")
    # Deliberately NOT converted: 工作日 (domain assumption), 月/年
    # (variable length). % carries no physical conversion — and note the
    # pre-existing _mechanical_normalize drops the sign, so "0.3%" reads as
    # the bare number (unchanged by this change).
    assert normalize_value("5个工作日") != normalize_value("40小时")
    assert normalize_value("3个月") != normalize_value("90天")
    assert normalize_value("0.3%") == "0.3"
    # Non-numeric shapes are identity.
    assert normalize_value("mysql") == "mysql"
    assert canonical_unit_value("") == ""


def test_numeric_channel_unit_equivalence_is_duplicate() -> None:
    """"500ms" vs "0.5s" canonicalises equal — a duplicate, never a conflict
    candidate; "500ms" vs "5s" stays a candidate."""
    from memory_arbiter.semantic_conflict import decide_evidence
    decision = decide_evidence("接口超时时间为 500ms。", "接口超时时间为 0.5s。")
    assert decision.action == "ignore"
    decision = decide_evidence("接口超时时间为 500ms。", "接口超时时间为 5s。")
    assert decision.action == "check" and decision.reason == "numeric_value_candidate"


# ── 2026-09-16 (owner): deterministic direct path ─────────────────────────

def test_direct_value_verdict_threshold_and_guards() -> None:
    from memory_arbiter.semantic_conflict import (
        EvidenceDecision, decide_evidence, direct_value_verdict,
    )
    # Same skeleton + single canonical value difference → verdict.
    decision = decide_evidence("接口超时时间为 500ms。", "接口超时时间为 200ms。")
    verdict = direct_value_verdict(
        "接口超时时间为 500ms。", "接口超时时间为 200ms。", decision, key_cosine=1.0,
    )
    assert verdict is not None and verdict[1] == "500ms" and verdict[2] == "200ms"
    assert verdict[0]  # non-empty attribute from the skeleton
    # Below the key threshold → back to Qwen (tested with DIFFERING
    # skeletons; identical skeletons shortcut at cosine 1.0 by design).
    differing = EvidenceDecision("check", "numeric_value_candidate", [], "500ms", "200ms")
    assert direct_value_verdict(
        "接口超时时间为 500ms。", "接口超时上限为 200ms。", differing, key_cosine=0.90,
    ) is None
    assert direct_value_verdict(
        "接口超时时间为 500ms。", "接口超时上限为 200ms。", differing, key_cosine=0.97,
    ) is not None
    # Canonically EQUAL values never reach the direct path (not a candidate).
    equal = decide_evidence("接口超时时间为 500ms。", "接口超时时间为 0.5s。")
    assert direct_value_verdict(
        "接口超时时间为 500ms。", "接口超时时间为 0.5s。", equal, key_cosine=1.0,
    ) is None
    # Version ordinals: evolution must not become a direct conflict.
    versioned = EvidenceDecision("check", "numeric_value_candidate", [], "500ms", "200ms")
    assert direct_value_verdict(
        "方案 v1 超时上限为 500ms。", "方案 v2 超时上限为 200ms。", versioned, key_cosine=1.0,
    ) is None
    # Evolution wording veto still applies on the direct path.
    evolving = decide_evidence("超时配置不再采用 500ms。", "超时配置改为 200ms。")
    if evolving.action == "check":
        assert direct_value_verdict(
            "超时配置不再采用 500ms。", "超时配置改为 200ms。", evolving, key_cosine=1.0,
        ) is None
    # Multi-value sentences are pairwise-ambiguous.
    multi = decide_evidence("连接池上限为 10，队列长度为 3。", "连接池上限为 99，队列长度为 5。")
    assert multi.left_value and ", " in multi.left_value
    assert direct_value_verdict(
        "连接池上限为 10，队列长度为 3。", "连接池上限为 99，队列长度为 5。", multi, key_cosine=1.0,
    ) is None


def test_english_pairs_rule_layer() -> None:
    """English evidence: numeric channel, identifier value-opposition, and
    the direct path all work language-agnostically."""
    from memory_arbiter.difference_classifier import classify_pair
    from memory_arbiter.semantic_conflict import decide_evidence, direct_value_verdict
    # Numeric channel with English units.
    decision = decide_evidence(
        "Refund requests over 5000 CNY require finance review.",
        "Refund requests over 500 CNY require finance review.",
    )
    assert decision.action == "check" and decision.reason == "numeric_value_candidate"
    assert classify_pair(
        "Refund requests over 5000 CNY require finance review.",
        "Refund requests over 500 CNY require finance review.",
        route=decision.reason,
    ) == "keep"
    # Identifier opposition (MySQL/PostgreSQL) via the value-opposition keep.
    left = "The production database uses MySQL with a dual-primary setup."
    right = "The production database uses PostgreSQL with a single primary."
    decision = decide_evidence(left, right)
    assert decision.action == "check"
    assert classify_pair(left, right, route=decision.reason) == "keep"
    # English numeric pair with identical skeleton → direct verdict.
    decision = decide_evidence(
        "The connection pool limit is 10.", "The connection pool limit is 99.",
    )
    verdict = direct_value_verdict(
        "The connection pool limit is 10.", "The connection pool limit is 99.",
        decision, key_cosine=1.0,
    )
    assert verdict is not None and verdict[1] == "10" and verdict[2] == "99"

