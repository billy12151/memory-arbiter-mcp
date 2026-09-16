"""Write-time duplicate hint: subject gate + content-confirmation gate
(owner redesign 2026-09-16; supersedes the 2026-09-02 subject+tags double
gate). Deterministic, model-free, info notice with an agent-first triage
instruction; deliberate series entries must stay quiet.
"""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    return MemoryTools(settings, MemoryDB(settings))


BODY_A = (
    "金营平台生产部署采用双可用区架构，发布走灰度批次，先切百分之五流量观察"
    "三十分钟再全量。回滚保留上一版本镜像，由值班工程师执行。"
)
BODY_A_REWRITE = (
    "金营平台生产部署采用双可用区架构，发布走灰度批次，先切百分之五流量观察"
    "三十分钟再全量推进。回滚时保留上一版本的镜像，由值班工程师负责执行。"
)
BODY_SERIAL = (
    "第三轮决策记录：用户确认归一策略改为保守档，高风险候选一律转人工复核，"
    "本周内完成灰度验证后同步到配置模板。"
)


def _write(tools: MemoryTools, subject: str, tags: list[str], workspace: str = "w", content: str = "body"):
    return tools.memory_write(content=content, subject=subject, tags=tags, workspace=workspace)


def _similar_notices(result: dict) -> list[dict]:
    return [n for n in result.get("notices") or [] if n.get("type") == "similar_active_memory"]


def test_duplicate_subject_and_body_fires_hint_with_content_cosine(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = _write(tools, "部署方案：金营项目上线流程", ["deployment", "金营"], content=BODY_A)
    assert _similar_notices(first) == []
    second = _write(
        tools, "部署方案：金营项目上线流程（终版）", ["deployment", "金营"], content=BODY_A_REWRITE,
    )
    hints = _similar_notices(second)
    assert len(hints) == 1
    match = hints[0]["matches"][0]
    assert match["memory_id"] == first["data"]["id"]
    assert match["subject_similarity"] >= 0.8
    assert match["content_cosine"] is not None and match["content_cosine"] >= 0.4
    assert match["low_confidence"] is False
    assert "Triage silently" in hints[0]["agent_instruction"]
    assert second["ok"] is True


def test_same_subject_dissimilar_body_stays_quiet(tmp_path: Path) -> None:
    """Same-subject serials (successive decision records) are the real
    library's dominant subject-similar shape — the content gate keeps
    them quiet even with identical tags."""
    tools = make_tools(tmp_path)
    _write(tools, "金营平台部署纪要", ["deployment", "金营"], content=BODY_A)
    serial = _write(tools, "金营平台部署纪要", ["deployment", "金营"], content=BODY_SERIAL)
    assert _similar_notices(serial) == []


def test_disjoint_tags_similar_body_fires(tmp_path: Path) -> None:
    """Tags no longer gate the hint (2026-09-16): cross-habit duplicate
    rewrites whose tags drifted must fire on subject + body alone."""
    tools = make_tools(tmp_path)
    first = _write(tools, "金营平台部署方案", ["deployment", "金营"], content=BODY_A)
    dup = _write(tools, "金营平台部署方案", ["ops", "上线"], content=BODY_A_REWRITE)
    hints = _similar_notices(dup)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["data"]["id"]


def test_short_body_skips_confirmation_as_low_confidence(tmp_path: Path) -> None:
    """Bodies under WRITE_SIMILAR_MIN_CONTENT_CHARS skip the content gate
    and hint anyway, flagged low_confidence."""
    tools = make_tools(tmp_path)
    first = _write(tools, "API token 轮换流程", [], content="rotate quarterly")
    dup = _write(tools, "API token 轮换流程", [], content="rotate every quarter")
    hints = _similar_notices(dup)
    assert len(hints) == 1
    match = hints[0]["matches"][0]
    assert match["low_confidence"] is True
    assert match["content_cosine"] is None


def test_series_entries_stay_quiet(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "mema-core Tier 1 功能方案：治理化 merge / 定时任务引导", ["mema-core", "roadmap", "plan"])
    series = _write(
        tools,
        "mema-core Tier 2 功能候选：访问信号排序 / 遗忘归档生命周期",
        ["mema-core", "roadmap", "plan"],
    )
    assert _similar_notices(series) == [], "deliberate series entries must not trigger the hint"


def test_tag_overlap_alone_does_not_fire(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "数据库选型记录", ["mema", "infra"])
    other = _write(tools, "完全不同主题的会议纪要", ["mema", "infra"])
    assert _similar_notices(other) == []


def test_no_tags_on_either_side_subject_decides(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = _write(tools, "API token 轮换流程", [])
    # 0.16.6: byte-identical content is replayed idempotently by the dedup
    # gate and never reaches the hint; a near-duplicate body keeps the
    # "subject decides" scenario intact.
    dup = _write(tools, "API token 轮换流程", [], content="body, second entry")
    hints = _similar_notices(dup)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["data"]["id"]


def test_cross_workspace_and_superseded_stay_quiet(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = _write(tools, "同名部署方案", ["ops"], workspace="w")
    # Different workspace: no leakage.
    cross = _write(tools, "同名部署方案", ["ops"], workspace="other")
    assert _similar_notices(cross) == []
    # Superseded originals no longer count as active duplicates.
    tools.memory_supersede(memory_id=first["data"]["id"], reason="gone", authorized=True)
    after = _write(tools, "同名部署方案", ["ops"])
    assert _similar_notices(after) == []


def test_digit_run_series_entries_stay_quiet(tmp_path: Path) -> None:
    """Version-number series (subjects identical modulo digit runs) are not
    duplicates: "项目 0.15.1 发版清单" vs "项目 0.15.2 发版清单" must not fire
    even with identical tags (raw ratio measured 0.9524 >= 0.95)."""
    tools = make_tools(tmp_path)
    _write(tools, "项目 0.15.1 发版清单", ["mema-core", "release"])
    series = _write(tools, "项目 0.15.2 发版清单", ["mema-core", "release"])
    assert _similar_notices(series) == [], "digit-run series entries must not trigger the hint"


def test_exact_digit_subject_duplicate_still_fires(tmp_path: Path) -> None:
    """The series suppression only applies when the subjects differ: writing
    the exact same version-numbered subject twice is a true duplicate."""
    tools = make_tools(tmp_path)
    first = _write(tools, "项目 0.15.1 发版清单", ["mema-core", "release"])
    dup = _write(tools, "项目 0.15.1 发版清单", ["mema-core", "release"], content="same checklist again")
    hints = _similar_notices(dup)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["data"]["id"]


def _make_strict_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "strict.sqlite3",
        backup_jsonl=tmp_path / "strict.jsonl",
        workspace="projA",
        isolation="strict",
    )
    return MemoryTools(settings, MemoryDB(settings))


def _confirm_pending(tools: MemoryTools, memory_id: int) -> None:
    record = tools.db.get_memory(memory_id)
    if record["status"] != "pending":
        return
    confirmed = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id,
        "canonical": record["workspace_canonical"] or record["workspace"],
        "authorized": True,
    })
    assert confirmed["ok"] is True, confirmed


def test_strict_confirm_pending_workspace_adds_similar_active_notice(tmp_path: Path) -> None:
    """Strict-mode pending memories are activated by confirm/activate; the
    write-time duplicate hint must still fire once a second active duplicate
    exists in the same workspace."""
    tools = _make_strict_tools(tmp_path)
    # Both writes target the same *unconfirmed* workspace, so both are pending.
    first = tools.memory_write(
        content="body one", subject="proj onboarding checklist", tags=["ops"],
        workspace="projA",
    )
    second = tools.memory_write(
        content="body two", subject="proj onboarding checklist", tags=["ops"],
        workspace="projA",
    )
    assert first["data"]["record"]["status"] == "pending"
    assert second["data"]["record"]["status"] == "pending"

    first_confirmed = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": first["data"]["id"],
        "canonical": "projA",
        "authorized": True,
    })
    assert first_confirmed["ok"] is True
    assert _similar_notices(first_confirmed) == [], "first active memory has no peers"

    second_confirmed = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": second["data"]["id"],
        "canonical": "projA",
        "authorized": True,
    })
    assert second_confirmed["ok"] is True
    hints = _similar_notices(second_confirmed)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["data"]["id"]


def test_strict_memory_activate_adds_similar_active_notice(tmp_path: Path) -> None:
    tools = _make_strict_tools(tmp_path)
    first = tools.memory_write(
        content="body one", subject="release checklist", tags=["release"],
        workspace="projA",
    )
    assert first["data"]["record"]["status"] == "pending"
    _confirm_pending(tools, first["data"]["id"])

    # Workspace is now confirmed; an explicit pending write can be activated
    # via memory_activate, and must still receive the duplicate hint.
    second = tools.memory_write(
        content="body two", subject="release checklist", tags=["release"],
        workspace="projA", status="pending",
    )
    assert second["data"]["record"]["status"] == "pending"

    activated_second = tools.memory_activate(second["data"]["id"], authorized=True)
    assert activated_second["ok"] is True, activated_second
    hints = _similar_notices(activated_second)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["data"]["id"]
