"""写入时 workspace 归一——真模型慢测试（owner 四条验收，2026-10-02）.

快测试（test_write_workspace_canonicalization.py）用字符直方图假 embedder
钉决策分支的字面行为；本文件换真 embedder（embeddinggemma-300m Q8_0 常驻
GGUF）走完整 Settings → MemoryTools 写入路径，钉四条端到端契约：

  1. notice 提示：新桶注册的 workspace_review notice 带完整 review/confirm
     调用指引到达响应 notices（用户/agent 可直接消费）；
  2. 近似值直接归一：真余弦距离 ≤ WORKSPACE_MATCH_DISTANCE(0.25) 的近邻名
     AUTO 折桶（vector_strong），不打扰人（钉对：金营项目/金科营销项目，
     resolver docstring 自带例，实测 d≈0.158）；
  3. Agent 裁决闭环：generic 名 ASK → write_hints.workspace_review 给出
     migrate_workspace / keep_separate 双选项 → agent 按 hint 调
     memory_govern(migrate_workspace) → confirmed alias 落库 → 后续写入跟随裁决；
  4. mema-twin 内部标识不被污染：twin 身份建立桶后，非 twin 调用方的 exact
     与机械变体写入全部改道 mema-twin-dev（protected_bucket_redirect
     notice），mema-twin 桶内永远只有 twin 自己的行。

CI 无 GGUF 时 skip（与 test_scan_capability_e2e 同约定）；teardown 遵守
owner 卸载纪律（close + del + gc），否则 pytest 全绿但 exit 134。
"""
from __future__ import annotations

import gc
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.tools import MemoryTools  # noqa: E402

_EMBED_MODEL = Path(
    "~/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf"
).expanduser()


def _make_real_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    if not _EMBED_MODEL.exists():
        pytest.skip(f"real embedding model not installed at {_EMBED_MODEL}")
    settings = Settings(
        db_path=tmp_path / "ws-slow.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=_EMBED_MODEL,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


@pytest.fixture
def real_tools(tmp_path: Path) -> Iterator[MemoryTools]:
    tools = _make_real_tools(tmp_path)
    yield tools
    # 真模型卸载纪律：不显式释放则 pytest 全绿但进程 exit 134（Metal 退出崩溃）。
    backend = getattr(tools, "_semantic_backend", None)
    if backend is not None:
        try:
            backend.unload(timeout=10.0)
        except Exception:
            pass
    tools._semantic_backend = None
    embedder = getattr(tools, "_embedder", None)
    if embedder is not None:
        embedder.close()
    tools._embedder = None
    tools._embedder_loaded = False
    del tools
    gc.collect()


def _rows(tools: MemoryTools) -> list[tuple[str, str]]:
    with tools.db.connection() as conn:
        return [
            (str(r["workspace"]), str(r["workspace_canonical"]))
            for r in conn.execute(
                "SELECT workspace, workspace_canonical FROM memories ORDER BY id"
            )
        ]


def _registered_canonicals(tools: MemoryTools) -> set[str]:
    with tools.db.connection() as conn:
        return {
            str(r["name"]) for r in conn.execute("SELECT name FROM workspace_canonicals")
        }


# ── 1. notice 提示给用户：新桶注册 notice 带可执行调用 ──────────────────────


def test_slow_new_workspace_notice_reaches_user(real_tools: MemoryTools) -> None:
    result = real_tools.memory_write(
        content="网关端口配置说明，超时 30 秒。",
        subject="s-billing", tags=[], workspace="billing-portal-zx9",
    )
    assert result["ok"], result
    data = result["data"]
    assert data["workspace_matched_by"] == "new"
    assert data["workspace_canonical"] == "billing-portal-zx9"
    assert data["workspace_decision"] == "AUTO"
    assert data["workspace_decision_reason"] == "new_specific_canonical"
    hints = data.get("write_hints") or {}
    assert hints["new_workspace_detected"]["canonical"] == "billing-portal-zx9"
    notice = next(
        n for n in result.get("notices") or [] if n.get("type") == "workspace_review"
    )
    assert notice["severity"] == "info"
    assert notice["workspace"] == "billing-portal-zx9"
    assert notice["action_required"] == "review_workspace_registry"
    assert notice["review_call"]["tool"] == "memory_review"
    assert notice["review_call"]["view"] == "doctor"
    assert notice["confirm_call"]["tool"] == "memory_govern"
    assert notice["confirm_call"]["action"] == "confirm_workspaces"
    assert notice["authorization_required"] is True


# ── 2. 近似值直接归一：真向量近邻 AUTO 折桶，不问人 ─────────────────────────


def test_slow_near_neighbor_name_folds_via_real_vectors(real_tools: MemoryTools) -> None:
    first = real_tools.memory_write(
        content="金营项目的渠道对账流程。",
        subject="s-jy", tags=[], workspace="金营项目",
    )
    assert first["ok"], first
    assert first["data"]["workspace_canonical"] == "金营项目"

    second = real_tools.memory_write(
        content="金科营销项目的渠道对账流程。",
        subject="s-jy2", tags=[], workspace="金科营销项目",
    )
    assert second["ok"], second
    data = second["data"]
    assert data["workspace_matched_by"] == "vector"
    assert data["workspace_canonical"] == "金营项目"
    assert data["workspace_decision"] == "AUTO"
    assert data["workspace_decision_reason"] == "vector_strong"
    assert "workspace_review" not in (data.get("write_hints") or {})
    assert _rows(real_tools)[1] == ("金科营销项目", "金营项目")  # raw 原样保留


# ── 3. Agent 裁决闭环：ASK → hint 双选项 → migrate → confirmed alias 落库 ───


def test_slow_ask_lets_agent_adjudicate_and_decision_sticks(
    real_tools: MemoryTools,
) -> None:
    first = real_tools.memory_write(
        content="apisvc 网关超时 30 秒。",
        subject="s-apisvc", tags=[], workspace="apisvc",
    )
    assert first["ok"], first
    assert first["data"]["workspace_canonical"] == "apisvc"

    ask = real_tools.memory_write(
        content="本周渠道对账进度记录。",
        subject="s-weekly", tags=[], workspace="周报",  # GENERIC_TERMS → 低信号
    )
    assert ask["ok"], ask
    data = ask["data"]
    assert data["workspace_decision"] == "ASK"
    assert data["workspace_decision_reason"] == "low_signal_generic"
    review = (data.get("write_hints") or {}).get("workspace_review")
    assert review is not None, data
    assert review["raw"] == "周报"
    assert [s["name"] for s in review["similar_workspaces"]] == ["apisvc"]
    merge_option, keep_option = review["options"]
    assert merge_option["decision"] == "merge"
    assert merge_option["call"]["tool"] == "memory_govern"
    assert merge_option["call"]["action"] == "migrate_workspace"
    assert merge_option["call"]["data"] == {"from": "周报", "to": "apisvc"}
    assert merge_option["authorization_required"] is True
    assert keep_option["decision"] == "keep_separate"
    assert "Merge only after user confirmation" in review["note"]
    # ASK 不静默合并：行先落自己的桶，等人裁决。
    assert _rows(real_tools)[1] == ("周报", "周报")

    # Agent 裁决：按 hint 给出的调用执行合并。
    merged = real_tools.memory_govern("migrate_workspace", {
        "workspace": "default",
        "from": "周报", "to": "apisvc",
        "reason": "agent adjudication: weekly rollup belongs to apisvc",
        "authorized": True,
    })
    assert merged["ok"], merged
    assert merged["data"]["migrated"] is True
    assert merged["data"]["memories_updated"] >= 1
    assert _rows(real_tools)[1] == ("周报", "apisvc")

    # 裁决持久化：后续同一 raw 写入走 confirmed alias AUTO，不再问。
    third = real_tools.memory_write(
        content="又一周的对账进度记录。",
        subject="s-weekly2", tags=[], workspace="周报",
    )
    assert third["ok"], third
    tdata = third["data"]
    assert tdata["workspace_matched_by"] == "confirmed_alias"
    assert tdata["workspace_canonical"] == "apisvc"
    assert tdata["workspace_decision"] == "AUTO"
    assert tdata["workspace_decision_reason"] == "confirmed_alias"
    assert _rows(real_tools)[2] == ("周报", "apisvc")


# ── 4. mema-twin 内部标识不被污染 ───────────────────────────────────────────


def test_slow_twin_bucket_not_polluted_by_other_callers(
    real_tools: MemoryTools,
) -> None:
    from memory_arbiter.request_identity import (
        RequestIdentity,
        request_identity_scope,
    )

    # twin 身份先建桶：mema-twin 归 twin 自己。
    with request_identity_scope(
        RequestIdentity(client="mema-twin", agent_id="mema-twin")
    ):
        twin = real_tools.memory_write(
            content="twin persona 沉淀。",
            subject="s-twin", tags=[], workspace="mema-twin",
        )
    assert twin["ok"], twin
    assert twin["data"]["workspace_matched_by"] == "new"
    assert twin["data"]["workspace_canonical"] == "mema-twin"

    # 非 twin 调用方 exact 同名 → 改道 mema-twin-dev + redirect notice。
    other = real_tools.memory_write(
        content="别人的记忆想进 twin 桶。",
        subject="s-other", tags=[], workspace="mema-twin",
    )
    assert other["ok"], other
    odata = other["data"]
    assert odata["workspace_matched_by"] == "twin_redirect"
    assert odata["workspace_canonical"] == "mema-twin-dev"
    redirect = next(
        n for n in other.get("notices") or []
        if n.get("type") == "protected_bucket_redirect"
    )
    assert redirect["requested_workspace"] == "mema-twin"
    assert redirect["workspace"] == "mema-twin-dev"
    assert _rows(real_tools)[1] == ("mema-twin", "mema-twin-dev")

    # 机械变体拼写（大小写/分隔符折到 mema-twin）同样吃改道。
    variant = real_tools.memory_write(
        content="变体拼写的记忆。",
        subject="s-variant", tags=[], workspace="Mema_Twin",
    )
    assert variant["ok"], variant
    vdata = variant["data"]
    assert vdata["workspace_matched_by"] == "twin_redirect"
    assert vdata["workspace_canonical"] == "mema-twin-dev"
    assert _rows(real_tools)[2] == ("Mema_Twin", "mema-twin-dev")

    # mema-twin 桶内只有 twin 自己的行；其余全在 dev。
    rows = _rows(real_tools)
    in_twin = [r for r in rows if r[1] == "mema-twin"]
    assert len(in_twin) == 1, rows
    assert _registered_canonicals(real_tools) == {"mema-twin", "mema-twin-dev"}
