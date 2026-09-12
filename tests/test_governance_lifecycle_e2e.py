"""Governance lifecycle end-to-end acceptance tests (0.15.15, mema #970).

Release gate: the deadlock fix (D1-D3) must hold under REAL Qwen extraction —
realistic long values (not hand-picked short ones like "sqlite") recorded,
judged, applied, and resolved through the product API. The conflict #50
incident (25↔534, "twine upload dist/*" vs explicit filenames) dead-ended at
apply-side grounding precisely because every fast-test fixture was short.

Real-model variants run under @pytest.mark.slow with the module-scoped
real_backend fixture (same contract as test_write_time_notice_e2e): one model
load shared across tests, clean pytest.skip when the GGUF is absent, explicit
unload + del + gc teardown so the process exits 0 (owner rule 2026-09-11).

Chapters (plan doc mema-01515 §2 commit 2):
  1  real Qwen extracts a conflicting upload recipe pair; the extracted
     values become a formal conflict group through record_conflict
  2  judge with the extracted group value; apply update_current_claim
     (rewrite) then use_as_resolution; resolve — the #50 deadlock path,
     which must complete with a long paraphrase-shaped value
  3  recovery drill: update_current_claim whose edit misses the chosen
     value fails grounded; replan (re-edit containing the chosen value)
     recovers and resolves
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import ConflictMember
from memory_arbiter.semantic_conflict import normalize_value
from memory_arbiter.tools import MemoryTools

_SLOW_MODEL = Path(
    "~/.local/share/memory-arbiter/models/semantic-conflict/"
    "Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf"
).expanduser()

_META = {"entity": "governance-e2e", "scope": "release-pipeline"}


class FakeEmbedder:
    embedding_space_id = "fake-governance-space"
    dim = 2
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars: str | int | None = None) -> EmbedResult:
        text = f"{prefix}\n{body}".casefold()
        vector = [1.0, 0.0] if "dist" in text else [0.0, 1.0]
        return EmbedResult(vector, False, len(text), len(text))


def _make_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "governance.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
        embedding_auto_write=True,
        embedding_auto_query=True,
        semantic_conflict_on_write="off",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = FakeEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(FakeEmbedder.dim) == []
    tools.db.init_vec_index_state(FakeEmbedder.embedding_space_id, True, active_dim=FakeEmbedder.dim)
    return tools


@pytest.fixture(scope="module")
def real_backend() -> "Any":
    """Load the GGUF model once per module; skip cleanly when absent.

    Teardown contract (owner rule, 2026-09-11): unload, drop the reference,
    gc — interpreter-exit finalization SIGABRTs in ggml_metal_device_free.
    """
    if not _SLOW_MODEL.exists():
        pytest.skip(f"real model not installed at {_SLOW_MODEL}")
    from memory_arbiter.constants import SEMANTIC_N_CTX
    from memory_arbiter.semantic_conflict import LocalGGUFSemanticBackend

    backend = LocalGGUFSemanticBackend(_SLOW_MODEL, n_ctx=SEMANTIC_N_CTX, n_threads=4, n_batch=128)
    backend.load()
    yield backend
    try:
        backend.unload()
    except Exception:
        pass
    del backend
    gc.collect()


_RELEASE_LEFT = "发版流程：mema-core 发版第 7 步使用 twine upload dist/* 一次性上传全部构建产物。"
_RELEASE_RIGHT = "发版流程：mema-core 发版经验是 twine 上传必须使用显式文件名，禁止 dist/* 通配，避免重传旧版本被 PyPI 拒绝。"


def _env(content: str, memory_id: int) -> dict[str, Any]:
    return {
        "quote": content[:400], "subject": "发版流程", "tags": ["release"],
        "workspace_canonical": "governance-e2e", "memory_id": memory_id, "version": 1,
        "event_time": None, "metadata": dict(_META),
    }


def _real_extraction(backend: Any, left: str, right: str, left_id: int, right_id: int) -> tuple[str, str]:
    """Real Qwen forward+reverse extraction; both directions must be valid
    attribute-value extractions with differing values (this pair is a genuine
    same-attribute contradiction)."""
    from memory_arbiter.semantic_conflict import signal_extraction

    forward = backend.classify_pair(_env(left, left_id), _env(right, right_id))
    reverse = backend.classify_pair(_env(right, right_id), _env(left, left_id))
    parsed_f = signal_extraction(forward)
    parsed_r = signal_extraction(reverse)
    assert parsed_f is not None and parsed_r is not None, (forward.candidate_type, reverse.candidate_type)
    assert forward.candidate_type not in {"invalid_json", "invalid_schema"}, getattr(forward, "raw", "")[:200]
    assert reverse.candidate_type not in {"invalid_json", "invalid_schema"}, getattr(reverse, "raw", "")[:200]
    value_left = str(parsed_f.value_a)
    value_right = str(parsed_f.value_b)
    assert value_left and value_right and value_left != value_right, parsed_f
    return value_left, value_right


def _record_group(
    tools: MemoryTools, left_id: int, right_id: int,
    value_left: str, value_right: str,
) -> int:
    """Record the formal group with the REAL extracted values, mechanically
    normalized (the D1 intake contract)."""
    def member(memory_id: int, value: str, quote: str) -> dict[str, Any]:
        return ConflictMember(
            memory_id=memory_id, version=1, attribute_raw="上传方式",
            value_raw=value, normalized_attribute="上传方式",
            normalized_value=normalize_value(value), evidence_quote=quote,
            evidence_span=(0, len(quote)),
            content_hash="0" * 64, direction="a_to_b", prompt_version="pair-v6",
            detector_version="attribute-value-v1",
        ).to_dict()

    created = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "governance-e2e", "attribute": "上传方式", "scope": "release"},
        "members": [member(left_id, value_left, _RELEASE_LEFT), member(right_id, value_right, _RELEASE_RIGHT)],
        "value_groups": [
            {"normalized_value": normalize_value(value_left), "display_value": value_left,
             "members": [f"{left_id}@1"]},
            {"normalized_value": normalize_value(value_right), "display_value": value_right,
             "members": [f"{right_id}@1"]},
        ],
        "detector_version": "attribute-value-v1", "prompt_version": "pair-v6",
        "source": "scheduled_scan", "reason": "同一发版上传步骤的配方冲突",
        "status": "open", "workspace": "governance-e2e",
    })
    assert created["ok"] is True, created["data"]
    return int(created["data"]["conflict_id"])


def _govern(tools: MemoryTools, action: str, data: dict[str, Any]) -> dict[str, Any]:
    result = tools.memory_govern(action, data)
    assert result["ok"] is True, result["data"]
    return result["data"]


def _release_real_models(tools: MemoryTools) -> None:
    """Teardown contract (owner rule, 2026-09-11): release the embedder
    explicitly; del + gc so the process exits 0."""
    try:
        embedder = getattr(tools, "_embedder", None)
        if embedder is not None:
            embedder.close()
    except Exception:
        pass
    tools._embedder = None
    tools._embedder_loaded = False
    gc.collect()


@pytest.mark.slow
def test_slow_governance_lifecycle_e2e_real_models(tmp_path: Path, real_backend: "Any") -> None:
    """Chapters 1+2: real extraction → group → judge → apply → resolve.

    The chosen value is a long normalized phrase (the #50 shape); the
    use_as_resolution step must complete without content grounding and the
    group must resolve. Pre-0.15.15 this exact flow dead-ended at
    chosen_value_not_grounded with no recovery.
    """
    tools = _make_tools(tmp_path)
    try:
        left = tools.memory_write(
            content=_RELEASE_LEFT, subject="发版流程", tags=["release"],
            metadata=dict(_META), workspace="governance-e2e",
        )["data"]
        right = tools.memory_write(
            content=_RELEASE_RIGHT, subject="发版流程", tags=["release"],
            metadata=dict(_META), workspace="governance-e2e",
        )["data"]
        assert tools.wait_evidence_worker_drained(timeout=5)

        value_left, value_right = _real_extraction(
            real_backend, _RELEASE_LEFT, _RELEASE_RIGHT, int(left["id"]), int(right["id"]),
        )
        conflict_id = _record_group(tools, int(left["id"]), int(right["id"]), value_left, value_right)

        chosen = normalize_value(value_right)
        judged = tools.memory("judge", {
            "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": chosen,
            "decided_by": "user", "ref": "release-e2e", "reason": "显式文件名是现行权威",
            "resolution_memory_id": int(right["id"]),
            "apply_plan": [
                {"memory_id": int(left["id"]), "action": "update_current_claim"},
                {"memory_id": int(right["id"]), "action": "use_as_resolution"},
            ],
            "authorized": True,
        })
        assert judged["ok"] is True, judged["data"]
        revision = int(judged["data"]["revision"])

        # update_current_claim: rewrite the wrong recipe to carry the chosen
        # value verbatim (the grounded edit path).
        rewrite = _govern(tools, "apply_conflict_action", {
            "conflict_id": conflict_id, "expected_revision": revision,
            "memory_id": int(left["id"]), "action": "update_current_claim", "authorized": True,
            "content": f"发版流程：mema-core 发版第 7 步按 {value_right} 上传（旧配方 twine upload dist/* 已废止）。",
        })
        assert rewrite["outcome"] == "completed"
        revision = int(rewrite["revision"])

        marker = _govern(tools, "apply_conflict_action", {
            "conflict_id": conflict_id, "expected_revision": revision,
            "memory_id": int(right["id"]), "action": "use_as_resolution", "authorized": True,
        })
        assert marker["outcome"] == "completed"

        resolved = _govern(tools, "resolve_conflict", {
            "conflict_id": conflict_id, "expected_revision": int(marker["revision"]),
            "reason": "e2e 完成", "authorized": True,
        })
        assert resolved["outcome"] == "resolved"
    finally:
        _release_real_models(tools)


@pytest.mark.slow
def test_slow_governance_recovery_e2e_real_models(tmp_path: Path, real_backend: "Any") -> None:
    """Chapter 3: the grounding-failure recovery drill.

    update_current_claim (the one action that still grounds) fails when the
    edit misses the chosen value; the failure names the replan recovery, and
    a replan with a re-edited plan completes and resolves. The failure is
    manufactured with update_current_claim per the plan doc — use_as_resolution
    no longer grounds (D2), so it cannot produce this error any more.
    """
    tools = _make_tools(tmp_path)
    try:
        left = tools.memory_write(
            content=_RELEASE_LEFT, subject="发版流程", tags=["release"],
            metadata=dict(_META), workspace="governance-e2e",
        )["data"]
        right = tools.memory_write(
            content=_RELEASE_RIGHT, subject="发版流程", tags=["release"],
            metadata=dict(_META), workspace="governance-e2e",
        )["data"]
        assert tools.wait_evidence_worker_drained(timeout=5)

        value_left, value_right = _real_extraction(
            real_backend, _RELEASE_LEFT, _RELEASE_RIGHT, int(left["id"]), int(right["id"]),
        )
        conflict_id = _record_group(tools, int(left["id"]), int(right["id"]), value_left, value_right)
        chosen = normalize_value(value_right)

        judged = tools.memory("judge", {
            "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": chosen,
            "decided_by": "user", "ref": "release-e2e", "reason": "恢复演练",
            "resolution_memory_id": int(right["id"]),
            "apply_plan": [{"memory_id": int(left["id"]), "action": "update_current_claim"}],
            "authorized": True,
        })
        assert judged["ok"] is True, judged["data"]
        revision = int(judged["data"]["revision"])

        # The failing edit: a real edit that does NOT contain the chosen value.
        failed = tools.memory_govern("apply_conflict_action", {
            "conflict_id": conflict_id, "expected_revision": revision,
            "memory_id": int(left["id"]), "action": "update_current_claim", "authorized": True,
            "content": "发版流程：上传方式待定，尚未裁决。",
        })["data"]
        assert failed["outcome"] == "apply_failed", failed
        assert failed["apply_summary"]["plan"][0]["error"] == "chosen_value_not_grounded"
        assert failed["action_required"] == "replan_conflict"
        assert "chosen_value" in failed["note"]

        # Recovery: re-edit so the chosen value is grounded, via replan.
        replanned = _govern(tools, "replan_conflict", {
            "conflict_id": conflict_id, "expected_revision": int(failed["revision"]),
            "apply_plan": [{"memory_id": int(left["id"]), "action": "update_current_claim"}],
            "authorized": True,
        })
        assert replanned["outcome"] == "replanned"
        applied = _govern(tools, "apply_conflict_action", {
            "conflict_id": conflict_id, "expected_revision": int(replanned["revision"]),
            "memory_id": int(left["id"]), "action": "update_current_claim", "authorized": True,
            "content": f"发版流程：mema-core 发版第 7 步按 {value_right} 上传（已裁决）。",
        })
        assert applied["outcome"] == "completed"
        resolved = _govern(tools, "resolve_conflict", {
            "conflict_id": conflict_id, "expected_revision": int(applied["revision"]),
            "reason": "恢复完成", "authorized": True,
        })
        assert resolved["outcome"] == "resolved"
    finally:
        _release_real_models(tools)
