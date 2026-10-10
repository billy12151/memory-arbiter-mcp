"""0.17.2 possible 灰区门（owner 拍板 2026-10-10）行为测试。

钉死的契约：
- _judge_outcome：argmax=possible 且 P(conflict)+P(possible_conflict) ≥
  mdeberta_possible_min_susp → 'possible'（info 通知）；低于 →
  'possible_below_gate'（静默 + model_possible_below_gate 计数）；签名缺省
  0.0 = 0.17.2 前的无门行为（纯函数/遗留调用方）；
- 写时 cross 腿：低 susp 的 possible 对不建 conflicts 行、回执带计数；过门
  对 info 组照建（回归）；internal 腿与 backlog drain 共用同一判定函数，
  below-gate 均静默落地；
- 配置：默认 0.7、文件键 semantic_conflict.mdeberta_possible_min_susp、
  越界钳制告警；置 0 = 旧行为回滚位。

定标依据（v58_ep3，合并标定=四卷考+LoCoMo 金标）：甜区 [0.65, 0.73] 取
0.70——疑似命中 40/66 全保、no→疑似误报 5→2（0.571/0.648/0.648 三条被
杀，0.96/0.968 两条任何阈值都压不掉）。
"""
from __future__ import annotations

from pathlib import Path

import tests.test_config as tc
import tests.test_vnext_evidence as tv
from memory_arbiter.pipeline._evidence_judge import _judge_outcome
from memory_arbiter.semantic_judge import PairVerdict


def _pv(label: str, probs: dict) -> PairVerdict:
    return PairVerdict(label, probs, None, "mdeberta-v58:test")


# ── 纯函数层 ────────────────────────────────────────────────────────────────

def test_judge_outcome_possible_gate_boundaries() -> None:
    below = _pv("possible_conflict", {"conflict": 0.05, "no_conflict": 0.30, "possible_conflict": 0.60})
    at = _pv("possible_conflict", {"conflict": 0.05, "no_conflict": 0.25, "possible_conflict": 0.65})
    above = _pv("possible_conflict", {"conflict": 0.05, "no_conflict": 0.05, "possible_conflict": 0.90})
    # susp = P(conflict)+P(possible_conflict)：below=0.65 / at=0.70 / above=0.95
    assert _judge_outcome(below, 0.5, 0.7) == "possible_below_gate"
    assert _judge_outcome(at, 0.5, 0.7) == "possible"
    assert _judge_outcome(above, 0.5, 0.7) == "possible"
    # 签名缺省 0.0 = 0.17.2 前无门行为
    assert _judge_outcome(below, 0.5) == "possible"
    # P(conflict) 计入 susp（argmax=possible 但冲突质量足够 → 过门）
    conflict_heavy = _pv("possible_conflict", {"conflict": 0.45, "no_conflict": 0.05, "possible_conflict": 0.50})
    assert _judge_outcome(conflict_heavy, 0.5, 0.7) == "possible"
    # 其他出口不受门影响
    assert _judge_outcome(_pv("conflict", {"conflict": 0.9, "no_conflict": 0.05, "possible_conflict": 0.05}), 0.5, 0.7) == "notice"
    assert _judge_outcome(_pv("no_conflict", {"conflict": 0.05, "no_conflict": 0.9, "possible_conflict": 0.05}), 0.5, 0.7) == "clear"


# ── 写时管线（cross 腿）─────────────────────────────────────────────────────

def _seed_pair(tools, monkeypatch, verdict):
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    peer = tools.memory_write(content="database is mysql 8 here", subject="a", tags=[], metadata=meta)["data"]
    new = tools.memory_write(content="database is mysql 8 there", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "database is mysql 8 here",
             "start_offset": 0, "end_offset": 22, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: tv._hits_with_metadata(tools, list(hits)))
    tv._pass_cos_gate(monkeypatch)

    class Backend:
        @staticmethod
        def judge_pair(text_a, text_b):
            return verdict

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: Backend())
    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))
    return result, new["id"]


def test_write_time_below_gate_possible_is_silent_and_counted(tmp_path: Path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path)
    verdict = _pv("possible_conflict", {"conflict": 0.05, "no_conflict": 0.30, "possible_conflict": 0.60})
    result, new_id = _seed_pair(tools, monkeypatch, verdict)
    assert result["status"] == "completed"
    assert result["outcome"] == "checked_no_notice"
    assert result["model_possible_below_gate"] == 1
    assert not [n for n in tools.db.list_semantic_notices() if n["memory_id"] == new_id]
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conflicts WHERE notice_type IS NOT NULL").fetchone()[0] == 0


def test_write_time_above_gate_possible_still_lands_info(tmp_path: Path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path)
    verdict = _pv("possible_conflict", {"conflict": 0.05, "no_conflict": 0.05, "possible_conflict": 0.90})
    result, new_id = _seed_pair(tools, monkeypatch, verdict)
    assert result["outcome"] == "notices_created"
    assert result["model_possible_count"] == 1
    assert "model_possible_below_gate" not in result
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT notice_severity, detection_reason FROM conflicts WHERE notice_type IS NOT NULL"
        ).fetchone()
    assert row["notice_severity"] == "info"
    assert row["detection_reason"] == "model_classified_possible_conflict"


def test_gate_zero_restores_pre_0172_behavior(tmp_path: Path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_mdeberta_possible_min_susp = 0.0
    verdict = _pv("possible_conflict", {"conflict": 0.05, "no_conflict": 0.30, "possible_conflict": 0.60})
    result, _ = _seed_pair(tools, monkeypatch, verdict)
    assert result["outcome"] == "notices_created"
    assert result["model_possible_count"] == 1


# ── 配置链 ─────────────────────────────────────────────────────────────────

def test_config_default_file_key_and_clamp(tmp_path: Path, monkeypatch) -> None:
    tc.clear_config_env(monkeypatch)
    ckpt = tmp_path / "mdeberta-v58_ep3_fp16.pt"
    ckpt.write_bytes(b"x")

    base = {"db_path": str(tmp_path / "b.sqlite3"), "backup_jsonl": str(tmp_path / "b.jsonl"),
            "semantic_conflict": {"mdeberta_ckpt": str(ckpt)}}
    cfg = tmp_path / "config.json"
    cfg.write_text(__import__("json").dumps(base), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg))
    settings = __import__("memory_arbiter.config", fromlist=["Settings"]).Settings.from_env()
    assert settings.semantic_conflict_mdeberta_possible_min_susp == 0.7  # 定标甜区中值

    base["semantic_conflict"]["mdeberta_possible_min_susp"] = 0.9
    cfg.write_text(__import__("json").dumps(base), encoding="utf-8")
    settings = __import__("memory_arbiter.config", fromlist=["Settings"]).Settings.from_env()
    assert settings.semantic_conflict_mdeberta_possible_min_susp == 0.9

    base["semantic_conflict"]["mdeberta_possible_min_susp"] = 1.5
    cfg.write_text(__import__("json").dumps(base), encoding="utf-8")
    settings = __import__("memory_arbiter.config", fromlist=["Settings"]).Settings.from_env()
    assert settings.semantic_conflict_mdeberta_possible_min_susp == 1.0
    assert any("mdeberta_possible_min_susp" in w for w in settings.config_warnings)
