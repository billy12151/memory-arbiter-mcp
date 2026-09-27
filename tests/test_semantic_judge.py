"""semantic_judge (0.17.1) unit tests: contract pinning, verdict shaping,
supervisor lifecycle with a fake child, and the crash breaker. No real model
load (torch/transformers imports inside the child are stubbed by the fake
process target) — real-model behaviour is covered by the E-gates offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.constants import SEMANTIC_MDEBERTA_LABELS
from memory_arbiter.semantic_judge import (
    IsolatedMDeBERTaBackend,
    PairVerdict,
    checkpoint_sha8,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_child_target(conn, config):  # pragma: no cover - runs in child process
    """Module-level so spawn can pickle it; behavior rides config['behavior']."""
    behavior = config.get("behavior")
    try:
        request = conn.recv()
        if behavior == "die_on_load" and request.get("command") == "load":
            conn.close()
            return
        if request.get("command") == "load":
            labels = list(config["labels"])
            if behavior == "wrong_labels":
                labels = ["no_conflict", "conflict", "possible_conflict"]
            conn.send({
                "ok": True,
                "result": {
                    "loaded": True, "labels": labels, "mechs": config["mechs"],
                    "model_version": f"mdeberta-v4m:{config['sha8']}",
                },
            })
        while True:
            request = conn.recv()
            if request.get("command") == "shutdown":
                return
            if behavior == "die_every_request":
                conn.close()
                return
            pairs = request["pairs"]
            results = []
            for _ in pairs:
                results.append({
                    "probs": {
                        "conflict": 0.91, "no_conflict": 0.06,
                        "possible_conflict": 0.03,
                    },
                    "mechanism": "numeric_value",
                })
            conn.send({"ok": True, "result": results})
    except (EOFError, BrokenPipeError, OSError):
        return


def _backend(tmp_path: Path, behavior: str = "ok") -> IsolatedMDeBERTaBackend:
    ckpt = tmp_path / "mdeberta-v4m.pt"
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    model_dir = tmp_path / "mdeberta-base"
    model_dir.mkdir(exist_ok=True)
    return IsolatedMDeBERTaBackend(
        ckpt, model_dir,
        batch_size=4, hard_timeout_ms=2000, load_timeout_ms=2000,
        process_target=_fake_child_target,
        child_config_extra={"behavior": behavior},
    )


# ── checkpoint identity ──────────────────────────────────────────────────────

def test_checkpoint_sha8_stable_and_sensitive(tmp_path: Path) -> None:
    a = tmp_path / "a.pt"
    a.write_bytes(b"alpha")
    b = tmp_path / "b.pt"
    b.write_bytes(b"beta")
    assert checkpoint_sha8(a) == checkpoint_sha8(a)
    assert checkpoint_sha8(a) != checkpoint_sha8(b)
    assert len(checkpoint_sha8(a)) == 8


# ── verdict shaping ──────────────────────────────────────────────────────────

def test_judge_pairs_batch_shapes_verdicts(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.load()
    verdicts = backend.judge_pairs([("端口 8080", "端口 9090"), ("甲", "乙")])
    assert len(verdicts) == 2
    for v in verdicts:
        assert isinstance(v, PairVerdict)
        assert v.label == "conflict"
        assert set(v.probs) == set(SEMANTIC_MDEBERTA_LABELS)
        assert v.mechanism == "numeric_value"
        assert v.model_version.startswith("mdeberta-v4m:")
        assert v.error is None


def test_judge_pair_single_is_batch_of_one(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.load()
    v = backend.judge_pair("a", "b")
    assert v.label == "conflict"


def test_judge_pairs_empty_input_noop(tmp_path: Path) -> None:
    assert _backend(tmp_path).judge_pairs([]) == []


def test_status_reports_identity(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.load()
    status = backend.status()
    assert status["device"] == "cpu"
    assert status["ckpt_sha8"] == checkpoint_sha8(tmp_path / "mdeberta-v4m.pt")
    assert status["labels"] == list(SEMANTIC_MDEBERTA_LABELS)
    assert status["disabled"] is False


# ── contract pinning ─────────────────────────────────────────────────────────

def test_label_contract_mismatch_disables_backend(tmp_path: Path) -> None:
    backend = _backend(tmp_path, behavior="wrong_labels")
    with pytest.raises(RuntimeError, match="label contract mismatch"):
        backend.load()
    # refused to serve: judge_pairs fails closed with error verdicts
    verdicts = backend.judge_pairs([("a", "b")])
    assert verdicts[0].error is not None
    assert backend.status()["disabled"] is False  # contract mismatch ≠ crash breaker


# ── lifecycle failure modes ──────────────────────────────────────────────────

def test_child_death_on_load_returns_error_verdicts(tmp_path: Path) -> None:
    backend = _backend(tmp_path, behavior="die_on_load")
    verdicts = backend.judge_pairs([("a", "b")])
    assert len(verdicts) == 1
    assert verdicts[0].error is not None
    # fail-open verdict labels the safe class; callers key off .error
    assert verdicts[0].probs == {}


def test_crash_breaker_disables_after_repeated_deaths(tmp_path: Path) -> None:
    backend = _backend(tmp_path, behavior="die_every_request")
    for _ in range(3):
        backend.judge_pairs([("a", "b")])
    assert backend.status()["disabled"] is True
    assert "crash breaker" in (backend.status()["last_error"] or "")
    # after the breaker trips, requests fail immediately (still error verdicts)
    verdicts = backend.judge_pairs([("a", "b")])
    assert verdicts[0].error is not None


def test_set_disabled_blocks_immediately(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.load()
    backend.set_disabled(True)
    verdicts = backend.judge_pairs([("a", "b")])
    assert verdicts[0].error is not None


def test_unload_disable_wins_over_inflight_flag(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.load()
    result = backend.unload(timeout=5.0, disable=True)
    assert result["ok"] is True
    assert backend.status()["disabled"] is True
