"""mDeBERTa three-class conflict judge (0.17.1): the write-time arbitration
engine that replaces the Qwen slot-extraction backend (owner 2026-09-28,
plan mema-mdeberta-v4m-judge-swap-0171 §3.1).

Design contract:
- The model is the ONLY judge engine: no selection key, no Qwen runtime
  fallback (rollback = git revert / 0.17.0). An unconfigured checkpoint means
  arbitration is disabled — the ``_ensure_semantic_backend``-returns-None path
  the rest of the pipeline already treats as fail-open.
- Deployment shape mirrors the retired IsolatedGGUFSemanticBackend
  lifecycle: spawn subprocess, single-flight lock, hard timeouts, preload,
  restart on child death. The protocol is JSON-lines-shaped dict exchange over
  a multiprocessing Pipe (the GGUF backend pickled dataclasses; here both
  sides exchange plain dicts so the child needs no imports from this module).
- Label contract is PINNED: the child echoes its LABELS after load; the
  parent refuses to serve if the echo differs from SEMANTIC_MDEBERTA_LABELS
  (a future checkpoint that reorders/renames classes must fail loudly, not
  misjudge silently).
- Device is CPU fp32 only (mDeBERTa MPS dtype assertion bug, train.py:106).
- Batch interface is primary: judge_pairs(list[(text_a, text_b)]) with
  length-sorted dynamic padding; single pair = batch of 1.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    SEMANTIC_INFERENCE_TIMEOUT_MS,
    SEMANTIC_LOAD_TIMEOUT_MS,
    SEMANTIC_MDEBERTA_BATCH,
    SEMANTIC_MDEBERTA_LABELS,
    SEMANTIC_MDEBERTA_MAX_LEN,
    SEMANTIC_N_THREADS,
)

# The restart circuit breaker (owner plan §3.1, adversarial review P2⑥): a
# child that dies repeatedly (OOM-kill, import crash) must disable itself
# instead of re-loading a 1.1GB model on every write job. 3 deaths inside the
# window → set_disabled + doctor reports; manual restart clears.
_CRASH_BREAKER_THRESHOLD = 3
_CRASH_BREAKER_WINDOW_S = 600.0


@dataclass(frozen=True)
class PairVerdict:
    """One judged pair. ``label`` is always one of SEMANTIC_MDEBERTA_LABELS;
    ``error`` is set only on technical failure (timeout / child death /
    contract mismatch) — the caller's fail-open paths key off it."""

    label: str
    probs: dict[str, float]
    mechanism: str | None
    model_version: str
    error: str | None = None


def _failed_verdict(error: str) -> PairVerdict:
    return PairVerdict("no_conflict", {}, None, "mdeberta:unavailable", error=error)


def checkpoint_sha8(ckpt_path: Path) -> str:
    """First 8 hex chars of the checkpoint digest. Read on every load so a
    swapped .pt changes the reported model_version without a code change."""
    digest = hashlib.sha256()
    with open(ckpt_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:8]


def _mdeberta_inference_process(conn: Any, config: dict[str, Any]) -> None:
    """Child entry point. Owns only torch state, never MemoryDB state.

    Loads AutoConfig + AutoModel from ``model_dir`` (config/tokenizer only —
    the base .safetensors is NOT read), builds the DualHead skeleton, loads
    the full state_dict from ``ckpt``. Echoes LABELS for the parent-side
    contract check. torch threads are pinned before model load.
    """
    import gc

    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        torch.set_num_threads(int(config["n_threads"]))
        model_dir = str(config["model_dir"])
        ckpt = str(config["ckpt"])
        max_len = int(config["max_len"])

        cfg = AutoConfig.from_pretrained(model_dir)
        base = AutoModel.from_config(cfg)
        hidden = base.config.hidden_size
        labels = list(config["labels"])
        mechs = list(config["mechs"])
        lab_head = torch.nn.Sequential(
            torch.nn.Dropout(0.1), torch.nn.Linear(hidden, len(labels)),
        )
        mech_head = torch.nn.Sequential(
            torch.nn.Dropout(0.1), torch.nn.Linear(hidden, len(mechs)),
        )

        state = torch.load(ckpt, map_location="cpu")
        missing, unexpected = base.load_state_dict(
            {k.removeprefix("encoder."): v for k, v in state.items()
             if k.startswith("encoder.")},
            strict=False,
        )
        lab_head.load_state_dict(
            {k.removeprefix("lab_head."): v for k, v in state.items()
             if k.startswith("lab_head.")},
        )
        mech_head.load_state_dict(
            {k.removeprefix("mech_head."): v for k, v in state.items()
             if k.startswith("mech_head.")},
        )
        # Buffers (lab_w/mech_w) and dropout prefixes are dropped on purpose;
        # a genuinely incompatible skeleton surfaces as a large ``missing``.
        if len(missing) > len(getattr(base, "_keys_to_ignore_on_load_missing", [])) + 8:
            raise RuntimeError(
                f"checkpoint skeleton mismatch: {len(missing)} missing encoder keys"
            )
        base.eval()
        lab_head.eval()
        mech_head.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        conn.send({
            "ok": True, "result": {
                "loaded": True, "labels": labels, "mechs": mechs,
                "model_version": f"mdeberta-v4m:{config['sha8']}",
            },
        })
    except BaseException as exc:  # noqa: BLE001 - child must report, not die silently
        try:
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except BaseException:
            pass
        return

    try:
        softmax = torch.nn.Softmax(dim=-1)
        while True:
            request = conn.recv()
            command = request.get("command")
            if command == "shutdown":
                return
            try:
                if command == "judge_pairs":
                    pairs = request["pairs"]
                    results: list[dict[str, Any]] = []
                    order = sorted(
                        range(len(pairs)),
                        key=lambda i: len(pairs[i][0]) + len(pairs[i][1]),
                    )
                    batch_outputs: dict[int, dict[str, Any]] = {}
                    with torch.inference_mode():
                        for start in range(0, len(order), int(config["batch_size"])):
                            chunk = order[start : start + int(config["batch_size"])]
                            texts_a = [pairs[i][0] for i in chunk]
                            texts_b = [pairs[i][1] for i in chunk]
                            enc = tokenizer(
                                texts_a, texts_b, truncation=True,
                                max_length=max_len, padding=True,
                                return_tensors="pt",
                            )
                            out = base(
                                input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"],
                            )
                            cls = out.last_hidden_state[:, 0]
                            lab_logits = lab_head(cls)
                            mech_logits = mech_head(cls)
                            probs = softmax(lab_logits, dim=-1).tolist()
                            mech_idx = mech_logits.argmax(dim=-1).tolist()
                            for pos, i in enumerate(chunk):
                                batch_outputs[i] = {
                                    "probs": probs[pos],
                                    "mech": mechs[int(mech_idx[pos])],
                                }
                    for i in range(len(pairs)):
                        item = batch_outputs.get(i)
                        if item is None:
                            results.append({"error": "judge_pairs internal gap"})
                            continue
                        results.append({
                            "probs": {
                                labels[j]: round(float(p), 4)
                                for j, p in enumerate(item["probs"])
                            },
                            "mechanism": item["mech"],
                        })
                    conn.send({"ok": True, "result": results})
                else:
                    raise ValueError(f"unknown judge child command: {command}")
            except BaseException as exc:  # noqa: BLE001
                conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        # No Metal/SIGABRT history for torch, but free deterministically
        # anyway — 1.1GB should return to the OS promptly.
        try:
            del base, lab_head, mech_head, tokenizer  # noqa: F821
        except BaseException:
            pass
        gc.collect()
        conn.close()


class IsolatedMDeBERTaBackend:
    """Single-flight supervisor over the torch child process. Mirrors the
    retired GGUF supervisor's lifecycle (spawn / lazy load / hard timeouts /
    terminate+restart / disabled flag) minus the notice/workspace scheduler
    queues — the judge has one request class."""

    def __init__(
        self,
        ckpt: Path,
        model_dir: Path,
        *,
        batch_size: int = SEMANTIC_MDEBERTA_BATCH,
        max_len: int = SEMANTIC_MDEBERTA_MAX_LEN,
        n_threads: int = SEMANTIC_N_THREADS,
        hard_timeout_ms: int = SEMANTIC_INFERENCE_TIMEOUT_MS,
        load_timeout_ms: int = SEMANTIC_LOAD_TIMEOUT_MS,
        process_target: Any = _mdeberta_inference_process,
        child_config_extra: dict[str, Any] | None = None,
    ) -> None:
        import multiprocessing

        self.ckpt = Path(ckpt).expanduser()
        self.model_dir = Path(model_dir).expanduser()
        self.batch_size = max(1, int(batch_size))
        self.max_len = int(max_len)
        self.n_threads = int(n_threads)
        self.hard_timeout_ms = max(1, int(hard_timeout_ms))
        self.load_timeout_ms = max(1, int(load_timeout_ms))
        self._process_target = process_target
        self._child_config_extra = dict(child_config_extra or {})
        self._ctx = multiprocessing.get_context("spawn")
        self._request_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._process: Any = None
        self._conn: Any = None
        self._child_loaded = False
        self._disabled = False
        self._generation = 0
        self._restarts = 0
        self._timed_out = 0
        self._last_error: str | None = None
        self._loaded_at: float | None = None
        self._labels: tuple[str, ...] | None = None
        self._model_version: str | None = None
        self._crash_times: list[float] = []
        self._sha8: str | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def _start_locked(self) -> None:
        if self._disabled:
            raise RuntimeError("mdeberta judge disabled")
        if self._process is not None and self._process.is_alive():
            return
        if self._process is not None:
            self._terminate_locked(count_restart=True)
        if self._sha8 is None:
            self._sha8 = checkpoint_sha8(self.ckpt)
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=self._process_target,
            args=(child, {
                "model_dir": str(self.model_dir),
                "ckpt": str(self.ckpt),
                "sha8": self._sha8,
                "labels": list(SEMANTIC_MDEBERTA_LABELS),
                "mechs": [
                    "affirmation_negation", "numeric_value", "numeric_range",
                    "quantifier_scope", "obligation_permission", "time_version",
                    "condition_exception", "reference_entity",
                    "definition_caliber", "paraphrase_compatibility",
                ],
                "max_len": self.max_len,
                "batch_size": self.batch_size,
                "n_threads": self.n_threads,
                **self._child_config_extra,
            }),
            name="memory-arbiter-mdeberta-judge",
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            try:
                process.close()
            except (AttributeError, ValueError):
                pass
            raise
        child.close()
        self._process = process
        self._conn = parent
        self._generation += 1
        self._child_loaded = False

    def _terminate_locked(self, *, count_restart: bool) -> None:
        process, conn = self._process, self._conn
        self._process = None
        self._conn = None
        self._loaded_at = None
        self._child_loaded = False
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():  # pragma: no cover - defensive OS fallback
                process.kill()
                process.join(timeout=1.0)
            try:
                process.close()
            except (AttributeError, ValueError):
                pass
            if count_restart:
                self._restarts += 1

    def _record_child_death(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            self._crash_times = [t for t in self._crash_times if now - t < _CRASH_BREAKER_WINDOW_S]
            self._crash_times.append(now)
            if len(self._crash_times) >= _CRASH_BREAKER_THRESHOLD:
                self._disabled = True
                self._last_error = (
                    "mdeberta child died "
                    f"{len(self._crash_times)}x in {_CRASH_BREAKER_WINDOW_S:.0f}s — "
                    "judge disabled (crash breaker); restart the service to clear"
                )

    def _exchange(self, command: str, timeout_ms: int, **payload: Any) -> Any:
        conn = self._conn
        conn.send({"command": command, **payload})
        deadline = time.monotonic() + max(0.001, timeout_ms / 1000.0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._state_lock:
                    self._timed_out += 1
                    phase = "load" if command == "load" else "inference"
                    self._last_error = f"mdeberta {phase} hard timeout after {timeout_ms}ms"
                    if self._conn is conn:
                        self._terminate_locked(count_restart=True)
                raise TimeoutError(self._last_error)
            if conn.poll(min(remaining, 0.05)):
                break
            with self._state_lock:
                if self._conn is not conn:
                    raise EOFError("mdeberta child connection was closed")
        response = conn.recv()
        if not response.get("ok"):
            self._last_error = str(response.get("error") or "mdeberta child error")
            raise RuntimeError(self._last_error)
        return response.get("result")

    def _request(self, command: str, timeout_ms: int, **payload: Any) -> Any:
        acquired = self._request_lock.acquire(timeout=0)
        if not acquired:
            raise RuntimeError("mdeberta judge admitted concurrent request")
        try:
            with self._state_lock:
                self._start_locked()
                conn = self._conn
            try:
                if not self._child_loaded:
                    # One load per child, lazily on the first request — an
                    # explicit load() is simply the first request.
                    self._verify_contract(self._exchange("load", self.load_timeout_ms))
                    with self._state_lock:
                        self._child_loaded = True
                        self._loaded_at = time.time()
                if command != "load":
                    return self._exchange(command, timeout_ms, **payload)
                return {"loaded": True}
            except (EOFError, BrokenPipeError, OSError) as exc:
                with self._state_lock:
                    self._last_error = f"mdeberta child exited: {exc}"
                    self._terminate_locked(count_restart=True)
                self._record_child_death()
                raise RuntimeError(self._last_error) from exc
            finally:
                with self._state_lock:
                    if (
                        self._process is not None
                        and not self._process.is_alive()
                    ):
                        self._terminate_locked(count_restart=True)
                        self._record_child_death()
        finally:
            self._request_lock.release()

    def _verify_contract(self, load_result: Any) -> None:
        if not isinstance(load_result, dict):
            raise RuntimeError("mdeberta load returned invalid payload")
        labels = tuple(load_result.get("labels") or ())
        if labels != tuple(SEMANTIC_MDEBERTA_LABELS):
            raise RuntimeError(
                "mdeberta label contract mismatch: child echoed "
                f"{labels}, expected {tuple(SEMANTIC_MDEBERTA_LABELS)} — refusing to serve"
            )
        with self._state_lock:
            self._labels = labels
            self._model_version = str(load_result.get("model_version") or "mdeberta:unknown")

    # ── public interface ────────────────────────────────────────────────────
    def judge_pairs(self, pairs: list[tuple[str, str]]) -> list[PairVerdict]:
        if not pairs:
            return []
        try:
            results = self._request("judge_pairs", self.hard_timeout_ms, pairs=list(pairs))
        except TimeoutError as exc:
            return [_failed_verdict(str(exc)) for _ in pairs]
        except RuntimeError as exc:
            return [_failed_verdict(str(exc)) for _ in pairs]
        verdicts: list[PairVerdict] = []
        with self._state_lock:
            version = self._model_version or "mdeberta:unknown"
        for item in results:
            if not isinstance(item, dict) or item.get("error"):
                verdicts.append(_failed_verdict(str(item.get("error") if isinstance(item, dict) else "invalid child response")))
                continue
            probs = item.get("probs")
            if not isinstance(probs, dict) or set(probs) != set(SEMANTIC_MDEBERTA_LABELS):
                verdicts.append(_failed_verdict("child probs violate label contract"))
                continue
            label = max(probs, key=lambda k: probs[k])
            verdicts.append(PairVerdict(
                label=label,
                probs={k: float(v) for k, v in probs.items()},
                mechanism=item.get("mechanism"),
                model_version=version,
            ))
        return verdicts

    def judge_pair(self, text_a: str, text_b: str) -> PairVerdict:
        return self.judge_pairs([(text_a, text_b)])[0]

    def load(self) -> None:
        self._request("load", self.load_timeout_ms)

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "ckpt": str(self.ckpt),
                "ckpt_sha8": self._sha8,
                "model_dir": str(self.model_dir),
                "model_version": self._model_version,
                "labels": list(self._labels or ()),
                "device": "cpu",
                "batch_size": self.batch_size,
                "max_len": self.max_len,
                "n_threads": self.n_threads,
                "loaded_at": self._loaded_at,
                "last_error": self._last_error,
                "restarts": self._restarts,
                "timed_out": self._timed_out,
                "disabled": self._disabled,
                "generation": self._generation,
            }

    def unload(self, timeout: float = 30.0, disable: bool = False) -> dict[str, Any]:
        if disable:
            with self._state_lock:
                self._disabled = True
        acquired = self._request_lock.acquire(timeout=max(0.0, float(timeout)))
        if not acquired:
            return {"ok": False, "unloaded": False, "timeout": True, "inflight": 1, "retry_hint": "retry after inference completes", "generation": self._generation}
        try:
            with self._state_lock:
                self._terminate_locked(count_restart=False)
                return {"ok": True, "unloaded": True, "timeout": False, "inflight": 0, "retry_hint": None, "generation": self._generation}
        finally:
            self._request_lock.release()

    def set_disabled(self, disabled: bool) -> None:
        with self._state_lock:
            self._disabled = bool(disabled)

    def force_terminate(self) -> dict[str, Any]:
        with self._state_lock:
            self._disabled = True
        return self.unload(timeout=5.0, disable=True)
