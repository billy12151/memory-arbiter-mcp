"""ManagedEmbedder device policy + token-budget clamping (mema 793, v0.14.11).

Contract under test:
  - the embed_text token budget is min(n_ctx - reserved_tokens, n_batch):
    the llama-cpp-python embed() path silently truncates input to n_batch
    tokens, so the budget must clamp to the same ceiling or diagnostics lie;
  - GPU offload is probed once at construction with CPU fallback, and a
    GPU-backed instance that fails at runtime degrades exactly once to a
    freshly built CPU instance, healing the failing call;
  - doctor's deep pass surfaces the device state and counts evidence units
    beyond the budget (a tail-loss tripwire).
"""
import sys
import types
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.doctor import open_ro_connection, run_all_checks
from memory_arbiter.embedder import EmbedResult, ManagedEmbedder, build_embedder


def _fake_closures(captured: list[str], dim: int = 4):
    def encode(text: str) -> list[float]:
        captured.append(text)
        return [0.5] * dim

    def tokenize(text: str) -> list[int]:
        return list(range(len(text)))

    return encode, tokenize


def _embedder(encode, tokenize, **kwargs) -> ManagedEmbedder:
    fields: dict = dict(
        encode_raw=encode,
        tokenize=tokenize,
        model_digest="digest",
        embedding_space_id="space",
        n_ctx=2048,
        reserved_tokens=64,
    )
    fields.update(kwargs)
    return ManagedEmbedder(**fields)


class TestTokenBudget:
    def test_budget_clamps_to_n_batch(self):
        emb = _embedder(*_fake_closures([]), n_batch=100)
        assert emb.token_budget() == 100

    def test_budget_falls_back_to_n_ctx_when_batch_larger(self):
        emb = _embedder(*_fake_closures([]), n_batch=4096, n_ctx=200)
        assert emb.token_budget() == 200 - 64

    def test_default_n_batch_matches_library_default(self):
        emb = _embedder(*_fake_closures([]))
        assert emb.n_batch == 512
        assert emb.token_budget() == 512

    def test_embed_text_truncates_to_n_batch_budget(self):
        captured: list[str] = []
        emb = _embedder(*_fake_closures(captured), n_batch=100)
        result = emb.embed_text(prefix="", body="汉" * 150)
        assert result.embedding == [0.5] * 4
        assert result.truncated is True
        assert result.original_tokens == 150
        assert result.used_tokens <= 100
        assert len(captured[-1]) <= 100

    def test_embed_text_untouched_when_under_budget(self):
        captured: list[str] = []
        emb = _embedder(*_fake_closures(captured), n_batch=100)
        result = emb.embed_text(prefix="", body="汉" * 80)
        assert result.truncated is False
        assert result.used_tokens == 80
        assert captured[-1] == "汉" * 80

    def test_prefix_counts_toward_budget(self):
        captured: list[str] = []
        emb = _embedder(*_fake_closures(captured), n_batch=100)
        # prefix (4 tokens) + sep (1) + body must fit 100 together
        result = emb.embed_text(prefix="abcd", body="汉" * 150)
        assert result.truncated is True
        assert result.used_tokens <= 100
        assert captured[-1].startswith("abcd\n")
        assert len(captured[-1]) <= 100


class TestThreadSafety:
    def test_concurrent_calls_stay_single_flight(self):
        import threading
        import time

        inflight = 0
        max_inflight = 0
        guard = threading.Lock()

        # BOTH encode and tokenize track in-flight calls, so the assertion
        # fails if either embed_text or tokenize_locked ever runs unlocked.
        def enter() -> None:
            nonlocal inflight, max_inflight
            with guard:
                inflight += 1
                max_inflight = max(max_inflight, inflight)

        def leave() -> None:
            nonlocal inflight
            with guard:
                inflight -= 1

        def encode(text: str) -> list[float]:
            enter()
            time.sleep(0.001)  # simulate the GIL-release a C call performs
            leave()
            return [0.5] * 4

        def tokenize(text: str) -> list[int]:
            enter()
            time.sleep(0.001)
            leave()
            return list(range(len(text)))

        emb = _embedder(encode, tokenize)
        errors: list[BaseException] = []
        threads = []
        for i in range(8):
            def worker(idx=i):
                try:
                    if idx % 2:
                        assert emb.embed_text(prefix="", body=f"text-{idx}" * 30).embedding
                    else:
                        assert emb.tokenize_locked("probe" * 30)
                except BaseException as exc:  # noqa: BLE001 — collected for assertion
                    errors.append(exc)
            threads.append(threading.Thread(target=worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert max_inflight == 1


class TestGpuSelfHeal:
    def _gpu_embedder(self, bad_encode, rebuild_calls: list[int]):
        def rebuild():
            rebuild_calls.append(1)
            return _fake_closures([])

        return _embedder(bad_encode, _fake_closures([])[1], gpu_backed=True, _cpu_rebuild=rebuild)

    def test_runtime_failure_degrades_once_and_heals_call(self):
        rebuild_calls: list[int] = []
        bad_calls: list[str] = []

        def bad_encode(text: str) -> list[float]:
            bad_calls.append(text)
            raise RuntimeError("metal device lost")

        emb = self._gpu_embedder(bad_encode, rebuild_calls)
        result = emb.embed_text(prefix="", body="hello")
        assert result.embedding == [0.5] * 4  # healed on the CPU instance
        assert emb.device_degraded is True
        assert emb.gpu_backed is False
        assert emb.device_degraded_at is not None
        assert rebuild_calls == [1]
        assert any("degraded to CPU" in w for w in emb.warnings)

    def test_degrade_is_one_shot(self):
        rebuild_calls: list[int] = []
        state = {"fail_cpu": False}

        def gpu_encode(text: str) -> list[float]:
            raise RuntimeError("metal device lost")

        def rebuild():
            rebuild_calls.append(1)

            def cpu_encode(text: str) -> list[float]:
                if state["fail_cpu"]:
                    raise RuntimeError("cpu also broken")
                return [0.5] * 4

            def cpu_tokenize(text: str) -> list[int]:
                return list(range(len(text)))

            return cpu_encode, cpu_tokenize

        emb = _embedder(gpu_encode, _fake_closures([])[1], gpu_backed=True, _cpu_rebuild=rebuild)
        healed = emb.embed_text(prefix="", body="a")
        assert healed.embedding == [0.5] * 4  # GPU failed → degraded → CPU retry OK
        state["fail_cpu"] = True
        result = emb.embed_text(prefix="", body="b")  # CPU now fails too
        assert result.embedding == []
        assert result.truncated is True
        assert emb.last_encode_error == "cpu also broken"
        assert rebuild_calls == [1]  # no second rebuild attempt

    def test_cpu_instance_never_rebuilds(self):
        rebuild_calls: list[int] = []

        def rebuild():
            rebuild_calls.append(1)
            return _fake_closures([])

        def bad_encode(text: str) -> list[float]:
            raise RuntimeError("plain failure")

        emb = _embedder(bad_encode, _fake_closures([])[1], gpu_backed=False, _cpu_rebuild=rebuild)
        result = emb.embed_text(prefix="", body="x")
        assert result.embedding == []
        assert rebuild_calls == []
        assert emb.device_degraded is False

    def test_rebuild_failure_returns_sentinel_with_warning(self):
        def broken_rebuild():
            raise RuntimeError("model file vanished")

        def bad_encode(text: str) -> list[float]:
            raise RuntimeError("metal device lost")

        emb = _embedder(bad_encode, _fake_closures([])[1], gpu_backed=True, _cpu_rebuild=broken_rebuild)
        result = emb.embed_text(prefix="", body="x")
        assert result.embedding == []
        assert emb.device_degraded is True  # latch closed even on failed rebuild
        assert any("rebuild also failed" in w for w in emb.warnings)


class _FakeLlama:
    instances: list["_FakeLlama"] = []
    gpu_construction_error: str | None = None
    gpu_dim: int | None = None
    gpu_encode_error: str | None = None
    cpu_encode_error: str | None = None

    def __init__(self, **kwargs):
        if kwargs.get("n_gpu_layers") and _FakeLlama.gpu_construction_error:
            raise RuntimeError(_FakeLlama.gpu_construction_error)
        self.kwargs = kwargs
        self.n_batch = int(kwargs.get("n_batch", 512))
        _FakeLlama.instances.append(self)

    def create_embedding(self, text):
        is_gpu = "n_gpu_layers" in self.kwargs
        if is_gpu and _FakeLlama.gpu_encode_error:
            raise RuntimeError(_FakeLlama.gpu_encode_error)
        if not is_gpu and _FakeLlama.cpu_encode_error:
            raise RuntimeError(_FakeLlama.cpu_encode_error)
        dim = _FakeLlama.gpu_dim if _FakeLlama.gpu_dim is not None and is_gpu else 4
        return {"data": [{"embedding": [0.5] * dim}]}

    def tokenize(self, data, add_bos=False):
        return list(range(max(1, len(data))))


@pytest.fixture()
def fake_llama_cpp(monkeypatch):
    _FakeLlama.instances = []
    _FakeLlama.gpu_construction_error = None
    _FakeLlama.gpu_dim = None
    _FakeLlama.gpu_encode_error = None
    _FakeLlama.cpu_encode_error = None
    module = types.ModuleType("llama_cpp")
    module.Llama = _FakeLlama
    module.llama_supports_gpu_offload = lambda: True
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return module


class TestBuildEmbedderDevicePolicy:
    def _model(self, tmp_path: Path) -> str:
        path = tmp_path / "model.gguf"
        path.write_bytes(b"gguf-bytes")
        return str(path)

    def test_gpu_backend_offloads_and_exposes_cpu_rebuild(self, fake_llama_cpp, tmp_path):
        embedder, warnings = build_embedder(self._model(tmp_path), expected_dim=4)
        assert embedder is not None and warnings == []
        assert embedder.gpu_backed is True
        assert _FakeLlama.instances[0].kwargs["n_gpu_layers"] == -1
        assert embedder.n_batch == 512
        encode, _ = embedder._cpu_rebuild()
        assert encode("probe") == [0.5] * 4
        assert "n_gpu_layers" not in _FakeLlama.instances[1].kwargs  # rebuild is CPU-only

    def test_gpu_construction_failure_falls_back_to_cpu(self, fake_llama_cpp, tmp_path):
        _FakeLlama.gpu_construction_error = "no metal device available"
        embedder, warnings = build_embedder(self._model(tmp_path), expected_dim=4)
        assert embedder is not None
        assert embedder.gpu_backed is False
        assert embedder._cpu_rebuild is None
        assert any("GPU offload construction failed" in w for w in warnings)

    def test_gpu_dim_mismatch_retries_on_cpu(self, fake_llama_cpp, tmp_path):
        _FakeLlama.gpu_dim = 7  # wrong dimension only on the GPU instance
        embedder, warnings = build_embedder(self._model(tmp_path), expected_dim=4)
        assert embedder is not None
        assert embedder.gpu_backed is False
        assert any("GPU dimension probe" in w for w in warnings)

    def test_gpu_probe_exception_retries_on_cpu(self, fake_llama_cpp, tmp_path):
        _FakeLlama.gpu_encode_error = "metal device removed"
        embedder, warnings = build_embedder(self._model(tmp_path), expected_dim=4)
        assert embedder is not None
        assert embedder.gpu_backed is False
        assert any("GPU dimension probe failed" in w for w in warnings)

    def test_cpu_probe_exception_disables_embedder(self, fake_llama_cpp, tmp_path):
        fake_llama_cpp.llama_supports_gpu_offload = lambda: False
        _FakeLlama.cpu_encode_error = "boom"
        embedder, warnings = build_embedder(self._model(tmp_path), expected_dim=4)
        assert embedder is None
        assert any("GGUF embedder load failed" in w for w in warnings)

    def test_cpu_only_wheel_never_passes_gpu_kwarg(self, fake_llama_cpp, tmp_path):
        fake_llama_cpp.llama_supports_gpu_offload = lambda: False
        embedder, warnings = build_embedder(self._model(tmp_path), expected_dim=4)
        assert embedder is not None and warnings == []
        assert embedder.gpu_backed is False
        assert all("n_gpu_layers" not in i.kwargs for i in _FakeLlama.instances)


class _DoctorProbeEmbedder:
    """Duck-typed embedder for doctor's deep pass."""

    def __init__(self, dim: int = 2, *, degraded: bool = False,
                 degrade_failed: bool = False, tokenize_raises: bool = False):
        self._dim = dim
        self.gpu_backed = not degraded
        self.device_degraded = degraded or degrade_failed
        self.device_degraded_at = None if degrade_failed else (
            "2026-08-31T00:00:00Z" if degraded else None
        )
        self._tokenize_raises = tokenize_raises

    def embed_text(self, prefix: str, body: str, max_body_chars: int | None = None) -> EmbedResult:
        return EmbedResult(embedding=[0.5] * self._dim, truncated=False,
                           original_tokens=len(body), used_tokens=len(body))

    def tokenize_locked(self, text: str) -> list[int]:
        if self._tokenize_raises:
            raise RuntimeError("tokenizer unavailable")
        return list(range(len(text)))  # 1 token/char — CJK density

    def token_budget(self) -> int:
        return 512


class TestDoctorDeepDeviceAndBudget:
    def _db_with_unit(self, tmp_path: Path, unit_text: str):
        settings = Settings(
            db_path=tmp_path / "budget.sqlite3",
            backup_jsonl=tmp_path / "budget.jsonl",
            enable_sqlite_vec=False,
            vec_dim=2,
        )
        db = MemoryDB(settings)
        with db.write_transaction() as conn:
            conn.execute(
                """INSERT INTO memories(content,agent_id,workspace,tags,source_type,
                   event_time,ingest_time,status,subject,metadata,version,created_at)
                   VALUES('body','agent','default','[]','agent_generated',
                   '2026-01-01T00:00:00Z','2026-01-01T00:00:00Z','active','subject','{}',1,
                   '2026-01-01T00:00:00Z')"""
            )
            memory_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            conn.execute(
                """INSERT INTO memory_evidence(memory_id,memory_version,content_hash,
                   unit_index,kind,text,start_offset,end_offset,created_at)
                   VALUES(?,1,'hash',0,'text',?,0,?,'2026-01-01T00:00:00Z')""",
                (memory_id, unit_text, len(unit_text)),
            )
        return settings, db

    def test_oversized_unit_warns_and_degraded_device_flags(self, tmp_path):
        settings, db = self._db_with_unit(tmp_path, "汉" * 600)
        probe = _DoctorProbeEmbedder(degraded=True)
        with open_ro_connection(Path(settings.db_path)) as conn:
            report = run_all_checks(conn, settings, deep=True, embedder_probe=lambda: (probe, []))
        findings = {f.check_id: f for f in report.findings}
        budget = findings["evidence.unit_budget"]
        assert budget.status == "warn"
        assert budget.evidence["over_budget"] == 1
        assert budget.evidence["checked_units"] == 1
        assert budget.evidence["worst_tokens"] == 600
        device = findings["vector.device"]
        assert device.status == "warn"
        assert "degraded to CPU" in device.detail

    def test_undersized_unit_passes_and_healthy_device_is_info(self, tmp_path):
        settings, db = self._db_with_unit(tmp_path, "汉" * 350)  # over the 128-char prefilter, under the 512 budget
        probe = _DoctorProbeEmbedder()
        with open_ro_connection(Path(settings.db_path)) as conn:
            report = run_all_checks(conn, settings, deep=True, embedder_probe=lambda: (probe, []))
        findings = {f.check_id: f for f in report.findings}
        budget = findings["evidence.unit_budget"]
        assert budget.status == "pass"
        assert budget.evidence["checked_units"] == 1
        assert findings["vector.device"].status == "pass"

    def test_prefilter_scales_with_budget(self, tmp_path):
        # prefilter = budget // 4 = 128 chars: a 130-char unit is checked even
        # though CJK density alone would never push it past 512 tokens.
        settings, db = self._db_with_unit(tmp_path, "汉" * 130)
        probe = _DoctorProbeEmbedder()
        with open_ro_connection(Path(settings.db_path)) as conn:
            report = run_all_checks(conn, settings, deep=True, embedder_probe=lambda: (probe, []))
        budget = {f.check_id: f for f in report.findings}["evidence.unit_budget"]
        assert budget.evidence["checked_units"] == 1
        assert budget.evidence["worst_tokens"] == 130

    def test_failed_cpu_rebuild_reports_unavailable_not_degraded(self, tmp_path):
        settings, db = self._db_with_unit(tmp_path, "汉" * 10)
        probe = _DoctorProbeEmbedder(degrade_failed=True)
        with open_ro_connection(Path(settings.db_path)) as conn:
            report = run_all_checks(conn, settings, deep=True, embedder_probe=lambda: (probe, []))
        device = {f.check_id: f for f in report.findings}["vector.device"]
        assert device.status == "warn"
        assert "unavailable until restart" in device.detail

    def test_raising_tokenizer_never_crashes_doctor(self, tmp_path):
        settings, db = self._db_with_unit(tmp_path, "汉" * 600)
        probe = _DoctorProbeEmbedder(tokenize_raises=True)
        with open_ro_connection(Path(settings.db_path)) as conn:
            report = run_all_checks(conn, settings, deep=True, embedder_probe=lambda: (probe, []))
        budget = {f.check_id: f for f in report.findings}["evidence.unit_budget"]
        assert budget.status == "pass"
        assert "scan failed" in budget.detail
