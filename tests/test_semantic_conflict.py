from pathlib import Path
import threading
import time

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.semantic_conflict import model_signal_from_text, pair_text_gate, pair_text_evidence
from memory_arbiter.tools import MemoryTools


def test_pair_text_gate_medium_and_strong_examples():
    medium = pair_text_gate(
        "待办：完成 Console Settings 页。",
        "Console Settings 页已经完成并随 v0.10.3 发布。",
        mode="medium",
    )
    assert medium.passed
    assert medium.severity == "high"
    assert "todo_done" in medium.reasons

    duplicate = pair_text_gate(
        "mema 中文名叫迷码，CLI alias 是 mema。",
        "mema 的中文名是迷码，命令行仍然用 mema。",
        mode="medium",
    )
    assert not duplicate.passed
    assert duplicate.reasons == ["duplicate_guard"]

    compatible = pair_text_gate(
        "本机测试固定用 Python 3.12 pytest。",
        "本机测试不要用 Homebrew Python 3.14，固定 Python 3.12 pytest。",
        mode="strong",
    )
    assert not compatible.passed
    assert compatible.reasons == ["compatible_guard"]


def test_pair_text_gate_medium_only_pass_is_normal_severity():
    # The default medium gate must be able to pass a pair at severity="normal"
    # (not "high") without meeting any strong criterion. This is the branch that
    # distinguishes the default medium gate from the strong gate (id=609); the
    # strong gate must reject the very same pair.
    left, right = "系统采用方案甲甲甲甲。", "系统采用方案乙乙乙乙。"
    medium = pair_text_gate(left, right, mode="medium")
    assert medium.passed is True
    assert medium.severity == "normal"
    assert "replacement" in medium.reasons

    strong = pair_text_gate(left, right, mode="strong")
    assert strong.passed is False
    assert strong.severity == "none"


def test_semantic_config_defaults_and_env(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config_path))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "m.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "m.jsonl"))
    monkeypatch.delenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED", raising=False)
    settings = Settings.from_env()
    assert settings.semantic_conflict_enabled is False
    assert settings.semantic_conflict_pair_text_gate == "medium"
    assert settings.semantic_conflict_backend == "local_gguf"

    monkeypatch.setenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_GATE", "strong")
    monkeypatch.setenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_MODEL_PATH", str(tmp_path / "qwen.gguf"))
    settings = Settings.from_env()
    assert settings.semantic_conflict_enabled is True
    assert settings.semantic_conflict_pair_text_gate == "strong"
    assert settings.semantic_conflict_model_path == tmp_path / "qwen.gguf"


def test_semantic_auto_enabled_when_model_path_set(monkeypatch, tmp_path: Path):
    """model_path configured but enabled not explicitly set → auto-enable."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config_path))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "m.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "m.jsonl"))
    monkeypatch.delenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED", raising=False)
    monkeypatch.setenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_MODEL_PATH", str(tmp_path / "qwen.gguf"))
    settings = Settings.from_env()
    assert settings.semantic_conflict_enabled is True
    assert any("auto-enabled" in w for w in settings.config_warnings)


def test_semantic_explicit_false_overrides_auto_enable(monkeypatch, tmp_path: Path):
    """model_path set + enabled=false explicitly → still disabled."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"semantic_conflict": {"model_path": "'
        + str(tmp_path / "qwen.gguf").replace("\\", "\\\\")
        + '", "enabled": false}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config_path))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "m.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "m.jsonl"))
    monkeypatch.delenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED", raising=False)
    settings = Settings.from_env()
    assert settings.semantic_conflict_enabled is False


def _tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "m.jsonl",
        semantic_conflict_enabled=True,
        semantic_conflict_model_path=tmp_path / "missing.gguf",
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_semantic_notice_db_and_repair_control(tmp_path: Path):
    tools = _tools(tmp_path)
    first = tools.memory_write(
        content="mema 中文名叫迷码，CLI alias 是 mema。",
        subject="mema naming",
        tags=["mema", "naming"],
    )["data"]
    second = tools.memory_write(
        content="mema 的中文名是迷码，命令行仍然用 mema。",
        subject="mema naming followup",
        tags=["mema", "naming"],
    )["data"]
    assert first["semantic_conflict_check"]["status"] == "queued"
    assert second["semantic_conflict_check"]["status"] == "queued"

    result = tools.db.record_semantic_notice(
        memory_id=second["id"], peer_id=first["id"], severity="normal",
        notice_type="semantic_pair", title="test", message="msg",
        payload={"x": 1}, dedupe_key="k1",
    )
    assert result["outcome"] == "created"
    assert tools.db.record_semantic_notice(
        memory_id=second["id"], peer_id=first["id"], severity="normal",
        notice_type="semantic_pair", title="test", message="msg",
        payload={"x": 1}, dedupe_key="k1",
    )["outcome"] == "deduped"

    listed = tools.memory_repair(task="notice", data={"action": "list"})["data"]["notices"]
    assert len(listed) == 1
    assert listed[0]["payload"] == {"x": 1}
    dismissed = tools.memory_repair(task="notice", data={"action": "dismiss", "notice_id": listed[0]["id"]})
    assert dismissed["ok"] is True
    assert tools.db.semantic_notice_counts().get("dismissed") == 1


class _FakeSemanticBackend:
    def __init__(self):
        self.disabled = False
        self.unload_calls = 0
        self.maybe_unload_calls = 0

    def load(self):
        pass

    def classify_pair(self, left, right):
        from memory_arbiter.semantic_conflict import ModelSignal
        if self.disabled:
            return ModelSignal(False, "backend_unavailable", None, "", None, "disabled")
        return ModelSignal(True, "replacement", 0.9, "{}", {"reason_code": "replacement"})

    def suggest_workspace_candidate(self, ws_raw, evidence, candidates):
        from memory_arbiter.semantic_conflict import WorkspaceCandidateSignal
        if self.disabled:
            return WorkspaceCandidateSignal(None, "uncertain", None, "", "", "disabled")
        return WorkspaceCandidateSignal(candidates[0] if candidates else None, "alias", 0.9, "fake")

    def status(self):
        return {"backend": "fake", "model_state": "resident", "disabled": self.disabled}

    def unload(self, timeout=30.0, disable=False):
        self.unload_calls += 1
        self.disabled = self.disabled or bool(disable)
        return {"ok": True, "unloaded": True, "timeout": False, "inflight": 0, "retry_hint": None, "generation": self.unload_calls}

    def maybe_unload_if_idle(self):
        self.maybe_unload_calls += 1
        return {"ok": True, "unloaded": True, "inflight": 0, "generation": self.maybe_unload_calls}

    def set_disabled(self, disabled):
        self.disabled = bool(disabled)


def test_process_semantic_job_writes_notice_with_fake_backend(tmp_path: Path):
    tools = _tools(tmp_path)
    old = tools.memory_write(
        content="旧设计：vector scan 是冲突检测主路径。",
        subject="vector conflict path",
        tags=["memory-arbiter", "vector-conflict"],
    )["data"]
    new = tools.memory_write(
        content="新设计：保留 LLM scan，旧 vector conflict candidate scan 下线。",
        subject="vector conflict path update",
        tags=["memory-arbiter", "vector-conflict"],
    )["data"]
    record = tools.db.get_memory(new["id"])
    assert record is not None
    tools._semantic_backend = _FakeSemanticBackend()
    snapshot = {
        "memory_id": new["id"],
        "version": record["version"],
        "claim_revision": record["claim_revision"],
        "content_hash": __import__("hashlib").sha256(record["content"].encode("utf-8")).hexdigest(),
    }
    tools._process_semantic_conflict_job(new["id"], snapshot)
    notices = tools.db.list_semantic_notices()
    assert len(notices) == 1
    assert notices[0]["memory_id"] == new["id"]
    assert notices[0]["peer_id"] == old["id"]
    assert notices[0]["severity"] in {"normal", "high"}


def test_gate_vetoes_model_candidate_so_no_notice_is_written(tmp_path: Path):
    # id=609 two-layer design: the small model is only a recall signal; the
    # pair-text gate has veto power. Here the backend ALWAYS returns
    # candidate=True, but the pair is a duplicate the gate rejects
    # (duplicate_guard), so the job must write NO notice.
    tools = _tools(tmp_path)
    tools.memory_write(
        content="mema 中文名叫迷码，CLI alias 是 mema。",
        subject="mema naming",
        tags=["mema", "naming"],
    )
    new = tools.memory_write(
        content="mema 的中文名是迷码，命令行仍然用 mema。",
        subject="mema naming followup",
        tags=["mema", "naming"],
    )["data"]
    record = tools.db.get_memory(new["id"])
    # Sanity: the gate must actually veto this pair (else the test proves nothing).
    duplicate = pair_text_gate(
        "mema 中文名叫迷码，CLI alias 是 mema。",
        "mema 的中文名是迷码，命令行仍然用 mema。",
        mode="medium",
    )
    assert duplicate.passed is False
    tools._semantic_backend = _FakeSemanticBackend()  # always candidate=True
    snapshot = {
        "memory_id": new["id"],
        "version": record["version"],
        "claim_revision": record["claim_revision"],
        "content_hash": __import__("hashlib").sha256(record["content"].encode("utf-8")).hexdigest(),
    }
    tools._process_semantic_conflict_job(new["id"], snapshot)
    assert tools.db.list_semantic_notices() == []

def test_memory_edit_enqueues_semantic_check(tmp_path: Path):
    tools = _tools(tmp_path)
    written = tools.memory_write(
        content="旧设计：vector scan 是冲突检测主路径。",
        subject="vector conflict path",
        tags=["memory-arbiter", "vector-conflict"],
    )["data"]
    edited = tools.memory_edit(
        memory_id=written["id"],
        new_content="新设计：保留 LLM scan，旧 vector conflict candidate scan 下线。",
    )["data"]
    assert edited["edited"] is True
    assert edited["semantic_conflict_check"]["status"] == "queued"


def test_semantic_control_disable_blocks_new_enqueue(tmp_path: Path):
    tools = _tools(tmp_path)
    disabled = tools.memory_repair(task="semantic_control", data={"action": "disable"})
    assert disabled["data"]["outcome"] == "runtime_disabled"
    written = tools.memory_write(content="x", subject="s", tags=["t"])["data"]
    assert written["semantic_conflict_check"]["status"] == "runtime_disabled"
    resume = tools.memory_repair(task="semantic_control", data={"action": "resume"})
    assert resume["data"]["outcome"] == "runtime_disabled_use_enable"
    enabled = tools.memory_repair(task="semantic_control", data={"action": "enable"})
    assert enabled["data"]["outcome"] == "enabled"


def test_on_write_off_does_not_start_worker_thread(tmp_path: Path):
    # enabled=true + on_write=off must not spin up the worker thread or preload:
    # writes never enqueue under "off", so a resident worker would idle forever.
    import threading

    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "m.jsonl",
        semantic_conflict_enabled=True,
        semantic_conflict_on_write="off",
        semantic_conflict_model_path=tmp_path / "missing.gguf",
        semantic_conflict_preload=True,
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    tools.start_semantic_worker()
    before = threading.active_count()
    written = tools.memory_write(content="x", subject="s", tags=["t"])["data"]
    assert written["semantic_conflict_check"]["status"] == "off"
    assert threading.active_count() == before
    assert tools._semantic_worker.status()["runtime_state"] == "on_write_off"


def test_record_semantic_notice_fk_violation_not_masked_as_deduped(tmp_path: Path):
    # An IntegrityError with dedupe_key=None (e.g. FK violation on a missing
    # memory_id) must surface as an error, not be swallowed as "deduped".
    tools = _tools(tmp_path)
    result = tools.db.record_semantic_notice(
        memory_id=999999,  # does not exist -> FK violation
        peer_id=None,
        severity="normal",
        notice_type="semantic_pair",
        title="t",
        message="m",
        payload={},
        dedupe_key=None,
    )
    assert result["outcome"] == "error"
    assert result.get("reason") == "integrity_constraint"


def test_semantic_max_concurrency_reserved_warning(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config_path))
    monkeypatch.setenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_MAX_CONCURRENCY", "4")
    settings = Settings.from_env()
    assert settings.semantic_conflict_max_concurrency == 1
    assert any("max_concurrency is reserved" in warning for warning in settings.config_warnings)


def test_notice_dedupe_key_is_pair_symmetric():
    from memory_arbiter.semantic_conflict import notice_dedupe_key
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_pair") == notice_dedupe_key(2, 1, 4, 3, "semantic_pair")
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_pair") != notice_dedupe_key(1, 2, 4, 4, "semantic_pair")


def test_dismissed_semantic_notice_closes_same_snapshot_pair(tmp_path: Path):
    tools = _tools(tmp_path)
    left = tools.memory_write(content="默认模型推荐 MiniCPM。", subject="model choice", tags=["mema", "model"])["data"]
    right = tools.memory_write(content="默认模型改为 Qwen。", subject="model choice", tags=["mema", "model"])["data"]
    lv = tools.db.get_memory_version(left["id"]) or 1
    rv = tools.db.get_memory_version(right["id"]) or 1
    lcr = (tools.db.get_memory(left["id"]) or {}).get("claim_revision") or 1
    rcr = (tools.db.get_memory(right["id"]) or {}).get("claim_revision") or 1
    notice = tools.db.record_semantic_notice(
        memory_id=right["id"], peer_id=left["id"], severity="normal",
        notice_type="semantic_pair", title="test", message="msg", payload={},
        dedupe_key="close-test", left_version=rv, right_version=lv,
        left_claim_revision=rcr, right_claim_revision=lcr,
    )
    assert notice["outcome"] == "created"
    tools.db.update_semantic_notice_status(notice["notice_id"], "dismissed")
    assert tools.db.is_semantic_pair_closed(
        left["id"], right["id"], lv, rv,
        left_claim_revision=lcr, right_claim_revision=rcr,
    )
    edited = tools.memory_edit(memory_id=left["id"], new_content="默认模型推荐 MiniCPM，但需复核。")
    assert edited["ok"] is True
    assert not tools.db.is_semantic_pair_closed(
        left["id"], right["id"], tools.db.get_memory_version(left["id"]), rv,
        left_claim_revision=(tools.db.get_memory(left["id"]) or {}).get("claim_revision"),
        right_claim_revision=rcr,
    )


def test_worker_resume_method_does_not_clear_disabled(tmp_path: Path):
    tools = _tools(tmp_path)
    tools._semantic_worker.disable_runtime()
    tools._semantic_worker.resume()
    assert tools._semantic_worker.status()["runtime_state"] == "disabled"


def test_model_signal_parses_nested_json_object():
    # A non-greedy {.*?} regex truncates nested objects and would drop this
    # candidate. Brace-balanced extraction must recover the full payload.
    nested = 'noise {"should_surface": true, "reason_code": "value_diff", "parsed": {"k": 1, "list": [1, 2]}} tail'
    signal = model_signal_from_text(nested)
    assert signal.candidate is True
    assert signal.candidate_type == "value_diff"
    assert signal.parsed == {"should_surface": True, "reason_code": "value_diff", "parsed": {"k": 1, "list": [1, 2]}}
    assert signal.error is None


def test_model_signal_extracts_first_object_when_two_present():
    raw = '{"should_surface": false, "reason_code": "duplicate"} ... {"reason_code": "replacement"}'
    signal = model_signal_from_text(raw)
    # First balanced object wins, and duplicate forces candidate=False.
    assert signal.candidate is False
    assert signal.candidate_type == "duplicate"


def test_model_signal_braces_inside_strings_do_not_affect_depth():
    raw = '{"reason_code": "value_diff", "note": "has } and { inside"}'
    signal = model_signal_from_text(raw)
    assert signal.candidate is True
    assert signal.candidate_type == "value_diff"


def test_todo_done_is_direction_agnostic_with_common_tokens():
    # New write = "todo", peer = "done" must still pair (gate passes new as
    # left). Previously todo_done only fired left=todo/right=done.
    new_todo = pair_text_evidence("待办：完成 Console Settings 页。", "Console Settings 页已经完成并随 v0.10.3 发布。")
    new_done = pair_text_evidence("Console Settings 页已经完成并随 v0.10.3 发布。", "待办：完成 Console Settings 页。")
    assert new_todo.todo_done is True
    assert new_done.todo_done is True


def test_todo_done_requires_common_tokens_to_pair_unrelated_items():
    # An unrelated todo and an unrelated done statement must not pair just
    # because one says 待办 and the other 已完成.
    unrelated = pair_text_evidence("待办：买牛奶。", "季度报告已经完成。")
    assert unrelated.todo_done is False


def test_worker_set_error_records_under_lock_and_surfaces_in_status(tmp_path: Path):
    tools = _tools(tmp_path)
    tools._semantic_worker.set_error("boom")
    status = tools._semantic_worker.status()
    assert status["last_error"] == "boom"


def test_semantic_status_exposes_budget_and_deadline_diagnostics(tmp_path: Path):
    tools = _tools(tmp_path)
    status = tools._semantic_status()
    assert status["min_pair_budget_ms"] == 1000
    assert "job_deadline_behavior" in status
    assert status["last_pair_duration_ms"] is None


def _tools_with_timeout(tmp_path: Path, job_timeout_ms: int, min_pair_budget_ms: int) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "m.jsonl",
        semantic_conflict_enabled=True,
        semantic_conflict_model_path=tmp_path / "missing.gguf",
        semantic_conflict_job_timeout_ms=job_timeout_ms,
        semantic_conflict_min_pair_budget_ms=min_pair_budget_ms,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


class _SlowFakeBackend:
    """Returns a candidate so the gate runs; records call count for budget tests."""

    def __init__(self):
        self.calls = 0
        self.disabled = False
        self.unload_calls = 0
        self.maybe_unload_calls = 0

    def load(self):
        pass

    def classify_pair(self, left, right):
        from memory_arbiter.semantic_conflict import ModelSignal
        self.calls += 1
        if self.disabled:
            return ModelSignal(False, "backend_unavailable", None, "", None, "disabled")
        return ModelSignal(True, "replacement", 0.9, "{}", {"reason_code": "replacement"})

    def suggest_workspace_candidate(self, ws_raw, evidence, candidates):
        from memory_arbiter.semantic_conflict import WorkspaceCandidateSignal
        return WorkspaceCandidateSignal(None, "uncertain", None, "", "", "disabled" if self.disabled else None)

    def status(self):
        return {"backend": "fake", "model_state": "resident", "disabled": self.disabled}

    def unload(self, timeout=30.0, disable=False):
        self.unload_calls += 1
        self.disabled = self.disabled or bool(disable)
        return {"ok": True, "unloaded": True, "timeout": False, "inflight": 0, "retry_hint": None, "generation": self.unload_calls}

    def maybe_unload_if_idle(self):
        self.maybe_unload_calls += 1
        return {"ok": True, "unloaded": True, "inflight": 0, "generation": self.maybe_unload_calls}

    def set_disabled(self, disabled):
        self.disabled = bool(disabled)


def test_budget_guard_stops_before_starting_inference_below_floor(tmp_path: Path):
    tools = _tools_with_timeout(tmp_path, job_timeout_ms=500, min_pair_budget_ms=600)
    # Two overlapping memories so there is a candidate pair to evaluate.
    tools.memory_write(content="旧设计：vector scan 主路径。", subject="x", tags=["mema", "vector"])
    new = tools.memory_write(content="新设计：vector scan 下线。", subject="x update", tags=["mema", "vector"])["data"]
    record = tools.db.get_memory(new["id"])
    backend = _SlowFakeBackend()
    tools._semantic_backend = backend
    snapshot = {
        "memory_id": new["id"],
        "version": record["version"],
        "claim_revision": record["claim_revision"],
        "content_hash": __import__("hashlib").sha256(record["content"].encode("utf-8")).hexdigest(),
    }
    tools._process_semantic_conflict_job(new["id"], snapshot)
    # 500ms total budget < 600ms floor -> inference must never start.
    assert backend.calls == 0
    assert tools._semantic_worker.status()["last_error"] is not None
    assert "below" in tools._semantic_worker.status()["last_error"]


def test_run_loop_preserves_job_diagnostic_on_clean_return(tmp_path: Path):
    # Regression: the worker's _run finally must NOT clobber a diagnostic that
    # the current job recorded via set_error() and then returned cleanly from
    # (timeout / min-pair-budget early stop). Drive the real _run worker loop
    # (not _process_semantic_conflict_job directly) so the finally executes.
    import time

    tools = _tools_with_timeout(tmp_path, job_timeout_ms=500, min_pair_budget_ms=600)
    tools.memory_write(content="旧设计：vector scan 主路径。", subject="x", tags=["mema", "vector"])
    new = tools.memory_write(content="新设计：vector scan 下线。", subject="x update", tags=["mema", "vector"])["data"]
    record = tools.db.get_memory(new["id"])
    tools._semantic_backend = _SlowFakeBackend()
    snapshot = {
        "memory_id": new["id"],
        "version": record["version"],
        "claim_revision": record["claim_revision"],
        "content_hash": __import__("hashlib").sha256(record["content"].encode("utf-8")).hexdigest(),
    }
    worker = tools._semantic_worker
    worker._ensure_thread()
    worker.enqueue(new["id"], snapshot)
    # Wait for the worker to process one job.
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = worker.status()
        if status["processed"] >= 1 and not status["inflight"]:
            break
        time.sleep(0.02)
    status = worker.status()
    assert status["processed"] >= 1
    # The budget-floor diagnostic the job set must survive the clean return.
    assert status["last_error"] is not None
    assert "below" in status["last_error"]


def test_run_loop_clears_stale_prior_error_on_clean_run(tmp_path: Path):
    # The complementary invariant: a genuinely clean job (no self-diagnostic)
    # still clears a stale prior error so it does not linger forever.
    import time

    tools = _tools_with_timeout(tmp_path, job_timeout_ms=5000, min_pair_budget_ms=1)
    tools.memory_write(content="旧设计：vector scan 主路径。", subject="x", tags=["mema", "vector"])
    new = tools.memory_write(content="新设计：vector scan 下线。", subject="x update", tags=["mema", "vector"])["data"]
    record = tools.db.get_memory(new["id"])
    tools._semantic_backend = _SlowFakeBackend()
    snapshot = {
        "memory_id": new["id"],
        "version": record["version"],
        "claim_revision": record["claim_revision"],
        "content_hash": __import__("hashlib").sha256(record["content"].encode("utf-8")).hexdigest(),
    }
    worker = tools._semantic_worker
    worker.set_error("stale prior error")
    worker._ensure_thread()
    worker.enqueue(new["id"], snapshot)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = worker.status()
        if status["processed"] >= 1 and not status["inflight"]:
            break
        time.sleep(0.02)
    status = worker.status()
    assert status["processed"] >= 1
    assert status["last_error"] is None


def test_budget_guard_runs_inference_when_budget_is_sufficient(tmp_path: Path):
    tools = _tools_with_timeout(tmp_path, job_timeout_ms=5000, min_pair_budget_ms=1)
    tools.memory_write(content="旧设计：vector scan 主路径。", subject="x", tags=["mema", "vector"])
    new = tools.memory_write(content="新设计：vector scan 下线。", subject="x update", tags=["mema", "vector"])["data"]
    record = tools.db.get_memory(new["id"])
    backend = _SlowFakeBackend()
    tools._semantic_backend = backend
    snapshot = {
        "memory_id": new["id"],
        "version": record["version"],
        "claim_revision": record["claim_revision"],
        "content_hash": __import__("hashlib").sha256(record["content"].encode("utf-8")).hexdigest(),
    }
    tools._process_semantic_conflict_job(new["id"], snapshot)
    assert backend.calls >= 1
    assert tools._last_pair_duration_ms is not None


class _BlockingPreloadBackend:
    def __init__(self, ready, release):
        self.ready = ready
        self.release = release
        self.disabled = False

    def load(self):
        self.ready.set()
        self.release.wait(2)

    def classify_pair(self, left, right):
        from memory_arbiter.semantic_conflict import ModelSignal
        return ModelSignal(False, "unrelated", None, "{}", {})

    def suggest_workspace_candidate(self, ws_raw, evidence, candidates):
        from memory_arbiter.semantic_conflict import WorkspaceCandidateSignal
        return WorkspaceCandidateSignal(None, "uncertain", None, "", "", None)

    def status(self):
        return {"backend": "blocking", "model_state": "loading", "disabled": self.disabled}

    def unload(self, timeout=30.0, disable=False):
        self.disabled = self.disabled or bool(disable)
        return {"ok": True, "unloaded": True, "timeout": False, "inflight": 0, "retry_hint": None, "generation": 1}

    def maybe_unload_if_idle(self):
        return {"ok": True, "unloaded": False, "inflight": 0, "generation": 1}

    def set_disabled(self, disabled):
        self.disabled = bool(disabled)


def test_semantic_preload_runs_in_background(tmp_path: Path):
    import threading
    import time

    tools = _tools(tmp_path)
    tools.settings.semantic_conflict_preload = True
    ready = threading.Event()
    release = threading.Event()
    tools._semantic_backend = _BlockingPreloadBackend(ready, release)

    start = time.monotonic()
    tools.start_semantic_worker()
    elapsed = time.monotonic() - start
    try:
        assert elapsed < 0.5
        assert ready.wait(1)
        assert tools._semantic_worker.status()["runtime_state"] == "running"
    finally:
        release.set()


def test_tags_only_without_claim_semantic_change_skips_semantic_enqueue(tmp_path: Path):
    tools = _tools(tmp_path)
    written = tools.memory_write(content="固定用 Python 3.12 pytest。", subject="tests", tags=["python"])["data"]
    edited = tools.memory_edit(memory_id=written["id"], tags_only=True, add_tags=["pytest"])["data"]
    assert edited["edited"] is True
    assert edited["semantic_conflict_check"] == {
        "status": "skipped",
        "reason": "tags_only_no_semantic_change",
    }


def test_unpinned_dismissed_semantic_notice_does_not_close_versioned_pair(tmp_path: Path):
    tools = _tools(tmp_path)
    left = tools.memory_write(content="默认模型推荐 MiniCPM。", subject="model choice", tags=["mema", "model"])["data"]
    right = tools.memory_write(content="默认模型改为 Qwen。", subject="model choice", tags=["mema", "model"])["data"]
    notice = tools.db.record_semantic_notice(
        memory_id=right["id"], peer_id=left["id"], severity="normal",
        notice_type="semantic_pair", title="legacy", message="msg", payload={},
        dedupe_key="legacy-unpinned",
    )
    assert notice["outcome"] == "created"
    tools.db.update_semantic_notice_status(notice["notice_id"], "dismissed")
    lv = tools.db.get_memory_version(left["id"]) or 1
    rv = tools.db.get_memory_version(right["id"]) or 1
    assert not tools.db.is_semantic_pair_closed(left["id"], right["id"], lv, rv)
    assert not tools.db.is_semantic_pair_closed(left["id"], right["id"])


def test_activate_pending_requeues_semantic_check(tmp_path: Path):
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "m.jsonl",
        semantic_conflict_enabled=True,
        semantic_conflict_model_path=tmp_path / "missing.gguf",
        isolation="strict",
        workspace="default",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    pending = tools.memory_write(
        content="新工作区默认模型推荐 MiniCPM。",
        subject="model choice",
        tags=["mema", "model"],
        workspace="brand-new-workspace",
    )["data"]
    assert pending["record"]["status"] == "pending"

    activated = tools.memory_activate(memory_id=pending["id"], authorized=True)["data"]
    assert activated["activated"] is True
    assert activated["record"]["status"] == "active"
    assert activated["semantic_conflict_check"]["status"] == "queued"


class _FakeLlama:
    def __init__(self, ready=None, release=None, raw='{"should_surface": true, "reason_code": "replacement", "confidence": 0.9}'):
        self.ready = ready
        self.release = release
        self.raw = raw
        self.calls = 0

    def create_chat_completion(self, **_kwargs):
        self.calls += 1
        if self.ready is not None:
            self.ready.set()
        if self.release is not None:
            self.release.wait(2)
        return {"choices": [{"message": {"content": self.raw}}]}


def _local_backend_with_llm(tmp_path: Path, llm):
    from memory_arbiter.semantic_conflict import LocalGGUFSemanticBackend
    model = tmp_path / "model.gguf"
    model.write_text("fake", encoding="utf-8")
    backend = LocalGGUFSemanticBackend(model)
    with backend._cond:
        backend._llm = llm
        backend._loaded_at = time.time()
    return backend


def test_local_backend_unload_waits_for_inflight_barrier(tmp_path: Path):
    ready = threading.Event()
    release = threading.Event()
    llm = _FakeLlama(ready=ready, release=release)
    backend = _local_backend_with_llm(tmp_path, llm)
    signal_holder = {}

    t = threading.Thread(target=lambda: signal_holder.setdefault("signal", backend.classify_pair({"content": "旧设计"}, {"content": "新设计"})))
    t.start()
    assert ready.wait(1)
    started = time.monotonic()
    unload_holder = {}
    u = threading.Thread(target=lambda: unload_holder.setdefault("result", backend.unload(timeout=1.0)))
    u.start()
    time.sleep(0.05)
    assert u.is_alive()
    assert backend.status()["inflight"] == 1
    release.set()
    t.join(1)
    u.join(1)
    assert unload_holder["result"]["ok"] is True
    assert unload_holder["result"]["timeout"] is False
    assert backend.status()["model_state"] == "unloaded"
    assert time.monotonic() - started >= 0.05
    assert signal_holder["signal"].candidate is True


def test_local_backend_unload_timeout_keeps_llm_until_retry(tmp_path: Path):
    ready = threading.Event()
    release = threading.Event()
    llm = _FakeLlama(ready=ready, release=release)
    backend = _local_backend_with_llm(tmp_path, llm)
    t = threading.Thread(target=lambda: backend.classify_pair({"content": "旧设计"}, {"content": "新设计"}))
    t.start()
    assert ready.wait(1)
    result = backend.unload(timeout=0.01)
    assert result["ok"] is False
    assert result["timeout"] is True
    assert result["inflight"] == 1
    assert backend.status()["model_state"] == "resident"
    release.set()
    t.join(1)
    retry = backend.unload(timeout=1.0)
    assert retry["ok"] is True
    assert backend.status()["model_state"] == "unloaded"


def test_local_backend_disable_then_enable_restores_jobs(tmp_path: Path):
    backend = _local_backend_with_llm(tmp_path, _FakeLlama())
    disabled = backend.unload(timeout=1.0, disable=True)
    assert disabled["ok"] is True
    signal = backend.classify_pair({"content": "a"}, {"content": "b"})
    assert signal.candidate is False
    assert signal.candidate_type == "backend_unavailable"
    backend.set_disabled(False)
    with backend._cond:
        backend._llm = _FakeLlama()
    signal2 = backend.classify_pair({"content": "旧设计"}, {"content": "新设计"})
    assert signal2.candidate is True


def test_local_backend_suggest_workspace_lifecycle_disable(tmp_path: Path):
    backend = _local_backend_with_llm(tmp_path, _FakeLlama(raw='{"candidate": "金营项目", "relation": "alias", "confidence": 0.9, "evidence": "同项目"}'))
    sig = backend.suggest_workspace_candidate("金营", {"title": "t", "key_sentences": ["k"]}, ["金营项目"])
    assert sig.candidate == "金营项目"
    backend.unload(timeout=1.0, disable=True)
    disabled = backend.suggest_workspace_candidate("金营", {}, ["金营项目"])
    assert disabled.candidate is None
    assert disabled.error == "disabled"


def test_semantic_control_disable_enable_toggles_backend(tmp_path: Path):
    tools = _tools(tmp_path)
    backend = _FakeSemanticBackend()
    tools._semantic_backend = backend
    disabled = tools.memory_repair(task="semantic_control", data={"action": "disable", "timeout": 1})["data"]
    assert disabled["outcome"] == "runtime_disabled"
    assert disabled["unload"]["ok"] is True
    assert backend.disabled is True
    enabled = tools.memory_repair(task="semantic_control", data={"action": "enable"})["data"]
    assert enabled["outcome"] == "enabled"
    assert backend.disabled is False


def test_worker_shutdown_discards_pending_and_waits_only_inflight(tmp_path: Path):
    tools = _tools(tmp_path)
    worker = tools._semantic_worker
    with worker._cond:
        worker._pending[1] = {"memory_id": 1}
        worker._pending[2] = {"memory_id": 2}
    result = worker.shutdown(discard_pending=True)
    assert result["discarded_pending"] == 2
    assert worker.wait_drained(0.1) is True
    assert worker.enqueue(3, {"memory_id": 3})["status"] == "shutdown"


def test_memory_tools_shutdown_is_idempotent_and_unloads_backend(tmp_path: Path):
    tools = _tools(tmp_path)
    backend = _FakeSemanticBackend()
    tools._semantic_backend = backend
    first = tools.shutdown(timeout=1)
    second = tools.shutdown(timeout=1)
    assert first["ok"] is True
    assert first["backend_unload"]["ok"] is True
    assert backend.unload_calls == 1
    assert second == {"ok": True, "already_shutdown": True}


def test_resident_false_uses_nonblocking_idle_cleanup(tmp_path: Path):
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "m.jsonl",
        semantic_conflict_enabled=True,
        semantic_conflict_model_path=tmp_path / "missing.gguf",
        semantic_conflict_resident=False,
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    tools.memory_write(content="旧设计：vector scan 主路径。", subject="x", tags=["mema", "vector"])
    new = tools.memory_write(content="新设计：vector scan 下线。", subject="x update", tags=["mema", "vector"])["data"]
    record = tools.db.get_memory(new["id"])
    backend = _FakeSemanticBackend()
    tools._semantic_backend = backend
    snapshot = {
        "memory_id": new["id"],
        "version": record["version"],
        "claim_revision": record["claim_revision"],
        "content_hash": __import__("hashlib").sha256(record["content"].encode("utf-8")).hexdigest(),
    }
    tools._process_semantic_conflict_job(new["id"], snapshot)
    assert backend.maybe_unload_calls >= 1
    assert backend.unload_calls == 0


def test_managed_embedder_has_no_unload_reload_or_llm_surface():
    from memory_arbiter.embedder import ManagedEmbedder
    assert not hasattr(ManagedEmbedder, "unload")
    assert not hasattr(ManagedEmbedder, "reload")
    assert not hasattr(ManagedEmbedder, "_llm")
