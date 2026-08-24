from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import memory_arbiter.post_release_smoke as production_smoke

ROOT = Path(__file__).resolve().parent.parent

LOCAL_HIDDEN_DIRS = (
    ".claude", ".codex", ".cursor", ".idea", ".vscode", ".windsurf",
    ".workbuddy", ".zcode",
)


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_local_hidden_state_is_ignored_and_pruned_from_sdists():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert ".env.*" in gitignore
    assert "!.env.example" in gitignore
    assert "include .env.example" in manifest
    for directory in LOCAL_HIDDEN_DIRS:
        assert f"{directory}/" in gitignore
        assert f"prune {directory}" in manifest
    assert "global-exclude .DS_Store Thumbs.db" in manifest


def test_workflow_python_script_dependencies_are_tracked_when_in_git_repo():
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        return
    workflows = ROOT / ".github" / "workflows"
    referenced: set[str] = set()
    for workflow in workflows.glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        for token in text.replace('"', " ").replace("'", " ").split():
            token = token.strip("()|&;\\")
            marker = "$GITHUB_WORKSPACE/"
            if token.startswith(marker):
                token = token[len(marker):]
            if token.startswith("scripts/") and token.endswith(".py"):
                referenced.add(token)
    assert referenced, "no workflow Python script dependencies discovered"
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *sorted(referenced)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "workflow Python dependencies must be tracked:\n" + proc.stdout + proc.stderr
    )


def test_release_metadata_is_consistent_and_check_is_read_only():
    tracked = [ROOT / "server.json", ROOT / "CHANGELOG.md", ROOT / "uv.lock"]
    before = {path: _digest(path) for path in tracked}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_version.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert {path: _digest(path) for path in tracked} == before


def test_sync_version_detects_manifest_changelog_and_lock_drift(monkeypatch, tmp_path):
    sync = _load_script("sync_version_test", "scripts/sync_version.py")
    server = tmp_path / "server.json"
    changelog = tmp_path / "CHANGELOG.md"
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    server.write_text(json.dumps({"version": "0", "packages": [{"version": "0"}]}), encoding="utf-8")
    changelog.write_text("# Changelog\n\n## [0.12.5] — 2026-08-10\n", encoding="utf-8")
    pyproject.write_text('[project]\nname="memory-arbiter-mcp"\n[project.optional-dependencies]\ntest=[]\nvec=[]\nsemantic-local=[]\n', encoding="utf-8")
    lock.write_text('version=1\n[[package]]\nname="memory-arbiter-mcp"\nversion="0.12.5"\nsource={editable="."}\n[package.optional-dependencies]\ntest=[]\n[package.metadata]\nprovides-extras=["test"]\n', encoding="utf-8")
    monkeypatch.setattr(sync, "SERVER_JSON", server)
    monkeypatch.setattr(sync, "CHANGELOG", changelog)
    monkeypatch.setattr(sync, "PYPROJECT", pyproject)
    monkeypatch.setattr(sync, "UV_LOCK", lock)
    assert not sync.sync_server_json("0.13.0", check=True)
    assert not sync.check_changelog("0.13.0")
    assert not sync.check_uv_lock("0.13.0")


def test_development_version_output_explains_lock_and_release_manifests(monkeypatch, capsys):
    sync = _load_script("sync_version_development_output", "scripts/sync_version.py")
    monkeypatch.setattr(sync, "read_authoritative_version", lambda: "0.14.0.dev1")
    monkeypatch.setattr(sync, "check_uv_lock", lambda _version: True)
    monkeypatch.setattr(sys, "argv", ["sync_version.py", "--check"])
    assert sync.main() == 0
    output = capsys.readouterr().out
    assert "uv.lock is valid" in output
    assert "release manifests intentionally remain at the latest release version" in output


def test_release_version_check_remains_strict(monkeypatch):
    sync = _load_script("sync_version_release_strict", "scripts/sync_version.py")
    calls: list[str] = []
    monkeypatch.setattr(sync, "read_authoritative_version", lambda: "0.14.0")
    monkeypatch.setattr(sync, "sync_server_json", lambda version, check: calls.append(f"manifest:{version}:{check}") or False)
    monkeypatch.setattr(sync, "check_changelog", lambda version: calls.append(f"changelog:{version}") or False)
    monkeypatch.setattr(sync, "check_uv_lock", lambda version: calls.append(f"lock:{version}") or True)
    monkeypatch.setattr(sys, "argv", ["sync_version.py", "--check"])
    assert sync.main() == 1
    assert calls == ["manifest:0.14.0:True", "changelog:0.14.0", "lock:0.14.0"]


def test_uv_lock_dynamic_version_record_is_valid(monkeypatch, tmp_path):
    sync = _load_script("sync_version_dynamic_lock", "scripts/sync_version.py")
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    pyproject.write_text(
        '[project]\nname="memory-arbiter-mcp"\n[project.optional-dependencies]\ntest=[]\nvec=[]\nsemantic-local=[]\n',
        encoding="utf-8",
    )
    lock.write_text(
        'version=1\n[[package]]\nname="memory-arbiter-mcp"\nsource={editable="."}\n'
        '[package.optional-dependencies]\ntest=[]\nvec=[]\nsemantic-local=[]\n'
        '[package.metadata]\nprovides-extras=["test","vec","semantic-local"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sync, "PYPROJECT", pyproject)
    monkeypatch.setattr(sync, "UV_LOCK", lock)
    assert sync.check_uv_lock("0.13.1")


def test_production_smoke_version_mismatch_has_no_runtime_side_effect(monkeypatch):
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("runtime must not initialize")

    monkeypatch.setattr(production_smoke.Settings, "from_env", forbidden)
    monkeypatch.setattr(sys, "argv", ["mema-production-smoke", "--expected-version", "999.0"])
    assert production_smoke.main() == 2
    assert not called


def test_production_smoke_always_cleans_up(monkeypatch):
    events: list[str] = []

    class FakeSettings:
        workspace = "release-test"

    class FakeTools:
        def __init__(self, settings):
            events.append("init")

        def memory_write(self, **kwargs):
            events.append("write")
            return {"ok": True, "data": {"id": 7}}

        def memory_get(self, **kwargs):
            events.append("read")
            raise RuntimeError("forced read failure")

        def memory_supersede(self, **kwargs):
            events.append("retire")
            return {"ok": True}

        def memory_search_expired(self, **kwargs):
            events.append("expired")
            return {"data": {"results": [{"id": 7}]}}

        def shutdown(self, **kwargs):
            events.append("shutdown")
            return {"ok": True}

    monkeypatch.setattr(production_smoke.Settings, "from_env", lambda: FakeSettings())
    monkeypatch.setattr(production_smoke, "MemoryTools", FakeTools)
    monkeypatch.setattr(sys, "argv", ["mema-production-smoke", "--expected-version", production_smoke.__version__])
    assert production_smoke.main() == 1
    assert events == ["init", "write", "read", "retire", "expired", "shutdown"]
