#!/usr/bin/env python3
"""Synchronize and validate release version metadata.

``memory_arbiter.__version__`` is authoritative. Normal mode updates the MCP
registry manifest and the release strings in the user-facing docs. ``--check``
is read-only and also validates the newest CHANGELOG release and the editable
project record/extras in ``uv.lock``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "memory_arbiter" / "__init__.py"
SERVER_JSON = ROOT / "server.json"
CHANGELOG = ROOT / "CHANGELOG.md"
PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"

# User-facing docs that claim the current release version. Each pattern must
# match exactly the version-bearing phrase; a reworded doc fails the check
# instead of silently losing coverage.
DOC_RELEASE_PATTERNS: tuple[tuple[Path, str], ...] = (
    (ROOT / "README.md", r"Current release: `(\d+\.\d+\.\d+)`"),
    (ROOT / "README.zh-CN.md", r"当前正式版本 `(\d+\.\d+\.\d+)`"),
    (ROOT / "INTRO.md", r"当前文档对应 `(\d+\.\d+\.\d+)` 正式版本"),
    (ROOT / "docs" / "INTEGRATION.md", r"describes the `(\d+\.\d+\.\d+)` contract"),
    (ROOT / "docs" / "INTEGRATION.zh-CN.md", r"本指南描述 `(\d+\.\d+\.\d+)` 的正式契约"),
)


def read_authoritative_version() -> str:
    text = INIT.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"could not find __version__ in {INIT}")
    return match.group(1)


def collect_server_json_versions(data: dict) -> list[str]:
    versions = [data.get("version")]
    versions.extend(package.get("version") for package in data.get("packages", []))
    return [value for value in versions if value is not None]


def sync_server_json(version: str, *, check: bool) -> bool:
    data = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    current = collect_server_json_versions(data)
    expected_count = 1 + len(data.get("packages", []))
    in_sync = len(current) == expected_count and all(value == version for value in current)
    if check:
        if not in_sync:
            print(f"server.json version(s) {current} != authoritative {version}", file=sys.stderr)
        return in_sync
    if in_sync:
        return True
    data["version"] = version
    for package in data.get("packages", []):
        package["version"] = version
    SERVER_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated server.json -> {version}")
    return True


def check_changelog(version: str) -> bool:
    text = CHANGELOG.read_text(encoding="utf-8")
    first_release = re.search(r"^## \[([^]]+)]\s+[—-]\s+\d{4}-\d{2}-\d{2}\s*$", text, re.MULTILINE)
    ok = first_release is not None and first_release.group(1) == version
    if not ok:
        found = first_release.group(1) if first_release else "missing"
        print(f"CHANGELOG newest release {found} != authoritative {version}", file=sys.stderr)
    return ok


def sync_doc_release_strings(version: str, *, check: bool) -> bool:
    ok = True
    for path, pattern in DOC_RELEASE_PATTERNS:
        text = path.read_text(encoding="utf-8")
        match = re.search(pattern, text)
        try:
            display = path.relative_to(ROOT)
        except ValueError:
            display = path
        if match is None:
            print(
                f"release version string missing in {display} (expected pattern: {pattern})",
                file=sys.stderr,
            )
            ok = False
        elif match.group(1) != version:
            if check:
                print(
                    f"{display} claims release {match.group(1)} != authoritative {version}",
                    file=sys.stderr,
                )
                ok = False
            else:
                path.write_text(text[: match.start(1)] + version + text[match.end(1):], encoding="utf-8")
                print(f"updated {display} -> {version}")
    return ok


def _project_lock_record(lock_data: dict) -> dict | None:
    for package in lock_data.get("package", []):
        if package.get("name") == "memory-arbiter-mcp" and package.get("source", {}).get("editable") == ".":
            return package
    return None


def check_uv_lock(version: str) -> bool:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    expected_extras = set(project.get("optional-dependencies", {}))
    lock_data = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    record = _project_lock_record(lock_data)
    if record is None:
        print("uv.lock has no editable memory-arbiter-mcp project record", file=sys.stderr)
        return False
    actual_version = record.get("version")
    actual_extras = set(record.get("optional-dependencies", {}))
    metadata_extras = set(record.get("metadata", {}).get("provides-extras", []))
    version_ok = actual_version is None or actual_version == version
    ok = version_ok and actual_extras == expected_extras and metadata_extras == expected_extras
    if not version_ok:
        print(f"uv.lock project version {actual_version} != authoritative {version}", file=sys.stderr)
    if actual_extras != expected_extras:
        print(f"uv.lock optional dependencies {sorted(actual_extras)} != pyproject extras {sorted(expected_extras)}", file=sys.stderr)
    if metadata_extras != expected_extras:
        print(f"uv.lock provides-extras {sorted(metadata_extras)} != pyproject extras {sorted(expected_extras)}", file=sys.stderr)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify all release metadata without writing")
    args = parser.parse_args()

    version = read_authoritative_version()
    development_version = ".dev" in version
    if not args.check:
        if development_version:
            if not check_uv_lock(version):
                return 1
            print(
                f"development version {version}: uv.lock is valid; release manifests "
                "intentionally remain at the latest release version"
            )
            return 0
        sync_server_json(version, check=False)
        return 0 if sync_doc_release_strings(version, check=False) else 1

    checks = (
        True if development_version else sync_server_json(version, check=True),
        True if development_version else check_changelog(version),
        True if development_version else sync_doc_release_strings(version, check=True),
        check_uv_lock(version),
    )
    if all(checks):
        if development_version:
            print(
                f"development version {version}: uv.lock is valid; release manifests "
                "intentionally remain at the latest release version"
            )
        else:
            print(f"release metadata in sync: {version}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
