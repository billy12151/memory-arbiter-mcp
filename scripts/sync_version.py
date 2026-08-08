#!/usr/bin/env python3
"""Single source of truth for the package version.

The authoritative version lives in ``memory_arbiter/__init__.py`` as
``__version__``. ``pyproject.toml`` reads it dynamically via
``[tool.setuptools.dynamic]``. Files that cannot reference Python
(``server.json``, the MCP registry manifest) are kept in sync by this script.

Usage:
    python scripts/sync_version.py           # rewrite server.json to match __version__
    python scripts/sync_version.py --check    # exit non-zero if anything is out of sync
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "memory_arbiter" / "__init__.py"
SERVER_JSON = ROOT / "server.json"


def read_authoritative_version() -> str:
    text = INIT.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"could not find __version__ in {INIT}")
    return match.group(1)


def collect_server_json_versions(data: dict) -> list[str]:
    versions = [data.get("version")]
    for package in data.get("packages", []):
        versions.append(package.get("version"))
    return [v for v in versions if v is not None]


def sync_server_json(version: str, *, check: bool) -> bool:
    data = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    current = collect_server_json_versions(data)
    in_sync = all(v == version for v in current)
    if check:
        if not in_sync:
            print(
                f"server.json version(s) {current} != authoritative {version}",
                file=sys.stderr,
            )
        return in_sync
    if in_sync:
        return True
    data["version"] = version
    for package in data.get("packages", []):
        if "version" in package:
            package["version"] = version
    SERVER_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"updated server.json -> {version}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify sync without writing; non-zero exit if out of sync",
    )
    args = parser.parse_args()

    version = read_authoritative_version()
    ok = sync_server_json(version, check=args.check)
    if args.check:
        if ok:
            print(f"version in sync: {version}")
            return 0
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
