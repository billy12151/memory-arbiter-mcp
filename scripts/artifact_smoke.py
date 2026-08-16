#!/usr/bin/env python3
"""Smoke an installed wheel without starting the MCP server."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.resources

PACKAGE = "memory-arbiter-mcp"
EXPECTED_ENTRY_POINTS = {
    "memory-arbiter-mcp": "memory_arbiter.server:main",
    "memory-arbiter": "memory_arbiter.server:main",
    "mema": "memory_arbiter.server:main",
    "mema-production-smoke": "memory_arbiter.post_release_smoke:main",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    import memory_arbiter

    installed = importlib.metadata.version(PACKAGE)
    assert installed == args.expected_version, (installed, args.expected_version)
    assert memory_arbiter.__version__ == args.expected_version
    scripts = {
        entry.name: entry.value
        for entry in importlib.metadata.entry_points(group="console_scripts")
        if entry.name in EXPECTED_ENTRY_POINTS
    }
    assert scripts == EXPECTED_ENTRY_POINTS, scripts
    guide = importlib.resources.files("memory_arbiter").joinpath("AGENT_ONBOARDING.md")
    assert guide.is_file() and guide.read_text(encoding="utf-8").strip()
    print(f"artifact smoke passed: {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
