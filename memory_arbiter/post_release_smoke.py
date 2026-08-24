"""Optional post-release smoke against the configured production database.

Run the installed ``mema-production-smoke`` entry point only after installing
the requested PyPI version in the dedicated Python 3.13 production environment
and restarting the MCP host. This script is not a
release gate. It always attempts to retire the temporary record.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from memory_arbiter import __version__
from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools

PRODUCTION_SMOKE_WORKSPACE = "测试"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    if __version__ != args.expected_version:
        print(f"version mismatch: installed={__version__} expected={args.expected_version}", file=sys.stderr)
        return 2
    settings = Settings.from_env()
    tools = MemoryTools(settings)
    marker = f"release-smoke:{__version__}:{datetime.now(timezone.utc).isoformat()}"
    memory_id = None
    failed = False
    try:
        written = tools.memory_write(
            content=marker,
            subject=f"mema PyPI production smoke {__version__}",
            tags=["release-smoke", __version__],
            source_type="agent_generated",
            source_ref="pypi-production-smoke",
            workspace=PRODUCTION_SMOKE_WORKSPACE,
        )
        if not written.get("ok") or written.get("data", {}).get("backup_only"):
            raise RuntimeError(f"write failed or backup-only: {written}")
        memory_id = int(written["data"]["id"])
        read = tools.memory_get(memory_id=memory_id, sections="none")
        if read.get("data", {}).get("memory", {}).get("content") != marker:
            raise RuntimeError("read-back mismatch")
        found = tools.memory_search(query=marker, limit=5)
        if memory_id not in [item.get("id") for item in found.get("data", {}).get("results", [])]:
            raise RuntimeError("active search did not return smoke record")
    except Exception as exc:
        failed = True
        print(f"production smoke failed: {exc}", file=sys.stderr)
    finally:
        if memory_id is not None:
            retired = tools.memory_supersede(
                memory_id=memory_id,
                reason="post-release production smoke cleanup",
                authorized=True,
            )
            if not retired.get("ok"):
                failed = True
                print(f"SMOKE RECORD LEFT ACTIVE: memory_id={memory_id} result={retired}", file=sys.stderr)
            else:
                expired = tools.memory_search_expired(query=marker, limit=5)
                if memory_id not in [item.get("id") for item in expired.get("data", {}).get("results", [])]:
                    failed = True
                    print(f"retired smoke record not found in expired recall: memory_id={memory_id}", file=sys.stderr)
        shutdown = tools.shutdown(timeout=5.0)
        if not shutdown.get("ok"):
            failed = True
            print(f"production smoke runtime shutdown failed: {shutdown}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
