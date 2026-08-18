"""MCP server exposing the four product surfaces."""
from __future__ import annotations

import atexit
import signal
import sys
from typing import Any, NamedTuple, Optional

from .config import Settings
from .tools import MemoryTools


class ServerBundle(NamedTuple):
    app: Any
    tools: MemoryTools


def build_runtime() -> ServerBundle:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError("MCP Python SDK is not installed") from exc

    app = FastMCP("memory-arbiter-mcp")
    tools = MemoryTools(Settings.from_env())
    tools.start_update_monitor()
    tools.start_evidence_worker()
    tools.start_semantic_worker()

    @app.tool()
    def memory(action: str = "help", data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Daily memory operations: remember, find, read, update, judge, status, help."""
        return tools.memory(action=action, data={} if data is None else data)

    @app.tool()
    def memory_review(view: str = "help", data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Read-only inspection: overview, doctor, conflicts, judgments, history, audit, help."""
        return tools.memory_review(view=view, data={} if data is None else data)

    @app.tool()
    def memory_govern(action: str = "help", data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Explicitly authorized governance for memories, conflicts, and workspaces."""
        return tools.memory_govern(action=action, data={} if data is None else data)

    @app.tool()
    def memory_repair(task: str = "help", data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Maintenance: history cleanup, evidence rebuild, backup replay, notices, and runtime control."""
        return tools.memory_repair(task=task, data={} if data is None else data)

    return ServerBundle(app, tools)


def run() -> None:
    build_runtime().app.run()


def build_server() -> Any:
    """Return the FastMCP app for embedders and tests."""
    return build_runtime().app


def main() -> None:
    if len(sys.argv) > 1:
        command = sys.argv[1].replace("_", "-")
        if command == "doctor":
            from .doctor_cli import run_cli

            run_cli(sys.argv[2:])
            return
        if command == "setup":
            from .setup_cli import run_cli

            raise SystemExit(run_cli(sys.argv[2:]))
        if command == "console":
            from .console_cli import run_cli

            raise SystemExit(run_cli(sys.argv[2:]))
        if command == "migrate-vnext":
            from .vnext_migration import run_cli

            raise SystemExit(run_cli(sys.argv[2:]))
        if command in {"-h", "--help", "help"}:
            print("Usage: mema [doctor|setup|console|migrate-vnext]")
            return

    try:
        bundle = build_runtime()

        def shutdown_runtime(*_: Any) -> None:
            bundle.tools.shutdown(timeout=30.0)

        atexit.register(shutdown_runtime)

        def handle_signal(signum: int, _frame: Any) -> None:
            shutdown_runtime()
            raise SystemExit(128 + int(signum))

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        bundle.app.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
