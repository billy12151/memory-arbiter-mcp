"""MCP server exposing the four product surfaces."""
from __future__ import annotations

import atexit
import os
import signal
import sys
from typing import Any, NamedTuple, Optional

from .config import Settings
from .db_generation import database_startup_lock, require_current_or_new_database
from .tools import MemoryTools


class ServerBundle(NamedTuple):
    app: Any
    tools: MemoryTools


def build_runtime() -> ServerBundle:
    settings = Settings.from_env()
    # Same startup lock as MemoryDB.__init__: never race another first-start's
    # in-flight schema creation with this pre-flight generation check.
    with database_startup_lock(settings.db_path):
        require_current_or_new_database(settings.db_path)
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError("MCP Python SDK is not installed") from exc

    app = FastMCP("memory-arbiter-mcp")
    tools = MemoryTools(settings)
    tools.start_update_monitor()
    tools.start_evidence_worker()
    tools.start_semantic_worker()

    @app.tool()
    def memory(action: str = "help", data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Daily memory operations: remember, find, read, update, judge, status, help.

        Call memory(action="help") to discover accepted fields, judge requirements,
        value enums, update modes, and action_required paths before relying on a
        result that requests attention.
        """
        return tools.memory(action=action, data={} if data is None else data)

    @app.tool()
    def memory_review(view: str = "help", data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Read-only inspection: overview, doctor, conflicts, judgments, history, expired memory, audit, entities, help.

        Use memory_review(view="help") for accepted fields. Use conflict_detail
        before formal judgments so both sides and snapshot context are visible.
        """
        return tools.memory_review(view=view, data={} if data is None else data)

    @app.tool()
    def memory_govern(action: str = "help", data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Explicitly authorized governance for memories, conflicts, and workspaces.

        State-changing governance requires the user to authorize that exact action;
        then retry with authorized=true. Call memory_govern(action="help") for
        impact notes, accepted fields, and confirm vs confirm_pending_workspace.
        """
        return tools.memory_govern(action=action, data={} if data is None else data)

    @app.tool()
    def memory_repair(task: str = "help", data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Maintenance: evidence rebuild, history cleanup, entity assignment, pending activation, backup replay, notices, and semantic runtime control.

        Use memory_repair(task="help") for notice handling and semantic_control
        actions. Semantic notices are advisory; read both memories before dismiss
        or resolve, and never pass a notice directly to judge or resolve_conflict.
        """
        return tools.memory_repair(task=task, data={} if data is None else data)

    return ServerBundle(app, tools)


def run() -> None:
    build_runtime().app.run()


def build_server() -> Any:
    """Return the FastMCP app for embedders and tests."""
    return build_runtime().app


def _terminate_after_shutdown(signum: int, shutdown: Any) -> None:
    """Clean up, then terminate without entering Python finalization.

    FastMCP's stdio reader can be blocked in an AnyIO worker thread. Raising
    ``SystemExit`` from a signal handler lets interpreter finalization race
    that thread while it still owns a buffered-I/O lock, which makes CPython
    abort in ``_enter_buffered_busy``. After application workers are drained,
    restore the signal's default action and re-deliver it so the OS terminates
    the process directly instead.
    """
    signal.signal(signum, signal.SIG_IGN)
    shutdown()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
    os._exit(128 + int(signum))  # pragma: no cover - fallback if kill returns


def main() -> None:
    if len(sys.argv) > 1:
        command = sys.argv[1].replace("_", "-")
        if command == "doctor":
            from .doctor_cli import run_cli as doctor_main

            doctor_main(sys.argv[2:])
            return
        if command == "setup":
            from .setup_cli import run_cli as setup_main

            raise SystemExit(setup_main(sys.argv[2:]))
        if command == "console":
            from .console_cli import run_cli as console_main

            raise SystemExit(console_main(sys.argv[2:]))
        if command == "migrate-vnext":
            from .vnext_migration import run_cli as migrate_main

            raise SystemExit(migrate_main(sys.argv[2:]))
        if command == "upgrade":
            from .upgrade_cli import run_cli as upgrade_main

            raise SystemExit(upgrade_main(sys.argv[2:]))
        if command in {"-h", "--help", "help"}:
            print("Usage: mema [doctor|setup|console|upgrade|migrate-vnext]")
            return

    try:
        bundle = build_runtime()

        def shutdown_runtime(*_: Any) -> None:
            bundle.tools.shutdown(timeout=30.0)

        atexit.register(shutdown_runtime)

        def handle_signal(signum: int, _frame: Any) -> None:
            _terminate_after_shutdown(signum, shutdown_runtime)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        bundle.app.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
