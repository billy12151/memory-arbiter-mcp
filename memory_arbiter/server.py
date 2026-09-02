"""MCP server exposing the four product surfaces."""
from __future__ import annotations

import atexit
import json
import os
import signal
import sys
from typing import Any, Awaitable, Callable, MutableMapping, NamedTuple

from . import __version__
from .config import Settings
from .constants import MCP_HTTP_BODY_LIMIT, MCP_HTTP_PATH
from .request_identity import (
    AGENT_ID_HEADER,
    CLIENT_HEADER,
    IdentityHeaderError,
    RequestIdentity,
    get_request_identity,
    host_header_name,
    is_loopback_host,
    parse_identity_headers,
    request_identity_scope,
)
from .tools import MemoryTools

ASGIScope = MutableMapping[str, Any]
ASGIReceive = Callable[[], Awaitable[MutableMapping[str, Any]]]
ASGISend = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]


class ServerBundle(NamedTuple):
    app: Any
    tools: MemoryTools


class MemoryIdentityMiddleware:
    """Require advisory caller identity on the localhost HTTP MCP endpoint."""

    def __init__(
        self, app: ASGIApp, *, path: str = "/mcp", max_request_body_size: int = 4 * 1024 * 1024,
    ) -> None:
        self.app = app
        self.path = path.rstrip("/") or "/"
        self.max_request_body_size = max_request_body_size

    def _matches(self, scope: ASGIScope) -> bool:
        path = str(scope.get("path") or "")
        return path == self.path or (self.path != "/" and path == self.path + "/")

    @staticmethod
    async def _json_error(
        send: ASGISend, *, status: int, error: str, message: str,
    ) -> None:
        body = json.dumps(
            {"ok": False, "error": error, "message": message},
            ensure_ascii=True,
        ).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _headers(scope: ASGIScope) -> dict[str, list[str]]:
        headers: dict[str, list[str]] = {}
        for raw_name, raw_value in scope.get("headers") or []:
            name = bytes(raw_name).decode("latin-1").casefold()
            value = bytes(raw_value).decode("latin-1")
            headers.setdefault(name, []).append(value)
        return headers

    async def __call__(self, scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        if scope.get("type") == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope.get("type") != "http":
            await self._json_error(
                send, status=404, error="not_found", message="HTTP endpoint not found."
            )
            return
        if not self._matches(scope):
            await self._json_error(
                send, status=404, error="not_found", message="HTTP endpoint not found."
            )
            return

        client = scope.get("client")
        client_host = str(client[0]) if isinstance(client, (tuple, list)) and client else ""
        if not is_loopback_host(client_host):
            await self._json_error(
                send,
                status=403,
                error="localhost_required",
                message=(
                    "Community Streamable HTTP MCP accepts localhost callers only; "
                    "X-Mema-* headers are advisory provenance, not authentication."
                ),
            )
            return

        raw_headers = self._headers(scope)
        host_values = raw_headers.get("host", [])
        if (
            len(host_values) != 1
            or not is_loopback_host(host_header_name(host_values[0]) or "")
        ):
            await self._json_error(
                send,
                status=421,
                error="invalid_host",
                message="Host must identify localhost for Community Streamable HTTP MCP.",
            )
            return
        content_length_values = raw_headers.get("content-length", [])
        if content_length_values:
            try:
                content_length = int(content_length_values[0]) if len(content_length_values) == 1 else -1
            except ValueError:
                content_length = -1
            if content_length < 0:
                await self._json_error(
                    send, status=400, error="invalid_content_length",
                    message="Content-Length must be one non-negative integer.",
                )
                return
            if content_length > self.max_request_body_size:
                await self._json_error(
                    send, status=413, error="request_too_large",
                    message=f"Request body exceeds {self.max_request_body_size} bytes.",
                )
                return

        class ScopeHeaders(dict[str, str]):
            def getlist(self, name: str) -> list[str]:
                return list(raw_headers.get(name.casefold(), []))

        headers = ScopeHeaders({name: values[0] for name, values in raw_headers.items() if values})
        try:
            identity = parse_identity_headers(headers)
        except IdentityHeaderError as exc:
            await self._json_error(
                send, status=400, error="invalid_mema_identity", message=str(exc),
            )
            return

        downstream_receive = receive
        if str(scope.get("method") or "").upper() in {"POST", "PUT", "PATCH"}:
            buffered: list[MutableMapping[str, Any]] = []
            received_bytes = 0
            while True:
                message = await receive()
                buffered.append(message)
                if message.get("type") != "http.request":
                    break
                received_bytes += len(message.get("body") or b"")
                if received_bytes > self.max_request_body_size:
                    await self._json_error(
                        send, status=413, error="request_too_large",
                        message=f"Request body exceeds {self.max_request_body_size} bytes.",
                    )
                    return
                if not message.get("more_body", False):
                    break
            replay_index = 0

            async def receive_buffered() -> MutableMapping[str, Any]:
                nonlocal replay_index
                if replay_index < len(buffered):
                    message = buffered[replay_index]
                    replay_index += 1
                    return message
                return await receive()

            downstream_receive = receive_buffered

        with request_identity_scope(identity):
            await self.app(scope, downstream_receive, send)


def _identity_for_tool(app: Any) -> RequestIdentity | None:
    get_context = getattr(app, "get_context", None)
    if callable(get_context):
        try:
            request = get_context().request_context.request
            headers = getattr(request, "headers", None)
        except (AttributeError, LookupError, TypeError, ValueError):
            # No live request (stdio, direct call) — ContextVar fallback below.
            headers = None
        if headers is not None:
            # Fail closed: an HTTP request whose identity headers do not parse
            # must surface as an error, never fall back to a possibly-stale
            # session ContextVar identity.
            return parse_identity_headers(headers)
    return get_request_identity()


def _identity_mismatch(
    tools: MemoryTools, *, field: str, expected: str, received: Any,
) -> dict[str, Any]:
    return tools.db.state.response(
        {
            "error": "identity_mismatch",
            "field": field,
            "expected": expected,
            "received": received,
            "reason": (
                f"Streamable HTTP identity comes from {CLIENT_HEADER} and "
                f"{AGENT_ID_HEADER}. Remove the conflicting tool-data identity "
                "or fix this MCP server's fixed headers, then retry."
            ),
        },
        ok=False,
    )


def _policy_denied(tools: MemoryTools, identity: RequestIdentity) -> dict[str, Any] | None:
    allowed, warnings = tools._allowed(identity.agent_id, identity.client)
    if allowed:
        return None
    return tools.db.state.response(
        {"error": "agent_policy_denied", "client": identity.client, "agent_id": identity.agent_id},
        ok=False,
        extra_warnings=warnings,
    )


def _repair_policy_check(task: str, data: dict[str, Any] | None) -> bool:
    operation = task.strip().lower()
    payload = data if isinstance(data, dict) else {}
    if operation == "help":
        return False
    if operation in {"rebuild_evidence", "replay_backup"}:
        dry_run = payload.get("dry_run", True)
        if isinstance(dry_run, str):
            is_dry_run = dry_run.strip().lower() in {"true", "1", "yes", "on"}
        elif isinstance(dry_run, (bool, int, float)):
            is_dry_run = bool(dry_run)
        else:
            is_dry_run = False
        return not is_dry_run
    if operation == "semantic_control":
        return str(payload.get("action") or "status").strip().lower() != "status"
    if operation == "notice":
        return str(payload.get("action") or "list").strip().lower() not in {"list", "read"}
    return operation != "help"


def _invoke_with_identity(
    tools: MemoryTools,
    identity: RequestIdentity | None,
    fn: Callable[..., dict[str, Any]],
    *,
    policy_check: bool,
    **kwargs: Any,
) -> dict[str, Any]:
    if identity is not None and policy_check:
        denied = _policy_denied(tools, identity)
        if denied is not None:
            return denied
    with request_identity_scope(identity):
        return fn(**kwargs)


def _data_with_request_identity(
    tools: MemoryTools,
    data: dict[str, Any] | None,
    identity: RequestIdentity | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if identity is None or not isinstance(data, dict):
        return data, None
    payload = dict(data)
    expected = {"client": identity.client, "agent_id": identity.agent_id}
    for field, value in expected.items():
        if field in payload and payload[field] != value:
            return None, _identity_mismatch(
                tools, field=field, expected=value, received=payload[field],
            )
        payload.pop(field, None)
    # Identity is never injected into the payload: tool-data identity fields are
    # caller input (rejected on mismatch, stripped otherwise); attribution is
    # applied from the trusted identity inside the write pipeline.
    return payload, None


def build_runtime() -> ServerBundle:
    settings = Settings.from_env()
    if settings.mcp_transport == "streamable-http" and not is_loopback_host(settings.mcp_http_host):
        raise RuntimeError(
            "Community Streamable HTTP MCP may bind only to localhost "
            "(127.0.0.1, ::1, or localhost); X-Mema-* headers are not authentication."
        )
    # Normalize the configured identity once: an unstripped config value would
    # otherwise be stored verbatim as attribution and silently miss exact-match
    # policy rules (header identities are strip-validated on the HTTP path).
    configured_client = settings.client.strip()
    configured_agent_id = settings.agent_id.strip()
    if not configured_client or not configured_agent_id:
        raise RuntimeError(
            "memory-arbiter requires a configured identity: set client and "
            "agent_id in config.json (or MEMORY_ARBITER_CLIENT / "
            "MEMORY_ARBITER_AGENT_ID) before starting the MCP server."
        )
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError("MCP Python SDK is not installed") from exc

    app = FastMCP(
        "memory-arbiter-mcp",
        host=settings.mcp_http_host,
        port=settings.mcp_http_port,
        # Fixed HTTP transport surface (frozen constants since 0.15.0).
        streamable_http_path=MCP_HTTP_PATH,
        stateless_http=True,
        json_response=False,
        max_request_body_size=MCP_HTTP_BODY_LIMIT,
    )
    # The SDK's FastMCP constructor has no version parameter, and the
    # lowlevel Server reports the MCP SDK's own version (e.g. 1.29.0) in the
    # initialize handshake unless its `version` is set explicitly. Guard so
    # SDK refactors or test doubles without the attribute degrade to the old
    # behavior instead of breaking boot.
    if getattr(app, "_mcp_server", None) is not None:
        app._mcp_server.version = __version__
    tools = MemoryTools(settings)
    tools.start_update_monitor()
    tools.start_evidence_worker()
    tools.start_semantic_worker()
    # stdio identity bridge: stdio has no per-request headers, so establish the
    # process-level identity from config once and apply it per tool call via
    # request_identity_scope in _invoke_with_identity. Without this bridge the
    # trusted source (ContextVar) would be empty on every stdio call.
    stdio_identity: RequestIdentity | None = None
    if settings.mcp_transport == "stdio":
        stdio_identity = RequestIdentity(
            client=configured_client, agent_id=configured_agent_id, transport="stdio",
        )

    @app.tool()
    def memory(action: str = "help", data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Daily memory operations: remember, find, read, update, judge, status, help.

        Call memory(action="help") to discover accepted fields, judge requirements,
        value enums, update modes, and action_required paths before relying on a
        result that requests attention.

        find responses carry a size block (tokens_estimate, matched-beyond-limit
        counts) with a display_hint; surface those numbers to the user when
        presenting the results.
        """
        identity = _identity_for_tool(app) or stdio_identity
        payload, error = _data_with_request_identity(
            tools, {} if data is None else data, identity,
        )
        return error or _invoke_with_identity(
            tools, identity, tools.memory,
            policy_check=action.strip().lower() in {"update", "judge"},
            action=action, data=payload,
        )

    @app.tool()
    def memory_review(view: str = "help", data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read-only inspection: overview, doctor, conflicts, conflict_detail, history, expired, audit, entities, help.

        Use memory_review(view="help") for accepted fields. Inspect conflict_detail
        before judging a conflict so its members, value groups, revision, and apply
        state are visible.
        """
        identity = _identity_for_tool(app) or stdio_identity
        payload, error = _data_with_request_identity(
            tools, {} if data is None else data, identity,
        )
        return error or _invoke_with_identity(
            tools, identity, tools.memory_review,
            policy_check=False, view=view, data=payload,
        )

    @app.tool()
    def memory_govern(action: str = "help", data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Authorized governance: retire, merge near-duplicates, apply/replan/resolve conflicts, confirm, and manage workspaces.

        Every state-changing action requires explicit user authorization for that
        action, then authorized=true. Call memory_govern(action="help") for exact
        actions, accepted fields, impact notes, and confirmation semantics.
        """
        identity = _identity_for_tool(app) or stdio_identity
        payload, error = _data_with_request_identity(
            tools, {} if data is None else data, identity,
        )
        return error or _invoke_with_identity(
            tools, identity, tools.memory_govern,
            policy_check=action.strip().lower() != "help", action=action, data=payload,
        )

    @app.tool()
    def memory_repair(task: str = "help", data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Maintenance: evidence rebuild, conflict scans (scheduled-task spec under help topic scheduled_tasks), full-library duplicate sweeps (scan_duplicates), history cleanup, entity assignment, pending activation, backup replay, notices, and semantic runtime control.

        Use memory_repair(task="help") for notice handling and semantic_control
        actions. Semantic notices are advisory; read both memories before dismiss
        or resolve, and never pass a notice directly to judge or resolve_conflict.
        """
        identity = _identity_for_tool(app) or stdio_identity
        payload, error = _data_with_request_identity(
            tools, {} if data is None else data, identity,
        )
        return error or _invoke_with_identity(
            tools, identity, tools.memory_repair,
            policy_check=_repair_policy_check(task, payload), task=task, data=payload,
        )

    return ServerBundle(app, tools)


def build_http_app(bundle: ServerBundle) -> ASGIApp:
    settings = bundle.tools.settings
    if not is_loopback_host(settings.mcp_http_host):
        raise RuntimeError(
            "Community Streamable HTTP MCP may bind only to localhost "
            "(127.0.0.1, ::1, or localhost)."
        )
    return MemoryIdentityMiddleware(
        bundle.app.streamable_http_app(),
        path=MCP_HTTP_PATH,
        max_request_body_size=MCP_HTTP_BODY_LIMIT,
    )


def run_streamable_http(bundle: ServerBundle) -> None:
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("uvicorn is required for Streamable HTTP MCP") from exc
    settings = bundle.tools.settings
    server = uvicorn.Server(uvicorn.Config(
        build_http_app(bundle),
        host=settings.mcp_http_host,
        port=settings.mcp_http_port,
        log_level="info",
    ))
    server.run()


def run() -> None:
    bundle = build_runtime()
    if bundle.tools.settings.mcp_transport == "streamable-http":
        run_streamable_http(bundle)
    else:
        bundle.app.run()


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
            print(
                "Usage: mema [command]\n\n"
                "Run the MCP server when no command is given.\n\n"
                "Commands:\n"
                "  doctor         Check database, configuration, and runtime health\n"
                "  setup          Generate a starter config and check local prerequisites\n"
                "  console        Open the read-only local web console\n"
                "  upgrade        Rebuild a legacy database into a side-by-side current database\n"
                "  migrate-vnext  Run the low-level migration workflow\n"
                "  help           Show this help\n\n"
                "Use `mema <command> --help` for command-specific options."
            )
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
        if bundle.tools.settings.mcp_transport == "streamable-http":
            run_streamable_http(bundle)
        else:
            bundle.app.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
