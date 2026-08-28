from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import AgentPolicy, Settings
from memory_arbiter.request_identity import (
    AGENT_ID_HEADER,
    CLIENT_HEADER,
    IdentityHeaderError,
    RequestIdentity,
    get_request_identity,
    parse_identity_headers,
    request_identity_scope,
)
from memory_arbiter.server import MemoryIdentityMiddleware


class MultiHeaders(dict[str, str]):
    def __init__(self, items: list[tuple[str, str]]):
        super().__init__()
        self._items = items
        for key, value in items:
            self[key.casefold()] = value

    def getlist(self, name: str) -> list[str]:
        return [value for key, value in self._items if key.casefold() == name.casefold()]


def test_identity_header_parser_accepts_case_insensitive_valid_values() -> None:
    identity = parse_identity_headers({
        "x-mema-client": "claude-code",
        "X-MEMA-AGENT-ID": "agent_1@example",
    })
    assert identity == RequestIdentity(client="claude-code", agent_id="agent_1@example")


@pytest.mark.parametrize(
    ("headers", "header", "fragment"),
    [
        ({AGENT_ID_HEADER: "agent"}, CLIENT_HEADER, "required"),
        ({CLIENT_HEADER: "client", AGENT_ID_HEADER: ""}, AGENT_ID_HEADER, "empty"),
        ({CLIENT_HEADER: " client", AGENT_ID_HEADER: "agent"}, CLIENT_HEADER, "whitespace"),
        ({CLIENT_HEADER: "a,b", AGENT_ID_HEADER: "agent"}, CLIENT_HEADER, "ASCII"),
        ({CLIENT_HEADER: "a/b", AGENT_ID_HEADER: "agent"}, CLIENT_HEADER, "ASCII"),
        ({CLIENT_HEADER: "客户端", AGENT_ID_HEADER: "agent"}, CLIENT_HEADER, "ASCII"),
        ({CLIENT_HEADER: "a" * 65, AGENT_ID_HEADER: "agent"}, CLIENT_HEADER, "64"),
        ({CLIENT_HEADER: "client", AGENT_ID_HEADER: "a" * 129}, AGENT_ID_HEADER, "128"),
    ],
)
def test_identity_header_parser_rejects_invalid_values(
    headers: dict[str, str], header: str, fragment: str,
) -> None:
    with pytest.raises(IdentityHeaderError) as exc_info:
        parse_identity_headers(headers)
    assert exc_info.value.header == header
    assert fragment in str(exc_info.value)
    assert CLIENT_HEADER in str(exc_info.value)
    assert AGENT_ID_HEADER in str(exc_info.value)


def test_identity_header_parser_rejects_duplicate_header() -> None:
    headers = MultiHeaders([
        (CLIENT_HEADER, "one"), (CLIENT_HEADER.lower(), "two"),
        (AGENT_ID_HEADER, "agent"),
    ])
    with pytest.raises(IdentityHeaderError, match="exactly once"):
        parse_identity_headers(headers)


async def _call_asgi(
    app: Any,
    *,
    path: str = "/mcp",
    client: tuple[str, int] = ("127.0.0.1", 4321),
    headers: list[tuple[bytes, bytes]] | None = None,
    body_chunks: list[bytes] | None = None,
    method: str = "POST",
) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    chunks = list(body_chunks) if body_chunks is not None else [b""]
    receive_index = 0

    async def receive() -> dict[str, Any]:
        nonlocal receive_index
        if receive_index < len(chunks):
            body = chunks[receive_index]
            receive_index += 1
            return {
                "type": "http.request", "body": body,
                "more_body": receive_index < len(chunks),
            }
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    supplied_headers = list(headers or [])
    if not any(name.lower() == b"host" for name, _value in supplied_headers):
        supplied_headers.append((b"host", b"127.0.0.1:8000"))
    scope: dict[str, Any] = {
        "type": "http", "method": method, "path": path,
        "headers": supplied_headers, "client": client,
    }
    await app(scope, receive, send)
    return sent


def _response_json(messages: list[dict[str, Any]]) -> dict[str, Any]:
    body = b"".join(message.get("body", b"") for message in messages)
    return json.loads(body)


def test_identity_middleware_fails_closed_and_hides_other_paths() -> None:
    reached = False

    async def inner(_scope: Any, _receive: Any, _send: Any) -> None:
        nonlocal reached
        reached = True

    middleware = MemoryIdentityMiddleware(inner)
    missing = asyncio.run(_call_asgi(middleware))
    assert missing[0]["status"] == 400
    assert CLIENT_HEADER in _response_json(missing)["message"]
    assert reached is False

    remote = asyncio.run(_call_asgi(
        middleware,
        client=("192.0.2.10", 1234),
        headers=[
            (CLIENT_HEADER.lower().encode(), b"client"),
            (AGENT_ID_HEADER.lower().encode(), b"agent"),
        ],
    ))
    assert remote[0]["status"] == 403

    bad_host = asyncio.run(_call_asgi(
        middleware,
        headers=[
            (b"host", b"evil.example"),
            (CLIENT_HEADER.lower().encode(), b"client"),
            (AGENT_ID_HEADER.lower().encode(), b"agent"),
        ],
    ))
    assert bad_host[0]["status"] == 421

    too_large = asyncio.run(_call_asgi(
        MemoryIdentityMiddleware(inner, max_request_body_size=16),
        headers=[
            (b"content-length", b"17"),
            (CLIENT_HEADER.lower().encode(), b"client"),
            (AGENT_ID_HEADER.lower().encode(), b"agent"),
        ],
    ))
    assert too_large[0]["status"] == 413

    hidden = asyncio.run(_call_asgi(middleware, path="/docs"))
    assert hidden[0]["status"] == 404


def test_identity_middleware_sets_and_resets_request_context() -> None:
    observed: list[RequestIdentity | None] = []

    async def inner(_scope: Any, _receive: Any, send: Any) -> None:
        observed.append(get_request_identity())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = MemoryIdentityMiddleware(inner)
    result = asyncio.run(_call_asgi(
        middleware,
        headers=[
            (CLIENT_HEADER.lower().encode(), b"claude"),
            (AGENT_ID_HEADER.lower().encode(), b"agent-a"),
        ],
    ))
    assert result[0]["status"] == 204
    assert observed == [RequestIdentity(client="claude", agent_id="agent-a")]
    assert get_request_identity() is None


def test_identity_middleware_enforces_actual_chunked_body_size() -> None:
    observed: list[bytes] = []

    async def inner(_scope: Any, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                break
            observed.append(message.get("body") or b"")
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    headers = [
        (CLIENT_HEADER.lower().encode(), b"client"),
        (AGENT_ID_HEADER.lower().encode(), b"agent"),
    ]
    accepted = asyncio.run(_call_asgi(
        MemoryIdentityMiddleware(inner, max_request_body_size=4),
        headers=headers, body_chunks=[b"ab", b"cd"],
    ))
    assert accepted[0]["status"] == 204
    assert observed == [b"ab", b"cd"]

    observed.clear()
    rejected = asyncio.run(_call_asgi(
        MemoryIdentityMiddleware(inner, max_request_body_size=4),
        headers=headers, body_chunks=[b"abc", b"de"],
    ))
    assert rejected[0]["status"] == 413
    assert observed == []


def test_identity_middleware_does_not_preread_sse_get_body() -> None:
    reached = False

    async def inner(_scope: Any, _receive: Any, send: Any) -> None:
        nonlocal reached
        reached = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    headers = [
        (CLIENT_HEADER.lower().encode(), b"client"),
        (AGENT_ID_HEADER.lower().encode(), b"agent"),
    ]
    result = asyncio.run(_call_asgi(
        MemoryIdentityMiddleware(inner), headers=headers, method="GET",
    ))
    assert result[0]["status"] == 200
    assert reached is True


def test_identity_middleware_isolates_concurrent_requests() -> None:
    observed: list[tuple[str, str]] = []
    ready = asyncio.Event()
    count = 0

    async def inner(_scope: Any, _receive: Any, send: Any) -> None:
        nonlocal count
        count += 1
        if count == 2:
            ready.set()
        await ready.wait()
        identity = get_request_identity()
        assert identity is not None
        await asyncio.sleep(0)
        observed.append((identity.client, identity.agent_id))
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = MemoryIdentityMiddleware(inner)

    async def run_both() -> None:
        await asyncio.gather(
            _call_asgi(middleware, headers=[
                (b"x-mema-client", b"one"), (b"x-mema-agent-id", b"agent-one"),
            ]),
            _call_asgi(middleware, headers=[
                (b"x-mema-client", b"two"), (b"x-mema-agent-id", b"agent-two"),
            ]),
        )

    asyncio.run(run_both())
    assert sorted(observed) == [("one", "agent-one"), ("two", "agent-two")]
    assert get_request_identity() is None


def _install_fake_fastmcp(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    class FakeFastMCP:
        last_kwargs: dict[str, Any] = {}

        def __init__(self, _name: str, **kwargs: Any) -> None:
            self.tools: dict[str, Any] = {}
            self.last_kwargs = kwargs
            type(self).last_kwargs = kwargs

        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = func
                return func
            return decorator

    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = FakeFastMCP
    fake_server = types.ModuleType("mcp.server")
    fake_mcp = types.ModuleType("mcp")
    fake_server.fastmcp = fake_fastmcp
    fake_mcp.server = fake_server
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)
    return FakeFastMCP


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any):
    from memory_arbiter import server

    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        update_check_enabled=False,
        client="settings-client",
        agent_id="settings-agent",
        **overrides,
    )
    monkeypatch.setattr(server.Settings, "from_env", classmethod(lambda cls: settings))
    _install_fake_fastmcp(monkeypatch)
    return server.build_runtime()


def test_http_identity_attributes_write_provenance_and_rejects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime(tmp_path, monkeypatch)
    identity = RequestIdentity(client="header-client", agent_id="header-agent")
    with request_identity_scope(identity):
        written = bundle.app.tools["memory"](
            action="remember", data={"content": "fact", "subject": "identity"},
        )
        matching = bundle.app.tools["memory"](
            action="remember",
            data={
                "content": "matching", "subject": "identity",
                "client": "header-client", "agent_id": "header-agent",
            },
        )
        mismatch = bundle.app.tools["memory"](
            action="remember",
            data={"content": "bad", "subject": "identity", "agent_id": "other"},
        )
        status = bundle.app.tools["memory"](action="status", data={})

    assert written["data"]["record"]["agent_id"] == "header-agent"
    assert matching["data"]["record"]["agent_id"] == "header-agent"
    assert mismatch["ok"] is False
    assert mismatch["data"]["error"] == "identity_mismatch"
    assert status["data"]["client"] == "header-client"
    assert status["data"]["agent_id"] == "header-agent"
    bundle.tools.shutdown(timeout=1)


def test_tool_identity_prefers_current_mcp_request_over_stale_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime(tmp_path, monkeypatch)

    class RequestContext:
        request = type("Request", (), {"headers": {
            CLIENT_HEADER: "current-client", AGENT_ID_HEADER: "current-agent",
        }})()

    class Context:
        request_context = RequestContext()

    bundle.app.get_context = lambda: Context()
    with request_identity_scope(RequestIdentity(client="stale-client", agent_id="stale-agent")):
        status = bundle.app.tools["memory"](action="status", data={})
        written = bundle.app.tools["memory"](
            action="remember", data={"content": "current", "subject": "session"},
        )
    assert status["data"]["client"] == "current-client"
    assert status["data"]["agent_id"] == "current-agent"
    assert written["data"]["record"]["agent_id"] == "current-agent"
    bundle.tools.shutdown(timeout=1)


def test_tool_identity_invalid_request_headers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime(tmp_path, monkeypatch)

    class RequestContext:
        request = type("Request", (), {"headers": {
            CLIENT_HEADER: "", AGENT_ID_HEADER: "current-agent",
        }})()

    class Context:
        request_context = RequestContext()

    bundle.app.get_context = lambda: Context()
    with request_identity_scope(RequestIdentity(client="stale-client", agent_id="stale-agent")):
        with pytest.raises(IdentityHeaderError):
            bundle.app.tools["memory"](action="status", data={})
    bundle.tools.shutdown(timeout=1)


def test_stdio_write_keeps_settings_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _runtime(tmp_path, monkeypatch)
    written = bundle.app.tools["memory"](
        action="remember", data={"content": "fact", "subject": "stdio"},
    )
    assert written["data"]["record"]["agent_id"] == "settings-agent"
    bundle.tools.shutdown(timeout=1)


def test_header_identity_controls_all_http_mutation_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = AgentPolicy(default_enabled=True, deny_agents=["blocked"])
    bundle = _runtime(tmp_path, monkeypatch, policy=policy)
    seed = bundle.tools.memory(
        "remember", {"content": "seed", "subject": "policy"},
    )
    memory_id = seed["data"]["id"]
    with request_identity_scope(RequestIdentity(client="header-client", agent_id="blocked")):
        readable = [
            bundle.app.tools["memory_repair"](
                task="semantic_control", data={"action": "status"},
            ),
            bundle.app.tools["memory_repair"](
                task="notice", data={"action": "list"},
            ),
            bundle.app.tools["memory_repair"](
                task="rebuild_evidence", data={"dry_run": True},
            ),
            bundle.app.tools["memory_repair"](
                task="rebuild_evidence", data={},
            ),
            bundle.app.tools["memory_repair"](
                task="replay_backup", data={"dry_run": "true"},
            ),
            bundle.app.tools["memory_repair"](
                task="replay_backup", data={},
            ),
        ]
        results = [
            bundle.app.tools["memory"](
                action="remember", data={"content": "fact", "subject": "policy"},
            ),
            bundle.app.tools["memory"](
                action="update", data={"memory_id": memory_id, "new_content": "changed"},
            ),
            bundle.app.tools["memory_govern"](
                action="confirm", data={"memory_id": memory_id, "authorized": True},
            ),
            bundle.app.tools["memory_repair"](
                task="set_entity", data={"memory_id": memory_id, "entity": "secret"},
            ),
            bundle.app.tools["memory_repair"](
                task="scan_candidates", data={"anchor_memory_id": 0},
            ),
        ]
    assert all(result["ok"] is True for result in readable)
    assert all(result["ok"] is False for result in results)
    assert all(any("agent_id=blocked" in warning for warning in result["warnings"]) for result in results)
    assert bundle.tools.db.get_memory(memory_id)["content"] == "seed"
    bundle.tools.shutdown(timeout=1)


def test_onboarding_notice_uses_header_agent_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _runtime(tmp_path, monkeypatch)

    class Monitor:
        def __init__(self) -> None:
            self.agent_ids: list[str] = []

        def consume_agent_onboarding_notice(self, agent_id: str) -> list[dict[str, Any]]:
            self.agent_ids.append(agent_id)
            return []

        def consume_notices(self) -> list[dict[str, Any]]:
            return []

        def update_status(self) -> dict[str, Any]:
            return {"enabled": False, "status": "test"}

    monitor = Monitor()
    bundle.tools._update_monitor = monitor  # type: ignore[assignment]
    with request_identity_scope(RequestIdentity(client="header-client", agent_id="notice-agent")):
        bundle.app.tools["memory"](action="status", data={})
    assert monitor.agent_ids == ["notice-agent"]
    bundle.tools.shutdown(timeout=1)


def test_settings_parse_http_transport_and_server_passes_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "db_path": str(tmp_path / "db.sqlite3"),
        "backup_jsonl": str(tmp_path / "backup.jsonl"),
        "client": "cfg-client",
        "agent_id": "cfg-agent",
        "mcp": {
            "transport": "streamable-http",
            "http": {"host": "localhost", "port": 8123, "path": "/memory/"},
        },
        "update_check": {"enabled": False},
    }), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg))
    settings = Settings.from_env()
    assert settings.mcp_transport == "streamable-http"
    assert settings.mcp_http_host == "localhost"
    assert settings.mcp_http_port == 8123
    assert settings.mcp_http_path == "/memory"
    assert settings.mcp_http_stateless is True

    from memory_arbiter import server
    monkeypatch.setattr(server.Settings, "from_env", classmethod(lambda cls: settings))
    fake = _install_fake_fastmcp(monkeypatch)
    bundle = server.build_runtime()
    assert fake.last_kwargs["host"] == "localhost"
    assert fake.last_kwargs["port"] == 8123
    assert fake.last_kwargs["streamable_http_path"] == "/memory"
    assert fake.last_kwargs["stateless_http"] is True
    bundle.tools.shutdown(timeout=1)


def test_http_stateful_mode_remains_an_explicit_compatibility_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "db_path": str(tmp_path / "db.sqlite3"),
        "backup_jsonl": str(tmp_path / "backup.jsonl"),
        "client": "cfg-client",
        "agent_id": "cfg-agent",
        "mcp": {
            "transport": "streamable-http",
            "http": {"stateless": False},
        },
        "update_check": {"enabled": False},
    }), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg))

    settings = Settings.from_env()
    assert settings.mcp_http_stateless is False

    from memory_arbiter import server
    monkeypatch.setattr(server.Settings, "from_env", classmethod(lambda cls: settings))
    fake = _install_fake_fastmcp(monkeypatch)
    bundle = server.build_runtime()
    assert fake.last_kwargs["stateless_http"] is False
    bundle.tools.shutdown(timeout=1)


def test_streamable_http_rejects_non_loopback_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_arbiter import server

    settings = Settings(
        db_path=tmp_path / "db.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        client="cfg-client",
        agent_id="cfg-agent",
        mcp_transport="streamable-http",
        mcp_http_host="0.0.0.0",
    )
    monkeypatch.setattr(server.Settings, "from_env", classmethod(lambda cls: settings))
    with pytest.raises(RuntimeError, match="localhost"):
        server.build_runtime()
    assert not settings.db_path.exists()


def test_build_runtime_requires_configured_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_arbiter import server

    settings = Settings(
        db_path=tmp_path / "db.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
    )
    monkeypatch.setattr(server.Settings, "from_env", classmethod(lambda cls: settings))
    with pytest.raises(RuntimeError, match="configured identity"):
        server.build_runtime()
    assert not settings.db_path.exists()


def test_stdio_bridge_status_and_policy_use_process_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime(tmp_path, monkeypatch)
    # No request scope at all: the stdio process identity established in
    # build_runtime is the trusted source for every tool call.
    status = bundle.app.tools["memory"](action="status", data={})
    assert status["data"]["client"] == "settings-client"
    assert status["data"]["agent_id"] == "settings-agent"
    assert status["data"]["policy"] == {"caller_allowed": True}
    bundle.tools.shutdown(timeout=1)

    denied_settings = Settings(
        db_path=tmp_path / "denied.sqlite3",
        backup_jsonl=tmp_path / "denied.jsonl",
        update_check_enabled=False,
        client="settings-client",
        agent_id="blocked",
        policy=AgentPolicy(default_enabled=True, deny_agents=["blocked"]),
    )
    from memory_arbiter import server
    monkeypatch.setattr(server.Settings, "from_env", classmethod(lambda cls: denied_settings))
    _install_fake_fastmcp(monkeypatch)
    denied_bundle = server.build_runtime()
    written = denied_bundle.app.tools["memory"](
        action="remember", data={"content": "fact", "subject": "stdio-policy"},
    )
    assert written["ok"] is False
    assert written["data"]["written"] is False
    # A payload agent_id cannot launder policy: it conflicts with the trusted
    # process identity and is rejected as a mismatch before any write happens.
    laundered = denied_bundle.app.tools["memory"](
        action="remember",
        data={"content": "fact", "subject": "stdio-policy", "agent_id": "unblocked"},
    )
    assert laundered["ok"] is False
    assert laundered["data"]["error"] == "identity_mismatch"
    denied_bundle.tools.shutdown(timeout=1)


def test_remember_payload_agent_id_is_ignored_without_request_identity(
    tmp_path: Path,
) -> None:
    from memory_arbiter.tools import MemoryTools

    tools = MemoryTools(Settings(
        db_path=tmp_path / "direct.sqlite3",
        backup_jsonl=tmp_path / "direct.jsonl",
        agent_id="env-agent",
        client="env-client",
    ))
    written = tools.memory_write(
        content="fact", subject="direct", agent_id="smuggled-agent",
        source_type="agent_generated",
    )
    assert written["ok"] is True
    # Payload provenance is dropped (unknown-field warning) and never reaches
    # the record; attribution falls back to the env/config identity.
    assert any("unknown field ignored: agent_id" in w for w in written["warnings"])
    assert written["data"]["record"]["agent_id"] == "env-agent"
    tools.shutdown(timeout=1)
