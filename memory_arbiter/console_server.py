from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .console_api import ConsoleAPI
from .console_static import INDEX_HTML


class ConsoleRequestHandler(BaseHTTPRequestHandler):
    api: ConsoleAPI

    server_version = "memory-arbiter-console/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; "
            "form-action 'self'",
        )
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        self._handle(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib hook name
        self._handle(send_body=False)

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib hook name
        address = self.server.server_address
        port = int(address[1]) if isinstance(address, tuple) and len(address) > 1 else 0
        if not _host_allowed(self.headers.get("Host"), port):
            self._json({"error": "forbidden host"}, status=HTTPStatus.FORBIDDEN, send_body=False)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        # R2 P1-1 pin: this 204 carries NO Access-Control-* headers on purpose —
        # that is the CSRF defense for the POST surface (preflight always
        # fails). Never "fix" this by adding CORS headers.
        self.send_header("Allow", "GET, HEAD, POST, OPTIONS")
        self.end_headers()

    _MAX_POST_BODY = 2 * 1024 * 1024

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        address = self.server.server_address
        port = int(address[1]) if isinstance(address, tuple) and len(address) > 1 else 0
        # Order matters (plan §B.3.2): Host gate BEFORE body parse — it is the
        # only DNS-rebinding defense.
        if not _host_allowed(self.headers.get("Host"), port):
            self._json({"error": "forbidden host"}, status=HTTPStatus.FORBIDDEN)
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._json({"error": "content-type must be application/json"}, status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        if self.headers.get("X-Console-Request") != "1":
            self._json({"error": "missing X-Console-Request header"}, status=HTTPStatus.FORBIDDEN)
            return
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header else -1
        except ValueError:
            length = -1
        if length < 0:
            # chunked / missing length: stdlib http.server does not decode
            # chunked bodies — reject rather than mis-parse.
            self._json({"error": "content-length required"}, status=HTTPStatus.LENGTH_REQUIRED)
            return
        if length > self._MAX_POST_BODY:
            self._json({"error": "body too large"}, status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._json({"error": "invalid json body"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not isinstance(body, dict):
            self._json({"error": "json body must be an object"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            self._route_post(urlparse(self.path).path, body)
        except Exception:
            self._json({"error": "internal server error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _route_post(self, path: str, body: dict[str, Any]) -> None:
        if path == "/api/knowledge/membership":
            payload = self.api.knowledge_membership(
                memory_id=body.get("memory_id") or 0,
                op=str(body.get("op") or ""),
                workspace=_optional_str(body.get("workspace")),
            )
            self._json_or_error(payload)
            return
        if path == "/api/settings/llm":
            self._json_or_error(self.api.llm_config_save(body))
            return
        if path == "/api/llm/ask":
            payload = self.api.llm_ask(
                memory_ids=body.get("memory_ids") or [],
                action=str(body.get("action") or ""),
                custom_prompt=str(body.get("custom_prompt") or ""),
                workspace=_optional_str(body.get("workspace")),
            )
            self._json_or_error(payload)
            return
        if path == "/api/llm/test":
            self._json_or_error(self.api.llm_test())
            return
        if path == "/api/knowledge/edit":
            payload = self.api.knowledge_edit(
                memory_id=body.get("memory_id") or 0,
                old_text=str(body.get("old_text") or ""),
                new_text=str(body.get("new_text") or ""),
                add_tags=body.get("add_tags") if isinstance(body.get("add_tags"), list) else [],
                remove_tags=body.get("remove_tags") if isinstance(body.get("remove_tags"), list) else [],
                reason=str(body.get("reason") or ""),
                workspace=_optional_str(body.get("workspace")),
                authorized=bool(body.get("authorized")),
            )
            self._json_or_error(payload)
            return
        if path == "/api/knowledge/retire":
            payload = self.api.knowledge_retire(
                memory_id=body.get("memory_id") or 0,
                reason=str(body.get("reason") or ""),
                workspace=_optional_str(body.get("workspace")),
                authorized=bool(body.get("authorized")),
            )
            self._json_or_error(payload)
            return
        if path == "/api/knowledge/save":
            tags = body.get("tags")
            payload = self.api.knowledge_save(
                subject=str(body.get("subject") or ""),
                content=str(body.get("content") or ""),
                tags=tags if isinstance(tags, list) else [],
                workspace=_optional_str(body.get("workspace")),
            )
            self._json_or_error(payload)
            return
        self._json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle(self, send_body: bool) -> None:
        address = self.server.server_address
        port = int(address[1]) if isinstance(address, tuple) and len(address) > 1 else 0
        if not _host_allowed(self.headers.get("Host"), port):
            self._json({"error": "forbidden host"}, status=HTTPStatus.FORBIDDEN, send_body=send_body)
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self._html(INDEX_HTML, send_body=send_body)
                return
            if parsed.path == "/api/health":
                self._json(self.api.health(), send_body=send_body)
                return
            if parsed.path == "/api/overview":
                qs = parse_qs(parsed.query)
                ws = _one(qs, "workspace", "") or None
                self._json_or_error(self.api.overview(workspace=ws) if ws else self.api.overview(), send_body=send_body)
                return
            if parsed.path == "/api/conflicts":
                qs = parse_qs(parsed.query)
                ws = _one(qs, "workspace", "") or None
                kwargs = {"status": _one(qs, "status", "open"), "limit": _one(qs, "limit", "50")}
                if ws:
                    kwargs["workspace"] = ws
                self._json_or_error(self.api.conflicts(**kwargs), send_body=send_body)
                return
            if parsed.path.startswith("/api/conflicts/"):
                conflict_id = _path_int(parsed.path)
                if conflict_id is None:
                    self._json({"error": "conflict id must be an integer"}, status=HTTPStatus.BAD_REQUEST, send_body=send_body)
                    return
                qs = parse_qs(parsed.query)
                ws = _one(qs, "workspace", "") or None
                self._json_or_error(self.api.conflict_detail(conflict_id, workspace=ws) if ws else self.api.conflict_detail(conflict_id), send_body=send_body)
                return
            if parsed.path == "/api/memories":
                qs = parse_qs(parsed.query)
                self._json_or_error(self.api.memories(
                    query=_one(qs, "query", ""),
                    status=_one(qs, "status", "active"),
                    workspace=_one(qs, "workspace", "") or None,
                    source_type=_one(qs, "source_type", "") or None,
                    tags=_one(qs, "tags", ""),
                    limit=_one(qs, "limit", "30"),
                    offset=_one(qs, "offset", "0"),
                ), send_body=send_body)
                return
            if parsed.path.startswith("/api/memories/") and parsed.path.endswith("/graph"):
                memory_id = _path_int(parsed.path[: -len("/graph")])
                if memory_id is None:
                    self._json({"error": "memory id must be an integer"}, status=HTTPStatus.BAD_REQUEST, send_body=send_body)
                    return
                qs = parse_qs(parsed.query)
                ws = _one(qs, "workspace", "") or None
                self._json_or_error(self.api.memory_graph(memory_id, workspace=ws) if ws else self.api.memory_graph(memory_id), send_body=send_body)
                return
            if parsed.path.startswith("/api/memories/"):
                qs = parse_qs(parsed.query)
                memory_id = _path_int(parsed.path)
                if memory_id is None:
                    self._json({"error": "memory id must be an integer"}, status=HTTPStatus.BAD_REQUEST, send_body=send_body)
                    return
                ws = _one(qs, "workspace", "") or None
                self._json_or_error(self.api.memory_detail(memory_id, workspace=ws) if ws else self.api.memory_detail(memory_id), send_body=send_body)
                return
            if parsed.path == "/api/knowledge/tree":
                qs = parse_qs(parsed.query)
                ws = _one(qs, "workspace", "") or None
                self._json_or_error(self.api.knowledge_tree(workspace=ws), send_body=send_body)
                return
            if parsed.path == "/api/knowledge/candidates":
                qs = parse_qs(parsed.query)
                ws = _one(qs, "workspace", "") or None
                self._json_or_error(
                    self.api.knowledge_candidates(workspace=ws, limit=_one(qs, "limit", "50")),
                    send_body=send_body,
                )
                return
            if parsed.path == "/api/doctor":
                self._json(self.api.doctor(), send_body=send_body)
                return
            if parsed.path == "/api/settings":
                self._json(self.api.settings_view(), send_body=send_body)
                return
            self._json({"error": "not found"}, status=HTTPStatus.NOT_FOUND, send_body=send_body)
        except Exception:
            self._json(
                {"error": "internal server error"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                send_body=send_body,
            )

    def _html(self, text: str, send_body: bool = True) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK, send_body: bool = True) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _json_or_error(self, payload: dict[str, Any], send_body: bool = True) -> None:
        if "error" not in payload:
            self._json(payload, send_body=send_body)
            return
        status = payload.pop("_http_status", HTTPStatus.BAD_REQUEST)
        try:
            self._json(payload, status=HTTPStatus(status), send_body=send_body)
        except ValueError:
            self._json(payload, status=HTTPStatus.BAD_REQUEST, send_body=send_body)


def _host_allowed(host_header: str | None, bound_port: int) -> bool:
    if not host_header:
        return False
    host = host_header.strip().lower()
    allowed = {
        f"127.0.0.1:{bound_port}",
        f"localhost:{bound_port}",
        "127.0.0.1",
        "localhost",
    }
    return host in allowed


def _path_int(path: str) -> int | None:
    try:
        return int(path.rsplit("/", 1)[-1])
    except (TypeError, ValueError):
        return None


def _one(qs: dict[str, list[str]], key: str, default: str) -> str:
    values = qs.get(key)
    return values[0] if values else default


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def build_http_server(host: str, port: int, api: ConsoleAPI | None = None) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Console is local-only in this version; use --host 127.0.0.1")

    class Handler(ConsoleRequestHandler):
        pass

    Handler.api = api or ConsoleAPI()
    return ThreadingHTTPServer((host, int(port)), Handler)
