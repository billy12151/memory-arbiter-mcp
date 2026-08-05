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

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        self._handle(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib hook name
        self._handle(send_body=False)

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib hook name
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, HEAD, OPTIONS")
        self.end_headers()

    def _handle(self, send_body: bool) -> None:
        if not _host_allowed(self.headers.get("Host"), self.server.server_address[1]):
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
                self._json(self.api.overview(), send_body=send_body)
                return
            if parsed.path == "/api/conflicts":
                qs = parse_qs(parsed.query)
                self._json(self.api.conflicts(status=_one(qs, "status", "open"), limit=_one(qs, "limit", "50")), send_body=send_body)
                return
            if parsed.path.startswith("/api/conflicts/"):
                conflict_id = _path_int(parsed.path)
                if conflict_id is None:
                    self._json({"error": "conflict id must be an integer"}, status=HTTPStatus.BAD_REQUEST, send_body=send_body)
                    return
                self._json_or_error(self.api.conflict_detail(conflict_id), send_body=send_body)
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
            if parsed.path.startswith("/api/memories/"):
                qs = parse_qs(parsed.query)
                memory_id = _path_int(parsed.path)
                if memory_id is None:
                    self._json({"error": "memory id must be an integer"}, status=HTTPStatus.BAD_REQUEST, send_body=send_body)
                    return
                self._json_or_error(self.api.memory_detail(memory_id, sections=_one(qs, "sections", "catalog")), send_body=send_body)
                return
            if parsed.path == "/api/doctor":
                self._json(self.api.doctor(), send_body=send_body)
                return
            if parsed.path == "/api/settings":
                self._json(self.api.settings_view(), send_body=send_body)
                return
            self._json({"error": "not found"}, status=HTTPStatus.NOT_FOUND, send_body=send_body)
        except Exception as exc:
            self._json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR, send_body=send_body)

    def _html(self, text: str, send_body: bool = True) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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


def build_http_server(host: str, port: int, api: ConsoleAPI | None = None) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Console is local-only in this version; use --host 127.0.0.1")

    class Handler(ConsoleRequestHandler):
        pass

    Handler.api = api or ConsoleAPI()
    return ThreadingHTTPServer((host, int(port)), Handler)
