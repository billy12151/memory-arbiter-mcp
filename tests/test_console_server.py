from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from memory_arbiter.console_server import build_http_server


class DummyAPI:
    def health(self):
        return {"ok": True}

    def conflict_detail(self, conflict_id: int):
        return {"error": f"conflict id {conflict_id} not found", "_http_status": 404}

    def memory_detail(self, memory_id: int, sections: str = "catalog"):
        return {"memory": {"id": memory_id}, "sections": sections}

    def memories(self, **kwargs):
        return {"error": "strict isolation requires workspace", "_http_status": 400}


def test_console_server_rejects_non_localhost() -> None:
    try:
        build_http_server("0.0.0.0", 8766)
    except ValueError as exc:
        assert "local-only" in str(exc)
    else:
        raise AssertionError("expected local-only host rejection")


def test_console_server_rejects_ipv6_until_supported() -> None:
    try:
        build_http_server("::1", 8766)
    except ValueError as exc:
        assert "local-only" in str(exc)
    else:
        raise AssertionError("expected unsupported IPv6 host rejection")


def test_console_server_builds_on_localhost(tmp_path) -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    try:
        host, port = server.server_address
        assert host == "127.0.0.1"
        assert isinstance(port, int)
    finally:
        server.server_close()


def test_console_server_rejects_untrusted_host_header() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/health",
            headers={"Host": f"evil.example:{port}"},
        )
        try:
            urllib.request.urlopen(request, timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "forbidden host"
        else:
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown()
        server.server_close()


def test_console_server_maps_list_endpoint_errors_to_400() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/memories", timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "strict isolation requires workspace"
        else:
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown()
        server.server_close()


def test_console_server_returns_400_for_bad_path_id() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/conflicts/not-an-id", timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "conflict id must be an integer"
        else:
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown()
        server.server_close()


def test_console_server_returns_404_for_missing_conflict() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/conflicts/123", timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            payload = json.loads(exc.read().decode("utf-8"))
            assert payload["error"] == "conflict id 123 not found"
        else:
            raise AssertionError("expected HTTPError")
    finally:
        server.shutdown()
        server.server_close()


def test_console_server_handles_head_and_options() -> None:
    server = build_http_server("127.0.0.1", 0, api=DummyAPI())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, port = server.server_address
        head = urllib.request.Request(f"http://127.0.0.1:{port}/api/health", method="HEAD")
        with urllib.request.urlopen(head, timeout=2) as resp:
            assert resp.code == 200
            assert int(resp.headers.get("Content-Length", "0")) > 0
            assert resp.read() == b""
        options = urllib.request.Request(f"http://127.0.0.1:{port}/api/health", method="OPTIONS")
        with urllib.request.urlopen(options, timeout=2) as resp:
            assert resp.code == 204
            assert "GET" in (resp.headers.get("Allow") or "")
    finally:
        server.shutdown()
        server.server_close()
