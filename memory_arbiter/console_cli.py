from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from typing import Sequence

from .console_server import build_http_server


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="memory-arbiter console",
        description="Start the read-only local mema / 迷码 Console.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. MVP is local-only; default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=18876, help="Bind port; default: 18876")
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser automatically")
    parser.add_argument("--json", action="store_true", help="Print startup errors as JSON")
    args = parser.parse_args(list(argv or []))

    try:
        httpd = build_http_server(args.host, args.port)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        else:
            print(f"memory-arbiter console: {exc}", file=sys.stderr)
        return 2

    url = f"http://{args.host}:{args.port}"
    if args.json:
        print(json.dumps({"ok": True, "url": url, "read_only": True, "local_only": True}, ensure_ascii=False))
    else:
        print("memory-arbiter console — mema / 迷码")
        print(f"Listening on {url}")
        print("Read-only local governance console. Do not expose this port publicly.")
        print("Press Ctrl+C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
