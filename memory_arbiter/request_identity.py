"""Request-scoped identity for the localhost Streamable HTTP transport."""
from __future__ import annotations

import ipaddress
import re
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional
from urllib.parse import urlsplit

CLIENT_HEADER = "X-Mema-Client"
AGENT_ID_HEADER = "X-Mema-Agent-Id"
CLIENT_MAX_LENGTH = 64
AGENT_ID_MAX_LENGTH = 128

_IDENTITY_RE = re.compile(r"^[A-Za-z0-9._:@-]+$")
_current_identity: ContextVar["RequestIdentity" | None] = ContextVar(
    "memory_arbiter_request_identity", default=None,
)


@dataclass(frozen=True)
class RequestIdentity:
    client: str
    agent_id: str
    transport: str = "streamable-http"


class IdentityHeaderError(ValueError):
    def __init__(self, header: str, reason: str):
        self.header = header
        self.reason = reason
        super().__init__(
            f"{header} {reason}. Configure fixed {CLIENT_HEADER} and "
            f"{AGENT_ID_HEADER} headers for this MCP server, then retry."
        )


def get_request_identity() -> RequestIdentity | None:
    return _current_identity.get()


def set_request_identity(identity: RequestIdentity) -> Token[RequestIdentity | None]:
    return _current_identity.set(identity)


def reset_request_identity(token: Token[RequestIdentity | None]) -> None:
    _current_identity.reset(token)


@contextmanager
def request_identity_scope(identity: RequestIdentity | None) -> Iterator[None]:
    if identity is None:
        yield
        return
    token = set_request_identity(identity)
    try:
        yield
    finally:
        reset_request_identity(token)


def _header_value(headers: Mapping[str, Any], name: str) -> str | None:
    getlist = getattr(headers, "getlist", None)
    if callable(getlist):
        values = list(getlist(name))
        if len(values) > 1:
            raise IdentityHeaderError(name, "must be sent exactly once")
        if values:
            return str(values[0])
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    if value is None:
        for key, candidate in headers.items():
            if str(key).casefold() == name.casefold():
                value = candidate
                break
    if value is None:
        return None
    return str(value)


def _validate_identity_value(
    value: str | None, *, header: str, max_length: int,
) -> str:
    if value is None:
        raise IdentityHeaderError(header, "is required")
    normalized = value.strip()
    if not normalized:
        raise IdentityHeaderError(header, "must not be empty")
    if len(normalized) > max_length:
        raise IdentityHeaderError(header, f"must be at most {max_length} characters")
    if normalized != value or not _IDENTITY_RE.fullmatch(normalized):
        raise IdentityHeaderError(
            header,
            "must use only ASCII letters, digits, '.', '_', ':', '@', or '-' with no whitespace",
        )
    return normalized


def parse_identity_headers(headers: Mapping[str, Any]) -> RequestIdentity:
    client = _validate_identity_value(
        _header_value(headers, CLIENT_HEADER),
        header=CLIENT_HEADER,
        max_length=CLIENT_MAX_LENGTH,
    )
    agent_id = _validate_identity_value(
        _header_value(headers, AGENT_ID_HEADER),
        header=AGENT_ID_HEADER,
        max_length=AGENT_ID_MAX_LENGTH,
    )
    return RequestIdentity(client=client, agent_id=agent_id)


def is_loopback_host(host: str) -> bool:
    normalized = str(host or "").strip().strip("[]").casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def host_header_name(host_header: str | None) -> str | None:
    if host_header is None or not str(host_header).strip():
        return None
    try:
        return urlsplit(f"//{str(host_header).strip()}").hostname
    except ValueError:
        return None
