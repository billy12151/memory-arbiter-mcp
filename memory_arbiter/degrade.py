from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class DegradeState:
    mode: str = "sqlite"
    warnings: list[str] = field(default_factory=list)
    sqlite_vec_available: bool = False
    fts5_available: bool = False
    sqlite_writable: bool = True
    jsonl_backup_active: bool = False
    notice_provider: Callable[[], list[dict[str, Any]]] | None = None

    @property
    def degraded(self) -> bool:
        return bool(self.warnings) or self.mode != "sqlite_vec"

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def response(self, data: Any, ok: bool = True, extra_warnings: list[str] | None = None) -> dict[str, Any]:
        warnings = list(self.warnings)
        for warning in extra_warnings or []:
            if warning not in warnings:
                warnings.append(warning)
        resp = {
            "ok": ok,
            "mode": self.mode,
            "warnings": warnings,
            "degraded": bool(warnings) or self.mode != "sqlite_vec",
            "data": data,
        }
        if ok and self.notice_provider is not None:
            try:
                notices = self.notice_provider()
            except Exception:
                notices = []
            if notices:
                resp["notices"] = notices
        return resp
