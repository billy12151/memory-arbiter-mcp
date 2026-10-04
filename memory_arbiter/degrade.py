from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class DegradeState:
    mode: str = "sqlite"
    warnings: list[str] = field(default_factory=list)
    sqlite_vec_available: bool = False
    fts5_available: bool = False
    sqlite_writable: bool = True
    jsonl_backup_active: bool = False
    # B5（0.17.1 优化批）：降级写入的观测时间戳。jsonl_backup_active 是
    # 单向闩（置 True 恒伴 sqlite_writable=False，全仓无复位点），故不改
    # 其语义，只补"最后一次用 JSONL 是什么时候"（status 输出可见）。
    jsonl_backup_last_used_at: str | None = None
    notice_provider: Callable[[], list[dict[str, Any]]] | None = None

    @property
    def degraded(self) -> bool:
        return bool(self.warnings) or self.mode != "sqlite_vec"

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def response(
        self, data: Any, ok: bool = True,
        extra_warnings: list[str] | None = None,
        extra_notices: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
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
        notices = list(extra_notices or [])
        if ok and self.notice_provider is not None:
            try:
                notices.extend(self.notice_provider())
            except Exception:
                pass
        if notices:
            resp["notices"] = notices
        return resp
