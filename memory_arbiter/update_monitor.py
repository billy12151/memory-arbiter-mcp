from __future__ import annotations

import json
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import __version__

PACKAGE_NAME = "memory-arbiter-mcp"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
CHECK_INTERVAL = timedelta(hours=24)
BACKGROUND_TIMEOUT_SECONDS = 10
RETRY_AFTER_FAILURE = timedelta(hours=6)
NOTICE_SUPPRESS = timedelta(days=7)

Fetcher = Callable[[str, float], str]


def default_state_path() -> Path:
    return Path.home() / ".local" / "share" / "memory-arbiter" / "update_state.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _version_parts(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw in version.split("."):
        digits = ""
        for ch in raw:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def compare_versions(left: str, right: str) -> int:
    left_parts = _version_parts(left)
    right_parts = _version_parts(right)
    max_len = max(len(left_parts), len(right_parts))
    left_full = left_parts + (0,) * (max_len - len(left_parts))
    right_full = right_parts + (0,) * (max_len - len(right_parts))
    if left_full < right_full:
        return -1
    if left_full > right_full:
        return 1
    return 0


def _default_fetcher(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": f"{PACKAGE_NAME}/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310: fixed PyPI HTTPS URL
        return resp.read().decode("utf-8")


class UpdateMonitor:
    def __init__(
        self,
        enabled: bool = True,
        state_path: Optional[Path] = None,
        current_version: str = __version__,
        fetcher: Fetcher = _default_fetcher,
        now_func: Callable[[], datetime] = _now,
    ) -> None:
        self.enabled = enabled
        self.state_path = state_path or default_state_path()
        self.current_version = current_version
        self._fetcher = fetcher
        self._now = now_func
        self._lock = threading.RLock()
        self._check_thread: Optional[threading.Thread] = None
        self._state = self._load_state()
        try:
            self._observe_installed_version()
        except Exception:
            pass

    def maybe_start_check_if_due(self) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                if self._check_thread is not None and self._check_thread.is_alive():
                    return
                if not self._check_due_locked():
                    return
                self._check_thread = threading.Thread(
                    target=self._run_one_check,
                    name="memory-arbiter-update-check",
                    daemon=True,
                )
                self._check_thread.start()
        except Exception:
            return

    def consume_notices(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        self.maybe_start_check_if_due()
        notices: list[dict[str, Any]] = []
        try:
            with self._lock:
                self._reload_state_locked()
                self._observe_installed_version(write=False)
                update_notice = self._update_available_notice_locked()
                if update_notice is not None:
                    notices.append(update_notice)
                    self._state["last_update_notified_version"] = self._state.get("latest_version")
                    self._state["last_update_notified_at"] = self._now().isoformat()
                doctor_notice = self._post_upgrade_doctor_notice_locked()
                if doctor_notice is not None:
                    notices.append(doctor_notice)
                    self._state["last_post_upgrade_doctor_notified_version"] = self.current_version
                    self._state["last_post_upgrade_doctor_notified_at"] = self._now().isoformat()
                if notices:
                    self._write_state_locked()
        except Exception:
            return []
        return notices

    def update_status(self) -> dict[str, Any]:
        try:
            with self._lock:
                self._reload_state_locked()
                self._observe_installed_version(write=False)
                latest = self._state.get("latest_version")
                status = "disabled" if not self.enabled else "unknown"
                update_available = False
                if self.enabled and isinstance(latest, str) and latest:
                    cmp = compare_versions(self.current_version, latest)
                    update_available = cmp < 0
                    status = "update_available" if update_available else "up_to_date"
                checked_at = self._state.get("latest_checked_at")
                checked_dt = _parse_time(checked_at)
                cache_stale = bool(checked_dt and self._now() - checked_dt >= CHECK_INTERVAL)
                return {
                    "enabled": self.enabled,
                    "status": status,
                    "current_version": self.current_version,
                    "latest_version": latest,
                    "latest_checked_at": checked_at,
                    "latest_source": self._state.get("latest_source"),
                    "cache_stale": cache_stale,
                    "last_check_error": self._state.get("last_check_error"),
                    "next_retry_after": self._state.get("next_retry_after"),
                    "last_doctor_run_version": self._state.get("last_doctor_run_version"),
                    "last_doctor_run_at": self._state.get("last_doctor_run_at"),
                }
        except Exception:
            return {"enabled": self.enabled, "status": "unavailable", "current_version": self.current_version}

    def record_doctor_run(self) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                self._reload_state_locked()
                self._observe_installed_version(write=False)
                self._state["last_doctor_run_version"] = self.current_version
                self._state["last_doctor_run_at"] = self._now().isoformat()
                self._write_state_locked()
        except Exception:
            return

    def _run_one_check(self) -> None:
        try:
            raw = self._fetcher(PYPI_JSON_URL, float(BACKGROUND_TIMEOUT_SECONDS))
            payload = json.loads(raw)
            latest = str((payload.get("info") or {}).get("version") or "").strip()
            if not latest:
                raise ValueError("PyPI response missing info.version")
            with self._lock:
                self._reload_state_locked()
                self._observe_installed_version(write=False)
                self._state["latest_version"] = latest
                self._state["latest_checked_at"] = self._now().isoformat()
                self._state["latest_source"] = "pypi"
                self._state["last_check_error"] = None
                self._state["next_retry_after"] = None
                self._write_state_locked()
        except Exception as exc:
            with self._lock:
                self._reload_state_locked()
                self._observe_installed_version(write=False)
                self._state["last_check_error"] = f"{type(exc).__name__}: {exc}"
                self._state["next_retry_after"] = (self._now() + RETRY_AFTER_FAILURE).isoformat()
                self._write_state_locked()
        finally:
            with self._lock:
                self._check_thread = None

    def _check_due_locked(self) -> bool:
        next_retry = _parse_time(self._state.get("next_retry_after"))
        if next_retry and self._now() < next_retry:
            return False
        checked = _parse_time(self._state.get("latest_checked_at"))
        if checked is None:
            return True
        return self._now() - checked >= CHECK_INTERVAL

    def _update_available_notice_locked(self) -> Optional[dict[str, Any]]:
        latest = self._state.get("latest_version")
        if not isinstance(latest, str) or not latest:
            return None
        if compare_versions(self.current_version, latest) >= 0:
            return None
        if self._suppressed_locked("last_update_notified_version", "last_update_notified_at", latest):
            return None
        return {
            "type": "update_available",
            "severity": "info",
            "current_version": self.current_version,
            "latest_version": latest,
            "message": (
                f"{PACKAGE_NAME} {latest} is available. Upgrade when convenient; "
                "restart the MCP server after upgrading."
            ),
            "suppress_days": NOTICE_SUPPRESS.days,
            "upgrade_hint": f"python -m pip install --upgrade {PACKAGE_NAME}",
            "agent_instruction": "Tell the user once in plain language. Do not auto-upgrade.",
        }

    def _post_upgrade_doctor_notice_locked(self) -> Optional[dict[str, Any]]:
        previous = self._state.get("previous_installed_version")
        if not isinstance(previous, str) or not previous or previous == self.current_version:
            return None
        if self._state.get("last_doctor_run_version") == self.current_version:
            return None
        if self._suppressed_locked(
            "last_post_upgrade_doctor_notified_version",
            "last_post_upgrade_doctor_notified_at",
            self.current_version,
        ):
            return None
        return {
            "type": "post_upgrade_doctor_recommended",
            "severity": "info",
            "previous_version": previous,
            "current_version": self.current_version,
            "message": (
                f"{PACKAGE_NAME} was upgraded from {previous} to {self.current_version}. "
                "Please run `memory-arbiter doctor` once to verify database/vector/schema health after the upgrade."
            ),
            "suppress_days": NOTICE_SUPPRESS.days,
            "agent_instruction": "Tell the user once in plain language. Do not run doctor automatically unless the user asks.",
        }

    def _suppressed_locked(self, version_key: str, at_key: str, version: str) -> bool:
        if self._state.get(version_key) != version:
            return False
        notified_at = _parse_time(self._state.get(at_key))
        if notified_at is None:
            return False
        return self._now() - notified_at < NOTICE_SUPPRESS

    def _observe_installed_version(self, write: bool = True) -> None:
        changed = False
        seen = self._state.get("installed_version_seen")
        if seen != self.current_version:
            if isinstance(seen, str) and seen:
                self._state["previous_installed_version"] = seen
            else:
                self._state["previous_installed_version"] = None
            self._state["installed_version_seen"] = self.current_version
            changed = True
        latest = self._state.get("latest_version")
        if isinstance(latest, str) and compare_versions(self.current_version, latest) >= 0:
            self._state["last_update_notified_version"] = None
            self._state["last_update_notified_at"] = None
            changed = True
            if compare_versions(self.current_version, latest) > 0:
                self._state["latest_version"] = self.current_version
                self._state["latest_source"] = "installed_version"
        if changed and write:
            self._write_state_locked()

    def _load_state(self) -> dict[str, Any]:
        try:
            with self.state_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _reload_state_locked(self) -> None:
        self._state = self._load_state()

    def _write_state_locked(self) -> None:
        if not self.enabled:
            return
        try:
            existing = self._load_state()
            if existing:
                existing.update(self._state)
                self._state = existing
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._state, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.write("\n")
            tmp.replace(self.state_path)
        except OSError:
            return
