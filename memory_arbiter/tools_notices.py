"""epoch 布防与通知投递 mixin（从 tools.py 搬出，拆分批 ⑥ 纯移动）。"""
from __future__ import annotations

import hashlib
import time
from typing import Any, TYPE_CHECKING
from .config import Settings
from .db import MemoryDB
from .scan_pipeline import ScanPipeline
from .constants import SCAN_TASK_RECHECK_SECONDS, SCAN_TASK_STALE_DAYS
from .scan_tasks import AGENT_INSTRUCTION as _SCAN_AGENT_INSTRUCTION, SCHEDULED_TASKS_SPEC as _SCAN_TASKS_SPEC

if TYPE_CHECKING:
    from .update_monitor import UpdateMonitor
    from .config import Settings
    from .db import MemoryDB
    from .scan_pipeline import ScanPipeline

class _ToolsNotices:
    if TYPE_CHECKING:
        db: "MemoryDB"
        settings: "Settings"
        _scan_pipeline: "ScanPipeline"
        _update_monitor: "UpdateMonitor | None"
        _last_backup_notice_signature: "tuple[int, int, int, bool] | None"
        _last_backup_source_signature: "tuple[int, int, int] | None"
        def current_agent_id(self) -> "str | None": ...
        def _setup_health(self) -> "dict[str, Any]": ...

    def _arm_scan_epoch_if_needed(self) -> None:
        """Boot-time epoch arm (plan §6⑪): detector bump ⇒ full round.

        Two insurance paths for the first full scan coexist by design: the
        watermark-NULL default on never-scanned memories, and THIS arm —
        clearing every watermark when the running detector identity is newer
        than the persisted one. Also expires pending queue rows whose
        identity semantics belong to the old detector epoch, so the queue is
        re-populated under the new semantics instead of mixing epochs.
        """
        try:
            from .db_generation import CONFLICT_DETECTOR_VERSION

            arm = self.db.meta.scan_epoch_arm()
            current_to = str((arm or {}).get("to") or "")
            if current_to == CONFLICT_DETECTOR_VERSION:
                return
            cleared = self.db.clear_all_scan_watermarks()
            try:
                with self.db.write_transaction() as conn:
                    # Owner standing rule (0.16.2): a new detector epoch
                    # re-judges the whole library, so EVERY queue row is
                    # stale-semantics residue — DELETE outright (work queue,
                    # not an archive; all decision outcomes live in
                    # conflicts/memories). Deleting also releases the
                    # candidate_key_hash identities, so the full round can
                    # re-enqueue every pair the new classifier keeps without
                    # UNIQUE collisions.
                    conn.execute("DELETE FROM scan_queue")
                    # NOT internal_conflicts: the owner's no-resurrection rule
                    # (judged pairs never re-judged across generations) wins
                    # over the P2-5 unit↔row index-key overlap — that overlap
                    # may suppress a bounded set of internal findings for
                    # never-edited memories (documented plan known-limit);
                    # version bumps on any edit rebuild the rows anyway.
            except Exception:
                pass
            self.db.meta.record_scan_epoch_arm(
                previous=current_to or None,
                current=CONFLICT_DETECTOR_VERSION,
                reason=f"detector identity change at boot (cleared {cleared} watermarks)",
            )
        except Exception:
            # Arm is best-effort at boot; the watermark-NULL path still covers
            # never-scanned memories even if this fails.
            pass

    def _detect_full_scan_notice(self) -> dict[str, Any] | None:
        """E9 ③ side-channel: one-shot full_scan_required notice.

        Fires while an armed epoch round has not completed, regardless of any
        scheduled task — agents without one learn immediately that a full
        scan is pending. One-shot per detector epoch (a 7-day suppression
        window keyed by the armed identity keeps the multi-kick first round
        from repeating the notice on every response) and self-closing: gone
        once the pipeline round completes under the armed detector.
        """
        arm = self.db.meta.scan_epoch_arm()
        if not arm:
            return None
        from .db_generation import CONFLICT_DETECTOR_VERSION

        if str(arm.get("to") or "") != CONFLICT_DETECTOR_VERSION:
            return None
        state = self.db.meta.scan_pipeline_state() or {}
        if state.get("complete") and str(state.get("detector_version") or "") == CONFLICT_DETECTOR_VERSION:
            return None
        monitor = self._update_monitor
        if monitor is not None:
            try:
                from .update_monitor import NOTICE_SUPPRESS

                state_key = f"full_scan_notice:{CONFLICT_DETECTOR_VERSION}"
                notice_state = monitor.read_state_key(state_key)
                notice_state = notice_state if isinstance(notice_state, dict) else {}
                last_at = float(notice_state.get("last_at") or 0)
                now = time.time()
                if last_at and now - last_at < NOTICE_SUPPRESS.total_seconds():
                    return None
                monitor.write_state_key(state_key, {"last_at": now})
            except Exception:
                pass
        return {
            "type": "full_scan_required",
            "severity": "warning",
            "notice_id": f"full-scan-required-{CONFLICT_DETECTOR_VERSION}",
            "message": (
                f"Conflict detection upgraded to {CONFLICT_DETECTOR_VERSION} "
                f"(from {arm.get('from') or 'none'}): one full scan round is pending. "
                "Run memory_repair(task='scan_pipeline', data={'action': 'kick'}) repeatedly "
                "until complete=true, then clear the judgment queue "
                "(memory_repair task='scan_queue')."
            ),
        }

    def _detect_scheduled_task_notice(self, now_epoch: float) -> dict[str, Any] | None:
        """Three-tier detection over conflict-scan activity evidence."""
        from .timeutil import parse_iso8601_utc
        from .update_monitor import NOTICE_SUPPRESS

        scan_state = self.db.conflict_scan_state()
        base = {
            "notice_id": "scheduled-scan-tasks",
            "notice_version": "v1",
            "suppress_days": NOTICE_SUPPRESS.days,
            "agent_instruction": _SCAN_AGENT_INSTRUCTION,
            "setup": _SCAN_TASKS_SPEC,
        }
        if scan_state.get("required"):
            return {
                **base, "type": "scan_required", "severity": "warning",
                "message": (
                    "A detector/boundary change requires one full conflict scan; run the "
                    "conflict_scan task (or page scan_candidates manually) to completion."
                ),
            }
        last = self.db._scan_log_last_completed()
        if last is None and not scan_state.get("progress"):
            return {
                **base, "type": "scan_never_run", "severity": "info",
                "message": (
                    "No conflict scan has ever completed on this library; the conflict-detection "
                    "pipeline is silently idle without the scheduled tasks."
                ),
            }
        if last is not None:
            scanned_at = parse_iso8601_utc(last.get("scan_time"))
            if scanned_at is not None:
                stale_seconds = float(SCAN_TASK_STALE_DAYS) * 86400.0
                if now_epoch - scanned_at.timestamp() > stale_seconds:
                    return {
                        **base, "type": "scan_stale", "severity": "info",
                        "message": (
                            f"Last completed conflict scan was {last.get('scan_time')}; "
                            f"scan activity has been stale for over {SCAN_TASK_STALE_DAYS} days."
                        ),
                        "last_scan_time": last.get("scan_time"),
                    }
        return None

    def _scheduled_task_notice_state_key(self) -> str:
        """Per-library suppression key.

        The notice CONDITION is per-library (scan_log.jsonl next to each
        db_path) but UpdateMonitor's state file is per-user home: a shared
        key would let one healthy library suppress another library's
        guidance for the whole window (round-2 finding M1).
        """
        digest = hashlib.sha256(str(self.settings.db_path).encode("utf-8")).hexdigest()[:12]
        return f"scheduled_task_notice:{digest}"

    def _scheduled_task_notices(self) -> list[dict[str, Any]]:
        """Scheduled-task guidance notice, self-closing on scan evidence.

        A completed full-scan boundary (a scan_candidates page returning
        next_anchor_memory_id=null with anchors scanned) appends one audit
        line to scan_log.jsonl, and a required rebuild records conflict_scan
        pages — either is the machine-checkable proof that a task exists,
        and once it appears the trigger condition (and this notice) goes
        away. Suppression rides the
        shared notice-state file; a negative-cache timestamp keeps the
        scan_log.jsonl re-read off every product response. Only an actual
        delivery advances the suppression window (last_at); a clean check
        refreshes just the 1h negative cache, so a library that later goes
        stale is re-detected within the hour, not after the whole 7-day
        window (round-2 finding M2).
        """
        monitor = self._update_monitor
        if monitor is None:
            return []
        try:
            from .update_monitor import NOTICE_SUPPRESS

            now = time.time()
            state_key = self._scheduled_task_notice_state_key()
            state = monitor.read_state_key(state_key)
            state = state if isinstance(state, dict) else {}
            last_at = float(state.get("last_at") or 0)
            if last_at and now - last_at < NOTICE_SUPPRESS.total_seconds():
                return []
            checked_at = float(state.get("checked_at") or 0)
            if checked_at and now - checked_at < SCAN_TASK_RECHECK_SECONDS:
                return []
            notice = self._detect_scheduled_task_notice(now)
            monitor.write_state_key(state_key, {
                "type": notice.get("type") if notice else None,
                "last_at": now if notice else last_at,
                "checked_at": now,
            })
            return [notice] if notice else []
        except Exception:
            return []

    def _consume_notices(self) -> list[dict[str, Any]]:
        from .tools import _BACKUP_NOTICE_DEGRADED_SIGNATURE, _BACKUP_NOTICE_EMPTY_SIGNATURE
        notices: list[dict[str, Any]] = []
        if self._update_monitor is not None:
            onboarding = self._update_monitor.consume_agent_onboarding_notice(self.current_agent_id())
            for notice in onboarding:
                if notice.get("type") == "agent_onboarding":
                    # First call per agent doubles as the health card (P1): the
                    # agent sees capability gaps immediately instead of waiting
                    # for a human to run doctor.
                    notice["health"] = self._setup_health()
            notices.extend(onboarding)
            notices.extend(self._update_monitor.consume_notices())
            notices.extend(self._scheduled_task_notices())
            try:
                full_scan = self._detect_full_scan_notice()
                if full_scan is not None:
                    notices.append(full_scan)
            except Exception:
                pass
        try:
            source_signature = self.db.backup_replay.state_signature()
            if source_signature == self._last_backup_source_signature:
                return notices
            self._last_backup_source_signature = source_signature
            inspection = self.db.backup_replay.inspect(limit=10_000, offset=0)
            signature = (
                int(inspection.get("importable") or 0),
                int(inspection.get("invalid") or 0),
                int(inspection.get("conflicts") or 0),
                bool(inspection.get("has_more")),
            )
            if signature != _BACKUP_NOTICE_EMPTY_SIGNATURE and signature != self._last_backup_notice_signature:
                notices.append({
                    "type": "backup_replay_pending",
                    "severity": "warning",
                    "pending_records": signature[0],
                    "invalid_records": signature[1],
                    "conflicting_receipts": signature[2],
                    "additional_pages": signature[3],
                    "action_required": "preview_backup_replay",
                    "suggested_call": {
                        "tool": "memory_repair", "task": "replay_backup",
                        "data": {"dry_run": True},
                    },
                })
                self._last_backup_notice_signature = signature
            elif signature == _BACKUP_NOTICE_EMPTY_SIGNATURE:
                self._last_backup_notice_signature = None
        except Exception as exc:
            if self._last_backup_notice_signature != _BACKUP_NOTICE_DEGRADED_SIGNATURE:
                notices.append({
                    "type": "backup_replay_notice_degraded",
                    "severity": "warning",
                    "reason": str(exc),
                    "action_required": "inspect_backup_replay_manually",
                    "suggested_call": {
                        "tool": "memory_repair", "task": "replay_backup",
                        "data": {"dry_run": True, "limit": 200, "offset": 0},
                    },
                })
                self._last_backup_notice_signature = _BACKUP_NOTICE_DEGRADED_SIGNATURE
        return notices
