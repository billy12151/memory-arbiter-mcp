"""Deterministic doctor fixtures shared by the golden generator and its test.

These live under tests/ rather than scripts/ because they are test
infrastructure: scripts/gen_golden_doctor.py is a thin CLI over them, so the
snapshot and the assertions are built from exactly the same fixtures.

Every timestamp is relative to "now", so derived values (age_days, idle_days,
the 7-day window) are stable whenever this runs. Fixtures deliberately never
create the sqlite-vec virtual tables, so a snapshot is identical with and
without the extension installed -- CI runs both.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.db_generation import CONFLICT_DETECTOR_VERSION
from memory_arbiter.doctor import report_to_dict, run_all_checks
from memory_arbiter.scan_tasks import SCHEDULED_TASKS_SPEC_VERSION

# Absolute values that legitimately differ between two runs of the same
# fixture. Masked by explicit path -- never by pattern matching, because a
# pattern would also swallow age_days / idle_days / counts, and those are the
# logic under test.
MASK_TS = "<TS>"
MASK_PATH = "<PATH>"
EVIDENCE_MASKS: dict[str, tuple[str, ...]] = {
    "conflicts.scan_stale": ("last_scan_time",),
    "conflicts.backlog": ("latest_triage_at",),
    "conflicts.scan_epoch": ("at",),
    "conflicts.scan_chain": ("at",),
    "database.schema_generation": ("migration_completed_at",),
}
PATH_MASKS: dict[str, tuple[str, ...]] = {
    "workspace.review": ("sidecar",),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(delta: timedelta) -> str:
    return (_now() + delta).isoformat()


def build_settings(root: Path, *, with_model: bool = True, config_warnings: list[str] | None = None) -> Settings:
    model = root / "fake.gguf"
    model.write_bytes(b"fake-model-bytes")
    settings = Settings(
        db_path=root / "m.sqlite3",
        backup_jsonl=root / "backup.jsonl",
        embedding_model_path=model if with_model else None,
        embedding_auto_write=True,
        client="golden",
        agent_id="golden",
    )
    if config_warnings:
        settings.config_warnings = list(config_warnings)
    return settings


def _exec(db: MemoryDB, sql: str, params: tuple[Any, ...] = ()) -> None:
    with db.write_transaction() as conn:
        conn.execute(sql, params)


def _memory(db: MemoryDB, mid: int, subject: str, workspace: str = "ws", tags: str = '["t"]') -> None:
    now = _iso(timedelta(0))
    _exec(
        db,
        """INSERT INTO memories(id,content,agent_id,workspace,workspace_canonical,tags,
             source_type,event_time,ingest_time,confidence,protection_level,status,
             subject,metadata,version,created_at)
           VALUES(?,?,'golden',?,?,?,'agent_generated',?,?,0.5,'normal','active',?,'{}',1,?)""",
        (mid, f"body for {subject}", workspace, workspace, tags, now, now, subject, now),
    )


def _evidence(db: MemoryDB, memory_id: int, text: str, *, unit_index: int = 0) -> None:
    now = _iso(timedelta(0))
    _exec(
        db,
        """INSERT INTO memory_evidence(memory_id,memory_version,content_hash,unit_index,
             kind,text,start_offset,end_offset,created_at)
           VALUES(?,1,?,?,'body',?,0,?,?)""",
        (memory_id, hashlib.sha256(text.encode()).hexdigest(), unit_index, text, len(text), now),
    )


def _conflict(db: MemoryDB, cid: int, status: str) -> None:
    now = _iso(timedelta(0))
    key = json.dumps({"cid": cid})
    slot = json.dumps({"slot": cid})
    # open/applying groups carry a slot key by schema constraint.
    _exec(
        db,
        """INSERT INTO conflicts(id,workspace_canonical,slot_key,slot_key_hash,candidate_key,
             candidate_key_hash,status,member_versions,member_fingerprint,value_groups,
             detection_reason,source,detector_version,created_at,refreshed_at)
           VALUES(?,'ws',?,?,?,?,?,'[]',?,'[]','golden fixture','scan_pipeline',?,?,?)""",
        (cid, slot, hashlib.sha256(slot.encode()).hexdigest(), key,
         hashlib.sha256(key.encode()).hexdigest(), status,
         hashlib.sha256(f"fp{cid}".encode()).hexdigest(), CONFLICT_DETECTOR_VERSION, now, now),
    )


# --------------------------------------------------------------------------
# Fixtures. Every timestamp is built relative to "now" so derived values
# (age_days, idle_days, 7-day windows) are stable whenever this runs.
# --------------------------------------------------------------------------

def fx_empty(root: Path) -> tuple[Settings, MemoryDB]:
    settings = build_settings(root)
    return settings, MemoryDB(settings)


def fx_indexed(root: Path) -> tuple[Settings, MemoryDB]:
    settings = build_settings(root)
    db = MemoryDB(settings)
    for mid in (1, 2, 3):
        _memory(db, mid, f"subject {mid}")
        _evidence(db, mid, f"unit for {mid}")
    return settings, db


def fx_stale_orphan_evidence(root: Path) -> tuple[Settings, MemoryDB]:
    settings, db = fx_indexed(root)
    # memory_version drift = stale.
    _exec(db, "UPDATE memory_evidence SET memory_version=99 WHERE memory_id=1")
    # Orphan = evidence whose memory is gone. The FK cascade normally prevents
    # this, so reproduce the real-world shape (a cascade that never fired) by
    # dropping the parent row with foreign keys off.
    raw = sqlite3.connect(str(settings.db_path))
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        raw.execute("DELETE FROM memories WHERE id=3")
        raw.commit()
    finally:
        raw.close()
    return settings, db


def fx_conflict_backlog(root: Path) -> tuple[Settings, MemoryDB]:
    settings, db = fx_indexed(root)
    now = _iso(timedelta(0))
    for cid, status in ((1, "open"), (2, "open"), (3, "applying"), (4, "not_a_conflict")):
        _conflict(db, cid, status)
    _exec(
        db,
        """INSERT INTO scan_queue(kind,workspace_canonical,status,candidate_key_hash,
             member_versions,evidence,reason,severity,source,detail,created_at,updated_at)
           VALUES('workspace','ws','pending',?, '[]','[]','vector vote','normal',
                  'scan_pipeline',NULL,?,?)""",
        ("a" * 64, now, now),
    )
    _exec(
        db,
        """INSERT INTO internal_conflicts(memory_id,memory_version,status,unit_a,unit_b,
             quote_a,quote_b,span_a,span_b,detector_version,created_at,updated_at)
           VALUES(1,1,'pending',0,1,'port 8080','port 9090','[0,9]','[10,19]',?,?,?)""",
        (CONFLICT_DETECTOR_VERSION, now, now),
    )
    return settings, db


def _arm_epoch(db: MemoryDB, armed_to: str) -> None:
    payload = json.dumps(
        {"from": "previous-detector", "to": armed_to, "at": _iso(timedelta(minutes=-5)),
         "reason": "detector version changed"}
    )
    _exec(db, "INSERT OR REPLACE INTO migration_state(key,value) VALUES('scan_epoch_armed',?)", (payload,))


def fx_epoch_fresh(root: Path) -> tuple[Settings, MemoryDB]:
    settings, db = fx_indexed(root)
    _arm_epoch(db, CONFLICT_DETECTOR_VERSION)
    return settings, db


def fx_epoch_stale(root: Path) -> tuple[Settings, MemoryDB]:
    settings, db = fx_indexed(root)
    _arm_epoch(db, "detector-from-an-older-binary")
    return settings, db


def fx_scan_chain_broken(root: Path) -> tuple[Settings, MemoryDB]:
    settings, db = fx_indexed(root)
    payload = json.dumps(
        {"complete": False, "after": 10, "next_anchor": 123,
         "at": _iso(timedelta(hours=-6)), "client": "golden", "groups": 2}
    )
    _exec(db, "INSERT OR REPLACE INTO migration_state(key,value) VALUES('scan_page_progress',?)", (payload,))
    return settings, db


def fx_spec_drift(root: Path) -> tuple[Settings, MemoryDB]:
    settings, db = fx_indexed(root)
    # A completed scan 20 days ago proves a task exists; the stale spec stamp
    # says it still runs the v1 contract. Also trips conflicts.scan_stale.
    log = Path(settings.db_path).parent / "scan_log.jsonl"
    log.write_text(
        json.dumps({"status": "completed", "scan_time": _iso(timedelta(days=-20))}) + "\n",
        encoding="utf-8",
    )
    _exec(
        db,
        "INSERT OR REPLACE INTO migration_state(key,value) VALUES('scheduled_tasks_spec_confirmed','1')",
    )
    return settings, db


def fx_scan_log_unparseable_time(root: Path) -> tuple[Settings, MemoryDB]:
    """A completed scan whose timestamp will not parse: no staleness verdict."""
    settings, db = fx_indexed(root)
    log = Path(settings.db_path).parent / "scan_log.jsonl"
    log.write_text(
        json.dumps({"status": "completed", "scan_time": "not-a-timestamp"}) + "\n",
        encoding="utf-8",
    )
    return settings, db


def fx_scan_chain_recent(root: Path) -> tuple[Settings, MemoryDB]:
    """An in-flight chain younger than the alarm threshold stays quiet."""
    settings, db = fx_indexed(root)
    payload = json.dumps(
        {"complete": False, "after": 10, "next_anchor": 77,
         "at": _iso(timedelta(minutes=-5)), "client": "golden", "groups": 1}
    )
    _exec(db, "INSERT OR REPLACE INTO migration_state(key,value) VALUES('scan_page_progress',?)", (payload,))
    return settings, db


def fx_scan_chain_unparseable_at(root: Path) -> tuple[Settings, MemoryDB]:
    """Progress kv present but its timestamp is unreadable: no verdict."""
    settings, db = fx_indexed(root)
    payload = json.dumps({"complete": False, "next_anchor": 5, "at": "not-a-timestamp"})
    _exec(db, "INSERT OR REPLACE INTO migration_state(key,value) VALUES('scan_page_progress',?)", (payload,))
    return settings, db


def fx_spec_stamp_current(root: Path) -> tuple[Settings, MemoryDB]:
    """A task running the served spec produces no drift finding."""
    settings, db = fx_indexed(root)
    log = Path(settings.db_path).parent / "scan_log.jsonl"
    log.write_text(
        json.dumps({"status": "completed", "scan_time": _iso(timedelta(days=-1))}) + "\n",
        encoding="utf-8",
    )
    _exec(
        db,
        "INSERT OR REPLACE INTO migration_state(key,value)"
        " VALUES('scheduled_tasks_spec_confirmed',?)",
        (str(SCHEDULED_TASKS_SPEC_VERSION),),
    )
    return settings, db


def fx_hygiene_backlog(root: Path) -> tuple[Settings, MemoryDB]:
    from memory_arbiter.constants import MAX_MEMORY_TOTAL_TAGS

    settings = build_settings(root, config_warnings=["deprecated key ignored: legacy.option"])
    db = MemoryDB(settings)
    _memory(db, 1, "over-tagged", tags=json.dumps([f"t{i}" for i in range(MAX_MEMORY_TOTAL_TAGS + 3)]))
    _memory(db, 2, "normal one", workspace="other-ws")
    attention = Path(settings.db_path).parent / "attention_log.jsonl"
    attention.write_text(
        "".join(json.dumps({"ts": _iso(timedelta(days=-1))}) + "\n" for _ in range(3))
        + json.dumps({"ts": _iso(timedelta(days=-30))}) + "\n"
        # A torn line (crash mid-append) is counted but not parsed.
        + "{not json\n",
        encoding="utf-8",
    )
    return settings, db


def fx_workspace_registry_unreadable(root: Path) -> tuple[Settings, MemoryDB]:
    """workspace.review degrades to pass when the registry cannot be read.

    deep=False only: doctor.py's vector.workspace_rows counts
    workspace_canonicals with no guard, so a deep run raises OperationalError
    here. That is current behaviour, recorded in the plan's appendix B.
    """
    settings, db = fx_indexed(root)
    _exec(db, "ALTER TABLE workspace_canonicals RENAME TO workspace_canonicals_hidden")
    return settings, db


def fx_corrupt_kv_and_tags(root: Path) -> tuple[Settings, MemoryDB]:
    """Every JSON-decode fallback at once.

    doctor must degrade rather than raise when a migration_state value or a
    memories.tags column is not parseable -- that is what keeps a health check
    usable on a library that is already damaged.
    """
    settings = build_settings(root)
    db = MemoryDB(settings)
    _memory(db, 1, "bad tags", tags="{not json")
    for key in ("scan_epoch_armed", "scan_page_progress"):
        _exec(db, "INSERT OR REPLACE INTO migration_state(key,value) VALUES(?,'{oops')", (key,))
    log = Path(settings.db_path).parent / "scan_log.jsonl"
    log.write_text(
        json.dumps({"status": "completed", "scan_time": _iso(timedelta(days=-3))}) + "\n",
        encoding="utf-8",
    )
    _exec(
        db,
        "INSERT OR REPLACE INTO migration_state(key,value)"
        " VALUES('scheduled_tasks_spec_confirmed','not-an-int')",
    )
    return settings, db


def fx_epoch_armed_not_an_object(root: Path) -> tuple[Settings, MemoryDB]:
    """Valid JSON that is not an object: parsed fine, then skipped."""
    settings, db = fx_indexed(root)
    _exec(db, "INSERT OR REPLACE INTO migration_state(key,value) VALUES('scan_epoch_armed','[1,2]')")
    return settings, db


def fx_applying_unparseable_timestamp(root: Path) -> tuple[Settings, MemoryDB]:
    """idle_days stays None when refreshed_at cannot be read."""
    settings, db = fx_indexed(root)
    _conflict(db, 1, "applying")
    _exec(db, "UPDATE conflicts SET refreshed_at='not-a-timestamp' WHERE id=1")
    return settings, db


def fx_applying_naive_timestamp(root: Path) -> tuple[Settings, MemoryDB]:
    """A refreshed_at written without an offset is read as UTC, not rejected."""
    settings, db = fx_indexed(root)
    _conflict(db, 1, "applying")
    naive = (_now() - timedelta(days=3)).replace(tzinfo=None).isoformat()
    _exec(db, "UPDATE conflicts SET refreshed_at=? WHERE id=1", (naive,))
    return settings, db


def fx_side_tables_missing(root: Path) -> tuple[Settings, MemoryDB]:
    """scan_queue / internal_conflicts / normalize_audit unreadable.

    These three are guarded (unlike workspace_canonicals, see
    fx_workspace_registry_unreadable), so doctor reports zeros instead of
    raising. Dropped with foreign keys off so the drop itself succeeds.
    """
    settings, db = fx_indexed(root)
    raw = sqlite3.connect(str(settings.db_path))
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        for table in ("scan_queue", "internal_conflicts", "normalize_audit"):
            raw.execute(f"DROP TABLE IF EXISTS {table}")
        raw.commit()
    finally:
        raw.close()
    return settings, db


def fx_default_fallback_audit(root: Path) -> tuple[Settings, MemoryDB]:
    """normalize.autonomy's fallback note only appears when the gate flag is set."""
    settings, db = fx_indexed(root)
    now = _iso(timedelta(0))
    for mid, status in ((1, "applied"), (2, "rolled_back")):
        _exec(
            db,
            "INSERT INTO normalize_audit(memory_id,from_workspace,to_workspace,gate,status,created_at)"
            " VALUES(?,'ws','other','{}',?,?)",
            (mid, status, now),
        )
    _exec(
        db,
        "INSERT INTO normalize_audit(memory_id,from_workspace,to_workspace,gate,status,created_at)"
        " VALUES(1,'ws','default',?, 'manual_move',?)",
        (json.dumps({"default_fallback": 1}), now),
    )
    return settings, db


FIXTURES: list[tuple[str, Callable[[Path], tuple[Settings, MemoryDB]], tuple[bool, ...]]] = [
    ("empty", fx_empty, (False, True)),
    ("indexed", fx_indexed, (False, True)),
    ("stale_orphan_evidence", fx_stale_orphan_evidence, (False, True)),
    ("conflict_backlog", fx_conflict_backlog, (False, True)),
    ("epoch_fresh", fx_epoch_fresh, (False, True)),
    ("epoch_stale", fx_epoch_stale, (False, True)),
    ("scan_chain_broken", fx_scan_chain_broken, (False, True)),
    ("spec_drift", fx_spec_drift, (False, True)),
    ("hygiene_backlog", fx_hygiene_backlog, (False, True)),
    ("workspace_registry_unreadable", fx_workspace_registry_unreadable, (False,)),
    ("corrupt_kv_and_tags", fx_corrupt_kv_and_tags, (False,)),
    ("epoch_armed_not_an_object", fx_epoch_armed_not_an_object, (False,)),
    ("applying_unparseable_timestamp", fx_applying_unparseable_timestamp, (False,)),
    ("applying_naive_timestamp", fx_applying_naive_timestamp, (False,)),
    ("side_tables_missing", fx_side_tables_missing, (False,)),
    ("default_fallback_audit", fx_default_fallback_audit, (False,)),
    ("scan_log_unparseable_time", fx_scan_log_unparseable_time, (False,)),
    ("scan_chain_recent", fx_scan_chain_recent, (False,)),
    ("scan_chain_unparseable_at", fx_scan_chain_unparseable_at, (False,)),
    ("spec_stamp_current", fx_spec_stamp_current, (False,)),
]


# --------------------------------------------------------------------------
# Probe stubs. Named for the outcome they force, not for their internals.
# --------------------------------------------------------------------------

class _StubEmbedder:
    """Drives every branch of the three embedder-dependent checks.

    vector.device reports four distinct states (GPU, CPU, degraded-to-CPU, and
    degraded-with-a-failed-rebuild, which is DOWN rather than merely slow), and
    evidence.unit_budget three (clean scan, units over the token budget, and a
    tokenizer that raises). All of them are reachable only through this stub.
    """

    def __init__(
        self,
        *,
        embed_raises: bool = False,
        expose_budget: bool = True,
        gpu_backed: bool = True,
        degraded: bool = False,
        degraded_at: str | None = None,
        tokens_per_char: int = 1,
        tokenize_raises: bool = False,
        budget: int = 512,
    ) -> None:
        self._embed_raises = embed_raises
        self._tokens_per_char = tokens_per_char
        self._tokenize_raises = tokenize_raises
        self._budget = budget
        self.gpu_backed = gpu_backed
        self.device_degraded = degraded
        self.device_degraded_at = degraded_at
        if not expose_budget:
            self.tokenize_locked = None  # type: ignore[assignment]

    def embed_text(self, **_: Any) -> Any:
        if self._embed_raises:
            raise RuntimeError("embed exploded")
        return type("R", (), {"embedding": [0.0] * 8})()

    def tokenize_locked(self, text: str) -> list[int]:
        if self._tokenize_raises:
            raise RuntimeError("tokenizer exploded")
        return [0] * (len(text) * self._tokens_per_char)

    def token_budget(self) -> int:
        return self._budget


def _probe_raises() -> tuple[Any, list[str]]:
    raise RuntimeError("probe exploded")


def _prep_active_dim(match: bool) -> Callable[[Settings, MemoryDB], None]:
    """Persist an active dim so vector.dimension_probe compares instead of
    reporting "no stored active dim yet"."""

    def prep(settings: Settings, db: MemoryDB) -> None:
        _exec(db, "INSERT OR REPLACE INTO _vec_index_meta(key,value) VALUES('active_dim',?)",
              (str(8 if match else 1536),))

    return prep


def _prep_unreadable_attention_log(settings: Settings, db: MemoryDB) -> None:
    """A directory where the log should be: exists() passes, open() raises."""
    (Path(settings.db_path).parent / "attention_log.jsonl").mkdir()


def _prep_unreadable_model_path(settings: Settings, db: MemoryDB) -> None:
    """A directory where the GGUF should be, so the configured-space probe
    raises OSError and vector.space falls back to "configured=none"."""
    model = Path(settings.db_path).parent / "model_dir.gguf"
    model.mkdir()
    settings.embedding_model_path = model


PROBES: list[tuple[str, Callable[[], tuple[Any, list[str]]] | None, bool, Callable[[Settings, MemoryDB], None] | None]] = [
    ("probe_absent", None, True, None),
    ("probe_raises", _probe_raises, True, None),
    ("probe_returns_none_model_configured", lambda: (None, []), True, None),
    ("probe_returns_none_model_absent", lambda: (None, []), False, None),
    ("embedder_ok", lambda: (_StubEmbedder(), []), True, None),
    ("embedder_embed_raises", lambda: (_StubEmbedder(embed_raises=True), []), True, None),
    ("embedder_without_budget", lambda: (_StubEmbedder(expose_budget=False), []), True, None),
    ("embedder_on_cpu", lambda: (_StubEmbedder(gpu_backed=False), []), True, None),
    ("embedder_degraded_to_cpu",
     lambda: (_StubEmbedder(degraded=True, degraded_at="2026-01-01T00:00:00+00:00"), []), True, None),
    ("embedder_degraded_rebuild_failed", lambda: (_StubEmbedder(degraded=True), []), True, None),
    # 200-char units with a 4x tokenizer against a 64-token budget: the
    # char prefilter (budget // 4) admits them and they blow the budget.
    ("embedder_units_over_budget",
     lambda: (_StubEmbedder(tokens_per_char=4, budget=64), []), True, None),
    ("embedder_tokenizer_raises", lambda: (_StubEmbedder(tokenize_raises=True), []), True, None),
    ("embedder_dim_matches_stored", lambda: (_StubEmbedder(), []), True, _prep_active_dim(True)),
    ("embedder_dim_mismatches_stored", lambda: (_StubEmbedder(), []), True, _prep_active_dim(False)),
    ("unreadable_attention_log", lambda: (_StubEmbedder(), []), True, _prep_unreadable_attention_log),
    ("unreadable_model_path", lambda: (_StubEmbedder(), []), True, _prep_unreadable_model_path),
]


# --------------------------------------------------------------------------

def _mask(report: dict[str, Any]) -> dict[str, Any]:
    report["snapshot_ts"] = MASK_TS
    for finding in report["findings"]:
        check = finding["check_id"]
        evidence = finding.get("evidence") or {}
        for key in EVIDENCE_MASKS.get(check, ()):  # absolute timestamps
            if key in evidence and evidence[key] is not None:
                original = str(evidence[key])
                evidence[key] = MASK_TS
                # The same timestamp is usually interpolated into detail; mask
                # it by value so a 20-day age or an anchor id is left intact.
                finding["detail"] = finding["detail"].replace(original, MASK_TS)
        for key in PATH_MASKS.get(check, ()):
            if key in evidence and evidence[key] is not None:
                original = str(evidence[key])
                evidence[key] = MASK_PATH
                finding["detail"] = finding["detail"].replace(original, MASK_PATH)
        if check == "conflicts.applying":
            for group in evidence.get("groups") or []:
                group["refreshed_at"] = MASK_TS
        # Version-bound values. CONFLICT_DETECTOR_VERSION is a distinctive
        # string and is safe to replace wholesale; SCHEDULED_TASKS_SPEC_VERSION
        # is the integer 4, so it is masked by key and only its prefixed form
        # ("serves v4") is touched in detail -- a bare "4" replacement would
        # corrupt the counts this gate exists to compare.
        if check == "conflicts.scan_epoch":
            evidence["to"] = "<DETECTOR_VER>"
            finding["detail"] = finding["detail"].replace(CONFLICT_DETECTOR_VERSION, "<DETECTOR_VER>")
        if check == "conflicts.spec_drift":
            evidence["served_spec_version"] = "<SPEC_VER>"
            finding["detail"] = finding["detail"].replace(
                f"serves v{SCHEDULED_TASKS_SPEC_VERSION}", "serves v<SPEC_VER>"
            )
    return report


def _snapshot(settings: Settings, db: MemoryDB, deep: bool, probe: Any = None) -> dict[str, Any]:
    with db.diagnostic_connection() as conn:
        report = run_all_checks(conn, settings, deep=deep, embedder_probe=probe)
    return _mask(report_to_dict(report))


def build_snapshots() -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for name, factory, deep_modes in FIXTURES:
        for deep in deep_modes:
            root = Path(tempfile.mkdtemp(prefix=f"golden_{name}_"))
            try:
                settings, db = factory(root)
                snapshots.append({
                    "case_id": f"{name}/deep={int(deep)}",
                    "report": _snapshot(settings, db, deep),
                })
            finally:
                shutil.rmtree(root, ignore_errors=True)

    for probe_name, probe, with_model, prep in PROBES:
        root = Path(tempfile.mkdtemp(prefix=f"golden_probe_{probe_name}_"))
        try:
            settings = build_settings(root, with_model=with_model)
            db = MemoryDB(settings)
            for mid in (1, 2):
                _memory(db, mid, f"probe subject {mid}")
                _evidence(db, mid, "x" * 200)
            if prep is not None:
                prep(settings, db)
            snapshots.append({
                "case_id": f"probe/{probe_name}",
                "report": _snapshot(settings, db, True, probe),
            })
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return snapshots


