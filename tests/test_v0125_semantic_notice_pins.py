from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.semantic_conflict import notice_dedupe_key
from memory_arbiter.tools import MemoryTools


def test_semantic_notice_dedupe_key_includes_claim_revisions():
    base = notice_dedupe_key(1, 2, 1, 1, "semantic_pair", 1, 1)
    bumped = notice_dedupe_key(1, 2, 1, 1, "semantic_pair", 2, 1)

    assert base == notice_dedupe_key(2, 1, 1, 1, "semantic_pair", 1, 1)
    assert base != bumped


def test_dismissed_semantic_notice_does_not_close_claim_bumped_pair(tmp_path: Path):
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "m.jsonl")
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    left = tools.memory_write(content="默认模型推荐 MiniCPM。", subject="model choice", tags=["mema", "model"])["data"]
    right = tools.memory_write(content="默认模型改为 Qwen。", subject="model choice", tags=["mema", "model"])["data"]
    lv = tools.db.get_memory_version(left["id"]) or 1
    rv = tools.db.get_memory_version(right["id"]) or 1
    lcr = (tools.db.get_memory(left["id"]) or {}).get("claim_revision") or 1
    rcr = (tools.db.get_memory(right["id"]) or {}).get("claim_revision") or 1
    notice = tools.db.record_semantic_notice(
        memory_id=left["id"], peer_id=right["id"], severity="normal",
        notice_type="semantic_pair", title="test", message="msg", payload={},
        dedupe_key="claim-pinned", left_version=lv, right_version=rv,
        left_claim_revision=lcr, right_claim_revision=rcr,
    )
    assert notice["outcome"] == "created"
    tools.db.update_semantic_notice_status(notice["notice_id"], "dismissed")

    assert tools.db.is_semantic_pair_closed(
        left["id"], right["id"], lv, rv,
        left_claim_revision=lcr, right_claim_revision=rcr,
    )
    assert not tools.db.is_semantic_pair_closed(
        left["id"], right["id"], lv, rv,
        left_claim_revision=lcr + 1, right_claim_revision=rcr,
    )
