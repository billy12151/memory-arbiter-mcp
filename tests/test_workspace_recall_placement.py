from pathlib import Path
import hashlib
import pytest
from memory_arbiter.acl import raw_workspace
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryStatus
from memory_arbiter.tools import MemoryTools
from tests.test_workspace_recall_isolation import make_tools  # noqa: E402

@pytest.fixture(autouse=True)
def _relevance_floor_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0.15.9: the tests in this file exercise scoping/preview/metering with
    # content-only fixtures; the query-recall relevance floor is orthogonal to
    # what they pin (its behavior lives in tests/test_recall_quality_0159.py).
    # Disable it here so weak fixtures stay recallable.
    monkeypatch.setattr("memory_arbiter.search.QUERY_RECALL_SCORE_FLOOR", -1.0)

class _StubEmbedder:
    embedding_space_id = "stub"
    last_encode_error = None

    @staticmethod
    def embed_text(prefix, body, max_body_chars=None):
        from memory_arbiter.embedder import EmbedResult
        return EmbedResult([1.0, 0.0], False, len(body), len(body))


def _placement_tools(tmp_path):
    t = make_tools(tmp_path, "none")
    t._embedder = _StubEmbedder()
    t._embedder_loaded = True
    t.db.state.sqlite_vec_available = True
    return t


def test_placement_suggestion_for_empty_workspace(tmp_path):
    from memory_arbiter.models import MemoryRecord
    t = _placement_tools(tmp_path)
    # Nearest neighbor lives in a real workspace.
    # 0.16.12 P2-T5: hits carry m.status/workspace fields (production SELECT)
    t.db.row_knn = lambda *a, **k: [{"memory_id": 42, "distance": 5.0, "status": "active",
                                          "workspace": "金营项目", "workspace_canonical": "金营项目"}]  # type: ignore
    t.db.get_memory = lambda mid: {"id": 42, "status": "active", "workspace": "金营项目",  # type: ignore
                                   "workspace_canonical": "金营项目"} if mid == 42 else None
    rec = MemoryRecord.from_input(
        {"content": "排期正文", "subject": "金营项目 Q3 排期", "workspace": ""},
        t.settings.defaults(),
    )
    ws = t._write_pipeline._resolve_write_workspace(rec)
    assert str(ws["canonical"]).casefold() in {"", "default"}   # NOT auto-assigned
    sug = ws["placement_suggestion"]
    assert sug and sug["suggested_workspace"] == "金营项目"
    assert sug["from_memory_id"] == 42


def test_no_placement_suggestion_when_neighbor_is_default(tmp_path):
    from memory_arbiter.models import MemoryRecord
    t = _placement_tools(tmp_path)
    # Only default-workspace neighbors → global memory stays in default, no hint.
    t.db.row_knn = lambda *a, **k: [{"memory_id": 7, "distance": 5.0, "status": "superseded",
                                          "workspace": "别家项目", "workspace_canonical": "别家项目"}]  # type: ignore
    t.db.get_memory = lambda mid: {"id": 7, "status": "active", "workspace": "default",  # type: ignore
                                   "workspace_canonical": "default"} if mid == 7 else None
    rec = MemoryRecord.from_input(
        {"content": "偏好正文", "subject": "今天想吃什么", "workspace": ""},
        t.settings.defaults(),
    )
    ws = t._write_pipeline._resolve_write_workspace(rec)
    assert ws["placement_suggestion"] is None


def test_no_placement_suggestion_without_subject(tmp_path):
    from memory_arbiter.models import MemoryRecord
    t = _placement_tools(tmp_path)
    t.db.row_knn = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not embed"))  # type: ignore
    rec = MemoryRecord.from_input(
        {"content": "正文", "subject": "", "workspace": ""},
        t.settings.defaults(),
    )
    ws = t._write_pipeline._resolve_write_workspace(rec)
    assert ws["placement_suggestion"] is None


def test_non_default_workspace_gets_no_placement_suggestion(tmp_path):
    from memory_arbiter.models import MemoryRecord
    t = _placement_tools(tmp_path)
    t.db.row_knn = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run for a real ws"))  # type: ignore
    rec = MemoryRecord.from_input(
        {"content": "正文", "subject": "某主题", "workspace": "some-real-project"},
        t.settings.defaults(),
    )
    ws = t._write_pipeline._resolve_write_workspace(rec)
    assert "placement_suggestion" not in ws or ws.get("placement_suggestion") is None


def test_blank_workspace_write_lands_in_default(tmp_path):
    # from_input does not strip: a whitespace-only workspace is truthy and
    # passes through the whole main path. Resolution must still collapse it to
    # the default pool instead of storing an empty/blank canonical.
    from memory_arbiter.models import MemoryRecord

    t = make_tools(tmp_path)
    rec = MemoryRecord.from_input(
        {"content": "空白 workspace 正文", "subject": "空白边界", "workspace": "   "},
        t.settings.defaults(),
    )
    assert rec.workspace == "   "  # truthy blank survives from_input
    memory_id, warnings = t.db.insert_memory(rec, "")
    assert memory_id is not None
    with t.db.connection() as conn:
        row = conn.execute(
            "SELECT workspace, workspace_canonical FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
    assert row["workspace"] == "   "
    assert row["workspace_canonical"] == "default"


def test_insert_memory_empty_canonical_falls_back_to_default(tmp_path):
    # canonical 空串兜底: an explicit empty canonical with a blank raw workspace
    # must become DEFAULT_WORKSPACE_NAME, never an empty string.
    from memory_arbiter.models import MemoryRecord

    t = make_tools(tmp_path)
    rec = MemoryRecord.from_input(
        {"content": "空 canonical 正文", "subject": "空串兜底", "workspace": ""},
        t.settings.defaults(),
    )
    memory_id, _warnings = t.db.insert_memory(rec, "  ")
    assert memory_id is not None
    with t.db.connection() as conn:
        row = conn.execute(
            "SELECT workspace_canonical FROM memories WHERE id=?", (memory_id,),
        ).fetchone()
        registered = conn.execute(
            "SELECT name FROM workspace_canonicals"
        ).fetchall()
    assert row["workspace_canonical"] == "default"
    assert all(str(r["name"]).strip() for r in registered)


# ── from test_workspace_strict_admission.py ──
# helper make_tools renamed: strict_admission_make_tools (collision with test_workspace_isolation.py)
