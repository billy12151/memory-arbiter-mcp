"""Shared test fixtures (structure batch, 2026-10-11).

clear_config_env was previously defined in 4 test files and drifted: only
test_config.py's copy carried the 4 WORKSPACE_RECALL_* keys added later, so
the other three leaked those env vars across files. One fixture here, one
key list.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "MEMORY_ARBITER_CONFIG",
        "MEMORY_ARBITER_DB_PATH",
        "MEMORY_ARBITER_BACKUP_JSONL",
        "MEMORY_ARBITER_POLICY",
        "MEMORY_ARBITER_CLIENT",
        "MEMORY_ARBITER_AGENT_ID",
        "MEMORY_ARBITER_WORKSPACE",
        "MEMORY_ARBITER_ENABLE_SQLITE_VEC",
        "MEMORY_ARBITER_VEC_DIM",
        "MEMORY_ARBITER_WORKSPACE_RECALL_ADMISSION",
        "MEMORY_ARBITER_WORKSPACE_RECALL_CUTOFF",
        "MEMORY_ARBITER_WORKSPACE_WEAK_VECTOR_WEIGHT",
        "MEMORY_ARBITER_WORKSPACE_MIN_NAME_LEN",
        "MEMORY_ARBITER_RECALL_POOL_CAP",
        "MEMORY_ARBITER_CONTENT_LIKE_CAP",
        "MEMORY_ARBITER_EMBEDDING_PROVIDER",
        "MEMORY_ARBITER_EMBEDDING_MODEL_PATH",
        "MEMORY_ARBITER_EMBEDDING_AUTO_QUERY",
        "MEMORY_ARBITER_EMBEDDING_AUTO_WRITE",
        "MEMORY_ARBITER_GGUF",
        "MEMORY_ARBITER_TOOL_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)
