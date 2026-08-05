from __future__ import annotations

from memory_arbiter.config_registry import CONFIG_DESCRIPTORS, grouped_descriptors


def test_config_descriptors_cover_required_console_settings() -> None:
    paths = {item["path"] for item in CONFIG_DESCRIPTORS}
    required = {
        "db_path",
        "backup_jsonl",
        "policy_path",
        "recall_pool_cap",
        "content_like_cap",
        "superseded_limit",
        "vec.enabled",
        "vec.dim",
        "embedding.provider",
        "embedding.model_path",
        "embedding.auto_query",
        "embedding.auto_write",
        "embedding.n_ctx",
        "embedding.reserved_tokens",
        "structured_claim_mode",
        "isolation",
        "workspace_match_distance",
        "split.threshold",
        "split.section_vec_distance_threshold",
        "split.section_fulltext_threshold",
        "split.max_sections",
        "split.max_section_chars",
        "update_check.enabled",
        "client",
        "agent_id",
        "workspace",
    }
    assert required <= paths


def test_config_descriptors_are_bilingual_and_read_only() -> None:
    for item in CONFIG_DESCRIPTORS:
        assert item["label_en"]
        assert item["label_zh"]
        assert item["desc_en"]
        assert item["desc_zh"]
        assert item["editable"] is False


def test_grouped_descriptors_preserve_items() -> None:
    grouped = grouped_descriptors()
    grouped_count = sum(len(group["items"]) for group in grouped)
    assert grouped_count == len(CONFIG_DESCRIPTORS)
    grouped_keys = {group["key"] for group in grouped}
    assert {"paths", "retrieval", "vector_embedding", "claims", "workspace", "split", "update", "client_identity"} <= grouped_keys
