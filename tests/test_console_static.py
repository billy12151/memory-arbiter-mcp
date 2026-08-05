from __future__ import annotations

from memory_arbiter.console_static import INDEX_HTML


def test_console_static_has_sidebar_language_and_branding() -> None:
    assert "sidebarNav" in INDEX_HTML
    assert "langZh" in INDEX_HTML
    assert "langEn" in INDEX_HTML
    assert "mema Console" in INDEX_HTML
    assert "迷码" in INDEX_HTML
    assert "#/settings" in INDEX_HTML
    assert "loadMemories().catch" in INDEX_HTML
    assert "catch(e)" in INDEX_HTML


def test_console_static_does_not_offer_write_actions() -> None:
    forbidden = ["memory_write", "memory_supersede", "memory_confirm", "memory_resolve_conflict"]
    lower = INDEX_HTML.lower()
    for word in forbidden:
        assert word not in lower
