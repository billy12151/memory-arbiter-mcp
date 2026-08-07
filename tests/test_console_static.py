from __future__ import annotations

import shutil
import subprocess

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


def test_console_static_shows_resolution_guidance_without_write_actions() -> None:
    assert "resolutionLabel" in INDEX_HTML
    assert "resolutionActionText" in INDEX_HTML
    assert "Partial update" in INDEX_HTML
    assert "完整替代" in INDEX_HTML
    assert "user authorization is still required" in INDEX_HTML


def test_console_static_shows_support_panel_without_github_api() -> None:
    assert "Support mema" in INDEX_HTML
    assert "支持迷码" in INDEX_HTML
    assert "Star on GitHub" in INDEX_HTML
    assert "Request feature" in INDEX_HTML
    assert "Report bug" in INDEX_HTML
    assert "UX feedback" in INDEX_HTML
    assert "体验反馈" in INDEX_HTML
    assert "openFeedback('ux_feedback')" in INDEX_HTML
    assert "['bug','feature','ux_feedback'].includes(type)" in INDEX_HTML
    assert "buildIssueUrl" in INDEX_HTML
    assert "encodeURIComponent" in INDEX_HTML or "URLSearchParams" in INDEX_HTML
    assert "/issues/new" in INDEX_HTML
    assert "Console does not upload your memory automatically" in INDEX_HTML
    forbidden = ["github_token", "oauth", "device flow", "api.github.com", "fetch(supportUrls"]
    lower = INDEX_HTML.lower()
    for word in forbidden:
        assert word.lower() not in lower
    assert "fetch(supporturls" not in lower
    assert "fetch(buildissueurl" not in lower


def test_console_static_javascript_parses(tmp_path) -> None:
    node = shutil.which("node")
    if not node:
        return
    start = INDEX_HTML.index("<script>") + len("<script>")
    end = INDEX_HTML.index("</script>")
    script = INDEX_HTML[start:end]
    script_path = tmp_path / "console.js"
    script_path.write_text(script, encoding="utf-8")
    subprocess.run([node, "--check", str(script_path)], check=True)


def test_console_static_does_not_offer_write_actions() -> None:
    forbidden = ["memory_write", "memory_supersede", "memory_confirm", "memory_resolve_conflict"]
    lower = INDEX_HTML.lower()
    for word in forbidden:
        assert word not in lower
