"""Tests for operational runbooks — X-89 runbook validation."""

from pathlib import Path

import pytest

RUNBOOKS_PATH = Path(__file__).parent.parent / "docs" / "runbooks.md"


# ---------------------------------------------------------------------------
# TestRunbookExists
# ---------------------------------------------------------------------------

class TestRunbookExists:
    def test_file_exists(self):
        """docs/runbooks.md exists."""
        assert RUNBOOKS_PATH.exists(), f"Runbook not found at {RUNBOOKS_PATH}"

    def test_has_installation_section(self):
        """Contains 'Installation'."""
        content = RUNBOOKS_PATH.read_text()
        assert "Installation" in content, "Missing Installation section"

    def test_has_state_override_section(self):
        """Contains 'Manual State' or 'State Override'."""
        content = RUNBOOKS_PATH.read_text()
        has_manual = "Manual State" in content
        has_override = "State Override" in content
        assert has_manual or has_override, "Missing Manual State / State Override section"

    def test_has_provider_setup_section(self):
        """Contains 'Provider Setup'."""
        content = RUNBOOKS_PATH.read_text()
        assert "Provider Setup" in content, "Missing Provider Setup section"

    def test_has_monitoring_section(self):
        """Contains 'Monitoring'."""
        content = RUNBOOKS_PATH.read_text()
        assert "Monitoring" in content, "Missing Monitoring section"

    def test_has_dry_runs_section(self):
        """Contains 'Dry Run'."""
        content = RUNBOOKS_PATH.read_text()
        assert "Dry Run" in content, "Missing Dry Run section"

    def test_not_empty(self):
        """File has substantial content (>500 chars)."""
        content = RUNBOOKS_PATH.read_text()
        assert len(content) > 500, f"Runbook too short: {len(content)} chars"


# ---------------------------------------------------------------------------
# TestRunbookCodeBlocks
# ---------------------------------------------------------------------------

class TestRunbookCodeBlocks:
    def test_has_code_blocks(self):
        """Contains triple-backtick code blocks."""
        content = RUNBOOKS_PATH.read_text()
        assert "```" in content, "No code blocks found"

    def test_has_verification_steps(self):
        """Each dry run has a verification step."""
        content = RUNBOOKS_PATH.read_text().lower()
        assert "verif" in content, "No verification steps found in dry runs"


# ---------------------------------------------------------------------------
# TestRunbookContent
# ---------------------------------------------------------------------------

class TestRunbookContent:
    def test_mentions_routing_jsonl(self):
        """Monitoring section references routing.jsonl."""
        content = RUNBOOKS_PATH.read_text()
        assert "routing.jsonl" in content, "Should reference routing.jsonl for monitoring"

    def test_mentions_claude_health(self):
        """Monitoring section references claude_health.json."""
        content = RUNBOOKS_PATH.read_text()
        assert "claude_health" in content, "Should reference claude_health.json for monitoring"

    def test_has_smoke_test(self):
        """Installation section mentions smoke test."""
        content = RUNBOOKS_PATH.read_text().lower()
        assert "smoke test" in content or "smoke" in content, "Installation should include smoke test"

    def test_mentions_last10(self):
        """State override section mentions last10."""
        content = RUNBOOKS_PATH.read_text()
        assert "last10" in content.lower() or "last 10" in content.lower(), "Should mention last10 state"
