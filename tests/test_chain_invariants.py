"""Tests for chain invariant validation (item 4.5)."""

import pytest

from router.models import (
    TaskMeta, ChainEntry, CodexState,
    TaskClass, TaskRisk, TaskModality,
)
from router.policy import validate_chain, build_chain
from router.errors import ChainInvariantViolation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**kwargs):
    defaults = dict(
        task_id="test-1",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT,
        requires_repo_write=False,
        requires_multimodal=False,
        has_screenshots=False,
        swarm=False,
        repo_path="/tmp/repo",
        cwd="/tmp/repo",
        summary="test",
    )
    defaults.update(kwargs)
    return TaskMeta(**defaults)


# ---------------------------------------------------------------------------
# validate_chain — openai_primary (all providers allowed)
# ---------------------------------------------------------------------------

class TestValidateChainOpenaiPrimary:
    """openai_primary state: all providers allowed."""

    def test_valid_chain_with_all_backends(self):
        chain = [
            ChainEntry(tool="codex_cli", backend="openai_native", model_profile="x"),
            ChainEntry(tool="claude_code", backend="anthropic", model_profile="x"),
            ChainEntry(tool="codex_cli", backend="openrouter", model_profile="x"),
        ]
        valid, reason = validate_chain(CodexState.OPENAI_PRIMARY, chain)
        assert valid is True

    def test_empty_chain_valid(self):
        valid, reason = validate_chain(CodexState.OPENAI_PRIMARY, [])
        assert valid is True


# ---------------------------------------------------------------------------
# validate_chain — openai_conservation (all providers allowed)
# ---------------------------------------------------------------------------

class TestValidateChainOpenaiConservation:
    """openai_conservation state: all providers allowed."""

    def test_valid_chain_with_all_backends(self):
        chain = [
            ChainEntry(tool="codex_cli", backend="openai_native", model_profile="x"),
            ChainEntry(tool="claude_code", backend="anthropic", model_profile="x"),
        ]
        valid, reason = validate_chain(CodexState.OPENAI_CONSERVATION, chain)
        assert valid is True


# ---------------------------------------------------------------------------
# validate_chain — claude_backup (no openai_native, must start with anthropic)
# ---------------------------------------------------------------------------

class TestValidateChainClaudeBackup:
    """claude_backup state: no openai_native, first entry must be anthropic."""

    def test_valid_chain(self):
        chain = [
            ChainEntry(tool="claude_code", backend="anthropic", model_profile="claude_sonnet"),
            ChainEntry(tool="codex_cli", backend="openrouter", model_profile="x"),
        ]
        valid, reason = validate_chain(CodexState.CLAUDE_BACKUP, chain)
        assert valid is True

    def test_rejects_openai_native(self):
        chain = [
            ChainEntry(tool="claude_code", backend="anthropic", model_profile="x"),
            ChainEntry(tool="codex_cli", backend="openai_native", model_profile="x"),
        ]
        valid, reason = validate_chain(CodexState.CLAUDE_BACKUP, chain)
        assert valid is False
        assert "openai_native" in reason

    def test_rejects_openai_native_first(self):
        chain = [
            ChainEntry(tool="codex_cli", backend="openai_native", model_profile="x"),
        ]
        valid, reason = validate_chain(CodexState.CLAUDE_BACKUP, chain)
        assert valid is False

    def test_rejects_non_anthropic_first(self):
        chain = [
            ChainEntry(tool="codex_cli", backend="openrouter", model_profile="x"),
        ]
        valid, reason = validate_chain(CodexState.CLAUDE_BACKUP, chain)
        assert valid is False
        assert "anthropic" in reason

    def test_empty_chain_valid(self):
        valid, reason = validate_chain(CodexState.CLAUDE_BACKUP, [])
        assert valid is True


# ---------------------------------------------------------------------------
# validate_chain — openrouter_fallback (only openrouter)
# ---------------------------------------------------------------------------

class TestValidateChainOpenrouterFallback:
    """openrouter_fallback state: only openrouter backend allowed."""

    def test_valid_chain(self):
        chain = [
            ChainEntry(tool="codex_cli", backend="openrouter", model_profile="x"),
        ]
        valid, reason = validate_chain(CodexState.OPENROUTER_FALLBACK, chain)
        assert valid is True

    def test_rejects_openai_native(self):
        chain = [
            ChainEntry(tool="codex_cli", backend="openai_native", model_profile="x"),
        ]
        valid, reason = validate_chain(CodexState.OPENROUTER_FALLBACK, chain)
        assert valid is False
        assert "openai_native" in reason

    def test_rejects_anthropic(self):
        chain = [
            ChainEntry(tool="claude_code", backend="anthropic", model_profile="x"),
        ]
        valid, reason = validate_chain(CodexState.OPENROUTER_FALLBACK, chain)
        assert valid is False
        assert "anthropic" in reason

    def test_empty_chain_valid(self):
        valid, reason = validate_chain(CodexState.OPENROUTER_FALLBACK, [])
        assert valid is True


# ---------------------------------------------------------------------------
# build_chain produces valid chains for all 4 states
# ---------------------------------------------------------------------------

class TestBuiltChainsAreValid:
    """All 4 states produce chains that pass invariant validation."""

    def test_openai_primary_chain_valid(self):
        task = _make_task()
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        valid, reason = validate_chain(CodexState.OPENAI_PRIMARY, chain)
        assert valid is True, f"openai_primary chain invalid: {reason}"

    def test_openai_conservation_chain_valid(self):
        task = _make_task()
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        valid, reason = validate_chain(CodexState.OPENAI_CONSERVATION, chain)
        assert valid is True, f"openai_conservation chain invalid: {reason}"

    def test_claude_backup_chain_valid(self):
        task = _make_task()
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        valid, reason = validate_chain(CodexState.CLAUDE_BACKUP, chain)
        assert valid is True, f"claude_backup chain invalid: {reason}"

    def test_openrouter_fallback_chain_valid(self):
        task = _make_task()
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        valid, reason = validate_chain(CodexState.OPENROUTER_FALLBACK, chain)
        assert valid is True, f"openrouter_fallback chain invalid: {reason}"


# ---------------------------------------------------------------------------
# ChainInvariantViolation exception
# ---------------------------------------------------------------------------

class TestChainInvariantViolation:
    """ChainInvariantViolation exception behavior."""

    def test_attributes(self):
        exc = ChainInvariantViolation("claude_backup", "contains openai_native")
        assert exc.state == "claude_backup"
        assert exc.reason == "contains openai_native"
        assert "claude_backup" in str(exc)

    def test_is_router_error(self):
        from router.errors import RouterError
        exc = ChainInvariantViolation("x", "y")
        assert isinstance(exc, RouterError)
