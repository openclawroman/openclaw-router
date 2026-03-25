"""Tests for error propagation with full fallback trail (item 5.4).

When routing falls back through multiple executors, the full error trail
is captured so the caller knows what failed before.
"""

import pytest
from unittest.mock import MagicMock

from router.models import (
    TaskMeta, RouteDecision, ExecutorResult, ChainEntry,
    TaskClass, TaskRisk, TaskModality, CodexState,
)
from router.policy import route_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**kwargs):
    defaults = dict(
        task_id="test-err-prop",
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
        summary="test error propagation",
    )
    defaults.update(kwargs)
    return TaskMeta(**defaults)


def _make_success_result(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini"):
    """Create a successful ExecutorResult."""
    return ExecutorResult(
        task_id="test-err-prop",
        tool=tool,
        backend=backend,
        model_profile=model_profile,
        success=True,
        normalized_error=None,
        final_summary="Task completed successfully",
    )


def _make_failure_result(tool="codex_cli", backend="openai_native",
                         model_profile="codex_gpt54_mini",
                         error_type="rate_limited",
                         final_summary="Rate limit exceeded"):
    """Create a failed ExecutorResult."""
    return ExecutorResult(
        task_id="test-err-prop",
        tool=tool,
        backend=backend,
        model_profile=model_profile,
        success=False,
        normalized_error=error_type,
        final_summary=final_summary,
    )


def _default_chain():
    """Default 3-entry chain matching openai_primary state."""
    return [
        MagicMock(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini"),
        MagicMock(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        MagicMock(tool="codex_cli", backend="openrouter", model_profile="openrouter_minimax"),
    ]


def _patch_router(monkeypatch, chain=None, executor_results=None):
    """Common monkeypatch setup for router tests."""
    monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
    monkeypatch.setattr(
        "router.policy.build_chain",
        lambda task, state: chain or _default_chain(),
    )
    monkeypatch.setattr(
        "router.policy.get_reliability_config",
        lambda: {"chain_timeout_s": 600, "max_fallbacks": 3},
    )

    # Mock breaker — always available
    breaker = MagicMock()
    breaker.is_available.return_value = True
    monkeypatch.setattr("router.policy.get_breaker", lambda: breaker)

    # Mock notifier
    notifier = MagicMock()
    monkeypatch.setattr("router.policy.get_notifier", lambda: notifier)

    # Mock attempt logger
    monkeypatch.setattr("router.policy.AttemptLogger", lambda: MagicMock())

    # Mock shutdown manager
    shutdown_mgr = MagicMock()
    shutdown_mgr.should_accept_new_tasks.return_value = True
    monkeypatch.setattr("router.health.get_shutdown_manager", lambda: shutdown_mgr)

    # Mock executor
    if executor_results:
        _results = iter(executor_results)
        monkeypatch.setattr(
            "router.policy._run_executor",
            lambda entry, task, trace_id="": next(_results),
        )


# ---------------------------------------------------------------------------
# Model fields
# ---------------------------------------------------------------------------

class TestErrorHistoryModelFields:
    """error_history field exists on ExecutorResult and RouteDecision."""

    def test_executor_result_has_error_history(self):
        result = ExecutorResult()
        assert hasattr(result, "error_history")
        assert result.error_history == []

    def test_route_decision_has_error_history(self):
        decision = RouteDecision()
        assert hasattr(decision, "error_history")
        assert decision.error_history == []

    def test_error_history_entry_schema(self):
        entry = {
            "tool": "codex_cli",
            "backend": "openai_native",
            "model_profile": "codex_gpt54_mini",
            "error_type": "rate_limited",
            "error_message": "Rate limit exceeded",
        }
        for key in ("tool", "backend", "model_profile", "error_type", "error_message"):
            assert key in entry


# ---------------------------------------------------------------------------
# Single executor succeeds → error_history is empty
# ---------------------------------------------------------------------------

class TestSingleExecutorSuccess:
    """When a single executor succeeds, error_history is empty."""

    def test_success_empty_error_history(self, monkeypatch):
        """Single successful executor → empty error_history."""
        chain = [MagicMock(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini")]
        _patch_router(monkeypatch, chain=chain, executor_results=[_make_success_result()])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert result.error_history == []
        assert decision.error_history == []


# ---------------------------------------------------------------------------
# First executor fails, second succeeds → error_history has 1 entry
# ---------------------------------------------------------------------------

class TestFirstFailsSecondSucceeds:
    """First executor fails (fallback-eligible), second succeeds."""

    def test_fallback_error_history_one_entry(self, monkeypatch):
        """First fails with rate_limited, second succeeds → 1 error entry."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="rate_limited",
                final_summary="Rate limit exceeded for openai",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert len(result.error_history) == 1
        assert result.error_history[0]["tool"] == "codex_cli"
        assert result.error_history[0]["backend"] == "openai_native"
        assert result.error_history[0]["error_type"] == "rate_limited"
        assert decision.error_history == result.error_history


# ---------------------------------------------------------------------------
# All executors fail → error_history has N entries
# ---------------------------------------------------------------------------

class TestAllExecutorsFail:
    """All executors in the chain fail → error_history has all failures."""

    def test_all_fail_error_history_three_entries(self, monkeypatch):
        """3-executor chain, all fail → error_history has 3 entries."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="rate_limited", final_summary="Rate limited",
            ),
            _make_failure_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
                error_type="quota_exhausted", final_summary="Quota exhausted",
            ),
            _make_failure_result(
                tool="codex_cli", backend="openrouter",
                model_profile="openrouter_minimax",
                error_type="provider_unavailable", final_summary="Provider down",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is False
        assert len(result.error_history) == 3

        assert result.error_history[0]["tool"] == "codex_cli"
        assert result.error_history[0]["error_type"] == "rate_limited"

        assert result.error_history[1]["tool"] == "claude_code"
        assert result.error_history[1]["error_type"] == "quota_exhausted"

        assert result.error_history[2]["tool"] == "codex_cli"
        assert result.error_history[2]["error_type"] == "provider_unavailable"

        assert decision.error_history == result.error_history


# ---------------------------------------------------------------------------
# Error history captures tool, backend, model_profile, error_type
# ---------------------------------------------------------------------------

class TestErrorHistoryCapturesAllFields:
    """Each error entry captures all 5 required fields."""

    def test_error_entry_fields(self, monkeypatch):
        """Each error_history entry has all required fields with correct values."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="auth_error",
                final_summary="Authentication token expired",
            ),
            _make_success_result(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        ])

        decision, result = route_task(_make_task())

        entry = result.error_history[0]
        assert entry["tool"] == "codex_cli"
        assert entry["backend"] == "openai_native"
        assert entry["model_profile"] == "codex_gpt54_mini"
        assert entry["error_type"] == "auth_error"
        assert entry["error_message"] == "Authentication token expired"

    def test_error_message_from_final_summary(self, monkeypatch):
        """error_message is populated from final_summary of the failed result."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="rate_limited",
                final_summary="429 Too Many Requests: retry after 60s",
            ),
            _make_success_result(),
        ])

        _, result = route_task(_make_task())

        assert result.error_history[0]["error_message"] == "429 Too Many Requests: retry after 60s"


# ---------------------------------------------------------------------------
# Error history is preserved in final ExecutorResult
# ---------------------------------------------------------------------------

class TestErrorHistoryPreservedInResult:
    """The final ExecutorResult carries the full error_history."""

    def test_success_result_has_history(self, monkeypatch):
        """Successful result after fallback still has error_history from prior failures."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(error_type="transient_network_error"),
            _make_failure_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary", error_type="rate_limited",
            ),
            _make_success_result(tool="codex_cli", backend="openrouter"),
        ])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert len(result.error_history) == 2
        # Verify it's the same object reference as decision
        assert decision.error_history is result.error_history

    def test_failure_result_has_history(self, monkeypatch):
        """Failed result (all executors fail) has error_history of all failures."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(error_type="auth_error"),
            _make_failure_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary", error_type="quota_exhausted",
            ),
            _make_failure_result(
                tool="codex_cli", backend="openrouter",
                model_profile="openrouter_minimax", error_type="provider_unavailable",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is False
        assert len(result.error_history) == 3
        assert decision.error_history is result.error_history


# ---------------------------------------------------------------------------
# RouteDecision also has error_history for logging
# ---------------------------------------------------------------------------

class TestRouteDecisionErrorHistory:
    """RouteDecision carries error_history for observability/logging."""

    def test_decision_has_error_history_on_success(self, monkeypatch):
        """RouteDecision.error_history is populated even on successful route."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(error_type="rate_limited"),
            _make_success_result(tool="claude_code", backend="anthropic"),
        ])

        decision, result = route_task(_make_task())

        assert len(decision.error_history) == 1
        assert decision.error_history[0]["error_type"] == "rate_limited"

    def test_decision_error_history_empty_on_no_failures(self, monkeypatch):
        """RouteDecision.error_history is empty when no failures occurred."""
        _patch_router(monkeypatch, executor_results=[_make_success_result()])

        decision, result = route_task(_make_task())

        assert decision.error_history == []
        assert result.error_history == []

    def test_decision_error_history_matches_result(self, monkeypatch):
        """decision.error_history and result.error_history are the same object."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(error_type="provider_unavailable"),
            _make_failure_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary", error_type="auth_error",
            ),
            _make_success_result(tool="codex_cli", backend="openrouter"),
        ])

        decision, result = route_task(_make_task())

        assert decision.error_history is result.error_history
        assert len(decision.error_history) == 2


# ---------------------------------------------------------------------------
# Non-fallback errors: still captured if they stop the chain
# ---------------------------------------------------------------------------

class TestNonFallbackErrors:
    """Non-fallback-eligible errors still get captured in error_history."""

    def test_non_fallback_error_in_history(self, monkeypatch):
        """A non-fallback error (e.g. toolchain_error) is still in error_history."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                error_type="toolchain_error",
                final_summary="gcc not found",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is False
        assert len(result.error_history) == 1
        assert result.error_history[0]["error_type"] == "toolchain_error"
        assert result.error_history[0]["error_message"] == "gcc not found"
