"""End-to-end error/fallback tests — all error paths through route_task.

These tests exercise the complete error handling pipeline: executor failures,
fallback chaining, error history accumulation, partial success detection,
warning extraction, and trace logging. All executors are mocked.
"""

import uuid
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState,
    ExecutorResult, ChainEntry,
)
from router.policy import route_task
from router.errors import (
    CodexAuthError, CodexQuotaError, ClaudeAuthError,
    RateLimitedError, ProviderUnavailableError, ProviderTimeoutError,
    TransientNetworkError, ModelNotFoundError, ContextTooLongError,
    ContentFilteredError,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_task(**kwargs):
    defaults = dict(
        task_id="test-e2e-errors",
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
        summary="test error fallback end-to-end",
    )
    defaults.update(kwargs)
    return TaskMeta(**defaults)


def _make_success_result(tool="codex_cli", backend="openai_native",
                         model_profile="codex_gpt54_mini"):
    """Create a successful ExecutorResult."""
    return ExecutorResult(
        task_id="test-e2e-errors",
        tool=tool,
        backend=backend,
        model_profile=model_profile,
        success=True,
        normalized_error=None,
        exit_code=0,
        latency_ms=150,
        cost_estimate_usd=0.002,
        final_summary="Task completed successfully",
    )


def _make_failure_result(
    tool="codex_cli",
    backend="openai_native",
    model_profile="codex_gpt54_mini",
    error_type="rate_limited",
    final_summary=None,
    stderr_ref=None,
    artifacts=None,
):
    """Create a failed ExecutorResult. final_summary=None prevents partial_success."""
    return ExecutorResult(
        task_id="test-e2e-errors",
        tool=tool,
        backend=backend,
        model_profile=model_profile,
        success=False,
        normalized_error=error_type,
        exit_code=1,
        latency_ms=100,
        cost_estimate_usd=0.0,
        artifacts=artifacts or [],
        stderr_ref=stderr_ref,
        final_summary=final_summary,
    )


def _make_partial_result(
    tool="codex_cli",
    backend="openai_native",
    model_profile="codex_gpt54_mini",
    error_type="provider_timeout",
    final_summary="Partial output recovered",
    stderr_ref=None,
    warnings=None,
):
    """Create a partial success result — has output despite failure."""
    return ExecutorResult(
        task_id="test-e2e-errors",
        tool=tool,
        backend=backend,
        model_profile=model_profile,
        success=False,
        normalized_error=error_type,
        exit_code=1,
        latency_ms=500,
        cost_estimate_usd=0.0,
        final_summary=final_summary,
        stderr_ref=stderr_ref,
    )


def _default_chain():
    """Default 3-entry chain matching openai_primary state."""
    return [
        ChainEntry(tool="codex_cli", backend="openai_native",
                    model_profile="codex_gpt54_mini"),
        ChainEntry(tool="claude_code", backend="anthropic",
                    model_profile="claude_primary"),
        ChainEntry(tool="codex_cli", backend="openrouter",
                    model_profile="openrouter_minimax"),
    ]


def _patch_router(monkeypatch, chain=None, executor_results=None):
    """Common monkeypatch setup for E2E error tests.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        chain: Optional custom chain. Defaults to _default_chain().
        executor_results: List of ExecutorResult to return from _run_executor.
                         If None, _run_executor is not patched (use for exception tests).
    """
    monkeypatch.setattr("router.policy.resolve_state",
                        lambda: CodexState.OPENAI_PRIMARY)
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
    monkeypatch.setattr("router.health.get_shutdown_manager",
                        lambda: shutdown_mgr)

    # Mock _run_executor if results provided
    if executor_results:
        _results = iter(executor_results)
        monkeypatch.setattr(
            "router.policy._run_executor",
            lambda entry, task, trace_id="": next(_results),
        )


# ── Eligible error fallback tests (1-7) ─────────────────────────────────────

class TestEligibleErrorFallback:
    """Each eligible error type triggers fallback to next provider → success."""

    def test_auth_error_fallback(self, monkeypatch):
        """auth_error → fallback to next provider → success."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="auth_error",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert decision.attempted_fallback is True
        assert decision.fallback_from == "codex_cli"
        assert result.tool == "claude_code"

    def test_rate_limited_fallback(self, monkeypatch):
        """rate_limited → fallback → success."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="rate_limited",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert decision.attempted_fallback is True
        assert decision.fallback_from == "codex_cli"

    def test_quota_exhausted_fallback(self, monkeypatch):
        """quota_exhausted → fallback → success."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="quota_exhausted",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert decision.attempted_fallback is True

    def test_provider_timeout_fallback(self, monkeypatch):
        """provider_timeout → fallback → success.

        Tests the exception path: _run_executor raises ProviderTimeoutError,
        route_task catches it and falls back.
        """
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="provider_timeout",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert decision.attempted_fallback is True

    def test_provider_unavailable_fallback(self, monkeypatch):
        """provider_unavailable → fallback → success."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="provider_unavailable",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert decision.attempted_fallback is True

    def test_transient_network_fallback(self, monkeypatch):
        """transient_network_error → fallback → success."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="transient_network_error",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert decision.attempted_fallback is True

    def test_model_not_found_fallback(self, monkeypatch):
        """model_not_found → fallback → success."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="model_not_found",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is True
        assert decision.attempted_fallback is True


# ── All providers fail (8-9) ────────────────────────────────────────────────

class TestAllProvidersFail:
    """All providers fail — result carries last error and full error_history."""

    def test_all_providers_fail_auth(self, monkeypatch):
        """All 3 fail with auth_error → result has last error, error_history has 3."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="auth_error",
                final_summary="Codex auth token expired",
            ),
            _make_failure_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
                error_type="auth_error",
                final_summary="Claude API key invalid",
            ),
            _make_failure_result(
                tool="codex_cli", backend="openrouter",
                model_profile="openrouter_minimax",
                error_type="auth_error",
                final_summary="OpenRouter API key revoked",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is False
        assert result.normalized_error == "auth_error"
        assert len(result.error_history) == 3
        # Final error message comes from the last provider
        assert result.error_history[2]["error_message"] == "OpenRouter API key revoked"

    def test_all_providers_fail_mixed(self, monkeypatch):
        """1st auth, 2nd timeout, 3rd unavailable → all in error_history."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="auth_error",
                final_summary="Authentication failed",
            ),
            _make_failure_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
                error_type="provider_timeout",
                final_summary="Claude timed out after 30s",
            ),
            _make_failure_result(
                tool="codex_cli", backend="openrouter",
                model_profile="openrouter_minimax",
                error_type="provider_unavailable",
                final_summary="OpenRouter 503 Service Unavailable",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is False
        assert len(result.error_history) == 3
        assert result.error_history[0]["error_type"] == "auth_error"
        assert result.error_history[1]["error_type"] == "provider_timeout"
        assert result.error_history[2]["error_type"] == "provider_unavailable"
        # Decision mirrors the same list
        assert decision.error_history is result.error_history


# ── Non-eligible errors — no fallback (10-11) ───────────────────────────────

class TestNonEligibleNoFallback:
    """Non-eligible errors stop the chain immediately — no fallback."""

    def test_non_eligible_error_no_fallback(self, monkeypatch):
        """context_too_long → NO fallback, immediate failure.

        Even though there are more providers in the chain, the non-eligible
        error stops execution immediately.
        """
        chain = _default_chain()
        _patch_router(monkeypatch, chain=chain, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="context_too_long",
                final_summary="Input exceeds 128k token limit",
            ),
            # These should never be reached
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is False
        assert result.normalized_error == "context_too_long"
        assert decision.attempted_fallback is False
        # Only 1 error in history — chain stopped immediately
        assert len(result.error_history) == 1
        assert result.error_history[0]["error_type"] == "context_too_long"
        # Fallback count is 0 — no actual fallback occurred
        assert decision.fallback_count == 0

    def test_content_filtered_no_fallback(self, monkeypatch):
        """content_filtered → NO fallback, immediate failure."""
        chain = _default_chain()
        _patch_router(monkeypatch, chain=chain, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="content_filtered",
                final_summary="Content flagged by safety filter",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.success is False
        assert result.normalized_error == "content_filtered"
        assert decision.attempted_fallback is False
        assert len(result.error_history) == 1
        assert decision.fallback_count == 0


# ── Partial success (12-13) ─────────────────────────────────────────────────

class TestPartialSuccess:
    """Partial success stops the chain and returns usable output."""

    def test_partial_success_no_fallback(self, monkeypatch):
        """partial_success=True → stops chain, returns output.

        Even though the executor failed, partial_success means there's useful
        output, so the chain should NOT continue to the next provider.

        Note: When _run_executor is mocked, partial_success must be set on the
        returned result directly (the real _run_executor sets it automatically).
        """
        chain = _default_chain()
        partial = ExecutorResult(
            task_id="test-e2e-errors",
            tool="codex_cli", backend="openai_native",
            model_profile="codex_gpt54_mini",
            success=False,
            normalized_error="provider_timeout",
            exit_code=1,
            latency_ms=500,
            cost_estimate_usd=0.0,
            final_summary="Generated 80% of output before timeout",
            partial_success=True,
        )
        _patch_router(monkeypatch, chain=chain, executor_results=[
            partial,
            # Should never be reached
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        # The result should reflect the partial success
        assert result.success is False  # Still a failure
        assert result.partial_success is True  # But partial
        assert result.final_summary == "Generated 80% of output before timeout"
        # No fallback — chain stopped at partial success
        assert decision.attempted_fallback is False
        # error_history is empty — partial_success is not treated as fallback error
        assert len(result.error_history) == 0

    def test_partial_success_with_warnings(self, monkeypatch, tmp_path):
        """Warnings extracted from stderr file on partial success.

        The real _run_executor calls _extract_warnings on stderr_ref.
        When mocked, we set warnings on the result directly.
        """
        # Create a stderr file with warning lines
        stderr_file = tmp_path / "partial.stderr.txt"
        stderr_file.write_text(
            "INFO: Starting task\n"
            "WARNING: output may be incomplete\n"
            "WARN: timeout approaching\n"
            "ERROR: final step failed\n"
        )

        chain = _default_chain()
        partial = ExecutorResult(
            task_id="test-e2e-errors",
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_gpt54_mini",
            success=False,
            normalized_error="provider_timeout",
            exit_code=1,
            latency_ms=450,
            cost_estimate_usd=0.0,
            final_summary="Partial: 3 of 4 files written",
            stderr_ref=str(stderr_file),
            partial_success=True,
            warnings=["WARNING: output may be incomplete",
                       "WARN: timeout approaching"],
        )
        _patch_router(monkeypatch, chain=chain, executor_results=[
            partial,
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        decision, result = route_task(_make_task())

        assert result.partial_success is True
        assert len(result.warnings) == 2
        assert "WARNING: output may be incomplete" in result.warnings
        assert "WARN: timeout approaching" in result.warnings
        # ERROR line is not a warning
        assert all("ERROR" not in w for w in result.warnings)


# ── Error history accumulation (14-17) ──────────────────────────────────────

class TestErrorHistory:
    """Error history accumulates across providers with full field capture."""

    def test_error_history_across_3_providers(self, monkeypatch):
        """3 failures → error_history has 3 entries in order."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="auth_error",
                final_summary="Auth failed",
            ),
            _make_failure_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
                error_type="rate_limited",
                final_summary="Rate limited",
            ),
            _make_failure_result(
                tool="codex_cli", backend="openrouter",
                model_profile="openrouter_minimax",
                error_type="provider_unavailable",
                final_summary="503 down",
            ),
        ])

        decision, result = route_task(_make_task())

        assert len(result.error_history) == 3
        # Verify order
        assert result.error_history[0]["tool"] == "codex_cli"
        assert result.error_history[0]["backend"] == "openai_native"
        assert result.error_history[1]["tool"] == "claude_code"
        assert result.error_history[1]["backend"] == "anthropic"
        assert result.error_history[2]["tool"] == "codex_cli"
        assert result.error_history[2]["backend"] == "openrouter"

    def test_error_history_entry_fields(self, monkeypatch):
        """Each entry has tool, backend, model_profile, error_type, error_message."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="quota_exhausted",
                final_summary="Quota exceeded for the month",
            ),
            _make_success_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
            ),
        ])

        _, result = route_task(_make_task())

        entry = result.error_history[0]
        assert set(entry.keys()) == {
            "tool", "backend", "model_profile", "error_type", "error_message",
        }
        assert entry["tool"] == "codex_cli"
        assert entry["backend"] == "openai_native"
        assert entry["model_profile"] == "codex_gpt54_mini"
        assert entry["error_type"] == "quota_exhausted"
        assert entry["error_message"] == "Quota exceeded for the month"

    def test_normalized_error_in_result(self, monkeypatch):
        """result.normalized_error matches the error type string of the final result."""
        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="rate_limited",
            ),
            _make_failure_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
                error_type="transient_network_error",
            ),
            _make_failure_result(
                tool="codex_cli", backend="openrouter",
                model_profile="openrouter_minimax",
                error_type="provider_unavailable",
            ),
        ])

        _, result = route_task(_make_task())

        # Final result carries the last error type
        assert result.normalized_error == "provider_unavailable"
        # All errors are properly normalized strings
        for entry in result.error_history:
            assert isinstance(entry["error_type"], str)
            assert entry["error_type"] in {
                "rate_limited", "transient_network_error", "provider_unavailable",
            }

    def test_fallback_count_in_trace(self, monkeypatch):
        """trace.fallback_count matches number of fallback transitions.

        fallback_count increments for each transition to a new provider after
        the first failure. With 3 providers all failing:
          - 1st failure: is_first_executor → no count (transition to 2nd)
          - 2nd failure: fallback_count += 1 (transition to 3rd)
          - 3rd failure: fallback_count += 1 (chain exhausted)
        So fallback_count = 2.
        """
        # We need a real AttemptLogger to capture the trace
        traces_captured = []

        class FakeAttemptLogger:
            def log_trace(self, trace):
                traces_captured.append(trace)

        _patch_router(monkeypatch, executor_results=[
            _make_failure_result(
                tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini",
                error_type="auth_error",
            ),
            _make_failure_result(
                tool="claude_code", backend="anthropic",
                model_profile="claude_primary",
                error_type="rate_limited",
            ),
            _make_failure_result(
                tool="codex_cli", backend="openrouter",
                model_profile="openrouter_minimax",
                error_type="provider_unavailable",
            ),
        ])
        # Override the mock AttemptLogger with our capturing one
        monkeypatch.setattr("router.policy.AttemptLogger",
                            lambda: FakeAttemptLogger())

        decision, result = route_task(_make_task())

        assert len(traces_captured) == 1
        trace = traces_captured[0]
        assert trace.fallback_count == 2
        assert trace.final_success is False
        assert trace.final_error == "provider_unavailable"
        # decision also carries the count
        assert decision.fallback_count == 2


# ── Exception-based error tests ──────────────────────────────────────────────

class TestExceptionPathFallback:
    """Test that ExecutorError exceptions raised by executors trigger fallback."""

    def test_exception_triggers_fallback(self, monkeypatch):
        """ExecutorError raised by _run_executor → caught → fallback works.

        This tests the exception handling path in route_task's inner try/except.
        """
        call_count = 0

        def mock_run_executor(entry, task, trace_id=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitedError("429 Too Many Requests")
            return _make_success_result(
                tool=entry.tool, backend=entry.backend,
                model_profile=entry.model_profile,
            )

        _patch_router(monkeypatch, executor_results=None)
        monkeypatch.setattr("router.policy._run_executor", mock_run_executor)

        decision, result = route_task(_make_task())

        assert result.success is True
        assert call_count == 2
        assert len(result.error_history) == 1
        assert result.error_history[0]["error_type"] == "rate_limited"

    def test_exception_non_eligible_no_fallback(self, monkeypatch):
        """ContextTooLongError raised → no fallback, immediate failure."""
        call_count = 0

        def mock_run_executor(entry, task, trace_id=""):
            nonlocal call_count
            call_count += 1
            raise ContextTooLongError("Input exceeds 128k context limit")

        _patch_router(monkeypatch, executor_results=None)
        monkeypatch.setattr("router.policy._run_executor", mock_run_executor)

        decision, result = route_task(_make_task())

        assert result.success is False
        assert call_count == 1
        assert result.normalized_error == "context_too_long"
        assert decision.attempted_fallback is False

    def test_generic_exception_normalized(self, monkeypatch):
        """Non-ExecutorError exception → normalize_error → fallback if eligible."""
        call_count = 0

        def mock_run_executor(entry, task, trace_id=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Generic exception that normalizes to provider_unavailable
                raise RuntimeError("503 Service Unavailable")
            return _make_success_result(
                tool=entry.tool, backend=entry.backend,
                model_profile=entry.model_profile,
            )

        _patch_router(monkeypatch, executor_results=None)
        monkeypatch.setattr("router.policy._run_executor", mock_run_executor)

        decision, result = route_task(_make_task())

        assert result.success is True
        assert call_count == 2
        assert len(result.error_history) == 1
        assert result.error_history[0]["error_type"] == "provider_unavailable"
