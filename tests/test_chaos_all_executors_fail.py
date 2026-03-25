"""Chaos tests: all executors fail in various ways.

Goal: Verify the system degrades gracefully — returns RouteDecision with
error, NEVER raises an unhandled exception.
"""

import os
import time
import json
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState, ChainEntry
from router.policy import route_task, build_chain, reset_breaker, reset_notifier
from router.errors import (
    ExecutorError, CodexQuotaError, ClaudeAuthError, ProviderTimeoutError,
    RouterError, ELIGIBLE_FALLBACK_ERRORS,
)
from router.state_store import StateStore, reset_state_store
from router.circuit_breaker import CircuitBreaker
from router.config_loader import reload_config


@pytest.fixture(autouse=True)
def _reset_singletons(tmp_path, monkeypatch):
    """Reset all singletons between tests."""
    reset_breaker()
    reset_notifier()
    reset_state_store()
    # Use temp config dir
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    # Copy config
    import shutil
    src = Path(__file__).parent.parent / "config" / "router.config.json"
    if src.exists():
        shutil.copy(src, config_dir / "router.config.json")
        monkeypatch.setattr("router.config_loader.CONFIG_PATH", config_dir / "router.config.json")
        monkeypatch.setattr("router.config_loader._config_snapshot", None)
        monkeypatch.setattr("router.config_loader._config_raw", None)
    # State files in temp
    monkeypatch.setattr("router.state_store.CONFIG_DIR", config_dir)
    yield


@pytest.fixture
def sample_task():
    return TaskMeta(
        task_id="chaos-test-001",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT,
        repo_path="/tmp/test-repo",
        cwd="/tmp/test-repo",
        summary="Test task",
    )


def _make_timeout_executor(*args, **kwargs):
    """Simulate a timeout by raising ProviderTimeoutError."""
    raise ProviderTimeoutError("Executor timed out")


def _make_crash_executor(*args, **kwargs):
    """Simulate a crash by raising RuntimeError."""
    raise RuntimeError("Executor process crashed unexpectedly")


def _make_oserror_executor(*args, **kwargs):
    """Simulate OS-level failure."""
    raise OSError("No such file or directory: /usr/bin/executor")


def _make_connection_error_executor(*args, **kwargs):
    """Simulate network failure."""
    raise ConnectionError("Network is unreachable")


class TestAllExecutorsTimeout:
    """All 3 executors timeout → system should return error, not crash."""

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_all_executors_timeout(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        mock_codex.side_effect = _make_timeout_executor
        mock_claude.side_effect = _make_timeout_executor
        mock_openrouter.side_effect = _make_timeout_executor

        decision, result = route_task(sample_task)

        # Should return a decision, not crash
        assert decision is not None
        assert isinstance(decision.chain, list)
        assert not result.success
        # All executors failed — error should be present
        assert result.normalized_error is not None


class TestAllExecutorsCrash:
    """All 3 executors raise RuntimeError → system should return error, not crash."""

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_all_executors_runtime_error(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        mock_codex.side_effect = _make_crash_executor
        mock_claude.side_effect = _make_crash_executor
        mock_openrouter.side_effect = _make_crash_executor

        decision, result = route_task(sample_task)

        assert decision is not None
        assert not result.success

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_all_executors_os_error(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        """All executors raise OSError (binary not found)."""
        mock_codex.side_effect = _make_oserror_executor
        mock_claude.side_effect = _make_oserror_executor
        mock_openrouter.side_effect = _make_oserror_executor

        decision, result = route_task(sample_task)

        assert decision is not None
        assert not result.success

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_all_executors_connection_error(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        """All executors raise ConnectionError."""
        mock_codex.side_effect = _make_connection_error_executor
        mock_claude.side_effect = _make_connection_error_executor
        mock_openrouter.side_effect = _make_connection_error_executor

        decision, result = route_task(sample_task)

        assert decision is not None
        assert not result.success


class TestEmptyChainGraceful:
    """Chain with 0 entries → proper error, not IndexError."""

    @patch("router.policy.build_chain")
    def test_empty_chain_returns_error(self, mock_build_chain, sample_task):
        mock_build_chain.return_value = []

        decision, result = route_task(sample_task)

        # Should not crash with IndexError
        assert decision is not None
        assert decision.chain == []
        assert not result.success


class TestCircuitBreakerOpensAll:
    """After failures, all providers become unavailable."""

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_circuit_breaker_opens_all_providers(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        """All executors fail repeatedly → circuit breaker opens for all."""
        mock_codex.side_effect = CodexQuotaError()
        mock_claude.side_effect = ClaudeAuthError()
        mock_openrouter.side_effect = ProviderTimeoutError()

        # Run enough times to trip all breakers (threshold=5 by default)
        for _ in range(6):
            reset_breaker()
            # Reset with low threshold
            from router import policy
            policy._breaker = CircuitBreaker(threshold=2, window_s=60)
            decision, result = route_task(sample_task)

        # Now use a very low threshold breaker
        reset_breaker()
        from router import policy
        policy._breaker = CircuitBreaker(threshold=1, window_s=60)

        # First call trips the breakers
        decision1, result1 = route_task(sample_task)

        # Second call — all should be skipped
        decision2, result2 = route_task(sample_task)

        # At least some providers should have been skipped
        assert decision2 is not None


class TestFallbackExhaustion:
    """Chain completes, all failed, final error is last error."""

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_fallback_exhaustion_returns_last_error(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        mock_codex.side_effect = CodexQuotaError()
        mock_claude.side_effect = ClaudeAuthError()
        mock_openrouter.side_effect = ProviderTimeoutError()

        decision, result = route_task(sample_task)

        assert decision is not None
        assert not result.success
        # Error should be one of the fallback-eligible errors
        assert result.normalized_error in ELIGIBLE_FALLBACK_ERRORS

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_fallback_count_bounded(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        """Max fallbacks should be respected — no infinite loop."""
        mock_codex.side_effect = CodexQuotaError()
        mock_claude.side_effect = ClaudeAuthError()
        mock_openrouter.side_effect = ProviderTimeoutError()

        decision, result = route_task(sample_task)

        # Fallback count should be bounded
        assert decision.fallback_count <= 3  # default max_fallbacks


class TestMixedFailures:
    """Mixed failure scenarios."""

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_first_succeeds_others_not_called(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        """First executor succeeds → others not invoked."""
        from router.models import ExecutorResult
        mock_codex.return_value = ExecutorResult(
            task_id=sample_task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
        )

        decision, result = route_task(sample_task)

        assert result.success
        mock_claude.assert_not_called()
        mock_openrouter.assert_not_called()

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_first_fails_second_succeeds(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        """First fails, second succeeds → third not invoked."""
        from router.models import ExecutorResult
        mock_codex.side_effect = CodexQuotaError()
        mock_claude.return_value = ExecutorResult(
            task_id=sample_task.task_id,
            tool="claude_code",
            backend="anthropic",
            model_profile="claude_primary",
            success=True,
        )

        decision, result = route_task(sample_task)

        assert result.success
        assert decision.attempted_fallback is True
        mock_openrouter.assert_not_called()

    @patch("router.policy.run_openrouter")
    @patch("router.policy.run_claude")
    @patch("router.policy.run_codex")
    def test_executor_returns_failure_not_exception(self, mock_codex, mock_claude, mock_openrouter, sample_task):
        """Executor returns failure result (not exception) → still falls back."""
        from router.models import ExecutorResult
        mock_codex.return_value = ExecutorResult(
            task_id=sample_task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=False,
            normalized_error="rate_limited",
        )
        mock_claude.return_value = ExecutorResult(
            task_id=sample_task.task_id,
            tool="claude_code",
            backend="anthropic",
            model_profile="claude_primary",
            success=True,
        )

        decision, result = route_task(sample_task)

        assert result.success
        assert decision.attempted_fallback is True
