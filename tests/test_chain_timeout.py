"""Tests for chain timeout and fallback loop guard."""

import time
import pytest
from unittest.mock import MagicMock

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    CodexState, ExecutorResult,
)
from router.errors import ExecutorError
from router.policy import route_task


class TestChainTimeout:
    """Tests for total chain time budget enforcement."""

    def test_chain_timeout_triggers(self, monkeypatch):
        """If chain timeout is 0s, first executor should be skipped and chain_timed_out=True."""
        monkeypatch.setattr(
            "router.policy.get_reliability_config",
            lambda: {"chain_timeout_s": 0, "max_fallbacks": 3},
        )
        monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
        monkeypatch.setattr(
            "router.policy.build_chain",
            lambda task, state: [
                MagicMock(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini"),
                MagicMock(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
            ],
        )

        def slow_executor(entry, task):
            time.sleep(0.1)
            return ExecutorResult(
                task_id="t1", tool=entry.tool, backend=entry.backend,
                model_profile=entry.model_profile, success=True,
            )

        monkeypatch.setattr("router.policy._run_executor", slow_executor)

        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        decision, result = route_task(task)

        assert decision.chain_timed_out is True
        assert result.normalized_error == "chain_timeout"

    def test_chain_completes_within_timeout(self, monkeypatch):
        """Normal execution within timeout should not trigger timeout."""
        monkeypatch.setattr(
            "router.policy.get_reliability_config",
            lambda: {"chain_timeout_s": 600, "max_fallbacks": 3},
        )
        monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
        monkeypatch.setattr(
            "router.policy.build_chain",
            lambda task, state: [
                MagicMock(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini"),
            ],
        )
        monkeypatch.setattr(
            "router.policy._run_executor",
            lambda entry, task, trace_id="": ExecutorResult(
                task_id="t1", tool=entry.tool, backend=entry.backend,
                model_profile=entry.model_profile, success=True,
            ),
        )

        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        decision, result = route_task(task)

        assert decision.chain_timed_out is False
        assert result.success is True


class TestFallbackLoopGuard:
    """Tests for max fallback attempts enforcement."""

    def test_max_fallbacks_limits_retries(self, monkeypatch):
        """After max_fallbacks failures, chain stops even if more entries exist."""
        monkeypatch.setattr(
            "router.policy.get_reliability_config",
            lambda: {"chain_timeout_s": 600, "max_fallbacks": 2},
        )
        monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
        monkeypatch.setattr(
            "router.policy.build_chain",
            lambda task, state: [
                MagicMock(tool="a", backend="a", model_profile="a"),
                MagicMock(tool="b", backend="b", model_profile="b"),
                MagicMock(tool="c", backend="c", model_profile="c"),
                MagicMock(tool="d", backend="d", model_profile="d"),
            ],
        )

        call_count = 0

        def always_fails(entry, task, trace_id=""):
            nonlocal call_count
            call_count += 1
            return ExecutorResult(
                task_id="t1", tool=entry.tool, backend=entry.backend,
                model_profile=entry.model_profile, success=False,
                normalized_error="rate_limited",
            )

        monkeypatch.setattr("router.policy._run_executor", always_fails)

        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        decision, result = route_task(task)

        assert decision.fallback_count == 2
        assert call_count == 3  # 1 initial + 2 fallbacks

    def test_no_fallback_when_success(self, monkeypatch):
        """Success on first executor should not count any fallbacks."""
        monkeypatch.setattr(
            "router.policy.get_reliability_config",
            lambda: {"chain_timeout_s": 600, "max_fallbacks": 3},
        )
        monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
        monkeypatch.setattr(
            "router.policy.build_chain",
            lambda task, state: [
                MagicMock(tool="a", backend="a", model_profile="a"),
                MagicMock(tool="b", backend="b", model_profile="b"),
            ],
        )
        monkeypatch.setattr(
            "router.policy._run_executor",
            lambda entry, task, trace_id="": ExecutorResult(
                task_id="t1", tool=entry.tool, backend=entry.backend,
                model_profile=entry.model_profile, success=True,
            ),
        )

        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        decision, result = route_task(task)

        assert decision.fallback_count == 0
        assert result.success is True

    def test_non_fallback_error_stops_immediately(self, monkeypatch):
        """Non-fallback-eligible error should stop chain immediately."""
        monkeypatch.setattr(
            "router.policy.get_reliability_config",
            lambda: {"chain_timeout_s": 600, "max_fallbacks": 3},
        )
        monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
        monkeypatch.setattr(
            "router.policy.build_chain",
            lambda task, state: [
                MagicMock(tool="a", backend="a", model_profile="a"),
                MagicMock(tool="b", backend="b", model_profile="b"),
            ],
        )
        monkeypatch.setattr(
            "router.policy._run_executor",
            lambda entry, task, trace_id="": ExecutorResult(
                task_id="t1", tool=entry.tool, backend=entry.backend,
                model_profile=entry.model_profile, success=False,
                normalized_error="toolchain_error",
            ),
        )

        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        decision, result = route_task(task)

        # toolchain_error is NOT fallback-eligible, so no fallback count
        assert decision.fallback_count == 0
        assert decision.attempted_fallback is False
        assert result.normalized_error == "toolchain_error"

    def test_fallback_eligible_error_counts(self, monkeypatch):
        """A fallback-eligible error on first executor should count as 1 fallback."""
        monkeypatch.setattr(
            "router.policy.get_reliability_config",
            lambda: {"chain_timeout_s": 600, "max_fallbacks": 3},
        )
        monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
        monkeypatch.setattr(
            "router.policy.build_chain",
            lambda task, state: [
                MagicMock(tool="a", backend="a", model_profile="a"),
                MagicMock(tool="b", backend="b", model_profile="b"),
            ],
        )
        monkeypatch.setattr(
            "router.policy._run_executor",
            lambda entry, task, trace_id="": ExecutorResult(
                task_id="t1", tool=entry.tool, backend=entry.backend,
                model_profile=entry.model_profile, success=False,
                normalized_error="auth_error",
            ),
        )

        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        decision, result = route_task(task)

        # auth_error IS fallback-eligible → fallback_count=1, attempted_fallback=True
        assert decision.fallback_count == 1
        assert decision.attempted_fallback is True


class TestReliabilityConfig:
    """Tests for get_reliability_config()."""

    def test_default_config_values(self):
        from router.config_loader import get_reliability_config
        config = get_reliability_config()
        assert "chain_timeout_s" in config
        assert "max_fallbacks" in config
        assert config["chain_timeout_s"] > 0
        assert config["max_fallbacks"] > 0
