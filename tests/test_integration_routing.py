"""Integration tests for the full routing path with simulated executors.

These tests exercise the complete flow: task → state resolution → chain building
→ executor dispatch → fallback → result. All executors are mocked with realistic
delays, structured output, and configurable failure modes.
"""

import json
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    RouteDecision, ExecutorResult, ChainEntry, CodexState,
)
from router.state_store import StateStore, reset_state_store
from router.policy import (
    route_task, build_chain, resolve_state, reset_breaker, reset_notifier,
)
from router.errors import (
    CodexQuotaError, ProviderTimeoutError, ExecutorError, can_fallback,
)

from tests.conftest import MockExecutor, MockExecutorChain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_result(task_id: str, *, success: bool = True, tool: str = "codex_cli",
                      backend: str = "openai_native", model_profile: str = "codex_primary",
                      cost_usd: float = 0.0023, latency_ms: int = 150,
                      error_type: str | None = None,
                      summary: str = "Task completed: 2 files changed, 85 insertions(+)",
                      artifacts: list[str] | None = None):
    """Build a realistic ExecutorResult.

    Note: on failure, final_summary is None so _run_executor doesn't
    mark it as partial_success (which would prevent fallback).
    """
    return ExecutorResult(
        task_id=task_id,
        tool=tool,
        backend=backend,
        model_profile=model_profile,
        success=success,
        normalized_error=error_type,
        exit_code=0 if success else 1,
        latency_ms=latency_ms,
        cost_estimate_usd=cost_usd if success else 0.0,
        artifacts=artifacts if success else (artifacts or []),
        final_summary=summary if success else None,
    )


def _patch_state_store(monkeypatch, state_dir):
    """Point the module-level state store paths to temp directory.

    After calling this, get_state_store() will return a fresh singleton
    backed by temp files. route_task/resolve_state will use it.
    """
    monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
    monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
    monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
    monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")
    reset_state_store()
    reset_breaker()
    reset_notifier()
    # Force-create the singleton so route_task uses it
    from router.state_store import get_state_store
    return get_state_store()


# ===================================================================
# Test 1: Full routing happy path
# ===================================================================

class TestFullRoutingHappyPath:
    """Task → route_task → success with mocked executor."""

    def test_single_task_succeeds(self, tmp_path, monkeypatch):
        """Basic happy path: codex_cli succeeds on first try."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="happy-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Add user registration endpoint",
        )

        mock_result = _make_mock_result("happy-001")
        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(task)

        assert decision.task_id == "happy-001"
        assert decision.state == "openai_primary"
        assert result.success is True
        assert result.cost_estimate_usd == 0.0023
        assert "codex" in result.tool or result.tool == "codex_cli"
        assert result.final_summary is not None
        assert len(decision.chain) >= 1

    def test_result_has_structured_output(self, tmp_path, monkeypatch):
        """Executor returns structured output — verify it propagates."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="struct-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Fix typo in README",
        )

        mock_result = _make_mock_result(
            "struct-001",
            summary="Fixed: 1 file changed, 1 insertion(+), 1 deletion(-)",
            cost_usd=0.0005,
            artifacts=["/tmp/struct-001.stdout.txt"],
        )
        with patch("router.policy.run_codex", return_value=mock_result):
            _, result = route_task(task)

        assert result.success is True
        assert result.cost_estimate_usd == 0.0005
        assert "Fixed" in result.final_summary
        assert result.latency_ms > 0


# ===================================================================
# Test 2: Routing with fallback
# ===================================================================

class TestRoutingWithFallback:
    """Primary fails → fallback succeeds."""

    def test_primary_fails_fallback_succeeds(self, tmp_path, monkeypatch):
        """Codex fails with provider_unavailable → Claude succeeds."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="fb-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Implement search functionality",
        )

        failed_result = _make_mock_result(
            "fb-001", success=False, error_type="provider_unavailable",
            tool="codex_cli", backend="openai_native", model_profile="codex_primary",
            summary="Error: provider_unavailable",
        )
        success_result = _make_mock_result(
            "fb-001", success=True, tool="claude_code",
            backend="anthropic", model_profile="claude_sonnet",
            summary="Search implemented with 3 files changed",
        )

        call_sequence = [failed_result, success_result]
        call_idx = [0]

        def mock_codex(meta, model=None, **kw):
            return call_sequence[0]

        def mock_claude(meta, model=None, **kw):
            return call_sequence[1]

        with patch("router.policy.run_codex", side_effect=mock_codex), \
             patch("router.policy.run_claude", side_effect=mock_claude):
            decision, result = route_task(task)

        assert result.success is True
        assert result.tool == "claude_code"
        assert decision.attempted_fallback is True
        assert decision.fallback_from == "codex_cli"
        assert len(decision.error_history) >= 1
        assert decision.error_history[0]["error_type"] == "provider_unavailable"

    def test_auth_error_triggers_fallback(self, tmp_path, monkeypatch):
        """Auth error → fallback to next provider."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="auth-fb-001", agent="coder",
            task_class=TaskClass.BUGFIX, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Fix auth token refresh bug",
        )

        failed = _make_mock_result("auth-fb-001", success=False, error_type="auth_error")
        success = _make_mock_result("auth-fb-001", success=True, tool="claude_code",
                                     backend="anthropic", model_profile="claude_sonnet")

        with patch("router.policy.run_codex", return_value=failed), \
             patch("router.policy.run_claude", return_value=success):
            decision, result = route_task(task)

        assert result.success is True
        assert decision.attempted_fallback is True


# ===================================================================
# Test 3: Chain exhaustion — all fail
# ===================================================================

class TestRoutingChainExhaustion:
    """All providers fail → returns last error."""

    def test_all_providers_fail(self, tmp_path, monkeypatch):
        """Every executor in the chain fails → final result is failure."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="exhaust-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Add caching layer",
        )

        fail_result = _make_mock_result(
            "exhaust-001", success=False, error_type="provider_unavailable",
            summary="Error: provider unavailable",
        )

        with patch("router.policy.run_codex", return_value=fail_result), \
             patch("router.policy.run_claude", return_value=fail_result), \
             patch("router.policy.run_openrouter", return_value=fail_result):
            decision, result = route_task(task)

        assert result.success is False
        assert result.normalized_error is not None
        # Chain should have been fully traversed
        assert len(decision.error_history) > 0

    def test_exhaustion_preserves_last_error(self, tmp_path, monkeypatch):
        """After full chain failure, last error is available in result."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="exhaust-err-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Implement GraphQL endpoint",
        )

        fail_quota = _make_mock_result("exhaust-err-001", success=False, error_type="quota_exhausted")
        fail_unavail = _make_mock_result("exhaust-err-001", success=False, error_type="provider_unavailable")

        with patch("router.policy.run_codex", return_value=fail_quota), \
             patch("router.policy.run_claude", return_value=fail_unavail), \
             patch("router.policy.run_openrouter", return_value=fail_unavail):
            decision, result = route_task(task)

        assert result.success is False
        assert result.normalized_error in ("quota_exhausted", "provider_unavailable")


# ===================================================================
# Test 4: Routing with state override
# ===================================================================

class TestRoutingWithStateOverride:
    """Manual state changes routing chain."""

    def test_manual_claude_backup_chain(self, tmp_path, monkeypatch):
        """Setting manual state to claude_backup → Claude first in chain."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        store = _patch_state_store(monkeypatch, state_dir)

        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        task = TaskMeta(
            task_id="manual-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Add rate limiting middleware",
        )

        mock_result = _make_mock_result("manual-001", tool="claude_code",
                                         backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock_result):
            decision, result = route_task(task)

        assert decision.state == "claude_backup"
        assert result.success is True
        # Chain should start with claude_code
        first = decision.chain[0]
        assert first.tool == "claude_code"
        assert first.backend == "anthropic"

    def test_manual_openrouter_fallback_chain(self, tmp_path, monkeypatch):
        """openrouter_fallback state → only openrouter entries in chain."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        store = _patch_state_store(monkeypatch, state_dir)

        store.set_manual_state(CodexState.OPENROUTER_FALLBACK)

        task = TaskMeta(
            task_id="manual-or-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Update deployment scripts",
        )

        mock_result = _make_mock_result("manual-or-001", tool="codex_cli",
                                         backend="openrouter", model_profile="openrouter_minimax")
        with patch("router.policy.run_openrouter", return_value=mock_result):
            decision, result = route_task(task)

        assert decision.state == "openrouter_fallback"
        assert result.success is True
        for entry in decision.chain:
            assert entry.backend == "openrouter"


# ===================================================================
# Test 5: Routing respects timeouts
# ===================================================================

class TestRoutingRespectsTimeouts:
    """Mock executor that takes too long → timeout error."""

    def test_executor_timeout_raises(self, tmp_path, monkeypatch):
        """Executor that sleeps too long triggers timeout error."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="timeout-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Complex data migration",
        )

        def slow_executor(meta, **kw):
            raise ProviderTimeoutError("Execution timed out")

        success = _make_mock_result("timeout-001", tool="claude_code",
                                     backend="anthropic", model_profile="claude_sonnet")

        with patch("router.policy.run_codex", side_effect=slow_executor), \
             patch("router.policy.run_claude", return_value=success):
            decision, result = route_task(task)

        # Should have fallen back to Claude after codex timeout
        assert result.success is True
        assert result.tool == "claude_code"
        assert decision.attempted_fallback is True

    def test_timeout_is_fallback_eligible(self):
        """Timeout errors should be eligible for fallback."""
        assert can_fallback("provider_timeout") is True


# ===================================================================
# Test 6: Concurrent tasks
# ===================================================================

class TestRoutingConcurrentTasks:
    """Multiple tasks routed in parallel → all succeed."""

    def test_five_tasks_in_parallel(self, tmp_path, monkeypatch):
        """5 tasks routed concurrently → all succeed, no race conditions."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        num_tasks = 5
        results = [None] * num_tasks
        errors = [None] * num_tasks

        def make_result(meta, model=None, **kw):
            time.sleep(0.05 + (hash(meta.task_id) % 10) * 0.01)  # Realistic 50-150ms
            return _make_mock_result(meta.task_id, latency_ms=80, cost_usd=0.001)

        def worker(idx):
            try:
                task = TaskMeta(
                    task_id=f"concurrent-{idx}", agent="coder",
                    task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
                    repo_path="/tmp/repo", cwd="/tmp/repo",
                    summary=f"Task {idx}: add logging",
                )
                with patch("router.policy.run_codex", side_effect=make_result):
                    decision, result = route_task(task)
                results[idx] = (decision, result)
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_tasks)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(e is None for e in errors), f"Errors: {errors}"
        assert all(r is not None for r in results)
        for decision, result in results:
            assert result.success is True
            assert decision.task_id.startswith("concurrent-")


# ===================================================================
# Test 7: Trace preservation
# ===================================================================

class TestRoutingPreservesTrace:
    """RouteDecision has trace_id, all fields populated."""

    def test_trace_id_populated(self, tmp_path, monkeypatch):
        """Every route_task produces a unique trace_id."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="trace-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Add health check endpoint",
        )

        mock_result = _make_mock_result("trace-001")
        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(task)

        assert decision.trace_id != ""
        assert len(decision.trace_id) == 12  # 12-char hex
        assert decision.task_id == "trace-001"
        assert decision.state in {s.value for s in CodexState}
        assert len(decision.chain) >= 1
        assert decision.reason != ""

    def test_trace_ids_are_unique(self, tmp_path, monkeypatch):
        """Two different tasks get different trace_ids."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        trace_ids = set()
        for i in range(5):
            task = TaskMeta(
                task_id=f"trace-uniq-{i}", agent="coder",
                task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
                repo_path="/tmp/repo", cwd="/tmp/repo",
                summary=f"Quick fix {i}",
            )
            mock_result = _make_mock_result(f"trace-uniq-{i}")
            with patch("router.policy.run_codex", return_value=mock_result):
                decision, _ = route_task(task)
            trace_ids.add(decision.trace_id)

        assert len(trace_ids) == 5

    def test_decision_fields_complete(self, tmp_path, monkeypatch):
        """RouteDecision has all expected fields populated."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="fields-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Add input validation",
        )

        mock_result = _make_mock_result("fields-001")
        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(task)

        # All RouteDecision fields
        assert decision.task_id == "fields-001"
        assert decision.state  # non-empty string
        assert isinstance(decision.chain, list)
        assert decision.reason  # non-empty string
        assert isinstance(decision.attempted_fallback, bool)
        assert isinstance(decision.providers_skipped, list)
        assert isinstance(decision.chain_timed_out, bool)
        assert isinstance(decision.fallback_count, int)
        assert decision.trace_id
        assert isinstance(decision.error_history, list)

        # All ExecutorResult fields
        assert result.task_id == "fields-001"
        assert result.tool
        assert result.backend
        assert result.model_profile
        assert isinstance(result.success, bool)
        assert result.latency_ms > 0
        assert result.cost_estimate_usd is not None


# ===================================================================
# Test 8: Cost tracking
# ===================================================================

class TestRoutingCostTracking:
    """Executor returns cost → cost is in result."""

    def test_cost_propagates_from_executor(self, tmp_path, monkeypatch):
        """Cost returned by executor is preserved in the result."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="cost-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Implement data pipeline",
        )

        mock_result = _make_mock_result("cost-001", cost_usd=0.0187)
        with patch("router.policy.run_codex", return_value=mock_result):
            _, result = route_task(task)

        assert result.cost_estimate_usd == 0.0187

    def test_zero_cost_on_failure(self, tmp_path, monkeypatch):
        """Failed executor returns zero cost (or no cost)."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="cost-fail-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Expensive computation",
        )

        fail = _make_mock_result("cost-fail-001", success=False, error_type="quota_exhausted")
        with patch("router.policy.run_codex", return_value=fail), \
             patch("router.policy.run_claude", return_value=fail), \
             patch("router.policy.run_openrouter", return_value=fail):
            _, result = route_task(task)

        assert result.success is False
        # Cost should be 0 or None on failure
        assert result.cost_estimate_usd in (0.0, None, 0)

    def test_cost_accumulates_across_fallback(self, tmp_path, monkeypatch):
        """When fallback occurs, costs from failed attempts are tracked."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="cost-fb-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Implement caching",
        )

        failed = _make_mock_result("cost-fb-001", success=False, error_type="provider_unavailable")
        success = _make_mock_result("cost-fb-001", success=True, cost_usd=0.0045,
                                     tool="claude_code", backend="anthropic", model_profile="claude_sonnet")

        with patch("router.policy.run_codex", return_value=failed), \
             patch("router.policy.run_claude", return_value=success):
            _, result = route_task(task)

        assert result.success is True
        assert result.cost_estimate_usd == 0.0045


# ===================================================================
# EDGE CASE: Primary fails, fallback also fails
# ===================================================================

class TestFallbackBothFail:
    """Primary fails → fallback triggers → fallback also fails → final error."""

    def test_primary_fails_fallback_also_fails(self, tmp_path, monkeypatch):
        """Codex fails → claude fallback also fails → result is failure with fallback metadata."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="both-fail-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Complex deployment automation",
        )

        codex_fail = _make_mock_result(
            "both-fail-001", success=False, error_type="provider_unavailable",
            tool="codex_cli", backend="openai_native", model_profile="codex_primary",
        )
        claude_fail = _make_mock_result(
            "both-fail-001", success=False, error_type="quota_exhausted",
            tool="claude_code", backend="anthropic", model_profile="claude_sonnet",
        )

        with patch("router.policy.run_codex", return_value=codex_fail), \
             patch("router.policy.run_claude", return_value=claude_fail), \
             patch("router.policy.run_openrouter", return_value=codex_fail):
            decision, result = route_task(task)

        assert result.success is False
        assert decision.attempted_fallback is True
        assert len(decision.error_history) >= 2
        error_types = {e["error_type"] for e in decision.error_history}
        assert "provider_unavailable" in error_types
        assert "quota_exhausted" in error_types
        assert decision.fallback_count >= 1


# ===================================================================
# EDGE CASE: Rate limit error triggers fallback
# ===================================================================

class TestRoutingWithRateLimit:
    """Rate limit (429) → fallback eligible → fallback succeeds."""

    def test_rate_limit_triggers_fallback(self, tmp_path, monkeypatch):
        """Executor returns rate_limited → falls back to next provider."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="rl-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Implement webhook handler",
        )

        rate_limited = _make_mock_result(
            "rl-001", success=False, error_type="rate_limited",
            tool="codex_cli", backend="openai_native", model_profile="codex_primary",
        )
        success = _make_mock_result(
            "rl-001", success=True, tool="claude_code",
            backend="anthropic", model_profile="claude_sonnet",
        )

        with patch("router.policy.run_codex", return_value=rate_limited), \
             patch("router.policy.run_claude", return_value=success):
            decision, result = route_task(task)

        assert result.success is True
        assert result.tool == "claude_code"
        assert decision.attempted_fallback is True
        assert any(
            e["error_type"] == "rate_limited" for e in decision.error_history
        )

    def test_rate_limit_is_fallback_eligible(self):
        """Verify rate_limited is in the fallback-eligible set."""
        from router.errors import can_fallback
        assert can_fallback("rate_limited") is True


# ===================================================================
# EDGE CASE: Partial success prevents fallback
# ===================================================================

class TestPartialSuccessPreventsFallback:
    """Executor returns partial_success=True → no fallback, result returned as-is."""

    def test_partial_success_stops_fallback(self, tmp_path, monkeypatch):
        """Partial success should NOT trigger fallback chain."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="partial-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Refactor database layer",
        )

        partial = ExecutorResult(
            task_id="partial-001",
            tool="codex_cli", backend="openai_native", model_profile="codex_primary",
            success=False,
            partial_success=True,
            normalized_error="provider_unavailable",
            exit_code=1, latency_ms=200,
            cost_estimate_usd=0.001,
            final_summary="Completed 2/3 subtasks",
        )

        claude_mock = _make_mock_result("partial-001", success=True, tool="claude_code",
                                          backend="anthropic", model_profile="claude_sonnet")

        with patch("router.policy.run_codex", return_value=partial), \
             patch("router.policy.run_claude", return_value=claude_mock):
            decision, result = route_task(task)

        assert result.partial_success is True
        assert result.tool == "codex_cli"
        assert decision.attempted_fallback is False


# ===================================================================
# EDGE CASE: Cost preserved across fallbacks
# ===================================================================

class TestCostPreservedAcrossFallbacks:
    """Failed attempts have zero cost; final result has real cost."""

    def test_cost_from_winning_executor(self, tmp_path, monkeypatch):
        """After fallback, cost comes from the successful executor."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="cost-pres-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Implement caching layer",
        )

        failed = _make_mock_result("cost-pres-001", success=False, error_type="provider_unavailable")
        succeeded = _make_mock_result("cost-pres-001", success=True, cost_usd=0.0067,
                                       tool="claude_code", backend="anthropic", model_profile="claude_sonnet")

        with patch("router.policy.run_codex", return_value=failed), \
             patch("router.policy.run_claude", return_value=succeeded):
            _, result = route_task(task)

        assert result.success is True
        assert result.cost_estimate_usd == 0.0067


# ===================================================================
# EDGE CASE: Error history accumulates across fallback
# ===================================================================

class TestErrorHistoryAccumulatesAcrossFallback:
    """Each failed attempt adds to error_history in RouteDecision."""

    def test_error_history_accumulates(self, tmp_path, monkeypatch):
        """Three providers fail, then one succeeds → error_history has all failures."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "runtime" / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", tmp_path / "runtime" / "alerts.jsonl")
        _patch_state_store(monkeypatch, state_dir)

        task = TaskMeta(
            task_id="err-hist-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Deploy microservice",
        )

        fail_quota = _make_mock_result("err-hist-001", success=False, error_type="quota_exhausted",
                                        tool="codex_cli")
        fail_rate = _make_mock_result("err-hist-001", success=False, error_type="rate_limited",
                                       tool="claude_code")
        success = _make_mock_result("err-hist-001", success=True,
                                     tool="codex_cli", backend="openrouter", model_profile="openrouter_minimax")

        with patch("router.policy.run_codex", return_value=fail_quota), \
             patch("router.policy.run_claude", return_value=fail_rate), \
             patch("router.policy.run_openrouter", return_value=success):
            decision, result = route_task(task)

        assert result.success is True
        assert len(decision.error_history) >= 2
        error_types = [e["error_type"] for e in decision.error_history]
        assert "quota_exhausted" in error_types
        assert "rate_limited" in error_types
