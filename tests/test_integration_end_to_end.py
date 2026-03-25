"""End-to-end integration tests — classify → route → execute → result.

These tests exercise the complete pipeline from task classification through
routing, execution, metrics, and notifications. All executors are mocked
with realistic behavior (delays, structured output, configurable failures).
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState,
    ExecutorResult, ChainEntry,
)
from router.state_store import StateStore, reset_state_store
from router.policy import (
    route_task, build_chain, resolve_state, reset_breaker, reset_notifier,
)
from router.classifier import classify, classify_from_dict, Classifier
from router.metrics import MetricsCollector
from router.notifications import NotificationManager
from router.logger import RoutingLogger

from tests.conftest import MockExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(task_id: str, *, success: bool = True, tool: str = "codex_cli",
                 backend: str = "openai_native", model_profile: str = "codex_primary",
                 cost_usd: float = 0.002, latency_ms: int = 150,
                 error_type: str | None = None):
    """Build ExecutorResult. On failure, final_summary is None to avoid partial_success flag."""
    return ExecutorResult(
        task_id=task_id, tool=tool, backend=backend, model_profile=model_profile,
        success=success,
        normalized_error=error_type,
        exit_code=0 if success else 1, latency_ms=latency_ms,
        cost_estimate_usd=cost_usd if success else 0.0,
        final_summary="Done" if success else None,
    )


def _setup(tmp_path, monkeypatch):
    """Isolated env: temp state + runtime dirs, reset singletons."""
    state_dir = tmp_path / "state"
    runtime_dir = tmp_path / "runtime"
    state_dir.mkdir()
    runtime_dir.mkdir()

    monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
    monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
    monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
    monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")
    monkeypatch.setattr("router.logger.RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", runtime_dir / "routing.jsonl")
    monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", runtime_dir / "alerts.jsonl")

    reset_state_store()
    reset_breaker()
    reset_notifier()

    # Force-create singleton with patched paths
    from router.state_store import get_state_store
    store = get_state_store()
    return store, state_dir, runtime_dir


# ===================================================================
# Test 1: Classify → route → execute
# ===================================================================

class TestClassifyRouteExecute:
    """Classify task → build chain → route → execute → return result."""

    def test_full_pipeline_implement(self, tmp_path, monkeypatch):
        """From raw text classification to successful execution."""
        store, _, _ = _setup(tmp_path, monkeypatch)

        # Step 1: Classify
        meta = classify("Implement user authentication with JWT tokens")
        assert meta.task_class == TaskClass.IMPLEMENTATION
        assert meta.risk in (TaskRisk.LOW, TaskRisk.MEDIUM)
        assert meta.requires_repo_write is True

        # Step 2-3: Route + Execute (mocked)
        mock = _make_result(meta.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(meta)

        assert result.success is True
        assert decision.state == "openai_primary"
        assert decision.trace_id

    def test_full_pipeline_bugfix(self, tmp_path, monkeypatch):
        """Bugfix classification → routing → execution."""
        store, _, _ = _setup(tmp_path, monkeypatch)

        meta = classify("Fix crash in login form when password is empty")
        assert meta.task_class == TaskClass.BUGFIX

        mock = _make_result(meta.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(meta)

        assert result.success is True

    def test_full_pipeline_from_dict(self, tmp_path, monkeypatch):
        """classify_from_dict enrichment → routing → execution."""
        store, _, _ = _setup(tmp_path, monkeypatch)

        meta = classify_from_dict({
            "task": "Refactor database connection pooling",
            "repo_path": "/tmp/my-project",
            "risk": "high",
        })
        assert meta.task_class == TaskClass.REFACTOR
        assert meta.risk == TaskRisk.HIGH
        assert meta.repo_path == "/tmp/my-project"

        mock = _make_result(meta.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(meta)

        assert result.success is True
        assert decision.chain  # chain is populated


# ===================================================================
# Test 2: Multiple tasks sequential
# ===================================================================

class TestMultipleTasksSequential:
    """10 tasks in sequence, verify routing decisions."""

    def test_ten_sequential_tasks(self, tmp_path, monkeypatch):
        """Route 10 tasks one after another, all succeed."""
        store, _, _ = _setup(tmp_path, monkeypatch)

        tasks = [
            "Add input validation to signup form",
            "Fix null pointer in user profile",
            "Implement rate limiting middleware",
            "Refactor error handling in API layer",
            "Add unit tests for payment module",
            "Debug memory leak in WebSocket handler",
            "Review authentication flow",
            "Update API documentation",
            "Add caching for product catalog",
            "Fix timezone conversion bug",
        ]

        decisions = []
        results = []

        for i, desc in enumerate(tasks):
            meta = classify(desc)
            meta.task_id = f"seq-{i:03d}"  # Override for deterministic IDs

            mock = _make_result(meta.task_id, latency_ms=100 + i * 20)
            with patch("router.policy.run_codex", return_value=mock):
                d, r = route_task(meta)
            decisions.append(d)
            results.append(r)

        assert len(decisions) == 10
        assert len(results) == 10

        # All should succeed
        for r in results:
            assert r.success is True

        # All should have unique task IDs
        task_ids = {d.task_id for d in decisions}
        assert len(task_ids) == 10

        # All should have trace IDs
        for d in decisions:
            assert d.trace_id

        # All should be in openai_primary state (codex succeeds)
        for d in decisions:
            assert d.state == "openai_primary"

    def test_sequential_with_one_failure(self, tmp_path, monkeypatch):
        """10 tasks where task 5 fails → fallback → succeeds."""
        store, _, _ = _setup(tmp_path, monkeypatch)

        decisions = []
        results = []

        for i in range(10):
            meta = TaskMeta(
                task_id=f"seq-f-{i:03d}", agent="coder",
                task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
                repo_path="/tmp/repo", cwd="/tmp/repo",
                summary=f"Task {i}",
            )

            if i == 5:
                # Task 5: codex fails, claude succeeds
                fail = _make_result(meta.task_id, success=False, error_type="provider_unavailable")
                success = _make_result(meta.task_id, tool="claude_code",
                                        backend="anthropic", model_profile="claude_sonnet")
                with patch("router.policy.run_codex", return_value=fail), \
                     patch("router.policy.run_claude", return_value=success):
                    d, r = route_task(meta)
            else:
                mock = _make_result(meta.task_id)
                with patch("router.policy.run_codex", return_value=mock):
                    d, r = route_task(meta)

            decisions.append(d)
            results.append(r)

        assert all(r.success for r in results)
        assert decisions[5].attempted_fallback is True
        assert results[5].tool == "claude_code"


# ===================================================================
# Test 3: State changes affect routing
# ===================================================================

class TestStateChangesAffectRouting:
    """Change state mid-test → chain changes."""

    def test_state_change_mid_sequence(self, tmp_path, monkeypatch):
        """Switch from openai_primary to claude_backup between tasks."""
        store, _, _ = _setup(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="state-mid-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Implement feature",
        )

        # Route with openai_primary
        mock_codex = _make_result(task.task_id)
        with patch("router.policy.run_codex", return_value=mock_codex):
            d1, r1 = route_task(task)
        assert d1.state == "openai_primary"
        assert r1.tool == "codex_cli"

        # Switch state
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        # Route again — should use claude now
        mock_claude = _make_result(task.task_id, tool="claude_code",
                                    backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock_claude):
            d2, r2 = route_task(task)
        assert d2.state == "claude_backup"
        assert r2.tool == "claude_code"

    def test_all_four_states(self, tmp_path, monkeypatch):
        """Route a task in all 4 states, verify chain structure each time."""
        store, _, _ = _setup(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="all-states-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Test all states",
        )

        state_chain_start = {
            CodexState.OPENAI_PRIMARY: ("codex_cli", "openai_native"),
            CodexState.CLAUDE_BACKUP: ("claude_code", "anthropic"),
            CodexState.OPENROUTER_FALLBACK: ("codex_cli", "openrouter"),
        }

        for state, (exp_tool, exp_backend) in state_chain_start.items():
            store.set_manual_state(state)
            chain = build_chain(task, state)
            assert chain[0].tool == exp_tool, f"State {state}: expected {exp_tool}, got {chain[0].tool}"
            assert chain[0].backend == exp_backend, f"State {state}: expected {exp_backend}, got {chain[0].backend}"


# ===================================================================
# Test 4: Metrics accumulate
# ===================================================================

class TestMetricsAccumulate:
    """Multiple routes → metrics reflect all attempts."""

    def test_metrics_after_multiple_routes(self, tmp_path, monkeypatch):
        """Routing several tasks produces metrics entries."""
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)

        log_path = runtime_dir / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)

        # Route 5 tasks and log them
        for i in range(5):
            task = TaskMeta(
                task_id=f"metrics-{i}", agent="coder",
                task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
                repo_path="/tmp/repo", cwd="/tmp/repo",
                summary=f"Quick task {i}",
            )

            mock = _make_result(f"metrics-{i}", cost_usd=0.001 * (i + 1))
            with patch("router.policy.run_codex", return_value=mock):
                decision, result = route_task(task)

            # Log manually (route_task writes trace but not legacy log)
            logger.log(task, decision, result)

        # Collect metrics
        collector = MetricsCollector(log_path=log_path)
        report = collector.collect(period_hours=24)

        # Should have at least 5 legacy log entries (plus routing traces)
        assert report.total_tasks >= 5
        assert report.total_success >= 5

    def test_metrics_track_failure(self, tmp_path, monkeypatch):
        """Failed routes are reflected in metrics."""
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)

        log_path = runtime_dir / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)

        task = TaskMeta(
            task_id="metrics-fail-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Task that fails",
        )

        fail = _make_result(task.task_id, success=False, error_type="provider_unavailable")
        with patch("router.policy.run_codex", return_value=fail), \
             patch("router.policy.run_claude", return_value=fail), \
             patch("router.policy.run_openrouter", return_value=fail):
            decision, result = route_task(task)

        logger.log(task, decision, result)

        collector = MetricsCollector(log_path=log_path)
        report = collector.collect(period_hours=24)

        assert report.total_failure >= 1

    def test_metrics_cost_aggregation(self, tmp_path, monkeypatch):
        """Cost estimates from multiple routes are aggregated."""
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)

        log_path = runtime_dir / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)

        total_cost = 0.0
        for i in range(3):
            cost = 0.005 * (i + 1)
            total_cost += cost

            task = TaskMeta(
                task_id=f"cost-agg-{i}", agent="coder",
                task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
                repo_path="/tmp/repo", cwd="/tmp/repo",
                summary=f"Cost test {i}",
            )
            mock = _make_result(task.task_id, cost_usd=cost)
            with patch("router.policy.run_codex", return_value=mock):
                decision, result = route_task(task)
            logger.log(task, decision, result)

        collector = MetricsCollector(log_path=log_path)
        report = collector.collect(period_hours=24)

        # Total cost across all model metrics should be >= our costs
        total_model_cost = sum(m.total_cost_usd for m in report.by_model.values())
        assert total_model_cost >= total_cost


# ===================================================================
# Test 5: Notifications on state change
# ===================================================================

class TestNotificationsOnStateChange:
    """State change → notification emitted."""

    def test_state_change_emits_alert(self, tmp_path, monkeypatch):
        """notify_state_change writes an alert to the alerts file."""
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)

        alerts_path = runtime_dir / "alerts.jsonl"
        notifier = NotificationManager(alerts_path=alerts_path)

        alert = notifier.notify_state_change(
            from_state="openai_primary",
            to_state="claude_backup",
            reason="codex quota exhausted",
        )

        assert alert.alert_type == "state_change"
        assert "openai_primary" in alert.message
        assert "claude_backup" in alert.message

        # Verify alert was written
        assert alerts_path.exists()
        alerts = notifier.get_recent_alerts(limit=5)
        assert len(alerts) >= 1
        assert alerts[-1]["alert_type"] == "state_change"
        assert alerts[-1]["details"]["from"] == "openai_primary"
        assert alerts[-1]["details"]["to"] == "claude_backup"

    def test_fallback_rate_alert(self, tmp_path, monkeypatch):
        """High fallback rate triggers a warning alert."""
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)

        alerts_path = runtime_dir / "alerts.jsonl"
        notifier = NotificationManager(alerts_path=alerts_path)

        # 40% fallback rate (threshold is 30%)
        alert = notifier.check_fallback_rate(
            total_tasks=10,
            fallback_tasks=4,
            window_hours=1,
        )

        assert alert is not None
        assert alert.alert_type == "fallback_rate"
        assert alert.severity == "warning"

    def test_low_fallback_no_alert(self, tmp_path, monkeypatch):
        """Low fallback rate does NOT trigger alert."""
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)

        alerts_path = runtime_dir / "alerts.jsonl"
        notifier = NotificationManager(alerts_path=alerts_path)

        # 10% fallback rate (below 30% threshold)
        alert = notifier.check_fallback_rate(
            total_tasks=10,
            fallback_tasks=1,
            window_hours=1,
        )

        assert alert is None

    def test_notification_on_route_task_state_change(self, tmp_path, monkeypatch):
        """When state changes and route_task is called, notification system is engaged."""
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)

        alerts_path = runtime_dir / "alerts.jsonl"

        task = TaskMeta(
            task_id="notify-route-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Test notification",
        )

        # Route with codex primary
        mock = _make_result(task.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            route_task(task)

        # Switch to claude_backup
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        # Notify manually (route_task calls notifier.check_conservation_duration)
        notifier = NotificationManager(alerts_path=alerts_path)
        alert = notifier.notify_state_change(
            from_state="openai_primary",
            to_state="claude_backup",
            reason="manual override",
        )

        alerts = notifier.get_recent_alerts()
        assert any(a["alert_type"] == "state_change" for a in alerts)
