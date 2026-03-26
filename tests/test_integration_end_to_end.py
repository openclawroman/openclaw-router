"""End-to-end integration tests — classify → route → execute → result."""

from unittest.mock import patch

import pytest

from router.models import TaskMeta, TaskClass, TaskRisk, CodexState
from router.policy import route_task, build_chain, reset_breaker, reset_notifier
from router.state_store import reset_state_store
from router.classifier import classify, classify_from_dict
from router.metrics import MetricsCollector
from router.notifications import NotificationManager
from router.logger import RoutingLogger
from tests.conftest import make_result, make_task


def _setup(tmp_path, monkeypatch):
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
    from router.state_store import get_state_store
    return get_state_store(), state_dir, runtime_dir


class TestClassifyRouteExecute:
    def test_full_pipeline_implement(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        meta = classify("Implement user authentication with JWT tokens")
        assert meta.task_class == TaskClass.IMPLEMENTATION
        assert meta.risk in (TaskRisk.LOW, TaskRisk.MEDIUM)

        mock = make_result(meta.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(meta)
        assert result.success is True
        assert decision.state == "openai_primary"

    def test_full_pipeline_bugfix(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        meta = classify("Fix crash in login form when password is empty")
        assert meta.task_class == TaskClass.BUGFIX

        mock = make_result(meta.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            _, result = route_task(meta)
        assert result.success is True

    def test_full_pipeline_from_dict(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        meta = classify_from_dict({
            "task": "Refactor database connection pooling",
            "repo_path": "/tmp/my-project",
            "risk": "high",
        })
        assert meta.task_class == TaskClass.REFACTOR
        assert meta.risk == TaskRisk.HIGH

        mock = make_result(meta.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(meta)
        assert result.success is True
        assert decision.chain


class TestMultipleTasksSequential:
    def test_ten_sequential_tasks(self, tmp_path, monkeypatch):
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

        decisions, results = [], []
        for i, desc in enumerate(tasks):
            meta = classify(desc)
            meta.task_id = f"seq-{i:03d}"
            mock = make_result(meta.task_id, latency_ms=100 + i * 20)
            with patch("router.policy.run_codex", return_value=mock):
                d, r = route_task(meta)
            decisions.append(d)
            results.append(r)

        assert all(r.success for r in results)
        assert len({d.task_id for d in decisions}) == 10
        assert all(d.trace_id for d in decisions)
        assert all(d.state == "openai_primary" for d in decisions)

    def test_sequential_with_one_failure(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        decisions, results = [], []

        for i in range(10):
            task = make_task(f"seq-f-{i:03d}", summary=f"Task {i}")
            if i == 5:
                fail = make_result(task.task_id, success=False, error_type="provider_unavailable")
                success = make_result(task.task_id, tool="claude_code",
                                      backend="anthropic", model_profile="claude_sonnet")
                with patch("router.policy.run_codex", return_value=fail), \
                     patch("router.policy.run_claude", return_value=success):
                    d, r = route_task(task)
            else:
                mock = make_result(task.task_id)
                with patch("router.policy.run_codex", return_value=mock):
                    d, r = route_task(task)
            decisions.append(d)
            results.append(r)

        assert all(r.success for r in results)
        assert decisions[5].attempted_fallback is True
        assert results[5].tool == "claude_code"


class TestStateChangesAffectRouting:
    def test_state_change_mid_sequence(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        task = make_task("state-mid-001", summary="Implement feature")

        mock_codex = make_result(task.task_id)
        with patch("router.policy.run_codex", return_value=mock_codex):
            d1, r1 = route_task(task)
        assert d1.state == "openai_primary"
        assert r1.tool == "codex_cli"

        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        mock_claude = make_result(task.task_id, tool="claude_code",
                                  backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock_claude):
            d2, r2 = route_task(task)
        assert d2.state == "claude_backup"
        assert r2.tool == "claude_code"

    def test_all_four_states(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        task = make_task("all-states-001", summary="Test all states")

        state_chain_start = {
            CodexState.OPENAI_PRIMARY: ("codex_cli", "openai_native"),
            CodexState.CLAUDE_BACKUP: ("claude_code", "anthropic"),
            CodexState.OPENROUTER_FALLBACK: ("codex_cli", "openrouter"),
        }
        for state, (exp_tool, exp_backend) in state_chain_start.items():
            store.set_manual_state(state)
            chain = build_chain(task, state)
            assert chain[0].tool == exp_tool
            assert chain[0].backend == exp_backend


class TestMetricsAccumulate:
    def test_metrics_after_multiple_routes(self, tmp_path, monkeypatch):
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)
        log_path = runtime_dir / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)

        for i in range(5):
            task = make_task(f"metrics-{i}", risk=TaskRisk.LOW, summary=f"Quick task {i}")
            mock = make_result(f"metrics-{i}", cost_usd=0.001 * (i + 1))
            with patch("router.policy.run_codex", return_value=mock):
                decision, result = route_task(task)
            logger.log(task, decision, result)

        collector = MetricsCollector(log_path=log_path)
        report = collector.collect(period_hours=24)
        assert report.total_tasks >= 5
        assert report.total_success >= 5

    def test_metrics_track_failure(self, tmp_path, monkeypatch):
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)
        log_path = runtime_dir / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)

        task = make_task("metrics-fail-001", summary="Task that fails")
        fail = make_result(task.task_id, success=False, error_type="provider_unavailable")
        with patch("router.policy.run_codex", return_value=fail), \
             patch("router.policy.run_claude", return_value=fail), \
             patch("router.policy.run_openrouter", return_value=fail):
            decision, result = route_task(task)
        logger.log(task, decision, result)

        collector = MetricsCollector(log_path=log_path)
        report = collector.collect(period_hours=24)
        assert report.total_failure >= 1

    def test_metrics_cost_aggregation(self, tmp_path, monkeypatch):
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)
        log_path = runtime_dir / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)

        total_cost = 0.0
        for i in range(3):
            cost = 0.005 * (i + 1)
            total_cost += cost
            task = make_task(f"cost-agg-{i}", risk=TaskRisk.LOW, summary=f"Cost test {i}")
            mock = make_result(task.task_id, cost_usd=cost)
            with patch("router.policy.run_codex", return_value=mock):
                decision, result = route_task(task)
            logger.log(task, decision, result)

        collector = MetricsCollector(log_path=log_path)
        report = collector.collect(period_hours=24)
        total_model_cost = sum(m.total_cost_usd for m in report.by_model.values())
        assert total_model_cost >= total_cost


class TestNotificationsOnStateChange:
    def test_state_change_emits_alert(self, tmp_path, monkeypatch):
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)
        notifier = NotificationManager(alerts_path=runtime_dir / "alerts.jsonl")

        alert = notifier.notify_state_change(
            from_state="openai_primary", to_state="claude_backup",
            reason="codex quota exhausted",
        )
        assert alert.alert_type == "state_change"
        assert "openai_primary" in alert.message

        alerts = notifier.get_recent_alerts(limit=5)
        assert len(alerts) >= 1
        assert alerts[-1]["details"]["from"] == "openai_primary"

    def test_fallback_rate_alert(self, tmp_path, monkeypatch):
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)
        notifier = NotificationManager(alerts_path=runtime_dir / "alerts.jsonl")

        alert = notifier.check_fallback_rate(total_tasks=10, fallback_tasks=4, window_hours=1)
        assert alert is not None
        assert alert.alert_type == "fallback_rate"
        assert alert.severity == "warning"

    def test_low_fallback_no_alert(self, tmp_path, monkeypatch):
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)
        notifier = NotificationManager(alerts_path=runtime_dir / "alerts.jsonl")

        alert = notifier.check_fallback_rate(total_tasks=10, fallback_tasks=1, window_hours=1)
        assert alert is None

    def test_notification_on_route_task_state_change(self, tmp_path, monkeypatch):
        store, _, runtime_dir = _setup(tmp_path, monkeypatch)
        task = make_task("notify-route-001", summary="Test notification")

        mock = make_result(task.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            route_task(task)

        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        notifier = NotificationManager(alerts_path=runtime_dir / "alerts.jsonl")
        notifier.notify_state_change(from_state="openai_primary",
                                     to_state="claude_backup", reason="manual override")
        alerts = notifier.get_recent_alerts()
        assert any(a["alert_type"] == "state_change" for a in alerts)


class TestEndToEndErrorPropagation:
    def test_error_history_across_full_chain(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        meta = classify("Implement distributed consensus protocol")
        meta.task_id = "err-prop-001"

        codex_fail = make_result("err-prop-001", success=False, error_type="provider_unavailable")
        claude_fail = make_result("err-prop-001", success=False, error_type="rate_limited",
                                  tool="claude_code")
        openrouter_fail = make_result("err-prop-001", success=False, error_type="quota_exhausted",
                                      tool="codex_cli", backend="openrouter",
                                      model_profile="openrouter_minimax")
        with patch("router.policy.run_codex", return_value=codex_fail), \
             patch("router.policy.run_claude", return_value=claude_fail), \
             patch("router.policy.run_openrouter", return_value=openrouter_fail):
            decision, result = route_task(meta)

        assert result.success is False
        assert len(decision.error_history) >= 2
        assert "provider_unavailable" in [e["error_type"] for e in decision.error_history]

    def test_error_chain_with_successful_fallback(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        meta = classify("Refactor microservice architecture")
        meta.task_id = "err-chain-001"

        codex_fail = make_result("err-chain-001", success=False, error_type="provider_unavailable")
        claude_fail = make_result("err-chain-001", success=False, error_type="rate_limited",
                                  tool="claude_code")
        openrouter_ok = make_result("err-chain-001", success=True,
                                    tool="codex_cli", backend="openrouter",
                                    model_profile="openrouter_minimax", cost_usd=0.0055)
        with patch("router.policy.run_codex", return_value=codex_fail), \
             patch("router.policy.run_claude", return_value=claude_fail), \
             patch("router.policy.run_openrouter", return_value=openrouter_ok):
            decision, result = route_task(meta)

        assert result.success is True
        assert result.cost_estimate_usd == 0.0055
        assert len(decision.error_history) == 2
        assert decision.attempted_fallback is True
