"""Tests for health check and graceful shutdown."""

import json
import time
import pytest
from router.health import (
    ShutdownManager,
    TaskHandle,
    health_check,
    get_shutdown_manager,
    install_signal_handlers,
)


class TestShutdownManager:
    def test_initial_state(self):
        mgr = ShutdownManager()
        assert mgr.should_accept_new_tasks()
        assert mgr.is_drained()
        assert mgr.get_in_flight() == []

    def test_register_unregister(self):
        mgr = ShutdownManager()
        mgr.register_task("t1", "openai_primary", "codex_cli")
        assert not mgr.is_drained()
        assert len(mgr.get_in_flight()) == 1
        mgr.unregister_task("t1")
        assert mgr.is_drained()

    def test_request_shutdown(self):
        mgr = ShutdownManager()
        mgr.request_shutdown()
        assert not mgr.should_accept_new_tasks()

    def test_drain_timeout(self):
        mgr = ShutdownManager()
        mgr._drain_timeout_s = 0.01
        mgr.register_task("t1", "openai_primary", "codex_cli")
        mgr.request_shutdown()
        time.sleep(0.02)
        assert mgr.drain_timed_out()

    def test_multiple_tasks(self):
        mgr = ShutdownManager()
        mgr.register_task("t1", "openai_primary", "codex_cli")
        mgr.register_task("t2", "claude_backup", "claude_code")
        assert len(mgr.get_in_flight()) == 2
        mgr.unregister_task("t1")
        assert len(mgr.get_in_flight()) == 1
        mgr.unregister_task("t2")
        assert mgr.is_drained()

    def test_unregister_nonexistent(self):
        """Unregistering a non-existent task should not raise."""
        mgr = ShutdownManager()
        mgr.unregister_task("nonexistent")  # should not raise

    def test_get_status(self):
        mgr = ShutdownManager()
        mgr.register_task("t1", "openai_primary", "codex_cli")
        status = mgr.get_status()
        assert status["in_flight_count"] == 1
        assert "t1" in status["in_flight_tasks"]
        assert not status["shutdown_requested"]
        assert not status["is_drained"]


class TestHealthCheck:
    def test_health_check_returns_dict(self, monkeypatch):
        """Health check should return a dict with expected keys."""
        monkeypatch.setattr("router.policy.resolve_state", lambda: type("S", (), {"value": "openai_primary"})())
        monkeypatch.setattr("router.config_loader.get_reliability_config", lambda: {"chain_timeout_s": 600})

        report = health_check()
        assert isinstance(report, dict)
        assert "status" in report
        assert "state" in report
        assert "providers" in report
        assert "reliability" in report
        assert "shutdown" in report
        assert "timestamp" in report

    def test_health_check_status_healthy(self, monkeypatch):
        monkeypatch.setattr("router.policy.resolve_state", lambda: type("S", (), {"value": "openai_primary"})())
        monkeypatch.setattr("router.config_loader.get_reliability_config", lambda: {"chain_timeout_s": 600})

        report = health_check()
        assert report["status"] == "healthy"

    def test_health_check_status_degraded(self, monkeypatch):
        monkeypatch.setattr("router.policy.resolve_state", lambda: type("S", (), {"value": "openai_primary"})())
        breaker = type("B", (), {"get_stats": lambda self: {
            "codex_cli:openai_native": {"state": "open", "failure_count": 5}
        }})()
        monkeypatch.setattr("router.policy.get_breaker", lambda: breaker)
        monkeypatch.setattr("router.config_loader.get_reliability_config", lambda: {"chain_timeout_s": 600})

        report = health_check()
        assert report["status"] == "degraded"

    def test_health_check_status_draining(self, monkeypatch):
        monkeypatch.setattr("router.policy.resolve_state", lambda: type("S", (), {"value": "openai_primary"})())
        monkeypatch.setattr("router.config_loader.get_reliability_config", lambda: {"chain_timeout_s": 600})

        mgr = get_shutdown_manager()
        try:
            mgr.request_shutdown()
            report = health_check()
            assert report["status"] == "draining"
        finally:
            # Reset
            mgr._shutdown_requested = False
            mgr._shutdown_time = None


class TestGracefulShutdown:
    def test_rejects_new_tasks_when_shutting_down(self, monkeypatch):
        """route_task should reject new tasks during shutdown."""
        from router.policy import route_task
        from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality

        mgr = get_shutdown_manager()
        try:
            mgr.request_shutdown()
            task = TaskMeta(task_id="t-shutdown", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT)
            decision, result = route_task(task)
            assert not result.success
            assert result.normalized_error == "shutdown_in_progress"
        finally:
            mgr._shutdown_requested = False
            mgr._shutdown_time = None

    def test_completes_in_flight_on_shutdown(self):
        """Tasks registered before shutdown should complete."""
        mgr = ShutdownManager()
        mgr.register_task("t1", "openai_primary", "codex_cli")
        mgr.request_shutdown()
        assert not mgr.should_accept_new_tasks()
        assert not mgr.is_drained()
        # Task completes
        mgr.unregister_task("t1")
        assert mgr.is_drained()
