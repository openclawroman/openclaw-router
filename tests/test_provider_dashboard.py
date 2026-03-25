"""Tests for provider health dashboard."""

import json
import pytest
from unittest.mock import MagicMock
from router.circuit_breaker import CircuitBreaker
from router.provider_dashboard import ProviderDashboard, DashboardReport, ProviderStatus


class TestCircuitBreakerHealthSummary:
    def test_empty_breaker(self):
        cb = CircuitBreaker()
        summary = cb.get_health_summary()
        assert summary["total_providers"] == 0
        assert summary["healthy"] == 0

    def test_healthy_providers(self):
        cb = CircuitBreaker()
        cb.record_success("codex_cli", "openai_native")
        cb.record_success("claude_code", "anthropic")
        
        summary = cb.get_health_summary()
        assert summary["total_providers"] == 2
        assert summary["healthy"] == 2
        assert summary["degraded"] == 0
        assert summary["unhealthy"] == 0

    def test_mixed_health(self):
        cb = CircuitBreaker(threshold=2)
        # Healthy
        cb.record_success("codex_cli", "openai_native")
        # Unhealthy (threshold = 2)
        cb.record_failure("claude_code", "anthropic", "rate_limited")
        cb.record_failure("claude_code", "anthropic", "rate_limited")
        
        summary = cb.get_health_summary()
        assert summary["total_providers"] == 2
        assert summary["healthy"] == 1
        assert summary["unhealthy"] == 1


class TestProviderDashboard:
    def test_get_report_empty(self):
        cb = CircuitBreaker()
        dashboard = ProviderDashboard(breaker=cb)
        report = dashboard.get_report()
        
        assert report.total_providers == 0
        assert report.overall_status == "healthy"
        assert len(report.providers) == 0

    def test_get_report_with_providers(self):
        cb = CircuitBreaker()
        cb.record_success("codex_cli", "openai_native")
        cb.record_success("claude_code", "anthropic")
        
        dashboard = ProviderDashboard(breaker=cb)
        report = dashboard.get_report()
        
        assert report.total_providers == 2
        assert report.healthy_count == 2
        assert report.overall_status == "healthy"

    def test_report_shows_degraded(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        # This should open the circuit
        
        dashboard = ProviderDashboard(breaker=cb)
        report = dashboard.get_report()
        
        assert report.overall_status == "critical"
        assert report.unhealthy_count == 1


class TestDashboardReport:
    def test_to_dict(self):
        report = DashboardReport(
            providers=[ProviderStatus(name="test", circuit_state="closed", failure_count=0, healthy=True)],
            total_providers=1, healthy_count=1, degraded_count=0, unhealthy_count=0,
            overall_status="healthy",
        )
        d = report.to_dict()
        assert d["overall_status"] == "healthy"
        assert d["total_providers"] == 1
        assert len(d["providers"]) == 1

    def test_to_table(self):
        report = DashboardReport(
            providers=[
                ProviderStatus(name="codex_cli:openai", circuit_state="closed", failure_count=0, healthy=True),
                ProviderStatus(name="claude_code:anthropic", circuit_state="open", failure_count=5, healthy=False),
            ],
            total_providers=2, healthy_count=1, degraded_count=0, unhealthy_count=1,
            overall_status="critical",
        )
        table = report.to_table()
        assert "codex_cli:openai" in table
        assert "claude_code:anthropic" in table
        assert "CRITICAL" in table
