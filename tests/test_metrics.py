"""Tests for metrics aggregation."""

import json
import pytest
from pathlib import Path
from router.metrics import MetricsCollector, MetricsReport, StateMetrics, ModelMetrics


class TestMetricsCollector:
    def test_empty_log_returns_zero_report(self, tmp_path):
        collector = MetricsCollector(log_path=tmp_path / "nonexistent.jsonl")
        report = collector.collect()
        assert report.total_tasks == 0
        assert report.success_rate == 0.0

    def test_collects_from_routing_trace(self, tmp_path):
        log_path = tmp_path / "routing.jsonl"
        entries = [
            {"type": "routing_trace", "trace_id": "a1", "task_id": "t1",
             "state": "openai_primary", "chain": [],
             "attempts": [{"tool": "codex_cli", "backend": "openai_native",
                          "model_profile": "codex_gpt54_mini", "success": True,
                          "latency_ms": 150, "cost_estimate_usd": 0.0}],
             "total_latency_ms": 150, "final_tool": "codex_cli",
             "final_success": True, "timestamp": "2026-03-25T20:00:00+00:00"},
            {"type": "routing_trace", "trace_id": "a2", "task_id": "t2",
             "state": "claude_backup", "chain": [],
             "attempts": [{"tool": "claude_code", "backend": "anthropic",
                          "model_profile": "claude_primary", "success": False,
                          "latency_ms": 200, "normalized_error": "rate_limited"},
                         {"tool": "codex_cli", "backend": "openrouter",
                          "model_profile": "openrouter_minimax", "success": True,
                          "latency_ms": 300, "cost_estimate_usd": 0.002}],
             "fallback_count": 1, "total_latency_ms": 500,
             "final_tool": "codex_cli", "final_success": True,
             "timestamp": "2026-03-25T20:01:00+00:00"},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        collector = MetricsCollector(log_path=log_path)
        report = collector.collect()

        assert report.total_tasks == 2
        assert report.total_success == 2
        assert report.success_rate == 1.0
        assert "openai_primary" in report.by_state
        assert report.by_state["openai_primary"].task_count == 1
        assert "claude_backup" in report.by_state
        assert report.by_state["claude_backup"].fallback_count == 1

    def test_circuit_breaker_skips_counted(self, tmp_path):
        log_path = tmp_path / "routing.jsonl"
        entry = {
            "type": "routing_trace", "trace_id": "b1", "task_id": "t1",
            "state": "openai_primary",
            "attempts": [
                {"tool": "codex_cli", "backend": "openai_native",
                 "model_profile": "codex_gpt54_mini", "success": False,
                 "latency_ms": 0, "skipped": True, "skip_reason": "circuit_breaker_open"},
                {"tool": "claude_code", "backend": "anthropic",
                 "model_profile": "claude_primary", "success": True, "latency_ms": 200},
            ],
            "providers_skipped": ["codex_cli:openai_native"],
            "total_latency_ms": 200, "final_tool": "claude_code",
            "final_success": True, "timestamp": "2026-03-25T20:00:00+00:00",
        }
        log_path.write_text(json.dumps(entry) + "\n")

        collector = MetricsCollector(log_path=log_path)
        report = collector.collect()

        assert report.circuit_breaker_skips == 1
        assert "codex_cli:openai_native" in report.by_provider
        assert report.by_provider["codex_cli:openai_native"].skip_count == 1

    def test_chain_timeout_counted(self, tmp_path):
        log_path = tmp_path / "routing.jsonl"
        entry = {
            "type": "routing_trace", "trace_id": "c1", "task_id": "t1",
            "state": "openai_primary",
            "attempts": [],
            "chain_timed_out": True,
            "total_latency_ms": 600000,
            "final_tool": "chain", "final_success": False,
            "final_error": "chain_timeout",
            "timestamp": "2026-03-25T20:00:00+00:00",
        }
        log_path.write_text(json.dumps(entry) + "\n")

        collector = MetricsCollector(log_path=log_path)
        report = collector.collect()

        assert report.chain_timeouts == 1
        assert report.total_failure == 1

    def test_to_dict(self):
        report = MetricsReport(total_tasks=10, total_success=8, total_failure=2, success_rate=0.8)
        d = report.to_dict()
        assert d["total_tasks"] == 10
        assert d["success_rate"] == 0.8
        assert "by_state" in d
