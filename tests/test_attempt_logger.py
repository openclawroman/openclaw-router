"""Tests for structured logging and attempt trail."""

import json
import pytest
from pathlib import Path
from router.attempt_logger import AttemptLogger, ExecutorAttempt, RoutingTrace


class TestAttemptLogger:
    def test_create_trace(self, tmp_path):
        logger = AttemptLogger(log_path=tmp_path / "test.jsonl")
        trace = logger.create_trace("abc123", "t1", "openai_primary", [{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_gpt54_mini"}])
        assert trace.trace_id == "abc123"
        assert trace.task_id == "t1"
        assert trace.state == "openai_primary"

    def test_log_trace_writes_jsonl(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        logger = AttemptLogger(log_path=log_path)
        trace = RoutingTrace(
            trace_id="abc123", task_id="t1", state="openai_primary",
            chain=[{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_gpt54_mini"}],
            attempts=[ExecutorAttempt(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini", success=True, latency_ms=150)],
            total_latency_ms=150, final_tool="codex_cli", final_success=True,
        )
        logger.log_trace(trace)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "routing_trace"
        assert entry["trace_id"] == "abc123"
        assert len(entry["attempts"]) == 1
        assert entry["attempts"][0]["success"] is True
        assert entry["attempts"][0]["latency_ms"] == 150

    def test_attempt_logger_append(self, tmp_path):
        """Multiple traces append to the same file."""
        log_path = tmp_path / "test.jsonl"
        logger = AttemptLogger(log_path=log_path)

        for i in range(3):
            trace = RoutingTrace(trace_id=f"trace{i}", task_id=f"t{i}", state="openai_primary")
            logger.log_trace(trace)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_skipped_executor_logged(self, tmp_path):
        """Skipped executors (circuit breaker) should appear in trace."""
        log_path = tmp_path / "test.jsonl"
        logger = AttemptLogger(log_path=log_path)
        trace = RoutingTrace(
            trace_id="abc", task_id="t1", state="openai_primary",
            attempts=[
                ExecutorAttempt(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini", success=False, latency_ms=0, skipped=True, skip_reason="circuit_breaker_open"),
                ExecutorAttempt(tool="claude_code", backend="anthropic", model_profile="claude_primary", success=True, latency_ms=200),
            ],
            providers_skipped=["codex_cli:openai_native"],
            fallback_count=1,
            total_latency_ms=200,
            final_tool="claude_code",
            final_success=True,
        )
        logger.log_trace(trace)

        entry = json.loads(log_path.read_text().strip().split("\n")[0])
        assert len(entry["attempts"]) == 2
        assert entry["attempts"][0]["skipped"] is True
        assert entry["attempts"][0]["skip_reason"] == "circuit_breaker_open"
        assert entry["providers_skipped"] == ["codex_cli:openai_native"]


class TestRoutingTrace:
    def test_to_dict_omits_final_error_when_none(self):
        """final_error should not appear in dict when it's None."""
        trace = RoutingTrace(
            trace_id="abc123", task_id="t1", state="openai_primary",
            final_tool="codex_cli", final_success=True,
            final_error=None,
        )
        d = trace.to_dict()
        assert "final_error" not in d

    def test_to_dict_includes_all_fields(self):
        trace = RoutingTrace(
            trace_id="abc123", task_id="t1", state="openai_primary",
            fallback_count=2, chain_timed_out=True,
            final_tool="claude_code", final_success=False,
            final_error="chain_timeout",
        )
        d = trace.to_dict()
        assert d["trace_id"] == "abc123"
        assert d["chain_timed_out"] is True
        assert d["fallback_count"] == 2
        assert d["final_error"] == "chain_timeout"
        assert "timestamp" in d


class TestAttemptLoggerDefaultPath:
    def test_default_path_creates_runtime_dir(self):
        """AttemptLogger with no args uses DEFAULT_LOG_PATH from logger."""
        logger = AttemptLogger()
        assert logger.log_path.exists() or logger.log_path.parent.exists()
        assert logger.log_path.parent.name == "runtime"
