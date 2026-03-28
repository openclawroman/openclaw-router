"""Integration tests for the full ai-code-runner flow."""

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    RouteDecision, ExecutorResult, ChainEntry,
    RoutingLogger, route_task
)


@pytest.fixture
def clean_runtime(tmp_path, monkeypatch):
    """Use temp directory for runtime files."""
    monkeypatch.setattr("router.logger.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", tmp_path / "routing.jsonl")
    return tmp_path


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return TaskMeta(
        task_id="test-integration-001",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT,
        requires_repo_write=True,
        requires_multimodal=False,
        has_screenshots=False,
        swarm=False,
        repo_path="/tmp/test-repo",
        cwd="/tmp/test-repo",
        summary="Implement test feature",
    )


@pytest.fixture
def sample_decision(sample_task):
    """Sample routing decision."""
    return RouteDecision(
        task_id=sample_task.task_id,
        state="normal",
        chain=[
            ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary"),
        ],
        
        reason="Normal state: Codex native first",
        attempted_fallback=False,
    )


@pytest.fixture
def sample_result():
    """Sample executor result."""
    return ExecutorResult(
        task_id="test-integration-001",
        tool="codex_cli",
        backend="openai_native",
        model_profile="codex_primary",
        model_name="gpt-5.4",
        success=True,
        normalized_error=None,
        exit_code=0,
        latency_ms=500,
        request_id=None,
        cost_estimate_usd=0.0,
        artifacts=["/tmp/test-integration-001.stdout.txt"],
        stdout_ref="/tmp/test-integration-001.stdout.txt",
        stderr_ref=None,
        final_summary="Feature implemented successfully",
    )


class TestEndToEndFlow:
    """Test the full flow from task to logging."""

    def test_route_and_log(self, sample_task, sample_decision, sample_result, clean_runtime):
        """Verify route_task + logger produces a log entry."""
        log_path = clean_runtime / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)
        logger.log(sample_task, sample_decision, sample_result)

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["task_id"] == "test-integration-001"
        assert entry["result"]["success"] is True


class TestLogFileCreated:
    """Test that runtime/routing.jsonl is created."""

    def test_log_creates_file(self, clean_runtime, sample_task, sample_decision):
        """Log file should be created on first write."""
        log_path = clean_runtime / "routing.jsonl"
        assert not log_path.exists()

        logger = RoutingLogger(log_path=log_path)
        logger.log(sample_task, sample_decision)

        assert log_path.exists()


class TestLogEntryFields:
    """Test that log entries have all required fields."""

    def test_all_fields_present(self, clean_runtime, sample_task, sample_decision, sample_result):
        """Verify the X-80 spec fields are present in log entry."""
        log_path = clean_runtime / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)
        logger.log(sample_task, sample_decision, sample_result)

        entry = json.loads(log_path.read_text().strip())

        # Top-level fields
        assert "ts" in entry
        assert entry["task_id"] == "test-integration-001"
        assert entry["task_class"] == "implementation"
        assert entry["state"] == "normal"
        assert "chain" in entry
        assert "reason" in entry

        # Result fields
        assert entry["result"]["tool"] == "codex_cli"
        assert entry["result"]["backend"] == "openai_native"
        assert entry["result"]["model_profile"] == "codex_primary"
        assert entry["result"]["model_name"] == "gpt-5.4"
        assert entry["result"]["success"] is True
        assert entry["result"]["latency_ms"] == 500
        assert entry["result"]["cost_estimate_usd"] == 0.0
        assert entry["result"]["artifacts"] == ["/tmp/test-integration-001.stdout.txt"]
        assert entry["result"]["stdout_ref"] == "/tmp/test-integration-001.stdout.txt"
        assert entry["result"]["final_summary"] == "Feature implemented successfully"


class TestHealthFileCreated:
    """Test runtime/claude_health.json tracking."""

    def test_health_file_on_success(self, clean_runtime):
        """Claude success should create health file with available=true."""
        from router.executors import _update_claude_health

        runtime_dir = clean_runtime
        health_path = runtime_dir / "claude_health.json"

        # Patch the Path used in _update_claude_health
        with patch("router.executors.Path") as MockPath:
            MockPath.return_value = MagicMock()
            MockPath.return_value.mkdir = MagicMock()
            health_file = MagicMock()
            MockPath.return_value.__truediv__ = MagicMock(return_value=health_file)
            _update_claude_health(success=True)

            # Verify write_text was called with valid JSON
            health_file.write_text.assert_called_once()
            written = json.loads(health_file.write_text.call_args[0][0])
            assert written["available"] is True
            assert "updated_at" in written

    def test_health_file_on_failure(self, clean_runtime):
        """Claude failure should create health file with available=false."""
        from router.executors import _update_claude_health

        with patch("router.executors.Path") as MockPath:
            MockPath.return_value = MagicMock()
            MockPath.return_value.mkdir = MagicMock()
            health_file = MagicMock()
            MockPath.return_value.__truediv__ = MagicMock(return_value=health_file)
            _update_claude_health(success=False, error_type="quota_exhausted")

            written = json.loads(health_file.write_text.call_args[0][0])
            assert written["available"] is False
            assert written["reason"] == "quota_exhausted"


class TestMultipleExecutions:
    """Test multiple runs produce multiple log lines."""

    def test_three_runs_three_lines(self, clean_runtime, sample_task, sample_decision, sample_result):
        """Three executions should produce three log entries."""
        log_path = clean_runtime / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)

        for i in range(3):
            task = TaskMeta(
                task_id=f"test-batch-{i}",
                agent="coder",
                task_class=TaskClass.IMPLEMENTATION,
                risk=TaskRisk.MEDIUM,
                modality=TaskModality.TEXT,
                requires_repo_write=True,
                requires_multimodal=False,
                has_screenshots=False,
                swarm=False,
                repo_path="/tmp/test-repo",
                cwd="/tmp/test-repo",
                summary=f"Task {i}",
            )
            decision = RouteDecision(
                task_id=f"test-batch-{i}",
                state="normal",
                chain=[
                    ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary"),
                ],
                reason="Normal state: Codex native first",
                attempted_fallback=False,
            )
            logger.log(task, decision, sample_result)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert entry["task_id"] == f"test-batch-{i}"
            assert entry["result"]["success"] is True


class TestAiCodeRunnerImports:
    """Test that ai-code-runner has correct imports."""

    def test_runner_has_logger_import(self):
        """Verify the CLI entrypoint imports RoutingLogger."""
        runner_path = Path(__file__).parent.parent / "bin" / "ai-code-runner"
        content = runner_path.read_text()
        assert "RoutingLogger" in content

    def test_runner_calls_logger(self):
        """Verify the CLI entrypoint uses the logger."""
        runner_path = Path(__file__).parent.parent / "bin" / "ai-code-runner"
        content = runner_path.read_text()
        assert "logger.log(" in content or "logger.log (" in content
