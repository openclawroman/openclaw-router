"""Tests for trace ID propagation."""

import re
import pytest
from unittest.mock import MagicMock
from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState, ExecutorResult
from router.policy import route_task


class TestTraceIdGeneration:
    def test_route_task_generates_trace_id(self, monkeypatch):
        """route_task should generate a trace_id and include it in the decision."""
        monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
        monkeypatch.setattr("router.policy.build_chain", lambda task, state: [
            MagicMock(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini"),
        ])
        monkeypatch.setattr("router.policy._run_executor", lambda entry, task, trace_id="": ExecutorResult(
            task_id="t1", tool="codex_cli", backend="openai_native",
            model_profile="codex_gpt54_mini", success=True, trace_id=trace_id))
        monkeypatch.setattr("router.policy.get_breaker", lambda: MagicMock(
            is_available=lambda t, b: True, record_success=lambda *a: None))

        task = TaskMeta(task_id="t1", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT)
        decision, result = route_task(task)

        assert decision.trace_id
        assert len(decision.trace_id) == 12
        assert re.match(r'^[a-f0-9]{12}$', decision.trace_id)

    def test_trace_id_propagated_to_result(self, monkeypatch):
        """trace_id should be stamped on the ExecutorResult."""
        monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
        monkeypatch.setattr("router.policy.build_chain", lambda task, state: [
            MagicMock(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini"),
        ])
        captured_trace = {}
        def mock_executor(entry, task, trace_id=""):
            captured_trace["trace_id"] = trace_id
            return ExecutorResult(
                task_id="t1", tool="codex_cli", backend="openai_native",
                model_profile="codex_gpt54_mini", success=True, trace_id=trace_id)
        monkeypatch.setattr("router.policy._run_executor", mock_executor)
        monkeypatch.setattr("router.policy.get_breaker", lambda: MagicMock(
            is_available=lambda t, b: True, record_success=lambda *a: None))

        task = TaskMeta(task_id="t1", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT)
        decision, result = route_task(task)

        assert decision.trace_id == captured_trace["trace_id"]
        assert result.trace_id == decision.trace_id


class TestTraceIdInFallback:
    def test_same_trace_id_across_fallbacks(self, monkeypatch):
        """All fallback attempts should have the same trace_id."""
        trace_ids = []

        def mock_executor(entry, task, trace_id=""):
            trace_ids.append(trace_id)
            if entry.tool == "codex_cli":
                return ExecutorResult(task_id="t1", tool=entry.tool, backend=entry.backend,
                    model_profile=entry.model_profile, success=False, normalized_error="rate_limited")
            return ExecutorResult(task_id="t1", tool=entry.tool, backend=entry.backend,
                model_profile=entry.model_profile, success=True)

        monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
        monkeypatch.setattr("router.policy.build_chain", lambda task, state: [
            MagicMock(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini"),
            MagicMock(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        ])
        monkeypatch.setattr("router.policy._run_executor", mock_executor)
        monkeypatch.setattr("router.policy.get_breaker", lambda: MagicMock(
            is_available=lambda t, b: True, record_success=lambda *a: None, record_failure=lambda *a: None))

        task = TaskMeta(task_id="t1", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT)
        decision, result = route_task(task)

        # All calls should have the same trace_id
        assert len(set(trace_ids)) == 1
        assert trace_ids[0] == decision.trace_id
