"""Tests for partial success handling in the router.

Partial success: executor returns success=False but has useful output
(final_summary or artifacts). This should stop the chain and NOT trigger
fallback to the next executor.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, ExecutorResult, ChainEntry,
    TaskClass, TaskRisk, TaskModality, CodexState,
)
from router.policy import _run_executor, _extract_warnings, route_task
from router.errors import can_fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(tool="codex_cli", backend="openai_native", profile="codex_gpt54_mini"):
    return ChainEntry(tool=tool, backend=backend, model_profile=profile)


def _make_task():
    return TaskMeta(
        task_id="test-ps-001",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
    )


# ---------------------------------------------------------------------------
# _detect_partial_success via _run_executor
# ---------------------------------------------------------------------------

class TestPartialSuccessDetection:
    """Detect partial success from executor results."""

    def test_success_false_with_final_summary(self):
        """success=False + final_summary → partial_success=True."""
        result = ExecutorResult(
            task_id="t1", tool="codex_cli", backend="openai_native",
            success=False, final_summary="Partially completed: 3/5 files edited",
        )
        with patch("router.policy.run_codex", return_value=result):
            out = _run_executor(_make_entry(), _make_task(), trace_id="abc")
        assert out.partial_success is True
        assert out.success is False

    def test_success_false_with_artifacts(self):
        """success=False + artifacts → partial_success=True."""
        result = ExecutorResult(
            task_id="t1", tool="codex_cli", backend="openai_native",
            success=False, artifacts=["/tmp/t1.stdout.txt"],
        )
        with patch("router.policy.run_codex", return_value=result):
            out = _run_executor(_make_entry(), _make_task(), trace_id="abc")
        assert out.partial_success is True
        assert out.success is False

    def test_success_false_no_summary_no_artifacts(self):
        """success=False, no summary, no artifacts → partial_success=False."""
        result = ExecutorResult(
            task_id="t1", tool="codex_cli", backend="openai_native",
            success=False,
        )
        with patch("router.policy.run_codex", return_value=result):
            out = _run_executor(_make_entry(), _make_task(), trace_id="abc")
        assert out.partial_success is False

    def test_success_true_no_partial(self):
        """success=True → partial_success=False."""
        result = ExecutorResult(
            task_id="t1", tool="codex_cli", backend="openai_native",
            success=True, final_summary="All done",
        )
        with patch("router.policy.run_codex", return_value=result):
            out = _run_executor(_make_entry(), _make_task(), trace_id="abc")
        assert out.partial_success is False
        assert out.success is True

    def test_success_false_with_both_summary_and_artifacts(self):
        """success=False + both summary and artifacts → partial_success=True."""
        result = ExecutorResult(
            task_id="t1", tool="codex_cli", backend="openai_native",
            success=False,
            final_summary="Partial output",
            artifacts=["/tmp/t1.out.txt"],
        )
        with patch("router.policy.run_codex", return_value=result):
            out = _run_executor(_make_entry(), _make_task(), trace_id="abc")
        assert out.partial_success is True


# ---------------------------------------------------------------------------
# Warnings extraction
# ---------------------------------------------------------------------------

class TestWarningsExtraction:
    """Extract warning lines from stderr files."""

    def test_no_stderr_ref(self):
        """stderr_ref=None → empty warnings."""
        assert _extract_warnings(None) == []

    def test_nonexistent_file(self):
        """stderr_ref points to missing file → empty warnings (no crash)."""
        assert _extract_warnings("/tmp/nonexistent-stderr-xyz.txt") == []

    def test_warnings_lowercase(self):
        """Lowercase 'warning' lines are captured."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("INFO: starting\nwarning: deprecated API used\nINFO: done\n")
            f.flush()
            path = f.name
        try:
            result = _extract_warnings(path)
            assert len(result) == 1
            assert "deprecated API used" in result[0]
        finally:
            os.unlink(path)

    def test_warnings_uppercase_warn(self):
        """Uppercase 'WARN' lines are captured."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("INFO: ok\nWARN: memory limit approaching\nDEBUG: fine\n")
            f.flush()
            path = f.name
        try:
            result = _extract_warnings(path)
            assert len(result) == 1
            assert "memory limit approaching" in result[0]
        finally:
            os.unlink(path)

    def test_multiple_warnings(self):
        """Multiple warning lines are all captured."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("warning: first\nINFO: ok\nWARN: second\nwarning: third\n")
            f.flush()
            path = f.name
        try:
            result = _extract_warnings(path)
            assert len(result) == 3
        finally:
            os.unlink(path)

    def test_no_warnings(self):
        """File with no warning lines → empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("INFO: all good\nDEBUG: nothing to see\n")
            f.flush()
            path = f.name
        try:
            assert _extract_warnings(path) == []
        finally:
            os.unlink(path)

    def test_warnings_stored_on_result(self):
        """Warnings from stderr appear on the ExecutorResult."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("WARN: something\n")
            f.flush()
            stderr_path = f.name
        try:
            result = ExecutorResult(
                task_id="t1", tool="codex_cli", backend="openai_native",
                success=True, stderr_ref=stderr_path,
            )
            with patch("router.policy.run_codex", return_value=result):
                out = _run_executor(_make_entry(), _make_task(), trace_id="abc")
            assert len(out.warnings) == 1
            assert "something" in out.warnings[0]
        finally:
            os.unlink(stderr_path)


# ---------------------------------------------------------------------------
# Chain / fallback behavior with partial success
# ---------------------------------------------------------------------------

class TestPartialSuccessChainBehavior:
    """Partial success stops the chain — does NOT fallback."""

    @pytest.fixture()
    def base_setup(self):
        """Common mocking for route_task tests."""
        return {
            "task": _make_task(),
            "state": CodexState.OPENAI_PRIMARY,
        }

    def test_partial_success_stops_chain(self, base_setup):
        """Partial success from first executor → no fallback to second."""
        partial_result = ExecutorResult(
            task_id="test-ps-001", tool="codex_cli", backend="openai_native",
            success=False, final_summary="3/5 tasks done",
            normalized_error="toolchain_error", partial_success=True,
        )
        second_result = ExecutorResult(
            task_id="test-ps-001", tool="claude_code", backend="anthropic",
            success=True, final_summary="Fixed everything",
        )

        call_count = 0

        def mock_run_executor(entry, task, trace_id=""):
            nonlocal call_count
            call_count += 1
            if entry.tool == "codex_cli" and entry.backend == "openai_native":
                return partial_result
            return second_result

        mock_sm = MagicMock(
            should_accept_new_tasks=MagicMock(return_value=True),
            register_task=MagicMock(),
            unregister_task=MagicMock(),
        )

        with patch("router.policy._run_executor", side_effect=mock_run_executor), \
             patch("router.policy.resolve_state", return_value=base_setup["state"]), \
             patch("router.policy.get_breaker") as mock_breaker, \
             patch("router.health.get_shutdown_manager", return_value=mock_sm), \
             patch("router.policy.get_notifier") as mock_notifier, \
             patch("router.policy.AttemptLogger"):
            mock_breaker.return_value = MagicMock(
                is_available=MagicMock(return_value=True),
                record_success=MagicMock(),
                record_failure=MagicMock(),
            )
            mock_notifier.return_value = MagicMock(check_conservation_duration=MagicMock())

            decision, result = route_task(base_setup["task"])

        # Should have only called _run_executor once (partial success stops chain)
        assert call_count == 1
        assert result.partial_success is True
        assert result.success is False
        assert decision.fallback_count == 0

    def test_partial_success_with_fallback_eligible_error_still_returns(self, base_setup):
        """Even if error is fallback-eligible, partial success wins — no fallback."""
        partial_result = ExecutorResult(
            task_id="test-ps-001", tool="codex_cli", backend="openai_native",
            success=False,
            final_summary="Got partial output before rate limit",
            normalized_error="rate_limited",  # Normally fallback-eligible!
            partial_success=True,
        )
        # Confirm rate_limited IS fallback-eligible
        assert can_fallback("rate_limited") is True

        call_count = 0

        def mock_run_executor(entry, task, trace_id=""):
            nonlocal call_count
            call_count += 1
            return partial_result

        mock_sm = MagicMock(
            should_accept_new_tasks=MagicMock(return_value=True),
            register_task=MagicMock(),
            unregister_task=MagicMock(),
        )

        with patch("router.policy._run_executor", side_effect=mock_run_executor), \
             patch("router.policy.resolve_state", return_value=base_setup["state"]), \
             patch("router.policy.get_breaker") as mock_breaker, \
             patch("router.health.get_shutdown_manager", return_value=mock_sm), \
             patch("router.policy.get_notifier") as mock_notifier, \
             patch("router.policy.AttemptLogger"):
            mock_breaker.return_value = MagicMock(
                is_available=MagicMock(return_value=True),
                record_success=MagicMock(),
                record_failure=MagicMock(),
            )
            mock_notifier.return_value = MagicMock(check_conservation_duration=MagicMock())

            decision, result = route_task(base_setup["task"])

        assert call_count == 1
        assert result.partial_success is True
        assert decision.fallback_count == 0

    def test_no_partial_success_falls_back_normally(self, base_setup):
        """success=False, no summary/artifacts → partial_success=False → fallback."""
        fail_result = ExecutorResult(
            task_id="test-ps-001", tool="codex_cli", backend="openai_native",
            success=False, normalized_error="auth_error",
        )
        success_result = ExecutorResult(
            task_id="test-ps-001", tool="claude_code", backend="anthropic",
            success=True, final_summary="Done via Claude",
        )

        call_count = 0

        def mock_run_executor(entry, task, trace_id=""):
            nonlocal call_count
            call_count += 1
            if entry.tool == "codex_cli":
                return fail_result
            return success_result

        mock_sm = MagicMock(
            should_accept_new_tasks=MagicMock(return_value=True),
            register_task=MagicMock(),
            unregister_task=MagicMock(),
        )

        with patch("router.policy._run_executor", side_effect=mock_run_executor), \
             patch("router.policy.resolve_state", return_value=base_setup["state"]), \
             patch("router.policy.get_breaker") as mock_breaker, \
             patch("router.health.get_shutdown_manager", return_value=mock_sm), \
             patch("router.policy.get_notifier") as mock_notifier, \
             patch("router.policy.AttemptLogger"):
            mock_breaker.return_value = MagicMock(
                is_available=MagicMock(return_value=True),
                record_success=MagicMock(),
                record_failure=MagicMock(),
            )
            mock_notifier.return_value = MagicMock(check_conservation_duration=MagicMock())

            decision, result = route_task(base_setup["task"])

        # Should have called both executors (fallback happened)
        assert call_count >= 2
        assert result.success is True
        assert result.partial_success is False
        assert decision.attempted_fallback is True
