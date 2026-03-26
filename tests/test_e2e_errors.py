"""End-to-end error/fallback tests — all error paths through route_task.

Exercises: executor failures, fallback chaining, error history, partial success,
warning extraction, and trace logging. All executors are mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState,
    ExecutorResult, ChainEntry,
)
from router.policy import route_task


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_task(**kwargs):
    defaults = dict(
        task_id="test-e2e-errors", agent="coder",
        task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT, requires_repo_write=False,
        requires_multimodal=False, has_screenshots=False, swarm=False,
        repo_path="/tmp/repo", cwd="/tmp/repo",
        summary="test error fallback end-to-end",
    )
    defaults.update(kwargs)
    return TaskMeta(**defaults)


def _make_result(success, error_type="rate_limited", tool="codex_cli",
                 backend="openai_native", model_profile="codex_gpt54_mini",
                 final_summary=None, partial_success=False, stderr_ref=None):
    return ExecutorResult(
        task_id="test-e2e-errors", tool=tool, backend=backend,
        model_profile=model_profile, success=success,
        normalized_error=error_type if not success else None,
        exit_code=0 if success else 1,
        latency_ms=150 if success else 100,
        cost_estimate_usd=0.002 if success else 0.0,
        final_summary=final_summary,
        partial_success=partial_success,
        stderr_ref=stderr_ref,
    )


def _default_chain():
    return [
        ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini"),
        ChainEntry(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        ChainEntry(tool="codex_cli", backend="openrouter", model_profile="openrouter_minimax"),
    ]


def _patch_router(monkeypatch, chain=None, executor_results=None):
    monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
    monkeypatch.setattr("router.policy.build_chain", lambda task, state: chain or _default_chain())
    monkeypatch.setattr("router.policy.get_reliability_config",
                        lambda: {"chain_timeout_s": 600, "max_fallbacks": 3})

    breaker = MagicMock()
    breaker.is_available.return_value = True
    monkeypatch.setattr("router.policy.get_breaker", lambda: breaker)
    monkeypatch.setattr("router.policy.get_notifier", lambda: MagicMock())
    monkeypatch.setattr("router.policy.AttemptLogger", lambda: MagicMock())

    shutdown_mgr = MagicMock()
    shutdown_mgr.should_accept_new_tasks.return_value = True
    monkeypatch.setattr("router.health.get_shutdown_manager", lambda: shutdown_mgr)

    if executor_results:
        _results = iter(executor_results)
        monkeypatch.setattr("router.policy._run_executor",
                            lambda entry, task, trace_id="": next(_results))


# ── Eligible error fallback (1 parametrized test, 7 cases) ──────────────────

class TestEligibleErrorFallback:
    """Each eligible error type triggers fallback to next provider → success."""

    @pytest.mark.parametrize("error_type", [
        "auth_error", "rate_limited", "quota_exhausted",
        "provider_timeout", "provider_unavailable",
        "transient_network_error", "model_not_found",
    ])
    def test_eligible_error_triggers_fallback(self, monkeypatch, error_type):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type=error_type, tool="codex_cli", backend="openai_native"),
            _make_result(success=True, tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        ])
        decision, result = route_task(_make_task())
        assert result.success is True
        assert decision.attempted_fallback is True
        assert decision.fallback_from == "codex_cli"
        assert result.tool == "claude_code"


# ── All providers fail (2 parametrized tests) ───────────────────────────────

class TestAllProvidersFail:
    """All providers fail — result carries last error and full error_history."""

    @pytest.mark.parametrize("errors,expected_msgs", [
        (
            [("auth_error", "Codex auth token expired"),
             ("auth_error", "Claude API key invalid"),
             ("auth_error", "OpenRouter API key revoked")],
            ["Codex auth token expired", "Claude API key invalid", "OpenRouter API key revoked"],
        ),
        (
            [("auth_error", "Authentication failed"),
             ("provider_timeout", "Claude timed out after 30s"),
             ("provider_unavailable", "OpenRouter 503 Service Unavailable")],
            None,  # mixed — just check types
        ),
    ], ids=["all_auth", "mixed_errors"])
    def test_all_providers_fail(self, monkeypatch, errors, expected_msgs):
        results = []
        for err_type, msg in errors:
            results.append(_make_result(success=False, error_type=err_type, final_summary=msg))
        _patch_router(monkeypatch, executor_results=results)
        decision, result = route_task(_make_task())
        assert result.success is False
        assert len(result.error_history) == 3
        for i, (err_type, msg) in enumerate(errors):
            assert result.error_history[i]["error_type"] == err_type
        if expected_msgs:
            assert result.error_history[2]["error_message"] == expected_msgs[2]
        assert decision.error_history is result.error_history


# ── Non-eligible errors — no fallback (1 parametrized test) ─────────────────

class TestNonEligibleNoFallback:
    """Non-eligible errors stop the chain immediately."""

    @pytest.mark.parametrize("error_type,summary", [
        ("context_too_long", "Input exceeds 128k token limit"),
        ("content_filtered", "Content flagged by safety filter"),
    ])
    def test_non_eligible_stops_chain(self, monkeypatch, error_type, summary):
        chain = _default_chain()
        _patch_router(monkeypatch, chain=chain, executor_results=[
            _make_result(success=False, error_type=error_type, final_summary=summary),
            _make_result(success=True),  # should never be reached
        ])
        decision, result = route_task(_make_task())
        assert result.success is False
        assert result.normalized_error == error_type
        assert decision.attempted_fallback is False
        assert len(result.error_history) == 1
        assert decision.fallback_count == 0


# ── Partial success (2 unique tests) ────────────────────────────────────────

class TestPartialSuccess:
    """Partial success stops the chain and returns usable output."""

    def test_partial_success_stops_chain(self, monkeypatch):
        chain = _default_chain()
        partial = _make_result(success=False, error_type="provider_timeout",
                               final_summary="Generated 80% of output before timeout",
                               partial_success=True)
        _patch_router(monkeypatch, chain=chain, executor_results=[
            partial,
            _make_result(success=True),  # should not be reached
        ])
        decision, result = route_task(_make_task())
        assert result.success is False
        assert result.partial_success is True
        assert result.final_summary == "Generated 80% of output before timeout"
        assert decision.attempted_fallback is False
        assert len(result.error_history) == 0

    def test_partial_success_with_warnings(self, monkeypatch, tmp_path):
        stderr_file = tmp_path / "partial.stderr.txt"
        stderr_file.write_text(
            "INFO: Starting task\n"
            "WARNING: output may be incomplete\n"
            "WARN: timeout approaching\n"
            "ERROR: final step failed\n"
        )
        partial = _make_result(success=False, error_type="provider_timeout",
                               final_summary="Partial: 3 of 4 files written",
                               partial_success=True, stderr_ref=str(stderr_file))
        partial.warnings = ["WARNING: output may be incomplete", "WARN: timeout approaching"]
        _patch_router(monkeypatch, chain=_default_chain(), executor_results=[
            partial,
            _make_result(success=True),
        ])
        decision, result = route_task(_make_task())
        assert result.partial_success is True
        assert len(result.warnings) == 2
        assert "WARNING: output may be incomplete" in result.warnings
        assert all("ERROR" not in w for w in result.warnings)


# ── Error history (2 tests) ─────────────────────────────────────────────────

class TestErrorHistory:
    """Error history accumulates across providers."""

    def test_error_history_across_providers(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="auth_error", tool="codex_cli", backend="openai_native"),
            _make_result(success=False, error_type="rate_limited", tool="claude_code", backend="anthropic"),
            _make_result(success=False, error_type="provider_unavailable", tool="codex_cli", backend="openrouter"),
        ])
        _, result = route_task(_make_task())
        assert len(result.error_history) == 3
        assert result.error_history[0]["tool"] == "codex_cli"
        assert result.error_history[0]["backend"] == "openai_native"
        assert result.error_history[1]["tool"] == "claude_code"
        assert result.error_history[2]["error_type"] == "provider_unavailable"
        # Verify entry fields
        entry = result.error_history[0]
        assert set(entry.keys()) == {"tool", "backend", "model_profile", "error_type", "error_message"}

    def test_fallback_count_in_trace(self, monkeypatch):
        traces_captured = []

        class FakeAttemptLogger:
            def log_trace(self, trace):
                traces_captured.append(trace)

        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="auth_error"),
            _make_result(success=False, error_type="rate_limited"),
            _make_result(success=False, error_type="provider_unavailable"),
        ])
        monkeypatch.setattr("router.policy.AttemptLogger", lambda: FakeAttemptLogger())
        decision, result = route_task(_make_task())
        assert len(traces_captured) == 1
        assert traces_captured[0].fallback_count == 2
        assert traces_captured[0].final_error == "provider_unavailable"
        assert decision.fallback_count == 2


# ── Exception-based error tests (1 parametrized test, 3 cases) ──────────────

class TestExceptionPathFallback:
    """ExecutorError exceptions trigger fallback or stop chain as expected."""

    @pytest.mark.parametrize("first_exc,second_exc,expect_success,error_type", [
        ("rate_limited", None, True, "rate_limited"),
        (None, None, False, "context_too_long"),  # context_too_long → no fallback
        ("generic_503", None, True, "provider_unavailable"),
    ], ids=["eligible_exception", "non_eligible_exception", "generic_exception"])
    def test_exception_handling(self, monkeypatch, first_exc, second_exc, expect_success, error_type):
        from router.errors import RateLimitedError, ContextTooLongError
        call_count = 0

        def mock_run_executor(entry, task, trace_id=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                if first_exc == "rate_limited":
                    raise RateLimitedError("429 Too Many Requests")
                elif first_exc == "generic_503":
                    raise RuntimeError("503 Service Unavailable")
                else:
                    raise ContextTooLongError("Input exceeds 128k context limit")
            return _make_result(success=True, tool=entry.tool, backend=entry.backend)

        _patch_router(monkeypatch, executor_results=None)
        monkeypatch.setattr("router.policy._run_executor", mock_run_executor)

        decision, result = route_task(_make_task())
        assert result.success is expect_success
        if expect_success:
            assert call_count == 2
            assert len(result.error_history) == 1
            assert result.error_history[0]["error_type"] == error_type
        else:
            assert result.normalized_error == error_type
            assert call_count == 1
            assert decision.attempted_fallback is False
