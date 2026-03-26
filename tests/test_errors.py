"""Tests for router.errors — normalized error types, fallback eligibility, mapping."""

import pytest
from unittest.mock import MagicMock

from router.errors import (
    RouterError, ExecutorError, normalize_error, can_fallback,
    ELIGIBLE_FALLBACK_ERRORS, NON_ELIGIBLE_ERRORS, NORMALIZED_ERROR_TYPES,
    CodexAuthError, CodexQuotaError, ClaudeAuthError, ClaudeQuotaError,
    RateLimitedError, ProviderUnavailableError, ProviderTimeoutError,
    TransientNetworkError, OpenRouterError, ModelNotFoundError,
    InvalidPayloadError, MissingRepoPathError, PermissionDeniedLocalError,
    GitConflictError, ToolchainError, TemplateRenderError, UnsupportedTaskError,
    ContextTooLongError, ContentFilteredError, CodexToolError, ClaudeToolError,
)
from router.models import (
    TaskMeta, RouteDecision, ExecutorResult, ChainEntry,
    TaskClass, TaskRisk, TaskModality, CodexState,
)
from router.policy import route_task


# ── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_eligible_fallback_errors_count(self):
        assert len(ELIGIBLE_FALLBACK_ERRORS) == 7

    def test_eligible_fallback_errors_contents(self):
        assert ELIGIBLE_FALLBACK_ERRORS == {
            "auth_error", "rate_limited", "quota_exhausted",
            "provider_unavailable", "provider_timeout",
            "transient_network_error", "model_not_found",
        }

    def test_non_eligible_errors_count(self):
        assert len(NON_ELIGIBLE_ERRORS) == 9

    def test_non_eligible_errors_contents(self):
        assert NON_ELIGIBLE_ERRORS == {
            "invalid_payload", "missing_repo_path", "permission_denied_local",
            "git_conflict", "toolchain_error", "template_render_error",
            "unsupported_task", "context_too_long", "content_filtered",
        }

    def test_normalized_error_types_is_union(self):
        assert NORMALIZED_ERROR_TYPES == ELIGIBLE_FALLBACK_ERRORS | NON_ELIGIBLE_ERRORS
        assert len(NORMALIZED_ERROR_TYPES) == 16


# ── can_fallback ─────────────────────────────────────────────────────────────

class TestCanFallback:
    @pytest.mark.parametrize("error_type", [
        "auth_error", "rate_limited", "quota_exhausted",
        "provider_unavailable", "provider_timeout",
        "transient_network_error", "model_not_found",
    ])
    def test_eligible_errors_can_fallback(self, error_type):
        assert can_fallback(error_type) is True

    @pytest.mark.parametrize("error_type", [
        "invalid_payload", "missing_repo_path", "permission_denied_local",
        "git_conflict", "toolchain_error", "template_render_error",
        "unsupported_task", "context_too_long", "content_filtered",
    ])
    def test_non_eligible_errors_cannot_fallback(self, error_type):
        assert can_fallback(error_type) is False

    def test_unknown_error_cannot_fallback(self):
        assert can_fallback("unknown_error") is False


# ── normalize_error — string mapping ─────────────────────────────────────────

class TestNormalizeErrorStrings:
    @pytest.mark.parametrize("msg,expected", [
        ("quota exceeded for the month", "quota_exhausted"),
        ("API limit reached", "quota_exhausted"),
        ("rate limit exceeded", "rate_limited"),
        ("HTTP 429 Too Many Requests", "rate_limited"),
        ("auth token invalid", "auth_error"),
        ("unauthorized access to API", "auth_error"),
        ("401 Unauthorized", "auth_error"),
        ("request timeout after 30s", "provider_timeout"),
        ("connection timed out", "provider_timeout"),
        ("service unavailable", "provider_unavailable"),
        ("503 Service Unavailable", "provider_unavailable"),
        ("500 Internal Server Error", "provider_unavailable"),
        ("network error", "transient_network_error"),
        ("connection refused", "transient_network_error"),
        ("invalid payload format", "invalid_payload"),
        ("400 Bad Request", "invalid_payload"),
        ("missing repo path", "missing_repo_path"),
        ("permission denied on /tmp/repo", "permission_denied_local"),
        ("403 Forbidden", "permission_denied_local"),
        ("git conflict in file.py", "git_conflict"),
        ("merge conflict detected", "git_conflict"),
        ("toolchain error: gcc not found", "toolchain_error"),
        ("template render failed: missing variable", "template_render_error"),
        ("unsupported task class", "unsupported_task"),
        ("something completely different", "unknown_error"),
        ("", "unknown_error"),
    ])
    def test_string_mapping(self, msg, expected):
        assert normalize_error(msg) == expected


# ── normalize_error — Exception handling ─────────────────────────────────────

class TestNormalizeErrorExceptions:
    def test_executor_error_passthrough(self):
        assert normalize_error(CodexAuthError("test auth")) == "auth_error"

    def test_generic_exception(self):
        assert normalize_error(ValueError("401 unauthorized")) == "auth_error"

    @pytest.mark.parametrize("code,expected", [
        (401, "auth_error"), (429, "rate_limited"),
        (500, "provider_unavailable"), (503, "provider_unavailable"),
        (400, "invalid_payload"), (403, "permission_denied_local"),
    ])
    def test_integer_http_code(self, code, expected):
        assert normalize_error(code) == expected


# ── normalize_error — HTTP code mapping ──────────────────────────────────────

class TestNormalizeErrorHttpCodes:
    @pytest.mark.parametrize("code,expected", [
        ("401", "auth_error"), ("429", "rate_limited"),
        ("500", "provider_unavailable"), ("502", "provider_unavailable"),
        ("503", "provider_unavailable"), ("504", "provider_timeout"),
        ("400", "invalid_payload"), ("422", "invalid_payload"),
        ("403", "permission_denied_local"),
    ])
    def test_http_code_mapping(self, code, expected):
        assert normalize_error(f"HTTP {code} error") == expected


# ── ExecutorError base class ─────────────────────────────────────────────────

class TestExecutorError:
    def test_has_error_type(self):
        err = ExecutorError("test", "auth_error")
        assert err.error_type == "auth_error"
        assert str(err) == "test"

    def test_is_router_error(self):
        assert isinstance(ExecutorError("test", "auth_error"), RouterError)

    @pytest.mark.parametrize("cls,expected", [
        (CodexAuthError, "auth_error"), (CodexQuotaError, "quota_exhausted"),
        (ClaudeAuthError, "auth_error"), (ClaudeQuotaError, "quota_exhausted"),
        (RateLimitedError, "rate_limited"), (ProviderUnavailableError, "provider_unavailable"),
        (ProviderTimeoutError, "provider_timeout"), (TransientNetworkError, "transient_network_error"),
        (InvalidPayloadError, "invalid_payload"), (MissingRepoPathError, "missing_repo_path"),
        (PermissionDeniedLocalError, "permission_denied_local"), (GitConflictError, "git_conflict"),
        (ToolchainError, "toolchain_error"), (TemplateRenderError, "template_render_error"),
        (UnsupportedTaskError, "unsupported_task"), (ModelNotFoundError, "model_not_found"),
        (ContextTooLongError, "context_too_long"), (ContentFilteredError, "content_filtered"),
    ])
    def test_error_type(self, cls, expected):
        assert cls().error_type == expected

    def test_codex_tool_error_maps_to_toolchain(self):
        assert CodexToolError().error_type == "toolchain_error"

    def test_claude_tool_error_maps_to_toolchain(self):
        assert ClaudeToolError().error_type == "toolchain_error"

    def test_openrouter_error_default_type(self):
        assert OpenRouterError().error_type == "provider_unavailable"

    def test_all_executor_error_types_in_normalized_set(self):
        classes = [
            CodexAuthError, CodexQuotaError, ClaudeAuthError, ClaudeQuotaError,
            RateLimitedError, ProviderUnavailableError, ProviderTimeoutError,
            TransientNetworkError, InvalidPayloadError, MissingRepoPathError,
            PermissionDeniedLocalError, GitConflictError, ToolchainError,
            TemplateRenderError, UnsupportedTaskError, CodexToolError, ClaudeToolError,
            ModelNotFoundError, ContextTooLongError, ContentFilteredError,
        ]
        for cls in classes:
            assert cls().error_type in NORMALIZED_ERROR_TYPES


# ── Error propagation helpers ────────────────────────────────────────────────

def _make_task(**kwargs):
    defaults = dict(
        task_id="test-err-prop", agent="coder",
        task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT, requires_repo_write=False,
        requires_multimodal=False, has_screenshots=False, swarm=False,
        repo_path="/tmp/repo", cwd="/tmp/repo", summary="test error propagation",
    )
    defaults.update(kwargs)
    return TaskMeta(**defaults)


def _make_result(success=True, error_type=None, tool="codex_cli", backend="openai_native",
                 model_profile="codex_gpt54_mini", final_summary="done"):
    return ExecutorResult(
        task_id="test-err-prop", tool=tool, backend=backend,
        model_profile=model_profile, success=success,
        normalized_error=error_type, final_summary=final_summary,
    )


def _default_chain():
    return [
        MagicMock(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini"),
        MagicMock(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        MagicMock(tool="codex_cli", backend="openrouter", model_profile="openrouter_minimax"),
    ]


def _patch_router(monkeypatch, chain=None, executor_results=None):
    monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
    monkeypatch.setattr("router.policy.build_chain", lambda task, state: chain or _default_chain())
    monkeypatch.setattr("router.policy.get_reliability_config", lambda: {"chain_timeout_s": 600, "max_fallbacks": 3})
    breaker = MagicMock(is_available=MagicMock(return_value=True))
    monkeypatch.setattr("router.policy.get_breaker", lambda: breaker)
    monkeypatch.setattr("router.policy.get_notifier", lambda: MagicMock())
    monkeypatch.setattr("router.policy.AttemptLogger", lambda: MagicMock())
    sm = MagicMock(should_accept_new_tasks=MagicMock(return_value=True))
    monkeypatch.setattr("router.health.get_shutdown_manager", lambda: sm)
    if executor_results:
        _results = iter(executor_results)
        monkeypatch.setattr("router.policy._run_executor", lambda entry, task, trace_id="": next(_results))


# ── Error history model fields ──────────────────────────────────────────────

class TestErrorHistoryModelFields:
    def test_executor_result_has_error_history(self):
        assert hasattr(ExecutorResult(), "error_history")

    def test_route_decision_has_error_history(self):
        assert hasattr(RouteDecision(), "error_history")

    def test_error_history_entry_schema(self):
        for key in ("tool", "backend", "model_profile", "error_type", "error_message"):
            assert key in {"tool": "x", "backend": "x", "model_profile": "x", "error_type": "x", "error_message": "x"}


# ── Single executor success → error_history empty ────────────────────────────

class TestSingleExecutorSuccess:
    def test_success_empty_error_history(self, monkeypatch):
        chain = [MagicMock(tool="codex_cli", backend="openai_native", model_profile="codex_gpt54_mini")]
        _patch_router(monkeypatch, chain=chain, executor_results=[_make_result()])
        decision, result = route_task(_make_task())
        assert result.success is True
        assert result.error_history == []
        assert decision.error_history == []


# ── First fails, second succeeds → 1 error entry ────────────────────────────

class TestFirstFailsSecondSucceeds:
    def test_fallback_error_history_one_entry(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="rate_limited", final_summary="Rate limit exceeded"),
            _make_result(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        ])
        decision, result = route_task(_make_task())
        assert result.success is True
        assert len(result.error_history) == 1
        assert result.error_history[0]["tool"] == "codex_cli"
        assert result.error_history[0]["error_type"] == "rate_limited"
        assert decision.error_history == result.error_history


# ── All executors fail → error_history has N entries ─────────────────────────

class TestAllExecutorsFail:
    def test_all_fail_error_history_three_entries(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="rate_limited", final_summary="Rate limited"),
            _make_result(success=False, error_type="quota_exhausted", tool="claude_code", backend="anthropic", model_profile="claude_primary", final_summary="Quota exhausted"),
            _make_result(success=False, error_type="provider_unavailable", backend="openrouter", model_profile="openrouter_minimax", final_summary="Provider down"),
        ])
        decision, result = route_task(_make_task())
        assert result.success is False
        assert len(result.error_history) == 3
        assert result.error_history[0]["error_type"] == "rate_limited"
        assert result.error_history[1]["error_type"] == "quota_exhausted"
        assert result.error_history[2]["error_type"] == "provider_unavailable"
        assert decision.error_history == result.error_history


# ── Error history captures all fields ────────────────────────────────────────

class TestErrorHistoryCapturesAllFields:
    def test_error_entry_fields(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="auth_error", final_summary="Authentication token expired"),
            _make_result(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        ])
        _, result = route_task(_make_task())
        entry = result.error_history[0]
        assert entry["tool"] == "codex_cli"
        assert entry["backend"] == "openai_native"
        assert entry["model_profile"] == "codex_gpt54_mini"
        assert entry["error_type"] == "auth_error"
        assert entry["error_message"] == "Authentication token expired"

    def test_error_message_from_final_summary(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="rate_limited", final_summary="429 Too Many Requests: retry after 60s"),
            _make_result(),
        ])
        _, result = route_task(_make_task())
        assert result.error_history[0]["error_message"] == "429 Too Many Requests: retry after 60s"


# ── Error history preserved in final result ──────────────────────────────────

class TestErrorHistoryPreservedInResult:
    def test_success_result_has_history(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="transient_network_error"),
            _make_result(success=False, error_type="rate_limited", tool="claude_code", backend="anthropic", model_profile="claude_primary"),
            _make_result(backend="openrouter"),
        ])
        decision, result = route_task(_make_task())
        assert result.success is True
        assert len(result.error_history) == 2
        assert decision.error_history is result.error_history

    def test_failure_result_has_history(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="auth_error"),
            _make_result(success=False, error_type="quota_exhausted", tool="claude_code", backend="anthropic", model_profile="claude_primary"),
            _make_result(success=False, error_type="provider_unavailable", backend="openrouter", model_profile="openrouter_minimax"),
        ])
        decision, result = route_task(_make_task())
        assert result.success is False
        assert len(result.error_history) == 3
        assert decision.error_history is result.error_history


# ── RouteDecision error history ──────────────────────────────────────────────

class TestRouteDecisionErrorHistory:
    def test_decision_has_error_history_on_success(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="rate_limited"),
            _make_result(tool="claude_code", backend="anthropic"),
        ])
        decision, result = route_task(_make_task())
        assert len(decision.error_history) == 1
        assert decision.error_history[0]["error_type"] == "rate_limited"

    def test_decision_error_history_empty_on_no_failures(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[_make_result()])
        decision, result = route_task(_make_task())
        assert decision.error_history == []

    def test_decision_error_history_matches_result(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="provider_unavailable"),
            _make_result(success=False, error_type="auth_error", tool="claude_code", backend="anthropic", model_profile="claude_primary"),
            _make_result(backend="openrouter"),
        ])
        decision, result = route_task(_make_task())
        assert decision.error_history is result.error_history
        assert len(decision.error_history) == 2


# ── Non-fallback errors in history ───────────────────────────────────────────

class TestNonFallbackErrors:
    def test_non_fallback_error_in_history(self, monkeypatch):
        _patch_router(monkeypatch, executor_results=[
            _make_result(success=False, error_type="toolchain_error", final_summary="gcc not found"),
        ])
        decision, result = route_task(_make_task())
        assert result.success is False
        assert len(result.error_history) == 1
        assert result.error_history[0]["error_type"] == "toolchain_error"


# ── Partial success detection ────────────────────────────────────────────────

from unittest.mock import patch
from router.policy import _run_executor, _extract_warnings


def _make_entry(tool="codex_cli", backend="openai_native", profile="codex_gpt54_mini"):
    return ChainEntry(tool=tool, backend=backend, model_profile=profile)


def _make_ps_task():
    return TaskMeta(task_id="test-ps-001", agent="coder", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM)


class TestPartialSuccessDetection:
    @pytest.mark.parametrize("kwargs,expected_partial", [
        ({"success": False, "final_summary": "3/5 files edited"}, True),
        ({"success": False, "artifacts": ["/tmp/t1.stdout.txt"]}, True),
        ({"success": False}, False),
        ({"success": True, "final_summary": "All done"}, False),
        ({"success": False, "final_summary": "Partial output", "artifacts": ["/tmp/t1.out.txt"]}, True),
    ])
    def test_partial_success_cases(self, kwargs, expected_partial):
        result = ExecutorResult(task_id="t1", tool="codex_cli", backend="openai_native", **kwargs)
        with patch("router.policy.run_codex", return_value=result):
            out = _run_executor(_make_entry(), _make_ps_task(), trace_id="abc")
        assert out.partial_success is expected_partial
        if "success" in kwargs:
            assert out.success == kwargs["success"]


# ── Warnings extraction ─────────────────────────────────────────────────────

import os, tempfile


class TestWarningsExtraction:
    def test_no_stderr_ref(self):
        assert _extract_warnings(None) == []

    def test_nonexistent_file(self):
        assert _extract_warnings("/tmp/nonexistent-stderr-xyz.txt") == []

    @pytest.mark.parametrize("content,expected_count", [
        ("INFO: starting\nwarning: deprecated API used\nINFO: done\n", 1),
        ("INFO: ok\nWARN: memory limit approaching\nDEBUG: fine\n", 1),
        ("warning: first\nINFO: ok\nWARN: second\nwarning: third\n", 3),
        ("INFO: all good\nDEBUG: nothing to see\n", 0),
    ])
    def test_warnings_content(self, content, expected_count):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            assert len(_extract_warnings(path)) == expected_count
        finally:
            os.unlink(path)

    def test_warnings_stored_on_result(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("WARN: something\n")
            f.flush()
            stderr_path = f.name
        try:
            result = ExecutorResult(task_id="t1", tool="codex_cli", backend="openai_native", success=True, stderr_ref=stderr_path)
            with patch("router.policy.run_codex", return_value=result):
                out = _run_executor(_make_entry(), _make_ps_task(), trace_id="abc")
            assert len(out.warnings) == 1
            assert "something" in out.warnings[0]
        finally:
            os.unlink(stderr_path)


# ── Chain / fallback behavior with partial success ───────────────────────────

class TestPartialSuccessChainBehavior:
    @pytest.fixture()
    def base_setup(self):
        return {"task": _make_ps_task(), "state": CodexState.OPENAI_PRIMARY}

    def _patch_route(self, monkeypatch, state, mock_run):
        mock_sm = MagicMock(should_accept_new_tasks=MagicMock(return_value=True), register_task=MagicMock(), unregister_task=MagicMock())
        monkeypatch.setattr("router.policy._run_executor", mock_run)
        monkeypatch.setattr("router.policy.resolve_state", lambda: state)
        breaker = MagicMock(is_available=MagicMock(return_value=True), record_success=MagicMock(), record_failure=MagicMock())
        monkeypatch.setattr("router.policy.get_breaker", lambda: breaker)
        monkeypatch.setattr("router.health.get_shutdown_manager", lambda: mock_sm)
        monkeypatch.setattr("router.policy.get_notifier", lambda: MagicMock(check_conservation_duration=MagicMock()))
        monkeypatch.setattr("router.policy.AttemptLogger", lambda: MagicMock())

    def test_partial_success_stops_chain(self, monkeypatch, base_setup):
        partial_result = ExecutorResult(task_id="test-ps-001", tool="codex_cli", backend="openai_native",
            success=False, final_summary="3/5 tasks done", normalized_error="toolchain_error", partial_success=True)
        call_count = [0]
        def mock_run(entry, task, trace_id=""):
            call_count[0] += 1
            return partial_result
        self._patch_route(monkeypatch, base_setup["state"], mock_run)
        decision, result = route_task(base_setup["task"])
        assert call_count[0] == 1
        assert result.partial_success is True
        assert decision.fallback_count == 0

    def test_partial_success_with_fallback_eligible_error_still_returns(self, monkeypatch, base_setup):
        partial_result = ExecutorResult(task_id="test-ps-001", tool="codex_cli", backend="openai_native",
            success=False, final_summary="Got partial output before rate limit",
            normalized_error="rate_limited", partial_success=True)
        assert can_fallback("rate_limited") is True
        call_count = [0]
        def mock_run(entry, task, trace_id=""):
            call_count[0] += 1
            return partial_result
        self._patch_route(monkeypatch, base_setup["state"], mock_run)
        decision, result = route_task(base_setup["task"])
        assert call_count[0] == 1
        assert result.partial_success is True
        assert decision.fallback_count == 0

    def test_no_partial_success_falls_back_normally(self, monkeypatch, base_setup):
        fail_result = ExecutorResult(task_id="test-ps-001", tool="codex_cli", backend="openai_native",
            success=False, normalized_error="auth_error")
        success_result = ExecutorResult(task_id="test-ps-001", tool="claude_code", backend="anthropic",
            success=True, final_summary="Done via Claude")
        call_count = [0]
        def mock_run(entry, task, trace_id=""):
            call_count[0] += 1
            return fail_result if entry.tool == "codex_cli" else success_result
        self._patch_route(monkeypatch, base_setup["state"], mock_run)
        decision, result = route_task(base_setup["task"])
        assert call_count[0] >= 2
        assert result.success is True
        assert decision.attempted_fallback is True
