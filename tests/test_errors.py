"""Tests for router.errors — normalized error types, fallback eligibility, mapping."""

import pytest

from router.errors import (
    RouterError,
    ExecutorError,
    normalize_error,
    can_fallback,
    ELIGIBLE_FALLBACK_ERRORS,
    NON_ELIGIBLE_ERRORS,
    NORMALIZED_ERROR_TYPES,
    # Eligible classes
    CodexAuthError,
    CodexQuotaError,
    ClaudeAuthError,
    ClaudeQuotaError,
    RateLimitedError,
    ProviderUnavailableError,
    ProviderTimeoutError,
    TransientNetworkError,
    OpenRouterError,
    # Non-eligible classes
    InvalidPayloadError,
    MissingRepoPathError,
    PermissionDeniedLocalError,
    GitConflictError,
    ToolchainError,
    TemplateRenderError,
    UnsupportedTaskError,
    # Legacy aliases
    CodexToolError,
    ClaudeToolError,
)


# ── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_eligible_fallback_errors_count(self):
        assert len(ELIGIBLE_FALLBACK_ERRORS) == 6

    def test_eligible_fallback_errors_contents(self):
        assert ELIGIBLE_FALLBACK_ERRORS == {
            "auth_error",
            "rate_limited",
            "quota_exhausted",
            "provider_unavailable",
            "provider_timeout",
            "transient_network_error",
        }

    def test_non_eligible_errors_count(self):
        assert len(NON_ELIGIBLE_ERRORS) == 7

    def test_non_eligible_errors_contents(self):
        assert NON_ELIGIBLE_ERRORS == {
            "invalid_payload",
            "missing_repo_path",
            "permission_denied_local",
            "git_conflict",
            "toolchain_error",
            "template_render_error",
            "unsupported_task",
        }

    def test_normalized_error_types_is_union(self):
        assert NORMALIZED_ERROR_TYPES == ELIGIBLE_FALLBACK_ERRORS | NON_ELIGIBLE_ERRORS
        assert len(NORMALIZED_ERROR_TYPES) == 13


# ── can_fallback ─────────────────────────────────────────────────────────────

class TestCanFallback:
    @pytest.mark.parametrize("error_type", [
        "auth_error",
        "rate_limited",
        "quota_exhausted",
        "provider_unavailable",
        "provider_timeout",
        "transient_network_error",
    ])
    def test_eligible_errors_can_fallback(self, error_type):
        assert can_fallback(error_type) is True

    @pytest.mark.parametrize("error_type", [
        "invalid_payload",
        "missing_repo_path",
        "permission_denied_local",
        "git_conflict",
        "toolchain_error",
        "template_render_error",
        "unsupported_task",
    ])
    def test_non_eligible_errors_cannot_fallback(self, error_type):
        assert can_fallback(error_type) is False

    def test_unknown_error_cannot_fallback(self):
        assert can_fallback("unknown_error") is False


# ── normalize_error — string mapping ─────────────────────────────────────────

class TestNormalizeErrorStrings:
    def test_quota_keyword(self):
        assert normalize_error("quota exceeded for the month") == "quota_exhausted"

    def test_limit_keyword(self):
        assert normalize_error("API limit reached") == "quota_exhausted"

    def test_rate_keyword(self):
        assert normalize_error("rate limit exceeded") == "rate_limited"

    def test_429_status(self):
        assert normalize_error("HTTP 429 Too Many Requests") == "rate_limited"

    def test_auth_keyword(self):
        assert normalize_error("auth token invalid") == "auth_error"

    def test_unauthorized_keyword(self):
        assert normalize_error("unauthorized access to API") == "auth_error"

    def test_401_status(self):
        assert normalize_error("401 Unauthorized") == "auth_error"

    def test_timeout_keyword(self):
        assert normalize_error("request timeout after 30s") == "provider_timeout"

    def test_timed_out_keyword(self):
        assert normalize_error("connection timed out") == "provider_timeout"

    def test_unavailable_keyword(self):
        assert normalize_error("service unavailable") == "provider_unavailable"

    def test_503_status(self):
        assert normalize_error("503 Service Unavailable") == "provider_unavailable"

    def test_500_status(self):
        assert normalize_error("500 Internal Server Error") == "provider_unavailable"

    def test_network_keyword(self):
        assert normalize_error("network error") == "transient_network_error"

    def test_connection_refused(self):
        assert normalize_error("connection refused") == "transient_network_error"

    def test_invalid_payload_keyword(self):
        assert normalize_error("invalid payload format") == "invalid_payload"

    def test_400_status(self):
        assert normalize_error("400 Bad Request") == "invalid_payload"

    def test_missing_repo_path(self):
        assert normalize_error("missing repo path") == "missing_repo_path"

    def test_permission_denied(self):
        assert normalize_error("permission denied on /tmp/repo") == "permission_denied_local"

    def test_403_status(self):
        assert normalize_error("403 Forbidden") == "permission_denied_local"

    def test_git_conflict(self):
        assert normalize_error("git conflict in file.py") == "git_conflict"

    def test_merge_conflict(self):
        assert normalize_error("merge conflict detected") == "git_conflict"

    def test_toolchain_error(self):
        assert normalize_error("toolchain error: gcc not found") == "toolchain_error"

    def test_template_error(self):
        assert normalize_error("template render failed: missing variable") == "template_render_error"

    def test_unsupported_task(self):
        assert normalize_error("unsupported task class") == "unsupported_task"

    def test_unknown_error(self):
        assert normalize_error("something completely different") == "unknown_error"

    def test_empty_string(self):
        assert normalize_error("") == "unknown_error"


# ── normalize_error — Exception handling ─────────────────────────────────────

class TestNormalizeErrorExceptions:
    def test_executor_error_passthrough(self):
        """If given an ExecutorError, return its error_type directly."""
        err = CodexAuthError("test auth")
        assert normalize_error(err) == "auth_error"

    def test_generic_exception(self):
        """Generic exceptions are converted to strings and pattern-matched."""
        err = ValueError("401 unauthorized")
        assert normalize_error(err) == "auth_error"

    def test_integer_http_code(self):
        """Integer HTTP status codes are matched."""
        assert normalize_error(401) == "auth_error"
        assert normalize_error(429) == "rate_limited"
        assert normalize_error(500) == "provider_unavailable"
        assert normalize_error(503) == "provider_unavailable"
        assert normalize_error(400) == "invalid_payload"
        assert normalize_error(403) == "permission_denied_local"


# ── normalize_error — HTTP code mapping ──────────────────────────────────────

class TestNormalizeErrorHttpCodes:
    @pytest.mark.parametrize("code,expected", [
        ("401", "auth_error"),
        ("429", "rate_limited"),
        ("500", "provider_unavailable"),
        ("502", "provider_unavailable"),
        ("503", "provider_unavailable"),
        ("504", "provider_timeout"),
        ("400", "invalid_payload"),
        ("422", "invalid_payload"),
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
        err = ExecutorError("test", "auth_error")
        assert isinstance(err, RouterError)

    def test_codex_auth_error_type(self):
        assert CodexAuthError().error_type == "auth_error"

    def test_codex_quota_error_type(self):
        assert CodexQuotaError().error_type == "quota_exhausted"

    def test_claude_auth_error_type(self):
        assert ClaudeAuthError().error_type == "auth_error"

    def test_claude_quota_error_type(self):
        assert ClaudeQuotaError().error_type == "quota_exhausted"

    def test_rate_limited_error_type(self):
        assert RateLimitedError().error_type == "rate_limited"

    def test_provider_unavailable_error_type(self):
        assert ProviderUnavailableError().error_type == "provider_unavailable"

    def test_provider_timeout_error_type(self):
        assert ProviderTimeoutError().error_type == "provider_timeout"

    def test_transient_network_error_type(self):
        assert TransientNetworkError().error_type == "transient_network_error"

    def test_invalid_payload_error_type(self):
        assert InvalidPayloadError().error_type == "invalid_payload"

    def test_missing_repo_path_error_type(self):
        assert MissingRepoPathError().error_type == "missing_repo_path"

    def test_permission_denied_local_error_type(self):
        assert PermissionDeniedLocalError().error_type == "permission_denied_local"

    def test_git_conflict_error_type(self):
        assert GitConflictError().error_type == "git_conflict"

    def test_toolchain_error_type(self):
        assert ToolchainError().error_type == "toolchain_error"

    def test_template_render_error_type(self):
        assert TemplateRenderError().error_type == "template_render_error"

    def test_unsupported_task_error_type(self):
        assert UnsupportedTaskError().error_type == "unsupported_task"

    def test_codex_tool_error_maps_to_toolchain(self):
        """Legacy CodexToolError now maps to toolchain_error."""
        assert CodexToolError().error_type == "toolchain_error"

    def test_claude_tool_error_maps_to_toolchain(self):
        """Legacy ClaudeToolError now maps to toolchain_error."""
        assert ClaudeToolError().error_type == "toolchain_error"

    def test_openrouter_error_default_type(self):
        assert OpenRouterError().error_type == "provider_unavailable"

    def test_all_executor_error_types_in_normalized_set(self):
        """Every ExecutorError subclass .error_type must be in NORMALIZED_ERROR_TYPES."""
        classes = [
            CodexAuthError, CodexQuotaError, ClaudeAuthError, ClaudeQuotaError,
            RateLimitedError, ProviderUnavailableError, ProviderTimeoutError,
            TransientNetworkError, InvalidPayloadError, MissingRepoPathError,
            PermissionDeniedLocalError, GitConflictError, ToolchainError,
            TemplateRenderError, UnsupportedTaskError, CodexToolError, ClaudeToolError,
        ]
        for cls in classes:
            err = cls()
            assert err.error_type in NORMALIZED_ERROR_TYPES, (
                f"{cls.__name__}.error_type={err.error_type!r} not in NORMALIZED_ERROR_TYPES"
            )
