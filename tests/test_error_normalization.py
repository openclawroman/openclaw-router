"""Tests for normalize_error() cross-CLI error normalization.

Verifies that errors from ALL 3 CLI tools (Codex, Claude, OpenRouter)
map to the SAME normalized type regardless of which CLI produced the error.
"""

import pytest

from router.errors import (
    normalize_error,
    NORMALIZED_ERROR_TYPES,
    ELIGIBLE_FALLBACK_ERRORS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Auth errors → auth_error
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthErrorNormalization:
    """All three CLIs produce different auth error formats — all must normalize to auth_error."""

    @pytest.mark.parametrize("raw,cli", [
        # Codex
        ("ERROR: OpenAI API key invalid", "codex"),
        ("401 Unauthorized", "codex"),
        # Claude
        ("claude: please run `claude auth` first", "claude"),
        ("AuthenticationError", "claude"),
        # OpenRouter
        ("401 Invalid API key", "openrouter"),
        ("AUTHENTICATION_REQUIRED", "openrouter"),
    ])
    def test_auth_errors_normalize_to_auth_error(self, raw, cli):
        assert normalize_error(raw) == "auth_error", (
            f"{cli} error {raw!r} did not normalize to auth_error"
        )

    def test_all_auth_clis_same_type(self):
        """Codex, Claude, and OpenRouter auth errors must resolve to identical type."""
        codex = normalize_error("ERROR: OpenAI API key invalid")
        claude = normalize_error("claude: please run `claude auth` first")
        openrouter = normalize_error("401 Invalid API key")
        assert codex == claude == openrouter == "auth_error"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Rate limits → rate_limited
# ═══════════════════════════════════════════════════════════════════════════════

class TestRateLimitNormalization:
    """Rate limit errors from all CLIs must normalize to rate_limited."""

    @pytest.mark.parametrize("raw,cli", [
        # Codex
        ("429 Too Many Requests", "codex"),
        ("rate_limit exceeded", "codex"),
        # Claude
        ("429 rate limit exceeded", "claude"),
        ("too many requests", "claude"),
        # OpenRouter
        ("429 Rate Limited", "openrouter"),
        ("throttled", "openrouter"),
    ])
    def test_rate_limits_normalize_to_rate_limited(self, raw, cli):
        assert normalize_error(raw) == "rate_limited", (
            f"{cli} error {raw!r} did not normalize to rate_limited"
        )

    def test_all_rate_limit_clis_same_type(self):
        codex = normalize_error("429 Too Many Requests")
        claude = normalize_error("429 rate limit exceeded")
        openrouter = normalize_error("throttled")
        assert codex == claude == openrouter == "rate_limited"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Quota errors → quota_exhausted
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuotaExhaustedNormalization:
    """Quota/billing errors from all CLIs must normalize to quota_exhausted."""

    @pytest.mark.parametrize("raw,cli", [
        # Codex
        ("Quota exceeded", "codex"),
        ("You have exceeded your monthly limit", "codex"),
        # Claude
        ("billing limit exceeded", "claude"),
        ("No remaining credits", "claude"),
        # OpenRouter
        ("402 Insufficient credits", "openrouter"),
        ("credits exhausted", "openrouter"),
    ])
    def test_quota_errors_normalize_to_quota_exhausted(self, raw, cli):
        assert normalize_error(raw) == "quota_exhausted", (
            f"{cli} error {raw!r} did not normalize to quota_exhausted"
        )

    def test_all_quota_clis_same_type(self):
        codex = normalize_error("Quota exceeded")
        claude = normalize_error("No remaining credits")
        openrouter = normalize_error("402 Insufficient credits")
        assert codex == claude == openrouter == "quota_exhausted"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Timeout errors → provider_timeout
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeoutNormalization:
    """Timeout errors from all CLIs must normalize to provider_timeout."""

    @pytest.mark.parametrize("raw,cli", [
        # Codex
        ("request timed out", "codex"),
        ("504 Gateway Timeout", "codex"),
        # Claude
        ("connection timed out", "claude"),
        ("TIMEOUT", "claude"),
        # OpenRouter
        ("upstream timeout", "openrouter"),
        ("504", "openrouter"),
    ])
    def test_timeouts_normalize_to_provider_timeout(self, raw, cli):
        assert normalize_error(raw) == "provider_timeout", (
            f"{cli} error {raw!r} did not normalize to provider_timeout"
        )

    def test_all_timeout_clis_same_type(self):
        codex = normalize_error("request timed out")
        claude = normalize_error("TIMEOUT")
        openrouter = normalize_error("upstream timeout")
        assert codex == claude == openrouter == "provider_timeout"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Provider unavailable → provider_unavailable
# ═══════════════════════════════════════════════════════════════════════════════

class TestProviderUnavailableNormalization:
    """Provider down/overloaded errors from all CLIs must normalize to provider_unavailable."""

    @pytest.mark.parametrize("raw,cli", [
        # Codex
        ("503 Service Unavailable", "codex"),
        ("502 Bad Gateway", "codex"),
        # Claude
        ("server error", "claude"),
        ("overloaded", "claude"),
        # OpenRouter
        ("500 Internal Server Error", "openrouter"),
        ("upstream unavailable", "openrouter"),
    ])
    def test_unavailable_errors_normalize_to_provider_unavailable(self, raw, cli):
        assert normalize_error(raw) == "provider_unavailable", (
            f"{cli} error {raw!r} did not normalize to provider_unavailable"
        )

    def test_all_unavailable_clis_same_type(self):
        codex = normalize_error("503 Service Unavailable")
        claude = normalize_error("overloaded")
        openrouter = normalize_error("500 Internal Server Error")
        assert codex == claude == openrouter == "provider_unavailable"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Network errors → transient_network_error
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetworkErrorNormalization:
    """Network-level errors (not CLI-specific) must normalize to transient_network_error."""

    @pytest.mark.parametrize("raw", [
        "ECONNREFUSED",
        "ENOTFOUND",
        "dns resolution failed",
        "connection reset",
    ])
    def test_network_errors_normalize_to_transient_network_error(self, raw):
        assert normalize_error(raw) == "transient_network_error", (
            f"Network error {raw!r} did not normalize to transient_network_error"
        )

    def test_case_insensitive_network_errors(self):
        assert normalize_error("econnrefused") == "transient_network_error"
        assert normalize_error("ECONNREFUSED") == "transient_network_error"
        assert normalize_error("Connection Reset") == "transient_network_error"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: empty input, None, integer HTTP codes."""

    def test_empty_string_returns_unknown_error(self):
        assert normalize_error("") == "unknown_error"

    def test_none_returns_unknown_error(self):
        """None must be handled gracefully without raising."""
        assert normalize_error(None) == "unknown_error"

    def test_whitespace_only_returns_unknown_error(self):
        assert normalize_error("   ") == "unknown_error"

    # Integer HTTP status codes
    @pytest.mark.parametrize("code,expected", [
        (401, "auth_error"),
        (429, "rate_limited"),
        (500, "provider_unavailable"),
        (502, "provider_unavailable"),
        (503, "provider_unavailable"),
        (504, "provider_timeout"),
        (403, "permission_denied_local"),
        (400, "invalid_payload"),
    ])
    def test_integer_http_status_codes(self, code, expected):
        assert normalize_error(code) == expected, (
            f"HTTP {code} did not normalize to {expected}"
        )

    def test_unknown_garbage_returns_unknown_error(self):
        assert normalize_error("xyzzy plugh") == "unknown_error"

    def test_all_normalized_types_are_valid(self):
        """Every non-unknown return value must be in NORMALIZED_ERROR_TYPES."""
        test_inputs = [
            "auth error", "rate limit", "quota", "timeout",
            "unavailable", "network", "invalid payload",
        ]
        for raw in test_inputs:
            result = normalize_error(raw)
            assert result in NORMALIZED_ERROR_TYPES or result == "unknown_error", (
                f"normalize_error({raw!r}) returned {result!r} which is not a valid type"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-CLI consistency: the core invariant
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossCLIConsistency:
    """
    The core invariant: for every error category, errors from different CLIs
    MUST normalize to the same type. This is what makes fallback routing work.
    """

    @pytest.mark.parametrize("category,codex_err,claude_err,openrouter_err,expected", [
        ("auth",
         "ERROR: OpenAI API key invalid",
         "claude: please run `claude auth` first",
         "AUTHENTICATION_REQUIRED",
         "auth_error"),
        ("rate_limit",
         "429 Too Many Requests",
         "too many requests",
         "throttled",
         "rate_limited"),
        ("quota",
         "Quota exceeded",
         "billing limit exceeded",
         "credits exhausted",
         "quota_exhausted"),
        ("timeout",
         "request timed out",
         "connection timed out",
         "upstream timeout",
         "provider_timeout"),
        ("unavailable",
         "503 Service Unavailable",
         "overloaded",
         "upstream unavailable",
         "provider_unavailable"),
    ])
    def test_all_clis_normalize_to_same_type(
        self, category, codex_err, claude_err, openrouter_err, expected
    ):
        """Errors from Codex, Claude, and OpenRouter must converge to the same normalized type."""
        codex_result = normalize_error(codex_err)
        claude_result = normalize_error(claude_err)
        openrouter_result = normalize_error(openrouter_err)

        assert codex_result == expected, f"Codex {category}: {codex_err!r} → {codex_result!r} (expected {expected})"
        assert claude_result == expected, f"Claude {category}: {claude_err!r} → {claude_result!r} (expected {expected})"
        assert openrouter_result == expected, f"OpenRouter {category}: {openrouter_err!r} → {openrouter_result!r} (expected {expected})"

        # The money check: all three are identical
        assert codex_result == claude_result == openrouter_result
