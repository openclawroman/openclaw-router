"""Tests for circuit breaker provider health tracking."""

import time
import pytest
from router.circuit_breaker import CircuitBreaker, ProviderState


class TestCircuitBreakerStates:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.get_state("codex_cli", "openai_native") == "closed"
        assert cb.is_available("codex_cli", "openai_native")

    def test_after_threshold_failures_opens(self):
        cb = CircuitBreaker(threshold=3, window_s=60)
        for _ in range(3):
            cb.record_failure("codex_cli", "openai_native", "rate_limited")
        assert cb.get_state("codex_cli", "openai_native") == "open"
        assert not cb.is_available("codex_cli", "openai_native")

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(threshold=3, window_s=60)
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        cb.record_success("codex_cli", "openai_native")
        assert cb.get_state("codex_cli", "openai_native") == "closed"
        # Need 3 more failures to open
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        assert cb.get_state("codex_cli", "openai_native") == "closed"  # only 2

    def test_window_expiry_resets_count(self):
        cb = CircuitBreaker(threshold=3, window_s=0.01)
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        time.sleep(0.02)  # window expired
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        assert cb.get_state("codex_cli", "openai_native") == "closed"  # count reset


class TestCircuitBreakerCooldown:
    def test_open_becomes_half_open_after_cooldown(self):
        cb = CircuitBreaker(threshold=2, window_s=60, cooldown_s=0.01)
        cb.record_failure("claude_code", "anthropic", "auth_error")
        cb.record_failure("claude_code", "anthropic", "auth_error")
        assert cb.get_state("claude_code", "anthropic") == "open"
        time.sleep(0.02)
        assert cb.is_available("claude_code", "anthropic")  # half_open now

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(threshold=2, window_s=60, cooldown_s=0.01)
        cb.record_failure("claude_code", "anthropic", "auth_error")
        cb.record_failure("claude_code", "anthropic", "auth_error")
        time.sleep(0.02)
        assert cb.is_available("claude_code", "anthropic")  # half_open
        cb.record_failure("claude_code", "anthropic", "auth_error")
        assert cb.get_state("claude_code", "anthropic") == "open"

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(threshold=2, window_s=60, cooldown_s=0.01)
        cb.record_failure("claude_code", "anthropic", "auth_error")
        cb.record_failure("claude_code", "anthropic", "auth_error")
        time.sleep(0.02)
        assert cb.is_available("claude_code", "anthropic")  # half_open
        cb.record_success("claude_code", "anthropic")
        assert cb.get_state("claude_code", "anthropic") == "closed"


class TestCircuitBreakerMultipleProviders:
    def test_independent_per_provider(self):
        cb = CircuitBreaker(threshold=2, window_s=60)
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        assert cb.get_state("codex_cli", "openai_native") == "open"
        # Claude should still be healthy
        assert cb.get_state("claude_code", "anthropic") == "closed"
        assert cb.is_available("claude_code", "anthropic")

    def test_get_stats(self):
        cb = CircuitBreaker(threshold=5, window_s=60)
        cb.record_failure("codex_cli", "openai_native", "rate_limited")
        cb.record_success("claude_code", "anthropic")
        stats = cb.get_stats()
        assert "codex_cli:openai_native" in stats
        assert "claude_code:anthropic" in stats
