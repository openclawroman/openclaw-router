"""Tests for rate limit header parsing."""

import pytest
from router.rate_limits import (
    RateLimitInfo, parse_openai_rate_limits, parse_anthropic_rate_limits,
    parse_openrouter_rate_limits, parse_rate_limits,
)


class TestRateLimitInfo:
    def test_empty(self):
        info = RateLimitInfo()
        assert info.requests_utilization is None
        assert info.tokens_utilization is None
        assert info.is_near_limit is False

    def test_requests_utilization(self):
        info = RateLimitInfo(requests_limit=100, requests_remaining=20)
        assert info.requests_utilization == 0.8

    def test_tokens_utilization(self):
        info = RateLimitInfo(tokens_limit=10000, tokens_remaining=1000)
        assert info.tokens_utilization == 0.9

    def test_is_near_limit_requests(self):
        info = RateLimitInfo(requests_limit=100, requests_remaining=10)
        assert info.is_near_limit is True

    def test_is_near_limit_tokens(self):
        info = RateLimitInfo(tokens_limit=10000, tokens_remaining=1000)
        assert info.is_near_limit is True

    def test_is_not_near_limit(self):
        info = RateLimitInfo(requests_limit=100, requests_remaining=50, tokens_limit=10000, tokens_remaining=5000)
        assert info.is_near_limit is False

    def test_to_dict(self):
        info = RateLimitInfo(requests_limit=100, requests_remaining=20, tokens_limit=10000, tokens_remaining=8000)
        d = info.to_dict()
        assert d["requests_limit"] == 100
        assert d["requests_utilization"] == 0.8
        assert d["tokens_utilization"] == 0.2


class TestOpenAIRateLimits:
    def test_parse_complete(self):
        headers = {
            "x-ratelimit-limit-requests": "5000",
            "x-ratelimit-remaining-requests": "4900",
            "x-ratelimit-reset-requests": "60s",
            "x-ratelimit-limit-tokens": "800000",
            "x-ratelimit-remaining-tokens": "750000",
            "x-ratelimit-reset-tokens": "60s",
        }
        info = parse_openai_rate_limits(headers)
        assert info.requests_limit == 5000
        assert info.requests_remaining == 4900
        assert info.requests_reset_seconds == 60.0
        assert info.tokens_limit == 800000
        assert info.tokens_remaining == 750000

    def test_parse_empty(self):
        info = parse_openai_rate_limits({})
        assert info.requests_limit is None
        assert info.is_near_limit is False

    def test_case_insensitive(self):
        headers = {"X-RateLimit-Limit-Requests": "100"}
        info = parse_openai_rate_limits(headers)
        assert info.requests_limit == 100


class TestAnthropicRateLimits:
    def test_parse_complete(self):
        headers = {
            "anthropic-ratelimit-requests-limit": "1000",
            "anthropic-ratelimit-requests-remaining": "950",
            "anthropic-ratelimit-requests-reset": "60s",
            "anthropic-ratelimit-tokens-limit": "100000",
            "anthropic-ratelimit-tokens-remaining": "90000",
            "anthropic-ratelimit-tokens-reset": "60s",
        }
        info = parse_anthropic_rate_limits(headers)
        assert info.requests_limit == 1000
        assert info.requests_remaining == 950
        assert info.tokens_limit == 100000


class TestOpenRouterRateLimits:
    def test_parse(self):
        headers = {"x-ratelimit-limit": "200", "x-ratelimit-remaining": "180"}
        info = parse_openrouter_rate_limits(headers)
        assert info.requests_limit == 200
        assert info.requests_remaining == 180


class TestParseRateLimits:
    def test_dispatch_openai(self):
        headers = {"x-ratelimit-limit-requests": "100"}
        info = parse_rate_limits("openai_native", headers)
        assert info.requests_limit == 100

    def test_dispatch_anthropic(self):
        headers = {"anthropic-ratelimit-requests-limit": "200"}
        info = parse_rate_limits("anthropic", headers)
        assert info.requests_limit == 200

    def test_dispatch_unknown(self):
        info = parse_rate_limits("unknown_backend", {"anything": "123"})
        assert info.requests_limit is None
