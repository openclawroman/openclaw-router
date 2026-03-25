"""Rate limit header parsing from API responses."""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RateLimitInfo:
    """Parsed rate limit information from API response headers."""
    requests_limit: Optional[int] = None
    requests_remaining: Optional[int] = None
    requests_reset_seconds: Optional[float] = None
    tokens_limit: Optional[int] = None
    tokens_remaining: Optional[int] = None
    tokens_reset_seconds: Optional[float] = None
    
    # Provider-specific fields
    rpm: Optional[int] = None  # requests per minute
    tpm: Optional[int] = None  # tokens per minute
    daily_limit: Optional[int] = None
    
    @property
    def requests_utilization(self) -> Optional[float]:
        """Fraction of request quota used (0.0-1.0)."""
        if self.requests_limit and self.requests_limit > 0:
            used = self.requests_limit - (self.requests_remaining or 0)
            return used / self.requests_limit
        return None
    
    @property
    def tokens_utilization(self) -> Optional[float]:
        """Fraction of token quota used (0.0-1.0)."""
        if self.tokens_limit and self.tokens_limit > 0:
            used = self.tokens_limit - (self.tokens_remaining or 0)
            return used / self.tokens_limit
        return None
    
    @property
    def is_near_limit(self) -> bool:
        """True if utilization > 80% for either requests or tokens."""
        req_util = self.requests_utilization
        tok_util = self.tokens_utilization
        return (req_util is not None and req_util > 0.8) or \
               (tok_util is not None and tok_util > 0.8)
    
    def to_dict(self) -> dict:
        """Serialize to dict, omitting None values."""
        d = {}
        if self.requests_limit is not None:
            d["requests_limit"] = self.requests_limit
        if self.requests_remaining is not None:
            d["requests_remaining"] = self.requests_remaining
        if self.requests_reset_seconds is not None:
            d["requests_reset_seconds"] = self.requests_reset_seconds
        if self.tokens_limit is not None:
            d["tokens_limit"] = self.tokens_limit
        if self.tokens_remaining is not None:
            d["tokens_remaining"] = self.tokens_remaining
        if self.tokens_reset_seconds is not None:
            d["tokens_reset_seconds"] = self.tokens_reset_seconds
        if self.requests_utilization is not None:
            d["requests_utilization"] = round(self.requests_utilization, 3)
        if self.tokens_utilization is not None:
            d["tokens_utilization"] = round(self.tokens_utilization, 3)
        return d


def parse_openai_rate_limits(headers: Dict[str, str]) -> RateLimitInfo:
    """Parse OpenAI rate limit headers.
    
    OpenAI uses:
    - x-ratelimit-limit-requests
    - x-ratelimit-remaining-requests
    - x-ratelimit-reset-requests
    - x-ratelimit-limit-tokens
    - x-ratelimit-remaining-tokens
    - x-ratelimit-reset-tokens
    """
    info = RateLimitInfo()
    _lower = {k.lower(): v for k, v in headers.items()}
    
    def _get_int(key: str) -> Optional[int]:
        val = _lower.get(key.lower())
        if val:
            try:
                return int(val)
            except ValueError:
                return None
        return None
    
    def _get_float(key: str) -> Optional[float]:
        val = _lower.get(key.lower())
        if val:
            try:
                return float(val.replace("s", ""))
            except ValueError:
                return None
        return None
    
    info.requests_limit = _get_int("x-ratelimit-limit-requests")
    info.requests_remaining = _get_int("x-ratelimit-remaining-requests")
    info.requests_reset_seconds = _get_float("x-ratelimit-reset-requests")
    info.tokens_limit = _get_int("x-ratelimit-limit-tokens")
    info.tokens_remaining = _get_int("x-ratelimit-remaining-tokens")
    info.tokens_reset_seconds = _get_float("x-ratelimit-reset-tokens")
    
    return info


def parse_anthropic_rate_limits(headers: Dict[str, str]) -> RateLimitInfo:
    """Parse Anthropic rate limit headers.
    
    Anthropic uses:
    - anthropic-ratelimit-requests-limit
    - anthropic-ratelimit-requests-remaining
    - anthropic-ratelimit-requests-reset
    - anthropic-ratelimit-tokens-limit
    - anthropic-ratelimit-tokens-remaining
    - anthropic-ratelimit-tokens-reset
    """
    info = RateLimitInfo()
    _lower = {k.lower(): v for k, v in headers.items()}
    
    def _get_int(key: str) -> Optional[int]:
        val = _lower.get(key.lower())
        if val:
            try:
                return int(val)
            except ValueError:
                return None
        return None
    
    def _get_float(key: str) -> Optional[float]:
        val = _lower.get(key.lower())
        if val:
            try:
                return float(val.replace("s", ""))
            except ValueError:
                return None
        return None
    
    info.requests_limit = _get_int("anthropic-ratelimit-requests-limit")
    info.requests_remaining = _get_int("anthropic-ratelimit-requests-remaining")
    info.requests_reset_seconds = _get_float("anthropic-ratelimit-requests-reset")
    info.tokens_limit = _get_int("anthropic-ratelimit-tokens-limit")
    info.tokens_remaining = _get_int("anthropic-ratelimit-tokens-remaining")
    info.tokens_reset_seconds = _get_float("anthropic-ratelimit-tokens-reset")
    
    return info


def parse_openrouter_rate_limits(headers: Dict[str, str]) -> RateLimitInfo:
    """Parse OpenRouter rate limit headers.
    
    OpenRouter uses:
    - x-ratelimit-limit
    - x-ratelimit-remaining
    - x-ratelimit-reset
    """
    info = RateLimitInfo()
    _lower = {k.lower(): v for k, v in headers.items()}
    
    def _get_int(key: str) -> Optional[int]:
        val = _lower.get(key.lower())
        if val:
            try:
                return int(val)
            except ValueError:
                return None
        return None
    
    info.requests_limit = _get_int("x-ratelimit-limit")
    info.requests_remaining = _get_int("x-ratelimit-remaining")
    
    reset_val = _lower.get("x-ratelimit-reset")
    if reset_val:
        try:
            info.requests_reset_seconds = float(reset_val.replace("s", ""))
        except ValueError:
            pass
    
    return info


def parse_rate_limits(backend: str, headers: Dict[str, str]) -> RateLimitInfo:
    """Parse rate limits based on provider backend."""
    if backend == "openai_native":
        return parse_openai_rate_limits(headers)
    elif backend == "anthropic":
        return parse_anthropic_rate_limits(headers)
    elif backend == "openrouter":
        return parse_openrouter_rate_limits(headers)
    else:
        return RateLimitInfo()
