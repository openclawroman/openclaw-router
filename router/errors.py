"""Normalized error classes for executor results."""

import re
from typing import Optional


class RouterError(Exception):
    """Base exception for router errors."""
    pass


class ExecutorError(RouterError):
    """Base class for executor-specific errors."""
    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type


class CodexQuotaError(ExecutorError):
    def __init__(self, message: str = "Codex quota exceeded"):
        super().__init__(message, "quota_error")


class CodexAuthError(ExecutorError):
    def __init__(self, message: str = "Codex authentication failed"):
        super().__init__(message, "auth_error")


class CodexToolError(ExecutorError):
    def __init__(self, message: str = "Codex tool execution failed"):
        super().__init__(message, "tool_error")


class ClaudeQuotaError(ExecutorError):
    def __init__(self, message: str = "Claude quota exceeded"):
        super().__init__(message, "quota_error")


class ClaudeAuthError(ExecutorError):
    def __init__(self, message: str = "Claude authentication failed"):
        super().__init__(message, "auth_error")


class ClaudeToolError(ExecutorError):
    def __init__(self, message: str = "Claude tool execution failed"):
        super().__init__(message, "tool_error")


class OpenRouterError(ExecutorError):
    def __init__(self, message: str = "OpenRouter API error"):
        super().__init__(message, "api_error")


class ConfigurationError(RouterError):
    """Configuration file is missing or invalid."""
    pass


class StateError(RouterError):
    """State file read/write error."""
    pass


# Normalized error types used across the router
NORMALIZED_ERROR_TYPES = [
    "quota_exhausted",
    "auth_error",
    "provider_timeout",
    "provider_unavailable",
    "transient_network_error",
    "rate_limited",
]


def normalize_error(error_message: str) -> str:
    """
    Map a provider error message/string to a normalized error type.

    Mapping rules:
      - "quota", "limit"         -> "quota_exhausted"
      - "rate", "429"            -> "rate_limited"
      - "auth", "unauthorized", "401" -> "auth_error"
      - "timeout", "timed out"   -> "provider_timeout"
      - "unavailable", "503", "connection" -> "provider_unavailable"
      - "network", "connection"  -> "transient_network_error"

    Returns one of NORMALIZED_ERROR_TYPES or "unknown_error".
    """
    msg = error_message.lower()

    # quota / limit -> quota_exhausted
    if "quota" in msg or "limit" in msg:
        return "quota_exhausted"

    # rate / 429 -> rate_limited
    if "rate" in msg or "429" in msg:
        return "rate_limited"

    # auth / unauthorized / 401 -> auth_error
    if "auth" in msg or "unauthorized" in msg or "401" in msg:
        return "auth_error"

    # timeout / timed out -> provider_timeout
    if "timeout" in msg or "timed out" in msg:
        return "provider_timeout"

    # unavailable / 503 / connection -> provider_unavailable
    if "unavailable" in msg or "503" in msg or ("connection" in msg and "refused" not in msg):
        return "provider_unavailable"

    # network / connection (refused) -> transient_network_error
    if "network" in msg or "connection" in msg:
        return "transient_network_error"

    return "unknown_error"
