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


# ── Eligible fallback errors ─────────────────────────────────────────────────

class CodexAuthError(ExecutorError):
    def __init__(self, message: str = "Codex authentication failed"):
        super().__init__(message, "auth_error")


class CodexQuotaError(ExecutorError):
    def __init__(self, message: str = "Codex quota exceeded"):
        super().__init__(message, "quota_exhausted")


class ClaudeAuthError(ExecutorError):
    def __init__(self, message: str = "Claude authentication failed"):
        super().__init__(message, "auth_error")


class ClaudeQuotaError(ExecutorError):
    def __init__(self, message: str = "Claude quota exceeded"):
        super().__init__(message, "quota_exhausted")


class RateLimitedError(ExecutorError):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "rate_limited")


class ProviderUnavailableError(ExecutorError):
    def __init__(self, message: str = "Provider unavailable"):
        super().__init__(message, "provider_unavailable")


class ProviderTimeoutError(ExecutorError):
    def __init__(self, message: str = "Provider timed out"):
        super().__init__(message, "provider_timeout")


class TransientNetworkError(ExecutorError):
    def __init__(self, message: str = "Transient network error"):
        super().__init__(message, "transient_network_error")


class OpenRouterError(ExecutorError):
    """Generic OpenRouter API error — subclasses set the specific type."""
    def __init__(self, message: str = "OpenRouter API error", error_type: str = "provider_unavailable"):
        super().__init__(message, error_type)


# ── Legacy aliases (kept for backward compat in executors.py) ────────────────

class CodexToolError(ExecutorError):
    """Generic Codex tool execution failure → toolchain_error."""
    def __init__(self, message: str = "Codex tool execution failed"):
        super().__init__(message, "toolchain_error")


class ClaudeToolError(ExecutorError):
    """Generic Claude tool execution failure → toolchain_error."""
    def __init__(self, message: str = "Claude tool execution failed"):
        super().__init__(message, "toolchain_error")


# ── Non-eligible errors (fail immediately) ───────────────────────────────────

class InvalidPayloadError(ExecutorError):
    def __init__(self, message: str = "Invalid payload"):
        super().__init__(message, "invalid_payload")


class MissingRepoPathError(ExecutorError):
    def __init__(self, message: str = "Missing repo path"):
        super().__init__(message, "missing_repo_path")


class PermissionDeniedLocalError(ExecutorError):
    def __init__(self, message: str = "Local permission denied"):
        super().__init__(message, "permission_denied_local")


class GitConflictError(ExecutorError):
    def __init__(self, message: str = "Git conflict detected"):
        super().__init__(message, "git_conflict")


class ToolchainError(ExecutorError):
    def __init__(self, message: str = "Toolchain execution failed"):
        super().__init__(message, "toolchain_error")


class TemplateRenderError(ExecutorError):
    def __init__(self, message: str = "Template render failed"):
        super().__init__(message, "template_render_error")


class UnsupportedTaskError(ExecutorError):
    def __init__(self, message: str = "Unsupported task type"):
        super().__init__(message, "unsupported_task")


# ── Non-executor router errors ───────────────────────────────────────────────

class ConfigurationError(RouterError):
    """Configuration file is missing or invalid."""
    pass


class StateError(RouterError):
    """State file read/write error."""
    pass


class ChainInvariantViolation(RouterError):
    """Routing chain violates state-specific invariants."""
    def __init__(self, state: str, reason: str):
        self.state = state
        self.reason = reason
        super().__init__(f"Chain invariant violation [{state}]: {reason}")


# ── Normalized error type sets ───────────────────────────────────────────────

ELIGIBLE_FALLBACK_ERRORS = {
    "auth_error",
    "rate_limited",
    "quota_exhausted",
    "provider_unavailable",
    "provider_timeout",
    "transient_network_error",
}

NON_ELIGIBLE_ERRORS = {
    "invalid_payload",
    "missing_repo_path",
    "permission_denied_local",
    "git_conflict",
    "toolchain_error",
    "template_render_error",
    "unsupported_task",
}

NORMALIZED_ERROR_TYPES = ELIGIBLE_FALLBACK_ERRORS | NON_ELIGIBLE_ERRORS


def can_fallback(error_type: str) -> bool:
    """Return True if the given error type is eligible for automatic fallback."""
    return error_type in ELIGIBLE_FALLBACK_ERRORS


# ── normalize_error ──────────────────────────────────────────────────────────

# Keyword → error type mapping (order matters: first match wins)
_ERROR_PATTERNS: list[tuple[str, str]] = [
    # HTTP status codes
    (r"\b401\b", "auth_error"),
    (r"\b429\b", "rate_limited"),
    (r"\b403\b", "permission_denied_local"),
    (r"\b500\b", "provider_unavailable"),
    (r"\b502\b", "provider_unavailable"),
    (r"\b503\b", "provider_unavailable"),
    (r"\b504\b", "provider_timeout"),
    (r"\b400\b", "invalid_payload"),
    (r"\b422\b", "invalid_payload"),
    # Auth
    (r"auth", "auth_error"),
    (r"unauthorized", "auth_error"),
    (r"forbidden", "permission_denied_local"),
    # Rate / quota
    (r"rate.?limit", "rate_limited"),
    (r"too.?many.?request", "rate_limited"),
    (r"throttl", "rate_limited"),
    (r"quota", "quota_exhausted"),
    (r"exceed", "quota_exhausted"),
    (r"\blimit\b", "quota_exhausted"),
    # Timeout
    (r"timeout", "provider_timeout"),
    (r"timed?\s*out", "provider_timeout"),
    # Provider availability
    (r"unavailable", "provider_unavailable"),
    (r"service.?down", "provider_unavailable"),
    (r"maintenance", "provider_unavailable"),
    # Network
    (r"network", "transient_network_error"),
    (r"connection.?refus", "transient_network_error"),
    (r"dns", "transient_network_error"),
    (r"econnrefused", "transient_network_error"),
    (r"econnreset", "transient_network_error"),
    (r"etimedout", "transient_network_error"),
    # Local errors
    (r"permission.?denied", "permission_denied_local"),
    (r"access.?denied", "permission_denied_local"),
    (r"git.?conflict", "git_conflict"),
    (r"merge.?conflict", "git_conflict"),
    (r"invalid.?payload", "invalid_payload"),
    (r"malformed", "invalid_payload"),
    (r"missing.?repo", "missing_repo_path"),
    (r"no.?repo.?path", "missing_repo_path"),
    (r"toolchain", "toolchain_error"),
    (r"tool.?error", "toolchain_error"),
    (r"template", "template_render_error"),
    (r"render.?error", "template_render_error"),
    (r"unsupported.?task", "unsupported_task"),
    (r"unknown.?task.?class", "unsupported_task"),
]


def normalize_error(raw) -> str:
    """
    Map a raw error (Exception, string, or error message) to a normalized error type string.

    Accepts:
      - str: raw error message / stack trace
      - Exception: uses str(exc)
      - int: HTTP status code

    Returns one of NORMALIZED_ERROR_TYPES or "unknown_error".
    """
    if isinstance(raw, int):
        raw = str(raw)

    if isinstance(raw, Exception):
        # If it's already an ExecutorError, return its type directly
        if isinstance(raw, ExecutorError):
            return raw.error_type
        raw = str(raw)

    if not isinstance(raw, str):
        raw = str(raw)

    msg = raw.lower().strip()

    # Direct match against known error types
    if msg in NORMALIZED_ERROR_TYPES:
        return msg

    # Pattern matching
    for pattern, error_type in _ERROR_PATTERNS:
        if re.search(pattern, msg):
            return error_type

    return "unknown_error"
