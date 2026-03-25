"""Normalized error classes for executor results."""


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