"""ai-code-runner - external routing layer for OpenClaw coding tasks."""

from .models import (
    TaskMeta,
    RouteDecision,
    ExecutorResult,
    TaskClass,
    TaskModality,
    ExecutorBackend,
    ModelProfile,
    TaskCriticality,
    TaskRisk,
    CodexState,
    Executor,
)
from .policy import route_task
from .state_store import StateStore
from .logger import RoutingLogger
from .classifier import Classifier, classify_from_dict
from .errors import (
    RouterError,
    ExecutorError,
    StateError,
    ConfigurationError,
    normalize_error,
    NORMALIZED_ERROR_TYPES,
)
from .executors import (
    run_codex,
    run_claude,
    run_openrouter,
    run_codex_openrouter_minimax,
    run_codex_openrouter_kimi,
)

__all__ = [
    # models
    "TaskMeta",
    "RouteDecision",
    "ExecutorResult",
    "TaskClass",
    "TaskModality",
    "ExecutorBackend",
    "ModelProfile",
    "TaskCriticality",
    "TaskRisk",
    "CodexState",
    "Executor",
    # policy
    "route_task",
    # state
    "StateStore",
    # logging
    "RoutingLogger",
    # classifier
    "Classifier",
    "classify_from_dict",
    # errors
    "RouterError",
    "ExecutorError",
    "StateError",
    "ConfigurationError",
    "normalize_error",
    "NORMALIZED_ERROR_TYPES",
    # executors
    "run_codex",
    "run_claude",
    "run_openrouter",
    "run_codex_openrouter_minimax",
    "run_codex_openrouter_kimi",
]
