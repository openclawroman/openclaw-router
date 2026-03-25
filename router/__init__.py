"""ai-code-runner - external routing layer for OpenClaw coding tasks."""

from .models import (
    TaskMeta,
    RouteDecision,
    ExecutorResult,
    ChainEntry,
    TaskClass,
    TaskModality,
    TaskRisk,
    CodexState,
    Executor,
    ExecutorBackend,
    ModelProfile,
)
from .policy import (
    route_task,
    resolve_state,
    build_chain,
    choose_openrouter_profile,
    can_fallback,
)
from .state_store import StateStore
from .logger import RoutingLogger
from .classifier import Classifier, classify, classify_from_dict
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
    "ChainEntry",
    "TaskClass",
    "TaskModality",
    "TaskRisk",
    "CodexState",
    "Executor",
    "ExecutorBackend",
    "ModelProfile",
    # policy
    "route_task",
    "resolve_state",
    "build_chain",
    "choose_openrouter_profile",
    "can_fallback",
    # state
    "StateStore",
    # logging
    "RoutingLogger",
    # classifier
    "Classifier",
    "classify",
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
