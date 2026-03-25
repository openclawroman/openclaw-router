"""Routing policy logic based on Codex state and task characteristics."""

import os
from pathlib import Path
from typing import Optional

from .models import (
    TaskMeta, RouteDecision, Executor, ExecutorBackend, ModelProfile,
    CodexState, TaskRisk, TaskCriticality, TaskClass, TaskModality
)
from .state_store import StateStore
from .executors import run_codex, run_claude, run_openrouter
from .errors import ExecutorError


# ---------------------------------------------------------------------------
# Helper: which backend/model-profile does each executor use?
# ---------------------------------------------------------------------------

def _executor_backend(executor: Executor) -> ExecutorBackend:
    if executor == Executor.CODEX_CLI:
        return ExecutorBackend.OPENAI_NATIVE
    elif executor == Executor.CLAUDE_CODE:
        return ExecutorBackend.ANTHROPIC
    else:
        return ExecutorBackend.OPENROUTER


def _executor_model_profile(executor: Executor, openrouter_profile: ModelProfile) -> ModelProfile:
    if executor == Executor.CODEX_CLI:
        return ModelProfile.CODEX_PRIMARY
    elif executor == Executor.CLAUDE_CODE:
        return ModelProfile.CLAUDE_PRIMARY
    else:
        return openrouter_profile


# ---------------------------------------------------------------------------
# Configurable model strings (can be overridden via config file)
# ---------------------------------------------------------------------------

OPENROUTER_MINIMAX_MODEL = "minimax/minimax-m2.5"
OPENROUTER_KIMI_MODEL = "moonshotai/kimi-k2.5"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def resolve_state() -> CodexState:
    """
    Resolve the active Codex state.

    Priority: manual_state > auto_state > default (normal)
    """
    store = StateStore()
    # manual_state takes precedence if set
    manual = store.get_manual_state()
    if manual is not None:
        return manual
    # fall back to auto-computed state
    auto = store.get_auto_state()
    if auto is not None:
        return auto
    return CodexState.NORMAL


def claude_available() -> bool:
    """
    Check whether Claude Code is available in PATH.
    """
    return Path("/usr/local/bin/claude").exists() or Path("/usr/bin/claude").exists() or \
        Path(os.path.expanduser("~/.local/bin/claude")).exists() or \
        os.system("command -v claude > /dev/null 2>&1") == 0


# ---------------------------------------------------------------------------
# OpenRouter profile selection
# ---------------------------------------------------------------------------

def choose_openrouter_profile(task: TaskMeta) -> ModelProfile:
    """
    Choose the OpenRouter model profile based on task class.

    use_kimi_for = [ui_from_screenshot, multimodal_code_task, swarm_code_task]
    Otherwise default to minimax.
    """
    kimi_task_classes = {
        TaskClass.UI_FROM_SCREENSHOT,
        TaskClass.MULTIMODAL_CODE_TASK,
        TaskClass.SWARM_CODE_TASK,
    }
    if task.task_class in kimi_task_classes:
        return ModelProfile.OPENROUTER_KIMI
    return ModelProfile.OPENROUTER_MINIMAX


def _openrouter_model(profile: ModelProfile) -> str:
    if profile == ModelProfile.OPENROUTER_KIMI:
        return OPENROUTER_KIMI_MODEL
    return OPENROUTER_MINIMAX_MODEL


# ---------------------------------------------------------------------------
# Fallback eligibility
# ---------------------------------------------------------------------------

# Errors that are eligible for automatic fallback
ELIGIBLE_FALLBACK_ERRORS = {
    "auth_error",
    "rate_limited",
    "quota_exhausted",
    "provider_unavailable",
    "provider_timeout",
    "transient_network_error",
}


def can_fallback(error_type: Optional[str]) -> bool:
    """Return True if the given error type is eligible for fallback."""
    if error_type is None:
        return False
    return error_type in ELIGIBLE_FALLBACK_ERRORS


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def _build_route(
    task: TaskMeta,
    state: CodexState,
    manual_override: bool = False,
) -> RouteDecision:
    """
    Build a RouteDecision by selecting the appropriate chain
    based on the current state.
    """
    task_id = task.task_id or ""

    if state == CodexState.NORMAL:
        # chain = [codex_cli:openai_native, claude_code:anthropic, codex_cli:openrouter]
        chain = [
            (Executor.CODEX_CLI, ExecutorBackend.OPENAI_NATIVE, ModelProfile.CODEX_PRIMARY),
            (Executor.CLAUDE_CODE, ExecutorBackend.ANTHROPIC, ModelProfile.CLAUDE_PRIMARY),
            (Executor.CODEX_CLI, ExecutorBackend.OPENROUTER, ModelProfile.CODEX_PRIMARY),
        ]
        reason = "normal: standard chain"
    else:  # LAST10
        # chain = [claude_code:anthropic, codex_cli:openrouter_dynamic]
        openrouter_profile = choose_openrouter_profile(task)
        chain = [
            (Executor.CLAUDE_CODE, ExecutorBackend.ANTHROPIC, ModelProfile.CLAUDE_PRIMARY),
            (Executor.CODEX_CLI, ExecutorBackend.OPENROUTER, openrouter_profile),
        ]
        reason = f"last10: using {'kimi' if openrouter_profile == ModelProfile.OPENROUTER_KIMI else 'minimax'} for openrouter"

    # Execute the chain in order, stopping on first success or non-fallback-eligible error
    fallback_from = None
    last_error: Optional[str] = None

    for executor, backend, model_profile in chain:
        model_str = _openrouter_model(model_profile) if backend == ExecutorBackend.OPENROUTER else ""
        last_error = None

        try:
            if executor == Executor.CODEX_CLI:
                if backend == ExecutorBackend.OPENAI_NATIVE:
                    decision = run_codex(task)
                else:
                    decision = run_openrouter(task, _openrouter_model(model_profile))
            elif executor == Executor.CLAUDE_CODE:
                decision = run_claude(task)
            else:
                decision = run_openrouter(task, _openrouter_model(model_profile))

            # Success - populate full fields and return
            decision.task_id = task_id
            decision.chain = [e.value for e, _, _ in chain]
            decision.backend = backend
            decision.model_profile = model_profile
            decision.reason = reason if not decision.reason else decision.reason
            if fallback_from:
                decision.attempted_fallback = True
                decision.fallback_from = fallback_from
            return decision

        except ExecutorError as e:
            last_error = e.error_type
            if not can_fallback(e.error_type):
                # Non-fallback-eligible error: stop here
                return RouteDecision(
                    task_id=task_id,
                    executor=executor,
                    model=model_str,
                    chain=[e.value for e, _, _ in chain],
                    reason=f"{reason}: {e.error_type} (no fallback)",
                    status="error",
                    error_type=e.error_type,
                    backend=backend,
                    model_profile=model_profile,
                    normalized_error=e.error_type,
                )
            # Record that we failed and will try next in chain
            if fallback_from is None:
                fallback_from = executor.value

    # All executors in chain failed with fallback-eligible errors
    final_executor, final_backend, final_profile = chain[-1]
    return RouteDecision(
        task_id=task_id,
        executor=final_executor,
        model=_openrouter_model(final_profile),
        chain=[e.value for e, _, _ in chain],
        reason=f"{reason}: all executors failed",
        status="error",
        error_type=last_error,
        backend=final_backend,
        model_profile=final_profile,
        normalized_error=last_error,
        attempted_fallback=True,
        fallback_from=fallback_from,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def route_task(task: TaskMeta) -> RouteDecision:
    """
    Main routing function.

    Applies policy based on resolved Codex state (manual > auto > default)
    and task characteristics to select an executor chain.
    """
    state = resolve_state()
    return _build_route(task, state)
