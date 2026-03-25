"""Routing policy logic based on Codex state and task characteristics."""

from .models import (
    TaskMeta, RouteDecision, Executor,
    CodexState, TaskRisk, TaskCriticality
)
from .state_store import StateStore
from .executors import run_codex, run_claude, run_openrouter
from .errors import ExecutorError


# Default models for OpenRouter
OPENROUTER_LARGE_CONTEXT = "moonshotai/kimi-k2.5"
OPENROUTER_CHEAP = "minimax/minimax-m2.5"


def route_task(task: TaskMeta) -> RouteDecision:
    """
    Main routing function.
    
    Applies policy based on current Codex state and task characteristics.
    """
    state_store = StateStore()
    state = state_store.get_state()
    
    # Route based on state
    if state == CodexState.INCLUDED_FIRST:
        return _route_included_first(task)
    elif state == CodexState.INCLUDED_STRETCH:
        return _route_included_stretch(task)
    elif state == CodexState.LAST10:
        return _route_last10(task)
    elif state == CodexState.EXHAUSTED:
        return _route_exhausted(task)
    else:
        # Default fallback
        return _route_included_first(task)


def _route_included_first(task: TaskMeta) -> RouteDecision:
    """Route in included_first state - default to Codex."""
    try:
        return run_codex(task)
    except ExecutorError as e:
        # Fallback not implemented for included_first in v0
        return RouteDecision(
            executor=Executor.CODEX_CLI,
            model="codex-default",
            reason=f"included_first: {e.error_type}",
            attempted_fallback=False,
            status="error",
            error_type=e.error_type
        )


def _route_included_stretch(task: TaskMeta) -> RouteDecision:
    """Route in included_stretch state - still primarily Codex."""
    try:
        return run_codex(task)
    except ExecutorError as e:
        # Fallback not implemented for included_stretch in v0
        return RouteDecision(
            executor=Executor.CODEX_CLI,
            model="codex-default",
            reason=f"included_stretch: {e.error_type}",
            attempted_fallback=False,
            status="error",
            error_type=e.error_type
        )


def _route_last10(task: TaskMeta) -> RouteDecision:
    """Route in last10 state - reserve Codex for critical work."""
    
    # Critical or high-risk tasks stay on Codex
    if task.criticality == TaskCriticality.CRITICAL or task.risk == TaskRisk.HIGH:
        try:
            return run_codex(task)
        except ExecutorError as e:
            # Fallback to Claude on critical task failure
            try:
                decision = run_claude(task)
                decision.attempted_fallback = True
                decision.fallback_from = Executor.CODEX_CLI.value
                decision.reason = f"last10: critical task fallback from Codex: {e.error_type}"
                return decision
            except ExecutorError as e2:
                # Final fallback to OpenRouter
                decision = run_openrouter(task, OPENROUTER_LARGE_CONTEXT)
                decision.attempted_fallback = True
                decision.fallback_from = Executor.CLAUDE_CODE.value
                decision.reason = f"last10: final fallback from Claude: {e2.error_type}"
                return decision
    
    # Normal tasks - use Claude Code as overflow
    try:
        return run_claude(task)
    except ExecutorError as e:
        # Fallback to OpenRouter Kimi
        try:
            decision = run_openrouter(task, OPENROUTER_LARGE_CONTEXT)
            decision.attempted_fallback = True
            decision.fallback_from = Executor.CLAUDE_CODE.value
            decision.reason = f"last10: noncritical fallback from Claude: {e.error_type}"
            return decision
        except ExecutorError as e2:
            return RouteDecision(
                executor=Executor.OPENROUTER,
                model=OPENROUTER_LARGE_CONTEXT,
                reason=f"last10: all executors failed: {e2.error_type}",
                attempted_fallback=True,
                fallback_from=Executor.CLAUDE_CODE.value,
                status="error",
                error_type=e2.error_type
            )


def _route_exhausted(task: TaskMeta) -> RouteDecision:
    """Route in exhausted state - Claude Code primary, OpenRouter fallback."""
    
    # Primary: Claude Code
    try:
        return run_claude(task)
    except ExecutorError as e:
        # Fallback: OpenRouter Kimi for coding, MiniMax for support
        model = OPENROUTER_LARGE_CONTEXT
        
        try:
            decision = run_openrouter(task, model)
            decision.attempted_fallback = True
            decision.fallback_from = Executor.CLAUDE_CODE.value
            decision.reason = f"exhausted: fallback from Claude: {e.error_type}"
            return decision
        except ExecutorError as e2:
            return RouteDecision(
                executor=Executor.OPENROUTER,
                model=model,
                reason=f"exhausted: all executors failed: {e2.error_type}",
                attempted_fallback=True,
                fallback_from=Executor.CLAUDE_CODE.value,
                status="error",
                error_type=e2.error_type
            )