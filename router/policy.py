"""Routing policy logic based on Codex state and task characteristics."""

import os
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple

from .models import (
    TaskMeta, RouteDecision, ExecutorResult, ChainEntry,
    Executor, ExecutorBackend, ModelProfile,
    CodexState, TaskRisk, TaskClass, TaskModality, TaskPhase
)
from .state_store import StateStore, get_state_store
from .executors import run_codex, run_claude, run_openrouter
from .errors import ExecutorError, ELIGIBLE_FALLBACK_ERRORS, can_fallback, ChainInvariantViolation
from .config_loader import get_model, get_reliability_config
from .attempt_logger import AttemptLogger, ExecutorAttempt, RoutingTrace
from .notifications import NotificationManager
from .circuit_breaker import CircuitBreaker
from .budget_manager import get_budget_manager, reset_budget_manager as _reset_budget_mgr

MAX_ERROR_MESSAGE_LENGTH = 200


# ---------------------------------------------------------------------------
# Intent-aware task helpers
# ---------------------------------------------------------------------------

def is_decision_task(task: TaskMeta) -> bool:
    """Decision tasks need strong reasoning models (planning, triage, final review)."""
    return task.inferred_phase() == TaskPhase.DECIDE


def is_visual_task(task: TaskMeta) -> bool:
    """Visual tasks need multimodal models."""
    return task.inferred_phase() == TaskPhase.VISUAL


def is_heavy_execution_task(task: TaskMeta) -> bool:
    """Hard execution tasks (critical writes, debug, architecture change)."""
    return (
        task.risk == TaskRisk.CRITICAL
        or task.task_class in (TaskClass.DEBUG, TaskClass.REPO_ARCHITECTURE_CHANGE)
    )


def _truncate_error_message(msg: str) -> str:
    """Truncate error messages to prevent code-path leakage in logs."""
    if len(msg) > MAX_ERROR_MESSAGE_LENGTH:
        return msg[:MAX_ERROR_MESSAGE_LENGTH] + "...[TRUNCATED]"
    return msg


# ---------------------------------------------------------------------------
# Circuit breaker singleton
# ---------------------------------------------------------------------------

_breaker: Optional[CircuitBreaker] = None


def get_breaker() -> CircuitBreaker:
    """Get or create the circuit breaker singleton."""
    global _breaker
    if _breaker is None:
        from .config_loader import load_config
        config = load_config()
        cb_config = config.get("reliability", {}).get("circuit_breaker", {})
        _breaker = CircuitBreaker(
            threshold=cb_config.get("threshold", 5),
            window_s=cb_config.get("window_s", 60),
            cooldown_s=cb_config.get("cooldown_s", 120),
        )
    return _breaker


def reset_breaker() -> None:
    """Reset the circuit breaker singleton. Primarily for testing."""
    global _breaker
    _breaker = None


# ---------------------------------------------------------------------------
# Notification manager singleton
# ---------------------------------------------------------------------------

_notifier: Optional[NotificationManager] = None


def get_notifier() -> NotificationManager:
    """Get or create the notification manager singleton."""
    global _notifier
    if _notifier is None:
        _notifier = NotificationManager()
    return _notifier


def reset_notifier() -> None:
    """Reset the notification manager singleton. For testing."""
    global _notifier
    _notifier = None


def reset_budget() -> None:
    """Reset the budget manager singleton. For testing."""
    _reset_budget_mgr()


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def resolve_state(store: Optional[StateStore] = None) -> CodexState:
    """
    Resolve the active Codex state.
    Priority: manual_state > auto_state > default (openai_primary)
    """
    store = store or get_state_store()
    manual = store.get_manual_state()
    if manual is not None:
        return manual
    auto = store.get_auto_state()
    if auto is not None:
        return auto
    return CodexState.OPENAI_PRIMARY


def claude_available() -> bool:
    """Check whether Claude Code is available in PATH."""
    return (
        Path("/usr/local/bin/claude").exists()
        or Path("/usr/bin/claude").exists()
        or Path(os.path.expanduser("~/.local/bin/claude")).exists()
        or os.system("command -v claude > /dev/null 2>&1") == 0
    )


# ---------------------------------------------------------------------------
# OpenRouter profile selection
# ---------------------------------------------------------------------------

def choose_openrouter_profile(task: TaskMeta) -> ModelProfile:
    """
    Choose the OpenRouter model profile based on task characteristics.
    Kimi for multimodal/screenshot, MiMo for hardest tasks, MiniMax default.
    """
    if is_visual_task(task):
        return ModelProfile.OPENROUTER_KIMI
    if is_decision_task(task) or is_heavy_execution_task(task):
        return ModelProfile.OPENROUTER_MIMO
    return ModelProfile.OPENROUTER_MINIMAX


def choose_openai_profile(task: TaskMeta) -> str:
    """
    Select OpenAI model based on task complexity.
    Returns model string from config.

    Heavy (gpt-5.4): decision tasks or heavy execution.
    Light (gpt-5.4-mini): everything else.
    """
    if is_decision_task(task) or is_heavy_execution_task(task):
        return get_model("gpt54")
    return get_model("gpt54_mini")


def choose_claude_model(task: TaskMeta) -> str:
    """Select Claude model based on task complexity. Sonnet default, Opus for hard cases."""
    if is_decision_task(task) or is_heavy_execution_task(task):
        return get_model("opus")
    return get_model("sonnet")


def choose_claude_profile(task: TaskMeta) -> str:
    """Return model_profile string for Claude in chain entry."""
    model = choose_claude_model(task)
    if model == get_model("opus"):
        return ModelProfile.CLAUDE_OPUS.value
    return ModelProfile.CLAUDE_SONNET.value


def _openrouter_model(profile: ModelProfile) -> str:
    if profile == ModelProfile.OPENROUTER_KIMI:
        return get_model("kimi")
    if profile == ModelProfile.OPENROUTER_MIMO:
        return get_model("mimo")
    return get_model("minimax")


# ---------------------------------------------------------------------------
# Chain building — 4 states
# ---------------------------------------------------------------------------

def _build_openai_primary_chain(task: TaskMeta) -> List[ChainEntry]:
    """openai_primary: Codex subscription first, smart model selection."""
    openai_profile = choose_openai_profile(task)
    openrouter_profile = choose_openrouter_profile(task)
    # Determine which model_profile string to use for the codex entry
    if openai_profile == get_model("gpt54"):
        codex_profile = "codex_gpt54"
    else:
        codex_profile = "codex_gpt54_mini"
    return [
        ChainEntry(tool="codex_cli", backend="openai_native", model_profile=codex_profile),
        ChainEntry(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        ChainEntry(tool="codex_cli", backend="openrouter", model_profile=openrouter_profile.value),
    ]


def _build_openai_conservation_chain(task: TaskMeta) -> List[ChainEntry]:
    """openai_conservation: Conserve OpenAI usage. Almost everything → gpt-5.4-mini.

    Spec: "default executor for almost everything → gpt-5.4-mini.
    Only planner / final review / very high-risk tasks → gpt-5.4."

    gpt-5.4 reserved for decision tasks and heavy execution.
    Chain order: Codex → Claude → OpenRouter (Claude is still a prepaid bucket,
    OpenRouter is paid — subscription preservation comes first).
    """
    if is_decision_task(task) or is_heavy_execution_task(task):
        openai_profile = "codex_gpt54"
    else:
        openai_profile = "codex_gpt54_mini"
    return [
        ChainEntry(tool="codex_cli", backend="openai_native", model_profile=openai_profile),
        ChainEntry(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        ChainEntry(tool="codex_cli", backend="openrouter", model_profile=ModelProfile.OPENROUTER_MINIMAX.value),
    ]


def _build_claude_backup_chain(task: TaskMeta) -> List[ChainEntry]:
    """claude_backup: Claude Code subscription, Sonnet default."""
    claude_profile = choose_claude_profile(task)
    return [
        ChainEntry(tool="claude_code", backend="anthropic", model_profile=claude_profile),
        ChainEntry(tool="codex_cli", backend="openrouter", model_profile=ModelProfile.OPENROUTER_DYNAMIC.value),
    ]


def _build_openrouter_fallback_chain(task: TaskMeta) -> List[ChainEntry]:
    """openrouter_fallback: Last resort paid usage."""
    openrouter_profile = choose_openrouter_profile(task)
    return [
        ChainEntry(tool="codex_cli", backend="openrouter", model_profile=openrouter_profile.value),
    ]


# ---------------------------------------------------------------------------
# Backward compat chain aliases
# ---------------------------------------------------------------------------

def _build_normal_chain(task: TaskMeta) -> List[ChainEntry]:
    """Backward compat: normal → openai_primary."""
    return _build_openai_primary_chain(task)


def _build_last10_chain(task: TaskMeta) -> List[ChainEntry]:
    """Backward compat: last10 → claude_backup."""
    return _build_claude_backup_chain(task)


def build_chain(task: TaskMeta, state: CodexState) -> List[ChainEntry]:
    """Build the routing chain for a task given the current state."""
    builders = {
        CodexState.OPENAI_PRIMARY: _build_openai_primary_chain,
        CodexState.OPENAI_CONSERVATION: _build_openai_conservation_chain,
        CodexState.CLAUDE_BACKUP: _build_claude_backup_chain,
        CodexState.OPENROUTER_FALLBACK: _build_openrouter_fallback_chain,
    }
    builder = builders.get(state, _build_openai_primary_chain)
    return builder(task)


# ---------------------------------------------------------------------------
# Chain invariant validation
# ---------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


def validate_chain(state: CodexState, chain: List[ChainEntry]) -> Tuple[bool, str]:
    """
    Validate that a routing chain respects state-specific invariants.

    Returns (valid, reason) where reason explains any violation.
    """
    state_val = state.value if hasattr(state, "value") else str(state)

    if state == CodexState.CLAUDE_BACKUP:
        for entry in chain:
            if entry.backend == "openai_native":
                return False, f"claude_backup chain must not contain openai_native entries (found: {entry.tool}:{entry.backend})"
        if chain and chain[0].backend not in ("anthropic",):
            return False, f"claude_backup chain must start with anthropic backend (got: {chain[0].backend})"

    elif state == CodexState.OPENROUTER_FALLBACK:
        for entry in chain:
            if entry.backend in ("openai_native", "anthropic"):
                return False, f"openrouter_fallback chain must only contain openrouter entries (found: {entry.backend})"

    # openai_primary and openai_conservation: all providers allowed
    return True, "valid"


# ---------------------------------------------------------------------------
# Executor dispatch
# ---------------------------------------------------------------------------

def _run_executor(entry: ChainEntry, task: TaskMeta, trace_id: str = "") -> ExecutorResult:
    """Dispatch to the appropriate executor based on chain entry."""
    if entry.tool == "codex_cli" and entry.backend == "openai_native":
        if entry.model_profile == "codex_gpt54":
            result = run_codex(task, model=get_model("gpt54"))
        elif entry.model_profile == "codex_gpt54_mini":
            result = run_codex(task, model=get_model("gpt54_mini"))
        else:
            result = run_codex(task)
    elif entry.tool == "claude_code" and entry.backend == "anthropic":
        model = None
        if entry.model_profile == "claude_opus":
            model = get_model("opus")
        elif entry.model_profile == "claude_sonnet":
            model = get_model("sonnet")
        result = run_claude(task, model=model)
    elif entry.tool == "codex_cli" and entry.backend == "openrouter":
        model = _openrouter_model(ModelProfile(entry.model_profile))
        result = run_openrouter(task, model=model, profile=entry.model_profile)
    else:
        # Fallback: generic openrouter call
        model = _openrouter_model(ModelProfile.OPENROUTER_MINIMAX)
        result = run_openrouter(task, model=model)

    # Stamp trace_id on the result
    result.trace_id = trace_id

    # Detect partial success: failure but has useful output
    if not result.success:
        if result.final_summary is not None:
            result.partial_success = True
        elif result.artifacts:
            result.partial_success = True

    # Extract warnings from stderr
    result.warnings = _extract_warnings(result.stderr_ref)

    return result


def _extract_warnings(stderr_ref: Optional[str]) -> List[str]:
    """Extract warning lines from a stderr file reference."""
    warnings: List[str] = []
    if not stderr_ref:
        return warnings
    try:
        content = Path(stderr_ref).read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            if "warning" in line.lower() or "WARN" in line:
                warnings.append(line.strip())
    except OSError:
        pass
    return warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def route_task(task: TaskMeta) -> Tuple[RouteDecision, ExecutorResult]:
    """
    Main routing function.

    Resolves state, builds chain, executes chain with fallback,
    and returns (RouteDecision, ExecutorResult).
    """
    from .health import get_shutdown_manager

    shutdown_mgr = get_shutdown_manager()

    if not shutdown_mgr.should_accept_new_tasks():
        return (
            RouteDecision(
                task_id=task.task_id,
                state="shutdown",
                chain=[],
                reason="Shutdown in progress",
            ),
            ExecutorResult(
                task_id=task.task_id,
                tool="router",
                backend="shutdown",
                model_profile="",
                success=False,
                normalized_error="shutdown_in_progress",
                final_summary="Router is shutting down, not accepting new tasks",
            ),
        )

    # Budget check: auto-transition if threshold exceeded
    budget_mgr = get_budget_manager()
    budget_mgr.check_and_transition()

    state = resolve_state()
    chain = build_chain(task, state)

    # Validate chain invariants
    valid, invariant_reason = validate_chain(state, chain)
    if not valid:
        logger.warning("Chain invariant violation for state %s: %s", state.value, invariant_reason)

    trace_id = uuid.uuid4().hex[:12]  # 12-char hex trace ID
    reliability = get_reliability_config()
    chain_timeout_s = reliability.get("chain_timeout_s", 600)
    max_fallbacks = reliability.get("max_fallbacks", 3)

    attempt_logger = AttemptLogger()
    attempt_start = time.monotonic()
    attempts: List = []

    start_time = time.monotonic()

    state_str = state.value
    reason = f"{state_str}: standard chain"
    if state in (CodexState.CLAUDE_BACKUP, CodexState.OPENROUTER_FALLBACK):
        # No OpenAI lane available in these states
        pass  # chain already doesn't include openai_native

    fallback_from = None
    last_error: Optional[str] = None
    result: Optional[ExecutorResult] = None
    providers_skipped: List[str] = []
    breaker = get_breaker()
    fallback_count = 0
    chain_timed_out = False
    is_first_executor = True
    error_history: List[dict] = []

    # Register task as in-flight
    shutdown_mgr.register_task(task.task_id, state_str, chain[0].tool if chain else "none")

    try:
        for entry in chain:
            # Check chain timeout
            elapsed = time.monotonic() - start_time
            if elapsed >= chain_timeout_s:
                chain_timed_out = True
                result = result or ExecutorResult(
                    task_id=task.task_id,
                    tool="chain",
                    backend="timeout",
                    model_profile="",
                    success=False,
                    normalized_error="chain_timeout",
                    trace_id=trace_id,
                )
                break

            # Circuit breaker: skip if provider is open
            if not breaker.is_available(entry.tool, entry.backend):
                providers_skipped.append(f"{entry.tool}:{entry.backend}")
                attempts.append(ExecutorAttempt(
                    tool=entry.tool,
                    backend=entry.backend,
                    model_profile=entry.model_profile,
                    success=False,
                    latency_ms=0,
                    skipped=True,
                    skip_reason="circuit_breaker_open",
                ))
                continue

            last_error = None
            try:
                exec_start = time.monotonic()
                result = _run_executor(entry, task, trace_id=trace_id)
                exec_latency = int((time.monotonic() - exec_start) * 1000)
                attempts.append(ExecutorAttempt(
                    tool=result.tool,
                    backend=result.backend,
                    model_profile=result.model_profile,
                    success=result.success,
                    latency_ms=exec_latency,
                    normalized_error=result.normalized_error,
                    cost_estimate_usd=result.cost_estimate_usd,
                ))
                if result.success:
                    breaker.record_success(entry.tool, entry.backend)
                    break
                if result.partial_success:
                    # Partial success has useful output — don't fallback
                    break
                # Non-success but no exception — check if fallback-eligible
                if result.normalized_error:
                    breaker.record_failure(entry.tool, entry.backend, result.normalized_error)
                if result.normalized_error and can_fallback(result.normalized_error):
                    error_history.append({
                        "tool": entry.tool,
                        "backend": entry.backend,
                        "model_profile": entry.model_profile,
                        "error_type": result.normalized_error,
                        "error_message": _truncate_error_message(str(result.final_summary) or ""),
                    })
                    if fallback_from is None:
                        fallback_from = entry.tool
                    if is_first_executor:
                        is_first_executor = False
                    else:
                        fallback_count += 1
                    if fallback_count >= max_fallbacks:
                        break
                    continue
                else:
                    # Non-fallback-eligible error — record and stop
                    if result.normalized_error:
                        error_history.append({
                            "tool": entry.tool,
                            "backend": entry.backend,
                            "model_profile": entry.model_profile,
                            "error_type": result.normalized_error,
                            "error_message": _truncate_error_message(str(result.final_summary) or ""),
                        })
                    break
            except ExecutorError as e:
                last_error = e.error_type
                breaker.record_failure(entry.tool, entry.backend, e.error_type)
                exec_latency = int((time.monotonic() - exec_start) * 1000)
                attempts.append(ExecutorAttempt(
                    tool=entry.tool,
                    backend=entry.backend,
                    model_profile=entry.model_profile,
                    success=False,
                    latency_ms=exec_latency,
                    normalized_error=e.error_type,
                ))
                result = ExecutorResult(
                    task_id=task.task_id,
                    tool=entry.tool,
                    backend=entry.backend,
                    model_profile=entry.model_profile,
                    success=False,
                    normalized_error=e.error_type,
                    trace_id=trace_id,
                )
                if not can_fallback(e.error_type):
                    error_history.append({
                        "tool": entry.tool,
                        "backend": entry.backend,
                        "model_profile": entry.model_profile,
                        "error_type": e.error_type,
                        "error_message": _truncate_error_message(str(e)),
                    })
                    break
                error_history.append({
                    "tool": entry.tool,
                    "backend": entry.backend,
                    "model_profile": entry.model_profile,
                    "error_type": e.error_type,
                    "error_message": _truncate_error_message(str(e)),
                })
                if fallback_from is None:
                    fallback_from = entry.tool
                if is_first_executor:
                    is_first_executor = False
                else:
                    fallback_count += 1
                if fallback_count >= max_fallbacks:
                    break
            except Exception as e:
                # Catch all non-ExecutorError exceptions (RuntimeError, OSError, etc.)
                from .errors import normalize_error
                norm_err = normalize_error(e)
                last_error = norm_err
                breaker.record_failure(entry.tool, entry.backend, norm_err)
                exec_latency = int((time.monotonic() - exec_start) * 1000)
                attempts.append(ExecutorAttempt(
                    tool=entry.tool,
                    backend=entry.backend,
                    model_profile=entry.model_profile,
                    success=False,
                    latency_ms=exec_latency,
                    normalized_error=norm_err,
                ))
                result = ExecutorResult(
                    task_id=task.task_id,
                    tool=entry.tool,
                    backend=entry.backend,
                    model_profile=entry.model_profile,
                    success=False,
                    normalized_error=norm_err,
                    trace_id=trace_id,
                )
                error_history.append({
                    "tool": entry.tool,
                    "backend": entry.backend,
                    "model_profile": entry.model_profile,
                    "error_type": norm_err,
                    "error_message": _truncate_error_message(str(e)),
                })
                if can_fallback(norm_err):
                    if fallback_from is None:
                        fallback_from = entry.tool
                    if is_first_executor:
                        is_first_executor = False
                    else:
                        fallback_count += 1
                    if fallback_count >= max_fallbacks:
                        break
                    continue
                else:
                    break

        if result is None:
            result = ExecutorResult(
                task_id=task.task_id,
                success=False,
                normalized_error=last_error or "unknown_error",
                trace_id=trace_id,
                error_history=error_history,
            )
        else:
            result.error_history = error_history

        # --- Notifications ---
        notifier = get_notifier()

        # State change notification (if state changed)
        # Fallback rate is checked by periodic monitoring (MetricsCollector), not per-request

        # Conservation duration check (if state_entered_at is tracked)
        notifier.check_conservation_duration(
            state=state_str,
            state_entered_at=None,  # Tracked from state history in future update
        )

        # Write attempt trace
        total_latency = int((time.monotonic() - attempt_start) * 1000)
        trace = RoutingTrace(
            trace_id=trace_id,
            task_id=task.task_id,
            state=state_str,
            chain=[{"tool": c.tool, "backend": c.backend, "model_profile": c.model_profile} for c in chain],
            attempts=attempts,
            providers_skipped=providers_skipped,
            chain_timed_out=chain_timed_out,
            fallback_count=fallback_count,
            total_latency_ms=total_latency,
            final_tool=result.tool if result else "none",
            final_success=result.success if result else False,
            final_error=result.normalized_error if result and not result.success else None,
            chain_invariant_violated=not valid,
            chain_invariant_reason=invariant_reason if not valid else None,
        )
        attempt_logger.log_trace(trace)

        decision = RouteDecision(
            task_id=task.task_id,
            state=state_str,
            chain=chain,
            reason=reason,
            attempted_fallback=fallback_from is not None,
            fallback_from=fallback_from,
            providers_skipped=providers_skipped,
            chain_timed_out=chain_timed_out,
            fallback_count=fallback_count,
            trace_id=trace_id,
            error_history=error_history,
        )

        return decision, result
    finally:
        shutdown_mgr.unregister_task(task.task_id)


# ---------------------------------------------------------------------------
# Reviewer independence
# ---------------------------------------------------------------------------

def get_review_chain(task: TaskMeta, generate_executor_key: str) -> List[ChainEntry]:
    """
    Build review chain excluding the executor that generated the code.
    generate_executor_key is like "codex_cli:openai_native" or "claude_code:anthropic".
    """
    state = resolve_state()
    full_chain = build_chain(task, state)
    gen_tool, gen_backend = generate_executor_key.split(":", 1)
    return [
        entry for entry in full_chain
        if not (entry.tool == gen_tool and entry.backend == gen_backend)
    ]


# ---------------------------------------------------------------------------
# Review modes
# ---------------------------------------------------------------------------

class ReviewMode(str, Enum):
    FAST = "fast"
    DEEP = "deep"


def select_review_mode(task: TaskMeta) -> ReviewMode:
    """Fast by default, deep for high risk or architecture changes."""
    if task.risk == TaskRisk.HIGH or task.task_class == TaskClass.REPO_ARCHITECTURE_CHANGE:
        return ReviewMode.DEEP
    return ReviewMode.FAST


# ---------------------------------------------------------------------------
# Merge gate
# ---------------------------------------------------------------------------

def merge_gate(
    generator_result: ExecutorResult,
    reviewer_result: ExecutorResult,
    task: TaskMeta,
) -> Tuple[bool, List[str]]:
    """
    Check all merge gate conditions. Returns (passed, list_of_failure_reasons).
    """
    reasons: List[str] = []

    # Generator must succeed
    if not generator_result.success:
        reasons.append("Generator result did not succeed")

    # Reviewer must succeed
    if not reviewer_result.success:
        reasons.append("Reviewer result did not succeed")

    # Reviewer must be independent (different executor/backend)
    gen_key = (generator_result.tool, generator_result.backend)
    rev_key = (reviewer_result.tool, reviewer_result.backend)
    if gen_key == rev_key:
        reasons.append("Reviewer is the same as generator — independent review required")

    passed = len(reasons) == 0
    return passed, reasons
