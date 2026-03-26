"""Tests for model selection by task phase across all states.

Verifies that decision tasks (PLANNER, FINAL_REVIEW) always get the
strongest model in each state, regardless of risk level.
"""

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    CodexState, ModelProfile,
)
from router.policy import (
    build_chain,
    choose_openrouter_profile,
    choose_openai_profile,
    choose_claude_model,
    select_review_mode,
    ReviewMode,
    route_task,
    resolve_state,
    reset_breaker,
    reset_notifier,
)
from router.state_store import StateStore, reset_state_store
from router.config_loader import get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, **kwargs):
    return TaskMeta(
        task_id="test-001",
        agent="coder",
        task_class=task_class,
        risk=risk,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Conservation state — gpt-5.4 for DECIDE, gpt-5.4-mini for EXECUTE
# ---------------------------------------------------------------------------

class TestConservationPhaseModelSelection:
    """openai_conservation: DECIDE → gpt-5.4, else → gpt-5.4-mini."""

    def test_conservation_planner_gets_gpt54(self):
        """PLANNER (DECIDE phase) must get gpt-5.4 in conservation mode."""
        task = _make_task(task_class=TaskClass.PLANNER)
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        first = chain[0]
        assert first.model_profile == "codex_gpt54"

    def test_conservation_final_review_gets_gpt54(self):
        """FINAL_REVIEW (DECIDE phase) must get gpt-5.4 in conservation mode."""
        task = _make_task(task_class=TaskClass.FINAL_REVIEW)
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        first = chain[0]
        assert first.model_profile == "codex_gpt54"

    def test_conservation_implementation_gets_mini(self):
        """IMPLEMENTATION (EXECUTE phase) should get gpt-5.4-mini in conservation mode."""
        task = _make_task(task_class=TaskClass.IMPLEMENTATION)
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        first = chain[0]
        assert first.model_profile == "codex_gpt54_mini"

    def test_conservation_low_risk_planner_still_gpt54(self):
        """PLANNER with LOW risk must still get gpt-5.4 (decision tasks always get strong model)."""
        task = _make_task(task_class=TaskClass.PLANNER, risk=TaskRisk.LOW)
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        first = chain[0]
        assert first.model_profile == "codex_gpt54"


# ---------------------------------------------------------------------------
# Claude backup — Opus for DECIDE, Sonnet for EXECUTE
# ---------------------------------------------------------------------------

class TestClaudeBackupPhaseModelSelection:
    """claude_backup: DECIDE → Opus, else → Sonnet."""

    def test_claude_backup_planner_gets_opus(self):
        """PLANNER must get Opus in claude_backup state."""
        task = _make_task(task_class=TaskClass.PLANNER)
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        first = chain[0]
        assert first.model_profile == "claude_opus"

    def test_claude_backup_final_review_gets_opus(self):
        """FINAL_REVIEW must get Opus in claude_backup state."""
        task = _make_task(task_class=TaskClass.FINAL_REVIEW)
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        first = chain[0]
        assert first.model_profile == "claude_opus"

    def test_claude_backup_implementation_gets_sonnet(self):
        """IMPLEMENTATION should get Sonnet in claude_backup state."""
        task = _make_task(task_class=TaskClass.IMPLEMENTATION)
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        first = chain[0]
        assert first.model_profile == "claude_sonnet"

    def test_claude_backup_low_risk_planner_still_opus(self):
        """PLANNER with LOW risk must still get Opus."""
        task = _make_task(task_class=TaskClass.PLANNER, risk=TaskRisk.LOW)
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        first = chain[0]
        assert first.model_profile == "claude_opus"

    def test_claude_backup_critical_gets_opus(self):
        """CRITICAL risk task must get Opus (HEAVY_EXEC)."""
        task = _make_task(risk=TaskRisk.CRITICAL)
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        first = chain[0]
        assert first.model_profile == "claude_opus"


# ---------------------------------------------------------------------------
# OpenRouter fallback — VISUAL → Kimi, DECIDE → MiMo, else → MiniMax
# ---------------------------------------------------------------------------

class TestOpenRouterFallbackPhaseModelSelection:
    """openrouter_fallback: VISUAL → Kimi, DECIDE → MiMo, else → MiniMax."""

    def test_openrouter_planner_gets_mimo(self):
        """PLANNER must get MiMo in openrouter_fallback state."""
        task = _make_task(task_class=TaskClass.PLANNER)
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        first = chain[0]
        assert first.model_profile == ModelProfile.OPENROUTER_MIMO.value

    def test_openrouter_final_review_gets_mimo(self):
        """FINAL_REVIEW must get MiMo in openrouter_fallback state."""
        task = _make_task(task_class=TaskClass.FINAL_REVIEW)
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        first = chain[0]
        assert first.model_profile == ModelProfile.OPENROUTER_MIMO.value

    def test_openrouter_visual_gets_kimi(self):
        """Visual/screenshot tasks must get Kimi."""
        task = _make_task(has_screenshots=True)
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        first = chain[0]
        assert first.model_profile == ModelProfile.OPENROUTER_KIMI.value

    def test_openrouter_implementation_gets_minimax(self):
        """Standard implementation should get MiniMax."""
        task = _make_task()
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        first = chain[0]
        assert first.model_profile == ModelProfile.OPENROUTER_MINIMAX.value

    def test_openrouter_multimodal_gets_kimi(self):
        """Multimodal tasks must get Kimi."""
        task = _make_task(requires_multimodal=True)
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        first = chain[0]
        assert first.model_profile == ModelProfile.OPENROUTER_KIMI.value


# ---------------------------------------------------------------------------
# Review mode selection
# ---------------------------------------------------------------------------

class TestReviewMode:
    """select_review_mode: DEEP for FINAL_REVIEW/HIGH risk, FAST otherwise."""

    def test_final_review_is_deep(self):
        """FINAL_REVIEW must always be DEEP review."""
        task = _make_task(task_class=TaskClass.FINAL_REVIEW)
        assert select_review_mode(task) == ReviewMode.DEEP

    def test_high_risk_is_deep(self):
        """HIGH risk must trigger DEEP review."""
        task = _make_task(risk=TaskRisk.HIGH)
        assert select_review_mode(task) == ReviewMode.DEEP

    def test_architecture_change_is_deep(self):
        """REPO_ARCHITECTURE_CHANGE must trigger DEEP review."""
        task = _make_task(task_class=TaskClass.REPO_ARCHITECTURE_CHANGE)
        assert select_review_mode(task) == ReviewMode.DEEP

    def test_medium_risk_is_fast(self):
        """Medium risk standard task should be FAST review."""
        task = _make_task(risk=TaskRisk.MEDIUM)
        assert select_review_mode(task) == ReviewMode.FAST

    def test_low_risk_final_review_still_deep(self):
        """FINAL_REVIEW with LOW risk must still be DEEP (task_class override)."""
        task = _make_task(task_class=TaskClass.FINAL_REVIEW, risk=TaskRisk.LOW)
        assert select_review_mode(task) == ReviewMode.DEEP


# ---------------------------------------------------------------------------
# Phase property on TaskMeta
# ---------------------------------------------------------------------------

class TestTaskPhase:
    """TaskMeta.phase derived from task_class."""

    def test_planner_phase_is_plan(self):
        task = _make_task(task_class=TaskClass.PLANNER)
        assert task.phase == "plan"

    def test_final_review_phase_is_validate(self):
        task = _make_task(task_class=TaskClass.FINAL_REVIEW)
        assert task.phase == "validate"

    def test_implementation_phase_is_execute(self):
        task = _make_task(task_class=TaskClass.IMPLEMENTATION)
        assert task.phase == "execute"

    def test_debug_phase_is_execute(self):
        task = _make_task(task_class=TaskClass.DEBUG)
        assert task.phase == "execute"


# ---------------------------------------------------------------------------
# Log explainability — reason string includes phase
# ---------------------------------------------------------------------------

class TestRouteReason:
    """route_task reason string includes phase and modality."""

    def test_reason_includes_phase(self, patched_routing, monkeypatch):
        """Reason string must contain phase=<phase>."""
        from unittest.mock import patch
        from router.policy import _run_executor
        from router.models import ExecutorResult

        task = _make_task(task_class=TaskClass.PLANNER)

        mock_result = ExecutorResult(
            task_id=task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_gpt54",
            success=True,
            final_summary="done",
        )

        with patch("router.policy._run_executor", return_value=mock_result):
            decision, result = route_task(task)

        assert "phase=plan" in decision.reason

    def test_reason_includes_state(self, patched_routing, monkeypatch):
        """Reason string must contain state=<state>."""
        from unittest.mock import patch
        from router.models import ExecutorResult

        task = _make_task()

        mock_result = ExecutorResult(
            task_id=task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_gpt54_mini",
            success=True,
            final_summary="done",
        )

        with patch("router.policy._run_executor", return_value=mock_result):
            decision, result = route_task(task)

        assert "state=" in decision.reason

    def test_reason_includes_modality(self, patched_routing, monkeypatch):
        """Reason string must contain modality=<modality>."""
        from unittest.mock import patch
        from router.models import ExecutorResult

        task = _make_task(modality=TaskModality.IMAGE)

        mock_result = ExecutorResult(
            task_id=task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_gpt54_mini",
            success=True,
            final_summary="done",
        )

        with patch("router.policy._run_executor", return_value=mock_result):
            decision, result = route_task(task)

        assert "modality=image" in decision.reason

    def test_reason_full_format(self, patched_routing, monkeypatch):
        """Reason string must match state=X, phase=Y, modality=Z format."""
        from unittest.mock import patch
        from router.models import ExecutorResult

        task = _make_task(task_class=TaskClass.FINAL_REVIEW, modality=TaskModality.MIXED)

        mock_result = ExecutorResult(
            task_id=task.task_id,
            tool="claude_code",
            backend="anthropic",
            model_profile="claude_opus",
            success=True,
            final_summary="done",
        )

        with patch("router.policy._run_executor", return_value=mock_result):
            decision, result = route_task(task)

        assert "phase=validate" in decision.reason
        assert "modality=mixed" in decision.reason
        assert "state=" in decision.reason


# ---------------------------------------------------------------------------
# OpenAI primary — DECIDE → gpt-5.4
# ---------------------------------------------------------------------------

class TestOpenAIPrimaryPhaseModelSelection:
    """openai_primary: DECIDE → gpt-5.4, else → gpt-5.4-mini."""

    def test_primary_planner_gets_gpt54(self):
        task = _make_task(task_class=TaskClass.PLANNER)
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        first = chain[0]
        assert first.model_profile == "codex_gpt54"

    def test_primary_implementation_gets_mini(self):
        task = _make_task()
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        first = chain[0]
        assert first.model_profile == "codex_gpt54_mini"

    def test_primary_heavy_exec_gets_gpt54(self):
        task = _make_task(risk=TaskRisk.CRITICAL)
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        first = chain[0]
        assert first.model_profile == "codex_gpt54"
