"""Tests for intent-aware helpers and refactored selectors.

Validates is_decision_task, is_visual_task, is_heavy_execution_task
and their integration with choose_openai/openrouter/claude selectors.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality, TaskPhase, ModelProfile,
)
from router.policy import (
    is_decision_task,
    is_visual_task,
    is_heavy_execution_task,
    choose_openai_profile,
    choose_openrouter_profile,
    choose_claude_model,
)


def _task(**overrides) -> TaskMeta:
    """Build a TaskMeta with sensible defaults."""
    defaults = dict(
        task_id="test-intent",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        phase=TaskPhase.EXECUTE,
        risk=TaskRisk.MEDIUM,
    )
    defaults.update(overrides)
    return TaskMeta(**defaults)


# ---------------------------------------------------------------------------
# is_decision_task
# ---------------------------------------------------------------------------

class TestIsDecisionTask:

    def test_planner_is_decision(self):
        assert is_decision_task(_task(phase=TaskPhase.DECIDE, task_class=TaskClass.PLANNER)) is True

    def test_final_review_is_decision(self):
        assert is_decision_task(_task(phase=TaskPhase.DECIDE, task_class=TaskClass.FINAL_REVIEW)) is True

    def test_implementation_is_not_decision(self):
        assert is_decision_task(_task(phase=TaskPhase.EXECUTE, task_class=TaskClass.IMPLEMENTATION)) is False

    def test_inferred_from_planner_task_class(self):
        """When phase is default EXECUTE but task_class is PLANNER, inferred_phase gives DECIDE."""
        task = TaskMeta(task_id="t", task_class=TaskClass.PLANNER)
        assert is_decision_task(task) is True

    def test_inferred_from_final_review_task_class(self):
        task = TaskMeta(task_id="t", task_class=TaskClass.FINAL_REVIEW)
        assert is_decision_task(task) is True


# ---------------------------------------------------------------------------
# is_visual_task
# ---------------------------------------------------------------------------

class TestIsVisualTask:

    def test_visual_phase_is_visual(self):
        assert is_visual_task(_task(phase=TaskPhase.VISUAL)) is True

    def test_text_phase_is_not_visual(self):
        assert is_visual_task(_task(phase=TaskPhase.EXECUTE)) is False

    def test_inferred_from_screenshots(self):
        task = TaskMeta(task_id="t", has_screenshots=True)
        assert is_visual_task(task) is True

    def test_inferred_from_multimodal(self):
        task = TaskMeta(task_id="t", requires_multimodal=True)
        assert is_visual_task(task) is True

    def test_inferred_from_ui_from_screenshot_class(self):
        task = TaskMeta(task_id="t", task_class=TaskClass.UI_FROM_SCREENSHOT)
        assert is_visual_task(task) is True


# ---------------------------------------------------------------------------
# is_heavy_execution_task
# ---------------------------------------------------------------------------

class TestIsHeavyExecutionTask:

    def test_critical_risk_is_heavy(self):
        assert is_heavy_execution_task(_task(risk=TaskRisk.CRITICAL)) is True

    def test_debug_task_class_is_heavy(self):
        assert is_heavy_execution_task(_task(task_class=TaskClass.DEBUG)) is True

    def test_architecture_change_is_heavy(self):
        assert is_heavy_execution_task(_task(task_class=TaskClass.REPO_ARCHITECTURE_CHANGE)) is True

    def test_medium_implementation_is_not_heavy(self):
        assert is_heavy_execution_task(_task(
            risk=TaskRisk.MEDIUM, task_class=TaskClass.IMPLEMENTATION
        )) is False

    def test_high_risk_implementation_is_not_heavy(self):
        """HIGH risk without DEBUG/ARCHITECTURE_CHANGE is not heavy."""
        assert is_heavy_execution_task(_task(
            risk=TaskRisk.HIGH, task_class=TaskClass.IMPLEMENTATION
        )) is False


# ---------------------------------------------------------------------------
# choose_openai_profile integration
# ---------------------------------------------------------------------------

class TestChooseOpenaiProfile:

    def test_returns_gpt54_for_decision_task(self):
        task = _task(phase=TaskPhase.DECIDE)
        assert choose_openai_profile(task) == "gpt-5.4"

    def test_returns_gpt54_for_heavy_execution(self):
        task = _task(risk=TaskRisk.CRITICAL)
        assert choose_openai_profile(task) == "gpt-5.4"

    def test_returns_gpt54_mini_for_execute_task(self):
        task = _task(phase=TaskPhase.EXECUTE, risk=TaskRisk.MEDIUM)
        assert choose_openai_profile(task) == "gpt-5.4-mini"

    def test_returns_gpt54_mini_for_visual_task(self):
        """Visual tasks without heavy execution get the lightweight model."""
        task = _task(phase=TaskPhase.VISUAL, risk=TaskRisk.LOW)
        assert choose_openai_profile(task) == "gpt-5.4-mini"


# ---------------------------------------------------------------------------
# choose_openrouter_profile integration
# ---------------------------------------------------------------------------

class TestChooseOpenrouterProfile:

    def test_kimi_for_visual_task(self):
        task = _task(phase=TaskPhase.VISUAL)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_KIMI

    def test_mimo_for_decision_task(self):
        task = _task(phase=TaskPhase.DECIDE)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_MIMO

    def test_mimo_for_heavy_execution(self):
        task = _task(risk=TaskRisk.CRITICAL)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_MIMO

    def test_minimax_for_default(self):
        task = _task(phase=TaskPhase.EXECUTE, risk=TaskRisk.MEDIUM)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_MINIMAX


# ---------------------------------------------------------------------------
# choose_claude_model integration
# ---------------------------------------------------------------------------

class TestChooseClaudeModel:

    def test_opus_for_decision_task(self):
        task = _task(phase=TaskPhase.DECIDE)
        assert choose_claude_model(task) == "claude-opus-4.6"

    def test_opus_for_heavy_execution(self):
        task = _task(risk=TaskRisk.CRITICAL)
        assert choose_claude_model(task) == "claude-opus-4.6"

    def test_sonnet_for_execute_task(self):
        task = _task(phase=TaskPhase.EXECUTE, risk=TaskRisk.MEDIUM)
        assert choose_claude_model(task) == "claude-sonnet-4.6"
