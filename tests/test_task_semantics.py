"""Tests for planner/final-review task semantics and routing.

TDD: These tests define the expected behavior before implementation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.classifier import detect_task_class, Classifier
from router.models import TaskClass, TaskRisk, TaskModality, TaskMeta
from router.policy import build_chain, choose_openai_profile
from router.models import CodexState


# ---------------------------------------------------------------------------
# Classifier: PLANNER detection
# ---------------------------------------------------------------------------

class TestPlannerDetection:
    """Planner tasks should be detected from natural language patterns."""

    @pytest.mark.parametrize("text", [
        "plan how to implement auth",
        "think about how to structure the project",
        "how should we approach the migration",
        "break down the work for the new feature",
    ])
    def test_planner_keywords(self, text):
        assert detect_task_class(text) == TaskClass.PLANNER

    def test_plan_before_implement(self):
        """'plan' should match PLANNER even if 'implement' appears later."""
        assert detect_task_class("plan how to implement the auth system") == TaskClass.PLANNER


# ---------------------------------------------------------------------------
# Classifier: FINAL_REVIEW detection
# ---------------------------------------------------------------------------

class TestFinalReviewDetection:
    """Final review tasks should be detected from natural language patterns."""

    @pytest.mark.parametrize("text", [
        "review the generated code",
        "check the implementation for correctness",
        "validate the output",
        "verify the result of the changes",
    ])
    def test_final_review_keywords(self, text):
        assert detect_task_class(text) == TaskClass.FINAL_REVIEW


# ---------------------------------------------------------------------------
# Classifier: existing classes still work
# ---------------------------------------------------------------------------

class TestExistingClassesPreserved:
    """Adding PLANNER/FINAL_REVIEW must not break existing detections."""

    @pytest.mark.parametrize("text,expected", [
        ("add test for login", TaskClass.TEST_GENERATION),
        ("fix the null pointer bug", TaskClass.BUGFIX),
        ("refactor the auth module", TaskClass.REFACTOR),
        ("diagnose the crash on startup", TaskClass.DEBUG),
        ("implement user authentication", TaskClass.IMPLEMENTATION),
        ("change the architecture to microservices", TaskClass.REPO_ARCHITECTURE_CHANGE),
        ("convert this screenshot to code", TaskClass.UI_FROM_SCREENSHOT),
        ("multimodal input processing", TaskClass.MULTIMODAL_CODE_TASK),
        ("swarm of coding agents", TaskClass.SWARM_CODE_TASK),
    ])
    def test_still_works(self, text, expected):
        assert detect_task_class(text) == expected


# ---------------------------------------------------------------------------
# Classifier: summary generation for new classes
# ---------------------------------------------------------------------------

class TestSummaryForNewClasses:
    """Summary generator should handle planner/final_review text."""

    def test_planner_summary(self):
        clf = Classifier()
        meta = clf.classify("plan how to implement auth")
        assert meta.task_class == TaskClass.PLANNER
        assert len(meta.summary) > 0
        assert len(meta.summary) <= 120

    def test_final_review_summary(self):
        clf = Classifier()
        meta = clf.classify("review the generated code")
        assert meta.task_class == TaskClass.FINAL_REVIEW
        assert len(meta.summary) > 0
        assert len(meta.summary) <= 120


# ---------------------------------------------------------------------------
# Classifier: requires_repo_write for new classes
# ---------------------------------------------------------------------------

class TestRepoWriteForNewClasses:
    """Planner and final_review tasks should not require repo writes."""

    def test_planner_no_repo_write(self):
        clf = Classifier()
        meta = clf.classify("plan how to implement auth")
        assert meta.requires_repo_write is False

    def test_final_review_no_repo_write(self):
        clf = Classifier()
        meta = clf.classify("review the generated code")
        assert meta.requires_repo_write is False


# ---------------------------------------------------------------------------
# Routing: planner tasks → gpt-5.4 (heavy lane)
# ---------------------------------------------------------------------------

class TestPlannerRouting:
    """Planner tasks should route to gpt-5.4 in both primary and conservation chains."""

    def test_planner_uses_gpt54_primary(self):
        task = TaskMeta(
            task_id="test-plan",
            task_class=TaskClass.PLANNER,
            risk=TaskRisk.MEDIUM,
            summary="plan how to implement auth",
        )
        model = choose_openai_profile(task)
        # Should be gpt54 (heavy), not gpt54_mini
        from router.config_loader import get_model
        assert model == get_model("gpt54")

    def test_planner_uses_gpt54_conservation(self):
        """In conservation mode, planner tasks should still get gpt-5.4, not mini."""
        task = TaskMeta(
            task_id="test-plan",
            task_class=TaskClass.PLANNER,
            risk=TaskRisk.MEDIUM,
            summary="plan how to implement auth",
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        # First entry should be codex with gpt54 profile
        assert chain[0].model_profile == "codex_gpt54"


# ---------------------------------------------------------------------------
# Routing: final_review tasks → gpt-5.4
# ---------------------------------------------------------------------------

class TestFinalReviewRouting:
    """Final review tasks should route to gpt-5.4 in both chains."""

    def test_final_review_uses_gpt54_primary(self):
        task = TaskMeta(
            task_id="test-review",
            task_class=TaskClass.FINAL_REVIEW,
            risk=TaskRisk.MEDIUM,
            summary="review the generated code",
        )
        model = choose_openai_profile(task)
        from router.config_loader import get_model
        assert model == get_model("gpt54")

    def test_final_review_uses_gpt54_conservation(self):
        task = TaskMeta(
            task_id="test-review",
            task_class=TaskClass.FINAL_REVIEW,
            risk=TaskRisk.MEDIUM,
            summary="review the generated code",
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54"


# ---------------------------------------------------------------------------
# Routing: existing task classes still route correctly
# ---------------------------------------------------------------------------

class TestExistingRoutingPreserved:
    """Existing task classes should still route to their expected models."""

    def test_implementation_uses_mini(self):
        task = TaskMeta(
            task_id="test-impl",
            task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM,
            summary="implement feature",
        )
        model = choose_openai_profile(task)
        from router.config_loader import get_model
        assert model == get_model("gpt54_mini")

    def test_critical_still_uses_gpt54(self):
        task = TaskMeta(
            task_id="test-crit",
            task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.CRITICAL,
            summary="critical production fix",
        )
        model = choose_openai_profile(task)
        from router.config_loader import get_model
        assert model == get_model("gpt54")

    def test_debug_still_uses_gpt54(self):
        task = TaskMeta(
            task_id="test-debug",
            task_class=TaskClass.DEBUG,
            risk=TaskRisk.MEDIUM,
            summary="debug the crash",
        )
        model = choose_openai_profile(task)
        from router.config_loader import get_model
        assert model == get_model("gpt54")

    def test_architecture_change_still_uses_gpt54(self):
        task = TaskMeta(
            task_id="test-arch",
            task_class=TaskClass.REPO_ARCHITECTURE_CHANGE,
            risk=TaskRisk.MEDIUM,
            summary="change architecture",
        )
        model = choose_openai_profile(task)
        from router.config_loader import get_model
        assert model == get_model("gpt54")
