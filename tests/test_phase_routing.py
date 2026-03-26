"""
Comprehensive E2E tests for the phase-aware routing system.

Tests verify FULL routing behavior with phases across all 4 Codex states.
Phase concept: DECIDE | EXECUTE | VISUAL | HEAVY_EXEC

These tests define the expected behavior for phase-aware routing.
Some tests may fail until the phase-aware routing branch is merged.

Dependencies:
  - router.models.TaskPhase enum (DECIDE, EXECUTE, VISUAL, HEAVY_EXEC)
  - router.classifier.classify_phase() function
  - Phase-aware chain builders in router.policy
  - TaskMeta.phase field
  - RouteDecision.phase field
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    RouteDecision, ExecutorResult, ChainEntry, CodexState,
)
from router.policy import (
    route_task, build_chain, resolve_state,
    choose_openai_profile, choose_claude_model,
    choose_openrouter_profile,
    reset_breaker, reset_notifier,
)
import router.config_loader as _cfg_mod


# ---------------------------------------------------------------------------
# Helpers — import phase components with graceful fallback
# ---------------------------------------------------------------------------

def _import_phase_enum():
    """Import TaskPhase enum, skip tests if not yet implemented."""
    try:
        from router.models import TaskPhase
        return TaskPhase
    except ImportError:
        pytest.skip("TaskPhase enum not yet implemented in router.models")


def _import_classify_phase():
    """Import classify_phase, skip tests if not yet implemented."""
    try:
        from router.classifier import classify_phase
        return classify_phase
    except ImportError:
        pytest.skip("classify_phase not yet implemented in router.classifier")


def _make_task(
    summary: str = "Test task",
    *,
    task_class: TaskClass = TaskClass.IMPLEMENTATION,
    risk: TaskRisk = TaskRisk.MEDIUM,
    modality: TaskModality = TaskModality.TEXT,
    has_screenshots: bool = False,
    requires_multimodal: bool = False,
    phase=None,
    task_id: str = "phase-test-001",
) -> TaskMeta:
    """Build a TaskMeta for phase tests, optionally with phase override."""
    kw = dict(
        task_id=task_id,
        agent="coder",
        task_class=task_class,
        risk=risk,
        modality=modality,
        requires_repo_write=task_class not in {
            TaskClass.CODE_REVIEW, TaskClass.DEBUG,
            TaskClass.PLANNER, TaskClass.FINAL_REVIEW,
        },
        requires_multimodal=requires_multimodal,
        has_screenshots=has_screenshots,
        swarm=False,
        repo_path="/tmp/test-repo",
        cwd="/tmp/test-repo",
        summary=summary,
    )
    meta = TaskMeta(**kw)
    # If phase field exists on TaskMeta, set it
    if phase is not None and hasattr(meta, "phase"):
        meta.phase = phase
    return meta


# ===================================================================
# SECTION 1: Phase Classification (tests 1–8)
# ===================================================================

class TestPhaseClassification:
    """Tests for classify_phase: summary text → TaskPhase."""

    def test_plan_architecture_is_decide(self):
        """'Plan the architecture for auth' → DECIDE"""
        classify_phase = _import_classify_phase()
        phase = classify_phase("Plan the architecture for auth")
        TaskPhase = _import_phase_enum()
        assert phase == TaskPhase.DECIDE

    def test_final_review_is_decide(self):
        """'Final review of the PR' → DECIDE"""
        classify_phase = _import_classify_phase()
        TaskPhase = _import_phase_enum()
        phase = classify_phase("Final review of the PR")
        assert phase == TaskPhase.DECIDE

    def test_triage_production_issue_is_decide(self):
        """'Triage the production issue' → DECIDE"""
        classify_phase = _import_classify_phase()
        TaskPhase = _import_phase_enum()
        phase = classify_phase("Triage the production issue")
        assert phase == TaskPhase.DECIDE

    def test_implement_login_page_is_execute(self):
        """'Implement login page' → EXECUTE"""
        classify_phase = _import_classify_phase()
        TaskPhase = _import_phase_enum()
        phase = classify_phase("Implement login page")
        assert phase == TaskPhase.EXECUTE

    def test_fix_bug_in_parser_is_execute(self):
        """'Fix bug in parser' → EXECUTE"""
        classify_phase = _import_classify_phase()
        TaskPhase = _import_phase_enum()
        phase = classify_phase("Fix bug in parser")
        assert phase == TaskPhase.EXECUTE

    def test_debug_memory_leak_is_heavy_exec(self):
        """'Debug memory leak' → EXECUTE but HEAVY_EXEC (debug is heavy)"""
        classify_phase = _import_classify_phase()
        TaskPhase = _import_phase_enum()
        phase = classify_phase("Debug memory leak")
        assert phase == TaskPhase.HEAVY_EXEC

    def test_review_screenshot_is_visual(self):
        """'Review this screenshot' → VISUAL"""
        classify_phase = _import_classify_phase()
        TaskPhase = _import_phase_enum()
        phase = classify_phase("Review this screenshot")
        assert phase == TaskPhase.VISUAL

    def test_build_ui_from_mockup_is_visual(self):
        """'Build UI from this mockup' → VISUAL"""
        classify_phase = _import_classify_phase()
        TaskPhase = _import_phase_enum()
        phase = classify_phase("Build UI from this mockup")
        assert phase == TaskPhase.VISUAL


# ===================================================================
# SECTION 2: openai_primary + phase (tests 9–11)
# ===================================================================

class TestOpenaiPrimaryPhase:
    """Phase-aware model selection in openai_primary state."""

    @pytest.fixture(autouse=True)
    def _set_state(self, monkeypatch):
        """Force state to openai_primary."""
        monkeypatch.setattr("router.policy.resolve_state", lambda *a, **kw: CodexState.OPENAI_PRIMARY)

    def test_decide_task_starts_with_gpt54(self):
        """DECIDE task → chain starts with gpt-5.4"""
        task = _make_task(
            "Plan the architecture for auth",
            task_class=TaskClass.PLANNER,
        )
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        # First entry should be codex with gpt54
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"
        assert chain[0].model_profile == "codex_gpt54"

    def test_execute_task_starts_with_gpt54_mini(self):
        """EXECUTE task → chain starts with gpt-5.4-mini"""
        task = _make_task(
            "Implement login page",
            task_class=TaskClass.IMPLEMENTATION,
        )
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"
        assert chain[0].model_profile == "codex_gpt54_mini"

    def test_heavy_exec_critical_starts_with_gpt54(self):
        """HEAVY_EXEC (CRITICAL) → chain starts with gpt-5.4"""
        task = _make_task(
            "Debug production memory leak",
            task_class=TaskClass.DEBUG,
            risk=TaskRisk.CRITICAL,
        )
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"
        assert chain[0].model_profile == "codex_gpt54"


# ===================================================================
# SECTION 3: openai_conservation + phase (tests 12–15)
# ===================================================================

class TestOpenaiConservationPhase:
    """Phase-aware model selection in openai_conservation state.

    KEY behavior: conservation raises the bar for gpt-5.4.
    Only PLANNER, FINAL_REVIEW, or CRITICAL risk get gpt-5.4.
    Regular EXECUTE gets gpt-5.4-mini to conserve budget.
    """

    @pytest.fixture(autouse=True)
    def _set_state(self, monkeypatch):
        monkeypatch.setattr("router.policy.resolve_state", lambda *a, **kw: CodexState.OPENAI_CONSERVATION)

    def test_decide_task_includes_gpt54(self):
        """DECIDE task → chain includes gpt-5.4 (NOT just mini).

        This is the KEY conservation test: even in conservation mode,
        DECIDE/PLANNER tasks get the full model because they need reasoning.
        """
        task = _make_task(
            "Plan the architecture for auth",
            task_class=TaskClass.PLANNER,
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"
        assert chain[0].model_profile == "codex_gpt54"

    def test_execute_task_uses_gpt54_mini(self):
        """EXECUTE task → chain uses gpt-5.4-mini (conserving budget)."""
        task = _make_task(
            "Implement login page",
            task_class=TaskClass.IMPLEMENTATION,
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54_mini"

    def test_heavy_exec_includes_gpt54(self):
        """HEAVY_EXEC → chain includes gpt-5.4 (critical warrants full model)."""
        task = _make_task(
            "Debug critical memory leak in production",
            task_class=TaskClass.DEBUG,
            risk=TaskRisk.CRITICAL,
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54"

    def test_decide_medium_risk_still_gets_gpt54(self):
        """DECIDE + MEDIUM risk → still gets gpt-5.4.

        Proves risk is NOT the only factor — task class (PLANNER)
        alone is enough to get gpt-5.4 in conservation mode.
        """
        task = _make_task(
            "Plan the refactor strategy",
            task_class=TaskClass.PLANNER,
            risk=TaskRisk.MEDIUM,
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54"

    def test_final_review_gets_gpt54_in_conservation(self):
        """FINAL_REVIEW in conservation → gpt-5.4 (review needs thorough model)."""
        task = _make_task(
            "Final review of the generated code",
            task_class=TaskClass.FINAL_REVIEW,
            risk=TaskRisk.MEDIUM,
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54"


# ===================================================================
# SECTION 4: claude_backup + phase (tests 16–18)
# ===================================================================

class TestClaudeBackupPhase:
    """Phase-aware model selection in claude_backup state."""

    @pytest.fixture(autouse=True)
    def _set_state(self, monkeypatch):
        monkeypatch.setattr("router.policy.resolve_state", lambda *a, **kw: CodexState.CLAUDE_BACKUP)

    def test_decide_task_uses_opus(self):
        """DECIDE task → uses Opus (needs reasoning power)."""
        task = _make_task(
            "Plan the architecture for auth",
            task_class=TaskClass.PLANNER,
        )
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        # Claude backup starts with claude_code
        assert chain[0].tool == "claude_code"
        assert chain[0].backend == "anthropic"
        assert chain[0].model_profile == "claude_opus"

    def test_execute_task_uses_sonnet(self):
        """EXECUTE task → uses Sonnet (efficient for implementation)."""
        task = _make_task(
            "Implement login page",
            task_class=TaskClass.IMPLEMENTATION,
        )
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        assert chain[0].tool == "claude_code"
        assert chain[0].backend == "anthropic"
        assert chain[0].model_profile == "claude_sonnet"

    def test_heavy_exec_uses_opus(self):
        """HEAVY_EXEC → uses Opus (needs full reasoning for complex debug)."""
        task = _make_task(
            "Debug critical memory leak",
            task_class=TaskClass.DEBUG,
            risk=TaskRisk.CRITICAL,
        )
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        assert chain[0].tool == "claude_code"
        assert chain[0].backend == "anthropic"
        assert chain[0].model_profile == "claude_opus"


# ===================================================================
# SECTION 5: openrouter_fallback + phase (tests 19–21)
# ===================================================================

class TestOpenrouterFallbackPhase:
    """Phase-aware model selection in openrouter_fallback state."""

    @pytest.fixture(autouse=True)
    def _set_state(self, monkeypatch):
        monkeypatch.setattr("router.policy.resolve_state", lambda *a, **kw: CodexState.OPENROUTER_FALLBACK)

    def test_decide_task_uses_mimo(self):
        """DECIDE task → uses MiMo (reasoning model on OpenRouter)."""
        task = _make_task(
            "Plan the architecture for auth",
            task_class=TaskClass.PLANNER,
        )
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openrouter"
        assert chain[0].model_profile == "openrouter_mimo"

    def test_visual_task_uses_kimi(self):
        """VISUAL task → uses Kimi (multimodal model)."""
        task = _make_task(
            "Build UI from this screenshot",
            task_class=TaskClass.UI_FROM_SCREENSHOT,
            has_screenshots=True,
            modality=TaskModality.IMAGE,
        )
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        assert chain[0].model_profile == "openrouter_kimi"

    def test_execute_task_uses_minimax(self):
        """EXECUTE task → uses MiniMax (default OpenRouter model)."""
        task = _make_task(
            "Implement login page",
            task_class=TaskClass.IMPLEMENTATION,
        )
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        assert chain[0].model_profile == "openrouter_minimax"


# ===================================================================
# SECTION 6: Integration tests — full route_task (tests 22–25)
# ===================================================================

class TestRouteTaskPhaseIntegration:
    """Full route_task integration with phase-aware routing."""

    def test_route_decide_in_conservation_uses_gpt54(
        self, patched_routing, monkeypatch
    ):
        """route_task with DECIDE in conservation → model is gpt-5.4"""
        monkeypatch.setattr(
            "router.policy.resolve_state",
            lambda *a, **kw: CodexState.OPENAI_CONSERVATION,
        )
        task = _make_task(
            "Plan the architecture for auth",
            task_class=TaskClass.PLANNER,
            risk=TaskRisk.MEDIUM,
        )
        # Mock executor to return success on first call
        with patch("router.policy.run_codex") as mock_codex:
            mock_codex.return_value = ExecutorResult(
                task_id=task.task_id,
                tool="codex_cli",
                backend="openai_native",
                model_profile="codex_gpt54",
                success=True,
                latency_ms=100,
                cost_estimate_usd=0.01,
                final_summary="Architecture plan created",
            )
            decision, result = route_task(task)

        assert result.success
        # Verify the chain started with gpt-5.4
        assert decision.chain[0].model_profile == "codex_gpt54"

    def test_route_execute_in_conservation_uses_gpt54_mini(
        self, patched_routing, monkeypatch
    ):
        """route_task with EXECUTE in conservation → model is gpt-5.4-mini"""
        monkeypatch.setattr(
            "router.policy.resolve_state",
            lambda *a, **kw: CodexState.OPENAI_CONSERVATION,
        )
        task = _make_task(
            "Implement login page",
            task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM,
        )
        with patch("router.policy.run_codex") as mock_codex:
            mock_codex.return_value = ExecutorResult(
                task_id=task.task_id,
                tool="codex_cli",
                backend="openai_native",
                model_profile="codex_gpt54_mini",
                success=True,
                latency_ms=100,
                cost_estimate_usd=0.005,
                final_summary="Login page implemented",
            )
            decision, result = route_task(task)

        assert result.success
        assert decision.chain[0].model_profile == "codex_gpt54_mini"

    def test_route_reason_includes_phase_value(
        self, patched_routing, monkeypatch
    ):
        """route_task reason includes phase value."""
        monkeypatch.setattr(
            "router.policy.resolve_state",
            lambda *a, **kw: CodexState.OPENAI_PRIMARY,
        )
        task = _make_task(
            "Plan the architecture for auth",
            task_class=TaskClass.PLANNER,
        )
        with patch("router.policy.run_codex") as mock_codex:
            mock_codex.return_value = ExecutorResult(
                task_id=task.task_id,
                tool="codex_cli",
                backend="openai_native",
                model_profile="codex_gpt54",
                success=True,
                latency_ms=100,
                cost_estimate_usd=0.01,
                final_summary="Plan created",
            )
            decision, result = route_task(task)

        # Reason should mention the phase
        reason_lower = decision.reason.lower()
        assert any(phase in reason_lower for phase in ["decide", "plan", "phase"]), \
            f"Expected phase in reason, got: {decision.reason}"

    def test_route_with_explicit_phase_override_uses_override(
        self, patched_routing, monkeypatch
    ):
        """route_task with explicit phase override in task_meta → uses override."""
        monkeypatch.setattr(
            "router.policy.resolve_state",
            lambda *a, **kw: CodexState.OPENAI_CONSERVATION,
        )
        # Even though this looks like EXECUTE, explicit phase override should win
        task = _make_task(
            "Implement login page",
            task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM,
        )
        # Try setting phase on task_meta — requires phase field to exist
        TaskPhase = _import_phase_enum()
        if hasattr(task, "phase"):
            task.phase = TaskPhase.DECIDE
        else:
            pytest.skip("TaskMeta does not have phase field yet")

        with patch("router.policy.run_codex") as mock_codex:
            mock_codex.return_value = ExecutorResult(
                task_id=task.task_id,
                tool="codex_cli",
                backend="openai_native",
                model_profile="codex_gpt54",
                success=True,
                latency_ms=100,
                cost_estimate_usd=0.01,
                final_summary="Overridden to DECIDE model",
            )
            decision, result = route_task(task)

        # Despite IMPLEMENTATION task_class, explicit DECIDE phase should give gpt-5.4
        assert decision.chain[0].model_profile == "codex_gpt54"


# ===================================================================
# SECTION 7: Edge cases (tests 26–28)
# ===================================================================

class TestPhaseEdgeCases:
    """Edge cases for phase classification and routing."""

    def test_visual_decide_visual_wins(self):
        """VISUAL + DECIDE (screenshot + plan) → VISUAL wins.

        Visual modality takes priority over planning keywords
        because multimodal tasks need multimodal-capable models.
        """
        classify_phase = _import_classify_phase()
        TaskPhase = _import_phase_enum()
        # Contains both "plan" (DECIDE) and "screenshot" (VISUAL)
        phase = classify_phase("Plan the UI redesign based on this screenshot")
        assert phase == TaskPhase.VISUAL, \
            f"VISUAL should take priority over DECIDE, got {phase}"

    def test_planner_critical_gives_decide_phase(self):
        """PLANNER + CRITICAL → DECIDE phase (not just heavy).

        A critical planner task should be classified as DECIDE (needs reasoning),
        not just HEAVY_EXEC (which implies execution-heavy work).
        """
        classify_phase = _import_classify_phase()
        TaskPhase = _import_phase_enum()
        phase = classify_phase("Critical: plan the disaster recovery architecture")
        assert phase == TaskPhase.DECIDE, \
            f"PLANNER + CRITICAL should be DECIDE, got {phase}"

    def test_phase_persisted_in_route_decision(self, patched_routing, monkeypatch):
        """Phase should be persisted in RouteDecision for observability."""
        monkeypatch.setattr(
            "router.policy.resolve_state",
            lambda *a, **kw: CodexState.OPENAI_PRIMARY,
        )
        task = _make_task(
            "Implement login page",
            task_class=TaskClass.IMPLEMENTATION,
        )
        with patch("router.policy.run_codex") as mock_codex:
            mock_codex.return_value = ExecutorResult(
                task_id=task.task_id,
                tool="codex_cli",
                backend="openai_native",
                model_profile="codex_gpt54_mini",
                success=True,
                latency_ms=100,
                cost_estimate_usd=0.005,
                final_summary="Login implemented",
            )
            decision, result = route_task(task)

        # RouteDecision should have a phase field
        assert hasattr(decision, "phase"), \
            "RouteDecision should have a 'phase' field for observability"
        if decision.phase is not None:
            TaskPhase = _import_phase_enum()
            assert decision.phase in {TaskPhase.DECIDE, TaskPhase.EXECUTE,
                                       TaskPhase.VISUAL, TaskPhase.HEAVY_EXEC}


# ===================================================================
# SECTION 8: Additional phase-aware routing behavior
# ===================================================================

class TestPhaseConservationBehavior:
    """Deeper tests on conservation state phase behavior."""

    @pytest.fixture(autouse=True)
    def _set_state(self, monkeypatch):
        monkeypatch.setattr(
            "router.policy.resolve_state",
            lambda *a, **kw: CodexState.OPENAI_CONSERVATION,
        )

    def test_bugfix_uses_mini_in_conservation(self):
        """BUGFIX (not critical) → gpt-5.4-mini in conservation."""
        task = _make_task(
            "Fix bug in parser",
            task_class=TaskClass.BUGFIX,
            risk=TaskRisk.MEDIUM,
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54_mini"

    def test_refactor_uses_mini_in_conservation(self):
        """REFACTOR (not critical) → gpt-5.4-mini in conservation."""
        task = _make_task(
            "Refactor the auth module",
            task_class=TaskClass.REFACTOR,
            risk=TaskRisk.MEDIUM,
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54_mini"

    def test_architecture_change_uses_mini_in_conservation(self):
        """REPO_ARCHITECTURE_CHANGE (not critical) → still mini in conservation.

        In conservation, only PLANNER/FINAL_REVIEW/CRITICAL get gpt-5.4.
        REPO_ARCHITECTURE_CHANGE alone is not enough — the bar is higher.
        """
        task = _make_task(
            "Restructure the module hierarchy",
            task_class=TaskClass.REPO_ARCHITECTURE_CHANGE,
            risk=TaskRisk.HIGH,
        )
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54_mini"

    def test_critical_any_class_gets_gpt54(self):
        """CRITICAL risk + any task_class → gpt-5.4 in conservation."""
        for task_class in [TaskClass.IMPLEMENTATION, TaskClass.BUGFIX,
                           TaskClass.REFACTOR, TaskClass.DEBUG]:
            task = _make_task(
                f"Critical: fix production {task_class.value}",
                task_class=task_class,
                risk=TaskRisk.CRITICAL,
            )
            chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
            assert chain[0].model_profile == "codex_gpt54", \
                f"CRITICAL + {task_class.value} should get gpt-5.4 in conservation"


class TestPhaseOpenaiPrimaryComprehensive:
    """Comprehensive phase behavior in openai_primary."""

    @pytest.fixture(autouse=True)
    def _set_state(self, monkeypatch):
        monkeypatch.setattr(
            "router.policy.resolve_state",
            lambda *a, **kw: CodexState.OPENAI_PRIMARY,
        )

    def test_debug_gets_gpt54_in_primary(self):
        """DEBUG task_class → gpt-5.4 in primary (needs reasoning for debugging)."""
        task = _make_task(
            "Debug memory leak in parser",
            task_class=TaskClass.DEBUG,
            risk=TaskRisk.MEDIUM,
        )
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert chain[0].model_profile == "codex_gpt54"

    def test_architecture_change_gets_gpt54_in_primary(self):
        """REPO_ARCHITECTURE_CHANGE → gpt-5.4 in primary."""
        task = _make_task(
            "Restructure the system architecture",
            task_class=TaskClass.REPO_ARCHITECTURE_CHANGE,
            risk=TaskRisk.HIGH,
        )
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert chain[0].model_profile == "codex_gpt54"

    def test_test_generation_uses_mini(self):
        """TEST_GENERATION → gpt-5.4-mini in primary (routine work)."""
        task = _make_task(
            "Add tests for the parser module",
            task_class=TaskClass.TEST_GENERATION,
            risk=TaskRisk.LOW,
        )
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert chain[0].model_profile == "codex_gpt54_mini"

    def test_code_review_uses_mini(self):
        """CODE_REVIEW → gpt-5.4-mini in primary (routine review)."""
        task = _make_task(
            "Review the pull request changes",
            task_class=TaskClass.CODE_REVIEW,
            risk=TaskRisk.MEDIUM,
        )
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert chain[0].model_profile == "codex_gpt54_mini"


class TestVisualPhaseRouting:
    """Visual/modality phase routing across states."""

    def test_visual_openrouter_uses_kimi(self):
        """VISUAL tasks in openrouter_fallback → Kimi."""
        task = _make_task(
            "Build UI from this mockup",
            task_class=TaskClass.UI_FROM_SCREENSHOT,
            has_screenshots=True,
            modality=TaskModality.IMAGE,
        )
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        assert chain[0].model_profile == "openrouter_kimi"

    def test_multimodal_openrouter_uses_kimi(self):
        """MULTIMODAL tasks in openrouter_fallback → Kimi."""
        task = _make_task(
            "Convert this design to code",
            task_class=TaskClass.MULTIMODAL_CODE_TASK,
            requires_multimodal=True,
            modality=TaskModality.MIXED,
        )
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        assert chain[0].model_profile == "openrouter_kimi"

    def test_critical_non_visual_openrouter_uses_mimo(self):
        """CRITICAL + not visual → MiMo (reasoning model)."""
        task = _make_task(
            "Critical: fix the auth system",
            task_class=TaskClass.BUGFIX,
            risk=TaskRisk.CRITICAL,
        )
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        assert chain[0].model_profile == "openrouter_mimo"

    def test_visual_primary_chain_has_openrouter_kimi_fallback(self):
        """VISUAL task in openai_primary → OpenRouter Kimi as fallback."""
        task = _make_task(
            "Build UI from screenshot",
            task_class=TaskClass.UI_FROM_SCREENSHOT,
            has_screenshots=True,
            modality=TaskModality.IMAGE,
        )
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        # The OpenRouter fallback entry in primary chain should use Kimi
        openrouter_entries = [e for e in chain if e.backend == "openrouter"]
        assert any(e.model_profile == "openrouter_kimi" for e in openrouter_entries), \
            "VISUAL task should have Kimi in OpenRouter fallback slot"


class TestClaudeBackupPhaseComprehensive:
    """Full phase coverage in claude_backup state."""

    @pytest.fixture(autouse=True)
    def _set_state(self, monkeypatch):
        monkeypatch.setattr(
            "router.policy.resolve_state",
            lambda *a, **kw: CodexState.CLAUDE_BACKUP,
        )

    def test_architecture_change_uses_opus(self):
        """REPO_ARCHITECTURE_CHANGE → Opus."""
        task = _make_task(
            "Restructure the system",
            task_class=TaskClass.REPO_ARCHITECTURE_CHANGE,
            risk=TaskRisk.HIGH,
        )
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        assert chain[0].model_profile == "claude_opus"

    def test_critical_uses_opus(self):
        """CRITICAL risk → Opus."""
        task = _make_task(
            "Critical: fix production issue",
            task_class=TaskClass.BUGFIX,
            risk=TaskRisk.CRITICAL,
        )
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        assert chain[0].model_profile == "claude_opus"

    def test_refactor_uses_sonnet(self):
        """REFACTOR → Sonnet (routine work)."""
        task = _make_task(
            "Refactor the auth module",
            task_class=TaskClass.REFACTOR,
            risk=TaskRisk.MEDIUM,
        )
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        assert chain[0].model_profile == "claude_sonnet"

    def test_debug_uses_sonnet_in_backup(self):
        """DEBUG (not critical) → Sonnet in backup (primary gives gpt-5.4 for debug)."""
        task = _make_task(
            "Debug memory leak",
            task_class=TaskClass.DEBUG,
            risk=TaskRisk.MEDIUM,
        )
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        # In claude_backup, debug without critical risk gets Sonnet
        assert chain[0].model_profile == "claude_sonnet"


# ===================================================================
# SECTION 9: Chain invariants with phases
# ===================================================================

class TestPhaseChainInvariants:
    """Verify chain invariants hold for phase-aware chains."""

    def test_claude_backup_no_openai_entries(self):
        """claude_backup chain must not contain openai_native entries."""
        task = _make_task(
            "Plan the architecture",
            task_class=TaskClass.PLANNER,
        )
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        for entry in chain:
            assert entry.backend != "openai_native", \
                f"claude_backup must not have openai_native: {entry}"

    def test_openrouter_fallback_only_openrouter(self):
        """openrouter_fallback chain must only contain openrouter entries."""
        task = _make_task(
            "Plan the architecture",
            task_class=TaskClass.PLANNER,
        )
        chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)
        for entry in chain:
            assert entry.backend == "openrouter", \
                f"openrouter_fallback must only have openrouter: {entry}"

    def test_openai_primary_starts_with_openai(self):
        """openai_primary chain starts with openai_native."""
        task = _make_task(summary="Test")
        chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert chain[0].backend == "openai_native"

    def test_openai_conservation_starts_with_openai(self):
        """openai_conservation chain starts with openai_native."""
        task = _make_task(summary="Test")
        chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        assert chain[0].backend == "openai_native"
