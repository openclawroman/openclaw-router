"""Tests for router.flow_control — 2-phase/3-phase pipeline selection and execution."""

import pytest

from router.models import TaskMeta, TaskClass, TaskRisk
from router.flow_control import (
    PipelinePhase,
    PhaseConfig,
    FlowConfig,
    PhaseResult,
    PipelineResult,
    requires_three_phase,
    select_flow,
    THREE_PHASE_TASK_CLASSES,
    THREE_PHASE_RISK_LEVELS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_task(
    task_class: TaskClass = TaskClass.IMPLEMENTATION,
    risk: TaskRisk = TaskRisk.MEDIUM,
    requires_multimodal: bool = False,
    **kwargs,
) -> TaskMeta:
    return TaskMeta(
        task_id="test_001",
        agent="coder",
        task_class=task_class,
        risk=risk,
        modality="text",
        requires_repo_write=True,
        requires_multimodal=requires_multimodal,
        has_screenshots=False,
        swarm=False,
        repo_path="/tmp/repo",
        cwd="/tmp/repo",
        summary="Test task",
        **kwargs,
    )


# ── Phase constants ──────────────────────────────────────────────────────────

class TestPipelinePhaseConstants:
    def test_phase_names(self):
        assert PipelinePhase.PLAN == "plan"
        assert PipelinePhase.EXECUTE == "execute"
        assert PipelinePhase.VALIDATE == "validate"


# ── Three-phase trigger logic ────────────────────────────────────────────────

class TestRequiresThreePhase:
    def test_implementation_medium_is_two_phase(self):
        task = _make_task(TaskClass.IMPLEMENTATION, TaskRisk.MEDIUM)
        assert requires_three_phase(task) is False

    def test_architecture_change_is_three_phase(self):
        task = _make_task(TaskClass.REPO_ARCHITECTURE_CHANGE, TaskRisk.MEDIUM)
        assert requires_three_phase(task) is True

    def test_swarm_is_three_phase(self):
        task = _make_task(TaskClass.SWARM_CODE_TASK, TaskRisk.MEDIUM)
        assert requires_three_phase(task) is True

    def test_multimodal_class_is_three_phase(self):
        task = _make_task(TaskClass.MULTIMODAL_CODE_TASK, TaskRisk.MEDIUM)
        assert requires_three_phase(task) is True

    def test_critical_risk_is_three_phase(self):
        task = _make_task(TaskClass.IMPLEMENTATION, TaskRisk.CRITICAL)
        assert requires_three_phase(task) is True

    def test_high_risk_is_three_phase(self):
        task = _make_task(TaskClass.BUGFIX, TaskRisk.HIGH)
        assert requires_three_phase(task) is True

    def test_requires_multimodal_flag_is_three_phase(self):
        task = _make_task(TaskClass.IMPLEMENTATION, TaskRisk.LOW, requires_multimodal=True)
        assert requires_three_phase(task) is True

    def test_low_risk_implementation_is_two_phase(self):
        task = _make_task(TaskClass.IMPLEMENTATION, TaskRisk.LOW)
        assert requires_three_phase(task) is False

    def test_code_review_is_two_phase(self):
        task = _make_task(TaskClass.CODE_REVIEW, TaskRisk.MEDIUM)
        assert requires_three_phase(task) is False

    def test_debug_is_two_phase(self):
        task = _make_task(TaskClass.DEBUG, TaskRisk.LOW)
        assert requires_three_phase(task) is False


# ── select_flow ──────────────────────────────────────────────────────────────

class TestSelectFlow:
    def test_two_phase_flow_structure(self):
        task = _make_task(TaskClass.IMPLEMENTATION, TaskRisk.MEDIUM)
        flow = select_flow(task)
        assert flow.flow_type == "two_phase"
        assert len(flow.phases) == 2
        assert flow.phases[0].name == PipelinePhase.EXECUTE
        assert flow.phases[1].name == PipelinePhase.VALIDATE

    def test_three_phase_flow_structure(self):
        task = _make_task(TaskClass.REPO_ARCHITECTURE_CHANGE, TaskRisk.HIGH)
        flow = select_flow(task)
        assert flow.flow_type == "three_phase"
        assert len(flow.phases) == 3
        assert flow.phases[0].name == PipelinePhase.PLAN
        assert flow.phases[1].name == PipelinePhase.EXECUTE
        assert flow.phases[2].name == PipelinePhase.VALIDATE

    def test_two_phase_execute_is_required(self):
        task = _make_task(TaskClass.IMPLEMENTATION)
        flow = select_flow(task)
        execute = flow.phases[0]
        assert execute.required is True
        assert execute.retry_on_failure is True
        assert execute.max_retries == 2

    def test_two_phase_validate_is_optional(self):
        task = _make_task(TaskClass.IMPLEMENTATION)
        flow = select_flow(task)
        validate = flow.phases[1]
        assert validate.required is False
        assert validate.retry_on_failure is False

    def test_three_phase_plan_is_required(self):
        task = _make_task(TaskClass.SWARM_CODE_TASK, TaskRisk.CRITICAL)
        flow = select_flow(task)
        plan = flow.phases[0]
        assert plan.name == PipelinePhase.PLAN
        assert plan.required is True
        assert plan.retry_on_failure is True
        assert plan.max_retries == 1

    def test_three_phase_execute_has_extended_timeout(self):
        task = _make_task(TaskClass.REPO_ARCHITECTURE_CHANGE, TaskRisk.HIGH)
        flow = select_flow(task)
        execute = flow.phases[1]
        assert execute.timeout_s == 600

    def test_flow_includes_task_metadata(self):
        task = _make_task(TaskClass.BUGFIX, TaskRisk.HIGH)
        flow = select_flow(task)
        assert flow.task_class == "bugfix"
        assert flow.risk == "high"

    def test_three_phase_reason_mentions_triggers(self):
        task = _make_task(TaskClass.SWARM_CODE_TASK, TaskRisk.HIGH)
        flow = select_flow(task)
        assert "three_phase" in flow.reason
        assert "swarm_code_task" in flow.reason
        assert "high" in flow.reason

    def test_two_phase_reason(self):
        task = _make_task(TaskClass.IMPLEMENTATION, TaskRisk.LOW)
        flow = select_flow(task)
        assert "two_phase" in flow.reason
        assert "implementation" in flow.reason
        assert "low" in flow.reason

    def test_all_critical_risk_triggers_three_phase(self):
        for tc in TaskClass:
            task = _make_task(tc, TaskRisk.CRITICAL)
            flow = select_flow(task)
            assert flow.flow_type == "three_phase"

    def test_swarm_regardless_of_risk(self):
        for risk in [TaskRisk.LOW, TaskRisk.MEDIUM]:
            task = _make_task(TaskClass.SWARM_CODE_TASK, risk)
            flow = select_flow(task)
            assert flow.flow_type == "three_phase"


# ── PhaseResult and PipelineResult ───────────────────────────────────────────

class TestPhaseResult:
    def test_success_result(self):
        result = PhaseResult(phase="execute", success=True, latency_ms=1000)
        assert result.success is True
        assert result.latency_ms == 1000
        assert result.error is None

    def test_failure_result(self):
        result = PhaseResult(
            phase="plan", success=False, latency_ms=500,
            error="planning failed",
        )
        assert result.success is False
        assert result.error == "planning failed"


class TestPipelineResult:
    def test_empty_pipeline(self):
        result = PipelineResult(flow_type="two_phase")
        assert result.overall_success is True
        assert result.total_latency_ms == 0
        assert result.failed_phase is None

    def test_add_success_phase(self):
        result = PipelineResult(flow_type="two_phase")
        result.add_phase(PhaseResult(phase="execute", success=True, latency_ms=1000))
        assert result.overall_success is True
        assert result.total_latency_ms == 1000

    def test_add_failure_phase_sets_overall_false(self):
        result = PipelineResult(flow_type="three_phase")
        result.add_phase(PhaseResult(phase="plan", success=False, latency_ms=500, error="fail"))
        result.add_phase(PhaseResult(phase="execute", success=True, latency_ms=1000))
        assert result.overall_success is False
        assert result.failed_phase == "plan"
        assert result.total_latency_ms == 1500

    def test_phase_names(self):
        result = PipelineResult(flow_type="three_phase")
        result.add_phase(PhaseResult(phase="plan", success=True, latency_ms=100))
        result.add_phase(PhaseResult(phase="execute", success=True, latency_ms=200))
        result.add_phase(PhaseResult(phase="validate", success=True, latency_ms=50))
        assert result.phase_names == ["plan", "execute", "validate"]

    def test_get_phase(self):
        result = PipelineResult(flow_type="two_phase")
        result.add_phase(PhaseResult(phase="execute", success=True, latency_ms=100))
        result.add_phase(PhaseResult(phase="validate", success=True, latency_ms=50))
        execute = result.get_phase("execute")
        assert execute is not None
        assert execute.success is True
        assert result.get_phase("plan") is None

    def test_first_failure_recorded(self):
        """Only the first failure should be recorded as failed_phase."""
        result = PipelineResult(flow_type="three_phase")
        result.add_phase(PhaseResult(phase="plan", success=False, latency_ms=100, error="e1"))
        result.add_phase(PhaseResult(phase="execute", success=False, latency_ms=200, error="e2"))
        assert result.failed_phase == "plan"
