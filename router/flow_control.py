"""Flow control for ai-code-runner pipeline phases.

Pipeline phases:
  Phase 1 — Plan:   Generate a plan/strategy for the task (optional)
  Phase 2 — Execute: Run the actual code generation/modification
  Phase 3 — Validate: Verify the output format and correctness (optional)

Flow types:
  - two_phase:   Execute → Validate (skip planning)
  - three_phase: Plan → Execute → Validate

Selection logic:
  - three_phase is used for: architecture changes, high/critical risk tasks,
    swarm tasks, and tasks with multimodal input
  - two_phase is the default for everything else
"""

from dataclasses import dataclass, field
from typing import Optional, List

from .models import TaskMeta, TaskClass, TaskRisk


# ── Phase definitions ────────────────────────────────────────────────────────

class PipelinePhase:
    """Represents a single phase of the pipeline."""
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"


@dataclass
class PhaseConfig:
    """Configuration for a single pipeline phase."""
    name: str
    timeout_s: int = 300
    retry_on_failure: bool = False
    max_retries: int = 0
    required: bool = True


@dataclass
class FlowConfig:
    """Configuration for the complete pipeline flow."""
    flow_type: str  # "two_phase" or "three_phase"
    phases: List[PhaseConfig] = field(default_factory=list)
    task_class: str = ""
    risk: str = ""
    reason: str = ""


# ── Flow selection ───────────────────────────────────────────────────────────

# Task classes that always require three-phase
THREE_PHASE_TASK_CLASSES = {
    TaskClass.REPO_ARCHITECTURE_CHANGE,
    TaskClass.SWARM_CODE_TASK,
    TaskClass.MULTIMODAL_CODE_TASK,
}

# Risk levels that always require three-phase
THREE_PHASE_RISK_LEVELS = {
    TaskRisk.CRITICAL,
    TaskRisk.HIGH,
}


def requires_three_phase(task: TaskMeta) -> bool:
    """
    Determine if a task requires the 3-phase pipeline.

    3-phase is required when:
      - Task class is architecture change, swarm, or multimodal
      - Risk level is critical or high
      - Task explicitly requires multimodal input
    """
    if task.task_class in THREE_PHASE_TASK_CLASSES:
        return True
    if task.risk in THREE_PHASE_RISK_LEVELS:
        return True
    if task.requires_multimodal:
        return True
    return False


def select_flow(task: TaskMeta) -> FlowConfig:
    """
    Select the appropriate pipeline flow for a task.

    Returns a FlowConfig with the phases to execute.
    """
    if requires_three_phase(task):
        return _build_three_phase_flow(task)
    return _build_two_phase_flow(task)


def _build_two_phase_flow(task: TaskMeta) -> FlowConfig:
    """Build a 2-phase flow: Execute → Validate."""
    phases = [
        PhaseConfig(
            name=PipelinePhase.EXECUTE,
            timeout_s=300,
            retry_on_failure=True,
            max_retries=2,
            required=True,
        ),
        PhaseConfig(
            name=PipelinePhase.VALIDATE,
            timeout_s=60,
            retry_on_failure=False,
            max_retries=0,
            required=False,
        ),
    ]
    reason = f"two_phase: {task.task_class.value} task with {task.risk.value} risk"
    return FlowConfig(
        flow_type="two_phase",
        phases=phases,
        task_class=task.task_class.value,
        risk=task.risk.value,
        reason=reason,
    )


def _build_three_phase_flow(task: TaskMeta) -> FlowConfig:
    """Build a 3-phase flow: Plan → Execute → Validate."""
    reasons = []
    if task.task_class in THREE_PHASE_TASK_CLASSES:
        reasons.append(f"task_class={task.task_class.value}")
    if task.risk in THREE_PHASE_RISK_LEVELS:
        reasons.append(f"risk={task.risk.value}")
    if task.requires_multimodal:
        reasons.append("requires_multimodal")
    reason_str = ", ".join(reasons) if reasons else "unknown"

    phases = [
        PhaseConfig(
            name=PipelinePhase.PLAN,
            timeout_s=120,
            retry_on_failure=True,
            max_retries=1,
            required=True,
        ),
        PhaseConfig(
            name=PipelinePhase.EXECUTE,
            timeout_s=600,
            retry_on_failure=True,
            max_retries=2,
            required=True,
        ),
        PhaseConfig(
            name=PipelinePhase.VALIDATE,
            timeout_s=60,
            retry_on_failure=False,
            max_retries=0,
            required=False,
        ),
    ]
    return FlowConfig(
        flow_type="three_phase",
        phases=phases,
        task_class=task.task_class.value,
        risk=task.risk.value,
        reason=f"three_phase: {reason_str}",
    )


# ── Phase result tracking ───────────────────────────────────────────────────

@dataclass
class PhaseResult:
    """Result of a single pipeline phase."""
    phase: str
    success: bool
    latency_ms: int = 0
    error: Optional[str] = None
    output: Optional[dict] = None


@dataclass
class PipelineResult:
    """Result of the complete pipeline execution."""
    flow_type: str
    phases: List[PhaseResult] = field(default_factory=list)
    overall_success: bool = True
    failed_phase: Optional[str] = None
    total_latency_ms: int = 0

    def add_phase(self, result: PhaseResult):
        """Add a phase result and update overall status."""
        self.phases.append(result)
        self.total_latency_ms += result.latency_ms
        if not result.success:
            self.overall_success = False
            if self.failed_phase is None:
                self.failed_phase = result.phase

    @property
    def phase_names(self) -> List[str]:
        """Return list of phase names that were executed."""
        return [p.phase for p in self.phases]

    def get_phase(self, name: str) -> Optional[PhaseResult]:
        """Get the result for a specific phase."""
        for p in self.phases:
            if p.phase == name:
                return p
        return None
