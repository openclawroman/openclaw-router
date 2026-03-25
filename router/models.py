"""Data models for the ai-code-runner router."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class TaskClass(str, Enum):
    """High-level task classification (replaces task_type)."""
    IMPLEMENTATION = "implementation"
    REFACTOR = "refactor"
    BUGFIX = "bugfix"
    DEBUG = "debug"
    CODE_REVIEW = "code_review"
    TEST_GENERATION = "test_generation"
    REPO_ARCHITECTURE_CHANGE = "repo_architecture_change"
    UI_FROM_SCREENSHOT = "ui_from_screenshot"
    MULTIMODAL_CODE_TASK = "multimodal_code_task"
    SWARM_CODE_TASK = "swarm_code_task"


class TaskModality(str, Enum):
    """Modality of the task input/output."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    MIXED = "mixed"


class ExecutorBackend(str, Enum):
    """Backend provider for an executor."""
    OPENAI_NATIVE = "openai_native"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class ModelProfile(str, Enum):
    """Model profile identifier."""
    CODEX_PRIMARY = "codex_primary"
    CLAUDE_PRIMARY = "claude_primary"
    OPENROUTER_MINIMAX = "openrouter_minimax"
    OPENROUTER_KIMI = "openrouter_kimi"


class TaskRisk(str, Enum):
    HIGH = "high_risk"
    MEDIUM = "medium"
    LOW = "low"


class TaskCriticality(str, Enum):
    CRITICAL = "critical"
    NORMAL = "normal"
    LOW = "low"


class CodexState(str, Enum):
    NORMAL = "normal"
    LAST10 = "last10"


class Executor(str, Enum):
    CODEX_CLI = "codex_cli"
    CLAUDE_CODE = "claude_code"
    OPENROUTER = "openrouter"


@dataclass
class TaskMeta:
    """Input task metadata from OpenClaw."""
    agent: str
    task_class: TaskClass  # renamed from task_type
    task_brief: str
    repo_path: str
    branch: str
    risk: TaskRisk
    criticality: TaskCriticality
    context_size: str  # "small", "medium", "large"
    # New fields
    task_id: str = ""
    modality: TaskModality = TaskModality.TEXT
    requires_multimodal: bool = False
    has_screenshots: bool = False
    swarm: bool = False
    cwd: str = ""


@dataclass
class RouteDecision:
    """Output decision from the router."""
    task_id: str = ""
    executor: Executor = Executor.CODEX_CLI
    model: str = ""
    chain: list = field(default_factory=list)  # list of executor names in chain
    reason: str = ""
    attempted_fallback: bool = False
    fallback_from: Optional[str] = None
    status: str = "success"  # "success" or "error"
    error_type: Optional[str] = None
    # New fields
    tool: Optional[str] = None
    backend: Optional[ExecutorBackend] = None
    model_profile: Optional[ModelProfile] = None
    normalized_error: Optional[str] = None
    exit_code: Optional[int] = None
    request_id: Optional[str] = None
    cost_estimate_usd: Optional[float] = None
    artifacts: list = field(default_factory=list)
    stdout_ref: Optional[str] = None
    stderr_ref: Optional[str] = None
    final_summary: Optional[str] = None


@dataclass
class ExecutorResult:
    """Result from an executor execution."""
    task_id: str = ""
    tool: str = ""  # executor name that was run
    backend: ExecutorBackend = ExecutorBackend.OPENROUTER
    model_profile: ModelProfile = ModelProfile.OPENROUTER_MINIMAX
    success: bool = True
    normalized_error: Optional[str] = None
    exit_code: Optional[int] = None
    request_id: Optional[str] = None
    cost_estimate_usd: Optional[float] = None
    artifacts: list = field(default_factory=list)
    stdout_ref: Optional[str] = None
    stderr_ref: Optional[str] = None
    final_summary: Optional[str] = None
