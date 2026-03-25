"""Data models for the ai-code-runner router."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class TaskClass(str, Enum):
    """High-level task classification."""
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


class TaskRisk(str, Enum):
    """Risk level of the task."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CodexState(str, Enum):
    """Codex usage state — 4 subscription-budget-driven states."""
    OPENAI_PRIMARY = "openai_primary"
    OPENAI_CONSERVATION = "openai_conservation"
    CLAUDE_BACKUP = "claude_backup"
    OPENROUTER_FALLBACK = "openrouter_fallback"

    # Backward compat aliases
    NORMAL = "openai_primary"
    LAST10 = "claude_backup"


class ErrorType(str, Enum):
    """Error type classification."""
    AUTH_ERROR = "auth_error"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"


class Executor(str, Enum):
    """Executor tool."""
    CODEX_CLI = "codex_cli"
    CLAUDE_CODE = "claude_code"
    OPENROUTER = "openrouter"


class ExecutorBackend(str, Enum):
    """Backend provider for an executor."""
    OPENAI_NATIVE = "openai_native"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class ModelProfile(str, Enum):
    """Model profile identifier."""
    CODEX_PRIMARY = "codex_primary"
    CODEX_GPT54 = "codex_gpt54"
    CODEX_GPT54_MINI = "codex_gpt54_mini"
    CLAUDE_PRIMARY = "claude_primary"
    CLAUDE_SONNET = "claude_sonnet"
    CLAUDE_OPUS = "claude_opus"
    OPENROUTER_MINIMAX = "openrouter_minimax"
    OPENROUTER_KIMI = "openrouter_kimi"
    OPENROUTER_MIMO = "openrouter_mimo"
    OPENROUTER_DYNAMIC = "openrouter_dynamic"


@dataclass
class ChainEntry:
    """A single entry in the routing chain."""
    tool: str
    backend: str
    model_profile: str


@dataclass
class TaskMeta:
    """Input task metadata from OpenClaw."""
    task_id: str = ""
    agent: str = "coder"  # coder|reviewer|architect|designer|worker
    task_class: TaskClass = TaskClass.IMPLEMENTATION
    risk: TaskRisk = TaskRisk.MEDIUM
    modality: TaskModality = TaskModality.TEXT
    requires_repo_write: bool = False
    requires_multimodal: bool = False
    has_screenshots: bool = False
    swarm: bool = False
    repo_path: str = ""
    cwd: str = ""
    summary: str = ""


@dataclass
class RouteDecision:
    """Output decision from the router — which chain was chosen and why."""
    task_id: str = ""
    state: str = "openai_primary"
    chain: List[ChainEntry] = field(default_factory=list)
    reason: str = ""
    attempted_fallback: bool = False
    fallback_from: Optional[str] = None
    providers_skipped: List[str] = field(default_factory=list)
    chain_timed_out: bool = False
    fallback_count: int = 0
    trace_id: str = ""


@dataclass
class ExecutorResult:
    """Result from an executor execution — outcome of running a tool."""
    task_id: str = ""
    tool: str = ""
    backend: str = ""
    model_profile: str = ""
    success: bool = True
    normalized_error: Optional[str] = None
    exit_code: Optional[int] = None
    latency_ms: int = 0
    request_id: Optional[str] = None
    cost_estimate_usd: Optional[float] = None
    artifacts: List[str] = field(default_factory=list)
    stdout_ref: Optional[str] = None
    stderr_ref: Optional[str] = None
    final_summary: Optional[str] = None
    trace_id: str = ""
    rate_limit_info: Optional[dict] = None  # Serialized RateLimitInfo
