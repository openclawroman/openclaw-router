"""Data models for the ai-code-runner router."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TaskType(str, Enum):
    IMPLEMENTATION = "implementation"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    REVIEW = "review"
    ARCHITECTURE = "architecture"
    SUPPORT = "support"


class TaskCriticality(str, Enum):
    CRITICAL = "critical"
    NORMAL = "normal"
    LOW = "low"


class TaskRisk(str, Enum):
    HIGH = "high_risk"
    MEDIUM = "medium"
    LOW = "low"


class CodexState(str, Enum):
    INCLUDED_FIRST = "included_first"
    INCLUDED_STRETCH = "included_stretch"
    LAST10 = "last10"
    EXHAUSTED = "exhausted"


class Executor(str, Enum):
    CODEX_CLI = "codex_cli"
    CLAUDE_CODE = "claude_code"
    OPENROUTER = "openrouter"


@dataclass
class TaskMeta:
    """Input task metadata from OpenClaw."""
    agent: str
    task_type: TaskType
    task_brief: str
    repo_path: str
    branch: str
    risk: TaskRisk
    criticality: TaskCriticality
    context_size: str  # "small", "medium", "large"


@dataclass
class RouteDecision:
    """Output decision from the router."""
    executor: Executor
    model: str
    reason: str
    attempted_fallback: bool = False
    fallback_from: Optional[str] = None
    status: str = "success"  # "success" or "error"
    error_type: Optional[str] = None