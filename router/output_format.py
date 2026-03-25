"""Output format definitions and validation for ai-code-runner.

Supported output formats:
  - diff:     Unified diff (patch format)
  - files:    Full file contents (write entire files)
  - inline:   Inline code snippets (insert at specific locations)
  - plan:     Planning/recommendation text only (no code changes)
  - mixed:    Combination of the above

Output format validation ensures:
  - Format is a valid OutputFormat enum value
  - Format is compatible with the task class
  - Format is compatible with the executor capabilities
"""

from enum import Enum
from typing import Optional, Set

from .models import TaskClass


class OutputFormat(Enum):
    """Supported output formats for code runner results."""
    DIFF = "diff"
    FILES = "files"
    INLINE = "inline"
    PLAN = "plan"
    MIXED = "mixed"


# ── Task class → allowed output formats ─────────────────────────────────────

TASK_CLASS_FORMATS: dict[TaskClass, Set[OutputFormat]] = {
    TaskClass.IMPLEMENTATION:          {OutputFormat.DIFF, OutputFormat.FILES, OutputFormat.INLINE},
    TaskClass.REFACTOR:                {OutputFormat.DIFF, OutputFormat.FILES},
    TaskClass.BUGFIX:                  {OutputFormat.DIFF, OutputFormat.FILES, OutputFormat.INLINE},
    TaskClass.DEBUG:                   {OutputFormat.PLAN, OutputFormat.INLINE},
    TaskClass.CODE_REVIEW:             {OutputFormat.PLAN},
    TaskClass.TEST_GENERATION:         {OutputFormat.DIFF, OutputFormat.FILES, OutputFormat.INLINE},
    TaskClass.REPO_ARCHITECTURE_CHANGE:{OutputFormat.DIFF, OutputFormat.FILES, OutputFormat.PLAN},
    TaskClass.UI_FROM_SCREENSHOT:      {OutputFormat.FILES, OutputFormat.INLINE},
    TaskClass.MULTIMODAL_CODE_TASK:    {OutputFormat.FILES, OutputFormat.INLINE, OutputFormat.MIXED},
    TaskClass.SWARM_CODE_TASK:         {OutputFormat.DIFF, OutputFormat.FILES, OutputFormat.MIXED},
}

# Default format per task class
DEFAULT_FORMAT: dict[TaskClass, OutputFormat] = {
    TaskClass.IMPLEMENTATION:           OutputFormat.DIFF,
    TaskClass.REFACTOR:                 OutputFormat.DIFF,
    TaskClass.BUGFIX:                   OutputFormat.DIFF,
    TaskClass.DEBUG:                    OutputFormat.PLAN,
    TaskClass.CODE_REVIEW:              OutputFormat.PLAN,
    TaskClass.TEST_GENERATION:          OutputFormat.DIFF,
    TaskClass.REPO_ARCHITECTURE_CHANGE: OutputFormat.PLAN,
    TaskClass.UI_FROM_SCREENSHOT:       OutputFormat.FILES,
    TaskClass.MULTIMODAL_CODE_TASK:     OutputFormat.FILES,
    TaskClass.SWARM_CODE_TASK:          OutputFormat.DIFF,
}


class FormatValidationError(Exception):
    """Raised when output format validation fails."""

    def __init__(self, message: str, format_value: Optional[str] = None, task_class: Optional[str] = None):
        super().__init__(message)
        self.format_value = format_value
        self.task_class = task_class


def parse_output_format(value: str) -> OutputFormat:
    """
    Parse a string into an OutputFormat enum value.

    Raises FormatValidationError if the value is not a valid format.
    """
    try:
        return OutputFormat(value.lower().strip())
    except ValueError:
        valid = [f.value for f in OutputFormat]
        raise FormatValidationError(
            f"Invalid output format: {value!r}. Valid formats: {valid}",
            format_value=value,
        )


def validate_format_for_task(output_format: OutputFormat, task_class: TaskClass) -> None:
    """
    Validate that an output format is compatible with a task class.

    Raises FormatValidationError if the format is not compatible.
    """
    allowed = TASK_CLASS_FORMATS.get(task_class)
    if allowed is None:
        raise FormatValidationError(
            f"Unknown task class: {task_class.value!r}",
            task_class=task_class.value,
        )
    if output_format not in allowed:
        allowed_values = sorted(f.value for f in allowed)
        raise FormatValidationError(
            f"Output format {output_format.value!r} is not compatible with "
            f"task class {task_class.value!r}. Allowed: {allowed_values}",
            format_value=output_format.value,
            task_class=task_class.value,
        )


def get_default_format(task_class: TaskClass) -> OutputFormat:
    """
    Get the default output format for a task class.

    Falls back to DIFF if no default is configured.
    """
    return DEFAULT_FORMAT.get(task_class, OutputFormat.DIFF)


def resolve_output_format(
    requested: Optional[str],
    task_class: TaskClass,
) -> OutputFormat:
    """
    Resolve the output format for a task.

    If requested is None, returns the default for the task class.
    If requested is provided, parses and validates it against the task class.

    Raises FormatValidationError on invalid or incompatible format.
    """
    if requested is None:
        return get_default_format(task_class)

    fmt = parse_output_format(requested)
    validate_format_for_task(fmt, task_class)
    return fmt
