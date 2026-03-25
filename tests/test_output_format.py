"""Tests for router.output_format — OutputFormat enum, validation, and resolution."""

import pytest

from router.models import TaskClass
from router.output_format import (
    OutputFormat,
    FormatValidationError,
    parse_output_format,
    validate_format_for_task,
    get_default_format,
    resolve_output_format,
    TASK_CLASS_FORMATS,
    DEFAULT_FORMAT,
)


# ── OutputFormat enum ────────────────────────────────────────────────────────

class TestOutputFormatEnum:
    def test_all_values_present(self):
        expected = {"diff", "files", "inline", "plan", "mixed"}
        actual = {f.value for f in OutputFormat}
        assert actual == expected

    def test_count(self):
        assert len(OutputFormat) == 5

    def test_values_are_lowercase(self):
        for fmt in OutputFormat:
            assert fmt.value == fmt.value.lower()


# ── parse_output_format ─────────────────────────────────────────────────────

class TestParseOutputFormat:
    @pytest.mark.parametrize("raw,expected", [
        ("diff", OutputFormat.DIFF),
        ("files", OutputFormat.FILES),
        ("inline", OutputFormat.INLINE),
        ("plan", OutputFormat.PLAN),
        ("mixed", OutputFormat.MIXED),
    ])
    def test_valid_formats(self, raw, expected):
        assert parse_output_format(raw) == expected

    def test_case_insensitive(self):
        assert parse_output_format("DIFF") == OutputFormat.DIFF
        assert parse_output_format("Files") == OutputFormat.FILES
        assert parse_output_format("PLAN") == OutputFormat.PLAN

    def test_whitespace_stripped(self):
        assert parse_output_format("  diff  ") == OutputFormat.DIFF

    def test_invalid_raises(self):
        with pytest.raises(FormatValidationError, match="Invalid output format"):
            parse_output_format("json")

    def test_empty_raises(self):
        with pytest.raises(FormatValidationError):
            parse_output_format("")

    def test_error_has_format_value(self):
        try:
            parse_output_format("xml")
        except FormatValidationError as e:
            assert e.format_value == "xml"
            assert e.task_class is None


# ── validate_format_for_task ────────────────────────────────────────────────

class TestValidateFormatForTask:
    """Test format-task compatibility matrix."""

    def test_diff_allowed_for_implementation(self):
        validate_format_for_task(OutputFormat.DIFF, TaskClass.IMPLEMENTATION)

    def test_files_allowed_for_implementation(self):
        validate_format_for_task(OutputFormat.FILES, TaskClass.IMPLEMENTATION)

    def test_inline_allowed_for_implementation(self):
        validate_format_for_task(OutputFormat.INLINE, TaskClass.IMPLEMENTATION)

    def test_plan_rejected_for_implementation(self):
        with pytest.raises(FormatValidationError, match="not compatible"):
            validate_format_for_task(OutputFormat.PLAN, TaskClass.IMPLEMENTATION)

    def test_mixed_rejected_for_implementation(self):
        with pytest.raises(FormatValidationError, match="not compatible"):
            validate_format_for_task(OutputFormat.MIXED, TaskClass.IMPLEMENTATION)

    def test_plan_only_for_code_review(self):
        validate_format_for_task(OutputFormat.PLAN, TaskClass.CODE_REVIEW)
        for fmt in [OutputFormat.DIFF, OutputFormat.FILES, OutputFormat.INLINE, OutputFormat.MIXED]:
            with pytest.raises(FormatValidationError):
                validate_format_for_task(fmt, TaskClass.CODE_REVIEW)

    def test_plan_and_inline_for_debug(self):
        validate_format_for_task(OutputFormat.PLAN, TaskClass.DEBUG)
        validate_format_for_task(OutputFormat.INLINE, TaskClass.DEBUG)
        for fmt in [OutputFormat.DIFF, OutputFormat.FILES, OutputFormat.MIXED]:
            with pytest.raises(FormatValidationError):
                validate_format_for_task(fmt, TaskClass.DEBUG)

    def test_diff_and_files_for_refactor(self):
        validate_format_for_task(OutputFormat.DIFF, TaskClass.REFACTOR)
        validate_format_for_task(OutputFormat.FILES, TaskClass.REFACTOR)
        for fmt in [OutputFormat.INLINE, OutputFormat.PLAN, OutputFormat.MIXED]:
            with pytest.raises(FormatValidationError):
                validate_format_for_task(fmt, TaskClass.REFACTOR)

    def test_files_and_inline_for_ui_from_screenshot(self):
        validate_format_for_task(OutputFormat.FILES, TaskClass.UI_FROM_SCREENSHOT)
        validate_format_for_task(OutputFormat.INLINE, TaskClass.UI_FROM_SCREENSHOT)
        for fmt in [OutputFormat.DIFF, OutputFormat.PLAN, OutputFormat.MIXED]:
            with pytest.raises(FormatValidationError):
                validate_format_for_task(fmt, TaskClass.UI_FROM_SCREENSHOT)

    def test_mixed_allowed_for_multimodal(self):
        validate_format_for_task(OutputFormat.MIXED, TaskClass.MULTIMODAL_CODE_TASK)

    def test_mixed_allowed_for_swarm(self):
        validate_format_for_task(OutputFormat.MIXED, TaskClass.SWARM_CODE_TASK)

    def test_architecture_change_allows_plan(self):
        validate_format_for_task(OutputFormat.PLAN, TaskClass.REPO_ARCHITECTURE_CHANGE)

    def test_all_task_classes_covered(self):
        """Every TaskClass must have an entry in TASK_CLASS_FORMATS."""
        for tc in TaskClass:
            assert tc in TASK_CLASS_FORMATS, f"Missing format config for {tc.value}"

    def test_error_includes_task_class(self):
        try:
            validate_format_for_task(OutputFormat.MIXED, TaskClass.CODE_REVIEW)
        except FormatValidationError as e:
            assert e.task_class == "code_review"
            assert e.format_value == "mixed"


# ── get_default_format ──────────────────────────────────────────────────────

class TestGetDefaultFormat:
    def test_implementation_defaults_to_diff(self):
        assert get_default_format(TaskClass.IMPLEMENTATION) == OutputFormat.DIFF

    def test_code_review_defaults_to_plan(self):
        assert get_default_format(TaskClass.CODE_REVIEW) == OutputFormat.PLAN

    def test_debug_defaults_to_plan(self):
        assert get_default_format(TaskClass.DEBUG) == OutputFormat.PLAN

    def test_ui_from_screenshot_defaults_to_files(self):
        assert get_default_format(TaskClass.UI_FROM_SCREENSHOT) == OutputFormat.FILES

    def test_all_task_classes_have_defaults(self):
        for tc in TaskClass:
            fmt = get_default_format(tc)
            assert isinstance(fmt, OutputFormat)

    def test_default_is_allowed_for_task(self):
        """The default format for each task class must be in its allowed set."""
        for tc in TaskClass:
            default = get_default_format(tc)
            allowed = TASK_CLASS_FORMATS[tc]
            assert default in allowed, (
                f"Default {default.value} not in allowed formats for {tc.value}"
            )


# ── resolve_output_format ───────────────────────────────────────────────────

class TestResolveOutputFormat:
    def test_none_returns_default(self):
        result = resolve_output_format(None, TaskClass.IMPLEMENTATION)
        assert result == OutputFormat.DIFF

    def test_valid_requested_format(self):
        result = resolve_output_format("files", TaskClass.IMPLEMENTATION)
        assert result == OutputFormat.FILES

    def test_invalid_format_raises(self):
        with pytest.raises(FormatValidationError, match="Invalid output format"):
            resolve_output_format("json", TaskClass.IMPLEMENTATION)

    def test_incompatible_format_raises(self):
        with pytest.raises(FormatValidationError, match="not compatible"):
            resolve_output_format("mixed", TaskClass.CODE_REVIEW)

    def test_case_insensitive_resolution(self):
        result = resolve_output_format("FILES", TaskClass.IMPLEMENTATION)
        assert result == OutputFormat.FILES
