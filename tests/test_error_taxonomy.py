"""Tests for new error taxonomy types (item 5.1)."""

import pytest

from router.errors import (
    ExecutorError,
    normalize_error,
    can_fallback,
    ELIGIBLE_FALLBACK_ERRORS,
    NON_ELIGIBLE_ERRORS,
    NORMALIZED_ERROR_TYPES,
    ModelNotFoundError,
    ContextTooLongError,
    ContentFilteredError,
)


# ── Error class properties ───────────────────────────────────────────────────

class TestNewErrorTypes:
    def test_model_not_found_error_type(self):
        err = ModelNotFoundError()
        assert err.error_type == "model_not_found"

    def test_context_too_long_error_type(self):
        err = ContextTooLongError()
        assert err.error_type == "context_too_long"

    def test_content_filtered_error_type(self):
        err = ContentFilteredError()
        assert err.error_type == "content_filtered"

    def test_all_are_executor_errors(self):
        for cls in (ModelNotFoundError, ContextTooLongError, ContentFilteredError):
            assert issubclass(cls, ExecutorError)

    def test_custom_messages(self):
        assert str(ModelNotFoundError("gpt-5 missing")) == "gpt-5 missing"
        assert str(ContextTooLongError("128k limit")) == "128k limit"
        assert str(ContentFilteredError("nsfw blocked")) == "nsfw blocked"


# ── Fallback eligibility ─────────────────────────────────────────────────────

class TestFallbackEligibility:
    def test_model_not_found_is_eligible(self):
        assert "model_not_found" in ELIGIBLE_FALLBACK_ERRORS
        assert can_fallback("model_not_found") is True

    def test_context_too_long_not_eligible(self):
        assert "context_too_long" in NON_ELIGIBLE_ERRORS
        assert can_fallback("context_too_long") is False

    def test_content_filtered_not_eligible(self):
        assert "content_filtered" in NON_ELIGIBLE_ERRORS
        assert can_fallback("content_filtered") is False

    def test_all_new_types_in_normalized_set(self):
        assert "model_not_found" in NORMALIZED_ERROR_TYPES
        assert "context_too_long" in NORMALIZED_ERROR_TYPES
        assert "content_filtered" in NORMALIZED_ERROR_TYPES


# ── Updated set sizes ────────────────────────────────────────────────────────

class TestUpdatedSetSizes:
    def test_eligible_fallback_errors_count(self):
        assert len(ELIGIBLE_FALLBACK_ERRORS) == 7

    def test_non_eligible_errors_count(self):
        assert len(NON_ELIGIBLE_ERRORS) == 9

    def test_normalized_error_types_count(self):
        assert len(NORMALIZED_ERROR_TYPES) == 16


# ── normalize_error: model_not_found ─────────────────────────────────────────

class TestNormalizeModelNotFound:
    @pytest.mark.parametrize("msg", [
        "model not found",
        "Model-Not-Found: gpt-5",
        "model not available",
        "model_not available",
        "no such model",
        "model does not exist",
        "model xyz not recognized",
    ])
    def test_model_not_found_patterns(self, msg):
        assert normalize_error(msg) == "model_not_found"


# ── normalize_error: context_too_long ────────────────────────────────────────

class TestNormalizeContextTooLong:
    @pytest.mark.parametrize("msg", [
        "context too long",
        "context-too-long",
        "maximum context length exceeded",
        "token limit exceeded",
        "input too long for model",
        "context length 200000 exceeds max",
        "max tokens exceeded",
    ])
    def test_context_too_long_patterns(self, msg):
        assert normalize_error(msg) == "context_too_long"


# ── normalize_error: content_filtered ────────────────────────────────────────

class TestNormalizeContentFiltered:
    @pytest.mark.parametrize("msg", [
        "content filter triggered",
        "safety filter violation",
        "content policy violation",
        "request was flagged",
        "blocked content detected",
        "response contained harmful material",
    ])
    def test_content_filtered_patterns(self, msg):
        assert normalize_error(msg) == "content_filtered"


# ── normalize_error: ExecutorError passthrough ───────────────────────────────

class TestNormalizeNewExecutorErrors:
    def test_model_not_found_passthrough(self):
        err = ModelNotFoundError()
        assert normalize_error(err) == "model_not_found"

    def test_context_too_long_passthrough(self):
        err = ContextTooLongError()
        assert normalize_error(err) == "context_too_long"

    def test_content_filtered_passthrough(self):
        err = ContentFilteredError()
        assert normalize_error(err) == "content_filtered"
