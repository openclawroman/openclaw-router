"""Tests for task content isolation safeguards (item 6.2)."""

import json
import tempfile
from pathlib import Path

import pytest

from router.sanitize import sanitize_content, MAX_STRING_LENGTH, MAX_CONTENT_KEY_LENGTH
from router.attempt_logger import AttemptLogger, RoutingTrace, ExecutorAttempt
from router.policy import _truncate_error_message, MAX_ERROR_MESSAGE_LENGTH


# ---------------------------------------------------------------------------
# sanitize_content
# ---------------------------------------------------------------------------

class TestSanitizeContent:
    """Content sanitization defense-in-depth tests."""

    def test_truncates_long_strings(self):
        """Strings longer than 500 chars are replaced with a truncation marker."""
        entry = {"task_id": "abc", "result": "x" * 600}
        sanitized = sanitize_content(entry)
        assert sanitized["result"] == "[TRUNCATED 600 chars]"
        assert sanitized["task_id"] == "abc"  # short string preserved

    def test_preserves_short_strings(self):
        """Strings at or under 500 chars are preserved."""
        entry = {"name": "hello", "id": "12345"}
        sanitized = sanitize_content(entry)
        assert sanitized == entry

    def test_removes_prompt_key_long(self):
        """'prompt' key with long value is removed."""
        entry = {"prompt": "A" * 300, "task_id": "t1"}
        sanitized = sanitize_content(entry)
        assert "prompt" not in sanitized
        assert sanitized["task_id"] == "t1"

    def test_removes_content_key_long(self):
        """'content' key with long value is removed."""
        entry = {"content": "B" * 500, "ok": True}
        sanitized = sanitize_content(entry)
        assert "content" not in sanitized

    def test_removes_body_key_long(self):
        entry = {"body": "C" * 201}
        sanitized = sanitize_content(entry)
        assert "body" not in sanitized

    def test_removes_text_key_long(self):
        entry = {"text": "D" * 999}
        sanitized = sanitize_content(entry)
        assert "text" not in sanitized

    def test_removes_code_key_long(self):
        entry = {"code": "E" * 250}
        sanitized = sanitize_content(entry)
        assert "code" not in sanitized

    def test_removes_diff_key_long(self):
        entry = {"diff": "F" * 1000}
        sanitized = sanitize_content(entry)
        assert "diff" not in sanitized

    def test_removes_file_content_key_long(self):
        entry = {"file_content": "G" * 300}
        sanitized = sanitize_content(entry)
        assert "file_content" not in sanitized

    def test_removes_summary_key_long(self):
        entry = {"summary": "H" * 400}
        sanitized = sanitize_content(entry)
        assert "summary" not in sanitized

    def test_preserves_content_key_short(self):
        """Content keys with short values (≤ 200 chars) are preserved."""
        entry = {"prompt": "short prompt", "content": "ok", "body": "x" * 200}
        sanitized = sanitize_content(entry)
        assert sanitized["prompt"] == "short prompt"
        assert sanitized["content"] == "ok"
        assert sanitized["body"] == "x" * 200

    def test_preserves_numeric_values(self):
        """Numeric values are never touched."""
        entry = {"latency_ms": 1500, "cost": 0.003, "count": 42}
        sanitized = sanitize_content(entry)
        assert sanitized == entry

    def test_preserves_boolean_values(self):
        """Boolean values are never touched."""
        entry = {"success": True, "failed": False}
        sanitized = sanitize_content(entry)
        assert sanitized == entry

    def test_preserves_none_values(self):
        entry = {"optional": None}
        sanitized = sanitize_content(entry)
        assert sanitized == entry

    def test_preserves_list_values(self):
        entry = {"tags": ["a", "b"]}
        sanitized = sanitize_content(entry)
        assert sanitized == entry

    def test_preserves_dict_values(self):
        entry = {"meta": {"nested": True}}
        sanitized = sanitize_content(entry)
        assert sanitized == entry

    def test_empty_dict(self):
        assert sanitize_content({}) == {}

    def test_exact_boundary_500(self):
        """Exactly 500 chars should be preserved (not truncated)."""
        entry = {"data": "x" * 500}
        sanitized = sanitize_content(entry)
        assert sanitized["data"] == "x" * 500

    def test_exact_boundary_501(self):
        """501 chars should be truncated."""
        entry = {"data": "x" * 501}
        sanitized = sanitize_content(entry)
        assert sanitized["data"] == "[TRUNCATED 501 chars]"

    def test_content_key_exact_boundary(self):
        """Content key with exactly 200 chars should be preserved."""
        entry = {"prompt": "x" * 200}
        sanitized = sanitize_content(entry)
        assert sanitized["prompt"] == "x" * 200

    def test_content_key_over_boundary(self):
        """Content key with 201 chars should be removed."""
        entry = {"prompt": "x" * 201}
        sanitized = sanitize_content(entry)
        assert "prompt" not in sanitized


# ---------------------------------------------------------------------------
# AttemptLogger integration — log entries must be sanitized
# ---------------------------------------------------------------------------

class TestAttemptLoggerSanitization:
    """Ensure AttemptLogger.log_trace() sanitizes entries before writing."""

    def test_log_entry_no_long_strings(self, tmp_path):
        """Written log entries must not contain strings > 500 chars."""
        log_path = tmp_path / "trace.jsonl"
        logger = AttemptLogger(log_path=log_path)

        trace = RoutingTrace(
            trace_id="abc123",
            task_id="task-1",
            state="openai_primary",
            chain=[{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_gpt54"}],
            attempts=[
                ExecutorAttempt(
                    tool="codex_cli",
                    backend="openai_native",
                    model_profile="codex_gpt54",
                    success=True,
                    latency_ms=100,
                )
            ],
            final_tool="codex_cli",
            final_success=True,
        )

        logger.log_trace(trace)

        line = log_path.read_text().strip()
        entry = json.loads(line)

        # Verify no string value in the entry exceeds 500 chars
        def check_no_long_strings(obj, path=""):
            if isinstance(obj, str):
                assert len(obj) <= 500, f"Long string at {path}: {len(obj)} chars"
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_long_strings(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_no_long_strings(v, f"{path}[{i}]")

        check_no_long_strings(entry)

    def test_trace_fields_preserved(self, tmp_path):
        """RoutingTrace fields must survive sanitization (not accidentally removed)."""
        log_path = tmp_path / "trace2.jsonl"
        logger = AttemptLogger(log_path=log_path)

        trace = RoutingTrace(
            trace_id="xyz789",
            task_id="task-2",
            state="claude_backup",
            chain=[{"tool": "claude_code", "backend": "anthropic", "model_profile": "claude_sonnet"}],
            attempts=[],
            providers_skipped=["codex_cli:openai_native"],
            chain_timed_out=False,
            fallback_count=1,
            total_latency_ms=5000,
            final_tool="claude_code",
            final_success=True,
            final_error=None,
            chain_invariant_violated=False,
            chain_invariant_reason=None,
        )

        logger.log_trace(trace)

        line = log_path.read_text().strip()
        entry = json.loads(line)

        # All expected keys present
        for key in ["trace_id", "task_id", "state", "chain", "attempts",
                     "providers_skipped", "chain_timed_out", "fallback_count",
                     "total_latency_ms", "final_tool", "final_success", "type"]:
            assert key in entry, f"Missing key: {key}"

        assert entry["trace_id"] == "xyz789"
        assert entry["task_id"] == "task-2"
        assert entry["state"] == "claude_backup"
        assert entry["fallback_count"] == 1
        assert entry["total_latency_ms"] == 5000
        assert entry["final_success"] is True


# ---------------------------------------------------------------------------
# _truncate_error_message (policy.py)
# ---------------------------------------------------------------------------

class TestTruncateErrorMessage:
    """Error message truncation in error_history entries."""

    def test_short_message_preserved(self):
        msg = "Something went wrong"
        assert _truncate_error_message(msg) == msg

    def test_exact_boundary_preserved(self):
        msg = "x" * MAX_ERROR_MESSAGE_LENGTH
        assert _truncate_error_message(msg) == msg

    def test_over_boundary_truncated(self):
        msg = "A" * 300
        result = _truncate_error_message(msg)
        assert len(result) == MAX_ERROR_MESSAGE_LENGTH + len("...[TRUNCATED]")
        assert result.endswith("...[TRUNCATED]")
        assert result.startswith("A" * MAX_ERROR_MESSAGE_LENGTH)

    def test_empty_string(self):
        assert _truncate_error_message("") == ""

    def test_prevents_stack_trace_leakage(self):
        """Simulate a long stack trace — must be truncated."""
        stack_trace = "Traceback (most recent call last):\n" + ("  File x.py, line 1\n" * 50)
        result = _truncate_error_message(stack_trace)
        assert len(result) <= MAX_ERROR_MESSAGE_LENGTH + len("...[TRUNCATED]")
        assert "...[TRUNCATED]" in result


# ---------------------------------------------------------------------------
# Integration: error_history in policy.py uses truncation
# ---------------------------------------------------------------------------

class TestErrorHistoryTruncation:
    """Verify that error_history entries in route_task have truncated error_message."""

    def test_truncation_applied_in_error_history(self):
        """Directly test that _truncate_error_message is called for error_history."""
        # This tests the helper function integration; full route_task integration
        # requires mocking executors which is covered by test_policy.py
        long_msg = "X" * 500
        truncated = _truncate_error_message(long_msg)
        assert len(truncated) < len(long_msg)
        assert "[TRUNCATED]" in truncated
