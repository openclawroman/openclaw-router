"""Tests for secret redaction in logs and error messages."""

import json
import pytest

from router.secrets import sanitize_secrets, redact_dict
from router.errors import normalize_error
from router.attempt_logger import AttemptLogger, ExecutorAttempt, RoutingTrace


# ── sanitize_secrets ─────────────────────────────────────────────────────────

class TestSanitizeSecrets:

    def test_redacts_sk_keys(self):
        text = "key is sk-abcdefghijklmnopqrstuvwx"
        assert "sk-" not in sanitize_secrets(text)
        assert "[REDACTED]" in sanitize_secrets(text)

    def test_redacts_sk_ant_keys(self):
        text = "using sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
        assert "sk-ant-" not in sanitize_secrets(text)
        assert "[REDACTED]" in sanitize_secrets(text)

    def test_redacts_sk_or_keys(self):
        text = "openrouter key sk-or-v1-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
        assert "sk-or-" not in sanitize_secrets(text)
        assert "[REDACTED]" in sanitize_secrets(text)

    def test_redacts_bearer_tokens(self):
        text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234"
        assert "Bearer" not in sanitize_secrets(text) or "Bearer" not in sanitize_secrets(text)
        assert "[REDACTED]" in sanitize_secrets(text)

    def test_redacts_api_key_context(self):
        text = "error: api_key=supersecretvalue12345678"
        assert "supersecretvalue" not in sanitize_secrets(text)
        assert "[REDACTED]" in sanitize_secrets(text)

    def test_redacts_api_key_with_underscore(self):
        text = "received api_key sk-abcdefghijklmnopqrstuvwx"
        assert "sk-" not in sanitize_secrets(text)
        assert "[REDACTED]" in sanitize_secrets(text)

    def test_preserves_non_secret_text(self):
        text = "Request to model gpt-4 failed with status 500"
        assert sanitize_secrets(text) == text

    def test_preserves_short_strings(self):
        text = "hello world"
        assert sanitize_secrets(text) == text

    def test_empty_string(self):
        assert sanitize_secrets("") == ""

    def test_none_passthrough(self):
        assert sanitize_secrets(None) is None

    def test_multiple_secrets_in_one_string(self):
        text = "sk-abcdefghijklmnopqrstuvwx and sk-ant-api03-QRSTUVWXYZabcdef123456"
        result = sanitize_secrets(text)
        assert "sk-" not in result
        assert result.count("[REDACTED]") == 2


# ── redact_dict ──────────────────────────────────────────────────────────────

class TestRedactDict:

    def test_redacts_sensitive_keys(self):
        data = {"api_key": "sk-abcdefghijklmnopqrstuvwx", "name": "test"}
        result = redact_dict(data)
        assert result["api_key"] == "[REDACTED]"
        assert result["name"] == "test"

    def test_redacts_token_key(self):
        data = {"token": "Bearer abcdefghijklmnopqrstuvwxyz1234"}
        result = redact_dict(data)
        assert result["token"] == "[REDACTED]"

    def test_redacts_authorization_key(self):
        data = {"authorization": "Bearer abcdefghijklmnopqrstuvwxyz1234"}
        result = redact_dict(data)
        assert result["authorization"] == "[REDACTED]"

    def test_redacts_secret_key(self):
        data = {"secret": "mysecretvalue1234567890"}
        result = redact_dict(data)
        assert result["secret"] == "[REDACTED]"

    def test_redacts_credential_key(self):
        data = {"credential": "some-credential-value"}
        result = redact_dict(data)
        assert result["credential"] == "[REDACTED]"

    def test_case_insensitive_keys(self):
        data = {"API_KEY": "sk-abcdefghijklmnopqrstuvwx", "Token": "abc"}
        result = redact_dict(data)
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Token"] == "[REDACTED]"

    def test_preserves_non_sensitive_keys(self):
        data = {"model": "gpt-4", "task_id": "abc123", "success": True, "latency_ms": 500}
        result = redact_dict(data)
        assert result == data

    def test_redacts_nested_dicts(self):
        data = {
            "request": {
                "api_key": "sk-abcdefghijklmnopqrstuvwx",
                "model": "gpt-4",
            }
        }
        result = redact_dict(data)
        assert result["request"]["api_key"] == "[REDACTED]"
        assert result["request"]["model"] == "gpt-4"

    def test_redacts_list_of_dicts(self):
        data = {
            "attempts": [
                {"tool": "codex", "token": "secret123456789012345"},
                {"tool": "claude", "api_key": "sk-abcdefghijklmnopqrstuvwx"},
            ]
        }
        result = redact_dict(data)
        assert result["attempts"][0]["token"] == "[REDACTED]"
        assert result["attempts"][1]["api_key"] == "[REDACTED]"

    def test_sanitizes_string_values(self):
        data = {"error": "Auth failed with sk-abcdefghijklmnopqrstuvwx"}
        result = redact_dict(data)
        assert "sk-" not in result["error"]
        assert "[REDACTED]" in result["error"]

    def test_preserves_non_string_values(self):
        data = {"count": 42, "flag": True, "rate": 3.14, "items": None}
        result = redact_dict(data)
        assert result == data


# ── normalize_error ──────────────────────────────────────────────────────────

class TestNormalizeErrorRedaction:

    def test_no_api_key_leak(self):
        raw = "Auth failed with api_key=sk-abcdefghijklmnopqrstuvwx"
        result = normalize_error(raw)
        # Should still classify as auth_error, but the raw string processing
        # should not expose the key
        assert result == "auth_error"

    def test_no_bearer_leak(self):
        raw = "Unauthorized: Bearer abcdefghijklmnopqrstuvwxyz1234 expired"
        result = normalize_error(raw)
        assert result == "auth_error"

    def test_preserves_normal_errors(self):
        raw = "Model not found: gpt-99"
        result = normalize_error(raw)
        assert result == "model_not_found"


# ── AttemptLogger integration ────────────────────────────────────────────────

class TestLogTraceRedaction:

    def test_log_entries_no_api_keys(self, tmp_path):
        """When a trace contains secrets, the written JSONL must not."""
        log_path = tmp_path / "test.jsonl"
        logger = AttemptLogger(log_path=log_path)

        trace = RoutingTrace(
            trace_id="abc123",
            task_id="t1",
            state="openai_primary",
            chain=[{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_gpt54_mini"}],
            attempts=[
                ExecutorAttempt(
                    tool="codex_cli",
                    backend="openai_native",
                    model_profile="codex_gpt54_mini",
                    success=False,
                    latency_ms=150,
                    normalized_error="Auth failed with sk-abcdefghijklmnopqrstuvwx",
                )
            ],
            total_latency_ms=150,
            final_tool="codex_cli",
            final_success=False,
            final_error="sk-abcdefghijklmnopqrstuvwx",
        )
        logger.log_trace(trace)

        raw = log_path.read_text()
        assert "sk-" not in raw
        assert "[REDACTED]" in raw

        # Verify it's still valid JSONL
        entry = json.loads(raw.strip().split("\n")[0])
        assert entry["type"] == "routing_trace"
        assert entry["trace_id"] == "abc123"

    def test_normal_traces_unchanged(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        logger = AttemptLogger(log_path=log_path)

        trace = RoutingTrace(
            trace_id="abc123",
            task_id="t1",
            state="openai_primary",
            chain=[{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_gpt54_mini"}],
            attempts=[
                ExecutorAttempt(
                    tool="codex_cli",
                    backend="openai_native",
                    model_profile="codex_gpt54_mini",
                    success=True,
                    latency_ms=150,
                )
            ],
            total_latency_ms=150,
            final_tool="codex_cli",
            final_success=True,
        )
        logger.log_trace(trace)

        raw = log_path.read_text()
        entry = json.loads(raw.strip().split("\n")[0])
        assert entry["trace_id"] == "abc123"
        assert entry["final_success"] is True
