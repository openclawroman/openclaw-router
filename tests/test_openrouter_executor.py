"""Tests for OpenRouter executor adapter."""

import json
import os
import time
import urllib.error
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO

import pytest

from router.models import TaskMeta
from router.executors import (
    run_openrouter,
    run_codex_openrouter_minimax,
    run_codex_openrouter_kimi,
    _openrouter_request,
    _save_request_response,
)
from router.config_loader import get_model


def _make_meta(task_id="task-001", summary="Write a hello world function"):
    return TaskMeta(task_id=task_id, summary=summary)


def _mock_response(status=200, body=None):
    """Create a mock HTTP response."""
    if body is None:
        body = json.dumps({
            "id": "gen-abc123",
            "choices": [{"message": {"content": "print('hello')"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        })
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body.encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ── Model strings ──────────────────────────────────────────────────────────

class TestModelStrings:
    def test_minimax_model_is_m25(self):
        assert get_model("minimax") == "minimax/minimax-m2.7"

    def test_kimi_model(self):
        assert get_model("kimi") == "moonshotai/kimi-k2.5"


# ── File saving ────────────────────────────────────────────────────────────

class TestFileSaving:
    def test_save_request_response_creates_file(self):
        path = _save_request_response("test-task", '{"key": "value"}')
        assert path == "/tmp/test-task.stdout.txt"
        with open(path) as f:
            content = f.read()
        assert json.loads(content) == {"key": "value"}
        os.remove(path)


# ── Successful API call ───────────────────────────────────────────────────

class TestSuccessfulCall:
    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_success_returns_result(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        meta = _make_meta()
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        assert result.success is True
        assert result.exit_code == 0
        assert result.backend == "openrouter"
        assert result.model_profile == "openrouter_minimax"
        # Cleanup
        os.remove(result.stdout_ref)

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_request_id_is_generated(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        meta = _make_meta()
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        assert result.request_id is not None
        assert len(result.request_id) == 36  # UUID format
        os.remove(result.stdout_ref)

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_latency_ms_recorded(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        meta = _make_meta()
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        assert result.latency_ms >= 0
        os.remove(result.stdout_ref)

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_artifacts_populated(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        meta = _make_meta(task_id="art-test")
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        assert len(result.artifacts) == 1
        assert result.artifacts[0] == "/tmp/art-test.stdout.txt"
        assert result.stdout_ref == "/tmp/art-test.stdout.txt"
        os.remove(result.stdout_ref)

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_file_saved_on_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        meta = _make_meta(task_id="save-test")
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        assert os.path.exists("/tmp/save-test.stdout.txt")
        with open("/tmp/save-test.stdout.txt") as f:
            saved = json.loads(f.read())
        assert "request" in saved
        assert "response" in saved
        assert saved["request"]["model"] == "minimax/minimax-m2.7"
        assert saved["request"]["max_tokens"] == 4000
        os.remove(result.stdout_ref)


# ── Request payload structure ─────────────────────────────────────────────

class TestRequestPayload:
    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_payload_structure(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()

        meta = _make_meta(task_id="payload-test", summary="Fix the bug")
        _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))

        assert payload["model"] == "minimax/minimax-m2.7"
        assert payload["messages"] == [{"role": "user", "content": "Fix the bug"}]
        assert payload["max_tokens"] == 4000
        os.remove("/tmp/payload-test.stdout.txt")

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_request_method_is_post(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()

        meta = _make_meta(task_id="method-test")
        _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"
        os.remove("/tmp/method-test.stdout.txt")


# ── Error handling ─────────────────────────────────────────────────────────

class TestErrorHandling:
    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_401_returns_auth_error(self, mock_urlopen):
        err = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=BytesIO(b'{"error": "Invalid API key"}'),
        )
        mock_urlopen.side_effect = err
        meta = _make_meta(task_id="err-401")
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        assert result.success is False
        assert result.normalized_error == "auth_error"
        assert result.exit_code == 401
        assert result.stdout_ref == "/tmp/err-401.stdout.txt"
        assert os.path.exists("/tmp/err-401.stdout.txt")
        os.remove("/tmp/err-401.stdout.txt")

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_429_returns_rate_limited(self, mock_urlopen):
        err = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=BytesIO(b'{"error": "Rate limit exceeded"}'),
        )
        mock_urlopen.side_effect = err
        meta = _make_meta(task_id="err-429")
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        assert result.success is False
        assert result.normalized_error == "rate_limited"
        assert result.exit_code == 429
        os.remove("/tmp/err-429.stdout.txt")

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_503_returns_provider_unavailable(self, mock_urlopen):
        err = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=BytesIO(b'{"error": "Model unavailable"}'),
        )
        mock_urlopen.side_effect = err
        meta = _make_meta(task_id="err-503")
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        assert result.success is False
        assert result.normalized_error == "provider_unavailable"
        assert result.exit_code == 503
        os.remove("/tmp/err-503.stdout.txt")

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_url_error_returns_transient_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        meta = _make_meta(task_id="err-url")
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        assert result.success is False
        assert result.normalized_error == "transient_network_error"
        assert result.exit_code == -1
        os.remove("/tmp/err-url.stdout.txt")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_returns_auth_error(self):
        meta = _make_meta(task_id="err-nokey")
        result = run_codex_openrouter_minimax(meta)

        assert result.success is False
        assert result.normalized_error == "auth_error"
        assert result.exit_code == -1
        os.remove("/tmp/err-nokey.stdout.txt")


# ── Error files saved ─────────────────────────────────────────────────────

class TestErrorFileSaving:
    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_error_body_saved_to_file(self, mock_urlopen):
        err = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=BytesIO(b'{"error": "bad key"}'),
        )
        mock_urlopen.side_effect = err
        meta = _make_meta(task_id="save-err")
        result = _openrouter_request(meta, "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")

        with open("/tmp/save-err.stdout.txt") as f:
            saved = json.loads(f.read())
        assert "error" in saved
        assert saved["status_code"] == 401
        os.remove("/tmp/save-err.stdout.txt")


# ── Model selection ───────────────────────────────────────────────────────

class TestModelSelection:
    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_minimax_uses_m25_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        meta = _make_meta(task_id="sel-minimax")
        result = run_codex_openrouter_minimax(meta)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "minimax/minimax-m2.7"
        assert result.model_profile == "openrouter_minimax"
        os.remove(result.stdout_ref)

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_kimi_uses_k25_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        meta = _make_meta(task_id="sel-kimi")
        result = run_codex_openrouter_kimi(meta)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "moonshotai/kimi-k2.5"
        assert result.model_profile == "openrouter_kimi"
        os.remove(result.stdout_ref)

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_run_openrouter_default_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        meta = _make_meta(task_id="sel-default")
        result = run_openrouter(meta)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "minimax/minimax-m2.7"
        os.remove(result.stdout_ref)


# ── run_openrouter generic ────────────────────────────────────────────────

class TestRunOpenrouter:
    @patch("router.executors.urllib.request.urlopen")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
    def test_generic_openrouter_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        meta = _make_meta(task_id="generic-test")
        result = run_openrouter(meta)

        assert result.success is True
        assert result.tool == "openrouter"
        assert result.backend == "openrouter"
        os.remove(result.stdout_ref)
