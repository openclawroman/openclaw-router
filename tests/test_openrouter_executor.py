"""Tests for OpenRouter executor adapter."""

import json
import os
import urllib.error
from unittest.mock import patch, MagicMock
from io import BytesIO
import pytest

from router.models import TaskMeta
from router.executors import run_openrouter, run_codex_openrouter_minimax, run_codex_openrouter_kimi, _openrouter_request, _save_request_response
from router.config_loader import get_model


def _make_meta(task_id="task-001", summary="Write a hello world function"):
    return TaskMeta(task_id=task_id, summary=summary)


def _mock_response(status=200, body=None):
    if body is None:
        body = json.dumps({"id": "gen-abc123", "choices": [{"message": {"content": "print('hello')"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body.encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_http_error(code, msg, body):
    return urllib.error.HTTPError(url="https://openrouter.ai/api/v1/chat/completions", code=code, msg=msg, hdrs={}, fp=BytesIO(body.encode()))


def _patch_urlopen(monkeypatch, return_value=None, side_effect=None):
    p = patch("router.executors.urllib.request.urlopen")
    mock_urlopen = p.start()
    if side_effect:
        mock_urlopen.side_effect = side_effect
    else:
        mock_urlopen.return_value = return_value
    monkeypatch.setattr("router.executors.urllib.request.urlopen", mock_urlopen)
    return mock_urlopen


class TestModelStrings:
    @pytest.mark.parametrize("name,expected", [("minimax", "minimax/minimax-m2.7"), ("kimi", "moonshotai/kimi-k2.5")])
    def test_model_strings(self, name, expected):
        assert get_model(name) == expected


class TestFileSaving:
    def test_save_request_response_creates_file(self):
        path = _save_request_response("test-task", '{"key": "value"}')
        assert path == "/tmp/test-task.stdout.txt"
        with open(path) as f:
            assert json.loads(f.read()) == {"key": "value"}
        os.remove(path)


@patch("router.executors.urllib.request.urlopen")
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
class TestSuccessfulCall:
    def test_success_returns_result(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        result = _openrouter_request(_make_meta(), "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")
        assert result.success is True
        assert result.exit_code == 0
        assert result.backend == "openrouter"
        assert result.model_profile == "openrouter_minimax"
        os.remove(result.stdout_ref)

    def test_request_id_is_uuid(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        result = _openrouter_request(_make_meta(), "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")
        assert result.request_id is not None
        assert len(result.request_id) == 36
        os.remove(result.stdout_ref)

    def test_latency_ms_recorded(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        result = _openrouter_request(_make_meta(), "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")
        assert result.latency_ms >= 0
        os.remove(result.stdout_ref)

    def test_file_saved_on_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        result = _openrouter_request(_make_meta(task_id="save-test"), "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")
        with open("/tmp/save-test.stdout.txt") as f:
            saved = json.loads(f.read())
        assert saved["request"]["model"] == "minimax/minimax-m2.7"
        os.remove(result.stdout_ref)


@patch("router.executors.urllib.request.urlopen")
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
class TestRequestPayload:
    def test_payload_structure(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        _openrouter_request(_make_meta(task_id="payload-test", summary="Fix the bug"), "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "minimax/minimax-m2.7"
        assert payload["messages"] == [{"role": "user", "content": "Fix the bug"}]
        assert payload["max_tokens"] == 4000
        assert req.method == "POST"
        os.remove("/tmp/payload-test.stdout.txt")


@patch("router.executors.urllib.request.urlopen")
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
class TestErrorHandling:
    @pytest.mark.parametrize("code,msg,expected_error", [
        (401, "Unauthorized", "auth_error"),
        (429, "Too Many Requests", "rate_limited"),
        (503, "Service Unavailable", "provider_unavailable"),
    ])
    def test_http_errors(self, mock_urlopen, code, msg, expected_error):
        mock_urlopen.side_effect = _make_http_error(code, msg, '{"error": "test"}')
        result = _openrouter_request(_make_meta(task_id=f"err-{code}"), "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")
        assert result.success is False
        assert result.normalized_error == expected_error
        assert result.exit_code == code
        os.remove(f"/tmp/err-{code}.stdout.txt")

    def test_url_error_returns_transient_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        result = _openrouter_request(_make_meta(task_id="err-url"), "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")
        assert result.success is False
        assert result.normalized_error == "transient_network_error"
        os.remove("/tmp/err-url.stdout.txt")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_returns_auth_error(self, mock_urlopen):
        result = run_codex_openrouter_minimax(_make_meta(task_id="err-nokey"))
        assert result.success is False
        assert result.normalized_error == "auth_error"
        os.remove("/tmp/err-nokey.stdout.txt")


@patch("router.executors.urllib.request.urlopen")
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
class TestErrorFileSaving:
    def test_error_body_saved_to_file(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(401, "Unauthorized", '{"error": "bad key"}')
        result = _openrouter_request(_make_meta(task_id="save-err"), "minimax/minimax-m2.7", "codex_cli", "openrouter", "openrouter_minimax")
        with open("/tmp/save-err.stdout.txt") as f:
            saved = json.loads(f.read())
        assert saved["status_code"] == 401
        os.remove("/tmp/save-err.stdout.txt")


@patch("router.executors.urllib.request.urlopen")
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key"})
class TestModelSelection:
    def test_minimax_uses_m25_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        result = run_codex_openrouter_minimax(_make_meta(task_id="sel-minimax"))
        payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        assert payload["model"] == "minimax/minimax-m2.7"
        assert result.model_profile == "openrouter_minimax"
        os.remove(result.stdout_ref)

    def test_kimi_uses_k25_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        result = run_codex_openrouter_kimi(_make_meta(task_id="sel-kimi"))
        payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        assert payload["model"] == "moonshotai/kimi-k2.5"
        assert result.model_profile == "openrouter_kimi"
        os.remove(result.stdout_ref)

    def test_run_openrouter_default_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        result = run_openrouter(_make_meta(task_id="sel-default"))
        payload = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        assert payload["model"] == "minimax/minimax-m2.7"
        os.remove(result.stdout_ref)


class TestRunOpenrouterTypeAnnotation:
    def test_model_none_accepted(self):
        import inspect
        assert inspect.signature(run_openrouter).parameters['model'].default is None
