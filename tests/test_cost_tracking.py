"""Tests for OpenRouter cost extraction and tracking."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from router.executors import _openrouter_request, run_codex, run_claude
from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality


@pytest.fixture
def sample_task():
    return TaskMeta(
        task_id="cost-test-001",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT,
        requires_repo_write=False,
        requires_multimodal=False,
        has_screenshots=False,
        swarm=False,
        repo_path="",
        cwd="",
        summary="Test cost tracking",
    )


def _mock_response(body_dict, status=200):
    """Create a mock HTTP response."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(body_dict).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestCostExtraction:
    """Test cost extraction from OpenRouter response."""

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_cost_in_usage_field(self, mock_urlopen, sample_task):
        """Cost in usage.cost is extracted correctly."""
        mock_urlopen.return_value = _mock_response({
            "id": "gen-123",
            "choices": [{"message": {"content": "done"}}],
            "usage": {"cost": 0.0042, "prompt_tokens": 100, "completion_tokens": 200},
        })

        result = _openrouter_request(sample_task, "minimax/minimax-m2.7", "openrouter", "openrouter", "openrouter_minimax")

        assert result.success is True
        assert result.cost_estimate_usd == 0.0042

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_cost_at_top_level(self, mock_urlopen, sample_task):
        """Cost at top level of response is extracted."""
        mock_urlopen.return_value = _mock_response({
            "id": "gen-456",
            "choices": [{"message": {"content": "done"}}],
            "cost": 0.0015,
        })

        result = _openrouter_request(sample_task, "minimax/minimax-m2.7", "openrouter", "openrouter", "openrouter_minimax")

        assert result.cost_estimate_usd == 0.0015

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_cost_defaults_to_zero(self, mock_urlopen, sample_task):
        """Missing cost field defaults to 0.0."""
        mock_urlopen.return_value = _mock_response({
            "id": "gen-789",
            "choices": [{"message": {"content": "done"}}],
        })

        result = _openrouter_request(sample_task, "minimax/minimax-m2.7", "openrouter", "openrouter", "openrouter_minimax")

        assert result.cost_estimate_usd == 0.0


class TestCostInResult:
    """Test cost appears in ExecutorResult."""

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_cost_in_result_fields(self, mock_urlopen, sample_task):
        """ExecutorResult contains cost_estimate_usd."""
        mock_urlopen.return_value = _mock_response({
            "id": "gen-abc",
            "choices": [{"message": {"content": "done"}}],
            "usage": {"cost": 0.003},
        })

        result = _openrouter_request(sample_task, "minimax/minimax-m2.7", "openrouter", "openrouter", "openrouter_minimax")

        assert hasattr(result, "cost_estimate_usd")
        assert result.cost_estimate_usd == 0.003


class TestCostWithLatency:
    """Test cost and latency are both tracked."""

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_both_tracked(self, mock_urlopen, sample_task):
        """Both latency and cost are present in result."""
        mock_urlopen.return_value = _mock_response({
            "id": "gen-xyz",
            "choices": [{"message": {"content": "done"}}],
            "usage": {"cost": 0.007},
        })

        result = _openrouter_request(sample_task, "minimax/minimax-m2.7", "openrouter", "openrouter", "openrouter_minimax")

        assert result.latency_ms >= 0
        assert result.cost_estimate_usd == 0.007


class TestCostInSavedFile:
    """Test cost appears in the saved /tmp/ file."""

    @patch("router.executors.urllib.request.urlopen")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_cost_in_file(self, mock_urlopen, sample_task):
        """Saved file content includes cost_estimate_usd."""
        mock_urlopen.return_value = _mock_response({
            "id": "gen-file",
            "choices": [{"message": {"content": "done"}}],
            "usage": {"cost": 0.0099},
        })

        result = _openrouter_request(sample_task, "minimax/minimax-m2.7", "openrouter", "openrouter", "openrouter_minimax")

        with open(result.artifacts[0]) as f:
            saved = json.loads(f.read())

        assert saved["cost_estimate_usd"] == 0.0099


class TestCodexNoCost:
    """Test Codex returns 0.0 cost (subscription model)."""

    @patch("router.executors.subprocess.run")
    def test_codex_cost_zero(self, mock_run, sample_task):
        """Codex CLI has no per-call cost tracking."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Task completed",
            stderr="",
        )

        result = run_codex(sample_task)

        assert result.success is True
        assert result.cost_estimate_usd == 0.0


class TestClaudeNoCost:
    """Test Claude returns 0.0 cost (no direct tracking)."""

    @patch("router.executors.subprocess.run")
    def test_claude_cost_zero(self, mock_run, sample_task):
        """Claude Code has no direct cost tracking."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Task completed",
            stderr="",
        )

        result = run_claude(sample_task)

        assert result.success is True
        assert result.cost_estimate_usd == 0.0
