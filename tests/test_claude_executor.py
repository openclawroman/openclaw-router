"""Tests for Claude Code executor adapter (X-77)."""

import os
import json
import subprocess
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from router.models import TaskMeta
from router.executors import run_claude, _update_claude_health
from router.errors import (
    ClaudeQuotaError,
    ClaudeAuthError,
    ClaudeToolError,
    ProviderTimeoutError,
)


def _meta(task_id="task-001", summary="do something", cwd="/tmp"):
    return TaskMeta(task_id=task_id, summary=summary, cwd=cwd)


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

class TestRunClaudeSuccess:
    @patch("router.executors.subprocess.run")
    def test_successful_execution(self, mock_run, tmp_path):
        # Use a temp runtime dir
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        with patch("router.executors.Path") as mock_path:
            mock_path.return_value = runtime_dir
            # We need Path("runtime") to return our temp dir
            original_path = Path
            def path_side_effect(p):
                if p == "runtime":
                    return runtime_dir
                return original_path(p)
            mock_path.side_effect = path_side_effect

            mock_run.return_value = MagicMock(
                returncode=0, stdout="claude output", stderr=""
            )
            result = run_claude(_meta(cwd=str(tmp_path)))

        assert result.success is True
        assert result.exit_code == 0
        assert result.tool == "claude_code"
        assert result.backend == "anthropic"
        assert result.model_profile == "claude_primary"
        assert result.final_summary == "claude output"

    @patch("router.executors.subprocess.run")
    def test_empty_stdout(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = run_claude(_meta())
        assert result.success is True
        assert result.final_summary is None
        assert result.stdout_ref is None


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestRunClaudeErrors:
    @patch("router.executors.subprocess.run")
    def test_file_not_found_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError("No such file or directory: 'claude'")
        with pytest.raises(ClaudeToolError):
            run_claude(_meta())

    @patch("router.executors.subprocess.run")
    def test_timeout_expired(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=300)
        with pytest.raises(ProviderTimeoutError):
            run_claude(_meta())

    @patch("router.executors.subprocess.run")
    def test_quota_exhausted(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Error: quota limit exceeded"
        )
        with pytest.raises(ClaudeQuotaError):
            run_claude(_meta())

    @patch("router.executors.subprocess.run")
    def test_auth_error(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Unauthorized: invalid API key"
        )
        with pytest.raises(ClaudeAuthError):
            run_claude(_meta())

    @patch("router.executors.subprocess.run")
    def test_generic_tool_error(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Something broke"
        )
        with pytest.raises(ClaudeToolError):
            run_claude(_meta())


# ---------------------------------------------------------------------------
# File saving & artifacts
# ---------------------------------------------------------------------------

class TestRunClaudeFileSaving:
    def test_stdout_saved_to_file(self, tmp_path):
        meta = _meta(task_id="claude-file-test", cwd=str(tmp_path))
        with patch("router.executors.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="hello stdout", stderr="hello stderr"
            )
            result = run_claude(meta)

        stdout_path = "/tmp/claude-file-test.stdout.txt"
        stderr_path = "/tmp/claude-file-test.stderr.txt"

        assert os.path.exists(stdout_path)
        assert os.path.exists(stderr_path)
        assert open(stdout_path).read() == "hello stdout"
        assert open(stderr_path).read() == "hello stderr"

        os.unlink(stdout_path)
        os.unlink(stderr_path)

    def test_artifacts_populated(self, tmp_path):
        meta = _meta(task_id="claude-artifacts", cwd=str(tmp_path))
        with patch("router.executors.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="out", stderr="err"
            )
            result = run_claude(meta)

        assert "/tmp/claude-artifacts.stdout.txt" in result.artifacts
        assert "/tmp/claude-artifacts.stderr.txt" in result.artifacts
        assert result.stdout_ref == "/tmp/claude-artifacts.stdout.txt"
        assert result.stderr_ref == "/tmp/claude-artifacts.stderr.txt"

        for f in result.artifacts:
            os.unlink(f)


# ---------------------------------------------------------------------------
# Health tracking
# ---------------------------------------------------------------------------

class TestClaudeHealthTracking:
    def _cleanup_health(self):
        p = Path("runtime/claude_health.json")
        if p.exists():
            p.unlink()
        d = Path("runtime")
        if d.exists() and not any(d.iterdir()):
            d.rmdir()

    def teardown_method(self):
        self._cleanup_health()

    @patch("router.executors.subprocess.run")
    def test_success_updates_health_available_true(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok", stderr=""
        )
        run_claude(_meta())

        health_path = Path("runtime/claude_health.json")
        assert health_path.exists()
        health = json.loads(health_path.read_text())
        assert health["available"] is True
        assert "updated_at" in health
        assert health["reason"] == "recent successful invocation"

    @patch("router.executors.subprocess.run")
    def test_failure_updates_health_available_false(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Something broke"
        )
        with pytest.raises(ClaudeToolError):
            run_claude(_meta())

        health_path = Path("runtime/claude_health.json")
        assert health_path.exists()
        health = json.loads(health_path.read_text())
        assert health["available"] is False
        assert health["reason"] == "toolchain_error"

    @patch("router.executors.subprocess.run")
    def test_quota_failure_health_reason(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="quota limit hit"
        )
        with pytest.raises(ClaudeQuotaError):
            run_claude(_meta())

        health = json.loads(Path("runtime/claude_health.json").read_text())
        assert health["available"] is False
        assert health["reason"] == "quota_exhausted"

    @patch("router.executors.subprocess.run")
    def test_auth_failure_health_reason(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="unauthorized access"
        )
        with pytest.raises(ClaudeAuthError):
            run_claude(_meta())

        health = json.loads(Path("runtime/claude_health.json").read_text())
        assert health["available"] is False
        assert health["reason"] == "auth_error"

    @patch("router.executors.subprocess.run")
    def test_timeout_health_reason(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=300)
        with pytest.raises(ProviderTimeoutError):
            run_claude(_meta())

        health = json.loads(Path("runtime/claude_health.json").read_text())
        assert health["available"] is False
        assert health["reason"] == "provider_timeout"

    def test_health_file_created_on_first_run(self):
        """_update_claude_health creates runtime/ dir and health file."""
        self._cleanup_health()
        assert not Path("runtime/claude_health.json").exists()

        _update_claude_health(success=True)

        assert Path("runtime/claude_health.json").exists()
        health = json.loads(Path("runtime/claude_health.json").read_text())
        assert health["available"] is True
