"""Tests for Codex CLI executor adapter (X-76)."""

import os
import json
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from router.models import TaskMeta
from router.executors import run_codex
from router.errors import (
    CodexQuotaError,
    CodexAuthError,
    CodexToolError,
    ProviderTimeoutError,
)


def _meta(task_id="task-001", summary="do something", cwd="/tmp"):
    return TaskMeta(task_id=task_id, summary=summary, cwd=cwd)


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

class TestRunCodexSuccess:
    @patch("router.executors.subprocess.run")
    def test_successful_execution(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="output here", stderr=""
        )
        result = run_codex(_meta())

        assert result.success is True
        assert result.exit_code == 0
        assert result.tool == "codex_cli"
        assert result.backend == "openai_native"
        assert result.model_profile == "codex_primary"
        assert result.final_summary == "output here"

    @patch("router.executors.subprocess.run")
    def test_empty_stdout(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = run_codex(_meta())
        assert result.success is True
        assert result.final_summary is None
        assert result.stdout_ref is None


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestRunCodexErrors:
    @patch("router.executors.subprocess.run")
    def test_file_not_found_error(self, mock_run):
        """FileNotFoundError from subprocess → returns -1, stderr set by _run_subprocess."""
        mock_run.side_effect = FileNotFoundError("No such file or directory: 'codex'")
        with pytest.raises(CodexToolError):
            run_codex(_meta())

    @patch("router.executors.subprocess.run")
    def test_timeout_expired(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["codex"], timeout=300)
        with pytest.raises(ProviderTimeoutError):
            run_codex(_meta())

    @patch("router.executors.subprocess.run")
    def test_quota_exhausted(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Error: quota limit exceeded"
        )
        with pytest.raises(CodexQuotaError):
            run_codex(_meta())

    @patch("router.executors.subprocess.run")
    def test_auth_error(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Unauthorized: invalid API key"
        )
        with pytest.raises(CodexAuthError):
            run_codex(_meta())

    @patch("router.executors.subprocess.run")
    def test_generic_tool_error(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Something broke"
        )
        with pytest.raises(CodexToolError):
            run_codex(_meta())


# ---------------------------------------------------------------------------
# File saving & artifacts
# ---------------------------------------------------------------------------

class TestRunCodexFileSaving:
    def test_stdout_saved_to_file(self, tmp_path):
        meta = _meta(task_id="codex-file-test", cwd=str(tmp_path))
        with patch("router.executors.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="hello stdout", stderr="hello stderr"
            )
            result = run_codex(meta)

        stdout_path = "/tmp/codex-file-test.stdout.txt"
        stderr_path = "/tmp/codex-file-test.stderr.txt"

        assert os.path.exists(stdout_path)
        assert os.path.exists(stderr_path)
        assert open(stdout_path).read() == "hello stdout"
        assert open(stderr_path).read() == "hello stderr"

        # cleanup
        os.unlink(stdout_path)
        os.unlink(stderr_path)

    def test_artifacts_populated(self, tmp_path):
        meta = _meta(task_id="codex-artifacts", cwd=str(tmp_path))
        with patch("router.executors.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="out", stderr="err"
            )
            result = run_codex(meta)

        assert "/tmp/codex-artifacts.stdout.txt" in result.artifacts
        assert "/tmp/codex-artifacts.stderr.txt" in result.artifacts
        assert result.stdout_ref == "/tmp/codex-artifacts.stdout.txt"
        assert result.stderr_ref == "/tmp/codex-artifacts.stderr.txt"

        # cleanup
        for f in result.artifacts:
            os.unlink(f)

    def test_no_files_when_no_task_id(self, tmp_path):
        meta = _meta(task_id="", cwd=str(tmp_path))
        with patch("router.executors.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="out", stderr="err"
            )
            result = run_codex(meta)

        assert result.artifacts == []
        assert result.stdout_ref is None

    def test_no_stderr_file_when_stderr_empty(self, tmp_path):
        meta = _meta(task_id="codex-no-stderr", cwd=str(tmp_path))
        with patch("router.executors.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="out", stderr=""
            )
            result = run_codex(meta)

        assert "/tmp/codex-no-stderr.stdout.txt" in result.artifacts
        assert result.stderr_ref is None
        assert len(result.artifacts) == 1

        os.unlink("/tmp/codex-no-stderr.stdout.txt")
