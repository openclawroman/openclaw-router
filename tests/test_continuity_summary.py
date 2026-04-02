"""Tests for continuity_summary support in TaskMeta and executors."""

import json
import pytest
from unittest.mock import patch, MagicMock

from router.models import TaskMeta
from router.executors import run_codex, run_claude, _build_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta(**kwargs):
    defaults = dict(task_id="task-001", summary="do something", cwd="/tmp")
    defaults.update(kwargs)
    return TaskMeta(**defaults)


# ---------------------------------------------------------------------------
# Parser: continuity_summary from JSON
# ---------------------------------------------------------------------------

class TestParseContinuitySummary:
    def test_parse_continuity_summary_from_json(self):
        """parse_task_meta should accept continuity_summary from top-level or task_meta."""
        from bin.ai_code_runner import parse_task_meta
        payload = {
            "task_id": "t1",
            "task_meta": {
                "task_id": "t1",
                "summary": "do stuff",
            },
            "continuity_summary": "Previous work: added auth module",
        }
        meta = parse_task_meta(payload)
        assert meta.continuity_summary == "Previous work: added auth module"

    def test_parse_continuity_summary_in_task_meta(self):
        """continuity_summary inside task_meta dict is picked up."""
        from bin.ai_code_runner import parse_task_meta
        payload = {
            "task_id": "t1",
            "task_meta": {
                "task_id": "t1",
                "summary": "do stuff",
                "continuity_summary": "From task_meta field",
            },
        }
        meta = parse_task_meta(payload)
        assert meta.continuity_summary == "From task_meta field"

    def test_backward_compat_no_continuity_summary(self):
        """Old payloads without continuity_summary still work (defaults to empty)."""
        from bin.ai_code_runner import parse_task_meta
        payload = {
            "task_id": "t1",
            "task_meta": {
                "task_id": "t1",
                "summary": "do stuff",
            },
        }
        meta = parse_task_meta(payload)
        assert meta.continuity_summary == ""
        assert meta.summary == "do stuff"

    def test_empty_continuity_summary_preserved(self):
        """Explicit empty string is kept as empty."""
        from bin.ai_code_runner import parse_task_meta
        payload = {
            "task_id": "t1",
            "task_meta": {"task_id": "t1", "summary": "x"},
            "continuity_summary": "",
        }
        meta = parse_task_meta(payload)
        assert meta.continuity_summary == ""


# ---------------------------------------------------------------------------
# TaskMeta defaults
# ---------------------------------------------------------------------------

class TestTaskMetaContinuityDefault:
    def test_default_is_empty(self):
        meta = TaskMeta(task_id="t1", summary="s")
        assert meta.continuity_summary == ""

    def test_can_set(self):
        meta = TaskMeta(task_id="t1", summary="s", continuity_summary="ctx")
        assert meta.continuity_summary == "ctx"


# ---------------------------------------------------------------------------
# _build_prompt helper
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_no_continuity(self):
        meta = _meta(continuity_summary="")
        prompt = _build_prompt(meta)
        assert prompt == "do something"

    def test_with_continuity(self):
        meta = _meta(continuity_summary="Added auth module")
        prompt = _build_prompt(meta)
        assert "[Continuity context from previous delegation]" in prompt
        assert "Added auth module" in prompt
        assert "--- Current task ---" in prompt
        assert "do something" in prompt

    def test_continuity_is_additive(self):
        """Continuity block is prepended; original task remains intact."""
        meta = _meta(
            summary="Fix login bug",
            continuity_summary="Previously: built auth flow",
        )
        prompt = _build_prompt(meta)
        # Current task appears after the separator
        idx_ctx = prompt.index("[Continuity context")
        idx_sep = prompt.index("--- Current task ---")
        idx_task = prompt.index("Fix login bug")
        assert idx_ctx < idx_sep < idx_task

    def test_fallback_to_task_id(self):
        meta = TaskMeta(task_id="t1", summary="", continuity_summary="ctx")
        prompt = _build_prompt(meta)
        assert "t1" in prompt


# ---------------------------------------------------------------------------
# Executor prompt injection
# ---------------------------------------------------------------------------

class TestCodexExecutorContinuity:
    @patch("router.executors.subprocess.run")
    def test_codex_prompt_has_continuity(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        meta = _meta(continuity_summary="Built auth flow")
        run_codex(meta)
        cmd = mock_run.call_args[0][0]
        prompt_arg = cmd[-1]
        assert "[Continuity context from previous delegation]" in prompt_arg
        assert "Built auth flow" in prompt_arg
        assert "--- Current task ---" in prompt_arg

    @patch("router.executors.subprocess.run")
    def test_codex_prompt_unchanged_without_continuity(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        meta = _meta(continuity_summary="")
        run_codex(meta)
        cmd = mock_run.call_args[0][0]
        prompt_arg = cmd[-1]
        assert prompt_arg == "do something"

    @patch("router.executors.subprocess.run")
    def test_codex_prompt_unchanged_with_default(self, mock_run):
        """Old-style TaskMeta without continuity_summary works identically."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        meta = _meta()
        run_codex(meta)
        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == "do something"


class TestClaudeExecutorContinuity:
    @patch("router.executors.subprocess.run")
    def test_claude_prompt_has_continuity(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        meta = _meta(continuity_summary="Refactored DB layer")
        run_claude(meta)
        cmd = mock_run.call_args[0][0]
        prompt_arg = cmd[cmd.index("-p") + 1]
        assert "[Continuity context from previous delegation]" in prompt_arg
        assert "Refactored DB layer" in prompt_arg

    @patch("router.executors.subprocess.run")
    def test_claude_prompt_unchanged_without_continuity(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        meta = _meta(continuity_summary="")
        run_claude(meta)
        cmd = mock_run.call_args[0][0]
        prompt_arg = cmd[cmd.index("-p") + 1]
        assert prompt_arg == "do something"


class TestOpenRouterContinuity:
    @patch("urllib.request.urlopen")
    def test_openrouter_prompt_has_continuity(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"cost": 0.001},
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        from router.executors import _openrouter_request
        meta = _meta(continuity_summary="Deployed v2 API")
        result = _openrouter_request(
            meta=meta, model="test-model",
            tool="codex_cli", backend="openrouter",
            model_profile="openrouter_minimax",
        )
        # Inspect the saved request file
        with open(result.stdout_ref) as f:
            saved = json.load(f)
        messages = saved["request"]["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert "[Continuity context from previous delegation]" in content
        assert "Deployed v2 API" in content

    @patch("urllib.request.urlopen")
    def test_openrouter_prompt_unchanged_without_continuity(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"cost": 0.001},
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        from router.executors import _openrouter_request
        meta = _meta(continuity_summary="")
        result = _openrouter_request(
            meta=meta, model="test-model",
            tool="codex_cli", backend="openrouter",
            model_profile="openrouter_minimax",
        )
        with open(result.stdout_ref) as f:
            saved = json.load(f)
        messages = saved["request"]["messages"]
        content = messages[0]["content"]
        assert "[Continuity context" not in content
        assert content == "do something"
