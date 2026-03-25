"""Tests for reviewer independence, review modes, and merge gates (X-85)."""

from unittest.mock import patch

import pytest

from router.models import (
    TaskMeta, RouteDecision, ExecutorResult, ChainEntry,
    CodexState, TaskClass, TaskRisk, TaskModality, ModelProfile,
)
from router.policy import (
    build_chain,
    get_review_chain,
    ReviewMode,
    select_review_mode,
    merge_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**kwargs):
    defaults = dict(
        task_id="test-1",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT,
        requires_repo_write=False,
        requires_multimodal=False,
        has_screenshots=False,
        swarm=False,
        repo_path="/tmp/repo",
        cwd="/tmp/repo",
        summary="test",
    )
    defaults.update(kwargs)
    return TaskMeta(**defaults)


def _make_result(task_id="test-1", tool="codex_cli", backend="openai_native",
                 model_profile="codex_primary", success=True, normalized_error=None):
    return ExecutorResult(
        task_id=task_id,
        tool=tool,
        backend=backend,
        model_profile=model_profile,
        success=success,
        normalized_error=normalized_error,
    )


# ---------------------------------------------------------------------------
# 85.1: Reviewer Independence
# ---------------------------------------------------------------------------

class TestReviewerIndependence:
    """get_review_chain() must exclude the generator executor."""

    def test_excludes_codex_native_review(self):
        """If generator was codex_cli:openai_native, review chain does NOT contain it."""
        task = _make_task()
        chain = get_review_chain(task, "codex_cli:openai_native")
        for entry in chain:
            assert not (entry.tool == "codex_cli" and entry.backend == "openai_native"), \
                "codex_cli:openai_native should be excluded from review chain"

    def test_excludes_claude_review(self):
        """If generator was claude_code:anthropic, review chain does NOT contain it."""
        task = _make_task()
        chain = get_review_chain(task, "claude_code:anthropic")
        for entry in chain:
            assert not (entry.tool == "claude_code" and entry.backend == "anthropic"), \
                "claude_code:anthropic should be excluded from review chain"

    def test_excludes_openrouter_review(self):
        """If generator was codex_cli:openrouter, review chain does NOT contain it."""
        task = _make_task()
        chain = get_review_chain(task, "codex_cli:openrouter")
        for entry in chain:
            assert not (entry.tool == "codex_cli" and entry.backend == "openrouter"), \
                "codex_cli:openrouter should be excluded from review chain"

    def test_review_has_fallbacks(self):
        """Review chain still has fallback options after exclusion."""
        task = _make_task()
        chain = get_review_chain(task, "codex_cli:openai_native")
        assert len(chain) >= 1, "Review chain must have at least one fallback entry"

    def test_claude_review_after_codex_gen(self):
        """After codex gen, review uses claude or openrouter."""
        task = _make_task()
        chain = get_review_chain(task, "codex_cli:openai_native")
        tools = {(e.tool, e.backend) for e in chain}
        assert ("claude_code", "anthropic") in tools or ("codex_cli", "openrouter") in tools
        # Must not contain codex native
        assert ("codex_cli", "openai_native") not in tools

    def test_codex_review_after_claude_gen(self):
        """After claude gen, review uses codex or openrouter."""
        task = _make_task()
        chain = get_review_chain(task, "claude_code:anthropic")
        tools = {(e.tool, e.backend) for e in chain}
        assert ("codex_cli", "openai_native") in tools or ("codex_cli", "openrouter") in tools
        # Must not contain claude
        assert ("claude_code", "anthropic") not in tools


# ---------------------------------------------------------------------------
# 85.2: Review Modes
# ---------------------------------------------------------------------------

class TestReviewModes:
    """ReviewMode enum and select_review_mode()."""

    def test_review_mode_enum(self):
        """Has FAST and DEEP values."""
        assert ReviewMode.FAST.value == "fast"
        assert ReviewMode.DEEP.value == "deep"

    def test_default_fast(self):
        """Returns FAST for normal tasks."""
        task = _make_task(task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM)
        assert select_review_mode(task) == ReviewMode.FAST

    def test_deep_for_high_risk(self):
        """Returns DEEP when risk=high."""
        task = _make_task(risk=TaskRisk.HIGH)
        assert select_review_mode(task) == ReviewMode.DEEP

    def test_deep_for_architecture(self):
        """Returns DEEP for task_class=repo_architecture_change."""
        task = _make_task(task_class=TaskClass.REPO_ARCHITECTURE_CHANGE)
        assert select_review_mode(task) == ReviewMode.DEEP

    def test_fast_for_low_risk(self):
        """Returns FAST for risk=low."""
        task = _make_task(risk=TaskRisk.LOW)
        assert select_review_mode(task) == ReviewMode.FAST


# ---------------------------------------------------------------------------
# 85.3: Merge Gate
# ---------------------------------------------------------------------------

class TestMergeGate:
    """merge_gate() checks all conditions for merging."""

    def test_all_gates_pass(self):
        """Both results successful + different executors → (True, [])."""
        gen = _make_result(tool="codex_cli", backend="openai_native")
        rev = _make_result(tool="claude_code", backend="anthropic")
        task = _make_task()
        passed, reasons = merge_gate(gen, rev, task)
        assert passed is True
        assert reasons == []

    def test_reviewer_same_as_generator(self):
        """Reviewer used same executor → fails with reason."""
        gen = _make_result(tool="codex_cli", backend="openai_native")
        rev = _make_result(tool="codex_cli", backend="openai_native")
        task = _make_task()
        passed, reasons = merge_gate(gen, rev, task)
        assert passed is False
        assert any("same" in r.lower() or "independent" in r.lower() for r in reasons)

    def test_generator_failed(self):
        """Generator result not success → fails."""
        gen = _make_result(success=False, normalized_error="auth_error")
        rev = _make_result(tool="claude_code", backend="anthropic")
        task = _make_task()
        passed, reasons = merge_gate(gen, rev, task)
        assert passed is False
        assert any("generator" in r.lower() or "gen" in r.lower() for r in reasons)

    def test_reviewer_failed(self):
        """Reviewer result not success → fails."""
        gen = _make_result(tool="codex_cli", backend="openai_native")
        rev = _make_result(tool="claude_code", backend="anthropic", success=False, normalized_error="rate_limited")
        task = _make_task()
        passed, reasons = merge_gate(gen, rev, task)
        assert passed is False
        assert any("reviewer" in r.lower() or "review" in r.lower() for r in reasons)

    def test_both_succeed_different_executors(self):
        """Passes when both succeed with different executors."""
        gen = _make_result(tool="codex_cli", backend="openai_native")
        rev = _make_result(tool="claude_code", backend="anthropic")
        task = _make_task()
        passed, reasons = merge_gate(gen, rev, task)
        assert passed is True
        assert reasons == []

    def test_multiple_failures(self):
        """Returns all failure reasons, not just first."""
        gen = _make_result(success=False, normalized_error="auth_error")
        rev = _make_result(success=False, normalized_error="rate_limited")
        task = _make_task()
        passed, reasons = merge_gate(gen, rev, task)
        assert passed is False
        # Should have at least 2 failure reasons (gen failed + reviewer failed, and possibly same executor)
        assert len(reasons) >= 2
