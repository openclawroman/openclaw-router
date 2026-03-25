"""Tests for router/policy.py — state-aware routing policy behavior."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, RouteDecision, ChainEntry, CodexState,
    TaskClass, TaskRisk, TaskModality, ModelProfile,
)
from router.policy import (
    resolve_state,
    claude_available,
    choose_openrouter_profile,
    can_fallback,
    build_chain,
)


# ---------------------------------------------------------------------------
# resolve_state
# ---------------------------------------------------------------------------

class TestResolveState:
    """State resolution: manual > auto > default normal."""

    def test_default_normal(self, tmp_path):
        """Returns 'normal' when no state files exist."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        with patch("router.policy.StateStore") as MockStore:
            store = MagicMock()
            store.get_manual_state.return_value = None
            store.get_auto_state.return_value = None
            MockStore.return_value = store
            assert resolve_state() == CodexState.NORMAL

    def test_manual_override(self, tmp_path):
        """Manual state takes precedence over auto."""
        with patch("router.policy.StateStore") as MockStore:
            store = MagicMock()
            store.get_manual_state.return_value = CodexState.LAST10
            store.get_auto_state.return_value = CodexState.NORMAL
            MockStore.return_value = store
            assert resolve_state() == CodexState.LAST10

    def test_auto_fallback(self, tmp_path):
        """Auto state used when manual missing."""
        with patch("router.policy.StateStore") as MockStore:
            store = MagicMock()
            store.get_manual_state.return_value = None
            store.get_auto_state.return_value = CodexState.LAST10
            MockStore.return_value = store
            assert resolve_state() == CodexState.LAST10

    def test_manual_clear_falls_to_auto(self, tmp_path):
        """Clear manual state, auto takes over."""
        with patch("router.policy.StateStore") as MockStore:
            store = MagicMock()
            store.get_manual_state.return_value = None
            store.get_auto_state.return_value = CodexState.LAST10
            MockStore.return_value = store
            assert resolve_state() == CodexState.LAST10


# ---------------------------------------------------------------------------
# claude_available
# ---------------------------------------------------------------------------

class TestClaudeAvailable:
    """Claude availability check."""

    def test_returns_bool(self):
        """Always returns bool."""
        result = claude_available()
        assert isinstance(result, bool)

    def test_no_claude(self):
        """Returns False when claude not in PATH (mock shutil.which)."""
        with patch("router.policy.Path") as MockPath, \
             patch("router.policy.os.system", return_value=1):
            # Make all Path checks return False
            instance = MockPath.return_value
            instance.exists.return_value = False
            # Also need to handle Path("/usr/local/bin/claude") etc.
            MockPath.side_effect = lambda p: MagicMock(exists=MagicMock(return_value=False))
            result = claude_available()
            # os.system returns 1 (not found), and no Path exists
            assert result is False


# ---------------------------------------------------------------------------
# choose_openrouter_profile
# ---------------------------------------------------------------------------

class TestChooseOpenrouterProfile:
    """OpenRouter model profile selection."""

    def _make_task(self, **kwargs):
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

    def test_minimax_default(self):
        """Returns MINIMAX for normal tasks."""
        task = self._make_task()
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_MINIMAX

    def test_kimi_for_screenshots(self):
        """Returns KIMI when has_screenshots=True."""
        task = self._make_task(has_screenshots=True)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_KIMI

    def test_kimi_for_multimodal(self):
        """Returns KIMI when requires_multimodal=True."""
        task = self._make_task(requires_multimodal=True)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_KIMI

    def test_kimi_for_ui_from_screenshot(self):
        """Returns KIMI for task_class=ui_from_screenshot (via has_screenshots)."""
        task = self._make_task(task_class=TaskClass.UI_FROM_SCREENSHOT, has_screenshots=True)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_KIMI

    def test_kimi_for_multimodal_code_task(self):
        """Returns KIMI for task_class=multimodal_code_task (via requires_multimodal)."""
        task = self._make_task(task_class=TaskClass.MULTIMODAL_CODE_TASK, requires_multimodal=True)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_KIMI

    def test_kimi_for_swarm(self):
        """Returns KIMI for task_class=swarm_code_task (via requires_multimodal)."""
        task = self._make_task(task_class=TaskClass.SWARM_CODE_TASK, requires_multimodal=True)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_KIMI

    def test_minimax_for_implementation(self):
        """Returns MINIMAX for task_class=implementation."""
        task = self._make_task(task_class=TaskClass.IMPLEMENTATION)
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_MINIMAX


# ---------------------------------------------------------------------------
# can_fallback
# ---------------------------------------------------------------------------

class TestCanFallback:
    """Fallback eligibility checks."""

    def test_eligible_errors(self):
        """All 6 eligible errors return True."""
        eligible = [
            "auth_error",
            "rate_limited",
            "quota_exhausted",
            "provider_unavailable",
            "provider_timeout",
            "transient_network_error",
        ]
        for err in eligible:
            assert can_fallback(err) is True, f"Expected {err} to be eligible"

    def test_non_eligible_errors(self):
        """invalid_payload, missing_repo_path, toolchain_error, etc. return False."""
        non_eligible = [
            "invalid_payload",
            "missing_repo_path",
            "toolchain_error",
            "permission_denied_local",
            "git_conflict",
            "template_render_error",
            "unsupported_task",
            "unknown_error",
        ]
        for err in non_eligible:
            assert can_fallback(err) is False, f"Expected {err} to be non-eligible"

    def test_none_returns_false(self):
        """None returns False."""
        assert can_fallback(None) is False


# ---------------------------------------------------------------------------
# build_chain
# ---------------------------------------------------------------------------

class TestBuildChain:
    """Chain building for different states."""

    def _make_task(self, **kwargs):
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

    def test_normal_chain_length(self):
        """Normal state chain has 3 entries."""
        task = self._make_task()
        chain = build_chain(task, CodexState.NORMAL)
        assert len(chain) == 3

    def test_normal_chain_first(self):
        """First entry is codex_cli:openai_native."""
        task = self._make_task()
        chain = build_chain(task, CodexState.NORMAL)
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"

    def test_normal_chain_second(self):
        """Second entry is claude_code:anthropic."""
        task = self._make_task()
        chain = build_chain(task, CodexState.NORMAL)
        assert chain[1].tool == "claude_code"
        assert chain[1].backend == "anthropic"

    def test_normal_chain_third(self):
        """Third entry is codex_cli:openrouter."""
        task = self._make_task()
        chain = build_chain(task, CodexState.NORMAL)
        assert chain[2].tool == "codex_cli"
        assert chain[2].backend == "openrouter"

    def test_last10_chain_length(self):
        """Last10 chain has 2 entries."""
        task = self._make_task()
        chain = build_chain(task, CodexState.LAST10)
        assert len(chain) == 2

    def test_last10_chain_first(self):
        """First entry is claude_code:anthropic."""
        task = self._make_task()
        chain = build_chain(task, CodexState.LAST10)
        assert chain[0].tool == "claude_code"
        assert chain[0].backend == "anthropic"

    def test_last10_chain_second(self):
        """Second entry is codex_cli:openrouter."""
        task = self._make_task()
        chain = build_chain(task, CodexState.LAST10)
        assert chain[1].tool == "codex_cli"
        assert chain[1].backend == "openrouter"

    def test_last10_skips_codex_native(self):
        """No openai_native in last10 chain."""
        task = self._make_task()
        chain = build_chain(task, CodexState.LAST10)
        for entry in chain:
            assert entry.backend != "openai_native"

    def test_normal_kimi_profile(self):
        """Screenshot tasks get kimi in openrouter entry."""
        task = self._make_task(has_screenshots=True)
        chain = build_chain(task, CodexState.NORMAL)
        openrouter_entry = chain[2]
        assert "kimi" in openrouter_entry.model_profile

    def test_normal_minimax_profile(self):
        """Implementation tasks get minimax in openrouter entry."""
        task = self._make_task(task_class=TaskClass.IMPLEMENTATION)
        chain = build_chain(task, CodexState.NORMAL)
        openrouter_entry = chain[2]
        assert "minimax" in openrouter_entry.model_profile


# ---------------------------------------------------------------------------
# RouteDecision
# ---------------------------------------------------------------------------

class TestRouteDecision:
    """RouteDecision data structure."""

    def test_decision_has_required_fields(self):
        """task_id, state, chain, reason, attempted_fallback, fallback_from."""
        decision = RouteDecision(
            task_id="t1",
            state="normal",
            chain=[],
            reason="test",
            attempted_fallback=False,
            fallback_from=None,
        )
        assert decision.task_id == "t1"
        assert decision.state == "normal"
        assert isinstance(decision.chain, list)
        assert decision.reason == "test"
        assert decision.attempted_fallback is False
        assert decision.fallback_from is None

    def test_decisions_match_state(self):
        """Normal gives normal chain, last10 gives last10 chain."""
        task = TaskMeta(task_id="t1")
        chain_normal = build_chain(task, CodexState.NORMAL)
        chain_last10 = build_chain(task, CodexState.LAST10)

        assert len(chain_normal) == 3
        assert len(chain_last10) == 2
        # Normal starts with codex native, last10 starts with claude
        assert chain_normal[0].tool == "codex_cli"
        assert chain_last10[0].tool == "claude_code"
