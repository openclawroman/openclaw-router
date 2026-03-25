"""Mock-based simulation harness for routing scenarios.

Covers X-86: Simulation harness for routing/failure scenarios.
Uses monkeypatch to replace real executors with configurable mocks.
"""

import pytest
from typing import Optional

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState,
    ChainEntry, ExecutorResult, RouteDecision,
)
from router.policy import route_task, build_chain, choose_openrouter_profile


# ═══════════════════════════════════════════════════════════════════════════════
# Mock Executor Config
# ═══════════════════════════════════════════════════════════════════════════════

class MockExecutorConfig:
    """Configure what each executor should return."""

    def __init__(self):
        self.configs = {}

    def set_executor(self, tool_backend: str, success: bool = True,
                     normalized_error: Optional[str] = None,
                     latency_ms: int = 100, cost_usd: float = 0.02):
        self.configs[tool_backend] = {
            'success': success,
            'normalized_error': normalized_error,
            'latency_ms': latency_ms,
            'cost_usd': cost_usd,
        }

    def get_result(self, entry, task):
        cfg = self.configs.get(
            f"{entry.tool}:{entry.backend}",
            {'success': True, 'normalized_error': None, 'latency_ms': 100, 'cost_usd': 0.02}
        )
        return ExecutorResult(
            task_id=task.task_id,
            tool=entry.tool,
            backend=entry.backend,
            model_profile=entry.model_profile,
            success=cfg['success'],
            normalized_error=cfg['normalized_error'],
            latency_ms=cfg['latency_ms'],
            cost_estimate_usd=cfg['cost_usd'],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helper to build mock executors
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock_executors(config: MockExecutorConfig):
    """Create mock executor functions bound to the given config."""

    def mock_run_codex(task):
        entry = ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary")
        return config.get_result(entry, task)

    def mock_run_claude(task):
        entry = ChainEntry(tool="claude_code", backend="anthropic", model_profile="claude_primary")
        return config.get_result(entry, task)

    def mock_run_openrouter(task, model=None, profile=None):
        entry = ChainEntry(tool="codex_cli", backend="openrouter", model_profile=profile or "openrouter_minimax")
        return config.get_result(entry, task)

    return mock_run_codex, mock_run_claude, mock_run_openrouter


def _patch_executors(monkeypatch, config: MockExecutorConfig):
    """Patch router.policy's executor functions with mocks from config."""
    mock_codex, mock_claude, mock_openrouter = _make_mock_executors(config)
    monkeypatch.setattr("router.policy.run_codex", mock_codex)
    monkeypatch.setattr("router.policy.run_claude", mock_claude)
    monkeypatch.setattr("router.policy.run_openrouter", mock_openrouter)


def _make_task(**kwargs):
    defaults = dict(
        task_id="sim-task",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT,
    )
    defaults.update(kwargs)
    return TaskMeta(**defaults)


def _set_state(monkeypatch, state: CodexState):
    """Monkeypatch resolve_state to return a fixed state."""
    monkeypatch.setattr("router.policy.resolve_state", lambda: state)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimulation:
    """Mock-based routing simulation scenarios."""

    def test_scenario_1_normal_all_succeed(self, monkeypatch):
        """All 3 executors succeed → uses codex_cli:openai_native (first in chain)."""
        config = MockExecutorConfig()
        config.set_executor("codex_cli:openai_native", success=True)
        config.set_executor("claude_code:anthropic", success=True)
        config.set_executor("codex_cli:openrouter", success=True)
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.NORMAL)

        task = _make_task(task_id="sim-1")
        decision, result = route_task(task)

        assert result.success
        assert result.tool == "codex_cli"
        assert result.backend == "openai_native"
        assert not decision.attempted_fallback

    def test_scenario_2_codex_quota_fallback_claude(self, monkeypatch):
        """Codex quota_exhausted → falls back to claude_code successfully."""
        config = MockExecutorConfig()
        config.set_executor("codex_cli:openai_native", success=False, normalized_error="quota_exhausted")
        config.set_executor("claude_code:anthropic", success=True)
        config.set_executor("codex_cli:openrouter", success=True)
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.NORMAL)

        task = _make_task(task_id="sim-2")
        decision, result = route_task(task)

        assert result.success
        assert result.tool == "claude_code"
        assert decision.attempted_fallback
        assert decision.fallback_from == "codex_cli"

    def test_scenario_3_all_fail(self, monkeypatch):
        """All 3 executors fail → final result has last error in chain."""
        config = MockExecutorConfig()
        config.set_executor("codex_cli:openai_native", success=False, normalized_error="quota_exhausted")
        config.set_executor("claude_code:anthropic", success=False, normalized_error="rate_limited")
        config.set_executor("codex_cli:openrouter", success=False, normalized_error="provider_unavailable")
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.NORMAL)

        task = _make_task(task_id="sim-3")
        decision, result = route_task(task)

        assert not result.success
        assert result.normalized_error == "provider_unavailable"
        assert decision.attempted_fallback
        assert decision.fallback_from == "codex_cli"

    def test_scenario_4_last10_claude_primary(self, monkeypatch):
        """In last10, claude_code is primary (first in chain)."""
        config = MockExecutorConfig()
        config.set_executor("claude_code:anthropic", success=True)
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.LAST10)

        task = _make_task(task_id="sim-4")
        decision, result = route_task(task)

        assert result.success
        assert result.tool == "claude_code"
        assert result.backend == "anthropic"
        assert decision.state == "last10"

    def test_scenario_5_last10_fallback_openrouter(self, monkeypatch):
        """Claude fails in last10 → openrouter succeeds."""
        config = MockExecutorConfig()
        config.set_executor("claude_code:anthropic", success=False, normalized_error="rate_limited")
        config.set_executor("codex_cli:openrouter", success=True)
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.LAST10)

        task = _make_task(task_id="sim-5")
        decision, result = route_task(task)

        assert result.success
        assert result.tool == "codex_cli"
        assert result.backend == "openrouter"
        assert decision.attempted_fallback
        assert decision.fallback_from == "claude_code"

    def test_scenario_6_screenshot_kimi(self, monkeypatch):
        """Screenshot task routes through kimi profile in openrouter."""
        config = MockExecutorConfig()
        config.set_executor("codex_cli:openai_native", success=False, normalized_error="auth_error")
        config.set_executor("claude_code:anthropic", success=False, normalized_error="auth_error")
        config.set_executor("codex_cli:openrouter", success=True)
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.NORMAL)

        task = _make_task(task_id="sim-6", has_screenshots=True)
        decision, result = route_task(task)

        # Should reach openrouter with kimi profile
        assert result.success
        assert result.model_profile == "openrouter_kimi"

    def test_scenario_7_fallback_on_auth(self, monkeypatch):
        """auth_error IS eligible for fallback — codex auth fails → claude succeeds."""
        config = MockExecutorConfig()
        config.set_executor("codex_cli:openai_native", success=False, normalized_error="auth_error")
        config.set_executor("claude_code:anthropic", success=True)
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.NORMAL)

        task = _make_task(task_id="sim-7")
        decision, result = route_task(task)

        # auth_error IS in ELIGIBLE_FALLBACK_ERRORS — falls through to claude
        assert result.success
        assert result.tool == "claude_code"
        assert decision.attempted_fallback
        assert decision.fallback_from == "codex_cli"

    def test_scenario_8_no_fallback_on_toolchain(self, monkeypatch):
        """toolchain_error stops immediately (non-eligible)."""
        config = MockExecutorConfig()
        config.set_executor("codex_cli:openai_native", success=False, normalized_error="toolchain_error")
        config.set_executor("claude_code:anthropic", success=True)
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.NORMAL)

        task = _make_task(task_id="sim-8")
        decision, result = route_task(task)

        # toolchain_error is NOT eligible — stops at codex
        assert not result.success
        assert result.tool == "codex_cli"
        assert result.normalized_error == "toolchain_error"
        assert not decision.attempted_fallback

    def test_scenario_9_single_fallback(self, monkeypatch):
        """Only one fallback per task — stops at first success."""
        config = MockExecutorConfig()
        config.set_executor("codex_cli:openai_native", success=False, normalized_error="rate_limited")
        config.set_executor("claude_code:anthropic", success=True)
        config.set_executor("codex_cli:openrouter", success=True)
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.NORMAL)

        task = _make_task(task_id="sim-9")
        decision, result = route_task(task)

        # Should succeed at claude_code (chain[1]) and NOT go to openrouter (chain[2])
        assert result.success
        assert result.tool == "claude_code"
        assert decision.attempted_fallback
        assert decision.fallback_from == "codex_cli"

    def test_scenario_10_kimi_for_multimodal(self, monkeypatch):
        """Multimodal task → kimi profile in openrouter entry."""
        config = MockExecutorConfig()
        config.set_executor("codex_cli:openai_native", success=False, normalized_error="quota_exhausted")
        config.set_executor("claude_code:anthropic", success=False, normalized_error="quota_exhausted")
        config.set_executor("codex_cli:openrouter", success=True)
        _patch_executors(monkeypatch, config)
        _set_state(monkeypatch, CodexState.NORMAL)

        task = _make_task(
            task_id="sim-10",
            task_class=TaskClass.MULTIMODAL_CODE_TASK,
            requires_multimodal=True,
        )
        decision, result = route_task(task)

        # Verify kimi profile was selected
        assert result.success
        assert result.model_profile == "openrouter_kimi"
        # Also verify chain was built with kimi
        openrouter_entries = [e for e in decision.chain if e.backend == "openrouter"]
        assert openrouter_entries[-1].model_profile == "openrouter_kimi"
