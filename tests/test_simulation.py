"""Mock-based simulation harness for routing scenarios."""

import pytest
from typing import Optional

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState,
    ChainEntry, ExecutorResult, RouteDecision,
)
from router.policy import route_task, build_chain, choose_openrouter_profile, reset_breaker


@pytest.fixture(autouse=True)
def _reset_breaker():
    reset_breaker()
    yield
    reset_breaker()


class MockExecutorConfig:
    def __init__(self):
        self.configs = {}

    def set_executor(self, tool_backend, success=True, normalized_error=None, latency_ms=100, cost_usd=0.02):
        self.configs[tool_backend] = {'success': success, 'normalized_error': normalized_error, 'latency_ms': latency_ms, 'cost_usd': cost_usd}

    def get_result(self, entry, task):
        cfg = self.configs.get(f"{entry.tool}:{entry.backend}", {'success': True, 'normalized_error': None, 'latency_ms': 100, 'cost_usd': 0.02})
        return ExecutorResult(task_id=task.task_id, tool=entry.tool, backend=entry.backend,
            model_profile=entry.model_profile, success=cfg['success'],
            normalized_error=cfg['normalized_error'], latency_ms=cfg['latency_ms'], cost_estimate_usd=cfg['cost_usd'])


def _make_mock_executors(config):
    def mock_run_codex(task, model=None):
        return config.get_result(ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary"), task)
    def mock_run_claude(task, *, model=None):
        return config.get_result(ChainEntry(tool="claude_code", backend="anthropic", model_profile="claude_primary"), task)
    def mock_run_openrouter(task, model=None, profile=None):
        return config.get_result(ChainEntry(tool="codex_cli", backend="openrouter", model_profile=profile or "openrouter_minimax"), task)
    return mock_run_codex, mock_run_claude, mock_run_openrouter


def _patch_executors(monkeypatch, config):
    codex, claude, openrouter = _make_mock_executors(config)
    monkeypatch.setattr("router.policy.run_codex", codex)
    monkeypatch.setattr("router.policy.run_claude", claude)
    monkeypatch.setattr("router.policy.run_openrouter", openrouter)


def _make_task(**kwargs):
    defaults = dict(task_id="sim-task", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT)
    defaults.update(kwargs)
    return TaskMeta(**defaults)


def _set_state(monkeypatch, state):
    monkeypatch.setattr("router.policy.resolve_state", lambda: state)


def _setup_sim(monkeypatch, task_id, state, executors, **task_kwargs):
    """Common setup: configure executors, state, return task + route result."""
    config = MockExecutorConfig()
    for key, opts in executors.items():
        config.set_executor(key, **opts)
    _patch_executors(monkeypatch, config)
    _set_state(monkeypatch, state)
    task = _make_task(task_id=task_id, **task_kwargs)
    return task, *route_task(task)


class TestSimulation:
    def test_scenario_1_normal_all_succeed(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-1", CodexState.NORMAL,
            {"codex_cli:openai_native": {"success": True}, "claude_code:anthropic": {"success": True}, "codex_cli:openrouter": {"success": True}})
        assert result.success
        assert result.tool == "codex_cli"
        assert result.backend == "openai_native"
        assert not decision.attempted_fallback

    def test_scenario_2_codex_quota_fallback_claude(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-2", CodexState.NORMAL,
            {"codex_cli:openai_native": {"success": False, "normalized_error": "quota_exhausted"}, "claude_code:anthropic": {"success": True}})
        assert result.success
        assert result.tool == "claude_code"
        assert decision.attempted_fallback
        assert decision.fallback_from == "codex_cli"

    def test_scenario_3_all_fail(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-3", CodexState.NORMAL,
            {"codex_cli:openai_native": {"success": False, "normalized_error": "quota_exhausted"},
             "claude_code:anthropic": {"success": False, "normalized_error": "rate_limited"},
             "codex_cli:openrouter": {"success": False, "normalized_error": "provider_unavailable"}})
        assert not result.success
        assert result.normalized_error == "provider_unavailable"
        assert decision.attempted_fallback

    def test_scenario_4_last10_claude_primary(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-4", CodexState.LAST10,
            {"claude_code:anthropic": {"success": True}})
        assert result.success
        assert result.tool == "claude_code"
        assert decision.state == "claude_backup"

    def test_scenario_5_last10_fallback_openrouter(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-5", CodexState.LAST10,
            {"claude_code:anthropic": {"success": False, "normalized_error": "rate_limited"}, "codex_cli:openrouter": {"success": True}})
        assert result.success
        assert result.tool == "codex_cli"
        assert decision.attempted_fallback
        assert decision.fallback_from == "claude_code"

    def test_scenario_6_screenshot_kimi(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-6", CodexState.NORMAL,
            {"codex_cli:openai_native": {"success": False, "normalized_error": "auth_error"},
             "claude_code:anthropic": {"success": False, "normalized_error": "auth_error"},
             "codex_cli:openrouter": {"success": True}}, has_screenshots=True)
        assert result.success
        assert result.model_profile == "openrouter_kimi"

    def test_scenario_7_fallback_on_auth(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-7", CodexState.NORMAL,
            {"codex_cli:openai_native": {"success": False, "normalized_error": "auth_error"}, "claude_code:anthropic": {"success": True}})
        assert result.success
        assert result.tool == "claude_code"
        assert decision.attempted_fallback

    def test_scenario_8_no_fallback_on_toolchain(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-8", CodexState.NORMAL,
            {"codex_cli:openai_native": {"success": False, "normalized_error": "toolchain_error"}, "claude_code:anthropic": {"success": True}})
        assert not result.success
        assert result.tool == "codex_cli"
        assert not decision.attempted_fallback

    def test_scenario_9_single_fallback(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-9", CodexState.NORMAL,
            {"codex_cli:openai_native": {"success": False, "normalized_error": "rate_limited"},
             "claude_code:anthropic": {"success": True}, "codex_cli:openrouter": {"success": True}})
        assert result.success
        assert result.tool == "claude_code"
        assert decision.attempted_fallback

    def test_scenario_10_kimi_for_multimodal(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-10", CodexState.NORMAL,
            {"codex_cli:openai_native": {"success": False, "normalized_error": "quota_exhausted"},
             "claude_code:anthropic": {"success": False, "normalized_error": "quota_exhausted"},
             "codex_cli:openrouter": {"success": True}}, task_class=TaskClass.MULTIMODAL_CODE_TASK, requires_multimodal=True)
        assert result.success
        assert result.model_profile == "openrouter_kimi"
        openrouter_entries = [e for e in decision.chain if e.backend == "openrouter"]
        assert openrouter_entries[-1].model_profile == "openrouter_kimi"

    # ── 4-state scenarios ──

    def test_scenario_11_openai_primary_success(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-11", CodexState.OPENAI_PRIMARY,
            {"codex_cli:openai_native": {"success": True}})
        assert result.success
        assert result.tool == "codex_cli"
        assert decision.state == "openai_primary"

    def test_scenario_12_openai_conservation_mini_handled(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-12", CodexState.OPENAI_CONSERVATION,
            {"codex_cli:openai_native": {"success": True}})
        assert result.success
        assert decision.chain[0].model_profile == "codex_gpt54_mini"

    def test_scenario_13_openai_conservation_escalates_to_full(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-13", CodexState.OPENAI_CONSERVATION,
            {"codex_cli:openai_native": {"success": True}}, risk=TaskRisk.CRITICAL)
        assert result.success
        assert decision.chain[0].model_profile == "codex_gpt54"

    def test_scenario_14_claude_backup_sonnet_handles(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-14", CodexState.CLAUDE_BACKUP,
            {"claude_code:anthropic": {"success": True}})
        assert result.success
        assert result.tool == "claude_code"
        assert decision.chain[0].model_profile == "claude_sonnet"
        assert decision.state == "claude_backup"

    def test_scenario_15_claude_backup_opus_hard_case(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-15", CodexState.CLAUDE_BACKUP,
            {"claude_code:anthropic": {"success": True}}, task_class=TaskClass.REPO_ARCHITECTURE_CHANGE)
        assert result.success
        assert decision.chain[0].model_profile == "claude_opus"

    def test_scenario_16_openrouter_fallback_minimax(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-16", CodexState.OPENROUTER_FALLBACK,
            {"codex_cli:openrouter": {"success": True}})
        assert result.success
        assert result.backend == "openrouter"
        assert result.model_profile == "openrouter_minimax"
        assert decision.state == "openrouter_fallback"

    def test_scenario_17_openrouter_fallback_mimo_hard(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-17", CodexState.OPENROUTER_FALLBACK,
            {"codex_cli:openrouter": {"success": True}}, risk=TaskRisk.CRITICAL)
        assert result.success
        assert result.model_profile == "openrouter_mimo"

    def test_scenario_18_openrouter_fallback_kimi_visual(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-18", CodexState.OPENROUTER_FALLBACK,
            {"codex_cli:openrouter": {"success": True}}, has_screenshots=True)
        assert result.success
        assert result.model_profile == "openrouter_kimi"

    def test_scenario_19_claude_backup_fallback_to_openrouter(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-19", CodexState.CLAUDE_BACKUP,
            {"claude_code:anthropic": {"success": False, "normalized_error": "rate_limited"}, "codex_cli:openrouter": {"success": True}})
        assert result.success
        assert result.backend == "openrouter"
        assert decision.attempted_fallback
        assert decision.fallback_from == "claude_code"

    def test_scenario_20_openai_conservation_fallback_chain(self, monkeypatch):
        task, decision, result = _setup_sim(monkeypatch, "sim-20", CodexState.OPENAI_CONSERVATION,
            {"codex_cli:openai_native": {"success": False, "normalized_error": "quota_exhausted"},
             "claude_code:anthropic": {"success": True}, "codex_cli:openrouter": {"success": True}})
        assert result.success
        assert result.backend == "anthropic"
        assert decision.attempted_fallback
        assert decision.fallback_from == "codex_cli"
