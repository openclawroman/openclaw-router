"""Tests for router schemas: TaskMeta, RouteDecision, ExecutorResult, enums, and config loading."""

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.models import (
    TaskMeta, RouteDecision, ExecutorResult, ChainEntry,
    TaskClass, TaskModality, TaskRisk, CodexState,
    Executor, ExecutorBackend, ModelProfile,
)


class TestEnums:
    def test_all_task_classes_present(self):
        expected = {"implementation", "refactor", "bugfix", "debug", "code_review",
                    "test_generation", "repo_architecture_change", "ui_from_screenshot",
                    "multimodal_code_task", "swarm_code_task", "planner", "final_review"}
        assert {tc.value for tc in TaskClass} == expected
        assert len(TaskClass) == 12

    def test_modality_values(self):
        assert {m.value for m in TaskModality} == {"text", "image", "video", "mixed"}

    def test_risk_values(self):
        assert {r.value for r in TaskRisk} == {"low", "medium", "high", "critical"}
        assert TaskRisk.HIGH.value == "high"

    def test_state_values(self):
        expected = {"openai_primary", "openai_conservation", "claude_backup", "openrouter_fallback"}
        assert {s.value for s in CodexState} == expected

    def test_backward_compat_aliases(self):
        assert CodexState.NORMAL == CodexState.OPENAI_PRIMARY
        assert CodexState.LAST10 == CodexState.CLAUDE_BACKUP


class TestTaskMeta:
    def test_all_required_fields(self):
        meta = TaskMeta(
            task_id="task_001", agent="coder", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT, requires_repo_write=True,
            requires_multimodal=False, has_screenshots=False, swarm=False,
            repo_path="/tmp/repo", cwd="/tmp/repo", summary="Add login endpoint",
        )
        assert meta.task_id == "task_001"
        assert meta.agent == "coder"
        assert meta.task_class == TaskClass.IMPLEMENTATION
        assert meta.requires_repo_write is True

    def test_field_names_match_spec(self):
        meta = TaskMeta()
        expected = {"task_id", "agent", "task_class", "phase", "risk", "modality",
                    "requires_repo_write", "requires_multimodal", "has_screenshots",
                    "swarm", "repo_path", "cwd", "summary"}
        assert set(vars(meta).keys()) == expected

    def test_no_legacy_fields(self):
        meta = TaskMeta()
        for removed in ["task_brief", "branch", "criticality", "context_size"]:
            assert not hasattr(meta, removed)

    @pytest.mark.parametrize("agent", ["coder", "reviewer", "architect", "designer", "worker"])
    def test_agent_values(self, agent):
        assert TaskMeta(agent=agent).agent == agent


class TestRouteDecision:
    def test_has_spec_fields(self):
        decision = RouteDecision(
            task_id="task_001", state="normal",
            chain=[ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary")],
            reason="normal: standard chain", attempted_fallback=False, fallback_from=None,
        )
        assert decision.task_id == "task_001"
        assert decision.state == "normal"
        assert len(decision.chain) == 1
        assert decision.chain[0].tool == "codex_cli"

    def test_no_execution_time_fields(self):
        forbidden = {"tool", "backend", "model_profile", "success", "normalized_error",
                     "exit_code", "latency_ms", "request_id", "cost_estimate_usd",
                     "artifacts", "stdout_ref", "stderr_ref", "final_summary",
                     "executor", "model", "status", "error_type"}
        assert not (set(vars(RouteDecision()).keys()) & forbidden)

    def test_spec_field_names(self):
        expected = {"task_id", "state", "chain", "reason", "attempted_fallback",
                    "fallback_from", "providers_skipped", "chain_timed_out",
                    "fallback_count", "trace_id", "error_history", "phase"}
        assert set(vars(RouteDecision()).keys()) == expected


class TestExecutorResult:
    def test_has_spec_fields(self):
        result = ExecutorResult(
            task_id="task_001", tool="codex_cli", backend="openai_native",
            model_profile="codex_primary", success=True, normalized_error=None,
            exit_code=0, latency_ms=1234, request_id="req_abc", cost_estimate_usd=0.001,
            artifacts=["file.py"], stdout_ref="output.txt", stderr_ref=None, final_summary="Done",
        )
        assert result.success is True
        assert result.latency_ms == 1234
        assert result.request_id == "req_abc"
        assert result.artifacts == ["file.py"]

    def test_has_latency_ms(self):
        assert ExecutorResult().latency_ms == 0

    def test_no_routing_fields(self):
        forbidden = {"state", "chain", "reason", "attempted_fallback", "fallback_from"}
        assert not (set(vars(ExecutorResult()).keys()) & forbidden)

    def test_spec_field_names(self):
        expected = {"task_id", "tool", "backend", "model_profile", "success",
                    "normalized_error", "exit_code", "latency_ms", "request_id",
                    "cost_estimate_usd", "artifacts", "stdout_ref", "stderr_ref",
                    "final_summary", "trace_id", "rate_limit_info",
                    "partial_success", "warnings", "error_history"}
        assert set(vars(ExecutorResult()).keys()) == expected

    def test_separate_from_route_decision(self):
        shared = set(vars(RouteDecision()).keys()) & set(vars(ExecutorResult()).keys())
        assert shared == {"task_id", "trace_id", "error_history"}


class TestChainEntry:
    def test_fields(self):
        entry = ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary")
        assert entry.tool == "codex_cli"
        assert entry.backend == "openai_native"
        assert entry.model_profile == "codex_primary"


class TestConfigLoading:
    @pytest.fixture(autouse=True)
    def load_config(self):
        config_path = Path(__file__).parent.parent / "config" / "router.config.json"
        with open(config_path) as f:
            self.config = json.load(f)

    def test_version_is_1(self):
        assert self.config["version"] == 1

    def test_has_required_top_level_keys(self):
        required = {"version", "state", "openclaw", "tools", "routing", "openrouter_dynamic_rules", "retry", "logging"}
        assert required.issubset(set(self.config.keys()))

    def test_state_config(self):
        state = self.config["state"]
        assert state["default"] == "openai_primary"
        assert "manual_state_file" in state
        assert "auto_state_file" in state

    def test_routing_chains(self):
        normal = self.config["routing"]["normal"]["chain"]
        assert len(normal) == 3
        assert normal[0]["tool"] == "codex_cli"
        assert normal[0]["backend"] == "openai_native"
        assert normal[1]["tool"] == "claude_code"
        assert normal[2]["tool"] == "codex_cli"

        last10 = self.config["routing"]["last10"]["chain"]
        assert len(last10) == 2
        assert last10[0]["tool"] == "claude_code"

    def test_openrouter_dynamic_rules(self):
        rules = self.config["openrouter_dynamic_rules"]
        assert rules["default_model"] == "minimax"
        for key in ["ui_from_screenshot", "multimodal_code_task", "swarm_code_task", "has_screenshots"]:
            assert rules["specialist_rules"][key] == "kimi"

    def test_retry_eligible_errors(self):
        eligible = set(self.config["retry"]["eligible_errors"])
        assert eligible == {"auth_error", "rate_limited", "quota_exhausted",
                            "provider_unavailable", "provider_timeout", "transient_network_error"}

    def test_tools_profiles(self):
        profiles = self.config["tools"]["codex_cli"]["profiles"]
        assert "openai_native" in profiles
        assert "openrouter" in profiles
        assert self.config["tools"]["claude_code"]["provider"] == "anthropic"


class TestImports:
    def test_exports(self):
        import router
        expected = {
            "TaskMeta", "RouteDecision", "ExecutorResult", "ChainEntry",
            "TaskClass", "TaskModality", "TaskRisk", "CodexState",
            "Executor", "ExecutorBackend", "ModelProfile",
            "route_task", "resolve_state", "build_chain",
            "choose_openrouter_profile", "can_fallback",
            "StateStore", "RoutingLogger",
            "Classifier", "classify", "classify_from_dict",
            "RouterError", "ExecutorError", "StateError", "ConfigurationError",
            "normalize_error", "NORMALIZED_ERROR_TYPES",
            "run_codex", "run_claude", "run_openrouter",
            "run_codex_openrouter_minimax", "run_codex_openrouter_kimi",
            "OutputFormat", "FormatValidationError",
            "parse_output_format", "validate_format_for_task",
            "get_default_format", "resolve_output_format",
            "PipelinePhase", "PhaseConfig", "FlowConfig",
            "PhaseResult", "PipelineResult",
            "requires_three_phase", "select_flow",
            "load_config", "get_model", "reload_config",
            "get_config_snapshot", "ConfigValidationError",
        }
        assert set(router.__all__) == expected
