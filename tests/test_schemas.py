"""Tests for router schemas: TaskMeta, RouteDecision, ExecutorResult, enums, and config loading."""

import json
import os
import sys
import unittest
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from router.models import (
    TaskMeta, RouteDecision, ExecutorResult, ChainEntry,
    TaskClass, TaskModality, TaskRisk, CodexState,
    Executor, ExecutorBackend, ModelProfile,
)


class TestTaskClassEnum(unittest.TestCase):
    """Test that TaskClass enum covers all values from the plan."""

    EXPECTED_VALUES = {
        "implementation",
        "refactor",
        "bugfix",
        "debug",
        "code_review",
        "test_generation",
        "repo_architecture_change",
        "ui_from_screenshot",
        "multimodal_code_task",
        "swarm_code_task",
    }

    def test_all_task_classes_present(self):
        actual = {tc.value for tc in TaskClass}
        self.assertEqual(actual, self.EXPECTED_VALUES)

    def test_task_class_count(self):
        self.assertEqual(len(TaskClass), 10)


class TestTaskModalityEnum(unittest.TestCase):
    """Test TaskModality enum."""

    def test_modality_values(self):
        expected = {"text", "image", "video", "mixed"}
        actual = {m.value for m in TaskModality}
        self.assertEqual(actual, expected)


class TestTaskRiskEnum(unittest.TestCase):
    """Test TaskRisk enum has correct values (not 'high_risk')."""

    def test_risk_values(self):
        expected = {"low", "medium", "high", "critical"}
        actual = {r.value for r in TaskRisk}
        self.assertEqual(actual, expected)

    def test_high_not_high_risk(self):
        self.assertEqual(TaskRisk.HIGH.value, "high")


class TestCodexStateEnum(unittest.TestCase):
    def test_state_values(self):
        expected = {"openai_primary", "openai_conservation", "claude_backup", "openrouter_fallback"}
        actual = {s.value for s in CodexState}
        self.assertEqual(actual, expected)

    def test_backward_compat_aliases(self):
        """NORMAL and LAST10 still work as aliases."""
        self.assertEqual(CodexState.NORMAL, CodexState.OPENAI_PRIMARY)
        self.assertEqual(CodexState.LAST10, CodexState.CLAUDE_BACKUP)


class TestTaskMeta(unittest.TestCase):
    """Test TaskMeta creation with all required fields."""

    def test_all_required_fields_exist(self):
        meta = TaskMeta(
            task_id="task_001",
            agent="coder",
            task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM,
            modality=TaskModality.TEXT,
            requires_repo_write=True,
            requires_multimodal=False,
            has_screenshots=False,
            swarm=False,
            repo_path="/tmp/repo",
            cwd="/tmp/repo",
            summary="Add login endpoint",
        )
        self.assertEqual(meta.task_id, "task_001")
        self.assertEqual(meta.agent, "coder")
        self.assertEqual(meta.task_class, TaskClass.IMPLEMENTATION)
        self.assertEqual(meta.risk, TaskRisk.MEDIUM)
        self.assertEqual(meta.modality, TaskModality.TEXT)
        self.assertTrue(meta.requires_repo_write)
        self.assertFalse(meta.requires_multimodal)
        self.assertFalse(meta.has_screenshots)
        self.assertFalse(meta.swarm)
        self.assertEqual(meta.repo_path, "/tmp/repo")
        self.assertEqual(meta.cwd, "/tmp/repo")
        self.assertEqual(meta.summary, "Add login endpoint")

    def test_all_field_names_match_spec(self):
        """Ensure TaskMeta has exactly the spec fields, no extras."""
        meta = TaskMeta()
        expected_fields = {
            "task_id", "agent", "task_class", "risk", "modality",
            "requires_repo_write", "requires_multimodal", "has_screenshots",
            "swarm", "repo_path", "cwd", "summary",
        }
        actual_fields = set(vars(meta).keys())
        self.assertEqual(actual_fields, expected_fields,
                         f"Extra: {actual_fields - expected_fields}, Missing: {expected_fields - actual_fields}")

    def test_no_legacy_fields(self):
        """TaskMeta must NOT have removed fields."""
        meta = TaskMeta()
        for removed in ["task_brief", "branch", "criticality", "context_size"]:
            self.assertFalse(hasattr(meta, removed),
                             f"TaskMeta should not have field '{removed}'")

    def test_agent_values(self):
        """Agent field should accept the spec values."""
        for agent in ["coder", "reviewer", "architect", "designer", "worker"]:
            meta = TaskMeta(agent=agent)
            self.assertEqual(meta.agent, agent)


class TestRouteDecision(unittest.TestCase):
    """Test RouteDecision is a pure routing decision with NO execution-time fields."""

    def test_has_spec_fields(self):
        decision = RouteDecision(
            task_id="task_001",
            state="normal",
            chain=[ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary")],
            reason="normal: standard chain",
            attempted_fallback=False,
            fallback_from=None,
        )
        self.assertEqual(decision.task_id, "task_001")
        self.assertEqual(decision.state, "normal")
        self.assertEqual(len(decision.chain), 1)
        self.assertEqual(decision.chain[0].tool, "codex_cli")
        self.assertEqual(decision.reason, "normal: standard chain")
        self.assertFalse(decision.attempted_fallback)
        self.assertIsNone(decision.fallback_from)

    def test_no_execution_time_fields(self):
        """RouteDecision must NOT have executor result fields."""
        decision = RouteDecision()
        forbidden = {
            "tool", "backend", "model_profile", "success", "normalized_error",
            "exit_code", "latency_ms", "request_id", "cost_estimate_usd",
            "artifacts", "stdout_ref", "stderr_ref", "final_summary",
            "executor", "model", "status", "error_type",
        }
        actual = set(vars(decision).keys())
        leaked = actual & forbidden
        self.assertEqual(leaked, set(),
                         f"RouteDecision has forbidden execution-time fields: {leaked}")

    def test_spec_field_names(self):
        decision = RouteDecision()
        expected = {
            "task_id", "state", "chain", "reason",
            "attempted_fallback", "fallback_from",
            "providers_skipped",
            "chain_timed_out", "fallback_count",
        }
        actual = set(vars(decision).keys())
        self.assertEqual(actual, expected)


class TestExecutorResult(unittest.TestCase):
    """Test ExecutorResult is a separate type from RouteDecision."""

    def test_has_spec_fields(self):
        result = ExecutorResult(
            task_id="task_001",
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
            normalized_error=None,
            exit_code=0,
            latency_ms=1234,
            request_id="req_abc",
            cost_estimate_usd=0.001,
            artifacts=["file.py"],
            stdout_ref="output.txt",
            stderr_ref=None,
            final_summary="Done",
        )
        self.assertEqual(result.task_id, "task_001")
        self.assertEqual(result.tool, "codex_cli")
        self.assertEqual(result.backend, "openai_native")
        self.assertEqual(result.model_profile, "codex_primary")
        self.assertTrue(result.success)
        self.assertEqual(result.latency_ms, 1234)
        self.assertEqual(result.request_id, "req_abc")
        self.assertEqual(result.cost_estimate_usd, 0.001)
        self.assertEqual(result.artifacts, ["file.py"])
        self.assertEqual(result.stdout_ref, "output.txt")
        self.assertEqual(result.final_summary, "Done")

    def test_has_latency_ms(self):
        """latency_ms is a required field in the spec."""
        result = ExecutorResult()
        self.assertTrue(hasattr(result, "latency_ms"))
        self.assertEqual(result.latency_ms, 0)

    def test_no_routing_fields(self):
        """ExecutorResult must NOT have RouteDecision fields."""
        result = ExecutorResult()
        forbidden = {"state", "chain", "reason", "attempted_fallback", "fallback_from"}
        actual = set(vars(result).keys())
        leaked = actual & forbidden
        self.assertEqual(leaked, set(),
                         f"ExecutorResult has forbidden routing fields: {leaked}")

    def test_spec_field_names(self):
        result = ExecutorResult()
        expected = {
            "task_id", "tool", "backend", "model_profile", "success",
            "normalized_error", "exit_code", "latency_ms", "request_id",
            "cost_estimate_usd", "artifacts", "stdout_ref", "stderr_ref",
            "final_summary",
        }
        actual = set(vars(result).keys())
        self.assertEqual(actual, expected)

    def test_separate_from_route_decision(self):
        """RouteDecision and ExecutorResult share NO fields except task_id."""
        rd_fields = set(vars(RouteDecision()).keys())
        er_fields = set(vars(ExecutorResult()).keys())
        shared = rd_fields & er_fields
        self.assertEqual(shared, {"task_id"},
                         f"Unexpected shared fields: {shared}")


class TestChainEntry(unittest.TestCase):
    """Test ChainEntry dataclass."""

    def test_fields(self):
        entry = ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary")
        self.assertEqual(entry.tool, "codex_cli")
        self.assertEqual(entry.backend, "openai_native")
        self.assertEqual(entry.model_profile, "codex_primary")


class TestConfigLoading(unittest.TestCase):
    """Test that router.config.json loads and matches expected schema."""

    def setUp(self):
        config_path = Path(__file__).parent.parent / "config" / "router.config.json"
        with open(config_path) as f:
            self.config = json.load(f)

    def test_version_is_1(self):
        self.assertEqual(self.config["version"], 1)

    def test_has_required_top_level_keys(self):
        required = {"version", "state", "openclaw", "tools", "routing",
                     "openrouter_dynamic_rules", "retry", "logging"}
        actual = set(self.config.keys())
        self.assertTrue(required.issubset(actual),
                        f"Missing keys: {required - actual}")

    def test_state_config(self):
        state = self.config["state"]
        self.assertEqual(state["default"], "openai_primary")
        self.assertIn("manual_state_file", state)
        self.assertIn("auto_state_file", state)
        self.assertIn("claude_health_file", state)

    def test_routing_normal_chain(self):
        chain = self.config["routing"]["normal"]["chain"]
        self.assertIsInstance(chain, list)
        self.assertEqual(len(chain), 3)
        # First: codex_cli openai_native
        self.assertEqual(chain[0]["tool"], "codex_cli")
        self.assertEqual(chain[0]["backend"], "openai_native")
        # Second: claude_code anthropic
        self.assertEqual(chain[1]["tool"], "claude_code")
        self.assertEqual(chain[1]["backend"], "anthropic")
        # Third: codex_cli openrouter
        self.assertEqual(chain[2]["tool"], "codex_cli")
        self.assertEqual(chain[2]["backend"], "openrouter")

    def test_routing_last10_chain(self):
        chain = self.config["routing"]["last10"]["chain"]
        self.assertIsInstance(chain, list)
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0]["tool"], "claude_code")
        self.assertEqual(chain[1]["tool"], "codex_cli")

    def test_openrouter_dynamic_rules(self):
        rules = self.config["openrouter_dynamic_rules"]
        self.assertEqual(rules["default_model"], "minimax")
        specialist = rules["specialist_rules"]
        for key in ["ui_from_screenshot", "multimodal_code_task", "swarm_code_task", "has_screenshots"]:
            self.assertEqual(specialist[key], "kimi")

    def test_retry_eligible_errors(self):
        eligible = self.config["retry"]["eligible_errors"]
        expected = {"auth_error", "rate_limited", "quota_exhausted",
                     "provider_unavailable", "provider_timeout", "transient_network_error"}
        self.assertEqual(set(eligible), expected)

    def test_tools_codex_cli_profiles(self):
        profiles = self.config["tools"]["codex_cli"]["profiles"]
        self.assertIn("openai_native", profiles)
        self.assertIn("openrouter", profiles)

    def test_tools_claude_code(self):
        claude = self.config["tools"]["claude_code"]
        self.assertEqual(claude["provider"], "anthropic")


class TestImports(unittest.TestCase):
    """Test that __init__.py exports match expected names."""

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
        }
        actual = set(router.__all__)
        self.assertEqual(actual, expected,
                         f"Extra: {actual - expected}, Missing: {expected - actual}")


if __name__ == "__main__":
    unittest.main()
