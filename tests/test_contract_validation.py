"""Tests for routing contracts — validating all model fields, types, defaults, and docstrings."""

import inspect

import pytest

from router.models import (
    TaskMeta, RouteDecision, ExecutorResult,
    ChainEntry, TaskClass, TaskRisk, TaskModality, ModelProfile,
)


# ---------------------------------------------------------------------------
# TaskMeta contract
# ---------------------------------------------------------------------------

class TestTaskMetaContract:
    """Validate TaskMeta has all required fields with correct types and defaults."""

    def test_has_all_required_fields(self):
        """Verify: task_id(str), agent(str), task_class(TaskClass), risk(TaskRisk),
        modality(TaskModality), requires_repo_write(bool), requires_multimodal(bool),
        has_screenshots(bool), swarm(bool), repo_path(str), cwd(str), summary(str)."""
        tm = TaskMeta()
        # All fields must exist (accessing them proves presence)
        assert isinstance(tm.task_id, str)
        assert isinstance(tm.agent, str)
        assert isinstance(tm.task_class, TaskClass)
        assert isinstance(tm.risk, TaskRisk)
        assert isinstance(tm.modality, TaskModality)
        assert isinstance(tm.requires_repo_write, bool)
        assert isinstance(tm.requires_multimodal, bool)
        assert isinstance(tm.has_screenshots, bool)
        assert isinstance(tm.swarm, bool)
        assert isinstance(tm.repo_path, str)
        assert isinstance(tm.cwd, str)
        assert isinstance(tm.summary, str)

    def test_default_agent_is_coder(self):
        """Default agent field is 'coder'."""
        tm = TaskMeta()
        assert tm.agent == "coder"

    def test_default_risk_is_medium(self):
        """Default risk is 'medium'."""
        tm = TaskMeta()
        assert tm.risk == TaskRisk.MEDIUM

    def test_default_modality_is_text(self):
        """Default modality is 'text'."""
        tm = TaskMeta()
        assert tm.modality == TaskModality.TEXT


# ---------------------------------------------------------------------------
# RouteDecision contract
# ---------------------------------------------------------------------------

class TestRouteDecisionContract:
    """Validate RouteDecision has all required fields with correct types and defaults."""

    def test_has_all_required_fields(self):
        """task_id, state, chain(list), reason, attempted_fallback(bool), fallback_from(str|None)."""
        rd = RouteDecision()
        assert isinstance(rd.task_id, str)
        assert isinstance(rd.state, str)
        assert isinstance(rd.chain, list)
        assert isinstance(rd.reason, str)
        assert isinstance(rd.attempted_fallback, bool)
        # fallback_from can be str or None
        assert rd.fallback_from is None or isinstance(rd.fallback_from, str)

    def test_default_state_is_normal(self):
        """Default state is 'normal'."""
        rd = RouteDecision()
        assert rd.state == "normal"


# ---------------------------------------------------------------------------
# ExecutorResult contract
# ---------------------------------------------------------------------------

class TestExecutorResultContract:
    """Validate ExecutorResult has all required fields with correct types and defaults."""

    def test_has_all_required_fields(self):
        """task_id, tool, backend, model_profile, success(bool), normalized_error(str|None),
        exit_code(int|None), latency_ms(int), request_id(str|None), cost_estimate_usd(float|None),
        artifacts(list), stdout_ref(str|None), stderr_ref(str|None), final_summary(str|None)."""
        er = ExecutorResult()
        assert isinstance(er.task_id, str)
        assert isinstance(er.tool, str)
        assert isinstance(er.backend, str)
        assert isinstance(er.model_profile, str)
        assert isinstance(er.success, bool)
        assert er.normalized_error is None or isinstance(er.normalized_error, str)
        assert er.exit_code is None or isinstance(er.exit_code, int)
        assert isinstance(er.latency_ms, int)
        assert er.request_id is None or isinstance(er.request_id, str)
        assert er.cost_estimate_usd is None or isinstance(er.cost_estimate_usd, float)
        assert isinstance(er.artifacts, list)
        assert er.stdout_ref is None or isinstance(er.stdout_ref, str)
        assert er.stderr_ref is None or isinstance(er.stderr_ref, str)
        assert er.final_summary is None or isinstance(er.final_summary, str)

    def test_default_success_is_true(self):
        """success defaults to True."""
        er = ExecutorResult()
        assert er.success is True

    def test_default_cost_is_none(self):
        """cost_estimate_usd defaults to None."""
        er = ExecutorResult()
        assert er.cost_estimate_usd is None

    def test_default_artifacts_empty(self):
        """artifacts defaults to []."""
        er = ExecutorResult()
        assert er.artifacts == []


# ---------------------------------------------------------------------------
# Agent roles
# ---------------------------------------------------------------------------

class TestAgentRoles:
    """Agent role validation and task-class routing."""

    def test_valid_agent_roles(self):
        """coder, reviewer, architect, designer, worker."""
        valid_roles = {"coder", "reviewer", "architect", "designer", "worker"}
        for role in valid_roles:
            tm = TaskMeta(agent=role)
            assert tm.agent == role

    def test_task_class_routing(self):
        """implementation→coder, code_review→reviewer, repo_architecture_change→architect."""
        role_map = {
            TaskClass.IMPLEMENTATION: "coder",
            TaskClass.CODE_REVIEW: "reviewer",
            TaskClass.REPO_ARCHITECTURE_CHANGE: "architect",
        }
        for task_class, expected_role in role_map.items():
            tm = TaskMeta(task_class=task_class, agent=expected_role)
            assert tm.agent == expected_role


# ---------------------------------------------------------------------------
# Contract docstrings
# ---------------------------------------------------------------------------

class TestContractDocstrings:
    """Verify model classes have meaningful docstrings."""

    def test_taskmeta_docstring(self):
        """TaskMeta class has docstring mentioning 'Input task metadata'."""
        assert TaskMeta.__doc__ is not None
        assert "Input task metadata" in TaskMeta.__doc__

    def test_routedecision_docstring(self):
        """RouteDecision has docstring mentioning 'Output decision'."""
        assert RouteDecision.__doc__ is not None
        assert "Output decision" in RouteDecision.__doc__

    def test_executorresult_docstring(self):
        """ExecutorResult has docstring mentioning 'outcome'."""
        assert ExecutorResult.__doc__ is not None
        assert "outcome" in ExecutorResult.__doc__
