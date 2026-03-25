"""Tests for OpenAI profile selection (gpt-5.4 vs gpt-5.4-mini)."""

import pytest
from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState
from router.policy import choose_openai_profile, build_chain


class TestChooseOpenaiProfile:
    def test_implementation_gets_mini(self):
        task = TaskMeta(task_id="t1", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"

    def test_low_risk_gets_mini(self):
        task = TaskMeta(task_id="t2", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"

    def test_medium_risk_gets_mini(self):
        task = TaskMeta(task_id="t3", task_class=TaskClass.BUGFIX, risk=TaskRisk.MEDIUM)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"

    def test_critical_risk_gets_full(self):
        task = TaskMeta(task_id="t4", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.CRITICAL)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4"

    def test_high_risk_gets_mini(self):
        """HIGH risk alone doesn't trigger gpt-5.4, only CRITICAL does."""
        task = TaskMeta(task_id="t5", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.HIGH)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"

    def test_architecture_gets_full(self):
        task = TaskMeta(task_id="t6", task_class=TaskClass.REPO_ARCHITECTURE_CHANGE, risk=TaskRisk.LOW)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4"

    def test_debug_gets_full(self):
        task = TaskMeta(task_id="t7", task_class=TaskClass.DEBUG, risk=TaskRisk.MEDIUM)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4"

    def test_refactor_gets_mini(self):
        task = TaskMeta(task_id="t8", task_class=TaskClass.REFACTOR, risk=TaskRisk.MEDIUM)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"

    def test_code_review_gets_mini(self):
        task = TaskMeta(task_id="t9", task_class=TaskClass.CODE_REVIEW, risk=TaskRisk.MEDIUM)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"

    def test_test_generation_gets_mini(self):
        task = TaskMeta(task_id="t10", task_class=TaskClass.TEST_GENERATION, risk=TaskRisk.MEDIUM)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"

    def test_ui_from_screenshot_gets_mini(self):
        task = TaskMeta(task_id="t11", task_class=TaskClass.UI_FROM_SCREENSHOT, risk=TaskRisk.MEDIUM)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"

    def test_multimodal_gets_mini(self):
        task = TaskMeta(task_id="t12", task_class=TaskClass.MULTIMODAL_CODE_TASK, risk=TaskRisk.MEDIUM)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"

    def test_swarm_gets_mini(self):
        task = TaskMeta(task_id="t13", task_class=TaskClass.SWARM_CODE_TASK, risk=TaskRisk.MEDIUM)
        model = choose_openai_profile(task)
        assert model == "gpt-5.4-mini"


class TestOpenaiProfileInChain:
    """Verify the chain uses the right profile based on task."""

    def test_normal_chain_has_openai_entry(self):
        task = TaskMeta(task_id="t20", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM)
        chain = build_chain(task, CodexState.NORMAL)
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"

    def test_critical_chain_uses_gpt54(self):
        task = TaskMeta(task_id="t21", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.CRITICAL)
        chain = build_chain(task, CodexState.NORMAL)
        assert chain[0].model_profile == "codex_gpt54"

    def test_normal_impl_uses_mini(self):
        task = TaskMeta(task_id="t22", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM)
        chain = build_chain(task, CodexState.NORMAL)
        assert chain[0].model_profile == "codex_gpt54_mini"

    def test_debug_chain_uses_gpt54(self):
        task = TaskMeta(task_id="t23", task_class=TaskClass.DEBUG, risk=TaskRisk.LOW)
        chain = build_chain(task, CodexState.NORMAL)
        assert chain[0].model_profile == "codex_gpt54"

    def test_architecture_chain_uses_gpt54(self):
        task = TaskMeta(task_id="t24", task_class=TaskClass.REPO_ARCHITECTURE_CHANGE, risk=TaskRisk.MEDIUM)
        chain = build_chain(task, CodexState.NORMAL)
        assert chain[0].model_profile == "codex_gpt54"

    def test_last10_no_openai(self):
        """Last10 state should not have openai_native entry."""
        task = TaskMeta(task_id="t25", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.CRITICAL)
        chain = build_chain(task, CodexState.LAST10)
        openai_entries = [e for e in chain if e.backend == "openai_native"]
        assert len(openai_entries) == 0


class TestModelConfig:
    """Verify model strings are in config."""

    def test_gpt54_model_in_config(self):
        from router.config_loader import get_model
        assert get_model("gpt54") == "gpt-5.4"

    def test_gpt54_mini_model_in_config(self):
        from router.config_loader import get_model
        assert get_model("gpt54_mini") == "gpt-5.4-mini"
