"""Tests for 4-state subscription-aware state machine."""

import json
import tempfile
import pytest

from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState, ModelProfile


class TestCodexStateEnum:
    def test_has_four_states(self):
        states = {s.value for s in CodexState}
        assert {"openai_primary", "openai_conservation", "claude_backup", "openrouter_fallback"}.issubset(states)

    def test_backward_compat_aliases(self):
        assert CodexState.NORMAL == CodexState.OPENAI_PRIMARY
        assert CodexState.LAST10 == CodexState.CLAUDE_BACKUP


class TestModelProfileEnum:
    @pytest.mark.parametrize("profile", ["claude_sonnet", "claude_opus", "openrouter_mimo"])
    def test_profiles_exist(self, profile):
        assert profile in [p.value for p in ModelProfile]


class TestStateStoreFourStates:
    def test_validates_all_four(self, tmp_path):
        from router.state_store import StateStore
        store = StateStore(manual_path=tmp_path / "manual.json", auto_path=tmp_path / "auto.json")
        for state in CodexState:
            if state in (CodexState.NORMAL, CodexState.LAST10):
                continue
            assert StateStore._validate_state(state.value) == state

    @pytest.mark.parametrize("alias,expected", [("normal", CodexState.OPENAI_PRIMARY), ("last10", CodexState.CLAUDE_BACKUP)])
    def test_backward_compat_input(self, tmp_path, alias, expected):
        from router.state_store import StateStore
        store = StateStore(manual_path=tmp_path / "manual.json", auto_path=tmp_path / "auto.json")
        assert StateStore._validate_state(alias) == expected

    def test_default_is_openai_primary(self, tmp_path):
        from router.state_store import StateStore
        store = StateStore(manual_path=tmp_path / "manual.json", auto_path=tmp_path / "auto.json")
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_set_and_get_new_states(self, tmp_path):
        from router.state_store import StateStore
        store = StateStore(manual_path=tmp_path / "manual.json", auto_path=tmp_path / "auto.json")
        for state in CodexState:
            if state in (CodexState.NORMAL, CodexState.LAST10):
                continue
            store.set_manual_state(state)
            assert store.get_manual_state() == state


class TestBuildChainFourStates:
    def _task(self, **kw):
        defaults = dict(task_id="t1", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT)
        defaults.update(kw)
        return TaskMeta(**defaults)

    def _chain(self, state, **task_kw):
        from router.policy import build_chain
        return build_chain(self._task(**task_kw), state)

    @pytest.mark.parametrize("state,expected_first", [
        (CodexState.OPENAI_PRIMARY, ("codex_cli", "openai_native")),
        (CodexState.OPENAI_CONSERVATION, ("codex_cli", "openai_native")),
        (CodexState.CLAUDE_BACKUP, ("claude_code", "anthropic")),
    ])
    def test_first_in_chain(self, state, expected_first):
        chain = self._chain(state)
        assert (chain[0].tool, chain[0].backend) == expected_first

    def test_openai_primary_has_three_entries(self):
        chain = self._chain(CodexState.OPENAI_PRIMARY)
        assert len(chain) == 3
        assert (chain[1].tool, chain[1].backend) == ("claude_code", "anthropic")
        assert (chain[2].tool, chain[2].backend) == ("codex_cli", "openrouter")

    def test_openai_conservation_model_selection(self):
        chain = self._chain(CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54_mini"
        critical_chain = self._chain(CodexState.OPENAI_CONSERVATION, risk=TaskRisk.CRITICAL)
        assert critical_chain[0].model_profile == "codex_gpt54"
        assert len(chain) == 3

    def test_claude_backup_model_selection(self):
        chain = self._chain(CodexState.CLAUDE_BACKUP)
        assert chain[0].model_profile == "claude_sonnet"
        assert len(chain) == 2
        assert chain[1].backend == "openrouter"
        critical_chain = self._chain(CodexState.CLAUDE_BACKUP, risk=TaskRisk.CRITICAL)
        assert critical_chain[0].model_profile == "claude_opus"

    def test_openrouter_fallback_only_openrouter(self):
        chain = self._chain(CodexState.OPENROUTER_FALLBACK)
        assert len(chain) == 1
        assert chain[0].backend == "openrouter"
        assert chain[0].model_profile == "openrouter_minimax"

    @pytest.mark.parametrize("alias_state,real_state", [
        (CodexState.NORMAL, CodexState.OPENAI_PRIMARY),
        (CodexState.LAST10, CodexState.CLAUDE_BACKUP),
    ])
    def test_backward_compat_chain(self, alias_state, real_state):
        chain_alias = self._chain(alias_state)
        chain_real = self._chain(real_state)
        assert len(chain_alias) == len(chain_real)
        for a, b in zip(chain_alias, chain_real):
            assert a.tool == b.tool
            assert a.backend == b.backend


class TestOpenRouterProfileSelection:
    def _task(self, **kw):
        defaults = dict(task_id="t1", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT)
        defaults.update(kw)
        return TaskMeta(**defaults)

    @pytest.mark.parametrize("kwargs,expected", [
        ({}, ModelProfile.OPENROUTER_MINIMAX),
        ({"has_screenshots": True}, ModelProfile.OPENROUTER_KIMI),
        ({"requires_multimodal": True}, ModelProfile.OPENROUTER_KIMI),
        ({"risk": TaskRisk.CRITICAL}, ModelProfile.OPENROUTER_MIMO),
        ({"task_class": TaskClass.DEBUG}, ModelProfile.OPENROUTER_MIMO),
        ({"task_class": TaskClass.REPO_ARCHITECTURE_CHANGE}, ModelProfile.OPENROUTER_MIMO),
    ])
    def test_profile_selection(self, kwargs, expected):
        from router.policy import choose_openrouter_profile
        assert choose_openrouter_profile(self._task(**kwargs)) == expected


class TestClaudeModelSelection:
    def _task(self, **kw):
        defaults = dict(task_id="t1", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT)
        defaults.update(kw)
        return TaskMeta(**defaults)

    @pytest.mark.parametrize("kwargs,expected_model,expected_profile", [
        ({}, "claude-sonnet-4.6", "claude_sonnet"),
        ({"risk": TaskRisk.CRITICAL}, "claude-opus-4.6", "claude_opus"),
        ({"task_class": TaskClass.REPO_ARCHITECTURE_CHANGE}, "claude-opus-4.6", "claude_opus"),
    ])
    def test_claude_selection(self, kwargs, expected_model, expected_profile):
        from router.policy import choose_claude_model, choose_claude_profile
        assert choose_claude_model(self._task(**kwargs)) == expected_model
        assert choose_claude_profile(self._task(**kwargs)) == expected_profile


class TestResolveState:
    def _store(self, tmp_path):
        from router.state_store import StateStore
        return StateStore(manual_path=tmp_path / "manual.json", auto_path=tmp_path / "auto.json")

    def test_default_is_openai_primary(self, tmp_path):
        from router.policy import resolve_state
        assert resolve_state(store=self._store(tmp_path)) == CodexState.OPENAI_PRIMARY

    def test_manual_overrides(self, tmp_path):
        from router.policy import resolve_state
        store = self._store(tmp_path)
        m_path = tmp_path / "manual.json"
        m_path.write_text(json.dumps({"state": "openai_conservation"}))
        assert resolve_state(store=store) == CodexState.OPENAI_CONSERVATION

    def test_manual_overrides_auto(self, tmp_path):
        from router.policy import resolve_state
        store = self._store(tmp_path)
        (tmp_path / "auto.json").write_text(json.dumps({"state": "claude_backup"}))
        (tmp_path / "manual.json").write_text(json.dumps({"state": "openai_conservation"}))
        assert resolve_state(store=store) == CodexState.OPENAI_CONSERVATION

    def test_auto_state_used_when_no_manual(self, tmp_path):
        from router.policy import resolve_state
        store = self._store(tmp_path)
        (tmp_path / "manual.json").write_text(json.dumps({"state": None}))
        (tmp_path / "auto.json").write_text(json.dumps({"state": "claude_backup"}))
        assert resolve_state(store=store) == CodexState.CLAUDE_BACKUP


class TestNewModelConfig:
    @pytest.mark.parametrize("name,expected", [("mimo", "xiaomi/mimo-v2-pro"), ("sonnet", "claude-sonnet-4.6"), ("opus", "claude-opus-4.6")])
    def test_model_configs(self, name, expected):
        from router.config_loader import get_model
        assert get_model(name) == expected
