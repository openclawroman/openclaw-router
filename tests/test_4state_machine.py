"""Tests for 4-state subscription-aware state machine."""

import json
import os
import tempfile

import pytest
from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    CodexState, ModelProfile, ChainEntry,
)


class TestCodexStateEnum:
    """Verify the 4-state enum exists with backward compat."""

    def test_has_four_states(self):
        states = [s.value for s in CodexState]
        assert "openai_primary" in states
        assert "openai_conservation" in states
        assert "claude_backup" in states
        assert "openrouter_fallback" in states

    def test_backward_compat_normal_alias(self):
        assert CodexState.NORMAL == CodexState.OPENAI_PRIMARY
        assert CodexState.NORMAL.value == "openai_primary"

    def test_backward_compat_last10_alias(self):
        assert CodexState.LAST10 == CodexState.CLAUDE_BACKUP
        assert CodexState.LAST10.value == "claude_backup"


class TestModelProfileEnum:
    """Verify new model profiles exist."""

    def test_claude_profiles(self):
        values = [p.value for p in ModelProfile]
        assert "claude_sonnet" in values
        assert "claude_opus" in values

    def test_openrouter_mimo(self):
        values = [p.value for p in ModelProfile]
        assert "openrouter_mimo" in values


class TestStateStoreFourStates:
    """StateStore should accept all 4 states + backward compat."""

    def test_validates_all_four(self, tmp_path):
        from router.state_store import StateStore
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )
        for state in CodexState:
            if state in (CodexState.NORMAL, CodexState.LAST10):
                continue  # skip aliases, they map to primary values
            assert StateStore._validate_state(state.value) == state

    def test_backward_compat_normal_input(self, tmp_path):
        from router.state_store import StateStore
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )
        # "normal" should map to OPENAI_PRIMARY
        result = StateStore._validate_state("normal")
        assert result == CodexState.OPENAI_PRIMARY

    def test_backward_compat_last10_input(self, tmp_path):
        from router.state_store import StateStore
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )
        # "last10" should map to CLAUDE_BACKUP
        result = StateStore._validate_state("last10")
        assert result == CodexState.CLAUDE_BACKUP

    def test_default_is_openai_primary(self, tmp_path):
        from router.state_store import StateStore
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_set_and_get_new_states(self, tmp_path):
        from router.state_store import StateStore
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )
        for state in CodexState:
            if state in (CodexState.NORMAL, CodexState.LAST10):
                continue  # aliases
            store.set_manual_state(state)
            assert store.get_manual_state() == state


class TestBuildChainFourStates:
    """Test chain building for each of the 4 states."""

    def _task(self, task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, **kw):
        return TaskMeta(
            task_id="t1", task_class=task_class, risk=risk,
            modality=TaskModality.TEXT, **kw,
        )

    # --- openai_primary ---
    def test_openai_primary_starts_with_codex(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENAI_PRIMARY)
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"

    def test_openai_primary_has_three_entries(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENAI_PRIMARY)
        assert len(chain) == 3

    def test_openai_primary_second_is_claude(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENAI_PRIMARY)
        assert chain[1].tool == "claude_code"
        assert chain[1].backend == "anthropic"

    def test_openai_primary_third_is_openrouter(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENAI_PRIMARY)
        assert chain[2].tool == "codex_cli"
        assert chain[2].backend == "openrouter"

    # --- openai_conservation ---
    def test_openai_conservation_starts_with_codex(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENAI_CONSERVATION)
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"

    def test_openai_conservation_uses_mini_by_default(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54_mini"

    def test_openai_conservation_uses_gpt54_for_critical(self):
        from router.policy import build_chain
        chain = build_chain(self._task(risk=TaskRisk.CRITICAL), CodexState.OPENAI_CONSERVATION)
        assert chain[0].model_profile == "codex_gpt54"

    def test_openai_conservation_has_three_entries(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENAI_CONSERVATION)
        assert len(chain) == 3

    def test_openai_conservation_second_is_openrouter(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENAI_CONSERVATION)
        assert chain[1].tool == "codex_cli"
        assert chain[1].backend == "openrouter"

    def test_openai_conservation_third_is_claude(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENAI_CONSERVATION)
        assert chain[2].tool == "claude_code"
        assert chain[2].backend == "anthropic"

    # --- claude_backup ---
    def test_claude_backup_starts_with_claude(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.CLAUDE_BACKUP)
        assert chain[0].tool == "claude_code"
        assert chain[0].backend == "anthropic"

    def test_claude_backup_uses_sonnet_by_default(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.CLAUDE_BACKUP)
        assert chain[0].model_profile == "claude_sonnet"

    def test_claude_backup_uses_opus_for_critical(self):
        from router.policy import build_chain
        chain = build_chain(self._task(risk=TaskRisk.CRITICAL), CodexState.CLAUDE_BACKUP)
        assert chain[0].model_profile == "claude_opus"

    def test_claude_backup_has_two_entries(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.CLAUDE_BACKUP)
        assert len(chain) == 2

    def test_claude_backup_second_is_openrouter(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.CLAUDE_BACKUP)
        assert chain[1].tool == "codex_cli"
        assert chain[1].backend == "openrouter"

    # --- openrouter_fallback ---
    def test_openrouter_fallback_only_openrouter(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENROUTER_FALLBACK)
        assert len(chain) == 1
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openrouter"

    def test_openrouter_fallback_uses_minimax_default(self):
        from router.policy import build_chain
        chain = build_chain(self._task(), CodexState.OPENROUTER_FALLBACK)
        assert chain[0].model_profile == "openrouter_minimax"

    # --- backward compat ---
    def test_normal_backward_compat_chain(self):
        from router.policy import build_chain
        chain_normal = build_chain(self._task(), CodexState.NORMAL)
        chain_primary = build_chain(self._task(), CodexState.OPENAI_PRIMARY)
        assert len(chain_normal) == len(chain_primary)
        for a, b in zip(chain_normal, chain_primary):
            assert a.tool == b.tool
            assert a.backend == b.backend

    def test_last10_backward_compat_chain(self):
        from router.policy import build_chain
        chain_last10 = build_chain(self._task(), CodexState.LAST10)
        chain_backup = build_chain(self._task(), CodexState.CLAUDE_BACKUP)
        assert len(chain_last10) == len(chain_backup)
        for a, b in zip(chain_last10, chain_backup):
            assert a.tool == b.tool
            assert a.backend == b.backend


class TestOpenRouterProfileSelection:
    """Test choose_openrouter_profile with MiMo support."""

    def test_minimax_default(self):
        from router.policy import choose_openrouter_profile
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_MINIMAX

    def test_kimi_for_screenshots(self):
        from router.policy import choose_openrouter_profile
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
            has_screenshots=True,
        )
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_KIMI

    def test_kimi_for_multimodal(self):
        from router.policy import choose_openrouter_profile
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
            requires_multimodal=True,
        )
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_KIMI

    def test_mimo_for_critical(self):
        from router.policy import choose_openrouter_profile
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.CRITICAL, modality=TaskModality.TEXT,
        )
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_MIMO

    def test_mimo_for_debug(self):
        from router.policy import choose_openrouter_profile
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.DEBUG,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_MIMO

    def test_mimo_for_architecture(self):
        from router.policy import choose_openrouter_profile
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.REPO_ARCHITECTURE_CHANGE,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        assert choose_openrouter_profile(task) == ModelProfile.OPENROUTER_MIMO


class TestClaudeModelSelection:
    """Test choose_claude_model and choose_claude_profile."""

    def test_sonnet_default(self):
        from router.policy import choose_claude_model
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        assert choose_claude_model(task) == "claude-sonnet-4.6"

    def test_opus_for_critical(self):
        from router.policy import choose_claude_model
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.CRITICAL, modality=TaskModality.TEXT,
        )
        assert choose_claude_model(task) == "claude-opus-4.6"

    def test_opus_for_architecture(self):
        from router.policy import choose_claude_model
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.REPO_ARCHITECTURE_CHANGE,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        assert choose_claude_model(task) == "claude-opus-4.6"

    def test_sonnet_profile(self):
        from router.policy import choose_claude_profile
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        assert choose_claude_profile(task) == "claude_sonnet"

    def test_opus_profile(self):
        from router.policy import choose_claude_profile
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.CRITICAL, modality=TaskModality.TEXT,
        )
        assert choose_claude_profile(task) == "claude_opus"


class TestResolveState:
    """Test resolve_state with budget-aware priority."""

    def test_default_is_openai_primary(self):
        from router.policy import resolve_state
        from router.state_store import StateStore
        from pathlib import Path as P
        with tempfile.TemporaryDirectory() as d:
            store = StateStore(
                manual_path=P(d) / "m.json",
                auto_path=P(d) / "a.json",
            )
            assert resolve_state(store=store) == CodexState.OPENAI_PRIMARY

    def test_manual_overrides(self):
        from router.policy import resolve_state
        from router.state_store import StateStore
        from pathlib import Path as P
        with tempfile.TemporaryDirectory() as d:
            m_path = P(d) / "m.json"
            a_path = P(d) / "a.json"
            store = StateStore(manual_path=m_path, auto_path=a_path)
            with open(m_path, "w") as f:
                json.dump({"state": "openai_conservation"}, f)
            assert resolve_state(store=store) == CodexState.OPENAI_CONSERVATION

    def test_manual_overrides_auto(self):
        from router.policy import resolve_state
        from router.state_store import StateStore
        from pathlib import Path as P
        with tempfile.TemporaryDirectory() as d:
            m_path = P(d) / "m.json"
            a_path = P(d) / "a.json"
            store = StateStore(manual_path=m_path, auto_path=a_path)
            # Set auto to claude_backup
            with open(a_path, "w") as f:
                json.dump({"state": "claude_backup"}, f)
            # Set manual to openai_conservation — manual wins
            with open(m_path, "w") as f:
                json.dump({"state": "openai_conservation"}, f)
            assert resolve_state(store=store) == CodexState.OPENAI_CONSERVATION

    def test_auto_state_used_when_no_manual(self):
        from router.policy import resolve_state
        from router.state_store import StateStore
        from pathlib import Path as P
        with tempfile.TemporaryDirectory() as d:
            m_path = P(d) / "m.json"
            a_path = P(d) / "a.json"
            store = StateStore(manual_path=m_path, auto_path=a_path)
            # Write manual as null
            with open(m_path, "w") as f:
                json.dump({"state": None}, f)
            # Set auto
            with open(a_path, "w") as f:
                json.dump({"state": "claude_backup"}, f)
            assert resolve_state(store=store) == CodexState.CLAUDE_BACKUP


class TestNewModelConfig:
    """Test that new model strings are in config."""

    def test_mimo_model(self):
        from router.config_loader import get_model
        assert get_model("mimo") == "xiaomi/mimo-v2-pro"

    def test_sonnet_model(self):
        from router.config_loader import get_model
        assert get_model("sonnet") == "claude-sonnet-4.6"

    def test_opus_model(self):
        from router.config_loader import get_model
        assert get_model("opus") == "claude-opus-4.6"
