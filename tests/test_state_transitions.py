"""Tests for subscription-budget state transitions."""

import json
from pathlib import Path

import pytest

from router.models import CodexState, TaskMeta, TaskClass, TaskRisk, TaskModality
from router.state_store import StateStore
from router.policy import resolve_state, build_chain, route_task


class TestManualStateTransitions:
    """Test manual state switching between all 4 states."""

    @pytest.fixture
    def store(self, tmp_path):
        return StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )

    def test_manual_switch_to_conservation(self, store):
        """User can manually switch to openai_conservation."""
        store.set_manual_state(CodexState.OPENAI_CONSERVATION)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION

    def test_manual_switch_to_claude_backup(self, store):
        """User can manually switch to claude_backup."""
        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store.get_state() == CodexState.CLAUDE_BACKUP

    def test_manual_switch_to_openrouter_fallback(self, store):
        """User can manually switch to openrouter_fallback."""
        store.set_manual_state(CodexState.OPENROUTER_FALLBACK)
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_manual_switch_back_to_primary(self, store):
        """User can switch back to openai_primary after any state."""
        for state in CodexState:
            if state in (CodexState.NORMAL, CodexState.LAST10):
                continue
            store.set_manual_state(state)
            assert store.get_state() == state
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_manual_via_file_write(self, tmp_path):
        """Writing state file directly works."""
        manual_path = tmp_path / "manual.json"
        auto_path = tmp_path / "auto.json"
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        # Write openai_conservation to manual file
        with open(manual_path, "w") as f:
            json.dump({"state": "openai_conservation"}, f)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION

    def test_manual_clear_null_falls_to_auto(self, tmp_path):
        """Clearing manual (null) lets auto take over."""
        manual_path = tmp_path / "manual.json"
        auto_path = tmp_path / "auto.json"
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_manual_state(CodexState.OPENAI_CONSERVATION)
        store.set_auto_state(CodexState.CLAUDE_BACKUP)
        # Clear manual
        store.set_manual_state(None)
        assert store.get_state() == CodexState.CLAUDE_BACKUP


class TestAutoStatePriority:
    """Auto state overrides default but not manual."""

    @pytest.fixture
    def store(self, tmp_path):
        return StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )

    def test_auto_state_overrides_default(self, store):
        """Auto state takes precedence over the default (openai_primary)."""
        store.set_manual_state(None)
        store.set_auto_state(CodexState.CLAUDE_BACKUP)
        assert store.get_state() == CodexState.CLAUDE_BACKUP

    def test_auto_does_not_override_manual(self, store):
        """Manual always wins over auto."""
        store.set_manual_state(CodexState.OPENAI_CONSERVATION)
        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION

    def test_auto_transitions_through_all_states(self, store):
        """Auto state can transition through all 4 states."""
        store.set_manual_state(None)
        for state in CodexState:
            if state in (CodexState.NORMAL, CodexState.LAST10):
                continue
            store.set_auto_state(state)
            assert store.get_state() == state


class TestResolveStatePriority:
    """Test resolve_state function with various state configurations."""

    def test_default_returns_openai_primary(self, tmp_path):
        from unittest.mock import patch, MagicMock
        with patch("router.policy.StateStore") as MockStore:
            mock = MagicMock()
            mock.get_manual_state.return_value = None
            mock.get_auto_state.return_value = None
            MockStore.return_value = mock
            assert resolve_state() == CodexState.OPENAI_PRIMARY

    def test_manual_conservation_overrides_all(self, tmp_path):
        from unittest.mock import patch, MagicMock
        with patch("router.policy.StateStore") as MockStore:
            mock = MagicMock()
            mock.get_manual_state.return_value = CodexState.OPENAI_CONSERVATION
            mock.get_auto_state.return_value = CodexState.OPENROUTER_FALLBACK
            MockStore.return_value = mock
            assert resolve_state() == CodexState.OPENAI_CONSERVATION

    def test_auto_claude_used_when_no_manual(self, tmp_path):
        from unittest.mock import patch, MagicMock
        with patch("router.policy.StateStore") as MockStore:
            mock = MagicMock()
            mock.get_manual_state.return_value = None
            mock.get_auto_state.return_value = CodexState.CLAUDE_BACKUP
            MockStore.return_value = mock
            assert resolve_state() == CodexState.CLAUDE_BACKUP

    def test_explicit_store_override(self, tmp_path):
        """resolve_state with explicit store param."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )
        store.set_manual_state(CodexState.OPENROUTER_FALLBACK)
        assert resolve_state(store=store) == CodexState.OPENROUTER_FALLBACK


class TestStateTransitionEdges:
    """Test that state transitions reflect budget-driven logic, not failures."""

    def test_primary_to_conservation_is_budget_driven(self):
        """Transition from primary to conservation is about budget, not failures.
        
        Both states use codex_cli as primary, but conservation uses mini
        more aggressively. Both keep Claude as second entry (subscription before paid).
        """
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        primary_chain = build_chain(task, CodexState.OPENAI_PRIMARY)
        conservation_chain = build_chain(task, CodexState.OPENAI_CONSERVATION)

        # Both use codex_cli first
        assert primary_chain[0].tool == conservation_chain[0].tool == "codex_cli"
        # Both use Claude as second entry (subscription before OpenRouter)
        assert conservation_chain[1].backend == "anthropic"
        assert primary_chain[1].backend == "anthropic"

    def test_conservation_to_claude_is_exhaustion_driven(self):
        """Transition from conservation to claude is about OpenAI exhaustion.
        
        In claude_backup, Claude Code is primary — no OpenAI lane at all.
        """
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        conservation_chain = build_chain(task, CodexState.OPENAI_CONSERVATION)
        claude_chain = build_chain(task, CodexState.CLAUDE_BACKUP)

        # Conservation starts with codex, claude_backup starts with claude
        assert conservation_chain[0].tool == "codex_cli"
        assert claude_chain[0].tool == "claude_code"
        # claude_backup has no openai_native entries
        for entry in claude_chain:
            assert entry.backend != "openai_native"

    def test_claude_to_openrouter_is_availability_driven(self):
        """Transition from claude to openrouter is about Claude availability.
        
        openrouter_fallback only uses openrouter — no codex native or claude.
        """
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        claude_chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        openrouter_chain = build_chain(task, CodexState.OPENROUTER_FALLBACK)

        # claude_backup has 2 entries (claude + openrouter)
        assert len(claude_chain) == 2
        # openrouter_fallback has only 1 entry (openrouter)
        assert len(openrouter_chain) == 1
        assert openrouter_chain[0].backend == "openrouter"

    def test_each_state_is_a_complete_lane(self):
        """Each state defines a self-contained routing chain."""
        task = TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )
        for state in CodexState:
            if state in (CodexState.NORMAL, CodexState.LAST10):
                continue
            chain = build_chain(task, state)
            assert len(chain) >= 1, f"{state.value} should have at least 1 entry"
            for entry in chain:
                assert entry.tool
                assert entry.backend
                assert entry.model_profile