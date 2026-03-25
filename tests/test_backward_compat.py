"""Backward compatibility tests for old state names."""

import pytest

from router.models import (
    CodexState, TaskMeta, TaskClass, TaskRisk, TaskModality, ChainEntry,
)
from router.policy import build_chain, route_task, _build_normal_chain, _build_last10_chain
from router.state_store import StateStore


class TestEnumAliases:
    """Old enum members NORMAL/LAST10 should exist and map correctly."""

    def test_normal_alias_exists(self):
        assert hasattr(CodexState, "NORMAL")

    def test_normal_maps_to_openai_primary(self):
        assert CodexState.NORMAL == CodexState.OPENAI_PRIMARY
        assert CodexState.NORMAL.value == "openai_primary"

    def test_last10_alias_exists(self):
        assert hasattr(CodexState, "LAST10")

    def test_last10_maps_to_claude_backup(self):
        assert CodexState.LAST10 == CodexState.CLAUDE_BACKUP
        assert CodexState.LAST10.value == "claude_backup"

    def test_normal_is_same_object(self):
        """NORMAL and OPENAI_PRIMARY are the same enum member."""
        assert CodexState.NORMAL is CodexState.OPENAI_PRIMARY

    def test_last10_is_same_object(self):
        """LAST10 and CLAUDE_BACKUP are the same enum member."""
        assert CodexState.LAST10 is CodexState.CLAUDE_BACKUP


class TestStateStoreBackwardCompat:
    """StateStore should accept old state string values."""

    def test_validate_state_normal_string(self):
        assert StateStore._validate_state("normal") == CodexState.OPENAI_PRIMARY

    def test_validate_state_last10_string(self):
        assert StateStore._validate_state("last10") == CodexState.CLAUDE_BACKUP

    def test_set_state_via_old_names(self, tmp_path):
        """Writing 'normal' to file should produce OPENAI_PRIMARY."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        store = StateStore(manual_path=manual, auto_path=auto)
        # Simulate old file format with "normal"
        import json
        with open(manual, "w") as f:
            json.dump({"state": "normal"}, f)
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_set_state_via_last10_string(self, tmp_path):
        """Writing 'last10' to file should produce CLAUDE_BACKUP."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        store = StateStore(manual_path=manual, auto_path=auto)
        import json
        with open(manual, "w") as f:
            json.dump({"state": "last10"}, f)
        assert store.get_state() == CodexState.CLAUDE_BACKUP


class TestChainBackwardCompat:
    """build_chain with NORMAL/LAST10 should produce same chains as new names."""

    def _task(self):
        return TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )

    def test_normal_alias_works_in_build_chain(self):
        task = self._task()
        chain = build_chain(task, CodexState.NORMAL)
        assert chain[0].tool == "codex_cli"

    def test_last10_alias_works_in_build_chain(self):
        task = self._task()
        chain = build_chain(task, CodexState.LAST10)
        assert chain[0].tool == "claude_code"

    def test_normal_chain_matches_openai_primary(self):
        task = self._task()
        chain_normal = build_chain(task, CodexState.NORMAL)
        chain_primary = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert len(chain_normal) == len(chain_primary)
        for a, b in zip(chain_normal, chain_primary):
            assert a.tool == b.tool
            assert a.backend == b.backend
            assert a.model_profile == b.model_profile

    def test_last10_chain_matches_claude_backup(self):
        task = self._task()
        chain_last10 = build_chain(task, CodexState.LAST10)
        chain_backup = build_chain(task, CodexState.CLAUDE_BACKUP)
        assert len(chain_last10) == len(chain_backup)
        for a, b in zip(chain_last10, chain_backup):
            assert a.tool == b.tool
            assert a.backend == b.backend
            assert a.model_profile == b.model_profile


class TestBackwardCompatFunctions:
    """Backward compat chain builder functions still work."""

    def _task(self):
        return TaskMeta(
            task_id="t1", task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT,
        )

    def test_build_normal_chain_exists(self):
        """_build_normal_chain should still be importable and work."""
        chain = _build_normal_chain(self._task())
        assert len(chain) == 3
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"

    def test_build_last10_chain_exists(self):
        """_build_last10_chain should still be importable and work."""
        chain = _build_last10_chain(self._task())
        assert len(chain) == 2
        assert chain[0].tool == "claude_code"
        assert chain[0].backend == "anthropic"

    def test_build_normal_equals_build_primary(self):
        chain_old = _build_normal_chain(self._task())
        chain_new = build_chain(self._task(), CodexState.OPENAI_PRIMARY)
        assert len(chain_old) == len(chain_new)
        for a, b in zip(chain_old, chain_new):
            assert a.tool == b.tool
            assert a.backend == b.backend

    def test_build_last10_equals_build_backup(self):
        chain_old = _build_last10_chain(self._task())
        chain_new = build_chain(self._task(), CodexState.CLAUDE_BACKUP)
        assert len(chain_old) == len(chain_new)
        for a, b in zip(chain_old, chain_new):
            assert a.tool == b.tool
            assert a.backend == b.backend
