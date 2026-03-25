"""Tests for StateStore — Codex state detection and manual override resolution."""

import json
from pathlib import Path

import pytest

from router.models import CodexState
from router.state_store import StateStore


@pytest.fixture
def tmp_config(tmp_path):
    """Provide temporary manual and auto state file paths."""
    manual = tmp_path / "codex_manual_state.json"
    auto = tmp_path / "codex_auto_state.json"
    return manual, auto


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class TestManualOverridePrecedence:
    """Manual state should take precedence over auto state."""

    def test_manual_last10_overrides_auto_normal(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": "last10"})
        _write_json(auto_path, {"state": "normal"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_state() == CodexState.LAST10

    def test_manual_normal_overrides_auto_last10(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": "normal"})
        _write_json(auto_path, {"state": "last10"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_state() == CodexState.NORMAL

    def test_manual_null_falls_through_to_auto(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": None})
        _write_json(auto_path, {"state": "last10"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_state() == CodexState.LAST10


class TestAutoStateFallback:
    """Auto state should be used when manual is absent or null."""

    def test_auto_used_when_manual_missing(self, tmp_config):
        manual_path, auto_path = tmp_config
        # Only write auto, delete manual (auto-created by __init__)
        _write_json(auto_path, {"state": "last10"})
        if manual_path.exists():
            manual_path.unlink()

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        # After init, manual exists with default. Clear it.
        store.set_manual_state(None)
        assert store.get_state() == CodexState.LAST10

    def test_auto_used_when_manual_state_is_null(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": None})
        _write_json(auto_path, {"state": "last10"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_state() == CodexState.LAST10


class TestDefaultFallback:
    """Should fall back to normal when neither manual nor auto is set."""

    def test_default_normal_when_both_missing(self, tmp_config):
        manual_path, auto_path = tmp_config
        if manual_path.exists():
            manual_path.unlink()
        if auto_path.exists():
            auto_path.unlink()

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_manual_state(None)
        # Delete auto so get_auto_state returns None
        auto_path.unlink()
        assert store.get_state() == CodexState.NORMAL

    def test_default_normal_when_both_null(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": None})
        _write_json(auto_path, {"state": None})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_state() == CodexState.NORMAL


class TestInvalidStateValues:
    """Invalid state values should be rejected (return None)."""

    def test_invalid_manual_state_returns_none(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": "invalid_state"})
        _write_json(auto_path, {"state": "normal"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_manual_state() is None
        # Should fall through to auto
        assert store.get_state() == CodexState.NORMAL

    def test_invalid_auto_state_returns_none(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": None})
        _write_json(auto_path, {"state": "bogus"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_auto_state() is None
        # Should fall through to default
        assert store.get_state() == CodexState.NORMAL

    def test_validate_state_rejects_unknown(self):
        assert StateStore._validate_state("unknown") is None
        assert StateStore._validate_state("") is None
        assert StateStore._validate_state("NORMAL") is None  # case sensitive

    def test_validate_state_accepts_valid(self):
        assert StateStore._validate_state("normal") == CodexState.NORMAL
        assert StateStore._validate_state("last10") == CodexState.LAST10


class TestMissingFilesReturnNone:
    """Missing files should cause state getters to return None, not raise."""

    def test_missing_manual_returns_none(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(auto_path, {"state": "normal"})
        if manual_path.exists():
            manual_path.unlink()

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        # Clear manual that was auto-created
        manual_path.unlink()
        assert store.get_manual_state() is None

    def test_missing_auto_returns_none(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": None})
        if auto_path.exists():
            auto_path.unlink()

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        auto_path.unlink()
        assert store.get_auto_state() is None


class TestSetState:
    """Setting state should write correct values."""

    def test_set_manual_state_normal(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_manual_state(CodexState.NORMAL)
        assert store.get_manual_state() == CodexState.NORMAL

    def test_set_manual_state_last10(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_manual_state(CodexState.LAST10)
        assert store.get_manual_state() == CodexState.LAST10

    def test_set_manual_state_none_clears(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_manual_state(CodexState.LAST10)
        store.set_manual_state(None)
        assert store.get_manual_state() is None

    def test_set_auto_state(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_auto_state(CodexState.LAST10)
        assert store.get_auto_state() == CodexState.LAST10


# ═══════════════════════════════════════════════════════════════════════════════
# 4-state validation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateAllFourStates:
    """_validate_state should accept all 4 new state strings."""

    def test_validate_state_openai_primary(self):
        assert StateStore._validate_state("openai_primary") == CodexState.OPENAI_PRIMARY

    def test_validate_state_openai_conservation(self):
        assert StateStore._validate_state("openai_conservation") == CodexState.OPENAI_CONSERVATION

    def test_validate_state_claude_backup(self):
        assert StateStore._validate_state("claude_backup") == CodexState.CLAUDE_BACKUP

    def test_validate_state_openrouter_fallback(self):
        assert StateStore._validate_state("openrouter_fallback") == CodexState.OPENROUTER_FALLBACK

    def test_backward_compat_normal_maps_to_openai_primary(self):
        assert StateStore._validate_state("normal") == CodexState.OPENAI_PRIMARY

    def test_backward_compat_last10_maps_to_claude_backup(self):
        assert StateStore._validate_state("last10") == CodexState.CLAUDE_BACKUP


class TestSetAllFourStates:
    """Manual/auto state setters should work with all 4 states."""

    def test_set_manual_openai_conservation(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_manual_state(CodexState.OPENAI_CONSERVATION)
        assert store.get_manual_state() == CodexState.OPENAI_CONSERVATION

    def test_set_manual_claude_backup(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store.get_manual_state() == CodexState.CLAUDE_BACKUP

    def test_set_manual_openrouter_fallback(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_manual_state(CodexState.OPENROUTER_FALLBACK)
        assert store.get_manual_state() == CodexState.OPENROUTER_FALLBACK

    def test_set_auto_openai_conservation(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        assert store.get_auto_state() == CodexState.OPENAI_CONSERVATION

    def test_set_auto_openrouter_fallback(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)
        assert store.get_auto_state() == CodexState.OPENROUTER_FALLBACK

    def test_manual_conservation_overrides_auto_primary(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(auto_path, {"state": "openai_primary"})
        _write_json(manual_path, {"state": "openai_conservation"})
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION

    def test_auto_fallback_used_when_manual_null(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": None})
        _write_json(auto_path, {"state": "openrouter_fallback"})
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_default_is_openai_primary_when_all_null(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": None})
        _write_json(auto_path, {"state": None})
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert store.get_state() == CodexState.OPENAI_PRIMARY
