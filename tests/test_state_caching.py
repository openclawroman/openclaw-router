"""Tests for StateStore singleton and in-memory state caching."""

import json
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.models import CodexState
from router.state_store import StateStore, get_state_store, reset_state_store, _global_store


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global singleton before and after each test."""
    reset_state_store()
    yield
    reset_state_store()


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


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetStateStoreSingleton:
    """get_state_store() returns the same instance on repeated calls."""

    def test_returns_same_instance(self):
        store1 = get_state_store()
        store2 = get_state_store()
        assert store1 is store2

    def test_returns_state_store_instance(self):
        store = get_state_store()
        assert isinstance(store, StateStore)

    def test_thread_safe_creation(self):
        """Multiple threads calling get_state_store() concurrently yield the same instance."""
        results = []
        errors = []

        def _get_store():
            try:
                results.append(get_state_store())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_get_store) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors during concurrent creation: {errors}"
        assert len(results) == 20
        first = results[0]
        for store in results:
            assert store is first


class TestResetStateStore:
    """reset_state_store() clears the singleton so a fresh one is created."""

    def test_reset_creates_new_instance(self):
        store1 = get_state_store()
        reset_state_store()
        store2 = get_state_store()
        assert store1 is not store2


# ═══════════════════════════════════════════════════════════════════════════════
# In-memory cache tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestManualStateCaching:
    """get_manual_state() should use in-memory cache when valid."""

    def test_second_call_uses_cache(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": "openai_primary"})
        _write_json(auto_path, {"state": "openai_primary"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        # First call reads from file
        result1 = store.get_manual_state()
        assert result1 == CodexState.OPENAI_PRIMARY

        # Now change the file directly (bypassing store)
        _write_json(manual_path, {"state": "claude_backup"})

        # Second call should return cached value, NOT the changed file
        result2 = store.get_manual_state()
        assert result2 == CodexState.OPENAI_PRIMARY  # cached, not re-read

    def test_set_manual_state_updates_cache(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)

        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        # Cache should reflect the new state immediately
        assert store.get_manual_state() == CodexState.CLAUDE_BACKUP

    def test_set_manual_state_none_updates_cache(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)

        store.set_manual_state(CodexState.OPENAI_CONSERVATION)
        assert store.get_manual_state() == CodexState.OPENAI_CONSERVATION

        store.set_manual_state(None)
        assert store.get_manual_state() is None

    def test_cache_valid_flag_set_after_get(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": "openai_primary"})
        _write_json(auto_path, {"state": "openai_primary"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert not store._manual_cache_valid
        store.get_manual_state()
        assert store._manual_cache_valid


class TestAutoStateCaching:
    """get_auto_state() should use in-memory cache when valid."""

    def test_second_call_uses_cache(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": "openai_primary"})
        _write_json(auto_path, {"state": "claude_backup"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        result1 = store.get_auto_state()
        assert result1 == CodexState.CLAUDE_BACKUP

        # Change file directly
        _write_json(auto_path, {"state": "openai_conservation"})

        result2 = store.get_auto_state()
        assert result2 == CodexState.CLAUDE_BACKUP  # cached

    def test_set_auto_state_updates_cache(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)

        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)
        assert store.get_auto_state() == CodexState.OPENROUTER_FALLBACK

    def test_cache_valid_flag_set_after_get(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(manual_path, {"state": "openai_primary"})
        _write_json(auto_path, {"state": "openai_primary"})

        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        assert not store._auto_cache_valid
        store.get_auto_state()
        assert store._auto_cache_valid


class TestCacheReturnCorrectState:
    """Cache should return correct state after set operations."""

    def test_manual_round_trip(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)

        for state in [
            CodexState.OPENAI_PRIMARY,
            CodexState.OPENAI_CONSERVATION,
            CodexState.CLAUDE_BACKUP,
            CodexState.OPENROUTER_FALLBACK,
            None,
        ]:
            store.set_manual_state(state)
            assert store.get_manual_state() == state

    def test_auto_round_trip(self, tmp_config):
        manual_path, auto_path = tmp_config
        store = StateStore(manual_path=manual_path, auto_path=auto_path)

        for state in [
            CodexState.OPENAI_PRIMARY,
            CodexState.OPENAI_CONSERVATION,
            CodexState.CLAUDE_BACKUP,
            CodexState.OPENROUTER_FALLBACK,
        ]:
            store.set_auto_state(state)
            assert store.get_auto_state() == state


class TestCacheNoneState:
    """Cache should correctly handle None (missing/null) states."""

    def test_manual_none_cached(self, tmp_config):
        manual_path, auto_path = tmp_config
        # No files exist yet
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_manual_state(None)

        # Change file to have a value — cache should still return None
        _write_json(manual_path, {"state": "claude_backup"})
        assert store.get_manual_state() is None

    def test_auto_none_cached(self, tmp_config):
        manual_path, auto_path = tmp_config
        _write_json(auto_path, {"state": "openai_primary"})
        store = StateStore(manual_path=manual_path, auto_path=auto_path)
        store.set_auto_state(CodexState.OPENAI_PRIMARY)

        # Now set manual to override, so we don't read auto
        _write_json(manual_path, {"state": None})
        _write_json(auto_path, {"state": None})
        store._auto_cache_valid = False  # force re-read
        assert store.get_auto_state() is None
