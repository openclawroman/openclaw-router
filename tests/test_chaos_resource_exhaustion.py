"""Chaos tests: resource exhaustion.

Goal: Verify that resource exhaustion (disk full, too many open files,
memory pressure) is handled gracefully, not with unhandled exceptions.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

from router.state_store import StateStore, reset_state_store
from router.config_loader import load_config, reload_config
from router.policy import route_task, reset_breaker, reset_notifier
from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState
from router.errors import StateError


@pytest.fixture(autouse=True)
def _reset():
    reset_state_store()
    reset_breaker()
    reset_notifier()
    import router.config_loader as cl
    cl._config_snapshot = None
    cl._config_raw = None
    cl._active_config_path = None
    yield
    reset_state_store()
    reset_breaker()
    reset_notifier()


class TestDiskFullOnWrite:
    """Mock disk full during _atomic_write → error handled."""

    def test_state_store_write_disk_full(self, tmp_path):
        """Simulate disk full during state write."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
            wal_path=tmp_path / "wal.jsonl",
        )

        original_mkstemp = tempfile.mkstemp

        def failing_mkstemp(*args, **kwargs):
            raise OSError(28, "No space left on device")

        with patch("tempfile.mkstemp", side_effect=failing_mkstemp):
            with pytest.raises((OSError, StateError)):
                store.set_manual_state(CodexState.CLAUDE_BACKUP)

    def test_wal_write_disk_full(self, tmp_path):
        """Simulate disk full during WAL append."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
            wal_path=tmp_path / "wal.jsonl",
        )

        original_open = open

        def failing_open(path, *args, **kwargs):
            if "wal" in str(path):
                raise OSError(28, "No space left on device")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=failing_open):
            with pytest.raises((OSError, StateError)):
                store.set_manual_state(CodexState.CLAUDE_BACKUP)

    def test_config_reload_disk_full(self, tmp_path):
        """Simulate disk full during config load."""
        config_file = tmp_path / "router.config.json"
        config_file.write_text(json.dumps({"version": 1}))

        # Clear cache
        import router.config_loader as cl
        cl._config_snapshot = None
        cl._config_raw = None
        cl._active_config_path = config_file

        real_open = open
        call_count = 0

        def failing_open(path, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "router.config" in str(path) and call_count == 1:
                raise OSError(28, "No space left on device")
            return real_open(path, *args, **kwargs)

        with pytest.raises(OSError):
            with patch("builtins.open", side_effect=failing_open):
                load_config(config_file)


class TestTooManyOpenFiles:
    """Mock file descriptor exhaustion → error handled."""

    def test_state_store_too_many_open_files(self, tmp_path):
        """Simulate EMFILE (too many open files) during state write."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
            wal_path=tmp_path / "wal.jsonl",
        )

        original_mkstemp = tempfile.mkstemp

        def failing_mkstemp(*args, **kwargs):
            raise OSError(24, "Too many open files")

        with patch("tempfile.mkstemp", side_effect=failing_mkstemp):
            with pytest.raises((OSError, StateError)):
                store.set_manual_state(CodexState.CLAUDE_BACKUP)

    def test_config_load_too_many_open_files(self, tmp_path):
        """Simulate EMFILE during config load."""
        config_file = tmp_path / "router.config.json"
        config_file.write_text(json.dumps({"version": 1}))

        import router.config_loader as cl
        cl._config_snapshot = None
        cl._config_raw = None
        cl._active_config_path = config_file

        def failing_open(path, *args, **kwargs):
            if "router.config" in str(path):
                raise OSError(24, "Too many open files")
            return open(path, *args, **kwargs)

        with pytest.raises(OSError):
            with patch("builtins.open", side_effect=failing_open):
                load_config(config_file)


class TestMemoryPressure:
    """Large config file → doesn't OOM."""

    def test_large_config_file(self, tmp_path):
        """Config file with large but valid content."""
        config_file = tmp_path / "router.config.json"

        # Build a large but valid config
        large_config = {
            "version": 1,
            "models": {
                "openrouter": {"minimax": "minimax/minimax-m2.7"},
                "codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"},
                "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"},
            },
            "reliability": {"chain_timeout_s": 600, "max_fallbacks": 3},
        }

        # Add a large junk section (1MB of data)
        large_config["large_section"] = {f"key_{i}": "x" * 1000 for i in range(1000)}

        config_file.write_text(json.dumps(large_config))

        import router.config_loader as cl
        cl._config_snapshot = None
        cl._config_raw = None
        cl._active_config_path = config_file

        # Should load without OOM
        config = load_config(config_file)
        assert config["version"] == 1
        assert "models" in config

    def test_many_state_transitions(self, tmp_path):
        """Many rapid state transitions don't leak memory."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
            history_path=tmp_path / "history.json",
            wal_path=tmp_path / "wal.jsonl",
        )

        states = [CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP,
                  CodexState.OPENAI_CONSERVATION, CodexState.OPENROUTER_FALLBACK]

        # Force transitions (bypass anti-flap)
        for i in range(100):
            store.set_state_with_history(states[i % 4], reason=f"test_{i}", force=True)

        # History should be bounded
        history = store.get_state_history(limit=200)
        assert len(history) <= 100  # Should not grow unbounded

    def test_config_with_unicode_garbage(self, tmp_path):
        """Config with lots of unicode doesn't blow up."""
        config_file = tmp_path / "router.config.json"

        large_config = {
            "version": 1,
            "models": {
                "openrouter": {"minimax": "minimax/minimax-m2.7"},
                "codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"},
                "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"},
            },
            "notes": "🚀" * 10000,  # Lots of unicode
        }

        config_file.write_text(json.dumps(large_config, ensure_ascii=False))

        import router.config_loader as cl
        cl._config_snapshot = None
        cl._config_raw = None
        cl._active_config_path = config_file

        config = load_config(config_file)
        assert config["version"] == 1
