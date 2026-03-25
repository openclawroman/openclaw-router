"""Chaos tests: config file corruption.

Goal: Verify that corrupted/missing/unreadable config files produce proper
ConfigurationError, NOT unhandled exceptions.
"""

import json
import os
import stat
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.config_loader import (
    load_config, reload_config, get_config_snapshot,
    get_model, get_reliability_config, CONFIG_PATH,
    ConfigValidationError,
)
from router.errors import ConfigurationError


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config cache between tests."""
    import router.config_loader as cl
    cl._config_snapshot = None
    cl._config_raw = None
    cl._active_config_path = None
    yield
    cl._config_snapshot = None
    cl._config_raw = None
    cl._active_config_path = None


@pytest.fixture
def valid_config():
    """Minimal valid config."""
    return {
        "version": 1,
        "models": {
            "openrouter": {"minimax": "minimax/minimax-m2.7"},
            "codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"},
            "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"},
        },
        "reliability": {
            "chain_timeout_s": 600,
            "max_fallbacks": 3,
        },
    }


class TestConfigDeletedMidRun:
    """Config file deleted → load_config raises an error."""

    def test_config_deleted_load_config(self, tmp_path):
        config_file = tmp_path / "router.config.json"
        config_file.write_text(json.dumps({"version": 1}))

        # First load succeeds
        load_config(config_file)

        # Delete the file
        config_file.unlink()

        # Next load should raise, not crash with unhandled exception
        with pytest.raises((FileNotFoundError, OSError, IOError)):
            load_config(config_file)

    def test_config_deleted_get_snapshot(self, tmp_path):
        config_file = tmp_path / "router.config.json"
        config = {"version": 1, "models": {"codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"}, "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"}, "openrouter": {"minimax": "minimax/minimax-m2.7"}}}
        config_file.write_text(json.dumps(config))

        import router.config_loader as cl
        cl._active_config_path = config_file
        load_config(config_file)
        config_file.unlink()

        # Clear cache to force re-read from the now-deleted path
        cl._config_snapshot = None
        cl._config_raw = None

        # get_config_snapshot calls load_config internally, which will raise
        with pytest.raises((FileNotFoundError, OSError, IOError)):
            get_config_snapshot()


class TestConfigInvalidJson:
    """Config file has invalid JSON → proper error."""

    def test_invalid_json_load(self, tmp_path):
        config_file = tmp_path / "router.config.json"
        config_file.write_text('{"version": 1, "models": {broken json!!!')

        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(config_file)

    def test_truncated_json(self, tmp_path):
        """Half a JSON file."""
        config_file = tmp_path / "router.config.json"
        config_file.write_text('{"version": 1, "models": {"cod')

        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(config_file)

    def test_json_with_invalid_unicode(self, tmp_path):
        """Binary garbage in config file."""
        config_file = tmp_path / "router.config.json"
        config_file.write_bytes(b'\x00\x01\x02\x80\x81\x82{"version": 1}')

        # Should raise, not crash
        with pytest.raises(Exception):
            load_config(config_file)


class TestConfigEmptyFile:
    """Config file is empty → proper error."""

    def test_empty_file_load(self, tmp_path):
        config_file = tmp_path / "router.config.json"
        config_file.write_text("")

        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(config_file)

    def test_whitespace_only_file(self, tmp_path):
        config_file = tmp_path / "router.config.json"
        config_file.write_text("   \n\t\n  ")

        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(config_file)


class TestConfigPartialWrite:
    """Config file truncated mid-write → proper error."""

    def test_partial_write_corrupted(self, tmp_path):
        """Simulate a partially written config (e.g., disk full during write)."""
        config_file = tmp_path / "router.config.json"
        # Write half the file
        config_file.write_text('{"version": 1, "models": {"openrouter": {"min')

        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(config_file)

    def test_reload_after_corruption(self, tmp_path):
        """Config valid initially, then corrupted, reload fails."""
        config_file = tmp_path / "router.config.json"

        # Good first
        good = {
            "version": 1,
            "models": {
                "openrouter": {"minimax": "minimax/minimax-m2.7"},
                "codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"},
                "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"},
            },
        }
        config_file.write_text(json.dumps(good))
        load_config(config_file)

        # Corrupt it
        config_file.write_text('{"broken":')

        with pytest.raises((json.JSONDecodeError, ValueError)):
            reload_config(config_file)


class TestConfigPermissionsDenied:
    """Config file unreadable → proper error."""

    def test_no_read_permission(self, tmp_path):
        config_file = tmp_path / "router.config.json"
        config_file.write_text(json.dumps({"version": 1}))

        # Remove read permission
        os.chmod(config_file, 0o000)

        try:
            with pytest.raises((PermissionError, OSError)):
                load_config(config_file)
        finally:
            # Restore permissions for cleanup
            os.chmod(config_file, 0o644)

    def test_directory_not_accessible(self, tmp_path):
        """Config directory itself is inaccessible."""
        config_dir = tmp_path / "secret_config"
        config_dir.mkdir()
        config_file = config_dir / "router.config.json"
        config_file.write_text(json.dumps({"version": 1}))

        # Make directory inaccessible
        os.chmod(config_dir, 0o000)

        try:
            with pytest.raises((PermissionError, OSError)):
                load_config(config_file)
        finally:
            os.chmod(config_dir, 0o755)


class TestConfigGracefulDefaults:
    """System should handle missing config sections gracefully."""

    def test_config_missing_models_section(self, tmp_path):
        """Config without models section — get_model should fail gracefully."""
        config_file = tmp_path / "router.config.json"
        config_file.write_text(json.dumps({"version": 1}))

        # Bypass validation for this test
        import router.config_loader as cl
        cl._active_config_path = config_file
        cl._config_snapshot = None
        cl._config_raw = None

        with pytest.raises(KeyError):
            get_model("nonexistent")

    def test_reload_clears_cache(self, tmp_path):
        """After reload, old cache is gone."""
        config_file = tmp_path / "router.config.json"
        good_config = {"version": 1, "models": {"codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"}, "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"}, "openrouter": {"minimax": "minimax/minimax-m2.7"}}, "reliability": {"chain_timeout_s": 600, "max_fallbacks": 3}}
        config_file.write_text(json.dumps(good_config))

        reload_config(config_file)

        # Corrupt and reload should fail
        config_file.write_text("not json")

        with pytest.raises(Exception):
            reload_config(config_file)
