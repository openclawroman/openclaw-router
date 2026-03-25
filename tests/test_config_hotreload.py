"""Tests for config hot-reload and immutable snapshots."""

import json
import pytest
from pathlib import Path
from types import MappingProxyType
from router.config_loader import (
    load_config, reload_config, get_config_snapshot,
    get_model, list_models, get_reliability_config,
)


class TestImmutableSnapshots:
    def test_load_config_returns_copy(self):
        """load_config should return a regular dict, not the snapshot."""
        config = load_config()
        assert isinstance(config, dict)
        # Modifying the copy should not affect the snapshot
        config["models"]["fake_key"] = "fake_value"
        config2 = load_config()
        assert "fake_key" not in config2.get("models", {})

    def test_get_config_snapshot_is_immutable(self):
        """get_config_snapshot returns MappingProxyType (immutable)."""
        snapshot = get_config_snapshot()
        assert isinstance(snapshot, MappingProxyType)
        with pytest.raises(TypeError):
            snapshot["models"]["fake"] = "value"

    def test_snapshot_reflects_reload(self, tmp_path):
        """After reload, snapshot should reflect new config."""
        # Create temp config
        config1 = {"version": 1, "models": {"openrouter": {"test": "v1"}, "codex": {"x": "y"}, "claude": {"x": "y"}}, 
                   "state": {"default": "openai_primary", "manual_state_file": "x", "auto_state_file": "y"},
                   "tools": {"codex_cli": {"profiles": {}}, "claude_code": {}}}
        config2 = config1.copy()
        config2["models"]["openrouter"]["test"] = "v2"
        
        path = tmp_path / "test_config.json"
        path.write_text(json.dumps(config1))
        
        reload_config(path)
        snap1 = get_config_snapshot()
        
        path.write_text(json.dumps(config2))
        snap2 = reload_config(path)
        
        assert snap2["models"]["openrouter"]["test"] == "v2"


class TestThreadSafety:
    def test_concurrent_reload_does_not_corrupt(self):
        """Multiple reload calls should not produce partial state."""
        import threading
        
        errors = []
        
        def reload_worker():
            try:
                config = reload_config()
                assert isinstance(config, dict)
                assert "models" in config
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=reload_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors, f"Thread errors: {errors}"


class TestReloadConfig:
    def test_reload_clears_and_reloads(self):
        """reload_config should load fresh config from disk."""
        config = reload_config()
        assert isinstance(config, dict)
        assert "models" in config

    def test_reload_from_path(self, tmp_path):
        """reload_config with custom path should load that file."""
        custom_config = {
            "version": 1, 
            "models": {"openrouter": {"custom": "custom/model"}, "codex": {"x": "y"}, "claude": {"x": "y"}},
            "state": {"default": "openai_primary", "manual_state_file": "x", "auto_state_file": "y"},
            "tools": {"codex_cli": {"profiles": {}}, "claude_code": {}}
        }
        path = tmp_path / "custom.json"
        path.write_text(json.dumps(custom_config))
        
        config = reload_config(path)
        assert config["models"]["openrouter"]["custom"] == "custom/model"
        
        # Reset to default for other tests
        reload_config(None)


class TestConfigMutationIsolation:
    def test_get_model_unaffected_by_mutations(self):
        """get_model should work even if someone mutates the returned config."""
        config = load_config()
        config["models"]["openrouter"]["minimax"] = "CORRUPTED"
        
        # get_model should still return the original value
        model = get_model("minimax")
        assert model != "CORRUPTED"

    def test_list_models_unaffected_by_mutations(self):
        """list_models should work even if config was mutated."""
        config = load_config()
        config["models"]["openrouter"]["FAKE"] = "fake/model"
        
        models = list_models()
        assert "FAKE" not in models
