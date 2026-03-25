"""Tests for config-driven model strings."""

import json
import pytest
from pathlib import Path
from router.config_loader import load_config, get_model, reload_config


class TestConfigLoader:
    def test_load_config_returns_dict(self, tmp_path):
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps({"models": {"openrouter": {"minimax": "minimax/test"}}}))
        config = load_config(config_path=config_file)
        assert isinstance(config, dict)
        assert config["models"]["openrouter"]["minimax"] == "minimax/test"

    def test_load_config_caches(self):
        # First load
        config1 = load_config()
        # Second load returns equivalent config (deep copy for mutation safety)
        config2 = load_config()
        assert config1 == config2

    def test_reload_clears_cache(self):
        config1 = load_config()
        reload_config()
        config2 = load_config()
        # After reload, should load fresh (different object)
        assert config1 is not config2


class TestGetModel:
    def test_minimax_model(self):
        model = get_model("minimax")
        assert model == "minimax/minimax-m2.7"

    def test_kimi_model(self):
        model = get_model("kimi")
        assert model == "moonshotai/kimi-k2.5"

    def test_unknown_profile_raises_keyerror(self):
        from router.config_loader import get_model
        with pytest.raises(KeyError, match="some_future_model"):
            get_model("some_future_model")

    def test_list_models(self):
        from router.config_loader import list_models
        models = list_models()
        assert "minimax" in models
        assert "gpt54" in models

    def test_models_from_config(self, tmp_path):
        """Verify models come from config, not hardcoded."""
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps({
            "models": {
                "openrouter": {
                    "minimax": "minimax/future-v3.0",
                    "kimi": "moonshotai/kimi-v3.0"
                }
            }
        }))
        reload_config(config_path=config_file)
        try:
            model = get_model("minimax")
            assert model == "minimax/future-v3.0"
            model = get_model("kimi")
            assert model == "moonshotai/kimi-v3.0"
        finally:
            reload_config()  # restore default cache


class TestNoHardcodedModels:
    """Verify model strings are NOT hardcoded in policy.py and executors.py."""

    def test_policy_no_hardcoded_minimax(self):
        source = Path("router/policy.py").read_text()
        assert '"minimax/minimax-m2.7"' not in source
        assert "'minimax/minimax-m2.7'" not in source

    def test_policy_no_hardcoded_kimi(self):
        source = Path("router/policy.py").read_text()
        assert '"moonshotai/kimi-k2.5"' not in source
        assert "'moonshotai/kimi-k2.5'" not in source

    def test_executors_no_hardcoded_minimax(self):
        source = Path("router/executors.py").read_text()
        assert '"minimax/minimax-m2.7"' not in source
        assert "'minimax/minimax-m2.7'" not in source

    def test_executors_no_hardcoded_kimi(self):
        source = Path("router/executors.py").read_text()
        assert '"moonshotai/kimi-k2.5"' not in source
        assert "'moonshotai/kimi-k2.5'" not in source

    def test_config_has_models(self):
        config = load_config()
        assert "models" in config
        assert "openrouter" in config["models"]
        assert "minimax" in config["models"]["openrouter"]
        assert "kimi" in config["models"]["openrouter"]


class TestModelChangeWorkflow:
    """Verify the workflow: change one config line, model changes everywhere."""

    def test_single_config_change_changes_all(self, tmp_path):
        """Changing model in config changes what get_model returns."""
        # Create test config with new model
        config_file = tmp_path / "test_config.json"
        new_model = "minimax/minimax-m9.9"
        config_file.write_text(json.dumps({
            "models": {
                "openrouter": {
                    "minimax": new_model,
                    "kimi": "moonshotai/kimi-k2.5"
                }
            }
        }))
        reload_config(config_path=config_file)
        try:
            # Now get_model returns the new string
            assert get_model("minimax") == new_model
        finally:
            reload_config()
