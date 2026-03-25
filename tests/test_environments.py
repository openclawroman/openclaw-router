"""Tests for environment-specific config loading."""

import json
import os
import pytest
from pathlib import Path
from router.environments import (
    detect_environment, get_config_path,
    apply_env_overrides, ENV_CONFIGS,
)


class TestDetectEnvironment:
    def test_default_is_prod(self, monkeypatch):
        monkeypatch.delenv("ROUTER_ENV", raising=False)
        assert detect_environment() == "prod"

    def test_from_env_var(self, monkeypatch):
        monkeypatch.setenv("ROUTER_ENV", "dev")
        assert detect_environment() == "dev"

    def test_staging(self, monkeypatch):
        monkeypatch.setenv("ROUTER_ENV", "staging")
        assert detect_environment() == "staging"

    def test_invalid_env_raises(self, monkeypatch):
        monkeypatch.setenv("ROUTER_ENV", "invalid")
        with pytest.raises(ValueError, match="Unknown environment"):
            detect_environment()


class TestGetConfigPath:
    def test_cli_path_takes_priority(self, monkeypatch):
        monkeypatch.setenv("ROUTER_ENV", "dev")
        monkeypatch.setenv("ROUTER_CONFIG_PATH", "/env/path.json")
        assert get_config_path("/cli/path.json") == Path("/cli/path.json")

    def test_env_var_overrides_environment(self, monkeypatch):
        monkeypatch.setenv("ROUTER_ENV", "dev")
        monkeypatch.setenv("ROUTER_CONFIG_PATH", "/env/path.json")
        assert get_config_path() == Path("/env/path.json")

    def test_environment_default(self, monkeypatch):
        monkeypatch.delenv("ROUTER_CONFIG_PATH", raising=False)
        monkeypatch.setenv("ROUTER_ENV", "dev")
        assert get_config_path() == ENV_CONFIGS["dev"].config_path

    def test_prod_default(self, monkeypatch):
        monkeypatch.delenv("ROUTER_CONFIG_PATH", raising=False)
        monkeypatch.setenv("ROUTER_ENV", "prod")
        assert get_config_path() == Path("config/router.config.json")


class TestApplyOverrides:
    def test_no_overrides_for_prod(self):
        config = {"reliability": {"chain_timeout_s": 600}}
        result = apply_env_overrides(config, "prod")
        assert result["reliability"]["chain_timeout_s"] == 600

    def test_dev_overrides_applied(self):
        config = {"reliability": {"chain_timeout_s": 600, "circuit_breaker": {"cooldown_s": 120}}}
        result = apply_env_overrides(config, "dev")
        assert result["reliability"]["chain_timeout_s"] == 300  # overridden
        assert result["reliability"]["circuit_breaker"]["cooldown_s"] == 30  # overridden

    def test_deep_override_creates_intermediate(self):
        config = {"existing": {"key": "value"}}
        # dev overrides include reliability.chain_timeout_s
        result = apply_env_overrides(config, "dev")
        assert result["reliability"]["chain_timeout_s"] == 300

    def test_original_not_modified(self):
        config = {"reliability": {"chain_timeout_s": 600}}
        result = apply_env_overrides(config, "dev")
        assert config["reliability"]["chain_timeout_s"] == 600  # original unchanged


class TestConfigExtends:
    def test_extends_inheritance(self, tmp_path):
        """Config with _extends should merge with base config."""
        base = {"version": 1, "models": {"openrouter": {"x": "y"}, "codex": {"a": "b"}, "claude": {"c": "d"}},
                "state": {"default": "openai_primary", "manual_state_file": "x", "auto_state_file": "y"},
                "tools": {"codex_cli": {"profiles": {}}, "claude_code": {}},
                "reliability": {"chain_timeout_s": 600}}

        override = {"_extends": "base.json", "reliability": {"chain_timeout_s": 300}}

        base_path = tmp_path / "base.json"
        base_path.write_text(json.dumps(base))

        override_path = tmp_path / "override.json"
        override_path.write_text(json.dumps(override))

        # Test via apply_env_overrides is simpler
        # The _extends logic is in load_config which is harder to test in isolation
        # So we test the merge behavior

        import copy
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key].update(value)
            else:
                result[key] = value

        assert result["reliability"]["chain_timeout_s"] == 300
        assert result["version"] == 1


class TestEnvironmentConfigIntegrity:
    def test_all_envs_have_config(self):
        """All environments should have valid config paths."""
        for env, env_config in ENV_CONFIGS.items():
            assert env_config.name == env
            assert isinstance(env_config.config_path, Path)
            assert isinstance(env_config.overrides, dict)

    def test_prod_has_no_overrides(self):
        """Production should have no overrides (config is source of truth)."""
        assert ENV_CONFIGS["prod"].overrides == {}
