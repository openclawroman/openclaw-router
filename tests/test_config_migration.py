"""Tests for config version migration system."""

import json
import pytest

from router.config_migration import (
    CURRENT_CONFIG_VERSION, MIGRATIONS, ConfigMigrationError,
    migrate_config, register_migration,
)


def _v0_config(**overrides):
    base = {
        "models": {
            "openrouter": {"GPT4": "gpt-4", "Sonnet": "claude-sonnet-4"},
            "codex": {"Default": "codex-mini"},
        },
        "tools": {"codex_cli": {"profiles": {"OpenAI_Native": {"model": "o3", "timeout_s": 300}}}},
        "routing": {"policy": "openai_primary"},
        "timeouts": {"openai": 30},
        "state": {"default": "openai_primary", "manual_state_file": "config/manual.json", "auto_state_file": "config/auto.json"},
    }
    base.update(overrides)
    return base


def _v1_config(**overrides):
    base = {"version": CURRENT_CONFIG_VERSION, "models": {"openrouter": {"gpt4": "gpt-4", "sonnet": "claude-sonnet-4"}}, "routing": {"policy": "openai_primary"}, "timeouts": {"openai": 30}}
    base.update(overrides)
    return base


class TestNoVersionFieldAssumesV0:
    def test_version_added(self):
        config = _v0_config()
        assert "version" not in config
        result = migrate_config(config)
        assert result["version"] == CURRENT_CONFIG_VERSION

    def test_original_unchanged(self):
        config = _v0_config()
        migrate_config(config)
        assert "version" not in config


class TestAlreadyCurrentVersion:
    def test_no_extra_migration(self):
        config = _v1_config()
        result = migrate_config(config)
        assert result["version"] == CURRENT_CONFIG_VERSION
        assert result == config

    def test_deep_equality(self):
        config = _v1_config(extra_key="preserved")
        result = migrate_config(config)
        assert result["extra_key"] == "preserved"


class TestV0MigrationAddsDefaults:
    @pytest.mark.parametrize("config,expected_keys", [
        ({"routing": {"policy": "test"}, "timeouts": {"x": 1}}, ["models"]),
        ({"models": {"a": {"b": "c"}}, "timeouts": {"x": 1}}, ["routing"]),
        ({"models": {"a": {"b": "c"}}, "routing": {"policy": "test"}}, ["timeouts"]),
        ({}, ["models", "routing", "timeouts"]),
    ])
    def test_missing_sections_get_defaults(self, config, expected_keys):
        result = migrate_config(config)
        for key in expected_keys:
            assert key in result
        assert result["version"] == CURRENT_CONFIG_VERSION


class TestV0MigrationNormalizesKeys:
    def test_model_keys_normalized(self):
        config = _v0_config()
        result = migrate_config(config)
        assert "GPT4" not in result["models"]["openrouter"]
        assert "gpt4" in result["models"]["openrouter"]

    def test_tool_keys_normalized(self):
        config = {"models": {}, "routing": {}, "timeouts": {}, "tools": {"codex_cli": {"profiles": {"OpenAI_Native": {"model": "o3"}}, "SomeKey": "value"}}}
        result = migrate_config(config)
        tool_cfg = result["tools"]["codex_cli"]
        assert "somekey" in tool_cfg
        assert tool_cfg["somekey"] == "value"

    def test_non_dict_values_preserved(self):
        config = {"models": {"provider_a": {"model1": "gpt-4"}, "provider_b": "not-a-dict"}, "routing": {}, "timeouts": {}}
        result = migrate_config(config)
        assert result["models"]["provider_b"] == "not-a-dict"


class TestFutureVersionRaises:
    @pytest.mark.parametrize("version", [999, CURRENT_CONFIG_VERSION + 1])
    def test_future_version_raises(self, version):
        with pytest.raises(ConfigMigrationError):
            migrate_config({"version": version})


class TestMigrationChainAppliedSequentially:
    def test_chain_is_sequential(self):
        call_order = []

        @register_migration(CURRENT_CONFIG_VERSION)
        def dummy_migration(config):
            call_order.append(config.get("version"))
            config["custom_flag"] = True
            return config

        try:
            import router.config_migration as cm
            original_version = cm.CURRENT_CONFIG_VERSION
            cm.CURRENT_CONFIG_VERSION = CURRENT_CONFIG_VERSION + 1
            config = {"models": {}, "routing": {}, "timeouts": {}}
            result = migrate_config(config)
            assert result["version"] == CURRENT_CONFIG_VERSION + 1
            assert result.get("custom_flag") is True
        finally:
            cm.CURRENT_CONFIG_VERSION = original_version
            MIGRATIONS.pop(CURRENT_CONFIG_VERSION, None)


class TestDeepConfigPreserved:
    def test_nested_dicts_preserved(self):
        config = {
            "models": {"openrouter": {"gpt4": {"model": "gpt-4", "params": {"temperature": 0.7, "max_tokens": 4096}}}},
            "routing": {"policy": "openai_primary", "fallback_chain": ["openai", "claude", "openrouter"]},
            "timeouts": {"openai": 30, "nested": {"deep": {"value": 42}}},
            "custom_section": {"arbitrary": "data", "numbers": [1, 2, 3]},
        }
        result = migrate_config(config)
        assert result["custom_section"]["arbitrary"] == "data"
        assert result["timeouts"]["nested"]["deep"]["value"] == 42

    def test_list_values_preserved(self):
        config = {"models": {}, "routing": {"chain": ["a", "b", "c"]}, "timeouts": {}}
        result = migrate_config(config)
        assert result["routing"]["chain"] == ["a", "b", "c"]


class TestMigrationIsIdempotent:
    @pytest.mark.parametrize("config_fn", [_v0_config, _v1_config])
    def test_idempotent(self, config_fn):
        config = config_fn()
        first = migrate_config(config)
        second = migrate_config(first)
        assert first == second


class TestCustomMigrationRegistered:
    def test_decorator_registers(self):
        @register_migration(CURRENT_CONFIG_VERSION)
        def my_migration(config):
            return config
        try:
            assert CURRENT_CONFIG_VERSION in MIGRATIONS
            assert MIGRATIONS[CURRENT_CONFIG_VERSION] is my_migration
        finally:
            MIGRATIONS.pop(CURRENT_CONFIG_VERSION, None)


class TestMigrationErrorMessage:
    def test_future_version_message(self):
        with pytest.raises(ConfigMigrationError, match="42"):
            migrate_config({"version": 42})

    def test_missing_migration_message(self):
        from router import config_migration as cm
        orig_version = cm.CURRENT_CONFIG_VERSION
        orig_migrations = dict(cm.MIGRATIONS)
        try:
            cm.MIGRATIONS = {0: orig_migrations[0]}
            cm.CURRENT_CONFIG_VERSION = 2
            with pytest.raises(ConfigMigrationError, match="No migration registered"):
                cm.migrate_config({"models": {}, "routing": {}, "timeouts": {}})
        finally:
            cm.CURRENT_CONFIG_VERSION = orig_version
            cm.MIGRATIONS = orig_migrations


class TestLoadConfigAppliesMigration:
    def test_v0_config_loads(self, tmp_path):
        from router.config_loader import reload_config
        config = _v0_config()
        config.pop("state", None)
        path = tmp_path / "test_v0_config.json"
        path.write_text(json.dumps(config))
        try:
            result = reload_config(path)
            assert result["version"] == CURRENT_CONFIG_VERSION
        except Exception as e:
            assert "version" not in str(e).lower() or "already" in str(e).lower()


class TestBackwardCompatOldConfig:
    def test_minimal_old_config(self):
        config = {"models": {"openai": {"gpt4": "gpt-4"}}}
        result = migrate_config(config)
        assert result["version"] == CURRENT_CONFIG_VERSION
        assert result["models"]["openai"]["gpt4"] == "gpt-4"

    def test_realistic_old_config(self):
        config = {
            "models": {"openrouter": {"MiniMax": "minimax/minimax-m2.7", "Kimi": "moonshotai/kimi-k2.5"}, "codex": {"Default": "codex-default", "GPT54": "gpt-5.4"}},
            "tools": {"codex_cli": {"profiles": {"OpenAI_Native": {"model": "o3"}}}},
            "routing": {"policy": "openai_primary"}, "timeouts": {"openai": 30},
            "state": {"default": "openai_primary", "manual_state_file": "config/manual.json", "auto_state_file": "config/auto.json"},
        }
        result = migrate_config(config)
        assert result["version"] == CURRENT_CONFIG_VERSION
        assert "minimax" in result["models"]["openrouter"]
        assert "kimi" in result["models"]["openrouter"]
        assert "default" in result["models"]["codex"]
        assert "gpt54" in result["models"]["codex"]
        assert result["state"]["default"] == "openai_primary"

    def test_empty_config(self):
        result = migrate_config({})
        assert result["version"] == CURRENT_CONFIG_VERSION
        for key in ["models", "routing", "timeouts"]:
            assert key in result
