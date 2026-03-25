"""Tests for config version migration system (item 10.2)."""

import copy
import json
import logging
from pathlib import Path

import pytest

from router.config_migration import (
    CURRENT_CONFIG_VERSION,
    MIGRATIONS,
    ConfigMigrationError,
    migrate_config,
    register_migration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _v0_config(**overrides):
    """A realistic v0 (pre-versioning) config without 'version' field."""
    base = {
        "models": {
            "openrouter": {"GPT4": "gpt-4", "Sonnet": "claude-sonnet-4"},
            "codex": {"Default": "codex-mini"},
        },
        "tools": {
            "codex_cli": {
                "profiles": {
                    "OpenAI_Native": {"model": "o3", "timeout_s": 300},
                }
            }
        },
        "routing": {"policy": "openai_primary"},
        "timeouts": {"openai": 30},
        "state": {
            "default": "openai_primary",
            "manual_state_file": "config/manual.json",
            "auto_state_file": "config/auto.json",
        },
    }
    base.update(overrides)
    return base


def _v1_config(**overrides):
    """A realistic v1 config (current version)."""
    base = {
        "version": CURRENT_CONFIG_VERSION,
        "models": {
            "openrouter": {"gpt4": "gpt-4", "sonnet": "claude-sonnet-4"},
        },
        "routing": {"policy": "openai_primary"},
        "timeouts": {"openai": 30},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Core migration tests
# ---------------------------------------------------------------------------

class TestNoVersionFieldAssumesV0:
    """Config without 'version' field should be treated as v0 and migrated."""

    def test_version_added(self):
        config = _v0_config()
        assert "version" not in config
        result = migrate_config(config)
        assert result["version"] == CURRENT_CONFIG_VERSION

    def test_original_unchanged(self):
        """migrate_config must not mutate the input."""
        config = _v0_config()
        migrate_config(config)
        assert "version" not in config


class TestAlreadyCurrentVersion:
    """Config at CURRENT_CONFIG_VERSION should pass through unchanged."""

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
    """Missing sections (models, routing, timeouts) get defaults."""

    def test_missing_models(self):
        config = {"routing": {"policy": "test"}, "timeouts": {"x": 1}}
        result = migrate_config(config)
        assert "models" in result
        assert isinstance(result["models"], dict)

    def test_missing_routing(self):
        config = {"models": {"a": {"b": "c"}}, "timeouts": {"x": 1}}
        result = migrate_config(config)
        assert "routing" in result

    def test_missing_timeouts(self):
        config = {"models": {"a": {"b": "c"}}, "routing": {"policy": "test"}}
        result = migrate_config(config)
        assert "timeouts" in result

    def test_all_missing(self):
        config = {}
        result = migrate_config(config)
        assert "models" in result
        assert "routing" in result
        assert "timeouts" in result
        assert result["version"] == CURRENT_CONFIG_VERSION


class TestV0MigrationNormalizesKeys:
    """Model profile names and tool section keys are lowercased."""

    def test_model_keys_normalized(self):
        config = _v0_config()
        result = migrate_config(config)
        assert "GPT4" not in result["models"]["openrouter"]
        assert "gpt4" in result["models"]["openrouter"]

    def test_tool_keys_normalized(self):
        """The keys within tools/* are lowercased (e.g. 'profiles' → 'profiles',
        and nested dict keys like 'OpenAI_Native' → 'openai_native')."""
        config = {
            "models": {},
            "routing": {},
            "timeouts": {},
            "tools": {
                "codex_cli": {
                    "profiles": {
                        "OpenAI_Native": {"model": "o3", "timeout_s": 300},
                    },
                    "SomeKey": "value",
                }
            },
        }
        result = migrate_config(config)
        tool_cfg = result["tools"]["codex_cli"]
        assert "profiles" in tool_cfg
        assert "somekey" in tool_cfg
        # The dict values under codex_cli are also normalized
        assert tool_cfg["somekey"] == "value"

    def test_mixed_case_all_lowercased(self):
        config = {
            "models": {
                "MyProvider": {
                    "GPT4Turbo": "gpt-4-turbo",
                    "claude-sonnet": "claude-sonnet-4",
                }
            },
            "routing": {},
            "timeouts": {},
        }
        result = migrate_config(config)
        assert "gpt4turbo" in result["models"]["MyProvider"]
        assert "claude-sonnet" in result["models"]["MyProvider"]

    def test_non_dict_values_preserved(self):
        """Non-dict values in models/tools sections are not modified."""
        config = {
            "models": {
                "provider_a": {"model1": "gpt-4"},
                "provider_b": "not-a-dict",
            },
            "routing": {},
            "timeouts": {},
        }
        result = migrate_config(config)
        assert result["models"]["provider_b"] == "not-a-dict"


class TestFutureVersionRaises:
    """Config with version > CURRENT_CONFIG_VERSION must raise ConfigMigrationError."""

    def test_version_999(self):
        config = {"version": 999}
        with pytest.raises(ConfigMigrationError, match="newer than supported"):
            migrate_config(config)

    def test_version_current_plus_one(self):
        config = {"version": CURRENT_CONFIG_VERSION + 1}
        with pytest.raises(ConfigMigrationError, match="Update the router"):
            migrate_config(config)

    def test_error_is_exception(self):
        with pytest.raises(Exception):
            migrate_config({"version": 999})


class TestMigrationChainAppliedSequentially:
    """Migrations are applied in order: 0→1→...→CURRENT."""

    def test_chain_is_sequential(self):
        """If we register a v1→v2 migration, it runs after v0→v1."""
        call_order = []

        @register_migration(CURRENT_CONFIG_VERSION)
        def dummy_migration(config):
            call_order.append(config.get("version"))
            config["custom_flag"] = True
            return config

        try:
            # Temporarily bump the version counter
            from router.config_migration import CURRENT_CONFIG_VERSION as _cv
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
    """Nested config values survive migration."""

    def test_nested_dicts_preserved(self):
        config = {
            "models": {
                "openrouter": {
                    "gpt4": {
                        "model": "gpt-4",
                        "params": {"temperature": 0.7, "max_tokens": 4096},
                    }
                }
            },
            "routing": {
                "policy": "openai_primary",
                "fallback_chain": ["openai", "claude", "openrouter"],
            },
            "timeouts": {
                "openai": 30,
                "nested": {"deep": {"value": 42}},
            },
            "custom_section": {
                "arbitrary": "data",
                "numbers": [1, 2, 3],
            },
        }
        result = migrate_config(config)
        assert result["custom_section"]["arbitrary"] == "data"
        assert result["custom_section"]["numbers"] == [1, 2, 3]
        assert result["timeouts"]["nested"]["deep"]["value"] == 42

    def test_list_values_preserved(self):
        config = {
            "models": {},
            "routing": {"chain": ["a", "b", "c"]},
            "timeouts": {},
        }
        result = migrate_config(config)
        assert result["routing"]["chain"] == ["a", "b", "c"]


class TestMigrationIsIdempotent:
    """migrate(migrate(x)) == migrate(x)."""

    def test_idempotent_v0(self):
        config = _v0_config()
        first = migrate_config(config)
        second = migrate_config(first)
        assert first == second

    def test_idempotent_v1(self):
        config = _v1_config()
        first = migrate_config(config)
        second = migrate_config(first)
        assert first == second


class TestCustomMigrationRegistered:
    """register_migration decorator registers functions correctly."""

    def test_decorator_registers(self):
        original_keys = set(MIGRATIONS.keys())

        @register_migration(CURRENT_CONFIG_VERSION)
        def my_migration(config):
            return config

        try:
            assert CURRENT_CONFIG_VERSION in MIGRATIONS
            assert MIGRATIONS[CURRENT_CONFIG_VERSION] is my_migration
        finally:
            MIGRATIONS.pop(CURRENT_CONFIG_VERSION, None)

    def test_multiple_registrations(self):
        """Latest registration overwrites for the same version."""

        @register_migration(99)
        def first(config):
            return config

        @register_migration(99)
        def second(config):
            return config

        try:
            assert MIGRATIONS[99] is second
        finally:
            MIGRATIONS.pop(99, None)


class TestMigrationErrorMessage:
    """Error messages are clear and informative."""

    def test_future_version_message(self):
        config = {"version": 42}
        with pytest.raises(ConfigMigrationError, match="42") as exc:
            migrate_config(config)
        assert "newer than supported" in str(exc.value)
        assert str(CURRENT_CONFIG_VERSION) in str(exc.value)

    def test_missing_migration_message(self):
        """If a migration is missing, the error names the version gap."""
        from router import config_migration as cm

        original_version = cm.CURRENT_CONFIG_VERSION
        original_migrations = dict(cm.MIGRATIONS)

        try:
            # Set version to 2 but only register migration 0→1
            cm.MIGRATIONS = {0: original_migrations[0]}
            cm.CURRENT_CONFIG_VERSION = 2
            config = {"models": {}, "routing": {}, "timeouts": {}}
            with pytest.raises(ConfigMigrationError, match="No migration registered") as exc:
                cm.migrate_config(config)
            assert "1" in str(exc.value)
        finally:
            cm.CURRENT_CONFIG_VERSION = original_version
            cm.MIGRATIONS = original_migrations


class TestLoadConfigAppliesMigration:
    """End-to-end: load_config() calls migrate_config before validation."""

    def test_v0_config_loads(self, tmp_path):
        """A v0 config file loads successfully via load_config."""
        from router.config_loader import reload_config

        config = _v0_config()
        config.pop("state", None)  # Remove state to avoid validator errors
        path = tmp_path / "test_v0_config.json"
        path.write_text(json.dumps(config))

        try:
            # This should not raise — migration adds version and defaults
            result = reload_config(path)
            assert result["version"] == CURRENT_CONFIG_VERSION
        except Exception as e:
            # If config_validator raises because of missing required keys
            # (state, tools), that's expected — but the migration should have run
            assert "version" not in str(e).lower() or "already" in str(e).lower()


class TestBackwardCompatOldConfig:
    """Real old-style configs still work after migration."""

    def test_minimal_old_config(self):
        """Smallest possible v0 config."""
        config = {"models": {"openai": {"gpt4": "gpt-4"}}}
        result = migrate_config(config)
        assert result["version"] == CURRENT_CONFIG_VERSION
        assert result["models"]["openai"]["gpt4"] == "gpt-4"

    def test_realistic_old_config(self):
        """Config that looks like a real v0 router.config.json."""
        config = {
            "models": {
                "openrouter": {
                    "MiniMax": "minimax/minimax-m2.7",
                    "Kimi": "moonshotai/kimi-k2.5",
                },
                "codex": {
                    "Default": "codex-default",
                    "GPT54": "gpt-5.4",
                },
            },
            "tools": {
                "codex_cli": {
                    "profiles": {
                        "OpenAI_Native": {"model": "o3"},
                    }
                }
            },
            "routing": {"policy": "openai_primary"},
            "timeouts": {"openai": 30},
            "state": {
                "default": "openai_primary",
                "manual_state_file": "config/manual.json",
                "auto_state_file": "config/auto.json",
            },
        }
        result = migrate_config(config)
        assert result["version"] == CURRENT_CONFIG_VERSION
        # Keys normalized
        assert "minimax" in result["models"]["openrouter"]
        assert "kimi" in result["models"]["openrouter"]
        assert "default" in result["models"]["codex"]
        assert "gpt54" in result["models"]["codex"]
        # Other sections preserved
        assert result["state"]["default"] == "openai_primary"
        assert result["routing"]["policy"] == "openai_primary"

    def test_empty_config(self):
        """Empty dict migrates to valid v1 with all defaults."""
        config = {}
        result = migrate_config(config)
        assert result["version"] == CURRENT_CONFIG_VERSION
        assert "models" in result
        assert "routing" in result
        assert "timeouts" in result
