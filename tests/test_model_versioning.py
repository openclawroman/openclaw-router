"""Tests for model version tracking and deprecation warnings."""

import json
import logging
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch
from types import MappingProxyType

from router.model_registry import (
    ModelVersion,
    KNOWN_MODELS,
    check_model_deprecation,
    validate_config_models,
    get_replacement,
    register_model,
    list_deprecated_models,
)


class TestModelVersion:
    """Tests for the ModelVersion dataclass."""

    def test_model_version_creation(self):
        mv = ModelVersion("gpt-4o", "openai")
        assert mv.name == "gpt-4o"
        assert mv.provider == "openai"
        assert mv.deprecated is False
        assert mv.replaced_by is None
        assert mv.sunset_date is None
        assert mv.notes == ""

    def test_model_version_deprecated_fields(self):
        mv = ModelVersion(
            "gpt-4", "openai",
            deprecated=True,
            deprecated_at="2024-05-13",
            replaced_by="gpt-4o",
            sunset_date="2025-06-01",
            notes="Old model",
        )
        assert mv.deprecated is True
        assert mv.deprecated_at == "2024-05-13"
        assert mv.replaced_by == "gpt-4o"
        assert mv.sunset_date == "2025-06-01"
        assert mv.notes == "Old model"


class TestCheckModelDeprecation:
    """Tests for check_model_deprecation()."""

    def test_deprecated_model_detected(self):
        """'gpt-4' is in the registry as deprecated."""
        msg = check_model_deprecation("gpt-4")
        assert msg is not None
        assert "deprecated" in msg
        assert "gpt-4" in msg
        assert "gpt-4o" in msg  # replacement mentioned

    def test_current_model_no_warning(self):
        """'gpt-4o' is current — no warning."""
        assert check_model_deprecation("gpt-4o") is None

    def test_unknown_model_no_warning(self):
        """Unknown/future models return None (not tracked as deprecated)."""
        assert check_model_deprecation("future-model-2030") is None

    def test_deprecated_without_replacement(self):
        """Deprecated model without replacement still warns."""
        register_model(ModelVersion("old-model-x", "openai", deprecated=True))
        msg = check_model_deprecation("old-model-x")
        assert msg is not None
        assert "deprecated" in msg
        assert "use " not in msg.lower().split("deprecated")[1] if "use" in msg else True

    def test_deprecated_with_sunset_date(self):
        """Deprecated model with sunset_date includes it in warning."""
        register_model(
            ModelVersion(
                "sunset-model", "openai",
                deprecated=True,
                replaced_by="new-model",
                sunset_date="2025-12-31",
            )
        )
        msg = check_model_deprecation("sunset-model")
        assert "sunset: 2025-12-31" in msg


class TestValidateConfigModels:
    """Tests for validate_config_models()."""

    def test_validate_config_finds_deprecated(self):
        """Config containing gpt-4 produces warnings."""
        config = {
            "models": {
                "openai": {
                    "primary": "gpt-4o",
                    "legacy": "gpt-4",
                }
            }
        }
        warnings = validate_config_models(config)
        assert len(warnings) >= 1
        assert any("gpt-4" in w and "openai.legacy" in w for w in warnings)

    def test_validate_config_clean(self):
        """Config with only current models produces no warnings."""
        config = {
            "models": {
                "openai": {
                    "primary": "gpt-4o",
                    "mini": "gpt-4o-mini",
                },
                "claude": {
                    "sonnet": "claude-sonnet-4-20250514",
                },
            }
        }
        warnings = validate_config_models(config)
        assert warnings == []

    def test_validate_config_multiple_providers(self):
        """Deprecated models across multiple providers are all detected."""
        # Add a deprecated claude model for this test
        original_registry = dict(KNOWN_MODELS)
        try:
            KNOWN_MODELS["claude-3-opus"] = ModelVersion(
                "claude-3-opus", "claude",
                deprecated=True, replaced_by="claude-sonnet-4-20250514",
            )
            config = {
                "models": {
                    "openai": {"legacy": "gpt-4"},
                    "claude": {"old": "claude-3-opus"},
                }
            }
            warnings = validate_config_models(config)
            assert len(warnings) == 2
            assert any("gpt-4" in w for w in warnings)
            assert any("claude-3-opus" in w for w in warnings)
        finally:
            KNOWN_MODELS.clear()
            KNOWN_MODELS.update(original_registry)

    def test_config_models_with_none_values(self):
        """None model values in config are skipped."""
        config = {
            "models": {
                "openai": {
                    "primary": "gpt-4o",
                    "unused": None,
                }
            }
        }
        warnings = validate_config_models(config)
        # Should not crash, and should not produce warnings for None
        assert isinstance(warnings, list)

    def test_config_models_section_nested(self):
        """Nested model configs with deprecated models at various depths."""
        config = {
            "models": {
                "openai": {
                    "profile_a": "gpt-4o",
                    "profile_b": "gpt-4",
                    "profile_c": "gpt-4o-mini",
                },
                "claude": {
                    "sonnet": "claude-sonnet-4-20250514",
                },
            },
            "tools": {
                "codex_cli": {
                    "profiles": {
                        "codex_gpt4": {"model": "gpt-4"},
                    }
                }
            }
        }
        warnings = validate_config_models(config)
        # Should detect gpt-4 in models.openai.profile_b
        assert any("gpt-4" in w and "openai.profile_b" in w for w in warnings)

    def test_validate_config_no_models_section(self):
        """Config without 'models' key returns no warnings."""
        config = {"other": "value"}
        warnings = validate_config_models(config)
        assert warnings == []

    def test_validate_config_empty_models(self):
        """Config with empty models returns no warnings."""
        config = {"models": {}}
        warnings = validate_config_models(config)
        assert warnings == []

    def test_validate_config_non_dict_provider(self):
        """Non-dict provider values are skipped gracefully."""
        config = {
            "models": {
                "openai": "some-string-value",
            }
        }
        warnings = validate_config_models(config)
        assert warnings == []

    def test_validate_config_models_is_list(self):
        """Config with models as list (wrong type) returns no warnings."""
        config = {"models": ["gpt-4", "gpt-4o"]}
        warnings = validate_config_models(config)
        assert warnings == []


class TestGetReplacement:
    """Tests for get_replacement()."""

    def test_get_replacement(self):
        """Deprecated 'gpt-4' returns 'gpt-4o'."""
        assert get_replacement("gpt-4") == "gpt-4o"

    def test_get_replacement_non_deprecated(self):
        """Current model returns None."""
        assert get_replacement("gpt-4o") is None

    def test_get_replacement_unknown(self):
        """Unknown model returns None."""
        assert get_replacement("nonexistent-model") is None


class TestRegisterModel:
    """Tests for register_model()."""

    def test_register_custom_model(self):
        """Register a custom model and verify it's found."""
        mv = ModelVersion("my-custom-model", "openrouter")
        register_model(mv)
        assert "my-custom-model" in KNOWN_MODELS
        assert check_model_deprecation("my-custom-model") is None
        # Cleanup
        del KNOWN_MODELS["my-custom-model"]

    def test_register_overwrites_existing(self):
        """Registering an existing model name overwrites the entry."""
        original = KNOWN_MODELS.get("gpt-4o")
        mv = ModelVersion("gpt-4o", "openai", deprecated=True, replaced_by="gpt-5")
        register_model(mv)
        assert KNOWN_MODELS["gpt-4o"].deprecated is True
        # Restore
        register_model(original)


class TestListDeprecatedModels:
    """Tests for list_deprecated_models()."""

    def test_list_deprecated_models(self):
        """Returns all deprecated models from the registry."""
        deprecated = list_deprecated_models()
        names = [m.name for m in deprecated]
        assert "gpt-4" in names
        # All returned models must be deprecated
        for m in deprecated:
            assert m.deprecated is True

    def test_no_current_models_in_deprecated_list(self):
        """Current (non-deprecated) models are not in the deprecated list."""
        deprecated = list_deprecated_models()
        names = [m.name for m in deprecated]
        assert "gpt-4o" not in names
        assert "gpt-4o-mini" not in names


class TestConfigLoaderIntegration:
    """Integration tests with config_loader."""

    @pytest.fixture(autouse=True)
    def _reset_config(self):
        """Reset config loader global state after each integration test."""
        yield
        from router.config_loader import reload_config
        reload_config()  # Reset to production config path (no arg = default)

    def test_get_model_with_deprecation_check_warns(self, caplog):
        """get_model_with_deprecation_check logs warning for deprecated models."""
        from router.config_loader import get_model_with_deprecation_check, reload_config

        config = {
            "models": {
                "openai": {"legacy": "gpt-4"},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            reload_config(Path(f.name))

        with caplog.at_level(logging.WARNING):
            model = get_model_with_deprecation_check("legacy")
        assert model == "gpt-4"
        assert any("deprecated" in r.message for r in caplog.records)

    def test_get_model_with_deprecation_check_no_warn(self, caplog):
        """No warning for current models."""
        from router.config_loader import get_model_with_deprecation_check, reload_config

        config = {
            "models": {
                "openai": {"primary": "gpt-4o"},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            reload_config(Path(f.name))

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            model = get_model_with_deprecation_check("primary")
        assert model == "gpt-4o"
        assert not any("deprecated" in r.message for r in caplog.records)

    def test_load_config_logs_deprecation_warnings(self, caplog):
        """load_config logs deprecation warnings for deprecated models in config."""
        from router.config_loader import reload_config

        config = {
            "models": {
                "openai": {"legacy": "gpt-4"},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            with caplog.at_level(logging.WARNING):
                reload_config(Path(f.name))
        assert any("deprecated" in r.message for r in caplog.records)
