"""Tests for model version tracking and deprecation warnings."""

import json
import logging
import tempfile
from pathlib import Path
import pytest

from router.model_registry import (
    ModelVersion, KNOWN_MODELS, check_model_deprecation,
    validate_config_models, get_replacement, register_model, list_deprecated_models,
)


class TestModelVersion:
    def test_creation(self):
        mv = ModelVersion("gpt-4o", "openai")
        assert mv.name == "gpt-4o"
        assert mv.deprecated is False
        assert mv.replaced_by is None

    def test_deprecated_fields(self):
        mv = ModelVersion("gpt-4", "openai", deprecated=True, deprecated_at="2024-05-13",
                          replaced_by="gpt-4o", sunset_date="2025-06-01", notes="Old model")
        assert mv.deprecated is True
        assert mv.replaced_by == "gpt-4o"
        assert mv.sunset_date == "2025-06-01"


class TestCheckModelDeprecation:
    @pytest.mark.parametrize("model,deprecated,warns", [
        ("gpt-4", True, True),
        ("gpt-4o", False, False),
        ("future-model-2030", False, False),
    ])
    def test_deprecation_check(self, model, deprecated, warns):
        msg = check_model_deprecation(model)
        if warns:
            assert msg is not None
            assert "deprecated" in msg
        else:
            assert msg is None

    def test_deprecated_with_sunset_date(self):
        register_model(ModelVersion("sunset-model", "openai", deprecated=True, replaced_by="new-model", sunset_date="2025-12-31"))
        msg = check_model_deprecation("sunset-model")
        assert "sunset: 2025-12-31" in msg


class TestValidateConfigModels:
    def test_finds_deprecated(self):
        config = {"models": {"openai": {"primary": "gpt-4o", "legacy": "gpt-4"}}}
        warnings = validate_config_models(config)
        assert len(warnings) >= 1
        assert any("gpt-4" in w and "openai.legacy" in w for w in warnings)

    @pytest.mark.parametrize("config", [
        {"models": {"openai": {"primary": "gpt-4o", "mini": "gpt-4o-mini"}}},
        {"other": "value"},
        {"models": {}},
        {"models": {"openai": "some-string-value"}},
        {"models": ["gpt-4", "gpt-4o"]},
    ])
    def test_clean_or_invalid_configs(self, config):
        assert validate_config_models(config) == []

    def test_multiple_providers(self):
        original = dict(KNOWN_MODELS)
        try:
            KNOWN_MODELS["claude-3-opus"] = ModelVersion("claude-3-opus", "claude", deprecated=True, replaced_by="claude-sonnet-4-20250514")
            config = {"models": {"openai": {"legacy": "gpt-4"}, "claude": {"old": "claude-3-opus"}}}
            warnings = validate_config_models(config)
            assert len(warnings) == 2
        finally:
            KNOWN_MODELS.clear()
            KNOWN_MODELS.update(original)

    def test_none_values_skipped(self):
        config = {"models": {"openai": {"primary": "gpt-4o", "unused": None}}}
        assert isinstance(validate_config_models(config), list)

    def test_nested_config(self):
        config = {
            "models": {"openai": {"profile_a": "gpt-4o", "profile_b": "gpt-4"}},
            "tools": {"codex_cli": {"profiles": {"codex_gpt4": {"model": "gpt-4"}}}},
        }
        warnings = validate_config_models(config)
        assert any("gpt-4" in w and "openai.profile_b" in w for w in warnings)


class TestGetReplacement:
    @pytest.mark.parametrize("model,expected", [("gpt-4", "gpt-4o"), ("gpt-4o", None), ("nonexistent-model", None)])
    def test_replacement(self, model, expected):
        assert get_replacement(model) == expected


class TestRegisterModel:
    def test_register_custom(self):
        mv = ModelVersion("my-custom-model", "openrouter")
        register_model(mv)
        assert "my-custom-model" in KNOWN_MODELS
        assert check_model_deprecation("my-custom-model") is None
        del KNOWN_MODELS["my-custom-model"]

    def test_register_overwrites(self):
        original = KNOWN_MODELS.get("gpt-4o")
        register_model(ModelVersion("gpt-4o", "openai", deprecated=True, replaced_by="gpt-5"))
        assert KNOWN_MODELS["gpt-4o"].deprecated is True
        register_model(original)


class TestListDeprecatedModels:
    def test_returns_deprecated_only(self):
        deprecated = list_deprecated_models()
        names = [m.name for m in deprecated]
        assert "gpt-4" in names
        for m in deprecated:
            assert m.deprecated is True
        assert "gpt-4o" not in names
        assert "gpt-4o-mini" not in names


class TestConfigLoaderIntegration:
    @pytest.fixture(autouse=True)
    def _reset_config(self):
        yield
        from router.config_loader import reload_config
        reload_config()

    def _load_config(self, models_dict):
        from router.config_loader import reload_config
        config = {"models": models_dict}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            reload_config(Path(f.name))
        return Path(f.name)

    def test_deprecated_model_warns(self, caplog):
        self._load_config({"openai": {"legacy": "gpt-4"}})
        from router.config_loader import get_model_with_deprecation_check
        with caplog.at_level(logging.WARNING):
            model = get_model_with_deprecation_check("legacy")
        assert model == "gpt-4"
        assert any("deprecated" in r.message for r in caplog.records)

    def test_current_model_no_warn(self, caplog):
        self._load_config({"openai": {"primary": "gpt-4o"}})
        from router.config_loader import get_model_with_deprecation_check
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            model = get_model_with_deprecation_check("primary")
        assert model == "gpt-4o"
        assert not any("deprecated" in r.message for r in caplog.records)

    def test_load_config_logs_deprecation(self, caplog):
        from router.config_loader import reload_config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"models": {"openai": {"legacy": "gpt-4"}}}, f)
            f.flush()
            with caplog.at_level(logging.WARNING):
                reload_config(Path(f.name))
        assert any("deprecated" in r.message for r in caplog.records)
