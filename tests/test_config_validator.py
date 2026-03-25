"""Tests for config validation."""

import json
import pytest
from router.config_validator import (
    validate_config, validate_config_file,
    ValidationResult, ConfigError,
    SUPPORTED_VERSIONS, VALID_STATES,
)


def _valid_config():
    """Return a minimal valid config for testing."""
    return {
        "version": 1,
        "models": {
            "openrouter": {"minimax": "minimax/minimax-m2.7"},
            "codex": {"gpt54": "gpt-5.4"},
            "claude": {"sonnet": "claude-sonnet-4.6"},
        },
        "state": {
            "default": "openai_primary",
            "manual_state_file": "config/codex_manual_state.json",
            "auto_state_file": "config/codex_auto_state.json",
        },
        "tools": {
            "codex_cli": {"profiles": {"openai_native": {"timeout_s": 300}}},
            "claude_code": {"provider": "anthropic", "timeout_s": 300},
        },
    }


class TestValidConfig:
    def test_valid_config_passes(self):
        result = validate_config(_valid_config())
        assert result.valid
        assert result.error_count == 0

    def test_real_config_passes(self, tmp_path):
        """Actual router.config.json should validate."""
        from router.config_validator import validate_config_file
        from router.config_loader import CONFIG_PATH
        result = validate_config_file(str(CONFIG_PATH))
        assert result.valid


class TestMissingKeys:
    def test_missing_version(self):
        config = _valid_config()
        del config["version"]
        result = validate_config(config)
        assert not result.valid
        assert any("version" in e.path for e in result.errors)

    def test_missing_models(self):
        config = _valid_config()
        del config["models"]
        result = validate_config(config)
        assert not result.valid

    def test_missing_models_section(self):
        config = _valid_config()
        del config["models"]["codex"]
        result = validate_config(config)
        assert not result.valid

    def test_missing_state_default(self):
        config = _valid_config()
        del config["state"]["default"]
        result = validate_config(config)
        assert not result.valid


class TestInvalidTypes:
    def test_version_wrong_type(self):
        config = _valid_config()
        config["version"] = "one"
        result = validate_config(config)
        assert not result.valid

    def test_models_wrong_type(self):
        config = _valid_config()
        config["models"] = "not a dict"
        result = validate_config(config)
        assert not result.valid


class TestVersionValidation:
    def test_unsupported_version(self):
        config = _valid_config()
        config["version"] = 99
        result = validate_config(config)
        assert not result.valid
        assert any("Unsupported version" in e.message for e in result.errors)

    def test_supported_versions(self):
        for v in SUPPORTED_VERSIONS:
            config = _valid_config()
            config["version"] = v
            result = validate_config(config)
            assert result.valid, f"Version {v} should be valid"


class TestStateValidation:
    def test_invalid_default_state(self):
        config = _valid_config()
        config["state"]["default"] = "invalid_state"
        result = validate_config(config)
        assert not result.valid

    def test_valid_states(self):
        for state in VALID_STATES:
            config = _valid_config()
            config["state"]["default"] = state
            result = validate_config(config)
            assert result.valid, f"State '{state}' should be valid"


class TestUnknownKeys:
    def test_unknown_key_is_warning(self):
        config = _valid_config()
        config["unknown_section"] = {"foo": "bar"}
        result = validate_config(config)
        assert result.valid  # warnings don't invalidate
        assert result.warning_count >= 1
        assert any("unknown_section" in e.path for e in result.errors)


class TestReliabilityValidation:
    def test_invalid_timeout_type(self):
        config = _valid_config()
        config["reliability"] = {"chain_timeout_s": "not_a_number"}
        result = validate_config(config)
        assert not result.valid

    def test_invalid_circuit_breaker(self):
        config = _valid_config()
        config["reliability"] = {
            "circuit_breaker": {"threshold": "five"}
        }
        result = validate_config(config)
        assert not result.valid


class TestEmptyModels:
    def test_empty_models_warning(self):
        config = _valid_config()
        config["models"]["openrouter"] = {}
        result = validate_config(config)
        assert result.valid  # warning, not error
        assert result.warning_count >= 1


class TestValidationResult:
    def test_summary_valid(self):
        result = validate_config(_valid_config())
        assert "OK" in result.summary()

    def test_summary_invalid(self):
        config = _valid_config()
        del config["version"]
        result = validate_config(config)
        assert "INVALID" in result.summary()


class TestValidateConfigFile:
    """Edge-case tests for validate_config_file()."""

    def test_missing_file_returns_file_error(self, tmp_path):
        """Missing file should return ValidationResult with a file error."""
        missing = tmp_path / "nonexistent_config.json"
        result = validate_config_file(str(missing))
        assert not result.valid
        assert result.error_count == 1
        err = result.errors[0]
        assert err.path == "file"
        assert err.severity == "error"
        assert "nonexistent_config.json" in err.message

    def test_invalid_json_returns_parse_error(self, tmp_path):
        """Invalid JSON file should return ValidationResult with a parse error."""
        bad_json = tmp_path / "bad_config.json"
        bad_json.write_text("{invalid json!!!")
        result = validate_config_file(str(bad_json))
        assert not result.valid
        assert result.error_count == 1
        err = result.errors[0]
        assert err.path == "file"
        assert err.severity == "error"
        assert "JSON" in err.message or "json" in err.message.lower() or "line" in err.message.lower()

    def test_valid_file_validates_and_returns_result(self, tmp_path):
        """A valid config file should pass validation."""
        good = tmp_path / "good_config.json"
        import json as _json
        good.write_text(_json.dumps(_valid_config(), indent=2))
        result = validate_config_file(str(good))
        assert result.valid
        assert result.error_count == 0

    def test_invalid_schema_returns_schema_errors(self, tmp_path):
        """A JSON file with invalid schema should fail validation."""
        bad_schema = tmp_path / "bad_schema.json"
        import json as _json
        config = _valid_config()
        del config["version"]
        bad_schema.write_text(_json.dumps(config))
        result = validate_config_file(str(bad_schema))
        assert not result.valid
        assert result.error_count >= 1
        assert any("version" in e.path for e in result.errors)
