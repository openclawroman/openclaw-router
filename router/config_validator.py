"""Config validation for router.config.json."""

from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field


@dataclass
class ConfigError:
    """A single config validation error."""
    path: str
    message: str
    severity: str = "error"  # error | warning


@dataclass
class ValidationResult:
    """Result of config validation."""
    valid: bool
    errors: List[ConfigError] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == "warning")

    def summary(self) -> str:
        if self.valid:
            return f"Config OK ({len(self.errors)} warnings)" if self.errors else "Config OK"
        return f"Config INVALID: {self.error_count} error(s), {self.warning_count} warning(s)"


# Minimum required schema
REQUIRED_TOP_LEVEL_KEYS = {
    "version": int,
    "models": dict,
    "state": dict,
    "tools": dict,
}

REQUIRED_MODELS_SECTIONS = {
    "openrouter": dict,
    "codex": dict,
    "claude": dict,
}

REQUIRED_STATE_KEYS = {
    "default": str,
    "manual_state_file": str,
    "auto_state_file": str,
}

VALID_STATES = {"openai_primary", "openai_conservation", "claude_backup", "openrouter_fallback"}

SUPPORTED_VERSIONS = {1, 2}


def validate_config(config: dict) -> ValidationResult:
    """Validate router config against expected schema.
    
    Returns ValidationResult with valid=True/False and list of errors.
    """
    errors: List[ConfigError] = []

    # Check required top-level keys
    for key, expected_type in REQUIRED_TOP_LEVEL_KEYS.items():
        if key not in config:
            errors.append(ConfigError(key, f"Missing required key: '{key}'", "error"))
        elif not isinstance(config[key], expected_type):
            errors.append(ConfigError(key, f"Expected {expected_type.__name__}, got {type(config[key]).__name__}", "error"))

    # Check version
    version = config.get("version")
    if version is not None and version not in SUPPORTED_VERSIONS:
        errors.append(ConfigError(
            "version",
            f"Unsupported version {version}. Supported: {SUPPORTED_VERSIONS}",
            "error",
        ))

    # Check models section
    models = config.get("models", {})
    if isinstance(models, dict):
        for section, expected_type in REQUIRED_MODELS_SECTIONS.items():
            if section not in models:
                errors.append(ConfigError(f"models.{section}", f"Missing required models section: '{section}'", "error"))
            elif not isinstance(models[section], expected_type):
                errors.append(ConfigError(f"models.{section}", f"Expected {expected_type.__name__}", "error"))

        # Check each model section has at least one entry
        for section_name, section in models.items():
            if isinstance(section, dict) and len(section) == 0:
                errors.append(ConfigError(f"models.{section_name}", "Empty models section (need at least one model)", "warning"))

    # Check state section
    state = config.get("state", {})
    if isinstance(state, dict):
        for key, expected_type in REQUIRED_STATE_KEYS.items():
            if key not in state:
                errors.append(ConfigError(f"state.{key}", f"Missing required key: 'state.{key}'", "error"))
            elif not isinstance(state[key], expected_type):
                errors.append(ConfigError(f"state.{key}", f"Expected {expected_type.__name__}", "error"))

        # Check default state is valid
        default_state = state.get("default")
        if default_state is not None and default_state not in VALID_STATES:
            errors.append(ConfigError(
                "state.default",
                f"Invalid default state '{default_state}'. Valid: {VALID_STATES}",
                "error",
            ))

    # Check tools section
    tools = config.get("tools", {})
    if isinstance(tools, dict):
        if "codex_cli" not in tools:
            errors.append(ConfigError("tools.codex_cli", "Missing required tool: 'codex_cli'", "error"))
        if "claude_code" not in tools:
            errors.append(ConfigError("tools.claude_code", "Missing required tool: 'claude_code'", "warning"))

    # Check reliability section (optional but if present, validate)
    reliability = config.get("reliability", {})
    if isinstance(reliability, dict):
        if "chain_timeout_s" in reliability and not isinstance(reliability["chain_timeout_s"], (int, float)):
            errors.append(ConfigError("reliability.chain_timeout_s", "Expected number", "error"))
        if "max_fallbacks" in reliability and not isinstance(reliability["max_fallbacks"], int):
            errors.append(ConfigError("reliability.max_fallbacks", "Expected integer", "error"))

        cb = reliability.get("circuit_breaker", {})
        if isinstance(cb, dict):
            for key in ("threshold", "window_s", "cooldown_s"):
                if key in cb and not isinstance(cb[key], (int, float)):
                    errors.append(ConfigError(f"reliability.circuit_breaker.{key}", "Expected number", "error"))

    # Check for unknown top-level keys (warning only)
    known_keys = {"version", "models", "state", "openclaw", "tools", "routing",
                  "openrouter_dynamic_rules", "retry", "reliability", "logging"}
    unknown = set(config.keys()) - known_keys
    for key in unknown:
        errors.append(ConfigError(key, f"Unknown top-level key: '{key}'", "warning"))

    is_valid = all(e.severity != "error" for e in errors)
    return ValidationResult(valid=is_valid, errors=errors)


def validate_config_file(path: Optional[str] = None) -> ValidationResult:
    """Load and validate a config file. Returns validation result."""
    import json
    from pathlib import Path
    
    if path is None:
        from .config_loader import CONFIG_PATH
        path = CONFIG_PATH
    
    try:
        with open(Path(path)) as f:
            config = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return ValidationResult(valid=False, errors=[ConfigError("file", str(e), "error")])
    
    return validate_config(config)
