"""Config loader for router.config.json."""

import copy
import json
from pathlib import Path
from typing import Optional, List

CONFIG_PATH = Path(__file__).parent.parent / "config" / "router.config.json"

_config_cache: Optional[dict] = None
_active_config_path: Optional[Path] = None


def _get_config_path() -> Path:
    """Return the active config path (overridable for testing)."""
    return _active_config_path or CONFIG_PATH


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _merge_extends(config: dict, path: Path) -> dict:
    """Handle _extends inheritance: deep merge base config with overrides."""
    if "_extends" not in config:
        return config

    base_path = path.parent / config["_extends"]
    with open(base_path) as f:
        base_config = json.load(f)

    return _deep_merge(base_config, config)


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load router config. Supports environment overrides.

    Path resolution priority:
    1. config_path argument (explicit)
    2. Active config path (from previous reload_config)
    3. Environment resolution (CLI flag > env var > environment default)

    Raises ConfigValidationError if config is invalid."""
    global _config_cache

    if _config_cache is not None and config_path is None:
        return _config_cache

    # Resolve path
    if config_path is not None:
        path = Path(config_path)
    elif _active_config_path is not None:
        path = _active_config_path
    else:
        from .environments import get_config_path, apply_env_overrides
        path = get_config_path()

    with open(path) as f:
        config = json.load(f)


    # Handle _extends (base config inheritance)
    config = _merge_extends(config, path)

    # Apply environment overrides (only when using environment resolution)
    if config_path is None and _active_config_path is None:
        try:
            from .environments import apply_env_overrides as _apply
            config = _apply(config)
        except ImportError:
            pass

    # Validate on load (skip when testing with a custom config path)
    if config_path is None and _active_config_path is None:
        from .config_validator import validate_config
        result = validate_config(config)
        if not result.valid:
            raise ConfigValidationError(result)
    if config_path is None:
        _config_cache = config
    return config


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    def __init__(self, validation_result):
        self.result = validation_result
        super().__init__(validation_result.summary())


def get_model(profile: str) -> str:
    """
    Get the model string for a given profile name.

    Examples:
        get_model("minimax") -> "minimax/minimax-m2.7"
        get_model("kimi") -> "moonshotai/kimi-k2.5"
        get_model("codex_default") -> "codex-default"
        get_model("claude_default") -> "claude-default"
    """
    config = load_config()
    models = config.get("models", {})

    for section in models.values():
        if isinstance(section, dict) and profile in section:
            return section[profile]

    # Also check top-level codex/claude/openrouter configs
    for tool_config in config.get("tools", {}).values():
        if isinstance(tool_config, dict):
            profiles = tool_config.get("profiles", {})
            if profile in profiles:
                profile_cfg = profiles[profile]
                if isinstance(profile_cfg, dict) and "model" in profile_cfg:
                    return profile_cfg["model"]

    raise KeyError(f"Model profile '{profile}' not found in config. Available profiles: {list_models()}")


def list_models() -> List[str]:
    """List all available model profile names."""
    config = load_config()
    names = []
    for section in config.get("models", {}).values():
        if isinstance(section, dict):
            names.extend(section.keys())
    return names


def get_reliability_config() -> dict:
    """Get reliability/fallback configuration with defaults."""
    config = load_config()
    reliability = config.get("reliability", {})
    return {
        "chain_timeout_s": reliability.get("chain_timeout_s", 600),
        "drain_timeout_s": reliability.get("drain_timeout_s", 30),
        "max_retries": reliability.get("max_retries", 2),
        "max_fallbacks": reliability.get("max_fallbacks", 3),
    }


def reload_config(config_path: Optional[Path] = None):
    """Clear config cache (for testing or hot-reload).

    If config_path is provided, it overrides the active config path
    until the next reload_config() call.
    """
    global _config_cache, _active_config_path
    _config_cache = None
    _active_config_path = config_path
