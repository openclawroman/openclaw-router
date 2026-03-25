"""Config loader for router.config.json."""

import json
from pathlib import Path
from typing import Optional, List

CONFIG_PATH = Path(__file__).parent.parent / "config" / "router.config.json"

_config_cache: Optional[dict] = None
_active_config_path: Optional[Path] = None


def _get_config_path() -> Path:
    """Return the active config path (overridable for testing)."""
    return _active_config_path or CONFIG_PATH


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load router config from JSON file. Caches after first load."""
    global _config_cache
    if _config_cache is not None and config_path is None:
        return _config_cache
    path = config_path or _get_config_path()
    with open(path) as f:
        config = json.load(f)
    if config_path is None:
        _config_cache = config
    return config


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


def reload_config(config_path: Optional[Path] = None):
    """Clear config cache (for testing or hot-reload).

    If config_path is provided, it overrides the active config path
    until the next reload_config() call.
    """
    global _config_cache, _active_config_path
    _config_cache = None
    _active_config_path = config_path
