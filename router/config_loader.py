"""Config loader for router.config.json."""

import copy
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Optional, List

from .config_migration import migrate_config, CURRENT_CONFIG_VERSION
from .model_registry import validate_config_models, check_model_deprecation

logger = logging.getLogger(__name__)

RESTRICTIVE_PERMISSIONS = 0o600  # Owner read/write only

CONFIG_PATH = Path(__file__).parent.parent / "config" / "router.config.json"

# Thread-safe immutable config snapshot
_config_snapshot: Optional[MappingProxyType] = None
_config_raw: Optional[dict] = None  # mutable source for deep copies
_config_lock = Lock()
_active_config_path: Optional[Path] = None


def _restrict_permissions(path: Path) -> None:
    """Best-effort: set file permissions to owner-only read/write (0o600).

    Silently skips on platforms/filesystems where chmod fails (e.g. Windows).
    """
    try:
        os.chmod(path, RESTRICTIVE_PERMISSIONS)
    except OSError:
        logger.warning("Failed to set restrictive permissions on %s (may not be supported on this platform)", path)


def _get_config_path() -> Path:
    """Return the active config path (overridable for testing)."""
    return _active_config_path or CONFIG_PATH


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    def __init__(self, validation_result):
        self.result = validation_result
        super().__init__(validation_result.summary())


def _freeze(obj):
    """Recursively convert nested dicts to MappingProxyType for full immutability.

    Lists are frozen as tuples to ensure complete immutability of the
    returned snapshot.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return tuple(_freeze(item) for item in obj)
    return obj


def _unfreeze(obj):
    """Recursively convert frozen objects back to mutable forms.

    MappingProxyType -> dict, tuple -> list.  Everything else passes
    through unchanged.
    """
    if isinstance(obj, MappingProxyType):
        return {k: _unfreeze(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [_unfreeze(item) for item in obj]
    return obj


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load router config from JSON file.

    Raises ConfigValidationError if config is invalid.
    Returns a regular dict copy for backward compatibility (read-only recommended).
    """
    global _config_snapshot, _config_raw

    with _config_lock:
        if _config_snapshot is not None and config_path is None:
            return copy.deepcopy(_config_raw)

    path = config_path or _get_config_path()

    with open(path) as f:
        config = json.load(f)

    # Apply config migrations before validation (backward compat)
    original_version = config.get("version", 0)
    config = migrate_config(config)
    if config.get("version") != original_version:
        logger.info(
            "Config migrated from version %d to %d",
            original_version,
            config.get("version", CURRENT_CONFIG_VERSION),
        )

    if config_path is None and _active_config_path is None:
        # Validate before caching (production path only)
        try:
            from .config_validator import validate_config
            result = validate_config(config)
            if not result.valid:
                raise ConfigValidationError(result)
        except ImportError:
            pass

    # Check for deprecated model warnings
    dep_warnings = validate_config_models(config)
    for w in dep_warnings:
        logger.warning(w)

    # Cache: atomic swap under lock
    with _config_lock:
        _config_raw = copy.deepcopy(config)
        _config_snapshot = _freeze(config)

    return copy.deepcopy(config)


def reload_config(config_path: Optional[Path] = None) -> dict:
    """Reload config from disk. Clears cache, re-validates, returns new config.

    Thread-safe. Atomic swap — callers never see partial state.
    """
    global _config_snapshot, _config_raw, _active_config_path

    with _config_lock:
        _config_snapshot = None
        _config_raw = None
        _active_config_path = config_path

    return load_config()


def get_config_snapshot() -> MappingProxyType:
    """Get the raw immutable config snapshot (no copy).

    For read-only access without the overhead of load_config()'s copy.
    """
    global _config_snapshot
    with _config_lock:
        if _config_snapshot is not None:
            return _config_snapshot
    # Trigger load outside lock to avoid deadlock
    load_config()
    with _config_lock:
        return _config_snapshot


def get_model(profile: str) -> str:
    """
    Get the model string for a given profile name.

    Uses immutable snapshot to prevent mutation during iteration.
    """
    config = get_config_snapshot()
    models = config.get("models", {})

    for section in models.values():
        if isinstance(section, Mapping) and profile in section:
            return section[profile]

    for tool_config in config.get("tools", {}).values():
        if isinstance(tool_config, Mapping):
            profiles = tool_config.get("profiles", {})
            if profile in profiles:
                profile_cfg = profiles[profile]
                if isinstance(profile_cfg, Mapping) and "model" in profile_cfg:
                    return profile_cfg["model"]

    raise KeyError(f"Model profile '{profile}' not found in config. Available profiles: {list_models()}")


def list_models() -> List[str]:
    """List all available model profile names."""
    config = get_config_snapshot()
    names = []
    for section in config.get("models", {}).values():
        if isinstance(section, Mapping):
            names.extend(section.keys())
    return names


def get_reliability_config() -> dict:
    """Get reliability/fallback configuration with defaults.

    Returns a mutable copy — safe for callers to use without affecting global config.
    """
    config = get_config_snapshot()
    reliability = config.get("reliability", {})
    return {
        "chain_timeout_s": reliability.get("chain_timeout_s", 600),
        "drain_timeout_s": reliability.get("drain_timeout_s", 30),
        "max_retries": reliability.get("max_retries", 2),
        "max_fallbacks": reliability.get("max_fallbacks", 3),
    }


def get_model_with_deprecation_check(profile: str) -> str:
    """Get model string for a profile, warning if model is deprecated."""
    model = get_model(profile)
    warning = check_model_deprecation(model)
    if warning:
        logger.warning("[profile=%s] %s", profile, warning)
    return model
