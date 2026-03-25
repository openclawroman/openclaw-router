"""Config version migration system for backward compatibility."""
from typing import Any, Callable, Dict, List, Optional
import copy

# Current config schema version
CURRENT_CONFIG_VERSION = 1

# Migration functions: old_version -> new_version
MIGRATIONS: Dict[int, Callable[[dict], dict]] = {}


def register_migration(from_version: int):
    """Decorator to register a migration function."""
    def decorator(fn: Callable[[dict], dict]):
        MIGRATIONS[from_version] = fn
        return fn
    return decorator


def migrate_config(config: dict) -> dict:
    """Migrate a config dict to the current schema version.

    If no version field, assumes version 0 (original format).
    Applies migrations sequentially: 0→1→2→...→CURRENT.
    """
    config = copy.deepcopy(config)
    version = config.get("version", 0)

    if not isinstance(version, int):
        # Non-integer version: skip migration, pass through for downstream validation
        return config

    if version > CURRENT_CONFIG_VERSION:
        raise ConfigMigrationError(
            f"Config version {version} is newer than supported ({CURRENT_CONFIG_VERSION}). "
            f"Update the router to handle this config."
        )

    while version < CURRENT_CONFIG_VERSION:
        if version not in MIGRATIONS:
            raise ConfigMigrationError(
                f"No migration registered for config version {version} → {version + 1}"
            )
        config = MIGRATIONS[version](config)
        version += 1
        config["version"] = version

    return config


class ConfigMigrationError(Exception):
    """Raised when config migration fails."""
    pass


# --- Version 0 → 1: Add version field, normalize model keys ---
@register_migration(0)
def migrate_v0_to_v1(config: dict) -> dict:
    """Migrate from original config format to version 1.

    Changes:
    - Add "version": 1
    - Ensure "models" section exists
    - Normalize model profile names to lowercase
    """
    config.setdefault("models", {})
    config.setdefault("routing", {})
    config.setdefault("timeouts", {})

    # Normalize model profile names to lowercase
    for section in ["models", "tools"]:
        if section in config:
            for key in list(config[section].keys()):
                if isinstance(config[section][key], dict):
                    normalized = {k.lower(): v for k, v in config[section][key].items()}
                    config[section][key] = normalized

    return config
