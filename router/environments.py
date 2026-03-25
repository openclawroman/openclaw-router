"""Environment-aware config loading for the router."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    name: str  # dev | staging | prod
    config_path: Path
    overrides: Dict[str, Any]


ENV_CONFIGS = {
    "dev": EnvironmentConfig(
        name="dev",
        config_path=Path("config/router.config.dev.json"),
        overrides={
            "reliability.chain_timeout_s": 300,  # shorter for dev
            "reliability.circuit_breaker.cooldown_s": 30,  # faster recovery in dev
        },
    ),
    "staging": EnvironmentConfig(
        name="staging",
        config_path=Path("config/router.config.staging.json"),
        overrides={},
    ),
    "prod": EnvironmentConfig(
        name="prod",
        config_path=Path("config/router.config.json"),
        overrides={},
    ),
}


def detect_environment() -> str:
    """Detect current environment from env vars.

    Priority:
    1. ROUTER_ENV env var
    2. Default to 'prod'
    """
    env = os.environ.get("ROUTER_ENV", "prod").lower()
    if env not in ENV_CONFIGS:
        raise ValueError(f"Unknown environment '{env}'. Valid: {list(ENV_CONFIGS.keys())}")
    return env


def get_config_path(cli_path: Optional[str] = None) -> Path:
    """Resolve config file path.

    Priority:
    1. CLI argument (--config path)
    2. ROUTER_CONFIG_PATH env var
    3. Environment-specific default
    """
    if cli_path:
        return Path(cli_path)

    env_path = os.environ.get("ROUTER_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    env = detect_environment()
    return ENV_CONFIGS[env].config_path


def apply_env_overrides(config: dict, env: Optional[str] = None) -> dict:
    """Apply environment-specific overrides to config.

    Returns a new dict with overrides applied (doesn't modify original).
    """
    if env is None:
        env = detect_environment()

    env_config = ENV_CONFIGS.get(env)
    if not env_config or not env_config.overrides:
        return config

    import copy
    result = copy.deepcopy(config)

    for dotted_key, value in env_config.overrides.items():
        keys = dotted_key.split(".")
        target = result
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    return result
