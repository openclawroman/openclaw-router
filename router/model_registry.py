"""Model version tracking and deprecation management."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import warnings


@dataclass
class ModelVersion:
    """A tracked model version."""
    name: str
    provider: str  # "openai", "claude", "openrouter"
    deprecated: bool = False
    deprecated_at: Optional[str] = None  # ISO date
    replaced_by: Optional[str] = None
    sunset_date: Optional[str] = None  # When model will be removed
    notes: str = ""


# Built-in registry of known models and their deprecation status
KNOWN_MODELS: Dict[str, ModelVersion] = {
    "gpt-4": ModelVersion("gpt-4", "openai", deprecated=True, replaced_by="gpt-4o", deprecated_at="2024-05-13"),
    "gpt-4o": ModelVersion("gpt-4o", "openai"),
    "gpt-4o-mini": ModelVersion("gpt-4o-mini", "openai"),
    "claude-sonnet-4-20250514": ModelVersion("claude-sonnet-4-20250514", "claude"),
    "claude-haiku-3.5": ModelVersion("claude-haiku-3.5", "claude"),
    "codex-mini": ModelVersion("codex-mini", "openai"),
}


def check_model_deprecation(model_name: str) -> Optional[str]:
    """Check if a model is deprecated. Returns warning message or None."""
    entry = KNOWN_MODELS.get(model_name)
    if entry and entry.deprecated:
        msg = f"Model '{model_name}' is deprecated"
        if entry.replaced_by:
            msg += f", use '{entry.replaced_by}' instead"
        if entry.sunset_date:
            msg += f" (sunset: {entry.sunset_date})"
        return msg
    return None


def validate_config_models(config: dict) -> List[str]:
    """Check all models in config against deprecation registry. Returns warnings."""
    warnings_list = []
    models = config.get("models", {})
    if not isinstance(models, dict):
        return warnings_list
    for provider, provider_models in models.items():
        if isinstance(provider_models, dict):
            for profile, model_name in provider_models.items():
                if isinstance(model_name, str):
                    warning = check_model_deprecation(model_name)
                    if warning:
                        warnings_list.append(f"[{provider}.{profile}] {warning}")
    return warnings_list


def get_replacement(model_name: str) -> Optional[str]:
    """Get the recommended replacement for a deprecated model."""
    entry = KNOWN_MODELS.get(model_name)
    if entry and entry.replaced_by:
        return entry.replaced_by
    return None


def register_model(version: ModelVersion):
    """Register a custom model in the registry."""
    KNOWN_MODELS[version.name] = version


def list_deprecated_models() -> List[ModelVersion]:
    """List all deprecated models in the registry."""
    return [v for v in KNOWN_MODELS.values() if v.deprecated]
