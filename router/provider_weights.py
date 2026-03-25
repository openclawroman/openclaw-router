"""Provider weight system for cost/reliability-aware routing."""

import json
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class ProviderWeight:
    """Weight configuration for a single provider."""
    name: str                          # "codex_cli:openai_native", etc.
    cost_per_1k_tokens: float = 0.0   # USD cost per 1K tokens
    reliability_score: float = 1.0    # 0.0-1.0, 1.0 = perfect
    speed_score: float = 1.0          # 0.0-1.0, 1.0 = fastest
    priority: int = 0                 # Manual priority override (higher = more preferred)
    
    # Default weights by provider
    DEFAULT_WEIGHTS = {
        "codex_cli:openai_native": {"cost_per_1k_tokens": 0.015, "reliability_score": 0.95, "speed_score": 0.8, "priority": 10},
        "claude_code:anthropic":   {"cost_per_1k_tokens": 0.015, "reliability_score": 0.90, "speed_score": 0.7, "priority": 8},
        "codex_cli:openrouter":    {"cost_per_1k_tokens": 0.005, "reliability_score": 0.85, "speed_score": 0.9, "priority": 6},
    }
    
    @property
    def composite_score(self) -> float:
        """Composite score: higher = better.
        
        Weighted formula:
          composite = (reliability * 0.4) + (speed * 0.2) + (cost_efficiency * 0.2) + (priority * 0.2)
        
        Cost efficiency: inverse of cost (lower cost = higher efficiency)
        """
        # Normalize cost (lower is better, max assumed 0.10 USD/1K)
        cost_efficiency = 1.0 - min(self.cost_per_1k_tokens / 0.10, 1.0)
        
        return (
            self.reliability_score * 0.4 +
            self.speed_score * 0.2 +
            cost_efficiency * 0.2 +
            (self.priority / 10.0) * 0.2
        )


class ProviderWeightManager:
    """Manages provider weights and ranking."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize with optional config override.
        
        Config format:
        {
            "codex_cli:openai_native": {"cost_per_1k_tokens": 0.015, ...},
            ...
        }
        """
        self._weights: Dict[str, ProviderWeight] = {}
        self._load_defaults()
        if config:
            self._apply_config(config)
    
    def _load_defaults(self) -> None:
        """Load default weights from ProviderWeight.DEFAULT_WEIGHTS."""
        for name, params in ProviderWeight.DEFAULT_WEIGHTS.items():
            self._weights[name] = ProviderWeight(name=name, **params)
    
    def _apply_config(self, config: dict) -> None:
        """Apply config overrides."""
        for name, params in config.items():
            if name in self._weights:
                existing = self._weights[name]
                if "cost_per_1k_tokens" in params:
                    existing.cost_per_1k_tokens = params["cost_per_1k_tokens"]
                if "reliability_score" in params:
                    existing.reliability_score = params["reliability_score"]
                if "speed_score" in params:
                    existing.speed_score = params["speed_score"]
                if "priority" in params:
                    existing.priority = params["priority"]
            else:
                self._weights[name] = ProviderWeight(name=name, **params)
    
    def get_weight(self, provider_name: str) -> Optional[ProviderWeight]:
        """Get weight for a provider."""
        return self._weights.get(provider_name)
    
    def get_ranked_providers(self, available: Optional[List[str]] = None) -> List[ProviderWeight]:
        """Get providers ranked by composite score (best first).
        
        Args:
            available: If provided, only include these providers.
        """
        providers = list(self._weights.values())
        if available is not None:
            providers = [p for p in providers if p.name in available]
        providers.sort(key=lambda p: p.composite_score, reverse=True)
        return providers
    
    def update_reliability(self, provider_name: str, success: bool, alpha: float = 0.1) -> None:
        """Update reliability score using exponential moving average.
        
        Args:
            alpha: Learning rate (0.0-1.0). Higher = more weight on recent.
        """
        pw = self._weights.get(provider_name)
        if pw is None:
            return
        
        # EMA: new = (1-alpha) * old + alpha * observation
        observation = 1.0 if success else 0.0
        pw.reliability_score = (1 - alpha) * pw.reliability_score + alpha * observation
    
    def to_dict(self) -> dict:
        """Serialize all weights."""
        return {name: asdict(pw) for name, pw in self._weights.items()}
    
    def get_provider_summary(self) -> List[dict]:
        """Get summary of all providers with scores."""
        summary = []
        for name, pw in sorted(self._weights.items(), key=lambda x: x[1].composite_score, reverse=True):
            summary.append({
                "provider": name,
                "cost_per_1k": pw.cost_per_1k_tokens,
                "reliability": round(pw.reliability_score, 3),
                "speed": round(pw.speed_score, 3),
                "priority": pw.priority,
                "composite": round(pw.composite_score, 3),
            })
        return summary
