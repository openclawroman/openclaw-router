"""Tests for provider weight system."""

import pytest
from router.provider_weights import ProviderWeight, ProviderWeightManager


class TestProviderWeight:
    def test_default_weights(self):
        pw = ProviderWeight(name="test")
        assert pw.name == "test"
        assert pw.reliability_score == 1.0
        assert pw.composite_score > 0

    def test_composite_score_high_reliability(self):
        pw_high = ProviderWeight(name="high", reliability_score=1.0, cost_per_1k_tokens=0.01)
        pw_low = ProviderWeight(name="low", reliability_score=0.5, cost_per_1k_tokens=0.01)
        assert pw_high.composite_score > pw_low.composite_score

    def test_composite_score_low_cost(self):
        pw_cheap = ProviderWeight(name="cheap", cost_per_1k_tokens=0.001, reliability_score=0.8)
        pw_expensive = ProviderWeight(name="expensive", cost_per_1k_tokens=0.05, reliability_score=0.8)
        assert pw_cheap.composite_score > pw_expensive.composite_score

    def test_default_weights_exist(self):
        assert "codex_cli:openai_native" in ProviderWeight.DEFAULT_WEIGHTS
        assert "claude_code:anthropic" in ProviderWeight.DEFAULT_WEIGHTS
        assert "codex_cli:openrouter" in ProviderWeight.DEFAULT_WEIGHTS


class TestProviderWeightManager:
    def test_loads_defaults(self):
        mgr = ProviderWeightManager()
        assert mgr.get_weight("codex_cli:openai_native") is not None
        assert mgr.get_weight("claude_code:anthropic") is not None

    def test_get_ranked_providers(self):
        mgr = ProviderWeightManager()
        ranked = mgr.get_ranked_providers()
        assert len(ranked) == 3
        # Best should be first
        assert ranked[0].composite_score >= ranked[1].composite_score
        assert ranked[1].composite_score >= ranked[2].composite_score

    def test_get_ranked_providers_filtered(self):
        mgr = ProviderWeightManager()
        ranked = mgr.get_ranked_providers(available=["codex_cli:openai_native", "codex_cli:openrouter"])
        assert len(ranked) == 2
        names = [p.name for p in ranked]
        assert "codex_cli:openai_native" in names
        assert "claude_code:anthropic" not in names

    def test_update_reliability_success(self):
        mgr = ProviderWeightManager()
        pw = mgr.get_weight("claude_code:anthropic")
        original = pw.reliability_score
        
        # Repeated failures should lower score
        for _ in range(50):
            mgr.update_reliability("claude_code:anthropic", success=False, alpha=0.1)
        
        assert pw.reliability_score < original

    def test_update_reliability_unknown_provider(self):
        mgr = ProviderWeightManager()
        # Should not raise
        mgr.update_reliability("nonexistent:provider", success=True)

    def test_config_override(self):
        config = {"codex_cli:openai_native": {"priority": 100}}
        mgr = ProviderWeightManager(config=config)
        pw = mgr.get_weight("codex_cli:openai_native")
        assert pw.priority == 100

    def test_config_new_provider(self):
        config = {"custom:provider": {"cost_per_1k_tokens": 0.01, "reliability_score": 0.9}}
        mgr = ProviderWeightManager(config=config)
        assert mgr.get_weight("custom:provider") is not None

    def test_to_dict(self):
        mgr = ProviderWeightManager()
        d = mgr.to_dict()
        assert "codex_cli:openai_native" in d
        assert "cost_per_1k_tokens" in d["codex_cli:openai_native"]

    def test_get_provider_summary(self):
        mgr = ProviderWeightManager()
        summary = mgr.get_provider_summary()
        assert len(summary) == 3
        assert "provider" in summary[0]
        assert "composite" in summary[0]
        # Should be sorted by composite score
        scores = [s["composite"] for s in summary]
        assert scores == sorted(scores, reverse=True)
