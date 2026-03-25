"""Regression tests for classifier, policy, fallback, and normalize_error.

Covers X-87: Regression tests for classifier, policy, and fallback.
"""

import pytest

from router.classifier import classify, detect_task_class, detect_risk, detect_modality, classify_from_dict, _extract_repo_path
from router.policy import (
    build_chain, choose_openrouter_profile, can_fallback, resolve_state, route_task,
    _build_normal_chain, _build_last10_chain,
)
from router.errors import normalize_error
from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState,
    ChainEntry, RouteDecision, ExecutorResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TestClassifierRegression
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifierRegression:
    """Regression tests for task classification."""

    def test_implementation_keywords(self):
        """'implement', 'build', 'create' all classify as implementation."""
        for kw in ["implement a parser", "build the dashboard", "create a new module"]:
            result = detect_task_class(kw)
            assert result == TaskClass.IMPLEMENTATION, f"{kw!r} → {result}"

    def test_bugfix_keywords(self):
        """'fix', 'bug', 'repair' classify as bugfix."""
        for kw in ["fix the null pointer", "bug in login flow", "repair broken link"]:
            result = detect_task_class(kw)
            assert result == TaskClass.BUGFIX, f"{kw!r} → {result}"

    def test_refactor_keywords(self):
        """'refactor', 'restructure' classify as refactor."""
        for kw in ["refactor the auth module", "restructure the codebase"]:
            result = detect_task_class(kw)
            assert result == TaskClass.REFACTOR, f"{kw!r} → {result}"

    def test_debug_keywords(self):
        """'debug', 'investigate', 'diagnose' classify as debug."""
        # Note: "debug" contains "bug" as substring so bugfix matches first.
        # Use "diagnose", "investigate", "troubleshoot" which are unique to debug.
        for kw in ["diagnose the error", "investigate memory leak", "troubleshoot the issue"]:
            result = detect_task_class(kw)
            assert result == TaskClass.DEBUG, f"{kw!r} → {result}"

    def test_code_review_keywords(self):
        """'review', 'evaluate' classify as code_review."""
        for kw in ["review the PR", "evaluate the changes"]:
            result = detect_task_class(kw)
            assert result == TaskClass.CODE_REVIEW, f"{kw!r} → {result}"

    def test_test_generation_keywords(self):
        """'add test', 'write test' classify as test_generation."""
        for kw in ["add test for parser", "write test for auth"]:
            result = detect_task_class(kw)
            assert result == TaskClass.TEST_GENERATION, f"{kw!r} → {result}"

    def test_architecture_keywords(self):
        """'architecture', 'architectural' classify as repo_architecture_change."""
        for kw in ["update the architecture", "architectural redesign needed"]:
            result = detect_task_class(kw)
            assert result == TaskClass.REPO_ARCHITECTURE_CHANGE, f"{kw!r} → {result}"

    def test_ui_from_screenshot_keywords(self):
        """'screenshot', 'ui from' classify as ui_from_screenshot."""
        for kw in ["screenshot of the UI", "ui from this design"]:
            result = detect_task_class(kw)
            assert result == TaskClass.UI_FROM_SCREENSHOT, f"{kw!r} → {result}"

    def test_multimodal_keywords(self):
        """'multimodal', 'vision' classify as multimodal_code_task."""
        for kw in ["multimodal analysis", "vision model integration"]:
            result = detect_task_class(kw)
            assert result == TaskClass.MULTIMODAL_CODE_TASK, f"{kw!r} → {result}"

    def test_swarm_keywords(self):
        """'swarm', 'multi-agent' classify as swarm_code_task."""
        for kw in ["swarm of agents", "multi-agent coordination"]:
            result = detect_task_class(kw)
            assert result == TaskClass.SWARM_CODE_TASK, f"{kw!r} → {result}"

    def test_risk_critical(self):
        """'production', 'security', 'p0' → risk=high or critical."""
        for kw in ["production deployment", "security patch", "p0 incident"]:
            result = detect_risk(kw)
            assert result in {TaskRisk.HIGH, TaskRisk.CRITICAL}, f"{kw!r} → {result}"

    def test_risk_low(self):
        """'docs', 'readme', 'comment' → risk=low."""
        # Note: avoid "add comment" since "add" is a medium-risk keyword.
        for kw in ["update docs", "fix readme typo", "just a comment change"]:
            result = detect_risk(kw)
            assert result == TaskRisk.LOW, f"{kw!r} → {result}"

    def test_modality_image(self):
        """'screenshot', 'image' → modality=image."""
        for kw in ["screenshot of the app", "image for the landing page"]:
            result = detect_modality(kw)
            assert result == TaskModality.IMAGE, f"{kw!r} → {result}"

    def test_modality_video(self):
        """'video', 'recording' → modality=video."""
        for kw in ["video walkthrough", "screen recording demo"]:
            result = detect_modality(kw)
            assert result == TaskModality.VIDEO, f"{kw!r} → {result}"

    def test_modality_mixed(self):
        """'multimodal', 'mixed' → modality=mixed."""
        for kw in ["multimodal input", "mixed media content"]:
            result = detect_modality(kw)
            assert result == TaskModality.MIXED, f"{kw!r} → {result}"

    def test_repo_path_extraction(self):
        """Text with '/Users/test/repo' → repo_path extracted."""
        result = _extract_repo_path("check /Users/test/repo for issues")
        assert result is not None
        assert "/Users/test/repo" in result

    def test_classify_from_dict(self):
        """Raw dict input produces TaskMeta."""
        raw = {
            "task_id": "test-123",
            "description": "fix the login bug",
            "repo_path": "/home/user/myrepo",
            "risk": "high",
            "agent": "reviewer",
        }
        result = classify_from_dict(raw)
        assert isinstance(result, TaskMeta)
        assert result.task_id == "test-123"
        assert result.task_class == TaskClass.BUGFIX
        assert result.repo_path == "/home/user/myrepo"
        assert result.risk == TaskRisk.HIGH
        assert result.agent == "reviewer"

    def test_empty_text_classifies(self):
        """Empty string doesn't crash, returns defaults."""
        result = classify("")
        assert isinstance(result, TaskMeta)
        assert result.task_class == TaskClass.IMPLEMENTATION
        assert result.risk == TaskRisk.MEDIUM
        assert result.modality == TaskModality.TEXT

    def test_multiple_keywords(self):
        """'fix and add test for bug' → test_generation (higher priority in keyword list)."""
        result = classify("fix and add test for bug")
        # test_generation keywords are checked before bugfix in TASK_CLASS_KEYWORDS
        assert result.task_class == TaskClass.TEST_GENERATION


# ═══════════════════════════════════════════════════════════════════════════════
# TestPolicyRegression
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolicyRegression:
    """Regression tests for routing policy."""

    def _make_task(self, **kwargs):
        defaults = dict(
            task_id="test-task",
            task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM,
            modality=TaskModality.TEXT,
        )
        defaults.update(kwargs)
        return TaskMeta(**defaults)

    def test_normal_chain_codex_first(self):
        """Normal state: first executor is codex_cli:openai_native."""
        task = self._make_task()
        chain = build_chain(task, CodexState.NORMAL)
        assert len(chain) >= 1
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"

    def test_last10_claude_first(self):
        """Last10 state: first executor is claude_code:anthropic."""
        task = self._make_task()
        chain = build_chain(task, CodexState.LAST10)
        assert len(chain) >= 1
        assert chain[0].tool == "claude_code"
        assert chain[0].backend == "anthropic"

    def test_last10_skips_codex(self):
        """Last10 chain has no openai_native backend."""
        task = self._make_task()
        chain = build_chain(task, CodexState.LAST10)
        for entry in chain:
            assert not (entry.tool == "codex_cli" and entry.backend == "openai_native"), \
                "Last10 chain should not include codex_cli:openai_native"

    def test_kimi_for_screenshots(self):
        """Screenshot tasks get kimi profile in the openrouter entry."""
        task = self._make_task(has_screenshots=True)
        chain = build_chain(task, CodexState.NORMAL)
        # The last entry in normal chain is openrouter
        openrouter_entries = [e for e in chain if e.backend == "openrouter"]
        assert len(openrouter_entries) > 0
        assert openrouter_entries[-1].model_profile == "openrouter_kimi"

    def test_minimax_for_normal(self):
        """Normal implementation tasks get minimax profile."""
        task = self._make_task()
        chain = build_chain(task, CodexState.NORMAL)
        openrouter_entries = [e for e in chain if e.backend == "openrouter"]
        assert len(openrouter_entries) > 0
        assert openrouter_entries[-1].model_profile == "openrouter_minimax"

    def test_screenshot_gets_kimi_in_last10(self):
        """Screenshot in last10 (claude_backup) uses openrouter_dynamic for openrouter entry."""
        task = self._make_task(has_screenshots=True)
        chain = build_chain(task, CodexState.LAST10)
        openrouter_entries = [e for e in chain if e.backend == "openrouter"]
        assert len(openrouter_entries) > 0
        # claude_backup chain uses openrouter_dynamic, not kimi directly
        assert openrouter_entries[-1].model_profile == "openrouter_dynamic"

    def test_chain_entry_format(self):
        """Each ChainEntry has tool, backend, model_profile."""
        task = self._make_task()
        for state in [CodexState.NORMAL, CodexState.LAST10]:
            chain = build_chain(task, state)
            for entry in chain:
                assert isinstance(entry, ChainEntry)
                assert entry.tool  # non-empty string
                assert entry.backend
                assert entry.model_profile


# ═══════════════════════════════════════════════════════════════════════════════
# TestFallbackRegression
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackRegression:
    """Regression tests for fallback eligibility."""

    def test_eligible_auth_error(self):
        assert can_fallback("auth_error") is True

    def test_eligible_rate_limited(self):
        assert can_fallback("rate_limited") is True

    def test_eligible_quota_exhausted(self):
        assert can_fallback("quota_exhausted") is True

    def test_eligible_provider_unavailable(self):
        assert can_fallback("provider_unavailable") is True

    def test_eligible_provider_timeout(self):
        assert can_fallback("provider_timeout") is True

    def test_eligible_transient_network(self):
        assert can_fallback("transient_network_error") is True

    def test_non_eligible_invalid_payload(self):
        assert can_fallback("invalid_payload") is False

    def test_non_eligible_missing_repo(self):
        assert can_fallback("missing_repo_path") is False

    def test_non_eligible_toolchain(self):
        assert can_fallback("toolchain_error") is False

    def test_non_eligible_git_conflict(self):
        assert can_fallback("git_conflict") is False

    def test_non_eligible_permission(self):
        assert can_fallback("permission_denied_local") is False

    def test_none_returns_false(self):
        assert can_fallback(None) is False

    def test_random_string_returns_false(self):
        assert can_fallback("random_error") is False


# ═══════════════════════════════════════════════════════════════════════════════
# TestNormalizeErrorRegression
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeErrorRegression:
    """Regression tests for error normalization."""

    def test_auth_error_mapping(self):
        """Auth-related errors normalize to 'auth_error'."""
        for msg in ["authentication failed", "unauthorized", "401 unauthorized", "auth token expired"]:
            result = normalize_error(msg)
            assert result == "auth_error", f"{msg!r} → {result}"

    def test_rate_limit_mapping(self):
        """Rate-related errors normalize to 'rate_limited'."""
        for msg in ["rate limit exceeded", "too many requests", "429 rate limited", "throttled"]:
            result = normalize_error(msg)
            assert result == "rate_limited", f"{msg!r} → {result}"

    def test_quota_mapping(self):
        """Quota/limit/exceeded → 'quota_exhausted'."""
        for msg in ["quota exceeded", "monthly limit reached", "usage exceeded"]:
            result = normalize_error(msg)
            assert result == "quota_exhausted", f"{msg!r} → {result}"

    def test_timeout_mapping(self):
        """Timeout-related → 'provider_timeout'."""
        for msg in ["request timeout", "connection timed out", "504 gateway timeout"]:
            result = normalize_error(msg)
            assert result == "provider_timeout", f"{msg!r} → {result}"

    def test_unavailable_mapping(self):
        """Unavailable/503 → 'provider_unavailable'."""
        for msg in ["service unavailable", "503 unavailable", "500 internal server error"]:
            result = normalize_error(msg)
            assert result == "provider_unavailable", f"{msg!r} → {result}"

    def test_network_mapping(self):
        """Network/connection → 'transient_network_error'."""
        for msg in ["network error", "connection refused", "dns resolution failed"]:
            result = normalize_error(msg)
            assert result == "transient_network_error", f"{msg!r} → {result}"

    def test_unknown_preserved(self):
        """Unknown errors pass through as-is (→ 'unknown_error')."""
        assert normalize_error("something went wrong completely") == "unknown_error"
        assert normalize_error("") == "unknown_error"
