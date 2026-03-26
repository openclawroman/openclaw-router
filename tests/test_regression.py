"""Regression tests for classifier, policy, fallback, and normalize_error."""

import pytest

from router.classifier import classify, detect_task_class, detect_risk, detect_modality, classify_from_dict, _extract_repo_path
from router.policy import build_chain, can_fallback, resolve_state, _build_normal_chain, _build_last10_chain
from router.errors import normalize_error
from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState, ChainEntry, RouteDecision, ExecutorResult


class TestClassifierRegression:
    @pytest.mark.parametrize("text,expected", [
        ("implement a parser", TaskClass.IMPLEMENTATION),
        ("build the dashboard", TaskClass.IMPLEMENTATION),
        ("fix the null pointer", TaskClass.BUGFIX),
        ("repair broken link", TaskClass.BUGFIX),
        ("refactor the auth module", TaskClass.REFACTOR),
        ("restructure the codebase", TaskClass.REFACTOR),
        ("diagnose the error", TaskClass.DEBUG),
        ("investigate memory leak", TaskClass.DEBUG),
        ("troubleshoot the issue", TaskClass.DEBUG),
        ("review the PR", TaskClass.CODE_REVIEW),
        ("evaluate the changes", TaskClass.CODE_REVIEW),
        ("add test for parser", TaskClass.TEST_GENERATION),
        ("write test for auth", TaskClass.TEST_GENERATION),
        ("update the architecture", TaskClass.REPO_ARCHITECTURE_CHANGE),
        ("architectural redesign needed", TaskClass.REPO_ARCHITECTURE_CHANGE),
        ("screenshot of the UI", TaskClass.UI_FROM_SCREENSHOT),
        ("ui from this design", TaskClass.UI_FROM_SCREENSHOT),
        ("multimodal analysis", TaskClass.MULTIMODAL_CODE_TASK),
        ("vision model integration", TaskClass.MULTIMODAL_CODE_TASK),
        ("swarm of agents", TaskClass.SWARM_CODE_TASK),
        ("multi-agent coordination", TaskClass.SWARM_CODE_TASK),
    ])
    def test_task_class_keywords(self, text, expected):
        assert detect_task_class(text) == expected

    def test_empty_text_classifies(self):
        result = classify("")
        assert isinstance(result, TaskMeta)
        assert result.task_class == TaskClass.IMPLEMENTATION
        assert result.risk == TaskRisk.MEDIUM

    def test_multiple_keywords(self):
        result = classify("fix and add test for bug")
        assert result.task_class == TaskClass.TEST_GENERATION

    @pytest.mark.parametrize("text,risk", [
        ("production deployment", TaskRisk.HIGH),
        ("security patch", TaskRisk.HIGH),
        ("p0 incident", TaskRisk.HIGH),
        ("update docs", TaskRisk.LOW),
        ("fix readme typo", TaskRisk.LOW),
    ])
    def test_risk_detection(self, text, risk):
        assert detect_risk(text) in {TaskRisk.HIGH, TaskRisk.CRITICAL} if risk in (TaskRisk.HIGH, TaskRisk.CRITICAL) else detect_risk(text) == risk

    @pytest.mark.parametrize("text,expected", [
        ("screenshot of the app", TaskModality.IMAGE),
        ("image for the landing page", TaskModality.IMAGE),
        ("video walkthrough", TaskModality.VIDEO),
        ("screen recording demo", TaskModality.VIDEO),
        ("multimodal input", TaskModality.MIXED),
        ("mixed media content", TaskModality.MIXED),
    ])
    def test_modality_detection(self, text, expected):
        assert detect_modality(text) == expected

    def test_repo_path_extraction(self):
        result = _extract_repo_path("check /Users/test/repo for issues")
        assert result is not None
        assert "/Users/test/repo" in result

    def test_classify_from_dict(self):
        raw = {"task_id": "test-123", "description": "fix the login bug", "repo_path": "/home/user/myrepo", "risk": "high", "agent": "reviewer"}
        result = classify_from_dict(raw)
        assert isinstance(result, TaskMeta)
        assert result.task_id == "test-123"
        assert result.task_class == TaskClass.BUGFIX
        assert result.risk == TaskRisk.HIGH
        assert result.agent == "reviewer"


class TestPolicyRegression:
    def _make_task(self, **kw):
        defaults = dict(task_id="test-task", task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, modality=TaskModality.TEXT)
        defaults.update(kw)
        return TaskMeta(**defaults)

    def test_normal_chain_codex_first(self):
        chain = build_chain(self._make_task(), CodexState.NORMAL)
        assert chain[0].tool == "codex_cli"
        assert chain[0].backend == "openai_native"

    def test_last10_claude_first(self):
        chain = build_chain(self._make_task(), CodexState.LAST10)
        assert chain[0].tool == "claude_code"
        assert chain[0].backend == "anthropic"

    def test_last10_skips_codex(self):
        chain = build_chain(self._make_task(), CodexState.LAST10)
        for entry in chain:
            assert not (entry.tool == "codex_cli" and entry.backend == "openai_native")

    def test_kimi_for_screenshots(self):
        chain = build_chain(self._make_task(has_screenshots=True), CodexState.NORMAL)
        openrouter = [e for e in chain if e.backend == "openrouter"]
        assert openrouter[-1].model_profile == "openrouter_kimi"

    def test_minimax_for_normal(self):
        chain = build_chain(self._make_task(), CodexState.NORMAL)
        openrouter = [e for e in chain if e.backend == "openrouter"]
        assert openrouter[-1].model_profile == "openrouter_minimax"

    def test_chain_entry_format(self):
        task = self._make_task()
        for state in [CodexState.NORMAL, CodexState.LAST10]:
            for entry in build_chain(task, state):
                assert isinstance(entry, ChainEntry)
                assert entry.tool
                assert entry.backend
                assert entry.model_profile


class TestFallbackRegression:
    @pytest.mark.parametrize("error_type,expected", [
        ("auth_error", True), ("rate_limited", True), ("quota_exhausted", True),
        ("provider_unavailable", True), ("provider_timeout", True), ("transient_network_error", True),
        ("invalid_payload", False), ("missing_repo_path", False), ("toolchain_error", False),
        ("git_conflict", False), ("permission_denied_local", False),
        (None, False), ("random_error", False),
    ])
    def test_fallback_eligibility(self, error_type, expected):
        assert can_fallback(error_type) is expected


class TestNormalizeErrorRegression:
    @pytest.mark.parametrize("msg,expected", [
        ("authentication failed", "auth_error"), ("unauthorized", "auth_error"),
        ("401 unauthorized", "auth_error"), ("auth token expired", "auth_error"),
        ("rate limit exceeded", "rate_limited"), ("too many requests", "rate_limited"),
        ("429 rate limited", "rate_limited"), ("throttled", "rate_limited"),
        ("quota exceeded", "quota_exhausted"), ("monthly limit reached", "quota_exhausted"),
        ("usage exceeded", "quota_exhausted"),
        ("request timeout", "provider_timeout"), ("connection timed out", "provider_timeout"),
        ("504 gateway timeout", "provider_timeout"),
        ("service unavailable", "provider_unavailable"), ("503 unavailable", "provider_unavailable"),
        ("500 internal server error", "provider_unavailable"),
        ("network error", "transient_network_error"), ("connection refused", "transient_network_error"),
        ("dns resolution failed", "transient_network_error"),
        ("something went wrong completely", "unknown_error"), ("", "unknown_error"),
    ])
    def test_error_normalization(self, msg, expected):
        assert normalize_error(msg) == expected
