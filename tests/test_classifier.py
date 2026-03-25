"""Tests for router.classifier — task class detection, risk/modality, repo path extraction, summary, classify_from_dict."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.classifier import (
    Classifier,
    classify,
    classify_from_dict,
    detect_task_class,
    detect_risk,
    detect_modality,
    _extract_repo_path,
    _generate_summary,
)
from router.models import TaskClass, TaskRisk, TaskModality


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def clf():
    return Classifier()


# ── Task class detection ─────────────────────────────────────────────────────

class TestDetectTaskClass:
    """Test keyword-based task class detection for all 10 classes."""

    @pytest.mark.parametrize("text", [
        "add test for login",
        "write tests for auth module",
        "generate test coverage",
    ])
    def test_test_generation(self, text):
        assert detect_task_class(text) == TaskClass.TEST_GENERATION

    @pytest.mark.parametrize("text", [
        "fix the null pointer bug",
        "hotfix production issue",
        "repair broken endpoint",
    ])
    def test_bugfix(self, text):
        assert detect_task_class(text) == TaskClass.BUGFIX

    @pytest.mark.parametrize("text", [
        "refactor the auth module",
        "restructure project layout",
        "reorganize into packages",
    ])
    def test_refactor(self, text):
        assert detect_task_class(text) == TaskClass.REFACTOR

    @pytest.mark.parametrize("text", [
        "diagnose the crash on startup",
        "investigate memory leak",
        "troubleshoot why CI is failing",
    ])
    def test_debug(self, text):
        assert detect_task_class(text) == TaskClass.DEBUG

    @pytest.mark.parametrize("text", [
        "review the pull request",
        "evaluate code quality",
        "assess the implementation",
    ])
    def test_code_review(self, text):
        assert detect_task_class(text) == TaskClass.CODE_REVIEW

    @pytest.mark.parametrize("text", [
        "implement user authentication",
        "build a REST API",
        "create a new endpoint",
    ])
    def test_implementation(self, text):
        assert detect_task_class(text) == TaskClass.IMPLEMENTATION

    @pytest.mark.parametrize("text", [
        "change the architecture to microservices",
        "architectural redesign needed",
    ])
    def test_repo_architecture_change(self, text):
        assert detect_task_class(text) == TaskClass.REPO_ARCHITECTURE_CHANGE

    @pytest.mark.parametrize("text", [
        "convert this screenshot to code",
        "image to code conversion",
    ])
    def test_ui_from_screenshot(self, text):
        assert detect_task_class(text) == TaskClass.UI_FROM_SCREENSHOT

    @pytest.mark.parametrize("text", [
        "multimodal input processing",
        "vision-based code task",
    ])
    def test_multimodal_code_task(self, text):
        assert detect_task_class(text) == TaskClass.MULTIMODAL_CODE_TASK

    @pytest.mark.parametrize("text", [
        "swarm of coding agents",
        "multi-agent coordination",
    ])
    def test_swarm_code_task(self, text):
        assert detect_task_class(text) == TaskClass.SWARM_CODE_TASK

    def test_no_keywords_defaults_to_implementation(self):
        assert detect_task_class("hello world") == TaskClass.IMPLEMENTATION

    def test_empty_string_defaults_to_implementation(self):
        assert detect_task_class("") == TaskClass.IMPLEMENTATION

    def test_test_generation_takes_priority_over_implementation(self):
        """Both 'write test' and 'build' appear, but test_generation is checked first."""
        assert detect_task_class("write test and build feature") == TaskClass.TEST_GENERATION


# ── Risk detection ───────────────────────────────────────────────────────────

class TestDetectRisk:
    def test_critical(self):
        assert detect_risk("critical production issue") == TaskRisk.CRITICAL

    def test_high(self):
        assert detect_risk("architecture migration needed") == TaskRisk.HIGH

    def test_medium(self):
        assert detect_risk("add a new feature") == TaskRisk.MEDIUM

    def test_low(self):
        assert detect_risk("fix docs typo") == TaskRisk.LOW

    def test_no_keywords_defaults_to_medium(self):
        assert detect_risk("hello world") == TaskRisk.MEDIUM


# ── Modality detection ──────────────────────────────────────────────────────

class TestDetectModality:
    def test_image(self):
        assert detect_modality("attach screenshot of UI") == TaskModality.IMAGE

    def test_video(self):
        assert detect_modality("screen recording shows the bug") == TaskModality.VIDEO

    def test_mixed(self):
        assert detect_modality("multimodal input with text and code") == TaskModality.MIXED

    def test_default_text(self):
        assert detect_modality("hello world") == TaskModality.TEXT


# ── Repo path extraction ────────────────────────────────────────────────────

class TestExtractRepoPath:
    def test_users_path(self):
        path = _extract_repo_path("Work on /Users/alice/projects/my-app")
        assert path == "/Users/alice/projects/my-app"

    def test_tilde_path(self):
        path = _extract_repo_path("Check ~/code/repo")
        assert path == "~/code/repo"

    def test_org_repo_path(self):
        path = _extract_repo_path("Clone /org/repo/src/main.py")
        assert path == "/org/repo/src/main.py"

    def test_no_path(self):
        assert _extract_repo_path("just some text") is None


# ── Summary generation ──────────────────────────────────────────────────────

class TestGenerateSummary:
    def test_strips_leading_keyword(self):
        assert _generate_summary("Implement user auth") == "user auth"

    def test_strips_whitespace(self):
        assert _generate_summary("fix   the   bug") == "the bug"

    def test_truncates_to_120_chars(self):
        long_text = "build " + "a" * 200
        result = _generate_summary(long_text)
        assert len(result) <= 120

    def test_empty_string(self):
        assert _generate_summary("") == ""


# ── Classifier.classify ─────────────────────────────────────────────────────

class TestClassify:
    """Test the full Classifier.classify method."""

    def test_returns_task_meta(self, clf):
        meta = clf.classify("add test for login")
        assert meta.task_class == TaskClass.TEST_GENERATION
        assert meta.risk == TaskRisk.MEDIUM
        assert meta.modality == TaskModality.TEXT

    def test_repo_path_extracted(self, clf):
        meta = clf.classify("fix bug in /Users/alice/projects/my-app")
        assert meta.repo_path == "/Users/alice/projects/my-app"
        assert meta.cwd == "/Users/alice/projects/my-app"

    def test_no_repo_path(self, clf):
        meta = clf.classify("just a normal task")
        assert meta.repo_path == ""
        assert meta.cwd == ""

    def test_summary_generated(self, clf):
        meta = clf.classify("implement user authentication")
        assert meta.summary == "user authentication"

    def test_task_id_is_8_chars(self, clf):
        meta = clf.classify("hello")
        assert len(meta.task_id) == 8

    def test_agent_defaults_to_coder(self, clf):
        meta = clf.classify("hello")
        assert meta.agent == "coder"

    def test_swarm_detection(self, clf):
        meta = clf.classify("swarm of agents working together")
        assert meta.swarm is True

    def test_no_swarm(self, clf):
        meta = clf.classify("regular task")
        assert meta.swarm is False

    def test_has_screenshots_detection(self, clf):
        meta = clf.classify("convert this screenshot to code")
        assert meta.has_screenshots is True

    def test_no_screenshots(self, clf):
        meta = clf.classify("regular task")
        assert meta.has_screenshots is False


# ── requires_repo_write ─────────────────────────────────────────────────────

class TestRequiresRepoWrite:
    def test_implementation_requires_write(self, clf):
        meta = clf.classify("implement new feature")
        assert meta.requires_repo_write is True

    def test_bugfix_requires_write(self, clf):
        meta = clf.classify("fix the bug")
        assert meta.requires_repo_write is True

    def test_refactor_requires_write(self, clf):
        meta = clf.classify("refactor the module")
        assert meta.requires_repo_write is True

    def test_code_review_no_write(self, clf):
        meta = clf.classify("review the code")
        assert meta.requires_repo_write is False

    def test_debug_no_write(self, clf):
        meta = clf.classify("diagnose the crash")
        assert meta.requires_repo_write is False


# ── requires_multimodal / has_screenshots ────────────────────────────────────

class TestRequiresMultimodal:
    def test_ui_from_screenshot_requires_multimodal(self, clf):
        meta = clf.classify("screenshot to code")
        assert meta.requires_multimodal is True
        assert meta.has_screenshots is True

    def test_multimodal_task_requires_multimodal(self, clf):
        meta = clf.classify("multimodal input processing")
        assert meta.requires_multimodal is True

    def test_implementation_no_multimodal(self, clf):
        meta = clf.classify("implement feature")
        assert meta.requires_multimodal is False
        assert meta.has_screenshots is False


# ── classify_from_dict ──────────────────────────────────────────────────────

class TestClassifyFromDict:
    def test_basic_dict_classification(self, clf):
        meta = clf.classify_from_dict({"summary": "fix the login bug"})
        assert meta.task_class == TaskClass.BUGFIX

    def test_override_task_id(self, clf):
        meta = clf.classify_from_dict({"task_id": "custom_01", "summary": "hello"})
        assert meta.task_id == "custom_01"

    def test_override_repo_path(self, clf):
        meta = clf.classify_from_dict({"repo_path": "/opt/myrepo", "summary": "hello"})
        assert meta.repo_path == "/opt/myrepo"
        assert meta.cwd == "/opt/myrepo"

    def test_override_risk(self, clf):
        meta = clf.classify_from_dict({"risk": "critical", "summary": "hello"})
        assert meta.risk == TaskRisk.CRITICAL

    def test_override_agent(self, clf):
        meta = clf.classify_from_dict({"agent": "reviewer", "summary": "hello"})
        assert meta.agent == "reviewer"

    def test_override_task_class(self, clf):
        meta = clf.classify_from_dict({"task_class": "debug", "summary": "hello"})
        assert meta.task_class == TaskClass.DEBUG

    def test_override_requires_repo_write(self, clf):
        meta = clf.classify_from_dict({"requires_repo_write": False, "summary": "implement feature"})
        assert meta.requires_repo_write is False

    def test_override_modality(self, clf):
        meta = clf.classify_from_dict({"modality": "image", "summary": "hello"})
        assert meta.modality == TaskModality.IMAGE

    def test_override_has_screenshots(self, clf):
        meta = clf.classify_from_dict({"has_screenshots": True, "summary": "hello"})
        assert meta.has_screenshots is True

    def test_override_swarm(self, clf):
        meta = clf.classify_from_dict({"swarm": True, "summary": "hello"})
        assert meta.swarm is True

    def test_override_requires_multimodal(self, clf):
        meta = clf.classify_from_dict({"requires_multimodal": True, "summary": "hello"})
        assert meta.requires_multimodal is True

    def test_description_key_fallback(self, clf):
        meta = clf.classify_from_dict({"description": "fix the bug"})
        assert meta.task_class == TaskClass.BUGFIX

    def test_task_brief_key_fallback(self, clf):
        meta = clf.classify_from_dict({"task_brief": "review code"})
        assert meta.task_class == TaskClass.CODE_REVIEW

    def test_task_key_fallback(self, clf):
        meta = clf.classify_from_dict({"task": "diagnose the issue"})
        assert meta.task_class == TaskClass.DEBUG

    def test_text_key_fallback(self, clf):
        meta = clf.classify_from_dict({"text": "refactor module"})
        assert meta.task_class == TaskClass.REFACTOR

    def test_empty_dict_defaults(self, clf):
        meta = clf.classify_from_dict({})
        assert meta.task_class == TaskClass.IMPLEMENTATION
        assert meta.risk == TaskRisk.MEDIUM
        assert meta.summary == ""


# ── Module-level convenience functions ──────────────────────────────────────

class TestModuleLevelFunctions:
    def test_classify_function(self):
        meta = classify("add test for auth")
        assert meta.task_class == TaskClass.TEST_GENERATION

    def test_classify_from_dict_function(self):
        meta = classify_from_dict({"summary": "fix bug", "task_id": "abc"})
        assert meta.task_id == "abc"
        assert meta.task_class == TaskClass.BUGFIX


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
