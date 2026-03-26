"""Tests for router.classifier — task class detection, risk/modality, repo path extraction, summary."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.classifier import Classifier, classify, classify_from_dict, detect_task_class, detect_risk, detect_modality, _extract_repo_path, _generate_summary
from router.models import TaskClass, TaskRisk, TaskModality


@pytest.fixture
def clf():
    return Classifier()


class TestDetectTaskClass:
    @pytest.mark.parametrize("text,expected", [
        ("add test for login", TaskClass.TEST_GENERATION),
        ("write tests for auth module", TaskClass.TEST_GENERATION),
        ("fix the null pointer bug", TaskClass.BUGFIX),
        ("hotfix production issue", TaskClass.BUGFIX),
        ("refactor the auth module", TaskClass.REFACTOR),
        ("restructure project layout", TaskClass.REFACTOR),
        ("diagnose the crash on startup", TaskClass.DEBUG),
        ("investigate memory leak", TaskClass.DEBUG),
        ("review the pull request", TaskClass.CODE_REVIEW),
        ("evaluate code quality", TaskClass.CODE_REVIEW),
        ("implement user authentication", TaskClass.IMPLEMENTATION),
        ("build a REST API", TaskClass.IMPLEMENTATION),
        ("change the architecture to microservices", TaskClass.REPO_ARCHITECTURE_CHANGE),
        ("convert this screenshot to code", TaskClass.UI_FROM_SCREENSHOT),
        ("multimodal input processing", TaskClass.MULTIMODAL_CODE_TASK),
        ("swarm of coding agents", TaskClass.SWARM_CODE_TASK),
        ("hello world", TaskClass.IMPLEMENTATION),
        ("", TaskClass.IMPLEMENTATION),
    ])
    def test_task_class_detection(self, text, expected):
        assert detect_task_class(text) == expected

    def test_test_generation_priority_over_implementation(self):
        assert detect_task_class("write test and build feature") == TaskClass.TEST_GENERATION


class TestDetectRisk:
    @pytest.mark.parametrize("text,expected", [
        ("critical production issue", TaskRisk.CRITICAL),
        ("architecture migration needed", TaskRisk.HIGH),
        ("add a new feature", TaskRisk.MEDIUM),
        ("fix docs typo", TaskRisk.LOW),
        ("hello world", TaskRisk.MEDIUM),
    ])
    def test_risk_detection(self, text, expected):
        assert detect_risk(text) == expected


class TestDetectModality:
    @pytest.mark.parametrize("text,expected", [
        ("attach screenshot of UI", TaskModality.IMAGE),
        ("screen recording shows the bug", TaskModality.VIDEO),
        ("multimodal input with text and code", TaskModality.MIXED),
        ("hello world", TaskModality.TEXT),
    ])
    def test_modality_detection(self, text, expected):
        assert detect_modality(text) == expected


class TestExtractRepoPath:
    @pytest.mark.parametrize("text,expected_substring", [
        ("Work on /Users/alice/projects/my-app", "/Users/alice/projects/my-app"),
        ("Check ~/code/repo", "~/code/repo"),
        ("Clone /org/repo/src/main.py", "/org/repo/src/main.py"),
    ])
    def test_path_extraction(self, text, expected_substring):
        result = _extract_repo_path(text)
        assert result is not None
        assert expected_substring in result

    def test_no_path(self):
        assert _extract_repo_path("just some text") is None


class TestGenerateSummary:
    @pytest.mark.parametrize("text,expected", [
        ("Implement user auth", "user auth"),
        ("fix   the   bug", "the bug"),
        ("", ""),
    ])
    def test_summary_generation(self, text, expected):
        assert _generate_summary(text) == expected

    def test_truncates_to_120_chars(self):
        result = _generate_summary("build " + "a" * 200)
        assert len(result) <= 120


class TestClassify:
    def test_returns_task_meta(self, clf):
        meta = clf.classify("add test for login")
        assert meta.task_class == TaskClass.TEST_GENERATION
        assert meta.risk == TaskRisk.MEDIUM
        assert meta.modality == TaskModality.TEXT

    def test_repo_path_extracted(self, clf):
        meta = clf.classify("fix bug in /Users/alice/projects/my-app")
        assert meta.repo_path == "/Users/alice/projects/my-app"

    def test_no_repo_path(self, clf):
        meta = clf.classify("just a normal task")
        assert meta.repo_path == ""

    def test_summary_generated(self, clf):
        meta = clf.classify("implement user authentication")
        assert meta.summary == "user authentication"

    def test_task_id_is_8_chars(self, clf):
        assert len(clf.classify("hello").task_id) == 8

    def test_agent_defaults_to_coder(self, clf):
        assert clf.classify("hello").agent == "coder"

    @pytest.mark.parametrize("text,swarm,screenshots", [
        ("swarm of agents working together", True, False),
        ("regular task", False, False),
        ("convert this screenshot to code", False, True),
    ])
    def test_flag_detection(self, clf, text, swarm, screenshots):
        meta = clf.classify(text)
        assert meta.swarm is swarm
        assert meta.has_screenshots is screenshots


class TestRequiresRepoWrite:
    @pytest.mark.parametrize("text,expected", [
        ("implement new feature", True),
        ("fix the bug", True),
        ("refactor the module", True),
        ("review the code", False),
        ("diagnose the crash", False),
    ])
    def test_requires_repo_write(self, clf, text, expected):
        assert clf.classify(text).requires_repo_write is expected


class TestRequiresMultimodal:
    @pytest.mark.parametrize("text,multimodal", [
        ("screenshot to code", True),
        ("multimodal input processing", True),
        ("implement feature", False),
    ])
    def test_requires_multimodal(self, clf, text, multimodal):
        meta = clf.classify(text)
        assert meta.requires_multimodal is multimodal


class TestClassifyFromDict:
    @pytest.mark.parametrize("input_dict,key,expected", [
        ({"summary": "fix the login bug"}, "task_class", TaskClass.BUGFIX),
        ({"task_id": "custom_01", "summary": "hello"}, "task_id", "custom_01"),
        ({"repo_path": "/opt/myrepo", "summary": "hello"}, "repo_path", "/opt/myrepo"),
        ({"risk": "critical", "summary": "hello"}, "risk", TaskRisk.CRITICAL),
        ({"agent": "reviewer", "summary": "hello"}, "agent", "reviewer"),
        ({"task_class": "debug", "summary": "hello"}, "task_class", TaskClass.DEBUG),
        ({"requires_repo_write": False, "summary": "implement feature"}, "requires_repo_write", False),
        ({"modality": "image", "summary": "hello"}, "modality", TaskModality.IMAGE),
        ({"has_screenshots": True, "summary": "hello"}, "has_screenshots", True),
        ({"swarm": True, "summary": "hello"}, "swarm", True),
        ({"requires_multimodal": True, "summary": "hello"}, "requires_multimodal", True),
    ])
    def test_override_fields(self, clf, input_dict, key, expected):
        meta = clf.classify_from_dict(input_dict)
        assert getattr(meta, key) == expected

    @pytest.mark.parametrize("input_dict,expected_class", [
        ({"description": "fix the bug"}, TaskClass.BUGFIX),
        ({"task_brief": "review code"}, TaskClass.CODE_REVIEW),
        ({"task": "diagnose the issue"}, TaskClass.DEBUG),
        ({"text": "refactor module"}, TaskClass.REFACTOR),
        ({}, TaskClass.IMPLEMENTATION),
    ])
    def test_key_fallbacks(self, clf, input_dict, expected_class):
        assert clf.classify_from_dict(input_dict).task_class == expected_class


class TestModuleLevelFunctions:
    def test_classify_function(self):
        assert classify("add test for auth").task_class == TaskClass.TEST_GENERATION

    def test_classify_from_dict_function(self):
        meta = classify_from_dict({"summary": "fix bug", "task_id": "abc"})
        assert meta.task_id == "abc"
        assert meta.task_class == TaskClass.BUGFIX
