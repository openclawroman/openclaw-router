"""Tests for TaskPhase detection and integration."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.classifier import Classifier, classify, classify_from_dict, detect_phase
from router.models import TaskClass, TaskModality, TaskPhase


@pytest.fixture
def clf():
    return Classifier()


class TestDetectPhase:
    def test_plan_description_is_decide(self):
        phase = detect_phase(
            "Plan the architecture of the system",
            TaskClass.IMPLEMENTATION,
            TaskModality.TEXT,
            False, False,
        )
        assert phase == TaskPhase.DECIDE

    def test_final_review_class_is_decide(self):
        phase = detect_phase(
            "Review the generated code",
            TaskClass.FINAL_REVIEW,
            TaskModality.TEXT,
            False, False,
        )
        assert phase == TaskPhase.DECIDE

    def test_architecture_change_class_is_decide(self):
        phase = detect_phase(
            "Change the project structure",
            TaskClass.REPO_ARCHITECTURE_CHANGE,
            TaskModality.TEXT,
            False, False,
        )
        assert phase == TaskPhase.DECIDE

    def test_planner_class_is_decide(self):
        phase = detect_phase(
            "Break down the task",
            TaskClass.PLANNER,
            TaskModality.TEXT,
            False, False,
        )
        assert phase == TaskPhase.DECIDE

    def test_fix_bug_is_execute(self):
        phase = detect_phase(
            "Fix this bug",
            TaskClass.BUGFIX,
            TaskModality.TEXT,
            False, False,
        )
        assert phase == TaskPhase.EXECUTE

    def test_implement_feature_is_execute(self):
        phase = detect_phase(
            "Implement user authentication",
            TaskClass.IMPLEMENTATION,
            TaskModality.TEXT,
            False, False,
        )
        assert phase == TaskPhase.EXECUTE

    def test_screenshot_is_visual(self):
        phase = detect_phase(
            "Convert this screenshot to code",
            TaskClass.UI_FROM_SCREENSHOT,
            TaskModality.IMAGE,
            True, True,
        )
        assert phase == TaskPhase.VISUAL

    def test_requires_multimodal_is_visual(self):
        phase = detect_phase(
            "Process this image",
            TaskClass.IMPLEMENTATION,
            TaskModality.TEXT,
            False, True,
        )
        assert phase == TaskPhase.VISUAL

    def test_mixed_modality_is_visual(self):
        phase = detect_phase(
            "Handle mixed input",
            TaskClass.IMPLEMENTATION,
            TaskModality.MIXED,
            False, False,
        )
        assert phase == TaskPhase.VISUAL

    def test_triage_keyword_is_decide(self):
        phase = detect_phase(
            "Triage the reported issues",
            TaskClass.IMPLEMENTATION,
            TaskModality.TEXT,
            False, False,
        )
        assert phase == TaskPhase.DECIDE

    def test_architect_keyword_is_decide(self):
        phase = detect_phase(
            "Architect a new service layer",
            TaskClass.IMPLEMENTATION,
            TaskModality.TEXT,
            False, False,
        )
        assert phase == TaskPhase.DECIDE


class TestClassifyPhase:
    def test_phase_in_classify_output(self, clf):
        meta = clf.classify("Fix this bug")
        assert hasattr(meta, "phase")
        assert meta.phase == TaskPhase.EXECUTE

    def test_plan_phase_is_decide(self, clf):
        meta = clf.classify("Plan the architecture of the new module")
        assert meta.phase == TaskPhase.DECIDE

    def test_review_implementation_is_decide(self, clf):
        meta = clf.classify("Review the implementation for correctness")
        assert meta.phase == TaskPhase.DECIDE

    def test_screenshot_task_is_visual(self, clf):
        meta = clf.classify("Convert this screenshot to React code")
        assert meta.phase == TaskPhase.VISUAL

    def test_default_is_execute(self, clf):
        meta = clf.classify("Build a REST API endpoint")
        assert meta.phase == TaskPhase.EXECUTE


class TestClassifyFromDictPhase:
    def test_explicit_phase_override(self, clf):
        meta = clf.classify_from_dict({"summary": "fix bug", "phase": "decide"})
        assert meta.phase == TaskPhase.DECIDE

    def test_explicit_phase_visual(self, clf):
        meta = clf.classify_from_dict({"summary": "implement feature", "phase": "visual"})
        assert meta.phase == TaskPhase.VISUAL

    def test_auto_detected_phase(self, clf):
        meta = clf.classify_from_dict({"summary": "Plan the architecture"})
        assert meta.phase == TaskPhase.DECIDE

    def test_auto_detected_visual(self, clf):
        meta = clf.classify_from_dict({"summary": "screenshot to code"})
        assert meta.phase == TaskPhase.VISUAL


class TestModuleLevelPhase:
    def test_classify_returns_phase(self):
        meta = classify("Fix this null pointer bug")
        assert meta.phase == TaskPhase.EXECUTE

    def test_classify_from_dict_returns_phase(self):
        meta = classify_from_dict({"summary": "Plan the new architecture"})
        assert meta.phase == TaskPhase.DECIDE
