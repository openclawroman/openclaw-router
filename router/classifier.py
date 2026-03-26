"""Task classifier for ai-code-runner.

Classifies incoming tasks into task_class, determines risk/modality,
and enriches with metadata for routing decisions.
"""

import uuid
import re
from typing import Optional

from .models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
)


# MiMo-as-brain design philosophy:
# The classifier uses MiMo for routing decisions (it runs locally, cheap).
# MiMo is the brain — it classifies and routes, but never executes tasks.
# It decides WHERE work goes, not HOW work is done.

# Order matters: more specific patterns must come before generic ones.
# PLANNER before IMPLEMENTATION (so "plan to implement" matches planner, not implementation).
# FINAL_REVIEW before CODE_REVIEW (so "review the generated code" matches final_review, not code_review).
TASK_CLASS_KEYWORDS = [
    ("planner",              ["plan", "think about", "how should we", "break down"]),
    ("final_review",         ["review the generated code", "check the implementation", "validate", "verify the result"]),
    ("test_generation",      ["add test", "add tests", "write test", "write tests", "generate test", "test for"]),
    ("bugfix",               ["fix", "fixing", "bug", "repair", "hotfix"]),
    ("refactor",             ["refactor", "restructure", "reorganize", "clean up"]),
    ("debug",                ["debug", "investigate", "diagnose", "troubleshoot"]),
    ("code_review",          ["review", "evaluate", "assess", "check"]),
    ("implementation",       ["implement", "build", "create", "add", "new feature"]),
    ("repo_architecture_change", ["architecture", "architectural", "structure change", "system redesign"]),
    ("ui_from_screenshot",  ["screenshot", "image to code", "ui from", "design image"]),
    ("multimodal_code_task", ["multimodal", "vision", "image to code"]),
    ("swarm_code_task",      ["swarm", "multi-agent", "coordinated"]),
]


RISK_KEYWORDS = {
    "critical": ["critical", "production", "data loss", "security", "breach", "p0", "p1"],
    "high":     ["architecture", "migration", "database", "api break", "breaking change", "refactor core"],
    "medium":   ["feature", "implement", "add", "create", "build"],
    "low":      ["docs", "documentation", "readme", "comment", "typo", "format"],
}


MODALITY_KEYWORDS = {
    TaskModality.IMAGE:  ["screenshot", "image", "picture", "design", "ui mockup", "figma"],
    TaskModality.VIDEO:  ["video", "recording", "screen record"],
    TaskModality.MIXED:  ["multimodal", "mixed"],
}


def detect_task_class(text: str) -> TaskClass:
    """Detect task class from text using keyword matching."""
    text_lower = text.lower()
    for task_class, keywords in TASK_CLASS_KEYWORDS:
        for kw in keywords:
            if kw in text_lower:
                return TaskClass(task_class)
    return TaskClass.IMPLEMENTATION


def detect_risk(text: str) -> TaskRisk:
    """Detect risk level from text."""
    text_lower = text.lower()
    for risk, keywords in RISK_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return TaskRisk(risk)
    return TaskRisk.MEDIUM


def detect_modality(text: str) -> TaskModality:
    """Detect modality from text."""
    text_lower = text.lower()
    for modality, keywords in MODALITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return modality
    return TaskModality.TEXT


def _extract_repo_path(text: str) -> Optional[str]:
    """Extract repo path from text if present."""
    patterns = [
        r'/[a-zA-Z0-9_./-]+/(?:ops-hub|projects?|repo|src|app)/[a-zA-Z0-9_./-]+',
        r'~/[a-zA-Z0-9_./-]+',
        r'/Users/[a-zA-Z0-9_./-]+',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(0)
    return None


def _generate_summary(text: str) -> str:
    """Generate a short summary from task text."""
    text = re.sub(r'^(plan|design|architect|implement|build|add|create|fix|refactor|debug|review|validate|verify)\s+', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:120]


class Classifier:
    """Main classifier for task metadata enrichment."""

    def classify(self, description: str) -> TaskMeta:
        """
        Classify a task description and produce enriched TaskMeta.

        Args:
            description: Free-text task description

        Returns:
            TaskMeta with all fields enriched
        """
        repo_path = _extract_repo_path(description)
        task_class = detect_task_class(description)

        return TaskMeta(
            task_id=str(uuid.uuid4())[:8],
            agent="coder",
            task_class=task_class,
            risk=detect_risk(description),
            modality=detect_modality(description),
            requires_repo_write=task_class not in {TaskClass.CODE_REVIEW, TaskClass.DEBUG, TaskClass.PLANNER, TaskClass.FINAL_REVIEW},
            requires_multimodal=task_class in {
                TaskClass.UI_FROM_SCREENSHOT,
                TaskClass.MULTIMODAL_CODE_TASK,
            },
            has_screenshots="screenshot" in description.lower(),
            swarm="swarm" in description.lower(),
            repo_path=repo_path or "",
            cwd=repo_path or "",
            summary=_generate_summary(description),
        )

    def classify_from_dict(self, raw: dict) -> TaskMeta:
        """
        Enrich a raw task dict.

        Returns TaskMeta with defaults filled in.
        """
        summary = (
            raw.get("summary")
            or raw.get("description")
            or raw.get("task_brief")
            or raw.get("task")
            or raw.get("text")
            or ""
        )
        meta = self.classify(summary)

        # Override with explicit values from raw dict if present
        if "task_id" in raw:
            meta.task_id = raw["task_id"]
        if "repo_path" in raw:
            meta.repo_path = raw["repo_path"]
        if "cwd" in raw:
            meta.cwd = raw["cwd"]
        elif "repo_path" in raw:
            meta.cwd = raw["repo_path"]
        if "risk" in raw:
            meta.risk = TaskRisk(raw["risk"])
        if "agent" in raw:
            meta.agent = raw["agent"]
        if "task_class" in raw:
            meta.task_class = TaskClass(raw["task_class"])
        if "requires_repo_write" in raw:
            meta.requires_repo_write = bool(raw["requires_repo_write"])
        if "requires_multimodal" in raw:
            meta.requires_multimodal = bool(raw["requires_multimodal"])
        if "has_screenshots" in raw:
            meta.has_screenshots = bool(raw["has_screenshots"])
        if "swarm" in raw:
            meta.swarm = bool(raw["swarm"])
        if "modality" in raw:
            meta.modality = TaskModality(raw["modality"])

        return meta


# Default classifier instance for module-level convenience
_default_classifier = Classifier()


def classify(description: str) -> TaskMeta:
    """Module-level classify function."""
    return _default_classifier.classify(description)


def classify_from_dict(raw: dict) -> TaskMeta:
    """Module-level classify_from_dict function."""
    return _default_classifier.classify_from_dict(raw)
