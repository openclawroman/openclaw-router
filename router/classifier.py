"""Task classifier for ai-code-runner.

Classifies incoming tasks into task_class, determines risk/modality,
and enriches with metadata for routing decisions.
"""

import uuid
import re
from typing import Optional

from .models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    TaskCriticality
)


TASK_CLASS_KEYWORDS = [
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
    # Match common path patterns
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
    # Strip common prefixes
    text = re.sub(r'^(implement|build|add|create|fix|refactor|debug|review)\s+', '', text.lower())
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
        
        return TaskMeta(
            task_id=str(uuid.uuid4())[:8],
            agent="coder",
            task_class=detect_task_class(description),
            task_brief=description,  # keep full description as brief
            repo_path=repo_path or "",
            branch="main",
            risk=detect_risk(description),
            criticality=TaskCriticality.NORMAL,
            context_size="medium",
            modality=detect_modality(description),
            requires_multimodal=detect_task_class(description) in {
                TaskClass.UI_FROM_SCREENSHOT,
                TaskClass.MULTIMODAL_CODE_TASK,
                TaskClass.SWARM_CODE_TASK,
            },
            has_screenshots="screenshot" in description.lower(),
            swarm="swarm" in description.lower(),
            cwd=repo_path or "",
        )

    def classify_from_dict(self, raw: dict) -> TaskMeta:
        """
        Enrich a raw task dict.

        Expected ``raw`` shape::

            {
                "task_id": "optional",
                "agent": "coder",
                "task_brief": "...",  # primary field for classification
                "repo_path": "...",   # optional
                "branch": "main",     # optional
                "risk": "medium",     # optional
                "criticality": "normal",  # optional
                "context_size": "medium", # optional
            }

        Returns TaskMeta (call .to_dict() to merge back).
        """
        brief = (
            raw.get("task_brief")
            or raw.get("description")
            or raw.get("task")
            or raw.get("text")
            or ""
        )
        meta = self.classify(brief)

        # Override with explicit values from raw dict if present
        if "task_id" in raw:
            meta.task_id = raw["task_id"]
        if "repo_path" in raw:
            meta.repo_path = raw["repo_path"]
            meta.cwd = raw["repo_path"]
        if "branch" in raw:
            meta.branch = raw["branch"]
        if "risk" in raw:
            meta.risk = TaskRisk(raw["risk"])
        if "criticality" in raw:
            meta.criticality = TaskCriticality(raw["criticality"])
        if "context_size" in raw:
            meta.context_size = raw["context_size"]
        if "agent" in raw:
            meta.agent = raw["agent"]

        return meta


# Default classifier instance for module-level convenience
_default_classifier = Classifier()


def classify(description: str) -> TaskMeta:
    """Module-level classify function."""
    return _default_classifier.classify(description)


def classify_from_dict(raw: dict) -> TaskMeta:
    """Module-level classify_from_dict function."""
    return _default_classifier.classify_from_dict(raw)
