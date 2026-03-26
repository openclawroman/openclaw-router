"""Tests for full envelope parsing from router-bridge."""
import json
import pytest
from bin.ai_code_runner import parse_task_meta
from router.models import TaskClass, TaskRisk, TaskModality


def test_parse_task_meta_extracts_core_fields():
    data = {
        "task_id": "task_001",
        "task_meta": {
            "task_id": "task_001",
            "task_class": "implementation",
            "risk": "medium",
            "modality": "text",
        },
        "prompt": "Write a function",
    }
    task = parse_task_meta(data)
    assert task.task_id == "task_001"
    assert task.task_class == TaskClass.IMPLEMENTATION
    assert task.risk == TaskRisk.MEDIUM


def test_parse_task_meta_extracts_cwd_from_root():
    data = {
        "task_meta": {"task_id": "t2", "task_class": "implementation"},
        "prompt": "test",
        "cwd": "/home/user/project",
    }
    task = parse_task_meta(data)
    assert task.cwd == "/home/user/project"


def test_parse_task_meta_extracts_cwd_from_context():
    data = {
        "task_meta": {"task_id": "t3", "task_class": "implementation"},
        "prompt": "test",
        "context": {"working_directory": "/home/user/repo"},
    }
    task = parse_task_meta(data)
    assert task.cwd == "/home/user/repo"


def test_parse_task_meta_falls_back_to_prompt_for_summary():
    data = {
        "task_meta": {"task_id": "t4", "task_class": "implementation"},
        "prompt": "Write a sorting algorithm in Python",
    }
    task = parse_task_meta(data)
    assert task.summary == "Write a sorting algorithm in Python"


def test_parse_task_meta_respects_existing_summary():
    data = {
        "task_meta": {"task_id": "t5", "task_class": "implementation", "summary": "Custom summary"},
        "prompt": "Different prompt",
    }
    task = parse_task_meta(data)
    assert task.summary == "Custom summary"


def test_parse_task_meta_cwd_priority():
    """cwd field should take precedence over context.working_directory"""
    data = {
        "task_meta": {"task_id": "t6", "task_class": "implementation"},
        "prompt": "test",
        "cwd": "/explicit/cwd",
        "context": {"working_directory": "/context/cwd"},
    }
    task = parse_task_meta(data)
    assert task.cwd == "/explicit/cwd"
