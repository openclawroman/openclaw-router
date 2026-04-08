import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

from router import ExecutorResult, RoutingLogger, TaskMeta, route_task
from router.health import get_shutdown_manager


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ai_code_runner_accepts_tilde_config_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config_dir = home / ".openclaw" / "router" / "config"
    config_dir.mkdir(parents=True)
    source_config = REPO_ROOT / "config" / "router.config.json"
    target_config = config_dir / "router.config.json"
    target_config.write_text(source_config.read_text(), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "bin" / "ai-code-runner"),
            "--config",
            "~/.openclaw/router/config/router.config.json",
            "--health",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    parsed = json.loads(proc.stdout)
    assert isinstance(parsed, dict)


def test_invalid_cwd_returns_explicit_error():
    runtime_log = REPO_ROOT / "runtime" / "routing.jsonl"
    try:
        runtime_log.unlink()
    except FileNotFoundError:
        pass
    payload = {
        "protocol_version": 1,
        "task_id": "task-invalid-cwd",
        "task_meta": {
            "task_id": "task-invalid-cwd",
            "task_class": "implementation",
            "risk": "medium",
            "modality": "text",
            "repo_path": "/definitely/missing/repo",
            "cwd": "/definitely/missing/cwd",
            "summary": "test invalid cwd",
        },
        "prompt": "do work",
        "trace": {
            "bridge_request_id": "bridge-invalid",
            "prompt_sha256": "abc123",
        },
    }

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "bin" / "ai-code-runner")],
        cwd=REPO_ROOT,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    parsed = json.loads(proc.stdout)
    assert parsed["success"] is False
    assert parsed["normalized_error"] == "invalid_working_directory"
    assert parsed["trace_id"]
    try:
        runtime_log.unlink()
    except FileNotFoundError:
        pass


def test_routing_log_contains_trace_and_bridge_request_id(tmp_path, monkeypatch):
    monkeypatch.setattr("router.policy.AttemptLogger", lambda: MagicMock(log_trace=lambda trace: None))
    monkeypatch.setattr(
        "router.policy._run_executor",
        lambda entry, task, trace_id="": ExecutorResult(
            task_id=task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
            trace_id=trace_id,
        ),
    )

    task = TaskMeta(
        task_id="task-log",
        repo_path=str(tmp_path),
        cwd=str(tmp_path),
        summary="log task",
        bridge_request_id="bridge-123",
        scope_id="scope-1",
        thread_id="thread-1",
        session_id="session-1",
        cwd_source="cwd",
        cwd_exists=True,
    )

    decision, result = route_task(task)
    log_path = tmp_path / "routing.jsonl"
    RoutingLogger(log_path=log_path).log(task, decision, result, latency_ms=12)

    entry = json.loads(log_path.read_text().splitlines()[-1])
    assert entry["trace_id"] == decision.trace_id == result.trace_id
    assert entry["bridge_request_id"] == "bridge-123"
    assert entry["scope_id"] == "scope-1"
    assert entry["cwd_exists"] is True


def test_invalid_cwd_does_not_leave_inflight_task(monkeypatch):
    shutdown_mgr = get_shutdown_manager()
    before = shutdown_mgr.get_status()["in_flight_count"]

    monkeypatch.setattr("router.policy.AttemptLogger", lambda: MagicMock(log_trace=lambda trace: None))

    task = TaskMeta(
        task_id="task-invalid-bookkeeping",
        repo_path="/definitely/missing/repo",
        cwd="/definitely/missing/cwd",
        summary="invalid bookkeeping task",
        cwd_source="cwd",
        cwd_exists=False,
    )

    decision, result = route_task(task)
    after = shutdown_mgr.get_status()["in_flight_count"]

    assert result.normalized_error == "invalid_working_directory"
    assert decision.reason.endswith("invalid cwd")
    assert after == before
