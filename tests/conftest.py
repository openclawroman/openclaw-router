"""Shared fixtures for integration tests."""

import json
import time
import uuid
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    RouteDecision, ExecutorResult, ChainEntry, CodexState,
)
from router.state_store import StateStore
from router.policy import route_task, build_chain, resolve_state, reset_breaker, reset_notifier


# ---------------------------------------------------------------------------
# Mock executor framework
# ---------------------------------------------------------------------------

class MockExecutor:
    """Realistic mock executor — configurable delays, outputs, failures, costs."""

    def __init__(self, *, delay_ms: int = 150, should_fail: bool = False,
                 error_type: str = "provider_unavailable",
                 cost_usd: float = 0.0023,
                 output_summary: str = "Task completed successfully",
                 artifacts: list[str] | None = None,
                 timeout_after_ms: int = 0):
        self.delay_ms = delay_ms
        self.should_fail = should_fail
        self.error_type = error_type
        self.cost_usd = cost_usd
        self.output_summary = output_summary
        self.artifacts = artifacts or []
        self.timeout_after_ms = timeout_after_ms  # 0 = no timeout
        self.call_count = 0
        self._lock = threading.Lock()

    def __call__(self, meta: TaskMeta, **kwargs) -> ExecutorResult:
        self.call_count += 1
        start = time.monotonic()

        if self.timeout_after_ms > 0:
            time.sleep(self.timeout_after_ms / 1000.0)
            from router.errors import ProviderTimeoutError
            raise ProviderTimeoutError("Execution timed out")

        time.sleep(self.delay_ms / 1000.0)
        latency = int((time.monotonic() - start) * 1000)

        if self.should_fail:
            return ExecutorResult(
                task_id=meta.task_id,
                tool="mock",
                backend="mock",
                model_profile="mock_default",
                success=False,
                normalized_error=self.error_type,
                exit_code=1,
                latency_ms=latency,
                cost_estimate_usd=0.0,
                final_summary=f"Error: {self.error_type}",
            )

        return ExecutorResult(
            task_id=meta.task_id,
            tool="mock",
            backend="mock",
            model_profile="mock_default",
            success=True,
            normalized_error=None,
            exit_code=0,
            latency_ms=latency,
            cost_estimate_usd=self.cost_usd,
            artifacts=self.artifacts,
            final_summary=self.output_summary,
        )


class MockExecutorChain:
    """A sequence of mock executors — one per chain position. Each call consumes the next."""

    def __init__(self, executors: list[MockExecutor]):
        self.executors = executors
        self._idx = 0
        self._lock = threading.Lock()

    def __call__(self, meta: TaskMeta, **kwargs) -> ExecutorResult:
        with self._lock:
            idx = min(self._idx, len(self.executors) - 1)
            self._idx += 1
        return self.executors[idx](meta, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_state_dir(tmp_path):
    """Provide a temp directory for state files."""
    return tmp_path / "state"


@pytest.fixture
def temp_runtime_dir(tmp_path):
    """Provide a temp directory for runtime files (logs, alerts)."""
    return tmp_path / "runtime"


@pytest.fixture
def default_config():
    """Default config dict for integration tests."""
    return {
        "models": {
            "openai": {"gpt4": "gpt-4", "codex": "codex-mini"},
            "claude": {"sonnet": "claude-sonnet-4-20250514"},
        },
        "tools": {
            "codex": {"profiles": {"default": "codex-mini"}},
            "claude": {"profiles": {"default": "claude-sonnet-4-20250514"}},
        },
        "routing": {
            "policy": "openai_primary",
            "fallback_chain": ["openai", "claude", "openrouter"],
        },
        "timeouts": {"openai": 30, "claude": 60, "openrouter": 30},
    }


@pytest.fixture
def state_store(temp_state_dir):
    """StateStore backed by temp directory."""
    store = StateStore(
        manual_path=temp_state_dir / "manual_state.json",
        auto_path=temp_state_dir / "auto_state.json",
        history_path=temp_state_dir / "state_history.json",
        wal_path=temp_state_dir / "state_wal.jsonl",
    )
    return store


@pytest.fixture
def sample_task():
    """A standard medium-risk implementation task."""
    return TaskMeta(
        task_id=str(uuid.uuid4())[:8],
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT,
        requires_repo_write=True,
        requires_multimodal=False,
        has_screenshots=False,
        swarm=False,
        repo_path="/tmp/test-repo",
        cwd="/tmp/test-repo",
        summary="Implement feature X with unit tests",
    )


@pytest.fixture
def critical_task():
    """A high-risk critical task."""
    return TaskMeta(
        task_id=str(uuid.uuid4())[:8],
        agent="coder",
        task_class=TaskClass.REPO_ARCHITECTURE_CHANGE,
        risk=TaskRisk.CRITICAL,
        modality=TaskModality.TEXT,
        requires_repo_write=True,
        requires_multimodal=False,
        has_screenshots=False,
        swarm=False,
        repo_path="/tmp/test-repo",
        cwd="/tmp/test-repo",
        summary="Redesign authentication system architecture",
    )


@pytest.fixture
def happy_executor():
    """Mock executor that always succeeds with realistic output."""
    return MockExecutor(
        delay_ms=200,
        cost_usd=0.0023,
        output_summary="Feature implemented: 3 files changed, 142 insertions(+)",
        artifacts=["/tmp/task-001.stdout.txt"],
    )


@pytest.fixture
def failing_executor():
    """Mock executor that always fails with provider_unavailable."""
    return MockExecutor(
        delay_ms=100,
        should_fail=True,
        error_type="provider_unavailable",
    )


@pytest.fixture
def timeout_executor():
    """Mock executor that times out (raises ProviderTimeoutError)."""
    return MockExecutor(
        delay_ms=0,
        timeout_after_ms=50,
    )


def patch_all_executors(happy: MockExecutor, failing: MockExecutor | None = None,
                         timeout: MockExecutor | None = None):
    """Return a dict of patches for all three executor entry points."""
    patches = {}
    patches["codex"] = patch("router.policy.run_codex", side_effect=happy)
    patches["claude"] = patch("router.policy.run_claude", side_effect=happy)
    patches["openrouter"] = patch("router.policy.run_openrouter", side_effect=happy)
    if failing:
        patches["codex_f"] = patch("router.policy.run_codex", side_effect=failing)
        patches["claude_f"] = patch("router.policy.run_claude", side_effect=failing)
        patches["openrouter_f"] = patch("router.policy.run_openrouter", side_effect=failing)
    if timeout:
        patches["codex_t"] = patch("router.policy.run_codex", side_effect=timeout)
        patches["claude_t"] = patch("router.policy.run_claude", side_effect=timeout)
        patches["openrouter_t"] = patch("router.policy.run_openrouter", side_effect=timeout)
    return patches


@pytest.fixture
def patched_routing(tmp_path, monkeypatch):
    """Set up isolated routing environment with temp runtime dir.

    Patches state store, logger paths, and executors so no real I/O happens.
    Returns a dict with 'runtime_dir' and 'state_store'.
    """
    runtime_dir = tmp_path / "runtime"
    state_dir = tmp_path / "state"
    runtime_dir.mkdir()
    state_dir.mkdir()

    # Patch logger runtime dir
    monkeypatch.setattr("router.logger.RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", runtime_dir / "routing.jsonl")

    # Patch notification alerts path
    monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", runtime_dir / "alerts.jsonl")

    # Patch state store paths
    monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
    monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
    monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
    monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")

    # Reset singletons
    from router.state_store import reset_state_store
    reset_state_store()
    reset_breaker()
    reset_notifier()

    store = StateStore(
        manual_path=state_dir / "manual.json",
        auto_path=state_dir / "auto.json",
        history_path=state_dir / "history.json",
        wal_path=state_dir / "wal.jsonl",
    )

    return {
        "runtime_dir": runtime_dir,
        "state_dir": state_dir,
        "state_store": store,
    }
