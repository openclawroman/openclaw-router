"""End-to-end happy-path tests for the router.

Covers the COMPLETE flow from task input through routing, execution, and result.
All tests use realistic mock executors — no real I/O or network calls.
"""

import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    RouteDecision, ExecutorResult, ChainEntry, CodexState,
)
from router.state_store import reset_state_store
from router.policy import route_task, reset_breaker, reset_notifier
from router.config_loader import get_config_snapshot
from router.classifier import classify, classify_from_dict

from tests.conftest import MockExecutor, MockExecutorChain


# ── Helpers ──────────────────────────────────────────────────────────────────

DEFAULT_CHAIN = [
    ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary"),
    ChainEntry(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
    ChainEntry(tool="codex_cli", backend="openrouter", model_profile="openrouter_dynamic"),
]


def _patch_env(tmp_path, monkeypatch):
    state_dir, runtime_dir = tmp_path / "state", tmp_path / "runtime"
    state_dir.mkdir()
    runtime_dir.mkdir()
    monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
    monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
    monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
    monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")
    monkeypatch.setattr("router.logger.RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", runtime_dir / "routing.jsonl")
    monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", runtime_dir / "alerts.jsonl")
    reset_state_store()
    reset_breaker()
    reset_notifier()


def _patch_router(monkeypatch, chain=None, executor_result=None):
    """Mock resolve_state, build_chain, and optionally _run_executor."""
    monkeypatch.setattr("router.policy.resolve_state", lambda: CodexState.OPENAI_PRIMARY)
    monkeypatch.setattr("router.policy.build_chain", lambda task, state: chain or DEFAULT_CHAIN)
    monkeypatch.setattr("router.policy.get_reliability_config",
                        lambda: {"chain_timeout_s": 600, "max_fallbacks": 3})
    breaker = MagicMock()
    breaker.is_available.return_value = True
    monkeypatch.setattr("router.policy.get_breaker", lambda: breaker)
    monkeypatch.setattr("router.policy.get_notifier", lambda: MagicMock())
    monkeypatch.setattr("router.policy.AttemptLogger", lambda: MagicMock())

    shutdown_mgr = MagicMock(
        should_accept_new_tasks=MagicMock(return_value=True),
        register_task=MagicMock(), unregister_task=MagicMock(),
    )
    monkeypatch.setattr("router.health.get_shutdown_manager", lambda: shutdown_mgr)

    if executor_result is not None:
        monkeypatch.setattr("router.policy._run_executor",
                            _make_run_executor(executor_result))
    return shutdown_mgr


def _make_result(task_id, *, success=True, tool="codex_cli", backend="openai_native",
                 model_profile="codex_primary", cost_usd=0.0023, latency_ms=150,
                 summary="Task completed", artifacts=None, error=None):
    return ExecutorResult(
        task_id=task_id, tool=tool, backend=backend, model_profile=model_profile,
        success=success, normalized_error=error,
        exit_code=0 if success else 1,
        latency_ms=latency_ms, cost_estimate_usd=cost_usd if success else 0.0,
        artifacts=artifacts or [], final_summary=summary if success else None,
    )


def _make_run_executor(result_fn):
    def _run_executor(entry, task, trace_id=""):
        result = result_fn(task.task_id)
        result.trace_id = trace_id
        return result
    return _run_executor


def _make_task(task_id="test", **kw):
    return TaskMeta(
        task_id=task_id, agent="coder",
        task_class=TaskClass.IMPLEMENTATION, risk=kw.get("risk", TaskRisk.MEDIUM),
        modality=kw.get("modality", TaskModality.TEXT),
        repo_path=kw.get("repo_path", "/tmp/repo"),
        cwd=kw.get("cwd", "/tmp/repo"),
        summary=kw.get("summary", "Test task"),
    )


# ── Full Pipeline (1 test) ──────────────────────────────────────────────────

class TestConfigLoadToResult:

    def test_full_pipeline(self, tmp_path, monkeypatch):
        _patch_env(tmp_path, monkeypatch)
        task = _make_task("cfg-001", summary="Implement feature X with tests")
        result_fn = lambda tid: _make_result(tid, summary="3 files changed, 42 insertions(+)")
        _patch_router(monkeypatch, executor_result=result_fn)
        decision, result = route_task(task)
        assert decision.task_id == "cfg-001"
        assert decision.state in {s.value for s in CodexState}
        assert len(decision.chain) >= 1
        assert result.success is True
        assert result.cost_estimate_usd > 0
        assert result.latency_ms > 0


# ── Task Classification & Routing (3 tests) ──────────────────────────────────

class TestTaskClassification:

    def test_implement_classifies_and_routes(self, tmp_path, monkeypatch):
        meta = classify("Implement user authentication with JWT tokens")
        assert meta.task_class == TaskClass.IMPLEMENTATION
        meta.repo_path = "/tmp/test-repo"
        artifacts = ["/tmp/task-implement.stdout.txt", "/tmp/task-implement.stderr.txt"]
        _patch_env(tmp_path, monkeypatch)
        result_fn = lambda tid: _make_result(tid, tool="codex_cli", backend="openai_native",
                                              model_profile="codex_primary", artifacts=artifacts,
                                              summary="Implemented JWT auth: 5 files changed, 180 insertions(+)")
        _patch_router(monkeypatch, executor_result=result_fn)
        decision, result = route_task(meta)
        assert result.success is True
        assert result.artifacts == artifacts
        assert len(decision.chain) >= 1

    def test_bugfix_classifies_and_routes(self, tmp_path, monkeypatch):
        meta = classify("Fix null pointer exception in login handler")
        assert meta.task_class == TaskClass.BUGFIX
        meta.repo_path = "/tmp/test-repo"
        _patch_env(tmp_path, monkeypatch)
        result_fn = lambda tid: _make_result(tid, summary="Fixed: 1 file changed, 3 insertions(+), 2 deletions(-)")
        _patch_router(monkeypatch, executor_result=result_fn)
        _, result = route_task(meta)
        assert result.success is True
        assert "Fixed" in result.final_summary

    def test_classify_from_dict_and_route(self, tmp_path, monkeypatch):
        meta = classify_from_dict({"summary": "Build a REST API for inventory management", "repo_path": "/tmp/repo"})
        assert meta.task_class == TaskClass.IMPLEMENTATION
        meta2 = classify_from_dict({"task_id": "dict-001", "summary": "Build a REST API", "repo_path": "/tmp/test-repo"})
        _patch_env(tmp_path, monkeypatch)
        result_fn = lambda tid: _make_result(tid, summary="REST API created: 8 files changed, 320 insertions(+)")
        _patch_router(monkeypatch, executor_result=result_fn)
        _, result = route_task(meta2)
        assert result.success is True
        assert "REST API" in result.final_summary


# ── Model Resolution & Chain Entry Fields (1 merged test) ────────────────────

class TestModelAndChain:

    def test_chain_entries_have_model_profiles(self, tmp_path, monkeypatch):
        _patch_env(tmp_path, monkeypatch)
        result_fn = lambda tid: _make_result(tid)
        _patch_router(monkeypatch, executor_result=result_fn)
        task = _make_task("model-001", summary="Test model resolution")
        decision, result = route_task(task)
        assert result.success is True
        assert len(decision.chain) >= 1
        for entry in decision.chain:
            assert entry.tool
            assert entry.backend
            assert entry.model_profile
            assert isinstance(entry.model_profile, str) and len(entry.model_profile) > 0


# ── Cost & Latency Propagation (1 merged test) ──────────────────────────────

class TestExecutorOutputPropagation:

    def test_cost_and_latency_propagate(self, tmp_path, monkeypatch):
        _patch_env(tmp_path, monkeypatch)
        task = _make_task("cost-001", summary="Test cost and latency")
        result_fn = lambda tid: _make_result(tid, cost_usd=0.05, latency_ms=200)
        _patch_router(monkeypatch, executor_result=result_fn)
        _, result = route_task(task)
        assert result.success is True
        assert result.cost_estimate_usd == 0.05
        assert result.latency_ms >= 200

    def test_artifacts_propagate(self, tmp_path, monkeypatch):
        _patch_env(tmp_path, monkeypatch)
        task = _make_task("struct-001", summary="Test structured output")
        expected = ["/tmp/struct-001.stdout.txt", "/tmp/struct-001.stderr.txt", "/tmp/struct-001.diff.txt"]
        result_fn = lambda tid: _make_result(tid, artifacts=expected, summary="Changed: 2 files, 45 insertions")
        _patch_router(monkeypatch, executor_result=result_fn)
        _, result = route_task(task)
        assert result.artifacts == expected
        assert len(result.artifacts) == 3


# ── Trace ID Uniqueness (1 test) ─────────────────────────────────────────────

class TestTraceIdUniqueness:

    def test_three_unique_trace_ids(self, tmp_path, monkeypatch):
        _patch_env(tmp_path, monkeypatch)
        result_fn = lambda tid: _make_result(tid)
        _patch_router(monkeypatch, executor_result=result_fn)
        trace_ids = set()
        for i in range(3):
            decision, result = route_task(_make_task(f"trace-{i:03d}", summary=f"Task {i}"))
            assert result.trace_id
            assert decision.trace_id == result.trace_id
            trace_ids.add(result.trace_id)
        assert len(trace_ids) == 3


# ── Sequential & Concurrent (2 tests) ────────────────────────────────────────

class TestScale:

    def test_hundred_sequential_successes(self, tmp_path, monkeypatch):
        _patch_env(tmp_path, monkeypatch)
        result_fn = lambda tid: _make_result(tid, latency_ms=10)
        _patch_router(monkeypatch, executor_result=result_fn)
        results = [route_task(_make_task(f"seq-{i:04d}", summary=f"Sequential task {i}")) for i in range(100)]
        assert all(r.success for _, r in results)
        for _, r in results:
            assert r.task_id and r.tool and r.backend and r.latency_ms > 0

    def test_ten_concurrent_successes(self, tmp_path, monkeypatch):
        _patch_env(tmp_path, monkeypatch)
        result_fn = lambda tid: _make_result(tid, latency_ms=50)
        _patch_router(monkeypatch, executor_result=result_fn)
        def route_one(i):
            return route_task(_make_task(f"con-{i:03d}", summary=f"Concurrent task {i}"))
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = [f.result() for f in [pool.submit(route_one, i) for i in range(10)]]
        for decision, result in results:
            assert result.success is True
            assert result.task_id.startswith("con-")


# ── Config Snapshot Consistency (1 test) ──────────────────────────────────────

class TestConfigSnapshot:

    def test_fifty_consistent_snapshots(self):
        snapshots = [dict(get_config_snapshot()) for _ in range(50)]
        first = snapshots[0]
        for i, snap in enumerate(snapshots[1:], 1):
            assert snap == first, f"Snapshot {i} differs"


# ── RouteDecision & ExecutorResult Fields (1 merged test) ────────────────────

class TestResultAndDecisionFields:

    def test_decision_and_result_fields_populated(self, tmp_path, monkeypatch):
        _patch_env(tmp_path, monkeypatch)
        task = _make_task("dec-001", summary="Test fields")
        result_fn = lambda tid: _make_result(tid, tool="codex_cli", backend="openai_native",
                                              model_profile="codex_primary", artifacts=["/tmp/out.txt"])
        _patch_router(monkeypatch, executor_result=result_fn)
        decision, result = route_task(task)

        # Decision fields
        assert decision.task_id == "dec-001"
        assert decision.state in {s.value for s in CodexState}
        assert len(decision.chain) >= 1
        assert decision.reason
        assert isinstance(decision.attempted_fallback, bool)
        assert isinstance(decision.fallback_count, int)
        assert isinstance(decision.providers_skipped, list)
        assert isinstance(decision.chain_timed_out, bool)
        assert isinstance(decision.trace_id, str)
        assert isinstance(decision.error_history, list)

        # Result fields
        assert result.task_id == "dec-001"
        assert result.tool and result.backend and result.model_profile
        assert result.success is True
        assert result.latency_ms >= 0
        assert result.exit_code is not None
        assert isinstance(result.cost_estimate_usd, (int, float, type(None)))
        assert isinstance(result.artifacts, list)
        assert isinstance(result.trace_id, str)
        assert isinstance(result.partial_success, bool)
        assert isinstance(result.warnings, list)
        assert isinstance(result.error_history, list)


# ── Shutdown Rejection (1 test) ──────────────────────────────────────────────

class TestShutdownRejectsTasks:

    def test_shutdown_returns_rejection(self, tmp_path, monkeypatch):
        _patch_env(tmp_path, monkeypatch)
        shutdown_mgr = _patch_router(monkeypatch)
        shutdown_mgr.should_accept_new_tasks.return_value = False
        task = _make_task("shut-001", summary="Should be rejected by shutdown")
        decision, result = route_task(task)
        assert decision.state == "shutdown"
        assert decision.task_id == "shut-001"
        assert len(decision.chain) == 0
        assert result.success is False
        assert result.normalized_error == "shutdown_in_progress"
        assert "shutdown" in result.final_summary.lower() or "shutting" in result.final_summary.lower()
        shutdown_mgr.register_task.assert_not_called()
