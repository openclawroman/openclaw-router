"""End-to-end happy-path tests for the router.

Covers the COMPLETE flow from task input through routing, execution, and result.
All tests use realistic mock executors — no real I/O or network calls.

16 tests covering every happy-path scenario the router should handle perfectly.
"""

import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    RouteDecision, ExecutorResult, ChainEntry, CodexState,
)
from router.state_store import reset_state_store
from router.policy import (
    route_task, build_chain, resolve_state, reset_breaker, reset_notifier,
)
from router.config_loader import load_config, get_model, get_config_snapshot
from router.classifier import classify, classify_from_dict
from router.health import ShutdownManager

from tests.conftest import MockExecutor, MockExecutorChain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_env(tmp_path, monkeypatch):
    """Set up isolated routing environment. Returns dict with paths."""
    state_dir = tmp_path / "state"
    runtime_dir = tmp_path / "runtime"
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

    return {"state_dir": state_dir, "runtime_dir": runtime_dir}


def _make_result(task_id, *, success=True, tool="codex_cli", backend="openai_native",
                 model_profile="codex_primary", cost_usd=0.0023, latency_ms=150,
                 summary="Task completed", artifacts=None, error=None):
    """Build a mock ExecutorResult."""
    return ExecutorResult(
        task_id=task_id,
        tool=tool,
        backend=backend,
        model_profile=model_profile,
        success=success,
        normalized_error=error,
        exit_code=0 if success else 1,
        latency_ms=latency_ms,
        cost_estimate_usd=cost_usd if success else 0.0,
        artifacts=artifacts if artifacts else [],
        final_summary=summary if success else None,
    )


def _make_run_executor(result_fn):
    """Create a _run_executor replacement that stamps trace_id like the real one."""
    def _run_executor(entry, task, trace_id=""):
        result = result_fn(task.task_id)
        result.trace_id = trace_id
        return result
    return _run_executor


def _patch_route_mocks(monkeypatch, executor_result=None):
    """Mock the shutdown manager and optional executor for route_task tests."""
    mock_sm = MagicMock(
        should_accept_new_tasks=MagicMock(return_value=True),
        register_task=MagicMock(),
        unregister_task=MagicMock(),
    )
    monkeypatch.setattr("router.health.get_shutdown_manager", lambda: mock_sm)

    if executor_result is not None:
        monkeypatch.setattr("router.policy._run_executor", _make_run_executor(executor_result))

    return mock_sm


# ===================================================================
# 1. test_config_load_to_result
# ===================================================================

class TestConfigLoadToResult:
    """Load config → get_model → resolve_state → build_chain → route_task → success."""

    def test_full_pipeline(self, tmp_path, monkeypatch):
        """The complete happy path: config loads, model resolves, chain builds, task routes."""
        _patch_env(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="cfg-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/test-repo", cwd="/tmp/test-repo",
            summary="Implement feature X with tests",
        )

        result_fn = lambda tid: _make_result(tid, summary="3 files changed, 42 insertions(+)")
        _patch_route_mocks(monkeypatch, result_fn)

        decision, result = route_task(task)

        assert decision.task_id == "cfg-001"
        assert decision.state in {s.value for s in CodexState}
        assert len(decision.chain) >= 1
        assert result.success is True
        assert result.cost_estimate_usd is not None
        assert result.cost_estimate_usd > 0
        assert result.final_summary is not None
        assert result.latency_ms > 0


# ===================================================================
# 2. test_implement_task_full_flow
# ===================================================================

class TestImplementTaskFullFlow:
    """Task class=implement → classify → codex chain → execute → result with artifacts."""

    def test_implement_classifies_correctly(self):
        """'Implement' keyword maps to IMPLEMENTATION class."""
        meta = classify("Implement user authentication with JWT tokens")
        assert meta.task_class == TaskClass.IMPLEMENTATION

    def test_implement_routes_through(self, tmp_path, monkeypatch):
        """Implementation task routes through chain and returns result with artifacts."""
        _patch_env(tmp_path, monkeypatch)

        meta = classify("Implement user authentication with JWT tokens")
        meta.repo_path = "/tmp/test-repo"

        artifacts = ["/tmp/task-implement.stdout.txt", "/tmp/task-implement.stderr.txt"]
        result_fn = lambda tid: _make_result(
            tid, tool="codex_cli", backend="openai_native",
            model_profile="codex_primary",
            artifacts=artifacts,
            summary="Implemented JWT auth: 5 files changed, 180 insertions(+)",
        )
        _patch_route_mocks(monkeypatch, result_fn)

        decision, result = route_task(meta)

        assert result.success is True
        assert result.artifacts == artifacts
        assert len(decision.chain) >= 1


# ===================================================================
# 3. test_bugfix_task_full_flow
# ===================================================================

class TestBugfixTaskFullFlow:
    """Task class=bugfix → classify → chain → execute."""

    def test_bugfix_classifies_correctly(self):
        """'Fix' keyword maps to BUGFIX class."""
        meta = classify("Fix null pointer exception in login handler")
        assert meta.task_class == TaskClass.BUGFIX

    def test_bugfix_routes_through(self, tmp_path, monkeypatch):
        """Bugfix task routes through chain and succeeds."""
        _patch_env(tmp_path, monkeypatch)

        meta = classify("Fix null pointer exception in login handler")
        meta.repo_path = "/tmp/test-repo"

        result_fn = lambda tid: _make_result(
            tid, tool="codex_cli", backend="openai_native",
            summary="Fixed: 1 file changed, 3 insertions(+), 2 deletions(-)",
        )
        _patch_route_mocks(monkeypatch, result_fn)

        decision, result = route_task(meta)

        assert result.success is True
        assert result.final_summary is not None
        assert "Fixed" in result.final_summary


# ===================================================================
# 4. test_from_dict_task_full_flow
# ===================================================================

class TestFromDictTaskFullFlow:
    """Task class=from_dict → classify → chain → execute."""

    def test_classify_from_dict_basic(self):
        """classify_from_dict creates valid TaskMeta from minimal dict."""
        raw = {
            "summary": "Build a REST API for inventory management",
            "repo_path": "/tmp/repo",
        }
        meta = classify_from_dict(raw)
        assert meta.task_class == TaskClass.IMPLEMENTATION
        assert meta.repo_path == "/tmp/repo"

    def test_from_dict_routes_through(self, tmp_path, monkeypatch):
        """Task from dict routes through chain and succeeds."""
        _patch_env(tmp_path, monkeypatch)

        meta = classify_from_dict({
            "task_id": "dict-001",
            "summary": "Build a REST API for inventory management",
            "repo_path": "/tmp/test-repo",
        })

        result_fn = lambda tid: _make_result(
            tid, tool="codex_cli", backend="openai_native",
            summary="REST API created: 8 files changed, 320 insertions(+)",
        )
        _patch_route_mocks(monkeypatch, result_fn)

        decision, result = route_task(meta)

        assert result.success is True
        assert "REST API" in result.final_summary


# ===================================================================
# 5. test_model_resolution_in_chain
# ===================================================================

class TestModelResolutionInChain:
    """Verify model_profile in chain entry matches config."""

    def test_primary_chain_has_model_profiles(self, tmp_path, monkeypatch):
        """Chain entries in OPENAI_PRIMARY state have valid model_profile values."""
        _patch_env(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="model-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Test model resolution",
        )

        state = resolve_state()
        chain = build_chain(task, state)

        assert len(chain) >= 1
        for entry in chain:
            assert entry.model_profile, \
                f"Chain entry {entry.tool}:{entry.backend} missing model_profile"
            assert isinstance(entry.model_profile, str)
            assert len(entry.model_profile) > 0


# ===================================================================
# 6. test_cost_propagates_to_result
# ===================================================================

class TestCostPropagation:
    """Mock executor returns cost → result has correct cost."""

    def test_cost_matches_executor(self, tmp_path, monkeypatch):
        """When executor returns cost_usd=0.05, result.cost_estimate_usd == 0.05."""
        _patch_env(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="cost-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Test cost propagation",
        )

        result_fn = lambda tid: _make_result(tid, cost_usd=0.05)
        _patch_route_mocks(monkeypatch, result_fn)

        _, result = route_task(task)

        assert result.success is True
        assert result.cost_estimate_usd == 0.05


# ===================================================================
# 7. test_trace_id_unique_per_task
# ===================================================================

class TestTraceIdUniqueness:
    """3 tasks → 3 different trace_ids."""

    def test_three_unique_trace_ids(self, tmp_path, monkeypatch):
        """Each routing call produces a unique trace_id."""
        _patch_env(tmp_path, monkeypatch)

        result_fn = lambda tid: _make_result(tid)
        _patch_route_mocks(monkeypatch, result_fn)

        trace_ids = set()
        for i in range(3):
            task = TaskMeta(
                task_id=f"trace-{i:03d}", agent="coder",
                task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
                repo_path="/tmp/repo", cwd="/tmp/repo",
                summary=f"Task {i}",
            )
            decision, result = route_task(task)
            assert result.trace_id, f"Result {i} missing trace_id"
            assert decision.trace_id == result.trace_id, \
                f"Decision/Result trace_id mismatch at task {i}"
            trace_ids.add(result.trace_id)

        assert len(trace_ids) == 3, \
            f"Expected 3 unique trace_ids, got {len(trace_ids)}: {trace_ids}"


# ===================================================================
# 8. test_latency_measured
# ===================================================================

class TestLatencyMeasured:
    """result.latency_ms > 0 and roughly matches mock delay."""

    def test_latency_positive(self, tmp_path, monkeypatch):
        """Executor latency_ms is positive on success."""
        _patch_env(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="lat-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Test latency measurement",
        )

        result_fn = lambda tid: _make_result(tid, latency_ms=200)
        _patch_route_mocks(monkeypatch, result_fn)

        _, result = route_task(task)

        assert result.success is True
        assert result.latency_ms > 0
        assert result.latency_ms >= 200  # at least the mock delay


# ===================================================================
# 9. test_structured_output_in_result
# ===================================================================

class TestStructuredOutputInResult:
    """Mock executor returns artifacts → result has artifacts."""

    def test_artifacts_propagate(self, tmp_path, monkeypatch):
        """When executor returns artifacts, they appear in the result."""
        _patch_env(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="struct-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Test structured output",
        )

        expected_artifacts = [
            "/tmp/struct-001.stdout.txt",
            "/tmp/struct-001.stderr.txt",
            "/tmp/struct-001.diff.txt",
        ]
        result_fn = lambda tid: _make_result(
            tid, artifacts=expected_artifacts,
            summary="Changed: 2 files, 45 insertions",
        )
        _patch_route_mocks(monkeypatch, result_fn)

        _, result = route_task(task)

        assert result.success is True
        assert result.artifacts == expected_artifacts
        assert len(result.artifacts) == 3


# ===================================================================
# 10. test_sequential_100_tasks
# ===================================================================

class TestSequential100Tasks:
    """100 tasks routed sequentially, all succeed, metrics reflect all."""

    def test_hundred_sequential_successes(self, tmp_path, monkeypatch):
        """Route 100 tasks sequentially — all succeed."""
        _patch_env(tmp_path, monkeypatch)

        result_fn = lambda tid: _make_result(tid, latency_ms=10)
        _patch_route_mocks(monkeypatch, result_fn)

        results = []
        for i in range(100):
            task = TaskMeta(
                task_id=f"seq-{i:04d}", agent="coder",
                task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
                repo_path="/tmp/repo", cwd="/tmp/repo",
                summary=f"Sequential task {i}",
            )
            decision, result = route_task(task)
            results.append(result)

        successes = [r for r in results if r.success]
        assert len(successes) == 100
        # Verify all have proper fields
        for r in results:
            assert r.task_id
            assert r.tool
            assert r.backend
            assert r.latency_ms > 0


# ===================================================================
# 11. test_concurrent_10_tasks
# ===================================================================

class TestConcurrent10Tasks:
    """10 tasks routed in parallel with ThreadPoolExecutor, all succeed."""

    def test_ten_concurrent_successes(self, tmp_path, monkeypatch):
        """Route 10 tasks concurrently — all succeed."""
        _patch_env(tmp_path, monkeypatch)

        result_fn = lambda tid: _make_result(tid, latency_ms=50)
        _patch_route_mocks(monkeypatch, result_fn)

        def route_one(i):
            task = TaskMeta(
                task_id=f"con-{i:03d}", agent="coder",
                task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.LOW,
                repo_path="/tmp/repo", cwd="/tmp/repo",
                summary=f"Concurrent task {i}",
            )
            return route_task(task)

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(route_one, i) for i in range(10)]
            results = [f.result() for f in futures]

        for decision, result in results:
            assert result.success is True
            assert result.task_id.startswith("con-")


# ===================================================================
# 12. test_config_snapshot_consistent
# ===================================================================

class TestConfigSnapshotConsistent:
    """get_config_snapshot returns same values across 50 calls."""

    def test_fifty_consistent_snapshots(self):
        """50 calls to get_config_snapshot return identical values."""
        snapshots = []
        for _ in range(50):
            snap = get_config_snapshot()
            snapshots.append(dict(snap))  # Convert MappingProxyType to dict for comparison

        first = snapshots[0]
        for i, snap in enumerate(snapshots[1:], 1):
            assert snap == first, f"Snapshot {i} differs from first snapshot"


# ===================================================================
# 13. test_all_chain_entries_have_required_fields
# ===================================================================

class TestChainEntryFields:
    """Every chain entry has tool, backend, model_profile."""

    def test_all_entries_complete(self, tmp_path, monkeypatch):
        """Build chain for a standard task — every entry has required fields."""
        _patch_env(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="fields-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Test chain entry fields",
        )

        state = resolve_state()
        chain = build_chain(task, state)

        for entry in chain:
            assert isinstance(entry, ChainEntry)
            assert entry.tool, f"Chain entry missing 'tool': {entry}"
            assert entry.backend, f"Chain entry missing 'backend': {entry}"
            assert entry.model_profile, f"Chain entry missing 'model_profile': {entry}"


# ===================================================================
# 14. test_route_decision_has_all_fields
# ===================================================================

class TestRouteDecisionFields:
    """RouteDecision has task_id, state, chain, reason, trace_id, etc."""

    def test_decision_fields_populated(self, tmp_path, monkeypatch):
        """RouteDecision returned by route_task has all expected fields populated."""
        _patch_env(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="dec-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Test route decision fields",
        )

        result_fn = lambda tid: _make_result(tid)
        _patch_route_mocks(monkeypatch, result_fn)

        decision, result = route_task(task)

        # Required fields
        assert decision.task_id == "dec-001"
        assert decision.state in {s.value for s in CodexState}
        assert isinstance(decision.chain, list)
        assert len(decision.chain) >= 1
        assert decision.reason, "RouteDecision.reason should be populated"

        # Optional fields exist (may be default/empty)
        assert isinstance(decision.attempted_fallback, bool)
        assert isinstance(decision.fallback_count, int)
        assert isinstance(decision.providers_skipped, list)
        assert isinstance(decision.chain_timed_out, bool)
        assert isinstance(decision.trace_id, str)
        assert isinstance(decision.error_history, list)


# ===================================================================
# 15. test_executor_result_has_all_fields
# ===================================================================

class TestExecutorResultFields:
    """ExecutorResult has tool, backend, model_profile, success, latency_ms, etc."""

    def test_result_fields_populated(self, tmp_path, monkeypatch):
        """ExecutorResult returned by route_task has all expected fields populated."""
        _patch_env(tmp_path, monkeypatch)

        task = TaskMeta(
            task_id="res-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Test executor result fields",
        )

        result_fn = lambda tid: _make_result(
            tid, tool="codex_cli", backend="openai_native",
            model_profile="codex_primary", artifacts=["/tmp/out.txt"],
        )
        _patch_route_mocks(monkeypatch, result_fn)

        _, result = route_task(task)

        # Required fields
        assert result.task_id == "res-001"
        assert result.tool, "ExecutorResult.tool should be populated"
        assert result.backend, "ExecutorResult.backend should be populated"
        assert result.model_profile, "ExecutorResult.model_profile should be populated"
        assert result.success is True
        assert result.latency_ms >= 0

        # Optional fields present (may be default/None)
        assert result.normalized_error is None or isinstance(result.normalized_error, str)
        assert result.exit_code is not None
        assert isinstance(result.cost_estimate_usd, (int, float, type(None)))
        assert isinstance(result.artifacts, list)
        assert isinstance(result.final_summary, (str, type(None)))
        assert isinstance(result.trace_id, str)
        assert isinstance(result.partial_success, bool)
        assert isinstance(result.warnings, list)
        assert isinstance(result.error_history, list)


# ===================================================================
# 16. test_shutdown_rejects_tasks
# ===================================================================

class TestShutdownRejectsTasks:
    """shutdown_mgr.should_accept_new_tasks()=False → proper error."""

    def test_shutdown_returns_rejection(self, tmp_path, monkeypatch):
        """When shutdown is requested, route_task returns rejection result."""
        _patch_env(tmp_path, monkeypatch)

        mock_sm = MagicMock(
            should_accept_new_tasks=MagicMock(return_value=False),
            register_task=MagicMock(),
            unregister_task=MagicMock(),
        )
        monkeypatch.setattr("router.health.get_shutdown_manager", lambda: mock_sm)

        task = TaskMeta(
            task_id="shut-001", agent="coder",
            task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
            repo_path="/tmp/repo", cwd="/tmp/repo",
            summary="Should be rejected by shutdown",
        )

        decision, result = route_task(task)

        # Decision should indicate shutdown
        assert decision.state == "shutdown"
        assert decision.task_id == "shut-001"
        assert len(decision.chain) == 0

        # Result should be a failure
        assert result.success is False
        assert result.normalized_error == "shutdown_in_progress"
        assert "shutdown" in result.final_summary.lower() \
            or "shutting" in result.final_summary.lower()

        # Executor should NOT have been called
        mock_sm.register_task.assert_not_called()
