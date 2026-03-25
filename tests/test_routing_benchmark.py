"""Routing benchmark + latency tests.

Measures routing decision latency and ensures it stays under the 10ms target.
Uses mocked executors and in-memory state to isolate routing decision overhead.
"""

import statistics
import time
from contextlib import ExitStack
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, RouteDecision, ExecutorResult, ChainEntry,
    CodexState, TaskClass, TaskRisk, TaskModality,
)
from router.policy import (
    route_task, resolve_state, build_chain, get_breaker, reset_breaker,
    choose_openrouter_profile, can_fallback,
    _build_openai_primary_chain, _build_claude_backup_chain,
    _build_openrouter_fallback_chain, _build_openai_conservation_chain,
)
from router.circuit_breaker import CircuitBreaker


# ---------------------------------------------------------------------------
# Helpers — mock factories
# ---------------------------------------------------------------------------

def _make_mock_store(state: CodexState = CodexState.OPENAI_PRIMARY):
    """Create a mock StateStore that returns the given state without disk I/O."""
    store = MagicMock()
    store.get_manual_state.return_value = None
    store.get_auto_state.return_value = state if state != CodexState.OPENAI_PRIMARY else None
    return store


def _success_result(task_id: str = "bench-1") -> ExecutorResult:
    """Return a successful ExecutorResult immediately."""
    return ExecutorResult(
        task_id=task_id,
        tool="codex_cli",
        backend="openai_native",
        model_profile="codex_primary",
        success=True,
        latency_ms=0,
    )


def _make_task(task_id: str = "bench-1", **overrides) -> TaskMeta:
    """Create a TaskMeta with sensible defaults."""
    defaults = dict(
        task_id=task_id,
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.LOW,
        modality=TaskModality.TEXT,
    )
    defaults.update(overrides)
    return TaskMeta(**defaults)


def _percentile(data, pct):
    """Compute percentile (0-100)."""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100.0)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _patch_route_mocks():
    """Return patches needed to make route_task() fast (no real execution).

    Uses patch.multiple on the router.policy module.
    """
    mock_breaker = CircuitBreaker(threshold=999)  # never trips
    mock_notifier = MagicMock()
    mock_notifier.check_conservation_duration.return_value = None

    return [
        patch("router.policy.resolve_state", return_value=CodexState.OPENAI_PRIMARY),
        patch("router.policy.get_breaker", return_value=mock_breaker),
        patch("router.health.get_shutdown_manager", return_value=MagicMock(
            should_accept_new_tasks=lambda: True,
            register_task=lambda *a: None,
            unregister_task=lambda *a: None,
        )),
        patch("router.policy.get_notifier", return_value=mock_notifier),
        patch("router.policy.AttemptLogger", return_value=MagicMock(log_trace=lambda trace: None)),
        patch("router.policy.get_reliability_config", return_value={"chain_timeout_s": 600, "max_fallbacks": 3}),
        patch("router.policy._run_executor", side_effect=lambda entry, task, trace_id="": _success_result(task.task_id)),
    ]


def _apply_route_mocks(exit_stack: ExitStack):
    """Apply all route mocks using the given ExitStack."""
    for p in _patch_route_mocks():
        exit_stack.enter_context(p)


# ---------------------------------------------------------------------------
# 1. route_task() benchmark — 100 calls, p95 < 10ms
# ---------------------------------------------------------------------------

class TestRouteTaskBenchmark:
    """Benchmark route_task() with fully mocked executors."""

    def test_100_calls_p95_under_10ms(self):
        """100 route_task() calls with mocked executors — p95 < 10ms."""
        with ExitStack() as stack:
            _apply_route_mocks(stack)

            # Warm up
            for _ in range(5):
                route_task(_make_task())

            latencies = []
            for i in range(100):
                task = _make_task(task_id=f"bench-{i}")
                start = time.perf_counter()
                decision, result = route_task(task)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            p95 = _percentile(latencies, 95)
            p50 = _percentile(latencies, 50)

            print(f"\n  route_task() 100 calls: p50={p50:.3f}ms  p95={p95:.3f}ms  max={max(latencies):.3f}ms")
            assert p95 < 10.0, f"p95 latency {p95:.3f}ms exceeds 10ms target"

    def test_route_decision_correct(self):
        """Sanity: mocked route_task returns valid RouteDecision."""
        with ExitStack() as stack:
            _apply_route_mocks(stack)

            task = _make_task()
            decision, result = route_task(task)

            assert isinstance(decision, RouteDecision)
            assert decision.task_id == "bench-1"
            assert decision.state == "openai_primary"
            assert len(decision.chain) > 0
            assert result.success is True


# ---------------------------------------------------------------------------
# 2. Individual component benchmarks
# ---------------------------------------------------------------------------

class TestComponentBenchmarks:
    """Benchmark individual routing components."""

    def test_resolve_state_under_1ms(self):
        """resolve_state() with mocked StateStore: < 1ms."""
        store = _make_mock_store()

        # Warm up
        for _ in range(10):
            resolve_state(store)

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            resolve_state(store)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p95 = _percentile(latencies, 95)
        print(f"\n  resolve_state() p95={p95:.4f}ms")
        assert p95 < 1.0, f"resolve_state p95 {p95:.4f}ms exceeds 1ms target"

    def test_build_chain_under_1ms(self):
        """build_chain() for all 4 states: < 1ms."""
        task = _make_task()
        states = [
            CodexState.OPENAI_PRIMARY,
            CodexState.OPENAI_CONSERVATION,
            CodexState.CLAUDE_BACKUP,
            CodexState.OPENROUTER_FALLBACK,
        ]

        # Warm up
        for state in states:
            build_chain(task, state)

        latencies = []
        for _ in range(100):
            for state in states:
                start = time.perf_counter()
                build_chain(task, state)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

        p95 = _percentile(latencies, 95)
        print(f"\n  build_chain() p95={p95:.4f}ms (400 calls across 4 states)")
        assert p95 < 1.0, f"build_chain p95 {p95:.4f}ms exceeds 1ms target"

    def test_get_model_lookup_under_1ms(self):
        """get_model() lookup with cached config: < 1ms."""
        from router.config_loader import get_model

        # Ensure config is loaded (cache warmup)
        get_model("gpt54")
        get_model("minimax")
        get_model("sonnet")

        profiles = ["gpt54", "gpt54_mini", "minimax", "kimi", "mimo", "sonnet", "opus"]
        latencies = []
        for _ in range(50):
            for profile in profiles:
                start = time.perf_counter()
                get_model(profile)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

        p95 = _percentile(latencies, 95)
        print(f"\n  get_model() p95={p95:.4f}ms ({len(latencies)} lookups)")
        assert p95 < 1.0, f"get_model p95 {p95:.4f}ms exceeds 1ms target"

    def test_circuit_breaker_check_under_1ms(self):
        """CircuitBreaker.is_available(): < 1ms."""
        breaker = CircuitBreaker(threshold=5, window_s=60, cooldown_s=120)

        # Warm up
        for _ in range(10):
            breaker.is_available("codex_cli", "openai_native")

        latencies = []
        providers = [
            ("codex_cli", "openai_native"),
            ("claude_code", "anthropic"),
            ("codex_cli", "openrouter"),
        ]
        for _ in range(100):
            for tool, backend in providers:
                start = time.perf_counter()
                breaker.is_available(tool, backend)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

        p95 = _percentile(latencies, 95)
        print(f"\n  circuit_breaker.is_available() p95={p95:.4f}ms ({len(latencies)} checks)")
        assert p95 < 1.0, f"circuit breaker p95 {p95:.4f}ms exceeds 1ms target"


# ---------------------------------------------------------------------------
# 3. Routing latency under load — 1000 calls < 5s
# ---------------------------------------------------------------------------

class TestRoutingLoad:
    """Routing latency under sustained load."""

    def test_1000_calls_under_5s(self):
        """1000 route_task() calls with mocked executors — total < 5s (5ms avg)."""
        with ExitStack() as stack:
            _apply_route_mocks(stack)

            # Warm up
            for _ in range(10):
                route_task(_make_task())

            start = time.perf_counter()
            for i in range(1000):
                task = _make_task(task_id=f"load-{i}")
                decision, result = route_task(task)
            total_ms = (time.perf_counter() - start) * 1000

            avg_ms = total_ms / 1000
            print(f"\n  route_task() 1000 calls: total={total_ms:.1f}ms  avg={avg_ms:.3f}ms")
            assert total_ms < 5000.0, f"1000 route_task() calls took {total_ms:.1f}ms, exceeds 5s target"

    def test_1000_calls_detailed_stats(self):
        """1000 route_task() calls with per-call latency measurement."""
        with ExitStack() as stack:
            _apply_route_mocks(stack)

            # Warm up
            for _ in range(10):
                route_task(_make_task())

            latencies = []
            for i in range(1000):
                task = _make_task(task_id=f"stats-{i}")
                start = time.perf_counter()
                decision, result = route_task(task)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            p50 = _percentile(latencies, 50)
            p95 = _percentile(latencies, 95)
            p99 = _percentile(latencies, 99)
            max_lat = max(latencies)

            print(f"\n  route_task() 1000 calls: p50={p50:.3f}ms  p95={p95:.3f}ms  p99={p99:.3f}ms  max={max_lat:.3f}ms")

            # All percentiles should be well under 10ms
            assert p95 < 10.0, f"p95 {p95:.3f}ms exceeds 10ms"
            assert p99 < 20.0, f"p99 {p99:.3f}ms exceeds 20ms (generous headroom)"


# ---------------------------------------------------------------------------
# Optional: benchmark runner (python3 -m router.benchmark)
# ---------------------------------------------------------------------------

def run_benchmarks():
    """Run the full benchmark suite and print results.

    Can be invoked via: python3 -m router.benchmark
    """
    print("=" * 60)
    print("  Routing Decision Latency Benchmark")
    print("=" * 60)

    # --- resolve_state ---
    store = _make_mock_store()
    for _ in range(10):
        resolve_state(store)
    rs_lat = []
    for _ in range(500):
        start = time.perf_counter()
        resolve_state(store)
        rs_lat.append((time.perf_counter() - start) * 1000)

    # --- build_chain ---
    task = _make_task()
    for state in CodexState:
        build_chain(task, state)
    bc_lat = []
    for _ in range(500):
        for state in CodexState:
            start = time.perf_counter()
            build_chain(task, state)
            bc_lat.append((time.perf_counter() - start) * 1000)

    # --- circuit breaker ---
    cb = CircuitBreaker(threshold=999)
    for _ in range(10):
        cb.is_available("codex_cli", "openai_native")
    cb_lat = []
    for _ in range(500):
        start = time.perf_counter()
        cb.is_available("codex_cli", "openai_native")
        cb_lat.append((time.perf_counter() - start) * 1000)

    # --- route_task ---
    with ExitStack() as stack:
        _apply_route_mocks(stack)

        for _ in range(20):
            route_task(_make_task())
        rt_lat = []
        for i in range(1000):
            task = _make_task(task_id=f"bm-{i}")
            start = time.perf_counter()
            route_task(task)
            rt_lat.append((time.perf_counter() - start) * 1000)

    def _report(name, data):
        data_sorted = sorted(data)
        n = len(data_sorted)
        p50 = data_sorted[int(n * 0.50)]
        p95 = data_sorted[int(n * 0.95)]
        p99 = data_sorted[int(n * 0.99)]
        mx = data_sorted[-1]
        print(f"\n  {name} ({n} calls)")
        print(f"    p50: {p50:.4f} ms")
        print(f"    p95: {p95:.4f} ms")
        print(f"    p99: {p99:.4f} ms")
        print(f"    max: {mx:.4f} ms")
        return p50, p95, p99, mx

    print()
    rs = _report("resolve_state()", rs_lat)
    bc = _report("build_chain()", bc_lat)
    cb_r = _report("circuit_breaker.is_available()", cb_lat)
    rt = _report("route_task()", rt_lat)

    print()
    print("-" * 60)
    # Verify targets
    checks = [
        ("resolve_state p95 < 1ms", rs[1] < 1.0),
        ("build_chain p95 < 1ms", bc[1] < 1.0),
        ("circuit_breaker p95 < 1ms", cb_r[1] < 1.0),
        ("route_task p95 < 10ms", rt[1] < 10.0),
    ]
    all_pass = all(ok for _, ok in checks)
    for label, ok in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}: {label}")
    print("-" * 60)

    if not all_pass:
        raise SystemExit(1)

    print("\n  All benchmarks passed! 🎉\n")


if __name__ == "__main__":
    run_benchmarks()
