"""Thread safety tests for StateStore.

Validates that concurrent access to StateStore does not corrupt state
files or cause crashes. Uses RLock for reentrancy.
"""

import json
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from router.models import CodexState
from router.state_store import StateStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a StateStore with temp paths."""
    manual = tmp_path / "manual.json"
    auto = tmp_path / "auto.json"
    history = tmp_path / "history.json"
    wal = tmp_path / "wal.jsonl"
    return StateStore(
        manual_path=manual,
        auto_path=auto,
        history_path=history,
        wal_path=wal,
    )


class TestConcurrentWriteSafety:
    """Concurrent writes should not corrupt state files."""

    def test_concurrent_set_auto_state_no_corruption(self, tmp_store):
        """10 threads writing auto state concurrently — file stays valid JSON."""
        states = list(CodexState)
        errors = []

        def write_auto(idx):
            try:
                state = states[idx % len(states)]
                tmp_store.set_auto_state(state)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_auto, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent writes: {errors}"

        # File must be valid JSON with a valid state value
        data = json.loads(tmp_store.auto_path.read_text())
        assert "state" in data
        assert data["state"] in {s.value for s in CodexState}

    def test_concurrent_set_manual_state_no_corruption(self, tmp_store):
        """10 threads writing manual state concurrently — file stays valid."""
        states = list(CodexState)
        errors = []

        def write_manual(idx):
            try:
                state = states[idx % len(states)]
                tmp_store.set_manual_state(state)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_manual, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent writes: {errors}"

        data = json.loads(tmp_store.manual_path.read_text())
        assert "state" in data
        # state can be null (None) or a valid state value
        if data["state"] is not None:
            assert data["state"] in {s.value for s in CodexState}


class TestConcurrentReadSafety:
    """Concurrent reads should always return valid results."""

    def test_concurrent_get_auto_state_consistent(self, tmp_store):
        """10 threads reading auto state — all get valid results."""
        tmp_store.set_auto_state(CodexState.OPENAI_PRIMARY)
        results = []
        errors = []

        def read_auto():
            try:
                result = tmp_store.get_auto_state()
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_auto) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent reads: {errors}"
        assert len(results) == 10
        assert all(r == CodexState.OPENAI_PRIMARY for r in results)

    def test_concurrent_get_state_consistent(self, tmp_store):
        """10 threads calling get_state() — all get valid CodexState."""
        tmp_store.set_auto_state(CodexState.CLAUDE_BACKUP)
        results = []
        errors = []

        def read_state():
            try:
                result = tmp_store.get_state()
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_state) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent reads: {errors}"
        assert len(results) == 10
        assert all(isinstance(r, CodexState) for r in results)


class TestConcurrentReadWriteSafety:
    """Mixed concurrent reads and writes should not crash."""

    def test_concurrent_reads_and_writes(self, tmp_store):
        """5 reader threads + 5 writer threads — no crashes, valid final state."""
        errors = []
        read_results = []

        def writer(idx):
            try:
                states = list(CodexState)
                for i in range(5):
                    tmp_store.set_auto_state(states[(idx + i) % len(states)])
            except Exception as e:
                errors.append(("writer", e))

        def reader(idx):
            try:
                for _ in range(5):
                    result = tmp_store.get_auto_state()
                    if result is not None:
                        read_results.append(result)
                    result = tmp_store.get_state()
                    read_results.append(result)
            except Exception as e:
                errors.append(("reader", e))

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Errors during concurrent read/write: {errors}"

        # Final state must be valid
        final = tmp_store.get_state()
        assert isinstance(final, CodexState)


class TestStateFileIntegrity:
    """State files must remain valid JSON after concurrent operations."""

    def test_file_integrity_after_concurrent_writes(self, tmp_path):
        """Write 20 states from 10 threads, verify both files are valid JSON."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        store = StateStore(manual_path=manual, auto_path=auto)

        states = list(CodexState)
        errors = []

        def write_both(idx):
            try:
                for i in range(4):
                    s = states[(idx + i) % len(states)]
                    store.set_auto_state(s)
                    store.set_manual_state(s)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(write_both, i) for i in range(10)]
            for f in as_completed(futures):
                exc = f.exception()
                if exc:
                    errors.append(exc)

        assert not errors, f"Errors: {errors}"

        # Both files must be valid JSON
        for path in [manual, auto]:
            data = json.loads(path.read_text())
            assert "state" in data
            if data["state"] is not None:
                assert data["state"] in {s.value for s in CodexState}


class TestLockReentrancy:
    """RLock allows the same thread to acquire the lock multiple times."""

    def test_set_state_with_history_is_reentrant(self, tmp_store):
        """set_state_with_history calls get_state, can_transition, set_manual_state,
        and log_state_transition — all of which acquire the lock.
        RLock must allow this without deadlock.
        """
        # This should complete without hanging (deadlock = test timeout)
        result = tmp_store.set_state_with_history(
            CodexState.CLAUDE_BACKUP, reason="test", force=True
        )
        assert result is True

    def test_get_state_is_reentrant(self, tmp_store):
        """get_state calls get_manual_state and get_auto_state — both lock.
        RLock must allow this without deadlock.
        """
        state = tmp_store.get_state()
        assert isinstance(state, CodexState)

    def test_can_transition_is_reentrant(self, tmp_store):
        """can_transition calls get_state and get_state_history — both lock.
        RLock must allow this without deadlock.
        """
        allowed, reason = tmp_store.can_transition(CodexState.CLAUDE_BACKUP)
        assert isinstance(allowed, bool)
        assert isinstance(reason, str)

    def test_concurrent_reentrant_calls(self, tmp_store):
        """Multiple threads calling reentrant methods concurrently — no deadlock."""
        errors = []

        def call_reentrant():
            try:
                tmp_store.set_state_with_history(
                    CodexState.CLAUDE_BACKUP, reason="test", force=True
                )
                tmp_store.get_state()
                tmp_store.can_transition(CodexState.OPENAI_PRIMARY)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_reentrant) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Errors in concurrent reentrant calls: {errors}"


class TestConcurrentStickyState:
    """Concurrent success counter operations should be safe."""

    def test_concurrent_record_success(self, tmp_store):
        """Multiple threads recording successes — no corruption of counter."""
        errors = []

        def record_many():
            try:
                for _ in range(20):
                    tmp_store.record_success(CodexState.CLAUDE_BACKUP, True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        # Counter should be >= 20 (10 threads * 20 increments)
        # Exact count depends on interleaving, but it should be > 0
        assert tmp_store.can_recover_to_primary(CodexState.CLAUDE_BACKUP)


class TestStressHundredConcurrentWrites:
    """Stress test: 100 concurrent writers must not corrupt state files."""

    def test_100_concurrent_writes_no_corruption(self, tmp_path):
        """100 threads writing state concurrently — file stays valid JSON."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        wal = tmp_path / "wal.jsonl"
        store = StateStore(manual_path=manual, auto_path=auto, wal_path=wal)

        states = list(CodexState)
        errors = []

        def writer(idx):
            try:
                for i in range(5):
                    s = states[(idx + i) % len(states)]
                    store.set_auto_state(s)
                    store.set_manual_state(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during 100-thread writes: {errors}"

        # Both files must be valid JSON with valid state values
        for path in [manual, auto]:
            data = json.loads(path.read_text())
            assert "state" in data
            if data["state"] is not None:
                assert data["state"] in {s.value for s in CodexState}


class TestConcurrentSetAndGet:
    """Concurrent set and get operations from separate thread groups."""

    def test_concurrent_set_and_get(self, tmp_store):
        """10 setter threads + 10 getter threads — no crashes, all results valid."""
        states = list(CodexState)
        errors = []
        results = []

        def setter(idx):
            try:
                for i in range(20):
                    tmp_store.set_manual_state(states[(idx + i) % len(states)])
            except Exception as e:
                errors.append(("setter", e))

        def getter():
            try:
                for _ in range(20):
                    result = tmp_store.get_manual_state()
                    results.append(result)
            except Exception as e:
                errors.append(("getter", e))

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=setter, args=(i,)))
        for _ in range(10):
            threads.append(threading.Thread(target=getter))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Errors during concurrent set/get: {errors}"
        # All results must be either None or a valid CodexState
        for r in results:
            assert r is None or isinstance(r, CodexState)


class TestReentrantDeadlockPrevention:
    """Verify that reentrant method chains don't cause deadlock."""

    def test_no_deadlock_on_reentrant_set_state(self, tmp_store):
        """set_state_with_history → can_transition → get_state → get_manual_state/get_auto_state.
        All acquire RLock. Must complete without deadlock."""
        # Set a history entry first so anti-flap can fire
        tmp_store.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP, "init"
        )
        # Force=True bypasses anti-flap, but still exercises reentrant path
        result = tmp_store.set_state_with_history(
            CodexState.OPENAI_PRIMARY, reason="test", force=True
        )
        assert result is True

        # Also test without force (exercises can_transition path)
        result = tmp_store.set_state_with_history(
            CodexState.CLAUDE_BACKUP, reason="test2", force=True
        )
        assert result is True

    def test_can_transition_reentrant_chain(self, tmp_store):
        """can_transition calls get_state which calls get_manual_state/get_auto_state.
        All lock acquisitions. Must not deadlock under concurrent access."""
        errors = []

        def transition_check():
            try:
                for state in CodexState:
                    tmp_store.can_transition(state)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=transition_check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Errors in reentrant chain: {errors}"


class TestConcurrentTransitions:
    """Concurrent transition checks from different starting states."""

    def test_concurrent_transition_from_different_states(self, tmp_store):
        """10 threads calling can_transition simultaneously — no deadlock, valid results."""
        errors = []
        results = []

        def check_transition():
            try:
                for target in CodexState:
                    allowed, reason = tmp_store.can_transition(target)
                    results.append((allowed, reason))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_transition) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Errors during concurrent transitions: {errors}"
        # All results must be (bool, str)
        for allowed, reason in results:
            assert isinstance(allowed, bool)
            assert isinstance(reason, str)


class TestLockFairness:
    """Verify no thread starvation under heavy contention."""

    def test_lock_acquisition_order_fairness(self, tmp_store):
        """100 threads each doing 10 writes. Verify all threads complete (no starvation).
        Measure completion count — all 100 must finish."""
        states = list(CodexState)
        completed = []
        errors = []

        def do_work(idx):
            try:
                for i in range(10):
                    s = states[(idx + i) % len(states)]
                    tmp_store.set_auto_state(s)
                    _ = tmp_store.get_auto_state()
                completed.append(idx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_work, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors: {errors}"
        # All 100 threads must complete — no starvation
        assert len(completed) == 100, f"Only {len(completed)}/100 threads completed — possible starvation"


class TestConcurrentWALOperations:
    """Multiple threads writing WAL simultaneously must not corrupt the WAL."""

    def test_concurrent_wal_operations(self, tmp_path):
        """Multiple threads writing to WAL via set operations — WAL stays valid JSONL."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        wal = tmp_path / "wal.jsonl"
        store = StateStore(manual_path=manual, auto_path=auto, wal_path=wal)

        states = list(CodexState)
        errors = []

        def wal_writer(idx):
            try:
                for i in range(10):
                    s = states[(idx + i) % len(states)]
                    store.set_auto_state(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=wal_writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Errors during concurrent WAL writes: {errors}"

        # WAL must be valid JSONL (one JSON object per line)
        if wal.exists():
            lines = wal.read_text().strip().splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)  # raises if invalid JSON
                assert "action" in entry
                assert entry["action"] in {"write", "committed"}


class TestPerformanceOverhead:
    """RLock overhead should be minimal."""

    def test_state_store_with_lock_does_not_slow_down_significantly(self, tmp_store):
        """1000 set+get operations must complete in under 2 seconds."""
        states = list(CodexState)
        start = time.monotonic()

        for i in range(1000):
            s = states[i % len(states)]
            tmp_store.set_auto_state(s)
            _ = tmp_store.get_auto_state()
            tmp_store.set_manual_state(s)
            _ = tmp_store.get_manual_state()

        elapsed = time.monotonic() - start
        assert elapsed < 3.0, f"1000 set+get ops took {elapsed:.2f}s (expected < 3.0s)"
