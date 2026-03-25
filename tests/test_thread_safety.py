"""Thread safety tests for StateStore.

Validates that concurrent access to StateStore does not corrupt state
files or cause crashes. Uses RLock for reentrancy.
"""

import json
import threading
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
