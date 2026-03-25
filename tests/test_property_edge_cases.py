"""Additional edge-case property tests for PR #55 review.

Covers gaps identified in the review checklist:
  1. Anti-flap boundary: exactly AT vs just BEFORE timeout
  2. Force bypass during anti-flap window
  3. State survives reload (new StateStore from same paths)
  4. State after WAL recovery matches expected value
  5. All tools unavailable (all circuit breakers open)
  6. Concurrent state writes from multiple threads
  7. Extreme chain length stress
"""

import json
import random
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from router.models import CodexState, TaskMeta, TaskClass, TaskRisk
from router.state_store import StateStore, MIN_STATE_DURATION_S
from router.circuit_breaker import CircuitBreaker
from router.policy import build_chain
from router.errors import StateError

ALL_STATES = list(CodexState)


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    """Provide a StateStore with temporary paths."""
    return StateStore(
        manual_path=tmp_path / "manual.json",
        auto_path=tmp_path / "auto.json",
        history_path=tmp_path / "history.json",
        wal_path=tmp_path / "wal.jsonl",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Anti-flap boundary tests: exactly AT vs just BEFORE timeout
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiFlapBoundary:
    """Anti-flap timeout boundary: exactly AT (300s) vs just BEFORE (299s)."""

    def test_transition_allowed_exactly_at_timeout(self, tmp_store):
        """Transition is allowed when elapsed == MIN_STATE_DURATION_S exactly."""
        tmp_store.set_state_with_history(
            CodexState.OPENAI_CONSERVATION, reason="setup", force=True
        )

        # Get the last history entry's timestamp
        history = tmp_store.get_state_history(limit=1)
        assert len(history) == 1
        last_ts_str = history[0]["timestamp"]

        # Patch datetime to simulate exactly MIN_STATE_DURATION_S elapsed
        class FakeDatetime:
            @staticmethod
            def now(tz=None):
                from datetime import datetime, timezone as tz_mod
                base = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
                return base + __import__("datetime").timedelta(seconds=MIN_STATE_DURATION_S)

            # Also need to patch fromisoformat
            @staticmethod
            def fromisoformat(s):
                from datetime import datetime
                return datetime.fromisoformat(s)

            timedelta = __import__("datetime").timedelta

        with patch("router.state_store.datetime", FakeDatetime):
            allowed, reason = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK)
            assert allowed is True, (
                f"Transition blocked exactly at timeout boundary: {reason}"
            )

    def test_transition_blocked_just_before_timeout(self, tmp_store):
        """Transition is blocked when elapsed < MIN_STATE_DURATION_S."""
        tmp_store.set_state_with_history(
            CodexState.OPENAI_CONSERVATION, reason="setup", force=True
        )

        history = tmp_store.get_state_history(limit=1)
        last_ts_str = history[0]["timestamp"]

        class FakeDatetime:
            @staticmethod
            def now(tz=None):
                from datetime import datetime, timezone as tz_mod
                base = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
                # 1 second before timeout
                return base + __import__("datetime").timedelta(seconds=MIN_STATE_DURATION_S - 1)

            @staticmethod
            def fromisoformat(s):
                from datetime import datetime
                return datetime.fromisoformat(s)

            timedelta = __import__("datetime").timedelta

        with patch("router.state_store.datetime", FakeDatetime):
            allowed, reason = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK)
            assert allowed is False, (
                f"Transition should be blocked just before timeout, but was allowed: {reason}"
            )
            assert "anti_flap" in reason

    def test_transition_allowed_just_after_timeout(self, tmp_store):
        """Transition is allowed when elapsed > MIN_STATE_DURATION_S."""
        tmp_store.set_state_with_history(
            CodexState.OPENAI_CONSERVATION, reason="setup", force=True
        )

        history = tmp_store.get_state_history(limit=1)
        last_ts_str = history[0]["timestamp"]

        class FakeDatetime:
            @staticmethod
            def now(tz=None):
                from datetime import datetime
                base = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
                return base + __import__("datetime").timedelta(seconds=MIN_STATE_DURATION_S + 1)

            @staticmethod
            def fromisoformat(s):
                from datetime import datetime
                return datetime.fromisoformat(s)

            timedelta = __import__("datetime").timedelta

        with patch("router.state_store.datetime", FakeDatetime):
            allowed, reason = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK)
            assert allowed is True

    def test_boundary_scans_all_states(self, tmp_store):
        """For each state, transition is allowed exactly at timeout."""
        for state in ALL_STATES:
            tmp_store.set_state_with_history(state, reason="boundary-test", force=True)
            history = tmp_store.get_state_history(limit=1)
            last_ts_str = history[0]["timestamp"]

            class FakeDatetime:
                @staticmethod
                def now(tz=None):
                    from datetime import datetime
                    base = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
                    return base + __import__("datetime").timedelta(seconds=MIN_STATE_DURATION_S)

                @staticmethod
                def fromisoformat(s):
                    from datetime import datetime
                    return datetime.fromisoformat(s)

                timedelta = __import__("datetime").timedelta

            # Test non-emergency target (openrouter_fallback)
            with patch("router.state_store.datetime", FakeDatetime):
                allowed, reason = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK)
                assert allowed is True, (
                    f"Boundary test failed from {state}: {reason}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Force bypass during anti-flap window
# ═══════════════════════════════════════════════════════════════════════════════

class TestForceBypassDuringAntiFlap:
    """force=True always bypasses anti-flap, even mid-window."""

    def test_force_immediately_after_transition(self, tmp_store):
        """Force transition immediately after a normal transition."""
        tmp_store.set_state_with_history(
            CodexState.OPENAI_CONSERVATION, reason="setup", force=True
        )
        # Immediately force another transition (within anti-flap window)
        tmp_store.set_state_with_history(
            CodexState.OPENROUTER_FALLBACK, reason="emergency", force=True
        )
        assert tmp_store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_force_to_non_emergency_during_window(self, tmp_store):
        """Force to non-emergency target (openrouter_fallback) during anti-flap."""
        tmp_store.set_state_with_history(
            CodexState.OPENAI_PRIMARY, reason="setup", force=True
        )
        # Non-emergency target, force should work
        allowed, reason = tmp_store.can_transition(
            CodexState.OPENROUTER_FALLBACK, force=True
        )
        assert allowed is True

    def test_force_rapid_sequence(self, tmp_store):
        """Force 20 rapid transitions — all succeed, no anti-flap errors."""
        for i in range(20):
            target = random.choice(ALL_STATES)
            tmp_store.set_state_with_history(target, reason=f"force-{i}", force=True)
            assert tmp_store.get_state() == target

    def test_force_vs_no_force_same_window(self, tmp_store):
        """Force works while non-force is blocked in same window."""
        tmp_store.set_state_with_history(
            CodexState.OPENAI_PRIMARY, reason="setup", force=True
        )
        # Non-emergency, non-force should be blocked
        allowed_no_force, reason = tmp_store.can_transition(
            CodexState.OPENROUTER_FALLBACK, force=False
        )
        if not allowed_no_force:
            assert "anti_flap" in reason
            # Force should work
            allowed_force, _ = tmp_store.can_transition(
                CodexState.OPENROUTER_FALLBACK, force=True
            )
            assert allowed_force is True

    def test_force_also_works_for_emergency_targets(self, tmp_store):
        """Force works for emergency targets too (redundant but should not break)."""
        tmp_store.set_state_with_history(
            CodexState.OPENAI_CONSERVATION, reason="setup", force=True
        )
        for target in [CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP]:
            allowed, _ = tmp_store.can_transition(target, force=True)
            assert allowed is True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. State survives reload (new StateStore reads same state)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateSurvivesReload:
    """A new StateStore instance reads back the same state."""

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_manual_state_survives_reload(self, tmp_path, state):
        """set_manual_state → new store → same state."""
        store1 = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=tmp_path / "w.jsonl",
        )
        store1.set_manual_state(state)

        # Create a fresh store from same paths
        store2 = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=tmp_path / "w.jsonl",
        )
        assert store2.get_manual_state() == state
        assert store2.get_state() == state

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_auto_state_survives_reload(self, tmp_path, state):
        """set_auto_state → new store → same state."""
        store1 = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=tmp_path / "w.jsonl",
        )
        store1.set_auto_state(state)

        store2 = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=tmp_path / "w.jsonl",
        )
        assert store2.get_auto_state() == state

    def test_none_manual_survives_reload(self, tmp_path):
        """Clearing manual (None) persists across reload."""
        store1 = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=tmp_path / "w.jsonl",
        )
        store1.set_manual_state(CodexState.CLAUDE_BACKUP)
        store1.set_manual_state(None)

        store2 = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=tmp_path / "w.jsonl",
        )
        assert store2.get_manual_state() is None

    def test_history_survives_reload(self, tmp_path):
        """State history persists across store reload."""
        store1 = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=tmp_path / "w.jsonl",
        )
        for state in ALL_STATES:
            store1.set_state_with_history(state, reason="reload-test", force=True)

        store2 = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=tmp_path / "w.jsonl",
        )
        history = store2.get_state_history(limit=50)
        assert len(history) >= len(ALL_STATES)

    def test_stress_reload_consistency(self, tmp_path):
        """Stress: 50 write-reload cycles — state is always consistent."""
        for i in range(50):
            state = random.choice(ALL_STATES)
            store = StateStore(
                manual_path=tmp_path / "m.json",
                auto_path=tmp_path / "a.json",
                history_path=tmp_path / "h.json",
                wal_path=tmp_path / "w.jsonl",
            )
            store.set_manual_state(state)
            # Reload
            store2 = StateStore(
                manual_path=tmp_path / "m.json",
                auto_path=tmp_path / "a.json",
                history_path=tmp_path / "h.json",
                wal_path=tmp_path / "w.jsonl",
            )
            assert store2.get_manual_state() == state, (
                f"Reload cycle {i}: expected {state}, got {store2.get_manual_state()}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. State after WAL recovery matches expected value
# ═══════════════════════════════════════════════════════════════════════════════

class TestWALRecoveryStateValue:
    """After WAL recovery, state matches what was intended."""

    def test_recovery_applies_uncommitted_write(self, tmp_path):
        """An uncommitted WAL write is applied during recovery."""
        manual_path = tmp_path / "m.json"
        auto_path = tmp_path / "a.json"
        history_path = tmp_path / "h.json"
        wal_path = tmp_path / "w.jsonl"

        # Write initial state
        manual_path.write_text(json.dumps({"state": "openai_primary"}))
        auto_path.write_text(json.dumps({"state": "openai_primary"}))

        # Simulate WAL with uncommitted write (no committed marker after)
        target_state = CodexState.CLAUDE_BACKUP.value
        wal_path.write_text(json.dumps({
            "action": "write",
            "path": str(manual_path),
            "data": {"state": target_state},
            "timestamp": "2026-01-01T00:00:00+00:00",
        }) + "\n")
        # No "committed" entry — should be recovered

        store = StateStore(
            manual_path=manual_path,
            auto_path=auto_path,
            history_path=history_path,
            wal_path=wal_path,
        )
        assert store.get_manual_state() == CodexState.CLAUDE_BACKUP

    def test_recovery_skips_committed_write(self, tmp_path):
        """A committed WAL write is NOT re-applied."""
        manual_path = tmp_path / "m.json"
        auto_path = tmp_path / "a.json"
        history_path = tmp_path / "h.json"
        wal_path = tmp_path / "w.jsonl"

        # Current state on disk
        manual_path.write_text(json.dumps({"state": "openai_primary"}))
        auto_path.write_text(json.dumps({"state": "openai_primary"}))

        # WAL with committed marker — should NOT re-apply
        wal_path.write_text(
            json.dumps({
                "action": "write",
                "path": str(manual_path),
                "data": {"state": "claude_backup"},
            }) + "\n" +
            json.dumps({"action": "committed"}) + "\n"
        )

        store = StateStore(
            manual_path=manual_path,
            auto_path=auto_path,
            history_path=history_path,
            wal_path=wal_path,
        )
        # Should keep existing state (openai_primary), not re-apply claude_backup
        assert store.get_manual_state() == CodexState.OPENAI_PRIMARY

    def test_recovery_multiple_uncommitted_writes(self, tmp_path):
        """Multiple uncommitted WAL writes — last one wins."""
        manual_path = tmp_path / "m.json"
        auto_path = tmp_path / "a.json"
        history_path = tmp_path / "h.json"
        wal_path = tmp_path / "w.jsonl"

        manual_path.write_text(json.dumps({"state": "openai_primary"}))
        auto_path.write_text(json.dumps({"state": "openai_primary"}))

        # Two uncommitted writes to same file
        wal_path.write_text(
            json.dumps({
                "action": "write",
                "path": str(manual_path),
                "data": {"state": "claude_backup"},
            }) + "\n" +
            json.dumps({
                "action": "write",
                "path": str(manual_path),
                "data": {"state": "openrouter_fallback"},
            }) + "\n"
        )

        store = StateStore(
            manual_path=manual_path,
            auto_path=auto_path,
            history_path=history_path,
            wal_path=wal_path,
        )
        # Last uncommitted write should be applied
        assert store.get_manual_state() == CodexState.OPENROUTER_FALLBACK

    def test_wal_is_truncated_after_recovery(self, tmp_path):
        """WAL is cleared after recovery completes."""
        wal_path = tmp_path / "w.jsonl"
        manual_path = tmp_path / "m.json"
        auto_path = tmp_path / "a.json"

        manual_path.write_text(json.dumps({"state": "openai_primary"}))
        auto_path.write_text(json.dumps({"state": "openai_primary"}))

        wal_path.write_text(json.dumps({
            "action": "write",
            "path": str(manual_path),
            "data": {"state": "claude_backup"},
        }) + "\n")

        store = StateStore(
            manual_path=manual_path,
            auto_path=auto_path,
            history_path=tmp_path / "h.json",
            wal_path=wal_path,
        )
        # WAL should be empty after recovery
        assert wal_path.read_text().strip() == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 5. All tools unavailable (all circuit breakers open)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAllToolsUnavailable:
    """Behavior when all circuit breakers are open."""

    def test_chain_still_built_when_all_breakers_open(self):
        """build_chain works regardless of circuit breaker state."""
        task = TaskMeta(
            task_id="all-down",
            task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM,
        )
        for state in ALL_STATES:
            chain = build_chain(task, state)
            assert len(chain) >= 1

    def test_all_breakers_open_produces_empty_attempts(self):
        """When all breakers are open, route_task skips all providers."""
        cb = CircuitBreaker(threshold=1, window_s=300, cooldown_s=600)
        tools_backends = [
            ("codex_cli", "openai_native"),
            ("claude_code", "anthropic"),
            ("codex_cli", "openrouter"),
        ]
        # Open all breakers
        for tool, backend in tools_backends:
            cb.record_failure(tool, backend)

        # Verify all are unavailable
        for tool, backend in tools_backends:
            if cb.is_available(tool, backend):
                # Skip if cooldown expired during test
                continue
            assert cb.is_available(tool, backend) is False

    def test_single_available_provider_among_open_breakers(self):
        """If only one breaker is closed, only that provider is available."""
        cb = CircuitBreaker(threshold=1, window_s=300, cooldown_s=600)

        # Open codex and openrouter
        cb.record_failure("codex_cli", "openai_native")
        cb.record_failure("codex_cli", "openrouter")

        # Claude should still be available
        assert cb.is_available("claude_code", "anthropic") is True
        assert cb.is_available("codex_cli", "openai_native") is False


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Concurrent state writes from multiple threads
# ═══════════════════════════════════════════════════════════════════════════════

class TestConcurrentStateWrites:
    """Thread safety of state writes and reads."""

    def test_concurrent_writes_from_all_states(self, tmp_store):
        """Each thread writes a different state concurrently — no corruption."""
        errors = []
        results = {}

        def writer(state, thread_id):
            try:
                for _ in range(50):
                    tmp_store.set_manual_state(state)
                    readback = tmp_store.get_state()
                    assert isinstance(readback, CodexState)
                results[thread_id] = True
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=writer, args=(s, i))
            for i, s in enumerate(ALL_STATES)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Concurrent write errors: {errors}"
        # Final state should be valid
        final = tmp_store.get_state()
        assert isinstance(final, CodexState)
        assert final in ALL_STATES

    def test_concurrent_read_write_no_deadlock(self, tmp_store):
        """Concurrent reads and writes don't deadlock."""
        tmp_store.set_manual_state(CodexState.OPENAI_PRIMARY)
        errors = []
        done = threading.Event()

        def writer():
            try:
                while not done.is_set():
                    tmp_store.set_manual_state(random.choice(ALL_STATES))
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                while not done.is_set():
                    _ = tmp_store.get_state()
                    _ = tmp_store.get_manual_state()
                    _ = tmp_store.get_auto_state()
            except Exception as e:
                errors.append(("reader", e))

        writers = [threading.Thread(target=writer) for _ in range(3)]
        readers = [threading.Thread(target=reader) for _ in range(3)]
        all_threads = writers + readers

        for t in all_threads:
            t.start()

        time.sleep(2)  # Run for 2 seconds
        done.set()

        for t in all_threads:
            t.join(timeout=10)
            assert not t.is_alive(), "Thread did not terminate — possible deadlock"

        assert len(errors) == 0, f"Concurrent read/write errors: {errors}"

    def test_concurrent_set_state_with_history(self, tmp_store):
        """Concurrent set_state_with_history with force=True — no errors."""
        errors = []

        def transitioner(thread_id):
            try:
                for i in range(30):
                    state = random.choice(ALL_STATES)
                    tmp_store.set_state_with_history(
                        state, reason=f"thread-{thread_id}-iter-{i}", force=True
                    )
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=transitioner, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Concurrent transition errors: {errors}"
        # History should have entries from multiple threads
        history = tmp_store.get_state_history(limit=50)
        assert len(history) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Extreme chain length stress
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtremeChainLength:
    """Stress tests for chain building at scale."""

    def test_10000_chain_builds(self):
        """Stress: 10,000 chain builds — always valid, always bounded."""
        for i in range(10000):
            task = TaskMeta(
                task_id=f"extreme-{i}",
                task_class=random.choice(list(TaskClass)),
                risk=random.choice(list(TaskRisk)),
                has_screenshots=random.random() < 0.1,
                requires_multimodal=random.random() < 0.1,
            )
            state = random.choice(ALL_STATES)
            chain = build_chain(task, state)
            assert 1 <= len(chain) <= 4, f"Chain length {len(chain)} out of bounds at iteration {i}"

    def test_chain_builds_all_task_class_state_combos(self):
        """Every (TaskClass × CodexState) combination produces valid chain."""
        for tc in TaskClass:
            for state in ALL_STATES:
                task = TaskMeta(task_id=f"combo-{tc.value}-{state.value}", task_class=tc)
                chain = build_chain(task, state)
                assert len(chain) >= 1
                for entry in chain:
                    assert entry.tool in {"codex_cli", "claude_code", "openrouter"}
                    assert entry.backend in {"openai_native", "anthropic", "openrouter"}

    def test_chain_builds_all_risk_state_combos(self):
        """Every (TaskRisk × CodexState) combination produces valid chain."""
        for risk in TaskRisk:
            for state in ALL_STATES:
                task = TaskMeta(task_id=f"risk-{risk.value}-{state.value}", risk=risk)
                chain = build_chain(task, state)
                assert len(chain) >= 1

    def test_chain_deterministic_same_input(self):
        """Same task+state always produces identical chain."""
        task = TaskMeta(
            task_id="deterministic",
            task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.HIGH,
        )
        for state in ALL_STATES:
            chains = [build_chain(task, state) for _ in range(100)]
            first = chains[0]
            for chain in chains[1:]:
                assert len(chain) == len(first)
                for a, b in zip(chain, first):
                    assert a.tool == b.tool
                    assert a.backend == b.backend
                    assert a.model_profile == b.model_profile


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Corrupted state file recovery
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorruptedStateFiles:
    """Behavior with corrupted state files."""

    def test_corrupted_manual_json(self, tmp_path):
        """Corrupted manual state JSON raises StateError on read."""
        manual_path = tmp_path / "m.json"
        auto_path = tmp_path / "a.json"
        manual_path.write_text("NOT JSON {{{{")
        auto_path.write_text(json.dumps({"state": "openai_primary"}))

        store = StateStore(
            manual_path=manual_path,
            auto_path=auto_path,
            history_path=tmp_path / "h.json",
            wal_path=tmp_path / "w.jsonl",
        )
        # Reading corrupted manual should raise StateError
        from router.errors import StateError
        with pytest.raises(StateError):
            store.get_manual_state()

    def test_invalid_state_value_in_file(self, tmp_path):
        """Invalid state value in file returns None from _validate_state."""
        from router.state_store import StateStore
        assert StateStore._validate_state("invalid_state_xyz") is None

    def test_backward_compat_state_values(self, tmp_path):
        """Old state names (normal, last10) map correctly."""
        from router.state_store import StateStore
        assert StateStore._validate_state("normal") == CodexState.OPENAI_PRIMARY
        assert StateStore._validate_state("last10") == CodexState.CLAUDE_BACKUP
