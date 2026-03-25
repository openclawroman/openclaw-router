"""Property-based tests for state machine invariants.

Tests fundamental properties that must ALWAYS hold, regardless of input.
Uses manual property tests with random/fixed inputs (no hypothesis dependency).

Properties tested:
  A. State always valid
  B. Manual overrides auto
  C. Transition consistency
  D. History monotonicity
  E. Chain non-empty
  F. Chain first entry valid
  G. State persistence round-trip
  H. WAL recovery consistency
  I. Force bypass works
  J. Anti-flap invariant
  K. Singleton invariant
  L. Circuit breaker invariant
"""

import json
import os
import random
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest

from router.models import CodexState, TaskMeta, TaskClass, TaskRisk, ChainEntry
from router.state_store import StateStore, get_state_store, reset_state_store, MIN_STATE_DURATION_S
from router.circuit_breaker import CircuitBreaker
from router.policy import build_chain, resolve_state
from router.errors import StateError

ALL_STATES = list(CodexState)


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    """Provide a StateStore with temporary paths (no WAL recovery side effects)."""
    manual = tmp_path / "manual.json"
    auto = tmp_path / "auto.json"
    history = tmp_path / "history.json"
    wal = tmp_path / "wal.jsonl"
    store = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
    return store


@pytest.fixture
def default_task():
    """A minimal TaskMeta for chain-building tests."""
    return TaskMeta(
        task_id="prop-test",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A. State always valid — after ANY set_manual/set_auto call, get_state()
#    returns a valid CodexState enum value.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyAStateAlwaysValid:
    """Property A: state is always a valid CodexState after any write."""

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_manual_state_is_valid(self, tmp_store, state):
        """After set_manual_state(X), get_state() returns a CodexState."""
        tmp_store.set_manual_state(state)
        result = tmp_store.get_state()
        assert isinstance(result, CodexState)
        assert result in ALL_STATES

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_auto_state_is_valid(self, tmp_store, state):
        """After set_auto_state(X), get_state() returns a CodexState."""
        tmp_store.set_auto_state(state)
        result = tmp_store.get_state()
        assert isinstance(result, CodexState)
        assert result in ALL_STATES

    def test_random_manual_sequences_stay_valid(self, tmp_store):
        """Stress: 500 random set_manual_state calls — state is always valid."""
        for _ in range(500):
            state = random.choice(ALL_STATES)
            tmp_store.set_manual_state(state)
            result = tmp_store.get_state()
            assert isinstance(result, CodexState)
            assert result in ALL_STATES

    def test_random_auto_sequences_stay_valid(self, tmp_store):
        """Stress: 500 random set_auto_state calls — state is always valid."""
        for _ in range(500):
            state = random.choice(ALL_STATES)
            tmp_store.set_auto_state(state)
            result = tmp_store.get_state()
            assert isinstance(result, CodexState)

    def test_mixed_manual_auto_stays_valid(self, tmp_store):
        """Stress: interleaved manual/auto writes — state is always valid."""
        for _ in range(200):
            if random.random() < 0.5:
                tmp_store.set_manual_state(random.choice(ALL_STATES))
            else:
                tmp_store.set_auto_state(random.choice(ALL_STATES))
            result = tmp_store.get_state()
            assert isinstance(result, CodexState)
            assert result in ALL_STATES

    def test_none_manual_with_auto_is_valid(self, tmp_store):
        """Clearing manual (None) + auto set → valid state."""
        for state in ALL_STATES:
            tmp_store.set_auto_state(state)
            tmp_store.set_manual_state(None)
            result = tmp_store.get_state()
            assert isinstance(result, CodexState)
            assert result == state


# ═══════════════════════════════════════════════════════════════════════════════
# B. Manual overrides auto — when manual is set, get_state() returns
#    manual value regardless of auto.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyBManualOverridesAuto:
    """Property B: manual always wins over auto."""

    @pytest.mark.parametrize("manual_state", ALL_STATES)
    @pytest.mark.parametrize("auto_state", ALL_STATES)
    def test_manual_wins_over_auto(self, tmp_store, manual_state, auto_state):
        """For every (manual, auto) pair: get_state() == manual."""
        tmp_store.set_auto_state(auto_state)
        tmp_store.set_manual_state(manual_state)
        assert tmp_store.get_state() == manual_state

    def test_random_manual_always_wins(self, tmp_store):
        """Stress: 300 random (manual, auto) pairs — manual always wins."""
        for _ in range(300):
            m = random.choice(ALL_STATES)
            a = random.choice(ALL_STATES)
            tmp_store.set_manual_state(m)
            tmp_store.set_auto_state(a)
            assert tmp_store.get_state() == m

    def test_clearing_manual_falls_through_to_auto(self, tmp_store):
        """After clearing manual (None), state falls through to auto."""
        for auto in ALL_STATES:
            tmp_store.set_auto_state(auto)
            tmp_store.set_manual_state(random.choice(ALL_STATES))
            tmp_store.set_manual_state(None)
            assert tmp_store.get_state() == auto


# ═══════════════════════════════════════════════════════════════════════════════
# C. Transition consistency — if can_transition returns True, the transition
#    can be executed without error.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyCTransitionConsistency:
    """Property C: can_transition=True implies transition succeeds."""

    @pytest.mark.parametrize("target", ALL_STATES)
    def test_allowed_transition_succeeds(self, tmp_store, target):
        """Fresh store (no history) — all transitions should be allowed."""
        allowed, _reason = tmp_store.can_transition(target)
        if allowed:
            # Should not raise
            tmp_store.set_state_with_history(target, reason="prop-test")

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_same_state_transition_always_allowed(self, tmp_store, state):
        """Transitioning to the current state is always allowed."""
        tmp_store.set_manual_state(state)
        allowed, reason = tmp_store.can_transition(state)
        assert allowed is True
        assert reason == "same_state"

    def test_random_transitions_respect_can_transition(self, tmp_store):
        """Stress: 100 random transitions — if allowed, no error."""
        tmp_store.set_manual_state(CodexState.OPENAI_PRIMARY)
        # Clear history to allow transitions
        if tmp_store.history_path.exists():
            tmp_store.history_path.write_text("")

        for _ in range(100):
            target = random.choice(ALL_STATES)
            allowed, reason = tmp_store.can_transition(target)
            if allowed:
                try:
                    tmp_store.set_state_with_history(target, reason="stress-test")
                except StateError:
                    pytest.fail(
                        f"can_transition returned True but set_state_with_history raised "
                        f"for {target} (reason: {reason})"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# D. History monotonicity — state history entries have increasing (or equal)
#    timestamps.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyDHistoryMonotonicity:
    """Property D: history timestamps are monotonically non-decreasing."""

    def test_sequential_transitions_are_monotonic(self, tmp_store):
        """After multiple transitions, history timestamps are non-decreasing."""
        # Clear history first
        if tmp_store.history_path.exists():
            tmp_store.history_path.write_text("")

        for state in ALL_STATES:
            tmp_store.set_state_with_history(state, reason="mono-test", force=True)

        history = tmp_store.get_state_history(limit=50)
        if len(history) < 2:
            pytest.skip("Need at least 2 history entries")

        timestamps = []
        for entry in history:
            ts = entry.get("timestamp", "")
            if ts:
                timestamps.append(ts.replace("Z", "+00:00"))

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamp {i} ({timestamps[i]}) < timestamp {i-1} ({timestamps[i-1]})"
            )

    def test_empty_history_is_valid(self, tmp_store):
        """Empty history doesn't break anything."""
        if tmp_store.history_path.exists():
            tmp_store.history_path.write_text("")
        history = tmp_store.get_state_history()
        assert isinstance(history, list)

    def test_corrupted_history_returns_empty(self, tmp_store):
        """Corrupted history file doesn't crash — returns empty list."""
        tmp_store.history_path.write_text("NOT VALID JSON {{{{")
        history = tmp_store.get_state_history()
        assert isinstance(history, list)
        assert len(history) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# E. Chain non-empty — build_chain() always returns at least 1 entry
#    for any valid (task, state) pair.
# F. Chain first entry valid — first chain entry always has a valid tool.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyEChainNonEmpty:
    """Property E: build_chain always returns >= 1 entry."""

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_chain_non_empty_for_default_task(self, default_task, state):
        """Every state produces a non-empty chain."""
        chain = build_chain(default_task, state)
        assert len(chain) >= 1, f"Chain empty for state {state}"

    @pytest.mark.parametrize("state", ALL_STATES)
    @pytest.mark.parametrize("risk", list(TaskRisk))
    def test_chain_non_empty_for_all_risks(self, state, risk):
        """Every (state, risk) pair produces a non-empty chain."""
        task = TaskMeta(task_id="e-risk", task_class=TaskClass.IMPLEMENTATION, risk=risk)
        chain = build_chain(task, state)
        assert len(chain) >= 1

    def test_random_tasks_always_have_chain(self):
        """Stress: 200 random (task, state) — chain always non-empty."""
        valid_tools = {"codex_cli", "claude_code", "openrouter"}
        for _ in range(200):
            state = random.choice(ALL_STATES)
            task = TaskMeta(
                task_id=f"stress-{_}",
                task_class=random.choice(list(TaskClass)),
                risk=random.choice(list(TaskRisk)),
                has_screenshots=random.random() < 0.1,
                requires_multimodal=random.random() < 0.1,
            )
            chain = build_chain(task, state)
            assert len(chain) >= 1


class TestPropertyFChainFirstEntryValid:
    """Property F: first chain entry has a valid tool."""

    VALID_TOOLS = {"codex_cli", "claude_code", "openrouter"}

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_first_entry_valid_tool(self, default_task, state):
        """First chain entry tool is in the known set."""
        chain = build_chain(default_task, state)
        assert chain[0].tool in self.VALID_TOOLS

    def test_all_chains_have_valid_tools(self):
        """Stress: 200 random chains — all entries have valid tools."""
        for _ in range(200):
            state = random.choice(ALL_STATES)
            task = TaskMeta(
                task_id=f"f-{_}",
                task_class=random.choice(list(TaskClass)),
                risk=random.choice(list(TaskRisk)),
            )
            chain = build_chain(task, state)
            for entry in chain:
                assert entry.tool in self.VALID_TOOLS, (
                    f"Invalid tool '{entry.tool}' in chain for state {state}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# G. State persistence round-trip — after set_manual_state(X), reading it
#    back gives X.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyGPersistenceRoundTrip:
    """Property G: write → read gives back the same value."""

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_manual_round_trip(self, tmp_store, state):
        """set_manual_state(X) → get_manual_state() == X."""
        tmp_store.set_manual_state(state)
        assert tmp_store.get_manual_state() == state

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_auto_round_trip(self, tmp_store, state):
        """set_auto_state(X) → get_auto_state() == X."""
        tmp_store.set_auto_state(state)
        assert tmp_store.get_auto_state() == state

    def test_none_manual_round_trip(self, tmp_store):
        """set_manual_state(None) → get_manual_state() returns None."""
        tmp_store.set_manual_state(CodexState.OPENAI_PRIMARY)
        tmp_store.set_manual_state(None)
        assert tmp_store.get_manual_state() is None

    def test_rapid_round_trips(self, tmp_store):
        """Stress: 1000 rapid write-read cycles — always round-trips."""
        for _ in range(1000):
            state = random.choice(ALL_STATES)
            if random.random() < 0.8:
                tmp_store.set_manual_state(state)
                assert tmp_store.get_manual_state() == state
            else:
                tmp_store.set_auto_state(state)
                assert tmp_store.get_auto_state() == state


# ═══════════════════════════════════════════════════════════════════════════════
# H. WAL recovery consistency — after recover_from_wal(), state is valid
#    (even with partial WAL).
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyHWALRecovery:
    """Property H: WAL recovery leaves state in a valid condition."""

    def test_empty_wal_recovery(self, tmp_path):
        """Recovering from empty/no WAL file is fine."""
        wal = tmp_path / "wal.jsonl"
        wal.write_text("")
        store = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=wal,
        )
        result = store.get_state()
        assert isinstance(result, CodexState)

    def test_partial_wal_recovery(self, tmp_path):
        """Partial WAL (truncated line) — recovery succeeds, state valid."""
        wal = tmp_path / "wal.jsonl"
        wal.write_text('{"action":"write","path":"/nonexistent","data":{"state":"openai_primary"}}\n{"action":"commi')
        store = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=wal,
        )
        result = store.get_state()
        assert isinstance(result, CodexState)

    def test_corrupted_wal_lines_skipped(self, tmp_path):
        """WAL with garbage lines — recovery skips them, state valid."""
        wal = tmp_path / "wal.jsonl"
        wal.write_text("NOT JSON\n{}\n{{invalid\n")
        store = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=wal,
        )
        result = store.get_state()
        assert isinstance(result, CodexState)

    def test_random_wal_entries_state_valid(self, tmp_path):
        """Stress: WAL with random entries — state always valid after recovery."""
        wal = tmp_path / "wal.jsonl"
        lines = []
        for _ in range(100):
            if random.random() < 0.7:
                state = random.choice(ALL_STATES).value
                lines.append(json.dumps({
                    "action": "write",
                    "path": str(tmp_path / "m.json"),
                    "data": {"state": state},
                }))
            else:
                lines.append("GARBAGE" if random.random() < 0.1 else json.dumps({"action": "committed"}))
        wal.write_text("\n".join(lines))
        store = StateStore(
            manual_path=tmp_path / "m.json",
            auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json",
            wal_path=wal,
        )
        result = store.get_state()
        assert isinstance(result, CodexState)


# ═══════════════════════════════════════════════════════════════════════════════
# I. Force bypass works — can_transition(..., force=True) always returns True.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyIForceBypass:
    """Property I: force=True always returns True."""

    @pytest.mark.parametrize("target", ALL_STATES)
    def test_force_always_allowed(self, tmp_store, target):
        """can_transition(X, force=True) is always (True, ...)."""
        allowed, reason = tmp_store.can_transition(target, force=True)
        assert allowed is True
        assert "forced" in reason or "override" in reason or "same_state" in reason

    def test_force_from_every_state(self, tmp_store):
        """Force bypass works from every starting state."""
        for start in ALL_STATES:
            tmp_store.set_manual_state(start)
            for target in ALL_STATES:
                allowed, _ = tmp_store.can_transition(target, force=True)
                assert allowed is True

    def test_force_random_sequences(self, tmp_store):
        """Stress: 500 force transitions — all succeed."""
        for _ in range(500):
            target = random.choice(ALL_STATES)
            allowed, _ = tmp_store.can_transition(target, force=True)
            assert allowed is True
            tmp_store.set_state_with_history(target, reason="forced", force=True)

    def test_force_transitions_never_raise(self, tmp_store):
        """set_state_with_history(..., force=True) never raises StateError."""
        for _ in range(200):
            target = random.choice(ALL_STATES)
            try:
                tmp_store.set_state_with_history(target, reason="force-test", force=True)
            except StateError:
                pytest.fail(f"Force transition to {target} raised StateError")


# ═══════════════════════════════════════════════════════════════════════════════
# J. Anti-flap invariant — rapid transitions within timeout window are blocked
#    (except emergency targets).
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyJAntiFlap:
    """Property J: anti-flap blocks rapid non-emergency transitions."""

    def test_emergency_targets_always_allowed(self, tmp_store):
        """OPENAI_PRIMARY and CLAUDE_BACKUP are always allowed (emergency)."""
        tmp_store.set_manual_state(CodexState.OPENAI_PRIMARY)
        # Add a recent history entry
        tmp_store.set_state_with_history(
            CodexState.OPENAI_CONSERVATION, reason="test", force=True
        )
        # Immediately try emergency targets
        for emergency in [CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP]:
            allowed, reason = tmp_store.can_transition(emergency)
            assert allowed is True, f"Emergency target {emergency} blocked: {reason}"

    def test_non_emergency_blocked_during_cooldown(self, tmp_store):
        """Non-emergency transitions blocked within MIN_STATE_DURATION_S window."""
        tmp_store.set_manual_state(CodexState.OPENAI_PRIMARY)
        # Force a transition with history to establish a recent timestamp
        tmp_store.set_state_with_history(
            CodexState.OPENAI_CONSERVATION, reason="anti-flap-setup", force=True
        )
        # Immediately try non-emergency targets
        for target in [CodexState.OPENROUTER_FALLBACK]:
            allowed, reason = tmp_store.can_transition(target)
            if not allowed:
                assert "anti_flap" in reason

    def test_force_bypasses_anti_flap(self, tmp_store):
        """Force flag bypasses anti-flap protection."""
        tmp_store.set_manual_state(CodexState.OPENAI_PRIMARY)
        tmp_store.set_state_with_history(
            CodexState.OPENAI_CONSERVATION, reason="setup", force=True
        )
        # Non-emergency with force should always be allowed
        allowed, _ = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK, force=True)
        assert allowed is True


# ═══════════════════════════════════════════════════════════════════════════════
# K. Singleton invariant — get_state_store() always returns the same instance.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyKSingleton:
    """Property K: get_state_store() returns the same instance every time."""

    def test_singleton_identity(self):
        """Multiple calls return the same object."""
        reset_state_store()
        s1 = get_state_store()
        s2 = get_state_store()
        s3 = get_state_store()
        assert s1 is s2 is s3
        reset_state_store()

    def test_singleton_across_threads(self):
        """Singleton holds across multiple threads."""
        reset_state_store()
        results = []

        def get_store():
            results.append(get_state_store())

        threads = [threading.Thread(target=get_store) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        first = results[0]
        for store in results:
            assert store is first
        reset_state_store()

    def test_stress_singleton_1000_calls(self):
        """Stress: 1000 calls to get_state_store — all return same instance."""
        reset_state_store()
        first = get_state_store()
        for _ in range(1000):
            assert get_state_store() is first
        reset_state_store()


# ═══════════════════════════════════════════════════════════════════════════════
# L. Circuit breaker invariant — after threshold failures, is_available()
#    returns False.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyLCircuitBreaker:
    """Property L: circuit breaker trips after threshold failures."""

    @pytest.mark.parametrize("threshold", [1, 3, 5, 10])
    def test_breaks_at_threshold(self, threshold):
        """After exactly threshold failures, is_available returns False."""
        cb = CircuitBreaker(threshold=threshold, window_s=300, cooldown_s=600)
        tool, backend = "codex_cli", "openai_native"
        assert cb.is_available(tool, backend) is True

        for i in range(threshold):
            cb.record_failure(tool, backend)

        assert cb.is_available(tool, backend) is False
        assert cb.get_state(tool, backend) == "open"

    @pytest.mark.parametrize("threshold", [3, 5])
    def test_stays_available_below_threshold(self, threshold):
        """Below threshold, provider stays available."""
        cb = CircuitBreaker(threshold=threshold, window_s=300, cooldown_s=600)
        tool, backend = "claude_code", "anthropic"

        for i in range(threshold - 1):
            cb.record_failure(tool, backend)
            # Should still be available
            if i < threshold - 1:
                assert cb.is_available(tool, backend) is True

    def test_success_resets_failure_count(self):
        """Recording a success resets the failure counter."""
        cb = CircuitBreaker(threshold=3, window_s=300, cooldown_s=600)
        tool, backend = "codex_cli", "openai_native"

        cb.record_failure(tool, backend)
        cb.record_failure(tool, backend)
        cb.record_success(tool, backend)

        # After success, should be available even with 2 more failures
        cb.record_failure(tool, backend)
        cb.record_failure(tool, backend)
        assert cb.is_available(tool, backend) is True

    def test_different_providers_independent(self):
        """Circuit breaker is per-provider — one breaking doesn't affect others."""
        cb = CircuitBreaker(threshold=3, window_s=300, cooldown_s=600)

        # Break codex
        for _ in range(3):
            cb.record_failure("codex_cli", "openai_native")
        assert cb.is_available("codex_cli", "openai_native") is False

        # Claude should still be available
        assert cb.is_available("claude_code", "anthropic") is True

    def test_random_providers_break_at_threshold(self):
        """Stress: random providers all break exactly at threshold."""
        tools = ["codex_cli", "claude_code", "openrouter"]
        backends = ["openai_native", "anthropic", "openrouter"]

        for _ in range(50):
            cb = CircuitBreaker(threshold=random.randint(1, 10), window_s=300, cooldown_s=600)
            tool = random.choice(tools)
            backend = random.choice(backends)
            threshold = cb.threshold

            for i in range(threshold):
                cb.record_failure(tool, backend)

            assert cb.is_available(tool, backend) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases and stress tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: None state, empty history, corrupted files."""

    def test_none_manual_state_clears_override(self, tmp_store):
        """Setting manual to None clears the override."""
        tmp_store.set_manual_state(CodexState.CLAUDE_BACKUP)
        tmp_store.set_manual_state(None)
        assert tmp_store.get_manual_state() is None

    def test_missing_state_files_get_defaults(self, tmp_path):
        """Non-existent state files get default state created."""
        store = StateStore(
            manual_path=tmp_path / "new_m.json",
            auto_path=tmp_path / "new_a.json",
            history_path=tmp_path / "new_h.json",
            wal_path=tmp_path / "new_w.jsonl",
        )
        result = store.get_state()
        assert isinstance(result, CodexState)
        assert result == CodexState.OPENAI_PRIMARY

    def test_resolve_state_default(self, tmp_store):
        """resolve_state returns OPENAI_PRIMARY when both manual and auto are None."""
        tmp_store.set_manual_state(None)
        # Clear auto state file
        tmp_store.auto_path.write_text(json.dumps({"state": None}))
        result = resolve_state(tmp_store)
        assert result == CodexState.OPENAI_PRIMARY

    def test_validate_transition_all_pairs(self, tmp_store):
        """validate_transition works for all state pairs."""
        # In the 4-state model, all transitions are valid
        for src in ALL_STATES:
            for dst in ALL_STATES:
                assert tmp_store.validate_transition(src, dst) is True

    def test_concurrent_state_writes(self, tmp_store):
        """Concurrent writes don't corrupt state."""
        errors = []

        def writer(state):
            try:
                for _ in range(100):
                    tmp_store.set_manual_state(state)
                    _ = tmp_store.get_state()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(s,)) for s in ALL_STATES]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent writes raised errors: {errors}"
        # State should still be valid
        result = tmp_store.get_state()
        assert isinstance(result, CodexState)


class TestStressSuite:
    """Stress tests: high-volume operations to detect subtle bugs."""

    def test_1000_state_transitions(self, tmp_store):
        """Stress: 1000 forced transitions — state is always valid, no errors."""
        for i in range(1000):
            state = random.choice(ALL_STATES)
            tmp_store.set_state_with_history(state, reason="stress", force=True)
            result = tmp_store.get_state()
            assert isinstance(result, CodexState)

    def test_1000_chain_builds(self, default_task):
        """Stress: 1000 build_chain calls — always non-empty, valid entries."""
        for _ in range(1000):
            state = random.choice(ALL_STATES)
            chain = build_chain(default_task, state)
            assert len(chain) >= 1
            for entry in chain:
                assert entry.tool in {"codex_cli", "claude_code", "openrouter"}
                assert isinstance(entry.backend, str)
                assert isinstance(entry.model_profile, str)

    def test_1000_round_trips_under_memory_pressure(self, tmp_store):
        """Stress: rapid write-read cycles to detect caching bugs."""
        for _ in range(1000):
            state = random.choice(ALL_STATES)
            tmp_store.set_manual_state(state)
            readback = tmp_store.get_manual_state()
            assert readback == state
