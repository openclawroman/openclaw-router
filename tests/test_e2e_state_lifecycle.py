"""Comprehensive E2E tests for the complete state lifecycle.

Covers: state transitions, manual/auto precedence, WAL crash recovery,
sticky anti-flap protection, success-count threshold, state history,
and circuit breaker integration.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    CodexState, TaskMeta, TaskClass, TaskRisk, TaskModality,
    ExecutorResult, ChainEntry,
)
from router.state_store import StateStore, MIN_STATE_DURATION_S, CONSECUTIVE_SUCCESSES_THRESHOLD
from router.circuit_breaker import CircuitBreaker
from router.errors import StateError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> StateStore:
    """Create a StateStore with isolated temp paths."""
    return StateStore(
        manual_path=tmp_path / "manual.json",
        auto_path=tmp_path / "auto.json",
        history_path=tmp_path / "history.json",
        wal_path=tmp_path / "wal.jsonl",
    )


def _make_task(task_id: str = "e2e-001", risk: TaskRisk = TaskRisk.LOW) -> TaskMeta:
    return TaskMeta(
        task_id=task_id,
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=risk,
        modality=TaskModality.TEXT,
    )


def _advance_time(history_path: Path, seconds: float):
    """Rewrite history entries to be older, bypassing anti-flap."""
    if not history_path.exists():
        return
    data = json.loads(history_path.read_text())
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    for entry in data.get("transitions", []):
        entry["timestamp"] = old_ts
    history_path.write_text(json.dumps(data, indent=2))


# ===================================================================
# 1. Full State Cycle
# ===================================================================

class TestFullStateCycle:
    """Exercise the complete 4-state transition cycle."""

    def test_full_state_cycle(self, tmp_path):
        """openai_primary → openai_conservation → claude_backup → openrouter_fallback → openai_primary."""
        store = _make_store(tmp_path)

        # Start at default
        assert store.get_state() == CodexState.OPENAI_PRIMARY

        # Transition through all states
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION

        store.set_auto_state(CodexState.CLAUDE_BACKUP)
        store.set_manual_state(None)
        assert store.get_state() == CodexState.CLAUDE_BACKUP

        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)
        store.set_manual_state(None)
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

        # Cycle back to primary
        store.set_auto_state(CodexState.OPENAI_PRIMARY)
        store.set_manual_state(None)
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_full_state_cycle_with_transitions(self, tmp_path):
        """Full cycle using set_state_with_history with force to bypass anti-flap."""
        store = _make_store(tmp_path)
        states = [
            CodexState.OPENAI_CONSERVATION,
            CodexState.CLAUDE_BACKUP,
            CodexState.OPENROUTER_FALLBACK,
            CodexState.OPENAI_PRIMARY,
        ]
        for i, state in enumerate(states):
            store.set_state_with_history(state, reason=f"step_{i}", force=True)
            assert store.get_state() == state

        # History should have 4 entries
        history = store.get_state_history(limit=10)
        assert len(history) == 4


# ===================================================================
# 2-3. Manual Override Lifecycle & Precedence
# ===================================================================

class TestManualOverride:
    """Manual state takes precedence over auto and supports lifecycle."""

    def test_manual_override_lifecycle(self, tmp_path):
        """Set manual claude_backup → route task → clear manual → route task (different chain)."""
        store = _make_store(tmp_path)

        # Set manual override
        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store.get_state() == CodexState.CLAUDE_BACKUP

        # Auto state is ignored while manual is set
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        assert store.get_state() == CodexState.CLAUDE_BACKUP  # manual wins

        # Clear manual → auto takes over
        store.set_manual_state(None)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION

        # Set manual again to a different state
        store.set_manual_state(CodexState.OPENROUTER_FALLBACK)
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

        # Clear manual → auto still conservation
        store.set_manual_state(None)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION

    def test_manual_override_takes_precedence(self, tmp_path):
        """auto=conservation, manual=claude_backup → state is claude_backup."""
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        assert store.get_state() == CodexState.CLAUDE_BACKUP

    def test_manual_null_falls_through_to_auto(self, tmp_path):
        """When manual is null, auto state is used."""
        store = _make_store(tmp_path)
        store.set_manual_state(None)
        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)

        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_default_state_when_neither_set(self, tmp_path):
        """When both manual and auto are null/absent, default to openai_primary."""
        store = _make_store(tmp_path)
        store.set_manual_state(None)
        store.set_auto_state(CodexState.OPENAI_PRIMARY)
        store.set_manual_state(None)
        # Reset auto by overwriting with primary (the default)
        assert store.get_state() == CodexState.OPENAI_PRIMARY


# ===================================================================
# 4-5. State Persistence
# ===================================================================

class TestStatePersistence:
    """State survives StateStore recreation."""

    def test_auto_state_persists(self, tmp_path):
        """Set auto → new StateStore → state persists."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        history = tmp_path / "history.json"
        wal = tmp_path / "wal.jsonl"

        store1 = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        store1.set_manual_state(None)
        store1.set_auto_state(CodexState.OPENAI_CONSERVATION)
        assert store1.get_state() == CodexState.OPENAI_CONSERVATION

        # New store instance — should pick up persisted auto state
        store2 = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        store2.set_manual_state(None)
        assert store2.get_state() == CodexState.OPENAI_CONSERVATION

    def test_manual_state_persists(self, tmp_path):
        """Set manual → new StateStore → state persists."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        history = tmp_path / "history.json"
        wal = tmp_path / "wal.jsonl"

        store1 = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        store1.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store1.get_state() == CodexState.CLAUDE_BACKUP

        # New store instance
        store2 = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        assert store2.get_state() == CodexState.CLAUDE_BACKUP

    def test_manual_null_persists(self, tmp_path):
        """Clearing manual (setting to None) persists across restarts."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        history = tmp_path / "history.json"
        wal = tmp_path / "wal.jsonl"

        store1 = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        store1.set_manual_state(CodexState.CLAUDE_BACKUP)
        store1.set_manual_state(None)

        store2 = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        assert store2.get_manual_state() is None


# ===================================================================
# 6-7. WAL Crash Recovery
# ===================================================================

class TestWALRecovery:
    """Write-Ahead Log recovers state after simulated crashes."""

    def test_wal_recovery_after_crash(self, tmp_path):
        """Write state → simulate crash (corrupt primary) → recover_from_wal → state correct."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        history = tmp_path / "history.json"
        wal = tmp_path / "wal.jsonl"

        store = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        # Verify state was written
        assert store.get_state() == CodexState.CLAUDE_BACKUP

        # Simulate crash: corrupt the primary state file
        manual.write_text("CORRUPTED{{{invalid json")

        # Truncate WAL to simulate: write intent landed but committed marker did not
        # Re-write WAL with only the write intent (no committed marker)
        wal.write_text(json.dumps({
            "action": "write",
            "path": str(manual),
            "data": {"state": "claude_backup"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")

        # New store should recover from WAL
        store2 = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        assert store2.get_state() == CodexState.CLAUDE_BACKUP

    def test_wal_recovery_multiple_entries(self, tmp_path):
        """3 writes → crash → recover → last state wins."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        history = tmp_path / "history.json"
        wal = tmp_path / "wal.jsonl"

        # Simulate 3 uncommitted writes (write intents without committed markers)
        entries = [
            {"action": "write", "path": str(manual), "data": {"state": "openai_conservation"}, "timestamp": "t1"},
            {"action": "write", "path": str(manual), "data": {"state": "claude_backup"}, "timestamp": "t2"},
            {"action": "write", "path": str(manual), "data": {"state": "openrouter_fallback"}, "timestamp": "t3"},
        ]
        wal.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        # Corrupt primary
        manual.write_text("BROKEN")

        # Recover — last write wins
        store = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_wal_recovery_skips_committed_entries(self, tmp_path):
        """Committed WAL entries should not be re-applied."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        history = tmp_path / "history.json"
        wal = tmp_path / "wal.jsonl"

        # Write intent + committed marker — should NOT be re-applied
        entries = [
            {"action": "write", "path": str(manual), "data": {"state": "claude_backup"}, "timestamp": "t1"},
            {"action": "committed", "timestamp": "t1"},
        ]
        wal.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        # Set primary to a different state — recovery should not overwrite
        manual.write_text(json.dumps({"state": "openai_primary"}))

        store = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        # The committed entry should not re-apply
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_wal_recovery_with_malformed_lines(self, tmp_path):
        """WAL with corrupted lines should skip them and recover valid entries."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        history = tmp_path / "history.json"
        wal = tmp_path / "wal.jsonl"

        # Mix valid and invalid lines
        wal.write_text(
            "NOT JSON AT ALL\n"
            '{"action": "write", "path": "' + str(manual) + '", "data": {"state": "claude_backup"}}\n'
            "```\n"
            "null\n"
        )
        manual.write_text("CORRUPT")

        store = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        assert store.get_state() == CodexState.CLAUDE_BACKUP


# ===================================================================
# 8-11. Anti-Flap / Sticky State Protection
# ===================================================================

class TestAntiFlapProtection:
    """Anti-flap prevents rapid state transitions."""

    def test_sticky_state_blocks_rapid_transition(self, tmp_path):
        """Anti-flap blocks transitions within timeout."""
        store = _make_store(tmp_path)

        # Make a transition and log it
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        store.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "budget_low"
        )

        # Immediately try to transition to a non-emergency state — should be blocked
        allowed, reason = store.can_transition(CodexState.OPENROUTER_FALLBACK)
        assert allowed is False
        assert "anti_flap" in reason

    def test_sticky_state_allows_after_timeout(self, tmp_path):
        """Transition allowed after anti-flap timeout elapses."""
        store = _make_store(tmp_path)

        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        store.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "budget_low"
        )

        # Advance history timestamps past the minimum duration
        _advance_time(store.history_path, MIN_STATE_DURATION_S + 60)

        allowed, reason = store.can_transition(CodexState.OPENROUTER_FALLBACK)
        assert allowed is True

    def test_force_bypasses_sticky_state(self, tmp_path):
        """can_transition(force=True) always allowed regardless of anti-flap."""
        store = _make_store(tmp_path)

        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        store.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "budget_low"
        )

        # Without force — blocked
        allowed_no_force, _ = store.can_transition(CodexState.OPENROUTER_FALLBACK, force=False)

        # With force — always allowed
        allowed_force, reason_force = store.can_transition(CodexState.OPENROUTER_FALLBACK, force=True)
        assert allowed_force is True

    def test_emergency_override_always_allowed(self, tmp_path):
        """openai_primary and claude_backup bypass sticky anti-flap."""
        store = _make_store(tmp_path)

        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)
        store.set_manual_state(None)
        store.log_state_transition(
            CodexState.CLAUDE_BACKUP, CodexState.OPENROUTER_FALLBACK, "all_failed"
        )

        # Emergency targets should always be allowed
        allowed_primary, reason_p = store.can_transition(CodexState.OPENAI_PRIMARY)
        assert allowed_primary is True
        assert "emergency" in reason_p.lower()

        allowed_claude, reason_c = store.can_transition(CodexState.CLAUDE_BACKUP)
        assert allowed_claude is True
        assert "emergency" in reason_c.lower()

    def test_non_emergency_blocked_by_anti_flap(self, tmp_path):
        """OPENROUTER_FALLBACK is not an emergency target — blocked by anti-flap."""
        store = _make_store(tmp_path)

        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        store.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "test"
        )

        # openrouter_fallback is not emergency
        allowed, reason = store.can_transition(CodexState.OPENROUTER_FALLBACK)
        assert allowed is False
        assert "anti_flap" in reason

    def test_set_state_with_history_respects_anti_flap(self, tmp_path):
        """set_state_with_history without force raises StateError during anti-flap window."""
        store = _make_store(tmp_path)

        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        store.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "test"
        )

        with pytest.raises(StateError, match="anti_flap"):
            store.set_state_with_history(CodexState.OPENROUTER_FALLBACK, reason="test")

    def test_set_state_with_history_force_bypasses(self, tmp_path):
        """set_state_with_history with force=True bypasses anti-flap."""
        store = _make_store(tmp_path)

        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        store.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "test"
        )

        result = store.set_state_with_history(
            CodexState.OPENROUTER_FALLBACK, reason="forced", force=True
        )
        assert result is True
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK


# ===================================================================
# 12-13. Consecutive Successes Recovery
# ===================================================================

class TestConsecutiveSuccesses:
    """Success-count threshold for recovery to primary."""

    def test_consecutive_successes_recovery(self, tmp_path):
        """3 consecutive successes → can_recover_to_primary."""
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)

        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False

        for i in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)

        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True

    def test_consecutive_successes_resets_on_failure(self, tmp_path):
        """Failure resets success counter — need to rebuild from 0."""
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)

        # Get close to threshold
        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD - 1):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)

        # One failure resets
        store.record_success(CodexState.OPENAI_CONSERVATION, False)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False

        # Need full threshold again
        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True

    def test_successes_are_per_state(self, tmp_path):
        """Success counters are independent per state."""
        store = _make_store(tmp_path)

        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)

        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True
        assert store.can_recover_to_primary(CodexState.OPENROUTER_FALLBACK) is False

    def test_reset_success_counter(self, tmp_path):
        """reset_success_counter clears counter for specific state."""
        store = _make_store(tmp_path)

        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True

        store.reset_success_counter(CodexState.OPENAI_CONSERVATION)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False


# ===================================================================
# 14-15. State History
# ===================================================================

class TestStateHistory:
    """State transition history tracking and persistence."""

    def test_state_history_records_transitions(self, tmp_path):
        """3 transitions → history has 3 entries."""
        store = _make_store(tmp_path)

        store.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "budget_low")
        store.log_state_transition(CodexState.OPENAI_CONSERVATION, CodexState.CLAUDE_BACKUP, "quota_hit")
        store.log_state_transition(CodexState.CLAUDE_BACKUP, CodexState.OPENROUTER_FALLBACK, "all_failed")

        history = store.get_state_history(limit=10)
        assert len(history) == 3

        assert history[0]["from"] == "openai_primary"
        assert history[0]["to"] == "openai_conservation"
        assert history[0]["reason"] == "budget_low"

        assert history[1]["from"] == "openai_conservation"
        assert history[1]["to"] == "claude_backup"

        assert history[2]["from"] == "claude_backup"
        assert history[2]["to"] == "openrouter_fallback"

    def test_state_history_persists(self, tmp_path):
        """History survives StateStore recreation."""
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        history_path = tmp_path / "history.json"
        wal = tmp_path / "wal.jsonl"

        store1 = StateStore(manual_path=manual, auto_path=auto, history_path=history_path, wal_path=wal)
        store1.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP, "manual")
        store1.log_state_transition(CodexState.CLAUDE_BACKUP, CodexState.OPENAI_CONSERVATION, "recovery")

        # New store instance
        store2 = StateStore(manual_path=manual, auto_path=auto, history_path=history_path, wal_path=wal)
        history = store2.get_state_history(limit=10)
        assert len(history) == 2
        assert history[0]["to"] == "claude_backup"
        assert history[1]["to"] == "openai_conservation"

    def test_state_history_max_entries(self, tmp_path):
        """History is capped at MAX_HISTORY_ENTRIES."""
        store = _make_store(tmp_path)

        from router.state_store import MAX_HISTORY_ENTRIES
        # Write more than max
        for i in range(MAX_HISTORY_ENTRIES + 10):
            store.log_state_transition(
                CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, f"entry_{i}"
            )

        history = store.get_state_history(limit=MAX_HISTORY_ENTRIES + 20)
        assert len(history) == MAX_HISTORY_ENTRIES

        # Oldest entries should be dropped — first entry should be entry_10
        assert history[0]["reason"] == "entry_10"

    def test_state_history_empty_on_fresh_store(self, tmp_path):
        """Fresh store has empty history."""
        store = _make_store(tmp_path)
        assert store.get_state_history() == []


# ===================================================================
# 16-17. Circuit Breaker
# ===================================================================

class TestCircuitBreakerLifecycle:
    """Circuit breaker trip and recovery lifecycle."""

    def test_circuit_breaker_trip(self, tmp_path):
        """Provider fails threshold times → circuit breaker opens."""
        breaker = CircuitBreaker(threshold=3, window_s=60, cooldown_s=120)

        assert breaker.is_available("codex_cli", "openai_native") is True
        assert breaker.get_state("codex_cli", "openai_native") == "closed"

        # Record failures up to threshold
        for _ in range(3):
            breaker.record_failure("codex_cli", "openai_native", "rate_limited")

        assert breaker.get_state("codex_cli", "openai_native") == "open"
        assert breaker.is_available("codex_cli", "openai_native") is False

    def test_circuit_breaker_recovery(self, tmp_path):
        """After cooldown timeout → circuit breaker allows retry (half_open)."""
        breaker = CircuitBreaker(threshold=3, window_s=60, cooldown_s=0.1)

        for _ in range(3):
            breaker.record_failure("codex_cli", "openai_native", "provider_unavailable")

        assert breaker.is_available("codex_cli", "openai_native") is False
        assert breaker.get_state("codex_cli", "openai_native") == "open"

        # Wait for cooldown
        time.sleep(0.15)

        # is_available() triggers the open→half_open transition
        assert breaker.is_available("codex_cli", "openai_native") is True
        assert breaker.get_state("codex_cli", "openai_native") == "half_open"

    def test_circuit_breaker_half_open_success_closes(self, tmp_path):
        """Success during half_open → circuit closes."""
        breaker = CircuitBreaker(threshold=2, window_s=60, cooldown_s=0.05)

        for _ in range(2):
            breaker.record_failure("codex_cli", "openai_native")

        time.sleep(0.1)
        # is_available() triggers the open→half_open transition
        assert breaker.is_available("codex_cli", "openai_native") is True
        assert breaker.get_state("codex_cli", "openai_native") == "half_open"

        breaker.record_success("codex_cli", "openai_native")
        assert breaker.get_state("codex_cli", "openai_native") == "closed"
        assert breaker.is_available("codex_cli", "openai_native") is True

    def test_circuit_breaker_half_open_failure_reopens(self, tmp_path):
        """Failure during half_open → circuit reopens."""
        breaker = CircuitBreaker(threshold=2, window_s=60, cooldown_s=0.05)

        for _ in range(2):
            breaker.record_failure("codex_cli", "openai_native")

        time.sleep(0.1)
        # is_available() triggers the open→half_open transition
        assert breaker.is_available("codex_cli", "openai_native") is True
        assert breaker.get_state("codex_cli", "openai_native") == "half_open"

        breaker.record_failure("codex_cli", "openai_native", "provider_unavailable")
        assert breaker.get_state("codex_cli", "openai_native") == "open"

    def test_circuit_breaker_per_provider_isolation(self, tmp_path):
        """Circuit breaker state is isolated per provider."""
        breaker = CircuitBreaker(threshold=2, window_s=60, cooldown_s=120)

        # Trip codex_cli:openai_native
        for _ in range(2):
            breaker.record_failure("codex_cli", "openai_native")

        assert breaker.is_available("codex_cli", "openai_native") is False
        # claude_code:anthropic should still be available
        assert breaker.is_available("claude_code", "anthropic") is True

    def test_circuit_breaker_window_reset(self, tmp_path):
        """Failures outside the window are reset."""
        breaker = CircuitBreaker(threshold=3, window_s=0.1, cooldown_s=120)

        breaker.record_failure("codex_cli", "openai_native")
        breaker.record_failure("codex_cli", "openai_native")

        # Wait for window to expire
        time.sleep(0.15)

        # Next failure should not trip because window reset the count
        breaker.record_failure("codex_cli", "openai_native")
        assert breaker.get_state("codex_cli", "openai_native") == "closed"

    def test_circuit_breaker_success_resets_failure_count(self, tmp_path):
        """Success resets failure counter."""
        breaker = CircuitBreaker(threshold=5, window_s=60, cooldown_s=120)

        breaker.record_failure("codex_cli", "openai_native")
        breaker.record_failure("codex_cli", "openai_native")
        breaker.record_failure("codex_cli", "openai_native")

        breaker.record_success("codex_cli", "openai_native")

        ps = breaker._providers["codex_cli:openai_native"]
        assert ps.failure_count == 0
        assert ps.state == "closed"

    def test_circuit_breaker_health_summary(self, tmp_path):
        """get_health_summary returns correct aggregate stats."""
        breaker = CircuitBreaker(threshold=2, window_s=60, cooldown_s=120)

        breaker.record_failure("codex_cli", "openai_native")
        breaker.record_failure("codex_cli", "openai_native")  # trips
        breaker.record_success("claude_code", "anthropic")

        summary = breaker.get_health_summary()
        assert summary["total_providers"] == 2
        assert summary["healthy"] == 1
        assert summary["unhealthy"] == 1


# ===================================================================
# Integration: State + Circuit Breaker together
# ===================================================================

class TestStateCircuitBreakerIntegration:
    """State lifecycle and circuit breaker working together."""

    def test_state_transition_with_breaker_trip(self, tmp_path):
        """When primary provider's circuit trips, state can transition to fallback."""
        store = _make_store(tmp_path)
        breaker = CircuitBreaker(threshold=3, window_s=60, cooldown_s=120)

        # Trip the primary provider
        for _ in range(3):
            breaker.record_failure("codex_cli", "openai_native", "quota_exhausted")

        assert breaker.is_available("codex_cli", "openai_native") is False

        # State transition to conservation (simulating router behavior)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION

        # Claude backup should still be available
        assert breaker.is_available("claude_code", "anthropic") is True

    def test_full_e2e_lifecycle(self, tmp_path):
        """End-to-end: state transitions, breaker trips, recovery, history."""
        store = _make_store(tmp_path)
        breaker = CircuitBreaker(threshold=2, window_s=60, cooldown_s=0.05)

        # Start: primary
        assert store.get_state() == CodexState.OPENAI_PRIMARY

        # Provider starts failing
        breaker.record_failure("codex_cli", "openai_native", "rate_limited")
        breaker.record_failure("codex_cli", "openai_native", "rate_limited")

        # Circuit trips → transition to conservation
        assert breaker.is_available("codex_cli", "openai_native") is False
        store.set_state_with_history(
            CodexState.OPENAI_CONSERVATION, reason="primary_tripped", force=True
        )
        assert store.get_state() == CodexState.OPENAI_CONSERVATION

        # Conservation also fails → go to claude backup
        breaker.record_failure("codex_cli", "openai_native", "rate_limited")
        store.set_state_with_history(
            CodexState.CLAUDE_BACKUP, reason="conservation_failed", force=True
        )
        assert store.get_state() == CodexState.CLAUDE_BACKUP

        # Claude works — record successes
        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.CLAUDE_BACKUP, True)
        assert store.can_recover_to_primary(CodexState.CLAUDE_BACKUP) is True

        # Wait for breaker cooldown
        time.sleep(0.1)

        # Recover to primary
        store.set_state_with_history(
            CodexState.OPENAI_PRIMARY, reason="recovered", force=True
        )
        assert store.get_state() == CodexState.OPENAI_PRIMARY

        # History should show the full journey
        history = store.get_state_history(limit=10)
        assert len(history) == 3
        assert history[0]["to"] == "openai_conservation"
        assert history[1]["to"] == "claude_backup"
        assert history[2]["to"] == "openai_primary"
