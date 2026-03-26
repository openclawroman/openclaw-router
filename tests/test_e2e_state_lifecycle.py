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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_store(tmp_path):
    return StateStore(
        manual_path=tmp_path / "manual.json",
        auto_path=tmp_path / "auto.json",
        history_path=tmp_path / "history.json",
        wal_path=tmp_path / "wal.jsonl",
    )


def _advance_time(history_path, seconds):
    if not history_path.exists():
        return
    data = json.loads(history_path.read_text())
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    for entry in data.get("transitions", []):
        entry["timestamp"] = old_ts
    history_path.write_text(json.dumps(data, indent=2))


# ── Full State Cycle (2 tests) ───────────────────────────────────────────────

class TestFullStateCycle:

    def test_full_cycle(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_state() == CodexState.OPENAI_PRIMARY
        for state in [CodexState.OPENAI_CONSERVATION, CodexState.CLAUDE_BACKUP,
                       CodexState.OPENROUTER_FALLBACK, CodexState.OPENAI_PRIMARY]:
            store.set_auto_state(state)
            store.set_manual_state(None)
            assert store.get_state() == state

    def test_cycle_with_transitions(self, tmp_path):
        store = _make_store(tmp_path)
        states = [CodexState.OPENAI_CONSERVATION, CodexState.CLAUDE_BACKUP,
                  CodexState.OPENROUTER_FALLBACK, CodexState.OPENAI_PRIMARY]
        for i, state in enumerate(states):
            store.set_state_with_history(state, reason=f"step_{i}", force=True)
            assert store.get_state() == state
        assert len(store.get_state_history(limit=10)) == 4


# ── Manual Override (4 tests) ────────────────────────────────────────────────

class TestManualOverride:

    def test_manual_override_lifecycle(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store.get_state() == CodexState.CLAUDE_BACKUP
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        assert store.get_state() == CodexState.CLAUDE_BACKUP  # manual wins
        store.set_manual_state(None)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION
        store.set_manual_state(CodexState.OPENROUTER_FALLBACK)
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_manual_precedence(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store.get_state() == CodexState.CLAUDE_BACKUP

    def test_manual_null_falls_through_to_auto(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_manual_state(None)
        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_default_state_when_neither_set(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_state() == CodexState.OPENAI_PRIMARY


# ── State Persistence (2 tests) ──────────────────────────────────────────────

class TestStatePersistence:

    def test_auto_state_persists(self, tmp_path):
        paths = {k: tmp_path / f"{k}.json" for k in ["manual", "auto", "history"]}
        paths["wal"] = tmp_path / "wal.jsonl"
        store1 = StateStore(**{f"{k}_path": v for k, v in paths.items()})
        store1.set_manual_state(None)
        store1.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store2 = StateStore(**{f"{k}_path": v for k, v in paths.items()})
        store2.set_manual_state(None)
        assert store2.get_state() == CodexState.OPENAI_CONSERVATION

    def test_manual_state_persists(self, tmp_path):
        paths = {k: tmp_path / f"{k}.json" for k in ["manual", "auto", "history"]}
        paths["wal"] = tmp_path / "wal.jsonl"
        store1 = StateStore(**{f"{k}_path": v for k, v in paths.items()})
        store1.set_manual_state(CodexState.CLAUDE_BACKUP)
        store2 = StateStore(**{f"{k}_path": v for k, v in paths.items()})
        assert store2.get_state() == CodexState.CLAUDE_BACKUP


# ── WAL Recovery (3 tests) ──────────────────────────────────────────────────

class TestWALRecovery:

    def test_recovery_after_crash(self, tmp_path):
        manual, auto = tmp_path / "manual.json", tmp_path / "auto.json"
        history, wal = tmp_path / "history.json", tmp_path / "wal.jsonl"
        store = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store.get_state() == CodexState.CLAUDE_BACKUP

        manual.write_text("CORRUPTED{{{invalid json")
        wal.write_text(json.dumps({"action": "write", "path": str(manual),
            "data": {"state": "claude_backup"}, "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")
        store2 = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        assert store2.get_state() == CodexState.CLAUDE_BACKUP

    def test_last_write_wins(self, tmp_path):
        manual, auto = tmp_path / "manual.json", tmp_path / "auto.json"
        history, wal = tmp_path / "history.json", tmp_path / "wal.jsonl"
        entries = [
            {"action": "write", "path": str(manual), "data": {"state": "openai_conservation"}, "timestamp": "t1"},
            {"action": "write", "path": str(manual), "data": {"state": "claude_backup"}, "timestamp": "t2"},
            {"action": "write", "path": str(manual), "data": {"state": "openrouter_fallback"}, "timestamp": "t3"},
        ]
        wal.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        manual.write_text("BROKEN")
        store = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_committed_entries_not_reapplied(self, tmp_path):
        manual, auto = tmp_path / "manual.json", tmp_path / "auto.json"
        history, wal = tmp_path / "history.json", tmp_path / "wal.jsonl"
        wal.write_text(
            json.dumps({"action": "write", "path": str(manual), "data": {"state": "claude_backup"}, "timestamp": "t1"}) + "\n" +
            json.dumps({"action": "committed", "timestamp": "t1"}) + "\n"
        )
        manual.write_text(json.dumps({"state": "openai_primary"}))
        store = StateStore(manual_path=manual, auto_path=auto, history_path=history, wal_path=wal)
        assert store.get_state() == CodexState.OPENAI_PRIMARY


# ── Anti-Flap Protection (4 tests) ──────────────────────────────────────────

class TestAntiFlapProtection:

    def test_sticky_state_blocks_rapid_transition(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        store.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "budget_low")
        allowed, reason = store.can_transition(CodexState.OPENROUTER_FALLBACK)
        assert allowed is False
        assert "anti_flap" in reason

    def test_allowed_after_timeout(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "budget_low")
        _advance_time(store.history_path, MIN_STATE_DURATION_S + 60)
        assert store.can_transition(CodexState.OPENROUTER_FALLBACK)[0] is True

    def test_force_bypasses_and_emergency_always_allowed(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "test")
        # Non-emergency without force → blocked
        assert store.can_transition(CodexState.OPENROUTER_FALLBACK, force=False)[0] is False
        # With force → allowed
        assert store.can_transition(CodexState.OPENROUTER_FALLBACK, force=True)[0] is True
        # Emergency targets → always allowed
        assert store.can_transition(CodexState.OPENAI_PRIMARY)[0] is True
        assert store.can_transition(CodexState.CLAUDE_BACKUP)[0] is True

    def test_set_state_respects_and_bypasses_anti_flap(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "test")
        with pytest.raises(StateError, match="anti_flap"):
            store.set_state_with_history(CodexState.OPENROUTER_FALLBACK, reason="test")
        assert store.set_state_with_history(CodexState.OPENROUTER_FALLBACK, reason="forced", force=True) is True
        assert store.get_state() == CodexState.OPENROUTER_FALLBACK


# ── Consecutive Successes (4 tests) ──────────────────────────────────────────

class TestConsecutiveSuccesses:

    def test_recovery_after_threshold(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False
        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True

    def test_failure_resets_counter(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD - 1):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENAI_CONSERVATION, False)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False
        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True

    def test_per_state_counters(self, tmp_path):
        store = _make_store(tmp_path)
        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True
        assert store.can_recover_to_primary(CodexState.OPENROUTER_FALLBACK) is False

    def test_reset_success_counter(self, tmp_path):
        store = _make_store(tmp_path)
        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True
        store.reset_success_counter(CodexState.OPENAI_CONSERVATION)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False


# ── State History (3 tests) ──────────────────────────────────────────────────

class TestStateHistory:

    def test_records_transitions(self, tmp_path):
        store = _make_store(tmp_path)
        store.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, "budget_low")
        store.log_state_transition(CodexState.OPENAI_CONSERVATION, CodexState.CLAUDE_BACKUP, "quota_hit")
        store.log_state_transition(CodexState.CLAUDE_BACKUP, CodexState.OPENROUTER_FALLBACK, "all_failed")
        history = store.get_state_history(limit=10)
        assert len(history) == 3
        assert history[0]["from"] == "openai_primary" and history[0]["to"] == "openai_conservation"
        assert history[1]["to"] == "claude_backup"
        assert history[2]["to"] == "openrouter_fallback"

    def test_history_persists(self, tmp_path):
        paths = {k: tmp_path / f"{k}.json" for k in ["manual", "auto", "history"]}
        paths["wal"] = tmp_path / "wal.jsonl"
        store1 = StateStore(**{f"{k}_path": v for k, v in paths.items()})
        store1.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP, "manual")
        store1.log_state_transition(CodexState.CLAUDE_BACKUP, CodexState.OPENAI_CONSERVATION, "recovery")
        store2 = StateStore(**{f"{k}_path": v for k, v in paths.items()})
        history = store2.get_state_history(limit=10)
        assert len(history) == 2
        assert history[0]["to"] == "claude_backup"

    def test_max_entries_capped(self, tmp_path):
        store = _make_store(tmp_path)
        from router.state_store import MAX_HISTORY_ENTRIES
        for i in range(MAX_HISTORY_ENTRIES + 10):
            store.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, f"entry_{i}")
        history = store.get_state_history(limit=MAX_HISTORY_ENTRIES + 20)
        assert len(history) == MAX_HISTORY_ENTRIES
        assert history[0]["reason"] == "entry_10"


# ── Circuit Breaker Lifecycle (4 parametrized tests) ─────────────────────────

class TestCircuitBreakerLifecycle:

    def test_trip(self, tmp_path):
        breaker = CircuitBreaker(threshold=3, window_s=60, cooldown_s=120)
        assert breaker.is_available("codex_cli", "openai_native") is True
        assert breaker.get_state("codex_cli", "openai_native") == "closed"
        for _ in range(3):
            breaker.record_failure("codex_cli", "openai_native", "rate_limited")
        assert breaker.get_state("codex_cli", "openai_native") == "open"
        assert breaker.is_available("codex_cli", "openai_native") is False

    def test_recovery_half_open_transition(self, tmp_path):
        breaker = CircuitBreaker(threshold=3, window_s=60, cooldown_s=0.1)
        for _ in range(3):
            breaker.record_failure("codex_cli", "openai_native", "provider_unavailable")
        assert breaker.is_available("codex_cli", "openai_native") is False
        time.sleep(0.15)
        assert breaker.is_available("codex_cli", "openai_native") is True
        assert breaker.get_state("codex_cli", "openai_native") == "half_open"

    @pytest.mark.parametrize("record_fn,expected_state", [
        (lambda b: b.record_success("codex_cli", "openai_native"), "closed"),
        (lambda b: b.record_failure("codex_cli", "openai_native", "provider_unavailable"), "open"),
    ], ids=["half_open_success_closes", "half_open_failure_reopens"])
    def test_half_open_outcome(self, tmp_path, record_fn, expected_state):
        breaker = CircuitBreaker(threshold=2, window_s=60, cooldown_s=0.05)
        for _ in range(2):
            breaker.record_failure("codex_cli", "openai_native")
        time.sleep(0.1)
        assert breaker.is_available("codex_cli", "openai_native") is True
        assert breaker.get_state("codex_cli", "openai_native") == "half_open"
        record_fn(breaker)
        assert breaker.get_state("codex_cli", "openai_native") == expected_state

    def test_isolation_window_reset_success_resets_health_summary(self, tmp_path):
        # Per-provider isolation
        breaker = CircuitBreaker(threshold=2, window_s=60, cooldown_s=120)
        for _ in range(2):
            breaker.record_failure("codex_cli", "openai_native")
        assert breaker.is_available("codex_cli", "openai_native") is False
        assert breaker.is_available("claude_code", "anthropic") is True

        # Window reset
        breaker2 = CircuitBreaker(threshold=3, window_s=0.1, cooldown_s=120)
        breaker2.record_failure("codex_cli", "openai_native")
        breaker2.record_failure("codex_cli", "openai_native")
        time.sleep(0.15)
        breaker2.record_failure("codex_cli", "openai_native")
        assert breaker2.get_state("codex_cli", "openai_native") == "closed"

        # Success resets failure count
        breaker3 = CircuitBreaker(threshold=5, window_s=60, cooldown_s=120)
        breaker3.record_failure("codex_cli", "openai_native")
        breaker3.record_failure("codex_cli", "openai_native")
        breaker3.record_failure("codex_cli", "openai_native")
        breaker3.record_success("codex_cli", "openai_native")
        ps = breaker3._providers["codex_cli:openai_native"]
        assert ps.failure_count == 0 and ps.state == "closed"

        # Health summary
        breaker4 = CircuitBreaker(threshold=2, window_s=60, cooldown_s=120)
        breaker4.record_failure("codex_cli", "openai_native")
        breaker4.record_failure("codex_cli", "openai_native")
        breaker4.record_success("claude_code", "anthropic")
        summary = breaker4.get_health_summary()
        assert summary["healthy"] == 1 and summary["unhealthy"] == 1


# ── Integration: State + Circuit Breaker (2 tests) ──────────────────────────

class TestStateCircuitBreakerIntegration:

    def test_state_transition_with_breaker_trip(self, tmp_path):
        store = _make_store(tmp_path)
        breaker = CircuitBreaker(threshold=3, window_s=60, cooldown_s=120)
        for _ in range(3):
            breaker.record_failure("codex_cli", "openai_native", "quota_exhausted")
        assert breaker.is_available("codex_cli", "openai_native") is False
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)
        assert store.get_state() == CodexState.OPENAI_CONSERVATION
        assert breaker.is_available("claude_code", "anthropic") is True

    def test_full_e2e_lifecycle(self, tmp_path):
        store = _make_store(tmp_path)
        breaker = CircuitBreaker(threshold=2, window_s=60, cooldown_s=0.05)
        assert store.get_state() == CodexState.OPENAI_PRIMARY
        for _ in range(2):
            breaker.record_failure("codex_cli", "openai_native", "rate_limited")
        assert breaker.is_available("codex_cli", "openai_native") is False
        store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="primary_tripped", force=True)
        store.set_state_with_history(CodexState.CLAUDE_BACKUP, reason="conservation_failed", force=True)
        for _ in range(CONSECUTIVE_SUCCESSES_THRESHOLD):
            store.record_success(CodexState.CLAUDE_BACKUP, True)
        assert store.can_recover_to_primary(CodexState.CLAUDE_BACKUP) is True
        time.sleep(0.1)
        store.set_state_with_history(CodexState.OPENAI_PRIMARY, reason="recovered", force=True)
        assert store.get_state() == CodexState.OPENAI_PRIMARY
        history = store.get_state_history(limit=10)
        assert len(history) == 3
        assert [h["to"] for h in history] == ["openai_conservation", "claude_backup", "openai_primary"]
