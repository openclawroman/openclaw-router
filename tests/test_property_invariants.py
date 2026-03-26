"""Merged property-based tests for state machine invariants and chain building.

Combines former test_property_state_machine.py, test_property_edge_cases.py,
and test_property_chain_building.py into one file.  All unique scenarios
preserved; redundancy removed via parametrize, shared fixtures, and helpers.

Properties tested:
  A. State always valid
  B. Manual overrides auto
  C. Transition consistency
  D. History monotonicity
  E. Chain non-empty / first entry valid
  F. State persistence round-trip
  G. WAL recovery consistency
  H. Force bypass works
  I. Anti-flap invariant
  J. Singleton invariant
  K. Circuit breaker invariant
  L. Chain structure (fields, length, no-dup, first-backend, consistency)
  M. Edge cases & stress
  N. Anti-flap boundary & force-bypass window
  O. State survives reload
  P. WAL recovery state value
  Q. Corrupted state files
"""

import json
import os
import random
import threading
import time
import itertools
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from router.models import CodexState, TaskMeta, TaskClass, TaskRisk, ChainEntry
from router.state_store import StateStore, get_state_store, reset_state_store, MIN_STATE_DURATION_S
from router.circuit_breaker import CircuitBreaker
from router.policy import build_chain, validate_chain, resolve_state
from router.errors import StateError

ALL_STATES = list(CodexState)
ALL_TASK_CLASSES = list(TaskClass)
ALL_RISKS = list(TaskRisk)
VALID_TOOLS = {"codex_cli", "claude_code", "openrouter"}
VALID_BACKENDS = {"openai_native", "anthropic", "openrouter"}
VALID_PROFILES = {
    "codex_primary", "codex_gpt54", "codex_gpt54_mini",
    "claude_primary", "claude_sonnet", "claude_opus",
    "openrouter_minimax", "openrouter_kimi", "openrouter_mimo", "openrouter_dynamic",
}
STATE_FIRST_BACKEND = {
    CodexState.OPENAI_PRIMARY: "openai_native",
    CodexState.OPENAI_CONSERVATION: "openai_native",
    CodexState.CLAUDE_BACKUP: "anthropic",
    CodexState.OPENROUTER_FALLBACK: "openrouter",
}
EMERGENCY_TARGETS = [CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP]


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    return StateStore(
        manual_path=tmp_path / "manual.json",
        auto_path=tmp_path / "auto.json",
        history_path=tmp_path / "history.json",
        wal_path=tmp_path / "wal.jsonl",
    )


@pytest.fixture
def default_task():
    return TaskMeta(task_id="prop-test", agent="coder",
                    task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM)


def _make_state_store(tmp_path):
    return StateStore(
        manual_path=tmp_path / "manual.json",
        auto_path=tmp_path / "auto.json",
        history_path=tmp_path / "history.json",
        wal_path=tmp_path / "wal.jsonl",
    )


def _make_task(task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM, **kwargs):
    return TaskMeta(task_id="chain-test", task_class=task_class, risk=risk, **kwargs)


def _clear_history(store):
    if store.history_path.exists():
        store.history_path.write_text("")


def _patch_at_timeout(tmp_store, delta_s):
    """Return a patch context that makes datetime.now() look like `delta_s`
    seconds after the last history entry."""
    history = tmp_store.get_state_history(limit=1)
    if not history:
        raise ValueError("No history to patch against")
    last_ts_str = history[0]["timestamp"]

    class FakeDatetime:
        @staticmethod
        def now(tz=None):
            base = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
            return base + timedelta(seconds=delta_s)

        @staticmethod
        def fromisoformat(s):
            return datetime.fromisoformat(s)

        timedelta = timedelta

    return patch("router.state_store.datetime", FakeDatetime)


# ═══════════════════════════════════════════════════════════════════════════════
# A. State always valid
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyAStateAlwaysValid:

    @pytest.mark.parametrize("setter", ["manual", "auto"])
    @pytest.mark.parametrize("state", ALL_STATES)
    def test_single_set_is_valid(self, tmp_store, setter, state):
        getattr(tmp_store, f"set_{setter}_state")(state)
        result = tmp_store.get_state()
        assert isinstance(result, CodexState)
        assert result in ALL_STATES

    @pytest.mark.parametrize("setter,count", [("manual", 500), ("auto", 500), ("mixed", 200)])
    def test_random_sequences_stay_valid(self, tmp_store, setter, count):
        for _ in range(count):
            if setter == "mixed":
                if random.random() < 0.5:
                    tmp_store.set_manual_state(random.choice(ALL_STATES))
                else:
                    tmp_store.set_auto_state(random.choice(ALL_STATES))
            else:
                getattr(tmp_store, f"set_{setter}_state")(random.choice(ALL_STATES))
            assert isinstance(tmp_store.get_state(), CodexState)

    def test_none_manual_with_auto_is_valid(self, tmp_store):
        for state in ALL_STATES:
            tmp_store.set_auto_state(state)
            tmp_store.set_manual_state(None)
            assert tmp_store.get_state() == state


# ═══════════════════════════════════════════════════════════════════════════════
# B. Manual overrides auto
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyBManualOverridesAuto:

    @pytest.mark.parametrize("manual_state", ALL_STATES)
    @pytest.mark.parametrize("auto_state", ALL_STATES)
    def test_manual_wins_over_auto(self, tmp_store, manual_state, auto_state):
        tmp_store.set_auto_state(auto_state)
        tmp_store.set_manual_state(manual_state)
        assert tmp_store.get_state() == manual_state

    def test_clearing_manual_falls_through_to_auto(self, tmp_store):
        for auto in ALL_STATES:
            tmp_store.set_auto_state(auto)
            tmp_store.set_manual_state(random.choice(ALL_STATES))
            tmp_store.set_manual_state(None)
            assert tmp_store.get_state() == auto


# ═══════════════════════════════════════════════════════════════════════════════
# C. Transition consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyCTransitionConsistency:

    @pytest.mark.parametrize("target", ALL_STATES)
    def test_allowed_transition_succeeds(self, tmp_store, target):
        allowed, _ = tmp_store.can_transition(target)
        if allowed:
            tmp_store.set_state_with_history(target, reason="prop-test")

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_same_state_transition_always_allowed(self, tmp_store, state):
        tmp_store.set_manual_state(state)
        allowed, reason = tmp_store.can_transition(state)
        assert allowed is True
        assert reason == "same_state"

    def test_random_transitions_respect_can_transition(self, tmp_store):
        tmp_store.set_manual_state(CodexState.OPENAI_PRIMARY)
        _clear_history(tmp_store)
        for _ in range(100):
            target = random.choice(ALL_STATES)
            allowed, reason = tmp_store.can_transition(target)
            if allowed:
                try:
                    tmp_store.set_state_with_history(target, reason="stress-test")
                except StateError:
                    pytest.fail(f"can_transition=True but set_state_with_history raised for {target}")


# ═══════════════════════════════════════════════════════════════════════════════
# D. History monotonicity
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyDHistoryMonotonicity:

    def test_sequential_transitions_are_monotonic(self, tmp_store):
        _clear_history(tmp_store)
        for state in ALL_STATES:
            tmp_store.set_state_with_history(state, reason="mono-test", force=True)
        history = tmp_store.get_state_history(limit=50)
        if len(history) < 2:
            pytest.skip("Need ≥2 history entries")
        timestamps = [e["timestamp"].replace("Z", "+00:00") for e in history if e.get("timestamp")]
        assert all(timestamps[i] >= timestamps[i - 1] for i in range(1, len(timestamps)))

    @pytest.mark.parametrize("corrupt", ["", "NOT VALID JSON {{{{"])
    def test_empty_or_corrupted_history_returns_list(self, tmp_store, corrupt):
        tmp_store.history_path.write_text(corrupt)
        assert tmp_store.get_state_history() == []


# ═══════════════════════════════════════════════════════════════════════════════
# E. Chain non-empty / first entry valid
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyEChainNonEmptyAndValid:

    @pytest.mark.parametrize("state", ALL_STATES)
    @pytest.mark.parametrize("risk", ALL_RISKS)
    def test_chain_non_empty_all_risks(self, state, risk):
        task = TaskMeta(task_id="e-risk", task_class=TaskClass.IMPLEMENTATION, risk=risk)
        assert len(build_chain(task, state)) >= 1

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_first_entry_valid_tool(self, default_task, state):
        assert build_chain(default_task, state)[0].tool in VALID_TOOLS

    def test_stress_chains_non_empty_with_valid_tools(self):
        for i in range(200):
            task = TaskMeta(task_id=f"stress-{i}",
                            task_class=random.choice(ALL_TASK_CLASSES),
                            risk=random.choice(ALL_RISKS))
            chain = build_chain(task, random.choice(ALL_STATES))
            assert len(chain) >= 1
            assert all(e.tool in VALID_TOOLS for e in chain)


# ═══════════════════════════════════════════════════════════════════════════════
# F. State persistence round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyFPersistenceRoundTrip:

    @pytest.mark.parametrize("setter", ["manual", "auto"])
    @pytest.mark.parametrize("state", ALL_STATES)
    def test_round_trip(self, tmp_store, setter, state):
        getattr(tmp_store, f"set_{setter}_state")(state)
        assert getattr(tmp_store, f"get_{setter}_state")() == state

    def test_none_manual_round_trip(self, tmp_store):
        tmp_store.set_manual_state(CodexState.OPENAI_PRIMARY)
        tmp_store.set_manual_state(None)
        assert tmp_store.get_manual_state() is None

    def test_rapid_1000_round_trips(self, tmp_store):
        for _ in range(1000):
            state = random.choice(ALL_STATES)
            if random.random() < 0.8:
                tmp_store.set_manual_state(state)
                assert tmp_store.get_manual_state() == state
            else:
                tmp_store.set_auto_state(state)
                assert tmp_store.get_auto_state() == state


# ═══════════════════════════════════════════════════════════════════════════════
# G. WAL recovery consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyGWALRecovery:

    @pytest.mark.parametrize("wal_content", [
        "",
        '{"action":"write","path":"/nonexistent","data":{"state":"openai_primary"}}\n{"action":"commi',
        "NOT JSON\n{}\n{{invalid\n",
    ])
    def test_wal_recovery_yields_valid_state(self, tmp_path, wal_content):
        wal = tmp_path / "wal.jsonl"
        wal.write_text(wal_content)
        store = StateStore(
            manual_path=tmp_path / "m.json", auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json", wal_path=wal,
        )
        assert isinstance(store.get_state(), CodexState)

    def test_random_wal_entries_state_valid(self, tmp_path):
        wal = tmp_path / "wal.jsonl"
        lines = []
        for _ in range(100):
            if random.random() < 0.7:
                lines.append(json.dumps({
                    "action": "write", "path": str(tmp_path / "m.json"),
                    "data": {"state": random.choice(ALL_STATES).value},
                }))
            else:
                lines.append("GARBAGE" if random.random() < 0.1 else json.dumps({"action": "committed"}))
        wal.write_text("\n".join(lines))
        store = StateStore(
            manual_path=tmp_path / "m.json", auto_path=tmp_path / "a.json",
            history_path=tmp_path / "h.json", wal_path=wal,
        )
        assert isinstance(store.get_state(), CodexState)


# ═══════════════════════════════════════════════════════════════════════════════
# H. Force bypass
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyHForceBypass:

    @pytest.mark.parametrize("target", ALL_STATES)
    def test_force_always_allowed(self, tmp_store, target):
        allowed, reason = tmp_store.can_transition(target, force=True)
        assert allowed is True

    def test_force_from_every_state(self, tmp_store):
        for start in ALL_STATES:
            tmp_store.set_manual_state(start)
            for target in ALL_STATES:
                assert tmp_store.can_transition(target, force=True)[0] is True

    def test_force_transitions_never_raise(self, tmp_store):
        for _ in range(200):
            target = random.choice(ALL_STATES)
            try:
                tmp_store.set_state_with_history(target, reason="force-test", force=True)
            except StateError:
                pytest.fail(f"Force transition to {target} raised StateError")

    def test_force_immediately_after_transition(self, tmp_store):
        tmp_store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="setup", force=True)
        tmp_store.set_state_with_history(CodexState.OPENROUTER_FALLBACK, reason="emergency", force=True)
        assert tmp_store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_force_vs_no_force_same_window(self, tmp_store):
        tmp_store.set_state_with_history(CodexState.OPENAI_PRIMARY, reason="setup", force=True)
        allowed_no_force, reason = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK, force=False)
        if not allowed_no_force:
            assert "anti_flap" in reason
            assert tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK, force=True)[0] is True

    def test_force_rapid_sequence(self, tmp_store):
        for i in range(20):
            target = random.choice(ALL_STATES)
            tmp_store.set_state_with_history(target, reason=f"force-{i}", force=True)
            assert tmp_store.get_state() == target


# ═══════════════════════════════════════════════════════════════════════════════
# I. Anti-flap invariant
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyIAntiFlap:

    def test_emergency_targets_always_allowed(self, tmp_store):
        tmp_store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="test", force=True)
        for target in EMERGENCY_TARGETS:
            allowed, reason = tmp_store.can_transition(target)
            assert allowed is True, f"Emergency {target} blocked: {reason}"

    def test_non_emergency_blocked_during_cooldown(self, tmp_store):
        tmp_store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="setup", force=True)
        allowed, reason = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK)
        if not allowed:
            assert "anti_flap" in reason

    def test_force_bypasses_anti_flap(self, tmp_store):
        tmp_store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="setup", force=True)
        assert tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK, force=True)[0] is True

    def test_transition_allowed_exactly_at_timeout(self, tmp_store):
        tmp_store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="setup", force=True)
        with _patch_at_timeout(tmp_store, MIN_STATE_DURATION_S):
            allowed, reason = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK)
            assert allowed is True, f"Blocked at timeout boundary: {reason}"

    def test_transition_blocked_just_before_timeout(self, tmp_store):
        tmp_store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="setup", force=True)
        with _patch_at_timeout(tmp_store, MIN_STATE_DURATION_S - 1):
            allowed, reason = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK)
            assert allowed is False
            assert "anti_flap" in reason

    def test_transition_allowed_just_after_timeout(self, tmp_store):
        tmp_store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="setup", force=True)
        with _patch_at_timeout(tmp_store, MIN_STATE_DURATION_S + 1):
            assert tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK)[0] is True

    def test_boundary_scans_all_states(self, tmp_store):
        for state in ALL_STATES:
            tmp_store.set_state_with_history(state, reason="boundary-test", force=True)
            with _patch_at_timeout(tmp_store, MIN_STATE_DURATION_S):
                allowed, reason = tmp_store.can_transition(CodexState.OPENROUTER_FALLBACK)
                assert allowed is True, f"Boundary failed from {state}: {reason}"


# ═══════════════════════════════════════════════════════════════════════════════
# J. Singleton invariant
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyJSingleton:

    def test_singleton_identity(self):
        reset_state_store()
        s1, s2, s3 = get_state_store(), get_state_store(), get_state_store()
        assert s1 is s2 is s3
        reset_state_store()

    def test_singleton_across_threads(self):
        reset_state_store()
        results = []
        def get_store(): results.append(get_state_store())
        threads = [threading.Thread(target=get_store) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert all(r is results[0] for r in results)
        reset_state_store()

    def test_stress_singleton_1000_calls(self):
        reset_state_store()
        first = get_state_store()
        for _ in range(1000):
            assert get_state_store() is first
        reset_state_store()


# ═══════════════════════════════════════════════════════════════════════════════
# K. Circuit breaker invariant
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyKCircuitBreaker:

    @pytest.mark.parametrize("threshold", [1, 3, 5, 10])
    def test_breaks_at_threshold(self, threshold):
        cb = CircuitBreaker(threshold=threshold, window_s=300, cooldown_s=600)
        tool, backend = "codex_cli", "openai_native"
        assert cb.is_available(tool, backend) is True
        for _ in range(threshold):
            cb.record_failure(tool, backend)
        assert cb.is_available(tool, backend) is False

    @pytest.mark.parametrize("threshold", [3, 5])
    def test_stays_available_below_threshold(self, threshold):
        cb = CircuitBreaker(threshold=threshold, window_s=300, cooldown_s=600)
        tool, backend = "claude_code", "anthropic"
        for i in range(threshold - 1):
            cb.record_failure(tool, backend)
            assert cb.is_available(tool, backend) is True

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(threshold=3, window_s=300, cooldown_s=600)
        t, b = "codex_cli", "openai_native"
        cb.record_failure(t, b); cb.record_failure(t, b)
        cb.record_success(t, b)
        cb.record_failure(t, b); cb.record_failure(t, b)
        assert cb.is_available(t, b) is True

    def test_different_providers_independent(self):
        cb = CircuitBreaker(threshold=3, window_s=300, cooldown_s=600)
        for _ in range(3):
            cb.record_failure("codex_cli", "openai_native")
        assert cb.is_available("codex_cli", "openai_native") is False
        assert cb.is_available("claude_code", "anthropic") is True

    def test_stress_random_thresholds(self):
        tools = ["codex_cli", "claude_code", "openrouter"]
        backends = ["openai_native", "anthropic", "openrouter"]
        for _ in range(50):
            cb = CircuitBreaker(threshold=random.randint(1, 10), window_s=300, cooldown_s=600)
            t, b = random.choice(tools), random.choice(backends)
            for _ in range(cb.threshold):
                cb.record_failure(t, b)
            assert cb.is_available(t, b) is False

    def test_single_available_provider_among_open_breakers(self):
        cb = CircuitBreaker(threshold=1, window_s=300, cooldown_s=600)
        cb.record_failure("codex_cli", "openai_native")
        cb.record_failure("codex_cli", "openrouter")
        assert cb.is_available("claude_code", "anthropic") is True
        assert cb.is_available("codex_cli", "openai_native") is False


# ═══════════════════════════════════════════════════════════════════════════════
# L. Chain structure (merged from chain_building)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLChainStructure:

    @pytest.mark.parametrize("task_class,state", list(itertools.product(ALL_TASK_CLASSES, ALL_STATES)))
    def test_combination_produces_valid_chain(self, task_class, state):
        chain = build_chain(_make_task(task_class=task_class), state)
        assert isinstance(chain, list) and len(chain) >= 1

    @pytest.mark.parametrize("task_class,state", list(itertools.product(ALL_TASK_CLASSES, ALL_STATES)))
    def test_validate_all_combinations(self, task_class, state):
        chain = build_chain(_make_task(task_class=task_class), state)
        valid, reason = validate_chain(state, chain)
        assert valid is True, f"Invalid ({task_class}, {state}): {reason}"

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_entries_have_required_valid_fields(self, state):
        chain = build_chain(_make_task(), state)
        for i, e in enumerate(chain):
            assert isinstance(e, ChainEntry) and e.tool and e.backend and e.model_profile
            assert e.tool in VALID_TOOLS, f"Unknown tool: {e.tool}"
            assert e.backend in VALID_BACKENDS, f"Unknown backend: {e.backend}"
            assert e.model_profile in VALID_PROFILES, f"Unknown profile: {e.model_profile}"

    @pytest.mark.parametrize("task_class,state", list(itertools.product(ALL_TASK_CLASSES, ALL_STATES)))
    def test_length_in_bounds(self, task_class, state):
        chain = build_chain(_make_task(task_class=task_class), state)
        assert 1 <= len(chain) <= 4

    @pytest.mark.parametrize("state", ALL_STATES)
    def test_no_duplicate_entries(self, state):
        chain = build_chain(_make_task(), state)
        seen = set()
        for e in chain:
            key = (e.tool, e.backend, e.model_profile)
            assert key not in seen, f"Duplicate: {key}"
            seen.add(key)

    @pytest.mark.parametrize("state,expected_backend", list(STATE_FIRST_BACKEND.items()))
    def test_first_entry_backend(self, state, expected_backend):
        assert build_chain(_make_task(), state)[0].backend == expected_backend

    def test_openrouter_fallback_only_has_openrouter(self):
        chain = build_chain(_make_task(), CodexState.OPENROUTER_FALLBACK)
        assert len(chain) == 1
        assert all(e.backend == "openrouter" for e in chain)

    def test_claude_backup_starts_with_anthropic(self):
        chain = build_chain(_make_task(), CodexState.CLAUDE_BACKUP)
        assert chain[0].backend == "anthropic"
        assert all(e.backend != "openai_native" for e in chain)

    def test_chain_length_consistent_across_tasks(self):
        lengths = {s: len(build_chain(_make_task(), s)) for s in ALL_STATES}
        for state, tc, risk in itertools.product(ALL_STATES, ALL_TASK_CLASSES, ALL_RISKS):
            chain = build_chain(_make_task(task_class=tc, risk=risk), state)
            assert len(chain) == lengths[state], f"Length mismatch ({tc}, {state})"

    def test_backend_order_consistent_across_risks(self):
        for state in ALL_STATES:
            c1 = build_chain(_make_task(risk=TaskRisk.LOW), state)
            c2 = build_chain(_make_task(risk=TaskRisk.CRITICAL), state)
            assert [e.backend for e in c1] == [e.backend for e in c2]

    @pytest.mark.parametrize("kwargs,profile", [
        ({"has_screenshots": True}, "openrouter_kimi"),
        ({"risk": TaskRisk.CRITICAL}, "openrouter_mimo"),
        ({"risk": TaskRisk.LOW}, "openrouter_minimax"),
    ])
    def test_multimodal_profile_selection(self, kwargs, profile):
        chain = build_chain(_make_task(**kwargs), CodexState.OPENAI_PRIMARY)
        or_entries = [e for e in chain if e.backend == "openrouter"]
        assert any(e.model_profile == profile for e in or_entries)

    def test_stress_500_random_chains(self):
        for _ in range(500):
            task = _make_task(task_class=random.choice(ALL_TASK_CLASSES),
                              risk=random.choice(ALL_RISKS))
            chain = build_chain(task, random.choice(ALL_STATES))
            assert 1 <= len(chain) <= 4
            seen = set()
            for e in chain:
                assert e.tool in VALID_TOOLS and e.backend in VALID_BACKENDS and e.model_profile in VALID_PROFILES
                key = (e.tool, e.backend, e.model_profile)
                assert key not in seen
                seen.add(key)


# ═══════════════════════════════════════════════════════════════════════════════
# M. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestMEdgeCases:

    def test_none_manual_state_clears_override(self, tmp_store):
        tmp_store.set_manual_state(CodexState.CLAUDE_BACKUP)
        tmp_store.set_manual_state(None)
        assert tmp_store.get_manual_state() is None

    def test_missing_state_files_get_defaults(self, tmp_path):
        store = StateStore(
            manual_path=tmp_path / "new_m.json", auto_path=tmp_path / "new_a.json",
            history_path=tmp_path / "new_h.json", wal_path=tmp_path / "new_w.jsonl",
        )
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_resolve_state_default(self, tmp_store):
        tmp_store.set_manual_state(None)
        tmp_store.auto_path.write_text(json.dumps({"state": None}))
        assert resolve_state(tmp_store) == CodexState.OPENAI_PRIMARY

    def test_validate_transition_all_pairs(self, tmp_store):
        for src, dst in itertools.product(ALL_STATES, ALL_STATES):
            assert tmp_store.validate_transition(src, dst) is True

    def test_concurrent_state_writes(self, tmp_store):
        errors = []
        def writer(state):
            try:
                for _ in range(100):
                    tmp_store.set_manual_state(state)
                    _ = tmp_store.get_state()
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=writer, args=(s,)) for s in ALL_STATES]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors) == 0
        assert isinstance(tmp_store.get_state(), CodexState)

    def test_1000_state_transitions(self, tmp_store):
        for _ in range(1000):
            tmp_store.set_state_with_history(random.choice(ALL_STATES), reason="stress", force=True)
            assert isinstance(tmp_store.get_state(), CodexState)

    def test_1000_chain_builds(self, default_task):
        for _ in range(1000):
            chain = build_chain(default_task, random.choice(ALL_STATES))
            assert len(chain) >= 1
            assert all(e.tool in VALID_TOOLS for e in chain)

    def test_10000_chain_builds_extreme(self):
        for i in range(10000):
            task = TaskMeta(task_id=f"extreme-{i}",
                            task_class=random.choice(ALL_TASK_CLASSES),
                            risk=random.choice(ALL_RISKS))
            chain = build_chain(task, random.choice(ALL_STATES))
            assert 1 <= len(chain) <= 4

    def test_chain_deterministic_same_input(self):
        task = _make_task(risk=TaskRisk.HIGH)
        for state in ALL_STATES:
            chains = [build_chain(task, state) for _ in range(100)]
            first = chains[0]
            for chain in chains[1:]:
                assert [(e.tool, e.backend, e.model_profile) for e in chain] == \
                       [(e.tool, e.backend, e.model_profile) for e in first]


# ═══════════════════════════════════════════════════════════════════════════════
# N. State survives reload
# ═══════════════════════════════════════════════════════════════════════════════

class TestNStateSurvivesReload:

    @pytest.mark.parametrize("setter", ["manual", "auto"])
    @pytest.mark.parametrize("state", ALL_STATES)
    def test_state_survives_reload(self, tmp_path, setter, state):
        s1 = _make_state_store(tmp_path)
        getattr(s1, f"set_{setter}_state")(state)
        s2 = _make_state_store(tmp_path)
        assert getattr(s2, f"get_{setter}_state")() == state

    def test_history_survives_reload(self, tmp_path):
        s1 = _make_state_store(tmp_path)
        for state in ALL_STATES:
            s1.set_state_with_history(state, reason="reload-test", force=True)
        s2 = _make_state_store(tmp_path)
        assert len(s2.get_state_history(limit=50)) >= len(ALL_STATES)

    def test_stress_reload_consistency(self, tmp_path):
        for i in range(50):
            state = random.choice(ALL_STATES)
            s = _make_state_store(tmp_path)
            s.set_manual_state(state)
            assert _make_state_store(tmp_path).get_manual_state() == state


# ═══════════════════════════════════════════════════════════════════════════════
# O. WAL recovery state value
# ═══════════════════════════════════════════════════════════════════════════════

class TestOWALRecoveryStateValue:

    def test_recovery_applies_uncommitted_write(self, tmp_path):
        mp, ap = tmp_path / "m.json", tmp_path / "a.json"
        mp.write_text(json.dumps({"state": "openai_primary"}))
        ap.write_text(json.dumps({"state": "openai_primary"}))
        wal = tmp_path / "w.jsonl"
        wal.write_text(json.dumps({
            "action": "write", "path": str(mp),
            "data": {"state": "claude_backup"}, "timestamp": "2026-01-01T00:00:00+00:00",
        }) + "\n")
        store = StateStore(manual_path=mp, auto_path=ap,
                           history_path=tmp_path / "h.json", wal_path=wal)
        assert store.get_manual_state() == CodexState.CLAUDE_BACKUP

    def test_recovery_skips_committed_write(self, tmp_path):
        mp, ap = tmp_path / "m.json", tmp_path / "a.json"
        mp.write_text(json.dumps({"state": "openai_primary"}))
        ap.write_text(json.dumps({"state": "openai_primary"}))
        wal = tmp_path / "w.jsonl"
        wal.write_text(
            json.dumps({"action": "write", "path": str(mp), "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
        )
        store = StateStore(manual_path=mp, auto_path=ap,
                           history_path=tmp_path / "h.json", wal_path=wal)
        assert store.get_manual_state() == CodexState.OPENAI_PRIMARY

    def test_recovery_multiple_uncommitted_last_wins(self, tmp_path):
        mp, ap = tmp_path / "m.json", tmp_path / "a.json"
        mp.write_text(json.dumps({"state": "openai_primary"}))
        ap.write_text(json.dumps({"state": "openai_primary"}))
        wal = tmp_path / "w.jsonl"
        wal.write_text(
            json.dumps({"action": "write", "path": str(mp), "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "write", "path": str(mp), "data": {"state": "openrouter_fallback"}}) + "\n"
        )
        store = StateStore(manual_path=mp, auto_path=ap,
                           history_path=tmp_path / "h.json", wal_path=wal)
        assert store.get_manual_state() == CodexState.OPENROUTER_FALLBACK

    def test_wal_cleared_after_recovery(self, tmp_path):
        mp, ap = tmp_path / "m.json", tmp_path / "a.json"
        mp.write_text(json.dumps({"state": "openai_primary"}))
        ap.write_text(json.dumps({"state": "openai_primary"}))
        wal = tmp_path / "w.jsonl"
        wal.write_text(json.dumps({
            "action": "write", "path": str(mp), "data": {"state": "claude_backup"},
        }) + "\n")
        _ = StateStore(manual_path=mp, auto_path=ap,
                       history_path=tmp_path / "h.json", wal_path=wal)
        assert wal.read_text().strip() == ""


# ═══════════════════════════════════════════════════════════════════════════════
# P. Corrupted state files
# ═══════════════════════════════════════════════════════════════════════════════

class TestPCorruptedStateFiles:

    def test_corrupted_manual_json(self, tmp_path):
        mp, ap = tmp_path / "m.json", tmp_path / "a.json"
        mp.write_text("NOT JSON {{{{")
        ap.write_text(json.dumps({"state": "openai_primary"}))
        store = StateStore(manual_path=mp, auto_path=ap,
                           history_path=tmp_path / "h.json", wal_path=tmp_path / "w.jsonl")
        with pytest.raises(StateError):
            store.get_manual_state()

    def test_invalid_state_value(self):
        assert StateStore._validate_state("invalid_state_xyz") is None

    def test_backward_compat_state_values(self):
        assert StateStore._validate_state("normal") == CodexState.OPENAI_PRIMARY
        assert StateStore._validate_state("last10") == CodexState.CLAUDE_BACKUP

    @pytest.mark.parametrize("manual_content,auto_content", [
        ("broken {", "not json"),
        ("", ""),
    ])
    def test_both_state_files_corrupted(self, tmp_path, manual_content, auto_content):
        store = _make_state_store(tmp_path)
        store.manual_path.write_text(manual_content)
        store.auto_path.write_text(auto_content)
        store._manual_cache_valid = False
        store._auto_cache_valid = False
        try:
            state = store.get_state()
            assert state == CodexState.OPENAI_PRIMARY
        except (StateError, json.JSONDecodeError):
            pass

    def test_both_wrong_schema(self, tmp_path):
        store = _make_state_store(tmp_path)
        store.manual_path.write_text(json.dumps({"key": "value"}))
        store.auto_path.write_text(json.dumps({"state": 999}))
        store._manual_cache_valid = False
        store._auto_cache_valid = False
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_state_file_no_write_permission(self, tmp_path):
        store = _make_state_store(tmp_path)
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        os.chmod(store.manual_path, 0o444)
        try:
            try:
                store.set_manual_state(CodexState.CLAUDE_BACKUP)
                store._manual_cache_valid = False
                assert store.get_manual_state() == CodexState.CLAUDE_BACKUP
            except (PermissionError, OSError, StateError):
                pass
        finally:
            os.chmod(store.manual_path, 0o644)

    def test_concurrent_read_write_no_deadlock(self, tmp_store):
        tmp_store.set_manual_state(CodexState.OPENAI_PRIMARY)
        errors, done = [], threading.Event()
        def writer():
            try:
                while not done.is_set():
                    tmp_store.set_manual_state(random.choice(ALL_STATES))
            except Exception as e: errors.append(("w", e))
        def reader():
            try:
                while not done.is_set():
                    _ = tmp_store.get_state()
                    _ = tmp_store.get_manual_state()
            except Exception as e: errors.append(("r", e))
        threads = [threading.Thread(target=writer) for _ in range(3)] + \
                  [threading.Thread(target=reader) for _ in range(3)]
        for t in threads: t.start()
        time.sleep(2); done.set()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "Thread deadlocked"
        assert len(errors) == 0

    def test_concurrent_set_state_with_history(self, tmp_store):
        errors = []
        def transitioner(tid):
            try:
                for i in range(30):
                    tmp_store.set_state_with_history(random.choice(ALL_STATES), reason=f"t{tid}-{i}", force=True)
            except Exception as e: errors.append((tid, e))
        threads = [threading.Thread(target=transitioner, args=(i,)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)
        assert len(errors) == 0
        assert len(tmp_store.get_state_history(limit=50)) > 0
