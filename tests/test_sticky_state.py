"""Tests for sticky state — success-count threshold for recovery to primary."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from router.models import CodexState
from router.state_store import StateStore


@pytest.fixture
def store(tmp_path):
    """Provide a StateStore with temporary paths."""
    manual = tmp_path / "codex_manual_state.json"
    auto = tmp_path / "codex_auto_state.json"
    history = tmp_path / "codex_state_history.json"
    return StateStore(manual_path=manual, auto_path=auto, history_path=history)


class TestSuccessCounterIncrement:
    """record_success should increment counter on success, reset on failure."""

    def test_initial_counter_is_zero(self, store):
        assert store._success_counter.get(CodexState.OPENAI_CONSERVATION.value, 0) == 0

    def test_success_increments_counter(self, store):
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store._success_counter[CodexState.OPENAI_CONSERVATION.value] == 1
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store._success_counter[CodexState.OPENAI_CONSERVATION.value] == 2

    def test_failure_resets_counter_to_zero(self, store):
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENAI_CONSERVATION, False)
        assert store._success_counter[CodexState.OPENAI_CONSERVATION.value] == 0

    def test_counters_are_per_state(self, store):
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENROUTER_FALLBACK, True)
        assert store._success_counter[CodexState.OPENAI_CONSERVATION.value] == 2
        assert store._success_counter[CodexState.OPENROUTER_FALLBACK.value] == 1


class TestRecoveryBlockedByLowCount:
    """Recovery to primary should be blocked when success count < threshold."""

    def test_zero_successes_blocks_recovery(self, store):
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False

    def test_one_success_blocks_recovery(self, store):
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False

    def test_two_successes_blocks_recovery(self, store):
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False

    def test_can_transition_does_not_block_sticky(self, store):
        """can_transition does NOT check sticky — emergency override always works.
        Sticky state is checked by callers via can_recover_to_primary()."""
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)  # Let auto state take effect
        # can_transition allows emergency override regardless of sticky
        allowed, reason = store.can_transition(CodexState.OPENAI_PRIMARY)
        assert allowed is True
        assert "emergency" in reason.lower()
        # But can_recover_to_primary still blocks
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False


class TestRecoveryAllowedAfterThreshold:
    """Recovery to primary should be allowed after N consecutive successes."""

    def test_three_successes_allows_recovery(self, store):
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        for _ in range(3):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True

    def test_can_transition_allowed_after_threshold(self, store):
        """can_transition always allows recovery to primary (emergency override).
        The sticky logic is enforced by callers via can_recover_to_primary()."""
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)  # Let auto state take effect
        for _ in range(3):
            store.record_success(CodexState.OPENAI_CONSERVATION, True)
        allowed, reason = store.can_transition(CodexState.OPENAI_PRIMARY)
        assert allowed is True
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True

    def test_failure_midway_resets_requires_full_threshold(self, store):
        """If a failure happens mid-count, need to rebuild from 0."""
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENAI_CONSERVATION, False)  # reset!
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False
        # Now need 3 more consecutive successes
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is False
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        assert store.can_recover_to_primary(CodexState.OPENAI_CONSERVATION) is True

    def test_recovery_from_fallback(self, store):
        """Sticky state should also work for OPENROUTER_FALLBACK → PRIMARY."""
        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)
        assert store.can_recover_to_primary(CodexState.OPENROUTER_FALLBACK) is False
        for _ in range(3):
            store.record_success(CodexState.OPENROUTER_FALLBACK, True)
        assert store.can_recover_to_primary(CodexState.OPENROUTER_FALLBACK) is True


class TestEmergencyOverrideBypassesSticky:
    """Emergency overrides to primary should always work, bypassing sticky state."""

    def test_force_bypasses_sticky_in_can_transition(self, store):
        """can_transition always allows emergency targets.
        Force parameter bypasses anti-flap for non-emergency targets."""
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        store.set_manual_state(None)  # Let auto state take effect
        # Emergency target always allowed
        allowed, reason = store.can_transition(CodexState.OPENAI_PRIMARY)
        assert allowed is True
        # Force also works
        allowed, reason = store.can_transition(CodexState.OPENAI_PRIMARY, force=True)
        assert allowed is True

    def test_set_state_with_history_force_bypasses_sticky(self, store):
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        # Should work with force=True even without meeting threshold
        result = store.set_state_with_history(
            CodexState.OPENAI_PRIMARY, reason="emergency", force=True
        )
        assert result is True
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_claude_backup_always_allowed(self, store):
        """Transition to CLAUDE_BACKUP should always be allowed (emergency target)."""
        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)
        allowed, reason = store.can_transition(CodexState.CLAUDE_BACKUP)
        assert allowed is True


class TestResetSuccessCounter:
    """reset_success_counter should clear the counter for a state."""

    def test_reset_clears_counter(self, store):
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.reset_success_counter(CodexState.OPENAI_CONSERVATION)
        assert store._success_counter.get(CodexState.OPENAI_CONSERVATION.value, 0) == 0

    def test_reset_only_affects_target_state(self, store):
        store.record_success(CodexState.OPENAI_CONSERVATION, True)
        store.record_success(CodexState.OPENROUTER_FALLBACK, True)
        store.record_success(CodexState.OPENROUTER_FALLBACK, True)
        store.reset_success_counter(CodexState.OPENAI_CONSERVATION)
        assert store._success_counter.get(CodexState.OPENAI_CONSERVATION.value, 0) == 0
        assert store._success_counter[CodexState.OPENROUTER_FALLBACK.value] == 2


class TestAntiFlapPreserved:
    """Existing anti-flap logic should still work alongside sticky state."""

    def test_anti_flap_blocks_downgrade(self, store):
        """Downgrades should still be blocked by anti-flap timing."""
        store.set_auto_state(CodexState.OPENAI_PRIMARY)
        # Log a recent transition
        store.log_state_transition(
            CodexState.OPENAI_CONSERVATION, CodexState.OPENAI_PRIMARY, "test"
        )
        # Try to downgrade — should be blocked by anti-flap (just transitioned)
        allowed, reason = store.can_transition(CodexState.OPENAI_CONSERVATION)
        assert allowed is False
        assert "anti_flap" in reason

    def test_sticky_not_applied_to_non_primary_targets(self, store):
        """Sticky state should only block transitions TO primary, not to other states."""
        store.set_auto_state(CodexState.OPENAI_PRIMARY)
        store.set_manual_state(None)  # Let auto state take effect
        # Transitioning from primary to conservation should not be affected by sticky
        allowed, reason = store.can_transition(CodexState.OPENAI_CONSERVATION)
        # May be blocked by anti-flap (no history) but not by sticky
        # With no history, should be allowed
        assert "sticky" not in reason.lower()
