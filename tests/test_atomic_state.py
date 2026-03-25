"""Tests for atomic state writes, history, and anti-flap."""

import json
import os
import time
import pytest
from pathlib import Path
from router.state_store import StateStore
from router.models import CodexState
from router.errors import StateError


class TestAtomicWrites:
    def test_state_write_is_atomic(self, tmp_path):
        """State should be written atomically via temp file + rename."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        # File should exist and contain valid JSON
        data = json.loads((tmp_path / "manual.json").read_text())
        assert data["state"] == "claude_backup"

        # No temp files should remain
        tmp_files = list(tmp_path.glob(".state_*"))
        assert len(tmp_files) == 0

    def test_concurrent_writes_dont_corrupt(self, tmp_path):
        """Multiple concurrent writes should not corrupt the file."""
        import threading

        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )

        states = [CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP, CodexState.OPENAI_CONSERVATION]
        errors = []

        def writer(state):
            try:
                for _ in range(10):
                    store.set_manual_state(state)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(s,)) for s in states * 3]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write errors: {errors}"

        # File should still be valid JSON
        data = json.loads((tmp_path / "manual.json").read_text())
        assert data["state"] in {"openai_primary", "claude_backup", "openai_conservation"}


class TestStateHistory:
    def test_transition_logged(self, tmp_path):
        """State transitions should be logged to history."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )
        store.set_state_with_history(CodexState.CLAUDE_BACKUP, reason="test_transition")

        history = store.get_state_history(limit=5)
        assert len(history) == 1
        assert history[0]["to"] == "claude_backup"
        assert history[0]["reason"] == "test_transition"
        assert "timestamp" in history[0]

    def test_history_limit(self, tmp_path):
        """History should keep only the last N entries."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )

        states = [CodexState.OPENAI_CONSERVATION, CodexState.CLAUDE_BACKUP, CodexState.OPENAI_PRIMARY]
        for i in range(60):
            store.set_state_with_history(states[i % 3], reason=f"transition_{i}", force=True)

        history = store.get_state_history(limit=100)
        assert len(history) <= 50  # MAX_HISTORY_ENTRIES


class TestAntiFlap:
    def test_anti_flap_blocks_rapid_transitions(self, tmp_path):
        """Rapid state transitions should be blocked by anti-flap."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )

        store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="first")

        # Try to transition again immediately to non-emergency target — should be blocked
        with pytest.raises(StateError, match="anti_flap"):
            store.set_state_with_history(CodexState.OPENROUTER_FALLBACK, reason="second")

    def test_force_bypasses_anti_flap(self, tmp_path):
        """Force flag should bypass anti-flap protection."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )

        store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="first")
        # Force should work even immediately to non-emergency target
        store.set_state_with_history(CodexState.OPENROUTER_FALLBACK, reason="forced", force=True)

        assert store.get_state() == CodexState.OPENROUTER_FALLBACK

    def test_emergency_override_allowed(self, tmp_path):
        """Emergency targets (OPENAI_PRIMARY, CLAUDE_BACKUP) should be allowed."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )

        # Patch MIN_STATE_DURATION_S for test
        import router.state_store as ss
        original = ss.MIN_STATE_DURATION_S
        ss.MIN_STATE_DURATION_S = 999999  # Effectively disable anti-flap except emergency

        try:
            store.set_state_with_history(CodexState.OPENAI_CONSERVATION, reason="first")

            # Emergency override to OPENAI_PRIMARY should work
            store.set_state_with_history(CodexState.OPENAI_PRIMARY, reason="emergency")
            assert store.get_state() == CodexState.OPENAI_PRIMARY
        finally:
            ss.MIN_STATE_DURATION_S = original


class TestInvariantValidation:
    def test_valid_transitions(self, tmp_path):
        """All expected transitions should be valid."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )

        all_states = list(CodexState)
        for from_s in all_states:
            for to_s in all_states:
                if from_s != to_s:
                    assert store.validate_transition(from_s, to_s), \
                        f"Transition {from_s} → {to_s} should be valid"

    def test_same_state_is_valid(self, tmp_path):
        """Staying in the same state should always be valid."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
        )
        for state in CodexState:
            assert store.validate_transition(state, state)
