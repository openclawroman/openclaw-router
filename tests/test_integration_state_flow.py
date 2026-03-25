"""Integration tests for state flow — transitions, persistence, WAL recovery, sticky state.

Exercises the full state lifecycle: set → resolve → route → transition → persist → recover.
All executors are mocked; state files use temp directories.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState,
    ExecutorResult,
)
from router.state_store import StateStore, reset_state_store
from router.classifier import classify
from router.policy import (
    route_task, resolve_state, build_chain, reset_breaker, reset_notifier,
)

from tests.conftest import MockExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(task_id: str, *, success: bool = True, tool: str = "codex_cli",
                 backend: str = "openai_native", model_profile: str = "codex_primary",
                 cost_usd: float = 0.002, error_type: str | None = None):
    """Build ExecutorResult. On failure, final_summary is None to avoid partial_success flag."""
    return ExecutorResult(
        task_id=task_id, tool=tool, backend=backend, model_profile=model_profile,
        success=success, normalized_error=error_type if not success else None,
        exit_code=0 if success else 1, latency_ms=120,
        cost_estimate_usd=cost_usd if success else 0.0,
        final_summary="Done" if success else None,
    )


def _setup_isolated_env(tmp_path, monkeypatch):
    """Set up temp state + runtime dirs, reset singletons."""
    state_dir = tmp_path / "state"
    runtime_dir = tmp_path / "runtime"
    state_dir.mkdir()
    runtime_dir.mkdir()

    monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
    monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
    monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
    monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")
    monkeypatch.setattr("router.logger.RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", runtime_dir / "routing.jsonl")
    monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", runtime_dir / "alerts.jsonl")

    reset_state_store()
    reset_breaker()
    reset_notifier()

    # Force-create singleton with patched paths
    from router.state_store import get_state_store
    store = get_state_store()
    return store, state_dir, runtime_dir


def _make_task(task_id: str = "sf-001", summary: str = "Implement feature"):
    return TaskMeta(
        task_id=task_id, agent="coder",
        task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
        repo_path="/tmp/repo", cwd="/tmp/repo",
        summary=summary,
    )


# ===================================================================
# Test 1: State transition flow
# ===================================================================

class TestStateTransitionFlow:
    """openai_primary → claude_backup → openai_primary."""

    def test_transition_round_trip(self, tmp_path, monkeypatch):
        """Set state to claude_backup, verify chain changes, then restore."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        # Initial: openai_primary
        assert store.get_state() == CodexState.OPENAI_PRIMARY

        # Transition to claude_backup
        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store.get_state() == CodexState.CLAUDE_BACKUP

        # Build chain — should start with claude
        task = _make_task()
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        assert chain[0].tool == "claude_code"
        assert chain[0].backend == "anthropic"

        # Transition back to openai_primary
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        assert store.get_state() == CodexState.OPENAI_PRIMARY

        chain2 = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert chain2[0].tool == "codex_cli"
        assert chain2[0].backend == "openai_native"

    def test_state_transitions_via_route_task(self, tmp_path, monkeypatch):
        """Changing state between route_task calls changes the chain."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        task = _make_task("trans-route-001")
        mock = _make_result("trans-route-001")

        # First route with openai_primary
        with patch("router.policy.run_codex", return_value=mock):
            d1, _ = route_task(task)
        assert d1.state == "openai_primary"
        assert d1.chain[0].tool == "codex_cli"

        # Switch to claude_backup
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        mock2 = _make_result("trans-route-001", tool="claude_code",
                              backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock2):
            d2, _ = route_task(task)
        assert d2.state == "claude_backup"
        assert d2.chain[0].tool == "claude_code"


# ===================================================================
# Test 2: Manual override flow
# ===================================================================

class TestManualOverrideFlow:
    """Set manual → route → clear manual → route (different chain)."""

    def test_manual_override_then_clear(self, tmp_path, monkeypatch):
        """Manual state overrides auto; clearing manual reverts to auto."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        # Clear manual and set auto to claude_backup
        store.set_manual_state(None)
        store.set_auto_state(CodexState.CLAUDE_BACKUP)

        # No manual → should resolve to claude_backup (auto)
        assert resolve_state(store) == CodexState.CLAUDE_BACKUP

        # Set manual to openai_primary → manual wins
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        assert resolve_state(store) == CodexState.OPENAI_PRIMARY

        # Clear manual → reverts to auto (claude_backup)
        store.set_manual_state(None)
        assert resolve_state(store) == CodexState.CLAUDE_BACKUP

    def test_manual_override_with_routing(self, tmp_path, monkeypatch):
        """Full routing with manual override, then clear, verify chain changes."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        task = _make_task("override-001")
        mock_codex = _make_result("override-001")
        mock_claude = _make_result("override-001", tool="claude_code",
                                    backend="anthropic", model_profile="claude_sonnet")

        # Manual override to claude_backup
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        with patch("router.policy.run_claude", return_value=mock_claude):
            d1, r1 = route_task(task)
        assert d1.state == "claude_backup"
        assert r1.tool == "claude_code"

        # Clear manual
        store.set_manual_state(None)

        with patch("router.policy.run_codex", return_value=mock_codex):
            d2, r2 = route_task(task)
        assert d2.state == "openai_primary"
        assert r2.tool == "codex_cli"


# ===================================================================
# Test 3: State persistence flow
# ===================================================================

class TestStatePersistenceFlow:
    """Set state → restart (new StateStore) → state persists."""

    def test_state_persists_across_restart(self, tmp_path, monkeypatch):
        """State set in one StateStore is visible after creating a new one."""
        state_dir = tmp_path / "state"
        runtime_dir = tmp_path / "runtime"
        state_dir.mkdir()
        runtime_dir.mkdir()

        monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
        monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
        monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
        monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")

        # First store: set state
        store1 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        store1.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store1.get_state() == CodexState.CLAUDE_BACKUP

        # "Restart" — create new store from same files
        store2 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        assert store2.get_state() == CodexState.CLAUDE_BACKUP

    def test_auto_state_persists(self, tmp_path, monkeypatch):
        """Auto state persists across store recreation."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
        monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
        monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
        monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")

        store1 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        store1.set_auto_state(CodexState.OPENAI_CONSERVATION)

        store2 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        assert store2.get_auto_state() == CodexState.OPENAI_CONSERVATION


# ===================================================================
# Test 4: WAL recovery flow
# ===================================================================

class TestWALRecoveryFlow:
    """Write state → corrupt primary → recover_from_wal → state correct."""

    def test_wal_recovers_uncommitted_write(self, tmp_path, monkeypatch):
        """An uncommitted WAL entry is applied on recovery."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
        monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
        monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
        monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")

        # Set initial state
        store1 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        store1.set_manual_state(CodexState.OPENAI_PRIMARY)

        # Simulate corruption: write a WAL entry without committed marker
        wal_path = state_dir / "wal.jsonl"
        wal_path.write_text("")  # Clear existing WAL

        corrupt_entry = json.dumps({
            "action": "write",
            "path": str(state_dir / "manual.json"),
            "data": {"state": "claude_backup"},
            "timestamp": "2025-01-01T00:00:00+00:00",
        }) + "\n"
        wal_path.write_text(corrupt_entry)

        # Corrupt the primary file
        (state_dir / "manual.json").write_text("CORRUPTED")

        # New store should recover from WAL
        store2 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        recovered = store2.recover_from_wal()  # Called in __init__, but calling again to verify
        # The recovery should have applied the uncommitted write during __init__
        # Now read the manual state — should be claude_backup
        state = store2.get_manual_state()
        assert state == CodexState.CLAUDE_BACKUP

    def test_wal_skips_committed_entries(self, tmp_path, monkeypatch):
        """Committed WAL entries are not re-applied on recovery."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
        monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
        monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
        monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")

        # Write a committed WAL entry
        wal_path = state_dir / "wal.jsonl"
        entries = [
            json.dumps({"action": "write", "path": str(state_dir / "manual.json"),
                         "data": {"state": "claude_backup"}, "timestamp": "2025-01-01T00:00:00"}),
            json.dumps({"action": "committed", "timestamp": "2025-01-01T00:00:01"}),
        ]
        wal_path.write_text("\n".join(entries) + "\n")

        # Write the expected state to primary
        (state_dir / "manual.json").write_text(json.dumps({"state": "openrouter_fallback"}))

        store = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        # Should NOT have re-applied the committed entry — primary has openrouter_fallback
        assert store.get_manual_state() == CodexState.OPENROUTER_FALLBACK


# ===================================================================
# Test 5: Sticky state flow
# ===================================================================

class TestStickyStateFlow:
    """Fail state → wait → recover to primary (threshold met)."""

    def test_sticky_state_requires_consecutive_successes(self, tmp_path, monkeypatch):
        """record_success increments counter; can_recover checks threshold."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        state = CodexState.CLAUDE_BACKUP
        assert store.can_recover_to_primary(state) is False

        # Record 2 successes — not enough (threshold is 3)
        store.record_success(state, True)
        store.record_success(state, True)
        assert store.can_recover_to_primary(state) is False

        # Third success → threshold met
        store.record_success(state, True)
        assert store.can_recover_to_primary(state) is True

    def test_failure_resets_sticky_counter(self, tmp_path, monkeypatch):
        """A failure resets the consecutive success counter."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        state = CodexState.OPENAI_CONSERVATION
        store.record_success(state, True)
        store.record_success(state, True)
        # Fail!
        store.record_success(state, False)
        assert store.can_recover_to_primary(state) is False

        # Need 3 more successes
        store.record_success(state, True)
        store.record_success(state, True)
        assert store.can_recover_to_primary(state) is False
        store.record_success(state, True)
        assert store.can_recover_to_primary(state) is True

    def test_sticky_state_with_routing(self, tmp_path, monkeypatch):
        """Full flow: fail → fallback state → record successes → recover."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        task = _make_task("sticky-001")

        # Fail the first executor → trigger fallback
        fail = _make_result("sticky-001", success=False, error_type="provider_unavailable")
        success_claude = _make_result("sticky-001", success=True, tool="claude_code",
                                       backend="anthropic", model_profile="claude_sonnet")

        with patch("router.policy.run_codex", return_value=fail), \
             patch("router.policy.run_claude", return_value=success_claude):
            d1, r1 = route_task(task)
        assert r1.success is True

        # Set state to claude_backup (simulating auto-transition)
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        # Record successes in claude_backup state
        store.record_success(CodexState.CLAUDE_BACKUP, True)
        store.record_success(CodexState.CLAUDE_BACKUP, True)
        assert store.can_recover_to_primary(CodexState.CLAUDE_BACKUP) is False
        store.record_success(CodexState.CLAUDE_BACKUP, True)
        assert store.can_recover_to_primary(CodexState.CLAUDE_BACKUP) is True

        # Now we can recover to primary
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        assert store.get_state() == CodexState.OPENAI_PRIMARY


# ===================================================================
# Test 6: History tracking flow
# ===================================================================

class TestHistoryTrackingFlow:
    """3 transitions → history has 3 entries."""

    def test_history_records_transitions(self, tmp_path, monkeypatch):
        """State transitions are logged to history."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        # Make 3 transitions
        store.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP,
            reason="codex quota exhausted",
        )
        store.log_state_transition(
            CodexState.CLAUDE_BACKUP, CodexState.OPENROUTER_FALLBACK,
            reason="claude rate limited",
        )
        store.log_state_transition(
            CodexState.OPENROUTER_FALLBACK, CodexState.OPENAI_PRIMARY,
            reason="recovery: quota reset",
        )

        history = store.get_state_history(limit=10)
        assert len(history) == 3

        assert history[0]["from"] == "openai_primary"
        assert history[0]["to"] == "claude_backup"
        assert history[0]["reason"] == "codex quota exhausted"

        assert history[1]["from"] == "claude_backup"
        assert history[1]["to"] == "openrouter_fallback"
        assert history[1]["reason"] == "claude rate limited"

        assert history[2]["from"] == "openrouter_fallback"
        assert history[2]["to"] == "openai_primary"
        assert history[2]["reason"] == "recovery: quota reset"

    def test_history_persists_across_restart(self, tmp_path, monkeypatch):
        """History file survives store recreation."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
        monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
        monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
        monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")

        store1 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        store1.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP, reason="test",
        )

        store2 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        history = store2.get_state_history()
        assert len(history) == 1
        assert history[0]["to"] == "claude_backup"

    def test_history_truncates_at_max(self, tmp_path, monkeypatch):
        """History is capped at MAX_HISTORY_ENTRIES."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        # Write 55 entries (exceeds MAX_HISTORY_ENTRIES=50)
        for i in range(55):
            from_state = CodexState.OPENAI_PRIMARY if i % 2 == 0 else CodexState.CLAUDE_BACKUP
            to_state = CodexState.CLAUDE_BACKUP if i % 2 == 0 else CodexState.OPENAI_PRIMARY
            store.log_state_transition(from_state, to_state, reason=f"transition {i}")

        history = store.get_state_history(limit=100)
        assert len(history) == 50  # Capped

    def test_history_has_timestamps(self, tmp_path, monkeypatch):
        """Each history entry has a timestamp field."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        store.log_state_transition(
            CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP, reason="test",
        )
        history = store.get_state_history()
        assert len(history) == 1
        assert "timestamp" in history[0]
        # Timestamp should be ISO format
        from datetime import datetime
        ts = datetime.fromisoformat(history[0]["timestamp"].replace("Z", "+00:00"))
        assert ts is not None


# ===================================================================
# EDGE CASE: Manual set → route → clear → route again
# ===================================================================

class TestManualSetRouteClearRoute:
    """Set manual state, route a task, clear state, route again — different chain."""

    def test_manual_set_route_clear_route_cycle(self, tmp_path, monkeypatch):
        """Full cycle: manual claude_backup → route → clear → route again with openai_primary."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        task = _make_task("cycle-001")

        # Phase 1: Set manual to claude_backup, route
        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        mock_claude = _make_result("cycle-001", tool="claude_code",
                                    backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock_claude):
            d1, r1 = route_task(task)
        assert d1.state == "claude_backup"
        assert r1.tool == "claude_code"

        # Phase 2: Clear manual state → default openai_primary
        store.set_manual_state(None)
        mock_codex = _make_result("cycle-001")
        with patch("router.policy.run_codex", return_value=mock_codex):
            d2, r2 = route_task(task)
        assert d2.state == "openai_primary"
        assert r2.tool == "codex_cli"


# ===================================================================
# EDGE CASE: WAL recovery then immediately route
# ===================================================================

class TestWALRecoveryThenRoute:
    """Recover from WAL → immediately route a task → success."""

    def test_wal_recovery_then_route(self, tmp_path, monkeypatch):
        """Corrupt state, recover from WAL, then route a task on the recovered state."""
        state_dir = tmp_path / "state"
        runtime_dir = tmp_path / "runtime"
        state_dir.mkdir()
        runtime_dir.mkdir()

        monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
        monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
        monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
        monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")
        monkeypatch.setattr("router.logger.RUNTIME_DIR", runtime_dir)
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", runtime_dir / "routing.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", runtime_dir / "alerts.jsonl")

        reset_state_store()
        reset_breaker()
        reset_notifier()

        # Step 1: Set up initial state
        store1 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        store1.set_manual_state(CodexState.OPENAI_PRIMARY)

        # Step 2: Simulate crash — corrupt primary, leave WAL with uncommitted write
        wal_path = state_dir / "wal.jsonl"
        wal_path.write_text("")
        corrupt_entry = json.dumps({
            "action": "write",
            "path": str(state_dir / "manual.json"),
            "data": {"state": "claude_backup"},
            "timestamp": "2025-06-01T00:00:00+00:00",
        }) + "\n"
        wal_path.write_text(corrupt_entry)
        (state_dir / "manual.json").write_text("CORRUPTED")

        # Step 3: New store recovers from WAL
        store2 = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        assert store2.get_manual_state() == CodexState.CLAUDE_BACKUP

        # Step 4: Force singleton for route_task
        import router.state_store as ss_mod
        ss_mod._instance = store2

        # Step 5: Route immediately — should use claude_backup (recovered state)
        task = _make_task("wal-route-001")
        mock = _make_result("wal-route-001", tool="claude_code",
                             backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock):
            decision, result = route_task(task)

        assert result.success is True
        assert decision.state == "claude_backup"


# ===================================================================
# EDGE CASE: Unknown task class defaults gracefully
# ===================================================================

class TestUnknownTaskClass:
    """Task with text that doesn\'t match any classifier keywords → defaults to IMPLEMENTATION."""

    def test_unknown_class_defaults_to_implementation(self, tmp_path, monkeypatch):
        """Ambiguous task text defaults to IMPLEMENTATION and routes normally."""
        store, _, _ = _setup_isolated_env(tmp_path, monkeypatch)

        meta = classify("xyzzy plugh nothing matches these keywords")
        assert meta.task_class == TaskClass.IMPLEMENTATION

        mock = _make_result(meta.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(meta)

        assert result.success is True
        assert decision.state == "openai_primary"
