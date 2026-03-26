"""Integration tests for state flow — transitions, persistence, WAL recovery, sticky state."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from router.models import TaskMeta, TaskClass, TaskRisk, CodexState
from router.state_store import StateStore, reset_state_store
from router.policy import route_task, build_chain, resolve_state, reset_breaker, reset_notifier
from router.classifier import classify
from tests.conftest import make_result, make_task


def _setup(tmp_path, monkeypatch):
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
    from router.state_store import get_state_store
    return get_state_store(), state_dir, runtime_dir


def _make_state_store(state_dir):
    return StateStore(
        manual_path=state_dir / "manual.json",
        auto_path=state_dir / "auto.json",
        history_path=state_dir / "history.json",
        wal_path=state_dir / "wal.jsonl",
    )


class TestStateTransitionFlow:
    def test_transition_round_trip(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        assert store.get_state() == CodexState.OPENAI_PRIMARY

        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        assert store.get_state() == CodexState.CLAUDE_BACKUP

        task = make_task()
        chain = build_chain(task, CodexState.CLAUDE_BACKUP)
        assert chain[0].tool == "claude_code"

        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        chain2 = build_chain(task, CodexState.OPENAI_PRIMARY)
        assert chain2[0].tool == "codex_cli"

    def test_state_transitions_via_route_task(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        task = make_task("trans-route-001")

        mock = make_result("trans-route-001")
        with patch("router.policy.run_codex", return_value=mock):
            d1, _ = route_task(task)
        assert d1.state == "openai_primary"

        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        mock2 = make_result("trans-route-001", tool="claude_code",
                            backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock2):
            d2, _ = route_task(task)
        assert d2.state == "claude_backup"
        assert d2.chain[0].tool == "claude_code"


class TestManualOverrideFlow:
    def test_manual_override_then_clear(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        store.set_manual_state(None)
        store.set_auto_state(CodexState.CLAUDE_BACKUP)
        assert resolve_state(store) == CodexState.CLAUDE_BACKUP

        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        assert resolve_state(store) == CodexState.OPENAI_PRIMARY

        store.set_manual_state(None)
        assert resolve_state(store) == CodexState.CLAUDE_BACKUP

    def test_manual_override_with_routing(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        task = make_task("override-001")

        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        mock_claude = make_result("override-001", tool="claude_code",
                                  backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock_claude):
            d1, r1 = route_task(task)
        assert d1.state == "claude_backup"
        assert r1.tool == "claude_code"

        store.set_manual_state(None)
        mock_codex = make_result("override-001")
        with patch("router.policy.run_codex", return_value=mock_codex):
            d2, r2 = route_task(task)
        assert d2.state == "openai_primary"
        assert r2.tool == "codex_cli"


class TestStatePersistenceFlow:
    def test_state_persists_across_restart(self, tmp_path, monkeypatch):
        _, state_dir, _ = _setup(tmp_path, monkeypatch)
        store1 = _make_state_store(state_dir)
        store1.set_manual_state(CodexState.CLAUDE_BACKUP)

        store2 = _make_state_store(state_dir)
        assert store2.get_state() == CodexState.CLAUDE_BACKUP

    def test_auto_state_persists(self, tmp_path, monkeypatch):
        _, state_dir, _ = _setup(tmp_path, monkeypatch)
        store1 = _make_state_store(state_dir)
        store1.set_auto_state(CodexState.OPENAI_CONSERVATION)

        store2 = _make_state_store(state_dir)
        assert store2.get_auto_state() == CodexState.OPENAI_CONSERVATION


class TestWALRecoveryFlow:
    def test_wal_recovers_uncommitted_write(self, tmp_path, monkeypatch):
        _, state_dir, _ = _setup(tmp_path, monkeypatch)
        store1 = _make_state_store(state_dir)
        store1.set_manual_state(CodexState.OPENAI_PRIMARY)

        wal_path = state_dir / "wal.jsonl"
        wal_path.write_text(json.dumps({
            "action": "write", "path": str(state_dir / "manual.json"),
            "data": {"state": "claude_backup"}, "timestamp": "2025-01-01T00:00:00+00:00",
        }) + "\n")
        (state_dir / "manual.json").write_text("CORRUPTED")

        store2 = _make_state_store(state_dir)
        store2.recover_from_wal()
        assert store2.get_manual_state() == CodexState.CLAUDE_BACKUP

    def test_wal_skips_committed_entries(self, tmp_path, monkeypatch):
        _, state_dir, _ = _setup(tmp_path, monkeypatch)
        wal_path = state_dir / "wal.jsonl"
        entries = [
            json.dumps({"action": "write", "path": str(state_dir / "manual.json"),
                         "data": {"state": "claude_backup"}, "timestamp": "2025-01-01T00:00:00"}),
            json.dumps({"action": "committed", "timestamp": "2025-01-01T00:00:01"}),
        ]
        wal_path.write_text("\n".join(entries) + "\n")
        (state_dir / "manual.json").write_text(json.dumps({"state": "openrouter_fallback"}))

        store = _make_state_store(state_dir)
        assert store.get_manual_state() == CodexState.OPENROUTER_FALLBACK


class TestStickyStateFlow:
    def test_sticky_state_requires_consecutive_successes(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        state = CodexState.CLAUDE_BACKUP
        assert store.can_recover_to_primary(state) is False

        store.record_success(state, True)
        store.record_success(state, True)
        assert store.can_recover_to_primary(state) is False

        store.record_success(state, True)
        assert store.can_recover_to_primary(state) is True

    def test_failure_resets_sticky_counter(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        state = CodexState.OPENAI_CONSERVATION
        store.record_success(state, True)
        store.record_success(state, True)
        store.record_success(state, False)
        assert store.can_recover_to_primary(state) is False

        store.record_success(state, True)
        store.record_success(state, True)
        assert store.can_recover_to_primary(state) is False
        store.record_success(state, True)
        assert store.can_recover_to_primary(state) is True

    def test_sticky_state_with_routing(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        task = make_task("sticky-001")

        fail = make_result("sticky-001", success=False, error_type="provider_unavailable")
        success_claude = make_result("sticky-001", success=True, tool="claude_code",
                                     backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_codex", return_value=fail), \
             patch("router.policy.run_claude", return_value=success_claude):
            d1, r1 = route_task(task)
        assert r1.success is True

        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        store.record_success(CodexState.CLAUDE_BACKUP, True)
        store.record_success(CodexState.CLAUDE_BACKUP, True)
        assert store.can_recover_to_primary(CodexState.CLAUDE_BACKUP) is False
        store.record_success(CodexState.CLAUDE_BACKUP, True)
        assert store.can_recover_to_primary(CodexState.CLAUDE_BACKUP) is True

        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        assert store.get_state() == CodexState.OPENAI_PRIMARY


class TestHistoryTrackingFlow:
    def test_history_records_transitions(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        store.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP,
                                   reason="codex quota exhausted")
        store.log_state_transition(CodexState.CLAUDE_BACKUP, CodexState.OPENROUTER_FALLBACK,
                                   reason="claude rate limited")
        store.log_state_transition(CodexState.OPENROUTER_FALLBACK, CodexState.OPENAI_PRIMARY,
                                   reason="recovery: quota reset")

        history = store.get_state_history(limit=10)
        assert len(history) == 3
        assert history[0]["from"] == "openai_primary"
        assert history[0]["to"] == "claude_backup"
        assert history[1]["reason"] == "claude rate limited"
        assert history[2]["to"] == "openai_primary"

    def test_history_persists_across_restart(self, tmp_path, monkeypatch):
        _, state_dir, _ = _setup(tmp_path, monkeypatch)
        store1 = _make_state_store(state_dir)
        store1.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP, reason="test")

        store2 = _make_state_store(state_dir)
        history = store2.get_state_history()
        assert len(history) == 1
        assert history[0]["to"] == "claude_backup"

    def test_history_truncates_at_max(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        for i in range(55):
            from_s = CodexState.OPENAI_PRIMARY if i % 2 == 0 else CodexState.CLAUDE_BACKUP
            to_s = CodexState.CLAUDE_BACKUP if i % 2 == 0 else CodexState.OPENAI_PRIMARY
            store.log_state_transition(from_s, to_s, reason=f"transition {i}")

        history = store.get_state_history(limit=100)
        assert len(history) == 50

    def test_history_has_timestamps(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        store.log_state_transition(CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP, reason="test")

        history = store.get_state_history()
        assert "timestamp" in history[0]
        from datetime import datetime
        ts = datetime.fromisoformat(history[0]["timestamp"].replace("Z", "+00:00"))
        assert ts is not None


class TestStateFlowEdgeCases:
    def test_manual_set_route_clear_route_cycle(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        task = make_task("cycle-001")

        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        mock_claude = make_result("cycle-001", tool="claude_code",
                                  backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock_claude):
            d1, r1 = route_task(task)
        assert d1.state == "claude_backup"
        assert r1.tool == "claude_code"

        store.set_manual_state(None)
        mock_codex = make_result("cycle-001")
        with patch("router.policy.run_codex", return_value=mock_codex):
            d2, r2 = route_task(task)
        assert d2.state == "openai_primary"
        assert r2.tool == "codex_cli"

    def test_wal_recovery_then_route(self, tmp_path, monkeypatch):
        _, state_dir, runtime_dir = _setup(tmp_path, monkeypatch)
        store1 = _make_state_store(state_dir)
        store1.set_manual_state(CodexState.OPENAI_PRIMARY)

        wal_path = state_dir / "wal.jsonl"
        wal_path.write_text(json.dumps({
            "action": "write", "path": str(state_dir / "manual.json"),
            "data": {"state": "claude_backup"}, "timestamp": "2025-06-01T00:00:00+00:00",
        }) + "\n")
        (state_dir / "manual.json").write_text("CORRUPTED")

        store2 = _make_state_store(state_dir)
        assert store2.get_manual_state() == CodexState.CLAUDE_BACKUP

        import router.state_store as ss_mod
        ss_mod._instance = store2

        task = make_task("wal-route-001")
        mock = make_result("wal-route-001", tool="claude_code",
                           backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock):
            decision, result = route_task(task)
        assert result.success is True
        assert decision.state == "claude_backup"

    def test_unknown_class_defaults_to_implementation(self, tmp_path, monkeypatch):
        store, _, _ = _setup(tmp_path, monkeypatch)
        meta = classify("xyzzy plugh nothing matches these keywords")
        assert meta.task_class == TaskClass.IMPLEMENTATION

        mock = make_result(meta.task_id)
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(meta)
        assert result.success is True
