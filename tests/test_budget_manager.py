"""Tests for BudgetManager — budget tracking, auto-transition, persistence."""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from router.budget_manager import (
    BudgetManager,
    _BudgetState,
    _period_start,
    get_budget_manager,
    reset_budget_manager,
)
from router.models import CodexState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    *,
    enabled: bool = True,
    period: str = "monthly",
    tokens_limit: int = 1_000_000,
    cost_limit: float = 100.0,
    warning: float = 0.80,
    critical: float = 0.90,
    exhausted: float = 1.0,
) -> dict:
    """Build a config dict for testing."""
    return {
        "budget": {
            "enabled": enabled,
            "period": period,
            "limits": {
                "tokens": tokens_limit,
                "cost_usd": cost_limit,
            },
            "thresholds": {
                "warning": warning,
                "critical": critical,
                "exhausted": exhausted,
            },
        }
    }


class MockStateStore:
    """Minimal state store mock for budget tests."""

    def __init__(self):
        self._manual: CodexState | None = None
        self._auto: CodexState | None = None

    def get_manual_state(self) -> CodexState | None:
        return self._manual

    def set_manual_state(self, state: CodexState | None):
        self._manual = state

    def get_auto_state(self) -> CodexState | None:
        return self._auto

    def set_auto_state(self, state: CodexState):
        self._auto = state

    def get_state(self) -> CodexState:
        """Effective state (manual > auto > default)."""
        if self._manual is not None:
            return self._manual
        if self._auto is not None:
            return self._auto
        return CodexState.OPENAI_PRIMARY


def _make_real_state_store(state_dir: Path):
    """Create a real StateStore with isolated paths and no default manual state."""
    from router.state_store import StateStore
    store = StateStore(
        manual_path=state_dir / "manual.json",
        auto_path=state_dir / "auto.json",
        history_path=state_dir / "history.json",
        wal_path=state_dir / "wal.jsonl",
    )
    # Clear default manual state so budget transitions aren't blocked
    store.set_manual_state(None)
    return store


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestBudgetTracking:
    """Budget tracking accumulates usage correctly."""

    def test_tokens_accumulate(self, tmp_path):
        config = _make_config()
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage(tokens=1000)
        bm.record_usage(tokens=2000)

        assert bm.tokens_used == 3000

    def test_cost_accumulates(self, tmp_path):
        config = _make_config()
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage(cost_usd=10.0)
        bm.record_usage(cost_usd=5.5)

        assert bm.cost_usd == pytest.approx(15.5)

    def test_tokens_and_cost_together(self, tmp_path):
        config = _make_config()
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage(tokens=500, cost_usd=2.0)
        bm.record_usage(tokens=300, cost_usd=1.0)

        assert bm.tokens_used == 800
        assert bm.cost_usd == pytest.approx(3.0)

    def test_zero_usage(self, tmp_path):
        config = _make_config()
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage()

        assert bm.tokens_used == 0
        assert bm.cost_usd == 0.0


class TestAutoTransitionWarning:
    """Auto-transition at warning threshold (80%)."""

    def test_warning_by_tokens(self, tmp_path):
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        # 79% — no transition
        bm.record_usage(tokens=790)
        result = bm.check_and_transition()
        assert result is None

        # 80% — triggers warning
        bm.record_usage(tokens=10)  # 800/1000 = 80%
        result = bm.check_and_transition()
        assert result == CodexState.OPENAI_CONSERVATION
        assert store.get_auto_state() == CodexState.OPENAI_CONSERVATION

    def test_warning_by_cost(self, tmp_path):
        config = _make_config(cost_limit=100.0)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(cost_usd=80.0)  # exactly 80%
        result = bm.check_and_transition()
        assert result == CodexState.OPENAI_CONSERVATION

    def test_warning_not_duplicated(self, tmp_path):
        """Warning fires only once per period."""
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(tokens=850)  # 85% — triggers warning (not yet critical)
        result1 = bm.check_and_transition()
        assert result1 == CodexState.OPENAI_CONSERVATION

        # Further check should not re-trigger at same level
        result2 = bm.check_and_transition()
        assert result2 is None


class TestAutoTransitionCritical:
    """Auto-transition at critical threshold (90%)."""

    def test_critical_from_warning(self, tmp_path):
        """Escalate from warning → critical."""
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(tokens=850)  # 85% → warning
        bm.check_and_transition()
        assert store.get_auto_state() == CodexState.OPENAI_CONSERVATION

        bm.record_usage(tokens=50)  # 900/1000 = 90% → critical
        result = bm.check_and_transition()
        assert result == CodexState.CLAUDE_BACKUP
        assert store.get_auto_state() == CodexState.CLAUDE_BACKUP

    def test_critical_skips_warning(self, tmp_path):
        """Jump directly to critical if usage crosses both thresholds at once."""
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(tokens=950)  # 95% — crosses both warning and critical
        result = bm.check_and_transition()
        # Should go to critical (highest breached)
        assert result == CodexState.CLAUDE_BACKUP


class TestAutoTransitionExhausted:
    """Auto-transition at exhausted threshold (100%)."""

    def test_exhausted_from_critical(self, tmp_path):
        """Escalate from critical → exhausted."""
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(tokens=950)
        bm.check_and_transition()  # → critical
        assert store.get_auto_state() == CodexState.CLAUDE_BACKUP

        bm.record_usage(tokens=100)  # 1050/1000 = 105% → exhausted
        result = bm.check_and_transition()
        assert result == CodexState.OPENROUTER_FALLBACK
        assert store.get_auto_state() == CodexState.OPENROUTER_FALLBACK

    def test_exhausted_direct(self, tmp_path):
        """Jump directly to exhausted if usage crosses all thresholds."""
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(tokens=1000)  # 100% — crosses all
        result = bm.check_and_transition()
        assert result == CodexState.OPENROUTER_FALLBACK


class TestResetOnNewPeriod:
    """Budget resets on new billing period."""

    def test_daily_reset(self, tmp_path):
        config = _make_config(period="daily", tokens_limit=1000)
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage(tokens=500)
        assert bm.tokens_used == 500

        # Simulate period rollover by manipulating the state
        with bm._lock:
            # Set period start to yesterday
            yesterday = bm._state.period_start
            from datetime import datetime, timezone, timedelta
            dt = datetime.fromisoformat(yesterday) - timedelta(days=1)
            bm._state.period_start = dt.isoformat()
            bm._persist()

        # Record new usage — should trigger reset
        bm.record_usage(tokens=100)
        assert bm.tokens_used == 100  # reset happened

    def test_weekly_reset(self, tmp_path):
        config = _make_config(period="weekly", tokens_limit=1000)
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage(tokens=500)

        with bm._lock:
            from datetime import datetime, timezone, timedelta
            dt = datetime.fromisoformat(bm._state.period_start) - timedelta(weeks=1)
            bm._state.period_start = dt.isoformat()
            bm._persist()

        bm.record_usage(tokens=200)
        assert bm.tokens_used == 200

    def test_monthly_reset(self, tmp_path):
        config = _make_config(period="monthly", tokens_limit=1000)
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage(tokens=500)

        with bm._lock:
            from datetime import datetime, timezone, timedelta
            dt = datetime.fromisoformat(bm._state.period_start) - timedelta(days=32)
            bm._state.period_start = dt.isoformat()
            bm._persist()

        bm.record_usage(tokens=300)
        assert bm.tokens_used == 300


class TestPersistRecover:
    """Persist to disk and recover from crash."""

    def test_persist_creates_file(self, tmp_path):
        config = _make_config()
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage(tokens=42, cost_usd=1.23)

        state_path = tmp_path / "budget_state.json"
        assert state_path.exists()

        with open(state_path) as f:
            data = json.load(f)
        assert data["tokens_used"] == 42
        assert data["cost_usd"] == pytest.approx(1.23)

    def test_recover_from_disk(self, tmp_path):
        config = _make_config()

        # First instance — write state
        bm1 = BudgetManager(config=config, state_dir=tmp_path)
        bm1.record_usage(tokens=999, cost_usd=7.77)

        # Second instance — simulates restart
        bm2 = BudgetManager(config=config, state_dir=tmp_path)
        assert bm2.tokens_used == 999
        assert bm2.cost_usd == pytest.approx(7.77)

    def test_corrupt_file_reinitializes(self, tmp_path):
        config = _make_config()
        state_path = tmp_path / "budget_state.json"
        state_path.write_text("not valid json{{{")
        _restrict_permissions = None  # not needed

        bm = BudgetManager(config=config, state_dir=tmp_path)
        assert bm.tokens_used == 0  # fresh state

    def test_state_file_permissions(self, tmp_path):
        config = _make_config()
        bm = BudgetManager(config=config, state_dir=tmp_path)

        state_path = tmp_path / "budget_state.json"
        assert state_path.exists()
        mode = state_path.stat().st_mode & 0o777
        assert mode == 0o600


class TestThreadSafety:
    """Concurrent access to BudgetManager is safe."""

    def test_concurrent_record_usage(self, tmp_path):
        config = _make_config(tokens_limit=10_000_000)
        bm = BudgetManager(config=config, state_dir=tmp_path)

        errors = []

        def worker():
            try:
                for _ in range(100):
                    bm.record_usage(tokens=1, cost_usd=0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert bm.tokens_used == 1000  # 10 threads × 100 × 1 token
        assert bm.cost_usd == pytest.approx(10.0)  # 10 × 100 × 0.01

    def test_concurrent_check_and_transition(self, tmp_path):
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        # Pre-load to just below threshold
        bm.record_usage(tokens=850)

        results = []
        lock = threading.Lock()

        def worker():
            r = bm.check_and_transition()
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Only one thread should have triggered the transition
        transitions = [r for r in results if r is not None]
        assert len(transitions) == 1
        assert transitions[0] == CodexState.OPENAI_CONSERVATION


class TestDisabledBudget:
    """Disabled budget does nothing."""

    def test_disabled_no_transition(self, tmp_path):
        config = _make_config(enabled=False, tokens_limit=100)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(tokens=1000)  # way over limit
        result = bm.check_and_transition()

        assert result is None
        assert store.get_auto_state() is None

    def test_disabled_still_tracks(self, tmp_path):
        """Even when disabled, usage tracking works (just no transitions)."""
        config = _make_config(enabled=False)
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage(tokens=500, cost_usd=10.0)
        assert bm.tokens_used == 500
        assert bm.cost_usd == pytest.approx(10.0)


class TestManualOverride:
    """Manual state always overrides budget auto-transitions."""

    def test_manual_state_blocks_transition(self, tmp_path):
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(tokens=900)  # 90% — would be critical
        result = bm.check_and_transition()

        assert result is None
        # Auto state should not be changed
        assert store.get_auto_state() is None

    def test_no_manual_allows_transition(self, tmp_path):
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(tokens=850)
        result = bm.check_and_transition()

        assert result == CodexState.OPENAI_CONSERVATION


class TestNotificationIntegration:
    """Budget transitions log to NotificationManager."""

    def test_notifier_called_on_transition(self, tmp_path):
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        notifier = MagicMock()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store, notifier=notifier)

        bm.record_usage(tokens=850)
        bm.check_and_transition()

        notifier.notify_state_change.assert_called_once()
        call_kwargs = notifier.notify_state_change.call_args
        assert call_kwargs[1]["from_state"] == "openai_primary"
        assert call_kwargs[1]["to_state"] == "openai_conservation"
        assert "budget_threshold_warning" in call_kwargs[1]["reason"]

    def test_notifier_escalation(self, tmp_path):
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        notifier = MagicMock()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store, notifier=notifier)

        bm.record_usage(tokens=850)  # 85% → warning
        bm.check_and_transition()  # → warning

        bm.record_usage(tokens=50)  # 900/1000 = 90% → critical
        bm.check_and_transition()  # → critical

        assert notifier.notify_state_change.call_count == 2


class TestDiagnostics:
    """Budget status reporting."""

    def test_get_status(self, tmp_path):
        config = _make_config(tokens_limit=1000, cost_limit=100.0)
        bm = BudgetManager(config=config, state_dir=tmp_path)

        bm.record_usage(tokens=500, cost_usd=25.0)
        status = bm.get_status()

        assert status["enabled"] is True
        assert status["tokens_used"] == 500
        assert status["tokens_pct"] == 50.0
        assert status["cost_usd"] == pytest.approx(25.0)
        assert status["cost_pct"] == 25.0
        assert status["current_threshold"] is None

    def test_get_status_after_transition(self, tmp_path):
        config = _make_config(tokens_limit=1000)
        store = MockStateStore()
        bm = BudgetManager(config=config, state_dir=tmp_path, state_store=store)

        bm.record_usage(tokens=850)
        bm.check_and_transition()

        status = bm.get_status()
        assert status["current_threshold"] == "warning"
        assert status["tokens_pct"] == 85.0


class TestPeriodStart:
    """Period start computation."""

    def test_daily(self):
        from datetime import datetime, timezone
        now = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
        ps = _period_start("daily", now)
        assert ps == datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_weekly_monday(self):
        from datetime import datetime, timezone
        # 2026-03-15 is a Sunday
        now = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
        ps = _period_start("weekly", now)
        # Monday 2026-03-09
        assert ps == datetime(2026, 3, 9, 0, 0, 0, tzinfo=timezone.utc)

    def test_monthly(self):
        from datetime import datetime, timezone
        now = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
        ps = _period_start("monthly", now)
        assert ps == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_invalid_period(self):
        with pytest.raises(ValueError, match="Unknown period"):
            _period_start("yearly")


class TestSingleton:
    """Budget manager singleton."""

    def test_get_and_reset(self, tmp_path):
        reset_budget_manager()
        config = _make_config()
        bm1 = get_budget_manager(config=config, state_dir=tmp_path)
        bm2 = get_budget_manager()
        assert bm1 is bm2

        reset_budget_manager()
        bm3 = get_budget_manager(config=config, state_dir=tmp_path)
        assert bm3 is not bm1


class TestPolicyIntegration:
    """Integration with policy.route_task()."""

    def test_budget_check_called_on_route(self, tmp_path, monkeypatch):
        """route_task calls budget_manager.check_and_transition() before building chain."""
        from router.policy import route_task, reset_breaker, reset_notifier
        from router.state_store import StateStore, reset_state_store
        from router.models import TaskMeta, TaskClass, TaskRisk, TaskModality
        from tests.conftest import MockExecutor
        from unittest.mock import patch, MagicMock
        import router.config_loader as _cfg_mod

        # Set up isolated dirs
        state_dir = tmp_path / "state"
        runtime_dir = tmp_path / "runtime"
        state_dir.mkdir()
        runtime_dir.mkdir()

        monkeypatch.setattr("router.state_store.MANUAL_STATE_PATH", state_dir / "manual.json")
        monkeypatch.setattr("router.state_store.AUTO_STATE_PATH", state_dir / "auto.json")
        monkeypatch.setattr("router.state_store.STATE_HISTORY_PATH", state_dir / "history.json")
        monkeypatch.setattr("router.state_store.WAL_PATH", state_dir / "wal.jsonl")
        monkeypatch.setattr("router.notifications.NotificationManager.ALERTS_PATH", runtime_dir / "alerts.jsonl")
        monkeypatch.setattr("router.logger.RUNTIME_DIR", runtime_dir)
        monkeypatch.setattr("router.logger.DEFAULT_LOG_PATH", runtime_dir / "routing.jsonl")

        reset_state_store()
        reset_breaker()
        reset_notifier()

        # Create a real StateStore with isolated paths, clear default manual state
        from router.state_store import StateStore
        store = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        store.set_manual_state(None)  # Clear default so budget can transition

        # Set up budget manager in "warning" state
        budget_config = _make_config(tokens_limit=1000)
        from router.budget_manager import BudgetManager, reset_budget_manager
        reset_budget_manager()
        bm = BudgetManager(
            config=budget_config,
            state_dir=state_dir,
            state_store=store,
        )
        bm.record_usage(tokens=850)  # 85% — triggers warning on next check

        # Patch get_budget_manager to return our instance
        import router.policy as _policy_mod
        monkeypatch.setattr(_policy_mod, "get_budget_manager", lambda: bm)

        # Mock build_chain to avoid config loading
        mock_chain_entry = MagicMock()
        mock_chain_entry.tool = "codex_cli"
        mock_chain_entry.backend = "openai_native"
        mock_chain_entry.model_profile = "auto"

        happy = MockExecutor(delay_ms=50, cost_usd=0.001)

        task = TaskMeta(
            task_id="budget-test-001",
            agent="coder",
            task_class=TaskClass.IMPLEMENTATION,
            risk=TaskRisk.MEDIUM,
            modality=TaskModality.TEXT,
            requires_repo_write=False,
            requires_multimodal=False,
            has_screenshots=False,
            swarm=False,
            repo_path="/tmp/test",
            cwd="/tmp/test",
            summary="Test budget integration",
        )

        with patch("router.policy.build_chain", return_value=[mock_chain_entry]):
            with patch("router.policy.run_codex", side_effect=happy):
                with patch("router.policy.run_claude", side_effect=happy):
                    with patch("router.policy.run_openrouter", side_effect=happy):
                        # Verify our patched budget manager is used
                        assert _policy_mod.get_budget_manager() is bm
                        assert bm.enabled
                        assert bm.current_threshold is None  # not yet checked
                        
                        decision, result = route_task(task)

        # Budget should have auto-transitioned to conservation
        final_state = store.get_auto_state()
        assert final_state == CodexState.OPENAI_CONSERVATION, (
            f"Expected OPENAI_CONSERVATION but got {final_state}, "
            f"budget threshold: {bm.current_threshold}"
        )
