"""Budget manager — track cumulative token/cost usage and auto-transition states."""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from .models import CodexState

logger = logging.getLogger(__name__)

RESTRICTIVE_PERMISSIONS = 0o600

# Budget state transition ladder: lower → higher conservation
_STATE_LADDER = [
    CodexState.OPENAI_PRIMARY,
    CodexState.OPENAI_CONSERVATION,
    CodexState.CLAUDE_BACKUP,
    CodexState.OPENROUTER_FALLBACK,
]

# Threshold → target state mapping
_THRESHOLD_TRANSITIONS = {
    "warning": CodexState.OPENAI_CONSERVATION,
    "critical": CodexState.CLAUDE_BACKUP,
    "exhausted": CodexState.OPENROUTER_FALLBACK,
}


def _restrict_permissions(path: Path) -> None:
    try:
        os.chmod(path, RESTRICTIVE_PERMISSIONS)
    except OSError:
        pass


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _period_start(period: str, now: Optional[datetime] = None) -> datetime:
    """Compute the start of the current billing period."""
    now = now or _now_utc()
    if period == "daily":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "weekly":
        # ISO week: Monday start
        days_since_monday = now.weekday()
        monday = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
        return monday
    elif period == "monthly":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Unknown period: {period}")


def _default_config() -> dict:
    return {
        "enabled": False,
        "period": "monthly",
        "limits": {
            "tokens": 6_500_000_000,
            "cost_usd": 250.0,
        },
        "thresholds": {
            "warning": 0.50,
            "critical": 0.75,
            "exhausted": 0.95,
        },
    }


@dataclass
class _BudgetState:
    """Serializable budget state."""
    period_start: str  # ISO format
    period: str
    tokens_used: int = 0
    cost_usd: float = 0.0
    current_threshold: Optional[str] = None  # None, "warning", "critical", "exhausted"

    def to_dict(self) -> dict:
        return {
            "period_start": self.period_start,
            "period": self.period,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "current_threshold": self.current_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "_BudgetState":
        return cls(
            period_start=data["period_start"],
            period=data["period"],
            tokens_used=data.get("tokens_used", 0),
            cost_usd=data.get("cost_usd", 0.0),
            current_threshold=data.get("current_threshold"),
        )


class BudgetManager:
    """Track cumulative token/cost usage and auto-transition router states.

    Thread-safe, opt-in (disabled by default). Stores state in JSON on disk
    for crash recovery.
    """

    BUDGET_STATE_FILE = "budget_state.json"

    def __init__(
        self,
        config: Optional[dict] = None,
        state_dir: Optional[Path] = None,
        state_store=None,
        notifier=None,
    ):
        self._lock = threading.RLock()
        self._config = config or _default_config()
        budget_cfg = self._config.get("budget", self._config)
        self._enabled = budget_cfg.get("enabled", False)
        self._period = budget_cfg.get("period", "monthly")
        self._limits = budget_cfg.get("limits", {"tokens": 6_500_000_000, "cost_usd": 250.0})
        self._thresholds = budget_cfg.get("thresholds", {"warning": 0.50, "critical": 0.75, "exhausted": 0.95})

        self._state_dir = Path(state_dir) if state_dir else Path("config")
        self._state_path = self._state_dir / self.BUDGET_STATE_FILE
        self._state_store = state_store
        self._notifier = notifier

        self._state: Optional[_BudgetState] = None
        self._load_or_init_state()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def tokens_used(self) -> int:
        with self._lock:
            return self._state.tokens_used if self._state else 0

    @property
    def cost_usd(self) -> float:
        with self._lock:
            return self._state.cost_usd if self._state else 0.0

    @property
    def current_threshold(self) -> Optional[str]:
        with self._lock:
            return self._state.current_threshold if self._state else None

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_or_init_state(self) -> None:
        """Load state from disk or initialize fresh."""
        with self._lock:
            if self._state_path.exists():
                try:
                    with open(self._state_path, "r") as f:
                        data = json.load(f)
                    self._state = _BudgetState.from_dict(data)
                    # Check if period has rolled over
                    if self._is_new_period():
                        self._reset_period()
                    return
                except (json.JSONDecodeError, KeyError, ValueError):
                    logger.warning("Corrupt budget state file, reinitializing")

            # Fresh state
            now = _now_utc()
            ps = _period_start(self._period, now)
            self._state = _BudgetState(
                period_start=ps.isoformat(),
                period=self._period,
            )
            self._persist()

    def _is_new_period(self) -> bool:
        """Check if current time is in a new billing period."""
        if not self._state:
            return True
        try:
            stored_start = datetime.fromisoformat(self._state.period_start)
            current_start = _period_start(self._state.period)
            return current_start > stored_start
        except (ValueError, TypeError):
            return True

    def _reset_period(self) -> None:
        """Reset counters for new billing period."""
        now = _now_utc()
        ps = _period_start(self._period, now)
        self._state = _BudgetState(
            period_start=ps.isoformat(),
            period=self._period,
        )
        logger.info("Budget period reset: new period starting %s", ps.isoformat())

    def _persist(self) -> None:
        """Write state to disk atomically."""
        if not self._state:
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)
        data = self._state.to_dict()
        fd, tmp = tempfile.mkstemp(dir=str(self._state_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(self._state_path))
            _restrict_permissions(self._state_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def record_usage(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Record token and/or cost usage."""
        with self._lock:
            if self._is_new_period():
                self._reset_period()
            self._state.tokens_used += tokens
            self._state.cost_usd += cost_usd
            self._persist()

    # ------------------------------------------------------------------
    # Threshold evaluation
    # ------------------------------------------------------------------

    def _evaluate_threshold(self) -> Optional[str]:
        """Return the highest breached threshold, or None."""
        if not self._state:
            return None

        token_limit = self._limits.get("tokens", float("inf"))
        cost_limit = self._limits.get("cost_usd", float("inf"))

        # Use the most-consumed resource ratio
        token_ratio = self._state.tokens_used / token_limit if token_limit > 0 else 0.0
        cost_ratio = self._state.cost_usd / cost_limit if cost_limit > 0 else 0.0
        ratio = max(token_ratio, cost_ratio)

        # Check thresholds in order (highest first)
        for name in ("exhausted", "critical", "warning"):
            threshold = self._thresholds.get(name)
            if threshold is not None and ratio >= threshold:
                return name
        return None

    def _threshold_lte(self, a: Optional[str], b: Optional[str]) -> bool:
        """Return True if threshold a is <= b in severity."""
        order = {None: -1, "warning": 0, "critical": 1, "exhausted": 2}
        return order.get(a, -1) <= order.get(b, -1)

    # ------------------------------------------------------------------
    # Auto-transition
    # ------------------------------------------------------------------

    def check_and_transition(
        self,
        state_store=None,
        notifier=None,
    ) -> Optional[CodexState]:
        """Check budget and auto-transition if threshold exceeded.

        Manual state always overrides budget auto-transitions:
        if manual_state is set, budget auto-transition is skipped.

        Returns the new state if transitioned, else None.
        """
        if not self._enabled:
            return None

        store = state_store or self._state_store
        notif = notifier or self._notifier

        with self._lock:
            if self._is_new_period():
                self._reset_period()
                self._persist()

            breached = self._evaluate_threshold()
            if breached is None:
                # Below all thresholds — if we were in a budget-driven state,
                # we could auto-recover, but the spec doesn't require it.
                return None

            # Don't escalate if already at same or higher threshold
            if self._threshold_lte(breached, self._state.current_threshold):
                return None

            target_state = _THRESHOLD_TRANSITIONS.get(breached)
            if target_state is None:
                return None

            # Check manual override
            if store is not None:
                manual = store.get_manual_state()
                if manual is not None:
                    logger.debug(
                        "Budget threshold %s breached but manual state %s overrides",
                        breached,
                        manual.value,
                    )
                    return None

            # Perform transition
            old_threshold = self._state.current_threshold
            self._state.current_threshold = breached
            self._persist()

            if store is not None:
                old_state = store.get_auto_state() or CodexState.OPENAI_PRIMARY
                store.set_auto_state(target_state)
                logger.info(
                    "Budget auto-transition: %s → %s (threshold: %s, tokens: %d/%d, cost: $%.2f/$%.2f)",
                    old_state.value,
                    target_state.value,
                    breached,
                    self._state.tokens_used,
                    self._limits.get("tokens", 0),
                    self._state.cost_usd,
                    self._limits.get("cost_usd", 0),
                )

                if notif is not None:
                    notif.notify_state_change(
                        from_state=old_state.value,
                        to_state=target_state.value,
                        reason=f"budget_threshold_{breached}: tokens={self._state.tokens_used}, cost=${self._state.cost_usd:.2f}",
                    )

            return target_state

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current budget status for observability."""
        with self._lock:
            if not self._state:
                return {"enabled": self._enabled}

            token_limit = self._limits.get("tokens", 0)
            cost_limit = self._limits.get("cost_usd", 0)
            token_pct = (self._state.tokens_used / token_limit * 100) if token_limit > 0 else 0
            cost_pct = (self._state.cost_usd / cost_limit * 100) if cost_limit > 0 else 0

            return {
                "enabled": self._enabled,
                "period": self._period,
                "period_start": self._state.period_start,
                "tokens_used": self._state.tokens_used,
                "tokens_limit": token_limit,
                "tokens_pct": round(token_pct, 1),
                "cost_usd": round(self._state.cost_usd, 4),
                "cost_limit": cost_limit,
                "cost_pct": round(cost_pct, 1),
                "current_threshold": self._state.current_threshold,
            }


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_budget_manager: Optional[BudgetManager] = None


def get_budget_manager(
    config: Optional[dict] = None,
    state_dir: Optional[Path] = None,
    state_store=None,
    notifier=None,
) -> BudgetManager:
    """Get or create the budget manager singleton."""
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = BudgetManager(
            config=config,
            state_dir=state_dir,
            state_store=state_store,
            notifier=notifier,
        )
    return _budget_manager


def reset_budget_manager() -> None:
    """Reset the budget manager singleton. For testing."""
    global _budget_manager
    _budget_manager = None
