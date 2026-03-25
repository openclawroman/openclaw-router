"""Health check and shutdown management for the router."""

import signal
import sys
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock


@dataclass
class TaskHandle:
    """Handle for tracking an in-flight task."""
    task_id: str
    started_at: float
    state: str
    executor: str


class ShutdownManager:
    """Manages graceful shutdown and in-flight task tracking."""

    def __init__(self):
        self._lock = Lock()
        self._in_flight: Dict[str, TaskHandle] = {}
        self._shutdown_requested = False
        self._shutdown_time: Optional[float] = None
        self._drain_timeout_s = 30.0

    def register_task(self, task_id: str, state: str, executor: str) -> None:
        """Register a task as in-flight."""
        with self._lock:
            self._in_flight[task_id] = TaskHandle(
                task_id=task_id,
                started_at=time.monotonic(),
                state=state,
                executor=executor,
            )

    def unregister_task(self, task_id: str) -> None:
        """Unregister a completed task."""
        with self._lock:
            self._in_flight.pop(task_id, None)

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        with self._lock:
            self._shutdown_requested = True
            self._shutdown_time = time.monotonic()

    def should_accept_new_tasks(self) -> bool:
        """Check if new tasks should be accepted."""
        return not self._shutdown_requested

    def is_drained(self) -> bool:
        """Check if all in-flight tasks have completed."""
        with self._lock:
            return len(self._in_flight) == 0

    def drain_timed_out(self) -> bool:
        """Check if drain timeout has been exceeded."""
        if not self._shutdown_requested or self._shutdown_time is None:
            return False
        return time.monotonic() - self._shutdown_time > self._drain_timeout_s

    def get_in_flight(self) -> List[Dict[str, Any]]:
        """Get list of currently in-flight tasks."""
        with self._lock:
            return [asdict(h) for h in self._in_flight.values()]

    def get_status(self) -> Dict[str, Any]:
        """Get shutdown manager status."""
        with self._lock:
            return {
                "shutdown_requested": self._shutdown_requested,
                "in_flight_count": len(self._in_flight),
                "in_flight_tasks": [h.task_id for h in self._in_flight.values()],
                "is_drained": len(self._in_flight) == 0,
                "drain_timed_out": self.drain_timed_out(),
            }


# Singleton instance
_shutdown_manager = ShutdownManager()


def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager."""
    return _shutdown_manager


def install_signal_handlers(manager: Optional[ShutdownManager] = None) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""
    mgr = manager or _shutdown_manager

    def _handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[{sig_name}] Graceful shutdown requested...", file=sys.stderr)
        mgr.request_shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def health_check() -> Dict[str, Any]:
    """Generate a health report for the router.

    Returns dict with:
    - status: "healthy" | "degraded" | "unhealthy"
    - state: current CodexState
    - providers: per-provider health from circuit breaker
    - reliability: timeout/fallback config
    - shutdown: shutdown manager status
    """
    from .policy import resolve_state
    from .config_loader import get_reliability_config

    try:
        state = resolve_state()
        state_ok = True
    except Exception as e:
        state = "error"
        state_ok = False

    # Provider health (from circuit breaker if available)
    providers = {}
    try:
        from .policy import get_breaker
        breaker = get_breaker()
        providers = breaker.get_stats()
    except Exception:
        providers = {"error": "circuit breaker not initialized"}

    # Reliability config
    reliability = get_reliability_config()

    # Shutdown status
    shutdown_mgr = get_shutdown_manager()
    shutdown_status = shutdown_mgr.get_status()

    # Determine overall status
    if not state_ok:
        overall = "unhealthy"
    elif shutdown_status["shutdown_requested"]:
        overall = "draining"
    elif any(p.get("state") == "open" for p in providers.values() if isinstance(p, dict)):
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "state": state.value if hasattr(state, "value") else str(state),
        "providers": providers,
        "reliability": reliability,
        "shutdown": shutdown_status,
        "timestamp": time.time(),
    }
