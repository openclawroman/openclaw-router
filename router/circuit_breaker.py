"""Circuit breaker for provider health tracking."""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class ProviderState:
    """Circuit breaker state for a single provider."""
    state: str = "closed"  # closed | open | half_open
    failure_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    opened_at: float = 0.0


class CircuitBreaker:
    """Per-provider circuit breaker for tracking health."""

    def __init__(
        self,
        threshold: int = 5,
        window_s: float = 60.0,
        cooldown_s: float = 120.0,
        state_file: Optional[Path] = None,
    ):
        self.threshold = threshold
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self.state_file = state_file
        self._providers: Dict[str, ProviderState] = {}

    def _provider_key(self, tool: str, backend: str) -> str:
        return f"{tool}:{backend}"

    def is_available(self, tool: str, backend: str) -> bool:
        """Check if a provider is available (not in open state)."""
        key = self._provider_key(tool, backend)
        ps = self._providers.get(key, ProviderState())

        if ps.state == "closed":
            return True

        if ps.state == "open":
            # Check if cool-down has passed
            if time.monotonic() - ps.opened_at >= self.cooldown_s:
                ps.state = "half_open"
                return True
            return False

        # half_open: allow one attempt
        return True

    def record_success(self, tool: str, backend: str) -> None:
        """Record a successful execution."""
        key = self._provider_key(tool, backend)
        ps = self._providers.get(key, ProviderState())
        ps.failure_count = 0
        ps.last_success_time = time.monotonic()
        ps.state = "closed"
        self._providers[key] = ps

    def record_failure(self, tool: str, backend: str, error_type: str = "") -> None:
        """Record a failed execution. May open the circuit."""
        key = self._provider_key(tool, backend)
        ps = self._providers.get(key, ProviderState())
        now = time.monotonic()

        # Reset if window has passed
        if now - ps.last_failure_time > self.window_s:
            ps.failure_count = 0

        ps.failure_count += 1
        ps.last_failure_time = now

        if ps.state == "half_open":
            # Failed during half-open: go back to open
            ps.state = "open"
            ps.opened_at = now
        elif ps.failure_count >= self.threshold:
            # Threshold reached: open the circuit
            ps.state = "open"
            ps.opened_at = now

        self._providers[key] = ps

    def get_state(self, tool: str, backend: str) -> str:
        """Get current circuit state for a provider."""
        key = self._provider_key(tool, backend)
        ps = self._providers.get(key, ProviderState())
        return ps.state

    def get_stats(self) -> Dict[str, Any]:
        """Get all provider stats."""
        return {k: asdict(v) for k, v in self._providers.items()}
