"""Per-executor attempt trail logger."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from .secrets import redact_dict

from .audit import AuditChain, init_chain
from .sanitize import sanitize_content


@dataclass
class ExecutorAttempt:
    """One attempt to run an executor."""
    tool: str
    backend: str
    model_profile: str
    success: bool
    latency_ms: int
    normalized_error: Optional[str] = None
    cost_estimate_usd: Optional[float] = None
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class RoutingTrace:
    """Complete trace of one routing decision."""
    trace_id: str = ""
    task_id: str = ""
    state: str = ""
    chain: List[Dict[str, str]] = field(default_factory=list)
    attempts: List[ExecutorAttempt] = field(default_factory=list)
    providers_skipped: List[str] = field(default_factory=list)
    chain_timed_out: bool = False
    fallback_count: int = 0
    total_latency_ms: int = 0
    final_tool: str = ""
    final_success: bool = False
    final_error: Optional[str] = None
    chain_invariant_violated: bool = False
    chain_invariant_reason: Optional[str] = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "state": self.state,
            "chain": self.chain,
            "attempts": [asdict(a) for a in self.attempts],
            "providers_skipped": self.providers_skipped,
            "chain_timed_out": self.chain_timed_out,
            "fallback_count": self.fallback_count,
            "total_latency_ms": self.total_latency_ms,
            "final_tool": self.final_tool,
            "final_success": self.final_success,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }
        if self.final_error:
            d["final_error"] = self.final_error
        return d


class AttemptLogger:
    """Collects executor attempts and writes structured trace."""

    def __init__(self, log_path: Optional[Path] = None):
        from .logger import DEFAULT_LOG_PATH
        self.log_path = log_path or DEFAULT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._chain = AuditChain(last_hash=init_chain(self.log_path))

    def log_trace(self, trace: RoutingTrace) -> None:
        """Write a routing trace to the log file (secrets redacted, content sanitized)."""
        entry = redact_dict(sanitize_content({"type": "routing_trace", **trace.to_dict()}))
        self._chain.chain_entry(entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def create_trace(
        self,
        trace_id: str,
        task_id: str,
        state: str,
        chain: List[Dict[str, str]],
    ) -> RoutingTrace:
        """Create a new routing trace."""
        return RoutingTrace(
            trace_id=trace_id,
            task_id=task_id,
            state=state,
            chain=chain,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
