"""Append-only routing log."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import TaskMeta, RouteDecision, Executor


DEFAULT_LOG_PATH = Path(__file__).parent.parent / "logs" / "routing.jsonl"


class RoutingLogger:
    """Append-only JSONL routing log."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or DEFAULT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, task: TaskMeta, decision: RouteDecision, latency_ms: Optional[int] = None):
        """Append a routing decision to the log."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent": task.agent,
            "task_type": task.task_type.value,
            "state": decision.reason.split(":")[0] if ":" in decision.reason else "unknown",
            "executor": decision.executor.value,
            "model": decision.model,
            "reason": decision.reason,
            "status": decision.status,
            "attempted_fallback": decision.attempted_fallback,
        }
        if latency_ms is not None:
            entry["latency_ms"] = latency_ms
        if decision.error_type:
            entry["error_type"] = decision.error_type

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_log_path(self) -> Path:
        """Return the log file path."""
        return self.log_path