"""Append-only routing log."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import TaskMeta, RouteDecision, Executor


RUNTIME_DIR = Path(__file__).parent.parent / "runtime"
DEFAULT_LOG_PATH = RUNTIME_DIR / "routing.jsonl"


class RoutingLogger:
    """Append-only JSONL routing log written to runtime/routing.jsonl."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or DEFAULT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        task: TaskMeta,
        decision: RouteDecision,
        latency_ms: Optional[int] = None,
    ):
        """Append a routing decision to the log."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task_id": decision.task_id or task.task_id or "",
            "agent": task.agent,
            "task_class": task.task_class.value if task.task_class else "",
            "task_brief": task.task_brief,
            "executor": decision.executor.value,
            "chain": decision.chain,
            "reason": decision.reason,
            "status": decision.status,
            "attempted_fallback": decision.attempted_fallback,
        }
        if latency_ms is not None:
            entry["latency_ms"] = latency_ms
        if decision.error_type:
            entry["error_type"] = decision.error_type
        if decision.backend:
            entry["backend"] = decision.backend.value
        if decision.model_profile:
            entry["model_profile"] = decision.model_profile.value
        if decision.normalized_error:
            entry["normalized_error"] = decision.normalized_error
        if decision.exit_code is not None:
            entry["exit_code"] = decision.exit_code
        if decision.request_id:
            entry["request_id"] = decision.request_id
        if decision.cost_estimate_usd is not None:
            entry["cost_estimate_usd"] = decision.cost_estimate_usd
        if decision.artifacts:
            entry["artifacts"] = decision.artifacts
        if decision.stdout_ref:
            entry["stdout_ref"] = decision.stdout_ref
        if decision.stderr_ref:
            entry["stderr_ref"] = decision.stderr_ref
        if decision.final_summary:
            entry["final_summary"] = decision.final_summary

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_log_path(self) -> Path:
        """Return the log file path."""
        return self.log_path
