"""Append-only routing log."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import TaskMeta, RouteDecision, ExecutorResult


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
        result: Optional[ExecutorResult] = None,
        latency_ms: Optional[int] = None,
    ):
        """Append a routing decision to the log."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task_id": decision.task_id or task.task_id or "",
            "agent": task.agent,
            "task_class": task.task_class.value if task.task_class else "",
            "summary": task.summary,
            "state": decision.state,
            "chain": [{"tool": c.tool, "backend": c.backend, "model_profile": c.model_profile} for c in decision.chain],
            "reason": decision.reason,
            "attempted_fallback": decision.attempted_fallback,
        }
        if decision.fallback_from:
            entry["fallback_from"] = decision.fallback_from

        if result:
            entry["result"] = {
                "tool": result.tool,
                "backend": result.backend,
                "model_profile": result.model_profile,
                "success": result.success,
                "latency_ms": result.latency_ms,
            }
            if result.normalized_error:
                entry["result"]["normalized_error"] = result.normalized_error
            if result.exit_code is not None:
                entry["result"]["exit_code"] = result.exit_code
            if result.request_id:
                entry["result"]["request_id"] = result.request_id
            if result.cost_estimate_usd is not None:
                entry["result"]["cost_estimate_usd"] = result.cost_estimate_usd
            if result.artifacts:
                entry["result"]["artifacts"] = result.artifacts
            if result.final_summary:
                entry["result"]["final_summary"] = result.final_summary

        if latency_ms is not None:
            entry["latency_ms"] = latency_ms

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_log_path(self) -> Path:
        """Return the log file path."""
        return self.log_path
