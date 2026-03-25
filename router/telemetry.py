"""Route quality reporting and telemetry analysis."""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ExecutorStats:
    tool: str
    backend: str
    total_calls: int
    success_count: int
    failure_count: int
    avg_latency_ms: float
    total_cost_usd: float
    success_rate: float


@dataclass
class RouteQualityReport:
    total_routes: int
    success_count: int
    failure_count: int
    overall_success_rate: float
    by_executor: List[ExecutorStats]
    by_task_class: Dict[str, Dict]
    by_state: Dict[str, Dict]
    most_common_errors: List[tuple]
    avg_latency_ms: float
    total_cost_usd: float


class RouteQualityReporter:
    """Analyze routing.jsonl to produce quality reports."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Path("runtime/routing.jsonl")

    def load_entries(self) -> List[dict]:
        """Load all log entries from routing.jsonl."""
        if not self.log_path.exists():
            return []
        entries = []
        for line in self.log_path.read_text().strip().split('\n'):
            if line.strip():
                entries.append(json.loads(line))
        return entries

    def generate_report(self) -> RouteQualityReport:
        """Generate a comprehensive route quality report."""
        entries = self.load_entries()

        if not entries:
            return RouteQualityReport(
                total_routes=0,
                success_count=0,
                failure_count=0,
                overall_success_rate=0.0,
                by_executor=[],
                by_task_class={},
                by_state={},
                most_common_errors=[],
                avg_latency_ms=0.0,
                total_cost_usd=0.0,
            )

        # Track totals
        success_count = 0
        failure_count = 0
        total_latency = 0.0
        total_cost = 0.0

        # Per-executor aggregation: key = (tool, backend)
        executor_data = defaultdict(lambda: {
            "total": 0, "success": 0, "failure": 0,
            "total_latency": 0.0, "total_cost": 0.0,
        })

        # Per-task-class aggregation
        task_class_data = defaultdict(lambda: {
            "total": 0, "success": 0, "failure": 0, "total_cost_usd": 0.0,
        })

        # Per-state aggregation
        state_data = defaultdict(lambda: {
            "total": 0, "success": 0, "failure": 0,
        })

        # Error tracking
        error_counts = defaultdict(int)

        for entry in entries:
            result = entry.get("result")
            task_class = entry.get("task_class", "unknown")
            state = entry.get("state", "unknown")

            if result:
                res_success = result.get("success", False)
                res_latency = result.get("latency_ms", 0)
                res_cost = result.get("cost_estimate_usd") or 0.0
                res_tool = result.get("tool", "unknown")
                res_backend = result.get("backend", "unknown")
                res_error = result.get("normalized_error")

                # Overall counts
                if res_success:
                    success_count += 1
                else:
                    failure_count += 1

                total_latency += res_latency
                total_cost += res_cost

                # Per-executor
                key = (res_tool, res_backend)
                executor_data[key]["total"] += 1
                if res_success:
                    executor_data[key]["success"] += 1
                else:
                    executor_data[key]["failure"] += 1
                executor_data[key]["total_latency"] += res_latency
                executor_data[key]["total_cost"] += res_cost

                # Per-task-class
                task_class_data[task_class]["total"] += 1
                if res_success:
                    task_class_data[task_class]["success"] += 1
                else:
                    task_class_data[task_class]["failure"] += 1
                task_class_data[task_class]["total_cost_usd"] += res_cost

                # Per-state
                state_data[state]["total"] += 1
                if res_success:
                    state_data[state]["success"] += 1
                else:
                    state_data[state]["failure"] += 1

                # Errors
                if not res_success and res_error:
                    error_counts[res_error] += 1
            else:
                # No result block — count as failure
                failure_count += 1

        total_routes = len(entries)
        overall_rate = success_count / total_routes if total_routes > 0 else 0.0

        # Build executor stats
        by_executor = []
        for (tool, backend), data in executor_data.items():
            rate = data["success"] / data["total"] if data["total"] > 0 else 0.0
            avg_lat = data["total_latency"] / data["total"] if data["total"] > 0 else 0.0
            by_executor.append(ExecutorStats(
                tool=tool,
                backend=backend,
                total_calls=data["total"],
                success_count=data["success"],
                failure_count=data["failure"],
                avg_latency_ms=avg_lat,
                total_cost_usd=data["total_cost"],
                success_rate=rate,
            ))

        # Build task class dict
        by_task_class = {}
        for tc, data in task_class_data.items():
            by_task_class[tc] = dict(data)

        # Build state dict
        by_state = {}
        for st, data in state_data.items():
            by_state[st] = dict(data)

        # Sort errors by frequency
        most_common_errors = sorted(error_counts.items(), key=lambda x: -x[1])

        avg_latency = total_latency / total_routes if total_routes > 0 else 0.0

        return RouteQualityReport(
            total_routes=total_routes,
            success_count=success_count,
            failure_count=failure_count,
            overall_success_rate=overall_rate,
            by_executor=by_executor,
            by_task_class=by_task_class,
            by_state=by_state,
            most_common_errors=most_common_errors,
            avg_latency_ms=avg_latency,
            total_cost_usd=total_cost,
        )
