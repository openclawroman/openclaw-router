"""Metrics aggregation from routing logs."""

import json
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta


@dataclass
class StateMetrics:
    """Metrics for a single routing state."""
    state: str
    task_count: int = 0
    success_count: int = 0
    fallback_count: int = 0
    timeout_count: int = 0
    avg_latency_ms: float = 0.0
    total_latency_ms: int = 0


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    model: str
    call_count: int = 0
    success_count: int = 0
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0


@dataclass
class ProviderMetrics:
    """Metrics for a provider (tool:backend)."""
    provider: str
    call_count: int = 0
    skip_count: int = 0
    success_count: int = 0
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0


@dataclass
class MetricsReport:
    """Aggregated metrics report."""
    total_tasks: int = 0
    total_success: int = 0
    total_failure: int = 0
    success_rate: float = 0.0
    by_state: Dict[str, StateMetrics] = field(default_factory=dict)
    by_model: Dict[str, ModelMetrics] = field(default_factory=dict)
    by_provider: Dict[str, ProviderMetrics] = field(default_factory=dict)
    circuit_breaker_skips: int = 0
    chain_timeouts: int = 0
    period_hours: int = 24

    def to_dict(self) -> dict:
        """Convert to dict for JSON output."""
        return {
            "total_tasks": self.total_tasks,
            "total_success": self.total_success,
            "total_failure": self.total_failure,
            "success_rate": round(self.success_rate, 3),
            "by_state": {k: asdict(v) for k, v in self.by_state.items()},
            "by_model": {k: asdict(v) for k, v in self.by_model.items()},
            "by_provider": {k: asdict(v) for k, v in self.by_provider.items()},
            "circuit_breaker_skips": self.circuit_breaker_skips,
            "chain_timeouts": self.chain_timeouts,
            "period_hours": self.period_hours,
        }


class MetricsCollector:
    """Read routing logs and compute aggregated metrics."""

    def __init__(self, log_path: Optional[Path] = None):
        from .logger import DEFAULT_LOG_PATH
        self.log_path = log_path or DEFAULT_LOG_PATH

    def collect(self, period_hours: int = 24) -> MetricsReport:
        """Collect metrics from log entries within the given time period.

        Handles both legacy log entries and new routing_trace entries.
        """
        report = MetricsReport(period_hours=period_hours)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=period_hours)

        if not self.log_path.exists():
            return report

        state_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0, "fallback": 0, "timeout": 0, "latency": 0})
        model_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "success": 0, "latency": 0, "cost": 0})
        provider_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "skips": 0, "success": 0, "latency": 0, "cost": 0})

        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Parse timestamp
                ts_str = entry.get("ts", entry.get("timestamp", ""))
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass

                # Handle routing_trace entries
                if entry.get("type") == "routing_trace":
                    report.total_tasks += 1
                    state = entry.get("state", "")
                    sc = state_counts[state]
                    sc["total"] += 1

                    if entry.get("final_success"):
                        report.total_success += 1
                        sc["success"] += 1
                    else:
                        report.total_failure += 1

                    if entry.get("chain_timed_out"):
                        report.chain_timeouts += 1
                        sc["timeout"] += 1

                    if entry.get("fallback_count", 0) > 0:
                        sc["fallback"] += 1

                    sc["latency"] += entry.get("total_latency_ms", 0)

                    # Per-executor attempts
                    for attempt in entry.get("attempts", []):
                        tool = attempt.get("tool", "")
                        backend = attempt.get("backend", "")
                        profile = attempt.get("model_profile", "")
                        provider_key = f"{tool}:{backend}"

                        if attempt.get("skipped"):
                            report.circuit_breaker_skips += 1
                            provider_counts[provider_key]["skips"] += 1
                            continue

                        # Model metrics
                        model_counts[profile]["calls"] += 1
                        model_counts[profile]["latency"] += attempt.get("latency_ms", 0)
                        if attempt.get("success"):
                            model_counts[profile]["success"] += 1
                        cost = attempt.get("cost_estimate_usd") or 0.0
                        model_counts[profile]["cost"] += int(cost * 1000)  # store as millis

                        # Provider metrics
                        provider_counts[provider_key]["calls"] += 1
                        provider_counts[provider_key]["latency"] += attempt.get("latency_ms", 0)
                        if attempt.get("success"):
                            provider_counts[provider_key]["success"] += 1
                        provider_counts[provider_key]["cost"] += int(cost * 1000)

                # Handle legacy entries
                elif entry.get("task_id"):
                    report.total_tasks += 1
                    state = entry.get("state", "")
                    sc = state_counts[state]
                    sc["total"] += 1
                    sc["latency"] += entry.get("latency_ms", 0) or 0

                    result = entry.get("result", {})
                    if result.get("success"):
                        report.total_success += 1
                        sc["success"] += 1
                    else:
                        report.total_failure += 1

                    if entry.get("attempted_fallback"):
                        sc["fallback"] += 1

                    # Model from result
                    profile = result.get("model_profile", "")
                    if profile:
                        model_counts[profile]["calls"] += 1
                        model_counts[profile]["latency"] += result.get("latency_ms", 0) or 0
                        if result.get("success"):
                            model_counts[profile]["success"] += 1
                        cost = result.get("cost_estimate_usd") or 0.0
                        model_counts[profile]["cost"] += int(cost * 1000)

        # Compute averages and build report
        if report.total_tasks > 0:
            report.success_rate = report.total_success / report.total_tasks

        for state, counts in state_counts.items():
            avg = counts["latency"] / max(counts["total"], 1)
            report.by_state[state] = StateMetrics(
                state=state, task_count=counts["total"],
                success_count=counts["success"], fallback_count=counts["fallback"],
                timeout_count=counts["timeout"], avg_latency_ms=round(avg, 1),
                total_latency_ms=counts["latency"],
            )

        for model, counts in model_counts.items():
            avg = counts["latency"] / max(counts["calls"], 1)
            report.by_model[model] = ModelMetrics(
                model=model, call_count=counts["calls"],
                success_count=counts["success"],
                total_latency_ms=counts["latency"],
                total_cost_usd=counts["cost"] / 1000.0,
            )

        for provider, counts in provider_counts.items():
            report.by_provider[provider] = ProviderMetrics(
                provider=provider, call_count=counts["calls"],
                skip_count=counts["skips"], success_count=counts["success"],
                total_latency_ms=counts["latency"],
                total_cost_usd=counts["cost"] / 1000.0,
            )

        return report
