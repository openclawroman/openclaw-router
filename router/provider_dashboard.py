"""Provider health dashboard — real-time provider status display."""

import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict

from .circuit_breaker import CircuitBreaker
from .metrics import MetricsCollector


@dataclass
class ProviderStatus:
    """Status of a single provider."""
    name: str
    circuit_state: str  # closed, open, half_open
    failure_count: int
    healthy: bool
    last_failure_age_s: Optional[float] = None
    last_success_age_s: Optional[float] = None
    
    @property
    def status_icon(self) -> str:
        if self.circuit_state == "closed":
            return "✅"
        elif self.circuit_state == "half_open":
            return "⚠️"
        else:
            return "❌"


@dataclass
class DashboardReport:
    """Full provider health dashboard report."""
    providers: List[ProviderStatus]
    total_providers: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    overall_status: str  # "healthy", "degraded", "critical"
    
    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "total_providers": self.total_providers,
            "healthy": self.healthy_count,
            "degraded": self.degraded_count,
            "unhealthy": self.unhealthy_count,
            "providers": [asdict(p) for p in self.providers],
        }
    
    def to_table(self) -> str:
        """Format as a human-readable table."""
        lines = [f"\n{'='*60}"]
        lines.append(f"  Provider Health Dashboard — {self.overall_status.upper()}")
        lines.append(f"{'='*60}")
        lines.append(f"  {'Provider':<30} {'State':<12} {'Fails':<6} {'Status'}")
        lines.append(f"  {'-'*58}")
        
        for p in self.providers:
            icon = p.status_icon
            lines.append(f"  {p.name:<30} {p.circuit_state:<12} {p.failure_count:<6} {icon}")
        
        lines.append(f"{'='*60}")
        lines.append(f"  Total: {self.total_providers} | ✅ {self.healthy_count} | ⚠️ {self.degraded_count} | ❌ {self.unhealthy_count}")
        lines.append(f"{'='*60}\n")
        
        return "\n".join(lines)


class ProviderDashboard:
    """Collects and displays provider health status."""
    
    def __init__(self, breaker: Optional[CircuitBreaker] = None):
        from .policy import get_breaker
        self.breaker = breaker or get_breaker()
    
    def get_report(self) -> DashboardReport:
        """Generate a provider health dashboard report."""
        summary = self.breaker.get_health_summary()
        
        providers = []
        now = time.monotonic()
        
        for name, state in summary["providers"].items():
            last_fail = state["last_failure_time"]
            last_ok = state["last_success_time"]
            
            providers.append(ProviderStatus(
                name=name,
                circuit_state=state["state"],
                failure_count=state["failure_count"],
                healthy=state["state"] == "closed",
                last_failure_age_s=round(now - last_fail, 1) if last_fail > 0 else None,
                last_success_age_s=round(now - last_ok, 1) if last_ok > 0 else None,
            ))
        
        # Sort: healthy first, then by name
        providers.sort(key=lambda p: (0 if p.healthy else 1, p.name))
        
        healthy = sum(1 for p in providers if p.healthy)
        degraded = summary["degraded"]
        unhealthy = summary["unhealthy"]
        
        if unhealthy > 0:
            overall = "critical"
        elif degraded > 0:
            overall = "degraded"
        else:
            overall = "healthy"
        
        return DashboardReport(
            providers=providers,
            total_providers=len(providers),
            healthy_count=healthy,
            degraded_count=degraded,
            unhealthy_count=unhealthy,
            overall_status=overall,
        )
    
    def get_combined_report(self) -> dict:
        """Get dashboard + metrics combined report."""
        dashboard = self.get_report()
        metrics = MetricsCollector().collect(period_hours=24)
        
        return {
            "dashboard": dashboard.to_dict(),
            "metrics_24h": metrics.to_dict(),
        }
