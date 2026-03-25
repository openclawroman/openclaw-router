"""State change notifications and alerting rules."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

RESTRICTIVE_PERMISSIONS = 0o600  # Owner read/write only


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    STATE_CHANGE = "state_change"
    FALLBACK_RATE = "fallback_rate"
    CONSERVATION_DURATION = "conservation_duration"
    PROVIDER_HEALTH = "provider_health"


@dataclass
class Alert:
    """An alert to be emitted."""
    alert_type: str
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


class NotificationManager:
    """Manages state change notifications and alerting rules.

    Alerts are written to a JSONL file and optionally printed to stderr.
    """

    ALERTS_PATH = Path("runtime/alerts.jsonl")

    # Thresholds
    FALLBACK_RATE_THRESHOLD = 0.30  # 30%
    CONSERVATION_MAX_HOURS = 24     # 24 hours

    def __init__(self, alerts_path: Optional[Path] = None):
        self.alerts_path = alerts_path or self.__class__.ALERTS_PATH
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)

    def notify_state_change(
        self,
        from_state: str,
        to_state: str,
        reason: str = "",
    ) -> Alert:
        """Emit a state change notification."""
        alert = Alert(
            alert_type=AlertType.STATE_CHANGE,
            severity=AlertSeverity.INFO,
            message=f"State transition: {from_state} \u2192 {to_state}",
            details={"from": from_state, "to": to_state, "reason": reason},
        )
        self._emit(alert)
        return alert

    def check_fallback_rate(
        self,
        total_tasks: int,
        fallback_tasks: int,
        window_hours: int = 1,
    ) -> Optional[Alert]:
        """Check if fallback rate exceeds threshold. Returns alert if threshold breached."""
        if total_tasks == 0:
            return None

        rate = fallback_tasks / total_tasks
        if rate > self.FALLBACK_RATE_THRESHOLD:
            alert = Alert(
                alert_type=AlertType.FALLBACK_RATE,
                severity=AlertSeverity.WARNING,
                message=f"Fallback rate {rate:.1%} exceeds {self.FALLBACK_RATE_THRESHOLD:.0%} threshold "
                        f"({fallback_tasks}/{total_tasks} in last {window_hours}h)",
                details={
                    "rate": round(rate, 3),
                    "threshold": self.FALLBACK_RATE_THRESHOLD,
                    "total_tasks": total_tasks,
                    "fallback_tasks": fallback_tasks,
                    "window_hours": window_hours,
                },
            )
            self._emit(alert)
            return alert
        return None

    def check_conservation_duration(
        self,
        state: str,
        state_entered_at: Optional[str] = None,
    ) -> Optional[Alert]:
        """Check if system has been in conservation state too long."""
        if state != "openai_conservation":
            return None

        if not state_entered_at:
            return None

        try:
            entered = datetime.fromisoformat(state_entered_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours = (now - entered).total_seconds() / 3600.0

            if hours > self.CONSERVATION_MAX_HOURS:
                alert = Alert(
                    alert_type=AlertType.CONSERVATION_DURATION,
                    severity=AlertSeverity.CRITICAL,
                    message=f"System in conservation state for {hours:.1f} hours "
                            f"(max: {self.CONSERVATION_MAX_HOURS}h)",
                    details={
                        "state": state,
                        "hours_in_state": round(hours, 1),
                        "max_hours": self.CONSERVATION_MAX_HOURS,
                        "entered_at": state_entered_at,
                    },
                )
                self._emit(alert)
                return alert
        except (ValueError, TypeError):
            pass

        return None

    def get_recent_alerts(self, limit: int = 20) -> List[dict]:
        """Get recent alerts from the alerts file."""
        if not self.alerts_path.exists():
            return []
        try:
            alerts = []
            with open(self.alerts_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            alerts.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return alerts[-limit:]
        except IOError:
            return []

    def get_alerts_by_type(self, alert_type: str, limit: int = 10) -> List[dict]:
        """Get recent alerts of a specific type."""
        all_alerts = self.get_recent_alerts(limit=100)
        return [a for a in all_alerts if a.get("alert_type") == alert_type][-limit:]

    def _emit(self, alert: Alert) -> None:
        """Write alert to file and optionally print to stderr."""
        entry = alert.to_dict()
        with open(self.alerts_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._restrict_permissions()
        print(f"[{alert.severity.upper()}] {alert.message}", file=sys.stderr)

    def _restrict_permissions(self) -> None:
        """Best-effort: set alert file permissions to owner-only read/write."""
        try:
            os.chmod(self.alerts_path, RESTRICTIVE_PERMISSIONS)
        except OSError:
            logger.warning("Failed to set restrictive permissions on %s (may not be supported)", self.alerts_path)
