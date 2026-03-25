"""Tests for state change notifications and alerting."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from router.notifications import NotificationManager, Alert, AlertType, AlertSeverity


class TestAlert:
    def test_alert_defaults(self):
        alert = Alert(alert_type="state_change", severity="info", message="test")
        assert alert.alert_type == "state_change"
        assert alert.severity == "info"
        assert alert.timestamp  # auto-set

    def test_alert_to_dict(self):
        alert = Alert(alert_type="state_change", severity="info", message="test", details={"from": "a"})
        d = alert.to_dict()
        assert d["alert_type"] == "state_change"
        assert d["details"]["from"] == "a"


class TestStateChangeNotification:
    def test_notify_state_change(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        alert = nm.notify_state_change("openai_primary", "openai_conservation", reason="budget pressure")

        assert alert.alert_type == "state_change"
        assert "openai_primary" in alert.message
        assert "openai_conservation" in alert.message
        assert alert.details["reason"] == "budget pressure"

        # Should be written to file
        lines = (tmp_path / "alerts.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["alert_type"] == "state_change"

    def test_notify_state_change_append(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        nm.notify_state_change("a", "b")
        nm.notify_state_change("b", "c")

        lines = (tmp_path / "alerts.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2


class TestFallbackRateAlert:
    def test_fallback_rate_below_threshold(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        alert = nm.check_fallback_rate(total_tasks=100, fallback_tasks=20)
        assert alert is None  # 20% < 30%

    def test_fallback_rate_above_threshold(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        alert = nm.check_fallback_rate(total_tasks=100, fallback_tasks=35)

        assert alert is not None
        assert alert.alert_type == "fallback_rate"
        assert alert.severity == "warning"
        assert "35%" in alert.message or "35.0%" in alert.message
        assert alert.details["rate"] == 0.35

    def test_fallback_rate_empty(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        alert = nm.check_fallback_rate(total_tasks=0, fallback_tasks=0)
        assert alert is None


class TestConservationDurationAlert:
    def test_not_in_conservation(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        alert = nm.check_conservation_duration("openai_primary")
        assert alert is None

    def test_conservation_under_threshold(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        from datetime import datetime, timezone
        recent = datetime.now(timezone.utc).isoformat()
        alert = nm.check_conservation_duration("openai_conservation", recent)
        assert alert is None

    def test_conservation_over_threshold(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        alert = nm.check_conservation_duration("openai_conservation", old)

        assert alert is not None
        assert alert.alert_type == "conservation_duration"
        assert alert.severity == "critical"
        assert "25" in alert.message
        assert alert.details["hours_in_state"] > 24


class TestGetAlerts:
    def test_get_recent_alerts(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        nm.notify_state_change("a", "b")
        nm.notify_state_change("c", "d")

        alerts = nm.get_recent_alerts(limit=10)
        assert len(alerts) == 2
        assert alerts[0]["alert_type"] == "state_change"

    def test_get_alerts_by_type(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        nm.notify_state_change("a", "b")
        nm.check_fallback_rate(10, 5)

        state_alerts = nm.get_alerts_by_type("state_change")
        assert len(state_alerts) == 1

        fb_alerts = nm.get_alerts_by_type("fallback_rate")
        assert len(fb_alerts) == 1
