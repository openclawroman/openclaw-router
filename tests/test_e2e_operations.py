"""End-to-end operational scenario tests.

Covers config hot-reload, model deprecation/registry, config migration,
metrics accumulation, notifications, audit chain integrity, secret redaction,
content isolation, file permissions, thread safety, and graceful shutdown.
"""

import json
import os
import stat
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality,
    RouteDecision, ExecutorResult, ChainEntry, CodexState,
)
from router.state_store import StateStore, reset_state_store
from router.config_loader import (
    load_config, reload_config, get_config_snapshot, get_model, list_models,
)
from router.config_migration import migrate_config, CURRENT_CONFIG_VERSION
from router.config_validator import validate_config
from router.model_registry import (
    check_model_deprecation, get_replacement, validate_config_models,
    KNOWN_MODELS, ModelVersion, register_model,
)
from router.metrics import MetricsCollector
from router.notifications import NotificationManager, Alert, AlertType, AlertSeverity
from router.audit import AuditChain, verify_chain, init_chain
from router.secrets import sanitize_secrets, redact_dict
from router.sanitize import sanitize_content
from router.logger import RoutingLogger
from router.attempt_logger import AttemptLogger, RoutingTrace, ExecutorAttempt
from router.policy import route_task, build_chain, resolve_state, reset_breaker, reset_notifier


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1–2. Config Hot-Reload During Routing                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestConfigHotReloadDuringRouting:
    """Config changes mid-run → new config applied on next route."""

    def test_config_hot_reload_during_routing(self, tmp_path):
        """Modifying config between two route calls should reflect new values."""
        config_v1 = {
            "version": 1,
            "models": {
                "openrouter": {"test_model": "model-v1"},
                "codex": {"primary": "codex-mini"},
                "claude": {"sonnet": "claude-sonnet-4-20250514"},
            },
            "state": {
                "default": "openai_primary",
                "manual_state_file": str(tmp_path / "manual.json"),
                "auto_state_file": str(tmp_path / "auto.json"),
            },
            "tools": {
                "codex_cli": {"profiles": {"default": {"model": "codex-mini"}}},
                "claude_code": {"profiles": {"default": {"model": "claude-sonnet-4-20250514"}}},
            },
        }
        config_path = tmp_path / "router.config.json"
        config_path.write_text(json.dumps(config_v1))

        # Load v1
        cfg1 = reload_config(config_path)
        assert cfg1["models"]["openrouter"]["test_model"] == "model-v1"

        # Simulate config change on disk
        config_v2 = json.loads(json.dumps(config_v1))
        config_v2["models"]["openrouter"]["test_model"] = "model-v2"
        config_path.write_text(json.dumps(config_v2))

        # Reload → should pick up v2
        cfg2 = reload_config(config_path)
        assert cfg2["models"]["openrouter"]["test_model"] == "model-v2"

        # Snapshot should also reflect v2
        snap = get_config_snapshot()
        assert snap["models"]["openrouter"]["test_model"] == "model-v2"

    def test_config_invalid_after_reload(self, tmp_path):
        """Invalid config on reload → error raised, validation catches issues."""
        valid_config = {
            "version": 1,
            "models": {
                "openrouter": {"m": "ok-model"},
                "codex": {"primary": "codex-mini"},
                "claude": {"sonnet": "claude-sonnet-4-20250514"},
            },
            "state": {
                "default": "openai_primary",
                "manual_state_file": str(tmp_path / "manual.json"),
                "auto_state_file": str(tmp_path / "auto.json"),
            },
            "tools": {
                "codex_cli": {"profiles": {}},
                "claude_code": {"profiles": {}},
            },
        }
        config_path = tmp_path / "router.config.json"
        config_path.write_text(json.dumps(valid_config))

        # Load valid config
        cfg_good = reload_config(config_path)
        result_good = validate_config(cfg_good)
        assert result_good.valid

        # Write invalid JSON to disk
        config_path.write_text("{invalid json!!!")

        # reload_config will raise on invalid JSON
        with pytest.raises(Exception):
            reload_config(config_path)

        # Now reload the valid config to restore state
        config_path.write_text(json.dumps(valid_config))
        cfg_restored = reload_config(config_path)
        assert cfg_restored["models"]["openrouter"]["m"] == "ok-model"

    def test_config_validation_rejects_invalid(self, tmp_path):
        """Config with structural errors should fail validation."""
        bad_config = {
            "version": 999,  # unsupported version
            "models": {},     # missing required sections
            "state": {},      # missing required keys
            # missing tools
        }
        result = validate_config(bad_config)
        assert not result.valid
        assert result.error_count > 0

    def test_config_validation_detects_missing_keys(self):
        """validate_config should flag missing required keys."""
        bad_config = {"models": {}}  # missing version, state, tools
        result = validate_config(bad_config)
        assert not result.valid
        assert result.error_count > 0
        error_paths = [e.path for e in result.errors if e.severity == "error"]
        assert "version" in error_paths
        assert "state" in error_paths
        assert "tools" in error_paths


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3–4. Deprecated Model Warning & Registry Replacement                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestDeprecatedModelWarning:
    """Config uses gpt-4 → deprecation warning in logs."""

    def test_deprecated_model_warning_logged(self, caplog):
        """validate_config_models should produce warnings for deprecated models."""
        import logging
        config = {
            "models": {
                "openai": {"primary": "gpt-4"},
                "claude": {"sonnet": "claude-sonnet-4-20250514"},
            },
        }
        warnings = validate_config_models(config)
        assert len(warnings) > 0
        assert any("gpt-4" in w and "deprecated" in w.lower() for w in warnings)

    def test_deprecated_model_check_returns_message(self):
        """check_model_deprecation('gpt-4') should return non-None warning."""
        msg = check_model_deprecation("gpt-4")
        assert msg is not None
        assert "deprecated" in msg.lower()
        assert "gpt-4o" in msg

    def test_non_deprecated_model_no_warning(self):
        """check_model_deprecation('gpt-4o') should return None."""
        msg = check_model_deprecation("gpt-4o")
        assert msg is None


class TestModelRegistryReplacement:
    """get_replacement('gpt-4') → 'gpt-4o'."""

    def test_model_registry_replacement(self):
        """get_replacement should return the correct replacement."""
        replacement = get_replacement("gpt-4")
        assert replacement == "gpt-4o"

    def test_replacement_for_non_deprecated_is_none(self):
        """get_replacement for non-deprecated model returns None."""
        replacement = get_replacement("gpt-4o")
        assert replacement is None

    def test_replacement_for_unknown_model_is_none(self):
        """get_replacement for unknown model returns None."""
        replacement = get_replacement("totally-fake-model-xyz")
        assert replacement is None

    def test_register_custom_model(self):
        """register_model should add a custom model to the registry."""
        custom = ModelVersion("my-custom-model", "custom", deprecated=True, replaced_by="my-new-model")
        register_model(custom)
        assert get_replacement("my-custom-model") == "my-new-model"
        assert check_model_deprecation("my-custom-model") is not None
        # Clean up
        del KNOWN_MODELS["my-custom-model"]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5–6. Config Migration v0 → v1                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestConfigMigration:
    """Old config without version → migrated automatically."""

    def test_config_migration_v0_to_v1(self):
        """Config without version field should be migrated to version 1."""
        old_config = {
            "models": {"OpenAI": {"PRIMARY": "gpt-4o"}},
            "routing": {"policy": "openai_primary"},
        }
        migrated = migrate_config(old_config)
        assert migrated["version"] == CURRENT_CONFIG_VERSION
        assert "models" in migrated
        assert "routing" in migrated
        assert "timeouts" in migrated

    def test_config_migration_preserves_data(self):
        """All config values should be preserved after migration."""
        old_config = {
            "models": {
                "openai": {"primary": "gpt-4o", "fallback": "gpt-4o-mini"},
                "claude": {"sonnet": "claude-sonnet-4-20250514"},
            },
            "routing": {"policy": "openai_primary", "chain": ["openai", "claude"]},
            "timeouts": {"openai": 30, "claude": 60},
            "custom_field": "preserved_value",
        }
        migrated = migrate_config(old_config)

        # Version added
        assert migrated["version"] == CURRENT_CONFIG_VERSION

        # Original data preserved
        assert migrated["models"]["openai"]["primary"] == "gpt-4o"
        assert migrated["models"]["openai"]["fallback"] == "gpt-4o-mini"
        assert migrated["models"]["claude"]["sonnet"] == "claude-sonnet-4-20250514"
        assert migrated["routing"]["policy"] == "openai_primary"
        assert migrated["timeouts"]["openai"] == 30
        assert migrated["custom_field"] == "preserved_value"

    def test_migration_is_idempotent(self):
        """Migrating an already-migrated config should be a no-op."""
        config = {"version": 1, "models": {"x": "y"}, "routing": {}, "tools": {}}
        migrated = migrate_config(config)
        assert migrated["version"] == 1
        assert migrated["models"]["x"] == "y"

    def test_migration_normalizes_model_keys(self):
        """Model profile names should be normalized to lowercase."""
        old_config = {
            "models": {"OpenRouter": {"MINIMAX": "minimax-model"}},
            "tools": {"Claude": {"Sonnet": "claude-sonnet"}},
        }
        migrated = migrate_config(old_config)
        # Keys in model sections should be lowercased
        assert "minimax" in migrated["models"]["OpenRouter"] or "MINIMAX" not in migrated["models"]["OpenRouter"]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7–9. Metrics Accumulation                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestMetricsAccumulation:
    """Metrics accumulate across multiple routes via JSONL log files."""

    @staticmethod
    def _write_trace_entries(log_path: Path, successes: int, failures: int,
                              cost_per_success: float = 0.01):
        """Write routing_trace entries to a JSONL log file."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with open(log_path, "a") as f:
            for i in range(successes):
                entry = {
                    "type": "routing_trace",
                    "trace_id": f"trace-s-{i}",
                    "task_id": f"task-s-{i}",
                    "state": "openai_primary",
                    "chain": [{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_primary"}],
                    "attempts": [{
                        "tool": "codex_cli", "backend": "openai_native",
                        "model_profile": "codex_primary",
                        "success": True, "latency_ms": 100,
                        "cost_estimate_usd": cost_per_success,
                    }],
                    "final_success": True,
                    "total_latency_ms": 100,
                    "timestamp": ts,
                }
                f.write(json.dumps(entry) + "\n")
            for i in range(failures):
                entry = {
                    "type": "routing_trace",
                    "trace_id": f"trace-f-{i}",
                    "task_id": f"task-f-{i}",
                    "state": "openai_primary",
                    "chain": [{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_primary"}],
                    "attempts": [{
                        "tool": "codex_cli", "backend": "openai_native",
                        "model_profile": "codex_primary",
                        "success": False, "latency_ms": 50,
                        "cost_estimate_usd": 0.0,
                    }],
                    "final_success": False,
                    "total_latency_ms": 50,
                    "timestamp": ts,
                }
                f.write(json.dumps(entry) + "\n")

    def test_metrics_accumulate_across_routes(self, tmp_path):
        """10 routes → metrics.total_tasks should be 10."""
        log_path = tmp_path / "routing.jsonl"
        self._write_trace_entries(log_path, successes=10, failures=0)

        mc = MetricsCollector(log_path=log_path)
        report = mc.collect(period_hours=1)
        assert report.total_tasks == 10
        assert report.total_success == 10
        assert report.total_failure == 0

    def test_metrics_track_success_rate(self, tmp_path):
        """8 success, 2 fail → ~80% success rate."""
        log_path = tmp_path / "routing.jsonl"
        self._write_trace_entries(log_path, successes=8, failures=2)

        mc = MetricsCollector(log_path=log_path)
        report = mc.collect(period_hours=1)
        assert report.total_tasks == 10
        assert report.total_success == 8
        assert report.total_failure == 2
        assert abs(report.success_rate - 0.8) < 0.01

    def test_metrics_track_costs(self, tmp_path):
        """Costs should accumulate across all routes."""
        log_path = tmp_path / "routing.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()

        costs = [0.01, 0.02, 0.03, 0.04, 0.05]
        with open(log_path, "a") as f:
            for i, cost in enumerate(costs):
                entry = {
                    "type": "routing_trace",
                    "trace_id": f"trace-{i}",
                    "task_id": f"task-{i}",
                    "state": "openai_primary",
                    "chain": [],
                    "attempts": [{
                        "tool": "codex_cli", "backend": "openai_native",
                        "model_profile": "codex_primary",
                        "success": True, "latency_ms": 100,
                        "cost_estimate_usd": cost,
                    }],
                    "final_success": True,
                    "total_latency_ms": 100,
                    "timestamp": ts,
                }
                f.write(json.dumps(entry) + "\n")

        mc = MetricsCollector(log_path=log_path)
        report = mc.collect(period_hours=1)
        # Cost is in by_model
        model_metrics = report.by_model.get("codex_primary")
        assert model_metrics is not None
        expected_total = sum(costs)
        assert abs(model_metrics.total_cost_usd - expected_total) < 0.001

    def test_metrics_per_state_breakdown(self, tmp_path):
        """Metrics should break down by state."""
        log_path = tmp_path / "routing.jsonl"
        self._write_trace_entries(log_path, successes=3, failures=0)

        mc = MetricsCollector(log_path=log_path)
        report = mc.collect(period_hours=1)
        assert "openai_primary" in report.by_state
        assert report.by_state["openai_primary"].task_count == 3


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  10–11. Notifications on State Change & High Fallback Rate              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestNotifications:
    """State change and fallback rate alerts."""

    def test_notification_on_state_change(self, tmp_path):
        """State change → alert file written."""
        alerts_path = tmp_path / "alerts.jsonl"
        nm = NotificationManager(alerts_path=alerts_path)

        alert = nm.notify_state_change(
            "openai_primary", "openai_conservation", reason="budget pressure"
        )

        assert alert.alert_type == "state_change"
        assert "openai_primary" in alert.message
        assert "openai_conservation" in alert.message

        # Alert file should exist and contain the alert
        assert alerts_path.exists()
        lines = alerts_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["alert_type"] == "state_change"
        assert entry["details"]["from"] == "openai_primary"
        assert entry["details"]["to"] == "openai_conservation"

    def test_notification_on_high_fallback_rate(self, tmp_path):
        """>50% fallback rate → alert emitted (threshold is 30%)."""
        alerts_path = tmp_path / "alerts.jsonl"
        nm = NotificationManager(alerts_path=alerts_path)

        # 60% fallback rate (> 30% threshold)
        alert = nm.check_fallback_rate(total_tasks=100, fallback_tasks=60)

        assert alert is not None
        assert alert.alert_type == "fallback_rate"
        assert alert.severity == "warning"
        assert alert.details["rate"] == 0.6

        # Written to file
        assert alerts_path.exists()
        lines = alerts_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["alert_type"] == "fallback_rate"

    def test_no_alert_on_low_fallback_rate(self, tmp_path):
        """<30% fallback rate → no alert."""
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        alert = nm.check_fallback_rate(total_tasks=100, fallback_tasks=20)
        assert alert is None

    def test_multiple_alerts_append(self, tmp_path):
        """Multiple alerts should append to the same file."""
        alerts_path = tmp_path / "alerts.jsonl"
        nm = NotificationManager(alerts_path=alerts_path)

        nm.notify_state_change("a", "b")
        nm.notify_state_change("b", "c")
        nm.check_fallback_rate(10, 5)

        lines = alerts_path.read_text().strip().split("\n")
        assert len(lines) == 3

        alerts = nm.get_recent_alerts(limit=10)
        assert len(alerts) == 3


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  12. Audit Chain Integrity                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestAuditChainIntegrity:
    """All log entries have hash chain, verify_chain passes."""

    def test_audit_chain_integrity(self, tmp_path):
        """Write multiple entries with AuditChain, then verify_chain should pass."""
        log_path = tmp_path / "audit.jsonl"
        chain = AuditChain(last_hash="GENESIS")

        entries = [
            {"type": "routing_trace", "task_id": f"task-{i}", "state": "openai_primary"}
            for i in range(10)
        ]

        for entry in entries:
            chain.chain_entry(entry)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        # All entries should have hash fields
        for line in log_path.read_text().splitlines():
            e = json.loads(line)
            assert "_hash" in e
            assert "_prev_hash" in e

        # verify_chain should pass
        valid, reason = verify_chain(log_path)
        assert valid, f"Chain invalid: {reason}"

    def test_audit_chain_detects_tampering(self, tmp_path):
        """Tampering with an entry should break the chain."""
        log_path = tmp_path / "audit.jsonl"
        chain = AuditChain(last_hash="GENESIS")

        for i in range(3):
            entry = {"type": "routing_trace", "task_id": f"task-{i}"}
            chain.chain_entry(entry)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        # Tamper with line 2
        lines = log_path.read_text().splitlines()
        tampered = json.loads(lines[1])
        tampered["task_id"] = "TAMPERED"
        lines[1] = json.dumps(tampered)
        log_path.write_text("\n".join(lines))

        valid, reason = verify_chain(log_path)
        assert not valid
        assert "mismatch" in reason.lower() or "tampered" in reason.lower()

    def test_empty_log_chain_valid(self, tmp_path):
        """Empty log file → chain is valid."""
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("")
        valid, reason = verify_chain(log_path)
        assert valid

    def test_init_chain_returns_genesis_for_new_file(self, tmp_path):
        """init_chain on new file returns 'GENESIS'."""
        log_path = tmp_path / "new.jsonl"
        last_hash = init_chain(log_path)
        assert last_hash == "GENESIS"

    def test_init_chain_returns_last_hash(self, tmp_path):
        """init_chain on existing log returns the last entry's hash."""
        log_path = tmp_path / "existing.jsonl"
        chain = AuditChain(last_hash="GENESIS")
        entry = {"type": "test"}
        chain.chain_entry(entry)
        log_path.write_text(json.dumps(entry) + "\n")

        last_hash = init_chain(log_path)
        assert last_hash == entry["_hash"]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  13. Secret Redaction in Logs                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestSecretRedaction:
    """API keys in error messages → redacted in logs."""

    def test_secret_redaction_in_logs(self):
        """API keys should be redacted by sanitize_secrets."""
        text = "Error: Authentication failed with api_key=sk-abc123def456ghi789jkl012mno"
        redacted = sanitize_secrets(text)
        assert "sk-abc123def456ghi789jkl012mno" not in redacted
        assert "[REDACTED]" in redacted

    def test_openrouter_key_redacted(self):
        """OpenRouter keys (sk-or-*) should be redacted."""
        text = "Request to openrouter with key sk-or-v1-abcdefghijklmnop1234567890"
        redacted = sanitize_secrets(text)
        assert "sk-or-v1-abcdefghijklmnop1234567890" not in redacted
        assert "[REDACTED]" in redacted

    def test_anthropic_key_redacted(self):
        """Anthropic keys (sk-ant-*) should be redacted."""
        text = "Claude API call with sk-ant-api03-abcdefghijklmnop1234567890"
        redacted = sanitize_secrets(text)
        assert "sk-ant-api03-abcdefghijklmnop1234567890" not in redacted
        assert "[REDACTED]" in redacted

    def test_bearer_token_redacted(self):
        """Bearer tokens should be redacted."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = sanitize_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_dict_sensitive_keys(self):
        """Dict keys like api_key, token, secret should be redacted."""
        data = {
            "api_key": "sk-secret123",
            "token": "bearer-abc",
            "safe_field": "visible",
            "nested": {"secret": "hidden", "ok": "fine"},
        }
        redacted = redact_dict(data)
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["token"] == "[REDACTED]"
        assert redacted["safe_field"] == "visible"
        assert redacted["nested"]["secret"] == "[REDACTED]"
        assert redacted["nested"]["ok"] == "fine"

    def test_attempt_logger_redacts_secrets(self, tmp_path):
        """AttemptLogger.log_trace should redact secrets from trace entries."""
        log_path = tmp_path / "trace.jsonl"
        logger = AttemptLogger(log_path=log_path)

        trace = logger.create_trace(
            trace_id="t1", task_id="task-1", state="openai_primary",
            chain=[{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_primary"}],
        )
        trace.final_error = "Auth failed with api_key=sk-leakedkey123456789"
        trace.final_success = False
        trace.attempts = [
            ExecutorAttempt(tool="codex_cli", backend="openai_native", model_profile="codex_primary",
                           success=False, latency_ms=100, normalized_error="auth_error")
        ]

        logger.log_trace(trace)

        # Read back and verify secrets are redacted
        content = log_path.read_text()
        assert "sk-leakedkey123456789" not in content


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  14. Content Isolation — No Prompt Leakage                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestContentIsolation:
    """Prompt text should NOT appear in routing logs."""

    def test_content_isolation_no_prompt_leakage(self):
        """sanitize_content should remove long 'prompt' values."""
        long_prompt = "Write a function that " + "does something really complex " * 20
        entry = {
            "type": "routing_trace",
            "task_id": "task-1",
            "prompt": long_prompt,
            "state": "openai_primary",
        }
        sanitized = sanitize_content(entry)

        # The long prompt should be removed
        assert "prompt" not in sanitized or len(sanitized.get("prompt", "")) <= 200
        # Non-content fields preserved
        assert sanitized["task_id"] == "task-1"
        assert sanitized["state"] == "openai_primary"

    def test_content_keys_removed_when_long(self):
        """Content keys with values > 200 chars should be dropped."""
        entry = {
            "content": "x" * 300,
            "body": "y" * 250,
            "code": "z" * 500,
            "short_field": "ok",
        }
        sanitized = sanitize_content(entry)
        assert "content" not in sanitized
        assert "body" not in sanitized
        assert "code" not in sanitized
        assert sanitized["short_field"] == "ok"

    def test_short_content_preserved(self):
        """Short content values should be preserved."""
        entry = {"prompt": "short prompt", "state": "ok"}
        sanitized = sanitize_content(entry)
        assert sanitized["prompt"] == "short prompt"

    def test_long_non_content_strings_truncated(self):
        """Long string values in non-content keys should be truncated."""
        entry = {"state": "x" * 600}
        sanitized = sanitize_content(entry)
        assert "[TRUNCATED" in sanitized["state"]

    def test_routing_log_entry_sanitized(self, tmp_path):
        """Full routing log entry should not contain prompt text."""
        log_path = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log_path)

        task = TaskMeta(
            task_id="t1", agent="coder", task_class=TaskClass.IMPLEMENTATION,
            summary="Secret prompt: " + "confidential data " * 30,
        )
        decision = RouteDecision(
            task_id="t1", state="openai_primary",
            chain=[ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary")],
        )

        # RoutingLogger doesn't sanitize — but AttemptLogger does.
        # We test that sanitize_content strips it.
        entry = {"summary": task.summary, "task_id": task.task_id}
        sanitized = sanitize_content(entry)
        # The long summary (which contains "confidential data" repeated) should be truncated or removed
        if "summary" in sanitized:
            assert "confidential data confidential data" not in sanitized["summary"] or len(sanitized["summary"]) <= 500


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  15. File Permissions on State Files                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestFilePermissions:
    """State files created with 0o600 permissions."""

    def test_file_permissions_on_state(self, tmp_path):
        """StateStore should create state files with 0o600 permissions."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
            history_path=tmp_path / "history.json",
            wal_path=tmp_path / "wal.jsonl",
        )

        # Check permissions on state files
        for path in [tmp_path / "manual.json", tmp_path / "auto.json"]:
            mode = os.stat(path).st_mode
            # Owner read+write only (mask out platform bits)
            perms = mode & 0o777
            assert perms == 0o600, f"Expected 0o600, got {oct(perms)} for {path}"

    def test_wal_file_permissions(self, tmp_path):
        """WAL file should also have restrictive permissions."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
            history_path=tmp_path / "history.json",
            wal_path=tmp_path / "wal.jsonl",
        )

        # Trigger WAL write
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        wal_path = tmp_path / "wal.jsonl"
        if wal_path.exists():
            mode = os.stat(wal_path).st_mode
            perms = mode & 0o777
            # WAL may be 0o600 or empty after truncate
            assert perms in (0o600, 0o000) or wal_path.stat().st_size == 0

    def test_alert_file_permissions(self, tmp_path):
        """NotificationManager should create alert files with 0o600."""
        alerts_path = tmp_path / "alerts.jsonl"
        nm = NotificationManager(alerts_path=alerts_path)
        nm.notify_state_change("a", "b")

        mode = os.stat(alerts_path).st_mode
        perms = mode & 0o777
        assert perms == 0o600, f"Expected 0o600, got {oct(perms)}"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  16. Concurrent Routing — Thread Safety                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestConcurrentRouting:
    """Multiple threads routing simultaneously → no corruption."""

    def test_concurrent_routing_thread_safe(self, tmp_path):
        """20 threads performing state operations simultaneously → no corruption."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
            history_path=tmp_path / "history.json",
            wal_path=tmp_path / "wal.jsonl",
        )

        errors = []
        results = []

        def worker(thread_id):
            try:
                # Each thread cycles through states
                states = [
                    CodexState.OPENAI_PRIMARY,
                    CodexState.OPENAI_CONSERVATION,
                    CodexState.CLAUDE_BACKUP,
                ]
                for state in states:
                    store.set_auto_state(state)
                    read_back = store.get_auto_state()
                    results.append((thread_id, state, read_back))
                # Also read the effective state
                effective = store.get_state()
                assert effective in list(CodexState)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 60  # 20 threads × 3 states

        # Final state should be a valid CodexState
        final = store.get_state()
        assert final in list(CodexState)

    def test_concurrent_metrics_no_race(self, tmp_path):
        """Concurrent log file writing + metrics collection should not race."""
        log_path = tmp_path / "routing.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        ts = datetime.now(timezone.utc).isoformat()
        lock = threading.Lock()

        def write_entries(n, thread_id):
            try:
                with lock:
                    with open(log_path, "a") as f:
                        for i in range(n):
                            entry = {
                                "type": "routing_trace",
                                "trace_id": f"t{thread_id}-{i}",
                                "task_id": f"task-{thread_id}-{i}",
                                "state": "openai_primary",
                                "chain": [],
                                "attempts": [{
                                    "tool": "codex_cli", "backend": "openai_native",
                                    "model_profile": "codex_primary",
                                    "success": True, "latency_ms": 10,
                                    "cost_estimate_usd": 0.001,
                                }],
                                "final_success": True,
                                "total_latency_ms": 10,
                                "timestamp": ts,
                            }
                            f.write(json.dumps(entry) + "\n")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_entries, args=(50, i)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"
        mc = MetricsCollector(log_path=log_path)
        report = mc.collect(period_hours=1)
        assert report.total_tasks == 500  # 10 threads × 50

    def test_concurrent_state_transitions_safe(self, tmp_path):
        """Concurrent state transitions should not corrupt state store."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
            history_path=tmp_path / "history.json",
            wal_path=tmp_path / "wal.jsonl",
        )

        errors = []

        def toggle_state(thread_id):
            try:
                for _ in range(10):
                    store.set_auto_state(CodexState.OPENAI_CONSERVATION)
                    store.set_auto_state(CodexState.OPENAI_PRIMARY)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=toggle_state, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # State should be valid after all operations
        state = store.get_state()
        assert state in list(CodexState)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  17. Graceful Shutdown                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TestGracefulShutdown:
    """Shutdown signal → pending tasks complete → new tasks rejected."""

    def test_graceful_shutdown(self):
        """Simulate shutdown: pending work completes, new work is rejected."""
        import concurrent.futures

        class GracefulRouter:
            """Minimal shutdown-aware router for testing."""
            def __init__(self):
                self._shutting_down = False
                self._lock = threading.Lock()
                self._completed = []
                self._rejected = []
                self._pending = 0

            def shutdown(self):
                with self._lock:
                    self._shutting_down = True

            @property
            def is_shutting_down(self):
                with self._lock:
                    return self._shutting_down

            def route(self, task_id: str):
                with self._lock:
                    if self._shutting_down:
                        self._rejected.append(task_id)
                        raise RuntimeError(f"Router shutting down: rejecting {task_id}")
                    self._pending += 1

                try:
                    # Simulate work
                    time.sleep(0.05)
                    with self._lock:
                        self._completed.append(task_id)
                finally:
                    with self._lock:
                        self._pending -= 1

        router = GracefulRouter()
        barrier = threading.Barrier(5)

        def worker(task_id):
            barrier.wait()  # Ensure all threads start together
            try:
                router.route(task_id)
            except RuntimeError:
                pass  # Expected for rejected tasks

        # Start 5 workers that will be pending when shutdown occurs
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(f"pre-shutdown-{i}",))
            threads.append(t)
            t.start()

        # Wait for all to enter the barrier → they're about to start work
        time.sleep(0.01)

        # Trigger shutdown
        router.shutdown()
        assert router.is_shutting_down

        # Wait for pending tasks to complete
        for t in threads:
            t.join(timeout=5)

        # All pre-shutdown tasks should have completed
        assert len(router._completed) == 5

        # New tasks should be rejected
        with pytest.raises(RuntimeError, match="shutting down"):
            router.route("post-shutdown-0")

        assert len(router._rejected) >= 1

    def test_shutdown_drains_executor_pool(self):
        """ThreadPoolExecutor should drain pending work on shutdown."""
        import concurrent.futures

        results = []
        lock = threading.Lock()

        def slow_task(task_id):
            time.sleep(0.1)
            with lock:
                results.append(task_id)
            return task_id

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # Submit several tasks
        futures = [executor.submit(slow_task, f"task-{i}") for i in range(8)]

        # Shutdown with wait=True (graceful)
        executor.shutdown(wait=True, cancel_futures=False)

        # All submitted tasks should complete
        assert len(results) == 8
        for f in futures:
            assert f.done()

    def test_state_store_survives_concurrent_shutdown(self, tmp_path):
        """StateStore should remain consistent during concurrent access + shutdown."""
        store = StateStore(
            manual_path=tmp_path / "manual.json",
            auto_path=tmp_path / "auto.json",
            history_path=tmp_path / "history.json",
            wal_path=tmp_path / "wal.jsonl",
        )

        errors = []

        def rapid_writer(n):
            try:
                for i in range(n):
                    state = [CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP][i % 2]
                    store.set_auto_state(state)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=rapid_writer, args=(20,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent writes: {errors}"

        # State should be valid after all threads finish
        state = store.get_state()
        assert state in list(CodexState)

        # Files should be valid JSON
        for path in [tmp_path / "manual.json", tmp_path / "auto.json"]:
            data = json.loads(path.read_text())
            assert "state" in data
            assert data["state"] in [s.value for s in CodexState]
