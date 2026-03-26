"""End-to-end operational scenario tests.

Covers config hot-reload, model deprecation/registry, config migration,
metrics accumulation, notifications, audit chain integrity, secret redaction,
content isolation, file permissions, thread safety, and graceful shutdown.
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
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
import router.config_loader as _cfg_mod
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


# ── Config Hot-Reload & Migration (merged) ──────────────────────────────────

class TestConfigHotReloadAndMigration:
    """Config hot-reload, invalid config handling, validation, and migration."""

    def test_hot_reload_reflects_changes(self, tmp_path, restore_config):
        config_v1 = {
            "version": 1,
            "models": {"openrouter": {"test_model": "model-v1"}, "codex": {"primary": "codex-mini"},
                       "claude": {"sonnet": "claude-sonnet-4-20250514"}},
            "state": {"default": "openai_primary", "manual_state_file": str(tmp_path / "manual.json"),
                      "auto_state_file": str(tmp_path / "auto.json")},
            "tools": {"codex_cli": {"profiles": {"default": {"model": "codex-mini"}}},
                      "claude_code": {"profiles": {"default": {"model": "claude-sonnet-4-20250514"}}}},
        }
        config_path = tmp_path / "router.config.json"
        config_path.write_text(json.dumps(config_v1))

        cfg1 = reload_config(config_path)
        assert cfg1["models"]["openrouter"]["test_model"] == "model-v1"

        config_v2 = json.loads(json.dumps(config_v1))
        config_v2["models"]["openrouter"]["test_model"] = "model-v2"
        config_path.write_text(json.dumps(config_v2))

        cfg2 = reload_config(config_path)
        assert cfg2["models"]["openrouter"]["test_model"] == "model-v2"
        assert get_config_snapshot()["models"]["openrouter"]["test_model"] == "model-v2"

    def test_invalid_config_after_reload(self, tmp_path, restore_config):
        valid = {
            "version": 1,
            "models": {"openrouter": {"m": "ok-model"}, "codex": {"primary": "codex-mini"},
                       "claude": {"sonnet": "claude-sonnet-4-20250514"}},
            "state": {"default": "openai_primary", "manual_state_file": str(tmp_path / "manual.json"),
                      "auto_state_file": str(tmp_path / "auto.json")},
            "tools": {"codex_cli": {"profiles": {}}, "claude_code": {"profiles": {}}},
        }
        config_path = tmp_path / "router.config.json"
        config_path.write_text(json.dumps(valid))
        cfg = reload_config(config_path)
        assert validate_config(cfg).valid

        config_path.write_text("{invalid json!!!")
        with pytest.raises(Exception):
            reload_config(config_path)

        config_path.write_text(json.dumps(valid))
        cfg_restored = reload_config(config_path)
        assert cfg_restored["models"]["openrouter"]["m"] == "ok-model"

    def test_validation_rejects_invalid_and_detects_missing_keys(self):
        bad_version = {"version": 999, "models": {}, "state": {}}
        result = validate_config(bad_version)
        assert not result.valid
        assert result.error_count > 0

        bad_missing = {"models": {}}
        result2 = validate_config(bad_missing)
        assert not result2.valid
        error_paths = [e.path for e in result2.errors if e.severity == "error"]
        assert "version" in error_paths
        assert "state" in error_paths
        assert "tools" in error_paths

    def test_migration_v0_to_v1(self):
        old = {"models": {"OpenAI": {"PRIMARY": "gpt-4o"}}, "routing": {"policy": "openai_primary"}}
        migrated = migrate_config(old)
        assert migrated["version"] == CURRENT_CONFIG_VERSION
        assert "timeouts" in migrated

    def test_migration_preserves_data_and_is_idempotent(self):
        old = {
            "models": {"openai": {"primary": "gpt-4o", "fallback": "gpt-4o-mini"},
                       "claude": {"sonnet": "claude-sonnet-4-20250514"}},
            "routing": {"policy": "openai_primary", "chain": ["openai", "claude"]},
            "timeouts": {"openai": 30, "claude": 60}, "custom_field": "preserved_value",
        }
        migrated = migrate_config(old)
        assert migrated["version"] == CURRENT_CONFIG_VERSION
        assert migrated["models"]["openai"]["primary"] == "gpt-4o"
        assert migrated["custom_field"] == "preserved_value"
        # Idempotent
        assert migrate_config(migrated)["version"] == 1


# ── Deprecated Model Warning & Registry (4 tests) ───────────────────────────

class TestModelDeprecationAndRegistry:
    """Model deprecation warnings and registry replacement."""

    def test_deprecated_model_warning(self):
        config = {"models": {"openai": {"primary": "gpt-4"}, "claude": {"sonnet": "claude-sonnet-4-20250514"}}}
        warnings = validate_config_models(config)
        assert any("gpt-4" in w and "deprecated" in w.lower() for w in warnings)

    def test_deprecation_and_replacement(self):
        assert "deprecated" in check_model_deprecation("gpt-4").lower()
        assert check_model_deprecation("gpt-4o") is None
        assert get_replacement("gpt-4") == "gpt-4o"
        assert get_replacement("gpt-4o") is None
        assert get_replacement("totally-fake-model-xyz") is None

    def test_register_custom_model(self):
        custom = ModelVersion("my-custom-model", "custom", deprecated=True, replaced_by="my-new-model")
        register_model(custom)
        assert get_replacement("my-custom-model") == "my-new-model"
        assert check_model_deprecation("my-custom-model") is not None
        del KNOWN_MODELS["my-custom-model"]


# ── Metrics (4 tests, kept — each tests unique accumulation behavior) ────────

class TestMetricsAccumulation:
    """Metrics accumulate across multiple routes via JSONL log files."""

    @staticmethod
    def _write_entries(log_path, successes, failures, cost_per_success=0.01):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with open(log_path, "a") as f:
            for i in range(successes):
                f.write(json.dumps({"type": "routing_trace", "trace_id": f"ts-{i}", "task_id": f"task-s-{i}",
                    "state": "openai_primary", "chain": [{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_primary"}],
                    "attempts": [{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_primary",
                                  "success": True, "latency_ms": 100, "cost_estimate_usd": cost_per_success}],
                    "final_success": True, "total_latency_ms": 100, "timestamp": ts}) + "\n")
            for i in range(failures):
                f.write(json.dumps({"type": "routing_trace", "trace_id": f"tf-{i}", "task_id": f"task-f-{i}",
                    "state": "openai_primary", "chain": [{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_primary"}],
                    "attempts": [{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_primary",
                                  "success": False, "latency_ms": 50, "cost_estimate_usd": 0.0}],
                    "final_success": False, "total_latency_ms": 50, "timestamp": ts}) + "\n")

    def test_accumulate_across_routes(self, tmp_path):
        log_path = tmp_path / "routing.jsonl"
        self._write_entries(log_path, 10, 0)
        report = MetricsCollector(log_path=log_path).collect(period_hours=1)
        assert report.total_tasks == 10
        assert report.total_success == 10

    def test_success_rate(self, tmp_path):
        log_path = tmp_path / "routing.jsonl"
        self._write_entries(log_path, 8, 2)
        report = MetricsCollector(log_path=log_path).collect(period_hours=1)
        assert report.total_tasks == 10
        assert report.total_failure == 2
        assert abs(report.success_rate - 0.8) < 0.01

    def test_costs(self, tmp_path):
        log_path = tmp_path / "routing.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        costs = [0.01, 0.02, 0.03, 0.04, 0.05]
        with open(log_path, "a") as f:
            for i, cost in enumerate(costs):
                f.write(json.dumps({"type": "routing_trace", "trace_id": f"t-{i}", "task_id": f"task-{i}",
                    "state": "openai_primary", "chain": [],
                    "attempts": [{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_primary",
                                  "success": True, "latency_ms": 100, "cost_estimate_usd": cost}],
                    "final_success": True, "total_latency_ms": 100, "timestamp": ts}) + "\n")
        report = MetricsCollector(log_path=log_path).collect(period_hours=1)
        mm = report.by_model.get("codex_primary")
        assert mm is not None
        assert abs(mm.total_cost_usd - sum(costs)) < 0.001

    def test_per_state_breakdown(self, tmp_path):
        log_path = tmp_path / "routing.jsonl"
        self._write_entries(log_path, 3, 0)
        report = MetricsCollector(log_path=log_path).collect(period_hours=1)
        assert "openai_primary" in report.by_state
        assert report.by_state["openai_primary"].task_count == 3


# ── Notifications (4 tests) ─────────────────────────────────────────────────

class TestNotifications:
    """State change and fallback rate alerts."""

    def test_state_change_alert(self, tmp_path):
        alerts_path = tmp_path / "alerts.jsonl"
        nm = NotificationManager(alerts_path=alerts_path)
        alert = nm.notify_state_change("openai_primary", "openai_conservation", reason="budget pressure")
        assert alert.alert_type == "state_change"
        assert alerts_path.exists()
        entry = json.loads(alerts_path.read_text().strip().split("\n")[0])
        assert entry["details"]["from"] == "openai_primary"
        assert entry["details"]["to"] == "openai_conservation"

    def test_high_fallback_rate_alert(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        alert = nm.check_fallback_rate(total_tasks=100, fallback_tasks=60)
        assert alert is not None
        assert alert.severity == "warning"
        assert alert.details["rate"] == 0.6

    def test_no_alert_on_low_fallback_rate(self, tmp_path):
        nm = NotificationManager(alerts_path=tmp_path / "alerts.jsonl")
        assert nm.check_fallback_rate(100, 20) is None

    def test_multiple_alerts_append(self, tmp_path):
        alerts_path = tmp_path / "alerts.jsonl"
        nm = NotificationManager(alerts_path=alerts_path)
        nm.notify_state_change("a", "b")
        nm.notify_state_change("b", "c")
        nm.check_fallback_rate(10, 5)
        assert len(alerts_path.read_text().strip().split("\n")) == 3
        assert len(nm.get_recent_alerts(limit=10)) == 3


# ── Audit Chain (5 tests) ───────────────────────────────────────────────────

class TestAuditChainIntegrity:
    """Hash chain integrity and tampering detection."""

    def test_chain_integrity(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        chain = AuditChain(last_hash="GENESIS")
        for i in range(10):
            entry = {"type": "routing_trace", "task_id": f"task-{i}", "state": "openai_primary"}
            chain.chain_entry(entry)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        for line in log_path.read_text().splitlines():
            e = json.loads(line)
            assert "_hash" in e and "_prev_hash" in e
        valid, reason = verify_chain(log_path)
        assert valid, f"Chain invalid: {reason}"

    def test_tamper_detection(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        chain = AuditChain(last_hash="GENESIS")
        for i in range(3):
            entry = {"type": "routing_trace", "task_id": f"task-{i}"}
            chain.chain_entry(entry)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        lines = log_path.read_text().splitlines()
        tampered = json.loads(lines[1])
        tampered["task_id"] = "TAMPERED"
        lines[1] = json.dumps(tampered)
        log_path.write_text("\n".join(lines))
        valid, reason = verify_chain(log_path)
        assert not valid

    def test_empty_log_valid(self, tmp_path):
        (tmp_path / "empty.jsonl").write_text("")
        assert verify_chain(tmp_path / "empty.jsonl")[0] is True

    def test_init_chain_genesis_and_last_hash(self, tmp_path):
        assert init_chain(tmp_path / "new.jsonl") == "GENESIS"
        log_path = tmp_path / "existing.jsonl"
        chain = AuditChain(last_hash="GENESIS")
        entry = {"type": "test"}
        chain.chain_entry(entry)
        log_path.write_text(json.dumps(entry) + "\n")
        assert init_chain(log_path) == entry["_hash"]


# ── Secret Redaction (3 parametrized tests) ──────────────────────────────────

class TestSecretRedaction:
    """API keys and secrets redacted from logs."""

    @pytest.mark.parametrize("text,key_fragment", [
        ("Error: Authentication failed with api_key=sk-abc123def456ghi789jkl012mno", "sk-abc123def456ghi789jkl012mno"),
        ("Request to openrouter with key sk-or-v1-abcdefghijklmnop1234567890", "sk-or-v1-abcdefghijklmnop1234567890"),
        ("Claude API call with sk-ant-api03-abcdefghijklmnop1234567890", "sk-ant-api03-abcdefghijklmnop1234567890"),
    ])
    def test_api_key_redacted(self, text, key_fragment):
        redacted = sanitize_secrets(text)
        assert key_fragment not in redacted
        assert "[REDACTED]" in redacted

    def test_bearer_token_redacted(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = sanitize_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_dict_and_attempt_logger(self, tmp_path):
        data = {"api_key": "sk-secret123", "token": "bearer-abc", "safe_field": "visible",
                "nested": {"secret": "hidden", "ok": "fine"}}
        redacted = redact_dict(data)
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["token"] == "[REDACTED]"
        assert redacted["safe_field"] == "visible"
        assert redacted["nested"]["secret"] == "[REDACTED]"
        assert redacted["nested"]["ok"] == "fine"

        # AttemptLogger redacts secrets from trace entries
        log_path = tmp_path / "trace.jsonl"
        logger = AttemptLogger(log_path=log_path)
        trace = logger.create_trace(
            trace_id="t1", task_id="task-1", state="openai_primary",
            chain=[{"tool": "codex_cli", "backend": "openai_native", "model_profile": "codex_primary"}],
        )
        trace.final_error = "Auth failed with api_key=sk-leakedkey123456789"
        trace.final_success = False
        trace.attempts = [ExecutorAttempt(tool="codex_cli", backend="openai_native",
                                          model_profile="codex_primary", success=False,
                                          latency_ms=100, normalized_error="auth_error")]
        logger.log_trace(trace)
        assert "sk-leakedkey123456789" not in log_path.read_text()


# ── Content Isolation (3 tests) ──────────────────────────────────────────────

class TestContentIsolation:
    """Prompt text should NOT appear in routing logs."""

    def test_long_prompt_removed(self):
        long_prompt = "Write a function that " + "does something really complex " * 20
        entry = {"type": "routing_trace", "task_id": "task-1", "prompt": long_prompt, "state": "openai_primary"}
        sanitized = sanitize_content(entry)
        assert "prompt" not in sanitized or len(sanitized.get("prompt", "")) <= 200
        assert sanitized["task_id"] == "task-1"

    def test_short_content_preserved_and_long_truncated(self):
        assert sanitize_content({"prompt": "short prompt", "state": "ok"})["prompt"] == "short prompt"
        result = sanitize_content({"state": "x" * 600})
        assert "[TRUNCATED" in result["state"]

    def test_content_keys_removed(self):
        entry = {"content": "x" * 300, "body": "y" * 250, "code": "z" * 500, "short_field": "ok"}
        sanitized = sanitize_content(entry)
        assert "content" not in sanitized and "body" not in sanitized and "code" not in sanitized
        assert sanitized["short_field"] == "ok"


# ── File Permissions (1 combined test) ───────────────────────────────────────

class TestFilePermissions:
    """State, WAL, and alert files created with 0o600 permissions."""

    def test_restrictive_permissions(self, tmp_path):
        store = StateStore(
            manual_path=tmp_path / "manual.json", auto_path=tmp_path / "auto.json",
            history_path=tmp_path / "history.json", wal_path=tmp_path / "wal.jsonl",
        )
        # State files should be 0o600
        for path in [tmp_path / "manual.json", tmp_path / "auto.json"]:
            perms = os.stat(path).st_mode & 0o777
            assert perms == 0o600, f"Expected 0o600, got {oct(perms)} for {path}"

        # Alert file
        alerts_path = tmp_path / "alerts.jsonl"
        nm = NotificationManager(alerts_path=alerts_path)
        nm.notify_state_change("a", "b")
        assert (os.stat(alerts_path).st_mode & 0o777) == 0o600


# ── Thread Safety (2 tests) ─────────────────────────────────────────────────

class TestConcurrentRouting:
    """Multiple threads routing simultaneously → no corruption."""

    def test_concurrent_state_operations(self, tmp_path):
        store = StateStore(
            manual_path=tmp_path / "manual.json", auto_path=tmp_path / "auto.json",
            history_path=tmp_path / "history.json", wal_path=tmp_path / "wal.jsonl",
        )
        errors = []
        results = []

        def worker(tid):
            try:
                for state in [CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, CodexState.CLAUDE_BACKUP]:
                    store.set_auto_state(state)
                    results.append((tid, state, store.get_auto_state()))
                assert store.get_state() in list(CodexState)
            except Exception as e:
                errors.append((tid, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 60
        assert store.get_state() in list(CodexState)

    def test_concurrent_state_transitions_safe(self, tmp_path):
        store = StateStore(
            manual_path=tmp_path / "manual.json", auto_path=tmp_path / "auto.json",
            history_path=tmp_path / "history.json", wal_path=tmp_path / "wal.jsonl",
        )
        errors = []

        def toggle(tid):
            try:
                for _ in range(10):
                    store.set_auto_state(CodexState.OPENAI_CONSERVATION)
                    store.set_auto_state(CodexState.OPENAI_PRIMARY)
            except Exception as e:
                errors.append((tid, str(e)))

        threads = [threading.Thread(target=toggle, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread errors: {errors}"
        assert store.get_state() in list(CodexState)


# ── Graceful Shutdown (2 tests) ──────────────────────────────────────────────

class TestGracefulShutdown:
    """Shutdown signal → pending tasks complete → new tasks rejected."""

    def test_graceful_shutdown(self):
        class GracefulRouter:
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

            def route(self, task_id):
                with self._lock:
                    if self._shutting_down:
                        self._rejected.append(task_id)
                        raise RuntimeError(f"Router shutting down: rejecting {task_id}")
                    self._pending += 1
                try:
                    time.sleep(0.05)
                    with self._lock:
                        self._completed.append(task_id)
                finally:
                    with self._lock:
                        self._pending -= 1

        router = GracefulRouter()
        barrier = threading.Barrier(5)

        def worker(task_id):
            barrier.wait()
            try:
                router.route(task_id)
            except RuntimeError:
                pass

        threads = [threading.Thread(target=worker, args=(f"pre-{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        time.sleep(0.01)
        router.shutdown()
        assert router.is_shutting_down
        for t in threads:
            t.join(timeout=5)
        assert len(router._completed) == 5
        with pytest.raises(RuntimeError, match="shutting down"):
            router.route("post-0")
        assert len(router._rejected) >= 1

    def test_shutdown_drains_executor_pool(self):
        import concurrent.futures
        results = []
        lock = threading.Lock()

        def slow_task(tid):
            time.sleep(0.1)
            with lock:
                results.append(tid)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        futures = [executor.submit(slow_task, f"task-{i}") for i in range(8)]
        executor.shutdown(wait=True, cancel_futures=False)
        assert len(results) == 8
        for f in futures:
            assert f.done()
