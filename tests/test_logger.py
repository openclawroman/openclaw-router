"""Tests for RoutingLogger — X-80 logger compliance."""

import json
from pathlib import Path

import pytest

from router.logger import RoutingLogger
from router.models import (
    ChainEntry,
    ExecutorResult,
    RouteDecision,
    TaskClass,
    TaskMeta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task(**kw):
    defaults = dict(
        task_id="t-001",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        summary="add logging",
    )
    defaults.update(kw)
    return TaskMeta(**defaults)


def _decision(**kw):
    defaults = dict(
        task_id="t-001",
        state="normal",
        chain=[ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary")],
        reason="best fit",
        attempted_fallback=False,
    )
    defaults.update(kw)
    return RouteDecision(**defaults)


def _result(**kw):
    defaults = dict(
        task_id="t-001",
        tool="codex_cli",
        backend="openai_native",
        model_profile="codex_primary",
        success=True,
        latency_ms=1200,
        request_id="req-abc",
        cost_estimate_usd=0.03,
        artifacts=["src/main.py"],
        final_summary="done",
    )
    defaults.update(kw)
    return ExecutorResult(**defaults)


def _read_lines(path: Path):
    return path.read_text().strip().splitlines()


# ---------------------------------------------------------------------------
# TestBasicLogging
# ---------------------------------------------------------------------------

class TestBasicLogging:
    def test_creates_file_on_first_write(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        logger.log(_task(), _decision())
        assert log.exists()

    def test_appends_on_second_write(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        logger.log(_task(), _decision())
        logger.log(_task(task_id="t-002"), _decision(task_id="t-002"))
        lines = _read_lines(log)
        assert len(lines) == 2

    def test_valid_json_each_line(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        logger.log(_task(), _decision())
        logger.log(_task(task_id="t-002"), _decision(task_id="t-002"))
        for line in _read_lines(log):
            parsed = json.loads(line)  # raises if invalid
            assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# TestLogFields
# ---------------------------------------------------------------------------

class TestLogFields:
    def test_all_required_fields_present(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        logger.log(_task(), _decision())
        entry = json.loads(log.read_text().strip())
        for key in ("ts", "task_id", "agent", "task_class", "summary", "state", "chain", "reason"):
            assert key in entry, f"missing field: {key}"


# ---------------------------------------------------------------------------
# TestWithResult
# ---------------------------------------------------------------------------

class TestWithResult:
    def test_result_fields_present(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        logger.log(_task(), _decision(), result=_result())
        entry = json.loads(log.read_text().strip())
        res = entry["result"]
        for key in ("tool", "backend", "model_profile", "success", "latency_ms",
                     "request_id", "cost_estimate_usd", "artifacts", "final_summary"):
            assert key in res, f"missing result field: {key}"

    def test_normalized_error_present_when_set(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        r = _result(normalized_error="TIMEOUT")
        logger.log(_task(), _decision(), result=r)
        entry = json.loads(log.read_text().strip())
        assert entry["result"]["normalized_error"] == "TIMEOUT"

    def test_normalized_error_absent_when_none(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        r = _result(normalized_error=None)
        logger.log(_task(), _decision(), result=r)
        entry = json.loads(log.read_text().strip())
        assert "normalized_error" not in entry["result"]


# ---------------------------------------------------------------------------
# TestStdoutStderrRef
# ---------------------------------------------------------------------------

class TestStdoutStderrRef:
    def test_refs_included_when_present(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        r = _result(stdout_ref="/logs/stdout.txt", stderr_ref="/logs/stderr.txt")
        logger.log(_task(), _decision(), result=r)
        entry = json.loads(log.read_text().strip())
        assert entry["result"]["stdout_ref"] == "/logs/stdout.txt"
        assert entry["result"]["stderr_ref"] == "/logs/stderr.txt"


# ---------------------------------------------------------------------------
# TestStdoutStderrRefOmitted
# ---------------------------------------------------------------------------

class TestStdoutStderrRefOmitted:
    def test_refs_omitted_when_none(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        r = _result(stdout_ref=None, stderr_ref=None)
        logger.log(_task(), _decision(), result=r)
        entry = json.loads(log.read_text().strip())
        assert "stdout_ref" not in entry["result"]
        assert "stderr_ref" not in entry["result"]


# ---------------------------------------------------------------------------
# TestFallbackLogging
# ---------------------------------------------------------------------------

class TestFallbackLogging:
    def test_fallback_from_included(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        d = _decision(attempted_fallback=True, fallback_from="codex_cli")
        logger.log(_task(), d)
        entry = json.loads(log.read_text().strip())
        assert entry["attempted_fallback"] is True
        assert entry["fallback_from"] == "codex_cli"

    def test_fallback_from_absent_when_no_fallback(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        logger.log(_task(), _decision())
        entry = json.loads(log.read_text().strip())
        assert "fallback_from" not in entry


# ---------------------------------------------------------------------------
# TestCustomLogPath
# ---------------------------------------------------------------------------

class TestCustomLogPath:
    def test_custom_path(self, tmp_path):
        custom = tmp_path / "custom" / "log.jsonl"
        logger = RoutingLogger(log_path=custom)
        assert logger.log_path == custom
        assert custom.parent.exists()


# ---------------------------------------------------------------------------
# TestGetLogPath
# ---------------------------------------------------------------------------

class TestGetLogPath:
    def test_returns_correct_path(self, tmp_path):
        p = tmp_path / "out.jsonl"
        logger = RoutingLogger(log_path=p)
        assert logger.get_log_path() == p


# ---------------------------------------------------------------------------
# TestTimestampFormat
# ---------------------------------------------------------------------------

class TestTimestampFormat:
    def test_ts_is_iso_format(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        logger.log(_task(), _decision())
        entry = json.loads(log.read_text().strip())
        ts = entry["ts"]
        # ISO format ends with Z or +HH:MM
        assert "T" in ts
        assert ts.endswith("Z") or "+" in ts.split("T")[1]


# ---------------------------------------------------------------------------
# TestEmptyTaskId
# ---------------------------------------------------------------------------

class TestEmptyTaskId:
    def test_handles_empty_task_id(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        t = _task(task_id="")
        d = _decision(task_id="")
        logger.log(t, d)
        entry = json.loads(log.read_text().strip())
        assert entry["task_id"] == ""


# ---------------------------------------------------------------------------
# TestChainFormat
# ---------------------------------------------------------------------------

class TestChainFormat:
    def test_chain_entries_have_required_keys(self, tmp_path):
        log = tmp_path / "routing.jsonl"
        logger = RoutingLogger(log_path=log)
        chain = [
            ChainEntry(tool="codex_cli", backend="openai_native", model_profile="codex_primary"),
            ChainEntry(tool="claude_code", backend="anthropic", model_profile="claude_primary"),
        ]
        d = _decision(chain=chain)
        logger.log(_task(), d)
        entry = json.loads(log.read_text().strip())
        assert len(entry["chain"]) == 2
        for ce in entry["chain"]:
            assert "tool" in ce
            assert "backend" in ce
            assert "model_profile" in ce
