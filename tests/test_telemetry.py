"""Tests for RouteQualityReporter — X-88 route quality reporting."""

import json
from pathlib import Path

import pytest

from router.telemetry import (
    RouteQualityReporter,
    RouteQualityReport,
    ExecutorStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, entries: list[dict]):
    """Write entries as JSONL to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_entry(
    task_id: str = "t-001",
    task_class: str = "implementation",
    state: str = "normal",
    tool: str = "codex_cli",
    backend: str = "openai_native",
    model_profile: str = "codex_primary",
    success: bool = True,
    latency_ms: int = 1000,
    cost_estimate_usd: float = 0.02,
    normalized_error: str | None = None,
) -> dict:
    """Create a routing.jsonl entry."""
    entry = {
        "ts": "2024-01-01T00:00:00+00:00",
        "task_id": task_id,
        "agent": "coder",
        "task_class": task_class,
        "summary": "test task",
        "state": state,
        "chain": [{"tool": tool, "backend": backend, "model_profile": model_profile}],
        "reason": "standard chain",
        "attempted_fallback": False,
        "result": {
            "tool": tool,
            "backend": backend,
            "model_profile": model_profile,
            "success": success,
            "latency_ms": latency_ms,
            "cost_estimate_usd": cost_estimate_usd,
        },
    }
    if normalized_error:
        entry["result"]["normalized_error"] = normalized_error
    return entry


# ---------------------------------------------------------------------------
# TestLoadEntries
# ---------------------------------------------------------------------------

class TestLoadEntries:
    def test_load_entries_empty_file(self, tmp_path):
        """No file → empty list."""
        reporter = RouteQualityReporter(log_path=tmp_path / "nonexistent.jsonl")
        entries = reporter.load_entries()
        assert entries == []

    def test_load_entries_with_data(self, tmp_path):
        """Writes test data, loads correctly."""
        log_path = tmp_path / "routing.jsonl"
        data = [
            _make_entry(task_id="t-001"),
            _make_entry(task_id="t-002"),
            _make_entry(task_id="t-003"),
        ]
        _write_jsonl(log_path, data)
        reporter = RouteQualityReporter(log_path=log_path)
        entries = reporter.load_entries()
        assert len(entries) == 3
        assert entries[0]["task_id"] == "t-001"
        assert entries[2]["task_id"] == "t-003"

    def test_load_entries_skips_blank_lines(self, tmp_path):
        """Blank lines are ignored."""
        log_path = tmp_path / "routing.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(_make_entry()) + "\n\n" + json.dumps(_make_entry(task_id="t-002")) + "\n"
        log_path.write_text(content)
        reporter = RouteQualityReporter(log_path=log_path)
        entries = reporter.load_entries()
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# TestGenerateReportFields
# ---------------------------------------------------------------------------

class TestGenerateReportFields:
    def test_generate_report_fields(self, tmp_path):
        """Report has all required fields."""
        log_path = tmp_path / "routing.jsonl"
        _write_jsonl(log_path, [_make_entry()])
        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        assert isinstance(report, RouteQualityReport)
        assert hasattr(report, "total_routes")
        assert hasattr(report, "success_count")
        assert hasattr(report, "failure_count")
        assert hasattr(report, "overall_success_rate")
        assert hasattr(report, "by_executor")
        assert hasattr(report, "by_task_class")
        assert hasattr(report, "by_state")
        assert hasattr(report, "most_common_errors")
        assert hasattr(report, "avg_latency_ms")
        assert hasattr(report, "total_cost_usd")


# ---------------------------------------------------------------------------
# TestOverallSuccessRate
# ---------------------------------------------------------------------------

class TestOverallSuccessRate:
    def test_overall_success_rate(self, tmp_path):
        """8 successes out of 10 → 80%."""
        log_path = tmp_path / "routing.jsonl"
        entries = []
        for i in range(8):
            entries.append(_make_entry(task_id=f"t-s{i}", success=True))
        for i in range(2):
            entries.append(_make_entry(task_id=f"f-f{i}", success=False, normalized_error="rate_limited"))
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        assert report.total_routes == 10
        assert report.success_count == 8
        assert report.failure_count == 2
        assert report.overall_success_rate == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# TestByExecutorStats
# ---------------------------------------------------------------------------

class TestByExecutorStats:
    def test_by_executor_stats(self, tmp_path):
        """Groups correctly by tool:backend."""
        log_path = tmp_path / "routing.jsonl"
        entries = [
            _make_entry(task_id="t-001", tool="codex_cli", backend="openai_native", success=True, latency_ms=1000, cost_estimate_usd=0.01),
            _make_entry(task_id="t-002", tool="codex_cli", backend="openai_native", success=True, latency_ms=2000, cost_estimate_usd=0.02),
            _make_entry(task_id="t-003", tool="claude_code", backend="anthropic", success=False, latency_ms=3000, cost_estimate_usd=0.03, normalized_error="timeout"),
        ]
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        codex_stats = [s for s in report.by_executor if s.tool == "codex_cli" and s.backend == "openai_native"]
        assert len(codex_stats) == 1
        assert codex_stats[0].total_calls == 2
        assert codex_stats[0].success_count == 2
        assert codex_stats[0].failure_count == 0

        claude_stats = [s for s in report.by_executor if s.tool == "claude_code" and s.backend == "anthropic"]
        assert len(claude_stats) == 1
        assert claude_stats[0].total_calls == 1
        assert claude_stats[0].success_count == 0
        assert claude_stats[0].failure_count == 1


# ---------------------------------------------------------------------------
# TestByTaskClass
# ---------------------------------------------------------------------------

class TestByTaskClass:
    def test_by_task_class(self, tmp_path):
        """Groups by task_class."""
        log_path = tmp_path / "routing.jsonl"
        entries = [
            _make_entry(task_id="t-001", task_class="implementation", success=True),
            _make_entry(task_id="t-002", task_class="implementation", success=True),
            _make_entry(task_id="t-003", task_class="bugfix", success=False, normalized_error="timeout"),
        ]
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        assert "implementation" in report.by_task_class
        assert "bugfix" in report.by_task_class
        assert report.by_task_class["implementation"]["total"] == 2
        assert report.by_task_class["implementation"]["success"] == 2
        assert report.by_task_class["bugfix"]["total"] == 1
        assert report.by_task_class["bugfix"]["failure"] == 1


# ---------------------------------------------------------------------------
# TestByState
# ---------------------------------------------------------------------------

class TestByState:
    def test_by_state(self, tmp_path):
        """Groups by normal vs last10."""
        log_path = tmp_path / "routing.jsonl"
        entries = [
            _make_entry(task_id="t-001", state="normal", success=True),
            _make_entry(task_id="t-002", state="normal", success=True),
            _make_entry(task_id="t-003", state="last10", success=False, normalized_error="auth_error"),
        ]
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        assert "normal" in report.by_state
        assert "last10" in report.by_state
        assert report.by_state["normal"]["total"] == 2
        assert report.by_state["normal"]["success"] == 2
        assert report.by_state["last10"]["total"] == 1
        assert report.by_state["last10"]["failure"] == 1


# ---------------------------------------------------------------------------
# TestMostCommonErrors
# ---------------------------------------------------------------------------

class TestMostCommonErrors:
    def test_most_common_errors(self, tmp_path):
        """Error frequency sorted."""
        log_path = tmp_path / "routing.jsonl"
        entries = [
            _make_entry(task_id="t-001", success=False, normalized_error="rate_limited"),
            _make_entry(task_id="t-002", success=False, normalized_error="rate_limited"),
            _make_entry(task_id="t-003", success=False, normalized_error="rate_limited"),
            _make_entry(task_id="t-004", success=False, normalized_error="timeout"),
            _make_entry(task_id="t-005", success=False, normalized_error="timeout"),
            _make_entry(task_id="t-006", success=False, normalized_error="auth_error"),
        ]
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        assert len(report.most_common_errors) >= 3
        # Sorted by frequency descending
        assert report.most_common_errors[0][0] == "rate_limited"
        assert report.most_common_errors[0][1] == 3
        assert report.most_common_errors[1][0] == "timeout"
        assert report.most_common_errors[1][1] == 2
        assert report.most_common_errors[2][0] == "auth_error"
        assert report.most_common_errors[2][1] == 1


# ---------------------------------------------------------------------------
# TestAvgLatency
# ---------------------------------------------------------------------------

class TestAvgLatency:
    def test_avg_latency(self, tmp_path):
        """Average latency across entries."""
        log_path = tmp_path / "routing.jsonl"
        entries = [
            _make_entry(task_id="t-001", latency_ms=1000),
            _make_entry(task_id="t-002", latency_ms=2000),
            _make_entry(task_id="t-003", latency_ms=3000),
        ]
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        assert report.avg_latency_ms == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# TestTotalCost
# ---------------------------------------------------------------------------

class TestTotalCost:
    def test_total_cost(self, tmp_path):
        """Sum of cost_estimate_usd."""
        log_path = tmp_path / "routing.jsonl"
        entries = [
            _make_entry(task_id="t-001", cost_estimate_usd=0.01),
            _make_entry(task_id="t-002", cost_estimate_usd=0.02),
            _make_entry(task_id="t-003", cost_estimate_usd=0.03),
        ]
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        assert report.total_cost_usd == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# TestSuccessRatePerExecutor
# ---------------------------------------------------------------------------

class TestSuccessRatePerExecutor:
    def test_success_rate_per_executor(self, tmp_path):
        """Per-executor success rate calculation."""
        log_path = tmp_path / "routing.jsonl"
        entries = [
            _make_entry(task_id="t-001", tool="codex_cli", backend="openai_native", success=True),
            _make_entry(task_id="t-002", tool="codex_cli", backend="openai_native", success=True),
            _make_entry(task_id="t-003", tool="codex_cli", backend="openai_native", success=True),
            _make_entry(task_id="t-004", tool="codex_cli", backend="openai_native", success=False, normalized_error="timeout"),
            _make_entry(task_id="t-005", tool="claude_code", backend="anthropic", success=True),
            _make_entry(task_id="t-006", tool="claude_code", backend="anthropic", success=False, normalized_error="auth_error"),
        ]
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        codex = [s for s in report.by_executor if s.tool == "codex_cli"][0]
        claude = [s for s in report.by_executor if s.tool == "claude_code"][0]

        assert codex.success_rate == pytest.approx(0.75)
        assert claude.success_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestZeroEntries
# ---------------------------------------------------------------------------

class TestZeroEntries:
    def test_zero_entries(self, tmp_path):
        """Empty report handles gracefully."""
        reporter = RouteQualityReporter(log_path=tmp_path / "nonexistent.jsonl")
        report = reporter.generate_report()

        assert report.total_routes == 0
        assert report.success_count == 0
        assert report.failure_count == 0
        assert report.overall_success_rate == 0.0
        assert report.by_executor == []
        assert report.by_task_class == {}
        assert report.by_state == {}
        assert report.most_common_errors == []
        assert report.avg_latency_ms == 0.0
        assert report.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# TestEntriesWithoutResult
# ---------------------------------------------------------------------------

class TestEntriesWithoutResult:
    def test_entry_without_result_skipped_in_stats(self, tmp_path):
        """Entries without result block are counted but don't contribute to executor stats."""
        log_path = tmp_path / "routing.jsonl"
        entry_no_result = {
            "ts": "2024-01-01T00:00:00+00:00",
            "task_id": "t-nr",
            "agent": "coder",
            "task_class": "implementation",
            "summary": "no result",
            "state": "normal",
            "chain": [],
            "reason": "standard",
            "attempted_fallback": False,
        }
        _write_jsonl(log_path, [entry_no_result, _make_entry()])

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        # Entry without result has no success/failure data — counted as failure
        assert report.total_routes == 2
        assert report.success_count == 1
        assert report.failure_count == 1


# ---------------------------------------------------------------------------
# TestCostPerTaskClass
# ---------------------------------------------------------------------------

class TestCostPerTaskClass:
    def test_cost_per_task_class(self, tmp_path):
        """Cost is aggregated per task class."""
        log_path = tmp_path / "routing.jsonl"
        entries = [
            _make_entry(task_id="t-001", task_class="implementation", cost_estimate_usd=0.01),
            _make_entry(task_id="t-002", task_class="implementation", cost_estimate_usd=0.02),
            _make_entry(task_id="t-003", task_class="bugfix", cost_estimate_usd=0.05),
        ]
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        assert report.by_task_class["implementation"]["total_cost_usd"] == pytest.approx(0.03)
        assert report.by_task_class["bugfix"]["total_cost_usd"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# TestLatencyPerExecutor
# ---------------------------------------------------------------------------

class TestLatencyPerExecutor:
    def test_avg_latency_per_executor(self, tmp_path):
        """Average latency is computed per executor."""
        log_path = tmp_path / "routing.jsonl"
        entries = [
            _make_entry(task_id="t-001", tool="codex_cli", backend="openai_native", latency_ms=1000),
            _make_entry(task_id="t-002", tool="codex_cli", backend="openai_native", latency_ms=3000),
            _make_entry(task_id="t-003", tool="claude_code", backend="anthropic", latency_ms=5000),
        ]
        _write_jsonl(log_path, entries)

        reporter = RouteQualityReporter(log_path=log_path)
        report = reporter.generate_report()

        codex = [s for s in report.by_executor if s.tool == "codex_cli"][0]
        claude = [s for s in report.by_executor if s.tool == "claude_code"][0]

        assert codex.avg_latency_ms == pytest.approx(2000.0)
        assert claude.avg_latency_ms == pytest.approx(5000.0)
