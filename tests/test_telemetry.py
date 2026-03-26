"""Tests for RouteQualityReporter — route quality reporting."""

import json
from pathlib import Path
import pytest

from router.telemetry import RouteQualityReporter, RouteQualityReport, ExecutorStats


def _write_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_entry(task_id="t-001", task_class="implementation", state="normal",
                tool="codex_cli", backend="openai_native", model_profile="codex_primary",
                success=True, latency_ms=1000, cost_estimate_usd=0.02, normalized_error=None):
    entry = {
        "ts": "2024-01-01T00:00:00+00:00", "task_id": task_id, "agent": "coder",
        "task_class": task_class, "summary": "test task", "state": state,
        "chain": [{"tool": tool, "backend": backend, "model_profile": model_profile}],
        "reason": "standard chain", "attempted_fallback": False,
        "result": {"tool": tool, "backend": backend, "model_profile": model_profile,
                   "success": success, "latency_ms": latency_ms, "cost_estimate_usd": cost_estimate_usd},
    }
    if normalized_error:
        entry["result"]["normalized_error"] = normalized_error
    return entry


class TestLoadEntries:
    def test_load_entries_empty_file(self, tmp_path):
        reporter = RouteQualityReporter(log_path=tmp_path / "nonexistent.jsonl")
        assert reporter.load_entries() == []

    def test_load_entries_with_data(self, tmp_path):
        log_path = tmp_path / "routing.jsonl"
        _write_jsonl(log_path, [_make_entry(task_id=f"t-{i:03d}") for i in range(3)])
        entries = RouteQualityReporter(log_path=log_path).load_entries()
        assert len(entries) == 3
        assert entries[0]["task_id"] == "t-000"
        assert entries[2]["task_id"] == "t-002"

    def test_load_entries_skips_blank_lines(self, tmp_path):
        log_path = tmp_path / "routing.jsonl"
        log_path.write_text(json.dumps(_make_entry()) + "\n\n" + json.dumps(_make_entry(task_id="t-002")) + "\n")
        assert len(RouteQualityReporter(log_path=log_path).load_entries()) == 2


class TestGenerateReportFields:
    def test_report_has_all_fields(self, tmp_path):
        _write_jsonl(tmp_path / "routing.jsonl", [_make_entry()])
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        assert isinstance(report, RouteQualityReport)
        for attr in ["total_routes", "success_count", "failure_count", "overall_success_rate",
                      "by_executor", "by_task_class", "by_state", "most_common_errors",
                      "avg_latency_ms", "total_cost_usd"]:
            assert hasattr(report, attr)


class TestOverallStats:
    def test_overall_success_rate(self, tmp_path):
        entries = [_make_entry(task_id=f"t-s{i}", success=True) for i in range(8)] + \
                  [_make_entry(task_id=f"f-f{i}", success=False, normalized_error="rate_limited") for i in range(2)]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        assert report.total_routes == 10
        assert report.success_count == 8
        assert report.failure_count == 2
        assert report.overall_success_rate == pytest.approx(0.8)

    def test_avg_latency(self, tmp_path):
        entries = [_make_entry(task_id="t-001", latency_ms=1000), _make_entry(task_id="t-002", latency_ms=2000), _make_entry(task_id="t-003", latency_ms=3000)]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        assert RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report().avg_latency_ms == pytest.approx(2000.0)

    def test_total_cost(self, tmp_path):
        entries = [_make_entry(task_id="t-001", cost_estimate_usd=0.01), _make_entry(task_id="t-002", cost_estimate_usd=0.02), _make_entry(task_id="t-003", cost_estimate_usd=0.03)]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        assert RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report().total_cost_usd == pytest.approx(0.06)


class TestByExecutorStats:
    def test_groups_correctly_by_tool_backend(self, tmp_path):
        entries = [
            _make_entry(task_id="t-001", tool="codex_cli", backend="openai_native", success=True),
            _make_entry(task_id="t-002", tool="codex_cli", backend="openai_native", success=True),
            _make_entry(task_id="t-003", tool="claude_code", backend="anthropic", success=False, normalized_error="timeout"),
        ]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        codex = [s for s in report.by_executor if s.tool == "codex_cli"][0]
        claude = [s for s in report.by_executor if s.tool == "claude_code"][0]
        assert codex.total_calls == 2
        assert codex.success_count == 2
        assert claude.total_calls == 1
        assert claude.failure_count == 1

    def test_success_rate_per_executor(self, tmp_path):
        entries = [
            _make_entry(task_id=f"t-{i}", tool="codex_cli", backend="openai_native", success=i < 3)
            for i in range(4)
        ] + [
            _make_entry(task_id=f"t-c{i}", tool="claude_code", backend="anthropic", success=i == 0)
            for i in range(2)
        ]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        codex = [s for s in report.by_executor if s.tool == "codex_cli"][0]
        claude = [s for s in report.by_executor if s.tool == "claude_code"][0]
        assert codex.success_rate == pytest.approx(0.75)
        assert claude.success_rate == pytest.approx(0.5)

    def test_avg_latency_per_executor(self, tmp_path):
        entries = [
            _make_entry(task_id="t-001", tool="codex_cli", backend="openai_native", latency_ms=1000),
            _make_entry(task_id="t-002", tool="codex_cli", backend="openai_native", latency_ms=3000),
            _make_entry(task_id="t-003", tool="claude_code", backend="anthropic", latency_ms=5000),
        ]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        codex = [s for s in report.by_executor if s.tool == "codex_cli"][0]
        claude = [s for s in report.by_executor if s.tool == "claude_code"][0]
        assert codex.avg_latency_ms == pytest.approx(2000.0)
        assert claude.avg_latency_ms == pytest.approx(5000.0)


class TestByTaskClass:
    def test_groups_by_task_class(self, tmp_path):
        entries = [
            _make_entry(task_id="t-001", task_class="implementation", success=True),
            _make_entry(task_id="t-002", task_class="implementation", success=True),
            _make_entry(task_id="t-003", task_class="bugfix", success=False, normalized_error="timeout"),
        ]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        assert report.by_task_class["implementation"]["total"] == 2
        assert report.by_task_class["implementation"]["success"] == 2
        assert report.by_task_class["bugfix"]["failure"] == 1

    def test_cost_per_task_class(self, tmp_path):
        entries = [
            _make_entry(task_id="t-001", task_class="implementation", cost_estimate_usd=0.01),
            _make_entry(task_id="t-002", task_class="implementation", cost_estimate_usd=0.02),
            _make_entry(task_id="t-003", task_class="bugfix", cost_estimate_usd=0.05),
        ]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        assert report.by_task_class["implementation"]["total_cost_usd"] == pytest.approx(0.03)
        assert report.by_task_class["bugfix"]["total_cost_usd"] == pytest.approx(0.05)


class TestByState:
    def test_groups_by_state(self, tmp_path):
        entries = [
            _make_entry(task_id="t-001", state="normal", success=True),
            _make_entry(task_id="t-002", state="normal", success=True),
            _make_entry(task_id="t-003", state="last10", success=False, normalized_error="auth_error"),
        ]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        assert report.by_state["normal"]["total"] == 2
        assert report.by_state["normal"]["success"] == 2
        assert report.by_state["last10"]["failure"] == 1


class TestMostCommonErrors:
    def test_error_frequency_sorted(self, tmp_path):
        entries = [
            _make_entry(task_id=f"t-{i}", success=False, normalized_error="rate_limited") for i in range(3)
        ] + [
            _make_entry(task_id=f"t-t{i}", success=False, normalized_error="timeout") for i in range(2)
        ] + [_make_entry(task_id="t-a0", success=False, normalized_error="auth_error")]
        _write_jsonl(tmp_path / "routing.jsonl", entries)
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        assert report.most_common_errors[0] == ("rate_limited", 3)
        assert report.most_common_errors[1] == ("timeout", 2)
        assert report.most_common_errors[2] == ("auth_error", 1)


class TestZeroEntries:
    def test_zero_entries(self, tmp_path):
        reporter = RouteQualityReporter(log_path=tmp_path / "nonexistent.jsonl")
        report = reporter.generate_report()
        assert report.total_routes == 0
        assert report.success_count == 0
        assert report.overall_success_rate == 0.0
        assert report.by_executor == []


class TestEntriesWithoutResult:
    def test_entry_without_result_counted_as_failure(self, tmp_path):
        entry_no_result = {
            "ts": "2024-01-01T00:00:00+00:00", "task_id": "t-nr", "agent": "coder",
            "task_class": "implementation", "summary": "no result", "state": "normal",
            "chain": [], "reason": "standard", "attempted_fallback": False,
        }
        _write_jsonl(tmp_path / "routing.jsonl", [entry_no_result, _make_entry()])
        report = RouteQualityReporter(log_path=tmp_path / "routing.jsonl").generate_report()
        assert report.total_routes == 2
        assert report.success_count == 1
        assert report.failure_count == 1
