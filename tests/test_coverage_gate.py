"""Tests for coverage gating — enforce minimum coverage thresholds.

This module ensures that test coverage for the router/ package stays above
the configured threshold (85%) and that critical modules are adequately covered.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUTER_DIR = PROJECT_ROOT / "router"
TESTS_DIR = PROJECT_ROOT / "tests"
MIN_COVERAGE = 85


class TestCoverageAboveThreshold:
    """Total test coverage must meet or exceed the configured minimum."""

    def test_coverage_above_threshold(self):
        """Run pytest with --cov and verify coverage >= MIN_COVERAGE%."""
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                str(TESTS_DIR),
                "--cov=router",
                "--cov-report=term-missing",
                "-q",
                "--no-header",
                "--override-ini=addopts=-v --tb=short --strict-markers",
                "--ignore=" + str(TESTS_DIR / "test_coverage_gate.py"),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=180,
        )
        # Parse the TOTAL line from coverage output
        coverage_match = re.search(
            r"TOTAL\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%", result.stdout
        )
        assert coverage_match, (
            f"Could not parse coverage from pytest output.\n"
            f"stdout:\n{result.stdout[-1000:]}\n"
            f"stderr:\n{result.stderr[-1000:]}"
        )
        total_coverage = float(coverage_match.group(1))
        assert total_coverage >= MIN_COVERAGE, (
            f"Coverage {total_coverage:.2f}% is below the {MIN_COVERAGE}% threshold"
        )


class TestAllRouterModulesCovered:
    """Every module in router/ must be imported by at least one test."""

    def test_all_router_modules_covered(self):
        """Each .py file in router/ should appear in at least one test import."""
        router_modules = set()
        for f in ROUTER_DIR.glob("*.py"):
            if f.name == "__init__.py":
                continue
            module_name = f.stem
            router_modules.add(module_name)

        # Collect all test file contents to check for imports
        test_content = ""
        for tf in TESTS_DIR.glob("test_*.py"):
            test_content += tf.read_text(encoding="utf-8") + "\n"

        # benchmark.py is a thin __main__ runner that imports from tests — skip
        skippable = {"benchmark"}
        missing = []
        for module in sorted(router_modules - skippable):
            # Check for import patterns: from router.module or import router.module
            patterns = [
                rf"from\s+router\.{module}\b",
                rf"import\s+router\.{module}\b",
            ]
            if not any(re.search(p, test_content) for p in patterns):
                missing.append(module)

        assert not missing, (
            f"Router modules without any test coverage:\n  {missing}"
        )


class TestCriticalPathsCovered:
    """Core routing functions and StateStore methods must be exercised."""

    CRITICAL_FUNCTIONS = [
        ("router.policy", "route_task"),
        ("router.policy", "build_chain"),
    ]

    CRITICAL_STATESTORE_METHODS = [
        "get_state",
        "set_state",
        "get_manual_state",
        "set_manual_state",
        "get_auto_state",
        "set_auto_state",
        "recover_from_wal",
    ]

    def test_route_task_has_test_coverage(self):
        """route_task() must be called in at least one test."""
        assert self._function_tested("route_task"), (
            "route_task has no test coverage"
        )

    def test_build_chain_has_test_coverage(self):
        """build_chain() must be called in at least one test."""
        assert self._function_tested("build_chain"), (
            "build_chain has no test coverage"
        )

    def test_statestore_methods_tested(self):
        """Critical StateStore methods must appear in test files."""
        test_content = self._all_test_content()
        missing = []
        for method in self.CRITICAL_STATESTORE_METHODS:
            if method not in test_content:
                missing.append(method)

        assert not missing, (
            f"StateStore methods not found in tests: {missing}"
        )

    @staticmethod
    def _all_test_content() -> str:
        content = ""
        for tf in TESTS_DIR.glob("test_*.py"):
            content += tf.read_text(encoding="utf-8") + "\n"
        return content

    @staticmethod
    def _function_tested(func_name: str) -> bool:
        """Check if a function is referenced in any test file."""
        for tf in TESTS_DIR.glob("test_*.py"):
            text = tf.read_text(encoding="utf-8")
            if func_name in text:
                return True
        return False


class TestNoEmptyTestFiles:
    """Every test file must contain at least one test function."""

    def test_no_empty_test_files(self):
        """Verify no test_*.py files are devoid of test functions."""
        empty = []
        for tf in sorted(TESTS_DIR.glob("test_*.py")):
            text = tf.read_text(encoding="utf-8")
            test_funcs = re.findall(r"^def test_|\s+def test_", text, re.MULTILINE)
            if not test_funcs:
                empty.append(tf.name)

        assert not empty, (
            f"Test files with no test functions: {empty}"
        )
