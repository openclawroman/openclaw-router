#!/usr/bin/env python3
"""check_coverage.py — Parse coverage.xml and report per-module coverage.

Fails if any critical module has < 70% coverage.

Usage:
    python3 -m pytest tests/ --cov=router --cov-report=xml -q
    python3 scripts/check_coverage.py
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_XML = PROJECT_ROOT / "coverage.xml"
MIN_MODULE_COVERAGE = 70

# Modules that must meet the per-module threshold
CRITICAL_MODULES = [
    "router/policy.py",
    "router/state_store.py",
    "router/executors.py",
    "router/errors.py",
    "router/models.py",
    "router/config_loader.py",
]


def parse_coverage_xml(path: Path) -> dict[str, float]:
    """Return {filename: line_rate} from a coverage.xml file."""
    tree = ET.parse(path)
    root = tree.getroot()
    results = {}
    for pkg in root.findall(".//package"):
        for cls in pkg.findall(".//class"):
            filename = cls.get("filename", "")
            line_rate = float(cls.get("line-rate", "0"))
            results[filename] = line_rate * 100  # Convert to percentage
    return results


def main():
    if not COVERAGE_XML.exists():
        print(f"ERROR: {COVERAGE_XML} not found.")
        print("Run: python3 -m pytest tests/ --cov=router --cov-report=xml -q")
        sys.exit(1)

    coverage = parse_coverage_xml(COVERAGE_XML)

    print("=" * 60)
    print("Per-Module Coverage Report")
    print("=" * 60)

    failures = []

    # Report all modules
    for filename in sorted(coverage.keys()):
        pct = coverage[filename]
        short = filename.replace("router/", "")
        marker = ""
        if filename in CRITICAL_MODULES:
            marker = " [CRITICAL]"
            if pct < MIN_MODULE_COVERAGE:
                failures.append((filename, pct))
        status = "✓" if pct >= MIN_MODULE_COVERAGE else "✗"
        print(f"  {status} {short:40s} {pct:6.2f}%{marker}")

    print("=" * 60)
    total = sum(coverage.values()) / len(coverage) if coverage else 0
    print(f"  Average: {total:.2f}%")
    print()

    if failures:
        print(f"FAILED: {len(failures)} critical module(s) below {MIN_MODULE_COVERAGE}%:")
        for filename, pct in failures:
            print(f"  ✗ {filename}: {pct:.2f}%")
        sys.exit(1)
    else:
        print("PASSED: All critical modules meet the coverage threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()
