"""Routing benchmark runner.

Simple benchmark that measures routing path latency using mocks
to isolate routing decision overhead from executor latency.

Usage:
    python3 -m router.benchmark
"""

from tests.test_routing_benchmark import run_benchmarks

if __name__ == "__main__":
    run_benchmarks()
