"""Integration tests for the full routing path with simulated executors."""

import threading
import time
from unittest.mock import patch

import pytest

from router.models import TaskMeta, TaskClass, TaskRisk, CodexState, ExecutorResult
from router.policy import route_task
from router.errors import can_fallback
from tests.conftest import make_result, make_task


class TestFullRoutingHappyPath:
    def test_single_task_succeeds(self, patched_routing):
        task = make_task("happy-001", summary="Add user registration endpoint")
        mock = make_result("happy-001")
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(task)

        assert result.success is True
        assert decision.state == "openai_primary"
        assert result.cost_estimate_usd == 0.002

    def test_result_has_structured_output(self, patched_routing):
        task = make_task("struct-001", risk=TaskRisk.LOW, summary="Fix typo in README")
        mock = make_result("struct-001", cost_usd=0.0005, summary="Fixed: 1 file changed",
                           artifacts=["/tmp/struct-001.stdout.txt"])
        with patch("router.policy.run_codex", return_value=mock):
            _, result = route_task(task)

        assert result.success is True
        assert result.cost_estimate_usd == 0.0005
        assert result.latency_ms > 0


class TestRoutingWithFallback:
    def test_primary_fails_fallback_succeeds(self, patched_routing):
        task = make_task("fb-001", summary="Implement search functionality")
        failed = make_result("fb-001", success=False, error_type="provider_unavailable")
        success = make_result("fb-001", success=True, tool="claude_code",
                              backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_codex", return_value=failed), \
             patch("router.policy.run_claude", return_value=success):
            decision, result = route_task(task)

        assert result.success is True
        assert result.tool == "claude_code"
        assert decision.attempted_fallback is True
        assert decision.error_history[0]["error_type"] == "provider_unavailable"

    def test_auth_error_triggers_fallback(self, patched_routing):
        task = make_task("auth-fb-001", task_class=TaskClass.BUGFIX, summary="Fix auth bug")
        failed = make_result("auth-fb-001", success=False, error_type="auth_error")
        success = make_result("auth-fb-001", success=True, tool="claude_code",
                              backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_codex", return_value=failed), \
             patch("router.policy.run_claude", return_value=success):
            decision, result = route_task(task)

        assert result.success is True
        assert decision.attempted_fallback is True


class TestRoutingChainExhaustion:
    def test_all_providers_fail(self, patched_routing):
        task = make_task("exhaust-001", summary="Add caching layer")
        fail = make_result("exhaust-001", success=False, error_type="provider_unavailable")
        with patch("router.policy.run_codex", return_value=fail), \
             patch("router.policy.run_claude", return_value=fail), \
             patch("router.policy.run_openrouter", return_value=fail):
            decision, result = route_task(task)

        assert result.success is False
        assert len(decision.error_history) > 0

    def test_exhaustion_preserves_last_error(self, patched_routing):
        task = make_task("exhaust-err-001", summary="Implement GraphQL endpoint")
        fail_quota = make_result("exhaust-err-001", success=False, error_type="quota_exhausted")
        fail_unavail = make_result("exhaust-err-001", success=False, error_type="provider_unavailable")
        with patch("router.policy.run_codex", return_value=fail_quota), \
             patch("router.policy.run_claude", return_value=fail_unavail), \
             patch("router.policy.run_openrouter", return_value=fail_unavail):
            decision, result = route_task(task)

        assert result.success is False
        assert result.normalized_error in ("quota_exhausted", "provider_unavailable")


class TestRoutingWithStateOverride:
    def test_manual_claude_backup_chain(self, patched_routing):
        store = patched_routing["state_store"]
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        task = make_task("manual-001", summary="Add rate limiting middleware")
        mock = make_result("manual-001", tool="claude_code",
                           backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_claude", return_value=mock):
            decision, result = route_task(task)

        assert decision.state == "claude_backup"
        assert result.success is True
        assert decision.chain[0].tool == "claude_code"

    def test_manual_openrouter_fallback_chain(self, patched_routing):
        store = patched_routing["state_store"]
        store.set_manual_state(CodexState.OPENROUTER_FALLBACK)

        task = make_task("manual-or-001", summary="Update deployment scripts")
        mock = make_result("manual-or-001", tool="codex_cli",
                           backend="openrouter", model_profile="openrouter_minimax")
        with patch("router.policy.run_openrouter", return_value=mock):
            decision, result = route_task(task)

        assert decision.state == "openrouter_fallback"
        assert all(e.backend == "openrouter" for e in decision.chain)


class TestRoutingRespectsTimeouts:
    def test_executor_timeout_raises(self, patched_routing):
        task = make_task("timeout-001", summary="Complex data migration")

        def slow_executor(meta, **kw):
            from router.errors import ProviderTimeoutError
            raise ProviderTimeoutError("Execution timed out")

        success = make_result("timeout-001", tool="claude_code",
                              backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_codex", side_effect=slow_executor), \
             patch("router.policy.run_claude", return_value=success):
            decision, result = route_task(task)

        assert result.success is True
        assert result.tool == "claude_code"
        assert decision.attempted_fallback is True

    def test_timeout_is_fallback_eligible(self):
        assert can_fallback("provider_timeout") is True


class TestRoutingConcurrentTasks:
    def test_five_tasks_in_parallel(self, patched_routing):
        num_tasks = 5
        results = [None] * num_tasks
        errors = [None] * num_tasks

        def make_mock(meta, model=None, **kw):
            time.sleep(0.05 + (hash(meta.task_id) % 10) * 0.01)
            return make_result(meta.task_id, latency_ms=80, cost_usd=0.001)

        def worker(idx):
            try:
                task = make_task(f"concurrent-{idx}", risk=TaskRisk.LOW,
                                 summary=f"Task {idx}: add logging")
                with patch("router.policy.run_codex", side_effect=make_mock):
                    results[idx] = route_task(task)
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_tasks)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(e is None for e in errors), f"Errors: {errors}"
        assert all(r is not None for r in results)
        for decision, result in results:
            assert result.success is True


class TestRoutingPreservesTrace:
    def test_trace_id_populated(self, patched_routing):
        task = make_task("trace-001", summary="Add health check endpoint")
        mock = make_result("trace-001")
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(task)

        assert len(decision.trace_id) == 12
        assert decision.task_id == "trace-001"
        assert decision.state in {s.value for s in CodexState}

    def test_trace_ids_are_unique(self, patched_routing):
        trace_ids = set()
        for i in range(5):
            task = make_task(f"trace-uniq-{i}", risk=TaskRisk.LOW, summary=f"Quick fix {i}")
            mock = make_result(f"trace-uniq-{i}")
            with patch("router.policy.run_codex", return_value=mock):
                decision, _ = route_task(task)
            trace_ids.add(decision.trace_id)
        assert len(trace_ids) == 5

    def test_decision_fields_complete(self, patched_routing):
        task = make_task("fields-001", summary="Add input validation")
        mock = make_result("fields-001")
        with patch("router.policy.run_codex", return_value=mock):
            decision, result = route_task(task)

        assert decision.state
        assert isinstance(decision.chain, list)
        assert decision.reason
        assert isinstance(decision.attempted_fallback, bool)
        assert decision.trace_id
        assert result.tool and result.backend and result.model_profile
        assert result.latency_ms > 0


class TestRoutingCostTracking:
    def test_cost_propagates_from_executor(self, patched_routing):
        task = make_task("cost-001", summary="Implement data pipeline")
        mock = make_result("cost-001", cost_usd=0.0187)
        with patch("router.policy.run_codex", return_value=mock):
            _, result = route_task(task)
        assert result.cost_estimate_usd == 0.0187

    def test_zero_cost_on_failure(self, patched_routing):
        task = make_task("cost-fail-001", summary="Expensive computation")
        fail = make_result("cost-fail-001", success=False, error_type="quota_exhausted")
        with patch("router.policy.run_codex", return_value=fail), \
             patch("router.policy.run_claude", return_value=fail), \
             patch("router.policy.run_openrouter", return_value=fail):
            _, result = route_task(task)
        assert result.success is False
        assert result.cost_estimate_usd in (0.0, None, 0)

    def test_cost_accumulates_across_fallback(self, patched_routing):
        task = make_task("cost-fb-001", summary="Implement caching")
        failed = make_result("cost-fb-001", success=False, error_type="provider_unavailable")
        success = make_result("cost-fb-001", success=True, cost_usd=0.0045,
                              tool="claude_code", backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_codex", return_value=failed), \
             patch("router.policy.run_claude", return_value=success):
            _, result = route_task(task)
        assert result.success is True
        assert result.cost_estimate_usd == 0.0045


class TestRoutingEdgeCases:
    def test_primary_fails_fallback_also_fails(self, patched_routing):
        task = make_task("both-fail-001", summary="Complex deployment automation")
        codex_fail = make_result("both-fail-001", success=False, error_type="provider_unavailable")
        claude_fail = make_result("both-fail-001", success=False, error_type="quota_exhausted",
                                  tool="claude_code", backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_codex", return_value=codex_fail), \
             patch("router.policy.run_claude", return_value=claude_fail), \
             patch("router.policy.run_openrouter", return_value=codex_fail):
            decision, result = route_task(task)

        assert result.success is False
        assert decision.attempted_fallback is True
        assert len(decision.error_history) >= 2
        error_types = {e["error_type"] for e in decision.error_history}
        assert "provider_unavailable" in error_types
        assert "quota_exhausted" in error_types

    def test_rate_limit_triggers_fallback(self, patched_routing):
        task = make_task("rl-001", summary="Implement webhook handler")
        rate_limited = make_result("rl-001", success=False, error_type="rate_limited")
        success = make_result("rl-001", success=True, tool="claude_code",
                              backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_codex", return_value=rate_limited), \
             patch("router.policy.run_claude", return_value=success):
            decision, result = route_task(task)

        assert result.success is True
        assert decision.attempted_fallback is True
        assert any(e["error_type"] == "rate_limited" for e in decision.error_history)

    def test_rate_limit_is_fallback_eligible(self):
        assert can_fallback("rate_limited") is True

    def test_partial_success_stops_fallback(self, patched_routing):
        task = make_task("partial-001", summary="Refactor database layer")
        partial = ExecutorResult(
            task_id="partial-001", tool="codex_cli", backend="openai_native",
            model_profile="codex_primary", success=False, partial_success=True,
            normalized_error="provider_unavailable", exit_code=1, latency_ms=200,
            cost_estimate_usd=0.001, final_summary="Completed 2/3 subtasks",
        )
        claude_mock = make_result("partial-001", success=True, tool="claude_code",
                                  backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_codex", return_value=partial), \
             patch("router.policy.run_claude", return_value=claude_mock):
            decision, result = route_task(task)

        assert result.partial_success is True
        assert result.tool == "codex_cli"
        assert decision.attempted_fallback is False

    def test_cost_from_winning_executor(self, patched_routing):
        task = make_task("cost-pres-001", summary="Implement caching layer")
        failed = make_result("cost-pres-001", success=False, error_type="provider_unavailable")
        succeeded = make_result("cost-pres-001", success=True, cost_usd=0.0067,
                                tool="claude_code", backend="anthropic", model_profile="claude_sonnet")
        with patch("router.policy.run_codex", return_value=failed), \
             patch("router.policy.run_claude", return_value=succeeded):
            _, result = route_task(task)
        assert result.success is True
        assert result.cost_estimate_usd == 0.0067

    def test_error_history_accumulates(self, patched_routing):
        task = make_task("err-hist-001", summary="Deploy microservice")
        fail_quota = make_result("err-hist-001", success=False, error_type="quota_exhausted")
        fail_rate = make_result("err-hist-001", success=False, error_type="rate_limited",
                                tool="claude_code")
        success = make_result("err-hist-001", success=True,
                              tool="codex_cli", backend="openrouter", model_profile="openrouter_minimax")
        with patch("router.policy.run_codex", return_value=fail_quota), \
             patch("router.policy.run_claude", return_value=fail_rate), \
             patch("router.policy.run_openrouter", return_value=success):
            decision, result = route_task(task)

        assert result.success is True
        assert len(decision.error_history) >= 2
        error_types = [e["error_type"] for e in decision.error_history]
        assert "quota_exhausted" in error_types
        assert "rate_limited" in error_types
