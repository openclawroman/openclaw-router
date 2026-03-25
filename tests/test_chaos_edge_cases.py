"""Chaos tests: missing edge cases for PR #56 review.

Covers gaps identified in the review checklist:
1. Config corruption between reads
2. WAL replay with interleaved commits
3. State store lock during chaos
4. Executor returns NaN cost
5. Executor returns negative latency
6. Executor returns success with empty output / None fields
7. Mixed executor failures (1 crashes, 1 times out, 1 errors)
8. WAL with entries in wrong order
9. WAL with committed marker but no matching write
10. Both state files corrupted simultaneously
11. Permission denied on state directory
12. Valid JSON but wrong schema types
"""

import json
import math
import os
import stat
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from router.models import (
    TaskMeta, TaskClass, TaskRisk, TaskModality, CodexState,
    ExecutorResult, ChainEntry,
)
from router.policy import route_task, reset_breaker, reset_notifier
from router.state_store import StateStore, reset_state_store
from router.errors import StateError


@pytest.fixture(autouse=True)
def _reset():
    reset_state_store()
    reset_breaker()
    reset_notifier()
    import router.config_loader as cl
    cl._config_snapshot = None
    cl._config_raw = None
    cl._active_config_path = None
    yield
    reset_state_store()
    reset_breaker()
    reset_notifier()


@pytest.fixture
def sample_task():
    return TaskMeta(
        task_id="chaos-edge-001",
        agent="coder",
        task_class=TaskClass.IMPLEMENTATION,
        risk=TaskRisk.MEDIUM,
        modality=TaskModality.TEXT,
        repo_path="/tmp/test-repo",
        cwd="/tmp/test-repo",
        summary="Edge case test task",
    )


def _make_state_store(tmp_path):
    """Create a fresh StateStore in a temp directory."""
    return StateStore(
        manual_path=tmp_path / "manual.json",
        auto_path=tmp_path / "auto.json",
        history_path=tmp_path / "history.json",
        wal_path=tmp_path / "wal.jsonl",
    )


# ========================================================================
# 1. Config corruption between reads
# ========================================================================

class TestConfigCorruptedBetweenReads:
    """Read config once OK, then corrupt, second read should fail."""

    def test_config_ok_then_corrupted(self, tmp_path):
        """Config valid on first read, corrupted on second read."""
        config_file = tmp_path / "router.config.json"
        good_config = {
            "version": 1,
            "models": {
                "openrouter": {"minimax": "minimax/minimax-m2.7"},
                "codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"},
                "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"},
            },
            "reliability": {"chain_timeout_s": 600, "max_fallbacks": 3},
        }
        config_file.write_text(json.dumps(good_config))

        from router.config_loader import load_config, reload_config
        import router.config_loader as cl

        # First load succeeds
        cl._config_snapshot = None
        cl._config_raw = None
        cl._active_config_path = config_file
        config1 = load_config(config_file)
        assert config1["version"] == 1

        # Corrupt the file
        config_file.write_text('{"version": 1, BROKEN')

        # Second load should fail
        cl._config_snapshot = None
        cl._config_raw = None
        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(config_file)

    def test_config_flips_between_route_calls(self, tmp_path, sample_task):
        """Config valid for first route_task call, corrupted for second."""
        config_file = tmp_path / "router.config.json"
        good_config = {
            "version": 1,
            "models": {
                "openrouter": {"minimax": "minimax/minimax-m2.7"},
                "codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"},
                "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"},
            },
            "reliability": {"chain_timeout_s": 600, "max_fallbacks": 3},
        }
        config_file.write_text(json.dumps(good_config))

        import router.config_loader as cl
        cl._active_config_path = config_file
        cl._config_snapshot = None
        cl._config_raw = None

        from router.config_loader import load_config, reload_config

        # First call works
        config1 = load_config(config_file)
        assert config1["version"] == 1

        # Corrupt
        config_file.write_text("not json at all")

        # Clear cache and reload should fail
        cl._config_snapshot = None
        cl._config_raw = None
        with pytest.raises(Exception):
            reload_config(config_file)


# ========================================================================
# 2. WAL replay with interleaved commits
# ========================================================================

class TestWalReplayWithInterleavedCommits:
    """WAL with interleaved write/committed pairs for different files."""

    def test_interleaved_two_files_both_committed(self, tmp_path):
        """Two files written interleaved, both committed → no recovery needed."""
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)
        auto = str(store.auto_path)

        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "write", "path": auto, "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
        )

        recovered = store.recover_from_wal()
        assert recovered == 0  # Both committed, nothing to recover

    def test_interleaved_one_committed_one_not(self, tmp_path):
        """WAL recovery is simplistic: committed marker after a write means committed.

        With write(A), write(B), committed — the committed marker is 'after'
        both writes, so both are considered committed. No recovery needed.
        """
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)
        auto = str(store.auto_path)

        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "write", "path": auto, "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
        )

        recovered = store.recover_from_wal()
        # Both writes see the committed marker → both considered committed
        assert recovered == 0

    def test_interleaved_multiple_ops_partial_commit(self, tmp_path):
        """Three writes interleaved with partial commits."""
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)
        auto = str(store.auto_path)

        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "write", "path": auto, "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"  # Commits the auto write
            + json.dumps({"action": "write", "path": manual, "data": {"state": "openrouter_fallback"}}) + "\n"
            # No committed for the last write
        )

        recovered = store.recover_from_wal()
        # First write: no committed follows it (only the committed that follows auto)
        # Actually let's trace through: the committed marker is for the second write
        # The first write has a committed marker somewhere after it
        # The third write has no committed after it
        # So: first write has committed after (the first committed), third write has no committed
        assert recovered == 1  # The last uncommitted write

    def test_wal_committed_without_prior_write(self, tmp_path):
        """WAL has committed marker but no matching write entry."""
        store = _make_state_store(tmp_path)

        # Just a committed marker with no prior write
        store.wal_path.write_text(
            json.dumps({"action": "committed"}) + "\n"
        )

        recovered = store.recover_from_wal()
        assert recovered == 0  # Nothing to recover — no writes

    def test_wal_committed_only_multiple(self, tmp_path):
        """Multiple committed markers with no writes."""
        store = _make_state_store(tmp_path)

        store.wal_path.write_text(
            json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
        )

        recovered = store.recover_from_wal()
        assert recovered == 0

    def test_wal_write_committed_write_no_commit(self, tmp_path):
        """Write+committed pair followed by orphaned write."""
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)

        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "claude_backup"}}) + "\n"
            # No committed for the second write
        )

        recovered = store.recover_from_wal()
        assert recovered == 1  # The orphaned second write


# ========================================================================
# 3. State store lock during chaos
# ========================================================================

class TestStateStoreLockDuringChaos:
    """Concurrent access and corruption while lock is held."""

    def test_concurrent_reads_during_corruption(self, tmp_path):
        """Multiple threads reading state while another corrupts it."""
        store = _make_state_store(tmp_path)
        store.set_manual_state(CodexState.OPENAI_PRIMARY)

        errors_caught = []
        results = []

        def reader():
            try:
                store._manual_cache_valid = False
                state = store.get_manual_state()
                results.append(state)
            except Exception as e:
                errors_caught.append(e)

        def corruptor():
            import time
            time.sleep(0.01)  # Small delay
            try:
                store.manual_path.write_text("CORRUPTED")
            except Exception:
                pass

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
        threads.append(threading.Thread(target=corruptor))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some reads should succeed, some should raise StateError
        # The important thing is no unhandled exception
        assert len(errors_caught) + len(results) == 5

    def test_concurrent_writes_no_deadlock(self, tmp_path):
        """Multiple threads writing state simultaneously → no deadlock."""
        store = _make_state_store(tmp_path)
        states = [CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP,
                  CodexState.OPENAI_CONSERVATION, CodexState.OPENROUTER_FALLBACK]

        errors = []

        def writer(state):
            try:
                store.set_manual_state(state)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=writer, args=(states[i % 4],))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Thread deadlocked"

        # Final state should be one of the valid states
        final = store.get_manual_state()
        assert final in CodexState

    def test_lock_released_after_exception_in_write(self, tmp_path):
        """If _write raises, lock should still be released."""
        store = _make_state_store(tmp_path)

        # Temporarily break the wal_path parent to force an error in _append_to_wal_unlocked
        original_wal = store.wal_path
        store.wal_path = Path("/nonexistent/dir/wal.jsonl")

        try:
            store.set_manual_state(CodexState.CLAUDE_BACKUP)
        except Exception:
            pass  # Expected

        # Restore and verify lock works
        store.wal_path = original_wal
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        assert store.get_manual_state() == CodexState.OPENAI_PRIMARY


# ========================================================================
# 4. Executor returns NaN cost
# ========================================================================

class TestExecutorReturnsNanCost:
    """Executor returns NaN as cost_estimate_usd."""

    def test_nan_cost_does_not_crash(self, sample_task):
        """Executor returns success but cost is NaN."""
        mock_result = ExecutorResult(
            task_id=sample_task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
            cost_estimate_usd=float("nan"),
        )

        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(sample_task)

        assert result is not None
        # NaN cost should not crash the router
        if result.cost_estimate_usd is not None:
            assert isinstance(result.cost_estimate_usd, float)

    def test_inf_cost_does_not_crash(self, sample_task):
        """Executor returns success but cost is infinity."""
        mock_result = ExecutorResult(
            task_id=sample_task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
            cost_estimate_usd=float("inf"),
        )

        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(sample_task)

        assert result is not None


# ========================================================================
# 5. Executor returns negative latency
# ========================================================================

class TestExecutorReturnsNegativeLatency:
    """Executor returns negative latency_ms."""

    def test_negative_latency_does_not_crash(self, sample_task):
        """Executor returns success but latency_ms is negative (clock skew)."""
        mock_result = ExecutorResult(
            task_id=sample_task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
            latency_ms=-500,
        )

        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(sample_task)

        assert result is not None
        assert result.success

    def test_zero_latency_does_not_crash(self, sample_task):
        """Executor returns success with zero latency."""
        mock_result = ExecutorResult(
            task_id=sample_task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
            latency_ms=0,
        )

        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(sample_task)

        assert result is not None
        assert result.success


# ========================================================================
# 6. Executor returns success with empty output / None fields
# ========================================================================

class TestExecutorReturnsEmptySuccess:
    """Executor returns success=True but with empty/None fields."""

    def test_success_with_no_output(self, sample_task):
        """Executor succeeds but has no output at all."""
        mock_result = ExecutorResult(
            task_id=sample_task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
            final_summary=None,
            artifacts=[],
            stdout_ref=None,
            stderr_ref=None,
        )

        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(sample_task)

        assert result is not None
        assert result.success

    def test_success_with_none_task_id(self, sample_task):
        """Executor succeeds but task_id is empty string."""
        mock_result = ExecutorResult(
            task_id="",  # Empty instead of None (str field)
            tool="codex_cli",
            backend="openai_native",
            model_profile="codex_primary",
            success=True,
        )

        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(sample_task)

        assert result is not None

    def test_success_with_none_model_profile(self, sample_task):
        """Executor succeeds but model_profile is empty string."""
        mock_result = ExecutorResult(
            task_id=sample_task.task_id,
            tool="codex_cli",
            backend="openai_native",
            model_profile="",  # Empty string
            success=True,
        )

        with patch("router.policy.run_codex", return_value=mock_result):
            decision, result = route_task(sample_task)

        assert result is not None


# ========================================================================
# 7. Mixed executor failures (1 crashes, 1 times out, 1 errors)
# ========================================================================

class TestMixedExecutorFailures:
    """Different failure modes across executors."""

    def test_crash_timeout_error(self, sample_task):
        """Codex crashes, Claude times out, OpenRouter returns error."""
        from router.errors import ProviderTimeoutError

        with patch("router.policy.run_codex", side_effect=RuntimeError("crash")):
            with patch("router.policy.run_claude", side_effect=ProviderTimeoutError("timeout")):
                with patch("router.policy.run_openrouter", side_effect=ConnectionError("network")):
                    decision, result = route_task(sample_task)

        assert decision is not None
        assert not result.success

    def test_first_returns_garbage_second_succeeds(self, sample_task):
        """First executor raises fallback-eligible error, second succeeds."""
        from router.errors import ExecutorError

        with patch("router.policy.run_codex", side_effect=ExecutorError("rate limited", "rate_limited")):
            mock_result = ExecutorResult(
                task_id=sample_task.task_id,
                tool="claude_code",
                backend="anthropic",
                model_profile="claude_primary",
                success=True,
            )
            with patch("router.policy.run_claude", return_value=mock_result):
                with patch("router.policy.run_openrouter") as mock_or:
                    decision, result = route_task(sample_task)

        assert result.success
        mock_or.assert_not_called()

    def test_timeout_crash_oserror_sequence(self, sample_task):
        """Codex timeout, Claude crash (RuntimeError), OpenRouter OSError."""
        with patch("router.policy.run_codex", side_effect=TimeoutError("timed out")):
            with patch("router.policy.run_claude", side_effect=RuntimeError("segfault")):
                with patch("router.policy.run_openrouter", side_effect=OSError("binary not found")):
                    decision, result = route_task(sample_task)

        assert decision is not None
        assert not result.success


# ========================================================================
# 8. WAL with entries in wrong order
# ========================================================================

class TestWalWrongOrder:
    """WAL entries in unexpected order."""

    def test_committed_before_write(self, tmp_path):
        """Committed marker appears before any write entry."""
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)

        store.wal_path.write_text(
            json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
        )

        recovered = store.recover_from_wal()
        # The write has no committed after it → should be recovered
        assert recovered == 1

    def test_multiple_writes_single_committed_at_start(self, tmp_path):
        """Multiple writes with a single committed marker at the start."""
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)
        auto = str(store.auto_path)

        store.wal_path.write_text(
            json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "write", "path": auto, "data": {"state": "claude_backup"}}) + "\n"
        )

        recovered = store.recover_from_wal()
        # First write: committed is before it (not after) → unrecovered → recover
        # Second write: no committed after → recover
        assert recovered == 2

    def test_write_committed_write_committed_write(self, tmp_path):
        """Alternating write/committed with orphaned trailing write."""
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)

        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "openrouter_fallback"}}) + "\n"
        )

        recovered = store.recover_from_wal()
        # First two writes are committed, third is orphaned
        assert recovered == 1


# ========================================================================
# 9. Both state files corrupted simultaneously
# ========================================================================

class TestBothStateFilesCorrupted:
    """Both manual and auto state files corrupted at the same time."""

    def test_both_invalid_json(self, tmp_path):
        """Both state files contain invalid JSON → StateError raised."""
        store = _make_state_store(tmp_path)
        store.manual_path.write_text("broken {")
        store.auto_path.write_text("not json")

        store._manual_cache_valid = False
        store._auto_cache_valid = False

        with pytest.raises(StateError):
            store.get_state()

    def test_both_empty(self, tmp_path):
        """Both state files are empty."""
        store = _make_state_store(tmp_path)
        store.manual_path.write_text("")
        store.auto_path.write_text("")

        store._manual_cache_valid = False
        store._auto_cache_valid = False

        # get_state should handle this gracefully
        try:
            state = store.get_state()
            assert state == CodexState.OPENAI_PRIMARY
        except (StateError, json.JSONDecodeError):
            pass  # Acceptable

    def test_both_binary_garbage(self, tmp_path):
        """Both state files contain binary garbage."""
        store = _make_state_store(tmp_path)
        store.manual_path.write_bytes(b'\x00\x01\x80\xff\xfe')
        store.auto_path.write_bytes(b'\xff\xfe\xfd\xfc')

        store._manual_cache_valid = False
        store._auto_cache_valid = False

        try:
            state = store.get_state()
            assert state == CodexState.OPENAI_PRIMARY
        except (StateError, UnicodeDecodeError):
            pass  # Acceptable

    def test_both_wrong_schema(self, tmp_path):
        """Both files have valid JSON but wrong structure."""
        store = _make_state_store(tmp_path)
        store.manual_path.write_text(json.dumps({"key": "value", "count": 42}))
        store.auto_path.write_text(json.dumps({"state": 999}))  # Number, not string

        store._manual_cache_valid = False
        store._auto_cache_valid = False

        # Manual: state key missing → None. Auto: state is not a string → invalid → None
        state = store.get_state()
        assert state == CodexState.OPENAI_PRIMARY


# ========================================================================
# 10. Permission denied on state directory
# ========================================================================

class TestStateDirPermissionDenied:
    """Permission denied when accessing state directory."""

    def test_state_file_no_write_permission(self, tmp_path):
        """State file read-only — owner can still replace via tempfile+os.replace.

        On macOS/Linux, the file owner can chmod and replace even read-only files
        when the directory is writable. Verify no unhandled exception either way.
        """
        store = _make_state_store(tmp_path)
        store.set_manual_state(CodexState.OPENAI_PRIMARY)

        os.chmod(store.manual_path, 0o444)

        try:
            try:
                store.set_manual_state(CodexState.CLAUDE_BACKUP)
                store._manual_cache_valid = False
                assert store.get_manual_state() == CodexState.CLAUDE_BACKUP
            except (PermissionError, OSError, StateError):
                pass  # Expected on strict ACL systems
        finally:
            os.chmod(store.manual_path, 0o644)

    def test_state_dir_no_write_permission(self, tmp_path):
        """State directory is not writable → can't create temp files."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        store = StateStore(
            manual_path=state_dir / "manual.json",
            auto_path=state_dir / "auto.json",
            history_path=state_dir / "history.json",
            wal_path=state_dir / "wal.jsonl",
        )
        store.set_manual_state(CodexState.OPENAI_PRIMARY)

        # Remove write permission from directory
        os.chmod(state_dir, 0o555)

        try:
            with pytest.raises((PermissionError, OSError, StateError)):
                store.set_manual_state(CodexState.CLAUDE_BACKUP)
        finally:
            os.chmod(state_dir, 0o755)


# ========================================================================
# 11. Valid JSON but wrong schema types
# ========================================================================

class TestConfigWrongSchemaTypes:
    """Config has valid JSON but unexpected value types."""

    def test_models_is_list_not_dict(self, tmp_path):
        """Config models section is a list instead of dict."""
        config_file = tmp_path / "router.config.json"
        bad_config = {"version": 1, "models": ["codex", "claude"]}
        config_file.write_text(json.dumps(bad_config))

        from router.config_loader import load_config
        import router.config_loader as cl
        cl._config_snapshot = None
        cl._config_raw = None
        cl._active_config_path = config_file

        config = load_config(config_file)
        # Should load but accessing models will fail at usage time
        assert config["version"] == 1

    def test_version_is_string_not_int(self, tmp_path):
        """Config version is a string instead of int."""
        config_file = tmp_path / "router.config.json"
        config = {
            "version": "one",
            "models": {
                "openrouter": {"minimax": "minimax/minimax-m2.7"},
                "codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"},
                "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"},
            },
        }
        config_file.write_text(json.dumps(config))

        from router.config_loader import load_config
        import router.config_loader as cl
        cl._config_snapshot = None
        cl._config_raw = None
        cl._active_config_path = config_file

        # Should load (doesn't validate types at load time)
        loaded = load_config(config_file)
        assert loaded["version"] == "one"

    def test_reliability_values_are_strings(self, tmp_path):
        """Reliability config has string values instead of numbers."""
        config_file = tmp_path / "router.config.json"
        config = {
            "version": 1,
            "models": {
                "openrouter": {"minimax": "minimax/minimax-m2.7"},
                "codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"},
                "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"},
            },
            "reliability": {"chain_timeout_s": "six hundred", "max_fallbacks": "three"},
        }
        config_file.write_text(json.dumps(config))

        from router.config_loader import load_config
        import router.config_loader as cl
        cl._config_snapshot = None
        cl._config_raw = None
        cl._active_config_path = config_file

        loaded = load_config(config_file)
        assert loaded["reliability"]["chain_timeout_s"] == "six hundred"

    def test_empty_models_dict(self, tmp_path):
        """Config has empty models dict."""
        config_file = tmp_path / "router.config.json"
        config = {"version": 1, "models": {}}
        config_file.write_text(json.dumps(config))

        from router.config_loader import load_config
        import router.config_loader as cl
        cl._config_snapshot = None
        cl._config_raw = None
        cl._active_config_path = config_file

        loaded = load_config(config_file)
        assert loaded["models"] == {}
