"""Chaos tests: edge cases for PR #56 review.

Covers gaps:
1. Config corruption between reads
2. WAL replay with interleaved commits
3. State store lock during chaos
4. Executor returns NaN/inf cost
5. Executor returns negative/zero latency
6. Executor returns success with empty/None fields
7. Mixed executor failures
8. WAL entries in wrong order
9. Both state files corrupted simultaneously
10. Permission denied on state directory
11. Valid JSON but wrong schema types
"""

import json
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


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset():
    reset_state_store(); reset_breaker(); reset_notifier()
    import router.config_loader as cl
    cl._config_snapshot = cl._config_raw = cl._active_config_path = None
    yield
    reset_state_store(); reset_breaker(); reset_notifier()


@pytest.fixture
def sample_task():
    return TaskMeta(task_id="chaos-edge-001", agent="coder",
                    task_class=TaskClass.IMPLEMENTATION, risk=TaskRisk.MEDIUM,
                    modality=TaskModality.TEXT, repo_path="/tmp/test-repo",
                    cwd="/tmp/test-repo", summary="Edge case test task")


def _make_state_store(tmp_path):
    return StateStore(manual_path=tmp_path / "manual.json",
                      auto_path=tmp_path / "auto.json",
                      history_path=tmp_path / "history.json",
                      wal_path=tmp_path / "wal.jsonl")


GOOD_CONFIG = {
    "version": 1,
    "models": {
        "openrouter": {"minimax": "minimax/minimax-m2.7"},
        "codex": {"default": "codex-default", "gpt54": "gpt-5.4", "gpt54_mini": "gpt-5.4-mini"},
        "claude": {"default": "claude-default", "sonnet": "claude-sonnet-4.6", "opus": "claude-opus-4.6"},
    },
    "reliability": {"chain_timeout_s": 600, "max_fallbacks": 3},
}


def _assert_executor_survives(sample_task, **result_kwargs):
    """Helper: patch run_codex with an ExecutorResult and verify route_task doesn't crash."""
    mock_result = ExecutorResult(
        task_id=result_kwargs.pop("task_id", sample_task.task_id),
        tool=result_kwargs.pop("tool", "codex_cli"),
        backend=result_kwargs.pop("backend", "openai_native"),
        model_profile=result_kwargs.pop("model_profile", "codex_primary"),
        success=result_kwargs.pop("success", True),
        **result_kwargs,
    )
    with patch("router.policy.run_codex", return_value=mock_result):
        decision, result = route_task(sample_task)
    assert result is not None
    return decision, result


# ========================================================================
# 1. Config corruption between reads
# ========================================================================

class TestConfigCorruptedBetweenReads:

    def test_config_ok_then_corrupted(self, tmp_path):
        cf = tmp_path / "router.config.json"
        cf.write_text(json.dumps(GOOD_CONFIG))
        import router.config_loader as cl
        from router.config_loader import load_config
        cl._config_snapshot = cl._config_raw = None
        cl._active_config_path = cf
        config1 = load_config(cf)
        assert config1["version"] == 1
        cf.write_text('{"version": 1, BROKEN')
        cl._config_snapshot = cl._config_raw = None
        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(cf)

    def test_config_flips_between_route_calls(self, tmp_path, sample_task):
        cf = tmp_path / "router.config.json"
        cf.write_text(json.dumps(GOOD_CONFIG))
        import router.config_loader as cl
        from router.config_loader import load_config, reload_config
        cl._active_config_path = cf
        cl._config_snapshot = cl._config_raw = None
        assert load_config(cf)["version"] == 1
        cf.write_text("not json at all")
        cl._config_snapshot = cl._config_raw = None
        with pytest.raises(Exception):
            reload_config(cf)


# ========================================================================
# 2. WAL replay with interleaved commits
# ========================================================================

class TestWalReplayWithInterleavedCommits:

    def test_both_writes_committed(self, tmp_path):
        store = _make_state_store(tmp_path)
        manual, auto = str(store.manual_path), str(store.auto_path)
        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "write", "path": auto, "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
        )
        assert store.recover_from_wal() == 0

    def test_committed_without_prior_write(self, tmp_path):
        store = _make_state_store(tmp_path)
        store.wal_path.write_text(json.dumps({"action": "committed"}) + "\n")
        assert store.recover_from_wal() == 0

    def test_multiple_committed_no_writes(self, tmp_path):
        store = _make_state_store(tmp_path)
        store.wal_path.write_text(
            json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
        )
        assert store.recover_from_wal() == 0

    def test_interleaved_one_committed_one_not(self, tmp_path):
        store = _make_state_store(tmp_path)
        manual, auto = str(store.manual_path), str(store.auto_path)
        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "write", "path": auto, "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
        )
        assert store.recover_from_wal() == 0

    def test_interleaved_multiple_ops_partial_commit(self, tmp_path):
        store = _make_state_store(tmp_path)
        manual, auto = str(store.manual_path), str(store.auto_path)
        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "write", "path": auto, "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "openrouter_fallback"}}) + "\n"
        )
        assert store.recover_from_wal() == 1

    def test_wal_write_committed_orphaned_write(self, tmp_path):
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)
        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "claude_backup"}}) + "\n"
        )
        assert store.recover_from_wal() == 1


# ========================================================================
# 3. State store lock during chaos
# ========================================================================

class TestStateStoreLockDuringChaos:

    def test_concurrent_reads_during_corruption(self, tmp_path):
        store = _make_state_store(tmp_path)
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        errors_caught, results = [], []
        def reader():
            try:
                store._manual_cache_valid = False
                results.append(store.get_manual_state())
            except Exception as e:
                errors_caught.append(e)
        def corruptor():
            import time; time.sleep(0.01)
            try: store.manual_path.write_text("CORRUPTED")
            except: pass
        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads.append(threading.Thread(target=corruptor))
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors_caught) + len(results) == 5

    def test_concurrent_writes_no_deadlock(self, tmp_path):
        store = _make_state_store(tmp_path)
        states = [CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP,
                  CodexState.OPENAI_CONSERVATION, CodexState.OPENROUTER_FALLBACK]
        errors = []
        def writer(state):
            try: store.set_manual_state(state)
            except Exception as e: errors.append(e)
        threads = [threading.Thread(target=writer, args=(states[i % 4],)) for i in range(10)]
        for t in threads: t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Thread deadlocked"
        assert store.get_manual_state() in CodexState

    def test_lock_released_after_exception(self, tmp_path):
        store = _make_state_store(tmp_path)
        original_wal = store.wal_path
        store.wal_path = Path("/nonexistent/dir/wal.jsonl")
        try:
            store.set_manual_state(CodexState.CLAUDE_BACKUP)
        except Exception:
            pass
        store.wal_path = original_wal
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        assert store.get_manual_state() == CodexState.OPENAI_PRIMARY


# ========================================================================
# 4-6. Executor returns weird values (NaN cost, negative latency, empty output)
# ========================================================================

class TestExecutorWeirdValues:
    """Executor returns unusual but type-correct values — router must not crash."""

    @pytest.mark.parametrize("kwargs", [
        {"cost_estimate_usd": float("nan")},
        {"cost_estimate_usd": float("inf")},
        {"latency_ms": -500},
        {"latency_ms": 0},
        {"final_summary": None, "artifacts": [], "stdout_ref": None, "stderr_ref": None},
        {"task_id": ""},  # empty task_id
        {"model_profile": ""},  # empty model_profile
    ])
    def test_executor_survives_weird_values(self, sample_task, kwargs):
        _, result = _assert_executor_survives(sample_task, **kwargs)
        if "cost_estimate_usd" in kwargs and result.cost_estimate_usd is not None:
            assert isinstance(result.cost_estimate_usd, float)


# ========================================================================
# 7. Mixed executor failures
# ========================================================================

class TestMixedExecutorFailures:

    def test_crash_timeout_error_sequence(self, sample_task):
        from router.errors import ProviderTimeoutError
        with patch("router.policy.run_codex", side_effect=RuntimeError("crash")):
            with patch("router.policy.run_claude", side_effect=ProviderTimeoutError("timeout")):
                with patch("router.policy.run_openrouter", side_effect=ConnectionError("network")):
                    decision, result = route_task(sample_task)
        assert decision is not None and not result.success

    def test_first_errors_second_succeeds(self, sample_task):
        from router.errors import ExecutorError
        with patch("router.policy.run_codex", side_effect=ExecutorError("rate limited", "rate_limited")):
            mock_result = ExecutorResult(task_id=sample_task.task_id, tool="claude_code",
                                         backend="anthropic", model_profile="claude_primary", success=True)
            with patch("router.policy.run_claude", return_value=mock_result):
                with patch("router.policy.run_openrouter") as mock_or:
                    decision, result = route_task(sample_task)
        assert result.success
        mock_or.assert_not_called()

    def test_timeout_crash_oserror(self, sample_task):
        with patch("router.policy.run_codex", side_effect=TimeoutError("timed out")):
            with patch("router.policy.run_claude", side_effect=RuntimeError("segfault")):
                with patch("router.policy.run_openrouter", side_effect=OSError("binary not found")):
                    decision, result = route_task(sample_task)
        assert decision is not None and not result.success


# ========================================================================
# 8. WAL entries in wrong order
# ========================================================================

class TestWalWrongOrder:

    def test_committed_before_write(self, tmp_path):
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)
        store.wal_path.write_text(
            json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
        )
        # Committed marker is BEFORE the write (not after) → write is uncommitted
        assert store.recover_from_wal() == 1

    def test_committed_at_start_two_writes(self, tmp_path):
        store = _make_state_store(tmp_path)
        manual, auto = str(store.manual_path), str(store.auto_path)
        store.wal_path.write_text(
            json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "write", "path": auto, "data": {"state": "claude_backup"}}) + "\n"
        )
        assert store.recover_from_wal() == 2

    def test_alternating_write_commit_orphaned_trailing(self, tmp_path):
        store = _make_state_store(tmp_path)
        manual = str(store.manual_path)
        store.wal_path.write_text(
            json.dumps({"action": "write", "path": manual, "data": {"state": "openai_primary"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "claude_backup"}}) + "\n"
            + json.dumps({"action": "committed"}) + "\n"
            + json.dumps({"action": "write", "path": manual, "data": {"state": "openrouter_fallback"}}) + "\n"
        )
        assert store.recover_from_wal() == 1


# ========================================================================
# 9. Both state files corrupted simultaneously
# ========================================================================

class TestBothStateFilesCorrupted:

    @pytest.mark.parametrize("manual_content,auto_content,expect_default", [
        ("broken {", "not json", True),
        ("", "", True),
        (json.dumps({"key": "value"}), json.dumps({"state": 999}), True),
    ])
    def test_corrupted_pairs(self, tmp_path, manual_content, auto_content, expect_default):
        store = _make_state_store(tmp_path)
        store.manual_path.write_text(manual_content)
        store.auto_path.write_text(auto_content)
        store._manual_cache_valid = store._auto_cache_valid = False
        try:
            state = store.get_state()
            if expect_default:
                assert state == CodexState.OPENAI_PRIMARY
        except (StateError, json.JSONDecodeError, UnicodeDecodeError):
            pass

    def test_both_binary_garbage(self, tmp_path):
        store = _make_state_store(tmp_path)
        store.manual_path.write_bytes(b'\x00\x01\x80\xff\xfe')
        store.auto_path.write_bytes(b'\xff\xfe\xfd\xfc')
        store._manual_cache_valid = store._auto_cache_valid = False
        try:
            state = store.get_state()
            assert state == CodexState.OPENAI_PRIMARY
        except (StateError, UnicodeDecodeError):
            pass


# ========================================================================
# 10. Permission denied on state directory
# ========================================================================

class TestStateDirPermissionDenied:

    def test_state_file_no_write_permission(self, tmp_path):
        store = _make_state_store(tmp_path)
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
        os.chmod(store.manual_path, 0o444)
        try:
            try:
                store.set_manual_state(CodexState.CLAUDE_BACKUP)
                store._manual_cache_valid = False
                assert store.get_manual_state() == CodexState.CLAUDE_BACKUP
            except (PermissionError, OSError, StateError):
                pass
        finally:
            os.chmod(store.manual_path, 0o644)

    def test_state_dir_no_write_permission(self, tmp_path):
        state_dir = tmp_path / "state"; state_dir.mkdir()
        store = StateStore(manual_path=state_dir / "manual.json",
                           auto_path=state_dir / "auto.json",
                           history_path=state_dir / "history.json",
                           wal_path=state_dir / "wal.jsonl")
        store.set_manual_state(CodexState.OPENAI_PRIMARY)
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

    @pytest.mark.parametrize("config_override", [
        {"models": ["codex", "claude"]},  # list not dict
        {"version": "one"},  # string not int
        {"reliability": {"chain_timeout_s": "six hundred", "max_fallbacks": "three"}},
        {"models": {}},  # empty dict
    ])
    def test_wrong_schema_types(self, tmp_path, config_override):
        cf = tmp_path / "router.config.json"
        config = {**GOOD_CONFIG, **config_override}
        cf.write_text(json.dumps(config))
        from router.config_loader import load_config
        import router.config_loader as cl
        cl._config_snapshot = cl._config_raw = None
        cl._active_config_path = cf
        loaded = load_config(cf)
        assert loaded is not None
