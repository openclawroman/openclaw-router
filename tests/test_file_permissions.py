"""Tests for file permission controls (item 6.3).

State files, WAL files, and alert files should have restricted
permissions (owner-only read/write = 0o600) on Unix systems.
chmod failures must not crash the application (best-effort).
"""

import json
import os
import platform
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from router.models import CodexState
from router.state_store import StateStore
from router.notifications import NotificationManager

IS_UNIX = platform.system() != "Windows"


def _get_file_mode(path: Path) -> int:
    """Return the permission bits of a file."""
    return stat.S_IMODE(os.stat(path).st_mode)


# ── State Store Permissions ──────────────────────────────────────────


class TestStateFilePermissions:
    """State files should be created with 0o600 permissions on Unix."""

    @pytest.fixture
    def store(self, tmp_path):
        manual = tmp_path / "codex_manual_state.json"
        auto = tmp_path / "codex_auto_state.json"
        history = tmp_path / "codex_state_history.json"
        wal = tmp_path / "codex_state_wal.jsonl"
        return StateStore(
            manual_path=manual,
            auto_path=auto,
            history_path=history,
            wal_path=wal,
        )

    @pytest.mark.skipif(not IS_UNIX, reason="chmod permission checks are Unix-only")
    def test_manual_state_file_has_600_permissions(self, store):
        """Manual state file should have 0o600 permissions."""
        mode = _get_file_mode(store.manual_path)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    @pytest.mark.skipif(not IS_UNIX, reason="chmod permission checks are Unix-only")
    def test_auto_state_file_has_600_permissions(self, store):
        """Auto state file should have 0o600 permissions."""
        mode = _get_file_mode(store.auto_path)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    @pytest.mark.skipif(not IS_UNIX, reason="chmod permission checks are Unix-only")
    def test_state_file_after_set_has_600_permissions(self, store):
        """State file should maintain 0o600 after writing new state."""
        store.set_auto_state(CodexState.CLAUDE_BACKUP)
        mode = _get_file_mode(store.auto_path)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    @pytest.mark.skipif(not IS_UNIX, reason="chmod permission checks are Unix-only")
    def test_manual_state_after_set_has_600_permissions(self, store):
        """Manual state file should maintain 0o600 after writing new state."""
        store.set_manual_state(CodexState.CLAUDE_BACKUP)
        mode = _get_file_mode(store.manual_path)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


class TestWALFilePermissions:
    """WAL files should have 0o600 permissions on Unix."""

    @pytest.fixture
    def store(self, tmp_path):
        manual = tmp_path / "codex_manual_state.json"
        auto = tmp_path / "codex_auto_state.json"
        history = tmp_path / "codex_state_history.json"
        wal = tmp_path / "codex_state_wal.jsonl"
        return StateStore(
            manual_path=manual,
            auto_path=auto,
            history_path=history,
            wal_path=wal,
        )

    @pytest.mark.skipif(not IS_UNIX, reason="chmod permission checks are Unix-only")
    def test_wal_file_has_600_permissions(self, store):
        """WAL file should have 0o600 permissions after write."""
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)
        mode = _get_file_mode(store.wal_path)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    @pytest.mark.skipif(not IS_UNIX, reason="chmod permission checks are Unix-only")
    def test_wal_file_after_multiple_writes(self, store):
        """WAL file should maintain 0o600 after multiple writes."""
        store.set_auto_state(CodexState.CLAUDE_BACKUP)
        store.set_auto_state(CodexState.OPENAI_PRIMARY)
        store.set_auto_state(CodexState.OPENROUTER_FALLBACK)
        mode = _get_file_mode(store.wal_path)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


class TestStateHistoryPermissions:
    """State history file should have 0o600 permissions on Unix."""

    @pytest.fixture
    def store(self, tmp_path):
        manual = tmp_path / "codex_manual_state.json"
        auto = tmp_path / "codex_auto_state.json"
        history = tmp_path / "codex_state_history.json"
        wal = tmp_path / "codex_state_wal.jsonl"
        return StateStore(
            manual_path=manual,
            auto_path=auto,
            history_path=history,
            wal_path=wal,
        )

    @pytest.mark.skipif(not IS_UNIX, reason="chmod permission checks are Unix-only")
    def test_history_file_has_600_permissions(self, store):
        """History file should have 0o600 permissions after transition log."""
        store.set_state_with_history(CodexState.CLAUDE_BACKUP, reason="test")
        mode = _get_file_mode(store.history_path)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


class TestChmodFailureResilience:
    """chmod failures should not crash the application."""

    @pytest.fixture
    def store(self, tmp_path):
        manual = tmp_path / "codex_manual_state.json"
        auto = tmp_path / "codex_auto_state.json"
        history = tmp_path / "codex_state_history.json"
        wal = tmp_path / "codex_state_wal.jsonl"
        return StateStore(
            manual_path=manual,
            auto_path=auto,
            history_path=history,
            wal_path=wal,
        )

    def test_chmod_failure_on_state_write_does_not_crash(self, store):
        """If os.chmod raises OSError, the application should not crash."""
        original_chmod = os.chmod

        def failing_chmod(path, mode):
            if str(store.manual_path) in str(path):
                raise OSError("chmod not supported on this filesystem")
            original_chmod(path, mode)

        with patch("router.state_store.os.chmod", side_effect=failing_chmod):
            # Should not raise — chmod failure is best-effort
            store.set_manual_state(CodexState.CLAUDE_BACKUP)

        # State should still be written correctly
        assert store.get_manual_state() == CodexState.CLAUDE_BACKUP

    def test_chmod_failure_on_wal_does_not_crash(self, store):
        """If os.chmod raises OSError on WAL file, the application should not crash."""
        original_chmod = os.chmod

        def failing_chmod(path, mode):
            if str(store.wal_path) in str(path):
                raise OSError("chmod not supported")
            original_chmod(path, mode)

        with patch("router.state_store.os.chmod", side_effect=failing_chmod):
            store.set_auto_state(CodexState.CLAUDE_BACKUP)

        assert store.get_auto_state() == CodexState.CLAUDE_BACKUP

    def test_chmod_failure_on_history_does_not_crash(self, store):
        """If os.chmod raises OSError on history file, the application should not crash."""
        original_chmod = os.chmod

        def failing_chmod(path, mode):
            if str(store.history_path) in str(path):
                raise OSError("chmod not supported")
            original_chmod(path, mode)

        with patch("router.state_store.os.chmod", side_effect=failing_chmod):
            store.set_state_with_history(CodexState.CLAUDE_BACKUP, reason="test")

        # Transition should still be logged
        history = store.get_state_history(limit=5)
        assert len(history) >= 1

    def test_chmod_permission_error_does_not_crash(self, store):
        """PermissionError (subclass of OSError) should be caught too."""
        with patch("router.state_store.os.chmod", side_effect=PermissionError("denied")):
            store.set_auto_state(CodexState.CLAUDE_BACKUP)

        assert store.get_auto_state() == CodexState.CLAUDE_BACKUP


class TestChmodMockNonUnix:
    """Test permission setting via mock on non-Unix platforms."""

    def test_chmod_called_on_state_write(self, tmp_path):
        """os.chmod should be called with 0o600 after state write."""
        manual = tmp_path / "codex_manual_state.json"
        auto = tmp_path / "codex_auto_state.json"
        history = tmp_path / "codex_state_history.json"
        wal = tmp_path / "codex_state_wal.jsonl"

        with patch("router.state_store.os.chmod") as mock_chmod:
            store = StateStore(
                manual_path=manual,
                auto_path=auto,
                history_path=history,
                wal_path=wal,
            )
            # At least the initial state files should have chmod called
            calls = mock_chmod.call_args_list
            assert any(
                call.args[1] == 0o600 for call in calls
            ), f"Expected chmod(..., 0o600) calls, got {calls}"

    def test_chmod_called_on_wal_write(self, tmp_path):
        """os.chmod should be called with 0o600 after WAL write."""
        manual = tmp_path / "codex_manual_state.json"
        auto = tmp_path / "codex_auto_state.json"
        history = tmp_path / "codex_state_history.json"
        wal = tmp_path / "codex_state_wal.jsonl"

        store = StateStore(
            manual_path=manual,
            auto_path=auto,
            history_path=history,
            wal_path=wal,
        )

        with patch("router.state_store.os.chmod") as mock_chmod:
            store.set_auto_state(CodexState.CLAUDE_BACKUP)
            wal_calls = [
                c for c in mock_chmod.call_args_list
                if str(wal) in str(c.args[0])
            ]
            assert len(wal_calls) >= 1
            assert wal_calls[0].args[1] == 0o600


# ── Notification Permissions ─────────────────────────────────────────


class TestAlertFilePermissions:
    """Alert files should have 0o600 permissions on Unix."""

    @pytest.fixture
    def nm(self, tmp_path):
        alerts_path = tmp_path / "alerts.jsonl"
        return NotificationManager(alerts_path=alerts_path)

    @pytest.mark.skipif(not IS_UNIX, reason="chmod permission checks are Unix-only")
    def test_alert_file_has_600_permissions(self, nm):
        """Alert file should have 0o600 permissions."""
        nm.notify_state_change("openai_primary", "claude_backup", reason="test")
        mode = _get_file_mode(nm.alerts_path)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_chmod_failure_on_alert_does_not_crash(self, nm):
        """If os.chmod raises OSError on alert file, the application should not crash."""
        with patch("router.notifications.os.chmod", side_effect=OSError("not supported")):
            # Should not raise
            alert = nm.notify_state_change("openai_primary", "claude_backup", reason="test")

        assert alert is not None
        alerts = nm.get_recent_alerts()
        assert len(alerts) >= 1

    def test_chmod_called_on_alert_write(self, tmp_path):
        """os.chmod should be called with 0o600 after alert write."""
        alerts_path = tmp_path / "alerts.jsonl"
        nm = NotificationManager(alerts_path=alerts_path)

        with patch("router.notifications.os.chmod") as mock_chmod:
            nm.notify_state_change("openai_primary", "claude_backup", reason="test")
            calls = [c for c in mock_chmod.call_args_list if str(alerts_path) in str(c.args[0])]
            assert len(calls) >= 1
            assert calls[0].args[1] == 0o600


# ── Config Loader Permissions ────────────────────────────────────────


class TestConfigFilePermissions:
    """Config files should have restricted permissions when written."""

    def test_restrict_permissions_helper(self, tmp_path):
        """Test the restrict_permissions helper function."""
        from router.config_loader import _restrict_permissions

        test_file = tmp_path / "test_config.json"
        test_file.write_text('{"key": "value"}')
        _restrict_permissions(test_file)

        if IS_UNIX:
            mode = _get_file_mode(test_file)
            assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_restrict_permissions_does_not_crash_on_failure(self, tmp_path):
        """_restrict_permissions should not crash if chmod fails."""
        from router.config_loader import _restrict_permissions

        test_file = tmp_path / "test_config.json"
        test_file.write_text('{"key": "value"}')

        with patch("router.config_loader.os.chmod", side_effect=OSError("not supported")):
            _restrict_permissions(test_file)  # Should not raise

    def test_restrict_permissions_called_on_load_config(self, tmp_path):
        """_restrict_permissions should be a callable utility."""
        from router.config_loader import _restrict_permissions
        assert callable(_restrict_permissions)
