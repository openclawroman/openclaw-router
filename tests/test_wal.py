"""Tests for WAL (write-ahead log) in StateStore."""

import json
from pathlib import Path

import pytest

from router.models import CodexState
from router.state_store import StateStore


@pytest.fixture
def tmp_store(tmp_path):
    """Provide a StateStore with temporary paths."""
    manual = tmp_path / "manual.json"
    auto = tmp_path / "auto.json"
    wal = tmp_path / "state_wal.jsonl"
    store = StateStore(manual_path=manual, auto_path=auto, wal_path=wal)
    return store, wal


class TestWALWrite:
    """WAL should record write intent before state write."""

    def test_wal_records_write_and_commit(self, tmp_store):
        store, wal = tmp_store
        # Perform a write via set_manual_state
        store.set_manual_state(CodexState.CLAUDE_BACKUP)

        lines = wal.read_text().strip().splitlines()
        entries = [json.loads(l) for l in lines]

        # Should have write + committed pairs (from init defaults + our write)
        write_entries = [e for e in entries if e.get("action") == "write"]
        commit_entries = [e for e in entries if e.get("action") == "committed"]

        assert len(write_entries) >= 1, "At least one write entry in WAL"
        assert len(commit_entries) >= 1, "At least one commit entry in WAL"

        # Last write should be for our manual state
        last_write = write_entries[-1]
        assert "path" in last_write
        assert last_write["data"]["state"] == "claude_backup"
        assert "timestamp" in last_write


class TestWALCommitMarker:
    """Each successful write should have a committed marker after it."""

    def test_committed_follows_write(self, tmp_store):
        store, wal = tmp_store
        store.set_auto_state(CodexState.OPENAI_CONSERVATION)

        lines = wal.read_text().strip().splitlines()
        entries = [json.loads(l) for l in lines]

        # Find our specific write (auto path)
        auto_writes = [
            i
            for i, e in enumerate(entries)
            if e.get("action") == "write" and e.get("data", {}).get("state") == "openai_conservation"
        ]
        assert len(auto_writes) == 1
        write_idx = auto_writes[0]

        # Next entry should be committed
        next_entry = entries[write_idx + 1]
        assert next_entry["action"] == "committed"
        assert "timestamp" in next_entry


class TestWALRecoveryFromCrash:
    """Simulate a crash: WAL has write without committed marker."""

    def test_recovery_applies_uncommitted_write(self, tmp_path):
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        wal = tmp_path / "state_wal.jsonl"

        # Create initial store (writes defaults)
        store = StateStore(manual_path=manual, auto_path=auto, wal_path=wal)

        # Simulate a crash: write to WAL but don't mark committed
        crashed_data = {"state": "openrouter_fallback"}
        wal.write_text(
            json.dumps({
                "action": "write",
                "path": str(manual),
                "data": crashed_data,
                "timestamp": "2026-03-25T18:00:00+00:00",
            })
            + "\n"
        )

        # Create new store — should recover
        recovered_store = StateStore(manual_path=manual, auto_path=auto, wal_path=wal)
        state = recovered_store.get_manual_state()
        assert state == CodexState.OPENROUTER_FALLBACK

    def test_recovery_skips_already_committed(self, tmp_path):
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        wal = tmp_path / "state_wal.jsonl"

        # WAL with committed write — should not re-apply
        wal.write_text(
            json.dumps({"action": "write", "path": str(manual), "data": {"state": "claude_backup"}, "timestamp": "2026-03-25T18:00:00+00:00"})
            + "\n"
            + json.dumps({"action": "committed", "timestamp": "2026-03-25T18:00:01+00:00"})
            + "\n"
        )

        # Create store with defaults (will overwrite manual)
        manual.write_text(json.dumps({"state": "openai_primary"}))
        auto.write_text(json.dumps({"state": "openai_primary"}))

        store = StateStore(manual_path=manual, auto_path=auto, wal_path=wal)
        # Should not have re-applied the committed write
        assert store.get_manual_state() == CodexState.OPENAI_PRIMARY


class TestWALEmpty:
    """Empty WAL should not cause issues."""

    def test_empty_wal_no_recovery(self, tmp_path):
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        wal = tmp_path / "state_wal.jsonl"

        # Empty WAL file
        wal.write_text("")

        store = StateStore(manual_path=manual, auto_path=auto, wal_path=wal)
        assert store.get_state() == CodexState.OPENAI_PRIMARY

    def test_no_wal_file(self, tmp_path):
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        wal = tmp_path / "nonexistent_wal.jsonl"

        store = StateStore(manual_path=manual, auto_path=auto, wal_path=wal)
        assert store.get_state() == CodexState.OPENAI_PRIMARY


class TestWALCorruptedEntries:
    """Corrupted WAL entries should be skipped gracefully."""

    def test_corrupted_lines_skipped(self, tmp_path):
        manual = tmp_path / "manual.json"
        auto = tmp_path / "auto.json"
        wal = tmp_path / "state_wal.jsonl"

        # Mix of valid and corrupted entries
        wal.write_text(
            "not json at all\n"
            + json.dumps({"action": "write", "path": str(manual), "data": {"state": "claude_backup"}, "timestamp": "2026-03-25T18:00:00+00:00"})
            + "\n"
            + "  \n"
            + "{{invalid json}}\n"
        )

        # Should recover the valid uncommitted entry and skip corrupted ones
        store = StateStore(manual_path=manual, auto_path=auto, wal_path=wal)
        assert store.get_manual_state() == CodexState.CLAUDE_BACKUP
