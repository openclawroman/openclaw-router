"""Tests for audit trail integrity verification (hash chain)."""

import json
import pytest
from pathlib import Path
from hashlib import sha256

from router.audit import AuditChain, verify_chain, init_chain
from router.attempt_logger import AttemptLogger, ExecutorAttempt, RoutingTrace


class TestAuditChainEntry:
    """AuditChain.chain_entry() adds integrity fields."""

    def test_chain_entry_adds_prev_hash_and_hash(self):
        chain = AuditChain()
        entry = {"type": "routing_trace", "trace_id": "t1"}
        result = chain.chain_entry(entry)
        assert "_prev_hash" in result
        assert "_hash" in result
        assert result["_prev_hash"] == "GENESIS"
        assert result["_hash"] == chain.last_hash

    def test_chain_is_deterministic(self):
        """Same input yields the same hash."""
        chain1 = AuditChain()
        chain2 = AuditChain()
        e1 = chain1.chain_entry({"data": "hello"})
        e2 = chain2.chain_entry({"data": "hello"})
        assert e1["_hash"] == e2["_hash"]
        assert e1["_prev_hash"] == e2["_prev_hash"]

    def test_chain_links_entries(self):
        chain = AuditChain()
        e1 = chain.chain_entry({"data": "first"})
        e2 = chain.chain_entry({"data": "second"})
        assert e2["_prev_hash"] == e1["_hash"]

    def test_hash_is_sha256(self):
        chain = AuditChain()
        entry = {"data": "test"}
        result = chain.chain_entry(entry)
        # Recompute
        check = dict(entry)  # has _prev_hash and _hash
        stored = check.pop("_hash")
        recomputed = sha256(
            json.dumps(check, sort_keys=True).encode()
        ).hexdigest()
        assert stored == recomputed


class TestVerifyChain:
    """verify_chain() validates or detects tampering."""

    def _write_log(self, path: Path, entries: list[dict]):
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def test_verify_valid_chain(self, tmp_path):
        chain = AuditChain()
        log_path = tmp_path / "valid.jsonl"
        entries = []
        for i in range(5):
            entries.append(chain.chain_entry({"data": f"entry-{i}"}))
        self._write_log(log_path, entries)
        valid, reason = verify_chain(log_path)
        assert valid is True
        assert reason == "chain valid"

    def test_verify_detects_tampered_content(self, tmp_path):
        chain = AuditChain()
        log_path = tmp_path / "tampered.jsonl"
        entries = []
        for i in range(3):
            entries.append(chain.chain_entry({"data": f"entry-{i}"}))
        self._write_log(log_path, entries)

        # Tamper with entry 1's content
        lines = log_path.read_text().splitlines()
        tampered = json.loads(lines[1])
        tampered["data"] = "MUTATED"
        lines[1] = json.dumps(tampered)
        log_path.write_text("\n".join(lines) + "\n")

        valid, reason = verify_chain(log_path)
        assert valid is False
        assert "hash mismatch" in reason

    def test_verify_detects_deleted_entry(self, tmp_path):
        chain = AuditChain()
        log_path = tmp_path / "deleted.jsonl"
        entries = []
        for i in range(3):
            entries.append(chain.chain_entry({"data": f"entry-{i}"}))
        self._write_log(log_path, entries)

        # Delete middle entry
        lines = log_path.read_text().splitlines()
        del lines[1]
        log_path.write_text("\n".join(lines) + "\n")

        valid, reason = verify_chain(log_path)
        assert valid is False
        assert "_prev_hash mismatch" in reason

    def test_verify_detects_inserted_entry(self, tmp_path):
        chain = AuditChain()
        log_path = tmp_path / "inserted.jsonl"
        entries = []
        for i in range(2):
            entries.append(chain.chain_entry({"data": f"entry-{i}"}))
        self._write_log(log_path, entries)

        # Insert a forged entry between the two
        lines = log_path.read_text().splitlines()
        forged = {"data": "forged", "_prev_hash": "fake", "_hash": "also_fake"}
        lines.insert(1, json.dumps(forged))
        log_path.write_text("\n".join(lines) + "\n")

        valid, reason = verify_chain(log_path)
        assert valid is False
        assert "mismatch" in reason.lower() or "hash mismatch" in reason

    def test_verify_empty_log(self, tmp_path):
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("")
        valid, reason = verify_chain(log_path)
        assert valid is True
        assert reason == "chain valid"

    def test_verify_missing_integrity_fields(self, tmp_path):
        log_path = tmp_path / "nointegrity.jsonl"
        log_path.write_text(json.dumps({"data": "no hash"}) + "\n")
        valid, reason = verify_chain(log_path)
        assert valid is False
        assert "missing integrity fields" in reason


class TestInitChain:
    """init_chain() resumes from the last entry."""

    def test_init_from_existing_log(self, tmp_path):
        chain = AuditChain()
        log_path = tmp_path / "resume.jsonl"
        entries = []
        for i in range(3):
            entries.append(chain.chain_entry({"data": f"entry-{i}"}))
        with open(log_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        last = init_chain(log_path)
        assert last == entries[-1]["_hash"]

    def test_init_empty_log(self, tmp_path):
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("")
        assert init_chain(log_path) == "GENESIS"

    def test_init_nonexistent_file(self, tmp_path):
        log_path = tmp_path / "nope.jsonl"
        assert init_chain(log_path) == "GENESIS"


class TestAttemptLoggerIntegrity:
    """AttemptLogger writes integrity fields on log entries."""

    def test_log_trace_includes_integrity_fields(self, tmp_path):
        log_path = tmp_path / "integrity.jsonl"
        logger = AttemptLogger(log_path=log_path)
        trace = RoutingTrace(
            trace_id="abc123",
            task_id="t1",
            state="openai_primary",
            final_tool="codex_cli",
            final_success=True,
        )
        logger.log_trace(trace)

        entry = json.loads(log_path.read_text().strip())
        assert "_prev_hash" in entry
        assert "_hash" in entry
        assert entry["_prev_hash"] == "GENESIS"

    def test_log_trace_chains_entries(self, tmp_path):
        log_path = tmp_path / "chain.jsonl"
        logger = AttemptLogger(log_path=log_path)

        for i in range(3):
            trace = RoutingTrace(
                trace_id=f"t{i}", task_id=f"task{i}", state="openai_primary",
                final_tool="codex_cli", final_success=True,
            )
            logger.log_trace(trace)

        lines = log_path.read_text().strip().splitlines()
        entries = [json.loads(l) for l in lines]
        for i in range(1, len(entries)):
            assert entries[i]["_prev_hash"] == entries[i - 1]["_hash"]

    def test_log_trace_verify_chain(self, tmp_path):
        log_path = tmp_path / "verify.jsonl"
        logger = AttemptLogger(log_path=log_path)

        for i in range(5):
            trace = RoutingTrace(
                trace_id=f"t{i}", task_id=f"task{i}", state="openai_primary",
                final_tool="codex_cli", final_success=True,
            )
            logger.log_trace(trace)

        valid, reason = verify_chain(log_path)
        assert valid is True
        assert reason == "chain valid"

    def test_routing_trace_format_unchanged(self, tmp_path):
        """The RoutingTrace dataclass itself should not have _hash/_prev_hash."""
        trace = RoutingTrace(
            trace_id="abc",
            task_id="t1",
            state="openai_primary",
            final_tool="codex_cli",
            final_success=True,
        )
        d = trace.to_dict()
        assert "_hash" not in d
        assert "_prev_hash" not in d
