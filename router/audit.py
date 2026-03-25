"""Audit trail integrity verification via hash chain."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
from hashlib import sha256


class AuditChain:
    """Maintains a hash chain over log entries for tamper detection."""

    def __init__(self, last_hash: str = "GENESIS"):
        self._last_hash = last_hash

    def chain_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Add integrity fields to an entry and advance the chain.

        Mutates the entry dict in-place:
          - ``_prev_hash`` — hash of the previous entry (or ``GENESIS``)
          - ``_hash``      — SHA-256 of the entry after adding ``_prev_hash``
        """
        entry["_prev_hash"] = self._last_hash
        digest = sha256(
            json.dumps(entry, sort_keys=True).encode()
        ).hexdigest()
        entry["_hash"] = digest
        self._last_hash = digest
        return entry

    @property
    def last_hash(self) -> str:
        return self._last_hash


def verify_chain(log_path: Path) -> Tuple[bool, str]:
    """Read a JSONL log and verify its hash chain.

    Returns ``(valid, reason)`` where *reason* is the first violation
    found, or the string ``"chain valid"`` when everything checks out.
    """
    prev_hash = "GENESIS"
    prev_hash_field = None

    for lineno, line in enumerate(log_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return False, f"line {lineno}: invalid JSON"

        # Check that integrity fields exist
        if "_hash" not in entry or "_prev_hash" not in entry:
            return False, f"line {lineno}: missing integrity fields"

        # Verify _prev_hash matches previous entry's _hash
        if entry["_prev_hash"] != prev_hash:
            return False, f"line {lineno}: _prev_hash mismatch (expected {prev_hash}, got {entry['_prev_hash']})"

        # Verify _hash matches recomputed hash
        stored_hash = entry.pop("_hash")
        recomputed = sha256(
            json.dumps(entry, sort_keys=True).encode()
        ).hexdigest()
        if stored_hash != recomputed:
            return False, f"line {lineno}: hash mismatch (tampered content)"

        prev_hash = stored_hash

    return True, "chain valid"


def init_chain(log_path: Path) -> str:
    """Return the last hash from an existing log, or ``"GENESIS"`` if empty."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return "GENESIS"

    lines = log_path.read_text().strip().splitlines()
    if not lines:
        return "GENESIS"

    last = json.loads(lines[-1])
    return last.get("_hash", "GENESIS")
