"""Persistent state store for Codex usage tracking."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from .models import CodexState
from .errors import StateError


CONFIG_DIR = Path(__file__).parent.parent / "config"
MANUAL_STATE_PATH = CONFIG_DIR / "codex_manual_state.json"
AUTO_STATE_PATH = CONFIG_DIR / "codex_auto_state.json"
STATE_HISTORY_PATH = CONFIG_DIR / "codex_state_history.json"
WAL_PATH = CONFIG_DIR / "codex_state_wal.jsonl"
MAX_HISTORY_ENTRIES = 50
MIN_STATE_DURATION_S = 300  # 5 minutes minimum in a state
CONSECUTIVE_SUCCESSES_THRESHOLD = 3  # consecutive successes required before recovery to primary

# All 4-state transitions are valid (subscription ladder)
VALID_STATE_TRANSITIONS = {
    CodexState.OPENAI_PRIMARY: {
        CodexState.OPENAI_CONSERVATION, CodexState.CLAUDE_BACKUP, CodexState.OPENROUTER_FALLBACK,
    },
    CodexState.OPENAI_CONSERVATION: {
        CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP, CodexState.OPENROUTER_FALLBACK,
    },
    CodexState.CLAUDE_BACKUP: {
        CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, CodexState.OPENROUTER_FALLBACK,
    },
    CodexState.OPENROUTER_FALLBACK: {
        CodexState.OPENAI_PRIMARY, CodexState.OPENAI_CONSERVATION, CodexState.CLAUDE_BACKUP,
    },
}


# Backward compat mapping: old state names → new ones
_STATE_BACKWARD_COMPAT = {
    "normal": "openai_primary",
    "last10": "claude_backup",
}


class StateStore:
    """Read/write persistent Codex state with manual and auto layers."""

    def __init__(
        self,
        manual_path: Optional[Path] = None,
        auto_path: Optional[Path] = None,
        history_path: Optional[Path] = None,
        wal_path: Optional[Path] = None,
    ):
        self.manual_path = Path(manual_path) if manual_path else MANUAL_STATE_PATH
        self.auto_path = Path(auto_path) if auto_path else AUTO_STATE_PATH
        self.history_path = Path(history_path) if history_path else self.manual_path.parent / "codex_state_history.json"
        self.wal_path = Path(wal_path) if wal_path else WAL_PATH
        self._success_counter: dict[str, int] = {}  # state value -> consecutive success count
        self.recover_from_wal()
        self._ensure_state_files()

    def _ensure_state_files(self):
        """Create default state files if they don't exist."""
        self.manual_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.manual_path.exists():
            self._write_default(self.manual_path, CodexState.OPENAI_PRIMARY)
        if not self.auto_path.exists():
            self._write_default(self.auto_path, CodexState.OPENAI_PRIMARY)

    def _write_default(self, path: Path, state: CodexState):
        """Write default state to a file."""
        data = {"state": state.value}
        self._write(path, data)

    @staticmethod
    def _validate_state(raw: str) -> Optional["CodexState"]:
        """Validate a state string. Returns CodexState or None if invalid.

        Supports backward compat: "normal" → OPENAI_PRIMARY, "last10" → CLAUDE_BACKUP.
        """
        # Apply backward compat mapping
        mapped = _STATE_BACKWARD_COMPAT.get(raw, raw)
        try:
            return CodexState(mapped)
        except ValueError:
            return None

    def _read(self, path: Path) -> Optional[dict]:
        """Read a state file. Returns None if file does not exist."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, IOError) as e:
            raise StateError(f"Failed to read state file {path}: {e}")

    def _append_to_wal(self, entry: dict) -> None:
        """Append an entry to the WAL (JSONL)."""
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.wal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _truncate_wal(self) -> None:
        """Truncate the WAL file."""
        if self.wal_path.exists():
            self.wal_path.write_text("")

    def recover_from_wal(self) -> int:
        """Recover from uncommitted WAL entries on restart.

        Reads the WAL, applies any uncommitted write intents, then
        truncates the WAL. Returns the number of recovered entries.
        """
        if not self.wal_path.exists():
            return 0

        try:
            lines = self.wal_path.read_text().strip().splitlines()
        except (IOError, OSError):
            return 0

        if not lines:
            return 0

        # Parse WAL entries, skip malformed lines
        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue  # corrupted line, skip

        # Find uncommitted writes (writes without a subsequent committed marker)
        recovered = 0
        for i, entry in enumerate(entries):
            if entry.get("action") == "write":
                # Check if a committed marker follows
                committed = any(
                    e.get("action") == "committed" for e in entries[i + 1 :]
                )
                if not committed:
                    # Apply the uncommitted write
                    target_path = Path(entry["path"])
                    data = entry["data"]
                    try:
                        # Direct atomic write without WAL re-append
                        self._atomic_write_only(target_path, data)
                        recovered += 1
                    except (IOError, OSError, StateError):
                        pass  # Best-effort recovery

        # Truncate WAL after recovery
        self._truncate_wal()
        return recovered

    def _atomic_write_only(self, path: Path, data: dict) -> None:
        """Atomic write without WAL logging (used during recovery)."""
        try:
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".state_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except IOError as e:
            raise StateError(f"Failed to write state file {path}: {e}")

    def _write(self, path: Path, data: dict):
        """Atomic write: write to temp file, then rename.

        Uses WAL: appends intent before write, committed marker after.
        """
        ts = datetime.now(timezone.utc).isoformat()
        # 1. Append write intent to WAL
        self._append_to_wal({
            "action": "write",
            "path": str(path),
            "data": data,
            "timestamp": ts,
        })
        try:
            # 2. Perform atomic write
            self._atomic_write_only(path, data)
            # 3. Mark WAL entry as committed
            self._append_to_wal({
                "action": "committed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            raise

    # -------------------------------------------------------------------------
    # Manual state (user-overridden, takes precedence)
    # -------------------------------------------------------------------------

    def get_manual_state(self) -> Optional[CodexState]:
        """Get the manually-set Codex state. Returns None if manual is not active or file missing."""
        data = self._read(self.manual_path)
        if data is None:
            return None
        raw = data.get("state")
        if raw is None or raw == "null":
            return None
        return self._validate_state(raw)

    def set_manual_state(self, state: Optional[CodexState]):
        """Set the manual override state. Pass None to clear."""
        if state is None:
            data = {"state": None}
        else:
            data = {"state": state.value}
        self._write(self.manual_path, data)

    # -------------------------------------------------------------------------
    # Auto state (computed, lower precedence than manual)
    # -------------------------------------------------------------------------

    def get_auto_state(self) -> Optional[CodexState]:
        """Get the auto-computed Codex state. Returns None if not set or file missing."""
        data = self._read(self.auto_path)
        if data is None:
            return None
        raw = data.get("state")
        if raw is None:
            return None
        return self._validate_state(raw)

    def set_auto_state(self, state: CodexState):
        """Set the auto-computed state."""
        data = {"state": state.value}
        self._write(self.auto_path, data)

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    def get_state(self) -> CodexState:
        """Get effective state (manual > auto > default openai_primary)."""
        manual = self.get_manual_state()
        if manual is not None:
            return manual
        auto = self.get_auto_state()
        if auto is not None:
            return auto
        return CodexState.OPENAI_PRIMARY

    def set_state(self, state: CodexState):
        """Set state (writes to manual for backward compat)."""
        self.set_manual_state(state)

    def get_paths(self) -> dict:
        """Return paths for both state files."""
        return {
            "manual": str(self.manual_path),
            "auto": str(self.auto_path),
        }

    # -------------------------------------------------------------------------
    # State history
    # -------------------------------------------------------------------------

    def log_state_transition(
        self,
        from_state: Optional[CodexState],
        to_state: CodexState,
        reason: str,
    ) -> None:
        """Log a state transition to the history file."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": from_state.value if from_state else None,
            "to": to_state.value,
            "reason": reason,
        }

        history = []
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    history = data.get("transitions", [])
                elif isinstance(data, list):
                    history = data
            except (json.JSONDecodeError, IOError):
                history = []

        history.append(entry)

        # Keep only last N entries
        if len(history) > MAX_HISTORY_ENTRIES:
            history = history[-MAX_HISTORY_ENTRIES:]

        self._write(self.history_path, {"transitions": history})

    def get_state_history(self, limit: int = 10) -> List[dict]:
        """Get recent state transitions."""
        if not self.history_path.exists():
            return []
        try:
            with open(self.history_path, "r") as f:
                data = json.load(f)
            return data.get("transitions", [])[-limit:]
        except (json.JSONDecodeError, IOError):
            return []

    # -------------------------------------------------------------------------
    # Sticky state — success-count threshold for recovery
    # -------------------------------------------------------------------------

    def record_success(self, state: CodexState, success: bool) -> None:
        """Record a success or failure for sticky-state tracking.

        On success, increment the consecutive success counter for the state.
        On failure, reset it to 0.
        """
        key = state.value
        if success:
            self._success_counter[key] = self._success_counter.get(key, 0) + 1
        else:
            self._success_counter[key] = 0

    def can_recover_to_primary(self, state: CodexState) -> bool:
        """Check if enough consecutive successes have been recorded to allow recovery to primary."""
        key = state.value
        count = self._success_counter.get(key, 0)
        return count >= CONSECUTIVE_SUCCESSES_THRESHOLD

    def reset_success_counter(self, state: CodexState) -> None:
        """Reset the consecutive success counter for a state."""
        self._success_counter[state.value] = 0

    # -------------------------------------------------------------------------
    # Anti-flap protection
    # -------------------------------------------------------------------------

    def can_transition(self, new_state: CodexState, force: bool = False) -> tuple[bool, str]:
        """Check if a state transition is allowed (anti-flap + sticky state).

        Returns (allowed, reason).

        Args:
            new_state: The target state.
            force: If True, bypass all checks (emergency override).
        """
        current = self.get_state()
        if current == new_state:
            return True, "same_state"

        # Force bypasses everything
        if force:
            return True, "emergency_override_forced"

        # Allow emergency overrides (any state → openai_primary or claude_backup is always OK)
        emergency_targets = {CodexState.OPENAI_PRIMARY, CodexState.CLAUDE_BACKUP}
        if new_state in emergency_targets:
            return True, "emergency_override"

        # Check history for recent transitions
        history = self.get_state_history(limit=5)
        if not history:
            return True, "no_history"

        last_entry = history[-1]
        try:
            last_ts = datetime.fromisoformat(last_entry["timestamp"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            elapsed = (now - last_ts).total_seconds()

            if elapsed < MIN_STATE_DURATION_S:
                return False, f"anti_flap: last transition {elapsed:.0f}s ago, minimum is {MIN_STATE_DURATION_S}s"
        except (ValueError, KeyError, TypeError):
            pass  # Can't parse history, allow transition

        return True, "allowed"

    def set_state_with_history(
        self,
        state: CodexState,
        reason: str = "manual",
        force: bool = False,
    ) -> bool:
        """Set state with history tracking and anti-flap protection.

        Returns True if the transition was made.
        """
        current = self.get_state()

        if not force:
            allowed, msg = self.can_transition(state)
            if not allowed:
                raise StateError(f"State transition blocked: {msg}")

        # Set the state
        self.set_manual_state(state)

        # Log the transition
        self.log_state_transition(current, state, reason)

        return True

    # -------------------------------------------------------------------------
    # Invariant validation
    # -------------------------------------------------------------------------

    def validate_transition(self, from_state: CodexState, to_state: CodexState) -> bool:
        """Validate that a state transition is allowed by the state machine."""
        if from_state == to_state:
            return True
        allowed = VALID_STATE_TRANSITIONS.get(from_state, set())
        return to_state in allowed
