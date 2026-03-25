"""Persistent state store for Codex usage tracking."""

import json
from pathlib import Path
from typing import Optional

from .models import CodexState
from .errors import StateError, ConfigurationError


DEFAULT_STATE_PATH = Path(__file__).parent.parent / "config" / "codex_manual_state.json"


class StateStore:
    """Read/write persistent Codex state."""

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or DEFAULT_STATE_PATH
        self._ensure_state_file()

    def _ensure_state_file(self):
        """Create default state file if it doesn't exist."""
        if not self.state_path.exists():
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_default()

    def _write_default(self):
        """Write default state."""
        default = {"state": CodexState.INCLUDED_FIRST.value}
        self._write(default)

    def _read(self) -> dict:
        """Read state file."""
        try:
            with open(self.state_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise StateError(f"Failed to read state: {e}")

    def _write(self, data: dict):
        """Write state file."""
        try:
            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            raise StateError(f"Failed to write state: {e}")

    def get_state(self) -> CodexState:
        """Get current Codex state."""
        data = self._read()
        state_str = data.get("state", CodexState.INCLUDED_FIRST.value)
        try:
            return CodexState(state_str)
        except ValueError:
            return CodexState.INCLUDED_FIRST

    def set_state(self, state: CodexState):
        """Set Codex state."""
        data = {"state": state.value}
        self._write(data)

    def get_state_path(self) -> Path:
        """Return the state file path."""
        return self.state_path