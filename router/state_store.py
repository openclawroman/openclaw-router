"""Persistent state store for Codex usage tracking."""

import json
from pathlib import Path
from typing import Optional

from .models import CodexState
from .errors import StateError, ConfigurationError


CONFIG_DIR = Path(__file__).parent.parent / "config"
MANUAL_STATE_PATH = CONFIG_DIR / "codex_manual_state.json"
AUTO_STATE_PATH = CONFIG_DIR / "codex_auto_state.json"


class StateStore:
    """Read/write persistent Codex state with manual and auto layers."""

    def __init__(
        self,
        manual_path: Optional[Path] = None,
        auto_path: Optional[Path] = None,
    ):
        self.manual_path = manual_path or MANUAL_STATE_PATH
        self.auto_path = auto_path or AUTO_STATE_PATH
        self._ensure_state_files()

    def _ensure_state_files(self):
        """Create default state files if they don't exist."""
        self.manual_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.manual_path.exists():
            self._write_default(self.manual_path, CodexState.NORMAL)
        if not self.auto_path.exists():
            self._write_default(self.auto_path, CodexState.NORMAL)

    def _write_default(self, path: Path, state: CodexState):
        """Write default state to a file."""
        data = {"state": state.value}
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            raise StateError(f"Failed to write state file {path}: {e}")

    @staticmethod
    def _validate_state(raw: str) -> Optional["CodexState"]:
        """Validate a state string. Returns CodexState or None if invalid."""
        try:
            return CodexState(raw)
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

    def _write(self, path: Path, data: dict):
        """Write a state file."""
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            raise StateError(f"Failed to write state file {path}: {e}")

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
        """Get effective state (manual > auto > default normal)."""
        manual = self.get_manual_state()
        if manual is not None:
            return manual
        auto = self.get_auto_state()
        if auto is not None:
            return auto
        return CodexState.NORMAL

    def set_state(self, state: CodexState):
        """Set state (writes to manual for backward compat)."""
        self.set_manual_state(state)

    def get_paths(self) -> dict:
        """Return paths for both state files."""
        return {
            "manual": str(self.manual_path),
            "auto": str(self.auto_path),
        }
