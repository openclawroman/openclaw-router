"""Secret redaction utilities — sanitize logs, error messages, and traces."""

import re
from typing import Any

# ── Patterns ─────────────────────────────────────────────────────────────────

_SECRET_PATTERNS: list[re.Pattern] = [
    # Order matters: more-specific prefixes first
    re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}"),
    re.compile(r"sk-or-[a-zA-Z0-9\-]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"Bearer\s+[a-zA-Z0-9\-_]{20,}"),
    re.compile(r"api[_-]?key[=:]\s*\S+", re.IGNORECASE),
]

_SENSITIVE_KEYS = frozenset({
    "api_key", "token", "secret", "authorization", "credential",
})


# ── String-level sanitization ────────────────────────────────────────────────

def sanitize_secrets(text: str) -> str:
    """Replace API key patterns in *text* with ``[REDACTED]``."""
    if not text:
        return text
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


# ── Dict-level sanitization ──────────────────────────────────────────────────

def redact_dict(data: dict) -> dict:
    """Return a shallow-copy of *data* with sensitive values redacted.

    Keys whose lowercased form matches *_SENSITIVE_KEYS* are replaced with
    ``"[REDACTED]"``.  All other string values are run through
    :func:`sanitize_secrets`.  Nested dicts are processed recursively.
    """
    out: dict = {}
    for key, value in data.items():
        if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
            out[key] = "[REDACTED]"
        elif isinstance(value, dict):
            out[key] = redact_dict(value)
        elif isinstance(value, str):
            out[key] = sanitize_secrets(value)
        elif isinstance(value, list):
            out[key] = [
                redact_dict(item) if isinstance(item, dict)
                else sanitize_secrets(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            out[key] = value
    return out
