"""Defense-in-depth content sanitization for routing logs.

Prevents accidental leakage of prompt text, task content, or file contents
into structured routing logs.
"""

from typing import Dict, Any

# Keys whose values are likely content and should be removed/truncated
CONTENT_KEYS = frozenset({
    "prompt", "content", "body", "text", "code", "diff",
    "file_content", "summary",
})

MAX_CONTENT_KEY_LENGTH = 200   # truncate content-key values above this
MAX_STRING_LENGTH = 500        # truncate any string value above this


def sanitize_content(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of *entry* with content fields sanitized.

    Rules
    -----
    1. Keys in ``CONTENT_KEYS`` are **removed** if their value is a string
       longer than ``MAX_CONTENT_KEY_LENGTH`` (200 chars).
    2. Any remaining string value longer than ``MAX_STRING_LENGTH`` (500 chars)
       is replaced with ``[TRUNCATED {N} chars]``.
    3. Non-string values and short strings are preserved untouched.
    """
    sanitized: Dict[str, Any] = {}

    for key, value in entry.items():
        # Rule 1: remove content-bearing keys with long string values
        if key in CONTENT_KEYS and isinstance(value, str) and len(value) > MAX_CONTENT_KEY_LENGTH:
            continue  # drop the key entirely

        # Rule 2: truncate any excessively long string value
        if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
            sanitized[key] = f"[TRUNCATED {len(value)} chars]"
            continue

        sanitized[key] = value

    return sanitized
