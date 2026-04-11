"""Simple API key authentication for the A2A server."""

from __future__ import annotations


def verify_api_key(provided: str, expected: str) -> bool:
    """Constant-time comparison of API keys."""
    if not expected:
        return True
    if len(provided) != len(expected):
        return False
    result = 0
    for a, b in zip(provided, expected):
        result |= ord(a) ^ ord(b)
    return result == 0
