from __future__ import annotations

import random
import string

_ALPHABET = string.digits + string.ascii_uppercase
_ALPHABET_NO_AMBIGUOUS = "".join(c for c in _ALPHABET if c not in {"0", "O", "1", "I", "L"})

_VALID_PREFIXES = {"R", "E"}


def generate_short_id(prefix: str, length: int = 4) -> str:
    if prefix not in _VALID_PREFIXES:
        raise ValueError(f"Invalid short ID prefix: {prefix!r}. Must be one of {_VALID_PREFIXES}")
    if length < 3 or length > 6:
        raise ValueError(f"Invalid short ID length: {length}. Must be between 3 and 6")
    suffix = "".join(random.choices(_ALPHABET_NO_AMBIGUOUS, k=length))
    return f"{prefix}{suffix}"


def strip_hash_prefix(short_id: str) -> str:
    return short_id.lstrip("#")
