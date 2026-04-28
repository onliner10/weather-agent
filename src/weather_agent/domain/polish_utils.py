from __future__ import annotations

import unicodedata

POLISH_ASCII: dict[str, str] = {
    "ą": "a",
    "ć": "c",
    "ę": "e",
    "ł": "l",
    "ń": "n",
    "ó": "o",
    "ś": "s",
    "ź": "z",
    "ż": "z",
    "Ą": "a",
    "Ć": "c",
    "Ę": "e",
    "Ł": "l",
    "Ń": "n",
    "Ó": "o",
    "Ś": "s",
    "Ź": "z",
    "Ż": "z",
}


def normalize_polish(text: str) -> str:
    return "".join(POLISH_ASCII.get(c, c) for c in text.lower())


def normalize_for_matching(s: str) -> str:
    normalized = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    ascii_friendly = "".join(POLISH_ASCII.get(ch, ch) for ch in stripped)
    return ascii_friendly.lower().strip()