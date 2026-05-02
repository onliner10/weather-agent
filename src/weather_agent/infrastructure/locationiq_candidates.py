from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

from weather_agent.domain.polish_utils import normalize_for_matching
from weather_agent.domain.weather import LocationRef

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({"w", "we", "na", "nad", "pod", "przy", "do", "z", "ze", "i"})
_PLACE_CLASSES = frozenset({"boundary", "place"})
_PLACE_TYPES = frozenset(
    {
        "administrative",
        "city",
        "city_district",
        "district",
        "hamlet",
        "locality",
        "neighbourhood",
        "neighborhood",
        "quarter",
        "suburb",
        "town",
        "village",
    }
)


def select_best_candidate(
    query: str, candidates: Sequence[Mapping[str, object]]
) -> LocationRef | None:
    tokens = _query_tokens(query)
    scored = [
        (_candidate_score(candidate, tokens), index, candidate)
        for index, candidate in enumerate(candidates)
    ]
    usable = [(score, index, candidate) for score, index, candidate in scored if score > 0]
    if not usable:
        return None
    _score, _index, candidate = max(usable, key=lambda item: (item[0], -item[1]))
    return _candidate_to_location_ref(candidate)


def _query_tokens(query: str) -> frozenset[str]:
    normalized = normalize_for_matching(query)
    return frozenset(
        token
        for token in _TOKEN_RE.findall(normalized)
        if len(token) >= 2 and token not in _STOPWORDS
    )


def _candidate_score(candidate: Mapping[str, object], query_tokens: frozenset[str]) -> float:
    if not _is_polish_candidate(candidate):
        return 0.0

    lat = _float_field(candidate, "lat")
    lon = _float_field(candidate, "lon")
    if lat is None or lon is None:
        return 0.0

    text = _candidate_text(candidate)
    matched = query_tokens.intersection(_query_tokens(text))
    if query_tokens and not matched:
        return 0.0

    is_place = _is_place_candidate(candidate)
    if not is_place and query_tokens and matched != query_tokens:
        return 0.0

    name_text = _string_field(candidate, "name") or _first_display_part(candidate) or ""
    name_matches = query_tokens.intersection(_query_tokens(name_text))
    score = float(len(matched) * 10 + len(name_matches) * 3)
    if is_place:
        score += 5.0
    score += _float_field(candidate, "importance") or 0.0
    return score


def _is_polish_candidate(candidate: Mapping[str, object]) -> bool:
    address = _address(candidate)
    country_code = _string_field(address, "country_code") or _string_field(
        candidate, "country_code"
    )
    return country_code is None or country_code.lower() == "pl"


def _is_place_candidate(candidate: Mapping[str, object]) -> bool:
    category = _string_field(candidate, "class") or _string_field(candidate, "category")
    candidate_type = _string_field(candidate, "type")
    addresstype = _string_field(candidate, "addresstype")
    return (
        category in _PLACE_CLASSES or candidate_type in _PLACE_TYPES or addresstype in _PLACE_TYPES
    )


def _candidate_to_location_ref(candidate: Mapping[str, object]) -> LocationRef | None:
    lat = _float_field(candidate, "lat")
    lon = _float_field(candidate, "lon")
    if lat is None or lon is None:
        return None
    location_id = _candidate_id(candidate)
    location_name = _string_field(candidate, "name") or _first_display_part(candidate)
    if location_name is None:
        location_name = _string_field(candidate, "display_name") or "Lokalizacja"
    return LocationRef(
        id=location_id,
        name=location_name,
        latitude=lat,
        longitude=lon,
    )


def _candidate_id(candidate: Mapping[str, object]) -> str:
    place_id = candidate.get("place_id")
    if place_id is not None:
        return str(place_id)
    osm_type = _string_field(candidate, "osm_type")
    osm_id = candidate.get("osm_id")
    if osm_type is not None and osm_id is not None:
        return f"{osm_type}:{osm_id}"
    return str(hash(_candidate_text(candidate)))


def _candidate_text(candidate: Mapping[str, object]) -> str:
    parts: list[str] = []
    for key in ("name", "display_name", "type", "class", "category", "addresstype"):
        value = _string_field(candidate, key)
        if value:
            parts.append(value)
    address = _address(candidate)
    parts.extend(value for value in _string_values(address) if value)
    return " ".join(parts)


def _first_display_part(candidate: Mapping[str, object]) -> str | None:
    display_name = _string_field(candidate, "display_name")
    if display_name is None:
        return None
    first_part = display_name.split(",", maxsplit=1)[0].strip()
    return first_part or None


def _address(candidate: Mapping[str, object]) -> Mapping[str, object]:
    address = candidate.get("address")
    if isinstance(address, Mapping):
        return cast(Mapping[str, object], address)
    return {}


def _string_field(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _float_field(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    try:
        if isinstance(value, str | int | float):
            return float(value)
    except ValueError:
        return None
    return None


def _string_values(data: Mapping[str, object]) -> list[str]:
    return [value.strip() for value in data.values() if isinstance(value, str) and value.strip()]
