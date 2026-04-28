from __future__ import annotations

import re

from weather_agent.domain.rules.short_id_generator import generate_short_id, strip_hash_prefix


class TestGenerateShortId:
    def test_rule_prefix(self) -> None:
        sid = generate_short_id("R")
        assert sid.startswith("R")
        assert len(sid) == 5

    def test_event_prefix(self) -> None:
        sid = generate_short_id("E")
        assert sid.startswith("E")
        assert len(sid) == 5

    def test_default_length(self) -> None:
        sid = generate_short_id("R")
        suffix = sid[1:]
        assert len(suffix) == 4

    def test_custom_length(self) -> None:
        sid = generate_short_id("R", length=3)
        suffix = sid[1:]
        assert len(suffix) == 3

    def test_uniqueness(self) -> None:
        ids = {generate_short_id("R") for _ in range(100)}
        assert len(ids) == 100

    def test_invalid_prefix_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Invalid short ID prefix"):
            generate_short_id("X")

    def test_invalid_length_too_short(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Invalid short ID length"):
            generate_short_id("R", length=2)

    def test_invalid_length_too_long(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Invalid short ID length"):
            generate_short_id("R", length=7)

    def test_no_ambiguous_characters(self) -> None:
        ambiguous = {"0", "O", "1", "I", "L"}
        for _ in range(100):
            sid = generate_short_id("R")
            suffix = sid[1:]
            for ch in suffix:
                assert ch not in ambiguous

    def test_alphanumeric_suffix(self) -> None:
        pattern = re.compile(r"^[RE][A-Z0-9]+$")
        for _ in range(100):
            sid = generate_short_id("R")
            assert pattern.match(sid), f"Short ID {sid} doesn't match pattern"


class TestStripHashPrefix:
    def test_strips_hash(self) -> None:
        assert strip_hash_prefix("#R7K2") == "R7K2"

    def test_no_hash(self) -> None:
        assert strip_hash_prefix("R7K2") == "R7K2"

    def test_multiple_hashes(self) -> None:
        assert strip_hash_prefix("##R7K2") == "R7K2"

    def test_empty_string(self) -> None:
        assert strip_hash_prefix("") == ""

    def test_only_hash(self) -> None:
        assert strip_hash_prefix("#") == ""