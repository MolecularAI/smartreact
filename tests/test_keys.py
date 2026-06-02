from __future__ import annotations

from smartreact.keys import extract_key_strings, orders_for_template
from smartreact.types import KeyMatch, KeysResult


class TestOrdersForTemplate:
    def test_no_match(self):
        assert orders_for_template({"X"}, {"Y"}, ("A", "B")) == []

    def test_forward_only(self):
        assert orders_for_template({"A"}, {"B"}, ("A", "B")) == [0]

    def test_reverse_only(self):
        assert orders_for_template({"B"}, {"A"}, ("A", "B")) == [1]

    def test_both(self):
        assert orders_for_template({"A", "B"}, {"A", "B"}, ("A", "B")) == [0, 1]

    def test_multiple_dots_in_key(self):
        assert orders_for_template({"A"}, {"B.C"}, ("A", "B.C")) == [0]


class TestExtractKeyStrings:
    def test_none_input(self):
        assert extract_key_strings(None) == set()

    def test_keys_result(self):
        kr = KeysResult(
            smiles="C",
            matches=[KeyMatch("A", "A1", "A1a"), KeyMatch("B", "B1", "B1a")],
        )
        keys = extract_key_strings(kr)
        assert keys == {"A1a", "B1a"}

    def test_excludes_higher_levels(self):
        """Only subsubcategory (Level 3) strings should be returned."""
        kr = KeysResult(
            smiles="C",
            matches=[
                KeyMatch(
                    "Phenol",
                    "PhenolHeteroaromatic6membered",
                    "PhenolHeteroaromatic6membered",
                )
            ],
        )
        keys = extract_key_strings(kr)
        assert "PhenolHeteroaromatic6membered" in keys
        assert "Phenol" not in keys

    def test_empty_matches(self):
        kr = KeysResult(smiles="C", matches=[])
        assert extract_key_strings(kr) == set()
