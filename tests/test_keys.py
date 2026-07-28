from __future__ import annotations

from smartreact.keys import (
    build_template_index,
    candidate_templates,
    extract_key_strings,
    orders_for_template,
)
from smartreact.types import KeyMatch, KeysResult, ReactionTemplate


def _make_template(
    left: str, right: str, template_id: int = 0, name: str = "t"
) -> ReactionTemplate:
    return ReactionTemplate(
        smarts="[#6:1]>>[#6:1]",
        reactant_categories=(left, right),
        name=name,
        template_id=template_id,
    )


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


class TestBuildTemplateIndex:
    def test_groups_by_requirement_pair(self):
        templates = [
            _make_template("A", "B", template_id=0),
            _make_template("A", "B", template_id=1),
            _make_template("C", "D", template_id=2),
        ]
        index = build_template_index(templates)
        assert index[("A", "B")] == [0, 1]
        assert index[("C", "D")] == [2]

    def test_empty_templates(self):
        assert build_template_index([]) == {}


class TestCandidateTemplates:
    def test_forward_only(self):
        index = build_template_index([_make_template("A", "B")])
        assert candidate_templates({"A"}, {"B"}, index) == {0: [0]}

    def test_reverse_only(self):
        index = build_template_index([_make_template("A", "B")])
        assert candidate_templates({"B"}, {"A"}, index) == {0: [1]}

    def test_both_orders(self):
        index = build_template_index([_make_template("A", "B")])
        assert candidate_templates({"A", "B"}, {"A", "B"}, index) == {0: [0, 1]}

    def test_no_match_returns_empty_dict(self):
        index = build_template_index([_make_template("A", "B")])
        assert candidate_templates({"X"}, {"Y"}, index) == {}

    def test_equivalent_to_scanning_orders_for_template(self):
        """candidate_templates must find exactly the same (template, orders) as
        scanning every template with orders_for_template, for arbitrary key sets.

        This is the correctness guarantee behind replacing the O(len(templates))
        per-pair scan with an O(|keys1| * |keys2|) index lookup.
        """
        templates = [
            _make_template("A", "B", template_id=0),
            _make_template("B", "A", template_id=1),
            _make_template("A", "A", template_id=2),
            _make_template("C", "D", template_id=3),
            _make_template("A", "B", template_id=4),  # duplicate requirement pair
        ]
        index = build_template_index(templates)

        key_sets = [{"A"}, {"B"}, {"A", "B"}, {"C", "D"}, {"A", "C"}, set(), {"Z"}]
        for keys1 in key_sets:
            for keys2 in key_sets:
                expected = {}
                for i, templ in enumerate(templates):
                    orders = orders_for_template(keys1, keys2, templ.reactant_categories)
                    if orders:
                        expected[i] = orders
                assert candidate_templates(keys1, keys2, index) == expected
