from __future__ import annotations

from smartreact.types import KeyMatch, KeysResult, ReactionResult, ReactionTemplate


class TestReactionTemplate:
    def test_frozen(self):
        t = ReactionTemplate(smarts="[C:1]>>[C:1]", reactant_categories=("A", "B"), name="test")
        assert t.smarts == "[C:1]>>[C:1]"
        assert t.reactant_categories == ("A", "B")
        assert t.name == "test"

    def test_hashable(self):
        t = ReactionTemplate(smarts="[C:1]>>[C:1]", reactant_categories=("A", "B"), name="test")
        assert hash(t) is not None
        assert {t}


class TestReactionResult:
    def test_str_with_products(self):
        r = ReactionResult(
            reactant_a="CCO",
            reactant_b="CC",
            reaction_name="test_rxn",
            products=["CCCO", "CCCCO"],
        )
        s = str(r)
        assert "test_rxn" in s
        assert "CCO" in s
        assert "CCCO" in s

    def test_str_no_products(self):
        r = ReactionResult(reactant_a="C", reactant_b="CC", reaction_name="x", products=[])
        assert "(no products)" in str(r)


class TestKeyMatch:
    def test_frozen(self):
        km = KeyMatch(category="cat", subcategory="sub", subsubcategory="subsub")
        assert km.category == "cat"


class TestKeysResult:
    def test_no_matches(self):
        kr = KeysResult(smiles="C", matches=[])
        assert "(no matches)" in str(kr)

    def test_categories(self):
        matches = [
            KeyMatch("A", "A1", "A1a"),
            KeyMatch("B", "B1", "B1a"),
        ]
        kr = KeysResult(smiles="CC", matches=matches)
        cats, subs, subsubs = kr.categories()
        assert cats == {"A", "B"}
        assert subs == {"A1", "B1"}
        assert subsubs == {"A1a", "B1a"}

    def test_to_dict(self):
        matches = [KeyMatch("A", "A1", "A1a")]
        kr = KeysResult(smiles="CC", matches=matches)
        d = kr.to_dict()
        assert d["smiles"] == "CC"
        assert len(d["matches"]) == 1
        assert d["matches"][0]["category"] == "A"

    def test_str_with_matches(self):
        matches = [KeyMatch("A", "A1", "A1a")]
        kr = KeysResult(smiles="CC", matches=matches)
        s = str(kr)
        assert "CC" in s
        assert "A" in s
