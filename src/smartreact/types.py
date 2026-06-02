from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class ReactionTemplate:
    """Single reaction template loaded from the TSV file."""

    smarts: str
    reactant_categories: tuple[str, str]
    name: str
    template_id: int = 0


@dataclass(frozen=True)
class ReactionResult:
    """Products from one template applied to one reactant pair."""

    reactant_a: str
    reactant_b: str
    reaction_name: str
    products: list[str]

    def __str__(self) -> str:
        return self._format()

    def __repr__(self) -> str:
        return self._format()

    def _format(self) -> str:
        products_str = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(self.products))
        if not products_str:
            products_str = "  (no products)"
        return (
            f"Reaction: {self.reaction_name}\n"
            f"Reactant A: {self.reactant_a}\n"
            f"Reactant B: {self.reactant_b}\n"
            f"Products ({len(self.products)}):\n{products_str}"
        )


@dataclass(frozen=True)
class KeyMatch:
    category: str
    subcategory: str
    subsubcategory: str


@dataclass
class KeysResult:
    """Classification result for one SMILES based on Key rules."""

    smiles: str
    matches: list[KeyMatch]

    def __str__(self) -> str:
        return self._format()

    def __repr__(self) -> str:
        return self._format()

    def _format(self) -> str:
        if not self.matches:
            return f"Smiles: {self.smiles}\n(no matches)"
        cats = sorted({m.category for m in self.matches})
        subs = sorted({m.subcategory for m in self.matches})
        subsubs = sorted({m.subsubcategory for m in self.matches})
        return (
            f"Smiles: {self.smiles}\n"
            f"Categories: {', '.join(cats)}\n"
            f"Subcategories: {', '.join(subs)}\n"
            f"Subsubcategories: {', '.join(subsubs)}"
        )

    def categories(self) -> tuple[set[str], set[str], set[str]]:
        cats = {m.category for m in self.matches}
        subs = {m.subcategory for m in self.matches}
        subsubs = {m.subsubcategory for m in self.matches}
        return cats, subs, subsubs

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "matches": [
                {
                    "category": m.category,
                    "subcategory": m.subcategory,
                    "subsubcategory": m.subsubcategory,
                }
                for m in self.matches
            ],
        }


@dataclass(frozen=True)
class Rule:
    """One Key assignment rule: SMARTS -> (category, subcategory, subsubcategory)."""

    category: str
    subcategory: str
    subsubcategory: str
    qmol: Chem.Mol


__all__ = [
    "ReactionTemplate",
    "ReactionResult",
    "KeyMatch",
    "KeysResult",
    "Rule",
]
