from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .types import ReactionTemplate


def extract_key_strings(res: Any) -> set[str]:
    """
    Turn a KeyGenerator classification result into a set of subsubcategory
    (Level 3) key strings.

    Template ``reactant_categories`` values reference Level 3 names, so only
    subsubcategories are returned.  Including higher levels would cause
    false-positive matches (e.g. the Level 1 name ``"Phenol"`` also appears
    as a Level 3 name for the strict-phenol rule, so a heteroaromatic phenol
    whose Level 1 is ``"Phenol"`` would incorrectly match that template).

    Supports objects with a ``.categories()`` method returning
    ``(cats, subs, subsubs)`` or a ``.matches`` list with a
    ``.subsubcategory`` attribute.
    """
    keys: set[str] = set()
    if res is None:
        return keys

    if hasattr(res, "categories") and callable(res.categories):
        try:
            _cats, _subs, subsubs = res.categories()
        except Exception:
            subsubs = None
        if subsubs:
            for v in subsubs:
                if v:
                    keys.add(str(v))
        if keys:
            return keys

    for m in getattr(res, "matches", []) or []:
        v = getattr(m, "subsubcategory", None)
        if v:
            keys.add(str(v))

    return keys


def orders_for_template(
    keys1: set[str],
    keys2: set[str],
    reactant_categories: tuple[str, str],
) -> list[int]:
    """
    Decide which orientation(s) of ``(mol1, mol2)`` satisfy the reactant_categories spec.

    Returns ``[]``, ``[0]``, ``[1]``, or ``[0, 1]``.
    """
    left_req, right_req = reactant_categories
    orders: list[int] = []
    if left_req in keys1 and right_req in keys2:
        orders.append(0)
    if left_req in keys2 and right_req in keys1:
        orders.append(1)
    return orders


def build_template_index(templates: Sequence[ReactionTemplate]) -> dict[tuple[str, str], list[int]]:
    """Map each (left_req, right_req) requirement pair to the template indices that need it.

    Built once from the template library. Lets :func:`candidate_templates` find
    applicable templates via lookups over a pair's own key sets, instead of
    scanning every template for every reactant pair.
    """
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, templ in enumerate(templates):
        index[templ.reactant_categories].append(i)
    return dict(index)


def candidate_templates(
    keys1: set[str],
    keys2: set[str],
    index: dict[tuple[str, str], list[int]],
) -> dict[int, list[int]]:
    """Find template indices whose reactant_categories are satisfied by (keys1, keys2).

    Equivalent to calling :func:`orders_for_template` for every template in the
    library and keeping the non-empty results, but costs O(|keys1| * |keys2|)
    index lookups rather than O(len(templates)).

    Returns
    -------
    dict[int, list[int]]
        Maps template index to its list of valid orders (``[0]``, ``[1]``, or
        ``[0, 1]``), keyed and ordered exactly as :func:`orders_for_template`
        would produce for that template.
    """
    candidates: dict[int, set[int]] = defaultdict(set)
    for ka in keys1:
        for kb in keys2:
            for t in index.get((ka, kb), ()):
                candidates[t].add(0)
    for ka in keys2:
        for kb in keys1:
            for t in index.get((ka, kb), ()):
                candidates[t].add(1)
    return {t: sorted(orders) for t, orders in candidates.items()}


__all__ = [
    "extract_key_strings",
    "orders_for_template",
    "build_template_index",
    "candidate_templates",
]
