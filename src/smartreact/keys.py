from __future__ import annotations

from typing import Any


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


__all__ = ["extract_key_strings", "orders_for_template"]
