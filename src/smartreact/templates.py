from __future__ import annotations

import csv
import io
import logging
from collections.abc import Iterable
from importlib.resources import files
from typing import Literal

from .types import ReactionTemplate

logger = logging.getLogger(__name__)

_DATA = files("smartreact") / "data"


def load_templates(
    reaction_list: Literal["all"] | Iterable[str],
) -> list[ReactionTemplate]:
    """
    Load and filter reaction templates from the bundled template file.

    Required columns: reaction_name, template_id, reaction_template, reactant_categories.
    """
    text = (_DATA / "templates.txt").read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")

    required = {"reaction_template", "reactant_categories", "reaction_name", "template_id"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Templates file must contain columns: " + ", ".join(sorted(required)))

    if reaction_list != "all":
        if isinstance(reaction_list, str):
            raise ValueError("reaction_list must be 'all' or an iterable of reaction names.")
        selected = set(reaction_list)
    else:
        selected = None

    templates: list[ReactionTemplate] = []
    for row in reader:
        rn = row["reaction_name"]
        if selected is not None and rn not in selected:
            continue
        raw = row["reactant_categories"]
        if "." not in raw:
            logger.error(
                "Skipping template %r: reactant_categories %r has no '.' separator.", rn, raw
            )
            continue
        left, right = raw.split(".", 1)
        templates.append(
            ReactionTemplate(
                smarts=row["reaction_template"],
                reactant_categories=(left.strip(), right.strip()),
                name=rn,
                template_id=int(row["template_id"]),
            )
        )

    if not templates:
        raise ValueError("No usable reaction templates after filtering.")

    return templates


__all__ = ["load_templates"]
