from __future__ import annotations

import logging

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def clean_smiles(smiles: str) -> str | None:
    """Standardize a SMILES string using RDKit molecule standardization.

    Steps applied in order:

    1. Remove isotope labels
    2. Normalize functional groups, sanitize, remove explicit Hs
    3. Strip known solvents and salts
    4. Keep the largest remaining fragment
    5. Neutralize charges

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    str or None
        Canonical SMILES of the cleaned molecule,
        or None if the input cannot be parsed.
    """
    if not smiles:
        logger.warning("Received empty SMILES string for cleaning.")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Could not parse SMILES for cleaning: %s", smiles)
        return None

    rdMolStandardize.IsotopeParentInPlace(mol)
    rdMolStandardize.CleanupInPlace(mol)
    rdMolStandardize.RemoveFragmentsInPlace(mol)
    rdMolStandardize.FragmentParentInPlace(mol, skipStandardize=True)
    rdMolStandardize.Uncharger().unchargeInPlace(mol)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


__all__ = ["clean_smiles"]
