from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def detect_format(input_str: str) -> str:
    """Detect molecule format from string content.

    Returns ``"inchi"``, ``"sdf"``, or ``"smiles"``.
    """
    stripped = input_str.strip()
    if stripped.startswith("InChI="):
        return "inchi"
    if "V2000" in stripped or "V3000" in stripped or "M  END" in stripped:
        return "sdf"
    return "smiles"


def to_smiles(input_str: str, fmt: str = "auto") -> str | None:
    """Convert a molecule string to canonical SMILES.

    Parameters
    ----------
    input_str:
        Molecule in one of the supported formats.
    fmt:
        ``"smiles"``, ``"inchi"``, ``"sdf"`` (single mol block), or
        ``"auto"`` (detect from content).

    Returns
    -------
    str or None
        Canonical SMILES, or ``None`` if parsing fails.
    """
    if not input_str or not input_str.strip():
        return None

    if fmt == "auto":
        fmt = detect_format(input_str)

    if fmt == "smiles":
        mol = Chem.MolFromSmiles(input_str.strip())
    elif fmt == "inchi":
        mol = Chem.MolFromInchi(input_str.strip(), sanitize=True, removeHs=True)
    elif fmt in ("sdf", "mol"):
        mol = Chem.MolFromMolBlock(input_str, sanitize=True, removeHs=True)
    else:
        logger.error("Unsupported format: %s", fmt)
        return None

    if mol is None:
        logger.warning("Failed to parse molecule (fmt=%s).", fmt)
        return None

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def read_sdf_block(text: str) -> list[str]:
    """Parse a multi-molecule SDF string and return canonical SMILES.

    Molecules that fail to parse are skipped with a warning.
    """
    supplier = Chem.SDMolSupplier()
    supplier.SetData(text, sanitize=True, removeHs=True)
    smiles: list[str] = []
    for i, mol in enumerate(supplier):
        if mol is None:
            logger.warning("Skipping molecule %d in SDF block: parse failed.", i)
            continue
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if smi:
            smiles.append(smi)
    return smiles


def read_sdf_file(path: str | Path) -> list[str]:
    """Read a multi-molecule SDF file and return canonical SMILES.

    Molecules that fail to parse are skipped with a warning.
    """
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
    smiles: list[str] = []
    for i, mol in enumerate(supplier):
        if mol is None:
            logger.warning("Skipping molecule %d in %s: parse failed.", i, path)
            continue
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if smi:
            smiles.append(smi)
    return smiles


__all__ = ["detect_format", "to_smiles", "read_sdf_block", "read_sdf_file"]
