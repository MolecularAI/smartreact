from __future__ import annotations

import csv
import logging
from pathlib import Path

from .cleaning import clean_smiles as _clean_smiles_fn
from .keygen import KeyGenerator
from .keys import extract_key_strings

logger = logging.getLogger(__name__)

_KEYS_SEPARATOR = "|"


def preprocess_smiles(
    smiles_list: list[str],
    keygen: KeyGenerator,
    clean_smiles: bool = False,
) -> dict[str, set[str]]:
    """Classify a list of SMILES and return a mapping of SMILES to their Keys.

    Use this to precompute Keys once for a large library, then pass the result
    to :meth:`~smartreact.ReactionEnumerator.enumerate_pairs` via the
    ``precomputed_keys`` argument to avoid repeated classification.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to preprocess.
    keygen : KeyGenerator
        Classifier used to assign Keys to each molecule.
    clean_smiles : bool, optional
        If True, standardize each SMILES before classification (default False).
        SMILES that fail parsing during cleaning are skipped with a warning.

    Returns
    -------
    dict[str, set[str]]
        Maps each (optionally cleaned) SMILES to its set of Key strings.
    """
    processed: list[str] = []
    for smi in smiles_list:
        if clean_smiles:
            cleaned = _clean_smiles_fn(smi)
            if cleaned is None:
                continue
            processed.append(cleaned)
        else:
            processed.append(smi)

    if not processed:
        return {}

    # Deduplicate while preserving order
    unique_smiles = list(dict.fromkeys(processed))
    key_results = keygen.classify_many(unique_smiles, parallel=(keygen.n_cores > 1))
    return {res.smiles: extract_key_strings(res) for res in key_results}


def save_preprocessed(keys_map: dict[str, set[str]], path: str | Path) -> None:
    """Save a precomputed SMILES→Keys mapping to a CSV file.

    The CSV has two columns:

    - ``smiles``: the SMILES string
    - ``keys``: pipe-separated Key strings (empty string if no keys matched)

    Parameters
    ----------
    keys_map : dict[str, set[str]]
        Mapping from SMILES to Key strings, as returned by :func:`preprocess_smiles`.
    path : str or Path
        Destination file path (e.g. ``"preprocessed.csv"``).
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "keys"])
        for smi, keys in keys_map.items():
            writer.writerow([smi, _KEYS_SEPARATOR.join(sorted(keys))])


def load_preprocessed(path: str | Path) -> dict[str, set[str]]:
    """Load a precomputed SMILES→Keys mapping from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to a CSV file written by :func:`save_preprocessed`.

    Returns
    -------
    dict[str, set[str]]
        Mapping from SMILES to Key strings.
    """
    result: dict[str, set[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys_str = row.get("keys") or ""
            keys = set(keys_str.split(_KEYS_SEPARATOR)) - {""} if keys_str else set()
            result[row["smiles"]] = keys
    return result


__all__ = ["preprocess_smiles", "save_preprocessed", "load_preprocessed"]
