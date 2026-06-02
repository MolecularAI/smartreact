from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from importlib.resources import files

from rdkit import Chem, RDLogger

from .types import KeyMatch, KeysResult, Rule

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

_DATA = files("smartreact") / "data"


def load_rules() -> list[Rule]:
    """Load Key assignment rules from the bundled keys file.

    The file is whitespace-separated with a header line.
    Columns: Level1 (category), Level2 (subcategory),
    Level3 (subsubcategory), SMARTS.
    """
    rules: list[Rule] = []
    text = (_DATA / "keys.txt").read_text(encoding="utf-8")

    lines = text.splitlines()
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split(None, 3)
        if len(cols) < 4:
            continue
        cat, subcat, subsubcat, smarts = cols
        qmol = Chem.MolFromSmarts(smarts)
        if qmol is None:
            continue
        rules.append(Rule(cat, subcat, subsubcat, qmol))

    return rules


_WORKER_RULES_CACHE: list[Rule] | None = None


def _classify_smiles_worker(smiles: str) -> KeysResult:
    """Worker function for parallel classification (loads rules once per process)."""
    global _WORKER_RULES_CACHE

    if _WORKER_RULES_CACHE is None:
        _WORKER_RULES_CACHE = load_rules()

    rules = _WORKER_RULES_CACHE

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return KeysResult(smiles=smiles, matches=[])

    matches: list[KeyMatch] = []
    for r in rules:
        if mol.HasSubstructMatch(r.qmol):
            matches.append(KeyMatch(r.category, r.subcategory, r.subsubcategory))
    return KeysResult(smiles=smiles, matches=matches)


class KeysLibrary:
    """Low-level classifier backed by a pre-loaded rule list."""

    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules

    def classify_smiles(self, smiles: str) -> KeysResult:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return KeysResult(smiles=smiles, matches=[])
        matches: list[KeyMatch] = []
        for r in self.rules:
            if mol.HasSubstructMatch(r.qmol):
                matches.append(KeyMatch(r.category, r.subcategory, r.subsubcategory))
        return KeysResult(smiles=smiles, matches=matches)


class KeyGenerator:
    """
    High-level facade to classify SMILES using the bundled Key rules.

    Parameters
    ----------
    n_cores:
        Number of cores for parallel classification.
        ``-1`` or ``0`` means "all available cores".
        Default is ``1`` (sequential).
    """

    def __init__(self, n_cores: int = 1) -> None:
        self.rules = load_rules()
        self._lib = KeysLibrary(self.rules)

        if n_cores in (-1, 0):
            self.n_cores = os.cpu_count() or 1
        else:
            self.n_cores = max(1, int(n_cores))

        self._pool: ProcessPoolExecutor | None = None

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.n_cores)
        return self._pool

    def close(self) -> None:
        """Shut down the worker pool, waiting for any running tasks to finish.

        Called automatically on context-manager exit. Safe to call more than once.
        """
        pool = getattr(self, "_pool", None)
        if pool is not None:
            self._pool = None
            pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> KeyGenerator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Must not raise: module globals may already be None during interpreter shutdown.
        try:
            pool = getattr(self, "_pool", None)
            if pool is not None:
                self._pool = None
                pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def classify(self, smiles: str) -> KeysResult:
        """Classify a single SMILES string."""
        return self._lib.classify_smiles(smiles)

    def classify_many(
        self,
        smiles_list: Iterable[str],
        parallel: bool = True,
        min_parallel_threshold: int = 10,
    ) -> list[KeysResult]:
        """
        Classify multiple SMILES strings.

        Parameters
        ----------
        smiles_list:
            Iterable of SMILES strings to classify.
        parallel:
            If ``True`` and ``n_cores > 1``, use parallel processing.
        min_parallel_threshold:
            Minimum number of SMILES to warrant parallel processing.
            Below this threshold, sequential processing is used.

        Returns
        -------
        list[KeysResult]
            Results in the same order as input *smiles_list*.
        """
        smiles = list(smiles_list)

        if not parallel or self.n_cores == 1 or len(smiles) < min_parallel_threshold:
            return [self.classify(s) for s in smiles]

        results_dict: dict[int, KeysResult] = {}
        executor = self._get_pool()
        future_to_idx = {
            executor.submit(_classify_smiles_worker, s): (idx, s) for idx, s in enumerate(smiles)
        }
        try:
            for future in as_completed(future_to_idx):
                idx, s = future_to_idx[future]
                try:
                    results_dict[idx] = future.result()
                except Exception:
                    results_dict[idx] = KeysResult(smiles=s, matches=[])
        except BrokenExecutor:
            # A worker crashed. Clear the broken pool so the next call spawns a
            # fresh one, then re-raise.
            self._pool = None
            raise

        return [results_dict[i] for i in range(len(smiles))]


__all__ = [
    "load_rules",
    "KeysLibrary",
    "KeyGenerator",
]
