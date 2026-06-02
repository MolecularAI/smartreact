from __future__ import annotations

import itertools
import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from functools import partial
from typing import Literal

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from .cleaning import clean_smiles as _clean_smiles
from .keygen import KeyGenerator
from .keys import extract_key_strings, orders_for_template
from .templates import load_templates
from .types import ReactionResult, ReactionTemplate

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

_RXN_CACHE: dict[str, AllChem.ChemicalReaction] = {}
_WORKER_TEMPLATES: list[ReactionTemplate] | None = None


def _get_rxn(smarts: str) -> AllChem.ChemicalReaction | None:
    rxn = _RXN_CACHE.get(smarts)
    if rxn is None:
        rxn = AllChem.ReactionFromSmarts(smarts)  # type: ignore[attr-defined]
        if rxn is not None:
            _RXN_CACHE[smarts] = rxn
    return rxn


def _collect_products(
    rxn: AllChem.ChemicalReaction,
    m1: Chem.Mol,
    m2: Chem.Mol,
    orders: Sequence[int],
) -> set[str]:
    """Run a reaction in the given orientations and collect unique product SMILES."""
    products: set[str] = set()
    for order in orders:
        args = (m1, m2) if order == 0 else (m2, m1)
        try:
            outcomes = rxn.RunReactants(args)
        except Exception:
            continue
        for outcome in outcomes or []:
            for pmol in outcome or []:
                if pmol is None:
                    continue
                try:
                    s = Chem.MolToSmiles(pmol, canonical=True, isomericSmiles=True)
                    if not s:
                        raise ValueError("MolToSmiles returned empty string")
                except Exception:
                    try:
                        r_smiles = [Chem.MolToSmiles(m) for m in args]
                        rxn_smarts = AllChem.ReactionToCXSmarts(rxn)  # type: ignore[attr-defined]
                    except Exception:
                        r_smiles = ["?", "?"]
                        rxn_smarts = "?"
                    logger.error(
                        "Product mol produced invalid SMILES. Reactants: %s, SMARTS: %s",
                        r_smiles,
                        rxn_smarts,
                    )
                    continue
                products.add(s)
    return products


def _worker_init(
    templates: list[ReactionTemplate] | None = None,
    log_file: str | None = None,
) -> None:
    """Configure templates and logging in worker processes."""
    global _WORKER_TEMPLATES
    if templates is not None:
        _WORKER_TEMPLATES = templates
    if not log_file:
        return
    try:
        handlers: list[logging.Handler] = [
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    except Exception:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def _worker_run_pair(
    s1: str,
    s2: str,
    keys1: set[str],
    keys2: set[str],
) -> list[ReactionResult]:
    """Run all applicable templates for one pair in a worker process.

    Templates are pre-loaded into ``_WORKER_TEMPLATES`` via the pool initializer,
    so they are not serialized per-task.  Molecules are parsed once per call.
    """
    assert _WORKER_TEMPLATES is not None, "_WORKER_TEMPLATES not initialised in worker"

    m1 = Chem.MolFromSmiles(s1)
    m2 = Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return []

    ra, rb = tuple(sorted([s1, s2]))
    results: list[ReactionResult] = []
    for templ in _WORKER_TEMPLATES:
        orders = orders_for_template(keys1, keys2, templ.reactant_categories)
        if not orders:
            continue
        rxn = _get_rxn(templ.smarts)
        if rxn is None:
            continue
        products = _collect_products(rxn, m1, m2, orders)
        if not products:
            logger.debug(
                "No valid products from %s reaction (template_id=%d): %s + %s. "
                "Keys matched (%s) but reaction failed to produce valid SMILES.",
                templ.name,
                templ.template_id,
                s1,
                s2,
                templ.reactant_categories,
            )
            continue
        results.append(
            ReactionResult(
                reactant_a=ra, reactant_b=rb, reaction_name=templ.name, products=sorted(products)
            )
        )
    return results


def _worker_run_pairs_batch(
    pairs: list[tuple[str, str]],
    smiles_keys: dict[str, set[str]],
) -> list[ReactionResult]:
    """Process a batch of pairs in a single worker call.

    Receiving a whole batch minimises IPC round-trips: the main process
    sends exactly ``n_cores`` tasks instead of one per pair.
    """
    results: list[ReactionResult] = []
    for s1, s2 in pairs:
        results.extend(_worker_run_pair(s1, s2, smiles_keys[s1], smiles_keys[s2]))
    return results


class ReactionEnumerator:
    """
    Enumerate products for pairs of SMILES using SMARTS templates + Keys.

    Parameters
    ----------
    reaction_list:
        Iterable of reaction names to use, or ``'all'`` (default).
        Defaults to all available templates.
    n_cores:
        Used only when ``parallel=True`` in ``enumerate_pairs()``.
        ``-1`` or ``0`` means "all available cores".
    log_file:
        Optional path to log file for worker processes (append mode).
    """

    def __init__(
        self,
        reaction_list: Literal["all"] | Iterable[str] = "all",
        n_cores: int = 1,
        log_file: str | None = None,
    ) -> None:
        if n_cores in (-1, 0):
            self.n_cores = os.cpu_count() or 1
        else:
            self.n_cores = max(1, int(n_cores))

        self.keygen = KeyGenerator(n_cores=self.n_cores)
        self.templates: list[ReactionTemplate] = load_templates(reaction_list=reaction_list)
        self.log_file = log_file
        self._pool: ProcessPoolExecutor | None = None

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_cores,
                initializer=partial(_worker_init, self.templates, self.log_file),
            )
        return self._pool

    def close(self) -> None:
        """Shut down the worker pool, waiting for any running tasks to finish.

        Called automatically on context-manager exit. Safe to call more than once.
        """
        keygen = getattr(self, "keygen", None)
        if keygen is not None:
            keygen.close()
        pool = getattr(self, "_pool", None)
        if pool is not None:
            self._pool = None
            pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> ReactionEnumerator:
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

    def _keys_for_smiles(self, smiles: str) -> set[str]:
        return extract_key_strings(self.keygen.classify(str(smiles)))

    def _enumerate_pair_from_keys(
        self,
        s1: str,
        s2: str,
        keys1: set[str],
        keys2: set[str],
    ) -> list[ReactionResult]:
        m1 = Chem.MolFromSmiles(s1)
        m2 = Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None:
            return []

        results: list[ReactionResult] = []
        for templ in self.templates:
            orders = orders_for_template(keys1, keys2, templ.reactant_categories)
            if not orders:
                continue

            rxn = _get_rxn(templ.smarts)
            if rxn is None:
                continue

            products = _collect_products(rxn, m1, m2, orders)
            if not products:
                logger.debug(
                    "No valid products from %s reaction (template_id=%d): %s + %s. "
                    "Keys matched (%s: %s + %s) but reaction failed.",
                    templ.name,
                    templ.template_id,
                    s1,
                    s2,
                    templ.reactant_categories,
                    keys1,
                    keys2,
                )
                continue

            ra, rb = tuple(sorted([s1, s2]))
            results.append(
                ReactionResult(
                    reactant_a=ra,
                    reactant_b=rb,
                    reaction_name=templ.name,
                    products=sorted(products),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enumerate_pair(self, s1: str, s2: str, clean_smiles: bool = False) -> list[ReactionResult]:
        """
        Enumerate all products for a single pair of SMILES across all
        configured reaction templates.

        Parameters
        ----------
        s1, s2:
            SMILES strings for the two reactants.
        clean_smiles:
            If ``True``, standardize each SMILES before enumeration.
            Returns ``[]`` if either molecule cannot be parsed after cleaning.
        """
        if clean_smiles:
            c1 = _clean_smiles(s1)
            c2 = _clean_smiles(s2)
            if c1 is None or c2 is None:
                logger.warning(
                    "Skipping pair (%s, %s): cleaning failed for %s.",
                    s1,
                    s2,
                    s1 if c1 is None else s2,
                )
                return []
            s1, s2 = c1, c2

        k1 = self._keys_for_smiles(s1)
        k2 = self._keys_for_smiles(s2)
        return self._enumerate_pair_from_keys(s1, s2, k1, k2)

    def products_for_pair(self, s1: str, s2: str, clean_smiles: bool = False) -> list[str]:
        """Return only unique product SMILES for a pair.

        Parameters
        ----------
        clean_smiles:
            If ``True``, standardize each SMILES before enumeration.
        """
        res = self.enumerate_pair(s1, s2, clean_smiles=clean_smiles)
        return sorted({p for r in res for p in r.products})

    def _process_chunk(
        self,
        pair_list: list[tuple[str, str]],
        parallel: bool,
        clean_smiles: bool,
        precomputed_keys: dict[str, set[str]] | None,
    ) -> Iterator[ReactionResult]:
        """Enumerate reactions for a materialised list of pairs and yield results."""
        if not pair_list:
            return

        if clean_smiles:
            unique_dirty = list({s for pair in pair_list for s in pair})
            if parallel and self.n_cores > 1:
                with ProcessPoolExecutor(max_workers=self.n_cores) as ex:
                    clean_results: dict[str, str | None] = dict(
                        zip(unique_dirty, ex.map(_clean_smiles, unique_dirty), strict=True)
                    )
            else:
                clean_results = {s: _clean_smiles(s) for s in unique_dirty}

            cleaned: list[tuple[str, str]] = []
            for a, b in pair_list:
                ca, cb = clean_results[a], clean_results[b]
                if ca is None or cb is None:
                    logger.warning(
                        "Skipping pair (%s, %s): cleaning failed for %s.",
                        a,
                        b,
                        a if ca is None else b,
                    )
                    continue
                cleaned.append((ca, cb))
            pair_list = cleaned
            if not pair_list:
                return

        unique_smiles = list({s for pair in pair_list for s in pair})

        if precomputed_keys is not None:
            missing = [s for s in unique_smiles if s not in precomputed_keys]
            if missing:
                logger.debug(
                    "%d SMILES not found in precomputed_keys; computing keys now.",
                    len(missing),
                )
                missing_results = self.keygen.classify_many(missing, parallel=(self.n_cores > 1))
                extra = {res.smiles: extract_key_strings(res) for res in missing_results}
                smiles_keys: dict[str, set[str]] = {**precomputed_keys, **extra}
            else:
                smiles_keys = precomputed_keys
        else:
            key_results = self.keygen.classify_many(unique_smiles, parallel=(self.n_cores > 1))
            smiles_keys = {res.smiles: extract_key_strings(res) for res in key_results}

        if not parallel or self.n_cores == 1:
            for a, b in pair_list:
                yield from self._enumerate_pair_from_keys(a, b, smiles_keys[a], smiles_keys[b])
            return

        chunk_size = max(1, len(pair_list) // self.n_cores)
        batches = [pair_list[i : i + chunk_size] for i in range(0, len(pair_list), chunk_size)]

        ex = self._get_pool()
        futures = [ex.submit(_worker_run_pairs_batch, batch, smiles_keys) for batch in batches]
        try:
            for fut in futures:
                yield from fut.result()
        except BrokenExecutor:
            # A worker crashed (e.g. OOM inside RDKit). Clear the broken pool so
            # the next call automatically spawns a fresh one, then re-raise.
            self._pool = None
            raise
        finally:
            # Cancel any futures not yet consumed (e.g. generator abandoned early).
            # cancel() is a no-op on already-running futures, so this is always safe.
            for fut in futures:
                fut.cancel()

    def enumerate_pairs_lazy(
        self,
        pairs: Iterable[tuple[str, str]],
        parallel: bool = False,
        clean_smiles: bool = False,
        precomputed_keys: dict[str, set[str]] | None = None,
        chunk_size: int = 50_000,
    ) -> Iterator[ReactionResult]:
        """
        Enumerate reactions for many reactant pairs, yielding results as each
        chunk completes.  Memory usage is bounded by ``chunk_size`` rather than
        the total number of pairs.

        Parameters
        ----------
        pairs:
            Iterable of ``(smiles_a, smiles_b)``.  Need not be materialised in
            memory — a generator is fine.
        parallel:
            If ``True``, use a ``ProcessPoolExecutor`` with ``self.n_cores`` workers.
        clean_smiles:
            If ``True``, standardize each SMILES before enumeration.
        precomputed_keys:
            Optional mapping of SMILES → key strings (see
            :func:`~smartreact.preprocessing.preprocess_smiles`).
        chunk_size:
            Number of pairs processed per iteration.  Larger values amortise
            key-classification overhead at the cost of higher peak memory.
        """
        pairs_iter = iter(pairs)
        while chunk := [(str(a), str(b)) for a, b in itertools.islice(pairs_iter, chunk_size)]:
            yield from self._process_chunk(chunk, parallel, clean_smiles, precomputed_keys)

    def enumerate_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        parallel: bool = False,
        clean_smiles: bool = False,
        precomputed_keys: dict[str, set[str]] | None = None,
    ) -> list[ReactionResult]:
        """
        Enumerate reactions for many reactant pairs.

        Parameters
        ----------
        pairs:
            Iterable of ``(smiles_a, smiles_b)``.
        parallel:
            If ``True``, use a ``ProcessPoolExecutor`` with ``self.n_cores`` workers.
        clean_smiles:
            If ``True``, standardize each SMILES before enumeration. Pairs where
            either molecule fails cleaning are skipped.
        precomputed_keys:
            Optional mapping of SMILES to pre-classified Key strings, as returned
            by :func:`~smartreact.preprocessing.preprocess_smiles`. Avoids
            re-classifying molecules that have already been processed. Keys for
            any SMILES not present in the mapping are computed on the fly.
        """
        pair_list = [(str(a), str(b)) for a, b in pairs]
        return list(self._process_chunk(pair_list, parallel, clean_smiles, precomputed_keys))


__all__ = ["ReactionEnumerator"]
