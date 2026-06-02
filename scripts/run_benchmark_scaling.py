"""
Generate scaling benchmark data: pairs/s vs number of templates,
for key-filtered and brute-force enumeration.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from smartreact.keygen import KeyGenerator
from smartreact.keys import extract_key_strings, orders_for_template
from smartreact.templates import load_templates
from smartreact.types import ReactionResult, ReactionTemplate

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Reaction helpers
# ---------------------------------------------------------------------------

_RXN_CACHE: dict[str, AllChem.ChemicalReaction] = {}


def _get_rxn(smarts: str) -> AllChem.ChemicalReaction | None:
    rxn = _RXN_CACHE.get(smarts)
    if rxn is None:
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is not None:
            _RXN_CACHE[smarts] = rxn
    return rxn


def _collect_products(
    rxn: AllChem.ChemicalReaction,
    m1: Chem.Mol,
    m2: Chem.Mol,
    orders: list[int],
) -> set[str]:
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
                    if s:
                        products.add(s)
                except Exception:
                    continue
    return products


# ---------------------------------------------------------------------------
# Strategy A: key-filtered
# ---------------------------------------------------------------------------

def enumerate_pair_filtered(
    s1: str, s2: str,
    m1: Chem.Mol, m2: Chem.Mol,
    keys1: set[str], keys2: set[str],
    templates: list[ReactionTemplate],
) -> list[ReactionResult]:
    results: list[ReactionResult] = []
    for templ in templates:
        orders = orders_for_template(keys1, keys2, templ.reactant_categories)
        if not orders:
            continue
        rxn = _get_rxn(templ.smarts)
        if rxn is None:
            continue
        products = _collect_products(rxn, m1, m2, orders)
        if products:
            ra, rb = tuple(sorted([s1, s2]))
            results.append(ReactionResult(ra, rb, templ.name, sorted(products)))
    return results


def run_filtered(
    pairs: list[tuple[str, str]],
    templates: list[ReactionTemplate],
    keygen: KeyGenerator,
) -> float:
    """Return elapsed seconds."""
    t0 = time.perf_counter()
    unique_smiles = list({s for pair in pairs for s in pair})
    smiles_mols = {s: Chem.MolFromSmiles(s) for s in unique_smiles}
    key_results = keygen.classify_many(unique_smiles, parallel=False)
    smiles_keys = {res.smiles: extract_key_strings(res) for res in key_results}
    for s1, s2 in pairs:
        m1, m2 = smiles_mols[s1], smiles_mols[s2]
        if m1 is None or m2 is None:
            continue
        enumerate_pair_filtered(s1, s2, m1, m2, smiles_keys[s1], smiles_keys[s2], templates)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Strategy B: brute-force
# ---------------------------------------------------------------------------

def enumerate_pair_brute(
    s1: str, s2: str,
    m1: Chem.Mol, m2: Chem.Mol,
    templates: list[ReactionTemplate],
) -> list[ReactionResult]:
    results: list[ReactionResult] = []
    for templ in templates:
        rxn = _get_rxn(templ.smarts)
        if rxn is None:
            continue
        products = _collect_products(rxn, m1, m2, [0, 1])
        if products:
            ra, rb = tuple(sorted([s1, s2]))
            results.append(ReactionResult(ra, rb, templ.name, sorted(products)))
    return results


def run_brute(
    pairs: list[tuple[str, str]],
    templates: list[ReactionTemplate],
) -> float:
    """Return elapsed seconds."""
    t0 = time.perf_counter()
    unique_smiles = list({s for pair in pairs for s in pair})
    smiles_mols = {s: Chem.MolFromSmiles(s) for s in unique_smiles}
    for s1, s2 in pairs:
        m1, m2 = smiles_mols[s1], smiles_mols[s2]
        if m1 is None or m2 is None:
            continue
        enumerate_pair_brute(s1, s2, m1, m2, templates)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

N_PAIRS  = 500
N_REPEATS = 3
SEED     = 42
INPUT    = ROOT / "data" / "building_blocks.parquet"
OUTPUT   = ROOT / "data" / "paper" / "benchmark_scaling.csv"


def main() -> None:
    import statistics

    # ---- load SMILES ---------------------------------------------------------
    smiles_pool = pd.read_parquet(INPUT)["smiles"].dropna().tolist()
    print(f"SMILES pool : {len(smiles_pool):,}")

    # ---- load templates ------------------------------------------------------
    all_templates = load_templates(reaction_list="all")
    n_total = len(all_templates)
    print(f"Templates   : {n_total}")

    # ---- fixed pair sample ---------------------------------------------------
    rng = random.Random(SEED)
    pairs_idx: set[tuple[int, int]] = set()
    while len(pairs_idx) < N_PAIRS:
        a, b = rng.sample(range(len(smiles_pool)), 2)
        pairs_idx.add((min(a, b), max(a, b)))
    pairs: list[tuple[str, str]] = [
        (smiles_pool[a], smiles_pool[b]) for a, b in sorted(pairs_idx)
    ]
    print(f"Pairs       : {len(pairs)}")

    # ---- choose template counts to sweep -------------------------------------
    candidates = [100, 500, 1000, 5000, n_total]
    template_counts = sorted({min(c, n_total) for c in candidates})
    print(f"Sweep points: {template_counts}\n")

    # ---- warm up caches ------------------------------------------------------
    keygen = KeyGenerator(n_cores=1)
    warmup = pairs[:10]
    run_filtered(warmup, all_templates, keygen)
    run_brute(warmup, all_templates)
    print("Warm-up done.\n")

    # ---- sweep ---------------------------------------------------------------
    rng_sweep = random.Random(SEED + 1)
    records: list[dict] = []

    for n in template_counts:
        pps_f_list: list[float] = []
        pps_b_list: list[float] = []

        for rep in range(N_REPEATS):
            subset = all_templates if n == n_total else rng_sweep.sample(all_templates, n)

            t_f = run_filtered(pairs, subset, keygen)
            t_b = run_brute(pairs, subset)

            pps_f_list.append(len(pairs) / t_f)
            pps_b_list.append(len(pairs) / t_b)

        mean_f = statistics.mean(pps_f_list)
        std_f  = statistics.stdev(pps_f_list) if len(pps_f_list) > 1 else 0.0
        mean_b = statistics.mean(pps_b_list)
        std_b  = statistics.stdev(pps_b_list) if len(pps_b_list) > 1 else 0.0

        records.append({
            "n_templates":   n,
            "filtered_mean": mean_f,
            "filtered_std":  std_f,
            "brute_mean":    mean_b,
            "brute_std":     std_b,
        })

        print(
            f"n={n:5d}  "
            f"filtered {mean_f:8.1f} ± {std_f:6.1f}  "
            f"brute {mean_b:8.1f} ± {std_b:6.1f}  pairs/s"
        )

    # ---- save ----------------------------------------------------------------
    pd.DataFrame(records).to_csv(OUTPUT, index=False)
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
