"""
Benchmark parallel scaling of enumeration throughput.

Cores sweep — pairs/s vs n_cores (powers of 2 up to MAX_CORES) using weak
scaling: n_pairs = PAIRS_PER_CORE * n_cores, so every step takes roughly the
same wall time (~2 min).  Adjust PAIRS_PER_CORE if your hardware is noticeably
faster or slower than ~50 pairs/s per core.

MAX_CORES is read from SLURM_CPUS_PER_TASK (default 32).

The naive vs. standard vs. precomputed-keys comparison lives in a separate
script: ``scripts/run_benchmark_precompute.py``.

Run with:
    pixi run python scripts/run_benchmark_parallel.py
On an HPC cluster:
    sbatch slurm/benchmark_parallel.sh
"""
from __future__ import annotations

import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from smartreact import ReactionEnumerator

ROOT = Path(__file__).parent.parent
INPUT = ROOT / "data" / "building_blocks.parquet"
OUTPUT_CORES = ROOT / "data" / "paper" / "benchmark_parallel_cores.csv"

# --- Scaling parameters -------------------------------------------------------
# Weak scaling: n_pairs per step = PAIRS_PER_CORE * n_cores.
# Baseline: ~50 pairs/s at 1 core → 500 pairs ≈ 10 s/rep × 3 reps ≈ 30 s/step.
# Increase if steps finish too quickly; decrease if 1-core step takes > 5 min.
PAIRS_PER_CORE = 1000
N_REPEATS = 3
SEED = 42

# Core counts: powers of 2 from 1 up to the allocated CPU count.
_MAX_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", 32))
CORE_COUNTS: list[int] = [2**i for i in range(int(math.log2(_MAX_CORES)) + 1)]
if CORE_COUNTS[-1] != _MAX_CORES:          # include MAX_CORES if not a power of 2
    CORE_COUNTS.append(_MAX_CORES)


def _sample_pairs(smiles_pool: list[str], n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs_idx: set[tuple[int, int]] = set()
    while len(pairs_idx) < n:
        a, b = rng.sample(range(len(smiles_pool)), 2)
        pairs_idx.add((min(a, b), max(a, b)))
    return [(smiles_pool[a], smiles_pool[b]) for a, b in sorted(pairs_idx)]


def main() -> None:
    # ---- load data -----------------------------------------------------------
    smiles_pool = pd.read_parquet(INPUT)["smiles"].dropna().tolist()
    print(f"SMILES pool : {len(smiles_pool):,}")
    print(f"Core sweep  : {CORE_COUNTS}")
    print(f"Pairs/core  : {PAIRS_PER_CORE}  (weak scaling — n_pairs = PAIRS_PER_CORE × n_cores)")
    print(f"Repeats     : {N_REPEATS}\n")

    # ---- measure pool spawn overhead -----------------------------------------
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=_MAX_CORES) as ex:
        list(ex.map(int, range(_MAX_CORES)))
    print(f"Pool spawn ({_MAX_CORES} workers): {time.perf_counter() - t0:.2f}s")

    # ---- warm up RDKit and key caches ----------------------------------------
    print("Warming up ...")
    warmup = _sample_pairs(smiles_pool, 50, SEED + 99)
    warmup_enum = ReactionEnumerator(n_cores=_MAX_CORES)
    warmup_enum.enumerate_pairs(warmup, parallel=True)
    warmup_enum.enumerate_pairs(warmup, parallel=False)
    print("Warm-up done.\n")

    # ==========================================================================
    # Experiment 1: weak-scaling cores sweep
    # ==========================================================================
    print("=" * 60)
    print("Experiment 1: cores sweep (weak scaling)")
    print("=" * 60)

    cores_records: list[dict] = []

    for n_cores in CORE_COUNTS:
        n_pairs = PAIRS_PER_CORE * n_cores
        pairs = _sample_pairs(smiles_pool, n_pairs, SEED)
        enum = ReactionEnumerator(n_cores=n_cores)
        use_parallel = n_cores > 1
        pps_list: list[float] = []

        for rep in range(N_REPEATS):
            t0 = time.perf_counter()
            enum.enumerate_pairs(pairs, parallel=use_parallel)
            elapsed = time.perf_counter() - t0
            pps_list.append(n_pairs / elapsed)
            print(
                f"  n_cores={n_cores:2d}  n_pairs={n_pairs:6d}"
                f"  rep={rep + 1}/{N_REPEATS}  {pps_list[-1]:8.1f} pairs/s  ({elapsed:.1f}s)"
            )

        mean_pps = statistics.mean(pps_list)
        std_pps = statistics.stdev(pps_list) if len(pps_list) > 1 else 0.0
        cores_records.append({
            "n_cores": n_cores,
            "n_pairs": n_pairs,
            "mean_pps": mean_pps,
            "std_pps": std_pps,
        })
        print(f"  → {mean_pps:.1f} ± {std_pps:.1f} pairs/s\n")

    pd.DataFrame(cores_records).to_csv(OUTPUT_CORES, index=False)
    print(f"Saved → {OUTPUT_CORES}")


if __name__ == "__main__":
    main()
