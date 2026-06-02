"""
Benchmark per-call deduplication and precomputed-key impact on enumeration
throughput.

Three conditions are compared at MAX_CORES:
  1. naive       — each pair is processed independently, classifying both
                   SMILES on every occurrence (no cross-pair reuse).
  2. standard    — ``enumerate_pairs(parallel=True)``: the package's default
                   per-call deduplication classifies each unique SMILES once.
  3. precomputed — ``preprocess_smiles`` builds the key map up-front, and
                   ``enumerate_pairs(..., precomputed_keys=...)`` skips
                   classification entirely; preprocessing time is reported
                   separately as a one-time amortised cost.

Pairs are sampled from a pool of N_BUILDING_BLOCKS molecules so that each
SMILES recurs in several pairs, mimicking a combinatorial library.  This is
the regime in which deduplication matters; with random pairs from a pool of
millions almost every SMILES is unique and the three conditions converge.
The pool is also kept large enough that the unique-SMILES classification
cost is a non-trivial fraction of total wall time, so the standard versus
precomputed gap remains visible.

MAX_CORES is read from SLURM_CPUS_PER_TASK (default 32).

Run with:
    pixi run python scripts/run_benchmark_precompute.py
"""
from __future__ import annotations

import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from smartreact import ReactionEnumerator
from smartreact.preprocessing import preprocess_smiles

ROOT = Path(__file__).parent.parent
INPUT = ROOT / "data" / "building_blocks.parquet"
OUTPUT = ROOT / "data" / "paper" / "benchmark_precomputed.csv"

# Pool of 100,000 building blocks with 50,000 pairs/core gives ~32 average
# pair occurrences per SMILES at 32 cores — enough reuse that the naive
# (no-dedup) worker re-classifies each molecule many times — and a unique
# SMILES count large enough that classification dominates several seconds of
# wall time, keeping the standard-vs-precomputed gap visible.
PAIRS_PER_CORE = 50_000
N_BUILDING_BLOCKS = 100_000
N_REPEATS = 3
SEED = 42
_MAX_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", 32))


# ---- naive worker ------------------------------------------------------------
# A fresh ReactionEnumerator(n_cores=1) is created once per worker and reused
# across pairs.  KeyGenerator.classify performs no memoisation, so each call to
# enumerate_pair re-classifies both SMILES — exactly the "no deduplication"
# baseline we want to measure.

_NAIVE_ENUM: ReactionEnumerator | None = None


def _naive_worker_init() -> None:
    global _NAIVE_ENUM
    _NAIVE_ENUM = ReactionEnumerator(n_cores=1)


def _naive_worker_batch(pairs: list[tuple[str, str]]):
    assert _NAIVE_ENUM is not None
    out = []
    for s1, s2 in pairs:
        out.extend(_NAIVE_ENUM.enumerate_pair(s1, s2))
    return out


_NAIVE_POOL: ProcessPoolExecutor | None = None


def _get_naive_pool(n_cores: int) -> ProcessPoolExecutor:
    global _NAIVE_POOL
    if _NAIVE_POOL is None:
        _NAIVE_POOL = ProcessPoolExecutor(
            max_workers=n_cores, initializer=_naive_worker_init
        )
    return _NAIVE_POOL


def _close_naive_pool() -> None:
    global _NAIVE_POOL
    if _NAIVE_POOL is not None:
        _NAIVE_POOL.shutdown(wait=True, cancel_futures=True)
        _NAIVE_POOL = None


def _run_naive(pairs: list[tuple[str, str]], n_cores: int) -> list:
    pool = _get_naive_pool(n_cores)
    chunk_size = max(1, len(pairs) // n_cores)
    batches = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    futures = [pool.submit(_naive_worker_batch, b) for b in batches]
    return [r for fut in futures for r in fut.result()]


# ---- pair sampling -----------------------------------------------------------


def _sample_pairs(
    smiles_pool: list[str], n: int, seed: int
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs_idx: set[tuple[int, int]] = set()
    while len(pairs_idx) < n:
        a, b = rng.sample(range(len(smiles_pool)), 2)
        pairs_idx.add((min(a, b), max(a, b)))
    return [(smiles_pool[a], smiles_pool[b]) for a, b in sorted(pairs_idx)]


def _stats(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    smiles_full = pd.read_parquet(INPUT)["smiles"].dropna().tolist()
    rng = random.Random(SEED)
    building_blocks = rng.sample(smiles_full, N_BUILDING_BLOCKS)

    n_pairs = PAIRS_PER_CORE * _MAX_CORES
    max_pairs = N_BUILDING_BLOCKS * (N_BUILDING_BLOCKS - 1) // 2
    if n_pairs > max_pairs:
        raise ValueError(
            f"Requested {n_pairs:,} pairs but only {max_pairs:,} unique pairs "
            f"are possible from {N_BUILDING_BLOCKS} building blocks."
        )

    pairs = _sample_pairs(building_blocks, n_pairs, SEED)
    unique_smiles = list({s for pair in pairs for s in pair})

    print(f"Building blocks  : {N_BUILDING_BLOCKS:,}")
    print(f"Pairs            : {n_pairs:,}")
    print(f"Unique SMILES    : {len(unique_smiles):,}")
    print(
        f"Avg pair count/sm: "
        f"{(2 * n_pairs) / max(1, len(unique_smiles)):.1f}"
    )
    print(f"Cores            : {_MAX_CORES}")
    print(f"Repeats          : {N_REPEATS}\n")

    enum = ReactionEnumerator(n_cores=_MAX_CORES)

    # ---- warmup --------------------------------------------------------------
    print("Warming up ...")
    warm = pairs[:50]
    _run_naive(warm, _MAX_CORES)
    enum.enumerate_pairs(warm, parallel=True)
    warm_keys = preprocess_smiles(list({s for p in warm for s in p}), enum.keygen)
    enum.enumerate_pairs(warm, parallel=True, precomputed_keys=warm_keys)
    print("Warm-up done.\n")

    # ---- benchmark -----------------------------------------------------------
    print("=" * 60)
    print(f"Benchmark: naive vs standard vs precomputed ({_MAX_CORES} cores)")
    print("=" * 60)

    naive_pps: list[float] = []
    standard_pps: list[float] = []
    precomputed_pps: list[float] = []
    preprocess_times: list[float] = []

    try:
        for rep in range(N_REPEATS):
            # naive: classify per pair occurrence
            t0 = time.perf_counter()
            _run_naive(pairs, _MAX_CORES)
            naive_pps.append(n_pairs / (time.perf_counter() - t0))
            print(
                f"  naive          rep={rep + 1}/{N_REPEATS}  "
                f"{naive_pps[-1]:8.1f} pairs/s"
            )

            # standard: per-call deduplication
            t0 = time.perf_counter()
            enum.enumerate_pairs(pairs, parallel=True)
            standard_pps.append(n_pairs / (time.perf_counter() - t0))
            print(
                f"  standard       rep={rep + 1}/{N_REPEATS}  "
                f"{standard_pps[-1]:8.1f} pairs/s"
            )

            # precomputed: classify once up-front, reuse across the call
            t0 = time.perf_counter()
            keys_map = preprocess_smiles(unique_smiles, enum.keygen)
            preprocess_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            enum.enumerate_pairs(pairs, parallel=True, precomputed_keys=keys_map)
            precomputed_pps.append(n_pairs / (time.perf_counter() - t0))
            print(
                f"  precomputed    rep={rep + 1}/{N_REPEATS}  "
                f"{precomputed_pps[-1]:8.1f} pairs/s"
                f"  (preprocess: {preprocess_times[-1]:.1f}s)"
            )
    finally:
        _close_naive_pool()
        enum.close()

    pre_mean, pre_std = _stats(preprocess_times)
    print(f"\n  preprocess (one-time cost): {pre_mean:.2f} ± {pre_std:.2f}s")

    records = []
    for method, pps in [
        ("naive", naive_pps),
        ("standard", standard_pps),
        ("precomputed", precomputed_pps),
    ]:
        m, s = _stats(pps)
        rec = {
            "method": method,
            "n_pairs": n_pairs,
            "mean_pps": m,
            "std_pps": s,
        }
        if method == "precomputed":
            rec["preprocess_mean"] = pre_mean
            rec["preprocess_std"] = pre_std
        records.append(rec)

    pd.DataFrame(records).to_csv(OUTPUT, index=False)
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
