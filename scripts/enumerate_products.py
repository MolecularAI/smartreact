"""Case study: enumerate products from sampled building blocks using SmartReact."""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

import pandas as pd

from smartreact import KeyGenerator, ReactionEnumerator, preprocess_smiles

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUILDING_BLOCKS_DIR = Path(__file__).resolve().parent.parent / "public_data" / "casestudy"
OUTPUT_DIR = DATA_DIR / "paper" / "casestudy"
SCRIPTS_DIR = Path(__file__).resolve().parent

SEED = 42


def load_or_sample_building_blocks(n_sample: int, bb_path: Path) -> list[str]:
    """Load cached building blocks or sample from the source parquet file."""
    if bb_path.exists():
        df = pd.read_csv(bb_path)
        print(f"Loaded {len(df)} building blocks from {bb_path}")
        return df["smiles"].tolist()

    print("Sampling building blocks from source parquet ...")
    df = pd.read_parquet(DATA_DIR / "building_blocks.parquet")
    sample = df.sample(n=n_sample, random_state=SEED)
    sample.to_csv(bb_path, index=False)
    print(f"Saved {len(sample)} building blocks to {bb_path}")
    return sample["smiles"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate products with SmartReact")
    parser.add_argument(
        "-n", "--n-sample", type=int, default=1000,
        help="Number of building blocks to sample (default: 1000)",
    )
    parser.add_argument(
        "-c", "--cores", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
        help="CPU cores for key classification and enumeration (default: SLURM_CPUS_PER_TASK or 4)",
    )
    args = parser.parse_args()
    n_sample = args.n_sample
    n_cores = args.cores

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BUILDING_BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    bb_path = BUILDING_BLOCKS_DIR / f"building_blocks_{n_sample}.csv"
    output_path = OUTPUT_DIR / "enumerated_products.parquet"

    smiles_list = load_or_sample_building_blocks(n_sample, bb_path)

    # Precompute keys
    print(f"\nPreprocessing {len(smiles_list)} building blocks ...")
    t_preprocess = time.perf_counter()
    keygen = KeyGenerator(n_cores=n_cores)
    keys_map = preprocess_smiles(smiles_list, keygen)
    keygen.close()
    preprocess_time = time.perf_counter() - t_preprocess
    print(f"  Keys computed for {len(keys_map)} unique SMILES ({preprocess_time:.1f} s)")

    # Generate all pairwise combinations (unordered)
    pairs = list(itertools.combinations(smiles_list, 2))
    n_pairs = len(pairs)
    print(f"\nEnumerating {n_pairs:,} pairs with {n_cores} cores ...")

    t_enum = time.perf_counter()
    with ReactionEnumerator(n_cores=n_cores) as enumerator:
        results = enumerator.enumerate_pairs(
            pairs, parallel=True, precomputed_keys=keys_map
        )
    enum_time = time.perf_counter() - t_enum
    total_time = preprocess_time + enum_time

    # Collect unique products per reaction type
    seen: dict[str, str] = {}  # smiles -> reaction_type
    for r in results:
        for p in r.products:
            if p not in seen:
                seen[p] = r.reaction_name

    df_out = pd.DataFrame(
        [{"smiles": s, "reaction_type": rt} for s, rt in seen.items()]
    )
    df_out.to_parquet(output_path, index=False)

    # Summary
    pairs_per_sec_enum = n_pairs / enum_time
    pairs_per_sec_total = n_pairs / total_time
    n_total_products = sum(len(r.products) for r in results)
    counts = df_out["reaction_type"].value_counts().sort_values(ascending=False)

    # Save metrics to CSV
    metrics_rows = [
        {"metric": "building_blocks", "value": len(smiles_list)},
        {"metric": "pairs_evaluated", "value": n_pairs},
        {"metric": "total_products", "value": n_total_products},
        {"metric": "unique_products", "value": len(df_out)},
        {"metric": "preprocess_time_s", "value": round(preprocess_time, 2)},
        {"metric": "enumeration_time_s", "value": round(enum_time, 2)},
        {"metric": "total_time_s", "value": round(total_time, 2)},
        {"metric": "pairs_per_second_enum", "value": round(pairs_per_sec_enum, 1)},
        {"metric": "pairs_per_second_total", "value": round(pairs_per_sec_total, 1)},
        {"metric": "n_cores", "value": n_cores},
    ]
    for rxn, count in counts.items():
        metrics_rows.append({"metric": f"products_{rxn}", "value": count})

    metrics_path = OUTPUT_DIR / "enumeration_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    print("\n" + "=" * 60)
    print("ENUMERATION SUMMARY")
    print("=" * 60)
    print(f"  Building blocks:   {len(smiles_list):>10,}")
    print(f"  Pairs evaluated:   {n_pairs:>10,}")
    print(f"  Preprocess time:   {preprocess_time:>10.1f} s")
    print(f"  Enumeration time:  {enum_time:>10.1f} s")
    print(f"  Total time:        {total_time:>10.1f} s")
    print(f"  Pairs/s (enum):    {pairs_per_sec_enum:>10,.0f}")
    print(f"  Pairs/s (total):   {pairs_per_sec_total:>10,.0f}")
    print(f"  Total products:    {n_total_products:>10,}")
    print(f"  Unique products:   {len(df_out):>10,}")
    print()
    print("Products per reaction type:")
    for rxn, count in counts.items():
        print(f"  {rxn:<40s} {count:>8,}")
    print("=" * 60)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
