"""Case study: visualize the chemical space of SmartReact enumeration products."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from umap import UMAP

RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # building_blocks.parquet lives here (not committed)
OUTPUT_DIR = DATA_DIR / "paper" / "casestudy"
PRODUCTS_PATH = OUTPUT_DIR / "enumerated_products.parquet"
OUTPUT_PNG = OUTPUT_DIR / "chemical_space.png"
OUTPUT_PDF = OUTPUT_DIR / "chemical_space.pdf"

FP_RADIUS = 2
FP_NBITS = 2048
PCA_VARIANCE = 0.80

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)


def smiles_to_fp(smiles: str) -> np.ndarray | None:
    """Compute Morgan fingerprint as a bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = _FP_GEN.GetFingerprintAsNumPy(mol)
    return fp


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze chemical space of SmartReact products")
    parser.add_argument(
        "-n", "--n-sample", type=int, default=1000,
        help="Number of building blocks used (matches enumerate_products.py -n)",
    )
    args = parser.parse_args()

    bb_path = OUTPUT_DIR / f"building_blocks_{args.n_sample}.csv"

    # Load data
    df_bb = pd.read_csv(bb_path)
    df_prod = pd.read_parquet(PRODUCTS_PATH)

    bb_smiles = df_bb["smiles"].tolist()
    prod_smiles = df_prod["smiles"].tolist()
    prod_types = df_prod["reaction_type"].tolist()

    print(f"Building blocks: {len(bb_smiles)}")
    print(f"Products: {len(prod_smiles)}")

    # Compute fingerprints in parallel (or load from cache)
    fp_cache = OUTPUT_DIR / f"morgan_fingerprints_{args.n_sample}.npz"
    all_smiles = bb_smiles + prod_smiles
    if fp_cache.exists():
        print(f"Loading cached Morgan fingerprints from {fp_cache} ...")
        cache = np.load(fp_cache)
        X = cache["X"]
        valid_idx = cache["valid_idx"].tolist()
        print(f"  Loaded {len(valid_idx)} fingerprints")
    else:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        print(f"Computing Morgan fingerprints for {len(all_smiles)} molecules ({n_workers} workers) ...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            fp_results = list(executor.map(smiles_to_fp, all_smiles, chunksize=500))
        fps = []
        valid_idx = []
        for i, fp in enumerate(fp_results):
            if fp is not None:
                fps.append(fp)
                valid_idx.append(i)
        X = np.vstack(fps)
        np.savez_compressed(fp_cache, X=X, valid_idx=np.array(valid_idx))
        print(f"  Computed {len(valid_idx)} fingerprints — saved to {fp_cache}")

    # Track which valid indices are building blocks vs products
    n_bb = len(bb_smiles)
    is_bb = np.array([idx < n_bb for idx in valid_idx])
    # Map valid product indices back to their reaction type
    prod_labels = []
    for idx in valid_idx:
        if idx >= n_bb:
            prod_labels.append(prod_types[idx - n_bb])
        else:
            prod_labels.append(None)

    # PCA
    print("Running PCA ...")
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_pca = pca.fit_transform(X)
    n_components = X_pca.shape[1]
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Retained {n_components} components ({explained:.1f}% variance explained)")

    # UMAP
    print("Running UMAP ...")
    umap = UMAP(n_components=2, random_state=42, n_jobs=1, metric="jaccard", n_neighbors=25, min_dist=0.5)
    X_umap = umap.fit_transform(X_pca)

    # Separate building blocks and products
    bb_mask = is_bb
    bb_coords = X_umap[bb_mask]
    prod_coords = X_umap[~is_bb]
    prod_labels_valid = [lbl for lbl, m in zip(prod_labels, ~is_bb) if m]

    # Sort reaction types by count descending for legend order (no "Other" grouping)
    label_series = pd.Series(prod_labels_valid)
    type_order = label_series.value_counts().sort_values(ascending=False).index.tolist()

    # Assign unique colors — tab20 + tab20b covers up to 40 distinct reaction types
    _cmap1 = plt.get_cmap("tab20")
    _cmap2 = plt.get_cmap("tab20b")
    color_map = {
        t: (_cmap1(i) if i < 20 else _cmap2(i - 20))
        for i, t in enumerate(type_order)
    }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Building blocks first so they appear at the top of the legend
    ax.scatter(
        bb_coords[:, 0], bb_coords[:, 1],
        c="black", s=15, alpha=0.8, marker="*", label="Building blocks", zorder=3,
    )

    # Products by reaction type (background)
    for rtype in type_order:
        mask = (label_series == rtype).values
        ax.scatter(
            prod_coords[mask, 0], prod_coords[mask, 1],
            c=[color_map[rtype]], s=4, alpha=0.5, label=rtype, zorder=1,
        )

    n_prod = len(label_series)
    n_bb_valid = int(bb_mask.sum())
    ax.set_title(
        f"Chemical space covered by SmartReact enumeration\n"
        f"({n_prod:,} products from {n_bb_valid:,} building blocks)",
        fontsize=13,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Legend outside plot
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, markerscale=2, frameon=True,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"\nSaved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
