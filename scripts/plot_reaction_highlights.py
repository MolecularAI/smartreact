"""Reaction-highlight chemical space plot.

Produces two figures:
  Main figure  — UMAP embedding (5 panels, one per highlighted reaction).
  Supplement   — MW vs logP scatter with marginal histograms (5 panels).

Each UMAP panel shows building blocks (black stars) and one reaction's
products (colour), with all other products as a light-grey background.

The UMAP embedding is cached to disk so re-runs for aesthetic tweaks are fast.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from umap import UMAP

RDLogger.DisableLog("rdApp.*")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "paper" / "casestudy"
PRODUCTS_PATH = OUTPUT_DIR / "enumerated_products.parquet"

N_SAMPLE = 1000
UMAP_METRIC = "jaccard"
UMAP_N_NEIGHBORS = 100
UMAP_MIN_DIST = 0.7

HIGHLIGHT_REACTIONS = ["C_C_decarboxylation", "snar_amine", "B-H", "mitsunobu", "amide_coupling"]
HIGHLIGHT_COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]  # red, blue, green, orange, purple

_md_str = f"{UMAP_MIN_DIST:.1f}".replace(".", "p")
EMBEDDING_CACHE = OUTPUT_DIR / f"umap_embedding_{UMAP_METRIC}_nn{UMAP_N_NEIGHBORS}_md{_md_str}_{N_SAMPLE}.npz"
OUTPUT_PNG = OUTPUT_DIR / "chemical_space_highlights.png"
OUTPUT_PDF = OUTPUT_DIR / "chemical_space_highlights.pdf"
OUTPUT_SUPP_PNG = OUTPUT_DIR / "chemical_space_supplement.png"
OUTPUT_SUPP_PDF = OUTPUT_DIR / "chemical_space_supplement.pdf"


def load_or_compute_embedding(X: np.ndarray) -> np.ndarray:
    if EMBEDDING_CACHE.exists():
        print(f"Loading cached embedding from {EMBEDDING_CACHE} ...")
        return np.load(EMBEDDING_CACHE)["X_umap"]

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(
        f"Computing UMAP (metric={UMAP_METRIC}, n_neighbors={UMAP_N_NEIGHBORS}, "
        f"min_dist={UMAP_MIN_DIST}, n_jobs={n_jobs}) ..."
    )
    reducer = UMAP(
        n_components=2,
        random_state=42,
        n_jobs=n_jobs,
        metric=UMAP_METRIC,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
    )
    X_umap = reducer.fit_transform(X)
    np.savez_compressed(EMBEDDING_CACHE, X_umap=X_umap)
    print(f"  Saved embedding to {EMBEDDING_CACHE}")
    return X_umap


def compute_properties(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mws, logps = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            mws.append(Descriptors.MolWt(mol))
            logps.append(Descriptors.MolLogP(mol))
        else:
            mws.append(np.nan)
            logps.append(np.nan)
    return np.array(mws, dtype=float), np.array(logps, dtype=float)


def main() -> None:
    # Load fingerprint cache
    fp_cache = OUTPUT_DIR / f"morgan_fingerprints_{N_SAMPLE}.npz"
    if not fp_cache.exists():
        raise FileNotFoundError(
            f"Fingerprint cache not found: {fp_cache}\n"
            "Run analyze_chemical_space.py first."
        )
    print(f"Loading fingerprints from {fp_cache} ...")
    cache = np.load(fp_cache)
    X = cache["X"]
    valid_idx = cache["valid_idx"].tolist()

    # Load labels and SMILES
    bb_path = OUTPUT_DIR / f"building_blocks_{N_SAMPLE}.csv"
    df_bb = pd.read_csv(bb_path)
    df_prod = pd.read_parquet(PRODUCTS_PATH)
    n_bb = len(df_bb)
    prod_smiles = df_prod["smiles"].tolist()
    prod_types = df_prod["reaction_type"].tolist()

    is_bb = np.array([idx < n_bb for idx in valid_idx])
    reaction_labels = np.array([
        prod_types[idx - n_bb] if idx >= n_bb else ""
        for idx in valid_idx
    ])
    print(f"  Building blocks: {is_bb.sum():,}  |  Products: {(~is_bb).sum():,}")

    # UMAP embedding
    X_umap = load_or_compute_embedding(X)

    bb_coords = X_umap[is_bb]
    prod_coords = X_umap[~is_bb]
    prod_reaction_labels = reaction_labels[~is_bb]

    xlim = (X_umap[:, 0].min() - 0.5, X_umap[:, 0].max() + 0.5)
    ylim = (X_umap[:, 1].min() - 0.5, X_umap[:, 1].max() + 0.5)

    # Compute chemical properties
    print("Computing MW and logP for building blocks ...")
    bb_mw, bb_logp = compute_properties(df_bb["smiles"].tolist())

    prop_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for reaction in HIGHLIGHT_REACTIONS:
        rxn_smiles = df_prod.loc[df_prod["reaction_type"] == reaction, "smiles"].tolist()
        print(f"Computing properties for {reaction} ({len(rxn_smiles):,} products) ...")
        prop_data[reaction] = compute_properties(rxn_smiles)

    # Shared property axis limits (consistent across all columns)
    all_mw = np.concatenate([bb_mw] + [prop_data[r][0] for r in HIGHLIGHT_REACTIONS])
    all_logp = np.concatenate([bb_logp] + [prop_data[r][1] for r in HIGHLIGHT_REACTIONS])
    all_mw = all_mw[np.isfinite(all_mw)]
    all_logp = all_logp[np.isfinite(all_logp)]
    mw_lim = (all_mw.min(), all_mw.max())
    logp_lim = (all_logp.min(), all_logp.max())
    mw_bins = np.linspace(*mw_lim, 60)
    logp_bins = np.linspace(*logp_lim, 60)

    # ── Main figure: UMAP panels ─────────────────────────────────────────────
    fig_main = plt.figure(figsize=(30, 6))
    gs_main = GridSpec(1, 5, figure=fig_main, wspace=0.25)

    for col, (reaction, color) in enumerate(zip(HIGHLIGHT_REACTIONS, HIGHLIGHT_COLORS)):
        ax_umap = fig_main.add_subplot(gs_main[0, col])

        highlight_mask = prod_reaction_labels == reaction
        other_mask = ~highlight_mask

        ax_umap.scatter(
            prod_coords[other_mask, 0], prod_coords[other_mask, 1],
            c="#CCCCCC", s=2, alpha=0.4, linewidths=0, zorder=1,
            label="Other products",
        )
        ax_umap.scatter(
            prod_coords[highlight_mask, 0], prod_coords[highlight_mask, 1],
            c=color, s=4, alpha=0.7, linewidths=0, zorder=2,
            label=f"Products ({reaction})",
        )
        ax_umap.scatter(
            bb_coords[:, 0], bb_coords[:, 1],
            c="black", s=15, alpha=0.9, marker="*", zorder=3,
            label="Building blocks",
        )
        ax_umap.set_title(
            f"{reaction}\n({int(highlight_mask.sum()):,} products)", fontsize=11
        )
        ax_umap.set_xlabel("UMAP 1", fontsize=9)
        ax_umap.set_xlim(xlim)
        ax_umap.set_ylim(ylim)
        ax_umap.legend(loc="upper right", fontsize=7, markerscale=2, frameon=True)
        if col == 0:
            ax_umap.set_ylabel("UMAP 2", fontsize=9)

    fig_main.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig_main.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"\nSaved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")

    # ── Supplement figure: MW vs logP with marginal histograms ───────────────
    fig_supp = plt.figure(figsize=(30, 7))
    gs_supp = GridSpec(1, 5, figure=fig_supp, wspace=0.35)

    for col, (reaction, color) in enumerate(zip(HIGHLIGHT_REACTIONS, HIGHLIGHT_COLORS)):
        inner_gs = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=gs_supp[0, col],
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
        )
        ax_main = fig_supp.add_subplot(inner_gs[1, 0])
        ax_top = fig_supp.add_subplot(inner_gs[0, 0], sharex=ax_main)
        ax_right = fig_supp.add_subplot(inner_gs[1, 1], sharey=ax_main)

        rxn_mw, rxn_logp = prop_data[reaction]
        valid = np.isfinite(rxn_mw) & np.isfinite(rxn_logp)
        bb_valid = np.isfinite(bb_mw) & np.isfinite(bb_logp)

        ax_main.scatter(
            rxn_mw[valid], rxn_logp[valid],
            c=color, s=3, alpha=0.3, linewidths=0, zorder=2,
        )
        ax_main.scatter(
            bb_mw[bb_valid], bb_logp[bb_valid],
            c="black", s=6, alpha=0.4, linewidths=0, zorder=3,
        )
        ax_main.set_xlabel("MW (Da)", fontsize=9)
        ax_main.set_xlim(mw_lim)
        ax_main.set_ylim(logp_lim)
        if col == 0:
            ax_main.set_ylabel("logP", fontsize=9)

        ax_top.set_title(reaction, fontsize=10)
        ax_top.hist(rxn_mw[valid], bins=mw_bins, color=color, alpha=0.5, linewidth=0, density=True)
        ax_top.hist(bb_mw[bb_valid], bins=mw_bins, color="black", alpha=0.4, linewidth=0, density=True)
        ax_top.set_xlim(mw_lim)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.set_yticks([])

        ax_right.hist(
            rxn_logp[valid], bins=logp_bins, color=color, alpha=0.5,
            linewidth=0, orientation="horizontal", density=True,
        )
        ax_right.hist(
            bb_logp[bb_valid], bins=logp_bins, color="black", alpha=0.4,
            linewidth=0, orientation="horizontal", density=True,
        )
        ax_right.set_ylim(logp_lim)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        ax_right.set_xticks([])

    fig_supp.savefig(OUTPUT_SUPP_PNG, dpi=300, bbox_inches="tight")
    fig_supp.savefig(OUTPUT_SUPP_PDF, bbox_inches="tight")
    print(f"Saved {OUTPUT_SUPP_PNG}")
    print(f"Saved {OUTPUT_SUPP_PDF}")


if __name__ == "__main__":
    main()
