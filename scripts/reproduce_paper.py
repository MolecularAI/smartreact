#!/usr/bin/env python3
"""Reproduce the benchmarks, results and figures in the SmartReact paper.

    pixi run -e baseline python scripts/reproduce_paper.py           # everything
    pixi run -e baseline python scripts/reproduce_paper.py --quick   # smoke test
    pixi run -e baseline python scripts/reproduce_paper.py --only baselines

Hardware: written for a 32-core node with >=128 GB RAM. Worker processes cache
compiled reaction templates without bound, up to about 1.2 GB each, so budget
roughly cores x 1.5 GB -- and twice that for `precompute`, which keeps two pools
of `cores` workers alive at once to time naive classification against the
library's own. With much less, the kernel will OOM-kill the run. Note that pool
workers are not reaped when the parent is killed: if a run reports far less free
memory than it should, look for orphaned python processes from an earlier one.

The route-vs-route timings in `baselines` and `scaling` ignore --cores and run
on one core, pinned in those two stage functions rather than configurable: the
three prefilters do not parallelise alike, so a multi-core comparison would
report thread counts as if they were filter quality. Every other stage uses the
cores it is given.

Network: `saturation` and `filterability` download USPTO-50k on first use. On a
cluster whose compute nodes have no outbound network, fetch it from the login
node first -- `_download_uspto()` is enough, or copy data/paper/uspto_50k_raw.csv
(or an already-extracted uspto_templates_n*_unique.csv) into place.

Needs the `baseline` pixi environment (superset of `casestudy`, plus rdchiral).
Everything runs from the 1,000 building blocks in public_data/, reads the
*installed* smartreact package, and writes to data/paper/.

The claim being measured: indexing templates by their reactant-side patterns is
standard, and RDKit ships two mechanisms that do it. What makes it work here is
that SmartReact's templates are built by combinatorial expansion over a fixed
SMARTS-RX vocabulary, so their distinct reactant patterns saturate -- the index
stops growing while the library grows. Auto-extracted templates share no such
vocabulary, so the index grows with the library. Same indexer, opposite result.

The workflow schematics are drawn by hand and are not produced here.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import platform
import random
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import (
    Descriptors,
    rdChemReactions,
    rdFingerprintGenerator,
    rdSubstructLibrary,
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogEntry, SmartsMatcher

import smartreact
from smartreact import (
    KeyGenerator,
    ReactionEnumerator,
    ReactionTemplate,
    build_template_index,
    candidate_templates,
    load_templates,
    preprocess_smiles,
)

# Every route below shares the library's own product collection, so a
# comparison measures the filter and not a reimplementation of RunReactants.
from smartreact.enumerator import _collect_products

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_BB = REPO_ROOT / "public_data" / "casestudy" / "building_blocks_1000.csv"
OUT = REPO_ROOT / "data" / "paper"
CASESTUDY_DIR = OUT / "casestudy"
FIGURES_DIR = OUT / "figures"
SEED = 42

USPTO_URL = (
    "https://raw.githubusercontent.com/connorcoley/retrosim/master/"
    "retrosim/data/data_processed.csv"
)
USPTO_RAW = OUT / "uspto_50k_raw.csv"

# --------------------------------------------------------------------------
# Figure style -- matches notebooks/paper_figures/benchmark_filtering.ipynb, so
# every figure in the paper reads as one set.
# --------------------------------------------------------------------------
FIG_SIZE = (8, 5)
FONT_LABEL, FONT_TICK, FONT_LEGEND = 15, 13, 13
# Markers are bigger than the notebook's 2 pt: these sweeps carry 5-9 points,
# not a dense curve, and at 2 pt the points vanish into the line.
MARKER_SIZE, LINE_WIDTH, BAND_ALPHA = 5, 1.5, 0.2

# Library names. Both cross-library figures screen with the *same* RDKit
# FilterCatalog, so these name the library under test, never the filter --
# "SmartReact" alone would read as "SmartReact's filter", which is the opposite
# of the point being made.
LIB_CONSTRUCTED = "Constructed (SmartReact library)"
LIB_EXTRACTED = "Auto-extracted (USPTO)"

C_BLUE = "#2196F3"   # SmartReact / filtered / the constructed library
C_RED = "#F44336"    # brute force / the auto-extracted library
C_GREY = "#9E9E9E"   # naive baseline
C_GREEN = "#4CAF50"  # precomputed keys

# Worker-process state.
_W_SMARTS: list[str] = []
_W_MOLS: list[Chem.Mol] = []
_NAIVE_ENUM: ReactionEnumerator | None = None


# ==========================================================================
# Entry point
# ==========================================================================


def main() -> int:
    # ---------------- CONFIG: edit here, or override with --only ----------
    STAGES = {
        "saturation": True,     # Fig 1a  reactant patterns vs library size
        "filterability": True,  # Fig 1b  one RDKit prefilter, two libraries
        "baselines": True,      # Table 2 four routes, one apply path
        "scaling": True,        # Fig 3a  throughput vs template count
        "parallel": True,       # Fig 3b  weak scaling over cores
        "precompute": True,     # Fig 3c  classification reuse sweep
        "casestudy": True,      # Sec 3.5 all pairs of the public blocks
        "chemspace": True,      # Fig 4   fingerprints, UMAP, highlights
        "coverage": True,       # App D,E chord diagrams, L3 table
    }
    CORES = 0                   # 0 = SLURM_CPUS_PER_TASK, else all cores
    REUSE_RESULTS = False       # True = redraw figures from existing CSVs
    BUILDING_BLOCKS = None      # None = the public 1,000; else a path
    SETTINGS = dict(
        uspto_limit=None,               # None = all 50,016 USPTO reactions
        saturation_repeats=5,
        n_filterability_molecules=250,
        # 5, not 3: the band now covers the molecule draw as well as the
        # template draw, and a std from three samples is barely an estimate.
        filterability_repeats=5,
        n_baseline_molecules=None,      # None = all building blocks
        n_brute_pairs=500,              # brute force cannot run at full scale
        n_scaling_molecules=60,
        scaling_repeats=3,
        pairs_per_core=5000,
        parallel_repeats=3,
        precompute_reuse=(2, 4, 8, 16, 32, 64, 128, 256),
        precompute_repeats=3,
        n_casestudy_molecules=1000,
    )
    # ---------------- end of config ---------------------------------------

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="smoke test on tiny inputs; the numbers are NOT the paper's")
    p.add_argument("-c", "--cores", type=int, default=0)
    p.add_argument("--reuse", action="store_true",
                   help="skip computation where a stage's CSV exists and just "
                        "redraw its figures")
    p.add_argument("--only", nargs="+", metavar="STAGE")
    p.add_argument("--skip", nargs="+", metavar="STAGE", default=[])
    p.add_argument("--building-blocks", type=Path, default=None)
    args = p.parse_args()

    registry = {
        "saturation": stage_saturation,
        "filterability": stage_filterability,
        "baselines": stage_baselines,
        "scaling": stage_scaling,
        "parallel": stage_parallel,
        "precompute": stage_precompute,
        "casestudy": stage_casestudy,
        "chemspace": stage_chemspace,
        "coverage": stage_coverage,
    }
    for name in (args.only or []) + args.skip:
        if name not in registry:
            p.error(f"unknown stage {name!r}. Known: {', '.join(registry)}")

    settings = dict(SETTINGS)
    if args.quick:
        settings.update(
            uspto_limit=3000, saturation_repeats=2,
            n_filterability_molecules=40, filterability_repeats=1,
            n_baseline_molecules=60, n_brute_pairs=100,
            n_scaling_molecules=20, scaling_repeats=1,
            pairs_per_core=200, parallel_repeats=1,
            # Enough points that the sweep still looks like a curve.
            precompute_reuse=(2, 4, 8, 16, 32), precompute_repeats=1,
            n_casestudy_molecules=150,
        )
    cores = (args.cores or CORES or int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
             or (os.cpu_count() or 1))
    cfg = Config(cores=cores, quick=args.quick, reuse=args.reuse or REUSE_RESULTS,
                 building_blocks=args.building_blocks or BUILDING_BLOCKS, **settings)

    # Registry order, not the order the flags were typed: chemspace reads the
    # parquet casestudy writes, so `--only chemspace casestudy` must not run
    # chemspace first.
    chosen = set(args.only) if args.only else {k for k, on in STAGES.items() if on}
    selected = [k for k in registry if k in chosen and k not in args.skip]

    # Line-buffer stdout: Python block-buffers when it is redirected to a file,
    # so a long stage's progress would otherwise sit invisible in an 8 KB buffer
    # and a running cluster job would look stalled.
    sys.stdout.reconfigure(line_buffering=True)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"smartreact {smartreact.__version__} | rdkit {rdBase.rdkitVersion} | "
          f"python {sys.version.split()[0]}")
    print(f"{platform.platform()} | {cores} cores | "
          f"{'QUICK smoke test' if args.quick else 'full run'}")
    print(f"stages: {', '.join(selected)}\n")

    failures = []
    for name in selected:
        print(f"{'=' * 70}\n[{name}]\n{'=' * 70}", flush=True)
        t0 = time.perf_counter()
        try:
            registry[name](cfg)
            print(f"  [{name}] ok in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:  # one stage must not sink the run
            import traceback

            traceback.print_exc()
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  [{name}] FAILED after {time.perf_counter() - t0:.1f}s")

    if failures:
        print("\nFAILED: " + "; ".join(failures))
        return 1
    print("\nAll stages completed." + (
        "\nThis was a smoke test; re-run without --quick for the paper's numbers."
        if args.quick else ""))
    return 0


@dataclass
class Config:
    cores: int
    quick: bool
    reuse: bool
    building_blocks: Path | None
    uspto_limit: int | None
    saturation_repeats: int
    n_filterability_molecules: int
    filterability_repeats: int
    n_baseline_molecules: int | None
    n_brute_pairs: int
    n_scaling_molecules: int
    scaling_repeats: int
    pairs_per_core: int
    parallel_repeats: int
    precompute_reuse: tuple[int, ...]
    precompute_repeats: int
    n_casestudy_molecules: int


# ==========================================================================
# Stage: saturation  -- distinct reactant patterns vs library size
# ==========================================================================

SATURATION_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]


def _download_uspto() -> None:
    if USPTO_RAW.exists():
        return
    USPTO_RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {USPTO_URL}")
    tmp = USPTO_RAW.with_suffix(".part")
    with urllib.request.urlopen(USPTO_URL) as response, tmp.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    tmp.rename(USPTO_RAW)


def _extract_retro_template(rxn_smiles: str) -> str | None:
    # Imported inside the worker: rdchiral pulls in RDKit state that does not
    # survive forking cleanly on every platform.
    from rdchiral.template_extractor import extract_from_reaction

    try:
        reactants, products = rxn_smiles.split(">>")
        result = extract_from_reaction(
            {"reactants": reactants, "products": products, "_id": "0"})
    except Exception:
        return None
    return (result or {}).get("reaction_smarts") or None


def prepare_uspto_templates(cores: int, limit: int | None = None) -> Path:
    """Auto-extracted comparison library: USPTO-50k forward templates.

    rdchiral emits retro templates, so each is reversed. Only bimolecular ones
    are kept, since SmartReact is pairwise, and identical templates are collapsed
    to one row.

    Deduplication matters for the comparison: USPTO reactions of the same class
    extract to the same template, so the raw output holds one row per *reaction*,
    not per template. Counting those repeats would put a library of n rows on the
    same axis as SmartReact's n distinct templates, and each repeat would raise
    the template count without contributing a distinct reactant pattern. The
    multiplicity is preserved in the n_source_reactions column.

    The cache filename carries the source-reaction count, so a truncated --quick
    extraction can never be picked up by a later full run.
    """
    _download_uspto()
    csv.field_size_limit(10**7)
    with USPTO_RAW.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit:
        rows = rows[:limit]
    reactions = [r["rxn_smiles"] for r in rows if r.get("rxn_smiles")]

    # "_unique" in the name so caches written before deduplication existed are
    # not silently reused.
    target = OUT / f"uspto_templates_n{len(reactions)}_unique.csv"
    if target.exists():
        print(f"  cached {target.name}")
        return target

    print(f"  extracting from {len(reactions):,} reactions on {cores} cores")
    with ProcessPoolExecutor(max_workers=cores) as pool:
        retro = list(pool.map(_extract_retro_template, reactions,
                              chunksize=max(1, len(reactions) // (cores * 8))))

    # Insertion-ordered, so the library is reproducible run to run.
    counts: Counter = Counter()
    for smarts in retro:
        if not smarts or ">>" not in smarts:
            continue
        product_side, reactant_side = smarts.split(">>", 1)
        if product_side.strip() and len(reactant_side.split(".")) == 2:
            counts[f"{reactant_side}>>{product_side}"] += 1
    if not counts:
        raise RuntimeError("No templates survived extraction.")
    records = [{"forward_template": t, "n_source_reactions": n} for t, n in counts.items()]
    print(f"  kept {len(records):,} distinct bimolecular templates from "
          f"{sum(counts.values()):,} extractions "
          f"(max {max(counts.values())} reactions share one template)")

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["forward_template", "n_source_reactions"])
        w.writeheader()
        w.writerows(records)
    return target


def _resolvable(smarts_list: Sequence[str], label: str) -> list[str]:
    """Templates RDKit can recover reactant-side patterns from.

    Run before match_library_sizes so that unusable templates cannot make the
    two libraries diverge again after they have been matched.
    """
    out = [s for s in smarts_list if reactant_patterns(s) is not None]
    if len(out) != len(smarts_list):
        print(f"  {label:26s} {len(smarts_list):,} -> {len(out):,} usable templates")
    return out


def _growth_curve(smarts_list: Sequence[str], label: str, repeats: int) -> list[dict]:
    resolved = [p for s in smarts_list if (p := reactant_patterns(s)) is not None]
    n_total = len(resolved)
    print(f"  [{label}] {n_total:,} usable templates")

    rng = random.Random(SEED)
    records = []
    for fraction in SATURATION_FRACTIONS:
        size = max(1, round(fraction * n_total))
        reps = 1 if size == n_total else repeats
        counts = []
        for _ in range(reps):
            patterns: set[str] = set()
            for i in rng.sample(range(n_total), size):
                patterns.update(resolved[i])
            counts.append(float(len(patterns)))
        mean, std = mean_std(counts)
        records.append({"library": label, "fraction": fraction, "n_templates": size,
                        "mean_patterns": round(mean, 2), "std_patterns": round(std, 2),
                        "patterns_per_template": round(mean / size, 4), "n_repeats": reps})
        print(f"  [{label[:12]:12s}] {size:7,} templates -> {mean:8.1f} patterns "
              f"({mean / size:.3f} each)")
    return records


def stage_saturation(cfg: Config) -> None:
    cached: list | None = None

    def libraries() -> list[tuple[str, list[str]]]:
        # Loaded lazily and shared by both CSVs below, so --reuse skips the
        # USPTO extraction entirely -- but only when both already exist.
        nonlocal cached
        if cached is None:
            with bundled("templates.txt").open(encoding="utf-8", newline="") as fh:
                sr = [r["reaction_template"] for r in csv.DictReader(fh, delimiter="\t")]
            path = prepare_uspto_templates(cfg.cores, cfg.uspto_limit)
            with path.open(encoding="utf-8") as fh:
                uspto = [r["forward_template"] for r in csv.DictReader(fh)]
            # Deduplicate, drop anything RDKit cannot recover reactant sides
            # from, then match the two sizes -- in that order, so the counts
            # that finally land on the x axis are the ones that were matched.
            sr = _resolvable(unique_templates(sr, LIB_CONSTRUCTED), LIB_CONSTRUCTED)
            uspto = _resolvable(unique_templates(uspto, LIB_EXTRACTED), LIB_EXTRACTED)
            sr, uspto = match_library_sizes(sr, uspto, LIB_CONSTRUCTED, LIB_EXTRACTED)
            cached = [(LIB_CONSTRUCTED, sr), (LIB_EXTRACTED, uspto)]
        return cached

    def curves() -> list[dict]:
        return [rec for label, smarts in libraries()
                for rec in _growth_curve(smarts, label, cfg.saturation_repeats)]

    def summarise() -> list[dict]:
        # Full-library counts with and without atom-map stripping, so the
        # normalisation choice is visible rather than buried.
        out = []
        for label, smarts in libraries():
            for strip in (True, False):
                patterns: set[str] = set()
                usable = 0
                for s in smarts:
                    if got := reactant_patterns(s, strip_maps=strip):
                        usable += 1
                        patterns.update(got)
                out.append({"library": label, "atom_maps_stripped": strip,
                            "n_templates": usable, "n_distinct_patterns": len(patterns),
                            "templates_per_pattern": round(usable / max(1, len(patterns)), 2)})
                print(f"  {label:24s} strip={strip!s:5s} {usable:7,} templates "
                      f"{len(patterns):6,} patterns")
        return out

    records = load_or_compute(OUT / "template_saturation.csv", curves, cfg)
    load_or_compute(OUT / "template_saturation_summary.csv", summarise, cfg)

    # Legacy keys kept so --reuse on a CSV written before the rename still
    # colours its curves correctly instead of silently falling back.
    styles = {LIB_CONSTRUCTED: (C_BLUE, "o"), LIB_EXTRACTED: (C_RED, "s"),
              "SmartReact": (C_BLUE, "o"), "USPTO (auto-extracted)": (C_RED, "s")}
    by_lib: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lib[str(r["library"])].append(r)

    # Log-log: the whole point is the *shape* of the two curves over two orders
    # of magnitude in library size, which linear axes would flatten out.
    fig, ax = new_figure()
    for label, recs in by_lib.items():
        recs = sorted(recs, key=lambda r: r["n_templates"])
        color, marker = styles.get(label, (C_GREEN, "^"))
        plot_series(ax, [r["n_templates"] for r in recs], [r["mean_patterns"] for r in recs],
                    color, label, marker, stds=[r["std_patterns"] for r in recs])
    # Every template in both libraries is bimolecular, so with no sharing at all
    # each would contribute two searches. The no-sharing ceiling is 2x, not x --
    # a line at y=x sits inside the sharing regime and is not a reference at all.
    allx = [r["n_templates"] for r in records]
    ax.plot([min(allx), max(allx)], [2 * min(allx), 2 * max(allx)], "--", color="0.55",
            linewidth=1.2, label="No sharing (2 searches per template)")
    ax.set(xscale="log", yscale="log")
    # Screening runs each distinct reactant query against each molecule exactly
    # once, so the count of distinct queries *is* the searches per molecule.
    finish_axes(ax, "Reaction templates in library", "Substructure searches per molecule",
                legend_loc="upper left")
    fig.tight_layout()
    savefig(fig, OUT / "template_saturation")


# ==========================================================================
# Stage: filterability -- one RDKit prefilter over two libraries
# ==========================================================================
#
# The CSV records index size, screening time and admitted work; the figure plots
# only the last of these -- template applications surviving the filter, per pair
# -- since that is what the downstream RunReactants cost is proportional to.
# NOT compared: product counts or apply time, since the two libraries cover
# different chemistry and any such comparison would be meaningless.
#
# Admitted-per-pair is an unbiased estimate of a property of the library, not of
# the molecule set: summed over templates its expectation is 2 * sum(p_a * p_b),
# in which the pair count cancels. So the molecule count sets the precision of
# each point, never its position -- and the band covers both the template draw
# and the molecule draw.

FILTERABILITY_COUNTS = [100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]


def usable_bimolecular(smarts_list: Sequence[str]) -> list[str]:
    """Templates that compile and have exactly two reactant sides."""
    out = []
    for s in smarts_list:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(s)
        except Exception:
            continue
        if rxn is not None and rxn.GetNumReactantTemplates() == 2:
            out.append(s)
    return out


def stage_filterability(cfg: Config) -> None:
    def compute() -> list[dict]:
        with bundled("templates.txt").open(encoding="utf-8", newline="") as fh:
            sr = [r["reaction_template"] for r in csv.DictReader(fh, delimiter="\t")]
        path = prepare_uspto_templates(cfg.cores, cfg.uspto_limit)
        with path.open(encoding="utf-8") as fh:
            uspto = [r["forward_template"] for r in csv.DictReader(fh)]
        # Same order as the saturation stage: deduplicate, keep what compiles,
        # then match the two sizes, so both sweeps share their x positions.
        sr_lib = usable_bimolecular(unique_templates(sr, LIB_CONSTRUCTED))
        up_lib = usable_bimolecular(unique_templates(uspto, LIB_EXTRACTED))
        sr_lib, up_lib = match_library_sizes(sr_lib, up_lib, LIB_CONSTRUCTED, LIB_EXTRACTED)
        libraries = [(LIB_CONSTRUCTED, sr_lib), (LIB_EXTRACTED, up_lib)]
        for label, s in libraries:
            print(f"  {label:26s} {len(s):,} usable templates")

        # A fresh molecule draw per repeat, so the band covers molecule sampling
        # as well as template sampling. The draws are shared by both libraries
        # and every template count -- repeat r screens the same molecules
        # everywhere -- so the two curves stay paired and the gap between them
        # is not an artefact of which library got which molecules.
        reps = cfg.filterability_repeats
        molecule_sets = [parse_molecules(load_building_blocks(
            cfg.building_blocks, n=cfg.n_filterability_molecules, seed=SEED + r))
            for r in range(reps)]
        n_molecules = len(molecule_sets[0])
        n_pairs = n_molecules * (n_molecules - 1) // 2
        print(f"  {reps} draw{'s' if reps != 1 else ''} of {n_molecules:,} molecules, "
              f"{n_pairs:,} pairs each")
        rng = random.Random(SEED)

        records = []
        for label, smarts in libraries:
            n_total = len(smarts)
            for n in sorted({min(c, n_total) for c in FILTERABILITY_COUNTS} | {n_total}):
                queries, seconds, admitted = [], [], []
                # The full library is identical every repeat, so there only the
                # molecules vary -- which is now a real source of spread, and
                # why this point is no longer exempted from repeating.
                for molecules in molecule_sets:
                    subset = smarts if n == n_total else rng.sample(smarts, n)
                    # FilterCatalog rather than SubstructLibrary: no fingerprint
                    # prescreen, so the time is the pattern-matching cost itself.
                    screen = screen_filtercatalog(subset, molecules)
                    queries.append(float(screen.n_queries))
                    seconds.append(screen.seconds)
                    # Averaged like every other column: a single repeat's value
                    # would describe a different template subset and a different
                    # molecule draw than the means.
                    admitted.append(float(len(screen.triples)))
                q, _ = mean_std(queries)
                s_mean, s_std = mean_std(seconds)
                a_mean, a_std = mean_std(admitted)
                full = n_pairs * n * 2
                records.append({
                    "library": label, "n_templates": n, "n_queries": round(q),
                    "queries_per_template": round(q / n, 4),
                    "screen_s_mean": round(s_mean, 3), "screen_s_std": round(s_std, 3),
                    "screen_ms_per_molecule": round(1000 * s_mean / n_molecules, 3),
                    "n_admitted": round(a_mean, 1), "n_admitted_std": round(a_std, 1),
                    "admitted_per_pair": round(a_mean / n_pairs, 4) if n_pairs else 0.0,
                    "admitted_per_pair_std": round(a_std / n_pairs, 4) if n_pairs else 0.0,
                    "full_space": full,
                    "admitted_fraction": round(a_mean / full, 8) if full else 0.0,
                    "n_molecules": n_molecules, "n_pairs": n_pairs, "n_repeats": reps})
                print(f"  [{label[:12]:12s}] {n:6,} templates -> {int(q):5,} queries  "
                      f"{s_mean:6.2f}s  ({1000 * s_mean / n_molecules:5.2f} ms/mol)  "
                      f"{a_mean / n_pairs:7.3f} applications/pair")
        return records

    records = load_or_compute(OUT / "benchmark_filterability.csv", compute, cfg)

    styles = {LIB_CONSTRUCTED: (C_BLUE, "o"), LIB_EXTRACTED: (C_RED, "s"),
              "SmartReact (constructed)": (C_BLUE, "o"),  # legacy CSV keys
              "USPTO (auto-extracted)": (C_RED, "s")}
    by_lib: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lib[str(r["library"])].append(r)

    fig, ax = new_figure()
    for label, recs in by_lib.items():
        recs = sorted(recs, key=lambda r: r["n_templates"])
        color, marker = styles.get(label, (C_GREEN, "^"))
        # Derived here rather than read from the CSV so that result files
        # written before this column existed still redraw under --reuse.
        ys = [r["n_admitted"] / r["n_pairs"] for r in recs]
        es = [r.get("n_admitted_std", 0.0) / r["n_pairs"] for r in recs]
        plot_series(ax, [r["n_templates"] for r in recs], ys, color, label, marker, stds=es)
    # Linear on both axes, anchored at the origin: the gap between the two
    # libraries is a ratio the reader should be able to read off directly.
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # "per pair" is load-bearing: without a denominator the number 1.11 has no
    # meaning. Oriented applications, so a template that matches a pair both
    # ways counts twice -- hence "applications", not "templates matched".
    finish_axes(ax, "Reaction templates in library",
                "Mean template applications per pair (after filtering)",
                legend_loc="upper left")
    fig.tight_layout()
    savefig(fig, OUT / "benchmark_filterability")


# ==========================================================================
# Stage: baselines -- four routes, one shared apply path
# ==========================================================================


def _compare(reference: dict, other: dict, ref_name: str, other_name: str) -> dict:
    """Diff two routes' products, pair by pair."""
    missing = extra = differing = 0
    example = ""
    for key in set(reference) | set(other):
        ref, oth = reference.get(key, frozenset()), other.get(key, frozenset())
        if ref == oth:
            continue
        differing += 1
        missing += len(ref - oth)
        extra += len(oth - ref)
        if not example and (ref - oth):
            example = f"{key[0]} + {key[1]} -> {sorted(ref - oth)[0]}"
    return {"comparison": f"{ref_name} vs {other_name}", "pairs_differing": differing,
            "products_only_in_reference": missing, "products_only_in_other": extra,
            "reference": ref_name, "other": other_name, "example_missing": example}


def _rationale_sample(templates, smiles, keys, triples, limit=25) -> list[dict]:
    """SMARTS-RX rationale next to the raw LHS SMARTS, for the same triples."""
    rng = random.Random(SEED)
    rows = []
    for m1, m2, t in sorted(rng.sample(list(triples), min(limit, len(triples))),
                            key=lambda x: x[2]):
        templ = templates[t]
        cats = templ.reactant_categories
        if isinstance(cats, str):
            cats = tuple(c.strip() for c in cats.split("."))
        lhs = reactant_patterns(templ.smarts, strip_maps=True) or ("?", "?")
        rows.append({
            "reaction_name": templ.name, "template_id": templ.template_id,
            "reactant_a": smiles[m1], "reactant_b": smiles[m2],
            "smartsrx_category_a": cats[0] if cats else "",
            "smartsrx_category_b": cats[1] if len(cats) > 1 else "",
            "smartsrx_keys_a": ";".join(sorted(keys.get(smiles[m1], set()))),
            "smartsrx_keys_b": ";".join(sorted(keys.get(smiles[m2], set()))),
            "exact_lhs_query_a": lhs[0][:200],
            "exact_lhs_query_b": lhs[1][:200] if len(lhs) > 1 else ""})
    return rows


def stage_baselines(cfg: Config) -> None:
    # No figure of its own, so --reuse just skips it.
    done = [OUT / "benchmark_baselines.csv", OUT / "benchmark_baselines_equivalence.csv",
            OUT / "benchmark_rationale_sample.csv"]
    if cfg.reuse and all(p.exists() for p in done):
        print("  reusing benchmark_baselines*.csv (--reuse)")
        return

    smiles = load_building_blocks(cfg.building_blocks, n=cfg.n_baseline_molecules)
    molecules = parse_molecules(smiles)
    templates = load_templates("all")
    smarts_list = [t.smarts for t in templates]
    n_pairs = len(smiles) * (len(smiles) - 1) // 2
    brute_n = cfg.n_brute_pairs
    print(f"  {len(smiles):,} molecules, {n_pairs:,} pairs, {len(templates):,} templates")

    # Hard-coded, not a setting: FilterCatalog.GetMatches cannot thread while
    # the other two routes can, so a multi-core comparison would report thread
    # counts as filter quality. The shared apply path would parallelise fairly,
    # but pairs_per_s is filter_s + apply_s, so both stay pinned.
    cc = 1
    print("  routes timed on one core so the filters are compared on equal footing")

    records: list[dict] = []
    outputs: dict[str, dict] = {}

    def run(screen: ScreenResult, pairs_covered: int) -> None:
        secs, productive, kept, by_pair = apply_triples(
            screen.triples, smarts_list, smiles, cc)
        outputs[screen.route] = by_pair
        unique = len({p for ps in by_pair.values() for p in ps})
        records.append({
            "method": screen.route, "n_pairs": pairs_covered,
            "n_queries": screen.n_queries, "n_queries_unit": screen.query_kind,
            "n_candidate_applications": len(screen.triples),
            "n_productive": productive,
            "n_products_summed_over_applications": kept,
            "n_unique_products": unique,
            "filter_s": screen.seconds, "apply_s": secs})
        print(f"    {screen.n_queries:,} queries, {len(screen.triples):,} admitted, "
              f"filter {screen.seconds:.2f}s, apply {secs:.2f}s, {unique:,} products")

    print("  [filtercatalog]")
    run(screen_filtercatalog(smarts_list, molecules), n_pairs)
    print("  [substructlibrary]")
    run(screen_substructlibrary(smarts_list, molecules, cc), n_pairs)
    print("  [smartreact]")
    sr_screen, sr_keys = screen_smartsrx(templates, smiles, cc)
    run(sr_screen, n_pairs)

    # Brute force cannot run at full scale. A *random* sample of the same pairs:
    # a contiguous prefix would draw every pair from the first few molecules and
    # its yield per pair would not be comparable.
    print(f"  [brute] on a random {brute_n:,}-pair subsample")
    sub_pairs = sample_pair_indices(len(smiles), brute_n)
    brute_pairs = {(smiles[a], smiles[b]) if smiles[a] <= smiles[b] else (smiles[b], smiles[a])
                   for a, b in sub_pairs}
    stream = ((a, b, t) for t in range(len(templates)) for m1, m2 in sub_pairs
              for a, b in ((m1, m2), (m2, m1)))
    secs, productive, kept, by_pair = apply_triples(
        stream, smarts_list, smiles, cc)
    outputs["brute"] = by_pair
    records.append({
        "method": "brute", "n_pairs": len(sub_pairs), "n_queries": 0,
        "n_queries_unit": "none (no prefilter)",
        "n_candidate_applications": len(sub_pairs) * len(templates) * 2,
        "n_productive": productive,
        "n_products_summed_over_applications": kept,
        "n_unique_products": len({p for ps in by_pair.values() for p in ps}),
        "filter_s": 0.0, "apply_s": secs})

    for rec in records:
        rec["total_s"] = round(rec["filter_s"] + rec["apply_s"], 2)
        rec["filter_s"] = round(rec["filter_s"], 2)
        rec["apply_s"] = round(rec["apply_s"], 2)
        rec["pairs_per_s"] = round(rec["n_pairs"] / rec["total_s"], 1) if rec["total_s"] else 0
        rec["applications_per_pair"] = round(
            rec["n_candidate_applications"] / rec["n_pairs"], 2)
        rec["n_molecules"] = len(smiles)
        rec["n_templates"] = len(templates)
        rec["comparison_cores"] = cc

    print(f"\n  {'method':18s} {'pairs':>8s} {'filter s':>9s} {'apply s':>9s} "
          f"{'pairs/s':>10s} {'appl/pair':>10s} {'products':>9s}")
    for r in records:
        print(f"  {r['method']:18s} {r['n_pairs']:8,} {r['filter_s']:9.2f} "
              f"{r['apply_s']:9.2f} {r['pairs_per_s']:10.1f} "
              f"{r['applications_per_pair']:10.2f} {r['n_unique_products']:9,}")
    print(f"\n  Note: brute ran on {brute_n:,} pairs, not {n_pairs:,}. Its product count is "
          f"smaller by\n  construction; the equivalence table below restricts every route to "
          f"brute's own pairs.")
    write_csv(records, OUT / "benchmark_baselines.csv", cfg)

    checks = [_compare(outputs["filtercatalog"], outputs[n], "filtercatalog", n)
              for n in ("substructlibrary", "smartreact")]
    checks += [_compare(outputs["brute"], {k: v for k, v in outputs[n].items()
                                           if k in brute_pairs}, "brute", n)
               for n in ("filtercatalog", "substructlibrary", "smartreact")]
    print()
    for c in checks:
        print(f"    {c['comparison']:38s} pairs differing {c['pairs_differing']:,}  "
              f"only-in-{c['reference']} {c['products_only_in_reference']:,}  "
              f"only-in-{c['other']} {c['products_only_in_other']:,}")
    write_csv(checks, OUT / "benchmark_baselines_equivalence.csv", cfg)
    write_csv(_rationale_sample(templates, smiles, sr_keys, sr_screen.triples),
              OUT / "benchmark_rationale_sample.csv", cfg)


# ==========================================================================
# Stage: scaling -- throughput vs template count, every route
# ==========================================================================

TEMPLATE_COUNTS = [100, 500, 1000, 5000]
ROUTES = ("brute", "filtercatalog", "substructlibrary", "smartreact")
# All four routes are measured and written to the CSV; the figure plots only
# smartreact against brute, which is the comparison the paper makes. The RDKit
# prefilters are the subject of the filterability stage instead.


def _timed_route(route, templates, smiles, molecules, cores) -> tuple[float, float, int]:
    """(screen seconds, apply seconds, admitted) -- all routes share apply_triples."""
    smarts = [t.smarts for t in templates]
    if route == "brute":
        n_pairs = len(smiles) * (len(smiles) - 1) // 2
        stream = ((a, b, t) for t in range(len(templates))
                  for m1, m2 in itertools.combinations(range(len(smiles)), 2)
                  for a, b in ((m1, m2), (m2, m1)))
        # Same batch size as the filtered routes: batching changes scheduling
        # overhead, which would otherwise leak into the timing comparison.
        secs, *_ = apply_triples(stream, smarts, smiles, cores, collect=False)
        return 0.0, secs, n_pairs * len(templates) * 2
    if route == "filtercatalog":
        screen = screen_filtercatalog(smarts, molecules)
    elif route == "substructlibrary":
        screen = screen_substructlibrary(smarts, molecules, cores)
    else:
        screen, _ = screen_smartsrx(templates, smiles, cores)
    secs, *_ = apply_triples(screen.triples, smarts, smiles, cores, collect=False)
    return screen.seconds, secs, len(screen.triples)


def stage_scaling(cfg: Config) -> None:
    def compute() -> list[dict]:
        smiles = load_building_blocks(cfg.building_blocks, n=cfg.n_scaling_molecules)
        molecules = parse_molecules(smiles)
        templates = load_templates("all")
        n_total, n_pairs = len(templates), len(smiles) * (len(smiles) - 1) // 2
        counts = sorted({min(c, n_total) for c in TEMPLATE_COUNTS} | {n_total})
        # Hard-coded for the same reason as in baselines: the three prefilters
        # do not thread alike, so a multi-core sweep would compare thread counts.
        cc = 1
        print(f"  {len(smiles):,} molecules, {n_pairs:,} pairs, sweep {counts}")
        print("  routes timed on one core so the filters are compared on equal footing")

        # Absorbs first-touch RDKit and allocator cost, and nothing more:
        # _apply_worker_init clears the compiled-reaction cache and screen_smartsrx
        # rebuilds the KeyGenerator on every call, so each timed run recompiles
        # what it touches. All four routes pay that equally, so it does not tilt
        # the comparison -- but these are not steady-state throughputs.
        _timed_route("smartreact", templates[:50], smiles[:5], molecules[:5], 1)
        _timed_route("brute", templates[:50], smiles[:5], molecules[:5], 1)

        rng = random.Random(SEED + 1)
        records = []
        for n in counts:
            row: dict = {"n_templates": n, "n_pairs": n_pairs, "n_molecules": len(smiles),
                         "n_repeats": cfg.scaling_repeats, "comparison_cores": cc}
            # Drawn once per n and shared by every route: drawing inside the
            # route loop would time each route on a different set of templates,
            # so the curves would not describe the same workload.
            subsets = [templates if n == n_total else rng.sample(templates, n)
                       for _ in range(cfg.scaling_repeats)]
            for route in ROUTES:
                pps, admitted = [], 0
                for subset in subsets:
                    s, a, admitted = _timed_route(route, subset, smiles, molecules, cc)
                    pps.append(n_pairs / (s + a))
                mean, std = mean_std(pps)
                row[f"{route}_mean"] = round(mean, 2)
                row[f"{route}_std"] = round(std, 2)
                row[f"{route}_admitted"] = admitted
            base = row["brute_mean"]
            for route in ROUTES:
                if route != "brute":
                    row[f"{route}_speedup_vs_brute"] = (
                        round(row[f"{route}_mean"] / base, 2) if base else 0)
            records.append(row)
            print(f"  n={n:5d}  " + "  ".join(f"{r[:6]} {row[f'{r}_mean']:8.1f}" for r in ROUTES))
        return records

    records = load_or_compute(OUT / "benchmark_scaling.csv", compute, cfg)

    # The CSV keeps all four routes; the figure shows only the two the paper
    # argues about. Log y, because brute force runs ~250x slower than SmartReact
    # at the full library and would otherwise lie flat on the axis.
    records = sorted(records, key=lambda r: r["n_templates"])
    xs = [r["n_templates"] for r in records]
    fig, ax = new_figure()
    for route, color, marker, label in (
            ("smartreact", C_BLUE, "o", "Filtered by functional groups"),
            ("brute", C_RED, "s", "Not filtered")):
        plot_series(ax, xs, [r[f"{route}_mean"] for r in records], color, label, marker,
                    stds=[r[f"{route}_std"] for r in records])
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    # Same label as Fig 1a/1b, but the count is not the same quantity: those
    # deduplicate before comparing libraries, this one runs the library as
    # shipped because the duplicate rows are part of the real workload.
    finish_axes(ax, "Reaction templates in library", "Pairs / second")
    fig.tight_layout()
    savefig(fig, OUT / "benchmark_scaling")


# ==========================================================================
# Stage: parallel -- weak scaling over cores
# ==========================================================================


def stage_parallel(cfg: Config) -> None:
    """Keys are precomputed once, so the timed region is reaction application.

    Under weak scaling the pair count grows with the core count while the
    molecule pool stays fixed, so classification -- once per molecule, not once
    per pair -- would otherwise be amortised over more pairs at every step up.
    That alone produced an apparent 10.7x speedup from 1 to 4 cores.
    """
    def compute() -> list[dict]:
        pool = load_building_blocks(cfg.building_blocks)
        core_counts = [2**i for i in range(int(math.log2(max(1, cfg.cores))) + 1)]
        if core_counts[-1] != cfg.cores:
            core_counts.append(cfg.cores)
        needed = cfg.pairs_per_core * max(core_counts)
        available = len(pool) * (len(pool) - 1) // 2
        if needed > available:
            raise ValueError(f"needs {needed:,} distinct pairs, pool has {available:,}")
        print(f"  pool {len(pool):,}, sweep {core_counts}, {cfg.pairs_per_core} pairs/core")

        with KeyGenerator(n_cores=cfg.cores) as keygen:
            keys = preprocess_smiles(list(pool), keygen)

        records = []
        for n_cores in core_counts:
            n_pairs = cfg.pairs_per_core * n_cores
            pairs = sample_pairs(pool, n_pairs)
            enum = ReactionEnumerator(n_cores=n_cores)
            try:
                # Warm *this* enumerator, on the pairs it is about to be timed
                # on: its pool is created lazily and its workers' compiled-
                # reaction caches start empty, so a cold first repeat charges
                # pool spawn and template compilation to the measurement. Both
                # grow with n_cores, and the 1-core point runs serially and
                # never spawns a pool at all, so leaving them in bends the
                # curve away from linear.
                enum.enumerate_pairs(pairs, parallel=n_cores > 1, precomputed_keys=keys)
                pps = []
                for _ in range(cfg.parallel_repeats):
                    t0 = time.perf_counter()
                    enum.enumerate_pairs(pairs, parallel=n_cores > 1, precomputed_keys=keys)
                    pps.append(n_pairs / (time.perf_counter() - t0))
            finally:
                enum.close()
            mean, std = mean_std(pps)
            records.append({"n_cores": n_cores, "n_pairs": n_pairs, "mean_pps": round(mean, 2),
                            "std_pps": round(std, 2), "n_building_blocks": len(pool),
                            "keys_precomputed": True, "n_repeats": cfg.parallel_repeats})
            print(f"  cores={n_cores:3d}  {mean:9.1f} +/- {std:6.1f} pairs/s")

        base = records[0]["mean_pps"]
        for r in records:
            r["speedup_vs_1_core"] = round(r["mean_pps"] / base, 2) if base else 0.0
        return records

    records = load_or_compute(OUT / "benchmark_parallel_cores.csv", compute, cfg)

    records = sorted(records, key=lambda r: r["n_cores"])
    xs = [r["n_cores"] for r in records]
    ys = [r["mean_pps"] for r in records]
    fig, ax = new_figure()
    plot_series(ax, xs, ys, C_BLUE, "Measured", "o", stds=[r["std_pps"] for r in records])
    ax.plot(xs, [ys[0] * x / xs[0] for x in xs], "--", color="0.55", linewidth=1.2,
            label="Ideal linear scaling")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    finish_axes(ax, "CPU cores", "Pairs / second", legend_loc="upper left")
    fig.tight_layout()
    savefig(fig, OUT / "benchmark_parallel_cores")


# ==========================================================================
# Stage: precompute -- classification reuse
# ==========================================================================


def _naive_worker_init() -> None:
    global _NAIVE_ENUM
    _NAIVE_ENUM = ReactionEnumerator(n_cores=1)


def _naive_worker_batch(pairs: list[tuple[str, str]]) -> list:
    """Classify both reactants on every pair occurrence -- no reuse at all."""
    assert _NAIVE_ENUM is not None
    return [r for s1, s2 in pairs for r in _NAIVE_ENUM.enumerate_pair(s1, s2)]


def stage_precompute(cfg: Config) -> None:
    """Reuse factor r = mean pair occurrences per molecule, so M molecules in N
    pairs gives r = 2N/M. Naive classifies 2N times, per-call deduplication once
    per distinct molecule, precomputed keys move that cost out of the call."""
    def compute() -> list[dict]:
        pool = load_building_blocks(cfg.building_blocks)
        n_blocks = len(pool)
        max_pairs = n_blocks * (n_blocks - 1) // 2
        print(f"  {n_blocks:,} blocks, {cfg.cores} cores, reuse {list(cfg.precompute_reuse)}")

        enum = ReactionEnumerator(n_cores=cfg.cores)
        naive_pool = ProcessPoolExecutor(max_workers=cfg.cores, initializer=_naive_worker_init)

        def run_naive(batch: list[tuple[str, str]]) -> None:
            chunk = max(1, len(batch) // cfg.cores)
            for f in [naive_pool.submit(_naive_worker_batch, batch[i:i + chunk])
                      for i in range(0, len(batch), chunk)]:
                f.result()

        records = []
        try:
            warm = sample_pairs(pool, 50, SEED + 7)
            run_naive(warm)
            enum.enumerate_pairs(warm, parallel=cfg.cores > 1)

            for reuse in cfg.precompute_reuse:
                n_pairs = int(reuse * n_blocks / 2)
                if n_pairs > max_pairs:
                    print(f"  reuse={reuse:3d} SKIPPED: needs {n_pairs:,} pairs")
                    continue
                pairs = sample_pairs(pool, n_pairs)
                unique = list({s for pair in pairs for s in pair})

                naive, standard, pre, pre_t = [], [], [], []
                for _ in range(cfg.precompute_repeats):
                    t0 = time.perf_counter()
                    run_naive(pairs)
                    naive.append(n_pairs / (time.perf_counter() - t0))
                    t0 = time.perf_counter()
                    enum.enumerate_pairs(pairs, parallel=cfg.cores > 1)
                    standard.append(n_pairs / (time.perf_counter() - t0))
                    t0 = time.perf_counter()
                    keys = preprocess_smiles(unique, enum.keygen)
                    pre_t.append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    enum.enumerate_pairs(pairs, parallel=cfg.cores > 1, precomputed_keys=keys)
                    pre.append(n_pairs / (time.perf_counter() - t0))

                pre_mean, pre_std = mean_std(pre_t)
                row = {"reuse_factor": reuse, "actual_reuse": round(2 * n_pairs / len(unique), 2),
                       "n_pairs": n_pairs, "n_building_blocks": n_blocks,
                       "n_unique_smiles": len(unique), "n_classifications_naive": 2 * n_pairs,
                       "n_classifications_dedup": len(unique), "cores": cfg.cores,
                       "preprocess_mean_s": round(pre_mean, 3),
                       "preprocess_std_s": round(pre_std, 3), "n_repeats": cfg.precompute_repeats}
                for name, vals in (("naive", naive), ("standard", standard), ("precomputed", pre)):
                    m, s = mean_std(vals)
                    row[f"{name}_mean_pps"] = round(m, 2)
                    row[f"{name}_std_pps"] = round(s, 2)
                row["dedup_speedup"] = (
                    round(row["standard_mean_pps"] / row["naive_mean_pps"], 2)
                    if row["naive_mean_pps"] else 0)
                row["precompute_speedup"] = round(
                    row["precomputed_mean_pps"] / row["standard_mean_pps"], 2
                ) if row["standard_mean_pps"] else 0
                records.append(row)
                print(f"  reuse={reuse:3d} ({n_pairs:7,} pairs)  "
                      f"naive {row['naive_mean_pps']:8.1f}  "
                      f"standard {row['standard_mean_pps']:8.1f} ({row['dedup_speedup']:5.2f}x)  "
                      f"precomputed {row['precomputed_mean_pps']:8.1f} "
                      f"({row['precompute_speedup']:4.2f}x)")
        finally:
            naive_pool.shutdown(wait=True, cancel_futures=True)
            enum.close()

        if not records:
            raise RuntimeError("No reuse factor fitted the building-block pool.")
        return records

    records = load_or_compute(OUT / "benchmark_precomputed.csv", compute, cfg)

    # The speedup panel is dropped: both ratios are already readable as the
    # vertical gaps between these three curves. dedup_speedup and
    # precompute_speedup stay in the CSV.
    records = sorted(records, key=lambda r: r["actual_reuse"])
    xs = [r["actual_reuse"] for r in records]
    fig, ax = new_figure()
    for name, label, color, marker in (
            ("naive", "Naive (every occurrence)", C_GREY, "^"),
            ("standard", "Per-call deduplication", C_BLUE, "o"),
            ("precomputed", "Precomputed keys", C_GREEN, "s")):
        plot_series(ax, xs, [r[f"{name}_mean_pps"] for r in records], color, label, marker,
                    stds=[r[f"{name}_std_pps"] for r in records])
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    # The three curves span the whole panel, so make headroom for the legend
    # rather than dropping it on top of one of them.
    top = max(r[f"{n}_mean_pps"] for r in records for n in ("naive", "standard", "precomputed"))
    ax.set_ylim(top=top * 4)
    finish_axes(ax, "Reuse factor (pair occurrences per molecule)", "Pairs / second",
                legend_loc="upper left")
    fig.tight_layout()
    savefig(fig, OUT / "benchmark_precomputed")


# ==========================================================================
# Stage: casestudy -- all pairs of the public building blocks
# ==========================================================================

# Separator inside the products file's reaction_types field. No reaction name
# contains it, so splitting on it round-trips the set of producing reactions.
REACTION_SEP = ";"


def produced_by(reaction_types: Iterable[str], name: str) -> np.ndarray:
    """Mask of products the named reaction reaches, shared ones included."""
    return np.array([name in str(t).split(REACTION_SEP) for t in reaction_types])


def casestudy_building_blocks(cfg: Config) -> tuple[list[str], Path]:
    """The committed 1,000 if that is what was asked for, else a subsample
    written under data/paper/ so a --quick run cannot pollute public_data/."""
    n = cfg.n_casestudy_molecules
    committed = PUBLIC_BB.parent / f"building_blocks_{n}.csv"
    if cfg.building_blocks is None and committed.exists():
        return load_building_blocks(committed), committed
    sampled = CASESTUDY_DIR / f"building_blocks_{n}.csv"
    # Only reuse a previous subsample when the source is the default one: this
    # file is named for n alone, so an explicit --building-blocks must re-draw
    # rather than silently enumerate whatever the last run happened to sample.
    if cfg.building_blocks is None and sampled.exists():
        return load_building_blocks(sampled), sampled
    smiles = load_building_blocks(cfg.building_blocks, n=n)
    CASESTUDY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": smiles}).to_csv(sampled, index=False)
    return smiles, sampled


def stage_casestudy(cfg: Config) -> None:
    products = CASESTUDY_DIR / "enumerated_products.parquet"
    metrics_path = CASESTUDY_DIR / "enumeration_metrics.csv"
    if cfg.reuse and products.exists() and metrics_path.exists():
        print(f"  reusing {products.name} (--reuse)")
        return

    smiles_list, bb_path = casestudy_building_blocks(cfg)
    CASESTUDY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {len(smiles_list):,} building blocks from {bb_path.name}")

    t0 = time.perf_counter()
    keygen = KeyGenerator(n_cores=cfg.cores)
    keys = preprocess_smiles(smiles_list, keygen)
    keygen.close()
    preprocess_time = time.perf_counter() - t0

    pairs = list(itertools.combinations(smiles_list, 2))
    print(f"  enumerating {len(pairs):,} pairs on {cfg.cores} cores")
    t0 = time.perf_counter()
    with ReactionEnumerator(n_cores=cfg.cores) as enum:
        results = enum.enumerate_pairs(pairs, parallel=True, precomputed_keys=keys)
    enum_time = time.perf_counter() - t0
    total = preprocess_time + enum_time

    # Every reaction that reaches a product is recorded, not just the first.
    #
    # 540 templates in the library appear under two reaction names apiece --
    # B-H/snar_amine, cross_electrophile_coupling/ullmann_X and
    # snar_alcohol/ullmann_phenol each encode an identical SMARTS -- so those
    # products come out of the enumerator under both names, byte for byte. A
    # first-writer-wins label would hand them to whichever result happened to
    # arrive first, which is an artefact of scheduling rather than chemistry.
    producers: dict[str, set[str]] = defaultdict(set)
    for r in results:
        for prod in r.products:
            producers[prod].add(r.reaction_name)
    df = pd.DataFrame([{"smiles": s, "reaction_types": REACTION_SEP.join(sorted(names)),
                        "n_reaction_types": len(names)}
                       for s, names in producers.items()])
    df.to_parquet(CASESTUDY_DIR / "enumerated_products.parquet", index=False)

    # Per name: products that reaction can reach. A product reachable by two
    # reactions counts once for each, so these sum past unique_products.
    per_name: Counter = Counter()
    for names in producers.values():
        per_name.update(names)
    combos = df["reaction_types"].value_counts()
    shared = combos[[REACTION_SEP in c for c in combos.index]]
    n_shared = int(df["n_reaction_types"].gt(1).sum())

    metrics = [
        {"metric": "building_blocks", "value": len(smiles_list)},
        {"metric": "pairs_evaluated", "value": len(pairs)},
        {"metric": "total_products", "value": sum(len(r.products) for r in results)},
        {"metric": "unique_products", "value": len(df)},
        {"metric": "products_reachable_by_one_reaction", "value": len(df) - n_shared},
        {"metric": "products_reachable_by_multiple_reactions", "value": n_shared},
        {"metric": "productive_reaction_types", "value": len(per_name)},
        {"metric": "distinct_reaction_type_combinations", "value": int(combos.size)},
        {"metric": "preprocess_time_s", "value": round(preprocess_time, 2)},
        {"metric": "enumeration_time_s", "value": round(enum_time, 2)},
        {"metric": "total_time_s", "value": round(total, 2)},
        {"metric": "pairs_per_second_enum", "value": round(len(pairs) / enum_time, 1)},
        {"metric": "pairs_per_second_total", "value": round(len(pairs) / total, 1)},
    ]
    metrics += [{"metric": f"products_{r}", "value": int(c)} for r, c in per_name.most_common()]
    # Spelled out separately so the overlaps are readable rather than implied.
    metrics += [{"metric": f"products_shared_{c}", "value": int(n)} for c, n in shared.items()]
    print(f"  {len(df):,} unique products across {len(per_name)} reaction types in "
          f"{total:.1f}s ({len(pairs) / total:,.0f} pairs/s)")
    print(f"  {n_shared:,} of them are reachable by more than one reaction "
          f"({combos.size} distinct reaction-name combinations)")
    for c, n in shared.items():
        print(f"    {c.replace(REACTION_SEP, ' + '):55s} {n:7,} products")
    write_csv(metrics, CASESTUDY_DIR / "enumeration_metrics.csv", cfg)


# ==========================================================================
# Stage: chemspace -- fingerprints, UMAP, highlight panels
# ==========================================================================

FP_RADIUS, FP_NBITS = 2, 2048
UMAP_METRIC, UMAP_N_NEIGHBORS, UMAP_MIN_DIST = "jaccard", 100, 0.7
HIGHLIGHT = ["C_C_decarboxylation", "snar_amine", "B-H", "mitsunobu", "amide_coupling"]
HIGHLIGHT_COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)


def _smiles_to_fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _FP_GEN.GetFingerprintAsNumPy(mol)


def _properties(smiles_list: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    mws, logps = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        mws.append(Descriptors.MolWt(mol) if mol else np.nan)
        logps.append(Descriptors.MolLogP(mol) if mol else np.nan)
    return np.array(mws, float), np.array(logps, float)


def stage_chemspace(cfg: Config) -> None:
    from umap import UMAP

    n = cfg.n_casestudy_molecules
    products_path = CASESTUDY_DIR / "enumerated_products.parquet"
    if not products_path.exists():
        raise FileNotFoundError(f"{products_path.name} missing -- run 'casestudy' first.")
    bb_smiles, _ = casestudy_building_blocks(cfg)
    df_prod = pd.read_parquet(products_path)
    if "reaction_types" not in df_prod.columns:
        raise RuntimeError(
            f"{products_path.name} predates multi-reaction labelling (columns: "
            f"{list(df_prod.columns)}). Re-run the 'casestudy' stage.")
    prod_smiles = df_prod["smiles"].tolist()
    # One entry per product, holding *every* reaction that reaches it.
    prod_types = df_prod["reaction_types"].tolist()
    n_shared = int(df_prod["n_reaction_types"].gt(1).sum())
    print(f"  {len(bb_smiles):,} building blocks, {len(prod_smiles):,} products "
          f"({n_shared:,} reachable by more than one reaction)")

    # Fingerprints and the embedding are cached: re-running for figure tweaks
    # should not repeat half a million fingerprints or a single-threaded UMAP.
    # Both names carry the molecule count as well as n, because casestudy
    # regenerates the product set -- upgrading the package changes it, which is
    # the whole point of reading the installed one -- and a cache keyed on n
    # alone would be reused against the new labels and mislabel every point.
    all_smiles = bb_smiles + prod_smiles
    n_all = len(all_smiles)
    fp_cache = CASESTUDY_DIR / f"morgan_fingerprints_{n}_{n_all}.npz"
    if fp_cache.exists():
        cache = np.load(fp_cache)
        X, valid = cache["X"], cache["valid_idx"].tolist()
    else:
        print(f"  fingerprinting {len(all_smiles):,} molecules on {cfg.cores} cores")
        with ProcessPoolExecutor(max_workers=cfg.cores) as ex:
            fps = list(ex.map(_smiles_to_fp, all_smiles, chunksize=500))
        valid = [i for i, fp in enumerate(fps) if fp is not None]
        X = np.vstack([fps[i] for i in valid])
        np.savez_compressed(fp_cache, X=X, valid_idx=np.array(valid))

    n_bb = len(bb_smiles)
    is_bb = np.array([i < n_bb for i in valid])
    labels = np.array([prod_types[i - n_bb] if i >= n_bb else "" for i in valid])

    md = f"{UMAP_MIN_DIST:.1f}".replace(".", "p")
    emb_cache = CASESTUDY_DIR / f"umap_{UMAP_METRIC}_nn{UMAP_N_NEIGHBORS}_md{md}_{n}_{n_all}.npz"
    if emb_cache.exists():
        X_umap = np.load(emb_cache)["X_umap"]
        if X_umap.shape[0] != X.shape[0]:
            raise RuntimeError(f"{emb_cache.name} has {X_umap.shape[0]:,} rows but "
                               f"{fp_cache.name} has {X.shape[0]:,}; delete both and re-run.")
    else:
        # UMAP silently drops to one thread whenever random_state is set. Kept
        # deliberately: a reproducible embedding is worth the wall time.
        print("  running UMAP (single-threaded, reproducible)")
        X_umap = UMAP(n_components=2, random_state=42, metric=UMAP_METRIC,
                      n_neighbors=min(UMAP_N_NEIGHBORS, max(2, len(valid) - 1)),
                      min_dist=UMAP_MIN_DIST).fit_transform(X)
        np.savez_compressed(emb_cache, X_umap=X_umap)

    bb_xy, prod_xy = X_umap[is_bb], X_umap[~is_bb]
    prod_labels = labels[~is_bb]
    xlim = (X_umap[:, 0].min() - 0.5, X_umap[:, 0].max() + 0.5)
    ylim = (X_umap[:, 1].min() - 0.5, X_umap[:, 1].max() + 0.5)

    series = pd.Series(prod_labels)
    order = series.value_counts().sort_values(ascending=False).index.tolist()
    c1, c2 = plt.get_cmap("tab20"), plt.get_cmap("tab20b")
    cmap = {t: (c1(i) if i < 20 else c2(i - 20)) for i, t in enumerate(order)}

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(bb_xy[:, 0], bb_xy[:, 1], c="black", s=15, alpha=0.8, marker="*",
               label="Building blocks", zorder=3)
    for rtype in order:
        mask = (series == rtype).values
        # A product reachable by two reactions is its own category, drawn and
        # labelled as such -- it is not silently folded into either one.
        ax.scatter(prod_xy[mask, 0], prod_xy[mask, 1], c=[cmap[rtype]], s=4,
                   alpha=0.5, label=rtype.replace(REACTION_SEP, " + "), zorder=1)
    ax.set_title(f"Chemical space covered by SmartReact enumeration\n"
                 f"({len(series):,} products from {int(is_bb.sum()):,} building blocks)",
                 fontsize=13)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, markerscale=2)
    fig.tight_layout()
    savefig(fig, CASESTUDY_DIR / "chemical_space")

    # Membership, not equality: a panel for snar_amine shows every product that
    # reaction reaches, including the ones B-H reaches identically.
    highlight_masks = {r: produced_by(prod_labels, r) for r in HIGHLIGHT}
    shared_mask = np.array([REACTION_SEP in str(t) for t in prod_labels])
    present = [r for r in HIGHLIGHT if highlight_masks[r].any()]
    if not present:
        print("  no highlight reactions present; skipping panels")
        return
    colors = dict(zip(HIGHLIGHT, HIGHLIGHT_COLORS, strict=True))

    fig = plt.figure(figsize=(6 * len(present), 6))
    gs = GridSpec(1, len(present), figure=fig, wspace=0.25)
    for col, reaction in enumerate(present):
        ax = fig.add_subplot(gs[0, col])
        hi = highlight_masks[reaction]
        ax.scatter(prod_xy[~hi, 0], prod_xy[~hi, 1], c="#CCCCCC", s=2, alpha=0.4,
                   linewidths=0, zorder=1, label="Other products")
        ax.scatter(prod_xy[hi, 0], prod_xy[hi, 1], c=colors[reaction], s=4, alpha=0.7,
                   linewidths=0, zorder=2, label=f"Products ({reaction})")
        ax.scatter(bb_xy[:, 0], bb_xy[:, 1], c="black", s=15, alpha=0.9, marker="*",
                   zorder=3, label="Building blocks")
        n_sh = int((hi & shared_mask).sum())
        ax.set_title(f"{reaction}\n({int(hi.sum()):,} products, "
                     f"{n_sh:,} also reachable by another reaction)", fontsize=11)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set(xlim=xlim, ylim=ylim)
        ax.legend(loc="upper right", fontsize=7, markerscale=2)
        if col == 0:
            ax.set_ylabel("UMAP 2", fontsize=9)
    savefig(fig, CASESTUDY_DIR / "chemical_space_highlights")

    print("  computing MW and logP")
    bb_mw, bb_logp = _properties(bb_smiles)
    props = {r: _properties(
        df_prod.loc[produced_by(df_prod["reaction_types"], r), "smiles"].tolist())
        for r in present}
    all_mw = np.concatenate([bb_mw] + [props[r][0] for r in present])
    all_logp = np.concatenate([bb_logp] + [props[r][1] for r in present])
    mw_lim = (np.nanmin(all_mw), np.nanmax(all_mw))
    logp_lim = (np.nanmin(all_logp), np.nanmax(all_logp))
    mw_bins, logp_bins = np.linspace(*mw_lim, 60), np.linspace(*logp_lim, 60)
    bb_ok = np.isfinite(bb_mw) & np.isfinite(bb_logp)

    fig = plt.figure(figsize=(6 * len(present), 7))
    gs = GridSpec(1, len(present), figure=fig, wspace=0.35)
    for col, reaction in enumerate(present):
        inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, col], width_ratios=[4, 1],
                                        height_ratios=[1, 4], hspace=0.05, wspace=0.05)
        ax_main = fig.add_subplot(inner[1, 0])
        ax_top = fig.add_subplot(inner[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)
        color = colors[reaction]
        mw, logp = props[reaction]
        ok = np.isfinite(mw) & np.isfinite(logp)

        ax_main.scatter(mw[ok], logp[ok], c=color, s=3, alpha=0.3, linewidths=0, zorder=2)
        ax_main.scatter(bb_mw[bb_ok], bb_logp[bb_ok], c="black", s=6, alpha=0.4,
                        linewidths=0, zorder=3)
        ax_main.set_xlabel("MW (Da)", fontsize=9)
        ax_main.set(xlim=mw_lim, ylim=logp_lim)
        if col == 0:
            ax_main.set_ylabel("logP", fontsize=9)
        ax_top.set_title(reaction, fontsize=10)
        ax_top.hist(mw[ok], bins=mw_bins, color=color, alpha=0.5, linewidth=0, density=True)
        ax_top.hist(bb_mw[bb_ok], bins=mw_bins, color="black", alpha=0.4, linewidth=0,
                    density=True)
        ax_top.set_xlim(mw_lim)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.set_yticks([])
        ax_right.hist(logp[ok], bins=logp_bins, color=color, alpha=0.5, linewidth=0,
                      orientation="horizontal", density=True)
        ax_right.hist(bb_logp[bb_ok], bins=logp_bins, color="black", alpha=0.4, linewidth=0,
                      orientation="horizontal", density=True)
        ax_right.set_ylim(logp_lim)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        ax_right.set_xticks([])
    savefig(fig, CASESTUDY_DIR / "chemical_space_supplement")


# ==========================================================================
# Stage: coverage -- SMARTS-RX statistics and chord diagrams
# ==========================================================================

# Paul Tol bright palette, indexed by position within a panel.
PANEL_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#EE7733"]

PANELS: dict[str, list[str]] = {
    "Pd-catalyzed C-C coupling": [
        "ART", "heck", "negishi_batch", "negishi_insitu",
        "pinacolatoborylation", "sonogashira", "suzuki"],
    "Other C-C forming": [
        "B-H", "C_C_decarboxylation", "chan_lam",
        "cross_electrophile_coupling", "deoxygenative_coupling", "horner_wadsworth_emmons"],
    "Amide & amine forming": [
        "amide_coupling", "amine_acetylation", "amine_sulfonation",
        "reductive_amination_aldehyde", "reductive_amination_ketone", "urea_formation"],
    "N/S nucleophilic substitution": [
        "mitsunobu", "sn2_nheterocycle", "snar_amine",
        "snar_thiol", "ullmann_X", "williamson"],
    "O-functionalization": [
        "ester_schotten_baumann", "ester_sulfonic_schotten_baumann",
        "esterification", "oxadiazole_condensation", "snar_alcohol", "ullmann_phenol"],
    "Heterocycle synthesis": [
        "imidazole_condensation_acid", "imidazole_condensation_amine",
        "imidazole_Xketone_synthesis", "tetrazole_synthesis",
        "triazole_synthesis_1", "triazole_synthesis_2"],
}

NODE_ORDER = [
    "X", "XF", "XKetone", "Mesylate", "AcidX", "Acid", "Thioacid", "Ester",
    "Sulfonyl", "Phosphate", "Aldehyde", "Ketone", "Alkene/MichaelAcceptor", "Alkyne",
    "Boronate", "Boronic", "Metals", "Nitrile", "Azide", "IsoCyanate", "ThioisoCyanate",
    "Amine", "Amidine", "HydroxyAmidine", "Hydrazine", "NitrogenHeterocycle",
    "Alcohol", "Phenol", "Thiol", "Thiophenol",
]

ANGLE_OFFSET_STEP = 0.04  # radians between arcs joining the same node pair


def _cubic_arc(ax, a1, a2, R, color, alpha=0.82, lw=2.1) -> None:
    x1, y1 = np.cos(a1) * R, np.sin(a1) * R
    x2, y2 = np.cos(a2) * R, np.sin(a2) * R
    verts = [(x1, y1), (x1 * 0.25, y1 * 0.25), (x2 * 0.25, y2 * 0.25), (x2, y2)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor="none", edgecolor=color,
                           alpha=alpha, lw=lw))


def _label_fontsize(name: str) -> float:
    n = len(name)
    return 19.0 if n <= 4 else 17.0 if n <= 7 else 15.0 if n <= 11 else 14.0 if n <= 15 else 13.0


def _draw_panel(ax, cat_pairs, l3_to_l1, node_order, l1_counts, reactions) -> None:
    reaction_color = {r: PANEL_COLORS[i] for i, r in enumerate(reactions)}
    reaction_set = set(reactions)

    seen: set[tuple[str, str, str]] = set()
    arcs: list[tuple[str, str, str]] = []
    for reaction, c1, c2 in cat_pairs:
        if reaction not in reaction_set:
            continue
        a, b = l3_to_l1.get(c1), l3_to_l1.get(c2)
        if a and b and (reaction, a, b) not in seen:
            seen.add((reaction, a, b))
            arcs.append((reaction, a, b))

    active = {n for _, a, b in arcs for n in (a, b)}
    nodes = [n for n in node_order if n in active]
    if not nodes:
        ax.axis("off")
        return
    # Half-slot offset so no node sits exactly at 12 o'clock.
    start = np.pi / 2 - np.pi / len(nodes)
    angles = {node: start - 2 * np.pi * i / len(nodes) for i, node in enumerate(nodes)}

    pair_arcs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for reaction, a, b in arcs:
        pair_arcs[(a, b)].append(reaction)

    R = 1.0
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="#DDDDDD", lw=0.8))

    for (a, b), here in pair_arcs.items():
        ordered = sorted(here, key=lambda r: reactions.index(r))
        for i, reaction in enumerate(ordered):
            offset = (i - (len(here) - 1) / 2) * ANGLE_OFFSET_STEP
            color = reaction_color[reaction]
            if a == b:
                ang = angles[a]
                x, y = np.cos(ang) * R, np.sin(ang) * R
                r_loop = 0.055 + i * 0.04
                ax.add_patch(plt.Circle((x * (1 + r_loop), y * (1 + r_loop)), r_loop,
                                        fill=False, color=color, alpha=0.82, lw=2.1))
            else:
                _cubic_arc(ax, angles[a] + offset, angles[b] + offset, R, color)

    max_count = max(l1_counts.values(), default=1)
    for node in nodes:
        ang = angles[node]
        x, y = np.cos(ang) * R, np.sin(ang) * R
        ax.scatter(x, y, s=35 + 160 * (l1_counts.get(node, 1) / max_count) ** 0.5,
                   color="#222222", zorder=6)
        rot, ha = np.degrees(ang), "left"
        if np.cos(ang) < 0:
            rot, ha = rot + 180, "right"
        ax.text(np.cos(ang) * 1.14, np.sin(ang) * 1.14, node, ha=ha, va="center",
                fontsize=_label_fontsize(node), fontweight="bold", color="#222222",
                rotation=rot, rotation_mode="anchor")

    ax.legend(handles=[mpatches.Patch(color=reaction_color[r], label=r.replace("_", " "))
                       for r in reactions],
              loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=15,
              framealpha=0.97, edgecolor="#CCCCCC", ncol=2, handlelength=1.8, handleheight=1.4)
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-2.8, 1.65)  # extra room keeps the legend inside the axes


def _save_l3_table(reaction_name: str, templates: pd.DataFrame, out_path: Path) -> None:
    df = templates[templates["reaction_name"] == reaction_name].copy()
    df[["cat1", "cat2"]] = df["reactant_categories"].str.split(".", expand=True)
    pairs = df[["cat1", "cat2"]].drop_duplicates().sort_values(["cat1", "cat2"])
    esc = lambda s: s.replace("_", r"\_")  # noqa: E731
    caption = (f"All Level 3 reactant category combinations for the "
               f"{reaction_name.replace('_', ' ').title()} reaction ({len(pairs)} combinations).")
    lines = [f"% {reaction_name} -- {len(pairs)} Level 3 combinations",
             r"\begin{longtable}{ll}",
             f"\\caption{{{caption}}} \\label{{tab:{reaction_name}_l3}} \\\\",
             r"\toprule", r"\textbf{Reactant 1 (L3)} & \textbf{Reactant 2 (L3)} \\",
             r"\midrule", r"\endfirsthead", r"\toprule",
             r"\textbf{Reactant 1 (L3)} & \textbf{Reactant 2 (L3)} \\",
             r"\midrule", r"\endhead", r"\midrule",
             r"\multicolumn{2}{r}{\textit{continued on next page}} \\",
             r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    prev = None
    for _, row in pairs.iterrows():
        if prev is not None and row.cat1 != prev:
            lines.append(r"\midrule")
        lines.append(f"{esc(row.cat1)} & {esc(row.cat2)} \\\\")
        prev = row.cat1
    lines.append(r"\end{longtable}")
    out_path.write_text("\n".join(lines))
    print(f"  saved -> {out_path.name}")


def stage_coverage(cfg: Config) -> None:
    rows = []
    with bundled("keys.txt").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 3:
                rows.append({"level1": parts[0], "level2": parts[1], "level3": parts[2]})
    keys = pd.DataFrame(rows)
    templates = pd.read_csv(bundled("templates.txt"), sep="\t", header=0)
    l3_to_l1 = dict(zip(keys["level3"], keys["level1"], strict=True))
    l3_to_l2 = dict(zip(keys["level3"], keys["level2"], strict=True))

    cat_pairs = [(r["reaction_name"], *str(r["reactant_categories"]).split("."))
                 for _, r in templates.iterrows()
                 if len(str(r["reactant_categories"]).split(".")) == 2]
    all_cats = {c for _, c1, c2 in cat_pairs for c in (c1, c2)}

    used = {
        "Level 1": ({l3_to_l1[c] for c in all_cats if c in l3_to_l1}, keys["level1"].nunique()),
        "Level 2": ({l3_to_l2[c] for c in all_cats if c in l3_to_l2}, keys["level2"].nunique()),
        "Level 3": (all_cats, keys["level3"].nunique()),
    }
    pair_counts = {
        "Level 1": len({frozenset([l3_to_l1.get(a), l3_to_l1.get(b)])
                        for _, a, b in cat_pairs if l3_to_l1.get(a) and l3_to_l1.get(b)}),
        "Level 2": len({frozenset([l3_to_l2.get(a), l3_to_l2.get(b)])
                        for _, a, b in cat_pairs if l3_to_l2.get(a) and l3_to_l2.get(b)}),
        "Level 3": len({frozenset([a, b]) for _, a, b in cat_pairs}),
    }
    n_reactions = len({r for r, *_ in cat_pairs})
    print(f"  {n_reactions} reactions, {len(templates):,} templates")
    records = []
    for level, (used_set, total) in used.items():
        print(f"  {level:<10}{len(used_set):>6} /{total:>6}  {pair_counts[level]:>8,} pairs")
        records.append({"level": level, "used": len(used_set), "total": total,
                        "percent": round(len(used_set) / total * 100, 1),
                        "unique_pairs": pair_counts[level], "n_reactions": n_reactions,
                        "n_templates": len(templates)})
    write_csv(records, OUT / "template_coverage.csv", cfg)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_l3_table("mitsunobu", templates, FIGURES_DIR / "mitsunobu_l3_combinations.tex")

    l1_counts: Counter = Counter()
    for _, c1, c2 in cat_pairs:
        for c in (c1, c2):
            if c in l3_to_l1:
                l1_counts[l3_to_l1[c]] += 1
    used_nodes = {l3_to_l1[c] for _, c1, c2 in cat_pairs for c in (c1, c2) if c in l3_to_l1}
    node_order = [n for n in NODE_ORDER if n in used_nodes]
    node_order += sorted(used_nodes - set(node_order))

    # Grid derived from PANELS rather than fixed at 2x3, so adding a seventh
    # panel gets a row instead of being dropped by the zip.
    ncols = 3
    nrows = math.ceil(len(PANELS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(28 / 3 * ncols, 11 * nrows), squeeze=False)
    fig.subplots_adjust(hspace=0.28, wspace=0.05, top=0.95, bottom=0.06)
    for ax, reactions in zip(axes.flat, PANELS.values(), strict=False):
        _draw_panel(ax, cat_pairs, l3_to_l1, node_order, dict(l1_counts), reactions)
    for ax in list(axes.flat)[len(PANELS):]:
        ax.axis("off")
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"template_coverage_chord_{len(PANELS)}panel.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  saved -> {path.name}")
    plt.close(fig)


# ==========================================================================
# Shared machinery
# ==========================================================================


def bundled(name: str) -> Path:
    """Path to a data file inside the installed package, not the source tree."""
    return Path(str(files("smartreact") / "data" / name))


def mean_std(xs: Sequence[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def write_csv(records: list[dict], path: Path, cfg: Config | None = None) -> None:
    """Write records, stamping each with the run's provenance."""
    if not records:
        raise ValueError(f"Refusing to write an empty CSV to {path}")
    for rec in records:
        rec.setdefault("smartreact_version", smartreact.__version__)
        rec.setdefault("rdkit_version", rdBase.rdkitVersion)
        rec.setdefault("platform", platform.platform())
        if cfg is not None:
            rec.setdefault("run_cores", cfg.cores)
            rec.setdefault("quick_mode", cfg.quick)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for rec in records:
        for key in rec:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"  saved -> {path.name}")


def load_or_compute(path: Path, compute, cfg: Config) -> list[dict]:
    """Reuse a stage's CSV instead of recomputing it, when --reuse is set.

    Every stage writes its numbers before drawing anything, so redrawing a
    figure only needs the CSV back. This is for iterating on figure design
    without paying for the benchmark each time -- not for production runs.
    """
    if cfg.reuse and path.exists():
        print(f"  reusing {path.name} (--reuse)")
        return pd.read_csv(path).to_dict("records")
    records = compute()
    write_csv(records, path, cfg)
    return records


def savefig(fig, stem: Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print(f"  saved -> {stem.name}.pdf/.png")
    plt.close(fig)


def new_figure(figsize: tuple[float, float] = FIG_SIZE):
    return plt.subplots(figsize=figsize)


def plot_series(ax, xs, ys, color, label, marker="o", stds=None) -> None:
    """One series in the shared style, with an optional +/-1 std band."""
    ys = np.asarray(ys, float)
    ax.plot(xs, ys, color=color, marker=marker, markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH, label=label)
    if stds is not None:
        es = np.asarray(stds, float)
        ax.fill_between(xs, ys - es, ys + es, color=color, alpha=BAND_ALPHA)


def finish_axes(ax, xlabel: str, ylabel: str, legend_loc: str | None = "best") -> None:
    """Labels, ticks and grid. No title: these carry captions in the paper."""
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, linestyle="--", alpha=0.5)
    if legend_loc is not None:
        ax.legend(fontsize=FONT_LEGEND, loc=legend_loc)


def parse_molecules(smiles: Sequence[str]) -> list[Chem.Mol]:
    """Parse to RDKit mols, failing loudly rather than handing None to RDKit."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if bad := [s for s, m in zip(smiles, mols, strict=True) if m is None]:
        raise RuntimeError(f"{len(bad)} building blocks failed to parse, "
                           f"first is {bad[0]!r}")
    return mols


def load_building_blocks(path: Path | None = None, n: int | None = None,
                         seed: int = SEED) -> list[str]:
    path = path or PUBLIC_BB
    if not path.exists():
        raise FileNotFoundError(f"No building blocks at {path}")
    if path.suffix == ".parquet":
        smiles = pd.read_parquet(path)["smiles"].dropna().tolist()
    else:
        with path.open(encoding="utf-8", newline="") as fh:
            smiles = [row["smiles"] for row in csv.DictReader(fh)]
    if n is not None and n < len(smiles):
        smiles = random.Random(seed).sample(smiles, n)
    return smiles


def sample_pair_indices(n_items: int, n: int, seed: int = SEED) -> list[tuple[int, int]]:
    """n distinct unordered index pairs. Guarded: without the check, asking for
    more pairs than exist spins in the rejection loop forever."""
    max_pairs = n_items * (n_items - 1) // 2
    if n > max_pairs:
        raise ValueError(f"Requested {n:,} pairs but only {max_pairs:,} exist.")
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < n:
        a, b = rng.sample(range(n_items), 2)
        seen.add((min(a, b), max(a, b)))
    return sorted(seen)


def sample_pairs(pool: Sequence[str], n: int, seed: int = SEED) -> list[tuple[str, str]]:
    return [(pool[a], pool[b]) for a, b in sample_pair_indices(len(pool), n, seed)]


def strip_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetAtomMapNum(0)
    return out


def reactant_patterns(reaction_smarts: str, strip_maps: bool = True) -> tuple[str, ...] | None:
    """SMARTS for each reactant side of a forward template, as RDKit writes them.

    This is the unit a prefilter works in: the number of *distinct* such
    patterns across a library, not the number of templates, sets its cost.
    Atom maps are stripped by default -- they are arbitrary per-template labels,
    and auto-extracted libraries number them independently, which would inflate
    their pattern count and bias the comparison.

    Not canonical: MolToSmarts has no canonical option, so two equivalent queries
    written in different atom orders count as two patterns. We found no
    straightforward fix -- SanitizeMol + CanonicalRankAtoms silently alters 340 of
    these queries -- and the effect is small and in the conservative direction:
    merging patterns that mutually query-query match drops the auto-extracted
    library's count by 4.4% and the constructed library's by 0%, moving the gap
    between them from 12.0x to 11.5x. Counting strings is also what an actual
    FilterCatalog index does.
    """
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    if rxn is None or rxn.GetNumReactantTemplates() == 0:
        return None
    out = []
    for i in range(rxn.GetNumReactantTemplates()):
        query = Chem.Mol(rxn.GetReactantTemplate(i))
        out.append(Chem.MolToSmarts(strip_atom_maps(query) if strip_maps else query))
    return tuple(out)


def unique_templates(smarts_list: Sequence[str], label: str) -> list[str]:
    """Collapse a library to distinct reaction SMARTS, first-seen order kept.

    Applied to *both* libraries before either cross-library comparison, because
    both repeat themselves: USPTO extracts the same template from many reactions,
    and SmartReact's table carries one row per (reaction, category pair), which
    can land on the same SMARTS twice. A repeat raises the template count without
    contributing a distinct reactant pattern, so counting repeats would flatter
    whichever library repeats itself more -- USPTO here, by a wide margin.

    Only the saturation and filterability stages do this. The enumeration stages
    run the library as shipped, repeats and all, since that is the real workload.
    """
    seen = dict.fromkeys(smarts_list)
    if len(seen) != len(smarts_list):
        print(f"  {label:26s} {len(smarts_list):,} rows -> {len(seen):,} distinct templates")
    return list(seen)


def match_library_sizes(a: Sequence[str], b: Sequence[str],
                        label_a: str, label_b: str) -> tuple[list[str], list[str]]:
    """Trim the larger library down to the size of the smaller one.

    Both sweeps then run over identical template counts, so every x position
    lines up between the two curves -- the absolute counts in
    FILTERABILITY_COUNTS and, because SATURATION_FRACTIONS is a fraction of the
    library, the percentage points too.

    A seeded random sample, not a prefix: the auto-extracted file is in
    source-reaction order, so its first n rows would be a biased slice of
    reaction classes rather than a cross-section of the library.
    """
    n = min(len(a), len(b))

    def cut(lib: Sequence[str], label: str) -> list[str]:
        if len(lib) == n:
            return list(lib)
        print(f"  {label:26s} {len(lib):,} -> {n:,} templates "
              f"(matched to the smaller library)")
        return [lib[i] for i in sorted(random.Random(SEED).sample(range(len(lib)), n))]

    return cut(a, label_a), cut(b, label_b)


def _iter_reactions(smarts_list: Sequence[str]) -> Iterator[rdChemReactions.ChemicalReaction]:
    """Compile one at a time: the full library costs ~1.2 GB held all at once."""
    for smarts in smarts_list:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        if rxn is None or rxn.GetNumReactantTemplates() != 2:
            raise ValueError(f"Not a two-reactant template: {smarts[:80]}")
        yield rxn


def _dedupe_lhs(smarts_list: Sequence[str], pooled: bool):
    """Deduplicate reactant-side queries.

    pooled=False keeps the two roles in separate sets (FilterCatalog screens
    each role with its own catalog); pooled=True merges them. Atom maps are kept
    here, unlike in the saturation analysis: these go straight to RDKit and must
    match what RunReactants sees.
    """
    q1: list[Chem.Mol] = []
    q2: list[Chem.Mol] = []
    i1: dict[str, int] = {}
    i2: dict[str, int] = {}
    lhs: list[tuple[int, int]] = []
    for rxn in _iter_reactions(smarts_list):
        a = Chem.Mol(rxn.GetReactantTemplate(0))
        b = Chem.Mol(rxn.GetReactantTemplate(1))
        sa, sb = Chem.MolToSmarts(a), Chem.MolToSmarts(b)
        if sa not in i1:
            i1[sa] = len(q1)
            q1.append(a)
        if pooled:
            if sb not in i1:
                i1[sb] = len(q1)
                q1.append(b)
            lhs.append((i1[sa], i1[sb]))
        else:
            if sb not in i2:
                i2[sb] = len(q2)
                q2.append(b)
            lhs.append((i1[sa], i2[sb]))
    return q1, q2, lhs


def _join_triples(hits1, hits2, template_lhs) -> list[tuple[int, int, int]]:
    """Every oriented (molecule, molecule, template) triple the filter admits.

    Template-major, which the apply workers rely on. m1 != m2 mirrors
    itertools.combinations; both orientations appear when both are compatible.
    """
    return [(m1, m2, t) for t, (a, b) in enumerate(template_lhs)
            for m1, m2 in itertools.product(hits1[a], hits2[b]) if m1 != m2]


@dataclass
class ScreenResult:
    route: str
    seconds: float
    n_queries: int
    triples: list[tuple[int, int, int]]
    # What n_queries counts. The RDKit routes index distinct reactant-side
    # SMARTS; SmartReact indexes SMARTS-RX rules. Comparable in spirit, not in
    # unit, so the unit travels with the number.
    query_kind: str = "distinct LHS SMARTS"


def screen_filtercatalog(smarts_list, molecules) -> ScreenResult:
    """RDKit's own screening machinery, one catalog per reactant role.

    Single-threaded, and takes no core count: FilterCatalog.GetMatches has no
    threading option, so this route is inherently serial while
    screen_substructlibrary and screen_smartsrx are not. Any cross-route timing
    comparison inherits that asymmetry.
    """
    q1, q2, lhs = _dedupe_lhs(smarts_list, pooled=False)
    t0 = time.perf_counter()
    cat1, cat2 = FilterCatalog(), FilterCatalog()
    for catalog, queries, tag in ((cat1, q1, "lhs1"), (cat2, q2, "lhs2")):
        for i, query in enumerate(queries):
            entry = FilterCatalogEntry(f"{tag}_{i}", SmartsMatcher(f"{tag}_{i}", query))
            entry.SetProp("query_index", str(i))
            catalog.AddEntry(entry)
    hits1: list[list[int]] = [[] for _ in q1]
    hits2: list[list[int]] = [[] for _ in q2]
    for mid, mol in enumerate(molecules):
        for e in cat1.GetMatches(mol):
            hits1[int(e.GetProp("query_index"))].append(mid)
        for e in cat2.GetMatches(mol):
            hits2[int(e.GetProp("query_index"))].append(mid)
    triples = _join_triples(hits1, hits2, lhs)
    return ScreenResult("filtercatalog", time.perf_counter() - t0, len(q1) + len(q2),
                        triples, "distinct LHS SMARTS")


def screen_substructlibrary(smarts_list, molecules, cores: int = 1) -> ScreenResult:
    """Same exact prefilter via a pattern-fingerprint SubstructLibrary."""
    queries, _, lhs = _dedupe_lhs(smarts_list, pooled=True)
    # The reaction's own match parameters, so screening matches what
    # RunReactants will later do. Taken from the first template and applied to
    # every query: reactions built the same way (all through
    # ReactionFromSmarts, no per-template flags) share these settings, so this
    # holds for both libraries here. It would not hold for a library assembled
    # from reactions configured individually.
    params = next(_iter_reactions(smarts_list[:1])).GetSubstructParams()
    t0 = time.perf_counter()
    library = rdSubstructLibrary.SubstructLibrary(
        rdSubstructLibrary.MolHolder(), rdSubstructLibrary.PatternHolder())
    for mol in molecules:
        library.AddMol(mol)
    # maxResults=-1 is explicit: some overloads default to a finite cap.
    hits = [list(library.GetMatches(q, params, numThreads=1 if cores <= 1 else -1,
                                    maxResults=-1)) for q in queries]
    triples = _join_triples(hits, hits, lhs)
    return ScreenResult("substructlibrary", time.perf_counter() - t0, len(queries), triples)


def screen_smartsrx(templates: Sequence[ReactionTemplate], smiles: Sequence[str],
                    cores: int = 1) -> tuple[ScreenResult, dict[str, set[str]]]:
    """SmartReact's route: classify once against SMARTS-RX, then index.

    Timed on the same footing as the RDKit routes -- classification plus the
    join that turns keys into candidate triples.
    """
    index = build_template_index(list(templates))
    keygen = KeyGenerator(n_cores=cores)
    try:
        t0 = time.perf_counter()
        keys = preprocess_smiles(list(smiles), keygen)
        n_rules = len(keygen.rules)
        triples = [(i, j, t) if order == 0 else (j, i, t)
                   for i, j in itertools.combinations(range(len(smiles)), 2)
                   for t, orders in candidate_templates(
                       keys[smiles[i]], keys[smiles[j]], index).items()
                   for order in orders]
        triples.sort(key=lambda x: x[2])  # template-major, as the other routes are
        elapsed = time.perf_counter() - t0
    finally:
        keygen.close()
    return ScreenResult("smartreact", elapsed, n_rules, triples, "SMARTS-RX rules"), keys


@lru_cache(maxsize=256)
def _w_rxn(index: int):
    """Compile one template on demand, keeping a bounded number hot.

    Compiling the whole library per worker costs ~124 KB x 9,770 x n_workers.
    Because every triple stream is template-major, a batch normally references
    one template, so this hits almost always while holding at most 256.
    """
    return rdChemReactions.ReactionFromSmarts(_W_SMARTS[index])


def _apply_worker_init(smarts_list: list[str], smiles: list[str]) -> None:
    global _W_SMARTS, _W_MOLS
    _W_SMARTS = smarts_list
    _W_MOLS = [Chem.MolFromSmiles(s) for s in smiles]
    _w_rxn.cache_clear()


def _apply_chunk(triples, collect=True):
    productive = kept = 0
    out = []
    for m1, m2, t in triples:
        rxn = _w_rxn(t)
        if rxn is None:
            continue
        products = _collect_products(rxn, _W_MOLS[m1], _W_MOLS[m2], [0])
        if products:
            productive += 1
            kept += len(products)
            # Not collected when the caller only wants the timing: shipping
            # every product back over the pipe would land in the measurement.
            if collect:
                out.append((m1, m2, tuple(sorted(products))))
    return productive, kept, out


def _batched(triples: Iterable, size: int) -> Iterator[list]:
    it = iter(triples)
    while batch := list(itertools.islice(it, size)):
        yield batch


def apply_triples(triples, smarts_list, smiles, cores, collect=True, batch=20_000):
    """Run RunReactants over admitted triples.

    Returns (seconds, productive, products_summed, products_by_pair), where
    products_summed adds up each application's own product set -- deduplicated
    within an application, not across them.

    Only orientation 0 runs per triple: every filter emits the reversed triple
    separately when that orientation is also compatible.
    """
    acc: dict[tuple[str, str], set[str]] = defaultdict(set)
    productive = kept = 0

    def fold(out):
        for m1, m2, products in out:
            a, b = smiles[m1], smiles[m2]
            acc[(a, b) if a <= b else (b, a)].update(products)

    t0 = time.perf_counter()
    if cores <= 1:
        _apply_worker_init(smarts_list, smiles)
        for chunk in _batched(triples, batch):
            p, k, out = _apply_chunk(chunk, collect)
            productive += p
            kept += k
            fold(out)
    else:
        with ProcessPoolExecutor(max_workers=cores, initializer=_apply_worker_init,
                                 initargs=(smarts_list, smiles)) as pool:
            # Bounded in-flight so a huge generator is consumed incrementally.
            batches = _batched(triples, batch)
            pending = []
            for chunk in itertools.islice(batches, cores * 4):
                pending.append(pool.submit(_apply_chunk, chunk, collect))
            while pending:
                p, k, out = pending.pop(0).result()
                productive += p
                kept += k
                fold(out)
                if (nxt := next(batches, None)) is not None:
                    pending.append(pool.submit(_apply_chunk, nxt, collect))
    return (time.perf_counter() - t0, productive, kept,
            {k: frozenset(v) for k, v in acc.items()})


if __name__ == "__main__":
    sys.exit(main())
