# Scripts

`reproduce_paper.py` regenerates every result and figure in the `smartreact` paper. It is the only script needed. `prepare_templates.py` is a maintenance tool, not a paper result: it rebuilds the bundled data files from an upstream template drop.

Neither is part of the library API, and neither is required to use `smartreact`.

## Running

Use the `baseline` pixi environment. It is a superset of `casestudy` and the only one carrying `rdchiral`, which the auto-extracted comparison library needs.

```bash
# check the pipeline end to end in a few minutes before committing hours
pixi run -e baseline python scripts/reproduce_paper.py --quick

# the real thing
pixi run -e baseline python scripts/reproduce_paper.py
```

`--quick` shrinks every input. It verifies that the pipeline runs; the numbers it produces are not the paper's and must not be cited. Every CSV records `quick_mode`, so a smoke-test result cannot be mistaken for a real one.

Other options:

```bash
python scripts/reproduce_paper.py --only baselines        # one or more stages
python scripts/reproduce_paper.py --skip chemspace        # all but these
python scripts/reproduce_paper.py --cores 32              # else SLURM_CPUS_PER_TASK, else all
python scripts/reproduce_paper.py --building-blocks bb.parquet
```

On a cluster: `sbatch slurm/run_paper.sh`.

## Choosing what runs

The `STAGES` block at the top of `main()` switches stages on and off, and `SETTINGS` below it holds every size. Editing them is equivalent to passing flags, and it survives in the file, so a cluster job records what it was asked to do.

```python
STAGES = {
    "saturation": True,     # Fig 1a  reactant patterns vs library size
    "filterability": True,  # Fig 1b  one RDKit prefilter, two libraries
    "baselines": True,      # Table 2 four routes, one apply path
    ...
}
```

## Stages

| Stage | Paper | What it measures |
|---|---|---|
| `saturation` | Fig 1a | Distinct reactant patterns vs library size, constructed vs auto-extracted |
| `filterability` | Fig 1b | RDKit's FilterCatalog over both libraries: index size, screening time, work admitted |
| `baselines` | Table 2 | Brute force, FilterCatalog, SubstructLibrary and SMARTS-RX keys over one shared apply path |
| `scaling` | Fig 3a | Throughput vs template count, every route |
| `parallel` | Fig 3b | Weak scaling over cores |
| `precompute` | Fig 3c | Classification reuse vs how often molecules recur |
| `casestudy` | Sec 3.5, App B | All pairs of the 1,000 public building blocks |
| `chemspace` | Fig 4, App C | Fingerprints, UMAP, reaction highlight panels |
| `coverage` | App D, E | SMARTS-RX coverage statistics, chord diagrams, L3 table |

`chemspace` reads what `casestudy` writes. Running it alone works only if that output is already on disk.

The workflow schematics — the pipeline overview and the worked template example — are drawn by hand and are not produced here.

## Data

Everything runs from the 1,000 building blocks committed at `public_data/casestudy/building_blocks_1000.csv`, so the paper reproduces from a clean checkout with no private data. Pass `--building-blocks` to substitute any CSV or parquet with a `smiles` column.

`saturation` and `filterability` download USPTO-50k (~23 MB) on first run and cache it under `data/paper/`. The extracted template cache is named for the number of source reactions it was built from, so a truncated `--quick` extraction can never be picked up by a later full run.

Outputs are written to `data/paper/`. Every CSV row carries the `smartreact_version`, `rdkit_version`, `platform`, `run_cores` and `quick_mode` that produced it, so provenance travels with the data.

## Provenance

Every stage reads the **installed** `smartreact` package and its bundled template and key files through `importlib.resources`, never from `src/`. Re-running after editing or upgrading the package therefore measures the new version, with no edits needed here.

## Hardware

Written for a 32-core node with at least 128 GB of RAM. Memory is the binding constraint, not CPU: compiled reaction templates cost about **1.2 GB per worker process** (roughly 124 KB each across 9,770 templates), so budget about `cores x 1.5 GB`. With much less, the kernel will OOM-kill the run.

Workers compile templates on demand behind a bounded cache and are fed work in template-major order, which is what keeps that figure at 1.2 GB rather than the full library per worker. There is no other memory management: size the job properly.
