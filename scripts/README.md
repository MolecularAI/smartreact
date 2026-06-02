# Scripts

These scripts reproduce the results and figures reported in the `smartreact` paper. They are not part of the library API and are not required to use `smartreact`. The computational resources under which each script was run are documented in the paper.

## Contents

- `enumerate_products.py` — case study: enumerate products from a sampled building-block library.
- `analyze_chemical_space.py` — Morgan fingerprints and UMAP embedding for the enumerated products.
- `analyze_template_coverage.py` — coverage statistics and chord diagrams over the SMARTS-RX hierarchy.
- `plot_reaction_highlights.py` — highlight figures for selected reactions.
- `run_benchmark_parallel.py` — scaling with worker count.
- `run_benchmark_precompute.py` — benefit of precomputing keys.
- `run_benchmark_scaling.py` — scaling with library size.

## Running

The case study and benchmark scripts expect a building-block parquet file at `data/building_blocks.parquet` (gitignored). Substitute any source of SMILES with a `smiles` column.

Use the `casestudy` pixi environment, which adds `scikit-learn`, `umap-learn`, `pyarrow`, and `matplotlib` on top of the runtime dependencies:

```bash
pixi run -e casestudy python scripts/enumerate_products.py
```
