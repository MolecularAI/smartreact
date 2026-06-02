## smartreact

[![CI](https://github.com/MolecularAI/smartreact/actions/workflows/test.yml/badge.svg)](https://github.com/MolecularAI/smartreact/actions/workflows/test.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Enumerate forward synthesis products at scale. Every prediction is tied to a named, chemically grounded reaction template.

### Installation

Only RDKit is pulled in as a runtime dependency.

**With pip** (from PyPI):

```bash
pip install smartreact
```

**With pixi** (handles the rdkit dependency via conda-forge). [Install pixi](https://pixi.sh/latest/#installation) first, then:

```bash
git clone https://github.com/MolecularAI/smartreact.git
cd smartreact
pixi install
```

### Quick Start

```python
from smartreact import ReactionEnumerator

enumerator = ReactionEnumerator()

# Suzuki coupling: bromobenzene + phenylboronic acid
results = enumerator.enumerate_pair("c1ccc(Br)cc1", "c1ccc(B(O)O)cc1")

for result in results:
    print(f"Reaction: {result.reaction_name}")
    print(f"Reactants: {result.reactant_a} + {result.reactant_b}")
    print(f"Products: {result.products}")
```

See [`notebooks/examples.ipynb`](notebooks/examples.ipynb) for more detailed examples.

### Usage

The `ReactionEnumerator` class generates products from reactant pairs using curated SMARTS templates. It uses SMARTS-RX functional group keys to filter compatible reactant pairs before applying reactions.

#### Single Pair

```python
from smartreact import ReactionEnumerator

enumerator = ReactionEnumerator()

# Or restrict to a specific subset of reactions
enumerator = ReactionEnumerator(reaction_list=["suzuki", "amide_coupling", "snar_amine"])

results = enumerator.enumerate_pair("c1ccc(Br)cc1", "c1ccc(B(O)O)cc1")

# Pass clean_smiles=True to standardize SMILES before enumeration if desired
results = enumerator.enumerate_pair("c1ccc(Br)cc1", "c1ccc(B(O)O)cc1", clean_smiles=True)
```

#### Batch Processing

```python
from smartreact import ReactionEnumerator

enumerator = ReactionEnumerator(n_cores=8)

pairs = [
    ("c1ccc(Br)cc1", "c1ccc(B(O)O)cc1"),
    ("c1ccc(I)cc1", "CCN"),
    ("CC(=O)O", "CCNC"),
]

# Returns a list — all results held in memory at once
results = enumerator.enumerate_pairs(pairs, parallel=True)

# Pass clean_smiles=True to standardize SMILES before enumeration if desired
results = enumerator.enumerate_pairs(pairs, parallel=True, clean_smiles=True)

for result in results:
    products_str = ", ".join(result.products)
    print(f"{result.reaction_name}: {result.reactant_a} + {result.reactant_b} -> {products_str}")
```

#### Large-Scale / Memory-Efficient Processing

For large datasets, use `enumerate_pairs_lazy` to stream results without
materialising everything in memory.  Pairs are processed in chunks of
`chunk_size` (default 50 000); peak memory scales with the chunk, not the
total input.

```python
from smartreact import ReactionEnumerator

enumerator = ReactionEnumerator(n_cores=8)

# pairs can be a generator — it is never fully materialised
def pair_generator():
    ...

for result in enumerator.enumerate_pairs_lazy(pair_generator(), parallel=True):
    print(result.reaction_name, result.products)

# Tune chunk_size to trade memory for fewer key-classification calls
for result in enumerator.enumerate_pairs_lazy(pairs, parallel=True, chunk_size=10_000):
    ...
```

#### Key Classification

```python
from smartreact import KeyGenerator

keygen = KeyGenerator()
result = keygen.classify("c1ccc(Br)cc1")
cats, subs, subsubs = result.categories()
print(cats, subs, subsubs)
```

#### Preprocessing Molecule Libraries

When each molecule appears in many pairs — for example when enumerating all
pairwise combinations of a compound library — it is wasteful to re-classify the
same molecule for every pair.  `preprocess_smiles` classifies each molecule once
and returns a mapping that can be reused across all enumeration calls.

```python
from smartreact import ReactionEnumerator, KeyGenerator
from smartreact.preprocessing import preprocess_smiles, save_preprocessed, load_preprocessed

keygen = KeyGenerator(n_cores=8)
library = ["c1ccc(Br)cc1", "c1ccc(B(O)O)cc1", "CC(=O)O", ...]

# Classify once.  Pass clean_smiles=True to standardize SMILES first
# (removes isotopes, strips salts/solvents, neutralizes charges, etc.).
keys_map = preprocess_smiles(library, keygen, clean_smiles=True)

# Optionally persist to disk and reload later
save_preprocessed(keys_map, "preprocessed.csv")
keys_map = load_preprocessed("preprocessed.csv")

# Pass precomputed keys — no re-classification happens during enumeration
enumerator = ReactionEnumerator(n_cores=8)
from itertools import combinations
pairs = list(combinations(library, 2))
results = enumerator.enumerate_pairs(pairs, precomputed_keys=keys_map)
```

Any SMILES not present in `keys_map` are classified on-the-fly as a fallback.

### Key Concepts

- **Parallelization**: Multi-core processing via `n_cores`. Use `parallel=True` in `enumerate_pairs()` / `enumerate_pairs_lazy()` and set `n_cores=-1` for all available cores.
- **Memory-efficient streaming**: `enumerate_pairs_lazy()` processes pairs in fixed-size chunks and yields results incrementally, keeping peak memory proportional to `chunk_size` rather than the total input size.
- **Preprocessing**: `preprocess_smiles()` classifies each molecule once and returns a `dict[str, set[str]]` that can be passed to `enumerate_pairs(precomputed_keys=...)`. This is most useful when the same molecules appear in many pairs, e.g. when enumerating all pairwise combinations of a library.
- **SMILES standardization**: All enumeration methods and `preprocess_smiles` accept `clean_smiles=True` to normalize SMILES before processing (removes isotopes, strips salts/solvents, neutralizes charges, keeps the largest fragment).
- **Reaction Selection**: All 36 available reaction types are used by default. Pass a custom list to restrict enumeration to specific reactions, e.g. `reaction_list=["suzuki", "amide_coupling"]`.
- **Result Format**: `ReactionResult` objects with `reactant_a`, `reactant_b` (sorted alphabetically), `reaction_name`, and `products` (canonical SMILES).

> **Note on RDKit logging.** Importing `smartreact` silences RDKit's C++ log channel (`rdApp.*`) process-wide via `RDLogger.DisableLog`. This keeps batch enumeration from flooding stderr when parsing unreliable inputs, but it also mutes RDKit warnings produced by *other* code in the same process. If you need RDKit's warnings, re-enable them after import with `RDLogger.EnableLog("rdApp.*")`.

#### Input Format Conversion

Molecules in InChI or SDF format can be converted to SMILES before enumeration:

```python
from smartreact import to_smiles, read_sdf_file

# InChI to SMILES
smi = to_smiles("InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H")  # -> "c1ccccc1"

# SDF mol block to SMILES
smi = to_smiles(mol_block, fmt="sdf")

# Auto-detect format
smi = to_smiles(input_string)  # detects InChI, SDF, or SMILES

# Load a multi-molecule SDF file as a SMILES list
library = read_sdf_file("compounds.sdf")
```

### Contributing

Contributions are welcome — bug reports, feature requests, documentation fixes, and new reaction templates. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow, code style, and guidance on adding reaction templates.

Authors and maintainers are listed in [`AUTHORS.md`](AUTHORS.md).

### License

Apache-2.0
