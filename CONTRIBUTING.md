# Contributing to smartreact

Thanks for your interest in contributing! `smartreact` is an open scientific tool, and we welcome contributions from chemists, cheminformaticians, and software developers alike.

## Ways to contribute

- **Report bugs** — open a GitHub issue with a minimal reproducible example (input SMILES, expected vs. actual output, traceback).
- **Suggest features** — open an issue describing the use case before sending a PR for larger changes.
- **Contribute reaction templates** — see [Adding reaction templates](#adding-reaction-templates) below.
- **Improve documentation** — fixes to the README, docstrings, or notebooks are always welcome.
- **Fix bugs or add features** — see [Development workflow](#development-workflow).

## Development workflow

### 1. Set up the environment

This project uses [pixi](https://pixi.sh) for dependency management.

```bash
git clone https://github.com/MolecularAI/smartreact.git
cd smartreact
pixi install -e dev
```

The `dev` environment adds `pytest` and `ruff` on top of the runtime dependencies. (The `default` environment installs only what end users get.) Other environments are available when needed:

```bash
pixi install -e notebook    # dev + jupyter + ipywidgets for working with notebooks/
pixi install -e casestudy   # dev + scikit-learn, umap-learn, pyarrow, matplotlib for the case study
```

All environments share a single dependency solve, so versions stay consistent across them.

### 2. Create a branch

```bash
git checkout -b your-feature-name
```

### 3. Make your changes

The pipeline is **SMILES → SMARTS-RX classification → template matching → RDKit reaction → products**. The core modules live in `src/smartreact/` — see the `README.md` for an overview.

### 4. Run tests, linting, and formatting

Run these in the `dev` environment:

```bash
pixi run -e dev test          # run the test suite
pixi run -e dev lint          # check for lint errors
pixi run -e dev format        # auto-format with ruff
pixi run -e dev format-check  # verify formatting without modifying files
```

All of these must pass before opening a pull request.

### 5. Open a pull request

- Reference any related issue (e.g. `Fixes #42`).
- Describe what changed and why.
- Include tests for new behavior.
- Keep PRs focused; split unrelated changes into separate PRs.

## Adding reaction templates

Reaction templates live in `src/smartreact/data/templates.txt` (TSV with columns `reaction_name`, `template_id`, `example`, `reaction_template`, `reactant_categories`).

When proposing a new template:

- Provide a clear `reaction_name` and a unique `template_id`.
- Include an `example` reaction (SMILES `reactants>>product`) so the template's intended behavior is obvious.
- Document the chemistry: cite a reference (paper, named reaction, etc.) in the PR description.
- Ensure `reactant_categories` correctly references SMARTS-RX functional group keys defined in `src/smartreact/data/keys.txt`.
- Add a test in `tests/` demonstrating the template fires on a known substrate pair and produces the expected product.

Template additions should be reviewed by a maintainer with chemistry expertise (currently Thierry Kogej).

## Code style

- Python 3.10+, ruff-formatted, line length 100.
- Lint rules: `E, F, W, I, UP, B, SIM` (configured in `pyproject.toml`).
- `smartreact` is treated as first-party for import sorting.
- Prefer explicit, well-named functions over clever one-liners. Avoid unnecessary comments; let identifiers speak.

## Testing

- Tests live in `tests/` and use `pytest`.
- New features should include unit tests; bug fixes should include a regression test.
- Run a single test with: `pixi run -e dev test -- tests/test_enumerator.py::test_name`.

## Reporting security issues

Please do not report security vulnerabilities through public GitHub issues. Instead, contact the maintainers directly.

## Code of conduct

Be respectful and constructive. We aim for a welcoming environment for contributors of all backgrounds and experience levels.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0 (see `LICENSE`).
