# Changelog for smartreact

This follows the guideline on [keep a changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] 2026-07-28

### Added

- `build_template_index` and `candidate_templates` for looking up applicable reaction templates from a reactant pair's key sets

### Changed

- `ReactionEnumerator` selects templates through the new index instead of scanning the full template library for every reactant pair
- Each unique SMILES is parsed once per call rather than once per pair occurrence, and parallel workers receive pre-parsed molecules restricted to the molecules their own batch references
- Enumeration results and their ordering are unchanged

## [1.0.0] 2026-06-08

### Added

- Initial release of smartreact
