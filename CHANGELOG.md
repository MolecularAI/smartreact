# Changelog for smartreact

This follows the guideline on [keep a changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Reaction templates updated with the inert `;!$([a3])` qualifier removed (11,848 occurrences across 2,368 templates). `[a3]` means "aromatic *and* isotope 3", which no realistic molecule satisfies
- The same qualifier removed from 18 SMARTS-RX definitions in `src/smartreact/data/keys.txt` (84 occurrences), keeping the keys consistent with the templates derived from them
- Key classification and enumeration results are unchanged: verified identical over the 1,000 public case-study building blocks and all 499,500 of their pairs

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
