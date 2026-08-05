# Changelog for smartreact

This follows the guideline on [keep a changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `negishi_insitu`, 176 templates covering the one-pot variant in which an aliphatic halide or triflate is zincated in situ, so both coupling partners are supplied as halides and no organozinc reagent is needed. Note that it overlaps `cross_electrophile_coupling`, which forms the same bond from the same substrate classes — pairs matched by both reactions now yield the same product under two reaction names

### Fixed

- `deoxygenative_coupling` and `cross_electrophile_coupling` no longer carry the reactant's configuration into the product at the reacting carbon. Both are Ni/photoredox couplings that break the bond homolytically and pass through a planar alkyl radical, so that centre is racemised — which is what makes the enantioconvergent variants of these reactions possible. The reacting carbon is now marked so RDKit discards its configuration: 120 of 120 `deoxygenative_coupling` templates at the carbinol carbon, and 52 of 196 `cross_electrophile_coupling` templates at whichever of the two coupling carbons is aliphatic (the remaining 144 couple two aromatic carbons and have no stereocentre to lose). Stereocentres elsewhere in the molecule are untouched, matching the chirality transfer reported for these reactions
- Williamson products now have the correct configuration at the halide carbon. Williamson is an SN2 displacement of the halide, so that carbon inverts while the alkoxide carbon — whose C–O bond is never broken — is retained. Both were previously retained. All 104 williamson templates now mark the halide carbon (atom map 4) as inverted relative to the input; the alkoxide carbon is deliberately left unmarked. Verified against two reference outcomes: butan-1-ol + (S)-2-bromooctane gives (R)-2-butoxyoctane, and (S)-octan-2-ol + iodomethane gives (S)-2-methoxyoctane
- Mitsunobu products now have the correct configuration at the carbinol carbon. The reaction proceeds by backside attack, so that centre inverts, but the templates left it unmarked and RDKit's default is to carry the reactant's configuration through unchanged — every stereodefined Mitsunobu product was the wrong enantiomer. All 60 mitsunobu templates now mark the centre as inverted relative to the input, so both input enantiomers are handled by the same template and an undefined centre stays undefined. No products are gained or lost; only their stereochemistry changes

### Changed

- **Breaking:** `negishi` is renamed `negishi_batch`, matching upstream. Its 44 templates are unchanged, but they always required a preformed organozinc reactant, which the new name makes explicit. `reaction_list=["negishi"]` must become `reaction_list=["negishi_batch"]`
- 191 templates revised upstream across `chan_lam` (9), `imidazole_Xketone_synthesis` (10), `imidazole_condensation_acid` (60), `imidazole_condensation_amine` (60), `reductive_amination_ketone` (36) and `sn2_nheterocycle` (16) — mostly tighter atom environments and aromatic/aliphatic corrections. No other reaction's templates change
- The `example` column was regenerated upstream for 1,316 rows across 10 reactions. Examples are illustrative only and do not affect enumeration; the refreshed ones are more accurate, reproducing their own template in every row sampled
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
