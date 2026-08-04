from __future__ import annotations

import itertools
import logging
from unittest.mock import patch

from rdkit import Chem
from rdkit.Chem import AllChem

from smartreact import ReactionEnumerator
from smartreact.enumerator import _collect_products
from smartreact.types import ReactionResult

from .conftest import (
    ACETIC_ACID,
    BIPHENYL,
    BROMOBENZENE,
    BUTAN_2_OL,
    DIMETHYLAMINE,
    ETHANOL,
    ETHYLAMINE,
    IODOBENZENE,
    PHENOL,
    PHENYLBORONIC_ACID,
    R_BUTAN_2_OL,
    S_BUTAN_2_OL,
)


class TestEnumeratePair:
    def test_suzuki_coupling(self, enumerator: ReactionEnumerator):
        """Bromobenzene + phenylboronic acid should yield biphenyl via Suzuki."""
        results = enumerator.enumerate_pair(BROMOBENZENE, PHENYLBORONIC_ACID)
        assert len(results) > 0
        all_products = {p for r in results for p in r.products}
        assert BIPHENYL in all_products

    def test_result_type(self, enumerator: ReactionEnumerator):
        results = enumerator.enumerate_pair(BROMOBENZENE, PHENYLBORONIC_ACID)
        for r in results:
            assert isinstance(r, ReactionResult)
            assert r.products

    def test_reactants_sorted(self, enumerator: ReactionEnumerator):
        results = enumerator.enumerate_pair(BROMOBENZENE, PHENYLBORONIC_ACID)
        for r in results:
            assert r.reactant_a <= r.reactant_b

    def test_invalid_smiles(self, enumerator: ReactionEnumerator):
        results = enumerator.enumerate_pair("invalid", BROMOBENZENE)
        assert results == []

    def test_no_reaction(self, enumerator: ReactionEnumerator):
        """Two molecules with no compatible reaction should return empty."""
        results = enumerator.enumerate_pair("C", "CC")
        assert results == []


class TestMitsunobuStereochemistry:
    """Mitsunobu displaces the alcohol by backside attack, so the carbinol carbon inverts."""

    def test_s_alcohol_gives_r_ether(self, enumerator: ReactionEnumerator):
        products = enumerator.products_for_pair(PHENOL, S_BUTAN_2_OL)
        assert products == ["CC[C@@H](C)Oc1ccccc1"]

    def test_r_alcohol_gives_s_ether(self, enumerator: ReactionEnumerator):
        """The same templates must invert the opposite enantiomer too, not fix an absolute tag."""
        products = enumerator.products_for_pair(PHENOL, R_BUTAN_2_OL)
        assert products == ["CC[C@H](C)Oc1ccccc1"]

    def test_undefined_centre_stays_undefined(self, enumerator: ReactionEnumerator):
        """Inverting an unknown configuration is still unknown; no stereochemistry is invented."""
        products = enumerator.products_for_pair(PHENOL, BUTAN_2_OL)
        assert products == ["CCC(C)Oc1ccccc1"]


class TestProductsForPair:
    def test_returns_strings(self, enumerator: ReactionEnumerator):
        products = enumerator.products_for_pair(BROMOBENZENE, PHENYLBORONIC_ACID)
        assert all(isinstance(p, str) for p in products)
        assert BIPHENYL in products

    def test_sorted(self, enumerator: ReactionEnumerator):
        products = enumerator.products_for_pair(BROMOBENZENE, PHENYLBORONIC_ACID)
        assert products == sorted(products)


class TestEnumeratePairs:
    def test_multiple_pairs(self, enumerator: ReactionEnumerator):
        pairs = [
            (BROMOBENZENE, PHENYLBORONIC_ACID),
            (IODOBENZENE, ETHYLAMINE),
        ]
        results = enumerator.enumerate_pairs(pairs)
        assert len(results) > 0

    def test_empty_input(self, enumerator: ReactionEnumerator):
        assert enumerator.enumerate_pairs([]) == []

    def test_parses_each_unique_molecule_once(self, enumerator: ReactionEnumerator):
        """Regression test: enumerate_pairs must parse each unique SMILES once,
        not once per pair occurrence, even when a molecule appears in many pairs.

        Keys are precomputed up front so the patched window only covers
        enumerator-level Mol parsing, not KeyGenerator's own (separate)
        internal MolFromSmiles call during classification.
        """
        from smartreact.preprocessing import preprocess_smiles

        pairs = [
            (BROMOBENZENE, PHENYLBORONIC_ACID),
            (BROMOBENZENE, ETHYLAMINE),
            (BROMOBENZENE, ACETIC_ACID),
            (IODOBENZENE, ETHYLAMINE),
        ]
        unique_smiles = {s for pair in pairs for s in pair}
        keys_map = preprocess_smiles(list(unique_smiles), enumerator.keygen)

        with patch(
            "smartreact.enumerator.Chem.MolFromSmiles",
            wraps=Chem.MolFromSmiles,
        ) as mock_parse:
            enumerator.enumerate_pairs(pairs, parallel=False, precomputed_keys=keys_map)

        assert mock_parse.call_count == len(unique_smiles)

    def test_parallel_matches_sequential(self):
        pairs = [
            (BROMOBENZENE, PHENYLBORONIC_ACID),
            (IODOBENZENE, ETHYLAMINE),
        ]
        enum_seq = ReactionEnumerator(n_cores=1)
        enum_par = ReactionEnumerator(n_cores=2)

        results_seq = enum_seq.enumerate_pairs(pairs, parallel=False)
        results_par = enum_par.enumerate_pairs(pairs, parallel=True)

        def _key(r):
            return (r.reactant_a, r.reactant_b, r.reaction_name, tuple(r.products))

        products_seq = {_key(r) for r in results_seq}
        products_par = {_key(r) for r in results_par}
        assert products_seq == products_par

    def test_parallel_with_clean_smiles_matches_sequential(self):
        """parallel=True + clean_smiles=True must produce the same results as sequential."""
        pairs = [
            (BROMOBENZENE, PHENYLBORONIC_ACID),
            (IODOBENZENE, ETHYLAMINE),
        ]
        enum_seq = ReactionEnumerator(n_cores=1)
        enum_par = ReactionEnumerator(n_cores=2)

        results_seq = enum_seq.enumerate_pairs(pairs, parallel=False, clean_smiles=True)
        results_par = enum_par.enumerate_pairs(pairs, parallel=True, clean_smiles=True)

        def _key(r):
            return (r.reactant_a, r.reactant_b, r.reaction_name, tuple(r.products))

        assert {_key(r) for r in results_seq} == {_key(r) for r in results_par}

    def test_workers_receive_only_their_own_batch_molecules(self):
        """Each submitted batch must carry only the molecules it references.

        ``submit`` pickles its arguments per call, so passing the chunk-wide
        mol/key dicts to every batch would serialise them n_cores times over.
        """
        molecules = [
            BROMOBENZENE,
            PHENYLBORONIC_ACID,
            ETHYLAMINE,
            ACETIC_ACID,
            IODOBENZENE,
            ETHANOL,
        ]
        pairs = list(itertools.combinations(molecules, 2))

        enum_par = ReactionEnumerator(n_cores=3)
        ex = enum_par._get_pool()
        real_submit = ex.submit
        captured: list[tuple[list[tuple[str, str]], set[str], set[str]]] = []

        def _spy(fn, batch, mols, keys):
            captured.append((batch, set(mols), set(keys)))
            return real_submit(fn, batch, mols, keys)

        with patch.object(ex, "submit", _spy):
            enum_par.enumerate_pairs(pairs, parallel=True)

        assert len(captured) > 1, "expected the pair list to be split across several batches"
        for batch, mols, keys in captured:
            needed = {s for pair in batch for s in pair}
            assert mols == needed
            assert keys == needed

        # At least one batch must be a strict subset, or the assertions above
        # would also pass for code that broadcasts everything to everyone.
        assert any(mols < set(molecules) for _, mols, _ in captured)
        assert set().union(*(mols for _, mols, _ in captured)) == set(molecules)


class TestEnumeratorInit:
    def test_default_reactions(self, enumerator: ReactionEnumerator):
        assert len(enumerator.templates) > 0

    def test_template_index_covers_all_templates(self, enumerator: ReactionEnumerator):
        """The key-pair index used for candidate lookup must reference every
        loaded template exactly once."""
        indexed_ids = sorted(idx for idxs in enumerator.template_index.values() for idx in idxs)
        assert indexed_ids == list(range(len(enumerator.templates)))

    def test_all_reactions(self, enumerator_all: ReactionEnumerator):
        default_enum = ReactionEnumerator(n_cores=1)
        assert len(enumerator_all.templates) >= len(default_enum.templates)

    def test_custom_reactions(self):
        enum = ReactionEnumerator(reaction_list=["suzuki"], n_cores=1)
        assert all(t.name == "suzuki" for t in enum.templates)

    def test_n_cores_auto(self):
        enum = ReactionEnumerator(n_cores=-1)
        assert enum.n_cores >= 1


# Suzuki coupling SMARTS: aryl halide + boronic acid -> biaryl
_SUZUKI_SMARTS = "[c:1][Br:2].[c:3][B:4]([OH:5])[OH:6]>>[c:1][c:3]"


class TestCollectProducts:
    """Tests for the _collect_products helper."""

    def test_valid_reaction_returns_products(self):
        rxn = AllChem.ReactionFromSmarts(_SUZUKI_SMARTS)
        m1 = Chem.MolFromSmiles(BROMOBENZENE)
        m2 = Chem.MolFromSmiles(PHENYLBORONIC_ACID)
        products = _collect_products(rxn, m1, m2, orders=[0])
        assert len(products) > 0
        assert all(Chem.MolFromSmiles(s) is not None for s in products)

    def test_order_swaps_reactants(self):
        rxn = AllChem.ReactionFromSmarts(_SUZUKI_SMARTS)
        m1 = Chem.MolFromSmiles(BROMOBENZENE)
        m2 = Chem.MolFromSmiles(PHENYLBORONIC_ACID)
        products_fwd = _collect_products(rxn, m1, m2, orders=[0])
        products_rev = _collect_products(rxn, m1, m2, orders=[1])
        # Swapped order puts boronic acid first — shouldn't match the template
        assert len(products_fwd) > 0
        assert len(products_rev) == 0

    def test_both_orders(self):
        rxn = AllChem.ReactionFromSmarts(_SUZUKI_SMARTS)
        m1 = Chem.MolFromSmiles(BROMOBENZENE)
        m2 = Chem.MolFromSmiles(PHENYLBORONIC_ACID)
        products = _collect_products(rxn, m1, m2, orders=[0, 1])
        assert len(products) > 0

    def test_no_matching_reactants_returns_empty(self):
        rxn = AllChem.ReactionFromSmarts(_SUZUKI_SMARTS)
        m1 = Chem.MolFromSmiles("C")
        m2 = Chem.MolFromSmiles("CC")
        products = _collect_products(rxn, m1, m2, orders=[0, 1])
        assert products == set()

    def test_invalid_product_smiles_logs_error(self, caplog):
        """When MolToSmiles raises, an error should be logged with reactant
        and reaction SMARTS context."""
        rxn = AllChem.ReactionFromSmarts(_SUZUKI_SMARTS)
        m1 = Chem.MolFromSmiles(BROMOBENZENE)
        m2 = Chem.MolFromSmiles(PHENYLBORONIC_ACID)

        with (
            patch(
                "smartreact.enumerator.Chem.MolToSmiles",
                side_effect=RuntimeError("bad mol"),
            ),
            caplog.at_level(logging.ERROR, logger="smartreact.enumerator"),
        ):
            products = _collect_products(rxn, m1, m2, orders=[0])

        assert products == set()
        assert any("invalid SMILES" in msg for msg in caplog.messages)
        # Log should contain reactant context
        assert any("Reactants:" in msg for msg in caplog.messages)
        assert any("SMARTS:" in msg for msg in caplog.messages)

    def test_empty_smiles_string_logs_error(self, caplog):
        """When MolToSmiles returns an empty string, an error should be logged."""
        rxn = AllChem.ReactionFromSmarts(_SUZUKI_SMARTS)
        m1 = Chem.MolFromSmiles(BROMOBENZENE)
        m2 = Chem.MolFromSmiles(PHENYLBORONIC_ACID)

        with (
            patch(
                "smartreact.enumerator.Chem.MolToSmiles",
                return_value="",
            ),
            caplog.at_level(logging.ERROR, logger="smartreact.enumerator"),
        ):
            products = _collect_products(rxn, m1, m2, orders=[0])

        assert products == set()
        assert any("invalid SMILES" in msg for msg in caplog.messages)

    def test_run_reactants_exception_is_skipped(self):
        """If RunReactants raises, that order is silently skipped."""
        rxn = AllChem.ReactionFromSmarts(_SUZUKI_SMARTS)
        m1 = Chem.MolFromSmiles(BROMOBENZENE)
        m2 = Chem.MolFromSmiles(PHENYLBORONIC_ACID)

        with patch.object(rxn, "RunReactants", side_effect=RuntimeError("fail")):
            products = _collect_products(rxn, m1, m2, orders=[0])

        assert products == set()


def _result_key(r: ReactionResult) -> tuple[str, str, str, tuple[str, ...]]:
    return (r.reactant_a, r.reactant_b, r.reaction_name, tuple(r.products))


class TestPerReactionCorrectness:
    """Lock down a few non-Suzuki templates so silent breakage is caught."""

    def test_suzuki_reaction_name_present(self, enumerator: ReactionEnumerator):
        results = enumerator.enumerate_pair(BROMOBENZENE, PHENYLBORONIC_ACID)
        assert any(r.reaction_name == "suzuki" for r in results)

    def test_esterification(self, enumerator: ReactionEnumerator):
        """Acetic acid + ethanol must fire 'esterification' and yield ethyl acetate."""
        results = enumerator.enumerate_pair(ACETIC_ACID, ETHANOL)
        ester_results = [r for r in results if r.reaction_name == "esterification"]
        assert ester_results, "no esterification result for acid + alcohol"
        ethyl_acetate = Chem.MolToSmiles(Chem.MolFromSmiles("CCOC(C)=O"))
        all_products = {p for r in ester_results for p in r.products}
        assert ethyl_acetate in all_products

    def test_amide_coupling(self, enumerator: ReactionEnumerator):
        """Acid + secondary amine must fire 'amide_coupling' and yield the expected amide."""
        results = enumerator.enumerate_pair(ACETIC_ACID, DIMETHYLAMINE)
        amide_results = [r for r in results if r.reaction_name == "amide_coupling"]
        assert amide_results, "no amide_coupling result for acid + secondary amine"
        dmac = Chem.MolToSmiles(Chem.MolFromSmiles("CN(C)C(C)=O"))
        all_products = {p for r in amide_results for p in r.products}
        assert dmac in all_products


class TestEnumeratePairsLazy:
    """Correctness coverage for the chunked lazy variant."""

    def test_empty_input_yields_nothing(self, enumerator: ReactionEnumerator):
        assert list(enumerator.enumerate_pairs_lazy([])) == []

    def test_single_chunk_matches_eager(self, enumerator: ReactionEnumerator):
        pairs = [
            (BROMOBENZENE, PHENYLBORONIC_ACID),
            (IODOBENZENE, ETHYLAMINE),
            (ACETIC_ACID, ETHANOL),
        ]
        eager = enumerator.enumerate_pairs(pairs)
        lazy = list(enumerator.enumerate_pairs_lazy(pairs, chunk_size=100))
        assert {_result_key(r) for r in eager} == {_result_key(r) for r in lazy}

    def test_multiple_chunks_matches_eager(self, enumerator: ReactionEnumerator):
        """chunk_size smaller than input forces multiple chunks; results must be identical."""
        pairs = [
            (BROMOBENZENE, PHENYLBORONIC_ACID),
            (IODOBENZENE, ETHYLAMINE),
            (ACETIC_ACID, ETHANOL),
            (ACETIC_ACID, DIMETHYLAMINE),
            (BROMOBENZENE, PHENYLBORONIC_ACID),
        ]
        eager = enumerator.enumerate_pairs(pairs)
        # chunk_size=2 → chunks of [2, 2, 1] — exercises non-divisible boundary
        lazy = list(enumerator.enumerate_pairs_lazy(pairs, chunk_size=2))
        assert {_result_key(r) for r in eager} == {_result_key(r) for r in lazy}

    def test_chunk_size_one(self, enumerator: ReactionEnumerator):
        pairs = [(BROMOBENZENE, PHENYLBORONIC_ACID), (ACETIC_ACID, ETHANOL)]
        eager = enumerator.enumerate_pairs(pairs)
        lazy = list(enumerator.enumerate_pairs_lazy(pairs, chunk_size=1))
        assert {_result_key(r) for r in eager} == {_result_key(r) for r in lazy}

    def test_accepts_generator(self, enumerator: ReactionEnumerator):
        """The pairs argument is documented as an Iterable — generators must work."""
        pairs = [(BROMOBENZENE, PHENYLBORONIC_ACID), (ACETIC_ACID, ETHANOL)]
        gen = (p for p in pairs)
        lazy = list(enumerator.enumerate_pairs_lazy(gen, chunk_size=10))
        eager = enumerator.enumerate_pairs(pairs)
        assert {_result_key(r) for r in eager} == {_result_key(r) for r in lazy}

    def test_parallel_lazy_matches_sequential_lazy(self):
        pairs = [
            (BROMOBENZENE, PHENYLBORONIC_ACID),
            (IODOBENZENE, ETHYLAMINE),
            (ACETIC_ACID, ETHANOL),
            (ACETIC_ACID, DIMETHYLAMINE),
        ]
        with ReactionEnumerator(n_cores=1) as enum_seq, ReactionEnumerator(n_cores=2) as enum_par:
            seq = list(enum_seq.enumerate_pairs_lazy(pairs, parallel=False, chunk_size=2))
            par = list(enum_par.enumerate_pairs_lazy(pairs, parallel=True, chunk_size=2))
        assert {_result_key(r) for r in seq} == {_result_key(r) for r in par}


class TestPrecomputedKeysImmutability:
    def test_enumerate_pairs_does_not_mutate_precomputed_keys(self):
        """enumerate_pairs must not modify the dict supplied via precomputed_keys."""
        from smartreact.preprocessing import preprocess_smiles

        with ReactionEnumerator(n_cores=1) as enumerator:
            keys_map = preprocess_smiles([BROMOBENZENE, PHENYLBORONIC_ACID], enumerator.keygen)
            before_ids = {k: id(v) for k, v in keys_map.items()}
            before_snapshot = {k: frozenset(v) for k, v in keys_map.items()}

            # Include a SMILES *not* in the map so the merge branch fires too.
            pairs = [(BROMOBENZENE, PHENYLBORONIC_ACID), (ACETIC_ACID, ETHANOL)]
            enumerator.enumerate_pairs(pairs, precomputed_keys=keys_map)

            assert set(keys_map.keys()) == set(before_snapshot.keys())
            for k, v in keys_map.items():
                assert frozenset(v) == before_snapshot[k]
                assert id(v) == before_ids[k]
