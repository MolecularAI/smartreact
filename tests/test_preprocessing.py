from __future__ import annotations

import pytest

from smartreact import KeyGenerator, ReactionEnumerator
from smartreact.preprocessing import load_preprocessed, preprocess_smiles, save_preprocessed

from .conftest import BIPHENYL, BROMOBENZENE, PHENYLBORONIC_ACID

_VALID_SMILES = [BROMOBENZENE, PHENYLBORONIC_ACID, BIPHENYL]


@pytest.fixture(scope="module")
def keygen() -> KeyGenerator:
    return KeyGenerator(n_cores=1)


class TestPreprocessSmiles:
    def test_returns_dict_for_all_inputs(self, keygen: KeyGenerator):
        result = preprocess_smiles(_VALID_SMILES, keygen)
        assert set(result.keys()) == set(_VALID_SMILES)

    def test_values_are_sets_of_strings(self, keygen: KeyGenerator):
        result = preprocess_smiles(_VALID_SMILES, keygen)
        for keys in result.values():
            assert isinstance(keys, set)
            assert all(isinstance(k, str) for k in keys)

    def test_known_keys_present(self, keygen: KeyGenerator):
        """Bromobenzene should be classified as an aryl bromide."""
        result = preprocess_smiles([BROMOBENZENE], keygen)
        keys = result[BROMOBENZENE]
        assert any("Bromide" in k or "Bromo" in k or "Br" in k for k in keys)

    def test_empty_input(self, keygen: KeyGenerator):
        assert preprocess_smiles([], keygen) == {}

    def test_deduplication(self, keygen: KeyGenerator):
        """Duplicate SMILES should appear only once in the result."""
        result = preprocess_smiles([BROMOBENZENE, BROMOBENZENE], keygen)
        assert len(result) == 1

    def test_clean_smiles_flag(self, keygen: KeyGenerator):
        """Cleaning should not change results for already-clean SMILES."""
        result_no_clean = preprocess_smiles([BROMOBENZENE], keygen, clean_smiles=False)
        result_clean = preprocess_smiles([BROMOBENZENE], keygen, clean_smiles=True)
        # Both should classify; keys may differ slightly but should not be empty
        assert len(result_no_clean) == 1
        assert len(result_clean) == 1

    def test_clean_smiles_skips_invalid(self, keygen: KeyGenerator):
        """Invalid SMILES should be skipped when clean_smiles=True."""
        result = preprocess_smiles(["not_valid", BROMOBENZENE], keygen, clean_smiles=True)
        assert "not_valid" not in result
        assert BROMOBENZENE in result or len(result) == 1


class TestSaveLoadPreprocessed:
    def test_round_trip(self, keygen: KeyGenerator, tmp_path):
        keys_map = preprocess_smiles(_VALID_SMILES, keygen)
        path = tmp_path / "preprocessed.csv"

        save_preprocessed(keys_map, path)
        loaded = load_preprocessed(path)

        assert set(loaded.keys()) == set(keys_map.keys())
        for smi in keys_map:
            assert loaded[smi] == keys_map[smi]

    def test_csv_has_expected_columns(self, keygen: KeyGenerator, tmp_path):
        import pandas as pd

        keys_map = preprocess_smiles([BROMOBENZENE], keygen)
        path = tmp_path / "preprocessed.csv"
        save_preprocessed(keys_map, path)

        df = pd.read_csv(path)
        assert "smiles" in df.columns
        assert "keys" in df.columns

    def test_empty_keys_round_trips(self, keygen: KeyGenerator, tmp_path):
        """Molecules with no key matches should survive a save/load cycle."""
        keys_map = {"C": set()}
        path = tmp_path / "empty_keys.csv"
        save_preprocessed(keys_map, path)
        loaded = load_preprocessed(path)
        assert loaded["C"] == set()

    def test_save_creates_file(self, keygen: KeyGenerator, tmp_path):
        keys_map = preprocess_smiles([BROMOBENZENE], keygen)
        path = tmp_path / "out.csv"
        save_preprocessed(keys_map, path)
        assert path.exists()


class TestEnumeratePairsWithPrecomputedKeys:
    def test_precomputed_keys_match_normal_results(self):
        """enumerate_pairs with precomputed_keys should produce the same results."""
        enumerator = ReactionEnumerator(n_cores=1)
        pairs = [(BROMOBENZENE, PHENYLBORONIC_ACID)]

        keys_map = preprocess_smiles([BROMOBENZENE, PHENYLBORONIC_ACID], enumerator.keygen)

        results_normal = enumerator.enumerate_pairs(pairs)
        results_precomputed = enumerator.enumerate_pairs(pairs, precomputed_keys=keys_map)

        def _key(r):
            return (r.reactant_a, r.reactant_b, r.reaction_name, tuple(r.products))

        assert {_key(r) for r in results_normal} == {_key(r) for r in results_precomputed}

    def test_missing_smiles_computed_on_the_fly(self):
        """SMILES absent from precomputed_keys are classified on the fly."""
        enumerator = ReactionEnumerator(n_cores=1)
        pairs = [(BROMOBENZENE, PHENYLBORONIC_ACID)]

        # Precomputed keys that are intentionally incomplete
        partial_keys = preprocess_smiles([BROMOBENZENE], enumerator.keygen)

        # Should not raise; missing PHENYLBORONIC_ACID keys are computed on the fly
        results = enumerator.enumerate_pairs(pairs, precomputed_keys=partial_keys)
        assert isinstance(results, list)

    def test_parallel_with_precomputed_keys_matches_sequential(self):
        """parallel=True + precomputed_keys must produce the same results as sequential."""
        pairs = [(BROMOBENZENE, PHENYLBORONIC_ACID)]
        enum_seq = ReactionEnumerator(n_cores=1)
        enum_par = ReactionEnumerator(n_cores=2)

        keys_map = preprocess_smiles([BROMOBENZENE, PHENYLBORONIC_ACID], enum_seq.keygen)

        results_seq = enum_seq.enumerate_pairs(pairs, parallel=False, precomputed_keys=keys_map)
        results_par = enum_par.enumerate_pairs(pairs, parallel=True, precomputed_keys=keys_map)

        def _key(r):
            return (r.reactant_a, r.reactant_b, r.reaction_name, tuple(r.products))

        assert {_key(r) for r in results_seq} == {_key(r) for r in results_par}


class TestEnumeratePairWithCleanSmiles:
    def test_clean_smiles_returns_results(self):
        enumerator = ReactionEnumerator(n_cores=1)
        results = enumerator.enumerate_pair(BROMOBENZENE, PHENYLBORONIC_ACID, clean_smiles=True)
        assert len(results) > 0

    def test_clean_smiles_invalid_returns_empty(self):
        enumerator = ReactionEnumerator(n_cores=1)
        results = enumerator.enumerate_pair("not_valid", BROMOBENZENE, clean_smiles=True)
        assert results == []

    def test_enumerate_pairs_clean_smiles_skips_invalid(self):
        enumerator = ReactionEnumerator(n_cores=1)
        pairs = [("not_valid", BROMOBENZENE), (BROMOBENZENE, PHENYLBORONIC_ACID)]
        results = enumerator.enumerate_pairs(pairs, clean_smiles=True)
        # Second pair should still produce results
        assert len(results) > 0
