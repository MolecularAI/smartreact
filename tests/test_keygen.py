from __future__ import annotations

from smartreact import KeyGenerator, load_rules
from smartreact.types import KeysResult


class TestLoadRules:
    def test_rules_non_empty(self):
        rules = load_rules()
        assert len(rules) > 0

    def test_rule_has_qmol(self):
        rules = load_rules()
        for r in rules[:5]:
            assert r.qmol is not None
            assert r.category
            assert r.subcategory
            assert r.subsubcategory


class TestKeyGenerator:
    def test_classify_valid_smiles(self, keygen: KeyGenerator):
        result = keygen.classify("c1ccc(Br)cc1")
        assert isinstance(result, KeysResult)
        assert result.smiles == "c1ccc(Br)cc1"

    def test_classify_invalid_smiles(self, keygen: KeyGenerator):
        result = keygen.classify("not_a_smiles")
        assert result.matches == []

    def test_classify_boronic_acid(self, keygen: KeyGenerator):
        result = keygen.classify("c1ccc(B(O)O)cc1")
        assert len(result.matches) > 0
        cats, _, _ = result.categories()
        assert len(cats) > 0

    def test_classify_many_sequential(self, keygen: KeyGenerator):
        smiles_list = ["c1ccc(Br)cc1", "CCN", "CC(=O)O"]
        results = keygen.classify_many(smiles_list, parallel=False)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, KeysResult)

    def test_classify_many_preserves_order(self, keygen: KeyGenerator):
        smiles_list = ["CCO", "CCN", "CCC"]
        results = keygen.classify_many(smiles_list, parallel=False)
        assert [r.smiles for r in results] == smiles_list


class TestClassifyManyParallel:
    """Cover the parallel path of KeyGenerator.classify_many."""

    def test_parallel_matches_sequential(self):
        smiles_list = [
            "c1ccc(Br)cc1",
            "c1ccc(B(O)O)cc1",
            "CCN",
            "CC(=O)O",
            "CCO",
            "CNC",
            "c1ccc(I)cc1",
            "CN",
            "CC(=O)OCC",
            "CC(=O)NC",
            "c1ccccc1",
            "CCC",
        ]
        kg_seq = KeyGenerator(n_cores=1)
        with KeyGenerator(n_cores=2) as kg_par:
            seq = kg_seq.classify_many(smiles_list, parallel=False)
            par = kg_par.classify_many(smiles_list, parallel=True)

        assert [r.smiles for r in par] == smiles_list  # order preserved
        for s, p in zip(seq, par, strict=True):
            assert s.smiles == p.smiles
            seq_keys = {(m.category, m.subcategory, m.subsubcategory) for m in s.matches}
            par_keys = {(m.category, m.subcategory, m.subsubcategory) for m in p.matches}
            assert seq_keys == par_keys

    def test_below_threshold_short_circuits_to_sequential(self):
        """With fewer SMILES than min_parallel_threshold, no pool is spawned."""
        with KeyGenerator(n_cores=2) as kg:
            results = kg.classify_many(["CCO", "CCN"], parallel=True, min_parallel_threshold=10)
        assert len(results) == 2
        assert [r.smiles for r in results] == ["CCO", "CCN"]
        # Short-circuit path must not have created a pool.
        assert kg._pool is None  # noqa: SLF001
