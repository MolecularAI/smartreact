from __future__ import annotations

from rdkit import Chem

from smartreact.cleaning import clean_smiles


class TestCleanSmiles:
    def test_valid_smiles_returns_string(self):
        result = clean_smiles("c1ccccc1")
        assert isinstance(result, str)
        assert Chem.MolFromSmiles(result) is not None

    def test_invalid_smiles_returns_none_and_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="smartreact.cleaning"):
            result = clean_smiles("not_a_smiles")
        assert result is None
        assert any("Could not parse" in msg for msg in caplog.messages)

    def test_empty_smiles_returns_none_and_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="smartreact.cleaning"):
            result = clean_smiles("")
        assert result is None
        assert any("empty" in msg.lower() for msg in caplog.messages)

    def test_output_is_canonical(self):
        """Different input representations of benzene should yield the same output."""
        result1 = clean_smiles("c1ccccc1")
        result2 = clean_smiles("C1=CC=CC=C1")
        assert result1 == result2

    def test_salt_removal(self):
        """Largest fragment is kept; counterion is stripped."""
        result = clean_smiles("c1ccccc1.O")  # benzene + water
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        # Only benzene should remain
        assert mol.GetNumAtoms() == 6

    def test_charge_neutralization(self):
        """Charged carboxylate should be neutralized to carboxylic acid."""
        result = clean_smiles("CC(=O)[O-]")
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        assert Chem.rdmolops.GetFormalCharge(mol) == 0

    def test_aspirin_different_atom_orderings_are_identical(self):
        """Aspirin written with different atom traversal orders must canonicalize identically."""
        assert clean_smiles("CC(=O)Oc1ccccc1C(=O)O") == clean_smiles("OC(=O)c1ccccc1OC(C)=O")

    def test_ethanol_different_orderings_are_identical(self):
        """Simple reordering of atoms in ethanol must produce the same canonical SMILES."""
        assert clean_smiles("CCO") == clean_smiles("OCC")

    def test_salt_stripping_removes_counterions(self):
        """Sodium acetate should reduce to acetic acid after fragment removal and neutralization."""
        assert clean_smiles("CC(=O)[O-].[Na+]") == clean_smiles("CC(=O)O")

    def test_stereoisomers_are_kept_distinct(self):
        """Cleaning must not destroy stereochemistry: L- and D-alanine must remain different."""
        assert clean_smiles("N[C@@H](C)C(=O)O") != clean_smiles("N[C@H](C)C(=O)O")

    def test_isotope_labels_are_stripped(self):
        """13C-labelled and unlabelled ethanol must clean to the same molecule."""
        assert clean_smiles("[13CH3]CO") == clean_smiles("CCO")

    def test_largest_fragment_kept_in_mixture(self):
        """In a multi-component SMILES the largest molecule must be returned."""
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        assert clean_smiles(f"{aspirin}.COCCO") == clean_smiles(aspirin)
