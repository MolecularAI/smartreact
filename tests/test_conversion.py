from rdkit import Chem

from smartreact.conversion import detect_format, read_sdf_block, read_sdf_file, to_smiles

BENZENE_INCHI = "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
ETHANOL_INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
BENZENE_SMILES = "c1ccccc1"
ETHANOL_SMILES = "CCO"


def _mol_block(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToMolBlock(mol)


def _sdf_text(*smiles_list: str) -> str:
    blocks = []
    for smi in smiles_list:
        blocks.append(_mol_block(smi))
        blocks.append("$$$$")
    return "\n".join(blocks)


# --- detect_format ---


def test_detect_format_inchi():
    assert detect_format(BENZENE_INCHI) == "inchi"


def test_detect_format_sdf():
    assert detect_format(_mol_block("CCO")) == "sdf"


def test_detect_format_smiles():
    assert detect_format("CCO") == "smiles"
    assert detect_format("c1ccccc1") == "smiles"


# --- to_smiles ---


def test_to_smiles_inchi():
    assert to_smiles(BENZENE_INCHI, fmt="inchi") == BENZENE_SMILES
    assert to_smiles(ETHANOL_INCHI, fmt="inchi") == ETHANOL_SMILES


def test_to_smiles_sdf():
    block = _mol_block("CCO")
    result = to_smiles(block, fmt="sdf")
    assert result == ETHANOL_SMILES


def test_to_smiles_passthrough():
    assert to_smiles("CCO", fmt="smiles") == ETHANOL_SMILES
    assert to_smiles(BENZENE_SMILES, fmt="smiles") == BENZENE_SMILES


def test_to_smiles_auto_inchi():
    assert to_smiles(BENZENE_INCHI) == BENZENE_SMILES


def test_to_smiles_auto_sdf():
    block = _mol_block("CCO")
    assert to_smiles(block) == ETHANOL_SMILES


def test_to_smiles_auto_smiles():
    assert to_smiles("CCO") == ETHANOL_SMILES


def test_to_smiles_invalid():
    assert to_smiles("not_a_molecule", fmt="inchi") is None
    assert to_smiles("", fmt="smiles") is None
    assert to_smiles("   ", fmt="auto") is None


def test_to_smiles_mol_alias():
    block = _mol_block("CCO")
    assert to_smiles(block, fmt="mol") == ETHANOL_SMILES


def test_to_smiles_unsupported_format():
    assert to_smiles("CCO", fmt="xyz") is None


# --- read_sdf_block ---


def test_read_sdf_block_single():
    text = _sdf_text("CCO")
    result = read_sdf_block(text)
    assert result == [ETHANOL_SMILES]


def test_read_sdf_block_multiple():
    text = _sdf_text("CCO", "c1ccccc1", "CC(=O)O")
    result = read_sdf_block(text)
    assert len(result) == 3
    assert ETHANOL_SMILES in result
    assert BENZENE_SMILES in result


def test_read_sdf_block_empty():
    assert read_sdf_block("") == []


# --- read_sdf_file ---


def test_read_sdf_file(tmp_path):
    sdf_path = tmp_path / "mols.sdf"
    sdf_path.write_text(_sdf_text("CCO", "c1ccccc1"))
    result = read_sdf_file(sdf_path)
    assert len(result) == 2
    assert ETHANOL_SMILES in result
    assert BENZENE_SMILES in result


def test_read_sdf_file_with_bad_entry(tmp_path):
    good_block = _mol_block("CCO")
    # Corrupt a mol block by mangling atom lines
    bad_block = "bad_molecule\n     RDKit\n\n  0  0  0  0  0  0  0  0  0  0  0 V2000\nM  END"
    text = good_block + "\n$$$$\n" + bad_block + "\n$$$$\n"
    sdf_path = tmp_path / "mixed.sdf"
    sdf_path.write_text(text)
    result = read_sdf_file(sdf_path)
    assert ETHANOL_SMILES in result
