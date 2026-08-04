import pytest

from smartreact import KeyGenerator, ReactionEnumerator

BROMOBENZENE = "c1ccc(Br)cc1"
PHENYLBORONIC_ACID = "c1ccc(B(O)O)cc1"
IODOBENZENE = "c1ccc(I)cc1"
ETHYLAMINE = "CCN"
ACETIC_ACID = "CC(=O)O"
METHYLAMINE = "CN"
BIPHENYL = "c1ccc(-c2ccccc2)cc1"
ETHANOL = "CCO"
DIMETHYLAMINE = "CNC"
PHENOL = "Oc1ccccc1"
S_BUTAN_2_OL = "C[C@H](O)CC"
R_BUTAN_2_OL = "C[C@@H](O)CC"
BUTAN_2_OL = "CCC(O)C"


@pytest.fixture(scope="session")
def keygen() -> KeyGenerator:
    return KeyGenerator(n_cores=1)


@pytest.fixture(scope="session")
def enumerator() -> ReactionEnumerator:
    return ReactionEnumerator(n_cores=1)


@pytest.fixture(scope="session")
def enumerator_all() -> ReactionEnumerator:
    return ReactionEnumerator(reaction_list="all", n_cores=1)
