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

# Williamson reference cases: SN2 inverts the halide carbon and leaves the
# alkoxide carbon alone, so (S)-2-bromooctane gives the (R) ether while
# (S)-octan-2-ol keeps its configuration.
BUTAN_1_OL = "CCCCO"
IODOMETHANE = "CI"
S_2_BROMOOCTANE = "CCCCCC[C@H](C)Br"
R_2_BUTOXYOCTANE = "CCCCCC[C@@H](C)OCCCC"
S_OCTAN_2_OL = "CCCCCC[C@H](C)O"
S_2_METHOXYOCTANE = "CCCCCC[C@H](C)OC"

# Negishi: the batch variant needs a preformed organozinc, the in-situ variant
# zincates an aliphatic halide on the fly and takes the two halides directly.
PHENYLZINC = "[Zn]c1ccccc1"
BROMOANISOLE = "COc1ccc(Br)cc1"
IODOBUTANE = "CCCCI"
METHOXYBIPHENYL = "COc1ccc(-c2ccccc2)cc1"
BUTYLANISOLE = "CCCCc1ccc(OC)cc1"


@pytest.fixture(scope="session")
def keygen() -> KeyGenerator:
    return KeyGenerator(n_cores=1)


@pytest.fixture(scope="session")
def enumerator() -> ReactionEnumerator:
    return ReactionEnumerator(n_cores=1)


@pytest.fixture(scope="session")
def enumerator_all() -> ReactionEnumerator:
    return ReactionEnumerator(reaction_list="all", n_cores=1)
