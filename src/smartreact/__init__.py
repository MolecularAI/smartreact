"""SmartReact: SMARTS-based reaction enumeration with SMARTS-RX functional group filtering."""

from .cleaning import clean_smiles
from .conversion import detect_format, read_sdf_block, read_sdf_file, to_smiles
from .enumerator import ReactionEnumerator
from .keygen import KeyGenerator, KeysLibrary, load_rules
from .keys import extract_key_strings, orders_for_template
from .preprocessing import load_preprocessed, preprocess_smiles, save_preprocessed
from .templates import load_templates
from .types import KeyMatch, KeysResult, ReactionResult, ReactionTemplate, Rule

try:
    from ._version import __version__, __version_tuple__  # noqa: F401
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "ReactionEnumerator",
    "KeyGenerator",
    "KeysLibrary",
    "load_templates",
    "load_rules",
    "ReactionTemplate",
    "ReactionResult",
    "KeyMatch",
    "KeysResult",
    "Rule",
    "extract_key_strings",
    "orders_for_template",
    "clean_smiles",
    "to_smiles",
    "detect_format",
    "read_sdf_file",
    "read_sdf_block",
    "preprocess_smiles",
    "save_preprocessed",
    "load_preprocessed",
]
